"""Overrides that each answer `issubclass(arg, C)` become one declared requirement (E1 S1)."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from nominal_refactor_advisor.codemod_payload import (
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
)
from nominal_refactor_advisor.codemod_reproof import (
    RepositorySourceReprovedOperation,
    SourceReproofDiagnostic,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_semantics import CodemodPreflightStatus
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)


@dataclass(frozen=True)
class RequirementAnswer:
    """A classmethod of one parameter whose body, past its docstring, is one
    return: `True` for the base, `issubclass(parameter, C)` for an override."""

    method: ast.FunctionDef
    docstring: str | None
    returned: ast.expr

    @classmethod
    def of(cls, node: ast.stmt, where: str) -> RequirementAnswer:
        if not isinstance(node, ast.FunctionDef):
            raise ValueError(f"{where} binds the name other than by a method")
        match node.decorator_list:
            case [ast.Name(id="classmethod")]:
                pass
            case _:
                raise ValueError(f"{where} is not a plain classmethod")
        arguments = node.args
        if (
            len(arguments.args) != 2
            or arguments.posonlyargs or arguments.kwonlyargs or arguments.vararg or arguments.kwarg
        ):
            raise ValueError(f"{where} does not take exactly one argument")
        body = list(node.body)
        docstring = ast.get_docstring(node, clean=False)
        if docstring is not None:
            body = body[1:]
        match body:
            case [ast.Return(value=returned)] if returned is not None:
                return cls(node, docstring, returned)
        raise ValueError(f"{where} does more than return its answer")

    @property
    def parameter(self) -> str:
        return self.method.args.args[1].arg

    def required(self, where: str) -> ast.expr:
        """The class the override requires its argument to be."""

        match self.returned:
            case ast.Call(func=ast.Name(id="issubclass"), args=[ast.Name(id=name), required], keywords=[]) if (
                name == self.parameter
            ):
                return required
        raise ValueError(
            f"{where} answers `{ast.unparse(self.returned)}`, not `issubclass({self.parameter}, C)`; "
            "name the capability it requires first"
        )


@dataclass(frozen=True, kw_only=True)
class DeclaredCapabilityRequirementOperation(RepositorySourceReprovedOperation):
    """Turn a family's per-member `issubclass` answers into one declared requirement
    (IMPL-3: each override re-derives, with a type test, a class it could name).

    The target is the base's classmethod, `Base.method(cls, arg)`, whose body
    returns `True`. Each descendant overriding it must return exactly
    `issubclass(arg, C)`; its override becomes the class attribute
    `attribute_name = C`, keeping its docstring after it. The base gains
    `attribute_name: ClassVar[type] = object` and answers
    `issubclass(arg, cls.attribute_name)`. `issubclass(x, object)` holds for every
    class, so the base's answer is unchanged, and attribute lookup follows the
    same MRO the overrides did. Any other override, a negated test among them,
    refuses by name.
    """

    attribute_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def preflight_reports(
        self, context: CodemodSelectorContext
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            self.required_reproof(lambda: self.source_edits_from_snapshot(context))
        except CodemodOperationPreflightError as error:
            return (error.report,)
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.PASSED,
                message="Every override declares the class it requires",
                detail=SourceReproofDiagnostic(self.target, ()),
            ),
        )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        source_path = self.required_source_path(snapshot, self.operation_key())
        class_name, _, method_name = (self.target.qualname or "").rpartition(".")
        if not class_name:
            raise ValueError("declared_capability_requirement requires a Class.method target_qualname")
        base = ResolvedClassTarget.from_rewrite_target(
            snapshot, SourceRewriteTarget(file_path=source_path, qualname=class_name)
        )
        index = snapshot.required_class_family_index
        base_symbol = base.required_symbol(snapshot)
        classes = index.classes_by_symbol
        family = (base_symbol, *index.descendant_symbols(base_symbol), *index.ancestor_symbols(base_symbol))
        bound = [symbol for symbol in family if self.attribute_name in classes[symbol].member_binding_names]
        if bound:
            raise ValueError(f"{sorted(bound)!r} already bind {self.attribute_name!r}")
        base_method = next((node for node in base.node.body if _binds(node, method_name)), None)
        if base_method is None:
            raise ValueError(f"`{base.qualname}` declares no {method_name!r}")
        answer = RequirementAnswer.of(base_method, f"{base.qualname}.{method_name}")
        if not (isinstance(answer.returned, ast.Constant) and answer.returned.value is True):
            raise ValueError(f"`{base.qualname}.{method_name}` does not answer True by default")
        geometries: dict[str, SourceTextGeometry] = {}
        replacements: dict[str, list[tuple[int, int, str]]] = {}

        def geometry(file_path: str) -> SourceTextGeometry:
            return geometries.setdefault(
                file_path, SourceTextGeometry(snapshot.sources_by_file_path[file_path])
            )

        for symbol in index.descendant_symbols(base_symbol):
            declaration = classes[symbol]
            if method_name not in declaration.member_binding_names:
                continue
            node = ResolvedClassTarget.from_rewrite_target(
                snapshot,
                SourceRewriteTarget(file_path=declaration.file_path, qualname=declaration.qualname),
            ).node
            where = f"{declaration.qualname}.{method_name}"
            overrides = [statement for statement in node.body if _binds(statement, method_name)]
            if len(overrides) != 1:
                raise ValueError(f"{where} is bound more than once")
            override = RequirementAnswer.of(overrides[0], where)
            required = ast.unparse(override.required(where))
            text = f"{self.attribute_name} = {required}"
            if override.docstring is not None:
                text += f"\n{_quoted(override.docstring)}"
            replacements.setdefault(declaration.file_path, []).append(
                (*_whole_lines(geometry(declaration.file_path), override.method), _indented(text, override.method))
            )
        if not replacements:
            raise ValueError(f"no descendant of `{base.qualname}` overrides {method_name!r}")
        assert isinstance(base_method, ast.FunctionDef)
        start, _end = _whole_lines(geometry(source_path), base_method)
        declared = (
            f"{self.attribute_name}: ClassVar[type] = object\n"
            f"\"\"\"The class `{method_name}` requires its argument to be; every class is an object.\"\"\"\n"
        )
        returned = base_method.body[-1]
        replacements.setdefault(source_path, []).extend((
            (start, start, _indented(declared, base_method) + "\n\n"),
            (
                *geometry(source_path).required_node_offsets(returned),
                f"return issubclass({answer.parameter}, cls.{self.attribute_name})",
            ),
        ))
        edits = [
            geometry(file_path).nominal_edit(
                file_path=file_path,
                replacements=tuple(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=start, end_offset=end, replacement_source=text
                    )
                    for start, end, text in spans
                ),
                rationale=self.rationale
                or f"Each member of `{base.qualname}` declares the class {method_name!r} requires.",
            )
            for file_path, spans in replacements.items()
        ]
        edits.extend(
            self.required_import_mutations(
                source_path,
                import_source="from typing import ClassVar",
                default_rationale="The requirement is a class variable.",
            )
        )
        return tuple(edits)


def _binds(statement: ast.stmt, name: str) -> bool:
    match statement:
        case ast.FunctionDef(name=bound) | ast.AsyncFunctionDef(name=bound) | ast.ClassDef(name=bound):
            return bound == name
        case ast.Assign(targets=targets):
            return any(isinstance(target, ast.Name) and target.id == name for target in targets)
        case ast.AnnAssign(target=ast.Name(id=bound)):
            return bound == name
    return False


def _whole_lines(geometry: SourceTextGeometry, method: ast.FunctionDef) -> tuple[int, int]:
    """A method's lines from its first decorator through its last, without the final newline."""

    lines = geometry.source.splitlines(keepends=True)
    first = min((decorator.lineno for decorator in method.decorator_list), default=method.lineno)
    start = sum(len(line) for line in lines[: first - 1])
    end = sum(len(line) for line in lines[: method.end_lineno or method.lineno])
    return start, end - (1 if lines[(method.end_lineno or method.lineno) - 1].endswith("\n") else 0)


def _indented(text: str, method: ast.FunctionDef) -> str:
    indent = " " * method.col_offset
    return "\n".join(f"{indent}{line}" if line else "" for line in text.splitlines())


def _quoted(docstring: str) -> str:
    return '"""' + docstring.replace('"""', r"\"\"\"") + '"""'
