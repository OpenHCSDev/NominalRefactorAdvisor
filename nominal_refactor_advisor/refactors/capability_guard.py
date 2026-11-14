"""A membership guard that raises becomes a method the capability overrides (E1 S2)."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from nominal_refactor_advisor.codemod_payload import (
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import (
    PreflightByReproof,
    RepositorySourceReprovedOperation,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.refactors.answered_method import AnsweredMethod


@dataclass(frozen=True)
class CapabilityGuard:
    """One statement of a function that refuses a value lacking a capability, read
    once from its source: `if not issubclass(x, C): raise ...`, `if not
    isinstance(x, C): raise ...` or `if x is not C: raise ...`, where `x` is a name
    or an attribute of the function's own receiver."""

    statement: ast.If
    subject: ast.expr
    capability: ast.expr
    raised: ast.Raise
    on_class: bool
    """Whether the subject is a class rather than an instance."""
    exact: bool
    """Whether the test is the class itself (`is not`) rather than a subclass."""

    @classmethod
    def in_function(cls, function: ast.FunctionDef | ast.AsyncFunctionDef) -> CapabilityGuard:
        for statement in function.body:
            match statement:
                case ast.If(
                    test=ast.UnaryOp(
                        op=ast.Not(),
                        operand=ast.Call(
                            func=ast.Name(id="issubclass" | "isinstance" as test),
                            args=[(ast.Name() | ast.Attribute()) as subject, capability],
                            keywords=[],
                        ),
                    ),
                    body=[ast.Raise() as raised],
                    orelse=[],
                ):
                    return cls(statement, subject, capability, raised, test == "issubclass", False)
                case ast.If(
                    test=ast.Compare(
                        left=(ast.Name() | ast.Attribute()) as subject,
                        ops=[ast.IsNot()],
                        comparators=[capability],
                    ),
                    body=[ast.Raise() as raised],
                    orelse=[],
                ):
                    return cls(statement, subject, capability, raised, True, True)
        raise ValueError(
            f"`{function.name}` has no capability guard that raises at its top level"
        )

    def raised_on(self, receiver: str) -> str:
        """The raise with the subject read as the method's own receiver."""

        subject = ast.dump(self.subject)

        class Receiver(ast.NodeTransformer):
            def visit(self, node: ast.AST) -> ast.AST:
                if isinstance(node, ast.expr) and ast.dump(node) == subject:
                    return ast.Name(id=receiver, ctx=ast.Load())
                return super().visit(node)

        return ast.unparse(Receiver().visit(copy.deepcopy(self.raised)))


@dataclass(frozen=True, kw_only=True)
class CapabilityGuardToOverrideOperation(
    PreflightByReproof, RepositorySourceReprovedOperation
):
    """Turn a guard that refuses every value lacking a capability into a method the
    capability answers (IMPL-3: the test re-derives at the consumer what the class
    already is).

    The target is the function holding `if not issubclass(x, Capability): raise E`
    (or `isinstance`); `base` is the class that answers the default, and
    `method_name` names the method. `base` gains `method_name`, which raises E,
    and `Capability` overrides it to pass. The guard becomes `x.method_name()`;
    when the target is `base.method_name` itself and the guard is all it does,
    its body becomes the raise.

    The proof is the MRO of every class `x` can be: the subject's declared family
    (the enclosing class for `cls` or `self`, else the parameter's annotation,
    `type[T]` or `T`). For each, `method_name` must resolve to the capability's
    override exactly when the class is one, and to `base` otherwise. Another
    declaration of the name on the way, or a class that reaches neither, refuses.
    """

    base: SourceRewriteTarget = codemod_payload_field(PayloadRecordValueCodec(SourceRewriteTarget))
    method_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    preflight_passed_message = (
        "Every class the subject can be dispatches to the capability's answer"
    )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        source_path = self.required_source_path(snapshot, self.operation_key())
        qualname = self.target.qualname or ""
        class_name, _, function_name = qualname.rpartition(".")
        module = snapshot.module_nodes_by_file_path[source_path]
        owner = (
            ResolvedClassTarget.from_rewrite_target(
                snapshot, SourceRewriteTarget(file_path=source_path, qualname=class_name)
            )
            if class_name
            else None
        )
        body = module.body if owner is None else owner.node.body
        function = next(
            (
                node
                for node in body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
            ),
            None,
        )
        if function is None:
            raise ValueError(f"{qualname!r} is not a function declared in {source_path!r}")
        guard = CapabilityGuard.in_function(function)
        base = ResolvedClassTarget.from_rewrite_target(snapshot, self.base)
        base_answer = AnsweredMethod(snapshot, base, function)
        parents = snapshot.parsed_module_for_source_path(source_path).fact(ModuleNodeIndex).parents
        capability = base_answer.class_symbol(source_path, guard.capability, parents)
        if capability is None:
            raise ValueError(f"`{ast.unparse(guard.capability)}` is not a class the scan declares")
        subject = self.subject_family(snapshot, base_answer, owner, function, guard, parents, source_path)
        base_symbol = base.required_symbol(snapshot)
        if guard.exact:
            below = subject & frozenset(snapshot.required_class_family_index.descendant_symbols(capability))
            if below:
                raise ValueError(
                    f"the guard tests `{ast.unparse(guard.capability)}` itself, and {sorted(below)!r} "
                    "descend from it: an override would let them pass"
                )
        self.require_dispatch(snapshot, subject, base_symbol, capability)
        capability_class = ResolvedClassTarget.from_rewrite_target(
            snapshot, self.class_target(snapshot, capability)
        )
        receiver = "cls" if guard.on_class else "self"
        decorator = "@classmethod\n" if guard.on_class else ""
        signature = f"def {self.method_name}({receiver}) -> None:"
        replacements: dict[str, list[tuple[int, int, str]]] = {}
        geometries: dict[str, SourceTextGeometry] = {}

        def geometry(file_path: str) -> SourceTextGeometry:
            return geometries.setdefault(
                file_path, SourceTextGeometry(snapshot.sources_by_file_path[file_path])
            )

        def add(file_path: str, start: int, end: int, text: str) -> None:
            replacements.setdefault(file_path, []).append((start, end, text))

        whole = (
            owner is not None
            and owner.required_symbol(snapshot) == base_symbol
            and function.name == self.method_name
            and [statement for statement in function.body if not _docstring(statement)] == [guard.statement]
        )
        if whole:
            start, end = geometry(source_path).required_node_offsets(guard.statement)
            add(source_path, start, end, guard.raised_on(receiver))
        else:
            if self.method_name in base.bound_names:
                raise ValueError(f"`{base.qualname}` already binds {self.method_name!r}")
            start, end = geometry(source_path).required_node_offsets(guard.statement)
            add(source_path, start, end, f"{ast.unparse(guard.subject)}.{self.method_name}()")
            raised = guard.raised_on(receiver)
            add(base.file_path, *self.appended(
                geometry(base.file_path), base.node,
                f"{decorator}{signature}\n    \"\"\"Refused: only a {capability_class.node.name} passes.\"\"\"\n    {raised}",
            ))
        if self.method_name in capability_class.bound_names:
            raise ValueError(f"`{capability_class.qualname}` already binds {self.method_name!r}")
        add(capability_class.file_path, *self.appended(
            geometry(capability_class.file_path), capability_class.node,
            f"{decorator}{signature}\n    \"\"\"Every {capability_class.node.name} passes.\"\"\"",
        ))
        return tuple(
            geometry(file_path).nominal_edit(
                file_path=file_path,
                replacements=tuple(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=start, end_offset=end, replacement_source=text
                    )
                    for start, end, text in spans
                ),
                rationale=self.rationale
                or f"{capability_class.node.name} answers {self.method_name!r}; the guard asks it.",
            )
            for file_path, spans in replacements.items()
        )

    def subject_family(self, snapshot, answered, owner, function, guard, parents, source_path) -> frozenset[str]:
        """Every class the guarded subject can be: a declared type and its descendants."""

        index = snapshot.required_class_family_index
        first = function.args.args[0].arg if function.args.args else None
        match guard.subject:
            case ast.Name(id=name) if owner is not None and name == first:
                declared = owner.required_symbol(snapshot)
            case ast.Name(id=name):
                declared = self.annotated(
                    answered, guard, parents, source_path,
                    next(
                        (
                            argument.annotation
                            for argument in (*function.args.args, *function.args.kwonlyargs)
                            if argument.arg == name
                        ),
                        None,
                    ),
                )
            case ast.Attribute(value=ast.Name(id=name), attr=field) if owner is not None and name == first:
                declared = self.annotated(
                    answered, guard, parents, source_path,
                    next(
                        (
                            statement.annotation
                            for statement in owner.node.body
                            if isinstance(statement, ast.AnnAssign)
                            and isinstance(statement.target, ast.Name)
                            and statement.target.id == field
                        ),
                        None,
                    ),
                )
            case _:
                declared = None
        if declared is None:
            raise ValueError(
                f"`{ast.unparse(guard.subject)}` has no annotation naming a class the scan declares"
            )
        return frozenset((declared, *index.descendant_symbols(declared)))

    @staticmethod
    def annotated(answered, guard, parents, source_path, annotation: ast.expr | None) -> str | None:
        """The class an annotation names for the subject: `type[T]` for a class, `T` for an instance."""

        match annotation:
            case ast.Subscript(value=ast.Name(id="type"), slice=inner) if guard.on_class:
                annotation = inner
        match annotation:
            case ast.Subscript(value=inner):
                annotation = inner
        return None if annotation is None else answered.class_symbol(source_path, annotation, parents)

    def require_dispatch(
        self, snapshot: CodemodSourceSnapshot, subject: frozenset[str], base: str, capability: str
    ) -> None:
        compact = snapshot.compact_class_index
        authority = compact.declared_mro_authority
        for symbol in sorted(subject):
            mro = authority.resolve(symbol).mro_type
            if mro is None:
                raise ValueError(f"`{symbol}` has an open MRO")
            order = [declaration.symbol for declaration in mro.declarations]
            if base not in order and capability not in order:
                raise ValueError(f"`{symbol}` reaches neither `{base}` nor `{capability}`")
            for declaration in mro.declarations:
                if declaration.symbol in (base, capability):
                    winner = declaration.symbol
                    break
                if self.method_name in declaration.member_binding_names:
                    raise ValueError(
                        f"`{declaration.symbol}` declares {self.method_name!r} ahead of the answer for `{symbol}`"
                    )
            else:
                raise ValueError(f"`{symbol}` reaches no answer")
            expected = capability if capability in order else base
            if winner != expected:
                raise ValueError(
                    f"`{symbol}` would dispatch {self.method_name!r} to `{winner}`, not `{expected}`: "
                    "put the capability ahead of the base in its bases, or name a base below it"
                )

    @staticmethod
    def class_target(snapshot: CodemodSourceSnapshot, symbol: str) -> SourceRewriteTarget:
        declaration = snapshot.required_class_family_index.classes_by_symbol[symbol]
        return SourceRewriteTarget(file_path=declaration.file_path, qualname=declaration.qualname)

    @staticmethod
    def appended(geometry: SourceTextGeometry, node: ast.ClassDef, member: str) -> tuple[int, int, str]:
        """A member appended to a class body, at the body's indentation."""

        last = node.body[-1]
        _start, end = geometry.required_node_offsets(last)
        indent = " " * last.col_offset
        text = "".join(
            f"\n{indent}{line}" if line else "\n" for line in ("", *member.splitlines())
        )
        return end, end, text


def _docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )
