"""Converting a module from 3.14's lazy annotations to stringized ones."""

from __future__ import annotations

import ast
import builtins
import symtable
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

from ..ast_tools import ModuleAnnotationEvaluationMode, Stringized
from ..codemod_operations import RefactorRecipeOperation
from ..codemod_payload import StringArrayPayloadValueCodec, codemod_payload_field
from ..codemod_preflight import CodemodOperationPreflightReport
from ..codemod_selection_context import CodemodSelectorContext
from ..codemod_semantics import CodemodPreflightStatus, CodemodSourceDependencyScope
from ..codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from ..json_reports import DataclassJsonReport

_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class _AnnotationNames:
    """The names a module's evaluated annotations load, each with its line, that a
    string evaluated in the module's globals would not find.

    A stringized annotation is evaluated later (by `typing.get_type_hints`) in the
    module's globals, plus the type parameters of the definition carrying it. A lazy
    one also sees an enclosing function's locals and the class body around a method.
    The two agree exactly when every name an annotation loads is bound at module
    scope, is a builtin, or is one of those type parameters. Annotations of locals
    inside a function body are never evaluated, in either mode, so they are skipped."""

    unresolved: tuple[tuple[str, int], ...]

    @classmethod
    def of(cls, module: ast.Module, source: str, file_path: str) -> _AnnotationNames:
        table = symtable.symtable(source, file_path, "exec")
        module_names = frozenset(
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_assigned() or symbol.is_imported()
        ) | frozenset(dir(builtins))
        return cls(
            tuple(
                sorted(
                    {
                        (node.id, node.lineno)
                        for annotation, type_parameters in cls._evaluated(module.body, ())
                        for node in ast.walk(annotation)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id not in module_names
                        and node.id not in type_parameters
                    }
                )
            )
        )

    @classmethod
    def _evaluated(
        cls,
        body: list[ast.stmt],
        type_parameters: tuple[str, ...],
    ) -> Iterator[tuple[ast.expr, frozenset[str]]]:
        for statement in body:
            if isinstance(statement, ast.AnnAssign):
                yield statement.annotation, frozenset(type_parameters)
            elif isinstance(statement, _DEFINITIONS):
                scope = (*type_parameters, *(param.name for param in statement.type_params))
                if isinstance(statement, ast.ClassDef):
                    yield from cls._evaluated(statement.body, scope)
                    continue
                arguments = statement.args
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    *((arguments.vararg,) if arguments.vararg else ()),
                    *((arguments.kwarg,) if arguments.kwarg else ()),
                ):
                    if argument.annotation is not None:
                        yield argument.annotation, frozenset(scope)
                if statement.returns is not None:
                    yield statement.returns, frozenset(scope)
                yield from cls._nested_definitions(statement.body, scope)
            else:
                for field in ("body", "orelse", "finalbody", "handlers"):
                    yield from cls._evaluated(
                        [
                            child
                            for child in getattr(statement, field, ())
                            if isinstance(child, ast.stmt)
                        ]
                        + [
                            stmt
                            for handler in getattr(statement, field, ())
                            if isinstance(handler, ast.ExceptHandler)
                            for stmt in handler.body
                        ],
                        type_parameters,
                    )

    @classmethod
    def _nested_definitions(
        cls,
        body: list[ast.stmt],
        type_parameters: tuple[str, ...],
    ) -> Iterator[tuple[ast.expr, frozenset[str]]]:
        """A definition inside a function body: its own annotations are evaluated,
        the annotations of the function's locals are not."""

        for statement in body:
            for node in ast.walk(statement):
                if isinstance(node, _DEFINITIONS):
                    yield from cls._evaluated([node], type_parameters)


@dataclass(frozen=True)
class _RawAnnotationReader:
    """A site that reads annotation values as stored: `__annotations__`, or
    `get_annotations` in VALUE (the default) or FORWARDREF format. Those values are
    strings in a stringized module and objects in a lazy one. STRING format and
    `typing.get_type_hints` read the same thing in both modes."""

    file_path: str
    line: int

    @property
    def site(self) -> str:
        return f"{self.file_path}:{self.line}"

    @classmethod
    def all_in(cls, context: CodemodSelectorContext) -> tuple[_RawAnnotationReader, ...]:
        return tuple(
            cls(source_file.file_path, node.lineno)
            for source_file in context.source_index.files
            if (module := context.module_nodes_by_file_path.get(source_file.file_path))
            is not None
            for node in ast.walk(module)
            if cls._reads_raw(node)
        )

    @staticmethod
    def _reads_raw(node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == "__annotations__"
        if not isinstance(node, ast.Call):
            return False
        name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else None
        )
        if name != "get_annotations":
            return False
        formats = tuple(keyword.value for keyword in node.keywords if keyword.arg == "format")
        return not any(
            isinstance(value, ast.Attribute) and value.attr == "STRING" for value in formats
        )


@dataclass(frozen=True)
class StringizedAnnotationsReport(DataclassJsonReport):
    """What a conversion to stringized annotations proved and what it was told."""

    file_path: str
    unresolved_annotation_names: tuple[str, ...]
    unreviewed_annotation_readers: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class StringizeAnnotationsOperation(RefactorRecipeOperation):
    """Give a lazily annotated module `from __future__ import annotations`, so code
    moving between it and a stringized module keeps one annotation mode.

    The conversion is proved where NRA can see it: every annotation the module
    evaluates must load only names bound at module scope, builtins, or the carrying
    definition's type parameters, so evaluating its string in the module's globals
    gives the object the lazy annotation gave. What a raw reader does with a stored
    annotation (a string afterwards, an object before) NRA cannot see, so every such
    reader in the index must be named in `reviewed_annotation_readers` as
    `path:line`, or the conversion is refused with the sites listed."""

    reviewed_annotation_readers: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec(),
        default=(),
    )
    source_dependency_scope: ClassVar[CodemodSourceDependencyScope] = (
        CodemodSourceDependencyScope.REPOSITORY
    )

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        source_path = self.required_source_path(context, self.operation_key())
        module = context.module_nodes_by_file_path[source_path]
        mode = ModuleAnnotationEvaluationMode.from_module(module)
        unresolved = _AnnotationNames.of(
            module, context.sources_by_file_path[source_path], source_path
        ).unresolved
        reviewed = frozenset(self.reviewed_annotation_readers)
        unreviewed = tuple(
            reader.site
            for reader in _RawAnnotationReader.all_in(context)
            if reader.site not in reviewed
        )
        message = (
            f"{source_path!r} already stores {mode.value} annotations"
            if mode is Stringized
            else "annotations load names a string evaluated in module globals would not "
            f"find: {unresolved!r}"
            if unresolved
            else "raw annotation readers would see strings instead of objects; review "
            f"each and name it in reviewed_annotation_readers: {unreviewed!r}"
            if unreviewed
            else "Every annotation evaluates to the same object from its string"
        )
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=(
                    CodemodPreflightStatus.FAILED
                    if mode is Stringized or unresolved or unreviewed
                    else CodemodPreflightStatus.PASSED
                ),
                message=message,
                detail=StringizedAnnotationsReport(
                    file_path=source_path,
                    unresolved_annotation_names=tuple(
                        f"{name}@{line}" for name, line in unresolved
                    ),
                    unreviewed_annotation_readers=unreviewed,
                ),
            ),
        )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        source_path = self.required_source_path(context, self.operation_key())
        module = context.module_nodes_by_file_path[source_path]
        source = context.sources_by_file_path[source_path]
        geometry = SourceTextGeometry(source)
        first = module.body[0]
        is_docstring = (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        )
        start, end = geometry.required_node_offsets(first)
        prelude = Stringized.new_module_prelude().rstrip("\n")
        written = source[start:end]
        replacement = f"{written}\n\n{prelude}" if is_docstring else f"{prelude}\n\n{written}"
        return (
            geometry.nominal_edit(
                file_path=source_path,
                replacements=(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=start,
                        end_offset=end,
                        replacement_source=replacement,
                    ),
                ),
                rationale=self.rationale
                or f"Store {source_path!r}'s annotations as strings.",
            ),
        )
