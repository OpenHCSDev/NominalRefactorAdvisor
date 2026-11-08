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


_KEY_READERS = frozenset({"len", "tuple", "set", "frozenset", "sorted", "list"})


@dataclass(frozen=True)
class _RawAnnotationRead:
    """A read of annotation values as stored: a load of `__annotations__`, or a call to
    `get_annotations` in VALUE (the default) or FORWARDREF format without
    `eval_str=True`. Those values are strings in a stringized module and objects in a
    lazy one; STRING format, `eval_str=True` and `typing.get_type_hints` read the same
    thing in both.

    A read is proved to see no difference when its dict is used only for its keys
    (iterated, tested with `in`, counted), or reaches `typing.get_type_hints`, which
    evaluates a string and returns the object a lazy annotation holds: directly, or
    through one call into a function of the same module that stores it as some
    object's `__annotations__` and asks `get_type_hints` for that object. Single
    values travel the same way when they are collected into a dict literal. Any other
    read is unproved, and named by the function holding it and its line."""

    qualname: str
    line: int

    @property
    def site(self) -> str:
        return f"{self.qualname}:{self.line}"

    @classmethod
    def unproved_in(cls, context: CodemodSelectorContext) -> tuple[_RawAnnotationRead, ...]:
        return tuple(
            sorted(
                {
                    read
                    for source_file in context.source_index.files
                    if (module := context.module_nodes_by_file_path.get(source_file.file_path))
                    is not None
                    for read in _ModuleAnnotationReads(module).unproved(source_file.module_name)
                },
                key=lambda read: (read.qualname, read.line),
            )
        )


@dataclass(frozen=True)
class _ModuleAnnotationReads:
    """The raw annotation reads of one module and the proof that each sees no
    difference between the two modes."""

    module: ast.Module

    def unproved(self, module_name: str) -> Iterator[_RawAnnotationRead]:
        yield from self._unproved_in(self.module.body, module_name, None)

    def _unproved_in(
        self,
        body: list[ast.stmt],
        scope: str,
        function: ast.FunctionDef | ast.AsyncFunctionDef | None,
    ) -> Iterator[_RawAnnotationRead]:
        for statement in body:
            if isinstance(statement, _DEFINITIONS):
                inner = (
                    function if isinstance(statement, ast.ClassDef) else statement
                )
                yield from self._unproved_in(statement.body, f"{scope}.{statement.name}", inner)
                roots: list[ast.AST] = list(statement.decorator_list)
            else:
                roots = [statement]
                yield from self._unproved_in(
                    [
                        child
                        for node in self._outside_definitions(statement)
                        for child in ast.iter_child_nodes(node)
                        if isinstance(child, _DEFINITIONS)
                    ],
                    scope,
                    function,
                )
            for root in roots:
                for node in self._outside_definitions(root):
                    if self._reads_raw(node) and not self._keys_or_evaluated(
                        node, function or self.module
                    ):
                        yield _RawAnnotationRead(scope, node.lineno)

    @staticmethod
    def _outside_definitions(root: ast.AST) -> Iterator[ast.AST]:
        pending = [root]
        while pending:
            node = pending.pop()
            yield node
            pending.extend(
                child for child in ast.iter_child_nodes(node) if not isinstance(child, _DEFINITIONS)
            )

    @staticmethod
    def _reads_raw(node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == "__annotations__" and isinstance(node.ctx, ast.Load)
        if not (isinstance(node, ast.Call) and _called_name(node) == "get_annotations"):
            return False
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        evaluates = keywords.get("eval_str")
        if isinstance(evaluates, ast.Constant) and evaluates.value is True:
            return False
        stored = keywords.get("format")
        return not (isinstance(stored, ast.Attribute) and stored.attr == "STRING")

    def _parents(self, scope: ast.AST) -> dict[ast.AST, ast.AST]:
        return {child: parent for parent in ast.walk(scope) for child in ast.iter_child_nodes(parent)}

    def _keys_or_evaluated(self, read: ast.expr, scope: ast.AST) -> bool:
        """`read` denotes an annotations dict: is every use of it keys-only or evaluated?"""

        parent = self._parents(scope).get(read)
        if isinstance(parent, ast.comprehension | ast.For) and parent.iter is read:
            return True
        if isinstance(parent, ast.Compare) and read in parent.comparators:
            return isinstance(parent.ops[parent.comparators.index(read)], ast.In | ast.NotIn)
        if isinstance(parent, ast.Call) and read in parent.args:
            return (
                _called_name(parent) in _KEY_READERS and len(parent.args) == 1
            ) or self._reaches_type_hints(parent, parent.args.index(read))
        if isinstance(parent, ast.Subscript) and parent.value is read:
            return self._evaluated_value(parent, scope)
        if (
            isinstance(parent, ast.Assign)
            and parent.value is read
            and len(parent.targets) == 1
            and isinstance(parent.targets[0], ast.Name)
        ):
            name = parent.targets[0].id
            loads = tuple(
                node
                for node in ast.walk(scope)
                if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
            )
            return bool(loads) and all(self._keys_or_evaluated(load, scope) for load in loads)
        return False

    def _evaluated_value(self, value: ast.expr, scope: ast.AST) -> bool:
        """`value` denotes one stored annotation: is it collected into a dict that is?"""

        parent = self._parents(scope).get(value)
        return (
            isinstance(parent, ast.Dict)
            and value in parent.values
            and self._keys_or_evaluated(parent, scope)
        )

    def _reaches_type_hints(self, call: ast.Call, position: int) -> bool:
        if _called_name(call) == "get_type_hints":
            return True
        callees = tuple(
            node
            for node in ast.walk(self.module)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == _called_name(call)
        )
        if len(callees) != 1:
            return False
        (callee,) = callees
        bound = (
            0
            if any(
                isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                for decorator in callee.decorator_list
            )
            else 1
            if isinstance(call.func, ast.Attribute)
            else 0
        )
        parameters = (*callee.args.posonlyargs, *callee.args.args)
        if position + bound >= len(parameters):
            return False
        name = parameters[position + bound].arg
        parents = self._parents(callee)
        hinted = {
            node.args[0].id
            for node in ast.walk(callee)
            if isinstance(node, ast.Call)
            and _called_name(node) == "get_type_hints"
            and node.args
            and isinstance(node.args[0], ast.Name)
        }
        loads = tuple(
            node
            for node in ast.walk(callee)
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
        )
        return bool(loads) and all(
            (
                isinstance(parent := parents.get(load), ast.Assign)
                and parent.value is load
                and all(
                    isinstance(target, ast.Attribute)
                    and target.attr == "__annotations__"
                    and isinstance(target.value, ast.Name)
                    and target.value.id in hinted
                    for target in parent.targets
                )
            )
            or (
                isinstance(parent, ast.Call)
                and _called_name(parent) == "get_type_hints"
                and load in parent.args
            )
            for load in loads
        )


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


@dataclass(frozen=True)
class StringizedAnnotationsReport(DataclassJsonReport):
    """What a conversion to stringized annotations proved and what it was told."""

    file_path: str
    unresolved_annotation_names: tuple[str, ...]
    unproved_annotation_reads: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class StringizeAnnotationsOperation(RefactorRecipeOperation):
    """Give a lazily annotated module `from __future__ import annotations`, so code
    moving between it and a stringized module keeps one annotation mode.

    The conversion is proved where NRA can see it: every annotation the module
    evaluates must load only names bound at module scope, builtins, or the carrying
    definition's type parameters, so evaluating its string in the module's globals
    gives the object the lazy annotation gave; and every read of stored annotations
    in the index (a string afterwards, an object before) must be proved unable to
    tell, by `_RawAnnotationRead`, or the conversion is refused with the reads
    listed."""

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
        unproved = tuple(read.site for read in _RawAnnotationRead.unproved_in(context))
        message = (
            f"{source_path!r} already stores {mode.value} annotations"
            if mode is Stringized
            else "annotations load names a string evaluated in module globals would not "
            f"find: {unresolved!r}"
            if unresolved
            else "these reads of stored annotations would see strings instead of objects, "
            f"and nothing proves they cannot tell: {unproved!r}"
            if unproved
            else "Every annotation evaluates to the same object from its string"
        )
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=(
                    CodemodPreflightStatus.FAILED
                    if mode is Stringized or unresolved or unproved
                    else CodemodPreflightStatus.PASSED
                ),
                message=message,
                detail=StringizedAnnotationsReport(
                    file_path=source_path,
                    unresolved_annotation_names=tuple(
                        f"{name}@{line}" for name, line in unresolved
                    ),
                    unproved_annotation_reads=unproved,
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
