"""Converting a module from 3.14's lazy annotations to stringized ones."""

from __future__ import annotations

import ast
import builtins
import symtable
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Self, cast

from ..ast_projection import ExpressionReference
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
from ..declared_family import ForeignAdapter
from ..json_reports import DataclassJsonReport

_KEY_READERS = frozenset({"len", "tuple", "set", "frozenset", "sorted", "list"})
_Annotations = Iterator[tuple[ast.expr, frozenset[str]]]


def _outside_definitions(root: ast.AST) -> Iterator[ast.AST]:
    """`root` and every node under it, without descending into a definition (a
    definition is yielded; its body and header belong to its own handling)."""

    pending = [root]
    while pending:
        node = pending.pop()
        yield node
        if not _Statement.decoded(node).opens_scope:
            pending.extend(ast.iter_child_nodes(node))


def _each(body: list[ast.stmt]) -> Iterator[ast.AST]:
    for statement in body:
        yield from _outside_definitions(statement)


class _Statement(ForeignAdapter):
    """A statement, read for the annotations it evaluates and the scopes its
    stored-annotation reads belong to. Anything that is not an annotated assignment
    or a definition carries neither (its nested ones are reached by the walk)."""

    __registry_key__ = "adapts"
    opens_scope: ClassVar[bool] = False
    is_function: ClassVar[bool] = False

    @classmethod
    def decoded(cls, node: object) -> Self:
        return cast(Self, super().decoded(node))

    def annotations(self, type_parameters: frozenset[str]) -> _Annotations:
        """Annotations it evaluates at module or class level."""

        return iter(())

    def annotations_in_function(self, type_parameters: frozenset[str]) -> _Annotations:
        """Annotations it evaluates inside a function body, where a local's
        annotation is never evaluated in either mode."""

        return iter(())

    def unproved_reads(
        self, reads: _ModuleAnnotationReads, scope: str, function: ast.AST | None
    ) -> Iterator[_RawAnnotationRead]:
        """Unproved reads in the scope this statement opens."""

        return iter(())


class _OtherStatement(_Statement):
    adapts = object


class _AnnotatedAssignment(_Statement):
    adapts = ast.AnnAssign

    def annotations(self, type_parameters: frozenset[str]) -> _Annotations:
        yield self.node.annotation, type_parameters


class _Definition(_Statement):
    opens_scope = True

    def scope_parameters(self, type_parameters: frozenset[str]) -> frozenset[str]:
        return type_parameters | {parameter.name for parameter in self.node.type_params}

    def annotations_in_function(self, type_parameters: frozenset[str]) -> _Annotations:
        return self.annotations(type_parameters)

    def unproved_reads(
        self, reads: _ModuleAnnotationReads, scope: str, function: ast.AST | None
    ) -> Iterator[_RawAnnotationRead]:
        yield from reads.unproved(
            self.node.body, f"{scope}.{self.node.name}", self.reading_function(function)
        )
        yield from reads.unproved_at(
            (child for child in ast.iter_child_nodes(self.node) if child not in self.node.body),
            scope,
            function,
        )

    def reading_function(self, function: ast.AST | None) -> ast.AST | None:
        return self.node


class _ClassDefinition(_Definition):
    adapts = ast.ClassDef

    def annotations(self, type_parameters: frozenset[str]) -> _Annotations:
        parameters = self.scope_parameters(type_parameters)
        for node in _each(self.node.body):
            yield from _Statement.decoded(node).annotations(parameters)

    def reading_function(self, function: ast.AST | None) -> ast.AST | None:
        return function


class _FunctionDefinition(_Definition):
    adapts = ast.FunctionDef
    is_function = True

    def annotations(self, type_parameters: frozenset[str]) -> _Annotations:
        parameters = self.scope_parameters(type_parameters)
        arguments = self.node.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *((arguments.vararg,) if arguments.vararg else ()),
            *((arguments.kwarg,) if arguments.kwarg else ()),
        ):
            if argument.annotation is not None:
                yield argument.annotation, parameters
        if self.node.returns is not None:
            yield self.node.returns, parameters
        for node in _each(self.node.body):
            yield from _Statement.decoded(node).annotations_in_function(parameters)


class _AsyncFunctionDefinition(_FunctionDefinition):
    adapts = ast.AsyncFunctionDef


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
                        for statement in _each(module.body)
                        for annotation, type_parameters in _Statement.decoded(
                            statement
                        ).annotations(frozenset())
                        for node in ast.walk(annotation)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id not in module_names
                        and node.id not in type_parameters
                    }
                )
            )
        )


class _StoredAnnotationsRead(ForeignAdapter):
    """A node that may read annotation values as stored: a load of `__annotations__`,
    or a call to `get_annotations` in VALUE (the default) or FORWARDREF format without
    `eval_str=True`. Those values are strings in a stringized module and objects in a
    lazy one; STRING format, `eval_str=True` and `typing.get_type_hints` read the same
    thing in both."""

    __registry_key__ = "adapts"

    @classmethod
    def decoded(cls, node: object) -> Self:
        return cast(Self, super().decoded(node))

    @property
    def is_raw(self) -> bool:
        return False


class _OtherNode(_StoredAnnotationsRead):
    adapts = object


class _AnnotationsAttribute(_StoredAnnotationsRead):
    adapts = ast.Attribute

    @property
    def is_raw(self) -> bool:
        return self.node.attr == "__annotations__" and isinstance(self.node.ctx, ast.Load)


class _AnnotationsCall(_StoredAnnotationsRead):
    adapts = ast.Call

    @property
    def is_raw(self) -> bool:
        if ExpressionReference.decoded(self.node.func).terminal_name != "get_annotations":
            return False
        keywords = {keyword.arg: keyword.value for keyword in self.node.keywords}
        evaluates = keywords.get("eval_str")
        if isinstance(evaluates, ast.Constant) and evaluates.value is True:
            return False
        return ExpressionReference.decoded(keywords.get("format")).terminal_name != "STRING"


@dataclass(frozen=True)
class _RawAnnotationRead:
    """A raw read of stored annotations that nothing proves unable to tell strings
    from objects, named by the function holding it and its line."""

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
                    for read in _ModuleAnnotationReads(module).unproved(
                        module.body, source_file.module_name, None
                    )
                },
                key=lambda read: (read.qualname, read.line),
            )
        )


@dataclass(frozen=True)
class _ModuleAnnotationReads:
    """One module's raw annotation reads, each proved or reported.

    A read is proved to see no difference when its dict is used only for its keys
    (iterated, tested with `in`, counted), or reaches `typing.get_type_hints`, which
    evaluates a string and returns the object a lazy annotation holds: directly, or
    through one call into a function of the same module that stores it as some
    object's `__annotations__` and asks `get_type_hints` for that object. Single
    values travel the same way when they are collected into a dict literal."""

    module: ast.Module
    _scopes: dict[int, _ReadScope] = field(default_factory=dict, compare=False)

    def scope(self, root: ast.AST) -> _ReadScope:
        """`root`'s parent map, built once however many reads it holds."""

        found = self._scopes.get(id(root))
        if found is None:
            found = _ReadScope(
                self,
                {child: parent for parent in ast.walk(root) for child in ast.iter_child_nodes(parent)},
                root,
            )
            self._scopes[id(root)] = found
        return found

    def unproved(
        self, body: list[ast.stmt], scope: str, function: ast.AST | None
    ) -> Iterator[_RawAnnotationRead]:
        for node in _each(body):
            yield from self._unproved_read(node, scope, function)
            yield from _Statement.decoded(node).unproved_reads(self, scope, function)

    def unproved_at(
        self, roots: Iterable[ast.AST], scope: str, function: ast.AST | None
    ) -> Iterator[_RawAnnotationRead]:
        for root in roots:
            for node in _outside_definitions(root):
                yield from self._unproved_read(node, scope, function)

    def _unproved_read(
        self, node: ast.AST, scope: str, function: ast.AST | None
    ) -> Iterator[_RawAnnotationRead]:
        if _StoredAnnotationsRead.decoded(node).is_raw and not self.scope(
            function or self.module
        ).dict_used(node):
            yield _RawAnnotationRead(scope, node.lineno)

    def type_hint_evaluator(self, call: ast.Call, position: int) -> _TypeHintEvaluator | None:
        """The one same-module function the call reaches, at the parameter the
        argument binds, or nothing when the callee is ambiguous or absent."""

        name = ExpressionReference.decoded(call.func).terminal_name
        callees = tuple(
            definition
            for definition in ast.walk(self.module)
            if _Statement.decoded(definition).is_function and definition.name == name
        )
        if len(callees) != 1:
            return None
        (callee,) = callees
        static = any(
            ExpressionReference.decoded(decorator).terminal_name == "staticmethod"
            for decorator in callee.decorator_list
        )
        bound = 0 if static or ExpressionReference.decoded(call.func).identifier else 1
        parameters = (*callee.args.posonlyargs, *callee.args.args)
        if position + bound >= len(parameters):
            return None
        return _TypeHintEvaluator(self.scope(callee), parameters[position + bound].arg)


@dataclass(frozen=True)
class _ReadScope:
    """One scope a read lives in, with its parent map."""

    reads: _ModuleAnnotationReads
    parents: Mapping[ast.AST, ast.AST]
    root: ast.AST

    def uses(self, node: ast.AST) -> AnnotationsUse:
        return AnnotationsUse.decoded(self.parents.get(node))

    def dict_used(self, read: ast.AST) -> bool:
        return self.uses(read).of_dict(read, self)

    def value_used(self, value: ast.AST) -> bool:
        return self.uses(value).of_value(value, self)

    def loads(self, name: str) -> tuple[ast.Name, ...]:
        return tuple(
            node
            for node in ast.walk(self.root)
            if ExpressionReference.decoded(node).identifier == name
            and isinstance(node.ctx, ast.Load)
        )


@dataclass(frozen=True)
class _TypeHintEvaluator:
    """A callee parameter that must reach `get_type_hints`: stored as some object's
    `__annotations__` that `get_type_hints` is asked for, or passed to it."""

    scope: _ReadScope
    parameter: str

    @property
    def hinted(self) -> frozenset[str]:
        return frozenset(
            name
            for node in ast.walk(self.scope.root)
            if isinstance(node, ast.Call)
            and ExpressionReference.decoded(node.func).terminal_name == "get_type_hints"
            and node.args
            and (name := ExpressionReference.decoded(node.args[0]).identifier) is not None
        )

    def evaluates(self) -> bool:
        loads = self.scope.loads(self.parameter)
        hinted = self.hinted
        return bool(loads) and all(
            self.scope.uses(load).reaches_type_hints(load, hinted) for load in loads
        )


class AnnotationsUse(ForeignAdapter):
    """How a parent node uses the annotations dict (or one value of it) below it."""

    __registry_key__ = "adapts"

    @classmethod
    def decoded(cls, node: object) -> Self:
        return cast(Self, super().decoded(node))

    def of_dict(self, read: ast.AST, scope: _ReadScope) -> bool:
        """Is this use of the whole dict keys-only or evaluated?"""

        return False

    def of_value(self, value: ast.AST, scope: _ReadScope) -> bool:
        """Is this use of one stored value collected into a dict that is?"""

        return False

    def reaches_type_hints(self, load: ast.AST, hinted: frozenset[str]) -> bool:
        """In a type-hint evaluator, does this use hand the parameter to
        `get_type_hints`?"""

        return False


class _UnprovedUse(AnnotationsUse):
    adapts = object


class _IteratedUse(AnnotationsUse):
    def of_dict(self, read: ast.AST, scope: _ReadScope) -> bool:
        return self.node.iter is read


class _ComprehensionUse(_IteratedUse):
    adapts = ast.comprehension


class _ForUse(_IteratedUse):
    adapts = ast.For


class _ComparedUse(AnnotationsUse):
    adapts = ast.Compare

    def of_dict(self, read: ast.AST, scope: _ReadScope) -> bool:
        return read in self.node.comparators and isinstance(
            self.node.ops[self.node.comparators.index(read)], ast.In | ast.NotIn
        )


class _PassedUse(AnnotationsUse):
    adapts = ast.Call

    @property
    def callee_name(self) -> str | None:
        return ExpressionReference.decoded(self.node.func).terminal_name

    def of_dict(self, read: ast.AST, scope: _ReadScope) -> bool:
        if read not in self.node.args:
            return False
        if self.callee_name in _KEY_READERS and len(self.node.args) == 1:
            return True
        if self.callee_name == "get_type_hints":
            return True
        evaluator = scope.reads.type_hint_evaluator(self.node, self.node.args.index(read))
        return evaluator is not None and evaluator.evaluates()

    def reaches_type_hints(self, load: ast.AST, hinted: frozenset[str]) -> bool:
        return self.callee_name == "get_type_hints" and load in self.node.args


class _SubscriptedUse(AnnotationsUse):
    adapts = ast.Subscript

    def of_dict(self, read: ast.AST, scope: _ReadScope) -> bool:
        return self.node.value is read and scope.value_used(self.node)


class _BoundUse(AnnotationsUse):
    adapts = ast.Assign

    def of_dict(self, read: ast.AST, scope: _ReadScope) -> bool:
        match self.node.targets:
            case (target,) if self.node.value is read:
                name = ExpressionReference.decoded(target).identifier
            case _:
                return False
        loads = () if name is None else scope.loads(name)
        return bool(loads) and all(scope.dict_used(load) for load in loads)

    def reaches_type_hints(self, load: ast.AST, hinted: frozenset[str]) -> bool:
        return self.node.value is load and all(
            (chain := ExpressionReference.decoded(target).attribute_chain) is not None
            and len(chain) == 2
            and chain[1] == "__annotations__"
            and chain[0] in hinted
            for target in self.node.targets
        )


class _CollectedUse(AnnotationsUse):
    adapts = ast.Dict

    def of_value(self, value: ast.AST, scope: _ReadScope) -> bool:
        return value in self.node.values and scope.dict_used(self.node)


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
