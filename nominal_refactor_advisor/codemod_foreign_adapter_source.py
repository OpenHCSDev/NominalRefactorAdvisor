"""Source edits that fold bind chains into one declared adapter family."""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from copy import deepcopy
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Self

from .ast_tools import BuiltinCallName, ParsedModule
from .cases_family import body_without_docstring
from .codemod_foreign_family_source import (
    FunctionNode,
    FunctionRewriteAuthority,
    ForeignFamilyReference,
)
from .codemod_source_edits import (
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .declared_family import ForeignAdapter
from .foreign_family import AssignmentStatement, FunctionDefinition
from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from .observation_families import ShapeFamily, TypeArgumentResolver
from .observation_shapes import ForeignFamilySiteModule, ForeignFamilySiteShape


def qualified_name(foreign: type) -> str:
    return f"{foreign.__module__}.{foreign.__qualname__}"


@dataclass(frozen=True)
class PropertyBody:
    """The single returned expression of a property, and the name it calls `self`."""

    expression: ast.expr
    receiver: str


@cache
def property_body(adapter: type[ForeignAdapter], name: str) -> PropertyBody | None:
    """The body `adapter` runs for property `name`, inherited when not overridden."""

    declaring = next(
        (vars(klass)[name] for klass in adapter.__mro__ if name in vars(klass)), None
    )
    match declaring:
        case property(fget=function) if function is not None:
            declaration = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        case _:
            return None
    match (declaration, body_without_docstring(declaration.body)):
        case (
            ast.FunctionDef(args=ast.arguments(args=[receiver, *_])),
            [ast.Return(value=ast.expr() as expression)],
        ):
            return PropertyBody(expression, receiver.arg)
    return None


@dataclass(frozen=True)
class SubjectAsNode(ast.NodeTransformer):
    """Reads of the chain's subject spelled as the adapter's `receiver.node`."""

    subject: str
    receiver: str

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id != self.subject:
            return node
        return ast.Attribute(
            value=ast.Name(id=self.receiver, ctx=ast.Load()),
            attr="node",
            ctx=ast.Load(),
        )


@dataclass(frozen=True)
class AdapterPropertyBodies:
    """What the properties of one adapter family compute, compared as expressions."""

    family: type[ForeignAdapter]

    @cached_property
    def property_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    name
                    for adapter in self.family.adapters()
                    for klass in adapter.__mro__[
                        : adapter.__mro__.index(self.family) + 1
                    ]
                    for name, declared in vars(klass).items()
                    if isinstance(declared, property)
                }
            )
        )

    @cached_property
    def adapters_by_name(self) -> dict[str, type[ForeignAdapter]]:
        return {qualified_name(adapter.adapts): adapter for adapter in self.family.adapters()}

    def implements(
        self, class_name: str, name: str, subject: str, expression: ast.expr
    ) -> bool:
        """Whether property `name` computes `expression` for the class `class_name`.

        The one comparison of a chain's arm expression with an adapter
        property: both are dumped without positions, the subject read as
        `receiver.node`.
        """

        body = property_body(self.adapters_by_name[class_name], name)
        if body is None:
            return False
        renamed = SubjectAsNode(subject, body.receiver).visit(deepcopy(expression))
        return ast.dump(renamed) == ast.dump(body.expression)


@dataclass(frozen=True)
class BindArm:
    """One arm: the classes it tests, what it binds, and any extra conjuncts."""

    classes: tuple[str, ...]
    bindings: tuple[tuple[str, ast.expr], ...]
    residuals: tuple[str, ...]

    @classmethod
    def parsed(
        cls,
        test: ast.expr,
        body: list[ast.stmt],
        resolver: TypeArgumentResolver,
    ) -> tuple[str, Self] | None:
        """(subject, arm) when the arm tests one subject and only binds names."""

        match test:
            case ast.BoolOp(op=ast.And(), values=[ast.Call() as call, *rest]):
                residuals = tuple(ast.dump(item) for item in rest)
            case ast.Call():
                call, residuals = test, ()
            case _:
                return None
        match BuiltinCallName.isinstance_arguments(call):
            case (ast.Name(id=subject), _):
                pass
            case _:
                return None
        classes = resolver.qualified_names_of_test(call)
        bindings = tuple(cls.binding(statement) for statement in body)
        if classes is None or None in bindings:
            return None
        return subject, cls(classes, bindings, residuals)

    @staticmethod
    def binding(statement: ast.stmt) -> tuple[str, ast.expr] | None:
        assignment = AssignmentStatement.decoded(statement)
        if (
            assignment is None
            or assignment.annotation is not None
            or assignment.single_name is None
        ):
            return None
        return assignment.single_name, assignment.value

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.bindings)

    def expression(self, name: str) -> ast.expr:
        return dict(self.bindings)[name]


@dataclass(frozen=True)
class BindChain:
    """`if isinstance(S, A): binds elif isinstance(S, B): binds [else: guard]`."""

    statement: ast.If
    subject: str
    arms: tuple[BindArm, ...]
    guard: ast.stmt | None

    @classmethod
    def in_scope(
        cls, function: FunctionNode, resolver: TypeArgumentResolver
    ) -> tuple[Self, ...]:
        """Every chain headed by a statement of `function`; `elif` tails are no heads."""

        branches = tuple(
            node for node in ast.walk(function) if isinstance(node, ast.If)
        )
        tails = {
            id(tail)
            for branch in branches
            for tail in branch.orelse
            if tail.col_offset == branch.col_offset and len(branch.orelse) == 1
        }
        return tuple(
            chain
            for branch in branches
            if id(branch) not in tails
            for chain in (cls.headed_by(branch, resolver),)
            if chain is not None
        )

    @classmethod
    def headed_by(
        cls, head: ast.If, resolver: TypeArgumentResolver
    ) -> Self | None:
        arms: list[BindArm] = []
        subjects: set[str] = set()
        node = head
        while True:
            parsed = BindArm.parsed(node.test, node.body, resolver)
            if parsed is None:
                return None
            subjects.add(parsed[0])
            arms.append(parsed[1])
            match node.orelse:
                case [ast.If() as tail] if tail.col_offset == node.col_offset:
                    node = tail
                    continue
                case []:
                    guard = None
                case [ast.Pass()]:
                    guard = None
                case [ast.Continue() | ast.Return(value=None | ast.Constant()) as other]:
                    guard = other
                case _:
                    return None
            break
        if len(subjects) != 1 or len(arms) < 2:
            return None
        chain = cls(head, subjects.pop(), tuple(arms), guard)
        return chain if chain.binds_one_set_of_names else None

    @cached_property
    def binds_one_set_of_names(self) -> bool:
        first = self.arms[0].names
        return (
            len(set(first)) == len(first)
            and self.subject not in first
            and all(set(arm.names) == set(first) and len(arm.names) == len(first)
                    for arm in self.arms)
        )

    @property
    def names(self) -> tuple[str, ...]:
        return self.arms[0].names

    @property
    def conjunct_free(self) -> bool:
        return not any(arm.residuals for arm in self.arms)



@dataclass(frozen=True)
class AdapterFold:
    """A bind chain proved to be a property read of one declared adapter family."""

    chain: BindChain
    family: type[ForeignAdapter]
    properties: tuple[tuple[str, str], ...]

    @classmethod
    def proving(cls, chain: BindChain, family: type[ForeignAdapter]) -> Self | None:
        """The fold when the arms test exactly the family's classes and each bound
        name is, for every tested class, one property's body."""

        tested = tuple(name for arm in chain.arms for name in arm.classes)
        adapted = family.member_names()
        if set(tested) != adapted or len(tested) != len(adapted):
            return None
        bodies = AdapterPropertyBodies(family)
        if any(
            first is not second and first in second.__mro__
            for first in family.__registry__
            for second in family.__registry__
        ):
            return None
        properties = []
        for name in chain.names:
            matching = [
                candidate
                for candidate in bodies.property_names
                if all(
                    bodies.implements(
                        class_name,
                        candidate,
                        chain.subject,
                        arm.expression(name),
                    )
                    for arm in chain.arms
                    for class_name in arm.classes
                )
            ]
            if len(matching) != 1:
                return None
            properties.append((name, matching[0]))
        return cls(chain, family, tuple(properties))


@dataclass(frozen=True)
class FreshNames:
    """Names no binding or read of the function or its module already uses."""

    taken: frozenset[str]

    @classmethod
    def around(cls, function: FunctionNode, parsed_module: ParsedModule) -> Self:
        read = {
            node.id for node in ast.walk(function) if isinstance(node, ast.Name)
        }
        return cls(
            frozenset(
                read
                | LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(function.body)
                | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(function)
                | LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                    parsed_module.module.body
                )
            )
        )

    def allocated(self, family_name: str) -> tuple[str, Self]:
        base = re.sub(r"(?<!^)(?=[A-Z])", "_", family_name).lower()
        name = next(
            candidate
            for index in range(1, len(self.taken) + 2)
            for candidate in (base if index == 1 else f"{base}_{index}",)
            if candidate not in self.taken
        )
        return name, FreshNames(self.taken | {name})


@dataclass(frozen=True)
class AdapterBindChainAuthority(FunctionRewriteAuthority):
    """Every foldable bind chain of one function, as source replacements."""

    reference: ForeignFamilyReference
    rewrite_name = "Adapter folding"

    @cached_property
    def folds(self) -> tuple[AdapterFold, ...]:
        resolver = TypeArgumentResolver(self.parsed_module)
        folds = (
            AdapterFold.proving(chain, self.reference.family)
            for chain in BindChain.in_scope(self.function, resolver)
            if chain.conjunct_free
        )
        return tuple(fold for fold in folds if fold is not None)

    def replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        if not self.folds:
            raise ValueError(
                f"No bind chain in {self.function.name!r} reads "
                f"{self.reference.import_path} properties"
            )
        names = FreshNames.around(self.function, self.parsed_module)
        replacements = []
        for fold in self.folds:
            fresh, names = names.allocated(self.reference.class_name)
            replacements.append(self.replacement(fold, fresh))
        return tuple(replacements)

    def replacement(self, fold: AdapterFold, fresh: str) -> SourceTextSpanReplacement:
        chain = fold.chain
        statement = chain.statement
        start, end = self.geometry.line_span_offsets(
            statement.lineno, statement.end_lineno or statement.lineno
        )
        if self.geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise ValueError("Bind chain rewrite would discard a comment")
        line = self.geometry.lines[statement.lineno - 1]
        indentation = line[: statement.col_offset]
        if indentation.strip():
            raise ValueError("Bind chain does not begin its own line")
        step = " " * (statement.body[0].col_offset - statement.col_offset)
        reads = [f"{name} = {fresh}.{prop}" for name, prop in fold.properties]
        decode = f"{fresh} = {self.reference.class_name}.decoded({chain.subject})"
        match chain.guard:
            case None:
                lines = [decode, f"if {fresh} is not None:", *(step + r for r in reads)]
            case _:
                lines = [
                    decode,
                    f"if {fresh} is None:",
                    step + ast.unparse(chain.guard),
                    *reads,
                ]
        text = "\n".join(indentation + item for item in lines)
        if not self.fit.fits("", text, ""):
            raise ValueError("Bind chain rewrite would exceed the line length")
        trailer = "\n" if self.geometry.source[start:end].endswith("\n") else ""
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start, end_offset=end, replacement_source=text + trailer
        )


class ForeignAdapterBindChainModuleFamily(ShapeFamily[ForeignFamilySiteModule]):
    """Bind chains that are property reads of a declared adapter family."""

    item_type = ForeignFamilySiteModule

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ForeignFamilySiteModule]:
        index = parsed_module.syntax_index
        node_indices = {
            statement: node_index
            for node_index, statement in index.indexed_nodes_of_type(ast.If)
        }
        resolver = TypeArgumentResolver(parsed_module)
        shapes = []
        for function in FunctionDefinition.within(ast.walk(parsed_module.module)):
            for chain in BindChain.in_scope(function, resolver):
                if not chain.conjunct_free:
                    continue
                for family in ForeignAdapter.concrete_families():
                    if AdapterFold.proving(chain, family) is not None:
                        shapes.append(
                            ForeignFamilySiteShape(
                                file_path=parsed_module.file_path,
                                lineno=chain.statement.lineno,
                                symbol=index.lexical_symbol(
                                    node_indices[chain.statement]
                                ),
                                family=family.import_path(),
                            )
                        )
        return [
            ForeignFamilySiteModule(
                module_name=parsed_module.module_name,
                sites=tuple(sorted(set(shapes), key=lambda shape: shape.lineno)),
            )
        ]
