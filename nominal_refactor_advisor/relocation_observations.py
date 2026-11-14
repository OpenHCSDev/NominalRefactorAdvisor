"""Relocation shapes: where an absence or kind probe reappears under another spelling.

Removing an `x is None` probe or a kind comparison does not remove the branch
when the fact moves somewhere a probe detector does not look: a family member
that answers which member it is, a property that derives a fixed answer from
the object's own state, a method that takes both branches as callables, a
0/1 tuple, or an `X | None` parameter.
These families collect each spelling so the count cannot drop while the branch
stays.
"""

from __future__ import annotations

import ast
import builtins
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar, Self

from nominal_refactor_advisor.annotation_semantics import TypeWrapper
from nominal_refactor_advisor.ast_field_decoders import variadic_parameters
from nominal_refactor_advisor.ast_projection import ExpressionReference
from nominal_refactor_advisor.ast_tools import ParsedModule, ParsedModuleFact
from nominal_refactor_advisor.class_index import (
    ABSTRACT_METHOD_DECORATOR,
    PROPERTY_DECORATORS,
    ModuleNominalBindingAuthority,
    ModuleNominalBindingSnapshot,
)
from nominal_refactor_advisor.class_symbols import ClassSymbolResolutionAuthority
from nominal_refactor_advisor.foreign_family import FunctionDefinition
from nominal_refactor_advisor.observation_families import ShapeFamily
from nominal_refactor_advisor.probe_owner import (
    ProbeOwner,
    ProbeOwnerResolver,
    UnresolvedProbeOwner,
)
from nominal_refactor_advisor.restated_fact_observations import (
    RestatingSiteFamily,
    RestatingSiteModule,
)
from nominal_refactor_advisor.observation_shapes import RestatingSite, SymbolSite

def scope_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """A statement's nodes in source order, stopping at each def, lambda and class it nests."""

    _scope_boundaries = (*FunctionDefinition.members, ast.ClassDef, ast.Lambda)
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        if not isinstance(current, _scope_boundaries):
            stack.extend(reversed(tuple(ast.iter_child_nodes(current))))


@dataclass(frozen=True)
class ScopedDefinition:
    """A def under its dotted name, with its own body read once."""

    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef

    @cached_property
    def nodes(self) -> tuple[ast.AST, ...]:
        """The nodes of the body, not of the functions, lambdas or classes it nests."""

        return tuple(
            node for statement in self.node.body for node in scope_nodes(statement)
        )

    @cached_property
    def absence_tuple_arities(self) -> frozenset[int]:
        """The arities the def returns when every return is `()` or `(x,)`
        (absence spelled as a collection); empty when any return is something else."""

        arities: set[int] = set()
        for child in self.nodes:
            match child:
                case ast.Return(value=ast.Tuple(elts=elements)) if len(
                    elements
                ) <= 1 and not any(isinstance(element, ast.Starred) for element in elements):
                    arities.add(len(elements))
                case ast.Return():
                    return frozenset()
        return frozenset(arities)


@dataclass(frozen=True)
class ModuleDefinitions(ParsedModuleFact):
    """Every def of one parse, in source order."""

    definitions: tuple[ScopedDefinition, ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        return cls(
            tuple(
                ScopedDefinition(qualname, node)
                for qualname, node in parsed_module.syntax_index.declarations(
                    FunctionDefinition.members
                )
            )
        )


def statements_after_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    match body:
        case [ast.Expr(value=ast.Constant(value=str())), *rest]:
            return rest
    return body


def takes_only_receiver(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    arguments = node.args
    return (
        len([*arguments.posonlyargs, *arguments.args]) == 1
        and not variadic_parameters(arguments)
        and not arguments.kwonlyargs
    )


def declares_abstract(
    node: ast.FunctionDef | ast.AsyncFunctionDef, bindings: ModuleNominalBindingSnapshot
) -> bool:
    """An abstract declaration asks the question; it answers nothing."""

    if bindings.decorated_by(node, (ABSTRACT_METHOD_DECORATOR,)):
        return True
    match statements_after_docstring(node.body):
        case [] | [ast.Pass()] | [ast.Expr(value=ast.Constant(value=builtins.Ellipsis))]:
            return True
        case [
            ast.Raise(
                exc=ast.Name(id="NotImplementedError")
                | ast.Call(func=ast.Name(id="NotImplementedError"))
            )
        ]:
            return True
    return False


def is_frozen_record(
    node: ast.ClassDef, bindings: ModuleNominalBindingSnapshot
) -> bool:
    """A frozen dataclass or a NamedTuple: its state is fixed once built.

    The decorator and the base are resolved through the module's bindings, so a
    local class named `dataclass` or `NamedTuple` is not one.
    """

    def resolves_to(reference: ast.expr, qualified_name: str) -> bool:
        parts = ExpressionReference.decoded(
            ClassSymbolResolutionAuthority.reference_node(reference)
        ).attribute_chain
        return parts is not None and bindings.reference_for(parts).has_bound_name(qualified_name)

    if any(resolves_to(base, "typing.NamedTuple") for base in node.bases):
        return True
    return any(
        isinstance(decorator, ast.Call)
        and resolves_to(decorator.func, "dataclasses.dataclass")
        and any(
            keyword.arg == "frozen"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in decorator.keywords
        )
        for decorator in node.decorator_list
    )


# ---- the class shapes every family join reads


@dataclass(frozen=True)
class MemberAnswer(ABC):
    """A member declaration read by how it answers."""

    member: str
    lineno: int

    @property
    @abstractmethod
    def detail(self) -> tuple[str, ...]:
        """What the answer is, as a report of the member spells it."""

        raise NotImplementedError


@dataclass(frozen=True)
class ConstantAnswer(MemberAnswer):
    """A member whose declaration answers with one constant."""

    value: str

    @property
    def detail(self) -> tuple[str, ...]:
        return (self.value,)


@dataclass(frozen=True)
class FoldDeclaration(MemberAnswer):
    """A method that returns a call of exactly one of its callable parameters."""

    called_parameter: str

    @property
    def detail(self) -> tuple[str, ...]:
        return (self.called_parameter,)


@dataclass(frozen=True)
class OwnStateAnswer(MemberAnswer):
    """A `-> bool` property or zero-argument method of a frozen record whose
    answer is a state of the record: its one return probes an own field for
    absence or a kind, and calls nothing but pure builtins."""

    accessed: str
    probe: str

    @property
    def detail(self) -> tuple[str, ...]:
        return (self.accessed, self.probe)


@dataclass
class ClassBodyReading:
    """One pass over a class body sorting each member declaration by how it answers.

    A constant answer is a zero-argument method or property returning one bool
    or str constant, a class attribute assigned a bool constant, or a `ClassVar`
    assigned a bool or str constant. Abstract declarations and bare annotations
    answer nothing; every other declaration is computed.
    """

    qualname: str
    node: ast.ClassDef
    bindings: ModuleNominalBindingSnapshot
    answers: dict[type[MemberAnswer], list[MemberAnswer]] = field(
        default_factory=lambda: defaultdict(list)
    )
    computed: set[str] = field(default_factory=set)
    declared: set[str] = field(default_factory=set)

    @classmethod
    def of(
        cls, qualname: str, node: ast.ClassDef, bindings: ModuleNominalBindingSnapshot
    ) -> ClassBodyReading:
        reading = cls(qualname, node, bindings)
        for statement in node.body:
            reading.read(statement)
        return reading

    @cached_property
    def frozen(self) -> bool:
        return is_frozen_record(self.node, self.bindings)

    def answer(self, answer: MemberAnswer) -> None:
        self.answers[type(answer)].append(answer)

    def read(self, statement: ast.stmt) -> None:
        match statement:
            case ast.FunctionDef(name=name, returns=ast.Name(id="bool")) if (
                takes_only_receiver(statement) and self.frozen
            ):
                self.read_own_state(name, statement)
        match statement:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.Assign(
                targets=[ast.Name(id=name)]
            ) | ast.AnnAssign(target=ast.Name(id=name)):
                self.declared.add(name)
        match statement:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                self.read_definition(name, statement)
            case ast.Assign(
                targets=[ast.Name(id=name)], value=ast.Constant(value=bool() as value)
            ) if not name.startswith("__"):
                self.answer(ConstantAnswer(name, statement.lineno, repr(value)))
            case ast.AnnAssign(
                target=ast.Name(id=name),
                annotation=annotation,
                value=ast.Constant(value=bool() | str() as value),
            ) if not name.startswith("__") and "ClassVar" in ast.unparse(annotation):
                self.answer(ConstantAnswer(name, statement.lineno, repr(value)))
            case ast.AnnAssign(target=ast.Name(id=name), value=ast.expr()) | ast.Assign(
                targets=[ast.Name(id=name)]
            ):
                self.computed.add(name)

    def read_definition(
        self, name: str, statement: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        if declares_abstract(statement, self.bindings):
            return
        parameters = [
            argument.arg
            for argument in [
                *statement.args.posonlyargs,
                *statement.args.args,
                *statement.args.kwonlyargs,
            ]
        ][1:]
        match statements_after_docstring(statement.body):
            case [ast.Return(value=ast.Constant(value=bool() | str() as value))] if (
                not name.startswith("__") and takes_only_receiver(statement)
            ):
                self.answer(ConstantAnswer(name, statement.lineno, repr(value)))
                return
            case [ast.Return(value=ast.Call(func=ast.Name(id=called)))] if (
                called in parameters and len(parameters) >= 2
            ):
                self.answer(FoldDeclaration(name, statement.lineno, called))
        self.computed.add(name)

    def read_own_state(self, name: str, statement: ast.FunctionDef) -> None:
        match statements_after_docstring(statement.body):
            case [ast.Return(value=ast.expr() as expression)] if calls_only_pure_builtins(
                expression, self.bindings
            ):
                probes = list(own_state_probes(expression))
            case _:
                return
        match probes:
            case [first, *_]:
                self.answer(
                    OwnStateAnswer(
                        name,
                        statement.lineno,
                        "property"
                        if self.bindings.decorated_by(statement, PROPERTY_DECORATORS)
                        else "method",
                        first,
                    )
                )


@dataclass(frozen=True)
class ModuleClassReadings(ParsedModuleFact):
    """Every class of one parse, in source order, its body read once under the
    module bindings its decorators resolve through, taken in one pass."""

    classes: tuple[ClassBodyReading, ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        classes = parsed_module.syntax_index.declarations(ast.ClassDef)
        snapshots = ModuleNominalBindingAuthority(parsed_module).snapshots_before(
            node.lineno for _qualname, node in classes
        )
        return cls(
            tuple(
                ClassBodyReading.of(qualname, node, snapshots[node.lineno])
                for qualname, node in classes
            )
        )


@dataclass(frozen=True)
class ClassShape:
    """The members one class declares, and which of them it computes.

    `symbol` is the class's module-qualified name, the key the compact class
    index (`CompactClassFamilyIndex`) resolves its bases under.
    """

    symbol: str
    declared: frozenset[str]
    computed: frozenset[str]

    @classmethod
    def of(cls, module_name: str, reading: ClassBodyReading) -> ClassShape:
        return cls(
            symbol=f"{module_name}.{reading.qualname}",
            declared=frozenset(reading.declared),
            computed=frozenset(reading.computed),
        )


@dataclass(frozen=True)
class ClassShapesModule:
    module_name: str
    classes: tuple[ClassShape, ...]


class ClassShapeFamily(ShapeFamily[ClassShapesModule]):
    """Each class's bases and the member answers family joins compare."""


    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ClassShapesModule]:
        return [
            ClassShapesModule(
                parsed_module.module_name,
                tuple(
                    ClassShape.of(parsed_module.module_name, reading)
                    for reading in parsed_module.fact(ModuleClassReadings).classes
                ),
            )
        ]


# ---- consumers: where an answer selects a branch, or a call passes lambdas


@dataclass(frozen=True)
class MemberUse(SymbolSite):
    """`x.member` used by a consumer: where, and what `x` is declared to be.

    `receiver` is `x` as its module declares it (`DeclaredPathProbeOwner`):
    a parameter's annotation, the method's own class, a member's annotation or
    a declared return, a declared collection's element, or a local bound from
    one. The scan's declarations resolve it to the class it is declared as
    (`ProbeOwner.resolved`); anything else says why it stays unresolved.
    """

    member: str
    receiver: ProbeOwner


@dataclass(frozen=True)
class BranchRead(MemberUse):
    """`x.member` (or `x.member()`) read in a test that selects a branch."""

    called: bool


@dataclass(frozen=True)
class LambdaCall(MemberUse):
    """`x.member(...)` called with two or more lambdas."""

    lambdas: int


@dataclass(frozen=True)
class ConsumerShapesModule:
    """A module's member uses."""

    module_name: str
    branch_reads: tuple[BranchRead, ...]
    lambda_calls: tuple[LambdaCall, ...]


def branch_tests(node: ast.AST) -> Iterator[ast.expr]:
    """The expressions whose truth selects a branch at this node."""

    match node:
        case ast.If(test=test) | ast.While(test=test) | ast.IfExp(test=test):
            yield test
        case ast.comprehension(ifs=tests):
            yield from tests
        case ast.match_case(guard=ast.expr() as guard):
            yield guard


def tested_reads(test: ast.expr) -> Iterator[tuple[ast.expr, str, bool]]:
    """Attribute reads a branch test decides on, through `not`, `and` and `or`."""

    match test:
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            yield from tested_reads(operand)
        case ast.BoolOp(values=values):
            for value in values:
                yield from tested_reads(value)
        case ast.Attribute(attr=member):
            yield test, member, False
        case ast.Call(func=ast.Attribute(attr=member), args=[], keywords=[]):
            yield test, member, True


class ConsumerShapeFamily(ShapeFamily[ConsumerShapesModule]):
    """Branch tests that read a member, and calls that pass two or more lambdas."""


    @staticmethod
    def receiver(node: ast.expr, resolver: ProbeOwnerResolver) -> ProbeOwner:
        """The object `node.member` or `node.member()` is read from, as its module declares it."""

        match node:
            case ast.Attribute(value=receiver) | ast.Call(
                func=ast.Attribute(value=receiver)
            ):
                return resolver.owner(receiver)
        return UnresolvedProbeOwner(f"`{ast.unparse(node)}` reads no member")

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ConsumerShapesModule]:
        index = parsed_module.syntax_index
        resolver = ProbeOwnerResolver.of(parsed_module)
        reads = []
        calls = []
        for node_index, node in enumerate(index.depth_first_nodes):
            for test in branch_tests(node):
                for read, member, called in tested_reads(test):
                    reads.append(
                        BranchRead(
                            parsed_module.file_path,
                            read.lineno,
                            index.evaluating_symbol(node_index),
                            member,
                            cls.receiver(read, resolver),
                            called,
                        )
                    )
            match node:
                case ast.Call(func=ast.Attribute(attr=member), args=arguments, keywords=keywords):
                    lambdas = sum(
                        isinstance(argument, ast.Lambda)
                        for argument in [*arguments, *(keyword.value for keyword in keywords)]
                    )
                    if lambdas >= 2:
                        calls.append(
                            LambdaCall(
                                parsed_module.file_path,
                                node.lineno,
                                index.evaluating_symbol(node_index),
                                member,
                                cls.receiver(node, resolver),
                                lambdas,
                            )
                        )
        return [
            ConsumerShapesModule(parsed_module.module_name, tuple(reads), tuple(calls))
        ]


# ---- site families, one per relocation detector


class MemberAnswerSiteFamily(RestatingSiteFamily[RestatingSiteModule], ABC):
    """Members of the module's classes that answer one way: `group` is the member."""

    _registry_skip = True
    answer: ClassVar[type[MemberAnswer]]

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for reading in parsed_module.fact(ModuleClassReadings).classes:
            for answer in reading.answers.get(cls.answer, ()):
                yield RestatingSite(
                    parsed_module.file_path,
                    answer.lineno,
                    f"{reading.qualname}.{answer.member}",
                    answer.member,
                    (f"{parsed_module.module_name}.{reading.qualname}", *answer.detail),
                )


class ConstantAnswerSiteFamily(MemberAnswerSiteFamily):
    """A member declaration that answers with one constant."""

    answer = ConstantAnswer


class OwnStateBoolSiteFamily(MemberAnswerSiteFamily):
    """A frozen record's `-> bool` member whose answer is a state of the record."""

    answer = OwnStateAnswer


class LambdaFoldSiteFamily(MemberAnswerSiteFamily):
    """A method returning a call of exactly one of its callable parameters."""

    answer = FoldDeclaration


def reads_through_self(expression: ast.expr) -> bool:
    """An attribute chain whose root is `self`: another of the record's own fields."""

    match expression:
        case ast.Name(id="self"):
            return True
        case ast.Attribute(value=inner):
            return reads_through_self(inner)
    return False


def own_state_probes(expression: ast.expr) -> Iterator[str]:
    """Each direct probe of one of the record's own fields: absence or a kind.

    A kind is a constant, a name or another object's attribute; a comparison
    with a computed value (`self.count == len(self.items)`) or with another own
    field (`self.start == self.end`) is a relation, not a state.
    """

    for node in ast.walk(expression):
        match node:
            case ast.Compare(
                left=ast.Attribute(value=ast.Name(id="self")) as field,
                ops=[ast.Is() | ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ):
                yield f"probes `{ast.unparse(field)}` for absence"
            case ast.Call(
                func=ast.Name(id="isinstance"),
                args=[ast.Attribute(value=ast.Name(id="self")) as field, _],
            ):
                yield f"tests the kind of `{ast.unparse(field)}`"
            case ast.Compare(
                left=ast.Attribute(value=ast.Name(id="self")),
                comparators=[ast.Attribute() as other],
            ) if reads_through_self(other):
                pass
            case ast.Compare(
                left=ast.Attribute(value=ast.Name(id="self")) as field,
                ops=[ast.Eq() | ast.NotEq() | ast.Is() | ast.IsNot() | ast.In() | ast.NotIn()],
                comparators=[ast.Constant() | ast.Name() | ast.Attribute()],
            ):
                yield f"compares `{ast.unparse(field)}` with a kind"


PURE_BUILTINS = frozenset(
    ("all", "any", "bool", "frozenset", "isinstance", "issubclass", "len", "max", "min", "set", "sorted", "sum", "tuple")
)


def calls_only_pure_builtins(
    expression: ast.expr, snapshot: ModuleNominalBindingSnapshot
) -> bool:
    """Every call in the expression is a pure builtin the module does not shadow.

    A call of anything else (a method of a field, a global's accessor) can read
    state the record does not hold, so the answer is not fixed at construction.
    """

    for node in ast.walk(expression):
        match node:
            case ast.Call(func=ast.Name(id=name)) if name in PURE_BUILTINS and (
                snapshot.resolves_unshadowed_builtin(name)
            ):
                pass
            case ast.Call():
                return False
    return True


class ZeroOneTupleSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """A function whose every return is `()` or a one-element tuple, with both.

    Overrides that each return one arity are left out: a family whose members
    contribute zero or one item already splits present from absent.
    """


    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for definition in parsed_module.fact(ModuleDefinitions).definitions:
            if definition.absence_tuple_arities != {0, 1}:
                continue
            yield RestatingSite(
                parsed_module.file_path,
                definition.node.lineno,
                definition.qualname,
                definition.node.name,
                (),
            )


def spells_none(node: ast.AST) -> bool:
    match node:
        case ast.Constant(value=None):
            return True
    return False


def optional_annotation(annotation: ast.expr | None) -> bool:
    match annotation:
        case ast.BinOp(op=ast.BitOr()):
            return any(spells_none(part) for part in ast.walk(annotation))
        case ast.Subscript(value=wrapper) if TypeWrapper.OPTIONAL.spelled_by(
            ExpressionReference.decoded(wrapper).terminal_name
        ):
            return True
        case ast.Constant(value=str() as text):
            return "| None" in text or "None |" in text or text.startswith("Optional[")
    return False


def undefaulted_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    positional = [*node.args.posonlyargs, *node.args.args]
    undefaulted = positional[: len(positional) - len(node.args.defaults)]
    undefaulted.extend(
        argument
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
        if not isinstance(default, ast.expr)
    )
    return undefaulted


class ThreadedAbsenceSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """An undefaulted `X | None` parameter that its own body compares with None.

    A probe choosing between the parameter itself and a replacement
    (`default if value is None else value`), or choosing the returned value of
    a function whose declared return is not optional, is where absence is
    decided, not asked again, so it is left out. So is a dunder method: the language's
    protocol fixes its signature and the interpreter is its caller.
    """


    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for definition in parsed_module.fact(ModuleDefinitions).definitions:
            name = definition.node.name
            if name.startswith("__") and name.endswith("__"):
                continue
            optional = {
                argument.arg
                for argument in undefaulted_parameters(definition.node)
                if optional_annotation(argument.annotation)
            }
            match definition.node.returns:
                case ast.expr() as declared:
                    settles = not optional_annotation(declared)
                case _:
                    settles = False
            deciding = {
                id(node.test)
                for node in definition.nodes
                if isinstance(node, ast.IfExp)
                and any(
                    isinstance(arm, ast.Name) and arm.id in optional
                    for arm in (node.body, node.orelse)
                )
            } | {
                id(node.value.test)
                for node in definition.nodes
                if settles
                and isinstance(node, ast.Return)
                and isinstance(node.value, ast.IfExp)
            }
            for node in definition.nodes:
                match node:
                    case ast.Compare(
                        left=ast.Name(id=name),
                        ops=[ast.Is() | ast.IsNot() as operator],
                        comparators=[ast.Constant(value=None)],
                    ) if name in optional and id(node) not in deciding:
                        yield RestatingSite(
                            parsed_module.file_path,
                            node.lineno,
                            definition.qualname,
                            name,
                            (
                                "is None"
                                if isinstance(operator, ast.Is)
                                else "is not None",
                            ),
                        )
