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
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from nominal_refactor_advisor.annotation_semantics import TypeWrapper
from nominal_refactor_advisor.ast_field_decoders import variadic_parameters
from nominal_refactor_advisor.ast_projection import ExpressionReference
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    ABSTRACT_METHOD_DECORATOR,
    PROPERTY_DECORATORS,
    ModuleNominalBindingAuthority,
    ModuleNominalBindingSnapshot,
)
from nominal_refactor_advisor.class_symbols import ClassSymbolResolutionAuthority
from nominal_refactor_advisor.foreign_family import FunctionDefinition
from nominal_refactor_advisor.observation_families import ShapeFamily, TypeArgumentResolver
from nominal_refactor_advisor.probe_owner import (
    ProbeOwner,
    ProbeOwnerResolver,
    UnresolvedProbeOwner,
)
from nominal_refactor_advisor.restated_fact_observations import (
    RestatingSiteFamily,
    RestatingSiteModule,
)
from nominal_refactor_advisor.observation_shapes import RestatingSite

def statements_of(scope: ast.AST) -> list[ast.stmt]:
    """The statement list of a module, class or def."""

    match scope:
        case ast.Module(body=body) | ast.ClassDef(body=body):
            return body
        case ast.FunctionDef(body=body) | ast.AsyncFunctionDef(body=body):
            return body
    return []


@dataclass(frozen=True)
class ScopedDefinition:
    """A def statement with its dotted name and immediate lexical scope."""

    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    owner: ast.AST


def scoped_definitions(parsed_module: ParsedModule) -> Iterator[ScopedDefinition]:
    """Every def in the module, named by its enclosing classes and functions."""

    children = parsed_module.syntax_index.named_scope_children
    for prefix, scope in walk_scopes(parsed_module):
        for nested in FunctionDefinition.within(children[scope]):
            yield ScopedDefinition(
                f"{prefix}{nested.name}",
                cast(ast.FunctionDef | ast.AsyncFunctionDef, nested),
                scope,
            )


def walk_scopes(parsed_module: ParsedModule) -> Iterator[tuple[str, ast.AST]]:
    """The module, every class and every def, each with the prefix of its members."""

    children = parsed_module.syntax_index.named_scope_children
    stack: list[tuple[str, ast.AST]] = [("", parsed_module.module)]
    while stack:
        prefix, scope = stack.pop()
        yield prefix, scope
        for nested in children[scope]:
            stack.append((f"{prefix}{nested.name}.", nested))


def scoped_classes(parsed_module: ParsedModule) -> Iterator[tuple[str, ast.ClassDef]]:
    """Every original class with its dotted name, retaining the scope walk's order."""

    children = parsed_module.syntax_index.named_scope_children
    for prefix, scope in walk_scopes(parsed_module):
        for nested in children[scope]:
            if isinstance(nested, ast.ClassDef):
                yield f"{prefix}{nested.name}", nested


def bound_classes(
    parsed_module: ParsedModule,
) -> Iterator[tuple[str, ast.ClassDef, ModuleNominalBindingSnapshot]]:
    """Every class in the module with its dotted name and the module bindings its
    body's decorators resolve through, taken in one pass."""

    classes = tuple(scoped_classes(parsed_module))
    snapshots = ModuleNominalBindingAuthority(parsed_module).snapshots_before(
        node.lineno for _qualname, node in classes
    )
    for qualname, node in classes:
        yield qualname, node, snapshots[node.lineno]


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


def function_nodes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
    """The nodes of one function body, not of the functions or classes it nests."""

    for statement in node.body:
        yield from scope_nodes(statement)


def scope_nodes(node: ast.AST) -> Iterator[ast.AST]:
    yield node
    if isinstance(node, (*FunctionDefinition.members, ast.ClassDef, ast.Lambda)):
        return
    for child in ast.iter_child_nodes(node):
        yield from scope_nodes(child)


# ---- the class shapes every family join reads


@dataclass(frozen=True)
class ConstantAnswer:
    """A member whose declaration answers with one constant."""

    member: str
    value: str
    lineno: int


@dataclass(frozen=True)
class FoldDeclaration:
    """A method that returns a call of exactly one of its callable parameters."""

    member: str
    called_parameter: str
    lineno: int


def absence_tuple_arities(node: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[int]:
    """The arities a function returns when every return is `()` or `(x,)`
    (absence spelled as a collection); empty when any return is something else."""

    arities: set[int] = set()
    for child in function_nodes(node):
        match child:
            case ast.Return(value=ast.Tuple(elts=elements)) if len(
                elements
            ) <= 1 and not any(isinstance(element, ast.Starred) for element in elements):
                arities.add(len(elements))
            case ast.Return():
                return frozenset()
    return frozenset(arities)


@dataclass(frozen=True)
class TupleArity:
    """A member whose every return is `()` or a one-element tuple, and which."""

    member: str
    arities: frozenset[int]
    lineno: int


@dataclass
class ClassBodyReading:
    """One pass over a class body sorting each member declaration by how it answers.

    A constant answer is a zero-argument method or property returning one bool
    or str constant, a class attribute assigned a bool constant, or a `ClassVar`
    assigned a bool or str constant. Abstract declarations and bare annotations
    answer nothing; every other declaration is computed.
    """

    bindings: ModuleNominalBindingSnapshot
    answers: list[ConstantAnswer]
    computed: set[str]
    declared: set[str]
    folds: list[FoldDeclaration]
    arities: list[TupleArity]

    @classmethod
    def of(
        cls, node: ast.ClassDef, bindings: ModuleNominalBindingSnapshot
    ) -> ClassBodyReading:
        reading = cls(bindings, [], set(), set(), [], [])
        for statement in node.body:
            reading.read(statement)
        return reading

    def read(self, statement: ast.stmt) -> None:
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
                self.answers.append(ConstantAnswer(name, repr(value), statement.lineno))
            case ast.AnnAssign(
                target=ast.Name(id=name),
                annotation=annotation,
                value=ast.Constant(value=bool() | str() as value),
            ) if not name.startswith("__") and "ClassVar" in ast.unparse(annotation):
                self.answers.append(ConstantAnswer(name, repr(value), statement.lineno))
            case ast.AnnAssign(target=ast.Name(id=name), value=ast.expr()) | ast.Assign(
                targets=[ast.Name(id=name)]
            ):
                self.computed.add(name)

    def read_definition(
        self, name: str, statement: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        arities = absence_tuple_arities(statement)
        if arities:
            self.arities.append(TupleArity(name, arities, statement.lineno))
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
                self.answers.append(ConstantAnswer(name, repr(value), statement.lineno))
                return
            case [ast.Return(value=ast.Call(func=ast.Name(id=called)))] if (
                called in parameters and len(parameters) >= 2
            ):
                self.folds.append(FoldDeclaration(name, called, statement.lineno))
        self.computed.add(name)


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
    def of(
        cls,
        parsed_module: ParsedModule,
        qualname: str,
        node: ast.ClassDef,
        bindings: ModuleNominalBindingSnapshot,
    ) -> ClassShape:
        reading = ClassBodyReading.of(node, bindings)
        return cls(
            symbol=f"{parsed_module.module_name}.{qualname}",
            declared=frozenset(reading.declared),
            computed=frozenset(reading.computed),
        )


@dataclass(frozen=True)
class ClassShapesModule:
    module_name: str
    classes: tuple[ClassShape, ...]


class ClassShapeFamily(ShapeFamily[ClassShapesModule]):
    """Each class's bases and the member answers family joins compare."""

    item_type = ClassShapesModule

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ClassShapesModule]:
        return [
            ClassShapesModule(
                parsed_module.module_name,
                tuple(
                    ClassShape.of(parsed_module, qualname, node, bindings)
                    for qualname, node, bindings in bound_classes(parsed_module)
                ),
            )
        ]


# ---- consumers: where an answer selects a branch, or a call passes lambdas


@dataclass(frozen=True)
class MemberUse:
    """`x.member` used by a consumer: where, and what `x` is declared to be.

    `receiver` is `x` as its module declares it (`DeclaredPathProbeOwner`):
    a parameter's annotation, the method's own class, a member's annotation or
    a declared return, a declared collection's element, or a local bound from
    one. The scan's declarations resolve it to the class it is declared as
    (`ProbeOwner.resolved`); anything else says why it stays unresolved.
    """

    file_path: str
    lineno: int
    symbol: str
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


def enclosing_symbols(parsed_module: ParsedModule) -> dict[int, str]:
    """The dotted name of the def or class enclosing each node."""

    children = parsed_module.syntax_index.named_scope_children
    symbols: dict[int, str] = {}
    for prefix, scope in walk_scopes(parsed_module):
        symbol = prefix.rstrip(".") or "<module>"
        for statement in statements_of(scope):
            for node in scope_nodes(statement):
                symbols[id(node)] = symbol
        for nested in children[scope]:
            for child in ast.iter_child_nodes(nested):
                if not isinstance(child, ast.stmt):
                    for node in ast.walk(child):
                        symbols.setdefault(id(node), symbol)
    return symbols


class ConsumerShapeFamily(ShapeFamily[ConsumerShapesModule]):
    """Branch tests that read a member, and calls that pass two or more lambdas."""

    item_type = ConsumerShapesModule

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
        symbols = enclosing_symbols(parsed_module)
        resolver = ProbeOwnerResolver.of(parsed_module)
        reads = []
        calls = []
        for node in ast.walk(parsed_module.module):
            symbol = symbols.get(id(node), "<module>")
            for test in branch_tests(node):
                for read, member, called in tested_reads(test):
                    reads.append(
                        BranchRead(
                            parsed_module.file_path,
                            read.lineno,
                            symbol,
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
                                symbol,
                                member,
                                cls.receiver(node, resolver),
                                lambdas,
                            )
                        )
        return [
            ConsumerShapesModule(parsed_module.module_name, tuple(reads), tuple(calls))
        ]


# ---- site families, one per relocation detector


class ConstantAnswerSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """A member declaration that answers with one constant: `group` is the member."""

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for qualname, node, bindings in bound_classes(parsed_module):
            for answer in ClassBodyReading.of(node, bindings).answers:
                yield RestatingSite(
                    parsed_module.file_path,
                    answer.lineno,
                    f"{qualname}.{answer.member}",
                    answer.member,
                    (f"{parsed_module.module_name}.{qualname}", answer.value),
                )


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


class OwnStateBoolSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """A `-> bool` property or zero-argument method of a frozen record whose
    answer is a state of the record: its one return probes an own field for
    absence or a kind, and calls nothing but pure builtins."""

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for qualname, node, bindings in bound_classes(parsed_module):
            if not is_frozen_record(node, bindings):
                continue
            for statement in node.body:
                match statement:
                    case ast.FunctionDef(
                        name=name,
                        returns=ast.Name(id="bool"),
                        body=body,
                    ) if takes_only_receiver(statement):
                        pass
                    case _:
                        continue
                match statements_after_docstring(body):
                    case [ast.Return(value=ast.expr() as expression)] if calls_only_pure_builtins(
                        expression, bindings
                    ):
                        probes = list(own_state_probes(expression))
                    case _:
                        continue
                match probes:
                    case [first, *_]:
                        accessed = (
                            "property"
                            if bindings.decorated_by(
                                statement, PROPERTY_DECORATORS
                            )
                            else "method"
                        )
                        yield RestatingSite(
                            parsed_module.file_path,
                            statement.lineno,
                            f"{qualname}.{name}",
                            name,
                            (
                                f"{parsed_module.module_name}.{qualname}",
                                accessed,
                                first,
                            ),
                        )


class LambdaFoldSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """A method returning a call of exactly one of its callable parameters."""

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for qualname, node, bindings in bound_classes(parsed_module):
            for fold in ClassBodyReading.of(node, bindings).folds:
                yield RestatingSite(
                    parsed_module.file_path,
                    fold.lineno,
                    f"{qualname}.{fold.member}",
                    fold.member,
                    (f"{parsed_module.module_name}.{qualname}", fold.called_parameter),
                )


class ZeroOneTupleSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """A function whose every return is `()` or a one-element tuple, with both.

    Overrides that each return one arity are left out: a family whose members
    contribute zero or one item already splits present from absent.
    """

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for definition in scoped_definitions(parsed_module):
            if absence_tuple_arities(definition.node) != {0, 1}:
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


REACTIVE_DECLARERS = frozenset({"reactive", "var", "Reactive"})
"""Textual's reactive descriptors: a class attribute assigned one of these is a reactive."""

REACTIVE_CALLBACK = re.compile(r"_?(?:watch|validate)_(?P<reactive>\w+)")
"""Textual calls `watch_x`/`validate_x` (or `_watch_x`/`_validate_x`) with reactive `x`'s value."""


def reactive_names(owner: ast.ClassDef) -> frozenset[str]:
    """The reactives the class body declares: `x = reactive(...)` or `x: T = var(...)`."""

    names: set[str] = set()
    for statement in owner.body:
        match statement:
            case ast.Assign(targets=[ast.Name(id=name)], value=ast.Call(func=callee)) | ast.AnnAssign(
                target=ast.Name(id=name), value=ast.Call(func=callee)
            ) if ExpressionReference.decoded(_called(callee)).terminal_name in REACTIVE_DECLARERS:
                names.add(name)
    return frozenset(names)


def _called(callee: ast.expr) -> ast.expr:
    """`reactive[int](...)` is a call of `reactive`."""

    return callee.value if isinstance(callee, ast.Subscript) else callee


def _optional_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.arg]:
    """Every parameter annotated `X | None`, defaulted or not."""

    return {
        argument.arg: argument
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        if optional_annotation(argument.annotation)
    }


def _none_default(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """Whether the parameter's default is `None`."""

    positional = [*node.args.posonlyargs, *node.args.args]
    defaulted = positional[len(positional) - len(node.args.defaults):]
    defaults: dict[str, ast.expr] = {
        argument.arg: default for argument, default in zip(defaulted, node.args.defaults)
    }
    defaults |= {
        argument.arg: default
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
        if default is not None
    }
    return name in defaults and spells_none(defaults[name])


@dataclass(frozen=True)
class OptionalParameterThread:
    """One function's undefaulted `X | None` parameters, and how a call passes them.

    A private function whose every call in the module passes, for one of them,
    an optional parameter of its caller is a copy of that caller's absence: the
    absence starts at the caller, so the copy is not reported and the caller's
    `None` default is, once (IDEN-5: the same sentinel restated at every level it
    is passed through)."""

    definition: ScopedDefinition
    optional: frozenset[str]

    @classmethod
    def of(cls, definition: ScopedDefinition) -> OptionalParameterThread:
        return cls(definition, frozenset(
            argument.arg
            for argument in undefaulted_parameters(definition.node)
            if optional_annotation(argument.annotation)
        ))

    @property
    def private(self) -> bool:
        name = self.definition.node.name
        return name.startswith("_") and not name.endswith("__")

    def passed(self, call: ast.Call, parameter: str, on_receiver: bool) -> ast.expr | None:
        """What the call passes for the parameter, when the call spells it."""

        for keyword in call.keywords:
            if keyword.arg == parameter:
                return keyword.value
        node = self.definition.node
        positional = [argument.arg for argument in (*node.args.posonlyargs, *node.args.args)]
        if on_receiver:
            positional = positional[1:]
        if parameter not in positional or positional.index(parameter) >= len(call.args):
            return None
        argument = call.args[positional.index(parameter)]
        return None if isinstance(argument, ast.Starred) else argument


class ThreadedAbsenceSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """An undefaulted `X | None` parameter that its own body compares with None.

    A probe choosing between the parameter itself and a replacement
    (`default if value is None else value`), or choosing the returned value of
    a function whose declared return is not optional, is where absence is
    decided, not asked again, so it is left out. So is a dunder method: the language's
    protocol fixes its signature and the interpreter is its caller. So is the
    decoder the factoring builds: a `@boundary_decoder`, and a probe whose None
    branch answers with a class (the site's second detail, which the detector
    checks against the scan's families). So is a Textual callback (`watch_x`,
    `validate_x`) of a reactive `x` its class declares: the framework calls it with
    the reactive's value.

    A parameter passed through from its callers' optional parameters is reported
    once, at the caller parameter whose `None` default starts the absence (see
    `OptionalParameterThread`).
    """

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        resolver = TypeArgumentResolver(parsed_module)
        definitions = list(scoped_definitions(parsed_module))
        threads = {id(definition.node): OptionalParameterThread.of(definition) for definition in definitions}
        copies = cls.threaded_copies(definitions, threads)
        for definition in definitions:
            name = definition.node.name
            if name.startswith("__") and name.endswith("__"):
                continue
            if any(
                ExpressionReference.decoded(decorator).terminal_name == "boundary_decoder"
                for decorator in definition.node.decorator_list
            ):
                continue
            if isinstance(definition.owner, ast.ClassDef) and (
                callback := REACTIVE_CALLBACK.fullmatch(name)
            ) and callback["reactive"] in reactive_names(definition.owner):
                continue
            optional = threads[id(definition.node)].optional - copies.get(id(definition.node), frozenset())
            yield from cls.probes(parsed_module, resolver, definition, optional)
        yield from cls.origins(parsed_module, definitions, threads, copies)

    @classmethod
    def probes(
        cls,
        parsed_module: ParsedModule,
        resolver: TypeArgumentResolver,
        definition: ScopedDefinition,
        optional: frozenset[str],
    ) -> Iterator[RestatingSite]:
        match definition.node.returns:
            case ast.expr() as declared:
                settles = not optional_annotation(declared)
            case _:
                settles = False
        deciding = {
            id(node.test)
            for node in function_nodes(definition.node)
            if isinstance(node, ast.IfExp)
            and any(
                isinstance(arm, ast.Name) and arm.id in optional
                for arm in (node.body, node.orelse)
            )
        } | {
            id(node.value.test)
            for node in function_nodes(definition.node)
            if settles
            and isinstance(node, ast.Return)
            and isinstance(node.value, ast.IfExp)
        }
        branches: dict[int, tuple[list[ast.AST], list[ast.AST]]] = {
            id(node.test): (list(node.body), list(node.orelse))
            for node in function_nodes(definition.node)
            if isinstance(node, ast.If)
        } | {
            id(node.test): ([node.body], [node.orelse])
            for node in function_nodes(definition.node)
            if isinstance(node, ast.IfExp)
        }
        for node in function_nodes(definition.node):
            match node:
                case ast.Compare(
                    left=ast.Name(id=name),
                    ops=[ast.Is() | ast.IsNot() as operator],
                    comparators=[ast.Constant(value=None)],
                ) if name in optional and id(node) not in deciding:
                    is_none = isinstance(operator, ast.Is)
                    yield RestatingSite(
                        parsed_module.file_path,
                        node.lineno,
                        definition.qualname,
                        name,
                        (
                            "is None" if is_none else "is not None",
                            cls.none_answer(resolver, definition, branches.get(id(node)), is_none),
                        ),
                    )

    @staticmethod
    def none_answer(
        resolver: TypeArgumentResolver,
        definition: ScopedDefinition,
        branches: tuple[list[ast.AST], list[ast.AST]] | None,
        is_none: bool,
    ) -> str:
        """The class the None branch answers with (`return C(...)`, `return C`), or ""."""

        if branches is None:
            return ""
        match branches[0] if is_none else branches[1]:
            case [ast.Return(value=ast.expr() as answer), *_] | [ast.expr() as answer]:
                pass
            case _:
                return ""
        named = answer.func if isinstance(answer, ast.Call) else answer
        if not isinstance(named, ast.Name | ast.Attribute):
            return ""
        return resolver.qualified_class_at(named, (definition.node,)) or ""

    @staticmethod
    def calls(
        definitions: list[ScopedDefinition], callee: ScopedDefinition
    ) -> Iterator[tuple[ScopedDefinition, ast.Call, bool]]:
        """Each call of `callee` in this module: by name for a module function,
        through `self`/`cls` from its own class's methods for a method."""

        for caller in definitions:
            for node in function_nodes(caller.node):
                if not isinstance(node, ast.Call):
                    continue
                match node.func:
                    case ast.Name(id=name) if name == callee.node.name and isinstance(callee.owner, ast.Module):
                        yield caller, node, False
                    case ast.Attribute(value=ast.Name(id="self" | "cls"), attr=name) if (
                        name == callee.node.name and caller.owner is callee.owner
                    ):
                        yield caller, node, True

    @classmethod
    def threaded_copies(
        cls, definitions: list[ScopedDefinition], threads: dict[int, OptionalParameterThread]
    ) -> dict[int, frozenset[str]]:
        """For each private function, the optional parameters that every call in the
        module passes an optional parameter of its caller."""

        copies: dict[int, frozenset[str]] = {}
        for definition in definitions:
            thread = threads[id(definition.node)]
            if not thread.private or not thread.optional:
                continue
            calls = list(cls.calls(definitions, definition))
            if not calls:
                continue
            copies[id(definition.node)] = frozenset(
                parameter
                for parameter in thread.optional
                if all(
                    isinstance(passed := thread.passed(call, parameter, on_receiver), ast.Name)
                    and passed.id in _optional_parameters(caller.node)
                    for caller, call, on_receiver in calls
                )
            )
        return copies

    @classmethod
    def origins(
        cls,
        parsed_module: ParsedModule,
        definitions: list[ScopedDefinition],
        threads: dict[int, OptionalParameterThread],
        copies: dict[int, frozenset[str]],
    ) -> Iterator[RestatingSite]:
        """One site per caller parameter whose `None` default reaches a threaded copy:
        the declaration that owns the absence."""

        seen: set[tuple[int, str]] = set()
        for definition in definitions:
            for parameter in sorted(copies.get(id(definition.node), ())):
                pending = [(definition, parameter)]
                while pending:
                    callee, name = pending.pop()
                    thread = threads[id(callee.node)]
                    for caller, call, on_receiver in cls.calls(definitions, callee):
                        passed = thread.passed(call, name, on_receiver)
                        if not isinstance(passed, ast.Name) or (id(caller.node), passed.id) in seen:
                            continue
                        seen.add((id(caller.node), passed.id))
                        if passed.id in copies.get(id(caller.node), frozenset()):
                            pending.append((caller, passed.id))
                        elif _none_default(caller.node, passed.id):
                            yield RestatingSite(
                                parsed_module.file_path,
                                _optional_parameters(caller.node)[passed.id].lineno,
                                caller.qualname,
                                passed.id,
                                (f"defaults to None, threaded to `{definition.node.name}`", ""),
                            )
