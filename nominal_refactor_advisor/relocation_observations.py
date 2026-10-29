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
from collections.abc import Iterator
from dataclasses import dataclass

from .annotation_semantics import TypeWrapper
from .ast_field_decoders import variadic_parameters
from .ast_projection import ExpressionReference
from .ast_tools import ParsedModule
from .class_identity_observations import DeclaredIdentityFamily, QualifiedNames
from .class_index import ModuleNominalBindingAuthority, ModuleNominalBindingSnapshot
from .observation_families import ShapeFamily
from .quoted_reference import ModuleReferenceScope
from .restated_fact_observations import (
    RestatingSite,
    RestatingSiteFamily,
    RestatingSiteModule,
)

DEFINITION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
PROPERTY_DECORATORS = frozenset(
    {"property", "cached_property", "functools.cached_property"}
)
ABSTRACT_DECORATORS = frozenset({"abstractmethod", "abc.abstractmethod"})



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
    """A def statement with its dotted name and the class or module declaring it."""

    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    owner: ast.AST


def scoped_definitions(module: ast.Module) -> Iterator[ScopedDefinition]:
    """Every def in the module, named by its enclosing classes and functions."""

    for prefix, scope in walk_scopes(module):
        for nested in nested_scopes(scope):
            if isinstance(nested, DEFINITION_TYPES):
                yield ScopedDefinition(f"{prefix}{nested.name}", nested, scope)


def nested_scopes(
    scope: ast.AST,
) -> Iterator[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    """The classes and defs a scope declares, inside blocks too, not their own."""

    for statement in statements_of(scope):
        for node in scope_nodes(statement):
            if isinstance(node, (ast.ClassDef, *DEFINITION_TYPES)):
                yield node


def walk_scopes(module: ast.Module) -> Iterator[tuple[str, ast.AST]]:
    """The module, every class and every def, each with the prefix of its members."""

    stack: list[tuple[str, ast.AST]] = [("", module)]
    while stack:
        prefix, scope = stack.pop()
        yield prefix, scope
        for nested in nested_scopes(scope):
            stack.append((f"{prefix}{nested.name}.", nested))


def scoped_classes(module: ast.Module) -> Iterator[tuple[str, ast.ClassDef]]:
    """Every class in the module with its dotted name."""

    for prefix, scope in walk_scopes(module):
        for nested in nested_scopes(scope):
            if isinstance(nested, ast.ClassDef):
                yield f"{prefix}{nested.name}", nested


def decorator_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> frozenset[str]:
    return frozenset(
        ast.unparse(decorator.func if isinstance(decorator, ast.Call) else decorator)
        for decorator in node.decorator_list
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


def declares_abstract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """An abstract declaration asks the question; it answers nothing."""

    if decorator_names(node) & ABSTRACT_DECORATORS:
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
    node: ast.ClassDef, bindings: ModuleNominalBindingAuthority
) -> bool:
    """A frozen dataclass or a NamedTuple: its state is fixed once built.

    The decorator and the base are resolved through the module's bindings, so a
    local class named `dataclass` or `NamedTuple` is not one.
    """

    def resolves_to(reference: ast.expr, qualified_name: str) -> bool:
        return bindings.qualified_name_at(reference, line=node.lineno) == qualified_name

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
    if isinstance(node, (*DEFINITION_TYPES, ast.ClassDef, ast.Lambda)):
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

    answers: list[ConstantAnswer]
    computed: set[str]
    declared: set[str]
    folds: list[FoldDeclaration]
    arities: list[TupleArity]

    @classmethod
    def of(cls, node: ast.ClassDef) -> ClassBodyReading:
        reading = cls([], set(), set(), [], [])
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
        if declares_abstract(statement):
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
        cls, parsed_module: ParsedModule, qualname: str, node: ast.ClassDef
    ) -> ClassShape:
        reading = ClassBodyReading.of(node)
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
                    ClassShape.of(parsed_module, qualname, node)
                    for qualname, node in scoped_classes(parsed_module.module)
                ),
            )
        ]


# ---- consumers: where an answer selects a branch, or a call passes lambdas


@dataclass(frozen=True)
class MemberUse:
    """`x.member` used by a consumer: where, and what `x` is declared to be.

    `receiver` is the module-qualified class `x` is declared as, the key the
    compact class index resolves under: the enclosing class for a method's
    first parameter, a parameter's annotation otherwise; empty when nothing
    declares it, or the function rebinds the name. `module` is the module
    holding the use.
    """

    file_path: str
    lineno: int
    symbol: str
    member: str
    receiver: str
    module: str


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
    """A module's member uses, and the modules its import statements name (with each `from` name as a submodule)."""

    module_name: str
    branch_reads: tuple[BranchRead, ...]
    lambda_calls: tuple[LambdaCall, ...]
    imported: tuple[str, ...] = ()


def declared_receivers(parsed_module: ParsedModule) -> dict[int, dict[str, str]]:
    """For each node of a function's own body, the classes its parameters are declared as.

    A method's first parameter (not under `staticmethod`) is its class; a
    parameter annotated with a bare or dotted name is the class that name
    resolves to (`QualifiedNames`). A parameter the body rebinds anywhere is
    left out: the value read may no longer be the one declared.
    """

    names = QualifiedNames.of(parsed_module)
    classes = {qualname for qualname, _node in scoped_classes(parsed_module.module)}
    found: dict[int, dict[str, str]] = {}
    for prefix, scope in walk_scopes(parsed_module.module):
        if not isinstance(scope, DEFINITION_TYPES):
            continue
        rebound = {
            name
            for statement in scope.body
            for node in ast.walk(statement)
            for name in ModuleReferenceScope.bound_names(node)
        }
        parameters = [*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs]
        owner = prefix.rstrip(".").rpartition(".")[0]
        declared: dict[str, str] = {}
        if (
            owner in classes
            and parameters
            and staticmethod.__name__ not in decorator_names(scope)
        ):
            declared[parameters[0].arg] = f"{parsed_module.module_name}.{owner}"
        for parameter in parameters:
            if isinstance(parameter.annotation, (ast.Name, ast.Attribute)):
                qualified = names.qualified(parameter.annotation)
                if qualified and not qualified.startswith(f"{builtins.__name__}."):
                    declared.setdefault(parameter.arg, qualified)
        declared = {
            name: receiver for name, receiver in declared.items() if name not in rebound
        }
        for statement in statements_of(scope):
            for node in scope_nodes(statement):
                found[id(node)] = declared
    return found


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


def enclosing_symbols(module: ast.Module) -> dict[int, str]:
    """The dotted name of the def or class enclosing each node."""

    symbols: dict[int, str] = {}
    for prefix, scope in walk_scopes(module):
        symbol = prefix.rstrip(".") or "<module>"
        for statement in statements_of(scope):
            for node in scope_nodes(statement):
                symbols[id(node)] = symbol
        for nested in nested_scopes(scope):
            for child in ast.iter_child_nodes(nested):
                if not isinstance(child, ast.stmt):
                    for node in ast.walk(child):
                        symbols.setdefault(id(node), symbol)
    return symbols


class ConsumerShapeFamily(ShapeFamily[ConsumerShapesModule]):
    """Branch tests that read a member, and calls that pass two or more lambdas."""

    item_type = ConsumerShapesModule

    @staticmethod
    def receiver(node: ast.expr, declared: dict[str, str]) -> str:
        """The declared class of the object `node.member` or `node.member()` is read from."""

        match node:
            case ast.Attribute(value=ast.Name(id=name)) | ast.Call(
                func=ast.Attribute(value=ast.Name(id=name))
            ):
                return declared.get(name, "")
        return ""

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ConsumerShapesModule]:
        symbols = enclosing_symbols(parsed_module.module)
        receivers = declared_receivers(parsed_module)
        module = parsed_module.module_name
        reads = []
        calls = []
        for node in ast.walk(parsed_module.module):
            symbol = symbols.get(id(node), "<module>")
            declared = receivers.get(id(node), {})
            for test in branch_tests(node):
                for read, member, called in tested_reads(test):
                    reads.append(
                        BranchRead(
                            parsed_module.file_path,
                            read.lineno,
                            symbol,
                            member,
                            cls.receiver(read, declared),
                            module,
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
                                cls.receiver(node, declared),
                                module,
                                lambdas,
                            )
                        )
        return [
            ConsumerShapesModule(
                module,
                tuple(reads),
                tuple(calls),
                tuple(
                    sorted(
                        {
                            found
                            for imported, taken in DeclaredIdentityFamily.imported(
                                parsed_module
                            )
                            for found in (
                                imported,
                                *(f"{imported}.{name}" for name in taken),
                            )
                        }
                    )
                ),
            )
        ]


# ---- site families, one per relocation detector


class ConstantAnswerSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """A member declaration that answers with one constant: `group` is the member."""

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for qualname, node in scoped_classes(parsed_module.module):
            for answer in ClassBodyReading.of(node).answers:
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
        bindings = ModuleNominalBindingAuthority(parsed_module)
        for qualname, node in scoped_classes(parsed_module.module):
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
                        expression, bindings.snapshot_before(node.lineno)
                    ):
                        probes = list(own_state_probes(expression))
                    case _:
                        continue
                match probes:
                    case [first, *_]:
                        accessed = (
                            "property"
                            if decorator_names(statement) & PROPERTY_DECORATORS
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
        for qualname, node in scoped_classes(parsed_module.module):
            for fold in ClassBodyReading.of(node).folds:
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
        for definition in scoped_definitions(parsed_module.module):
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


class ThreadedAbsenceSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """An undefaulted `X | None` parameter that its own body compares with None.

    A probe choosing between the parameter itself and a replacement
    (`default if value is None else value`), or choosing the returned value of
    a function whose declared return is not optional, is where absence is
    decided, not asked again, so it is left out. So is a dunder method: the language's
    protocol fixes its signature and the interpreter is its caller.
    """

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for definition in scoped_definitions(parsed_module.module):
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
            for node in function_nodes(definition.node):
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
