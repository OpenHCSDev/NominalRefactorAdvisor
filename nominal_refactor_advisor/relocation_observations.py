"""Relocation shapes: where an absence or kind probe reappears under another spelling.

Removing an `x is None` probe or a kind comparison does not remove the branch
when the fact moves somewhere a probe detector does not look: a family member
that answers which member it is, a property that derives a fixed answer from
the object's own state, an exception caught around one accessor, a method that
takes both branches as callables, a 0/1 tuple, or an `X | None` parameter.
These families collect each spelling so the count cannot drop while the branch
stays.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Iterator
from dataclasses import dataclass

from .ast_tools import ParsedModule
from .observation_families import ShapeFamily
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
BUILTIN_EXCEPTIONS = frozenset(
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
)


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
        and arguments.vararg is None
        and not arguments.kwonlyargs
        and arguments.kwarg is None
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


def base_names(node: ast.ClassDef) -> tuple[str, ...]:
    """Each base as the last name it is spelled with (`pkg.Base[T]` is `Base`)."""

    names = []
    for base in node.bases:
        unsubscripted = base.value if isinstance(base, ast.Subscript) else base
        match unsubscripted:
            case ast.Name(id=name) | ast.Attribute(attr=name):
                names.append(name)
    return tuple(names)


def is_frozen_record(node: ast.ClassDef) -> bool:
    """A frozen dataclass or a NamedTuple: its state is fixed once built."""

    if "NamedTuple" in base_names(node):
        return True
    return any(
        isinstance(decorator, ast.Call)
        and ast.unparse(decorator.func) in ("dataclass", "dataclasses.dataclass")
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
    folds: list[FoldDeclaration]
    arities: list[TupleArity]

    @classmethod
    def of(cls, node: ast.ClassDef) -> ClassBodyReading:
        reading = cls([], set(), [], [])
        for statement in node.body:
            reading.read(statement)
        return reading

    def read(self, statement: ast.stmt) -> None:
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
    """What one class declares that a family join needs: its bases and its answers."""

    file_path: str
    lineno: int
    name: str
    qualname: str
    bases: tuple[str, ...]
    constant_answers: tuple[ConstantAnswer, ...]
    computed_members: frozenset[str]
    folds: tuple[FoldDeclaration, ...]
    tuple_arities: tuple[TupleArity, ...]

    @classmethod
    def of(
        cls, parsed_module: ParsedModule, qualname: str, node: ast.ClassDef
    ) -> ClassShape:
        reading = ClassBodyReading.of(node)
        return cls(
            file_path=parsed_module.file_path,
            lineno=node.lineno,
            name=node.name,
            qualname=qualname,
            bases=base_names(node),
            constant_answers=tuple(reading.answers),
            computed_members=frozenset(reading.computed),
            folds=tuple(reading.folds),
            tuple_arities=tuple(reading.arities),
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
class BranchRead:
    """`x.member` (or `x.member()`) read in a test that selects a branch."""

    file_path: str
    lineno: int
    symbol: str
    member: str
    called: bool


@dataclass(frozen=True)
class LambdaCall:
    """`x.member(...)` called with two or more lambdas."""

    file_path: str
    lineno: int
    symbol: str
    member: str
    lambdas: int


@dataclass(frozen=True)
class ConsumerShapesModule:
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

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ConsumerShapesModule]:
        symbols = enclosing_symbols(parsed_module.module)
        reads = []
        calls = []
        for node in ast.walk(parsed_module.module):
            symbol = symbols.get(id(node), "<module>")
            for test in branch_tests(node):
                for read, member, called in tested_reads(test):
                    reads.append(
                        BranchRead(
                            parsed_module.file_path, read.lineno, symbol, member, called
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
                                parsed_module.file_path, node.lineno, symbol, member, lambdas
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
        for qualname, node in scoped_classes(parsed_module.module):
            for answer in ClassBodyReading.of(node).answers:
                yield RestatingSite(
                    parsed_module.file_path,
                    answer.lineno,
                    f"{qualname}.{answer.member}",
                    answer.member,
                    (node.name, answer.value),
                )


def own_state_shape(expression: ast.expr) -> str:
    """How a derived answer reads the object: an optional probe, a kind, or fields."""

    shapes = []
    for node in ast.walk(expression):
        match node:
            case ast.Compare(
                left=ast.Attribute(value=ast.Name(id="self")) as left,
                ops=[ast.Is() | ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ):
                shapes.append(f"probes `{ast.unparse(left)}` for absence")
            case ast.Call(func=ast.Name(id="isinstance"), args=[subject, *_]):
                shapes.append(f"tests the kind of `{ast.unparse(subject)}`")
            case ast.Compare(
                left=ast.Attribute(value=ast.Name(id="self")) as left,
                ops=[ast.Eq() | ast.NotEq() | ast.Is() | ast.IsNot() | ast.In() | ast.NotIn()],
            ):
                shapes.append(f"compares `{ast.unparse(left)}` with a kind")
    return shapes[0] if shapes else "derives it from the record's own fields"


class OwnStateBoolSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """A `-> bool` property or zero-argument method of a frozen record whose
    answer is derived from the record's own state."""

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for qualname, node in scoped_classes(parsed_module.module):
            if not is_frozen_record(node):
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
                    case [ast.Return(value=ast.expr() as expression)] if not isinstance(
                        expression, ast.Constant
                    ) and any(
                        isinstance(part, ast.Name) and part.id == "self"
                        for part in ast.walk(expression)
                    ):
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
                            (node.name, accessed, own_state_shape(expression)),
                        )


def handled_names(handler: ast.ExceptHandler) -> tuple[str, ...]:
    match handler.type:
        case ast.Tuple(elts=elements):
            caught = elements
        case ast.expr() as single:
            caught = [single]
        case _:
            caught = []
    return tuple(ast.unparse(name).rsplit(".", 1)[-1] for name in caught)


class ExceptionAbsenceSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """`try:` around one statement whose `except` arm is behaviour, not a re-raise.

    `group` is the caught name and `detail` the guarded statement; the join keeps
    the sites whose caught name the scan declares as an exception.
    """

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        symbols = enclosing_symbols(parsed_module.module)
        for node in ast.walk(parsed_module.module):
            match node:
                case ast.Try(body=[guarded], handlers=handlers) if any(
                    isinstance(child, ast.Call) for child in ast.walk(guarded)
                ):
                    pass
                case _:
                    continue
            spelled = ast.unparse(guarded).splitlines()[0]
            for handler in handlers:
                if all(isinstance(statement, ast.Raise) for statement in handler.body):
                    continue
                for name in handled_names(handler):
                    yield RestatingSite(
                        parsed_module.file_path,
                        handler.lineno,
                        symbols.get(id(node), "<module>"),
                        name,
                        (spelled,),
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
                    (node.name, fold.called_parameter),
                )


class ZeroOneTupleSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """A function whose every return is `()` or a one-element tuple.

    `detail` is the declaring class (empty for a module function) and the arities.
    """

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for definition in scoped_definitions(parsed_module.module):
            arities = absence_tuple_arities(definition.node)
            if not arities:
                continue
            owner = (
                definition.owner.name
                if isinstance(definition.owner, ast.ClassDef)
                else ""
            )
            yield RestatingSite(
                parsed_module.file_path,
                definition.node.lineno,
                definition.qualname,
                definition.node.name,
                (owner, *(str(arity) for arity in sorted(arities))),
            )


def optional_annotation(annotation: ast.expr | None) -> bool:
    match annotation:
        case ast.BinOp(op=ast.BitOr()):
            return any(
                isinstance(part, ast.Constant) and part.value is None
                for part in ast.walk(annotation)
            )
        case ast.Subscript(value=ast.Name(id="Optional") | ast.Attribute(attr="Optional")):
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
        if default is None
    )
    return undefaulted


class ThreadedAbsenceSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """An undefaulted `X | None` parameter that its own body compares with None."""

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for definition in scoped_definitions(parsed_module.module):
            optional = {
                argument.arg
                for argument in undefaulted_parameters(definition.node)
                if optional_annotation(argument.annotation)
            }
            for node in function_nodes(definition.node):
                match node:
                    case ast.Compare(
                        left=ast.Name(id=name),
                        ops=[ast.Is() | ast.IsNot() as operator],
                        comparators=[ast.Constant(value=None)],
                    ) if name in optional:
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
