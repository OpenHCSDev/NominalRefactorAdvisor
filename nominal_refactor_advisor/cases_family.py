"""An enum that declares behaviour becomes the class family its members describe.

``CasesFamily`` is the ``Cases`` input form with its own end state: the enum's
name stays as an abstract root, each member becomes a subclass carrying its data
as class attributes, and every method that read ``self`` to tell members apart
splits into one body per member, decided once by folding the member's literals
into its tests. Branches on member data outside the enum become calls of a new
family method; the names for those methods are questions the caller answers.

Members are classes, so identity (``status is Hit``), hashing and pickling keep
their meaning; ``value`` stays a class attribute so wire names survive.
"""

from __future__ import annotations

import ast
import builtins
import operator
import textwrap
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import cached_property

from .ast_tools import ParsedModule, PythonSourcePathPolicy
from .codemod_runtime import CodemodSourceSnapshot
from .descent import (
    Answers,
    Cases,
    Descent,
    EnumDeclaration,
    Member,
    MethodName,
    ModuleNames,
    Question,
    Rewrite,
    TailName,
)
from .semantic_descent import AuthorityClaim, SemanticAuthorityKind

LINE_LENGTH = 88
ITERATING_CALLS = frozenset({"tuple", "list", "set", "frozenset", "sorted"})
BOOLEAN_CALLS = frozenset({"isinstance", "issubclass", "bool", "callable", "hasattr"})
COMPARISONS: dict[type[ast.cmpop], Callable[[object, object], object]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}
IDENTITIES: dict[type[ast.cmpop], Callable[[object, object], object]] = {
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
INVERSE: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


class CallbackName(Question):
    @property
    def prompt(self) -> str:
        return "parameter name for the fall-through the family calls back"


class Unknown:
    """A value folding cannot decide from one member's literals."""


UNKNOWN = Unknown()


def handback(module: ParsedModule, node: ast.AST, reason: str) -> ValueError:
    return ValueError(f"handback: {module.file_path}:{node.lineno}: {reason}")


def body_without_docstring(statements: list[ast.stmt]) -> list[ast.stmt]:
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        return statements[1:]
    return statements


def docstring(statements: list[ast.stmt]) -> ast.Expr | None:
    if body_without_docstring(statements) is statements:
        return None
    first = statements[0]
    return first if isinstance(first, ast.Expr) else None


def decorator_names(function: ast.FunctionDef) -> tuple[str, ...]:
    return tuple(ast.unparse(decorator) for decorator in function.decorator_list)


def is_name(node: ast.AST | None, names: Iterable[str]) -> bool:
    return isinstance(node, ast.Name) and node.id in set(names)


def loaded_names(nodes: Iterable[ast.AST]) -> frozenset[str]:
    return frozenset(
        node.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )


def stored_names(nodes: Iterable[ast.AST]) -> frozenset[str]:
    return frozenset(
        node.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del)
    )


def module_bound_names(module: ast.Module) -> frozenset[str]:
    """Names one module binds at top level: definitions, imports and assignments."""

    names: set[str] = set()
    for statement in module.body:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(statement.name)
        elif isinstance(statement, ast.Import | ast.ImportFrom):
            names.update(
                (alias.asname or alias.name).split(".")[0] for alias in statement.names
            )
        elif isinstance(statement, ast.If):
            names.update(module_bound_names(ast.Module(statement.body, [])))
        else:
            names.update(stored_names([statement]))
    return frozenset(names)


def literal_constant(value: object) -> bool:
    return value is None or isinstance(value, bool | int | float | str)


def negated(node: ast.expr) -> ast.expr:
    """``not node`` read as a test, with one comparison inverted where exact."""

    if isinstance(node, ast.Constant):
        return ast.Constant(not node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return node.operand
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and type(node.ops[0]) in INVERSE
    ):
        return ast.Compare(
            node.left, [INVERSE[type(node.ops[0])]()], list(node.comparators)
        )
    return ast.UnaryOp(ast.Not(), node)


def rebuilt(node: ast.AST, renames: dict[int, str]) -> ast.AST:
    """A copy of ``node`` with each renamed node replaced by the source it becomes."""

    if id(node) in renames:
        return ast.parse(renames[id(node)], mode="eval").body
    fields: dict[str, object] = {}
    for name, value in ast.iter_fields(node):
        if isinstance(value, list):
            fields[name] = [
                rebuilt(item, renames) if isinstance(item, ast.AST) else item
                for item in value
            ]
        elif isinstance(value, ast.AST):
            fields[name] = rebuilt(value, renames)
        else:
            fields[name] = value
    return type(node)(**fields)


def is_boolean(node: ast.expr, boolean_names: frozenset[str]) -> bool:
    """Whether ``node`` evaluates to a ``bool``, read from its syntax alone."""

    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    if isinstance(node, ast.Constant):
        return isinstance(node.value, bool)
    if isinstance(node, ast.BoolOp):
        return all(is_boolean(value, boolean_names) for value in node.values)
    if isinstance(node, ast.Name):
        return node.id in boolean_names
    return isinstance(node, ast.Call) and is_name(node.func, BOOLEAN_CALLS)


def member_type(root: str, source: str) -> str:
    """Annotation ``source`` with ``root``, a type of members, read as ``type[root]``.

    String annotations are rewritten inside their quotes.
    """

    tree = ast.parse(source, mode="eval")
    lines = SourceLines(source)
    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    replacements: list[tuple[int, int, str]] = []
    for part in ast.walk(tree):
        if is_name(part, [root]):
            parent = parents.get(id(part))
            if isinstance(parent, ast.Subscript) and is_name(parent.value, ["type"]):
                continue
            replacements.append((*lines.offsets(part), f"type[{root}]"))
        elif isinstance(part, ast.Constant) and isinstance(part.value, str):
            inner = member_type(root, part.value)
            if inner != part.value:
                replacements.append((*lines.offsets(part), f'"{inner}"'))
    for start, end, text in sorted(replacements, reverse=True):
        source = source[:start] + text + source[end:]
    return source


@dataclass
class Folding:
    """One member's reading of a block: decided tests folded, every read renamed.

    ``literal`` answers a read the member decides; ``renames`` maps a node to the
    source it becomes. Tests are rebuilt from the AST, every other statement keeps
    its own source text.
    """

    module: ParsedModule
    rewrite: Rewrite
    literal: Callable[[ast.AST], object]
    renames: dict[int, str]
    boolean_names: frozenset[str]

    def expression(self, node: ast.expr) -> ast.expr:
        copy = rebuilt(node, self.renames)
        return copy if isinstance(copy, ast.expr) else node

    def decides(self, node: ast.AST) -> bool:
        return any(self.literal(part) is not UNKNOWN for part in ast.walk(node))

    def value(self, node: ast.expr) -> ast.expr:
        known = self.literal(node)
        if known is not UNKNOWN and literal_constant(known):
            return ast.Constant(known)
        if isinstance(node, ast.Compare):
            return self.compare(node)
        return self.expression(node)

    def truth(self, node: ast.expr) -> ast.expr:
        """``node`` read only for its truth, with every decided part folded."""

        known = self.literal(node)
        if known is not UNKNOWN:
            return ast.Constant(bool(known))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return negated(self.truth(node.operand))
        if isinstance(node, ast.BoolOp):
            conjunction = isinstance(node.op, ast.And)
            parts: list[ast.expr] = []
            for value in node.values:
                part = self.truth(value)
                if isinstance(part, ast.Constant):
                    if bool(part.value) != conjunction:
                        return ast.Constant(not conjunction)
                    continue
                parts.append(part)
            if not parts:
                return ast.Constant(conjunction)
            if len(parts) == 1:
                return parts[0]
            return ast.BoolOp(type(node.op)(), parts)
        if isinstance(node, ast.Compare):
            return self.compare(node)
        return self.expression(node)

    def compare(self, node: ast.Compare) -> ast.expr:
        operands = [self.value(node.left), *map(self.value, node.comparators)]
        if all(isinstance(operand, ast.Constant) for operand in operands):
            known = self.evaluated_chain(
                [
                    operand.value
                    for operand in operands
                    if isinstance(operand, ast.Constant)
                ],
                node.ops,
            )
            if known is not UNKNOWN:
                return ast.Constant(bool(known))
        if len(node.ops) == 1 and isinstance(
            node.ops[0], ast.Is | ast.IsNot | ast.Eq | ast.NotEq
        ):
            left, right = operands
            for constant, other in ((left, right), (right, left)):
                if (
                    isinstance(constant, ast.Constant)
                    and isinstance(constant.value, bool)
                    and is_boolean(other, self.boolean_names)
                ):
                    positive = (
                        isinstance(node.ops[0], ast.Is | ast.Eq) == constant.value
                    )
                    return other if positive else negated(other)
        return ast.Compare(operands[0], list(node.ops), operands[1:])

    @staticmethod
    def evaluated_chain(values: list[object], ops: list[ast.cmpop]) -> object:
        for left, op, right in zip(values, ops, values[1:]):
            if type(op) in IDENTITIES:
                if not (
                    (left is None or isinstance(left, bool))
                    and (right is None or isinstance(right, bool))
                ):
                    return UNKNOWN
                result = IDENTITIES[type(op)](left, right)
            else:
                try:
                    result = COMPARISONS[type(op)](left, right)
                except TypeError:
                    return UNKNOWN
            if not result:
                return False
        return True

    def block(self, statements: list[ast.stmt]) -> tuple[list[str], bool]:
        """Source lines of ``statements`` and whether every path through them ends."""

        lines: list[str] = []
        for statement in statements:
            emitted, ends = self.statement(statement)
            lines.extend(emitted)
            if ends:
                return lines, True
        return lines, False

    def statement(self, statement: ast.stmt) -> tuple[list[str], bool]:
        if isinstance(statement, ast.If) and self.decides(statement):
            return self.chain(statement, "if")
        source = self.rewrite.statements_source(
            self.module,
            [statement],
            [
                (node, self.renames[id(node)])
                for node in outermost(statement, self.renames)
            ],
        )
        return source.rstrip("\n").split("\n"), isinstance(
            statement, ast.Return | ast.Raise
        )

    def chain(self, statement: ast.If, keyword: str) -> tuple[list[str], bool]:
        test = self.truth(statement.test)
        if isinstance(test, ast.Constant):
            chosen = statement.body if test.value else statement.orelse
            if not test.value and len(chosen) == 1 and isinstance(chosen[0], ast.If):
                # An ``elif`` node's source starts at ``elif``: never slice it.
                return self.chain(chosen[0], keyword)
            if keyword == "if":
                return self.block(chosen)
            lines, ends = self.block(chosen)
            return (["else:", *indented(lines)], ends) if lines else ([], False)
        body, body_ends = self.block(statement.body)
        lines = [f"{keyword} {ast.unparse(test)}:", *indented(body or ["pass"])]
        if not statement.orelse:
            return lines, False
        if len(statement.orelse) == 1 and isinstance(statement.orelse[0], ast.If):
            rest, rest_ends = self.chain(statement.orelse[0], "elif")
            return lines + rest, body_ends and rest_ends
        orelse, orelse_ends = self.block(statement.orelse)
        if not orelse:
            return lines, False
        return lines + ["else:", *indented(orelse)], body_ends and orelse_ends


def indented(lines: list[str], prefix: str = "    ") -> list[str]:
    return [f"{prefix}{line}" if line else line for line in lines]


def outermost(root: ast.AST, renames: dict[int, str]) -> Iterator[ast.AST]:
    """The renamed nodes under ``root`` that no other renamed node encloses."""

    pending = [root]
    while pending:
        node = pending.pop()
        if id(node) in renames:
            yield node
            continue
        pending.extend(ast.iter_child_nodes(node))


def parameter_sources(
    arguments: ast.arguments, first: str | None, text: Callable[[ast.expr], str]
) -> list[str]:
    """Each parameter as Black renders it, the first renamed to ``first``.

    ``text`` renders an annotation or default from its source.
    """

    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(arguments.defaults)
    )
    defaults.extend(arguments.defaults)
    sources: list[str] = []
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        name = first if index == 0 and first is not None else argument.arg
        sources.append(argument_source(name, argument.annotation, default, text))
        if index + 1 == len(arguments.posonlyargs):
            sources.append("/")
    if arguments.vararg is not None:
        sources.append(
            "*"
            + argument_source(
                arguments.vararg.arg, arguments.vararg.annotation, None, text
            )
        )
    elif arguments.kwonlyargs:
        sources.append("*")
    for argument, kw_default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        sources.append(
            argument_source(argument.arg, argument.annotation, kw_default, text)
        )
    if arguments.kwarg is not None:
        sources.append(
            "**"
            + argument_source(
                arguments.kwarg.arg, arguments.kwarg.annotation, None, text
            )
        )
    return sources


def argument_source(
    name: str,
    annotation: ast.expr | None,
    default: ast.expr | None,
    text: Callable[[ast.expr], str],
) -> str:
    if annotation is None:
        return name if default is None else f"{name}={text(default)}"
    source = f"{name}: {text(annotation)}"
    return source if default is None else f"{source} = {text(default)}"


def segment(module: ParsedModule, node: ast.expr) -> str:
    """``node``'s own source on one line, else its unparsed form."""

    found = ast.get_source_segment(module.source, node)
    return found if found is not None and "\n" not in found else ast.unparse(node)


def signature(
    name: str, parameters: list[str], returns: str | None, indent: str
) -> list[str]:
    """``def`` header lines, exploded one parameter per line past the line length."""

    suffix = "" if returns is None else f" -> {returns}"
    line = f"def {name}({', '.join(parameters)}){suffix}:"
    if len(indent) + len(line) <= LINE_LENGTH:
        return [line]
    return [
        f"def {name}(",
        *(f"    {parameter}," for parameter in parameters),
        f"){suffix}:",
    ]


@dataclass(frozen=True)
class Datum:
    """One attribute every member class declares: its name, annotation and store."""

    name: str
    annotation: str | None
    stored: str


@dataclass
class MethodBodies:
    """One family method: its header, each member's folded body and the unfolded one.

    The root takes the body most members share and each other member overrides
    it, unless two members would carry the same override: that copy restates a
    sibling, so the root keeps the unfolded body, reading each member's own data,
    and no member overrides.
    """

    name: str
    header: list[str]
    bodies: dict[str, list[str]]
    generic: list[str]
    docstring: list[str] = field(default_factory=list)

    def root_body(self, members: tuple[Member, ...]) -> list[str]:
        counts = Counter(tuple(self.bodies[member.name]) for member in members)
        most = max(counts.values())
        root = next(
            self.bodies[member.name]
            for member in members
            if counts[tuple(self.bodies[member.name])] == most
        )
        if any(count > 1 for body, count in counts.items() if list(body) != root):
            return self.generic
        return root

    def override(self, member: Member, members: tuple[Member, ...]) -> list[str] | None:
        root = self.root_body(members)
        body = self.bodies[member.name]
        return None if root is self.generic or body == root else body

    def source(self, body: list[str], with_docstring: bool) -> list[str]:
        lines = [
            *self.header,
            *indented(
                [*(self.docstring if with_docstring else []), *(body or ["pass"])]
            ),
        ]
        return indented(lines)


@dataclass
class CaseModel:
    """The enum read as a family: per-member data, literals and folded properties."""

    enum: EnumDeclaration
    data: list[Datum]
    expressions: dict[str, dict[str, ast.expr]]
    literals: dict[str, dict[str, object]]
    folded: dict[str, str | None]
    forwarded: dict[str, str]
    called: frozenset[str]

    @property
    def root(self) -> str:
        return self.enum.name

    @property
    def members(self) -> tuple[Member, ...]:
        return self.enum.members

    @cached_property
    def class_names(self) -> dict[str, str]:
        return {member.name: member.class_name for member in self.members}

    @property
    def decided_names(self) -> frozenset[str]:
        """Attributes every member answers with a literal: reads a fold decides."""

        return frozenset(
            name
            for name in (*(datum.name for datum in self.data), *self.folded)
            if all(name in self.literals[member.name] for member in self.members)
        )

    def attribute_name(self, attribute: str) -> str:
        return self.forwarded.get(attribute, attribute)


class CasesFamily(Cases, Descent):
    """An enum that declares behaviour, read through its members' data and identity.

    The enum's name becomes an abstract root; members become its subclasses.
    """

    def admitted(self, enum: EnumDeclaration) -> EnumDeclaration:
        """Every method has a place: on the root, or split across the members."""

        return enum

    def claims(self, snapshot: CodemodSourceSnapshot) -> tuple[AuthorityClaim, ...]:
        enum = self.enum(snapshot)
        return (
            AuthorityClaim(
                claimed_symbol=enum.name,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                file_path=enum.module.file_path,
                qualname=enum.name,
            ),
        )

    def rewrite(self, snapshot: CodemodSourceSnapshot) -> Rewrite:
        enum = self.enum(snapshot)
        model = self.model(enum, snapshot)
        rewrite = Rewrite(snapshot)
        answer = Answers(self.names)
        aliases = self.aliases(snapshot, enum, enum.module, enum.name)
        self.require_free_class_names(model)
        methods = [self.method(model, rewrite, function) for function in enum.methods]
        handled: set[int] = {id(node) for node in ast.walk(enum.node)}
        for site in self.sites(snapshot, model, aliases, answer):
            methods.append(site.install(self, model, rewrite, answer, handled))
        rewrite.replace(
            enum.module, enum.node, self.family_source(model, rewrite, methods)
        )
        for module, names in aliases.items():
            self.rewrite_reads(rewrite, module, names, model, handled)
        self.rewrite_called(rewrite, snapshot, model, handled)
        answer.require_complete()
        self.ensure_import(rewrite, enum.module, "abc", "ABC")
        for module, names in aliases.items():
            used = tuple(sorted(rewrite.used_members[module.file_path]))
            replaced = {
                local: tuple(sorted({local, *used}))
                for local in names.enum
                if module is not enum.module and used
            }
            rewrite.prune_imports(module, replaced)
        return rewrite

    # -- the enum read as a family -------------------------------------------------

    def model(
        self, enum: EnumDeclaration, snapshot: CodemodSourceSnapshot
    ) -> CaseModel:
        module = enum.module
        if enum.node.decorator_list or enum.node.keywords:
            raise handback(
                module,
                enum.node,
                f"enum {enum.name} is decorated or takes class keywords",
            )
        for statement in body_without_docstring(enum.node.body):
            if isinstance(statement, ast.AsyncFunctionDef):
                raise handback(
                    module, statement, "async enum method; no family form derived"
                )
            if not isinstance(
                statement, ast.Assign | ast.AnnAssign | ast.FunctionDef
            ) or (isinstance(statement, ast.AnnAssign) and statement.value is not None):
                raise handback(
                    module,
                    statement,
                    "enum body statement is neither member, field, nor method",
                )
            if isinstance(statement, ast.Assign) and not all(
                isinstance(argument, ast.expr)
                for member in enum.members
                for argument in member.arguments
            ):
                raise handback(module, statement, "enum member is not a value tuple")
        data, expressions = self.data(enum)
        literals = {
            member.name: {
                name: value
                for name, expression in expressions[member.name].items()
                if (value := literal_value(expression)) is not UNKNOWN
            }
            for member in enum.members
        }
        forwarded = self.forwarded(enum, data)
        data = [
            Datum(
                forwarded.get(datum.stored, datum.name), datum.annotation, datum.stored
            )
            for datum in data
        ]
        for member in enum.members:
            literals[member.name] = {
                forwarded.get(name, name): value
                for name, value in literals[member.name].items()
            }
        model = CaseModel(enum, data, expressions, literals, {}, forwarded, frozenset())
        self.fold_properties(model)
        model.called = frozenset(
            function.name
            for function in enum.methods
            if isinstance(function, ast.FunctionDef)
            and decorator_names(function) == ("property",)
            and function.name not in model.folded
            and function.name not in forwarded.values()
        )
        return model

    def data(
        self, enum: EnumDeclaration
    ) -> tuple[list[Datum], dict[str, dict[str, ast.expr]]]:
        """``value`` then each field ``__new__`` stores, with every member's expression."""

        module = enum.module
        annotations = {
            statement.target.id: ast.unparse(statement.annotation)
            for statement in enum.node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        }
        new = enum.new
        if new is None:
            if not all(len(member.arguments) == 1 for member in enum.members):
                raise handback(module, enum.node, "tuple members without __new__")
            return [Datum("value", "str", "value")], {
                member.name: {"value": member.arguments[0]} for member in enum.members
            }
        body = new.body
        parameters = [argument.arg for argument in new.args.args[1:]]
        stores: dict[str, str] = {}
        if not (
            len(body) >= 2
            and isinstance(body[0], ast.Assign)
            and len(body[0].targets) == 1
            and isinstance(body[0].targets[0], ast.Name)
            and isinstance(body[-1], ast.Return)
            and is_name(body[-1].value, [body[0].targets[0].id])
        ):
            raise handback(module, new, "__new__ is not create, store fields, return")
        created = body[0].targets[0].id
        for statement in body[1:-1]:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Attribute)
                and is_name(statement.targets[0].value, [created])
                and is_name(statement.value, parameters)
                and isinstance(statement.value, ast.Name)
            ):
                raise handback(
                    module, statement, "__new__ statement does not store one parameter"
                )
            stores[statement.value.id] = statement.targets[0].attr
        if set(stores) != set(parameters) or "_value_" not in stores.values():
            raise handback(
                module,
                new,
                "__new__ does not store every parameter once, value included",
            )
        value_parameter = next(
            name for name, attribute in stores.items() if attribute == "_value_"
        )
        if parameters[0] != value_parameter:
            raise handback(
                module, new, "the member value is not __new__'s first parameter"
            )
        annotations_by_parameter = {
            argument.arg: (
                None
                if argument.annotation is None
                else ast.unparse(argument.annotation)
            )
            for argument in new.args.args[1:]
        }
        data = [Datum("value", annotations_by_parameter[value_parameter], "value")]
        data.extend(
            Datum(
                stores[parameter],
                annotations.get(stores[parameter], annotations_by_parameter[parameter]),
                stores[parameter],
            )
            for parameter in parameters[1:]
        )
        expressions: dict[str, dict[str, ast.expr]] = {}
        for member in enum.members:
            fields = dict(member.fields)
            values: dict[str, ast.expr] = {"value": member.arguments[0]}
            for parameter in parameters[1:]:
                expression = fields[parameter]
                if expression is None:
                    raise handback(
                        module,
                        member.node,
                        f"member {member.name} leaves {parameter} unset",
                    )
                values[stores[parameter]] = expression
            expressions[member.name] = values
        return data, expressions

    @staticmethod
    def forwarded(enum: EnumDeclaration, data: list[Datum]) -> dict[str, str]:
        """Stored field -> the property that only returns it, which takes its name."""

        stored = {datum.stored for datum in data}
        forwarded: dict[str, str] = {}
        for function in enum.methods:
            if not (
                isinstance(function, ast.FunctionDef)
                and decorator_names(function) == ("property",)
            ):
                continue
            body = body_without_docstring(function.body)
            receiver = function.args.args[0].arg
            if (
                len(body) == 1
                and isinstance(body[0], ast.Return)
                and isinstance(body[0].value, ast.Attribute)
                and is_name(body[0].value.value, [receiver])
                and body[0].value.attr in stored
                and body[0].value.attr not in forwarded
            ):
                forwarded[body[0].value.attr] = function.name
        return forwarded

    def fold_properties(self, model: CaseModel) -> None:
        """Properties whose one return every member answers with a literal become data."""

        pending = [
            function
            for function in model.enum.methods
            if isinstance(function, ast.FunctionDef)
            and decorator_names(function) == ("property",)
            and function.name not in model.forwarded.values()
        ]
        progressed = True
        while progressed:
            progressed = False
            for function in list(pending):
                body = body_without_docstring(function.body)
                if (
                    len(body) != 1
                    or not isinstance(body[0], ast.Return)
                    or body[0].value is None
                ):
                    pending.remove(function)
                    continue
                receiver = function.args.args[0].arg
                values = {
                    member.name: self.evaluated(model, member, receiver, body[0].value)
                    for member in model.members
                }
                if any(value is UNKNOWN for value in values.values()):
                    continue
                for member_name, value in values.items():
                    model.literals[member_name][function.name] = value
                model.folded[function.name] = (
                    None if function.returns is None else ast.unparse(function.returns)
                )
                pending.remove(function)
                progressed = True

    def evaluated(
        self, model: CaseModel, member: Member, receiver: str, node: ast.expr
    ) -> object:
        """``node``'s exact value for ``member``, or UNKNOWN."""

        known = self.member_literal(model, member, receiver, node)
        if known is not UNKNOWN:
            return known
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            inner = self.evaluated(model, member, receiver, node.operand)
            return UNKNOWN if inner is UNKNOWN else not inner
        if isinstance(node, ast.BoolOp):
            result: object = UNKNOWN
            for value in node.values:
                result = self.evaluated(model, member, receiver, value)
                if result is UNKNOWN:
                    return UNKNOWN
                if bool(result) != isinstance(node.op, ast.And):
                    return result
            return result
        if isinstance(node, ast.Compare):
            values = [
                self.evaluated(model, member, receiver, operand)
                for operand in (node.left, *node.comparators)
            ]
            if any(value is UNKNOWN for value in values):
                return UNKNOWN
            return Folding.evaluated_chain(values, node.ops)
        return UNKNOWN

    def member_literal(
        self, model: CaseModel, member: Member, receiver: str, node: ast.AST
    ) -> object:
        """What ``member`` answers for a read of itself through ``receiver``."""

        if (
            isinstance(node, ast.Attribute)
            and is_name(node.value, [receiver])
            and isinstance(node.ctx, ast.Load)
        ):
            name = model.attribute_name(node.attr)
            return model.literals[member.name].get(name, UNKNOWN)
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and is_name(node.left, [receiver])
        ):
            members = self.member_refs(model, node.comparators[0], receiver)
            if members is None:
                return UNKNOWN
            operation = node.ops[0]
            if isinstance(operation, ast.Is | ast.Eq | ast.In):
                return member.name in members
            if isinstance(operation, ast.IsNot | ast.NotEq | ast.NotIn):
                return member.name not in members
        return UNKNOWN

    @staticmethod
    def member_refs(
        model: CaseModel, node: ast.AST, receiver: str
    ) -> frozenset[str] | None:
        """Members ``node`` names: ``E.X``, ``cls.X`` or ``self.X``, alone or in a literal."""

        elements = (
            node.elts if isinstance(node, ast.Tuple | ast.List | ast.Set) else [node]
        )
        names = [
            element.attr
            for element in elements
            if isinstance(element, ast.Attribute)
            and is_name(element.value, [model.root, receiver, "cls"])
            and element.attr in model.class_names
        ]
        return frozenset(names) if len(names) == len(elements) else None

    @staticmethod
    def require_free_class_names(model: CaseModel) -> None:
        bound = module_bound_names(model.enum.module.module)
        for member in model.members:
            if member.class_name in bound:
                raise handback(
                    model.enum.module,
                    member.node,
                    f"{member.class_name} is already bound in the enum's module",
                )

    # -- the enum's own methods ----------------------------------------------------

    def method(
        self,
        model: CaseModel,
        rewrite: Rewrite,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> MethodBodies | None:
        """One enum method on the family; ``None`` for a property that became data."""

        module = model.enum.module
        if not isinstance(function, ast.FunctionDef):
            raise handback(module, function, "async enum method")
        decorators = decorator_names(function)
        if function.name in model.folded or function.name in model.forwarded.values():
            return None
        if decorators not in ((), ("property",), ("classmethod",), ("staticmethod",)):
            raise handback(module, function, f"enum method decorated {decorators}")
        static = decorators == ("staticmethod",)
        receiver = None if static else function.args.args[0].arg
        first = None if static else "cls"
        header_decorator = ["@staticmethod"] if static else ["@classmethod"]
        parameters = parameter_sources(
            function.args,
            first,
            lambda node: member_type(model.root, segment(module, node)),
        )
        returns = (
            None
            if function.returns is None
            else member_type(model.root, segment(module, function.returns))
        )
        header = [
            *header_decorator,
            *signature(function.name, parameters, returns, "    "),
        ]
        statements = body_without_docstring(function.body)
        documentation = docstring(function.body)
        renames = self.method_renames(
            model, function, receiver, decorators == ("classmethod",)
        )
        bodies: dict[str, list[str]] = {}
        for member in model.members:
            folding = Folding(
                module,
                rewrite,
                (
                    (
                        lambda node, member=member: self.member_literal(
                            model, member, receiver, node
                        )
                    )
                    if receiver is not None and decorators != ("classmethod",)
                    else (lambda node: UNKNOWN)
                ),
                renames,
                self.boolean_names(function),
            )
            bodies[member.name], _ends = folding.block(statements)
        generic, _ends = Folding(
            module,
            rewrite,
            lambda node: UNKNOWN,
            renames,
            self.boolean_names(function),
        ).block(statements)
        return MethodBodies(
            function.name,
            header,
            bodies,
            generic,
            (
                []
                if documentation is None
                else rewrite.statements_source(module, [documentation])
                .rstrip("\n")
                .split("\n")
            ),
        )

    def method_renames(
        self,
        model: CaseModel,
        function: ast.FunctionDef,
        receiver: str | None,
        is_classmethod: bool,
    ) -> dict[int, str]:
        """Member references become classes, the receiver ``cls``, and reads of
        properties now methods become calls."""

        renames: dict[int, str] = {}
        parents = {
            id(child): parent
            for parent in ast.walk(function)
            for child in ast.iter_child_nodes(parent)
        }
        for statement in function.body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    if (
                        is_name(
                            node.value, [model.root, *([receiver] if receiver else [])]
                        )
                        and node.attr in model.class_names
                    ):
                        renames[id(node)] = model.class_names[node.attr]
                        continue
                    if receiver is not None and is_name(node.value, [receiver]):
                        name = model.attribute_name(node.attr)
                        parent = parents.get(id(node))
                        called = name in model.called and not (
                            isinstance(parent, ast.Call) and parent.func is node
                        )
                        if name != node.attr or called:
                            renames[id(node)] = f"cls.{name}" + ("()" if called else "")
                        continue
                if (
                    isinstance(node, ast.Name)
                    and receiver is not None
                    and node.id == receiver
                    and id(node) not in renames
                ):
                    parent = parents.get(id(node))
                    if is_classmethod and self.iterated(node, parent):
                        renames[id(node)] = f"{model.root}.__subclasses__()"
                    elif not is_classmethod and isinstance(node.ctx, ast.Load):
                        if not (
                            isinstance(parent, ast.Attribute) and id(parent) in renames
                        ):
                            renames[id(node)] = "cls"
                    elif (
                        is_classmethod
                        and isinstance(parent, ast.Call)
                        and parent.func is node
                    ):
                        raise handback(
                            model.enum.module, node, "enum looked up by value"
                        )
        return renames

    @staticmethod
    def iterated(node: ast.AST, parent: ast.AST | None) -> bool:
        if isinstance(parent, ast.For | ast.comprehension) and parent.iter is node:
            return True
        return (
            isinstance(parent, ast.Call)
            and is_name(parent.func, ITERATING_CALLS)
            and parent.args == [node]
        )

    def boolean_names(self, function: ast.FunctionDef) -> frozenset[str]:
        """Parameters annotated ``bool`` and locals bound once to a boolean expression."""

        names = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
            if is_name(argument.annotation, ["bool"])
        }
        assignments: dict[str, list[ast.expr]] = {}
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                assignments.setdefault(node.targets[0].id, []).append(node.value)
        for name, values in assignments.items():
            if len(values) == 1 and is_boolean(values[0], frozenset(names)):
                names.add(name)
        return frozenset(names)

    # -- branches on member data outside the enum ----------------------------------

    def sites(
        self,
        snapshot: CodemodSourceSnapshot,
        model: CaseModel,
        aliases: dict[ParsedModule, ModuleNames],
        answer: Answers,
    ) -> list[Site]:
        """Every production block whose tests read member data through a field."""

        found: list[Site] = []
        decided = model.decided_names
        for module in sorted(aliases, key=lambda module: module.file_path):
            if PythonSourcePathPolicy.is_test_path(module.path):
                continue
            names = aliases[module]
            for owner in module.module.body:
                if not isinstance(owner, ast.ClassDef) or owner.name == model.root:
                    continue
                fields = {
                    statement.target.id: statement
                    for statement in owner.body
                    if isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                }
                held = {
                    name
                    for name, statement in fields.items()
                    if is_name(statement.annotation, names.enum)
                }
                for function in owner.body:
                    if not isinstance(function, ast.FunctionDef):
                        continue
                    self.refuse_member_comparisons(module, function, model, names)
                    if (
                        not held
                        or "staticmethod" in decorator_names(function)
                        or not function.args.args
                    ):
                        continue
                    receiver = function.args.args[0].arg
                    found.extend(
                        self.function_sites(
                            module, owner, function, receiver, held, fields, decided
                        )
                    )
        for site in found:
            site.ask(answer)
        return found

    def refuse_member_comparisons(
        self,
        module: ParsedModule,
        function: ast.FunctionDef,
        model: CaseModel,
        names: ModuleNames,
    ) -> None:
        for node in ast.walk(function):
            if isinstance(node, ast.If | ast.While | ast.IfExp | ast.match_case):
                tests = (
                    [node.guard] if isinstance(node, ast.match_case) else [node.test]
                )
                for test in tests:
                    for part in ast.walk(test) if test is not None else ():
                        if (
                            isinstance(part, ast.Attribute)
                            and is_name(part.value, names.enum)
                            and part.attr in model.class_names
                        ):
                            raise handback(
                                module,
                                part,
                                f"member {part.attr} compared outside the family",
                            )

    def function_sites(
        self,
        module: ParsedModule,
        owner: ast.ClassDef,
        function: ast.FunctionDef,
        receiver: str,
        held: set[str],
        fields: dict[str, ast.AnnAssign],
        decided: frozenset[str],
    ) -> Iterator[Site]:
        body = body_without_docstring(function.body)

        def reads(statement: ast.stmt) -> str | None:
            for node in ast.walk(statement):
                tests = (
                    [node.test]
                    if isinstance(node, ast.If | ast.While | ast.IfExp)
                    else []
                )
                for test in tests:
                    for part in ast.walk(test):
                        if (
                            isinstance(part, ast.Attribute)
                            and part.attr in decided
                            and isinstance(part.value, ast.Attribute)
                            and part.value.attr in held
                            and is_name(part.value.value, [receiver])
                        ):
                            return part.value.attr
            return None

        index = 0
        while index < len(body):
            held_field = reads(body[index])
            if held_field is None:
                index += 1
                continue
            if not isinstance(body[index], ast.If):
                raise handback(
                    module,
                    body[index],
                    "member data decided inside a compound statement",
                )
            end = index + 1
            while (
                end < len(body)
                and isinstance(body[end], ast.If)
                and reads(body[end]) == held_field
            ):
                end += 1
            # A returning run moves whole: its ifs end the function together.
            # Otherwise each if decides its own fact and becomes its own method,
            # so no override repeats a sibling's answer to another decision.
            returning = any(
                isinstance(node, ast.Return)
                for statement in body[index:end]
                for node in ast.walk(statement)
            )
            for start, stop in (
                [(index, end)]
                if returning
                else [(position, position + 1) for position in range(index, end)]
            ):
                yield Site.decoded(
                    module,
                    owner,
                    function,
                    receiver,
                    held_field,
                    fields,
                    body,
                    start,
                    stop,
                )
            index = end

    # -- the family source ---------------------------------------------------------

    def family_source(
        self, model: CaseModel, rewrite: Rewrite, methods: list[MethodBodies | None]
    ) -> str:
        enum = model.enum
        module = enum.module
        rewrite_methods = [method for method in methods if method is not None]
        documentation = docstring(enum.node.body)
        root: list[str] = [f"class {model.root}(ABC):"]
        if documentation is not None:
            root.extend(
                indented(
                    rewrite.statements_source(module, [documentation])
                    .rstrip("\n")
                    .split("\n")
                )
            )
            root.append("")
        annotated = [
            *((datum.name, datum.annotation) for datum in model.data),
            *model.folded.items(),
        ]
        root.extend(
            f"    {name}: {member_type(model.root, annotation or 'object')}"
            for name, annotation in annotated
        )
        for method in rewrite_methods:
            root.append("")
            root.extend(method.source(method.root_body(model.members), True))
        classes = ["\n".join(root)]
        for member in model.members:
            lines = [f"class {member.class_name}({model.root}):"]
            for datum in model.data:
                expression = model.expressions[member.name][datum.stored]
                lines.extend(indented(self.assignment(module, datum.name, expression)))
            for name in model.folded:
                lines.append(f"    {name} = {model.literals[member.name][name]!r}")
            for method in rewrite_methods:
                body = method.override(member, model.members)
                if body is not None:
                    lines.append("")
                    lines.extend(method.source(body, False))
            classes.append("\n".join(lines))
        return "\n\n\n".join(classes)

    @staticmethod
    def assignment(module: ParsedModule, name: str, expression: ast.expr) -> list[str]:
        segment = ast.get_source_segment(module.source, expression)
        if segment is None:
            segment = ast.unparse(expression)
        first, *rest = segment.split("\n")
        return (
            [f"{name} = {first}", *textwrap.dedent("\n".join(rest)).split("\n")]
            if rest
            else [f"{name} = {first}"]
        )

    # -- reads of the enum everywhere else -----------------------------------------

    def rewrite_reads(
        self,
        rewrite: Rewrite,
        module: ParsedModule,
        names: ModuleNames,
        model: CaseModel,
        handled: set[int],
    ) -> None:
        """Member references, annotations and iteration; the root keeps the enum's name."""

        parents = {
            id(child): parent
            for parent in ast.walk(module.module)
            for child in ast.iter_child_nodes(parent)
        }
        annotations = {
            id(node): annotation
            for annotated in ast.walk(module.module)
            for annotation in annotations_of(annotated)
            for node in ast.walk(annotation)
        }
        bound = module_bound_names(module.module)
        local_names = set(names.enum)
        if module is model.enum.module:
            local_names.add(model.root)
        for node in ast.walk(module.module):
            if id(node) in handled:
                continue
            if (
                id(node) in annotations
                and isinstance(node, ast.Constant)
                and isinstance(node.value, str)
            ):
                text = (
                    member_type(model.root, node.value)
                    if model.root in local_names
                    else node.value
                )
                if text != node.value:
                    rewrite.replace(module, node, f'"{text}"')
                    handled.add(id(node))
                continue
            if not is_name(node, local_names):
                continue
            parent = parents.get(id(node))
            if (
                isinstance(parent, ast.Attribute)
                and parent.attr in model.class_names
                and id(parent) not in handled
            ):
                member_class = model.class_names[parent.attr]
                if module is not model.enum.module and member_class in bound:
                    raise handback(
                        module,
                        parent,
                        f"{member_class} is already bound in this module",
                    )
                rewrite.replace(module, parent, member_class)
                if module is not model.enum.module:
                    rewrite.used_members[module.file_path].add(member_class)
                handled.add(id(parent))
                continue
            if id(node) in annotations:
                if isinstance(parent, ast.Subscript) and is_name(
                    parent.value, ["type"]
                ):
                    continue
                rewrite.replace(module, node, f"type[{node.id}]")
                rewrite.annotations_in.add(module.file_path)
                continue
            if self.iterated(node, parent):
                rewrite.replace(module, node, f"{node.id}.__subclasses__()")
                continue
            if isinstance(parent, ast.Call) and (
                parent.func is node
                or (is_name(parent.func, ["isinstance", "issubclass"]))
            ):
                raise handback(
                    module,
                    node,
                    f"enum {node.id} called or instance-checked; members are classes now",
                )

    def rewrite_called(
        self,
        rewrite: Rewrite,
        snapshot: CodemodSourceSnapshot,
        model: CaseModel,
        handled: set[int],
    ) -> None:
        """A property that is not data per member becomes a method; its reads, calls.

        Reads are matched by attribute name, so no other class may declare the name.
        """

        if not model.called:
            return
        for module in snapshot.parsed_modules:
            for node in ast.walk(module.module):
                if (
                    isinstance(node, ast.ClassDef)
                    and node is not model.enum.node
                    and any(
                        isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
                        and statement.name in model.called
                        or isinstance(statement, ast.AnnAssign | ast.Assign)
                        and stored_names([statement]) & model.called
                        for statement in node.body
                    )
                ):
                    raise handback(
                        module,
                        node,
                        f"{node.name} also declares one of {sorted(model.called)}",
                    )
        for module in snapshot.parsed_modules:
            parents = {
                id(child): parent
                for parent in ast.walk(module.module)
                for child in ast.iter_child_nodes(parent)
            }
            for node in ast.walk(module.module):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in model.called
                    and isinstance(node.ctx, ast.Load)
                    and id(node) not in handled
                ):
                    parent = parents.get(id(node))
                    if isinstance(parent, ast.Call) and parent.func is node:
                        raise handback(module, node, f"{node.attr} is already called")
                    end = rewrite.geometry(module).required_node_offsets(node)[1]
                    rewrite.span(module, end, end, "()")
                    handled.add(id(node))

    def ensure_import(
        self, rewrite: Rewrite, module: ParsedModule, source: str, name: str
    ) -> None:
        """Extend the module's absolute from-import of ``source``, or add one."""

        statements = [
            statement
            for statement in module.module.body
            if isinstance(statement, ast.ImportFrom)
            and statement.module == source
            and statement.level == 0
        ]
        if any(
            alias.name == name and alias.asname is None
            for statement in statements
            for alias in statement.names
        ):
            return
        if statements:
            rewrite.require_import(module, source, name)
            return
        # Ruff's order: from-imports by module, after plain imports.
        from_imports = [
            statement
            for statement in module.module.body
            if isinstance(statement, ast.ImportFrom)
            and statement.level == 0
            and statement.module != "__future__"
        ]
        earlier = [
            statement
            for statement in from_imports
            if (statement.module or "").lower() < source.lower()
        ]
        geometry = rewrite.geometry(module)
        if earlier:
            offset = geometry.line_offsets[earlier[-1].end_lineno]
        else:
            anchor = next(
                iter(from_imports),
                next(
                    statement
                    for statement in module.module.body
                    if isinstance(statement, ast.Import | ast.ImportFrom)
                    and not (
                        isinstance(statement, ast.ImportFrom)
                        and statement.module == "__future__"
                    )
                ),
            )
            offset = geometry.line_offsets[anchor.lineno - 1]
        rewrite.span(module, offset, offset, f"from {source} import {name}\n")


def literal_value(expression: ast.expr) -> object:
    try:
        value = ast.literal_eval(expression)
    except ValueError:
        return UNKNOWN
    return value if literal_constant(value) else UNKNOWN


def annotations_of(node: ast.AST) -> tuple[ast.expr, ...]:
    if isinstance(node, ast.arg):
        return () if node.annotation is None else (node.annotation,)
    if isinstance(node, ast.AnnAssign):
        return (node.annotation,)
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return () if node.returns is None else (node.returns,)
    return ()


class SourceLines:
    """Offsets of nodes parsed from one source string."""

    def __init__(self, source: str) -> None:
        self.starts = [0]
        for line in source.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line))
        self.source = source

    def offsets(self, node: ast.AST) -> tuple[int, int]:
        start = self.starts[node.lineno - 1] + len(
            self.source[self.starts[node.lineno - 1] :]
            .encode()[: node.col_offset]
            .decode()
        )
        end = self.starts[node.end_lineno - 1] + len(
            self.source[self.starts[node.end_lineno - 1] :]
            .encode()[: node.end_col_offset]
            .decode()
        )
        return start, end


@dataclass
class Site:
    """A run of a reader's ``if`` statements deciding member data through one field."""

    module: ParsedModule
    owner: ast.ClassDef
    function: ast.FunctionDef
    receiver: str
    field_name: str
    fields: dict[str, ast.AnnAssign]
    body: list[ast.stmt]
    start: int
    end: int
    method: str = ""

    @classmethod
    def decoded(
        cls,
        module: ParsedModule,
        owner: ast.ClassDef,
        function: ast.FunctionDef,
        receiver: str,
        field_name: str,
        fields: dict[str, ast.AnnAssign],
        body: list[ast.stmt],
        start: int,
        end: int,
    ) -> Site:
        """A lone ``if ...: return`` with statements after it is a guard; else a block."""

        statement = body[start]
        if (
            end == start + 1
            and end < len(body)
            and isinstance(statement, ast.If)
            and not statement.orelse
            and len(statement.body) == 1
            and isinstance(statement.body[0], ast.Return)
            and literal_value(statement.body[0].value or ast.Constant(None)) is None
        ):
            return Guard(
                module, owner, function, receiver, field_name, fields, body, start, end
            )
        return Block(
            module, owner, function, receiver, field_name, fields, body, start, end
        )

    @property
    def statements(self) -> list[ast.stmt]:
        return self.body[self.start : self.end]

    @property
    def is_last(self) -> bool:
        return self.end == len(self.body)

    @property
    def node(self) -> ast.stmt:
        return self.body[self.start]

    def ask(self, answer: Answers) -> None:
        self.method = answer(MethodName(self.module, self.node))

    def field_read(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == self.field_name
            and is_name(node.value, [self.receiver])
        )

    def literal(self, model: CaseModel, member: Member) -> Callable[[ast.AST], object]:
        def read(node: ast.AST) -> object:
            if isinstance(node, ast.Attribute) and self.field_read(node.value):
                return model.literals[member.name].get(node.attr, UNKNOWN)
            return UNKNOWN

        return read

    def install(
        self,
        family: CasesFamily,
        model: CaseModel,
        rewrite: Rewrite,
        answer: Answers,
        handled: set[int],
    ) -> MethodBodies:
        raise NotImplementedError

    def resolvable(self, model: CaseModel, annotation: str) -> str:
        """``annotation``, if every name in it is bound where the family lives."""

        bound = (
            module_bound_names(model.enum.module.module)
            | {model.root}
            | set(dir(builtins))
        )
        missing = sorted(loaded_names([ast.parse(annotation, mode="eval")]) - bound)
        if missing:
            raise handback(
                self.module,
                self.node,
                f"annotation {annotation!r} names {missing}, unbound in the family's module",
            )
        return member_type(model.root, annotation)

    def replace_statements(
        self,
        rewrite: Rewrite,
        statements: list[ast.stmt],
        source: str,
        handled: set[int],
    ) -> None:
        geometry = rewrite.geometry(self.module)
        start = geometry.line_offsets[statements[0].lineno - 1]
        end = geometry.required_node_offsets(statements[-1])[1]
        indent = " " * statements[0].col_offset
        rewrite.span(
            self.module, start, end, textwrap.indent(source, indent).rstrip("\n")
        )
        handled.update(
            id(node) for statement in statements for node in ast.walk(statement)
        )


@dataclass
class Block(Site):
    """The run moves whole: free reads become parameters, its returns the method's."""

    def install(
        self,
        family: CasesFamily,
        model: CaseModel,
        rewrite: Rewrite,
        answer: Answers,
        handled: set[int],
    ) -> MethodBodies:
        statements = self.statements
        module = self.module
        if any(
            isinstance(
                node, ast.Yield | ast.YieldFrom | ast.Await | ast.Global | ast.Nonlocal
            )
            for statement in statements
            for node in ast.walk(statement)
        ):
            raise handback(
                module, self.node, "branch yields, awaits or rebinds an outer scope"
            )
        returns = any(
            isinstance(node, ast.Return)
            for statement in statements
            for node in ast.walk(statement)
        )
        if returns and not self.is_last:
            raise handback(module, self.node, "branch returns before the function ends")
        if stored_names(statements) & loaded_names(self.body[self.end :]):
            raise handback(
                module, self.node, "branch binds a name the function reads after it"
            )
        renames: dict[int, str] = {}
        parameters: dict[str, str] = {}
        arguments: dict[str, str] = {}
        prefix = self.body[: self.start]
        bound_before = stored_names(prefix) | {
            argument.arg
            for argument in (
                *self.function.args.args[1:],
                *self.function.args.kwonlyargs,
            )
        }
        boolean_names = family.boolean_names(self.function)
        local_annotations = {
            argument.arg: argument.annotation
            for argument in (
                *self.function.args.args[1:],
                *self.function.args.kwonlyargs,
            )
        }
        parents = {
            id(child): parent
            for statement in statements
            for parent in ast.walk(statement)
            for child in ast.iter_child_nodes(parent)
        }
        for statement in statements:
            for node in ast.walk(statement):
                if isinstance(node, ast.Attribute) and self.field_read(node.value):
                    renames[id(node)] = f"cls.{node.attr}"
                elif self.field_read(node):
                    parent = parents.get(id(node))
                    if not (
                        isinstance(parent, ast.Attribute) and id(parent) in renames
                    ):
                        renames[id(node)] = "cls"
                elif isinstance(node, ast.Attribute) and is_name(
                    node.value, [self.receiver]
                ):
                    if node.attr not in self.fields or not isinstance(
                        node.ctx, ast.Load
                    ):
                        raise handback(
                            module,
                            node,
                            f"branch reads {self.receiver}.{node.attr}, not a declared field",
                        )
                    renames[id(node)] = node.attr
                    parameters.setdefault(
                        node.attr, ast.unparse(self.fields[node.attr].annotation)
                    )
                    arguments.setdefault(node.attr, f"{self.receiver}.{node.attr}")
                elif is_name(node, [self.receiver]):
                    parent = parents.get(id(node))
                    if not (
                        isinstance(parent, ast.Attribute)
                        and (id(parent) in renames or self.field_read(parent))
                    ):
                        raise handback(
                            module, node, f"branch reads {self.receiver} itself"
                        )
                elif (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id in bound_before
                ):
                    annotation = local_annotations.get(node.id)
                    if annotation is not None:
                        parameters.setdefault(node.id, ast.unparse(annotation))
                    elif node.id in boolean_names:
                        parameters.setdefault(node.id, "bool")
                    else:
                        raise handback(
                            module,
                            node,
                            f"branch reads local {node.id!r} of undeclared type",
                        )
                    arguments.setdefault(node.id, node.id)
        family_bound = module_bound_names(model.enum.module.module) | set(dir(builtins))
        free = (
            loaded_names(statements)
            - stored_names(statements)
            - set(parameters)
            - {self.receiver}
            - family_bound
        )
        if free:
            raise handback(
                module,
                self.node,
                f"branch reads {sorted(free)}, unbound in the family's module",
            )
        if "cls" in parameters:
            raise handback(module, self.node, "branch parameters collide")
        family_returns = (
            self.resolvable(model, ast.unparse(self.function.returns))
            if returns and self.function.returns is not None
            else "None"
        )
        header = [
            "@classmethod",
            *signature(
                self.method,
                [
                    "cls",
                    *(
                        f"{name}: {self.resolvable(model, annotation)}"
                        for name, annotation in parameters.items()
                    ),
                ],
                family_returns,
                "    ",
            ),
        ]
        booleans = boolean_names | {
            name for name, annotation in parameters.items() if annotation == "bool"
        }
        bodies: dict[str, list[str]] = {}
        for member in model.members:
            folding = Folding(
                module, rewrite, self.literal(model, member), renames, booleans
            )
            bodies[member.name], _ends = folding.block(statements)
        generic, _ends = Folding(
            module, rewrite, lambda node: UNKNOWN, renames, booleans
        ).block(statements)
        prefix = ("return " if returns else "") + (
            f"{self.receiver}.{self.field_name}.{self.method}("
        )
        call = f"{prefix}{', '.join(arguments.values())})"
        if statements[0].col_offset + len(call) > 88:
            call = "\n".join(
                [prefix, *(f"    {argument}," for argument in arguments.values()), ")"]
            )
        self.replace_statements(rewrite, statements, call + "\n", handled)
        return MethodBodies(self.method, header, bodies, generic)


@dataclass
class Guard(Site):
    """``if decided: return`` before the rest: the family runs the rest or not.

    The rest stays on its class as the tail method; members that fall through
    call it back through the parameter the family method takes.
    """

    tail: str = ""
    callback: str = ""

    def ask(self, answer: Answers) -> None:
        super().ask(answer)
        self.tail = answer(TailName(self.module, self.node))
        self.callback = answer(CallbackName(self.module, self.node))

    def install(
        self,
        family: CasesFamily,
        model: CaseModel,
        rewrite: Rewrite,
        answer: Answers,
        handled: set[int],
    ) -> MethodBodies:
        module = self.module
        guard = self.node
        assert isinstance(guard, ast.If)
        rest = self.body[self.end :]
        if (
            len(self.function.args.args) != 1
            or self.function.args.kwonlyargs
            or self.function.args.vararg
            or self.function.args.kwarg
        ):
            raise handback(
                module,
                guard,
                "guarded function takes parameters; the tail callback would need them",
            )
        if stored_names(self.body[: self.start]) & loaded_names(rest):
            raise handback(
                module, guard, "the rest reads a local bound before the guard"
            )
        if model.root in loaded_names(rest):
            raise handback(module, guard, "the rest reads the enum itself")
        if any(
            statement.name == self.tail
            for statement in self.owner.body
            if isinstance(statement, ast.FunctionDef)
        ):
            raise handback(
                module, guard, f"{self.owner.name} already declares {self.tail}"
            )
        returns = (
            "None"
            if self.function.returns is None
            else self.resolvable(model, ast.unparse(self.function.returns))
        )
        bodies: dict[str, list[str]] = {}
        for member in model.members:
            folding = Folding(
                module, rewrite, self.literal(model, member), {}, frozenset()
            )
            test = folding.truth(guard.test)
            if not isinstance(test, ast.Constant):
                raise handback(
                    module, guard, "guard is not decided by member data alone"
                )
            bodies[member.name] = (
                ["return"] if test.value else [f"return {self.callback}()"]
            )
        renames = {
            id(node): f"cls.{node.attr}"
            for node in ast.walk(guard.test)
            if isinstance(node, ast.Attribute) and self.field_read(node.value)
        }
        generic = [
            f"if {ast.unparse(rebuilt(guard.test, renames))}:",
            "    return",
            f"return {self.callback}()",
        ]
        callable_name = next(
            (
                alias.asname or alias.name
                for statement in model.enum.module.module.body
                if isinstance(statement, ast.ImportFrom)
                and statement.module in {"typing", "collections.abc"}
                for alias in statement.names
                if alias.name == "Callable"
            ),
            None,
        )
        if callable_name is None:
            family.ensure_import(
                rewrite, model.enum.module, "collections.abc", "Callable"
            )
            callable_name = "Callable"
        header = [
            "@classmethod",
            *signature(
                self.method,
                ["cls", f"{self.callback}: {callable_name}[[], {returns}]"],
                returns,
                "    ",
            ),
        ]
        tail_source = rewrite.statements_source(module, rest)
        indent = " " * self.function.col_offset
        tail = textwrap.indent(
            "\n".join(
                signature(
                    self.tail,
                    [self.receiver],
                    (
                        None
                        if self.function.returns is None
                        else ast.unparse(self.function.returns)
                    ),
                    indent,
                )
            )
            + "\n"
            + textwrap.indent(tail_source, "    "),
            indent,
        )
        call = f"return {self.receiver}.{self.field_name}.{self.method}({self.receiver}.{self.tail})\n"
        geometry = rewrite.geometry(module)
        start = geometry.line_offsets[guard.lineno - 1]
        end = geometry.required_node_offsets(self.function)[1]
        rewrite.span(
            module,
            start,
            end,
            textwrap.indent(call, " " * guard.col_offset) + "\n" + tail.rstrip("\n"),
        )
        handled.update(
            id(node)
            for statement in self.body[self.start :]
            for node in ast.walk(statement)
        )
        return MethodBodies(self.method, header, bodies, generic)
