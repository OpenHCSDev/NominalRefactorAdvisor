"""Restating sites: places that spell out a fact some other declaration should own.

Every site records where a fact is written again (`RestatingSite`) and which
restater it belongs to (`group`), so one detector base can turn any family of
sites into findings. A family differs from another only in what it matches.
"""

from __future__ import annotations

import ast
import re
import tomllib
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .ast_tools import BuiltinCallName, ParsedModule, module_syntax_index
from .declared_root import DeclaredRoot
from .models import SourceLocation
from .observation_families import ShapeFamily
from .python_module_identity import PythonModulePathIdentity

# A subject needs this many distinct string arms before it is a family spelled out.
STRING_DISPATCH_MIN_ARMS = 3
# A key set needs this many keys before it is a record rather than one probe.
EXACT_KEY_SET_MIN_KEYS = 2


@dataclass(frozen=True)
class RestatingSite:
    """One place that writes a fact again.

    `group` names what one finding covers inside `symbol` (the dispatch subject,
    the receiver of a probe, the declared name); `detail` is what the site says.
    """

    file_path: str
    lineno: int
    symbol: str
    group: str
    detail: tuple[str, ...]

    @property
    def location(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.lineno, self.symbol)


@dataclass(frozen=True)
class RestatingSiteModule:
    """One module's restating sites, kept even when empty so a scan knows its modules."""

    module_name: str
    sites: tuple[RestatingSite, ...]


class RestatingSiteFamily(ShapeFamily[RestatingSiteModule], ABC):
    """A family of restating sites; subclasses declare only what they match."""

    _registry_skip = True
    item_type = RestatingSiteModule

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[RestatingSiteModule]:
        return [
            RestatingSiteModule(
                module_name=parsed_module.module_name,
                sites=tuple(cls.sites(parsed_module)),
            )
        ]

    @classmethod
    @abstractmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        raise NotImplementedError

    @staticmethod
    def site(
        parsed_module: ParsedModule,
        node: ast.expr | ast.stmt | ast.arg | ast.alias,
        symbol: str,
        group: str,
        detail: tuple[str, ...],
    ) -> RestatingSite:
        return RestatingSite(
            file_path=parsed_module.file_path,
            lineno=node.lineno,
            symbol=symbol,
            group=group,
            detail=detail,
        )


def _string_constants(elements: list[ast.expr]) -> tuple[str, ...] | None:
    """The string values of a literal collection, or None if any element is not one."""

    values = tuple(
        element.value
        for element in elements
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    )
    return values if len(values) == len(elements) else None


def _literal_strings(node: ast.expr) -> tuple[str, ...] | None:
    """The string keys of a literal set, tuple, list or frozenset/set call."""

    match node:
        case (
            ast.Set(elts=elements) | ast.Tuple(elts=elements) | ast.List(elts=elements)
        ):
            return _string_constants(elements)
        case ast.Call(
            func=ast.Name(id=name),
            args=[
                ast.Set(elts=elements)
                | ast.Tuple(elts=elements)
                | ast.List(elts=elements)
            ],
        ) if name in (BuiltinCallName.SET, BuiltinCallName.FROZENSET):
            return _string_constants(elements)
    return None


class StringDispatchSiteFamily(RestatingSiteFamily):
    """One subject compared against several string literals inside one function."""

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        index = module_syntax_index(parsed_module.module)
        # (function, subject) -> node -> (symbol, literals compared at that node)
        arms: dict[tuple[int, str], dict[ast.AST, tuple[str, set[str]]]] = defaultdict(
            dict
        )
        for node_index, node in (
            *index.indexed_nodes_of_type(ast.Compare),
            *index.indexed_nodes_of_type(ast.Match),
        ):
            function = index.scopes[
                index.scope_ids[node_index]
            ].executable_function_index
            if function < 0:
                continue
            symbol = index.lexical_symbol(node_index)
            for subject, literal in cls.arms(node):
                _symbol, literals = arms[(function, subject)].setdefault(
                    node, (symbol, set())
                )
                literals.add(literal)
        for (_function, subject), nodes in arms.items():
            if (
                len(set().union(*(literals for _symbol, literals in nodes.values())))
                < STRING_DISPATCH_MIN_ARMS
            ):
                continue
            for node, (symbol, literals) in nodes.items():
                yield cls.site(
                    parsed_module, node, symbol, subject, tuple(sorted(literals))
                )

    @classmethod
    def arms(cls, node: ast.AST) -> Iterator[tuple[str, str]]:
        match node:
            case ast.Compare(left=left, ops=operators, comparators=comparators):
                for operator, right in zip(operators, comparators):
                    yield from cls.compare_arms(left, operator, right)
            case ast.Match(subject=subject, cases=cases):
                for case in cases:
                    for literal in cls.pattern_literals(case.pattern):
                        yield ast.unparse(subject), literal

    @staticmethod
    def compare_arms(
        left: ast.expr, operator: ast.cmpop, right: ast.expr
    ) -> Iterator[tuple[str, str]]:
        match operator, left, right:
            case ast.Eq() | ast.NotEq(), _, ast.Constant(value=str() as literal):
                if not isinstance(left, ast.Constant):
                    yield ast.unparse(left), literal
            case ast.Eq() | ast.NotEq(), ast.Constant(value=str() as literal), _:
                yield ast.unparse(right), literal
            case (
                ast.In() | ast.NotIn(),
                _,
                ast.Set(elts=elements)
                | ast.Tuple(elts=elements)
                | ast.List(elts=elements),
            ):
                for literal in _string_constants(elements) or ():
                    yield ast.unparse(left), literal

    @classmethod
    def pattern_literals(cls, pattern: ast.pattern) -> Iterator[str]:
        match pattern:
            case ast.MatchValue(value=ast.Constant(value=str() as literal)):
                yield literal
            case ast.MatchOr(patterns=alternatives):
                for alternative in alternatives:
                    yield from cls.pattern_literals(alternative)


def imported_module_names(parsed_module: ParsedModule) -> frozenset[str]:
    """Names this module binds with `import x` / `import x as y`: modules, not instances."""

    return frozenset(
        alias.asname or alias.name.partition(".")[0]
        for _node_index, node in module_syntax_index(
            parsed_module.module
        ).indexed_nodes_of_type(ast.Import)
        for alias in node.names
    )


class NamedAttributeSiteFamily(RestatingSiteFamily):
    """getattr/hasattr/setattr/delattr by a literal name, or by a module constant holding one.

    A name computed at run time (read from `_fields`, a loop variable) restates
    nothing: the attribute list is whatever the owner supplies. A literal, or a
    module-level string constant, spells one attribute of the receiver's type.
    An imported module is not a type of ours: `hasattr(os, "pidfd_open")` asks
    what this platform build provides, which no declaration of ours can hold.
    """

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        index = module_syntax_index(parsed_module.module)
        constants = cls.string_constants(parsed_module)
        modules = imported_module_names(parsed_module)
        names = BuiltinCallName.attribute_access_names()
        for node_index, call in index.indexed_nodes_of_type(ast.Call):
            callee = call.func
            if not isinstance(callee, ast.Name) or callee.id not in names:
                continue
            if len(call.args) < 2:
                continue
            receiver, name_expression, *_rest = call.args
            if isinstance(receiver, ast.Name) and receiver.id in modules:
                continue
            spelled = cls.attribute_name(name_expression, constants)
            if spelled is None:
                continue
            defaulted = callee.id == BuiltinCallName.GETATTR and len(call.args) == 3
            yield cls.site(
                parsed_module,
                call,
                index.lexical_symbol(node_index),
                ast.unparse(receiver),
                (f"{callee.id}{' with default' if defaulted else ''}", spelled),
            )

    @staticmethod
    def string_constants(parsed_module: ParsedModule) -> dict[str, str]:
        """Module-level `NAME = "literal"` bindings."""

        constants: dict[str, str] = {}
        for statement in parsed_module.module.body:
            match statement:
                case ast.Assign(
                    targets=[ast.Name(id=name)],
                    value=ast.Constant(value=str() as value),
                ) | ast.AnnAssign(
                    target=ast.Name(id=name), value=ast.Constant(value=str() as value)
                ):
                    constants[name] = value
        return constants

    @staticmethod
    def attribute_name(expression: ast.expr, constants: dict[str, str]) -> str | None:
        match expression:
            case ast.Constant(value=str() as value):
                return value
            case ast.Name(id=name):
                return constants.get(name)
        return None


class _KeyViewSyntax:
    """Recognise a mapping's key collection and a literal key collection."""

    WRAPPERS = frozenset(
        (
            BuiltinCallName.SET,
            BuiltinCallName.FROZENSET,
            BuiltinCallName.SORTED,
            BuiltinCallName.LIST,
            BuiltinCallName.TUPLE,
        )
    )
    SET_METHODS = frozenset(
        ("issubset", "issuperset", "isdisjoint", "difference", "symmetric_difference")
    )
    KEYS = "keys"

    @classmethod
    def mapping_of(cls, node: ast.expr) -> tuple[ast.expr, bool] | None:
        """The expression whose keys `node` collects, and whether `.keys()` says it is a mapping.

        A bare `set(x)` collects a mapping's keys only if `x` is a mapping, which the
        caller must establish; `x.keys()` establishes it.
        """

        match node:
            case ast.Call(func=ast.Attribute(value=receiver, attr=cls.KEYS), args=[]):
                return receiver, True
            case ast.Call(func=ast.Name(id=name), args=[inner]) if name in cls.WRAPPERS:
                keys_of = cls.mapping_of(inner)
                if keys_of is not None:
                    return keys_of
                if isinstance(inner, (ast.Name, ast.Attribute, ast.Subscript)):
                    return inner, False
        return None

    @classmethod
    def literal_keys(cls, node: ast.expr) -> tuple[str, ...] | None:
        keys = _literal_strings(node)
        if keys is None or len(set(keys)) < EXACT_KEY_SET_MIN_KEYS:
            return None
        return tuple(sorted(set(keys)))


class ExactKeySetSiteFamily(RestatingSiteFamily):
    """A literal key collection compared with, or subtracted from, a mapping's keys."""

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        index = module_syntax_index(parsed_module.module)
        keyed = cls.string_keyed_receivers(index)
        for node_index, node in (
            *index.indexed_nodes_of_type(ast.Compare),
            *index.indexed_nodes_of_type(ast.BinOp),
            *index.indexed_nodes_of_type(ast.Call),
            *index.indexed_nodes_of_type(ast.GeneratorExp),
        ):
            function = index.scopes[
                index.scope_ids[node_index]
            ].executable_function_index
            for mapping, keys in cls.restatements(node, keyed[function]):
                yield cls.site(
                    parsed_module,
                    node,
                    index.lexical_symbol(node_index),
                    f"{node.lineno}:{node.col_offset}",
                    (ast.unparse(mapping), *keys),
                )

    @staticmethod
    def string_keyed_receivers(index) -> dict[int, set[str]]:
        """Per function, the expressions it reads by a string key: those are mappings."""

        keyed: dict[int, set[str]] = defaultdict(set)
        for node_index, node in index.indexed_nodes_of_type(ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(
                node.slice.value, str
            ):
                function = index.scopes[
                    index.scope_ids[node_index]
                ].executable_function_index
                keyed[function].add(ast.unparse(node.value))
        for node_index, node in index.indexed_nodes_of_type(ast.Call):
            match node:
                case ast.Call(
                    func=ast.Attribute(
                        value=receiver, attr="get" | "pop" | "setdefault"
                    ),
                    args=[ast.Constant(value=str()), *_],
                ):
                    function = index.scopes[
                        index.scope_ids[node_index]
                    ].executable_function_index
                    keyed[function].add(ast.unparse(receiver))
        return keyed

    @classmethod
    def restatements(
        cls, node: ast.AST, keyed: set[str]
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        match node:
            case ast.Compare(
                left=left,
                ops=[
                    ast.Eq() | ast.NotEq() | ast.Lt() | ast.LtE() | ast.Gt() | ast.GtE()
                ],
                comparators=[right],
            ):
                yield from cls.pair(left, right, keyed)
            case ast.BinOp(
                left=left,
                op=ast.Sub() | ast.BitAnd() | ast.BitOr() | ast.BitXor(),
                right=right,
            ):
                yield from cls.pair(left, right, keyed)
            case ast.Call(
                func=ast.Attribute(value=viewed, attr=method), args=[literal]
            ) if (method in _KeyViewSyntax.SET_METHODS):
                yield from cls.pair(viewed, literal, keyed)
            case ast.GeneratorExp(
                elt=ast.Compare(
                    left=ast.Name(id=key),
                    ops=[ast.In() | ast.NotIn()],
                    comparators=[mapping],
                ),
                generators=[
                    ast.comprehension(target=ast.Name(id=target), iter=literal)
                ],
            ) if (
                key == target
            ):
                keys = _KeyViewSyntax.literal_keys(literal)
                if keys is not None and ast.unparse(mapping) in keyed:
                    yield mapping, keys

    @staticmethod
    def pair(
        first: ast.expr, second: ast.expr, keyed: set[str]
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        for viewed, literal in ((first, second), (second, first)):
            found = _KeyViewSyntax.mapping_of(viewed)
            keys = _KeyViewSyntax.literal_keys(literal)
            if found is None or keys is None:
                continue
            mapping, by_keys_view = found
            if by_keys_view or ast.unparse(mapping) in keyed:
                yield mapping, keys
                return


class ForeignAbsenceSiteFamily(RestatingSiteFamily):
    """`other.attr is None` and `is not None` on an object that is not the method's own."""

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        index = module_syntax_index(parsed_module.module)
        modules = imported_module_names(parsed_module)
        for node_index, node in index.indexed_nodes_of_type(ast.Compare):
            match node:
                case ast.Compare(
                    left=ast.Attribute(value=receiver, attr=attribute),
                    ops=[ast.Is() | ast.IsNot() as operator],
                    comparators=[ast.Constant(value=None)],
                ):
                    pass
                case _:
                    continue
            if cls.is_own_or_module(receiver, node_index, index, modules):
                continue
            yield cls.site(
                parsed_module,
                node,
                index.lexical_symbol(node_index),
                ast.unparse(receiver),
                (
                    attribute,
                    "is None" if isinstance(operator, ast.Is) else "is not None",
                ),
            )

    @staticmethod
    def is_own_or_module(
        receiver: ast.expr, node_index: int, index, modules: frozenset[str]
    ) -> bool:
        root = receiver
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name):
            return False
        if root.id in modules and root is receiver:
            return True
        function = index.scopes[index.scope_ids[node_index]].executable_function_index
        if function < 0:
            return False
        method = index.depth_first_nodes[function]
        owner = index.parent_node(function)
        arguments = [*method.args.posonlyargs, *method.args.args]
        return (
            isinstance(owner, ast.ClassDef)
            and bool(arguments)
            and arguments[0].arg == root.id
        )


class ForwardingFacadeSiteFamily(RestatingSiteFamily):
    """A class whose members only forward to an object it holds in an attribute.

    The first site of a facade is the class itself (detail: held class, or "" when
    unresolved); every further site is one forwarding member (detail: member name,
    kind, target). A facade needs two forwards to one held attribute.
    """

    MIN_FORWARDS = 2
    CLASS_SITE = "class"

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        index = module_syntax_index(parsed_module.module)
        for node_index, node in index.indexed_nodes_of_type(ast.ClassDef):
            held = cls.held_attributes(node)
            forwards: dict[str, list[tuple[ast.FunctionDef, str, str]]] = defaultdict(
                list
            )
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    forwarded = cls.forward(member, frozenset(held))
                    if forwarded is not None:
                        forwards[forwarded[0]].append(
                            (member, forwarded[1], forwarded[2])
                        )
            outer = index.lexical_symbol(node_index)
            symbol = node.name if outer == "<module>" else f"{outer}.{node.name}"
            for attribute, members in forwards.items():
                if len(members) < cls.MIN_FORWARDS:
                    continue
                yield cls.site(
                    parsed_module,
                    node,
                    symbol,
                    attribute,
                    (cls.CLASS_SITE, held[attribute]),
                )
                for member, kind, target in members:
                    yield cls.site(
                        parsed_module,
                        member,
                        symbol,
                        attribute,
                        (member.name, kind, target),
                    )

    @classmethod
    def held_attributes(cls, node: ast.ClassDef) -> dict[str, str]:
        """Attributes the class declares as fields or sets in __init__, with their class text."""

        held: dict[str, str] = {}
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                held[statement.target.id] = cls.annotated_class(statement.annotation)
            if isinstance(statement, ast.FunctionDef) and statement.name == "__init__":
                annotations = {
                    argument.arg: cls.annotated_class(argument.annotation)
                    for argument in (
                        *statement.args.posonlyargs,
                        *statement.args.args,
                        *statement.args.kwonlyargs,
                    )
                    if argument.annotation is not None
                }
                for assigned in statement.body:
                    match assigned:
                        case ast.Assign(
                            targets=[
                                ast.Attribute(value=ast.Name(id="self"), attr=name)
                            ],
                            value=value,
                        ) | ast.AnnAssign(
                            target=ast.Attribute(value=ast.Name(id="self"), attr=name),
                            value=value,
                        ) if (
                            value is not None
                        ):
                            held.setdefault(name, cls.class_text(value, annotations))
        return held

    @staticmethod
    def annotated_class(annotation: ast.expr) -> str:
        """The class text of an annotation, without an `Annotated[...]` wrapper's metadata."""

        match annotation:
            case ast.Subscript(
                value=ast.Name(id="Annotated") | ast.Attribute(attr="Annotated"),
                slice=ast.Tuple(elts=[wrapped, *_]),
            ):
                return ast.unparse(wrapped)
        return ast.unparse(annotation)

    @staticmethod
    def class_text(value: ast.expr, annotations: dict[str, str]) -> str:
        match value:
            case ast.Name(id=name):
                return annotations.get(name, "")
            case ast.Call(func=callee):
                return f"{ast.unparse(callee)}(...)"
        return ""

    @classmethod
    def forward(
        cls, member: ast.FunctionDef, held: frozenset[str]
    ) -> tuple[str, str, str] | None:
        """(held attribute, kind, target) when the member only forwards to a held attribute."""

        body = [
            statement
            for statement in member.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
            )
        ]
        match body:
            case [ast.Return(value=value)] if value is not None:
                pass
            case _:
                return None
        decorators = [ast.unparse(decorator) for decorator in member.decorator_list]
        match value, decorators:
            case ast.Attribute(
                value=ast.Attribute(value=ast.Name(id="self"), attr=attribute),
                attr=target,
            ), ["property"] if (
                attribute in held
            ):
                return attribute, "property", f"{attribute}.{target}"
            case (
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(value=ast.Name(id="self"), attr=attribute),
                        attr=target,
                    ),
                    args=arguments,
                    keywords=keywords,
                ),
                [],
            ) if attribute in held and cls.passes_own_parameters(
                member, arguments, keywords
            ):
                return attribute, "method", f"{attribute}.{target}()"
        return None

    @staticmethod
    def passes_own_parameters(
        member: ast.FunctionDef, arguments: list[ast.expr], keywords: list[ast.keyword]
    ) -> bool:
        """Whether the call passes exactly the member's own parameters, unchanged and in order."""

        signature = member.args
        own = [*signature.posonlyargs, *signature.args][1:]
        expected_arguments = [
            ast.Name(id=argument.arg, ctx=ast.Load()) for argument in own
        ]
        if signature.vararg is not None:
            expected_arguments.append(
                ast.Starred(
                    value=ast.Name(id=signature.vararg.arg, ctx=ast.Load()),
                    ctx=ast.Load(),
                )
            )
        expected_keywords = [
            ast.keyword(
                arg=argument.arg, value=ast.Name(id=argument.arg, ctx=ast.Load())
            )
            for argument in signature.kwonlyargs
        ]
        if signature.kwarg is not None:
            expected_keywords.append(
                ast.keyword(
                    arg=None, value=ast.Name(id=signature.kwarg.arg, ctx=ast.Load())
                )
            )
        return [ast.dump(item) for item in arguments] == [
            ast.dump(item) for item in expected_arguments
        ] and [ast.dump(item) for item in keywords] == [
            ast.dump(item) for item in expected_keywords
        ]


class LegacyWord(StrEnum):
    """Words with which a path declares itself the old one, kept for an old caller."""

    LEGACY = "legacy"
    COMPAT = "compat"
    BACKCOMPAT = "backcompat"
    DEPRECATED = "deprecated"
    DEPRECATE = "deprecate"
    DEPRECATION = "deprecation"
    SHIM = "shim"

    @classmethod
    def words(cls, identifier: str) -> list[str]:
        return [
            word.lower()
            for word in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", identifier)
        ]

    @classmethod
    def in_identifier(cls, identifier: str) -> bool:
        """Whether a snake_case or CamelCase identifier has one of these as a word."""

        return any(word in cls for word in cls.words(identifier))

    @classmethod
    def stem(cls, identifier: str) -> str:
        """The identifier without these words: what the old path is the old form of."""

        return "".join(word for word in cls.words(identifier) if word not in cls)

    @classmethod
    def in_prose(cls, text: str) -> bool:
        """Whether prose says the path is old: one of these as a whole word, or an old-caller phrase."""

        lowered = text.lower()
        return any(word in cls for word in re.findall(r"[a-z]+", lowered)) or bool(
            re.search(r"backwards?[ -]compatib", lowered)
        )


class LegacyMarkerSiteFamily(RestatingSiteFamily):
    """A path that says it is the old one, beside what replaces it.

    Three self-declarations count: a deprecation mechanism (a `warn` of a
    deprecation class, a deprecation decorator), prose in the docstring's summary line, and an
    identifier with a marker word *beside a sibling of the same stem* (the old
    path kept next to its replacement), or an alias `old = new` of a replacement.
    A name that merely contains a marker word, with no sibling, is not one.
    """

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        index = module_syntax_index(parsed_module.module)
        for node_type in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef):
            for node_index, node in index.indexed_nodes_of_type(node_type):
                yield from cls.declaration(parsed_module, index, node_index, node)
        for node_index, node in index.indexed_nodes_of_type(ast.arguments):
            siblings = {
                argument.arg
                for argument in (
                    *node.posonlyargs,
                    *node.args,
                    *node.kwonlyargs,
                    *([node.vararg] if node.vararg else []),
                    *([node.kwarg] if node.kwarg else []),
                )
            }
            for argument in sorted(siblings):
                if cls.has_replacement(argument, siblings):
                    argument_node = next(
                        candidate
                        for candidate in (
                            *node.posonlyargs,
                            *node.args,
                            *node.kwonlyargs,
                        )
                        if candidate.arg == argument
                    )
                    yield cls.site(
                        parsed_module,
                        argument_node,
                        cls.declared_symbol(index.lexical_symbol(node_index), argument),
                        argument,
                        ("parameter", "beside its replacement"),
                    )
        for node_index, node in (
            *index.indexed_nodes_of_type(ast.Assign),
            *index.indexed_nodes_of_type(ast.AnnAssign),
        ):
            owner = index.parent_node(node_index)
            if node.value is not None and isinstance(owner, (ast.Module, ast.ClassDef)):
                replacement = cls.aliased(node.value)
                for target in cls.assigned_names(node):
                    if replacement is not None and LegacyWord.in_identifier(target):
                        yield cls.site(
                            parsed_module,
                            node,
                            cls.declared_symbol(
                                index.lexical_symbol(node_index), target
                            ),
                            target,
                            ("alias", f"of {replacement}"),
                        )
        for node_index, node in index.indexed_nodes_of_type(ast.alias):
            # An alias renames the old path; a plain import of a marked name is its consumer.
            bound = node.asname or ""
            if (
                bound
                and LegacyWord.in_identifier(bound)
                and not LegacyWord.in_identifier(node.name)
            ):
                yield cls.site(
                    parsed_module,
                    node,
                    cls.declared_symbol(index.lexical_symbol(node_index), bound),
                    bound,
                    ("import alias", f"of {node.name}"),
                )
        for node_index, call in index.indexed_nodes_of_type(ast.Call):
            warned = cls.warned_legacy_class(call)
            if warned is not None:
                symbol = index.lexical_symbol(node_index)
                yield cls.site(
                    parsed_module,
                    call,
                    symbol,
                    symbol.rpartition(".")[2],
                    ("warning", warned),
                )

    @classmethod
    def declaration(
        cls,
        parsed_module: ParsedModule,
        index,
        node_index: int,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> Iterator[RestatingSite]:
        parent = index.parent_node(node_index)
        siblings = {
            name
            for statement in index.children_by_node.get(parent, ())
            if isinstance(statement, ast.stmt)
            for name in cls.declared_names(statement)
        }
        why = cls.reason(node, siblings)
        if why is None:
            return
        symbol = cls.declared_symbol(index.lexical_symbol(node_index), node.name)
        yield cls.site(
            parsed_module, node, symbol, node.name, (type(node).__name__, why)
        )

    @staticmethod
    def declared_symbol(owner: str, name: str) -> str:
        """The dotted symbol of a name declared inside `owner`."""

        return name if owner == "<module>" else f"{owner}.{name}"

    @staticmethod
    def has_replacement(name: str, siblings: set[str]) -> bool:
        """Whether `name` has a marker word and a sibling of the same stem without one."""

        stem = LegacyWord.stem(name)
        return (
            LegacyWord.in_identifier(name)
            and bool(stem)
            and any(
                sibling != name
                and not LegacyWord.in_identifier(sibling)
                and LegacyWord.stem(sibling) == stem
                for sibling in siblings
            )
        )

    @classmethod
    def reason(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        siblings: set[str],
    ) -> str | None:
        if cls.has_replacement(node.name, siblings):
            return "named so, beside its replacement"
        if any(
            LegacyWord.in_identifier(ast.unparse(decorator))
            for decorator in node.decorator_list
        ):
            return "decorated so"
        docstring = ast.get_docstring(node)
        # The summary line says what the thing is; later lines may merely discuss the idea.
        if docstring is not None and LegacyWord.in_prose(docstring.splitlines()[0]):
            return "documented so"
        return None

    @staticmethod
    def declared_names(statement: ast.stmt) -> Iterator[str]:
        match statement:
            case (
                ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
            ):
                yield name
            case ast.Assign(targets=targets):
                for target in targets:
                    if isinstance(target, ast.Name):
                        yield target.id
            case ast.AnnAssign(target=ast.Name(id=name)):
                yield name

    @staticmethod
    def aliased(value: ast.expr) -> str | None:
        """The replacement an assignment aliases, when its value is a plain reference."""

        return (
            ast.unparse(value) if isinstance(value, (ast.Name, ast.Attribute)) else None
        )

    @staticmethod
    def assigned_names(node: ast.Assign | ast.AnnAssign) -> Iterator[str]:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                yield target.id

    @staticmethod
    def warned_legacy_class(call: ast.Call) -> str | None:
        """The warning class, named for an old path, that a `warn(...)` call raises."""

        match call.func:
            case ast.Attribute(attr="warn") | ast.Name(id="warn"):
                pass
            case _:
                return None
        for argument in (*call.args, *(keyword.value for keyword in call.keywords)):
            spelled = ast.unparse(argument)
            if isinstance(
                argument, (ast.Name, ast.Attribute)
            ) and LegacyWord.in_identifier(spelled):
                return spelled
        return None


@dataclass(frozen=True)
class ModuleReferenceFacts:
    """What one module imports, names and uses, for finding who still reaches a path."""

    module_name: str
    file_path: str
    is_package_init: bool
    runs_as_script: bool
    enumerates_modules: bool
    imported: tuple[str, ...]
    named: tuple[str, ...]
    referenced: tuple[tuple[str, int], ...]

    @property
    def package_name(self) -> str:
        """The package this module sits in (itself, if it is a package's `__init__`)."""

        return (
            self.module_name
            if self.is_package_init
            else self.module_name.rpartition(".")[0]
        )

    @property
    def is_package_entry_file(self) -> bool:
        """`__init__.py` and `__main__.py` are reached by their package, not by import."""

        return self.is_package_init or Path(self.file_path).name == "__main__.py"


class ModuleReferenceFamily(ShapeFamily[ModuleReferenceFacts]):
    item_type = ModuleReferenceFacts
    DOTTED_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
    ENUMERATORS = frozenset(("iter_modules", "walk_packages"))

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ModuleReferenceFacts]:
        index = module_syntax_index(parsed_module.module)
        identity = PythonModulePathIdentity(
            path=parsed_module.path,
            import_name=parsed_module.module_name,
            is_package_init=parsed_module.is_package_init,
        )
        imported: set[str] = set()
        for _node_index, node in index.indexed_nodes_of_type(ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                imported.update(
                    ".".join(parts[: stop + 1]) for stop in range(len(parts))
                )
        for _node_index, node in index.indexed_nodes_of_type(ast.ImportFrom):
            base = identity.resolve_import_from_module(
                imported_module=node.module, level=node.level
            )
            if base is None:
                continue
            imported.add(base)
            imported.update(
                f"{base}.{alias.name}" for alias in node.names if alias.name != "*"
            )
        used: list[tuple[str, int]] = [
            *(
                (node.id, node.lineno)
                for _i, node in index.indexed_nodes_of_type(ast.Name)
            ),
            *(
                (node.attr, node.lineno)
                for _i, node in index.indexed_nodes_of_type(ast.Attribute)
            ),
            *(
                (node.arg, node.value.lineno)
                for _i, node in index.indexed_nodes_of_type(ast.keyword)
                if node.arg is not None
            ),
        ]
        referenced: dict[str, int] = {}
        for name, line in used:
            referenced[name] = min(line, referenced.get(name, line))
        named = {
            node.value
            for _node_index, node in index.indexed_nodes_of_type(ast.Constant)
            if isinstance(node.value, str) and cls.DOTTED_NAME.fullmatch(node.value)
        }
        return [
            ModuleReferenceFacts(
                module_name=parsed_module.module_name,
                file_path=parsed_module.file_path,
                is_package_init=parsed_module.is_package_init,
                runs_as_script=any(
                    cls.is_main_guard(node)
                    for node in parsed_module.module.body
                    if isinstance(node, ast.If)
                ),
                enumerates_modules=any(
                    cls.is_enumerator(call)
                    for _i, call in index.indexed_nodes_of_type(ast.Call)
                ),
                imported=tuple(sorted(imported)),
                named=tuple(sorted(named)),
                referenced=tuple(sorted(referenced.items())),
            )
        ]

    @classmethod
    def is_enumerator(cls, call: ast.Call) -> bool:
        """Whether the call lists a package's modules, which a loader then imports."""

        match call.func:
            case ast.Attribute(attr=name) | ast.Name(id=name):
                return name in cls.ENUMERATORS
        return False

    @staticmethod
    def is_main_guard(node: ast.If) -> bool:
        match node.test:
            case ast.Compare(
                left=ast.Name(id="__name__"),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value="__main__")],
            ):
                return True
        return False


@dataclass(frozen=True)
class ProjectEntryPoints:
    """Module names a project declares as entry points in its pyproject.toml."""

    module_names: frozenset[str]

    SCRIPT_TABLES = ("scripts", "gui-scripts")

    @classmethod
    def declared_for(cls, source_paths: Iterable[Path]) -> "ProjectEntryPoints":
        pyprojects = {
            pyproject
            for source_path in source_paths
            if (pyproject := DeclaredRoot.nearest_pyproject(source_path)) is not None
        }
        names: set[str] = set()
        for pyproject in sorted(pyprojects):
            project = tomllib.loads(pyproject.read_text()).get("project", {})
            targets = [
                target
                for table in cls.SCRIPT_TABLES
                for target in project.get(table, {}).values()
            ]
            targets += [
                target
                for group in project.get("entry-points", {}).values()
                for target in group.values()
            ]
            names.update(target.partition(":")[0].strip() for target in targets)
        return cls(frozenset(names))


@dataclass(frozen=True)
class UnreachedModules:
    """Scanned modules that nothing imports, names, lists as a package member, or declares as an entry point."""

    module_names: frozenset[str]

    @classmethod
    def from_facts(cls, facts: Sequence[ModuleReferenceFacts]) -> "UnreachedModules":
        entry_points = ProjectEntryPoints.declared_for(
            Path(module.file_path) for module in facts
        )
        reached_by: dict[str, set[str]] = defaultdict(set)
        for module in facts:
            for name in (*module.imported, *module.named):
                reached_by[name].add(module.module_name)
        # A loader that lists modules reaches those of the packages it can list: its
        # own, and any a string names. Which one it lists is not decidable here.
        listable = (
            {module.package_name for module in facts if module.enumerates_modules}
            | {name for module in facts for name in module.named}
            if any(module.enumerates_modules for module in facts)
            else set()
        )
        return cls(
            frozenset(
                module.module_name
                for module in facts
                if not module.is_package_entry_file
                and not module.runs_as_script
                and module.module_name not in entry_points.module_names
                and not reached_by[module.module_name] - {module.module_name}
                and not any(
                    module.module_name.startswith(f"{package}.") for package in listable
                )
            )
        )


@dataclass(frozen=True)
class ReferenceIndex:
    """Where scanned modules use each identifier: first use per module."""

    uses_by_name: Mapping[str, tuple[tuple[str, int], ...]]

    @classmethod
    def from_facts(cls, facts: Sequence[ModuleReferenceFacts]) -> "ReferenceIndex":
        uses: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for module in facts:
            for name, line in module.referenced:
                uses[name].append((module.file_path, line))
        return cls({name: tuple(found) for name, found in uses.items()})

    def consumers_of(
        self, name: str, declared_at: tuple[str, int]
    ) -> tuple[tuple[str, int], ...]:
        return tuple(
            use for use in self.uses_by_name.get(name, ()) if use != declared_at
        )


class DeadModuleSiteFamily(RestatingSiteFamily):
    """A module's own site, so that a finding can point at the module file."""

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        yield RestatingSite(
            file_path=parsed_module.file_path,
            lineno=1,
            symbol=parsed_module.module_name,
            group=parsed_module.module_name,
            detail=(),
        )
