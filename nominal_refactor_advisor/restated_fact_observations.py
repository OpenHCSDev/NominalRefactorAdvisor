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
        case ast.Set(elts=elements) | ast.Tuple(elts=elements) | ast.List(elts=elements):
            return _string_constants(elements)
        case ast.Call(
            func=ast.Name(id=name), args=[ast.Set(elts=elements) | ast.Tuple(elts=elements) | ast.List(elts=elements)]
        ) if name in (BuiltinCallName.SET, BuiltinCallName.FROZENSET):
            return _string_constants(elements)
    return None


class StringDispatchSiteFamily(RestatingSiteFamily):
    """One subject compared against several string literals inside one function."""

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        index = module_syntax_index(parsed_module.module)
        # (function, subject) -> node -> (symbol, literals compared at that node)
        arms: dict[tuple[int, str], dict[ast.AST, tuple[str, set[str]]]] = defaultdict(dict)
        for node_index, node in (
            *index.indexed_nodes_of_type(ast.Compare),
            *index.indexed_nodes_of_type(ast.Match),
        ):
            function = index.scopes[index.scope_ids[node_index]].executable_function_index
            if function < 0:
                continue
            symbol = index.lexical_symbol(node_index)
            for subject, literal in cls.arms(node):
                _symbol, literals = arms[(function, subject)].setdefault(node, (symbol, set()))
                literals.add(literal)
        for (_function, subject), nodes in arms.items():
            if len(set().union(*(literals for _symbol, literals in nodes.values()))) < STRING_DISPATCH_MIN_ARMS:
                continue
            for node, (symbol, literals) in nodes.items():
                yield cls.site(parsed_module, node, symbol, subject, tuple(sorted(literals)))

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
            case ast.In() | ast.NotIn(), _, ast.Set(elts=elements) | ast.Tuple(
                elts=elements
            ) | ast.List(elts=elements):
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
        for _node_index, node in module_syntax_index(parsed_module.module).indexed_nodes_of_type(ast.Import)
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
                case ast.Assign(targets=[ast.Name(id=name)], value=ast.Constant(value=str() as value)) | ast.AnnAssign(
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


