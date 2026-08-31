"""Environment-boolean authority drift detection."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from ..ast_tools import (
    BuiltinCallName,
    CollectedFamily,
    ParsedModule,
    SourceModule,
    walk_function_body_nodes,
)
from ..class_index import ATTRIBUTE_CHAIN_AUTHORITY
from ..models import RefactorFinding, SourceLocation
from ..name_algebra import CLASS_NAME_ALGEBRA
from ..native_syntax import NativePythonSyntaxIndex
from ..patterns import PatternId
from ..semantic_match import single_item
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    CompactModuleProjectionDetectorMixin,
    DetectorConfig,
    SemanticMirrorIssueDetector,
    high_confidence_spec,
)

_BOOLEAN_TOKEN_VOCABULARY = frozenset(
    (
        "",
        "0",
        "1",
        "disable",
        "disabled",
        "enable",
        "enabled",
        "f",
        "false",
        "n",
        "no",
        "none",
        "off",
        "on",
        "t",
        "true",
        "y",
        "yes",
    )
)
_ENVIRONMENT_KEY_PARAMETER_TOKENS = frozenset(
    ("env", "environment", "flag", "key", "name", "variable")
)
_DECLARED_AUTHORITY_TOKENS = frozenset(
    ("authority", "contract", "declared", "decision", "policy", "resolver")
)
_DECLARED_AUTHORITY_CALL_TOKENS = frozenset(("declared", "decision"))
_ENVIRONMENT_BOOLEAN_TOKENS = frozenset(("boolean", "decision", "enabled", "flag"))
_INSTANCE_PARAMETER_NAMES = frozenset(("cls", "self"))

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


class EnvironmentReadKind(StrEnum):
    """Direct Python environment access forms recognized by the detector."""

    GETENV = "getenv"
    ENVIRON_GET = "environ.get"
    ENVIRON_SUBSCRIPT = "environ[...]"

    @property
    def os_member_name(self) -> str:
        return self.value.partition(".")[0].partition("[")[0]

    @property
    def method_name(self) -> str | None:
        _owner, separator, method_name = self.value.partition(".")
        return method_name if separator else None


AbsentDecisionSelector = Callable[[bool | None, bool | None], bool | None]
AbsentSourceSelector = Callable[[str | None], str | None]


class MissingValueMode(Enum):
    """Statically visible behavior when an environment key is absent."""

    IMPLICIT_NONE = (
        "implicit None",
        True,
        False,
        lambda _literal_decision, implicit_none_decision: implicit_none_decision,
        lambda _literal: "implicit missing `None` value",
    )
    LITERAL = (
        "literal fallback",
        True,
        True,
        lambda literal_decision, _implicit_none_decision: literal_decision,
        lambda literal: f"environment-read default {literal!r}",
    )
    RAISES = (
        "subscript raises",
        False,
        False,
        lambda _literal_decision, _implicit_none_decision: None,
        lambda _literal: None,
    )
    UNRESOLVED = (
        "unresolved fallback",
        False,
        False,
        lambda _literal_decision, _implicit_none_decision: None,
        lambda _literal: None,
    )

    def __new__(
        cls,
        label: str,
        accepts_boolean_or_fallback: bool,
        carries_literal: bool,
        decision_selector: AbsentDecisionSelector,
        source_selector: AbsentSourceSelector,
    ) -> "MissingValueMode":
        member = object.__new__(cls)
        member._value_ = label
        member._accepts_boolean_or_fallback = accepts_boolean_or_fallback
        member._carries_literal = carries_literal
        member._decision_selector = decision_selector
        member._source_selector = source_selector
        return member

    @property
    def accepts_boolean_or_fallback(self) -> bool:
        return self._accepts_boolean_or_fallback

    @property
    def carries_literal(self) -> bool:
        return self._carries_literal

    def with_resolved_literal(self, literal: str | None) -> "MissingValueMode":
        return type(self).LITERAL if literal is not None else self

    def select_absent_decision(
        self,
        *,
        literal_decision: bool | None,
        implicit_none_decision: bool | None,
    ) -> bool | None:
        return self._decision_selector(literal_decision, implicit_none_decision)

    def absent_source(self, literal: str | None) -> str | None:
        return self._source_selector(literal)


class EnvironmentReadCallArgument(StrEnum):
    """Declared keyword surface of Python environment read calls."""

    KEY = "key"
    DEFAULT = "default"


class EnvironmentBooleanDriftKind(StrEnum):
    """Environment authority drift shapes emitted under one detector family."""

    LOCAL_TOKEN_PARSER = "local token parser"
    FIXED_KEY_AUTHORITY_WRAPPER = "fixed-key authority wrapper"


class EnvironmentFlagDecision(Enum):
    """Nominal labels for statically derived enabled-state decisions."""

    DISABLED = False
    ENABLED = True

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class _EnvironmentImportAliases:
    os_modules: frozenset[str]
    imported_member_names: dict[str, frozenset[str]]

    @property
    def environment_read_names(self) -> frozenset[str]:
        """Local names that can begin a registered environment read."""

        return self.os_modules | frozenset(
            name
            for read_kind in EnvironmentReadKind
            for name in self.names_for(read_kind)
        )

    def names_for(self, read_kind: EnvironmentReadKind) -> frozenset[str]:
        return self.imported_member_names.get(read_kind.os_member_name, frozenset())

    @classmethod
    def from_module(cls, module: ast.Module) -> "_EnvironmentImportAliases":
        os_modules: set[str] = set()
        imported_members: dict[str, set[str]] = {}
        for statement in module.body:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    if alias.name == "os":
                        os_modules.add(alias.asname or alias.name)
            elif isinstance(statement, ast.ImportFrom) and statement.module == "os":
                for alias in statement.names:
                    imported_members.setdefault(alias.name, set()).add(
                        alias.asname or alias.name
                    )
        return cls(
            os_modules=frozenset(os_modules),
            imported_member_names={
                member_name: frozenset(local_names)
                for member_name, local_names in imported_members.items()
            },
        )


@dataclass(frozen=True)
class _FunctionScope:
    module: ParsedModule
    node: FunctionNode
    class_node: ast.ClassDef | None
    class_method_count: int

    @property
    def class_name(self) -> str | None:
        return None if self.class_node is None else self.class_node.name

    @property
    def symbol(self) -> str:
        if self.class_name is None:
            return self.node.name
        return f"{self.class_name}.{self.node.name}"

    @property
    def parameter_names(self) -> frozenset[str]:
        arguments = self.node.args
        return frozenset(
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        )

    def nodes(self) -> tuple[ast.AST, ...]:
        return walk_function_body_nodes(self.node)

    def references_any_name(self, names: frozenset[str]) -> bool:
        return any(
            isinstance(node, ast.Name) and node.id in names for node in self.nodes()
        )

    @staticmethod
    def assignment_nodes(
        nodes: Iterable[ast.AST],
    ) -> tuple[ast.Assign | ast.AnnAssign, ...]:
        return tuple(
            node
            for node in nodes
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
        )


class _FunctionScopeCollector(ast.NodeVisitor):
    def __init__(self, module: ParsedModule) -> None:
        self.module = module
        self.class_stack: list[ast.ClassDef] = []
        self.scopes: list[_FunctionScope] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node)
        for statement in node.body:
            self.visit(statement)
        self.class_stack.pop()

    def _record_function(self, node: FunctionNode) -> None:
        class_node = self.class_stack[-1] if self.class_stack else None
        class_method_count = 0
        if class_node is not None:
            class_method_count = sum(
                isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                for statement in class_node.body
            )
        self.scopes.append(
            _FunctionScope(
                module=self.module,
                node=node,
                class_node=class_node,
                class_method_count=class_method_count,
            )
        )

    def visit_FunctionDef(self, node: FunctionNode) -> None:
        self._record_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def _function_scopes(module: ParsedModule) -> tuple[_FunctionScope, ...]:
    collector = _FunctionScopeCollector(module)
    collector.visit(module.module)
    return tuple(collector.scopes)


def _assignment_target_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = (node.target,) if isinstance(node, ast.AnnAssign) else tuple(node.targets)
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


@dataclass(frozen=True)
class _LiteralResolver:
    values_by_name: dict[str, ast.AST]

    @classmethod
    def for_scope(cls, scope: _FunctionScope) -> "_LiteralResolver":
        assignments: list[ast.Assign | ast.AnnAssign] = list(
            scope.assignment_nodes(scope.module.module.body)
        )
        if scope.class_node is not None:
            assignments.extend(scope.assignment_nodes(scope.class_node.body))
        assignments.extend(scope.assignment_nodes(scope.nodes()))
        values_by_name: dict[str, ast.AST] = {}
        for assignment in assignments:
            value = assignment.value
            if value is None:
                continue
            for name in _assignment_target_names(assignment):
                values_by_name[name] = value
        return cls(values_by_name)

    def string(self, node: ast.AST, seen: frozenset[str] = frozenset()) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        name = _terminal_name(node)
        if name is None or name in seen or name not in self.values_by_name:
            return None
        return self.string(self.values_by_name[name], seen | {name})

    def boolean_tokens(
        self,
        node: ast.AST,
        seen: frozenset[str] = frozenset(),
    ) -> tuple[str, ...] | None:
        elements: Sequence[ast.AST] | None = None
        if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            elements = node.elts
        elif (
            isinstance(node, ast.Call)
            and _terminal_name(node.func) in BuiltinCallName.collection_factory_names()
            and len(node.args) == 1
            and not node.keywords
        ):
            return self.boolean_tokens(node.args[0], seen)
        if elements is not None:
            values = tuple(self.string(element, seen) for element in elements)
            if any(value is None for value in values):
                return None
            raw_values = tuple(dict.fromkeys(str(value) for value in values))
            normalized = tuple(value.strip().lower() for value in raw_values)
            if len(normalized) >= 2 and all(
                value in _BOOLEAN_TOKEN_VOCABULARY for value in normalized
            ):
                return raw_values
            return None
        name = _terminal_name(node)
        if name is None or name in seen or name not in self.values_by_name:
            return None
        return self.boolean_tokens(self.values_by_name[name], seen | {name})


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


@dataclass(frozen=True)
class _MissingValueSemantics:
    environment_read_missing_mode: MissingValueMode


@dataclass(frozen=True)
class _EnvironmentRead(_MissingValueSemantics):
    line: int
    column: int
    kind: EnvironmentReadKind
    key: str
    missing_literal: str | None = None

    @property
    def identity(self) -> tuple[int, int, EnvironmentReadKind]:
        return (self.line, self.column, self.kind)

    def with_missing_literal(self, literal: str) -> "_EnvironmentRead":
        return type(self)(
            line=self.line,
            column=self.column,
            kind=self.kind,
            key=self.key,
            environment_read_missing_mode=MissingValueMode.LITERAL,
            missing_literal=literal,
        )


@dataclass(frozen=True)
class _EnvironmentReadSyntax(_MissingValueSemantics):
    node: ast.Call | ast.Subscript
    kind: EnvironmentReadKind
    key_node: ast.AST
    default_node: ast.AST | None = None

    @classmethod
    def from_call(
        cls,
        kind: EnvironmentReadKind,
        node: ast.Call,
    ) -> "_EnvironmentReadSyntax | None":
        keyword_values = {
            keyword.arg: keyword.value
            for keyword in node.keywords
            if keyword.arg is not None
        }
        key_node = (
            node.args[0]
            if node.args
            else keyword_values.get(EnvironmentReadCallArgument.KEY.value)
        )
        if key_node is None:
            return None
        default_node = (
            node.args[1]
            if len(node.args) >= 2
            else keyword_values.get(EnvironmentReadCallArgument.DEFAULT.value)
        )
        environment_read_missing_mode = (
            MissingValueMode.UNRESOLVED
            if default_node is not None
            else MissingValueMode.IMPLICIT_NONE
        )
        return cls(
            node=node,
            kind=kind,
            key_node=key_node,
            environment_read_missing_mode=environment_read_missing_mode,
            default_node=default_node,
        )

    def materialize(self, resolver: _LiteralResolver) -> _EnvironmentRead:
        key_literal = resolver.string(self.key_node)
        key = ast.unparse(self.key_node)
        if key_literal is not None:
            key = key_literal
        missing_literal = (
            None if self.default_node is None else resolver.string(self.default_node)
        )
        return _EnvironmentRead(
            line=self.node.lineno,
            column=self.node.col_offset,
            kind=self.kind,
            key=key,
            environment_read_missing_mode=(
                self.environment_read_missing_mode.with_resolved_literal(
                    missing_literal
                )
            ),
            missing_literal=missing_literal,
        )


class _EnvironmentReadRecognizer(ABC, metaclass=AutoRegisterMeta):
    __registry__: ClassVar[
        dict[EnvironmentReadKind, type["_EnvironmentReadRecognizer"]]
    ] = {}
    __registry_key__ = "kind"
    __skip_if_no_key__ = True

    kind: ClassVar[EnvironmentReadKind | None] = None

    @classmethod
    def recognizer_types(cls) -> tuple[type["_EnvironmentReadRecognizer"], ...]:
        return tuple(cls.__registry__.values())

    @classmethod
    @abstractmethod
    def recognize(
        cls,
        node: ast.AST,
        aliases: _EnvironmentImportAliases,
    ) -> _EnvironmentReadSyntax | None:
        raise NotImplementedError


class _GetenvReadRecognizer(_EnvironmentReadRecognizer):
    kind = EnvironmentReadKind.GETENV

    @classmethod
    def recognize(
        cls,
        node: ast.AST,
        aliases: _EnvironmentImportAliases,
    ) -> _EnvironmentReadSyntax | None:
        if not isinstance(node, ast.Call):
            return None
        chain = ATTRIBUTE_CHAIN_AUTHORITY.project(node.func)
        module_call = (
            chain is not None
            and len(chain) == 2
            and chain[0] in aliases.os_modules
            and chain[1] == cls.kind.os_member_name
        )
        imported_call = isinstance(
            node.func, ast.Name
        ) and node.func.id in aliases.names_for(cls.kind)
        if not (module_call or imported_call):
            return None
        return _EnvironmentReadSyntax.from_call(cls.kind, node)


class _EnvironGetReadRecognizer(_EnvironmentReadRecognizer):
    kind = EnvironmentReadKind.ENVIRON_GET

    @classmethod
    def recognize(
        cls,
        node: ast.AST,
        aliases: _EnvironmentImportAliases,
    ) -> _EnvironmentReadSyntax | None:
        if not isinstance(node, ast.Call):
            return None
        chain = ATTRIBUTE_CHAIN_AUTHORITY.project(node.func)
        if chain is None or chain[-1] != cls.kind.method_name:
            return None
        owner = chain[:-1]
        module_owner = (
            len(owner) == 2
            and owner[0] in aliases.os_modules
            and owner[1] == cls.kind.os_member_name
        )
        imported_owner = len(owner) == 1 and owner[0] in aliases.names_for(cls.kind)
        if not (module_owner or imported_owner):
            return None
        return _EnvironmentReadSyntax.from_call(cls.kind, node)


class _EnvironSubscriptReadRecognizer(_EnvironmentReadRecognizer):
    kind = EnvironmentReadKind.ENVIRON_SUBSCRIPT

    @classmethod
    def recognize(
        cls,
        node: ast.AST,
        aliases: _EnvironmentImportAliases,
    ) -> _EnvironmentReadSyntax | None:
        if not isinstance(node, ast.Subscript):
            return None
        chain = ATTRIBUTE_CHAIN_AUTHORITY.project(node.value)
        module_owner = (
            chain is not None
            and len(chain) == 2
            and chain[0] in aliases.os_modules
            and chain[1] == cls.kind.os_member_name
        )
        imported_owner = (
            chain is not None
            and len(chain) == 1
            and chain[0] in aliases.names_for(cls.kind)
        )
        if not (module_owner or imported_owner):
            return None
        return _EnvironmentReadSyntax(
            node=node,
            kind=cls.kind,
            key_node=node.slice,
            environment_read_missing_mode=MissingValueMode.RAISES,
        )


class EnvironmentReadSyntaxAuthority:
    """Enumerate inherited read recognizers and materialize one exact match."""

    @staticmethod
    def read(
        node: ast.AST,
        aliases: _EnvironmentImportAliases,
        resolver: _LiteralResolver,
    ) -> _EnvironmentRead | None:
        syntax = single_item(
            tuple(
                match
                for recognizer_type in _EnvironmentReadRecognizer.recognizer_types()
                if (match := recognizer_type.recognize(node, aliases)) is not None
            )
        )
        return None if syntax is None else syntax.materialize(resolver)


def _none_checked_names(node: ast.AST) -> tuple[str, ...]:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return ()
    if not isinstance(node.ops[0], ast.Is):
        return ()
    operands = (node.left, node.comparators[0])
    if not any(
        isinstance(operand, ast.Constant) and operand.value is None
        for operand in operands
    ):
        return ()
    return tuple(operand.id for operand in operands if isinstance(operand, ast.Name))


@dataclass(frozen=True)
class _MissingOverride:
    literal: str | None = None
    decision: bool | None = None


def _missing_overrides(
    scope: _FunctionScope,
    resolver: _LiteralResolver,
) -> dict[str, _MissingOverride]:
    overrides: dict[str, _MissingOverride] = {}
    for node in scope.nodes():
        if not isinstance(node, ast.If):
            continue
        checked_names = _none_checked_names(node.test)
        for checked_name in checked_names:
            for statement in node.body:
                if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    if checked_name not in _assignment_target_names(statement):
                        continue
                    value = statement.value
                    if (
                        value is not None
                        and (literal := resolver.string(value)) is not None
                    ):
                        overrides[checked_name] = _MissingOverride(literal=literal)
                elif (
                    isinstance(statement, ast.Return)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, bool)
                ):
                    overrides[checked_name] = _MissingOverride(
                        decision=statement.value.value
                    )
    return overrides


def _unique_read(
    reads: Iterable[_EnvironmentRead],
) -> _EnvironmentRead | None:
    reads_by_identity = {read.identity: read for read in reads}
    if len(reads_by_identity) != 1:
        return None
    return next(iter(reads_by_identity.values()))


@dataclass(frozen=True)
class _EnvironmentReadAuthority:
    aliases: _EnvironmentImportAliases
    resolver: _LiteralResolver

    def in_expression(self, node: ast.AST) -> tuple[_EnvironmentRead, ...]:
        reads_by_identity: dict[
            tuple[int, int, EnvironmentReadKind], _EnvironmentRead
        ] = {}
        for descendant in ast.walk(node):
            read = EnvironmentReadSyntaxAuthority.read(
                descendant,
                self.aliases,
                self.resolver,
            )
            if read is not None:
                reads_by_identity[read.identity] = read
        return tuple(reads_by_identity.values())

    def with_expression_fallback(
        self,
        read: _EnvironmentRead,
        node: ast.AST,
        lineage: dict[str, _EnvironmentRead],
    ) -> _EnvironmentRead:
        if not read.environment_read_missing_mode.accepts_boolean_or_fallback or (
            read.environment_read_missing_mode.carries_literal
            and read.missing_literal not in {None, ""}
        ):
            return read
        for descendant in ast.walk(node):
            if not isinstance(descendant, ast.BoolOp) or not isinstance(
                descendant.op, ast.Or
            ):
                continue
            source_value, *fallback_values = descendant.values
            source_reads = (
                *self.in_expression(source_value),
                *(
                    lineage[name.id]
                    for name in ast.walk(source_value)
                    if isinstance(name, ast.Name) and name.id in lineage
                ),
            )
            if not any(source.identity == read.identity for source in source_reads):
                continue
            for fallback_value in fallback_values:
                fallback = self.resolver.string(fallback_value)
                if fallback is not None:
                    return read.with_missing_literal(fallback)
        return read

    def lineage(self, scope: _FunctionScope) -> dict[str, _EnvironmentRead]:
        lineage: dict[str, _EnvironmentRead] = {}
        assignments = sorted(
            scope.assignment_nodes(scope.nodes()),
            key=lambda node: (node.lineno, node.col_offset),
        )
        for assignment in assignments:
            value = assignment.value
            if value is None:
                continue
            source_read = _unique_read(
                (
                    *self.in_expression(value),
                    *(
                        lineage[name.id]
                        for name in ast.walk(value)
                        if isinstance(name, ast.Name) and name.id in lineage
                    ),
                )
            )
            if source_read is None:
                continue
            source_read = self.with_expression_fallback(
                source_read,
                value,
                lineage,
            )
            for target_name in _assignment_target_names(assignment):
                lineage[target_name] = source_read
        return lineage

    def source_for_expression(
        self,
        node: ast.AST,
        lineage: dict[str, _EnvironmentRead],
    ) -> _EnvironmentRead | None:
        source_read = _unique_read(
            (
                *self.in_expression(node),
                *(
                    lineage[name.id]
                    for name in ast.walk(node)
                    if isinstance(name, ast.Name) and name.id in lineage
                ),
            )
        )
        if source_read is None:
            return None
        return self.with_expression_fallback(source_read, node, lineage)


def _expression_lineage_names(
    node: ast.AST,
    read: _EnvironmentRead,
    lineage: dict[str, _EnvironmentRead],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            name.id
            for name in ast.walk(node)
            if isinstance(name, ast.Name)
            and name.id in lineage
            and lineage[name.id].identity == read.identity
        )
    )


class EnvironmentValueNormalizer(Enum):
    """Nominal strategy family for recognized string normalizers."""

    CASEFOLD = ("casefold", str.casefold)
    LOWER = ("lower", str.lower)
    STRIP = ("strip", str.strip)
    UPPER = ("upper", str.upper)

    def __new__(
        cls,
        method_name: str,
        normalizer: Callable[[str], str],
    ) -> "EnvironmentValueNormalizer":
        member = object.__new__(cls)
        member._value_ = method_name
        member.normalizer: Callable[[str], str] = normalizer
        return member

    @property
    def method_name(self) -> str:
        return str(self.value)

    def apply(self, value: str) -> str:
        return self.normalizer(value)

    @classmethod
    def from_method_name(cls, method_name: str) -> "EnvironmentValueNormalizer | None":
        return next(
            (normalizer for normalizer in cls if normalizer.method_name == method_name),
            None,
        )


class EnvironmentValueNormalizationAuthority:
    """Project and apply the string normalization chain on an env value."""

    @classmethod
    def method_chain(
        cls,
        node: ast.AST,
        source_names: frozenset[str],
        read: _EnvironmentRead,
    ) -> tuple[EnvironmentValueNormalizer, ...] | None:
        if isinstance(node, ast.Name) and node.id in source_names:
            return ()
        if (
            isinstance(node, (ast.Call, ast.Subscript))
            and node.lineno == read.line
            and node.col_offset == read.column
        ):
            return ()
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and not node.args
            and not node.keywords
        ):
            normalizer = EnvironmentValueNormalizer.from_method_name(node.func.attr)
            if normalizer is not None:
                parent_chain = cls.method_chain(node.func.value, source_names, read)
                if parent_chain is not None:
                    return (*parent_chain, normalizer)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            return next(
                (
                    chain
                    for value in node.values
                    if (chain := cls.method_chain(value, source_names, read))
                    is not None
                ),
                None,
            )
        return None

    @staticmethod
    def apply(
        value: str,
        method_chain: tuple[EnvironmentValueNormalizer, ...],
    ) -> str:
        normalized = value
        for normalizer in method_chain:
            normalized = normalizer.apply(normalized)
        return normalized


@dataclass(frozen=True)
class _EnvironmentBooleanParserSite(SourceLocation):
    read_kind: EnvironmentReadKind
    environment_key: str
    token_values: tuple[str, ...]
    matched_decision: bool
    absent_decision: bool | None
    absent_source: str | None


def _absent_semantics(
    read: _EnvironmentRead,
    source_names: tuple[str, ...],
    value_expression: ast.AST,
    overrides: dict[str, _MissingOverride],
    token_values: tuple[str, ...],
    matched_decision: bool,
) -> tuple[bool | None, str | None]:
    method_chain = EnvironmentValueNormalizationAuthority.method_chain(
        value_expression,
        frozenset(source_names),
        read,
    )

    def decision_for_literal(literal: str) -> bool | None:
        if method_chain is None:
            return None
        normalized = EnvironmentValueNormalizationAuthority.apply(
            literal,
            method_chain,
        )
        return matched_decision if normalized in token_values else not matched_decision

    for source_name in source_names:
        override = overrides.get(source_name)
        if override is None:
            continue
        if override.decision is not None:
            return override.decision, f"explicit `{source_name} is None` return"
        if override.literal is not None:
            decision = decision_for_literal(override.literal)
            if decision is not None:
                return (
                    decision,
                    f"`{source_name} is None` fallback {override.literal!r}",
                )
    literal_decision = (
        None
        if read.missing_literal is None
        else decision_for_literal(read.missing_literal)
    )
    implicit_none_decision = not matched_decision if method_chain == () else None
    read_decision = read.environment_read_missing_mode.select_absent_decision(
        literal_decision=literal_decision,
        implicit_none_decision=implicit_none_decision,
    )
    read_source = read.environment_read_missing_mode.absent_source(read.missing_literal)
    if read_decision is not None and read_source is not None:
        return read_decision, read_source
    return None, None


def _environment_boolean_parser_sites(
    scope: _FunctionScope,
    aliases: _EnvironmentImportAliases,
) -> tuple[_EnvironmentBooleanParserSite, ...]:
    resolver = _LiteralResolver.for_scope(scope)
    read_authority = _EnvironmentReadAuthority(aliases, resolver)
    lineage = read_authority.lineage(scope)
    overrides = _missing_overrides(scope, resolver)
    sites: list[_EnvironmentBooleanParserSite] = []
    seen: set[tuple[int, int]] = set()
    for node in scope.nodes():
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        operator = node.ops[0]
        if not isinstance(operator, (ast.In, ast.NotIn)):
            continue
        token_values = resolver.boolean_tokens(node.comparators[0])
        if token_values is None:
            continue
        read = read_authority.source_for_expression(
            node.left,
            lineage,
        )
        if read is None:
            continue
        identity = (node.lineno, node.col_offset)
        if identity in seen:
            continue
        seen.add(identity)
        matched_decision = isinstance(operator, ast.In)
        source_names = _expression_lineage_names(node.left, read, lineage)
        absent_decision, absent_source = _absent_semantics(
            read,
            source_names,
            node.left,
            overrides,
            token_values,
            matched_decision,
        )
        sites.append(
            _EnvironmentBooleanParserSite(
                file_path=scope.module.file_path,
                line=node.lineno,
                symbol=scope.symbol,
                read_kind=read.kind,
                environment_key=read.key,
                token_values=token_values,
                matched_decision=matched_decision,
                absent_decision=absent_decision,
                absent_source=absent_source,
            )
        )
    return tuple(sites)


class EnvironmentSemanticNameAuthority:
    """Own canonical name-token projection for environment flag symbols."""

    @staticmethod
    def tokens(value: str) -> frozenset[str]:
        return frozenset(
            "environment" if token == "env" else token
            for token in CLASS_NAME_ALGEBRA.token_set(value)
        )


def _annotation_is_bool(node: ast.AST | None) -> bool:
    return node is not None and _terminal_name(node) == "bool"


def _scope_has_declared_authority_shape(scope: _FunctionScope) -> bool:
    semantic_tokens = EnvironmentSemanticNameAuthority.tokens(scope.symbol)
    if "environment" not in semantic_tokens:
        return False
    if not semantic_tokens & _ENVIRONMENT_BOOLEAN_TOKENS:
        return False
    if not semantic_tokens & _DECLARED_AUTHORITY_TOKENS:
        return False
    dynamic_parameters = scope.parameter_names - _INSTANCE_PARAMETER_NAMES
    if not any(
        EnvironmentSemanticNameAuthority.tokens(parameter_name)
        & _ENVIRONMENT_KEY_PARAMETER_TOKENS
        for parameter_name in dynamic_parameters
    ):
        return False
    if not _annotation_is_bool(scope.node.returns) and not (
        semantic_tokens & frozenset(("decision", "enabled"))
    ):
        return False
    nodes = scope.nodes()
    return any(isinstance(node, ast.Raise) for node in nodes) or any(
        not EnvironmentSemanticNameAuthority.tokens(ast.unparse(call.func)).isdisjoint(
            _DECLARED_AUTHORITY_CALL_TOKENS
        )
        for call in nodes
        if isinstance(call, ast.Call)
    )


@dataclass(frozen=True)
class _DeclaredEnvironmentFlagAuthority(SourceLocation):
    selectors: tuple[str, ...]


def _declared_environment_flag_authorities(
    scopes: Iterable[_FunctionScope],
) -> tuple[_DeclaredEnvironmentFlagAuthority, ...]:
    authorities: list[_DeclaredEnvironmentFlagAuthority] = []
    for scope in scopes:
        if not _scope_has_declared_authority_shape(scope):
            continue
        selectors = (scope.node.name,)
        if scope.class_name is not None:
            selectors = (scope.node.name, f"{scope.class_name}.{scope.node.name}")
        authorities.append(
            _DeclaredEnvironmentFlagAuthority(
                file_path=scope.module.file_path,
                line=scope.node.lineno,
                symbol=scope.symbol,
                selectors=selectors,
            )
        )
    return tuple(authorities)


def _authority_for_parser(
    site: _EnvironmentBooleanParserSite,
    authorities: tuple[_DeclaredEnvironmentFlagAuthority, ...],
) -> _DeclaredEnvironmentFlagAuthority | None:
    parser_tokens = EnvironmentSemanticNameAuthority.tokens(site.symbol)
    scored = sorted(
        (
            (
                len(
                    parser_tokens
                    & EnvironmentSemanticNameAuthority.tokens(authority.symbol)
                ),
                authority,
            )
            for authority in authorities
            if authority.symbol != site.symbol
        ),
        key=lambda row: (-row[0], row[1].file_path, row[1].line, row[1].symbol),
    )
    if not scored or scored[0][0] < 2:
        return None
    return scored[0][1]


def _returned_call(scope: _FunctionScope) -> ast.Call | None:
    body = tuple(
        statement
        for statement in scope.node.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    )
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    return body[0].value if isinstance(body[0].value, ast.Call) else None


def _authority_for_selector_chain(
    chain: tuple[str, ...],
    authorities: tuple[_DeclaredEnvironmentFlagAuthority, ...],
) -> _DeclaredEnvironmentFlagAuthority | None:
    qualified_selector = ".".join(chain[-2:])
    qualified_matches = tuple(
        authority
        for authority in authorities
        if qualified_selector in authority.selectors
    )
    qualified_match = single_item(qualified_matches)
    if qualified_match is not None:
        return qualified_match
    terminal_matches = tuple(
        authority for authority in authorities if chain[-1] in authority.selectors
    )
    return single_item(terminal_matches)


def _expression_uses_parameters(
    node: ast.AST,
    parameter_names: frozenset[str],
) -> bool:
    return any(
        isinstance(descendant, ast.Name) and descendant.id in parameter_names
        for descendant in ast.walk(node)
    )


@dataclass(frozen=True)
class _FixedKeyAuthorityWrapperSite(SourceLocation):
    environment_key: str
    authority: _DeclaredEnvironmentFlagAuthority


@dataclass(frozen=True)
class _FixedKeyAuthorityWrapperFact(SourceLocation):
    environment_key: str
    selector_chain: tuple[str, ...]


def _fixed_key_authority_wrapper_facts(
    scopes: Iterable[_FunctionScope],
) -> tuple[_FixedKeyAuthorityWrapperFact, ...]:
    facts: list[_FixedKeyAuthorityWrapperFact] = []
    for scope in scopes:
        if scope.class_name is None or scope.class_method_count > 2:
            continue
        call = _returned_call(scope)
        if call is None:
            continue
        selector_chain = ATTRIBUTE_CHAIN_AUTHORITY.project(call.func)
        if selector_chain is None:
            continue
        key_expression = next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg is not None
                and EnvironmentSemanticNameAuthority.tokens(keyword.arg)
                & _ENVIRONMENT_KEY_PARAMETER_TOKENS
            ),
            None,
        )
        if call.args:
            key_expression = call.args[0]
        if key_expression is None:
            continue
        dynamic_parameters = scope.parameter_names - _INSTANCE_PARAMETER_NAMES
        if _expression_uses_parameters(key_expression, dynamic_parameters):
            continue
        environment_key = _LiteralResolver.for_scope(scope).string(key_expression)
        if environment_key is None:
            continue
        facts.append(
            _FixedKeyAuthorityWrapperFact(
                file_path=scope.module.file_path,
                line=scope.node.lineno,
                symbol=scope.symbol,
                environment_key=environment_key,
                selector_chain=selector_chain,
            )
        )
    return tuple(facts)


@dataclass(frozen=True)
class _EnvironmentBooleanDriftCandidate(SourceLocation):
    kind: EnvironmentBooleanDriftKind
    environment_key: str
    summary_detail: str
    authority: _DeclaredEnvironmentFlagAuthority | None = None

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        local = SourceLocation(
            self.file_path,
            self.line,
            f"{self.symbol}:{self.environment_key}",
        )
        if self.authority is None:
            return (local,)
        return (
            local,
            SourceLocation(
                self.authority.file_path,
                self.authority.line,
                self.authority.symbol,
            ),
        )


def _parser_candidate(
    site: _EnvironmentBooleanParserSite,
    authority: _DeclaredEnvironmentFlagAuthority | None,
) -> _EnvironmentBooleanDriftCandidate:
    token_role = str(site.matched_decision).lower()
    detail = (
        f"interprets `{site.read_kind.value}` through local {token_role}-token "
        f"values {site.token_values}"
    )
    if site.absent_decision is not None:
        absent_state = EnvironmentFlagDecision(site.absent_decision).label
        detail += f"; {site.absent_source} makes absence {absent_state}"
    if authority is not None:
        detail += f" despite existing declared authority `{authority.symbol}`"
    return _EnvironmentBooleanDriftCandidate(
        kind=EnvironmentBooleanDriftKind.LOCAL_TOKEN_PARSER,
        file_path=site.file_path,
        line=site.line,
        symbol=site.symbol,
        environment_key=site.environment_key,
        summary_detail=detail,
        authority=authority,
    )


def _wrapper_candidate(
    site: _FixedKeyAuthorityWrapperSite,
) -> _EnvironmentBooleanDriftCandidate:
    return _EnvironmentBooleanDriftCandidate(
        kind=EnvironmentBooleanDriftKind.FIXED_KEY_AUTHORITY_WRAPPER,
        file_path=site.file_path,
        line=site.line,
        symbol=site.symbol,
        environment_key=site.environment_key,
        summary_detail=(
            "is a one-return fixed-key wrapper around existing declared authority "
            f"`{site.authority.symbol}`"
        ),
        authority=site.authority,
    )


@dataclass(frozen=True)
class _EnvironmentBooleanModuleProjection:
    parser_sites: tuple[_EnvironmentBooleanParserSite, ...]
    authorities: tuple[_DeclaredEnvironmentFlagAuthority, ...]
    wrapper_facts: tuple[_FixedKeyAuthorityWrapperFact, ...]

    @classmethod
    def from_scopes(
        cls,
        scopes: tuple[_FunctionScope, ...],
        aliases: _EnvironmentImportAliases,
    ) -> "_EnvironmentBooleanModuleProjection":
        environment_read_names = aliases.environment_read_names
        return cls(
            parser_sites=(
                tuple(
                    site
                    for scope in scopes
                    if scope.references_any_name(environment_read_names)
                    for site in _environment_boolean_parser_sites(scope, aliases)
                )
                if environment_read_names
                else ()
            ),
            authorities=_declared_environment_flag_authorities(scopes),
            wrapper_facts=_fixed_key_authority_wrapper_facts(scopes),
        )


def _native_environment_function_may_declare_authority(name: str) -> bool:
    tokens = EnvironmentSemanticNameAuthority.tokens(name)
    return (
        "environment" in tokens
        and bool(tokens & _ENVIRONMENT_BOOLEAN_TOKENS)
        and bool(tokens & _DECLARED_AUTHORITY_TOKENS)
    )


def _native_environment_import_aliases(
    syntax_index: NativePythonSyntaxIndex,
) -> tuple[_EnvironmentImportAliases, list[ast.stmt]]:
    imports = [
        syntax_index.statement_for(node)
        for node in syntax_index.tree.root_node.named_children
        if node.type in {"import_from_statement", "import_statement"}
    ]
    module = ast.Module(body=imports, type_ignores=[])
    return _EnvironmentImportAliases.from_module(module), imports


def _native_environment_module_projection(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[_EnvironmentBooleanModuleProjection] | None:
    """Project environment semantics from selected function fragments."""

    if not syntax_index.is_complete:
        return None
    try:
        aliases, imports = _native_environment_import_aliases(syntax_index)
        module_assignments = [
            syntax_index.statement_for(node)
            for node in syntax_index.top_level_assignment_statements()
        ]
        parsed_module = source_module.parsed_module(
            ast.Module(
                body=[*imports, *module_assignments],
                type_ignores=[],
            ),
        )
        captures = syntax_index.common_captures()
        functions = tuple(
            sorted(
                captures.get("function", ()),
                key=lambda node: (node.start_byte, -node.end_byte),
            )
        )
        direct_method_counts: dict[object, int] = defaultdict(int)
        for function_node in functions:
            class_node = syntax_index.direct_enclosing_class(function_node)
            if class_node is not None:
                direct_method_counts[class_node] += 1
        class_assignments: dict[object, list[ast.stmt]] = defaultdict(list)
        for assignment in captures.get("assignment", ()):
            statement_node = assignment.parent
            if statement_node is None or statement_node.type != "expression_statement":
                continue
            block = statement_node.parent
            if block is None or block.type != "block":
                continue
            class_node = block.parent
            if class_node is None or class_node.type != "class_definition":
                continue
            statement = syntax_index.statement_for(statement_node)
            if statement not in class_assignments[class_node]:
                class_assignments[class_node].append(statement)

        imported_read_names = frozenset(
            name
            for read_kind in EnvironmentReadKind
            for name in aliases.names_for(read_kind)
        )
        scopes: list[_FunctionScope] = []
        for function_node in functions:
            lexical_scopes = syntax_index.named_scope_nodes(function_node)
            if any(
                scope.type == "function_definition" for scope in lexical_scopes
            ):
                continue
            class_node = next(
                (
                    scope
                    for scope in reversed(lexical_scopes)
                    if scope.type == "class_definition"
                ),
                None,
            )
            class_method_count = (
                0 if class_node is None else direct_method_counts.get(class_node, 0)
            )
            function_name = syntax_index.declared_name(function_node)
            function_symbol = (
                function_name
                if class_node is None
                else f"{syntax_index.declared_name(class_node)}.{function_name}"
            )
            function_source = syntax_index.source_for(function_node)
            may_read_environment = (
                b"getenv" in function_source
                or b"environ" in function_source
                or any(
                    name.encode("utf-8") in function_source
                    for name in imported_read_names
                )
            )
            may_declare_authority = (
                _native_environment_function_may_declare_authority(function_symbol)
            )
            may_be_wrapper = class_node is not None and class_method_count <= 2
            if not (may_read_environment or may_declare_authority or may_be_wrapper):
                continue
            function = syntax_index.function_for(function_node)
            synthetic_class: ast.ClassDef | None = None
            if class_node is not None:
                synthetic_class = ast.ClassDef(
                    name=syntax_index.declared_name(class_node),
                    bases=[],
                    keywords=[],
                    body=list(class_assignments.get(class_node, ())),
                    decorator_list=[],
                )
            scopes.append(
                _FunctionScope(
                    module=parsed_module,
                    node=function,
                    class_node=synthetic_class,
                    class_method_count=class_method_count,
                )
            )
        scopes_tuple = tuple(scopes)
        return [_EnvironmentBooleanModuleProjection.from_scopes(scopes_tuple, aliases)]
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


class _EnvironmentBooleanModuleProjectionFamily(
    CollectedFamily[_EnvironmentBooleanModuleProjection]
):
    item_type = _EnvironmentBooleanModuleProjection
    report_presence_predicate = staticmethod(
        lambda items, config: any(
            item.parser_sites or item.authorities or item.wrapper_facts
            for item in items
            if isinstance(item, _EnvironmentBooleanModuleProjection)
        )
    )
    source_collector = staticmethod(_native_environment_module_projection)

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[_EnvironmentBooleanModuleProjection]:
        del cls
        scopes = _function_scopes(parsed_module)
        aliases = _EnvironmentImportAliases.from_module(parsed_module.module)
        return [_EnvironmentBooleanModuleProjection.from_scopes(scopes, aliases)]


def _fixed_key_authority_wrapper_sites_from_facts(
    facts: Iterable[_FixedKeyAuthorityWrapperFact],
    authorities: tuple[_DeclaredEnvironmentFlagAuthority, ...],
) -> tuple[_FixedKeyAuthorityWrapperSite, ...]:
    authority_symbols = {authority.symbol for authority in authorities}
    sites: list[_FixedKeyAuthorityWrapperSite] = []
    for fact in facts:
        if fact.symbol in authority_symbols:
            continue
        authority = _authority_for_selector_chain(fact.selector_chain, authorities)
        if authority is None:
            continue
        sites.append(
            _FixedKeyAuthorityWrapperSite(
                file_path=fact.file_path,
                line=fact.line,
                symbol=fact.symbol,
                environment_key=fact.environment_key,
                authority=authority,
            )
        )
    return tuple(sites)


class EnvironmentBooleanAuthorityDriftDetector(
    CompactModuleProjectionDetectorMixin[_EnvironmentBooleanModuleProjection],
    SemanticMirrorIssueDetector,
):
    """Detect local environment flag semantics outside a declared authority."""

    module_projection_family = _EnvironmentBooleanModuleProjectionFamily

    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Environment booleans should have one declared fail-loud authority",
        "Local environment reads coupled to ad hoc true/false token collections create independent enabled-state semantics. Missing-value defaults can invert that state, and fixed-key authority wrappers preserve multiple parsing entry points even when a parameterized declared authority already exists.",
        "one process-boundary environment flag authority that owns declared tokens and absent decisions, followed by typed immutable configuration",
        "direct environment reads or fixed-key wrappers repeat boolean flag semantics outside the declared authority boundary",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    )
    detector_priority = -2

    def _findings_from_compact_projections(
        self,
        projections: tuple[_EnvironmentBooleanModuleProjection, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        authorities = tuple(
            authority
            for projection in projections
            for authority in projection.authorities
        )
        candidates = [
            _parser_candidate(site, _authority_for_parser(site, authorities))
            for projection in projections
            for site in projection.parser_sites
        ]
        candidates.extend(
            _wrapper_candidate(site)
            for site in _fixed_key_authority_wrapper_sites_from_facts(
                (
                    fact
                    for projection in projections
                    for fact in projection.wrapper_facts
                ),
                authorities,
            )
        )
        return [
            self.build_finding(
                f"`{candidate.symbol}` {candidate.summary_detail} for `{candidate.environment_key}`.",
                candidate.evidence,
                scaffold=(
                    "@dataclass(frozen=True)\n"
                    "class RuntimeFlagConfig:\n"
                    "    feature_enabled: bool\n\n"
                    "# Resolve each environment flag once through the declared "
                    "authority at the process boundary, then pass RuntimeFlagConfig."
                ),
                codemod_patch=(
                    f"# Delete local environment-boolean semantics in `{candidate.symbol}`.\n"
                    "# Route the flag through the existing declared authority once, "
                    "materialize typed immutable configuration, and pass that value "
                    "to runtime consumers."
                ),
                relation_context=(
                    "AST dataflow links a direct Python environment read or fixed-key "
                    "wrapper to independently owned boolean flag semantics"
                ),
            )
            for candidate in sorted(
                candidates,
                key=lambda item: (
                    item.file_path,
                    item.line,
                    item.symbol,
                    item.kind.value,
                ),
            )
        ]


__all__ = ("EnvironmentBooleanAuthorityDriftDetector",)
