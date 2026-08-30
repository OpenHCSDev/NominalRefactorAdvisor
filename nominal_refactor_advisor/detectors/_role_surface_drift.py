"""Role-surface drift detector implementation."""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from types import EllipsisType
from typing import Iterable, Sequence

from ..ast_tools import (
    CollectedFamily,
    ParsedModule,
    SourceModule,
)
from ..export_tools import PublicExportPolicy, derive_public_exports
from ..semantic_algebra import FiniteAxisSystem, ObjectFamilyShape
from ..semantic_description_length import CompressionCertificate
from ..native_syntax import NativePythonSyntaxIndex
from ._base import *
from ._helpers import *

_ROLE_SURFACE_DRIFT_TOKEN_STOPWORDS = frozenset(
    {
        "arg",
        "args",
        "attr",
        "attrs",
        "build",
        "builder",
        "class",
        "classes",
        "cls",
        "collection",
        "collections",
        "component",
        "components",
        "config",
        "configs",
        "context",
        "contexts",
        "count",
        "counts",
        "data",
        "entry",
        "entries",
        "field",
        "fields",
        "for",
        "from",
        "function",
        "functions",
        "get",
        "has",
        "id",
        "ids",
        "index",
        "indices",
        "input",
        "inputs",
        "item",
        "items",
        "key",
        "keys",
        "kind",
        "metadata",
        "mode",
        "model",
        "models",
        "name",
        "names",
        "number",
        "numbers",
        "object",
        "objects",
        "output",
        "outputs",
        "path",
        "paths",
        "payload",
        "payloads",
        "post",
        "position",
        "positions",
        "property",
        "record",
        "records",
        "request",
        "requests",
        "response",
        "responses",
        "result",
        "results",
        "self",
        "set",
        "source",
        "state",
        "states",
        "target",
        "targets",
        "to",
        "type",
        "types",
        "value",
        "values",
        "with",
    }
)
_GENERIC_ROLE_CASE_LITERAL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_GENERIC_ROLE_CASE_CONTEXT_MAP_KEY = "mapping_key"
_GENERIC_ROLE_CASE_CONTEXT_COMPARE = "compare_case"
_GENERIC_ROLE_CASE_SENTINEL_TOKENS = frozenset({"false", "none", "null", "true"})


@dataclass(frozen=True)
class BroadSemanticAxisTokenBoundary:
    """Broad semantic role tokens shared by local case tables and logic."""

    broad_semantic_axis_tokens: tuple[str, ...]

    @classmethod
    def from_context(
        cls,
        *,
        owner_tokens: tuple[str, ...],
        body_tokens: tuple[str, ...],
        module_tokens: frozenset[str],
    ) -> "BroadSemanticAxisTokenBoundary | None":
        broad_semantic_axis_tokens = tuple(
            token
            for token in owner_tokens
            if token in body_tokens and token not in {"self", "cls"} | module_tokens
        )
        if not broad_semantic_axis_tokens:
            return None
        return cls(broad_semantic_axis_tokens)

    @property
    def label(self) -> str:
        return ", ".join(self.broad_semantic_axis_tokens)

    @property
    def token_set(self) -> frozenset[str]:
        return frozenset(self.broad_semantic_axis_tokens)

    def camel_case_name(self, default_tokens: tuple[str, ...]) -> str:
        return _camel_case(
            "_".join(self.broad_semantic_axis_tokens[:2] or default_tokens)
        )


@dataclass(frozen=True)
class RoleCaseLiteralBoundary:
    """Concrete role-case literals observed under one semantic axis."""

    case_literals: tuple[str, ...]

    @property
    def case_label(self) -> str:
        return ", ".join(self.case_literals)

    @property
    def short_case_label(self) -> str:
        return ",".join(self.case_literals[:4])


@dataclass(frozen=True)
class RoleCaseTokenBoundary:
    """Token projection of concrete role-case literals."""

    case_tokens: tuple[str, ...]

    @property
    def case_token_label(self) -> str:
        return ", ".join(self.case_tokens)


@dataclass(frozen=True)
class GenericRoleCaseTableSite(
    BroadSemanticAxisTokenBoundary,
    RoleCaseLiteralBoundary,
    RoleCaseTokenBoundary,
    LineWitnessCandidate,
):
    owner_symbol: str
    owner_tokens: tuple[str, ...]
    context_kinds: tuple[str, ...]

    @property
    def symbol(self) -> str:
        return f"{self.owner_symbol}:role_cases:{self.short_case_label}"


@dataclass(frozen=True)
class GenericRoleCaseTableCandidate(
    BroadSemanticAxisTokenBoundary,
    RoleCaseLiteralBoundary,
    LineWitnessCandidate,
):
    shared_case_tokens: tuple[str, ...]
    owner_symbols: tuple[str, ...]
    sites: tuple[GenericRoleCaseTableSite, ...]
    compression_certificate: CompressionCertificate

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(site.file_path, site.line, site.symbol)
            for site in self.sites[:8]
        )

    @property
    def scaffold(self) -> str:
        broad = self.camel_case_name(("role", "case"))
        return (
            f"@dataclass(frozen=True)\n"
            f"class {broad}CaseAuthority:\n"
            f"    case_name: str\n\n"
            f"    def label_for(self, value): ...\n\n"
            "# Keep concrete case tables behind one authority for the broader "
            "semantic axis, and let adapters/renderers depend on that authority."
        )

    @property
    def codemod_patch(self) -> str:
        cases = ", ".join(self.shared_case_tokens)
        return (
            f"# Concrete case table(s) for broad role token(s) {self.label} repeat "
            f"case token(s) {cases} across {len(self.owner_symbols)} owner(s).\n"
            "# Move the case table to one role-neutral authority owned by the "
            "broad semantic axis, then have concrete viewers/adapters call that "
            "authority instead of carrying local case knowledge."
        )


@dataclass(frozen=True, slots=True)
class CompactRoleSurfaceModuleProjection:
    """AST-free local role-case tables for one module."""

    generic_role_case_table_sites: tuple[GenericRoleCaseTableSite, ...]


@dataclass(frozen=True)
class LocalRoleCaseLogicCandidate(
    BroadSemanticAxisTokenBoundary,
    RoleCaseLiteralBoundary,
    RoleCaseTokenBoundary,
    LineWitnessCandidate,
):
    owner_symbol: str
    owner_tokens: tuple[str, ...]
    context_kinds: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(
                self.file_path,
                self.line,
                f"{self.owner_symbol}:local_role_cases:{self.short_case_label}",
            ),
        )

    @property
    def compression_certificate(self) -> CompressionCertificate:
        return CompressionCertificate.from_object_family(
            manual_object_count=max(
                (len(self.case_literals) * max(len(self.context_kinds), 1))
                + len(self.case_tokens)
                + len(self.broad_semantic_axis_tokens),
                8,
            ),
            replacement_shape=ObjectFamilyShape.from_roles(
                ("role_axis_projection_authority",),
                axis=("semantic_case",),
                source=("owner_scope",),
            ),
            semantic_axes=(
                ("broad_semantic_axis_tokens", self.broad_semantic_axis_tokens),
                ("case_tokens", self.case_tokens),
                ("context_kinds", self.context_kinds),
            ),
        )

    @property
    def scaffold(self) -> str:
        broad = self.camel_case_name(("role", "axis"))
        return (
            f"@dataclass(frozen=True)\n"
            f"class {broad}ProjectionAuthority:\n"
            f"    case_name: str\n\n"
            f"    def project(self, payload): ...\n\n"
            "# Behavior methods should depend on this role-axis authority instead "
            "of embedding concrete case literals."
        )

    @property
    def codemod_patch(self) -> str:
        return (
            f"# `{self.owner_symbol}` embeds concrete case literal(s) {self.case_label} "
            f"inside a broad role axis ({self.label}).\n"
            "# Move the concrete case knowledge behind a nominal role-axis authority "
            "and have this behavior surface query that authority instead of owning "
            "local map/guard cases."
        )


class RoleSurfaceTokenProjection:
    @lru_cache(maxsize=None)
    def identifier_tokens(self, text: str) -> tuple[str, ...]:
        return tuple(
            self.canonical_token(token)
            for token in CLASS_NAME_ALGEBRA.ordered_tokens(text)
            if len(token) >= 2 and not token.isdigit()
        )

    def canonical_token(self, token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return f"{token[:-3]}y"
        if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            return token[:-1]
        return token

    def semantic_tokens(self, text: str) -> tuple[str, ...]:
        return tuple(
            token
            for token in self.identifier_tokens(text)
            if token not in _ROLE_SURFACE_DRIFT_TOKEN_STOPWORDS
        )

    @lru_cache(maxsize=None)
    def node_tokens(self, node: ast.AST | None) -> tuple[str, ...]:
        if node is None:
            return ()
        tokens: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                tokens.extend(self.semantic_tokens(child.id))
            elif isinstance(child, ast.Attribute):
                tokens.extend(self.semantic_tokens(child.attr))
            elif isinstance(child, ast.keyword) and child.arg is not None:
                tokens.extend(self.semantic_tokens(child.arg))
        return tuple(sorted(set(tokens)))

    def target_tokens(self, targets: Iterable[ast.AST]) -> tuple[str, ...]:
        return tuple(
            sorted({token for target in targets for token in self.node_tokens(target)})
        )


ROLE_SURFACE_TOKEN_PROJECTION = RoleSurfaceTokenProjection()


@dataclass(frozen=True)
class _GenericRoleCaseTableProjection:
    site: GenericRoleCaseTableSite
    broad_semantic_axis_token: str


@lru_cache(maxsize=None)
def _generic_role_case_body_tokens(root: ast.AST) -> tuple[str, ...]:
    tokens: set[str] = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Name):
            tokens.update(ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(node.id))
        elif isinstance(node, ast.Attribute):
            tokens.update(ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(node.attr))
        elif isinstance(node, ast.arg):
            tokens.update(ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(node.arg))
        elif isinstance(node, ast.keyword) and node.arg is not None:
            tokens.update(ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(node.arg))
    return tuple(sorted(tokens))


@lru_cache(maxsize=None)
def _generic_role_case_literal_tokens(
    value: str | bytes | int | float | complex | bool | None | EllipsisType,
) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    if not _GENERIC_ROLE_CASE_LITERAL_RE.fullmatch(value):
        return ()
    tokens = ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(value)
    if not tokens:
        return ()
    return tuple(
        token for token in tokens if token not in _GENERIC_ROLE_CASE_SENTINEL_TOKENS
    )


def _generic_role_case_context(
    node: ast.Constant,
    parents: Sequence[ast.AST],
) -> str | None:
    parent = parents[-1] if parents else None
    if isinstance(parent, ast.Dict) and node in parent.keys:
        if any(
            isinstance(parent_node, ast.Return) for parent_node in reversed(parents)
        ):
            return None
        return _GENERIC_ROLE_CASE_CONTEXT_MAP_KEY
    if any(isinstance(parent_node, ast.Compare) for parent_node in reversed(parents)):
        return _GENERIC_ROLE_CASE_CONTEXT_COMPARE
    return None


class _GenericRoleCaseLiteralVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: list[ast.AST] = []
        self.records: list[tuple[int, str, tuple[str, ...], str]] = []
        self._body_tokens: set[str] = set()

    @property
    def body_tokens(self) -> tuple[str, ...]:
        return tuple(sorted(self._body_tokens))

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self._body_tokens.update(
                ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(node.id)
            )
        elif isinstance(node, ast.Attribute):
            self._body_tokens.update(
                ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(node.attr)
            )
        elif isinstance(node, ast.arg):
            self._body_tokens.update(
                ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(node.arg)
            )
        elif isinstance(node, ast.keyword) and node.arg is not None:
            self._body_tokens.update(
                ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(node.arg)
            )
        self.parents.append(node)
        try:
            super().visit(node)
        finally:
            self.parents.pop()

    def visit_Constant(self, node: ast.Constant) -> None:
        tokens = _generic_role_case_literal_tokens(node.value)
        if tokens:
            context = _generic_role_case_context(node, self.parents[:-1])
            if context is not None:
                self.records.append((node.lineno, str(node.value), tokens, context))
        self.generic_visit(node)


@dataclass(frozen=True)
class _LocalRoleCaseLiteralRecord:
    line: int
    literal: str
    literal_tokens: tuple[str, ...]
    context_kind: str


class _LocalRoleCaseLiteralCollector(ast.NodeVisitor):
    def __init__(self, role_boundary: BroadSemanticAxisTokenBoundary) -> None:
        self.role_boundary = role_boundary
        self.mapping_records_by_name: dict[str, list[_LocalRoleCaseLiteralRecord]] = (
            defaultdict(list)
        )
        self.axis_indexed_mapping_names: set[str] = set()
        self.compare_records: list[_LocalRoleCaseLiteralRecord] = []

    @property
    def records(self) -> tuple[_LocalRoleCaseLiteralRecord, ...]:
        mapping_records = tuple(
            record
            for mapping_name in sorted(self.axis_indexed_mapping_names)
            for record in self.mapping_records_by_name.get(mapping_name, ())
        )
        return (*mapping_records, *self.compare_records)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._record_mapping_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_mapping_assignment((node.target,), node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.attr in {"get", "pop", "setdefault"}
            and node.args
            and self._expression_has_broad_axis_token(node.args[0])
        ):
            self.axis_indexed_mapping_names.add(node.func.value.id)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and self._expression_has_broad_axis_token(
            node.slice
        ):
            self.axis_indexed_mapping_names.add(node.value.id)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        expressions = (node.left, *node.comparators)
        for left, right in zip(expressions, expressions[1:], strict=False):
            self._record_axis_compare(left, right, node.lineno)
            self._record_axis_compare(right, left, node.lineno)
        self.generic_visit(node)

    def _record_mapping_assignment(
        self,
        targets: Iterable[ast.AST],
        value: ast.AST | None,
    ) -> None:
        if value is None:
            return
        items = _string_dict_items(value)
        if items is None:
            return
        target_names = tuple(
            target.id for target in targets if isinstance(target, ast.Name)
        )
        if not target_names:
            return
        records = tuple(
            _LocalRoleCaseLiteralRecord(
                line=value.lineno,
                literal=literal,
                literal_tokens=literal_tokens,
                context_kind=_GENERIC_ROLE_CASE_CONTEXT_MAP_KEY,
            )
            for literal in sorted(items)
            if (literal_tokens := _generic_role_case_literal_tokens(literal))
        )
        for target_name in target_names:
            self.mapping_records_by_name[target_name].extend(records)

    def _record_axis_compare(
        self,
        possible_literal: ast.AST,
        possible_axis: ast.AST,
        line: int,
    ) -> None:
        if (
            not isinstance(possible_literal, ast.Constant)
            or not isinstance(possible_literal.value, str)
            or not self._expression_has_broad_axis_token(possible_axis)
        ):
            return
        literal_tokens = _generic_role_case_literal_tokens(possible_literal.value)
        if not literal_tokens:
            return
        self.compare_records.append(
            _LocalRoleCaseLiteralRecord(
                line=line,
                literal=possible_literal.value,
                literal_tokens=literal_tokens,
                context_kind=_GENERIC_ROLE_CASE_CONTEXT_COMPARE,
            )
        )

    def _expression_has_broad_axis_token(self, node: ast.AST) -> bool:
        return bool(
            self.role_boundary.token_set
            & set(ROLE_SURFACE_TOKEN_PROJECTION.node_tokens(node))
        )


def _generic_role_case_table_site(
    *,
    module: ParsedModule,
    owner_symbol: str,
    owner_name: str,
    line: int,
    root: ast.AST,
    minimum_case_count: int,
) -> GenericRoleCaseTableSite | None:
    owner_tokens = ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(owner_name)
    if len(owner_tokens) < 2:
        return None
    visitor = _GenericRoleCaseLiteralVisitor()
    visitor.visit(root)
    module_tokens = {
        token
        for part in module.path.with_suffix("").parts
        for token in ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(part)
    }
    role_boundary = BroadSemanticAxisTokenBoundary.from_context(
        owner_tokens=owner_tokens,
        body_tokens=visitor.body_tokens,
        module_tokens=frozenset(module_tokens),
    )
    if role_boundary is None:
        return None

    literal_records = visitor.records
    if not literal_records:
        return None
    case_tokens = tuple(
        sorted(
            {
                token
                for _, _, literal_tokens, _ in literal_records
                for token in literal_tokens
                if token not in role_boundary.broad_semantic_axis_tokens
            }
        )
    )
    if len(case_tokens) < minimum_case_count:
        return None
    case_literals = tuple(sorted({literal for _, literal, _, _ in literal_records}))
    context_kinds = tuple(sorted({context for *_, context in literal_records}))
    return GenericRoleCaseTableSite(
        file_path=str(module.path),
        line=line,
        owner_symbol=owner_symbol,
        owner_tokens=owner_tokens,
        broad_semantic_axis_tokens=role_boundary.broad_semantic_axis_tokens,
        case_tokens=case_tokens,
        case_literals=case_literals,
        context_kinds=context_kinds,
    )


def _generic_role_case_table_sites_with_minimum(
    module: ParsedModule,
    minimum_case_count: int,
    *,
    demanded_axis_tokens: frozenset[str] = frozenset(),
    demanded_case_tokens: frozenset[str] = frozenset(),
    demanded_case_count: int = 0,
) -> tuple[GenericRoleCaseTableSite, ...]:
    sites: list[GenericRoleCaseTableSite] = []
    source_lines = module.source.splitlines(keepends=True)
    module_axis_source = " ".join(module.path.with_suffix("").parts).casefold()
    top_level_functions = {
        statement
        for statement in module.module.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in _walk_nodes(module.module):
        site: GenericRoleCaseTableSite | None = None
        is_site_root = isinstance(node, ast.ClassDef) or (
            node in top_level_functions
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if not is_site_root:
            continue
        if demanded_axis_tokens or demanded_case_tokens:
            end_line = node.end_lineno or node.lineno
            root_source = "".join(source_lines[node.lineno - 1 : end_line]).casefold()
            if demanded_axis_tokens and not any(
                token in root_source or token in module_axis_source
                for token in demanded_axis_tokens
            ):
                continue
            if demanded_case_tokens and (
                sum(token in root_source for token in demanded_case_tokens)
                < demanded_case_count
            ):
                continue
        if isinstance(node, ast.ClassDef):
            site = _generic_role_case_table_site(
                module=module,
                owner_symbol=node.name,
                owner_name=node.name,
                line=node.lineno,
                root=node,
                minimum_case_count=minimum_case_count,
            )
        else:
            site = _generic_role_case_table_site(
                module=module,
                owner_symbol=node.name,
                owner_name=node.name,
                line=node.lineno,
                root=node,
                minimum_case_count=minimum_case_count,
            )
        if site is not None:
            sites.append(site)
    return tuple(sorted(sites, key=lambda item: (item.file_path, item.line)))


def _generic_role_case_table_certificate(
    *,
    sites: tuple[GenericRoleCaseTableSite, ...],
    role_boundary: BroadSemanticAxisTokenBoundary,
    shared_case_tokens: tuple[str, ...],
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=sum(len(site.case_tokens) for site in sites),
        replacement_shape=ObjectFamilyShape.from_roles(
            ("generic_role_case_authority",),
            axis=("semantic_case",),
            source=("owner_projection",),
        ),
        semantic_axes=(
            ("broad_semantic_axis_tokens", role_boundary.broad_semantic_axis_tokens),
            ("shared_case_tokens", shared_case_tokens),
        ),
        independent_source_count=len({site.owner_symbol for site in sites}),
    )


def _generic_role_case_table_candidates_from_sites(
    sites: Iterable[GenericRoleCaseTableSite],
    config: DetectorConfig,
) -> tuple[GenericRoleCaseTableCandidate, ...]:
    projections = tuple(
        _GenericRoleCaseTableProjection(
            site=site, broad_semantic_axis_token=broad_semantic_axis_token
        )
        for site in sites
        if len(site.case_tokens) >= config.min_generic_role_case_table_cases
        for broad_semantic_axis_token in site.broad_semantic_axis_tokens
    )
    if len(projections) < 2:
        return ()

    axis_system = FiniteAxisSystem.from_rows(
        (
            (
                projection,
                {
                    "broad_semantic_axis_token": projection.broad_semantic_axis_token,
                    "context_kinds": projection.site.context_kinds,
                },
            )
            for projection in projections
        )
    )
    components = axis_system.confusability_components(
        (("broad_semantic_axis_token", "context_kinds"),)
    )

    candidates: list[GenericRoleCaseTableCandidate] = []
    for component in components:
        unique_sites = tuple(dict.fromkeys(projection.site for projection in component))
        if len(unique_sites) < config.min_generic_role_case_table_owners:
            continue
        owner_symbols = tuple(sorted({site.owner_symbol for site in unique_sites}))
        if len(owner_symbols) < config.min_generic_role_case_table_owners:
            continue
        shared_broad_semantic_axis_tokens = tuple(
            sorted(
                set.intersection(
                    *(set(site.broad_semantic_axis_tokens) for site in unique_sites)
                )
            )
        )
        shared_role_boundary = BroadSemanticAxisTokenBoundary(
            shared_broad_semantic_axis_tokens
        )
        case_counts = Counter(
            token for site in unique_sites for token in site.case_tokens
        )
        shared_case_tokens = tuple(
            token
            for token, count in sorted(case_counts.items())
            if count >= config.min_generic_role_case_table_owners
        )
        if len(shared_case_tokens) < config.min_generic_role_case_table_cases:
            continue
        certificate = _generic_role_case_table_certificate(
            sites=unique_sites,
            role_boundary=shared_role_boundary,
            shared_case_tokens=shared_case_tokens,
        )
        if not certificate.pays_rent:
            continue
        first_site = min(unique_sites, key=lambda item: (item.file_path, item.line))
        candidates.append(
            GenericRoleCaseTableCandidate(
                file_path=first_site.file_path,
                line=first_site.line,
                broad_semantic_axis_tokens=shared_role_boundary.broad_semantic_axis_tokens,
                shared_case_tokens=shared_case_tokens,
                owner_symbols=owner_symbols,
                case_literals=tuple(
                    sorted(
                        {
                            literal
                            for site in unique_sites
                            for literal in site.case_literals
                        }
                    )
                ),
                sites=unique_sites,
                compression_certificate=certificate,
            )
        )

    deduped: dict[
        tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
        GenericRoleCaseTableCandidate,
    ] = {}
    for candidate in candidates:
        key = (
            candidate.broad_semantic_axis_tokens,
            candidate.shared_case_tokens,
            candidate.owner_symbols,
        )
        if key not in deduped:
            deduped[key] = candidate
    return tuple(
        sorted(
            deduped.values(),
            key=lambda item: (item.file_path, item.line, item.owner_symbols),
        )
    )


def _local_role_case_logic_site(
    *,
    module: ParsedModule,
    owner_symbol: str,
    owner_name: str,
    root: ast.FunctionDef | ast.AsyncFunctionDef,
    config: DetectorConfig,
) -> LocalRoleCaseLogicCandidate | None:
    owner_tokens = ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(owner_name)
    if len(owner_tokens) < 2:
        return None
    body_tokens = _generic_role_case_body_tokens(root)
    module_tokens = {
        token
        for part in module.path.with_suffix("").parts
        for token in ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(part)
    }
    role_boundary = BroadSemanticAxisTokenBoundary.from_context(
        owner_tokens=owner_tokens,
        body_tokens=body_tokens,
        module_tokens=frozenset(module_tokens),
    )
    if role_boundary is None:
        return None

    visitor = _LocalRoleCaseLiteralCollector(role_boundary)
    visitor.visit(root)
    literal_records = visitor.records
    if not literal_records:
        return None

    case_tokens = tuple(
        sorted(
            {
                token
                for record in literal_records
                for token in record.literal_tokens
                if token not in role_boundary.broad_semantic_axis_tokens
            }
        )
    )
    if len(case_tokens) < config.min_local_role_case_logic_cases:
        return None
    case_literals = tuple(sorted({record.literal for record in literal_records}))
    context_kinds = tuple(sorted({record.context_kind for record in literal_records}))
    candidate = LocalRoleCaseLogicCandidate(
        file_path=str(module.path),
        line=root.lineno,
        owner_symbol=owner_symbol,
        owner_tokens=owner_tokens,
        broad_semantic_axis_tokens=role_boundary.broad_semantic_axis_tokens,
        case_tokens=case_tokens,
        case_literals=case_literals,
        context_kinds=context_kinds,
    )
    if not candidate.compression_certificate.pays_rent:
        return None
    return candidate


def _local_role_case_logic_candidates_for_module(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[LocalRoleCaseLogicCandidate, ...]:
    candidates: list[LocalRoleCaseLogicCandidate] = []

    class Visitor(ClassFunctionStackNodeVisitor):
        def __init__(self, module: ParsedModule) -> None:
            super().__init__()
            self.module = module

        def before_visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            owner_parts = (*self.class_stack, *self.function_stack, node.name)
            owner_symbol = ".".join(owner_parts)
            candidate = _local_role_case_logic_site(
                module=self.module,
                owner_symbol=owner_symbol,
                owner_name=owner_symbol,
                root=node,
                config=config,
            )
            if candidate is not None:
                candidates.append(candidate)

    Visitor(module).visit(module.module)
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.line, item.owner_symbol),
        )
    )


def _local_role_case_logic_candidates(
    modules: Sequence[ParsedModule],
    config: DetectorConfig,
) -> tuple[LocalRoleCaseLogicCandidate, ...]:
    return tuple(
        candidate
        for module in modules
        for candidate in _local_role_case_logic_candidates_for_module(module, config)
    )


def _native_role_surface_module(
    source_module: SourceModule,
) -> ParsedModule:
    return source_module.parsed_module(
        ast.Module(body=[], type_ignores=[]),
    )


@dataclass(frozen=True)
class CompactRoleSurfaceProjectionDemand:
    """Report-target keys that can participate in the global role-case join."""

    generic_axis_tokens: frozenset[str]
    generic_case_tokens: frozenset[str]
    minimum_generic_case_count: int


def _role_surface_report_demand(
    target_items: tuple[object, ...],
    config: object,
) -> CompactRoleSurfaceProjectionDemand:
    if not isinstance(config, DetectorConfig):
        raise TypeError("role-surface report demand requires DetectorConfig")
    projections = tuple(
        item
        for item in target_items
        if isinstance(item, CompactRoleSurfaceModuleProjection)
    )
    return CompactRoleSurfaceProjectionDemand(
        generic_axis_tokens=frozenset(
            token
            for projection in projections
            for site in projection.generic_role_case_table_sites
            for token in site.broad_semantic_axis_tokens
        ),
        generic_case_tokens=frozenset(
            token
            for projection in projections
            for site in projection.generic_role_case_table_sites
            for token in site.case_tokens
        ),
        minimum_generic_case_count=config.min_generic_role_case_table_cases,
    )


def _cached_role_surface_demand_projection(
    items: tuple[object, ...],
    demand: object,
) -> tuple[object, ...]:
    if not isinstance(demand, CompactRoleSurfaceProjectionDemand):
        raise TypeError("role-surface projection demand has the wrong authority type")
    return tuple(
        CompactRoleSurfaceModuleProjection(
            generic_role_case_table_sites=tuple(
                site
                for site in item.generic_role_case_table_sites
                if set(site.broad_semantic_axis_tokens)
                & set(demand.generic_axis_tokens)
                and len(set(site.case_tokens) & set(demand.generic_case_tokens))
                >= demand.minimum_generic_case_count
            ),
        )
        for item in items
        if isinstance(item, CompactRoleSurfaceModuleProjection)
    )


def _native_demanded_role_surface_projection(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    *,
    generic_axis_tokens: frozenset[str],
    generic_case_tokens: frozenset[str] = frozenset(),
    minimum_generic_case_count: int = 2,
) -> CompactRoleSurfaceModuleProjection | None:
    """Project exact report-correlated role facts without a module AST."""

    if not syntax_index.is_complete:
        return None
    source_text = source_module.source
    module_axis_source = " ".join(source_module.path.with_suffix("").parts).casefold()
    folded_source = source_text.casefold()
    has_generic_demand = bool(generic_axis_tokens) and any(
        token in folded_source or token in module_axis_source
        for token in generic_axis_tokens
    )
    if has_generic_demand and generic_case_tokens:
        has_generic_demand = (
            sum(token in folded_source for token in generic_case_tokens)
            >= minimum_generic_case_count
        )
    generic_sites: list[GenericRoleCaseTableSite] = []
    if has_generic_demand:
        module = _native_role_surface_module(source_module)
        for class_node in syntax_index.common_captures().get("class", ()):
            root_source = syntax_index.source_for(class_node).decode("utf-8").casefold()
            if not any(
                token in root_source or token in module_axis_source
                for token in generic_axis_tokens
            ):
                continue
            if (
                generic_case_tokens
                and sum(token in root_source for token in generic_case_tokens)
                < minimum_generic_case_count
            ):
                continue
            root = syntax_index.class_for(class_node)
            site = _generic_role_case_table_site(
                module=module,
                owner_symbol=root.name,
                owner_name=root.name,
                line=root.lineno,
                root=root,
                minimum_case_count=1,
            )
            if site is not None and set(site.broad_semantic_axis_tokens) & set(
                generic_axis_tokens
            ):
                generic_sites.append(site)
        for function_node in syntax_index.top_level_declarations("function"):
            root_source = (
                syntax_index.source_for(function_node).decode("utf-8").casefold()
            )
            if not any(
                token in root_source or token in module_axis_source
                for token in generic_axis_tokens
            ):
                continue
            if (
                generic_case_tokens
                and sum(token in root_source for token in generic_case_tokens)
                < minimum_generic_case_count
            ):
                continue
            root = syntax_index.function_for(function_node)
            site = _generic_role_case_table_site(
                module=module,
                owner_symbol=root.name,
                owner_name=root.name,
                line=root.lineno,
                root=root,
                minimum_case_count=1,
            )
            if site is not None and set(site.broad_semantic_axis_tokens) & set(
                generic_axis_tokens
            ):
                generic_sites.append(site)
    return CompactRoleSurfaceModuleProjection(
        generic_role_case_table_sites=tuple(
            sorted(generic_sites, key=lambda item: (item.file_path, item.line))
        ),
    )


def _native_demanded_role_surface_projection_items(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    demand: object,
) -> list[CompactRoleSurfaceModuleProjection] | None:
    if not isinstance(demand, CompactRoleSurfaceProjectionDemand):
        raise TypeError("role-surface projection demand has the wrong authority type")
    projection = _native_demanded_role_surface_projection(
        source_module,
        syntax_index,
        generic_axis_tokens=demand.generic_axis_tokens,
        generic_case_tokens=demand.generic_case_tokens,
        minimum_generic_case_count=demand.minimum_generic_case_count,
    )
    return None if projection is None else [projection]


def _ast_demanded_role_surface_projection_items(
    parsed_module: ParsedModule,
    demand: object,
) -> list[CompactRoleSurfaceModuleProjection]:
    if not isinstance(demand, CompactRoleSurfaceProjectionDemand):
        raise TypeError("role-surface projection demand has the wrong authority type")
    module_axis_source = " ".join(
        parsed_module.path.with_suffix("").parts
    ).casefold()
    folded_source = parsed_module.source.casefold()
    has_generic_demand = bool(demand.generic_axis_tokens) and any(
        token in folded_source or token in module_axis_source
        for token in demand.generic_axis_tokens
    )
    if has_generic_demand and demand.generic_case_tokens:
        has_generic_demand = (
            sum(token in folded_source for token in demand.generic_case_tokens)
            >= demand.minimum_generic_case_count
        )
    return [
        CompactRoleSurfaceModuleProjection(
            generic_role_case_table_sites=(
                tuple(
                    site
                    for site in _generic_role_case_table_sites_with_minimum(
                        parsed_module,
                        1,
                        demanded_axis_tokens=demand.generic_axis_tokens,
                        demanded_case_tokens=demand.generic_case_tokens,
                        demanded_case_count=demand.minimum_generic_case_count,
                    )
                    if set(site.broad_semantic_axis_tokens)
                    & set(demand.generic_axis_tokens)
                    and len(set(site.case_tokens) & set(demand.generic_case_tokens))
                    >= demand.minimum_generic_case_count
                )
                if has_generic_demand
                else ()
            ),
        )
    ]


class CompactRoleSurfaceModuleProjectionFamily(
    CollectedFamily[CompactRoleSurfaceModuleProjection]
):
    item_type = CompactRoleSurfaceModuleProjection
    cache_payload_max_bytes = 1_000_000
    source_demand_collector = staticmethod(
        _native_demanded_role_surface_projection_items
    )
    ast_demand_collector = staticmethod(_ast_demanded_role_surface_projection_items)
    report_demand_builder = staticmethod(_role_surface_report_demand)
    cached_demand_projector = staticmethod(_cached_role_surface_demand_projection)

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactRoleSurfaceModuleProjection]:
        del cls
        return [
            CompactRoleSurfaceModuleProjection(
                generic_role_case_table_sites=(
                    _generic_role_case_table_sites_with_minimum(parsed_module, 1)
                ),
            )
        ]


class GenericRoleCaseTableDetector(
    SemanticMirrorIssueDetector,
    CompactProjectionCandidateDetector[
        CompactRoleSurfaceModuleProjection,
        GenericRoleCaseTableCandidate,
    ],
):
    module_projection_family = CompactRoleSurfaceModuleProjectionFamily
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Concrete role-case tables should move behind one generic axis authority",
        "Multiple owners repeat concrete case tables under the same broader semantic axis. That keeps variant knowledge in local surfaces instead of one role-neutral authority, so adding a new concrete case requires synchronized edits and makes semantic axes easy to confuse.",
        "one generic case-table authority owned by the broader semantic axis",
        "case-table literals are algebraically confusable under the same broad owner/context token axes",
        _NOMINAL_IDENTITY_PROVENANCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _CLASS_FAMILY_KEYWORD_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    )
    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactRoleSurfaceModuleProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[GenericRoleCaseTableCandidate]:
        return _generic_role_case_table_candidates_from_sites(
            (
                site
                for projection in projections
                for site in projection.generic_role_case_table_sites
            ),
            config,
        )

    def _finding_for_candidate(
        self, candidate: GenericRoleCaseTableCandidate
    ) -> RefactorFinding:
        cases = ", ".join(candidate.shared_case_tokens)
        owners = ", ".join(candidate.owner_symbols)
        return self.build_finding(
            (
                f"Owners {owners} repeat concrete case token(s) {cases} "
                f"under broad semantic token(s) {candidate.label}; centralize the "
                "case table behind one generic axis authority."
            ),
            candidate.evidence,
            scaffold=candidate.scaffold,
            codemod_patch=candidate.codemod_patch,
            compression_certificate=candidate.compression_certificate,
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(candidate.sites),
                mapping_name="generic_role_case_table",
                field_names=candidate.shared_case_tokens,
                source_name=",".join(candidate.broad_semantic_axis_tokens),
            ),
        )


class LocalRoleCaseLogicDetector(
    SemanticMirrorIssueDetector,
    ConfiguredModuleCollectorCandidateDetector[LocalRoleCaseLogicCandidate],
):
    finding_spec = high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Broad behavior surface embeds concrete role-case logic",
        "A method or function whose owner names a broad semantic axis contains local concrete case literals. That hardcodes variant semantics in behavior code instead of routing through a nominal axis authority, so role meanings can diverge across viewers, serializers, or execution backends.",
        "nominal role-axis authority or policy object owned by the semantic axis",
        "local map/guard literals are algebraically confusable under the broad owner/body token axis",
        _NOMINAL_IDENTITY_PROVENANCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _CLASS_FAMILY_KEYWORD_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    )
    candidate_collector = staticmethod(_local_role_case_logic_candidates_for_module)

    def _finding_for_candidate(
        self, candidate: LocalRoleCaseLogicCandidate
    ) -> RefactorFinding:
        contexts = ", ".join(candidate.context_kinds)
        return self.build_finding(
            (
                f"`{candidate.owner_symbol}` embeds concrete case literal(s) "
                f"{candidate.case_label} under broad semantic token(s) {candidate.label} via {contexts}; "
                "move those semantics behind a nominal axis authority."
            ),
            candidate.evidence,
            scaffold=candidate.scaffold,
            codemod_patch=candidate.codemod_patch,
            compression_certificate=candidate.compression_certificate,
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=1,
                mapping_name="local_role_case_logic",
                field_names=candidate.case_tokens,
                source_name=",".join(candidate.broad_semantic_axis_tokens),
            ),
        )


_PUBLIC_EXPORT_POLICY = PublicExportPolicy(
    module_name=__name__,
    root_types=(IssueDetector,),
)


__all__ = derive_public_exports(globals(), _PUBLIC_EXPORT_POLICY)
