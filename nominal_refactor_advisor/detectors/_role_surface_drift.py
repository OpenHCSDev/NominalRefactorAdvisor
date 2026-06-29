"""Role-surface drift detector implementation."""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from types import EllipsisType
from typing import Iterable, Sequence

from ..ast_tools import BuiltinCallName
from ..export_tools import PublicExportPolicy, derive_public_exports
from ..semantic_algebra import FiniteAxisSystem, ObjectFamilyShape
from ..semantic_description_length import CompressionCertificate
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
_ROLE_SURFACE_DRIFT_ITERATION_CALLS = (
    BuiltinCallName.role_surface_iteration_call_names()
)
_ROLE_SURFACE_OPERATION_ASSIGNED_FROM = "assigned_from"
_ROLE_SURFACE_OPERATION_COUNTED = "counted"
_ROLE_SURFACE_OPERATION_INDEXED = "indexed"
_ROLE_SURFACE_OPERATION_ITERATED = "iterated"
_ROLE_SURFACE_OPERATION_KEYWORD_FORWARDED = "keyword_forwarded"
_ROLE_SURFACE_DRIFT_STRUCTURAL_OPERATIONS = frozenset(
    {
        _ROLE_SURFACE_OPERATION_ASSIGNED_FROM,
        _ROLE_SURFACE_OPERATION_COUNTED,
        _ROLE_SURFACE_OPERATION_INDEXED,
        _ROLE_SURFACE_OPERATION_ITERATED,
        _ROLE_SURFACE_OPERATION_KEYWORD_FORWARDED,
    }
)
_ROLE_SURFACE_BROAD_CARRIER_TOKENS = frozenset(
    {
        "context",
        "model",
        "payload",
        "semantic",
        "semantics",
    }
)
_ROLE_SURFACE_PRESENTATION_CONTEXT_TOKENS = frozenset(
    (
        *CandidateFindingRenderer.presentation_context_tokens(),
        "finding",
        "metric",
        "renderer",
    )
)
_GENERIC_ROLE_CASE_LITERAL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_GENERIC_ROLE_CASE_CONTEXT_MAP_KEY = "mapping_key"
_GENERIC_ROLE_CASE_CONTEXT_COMPARE = "compare_case"
_GENERIC_ROLE_CASE_SENTINEL_TOKENS = frozenset({"false", "none", "null", "true"})


@dataclass(frozen=True)
class RoleSurfaceFieldWitness(LineWitnessCandidate):
    field_name: str


@dataclass(frozen=True)
class RoleSurfaceDeclaration(RoleSurfaceFieldWitness):
    class_name: str
    surface_tokens: tuple[str, ...]
    role_tokens: tuple[str, ...]

    @property
    def symbol(self) -> str:
        return f"{self.class_name}.{self.field_name}"


@dataclass(frozen=True)
class RoleSurfaceUseSite(RoleSurfaceFieldWitness):
    symbol: str
    operation_kind: str
    context_tokens: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        token_summary = ",".join(self.context_tokens[:4])
        if not token_summary:
            token_summary = "structural-use"
        return SourceLocation(
            self.file_path,
            self.line,
            f"{self.symbol}:{self.field_name}:{self.operation_kind}:{token_summary}",
        )


@dataclass(frozen=True)
class RoleSurfaceDriftCandidate(RoleSurfaceFieldWitness):
    class_names: tuple[str, ...]
    declared_role_tokens: tuple[str, ...]
    observed_role_tokens: tuple[str, ...]
    operation_kinds: tuple[str, ...]
    declarations: tuple[RoleSurfaceDeclaration, ...]
    use_sites: tuple[RoleSurfaceUseSite, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        declaration_evidence = tuple(
            SourceLocation(declaration.file_path, declaration.line, declaration.symbol)
            for declaration in self.declarations[:3]
        )
        use_evidence = tuple(use_site.evidence for use_site in self.use_sites[:7])
        return (*declaration_evidence, *use_evidence)

    @property
    def compression_certificate(self) -> CompressionCertificate:
        semantic_axes = (
            *(f"declared:{token}" for token in self.declared_role_tokens),
            *(f"observed:{token}" for token in self.observed_role_tokens),
        )
        return CompressionCertificate.from_object_family(
            manual_object_count=max(
                len(self.use_sites) * (1 + len(self.observed_role_tokens))
                + len(self.declared_role_tokens),
                len(self.use_sites) + len(semantic_axes),
            ),
            replacement_shape=ObjectFamilyShape(
                shared_objects=("role_neutral_surface", "role_projection_authority"),
                per_axis_objects=("role_axis_case",),
            ),
            semantic_axes=semantic_axes,
            independent_source_count=max(len(self.class_names), 1),
        )

    @property
    def scaffold(self) -> str:
        name_tokens = self.observed_role_tokens[:2]
        if not name_tokens:
            name_tokens = ("role",)
        surface_name = _camel_case("_".join(name_tokens))
        return (
            f"@dataclass(frozen=True)\n"
            f"class {surface_name}SourceProvenance:\n"
            f"    values: tuple[object, ...]\n\n"
            f"    def for_role_index(self, index: int) -> object:\n"
            f"        return self.values[index]\n\n"
            f"# Rename `{self.field_name}` behind a role-neutral carrier and keep the\n"
            f"# concrete role axis explicit at the call boundary."
        )

    @property
    def codemod_patch(self) -> str:
        observed = ", ".join(self.observed_role_tokens)
        declared = ", ".join(self.declared_role_tokens)
        return (
            f"# `{self.field_name}` declares role token(s) {declared}, but broad "
            f"use sites repeatedly introduce observed role token(s) {observed}.\n"
            "# Introduce one role-neutral provenance/surface carrier, move the concrete "
            "axis name to the consuming policy, and keep per-role projection explicit."
        )


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
            isinstance(parent_node, ast.Return)
            and parent_node.value is not None
            and _role_surface_contains_node(parent_node.value, parent)
            for parent_node in reversed(parents)
        ):
            return None
        return _GENERIC_ROLE_CASE_CONTEXT_MAP_KEY
    if any(
        isinstance(parent_node, ast.Compare)
        and _role_surface_contains_node(parent_node, node)
        for parent_node in reversed(parents)
    ):
        return _GENERIC_ROLE_CASE_CONTEXT_COMPARE
    return None


class _GenericRoleCaseLiteralVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: list[ast.AST] = []
        self.records: list[tuple[int, str, tuple[str, ...], str]] = []

    def visit(self, node: ast.AST) -> None:
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
    config: DetectorConfig,
) -> GenericRoleCaseTableSite | None:
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

    visitor = _GenericRoleCaseLiteralVisitor()
    visitor.visit(root)
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
    if len(case_tokens) < config.min_generic_role_case_table_cases:
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


def _generic_role_case_table_sites(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[GenericRoleCaseTableSite, ...]:
    sites: list[GenericRoleCaseTableSite] = []
    top_level_functions = {
        statement
        for statement in module.module.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in _walk_nodes(module.module):
        site: GenericRoleCaseTableSite | None = None
        if isinstance(node, ast.ClassDef):
            site = _generic_role_case_table_site(
                module=module,
                owner_symbol=node.name,
                owner_name=node.name,
                line=node.lineno,
                root=node,
                config=config,
            )
        elif node in top_level_functions and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            site = _generic_role_case_table_site(
                module=module,
                owner_symbol=node.name,
                owner_name=node.name,
                line=node.lineno,
                root=node,
                config=config,
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


def _generic_role_case_table_candidates(
    modules: Sequence[ParsedModule],
    config: DetectorConfig,
) -> tuple[GenericRoleCaseTableCandidate, ...]:
    projections = tuple(
        _GenericRoleCaseTableProjection(
            site=site, broad_semantic_axis_token=broad_semantic_axis_token
        )
        for module in modules
        for site in _generic_role_case_table_sites(module, config)
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
    graph = axis_system.confusability_graph(
        (("broad_semantic_axis_token", "context_kinds"),)
    )

    candidates: list[GenericRoleCaseTableCandidate] = []
    for component in graph.connected_components:
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


def _role_surface_class_field_declarations(
    module: ParsedModule,
) -> tuple[RoleSurfaceDeclaration, ...]:
    declarations: list[RoleSurfaceDeclaration] = []
    seen: set[tuple[str, str]] = set()

    def add_field(
        class_name: str,
        field_name: str,
        line: int,
    ) -> None:
        key = (class_name, field_name)
        if key in seen:
            return
        seen.add(key)
        surface_tokens = ROLE_SURFACE_TOKEN_PROJECTION.identifier_tokens(field_name)
        role_tokens = ROLE_SURFACE_TOKEN_PROJECTION.semantic_tokens(field_name)
        if field_name.startswith("_") or len(surface_tokens) < 2 or not role_tokens:
            return
        declarations.append(
            RoleSurfaceDeclaration(
                file_path=str(module.path),
                class_name=class_name,
                field_name=field_name,
                line=line,
                surface_tokens=surface_tokens,
                role_tokens=role_tokens,
            )
        )

    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                add_field(node.name, statement.target.id, statement.lineno)
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        add_field(node.name, target.id, statement.lineno)
            elif (
                isinstance(statement, ast.FunctionDef) and statement.name == "__init__"
            ):
                for child in _walk_nodes(statement):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                            ):
                                add_field(node.name, target.attr, child.lineno)
                    elif (
                        isinstance(child, ast.AnnAssign)
                        and isinstance(child.target, ast.Attribute)
                        and isinstance(child.target.value, ast.Name)
                        and child.target.value.id == "self"
                    ):
                        add_field(node.name, child.target.attr, child.lineno)
    return tuple(declarations)


def _role_surface_contains_node(root: ast.AST, target: ast.AST) -> bool:
    return any(child is target for child in ast.walk(root))


def _role_surface_call_name(call: ast.Call | None) -> str | None:
    if call is None:
        return None
    return _call_name(call.func)


def _role_surface_assignment_target_tokens(
    parents: Sequence[ast.AST],
    node: ast.AST,
) -> tuple[str, ...]:
    for parent in reversed(parents):
        if isinstance(parent, ast.Assign) and _role_surface_contains_node(
            parent.value, node
        ):
            return ROLE_SURFACE_TOKEN_PROJECTION.target_tokens(parent.targets)
        if (
            isinstance(parent, ast.AnnAssign)
            and parent.value is not None
            and _role_surface_contains_node(parent.value, node)
        ):
            return ROLE_SURFACE_TOKEN_PROJECTION.target_tokens((parent.target,))
    return ()


class _RoleSurfaceUseVisitor(ClassFunctionStackNodeVisitor):
    def __init__(self, file_path: str, field_names: frozenset[str]) -> None:
        super().__init__()
        self.file_path = file_path
        self.field_names = field_names
        self.node_stack: list[ast.AST] = []
        self.use_sites: list[RoleSurfaceUseSite] = []
        self._seen: set[tuple[str, int, str, str, tuple[str, ...]]] = set()

    def visit(self, node: ast.AST) -> None:
        self.node_stack.append(node)
        try:
            super().visit(node)
        finally:
            self.node_stack.pop()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.field_names:
            use_site = self._use_site_for_attribute(node)
            if use_site is not None:
                key = (
                    use_site.field_name,
                    use_site.line,
                    use_site.symbol,
                    use_site.operation_kind,
                    use_site.context_tokens,
                )
                if key not in self._seen:
                    self._seen.add(key)
                    self.use_sites.append(use_site)
        self.generic_visit(node)

    def _use_site_for_attribute(self, node: ast.Attribute) -> RoleSurfaceUseSite | None:
        parents = self.node_stack[:-1]
        context_tokens: set[str] = set()
        operation_kind: str | None = None

        for parent in reversed(parents):
            if isinstance(parent, ast.Subscript) and _role_surface_contains_node(
                parent.value, node
            ):
                operation_kind = _ROLE_SURFACE_OPERATION_INDEXED
                context_tokens.update(
                    ROLE_SURFACE_TOKEN_PROJECTION.node_tokens(parent.slice)
                )
                break
            if isinstance(parent, ast.For) and _role_surface_contains_node(
                parent.iter, node
            ):
                operation_kind = _ROLE_SURFACE_OPERATION_ITERATED
                context_tokens.update(
                    ROLE_SURFACE_TOKEN_PROJECTION.node_tokens(parent.target)
                )
                break
            if isinstance(parent, ast.comprehension) and _role_surface_contains_node(
                parent.iter, node
            ):
                operation_kind = _ROLE_SURFACE_OPERATION_ITERATED
                context_tokens.update(
                    ROLE_SURFACE_TOKEN_PROJECTION.node_tokens(parent.target)
                )
                break
            if isinstance(parent, ast.keyword) and _role_surface_contains_node(
                parent.value, node
            ):
                operation_kind = _ROLE_SURFACE_OPERATION_KEYWORD_FORWARDED
                if parent.arg is not None:
                    context_tokens.update(
                        ROLE_SURFACE_TOKEN_PROJECTION.semantic_tokens(parent.arg)
                    )
                break
            if isinstance(parent, ast.Call):
                call_name = _role_surface_call_name(parent)
                if call_name in _ROLE_SURFACE_DRIFT_ITERATION_CALLS:
                    if call_name == "len":
                        operation_kind = _ROLE_SURFACE_OPERATION_COUNTED
                    else:
                        operation_kind = _ROLE_SURFACE_OPERATION_ITERATED
                    break

        assigned_tokens = _role_surface_assignment_target_tokens(parents, node)
        if assigned_tokens and operation_kind is None:
            context_tokens.update(assigned_tokens)
            operation_kind = _ROLE_SURFACE_OPERATION_ASSIGNED_FROM

        if operation_kind not in _ROLE_SURFACE_DRIFT_STRUCTURAL_OPERATIONS:
            return None

        context_tokens = {
            token
            for token in context_tokens
            if token not in _ROLE_SURFACE_DRIFT_TOKEN_STOPWORDS
            and token not in _ROLE_SURFACE_PRESENTATION_CONTEXT_TOKENS
        }
        if not context_tokens:
            return None
        return RoleSurfaceUseSite(
            file_path=self.file_path,
            line=node.lineno,
            symbol=self.qualname,
            field_name=node.attr,
            operation_kind=operation_kind,
            context_tokens=tuple(sorted(context_tokens)),
        )


def _role_surface_use_sites(
    module: ParsedModule,
    field_names: frozenset[str],
) -> tuple[RoleSurfaceUseSite, ...]:
    visitor = _RoleSurfaceUseVisitor(str(module.path), field_names)
    visitor.visit(module.module)
    return tuple(visitor.use_sites)


def _role_surface_drift_candidates(
    modules: Sequence[ParsedModule],
    config: DetectorConfig,
) -> tuple[RoleSurfaceDriftCandidate, ...]:
    declarations_by_field: dict[str, list[RoleSurfaceDeclaration]] = defaultdict(list)
    for module in modules:
        for declaration in _role_surface_class_field_declarations(module):
            declarations_by_field[declaration.field_name].append(declaration)
    if not declarations_by_field:
        return ()

    field_names = frozenset(declarations_by_field)
    uses_by_field: dict[str, list[RoleSurfaceUseSite]] = defaultdict(list)
    for module in modules:
        for use_site in _role_surface_use_sites(module, field_names):
            uses_by_field[use_site.field_name].append(use_site)

    candidates: list[RoleSurfaceDriftCandidate] = []
    for field_name, declarations in sorted(declarations_by_field.items()):
        if field_name in uses_by_field:
            field_use_sites: Sequence[RoleSurfaceUseSite] = uses_by_field[field_name]
        else:
            field_use_sites = ()
        use_sites = tuple(
            sorted(
                field_use_sites,
                key=lambda item: (
                    item.file_path,
                    item.line,
                    item.symbol,
                    item.operation_kind,
                ),
            )
        )
        if len(use_sites) < config.min_role_drift_use_sites:
            continue
        surface_tokens = frozenset(
            token
            for declaration in declarations
            for token in declaration.surface_tokens
        )
        declared_role_tokens = tuple(
            sorted(
                {
                    token
                    for declaration in declarations
                    for token in declaration.role_tokens
                }
            )
        )
        if not declared_role_tokens:
            continue
        if set(declared_role_tokens) & _ROLE_SURFACE_BROAD_CARRIER_TOKENS:
            continue
        context_token_counts = Counter(
            token
            for use_site in use_sites
            for token in use_site.context_tokens
            if token not in surface_tokens
        )
        observed_role_tokens = tuple(
            token
            for token, count in sorted(
                context_token_counts.items(), key=lambda item: (-item[1], item[0])
            )
            if count >= config.min_role_drift_token_support
        )
        if not observed_role_tokens:
            continue
        operation_kinds = tuple(
            sorted({use_site.operation_kind for use_site in use_sites})
        )
        if operation_kinds == (_ROLE_SURFACE_OPERATION_ASSIGNED_FROM,):
            continue
        class_names = tuple(
            sorted({declaration.class_name for declaration in declarations})
        )
        candidate = RoleSurfaceDriftCandidate(
            file_path=declarations[0].file_path,
            line=min(declaration.line for declaration in declarations),
            class_names=class_names,
            field_name=field_name,
            declared_role_tokens=declared_role_tokens,
            observed_role_tokens=observed_role_tokens,
            operation_kinds=operation_kinds,
            declarations=tuple(declarations),
            use_sites=use_sites,
        )
        if candidate.compression_certificate.pays_rent:
            candidates.append(candidate)
    return tuple(
        sorted(
            candidates, key=lambda item: (item.file_path, item.line, item.field_name)
        )
    )


class RoleSurfaceDriftDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[RoleSurfaceDriftCandidate]
):
    ssot_authority_boundary = True
    finding_spec = high_confidence_certified_spec(
        PatternId.NOMINAL_WITNESS_CARRIER,
        "Role-specific surface is carrying broader latent role semantics",
        "A field or API surface whose name declares one concrete role is used by several structural operations under a broader observed role family. That lets the carrier drift into a generic abstraction without a neutral boundary, so later consumers can confuse axes and provenance.",
        "role-neutral carrier or nominal witness for the broader observed role family",
        "same role-specific field is indexed, iterated, counted, or forwarded under broader role contexts",
        _NOMINAL_IDENTITY_PROVENANCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _CLASS_FAMILY_KEYWORD_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    )
    candidate_collector = staticmethod(_role_surface_drift_candidates)

    def _finding_for_candidate(
        self, candidate: RoleSurfaceDriftCandidate
    ) -> RefactorFinding:
        declared = ", ".join(candidate.declared_role_tokens)
        observed = ", ".join(candidate.observed_role_tokens)
        operations = ", ".join(candidate.operation_kinds)
        classes = ", ".join(candidate.class_names)
        return self.build_finding(
            (
                f"`{candidate.field_name}` on {classes} declares role token(s) "
                f"{declared}, but structural uses ({operations}) repeatedly "
                f"introduce broader role token(s) {observed}."
            ),
            candidate.evidence,
            scaffold=candidate.scaffold,
            codemod_patch=candidate.codemod_patch,
            compression_certificate=candidate.compression_certificate,
            metrics=FieldFamilyMetrics(
                class_count=len(candidate.class_names),
                field_count=1,
                class_names=candidate.class_names,
                field_names=(candidate.field_name,),
                execution_level="role_surface_use_graph",
                dataclass_count=0,
            ),
        )


class GenericRoleCaseTableDetector(
    SemanticMirrorIssueDetector,
    ConfiguredCrossModuleCollectorCandidateDetector[GenericRoleCaseTableCandidate]
):
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Concrete role-case tables should move behind one generic axis authority",
        "Multiple owners repeat concrete case tables under the same broader semantic axis. That keeps variant knowledge in local surfaces instead of one role-neutral authority, so adding a new concrete case requires synchronized edits and makes semantic axes easy to confuse.",
        "one generic case-table authority owned by the broader semantic axis",
        "case-table literals are algebraically confusable under the same broad owner/context token axes",
        _NOMINAL_IDENTITY_PROVENANCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _CLASS_FAMILY_KEYWORD_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    )
    candidate_collector = staticmethod(_generic_role_case_table_candidates)

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
            metrics=MappingMetrics(
                mapping_site_count=len(candidate.sites),
                field_count=len(candidate.shared_case_tokens),
                mapping_name="generic_role_case_table",
                field_names=candidate.shared_case_tokens,
                source_name=",".join(candidate.broad_semantic_axis_tokens),
            ),
        )


class LocalRoleCaseLogicDetector(
    SemanticMirrorIssueDetector,
    ConfiguredModuleCollectorCandidateDetector[LocalRoleCaseLogicCandidate]
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
            metrics=MappingMetrics(
                mapping_site_count=1,
                field_count=len(candidate.case_tokens),
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
