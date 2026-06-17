"""Role-surface drift detector implementation."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..semantic_algebra import ObjectFamilyShape
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
_ROLE_SURFACE_DRIFT_ITERATION_CALLS = frozenset(
    {
        "all",
        "any",
        "dict",
        "enumerate",
        "len",
        "list",
        "max",
        "min",
        "set",
        "sum",
        "tuple",
        "zip",
    }
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


class RoleSurfaceTokenProjection:
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
            sorted(
                {
                    token
                    for target in targets
                    for token in self.node_tokens(target)
                }
            )
        )


ROLE_SURFACE_TOKEN_PROJECTION = RoleSurfaceTokenProjection()


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
        if assigned_tokens:
            context_tokens.update(assigned_tokens)
            if operation_kind is None:
                operation_kind = _ROLE_SURFACE_OPERATION_ASSIGNED_FROM

        if operation_kind not in _ROLE_SURFACE_DRIFT_STRUCTURAL_OPERATIONS:
            return None

        context_tokens = {
            token
            for token in context_tokens
            if token not in _ROLE_SURFACE_DRIFT_TOKEN_STOPWORDS
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
        class_names = tuple(sorted({declaration.class_name for declaration in declarations}))
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
        sorted(candidates, key=lambda item: (item.file_path, item.line, item.field_name))
    )


class RoleSurfaceDriftDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[RoleSurfaceDriftCandidate]
):
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


__all__ = ("RoleSurfaceDriftDetector",)
