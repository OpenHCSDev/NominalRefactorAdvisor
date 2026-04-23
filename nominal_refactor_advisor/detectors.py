"""Detector implementations and detection substrate.

This module houses both the detector registry and the concrete detector families
used by the advisor. The public base classes below are the intended extension
points for future detector work.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Any, ClassVar, Sequence, TypeVar, cast

from metaclass_registry import AutoRegisterMeta

from .ast_tools import (
    AccessorWrapperCandidate,
    AccessorWrapperObservationFamily,
    AttributeProbeObservation,
    AttributeProbeObservationFamily,
    BuilderCallShape,
    BuilderCallShapeFamily,
    ClassMarkerObservation,
    ClassMarkerObservationFamily,
    CollectedFamily,
    ConfigDispatchObservation,
    FieldObservation,
    FieldObservationFamily,
    ConfigDispatchObservationFamily,
    DualAxisResolutionObservation,
    DualAxisResolutionObservationFamily,
    DynamicMethodInjectionObservation,
    DynamicMethodInjectionObservationFamily,
    ExportDictShapeFamily,
    InterfaceGenerationObservation,
    InterfaceGenerationObservationFamily,
    LiteralDispatchObservation,
    ExportDictShape,
    LineageMappingObservation,
    LineageMappingObservationFamily,
    MethodShape,
    MethodShapeFamily,
    ParsedModule,
    ProjectionHelperShape,
    ProjectionHelperObservationFamily,
    RegistrationShape,
    RegistrationShapeFamily,
    RuntimeTypeGenerationObservation,
    RuntimeTypeGenerationObservationFamily,
    ScopedShapeWrapperFunction,
    ScopedShapeWrapperFunctionFamily,
    ScopedShapeWrapperSpec,
    ScopedShapeWrapperSpecFamily,
    SentinelTypeObservation,
    SentinelTypeObservationFamily,
    StringLiteralDispatchObservationFamily,
    NumericLiteralDispatchObservationFamily,
    InlineStringLiteralDispatchObservationFamily,
    collect_family_items,
    _walk_nodes,
)
from .class_index import (
    ClassFamilyIndex,
    IndexedClass,
    _module_import_aliases,
    build_class_family_index,
)
from .models import (
    CERTIFIED,
    SPECULATIVE,
    STRONG_HEURISTIC,
    BranchCountMetrics,
    DispatchCountMetrics,
    FieldFamilyMetrics,
    FindingMetrics,
    FindingSpec,
    HierarchyCandidateMetrics,
    ImpactDelta,
    MappingMetrics,
    OrchestrationMetrics,
    ParameterThreadMetrics,
    ProbeCountMetrics,
    RefactorFinding,
    RegistrationMetrics,
    RepeatedMethodMetrics,
    ResolutionAxisMetrics,
    SemanticBagDescriptor,
    SentinelSimulationMetrics,
    SourceLocation,
    WitnessCarrierMetrics,
    impact_delta_semantic_bag_descriptor,
    metric_semantic_bag_descriptors,
)
from .observation_graph import (
    ObservationGraph,
    ObservationKind,
    StructuralExecutionLevel,
    StructuralObservation,
    StructuralObservationCarrier,
)
from .patterns import PatternId
from .taxonomy import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    CapabilityTag,
    CertificationLevel,
    ObservationTag,
)


_GETATTR_BUILTIN = "getattr"
_HASATTR_BUILTIN = "hasattr"
_SETATTR_BUILTIN = "setattr"
_DELATTR_BUILTIN = "delattr"
_REFLECTIVE_SELF_BUILTINS = frozenset(
    {_GETATTR_BUILTIN, _HASATTR_BUILTIN, _SETATTR_BUILTIN, _DELATTR_BUILTIN}
)
_PIPELINE_ASSIGN_STAGE = "assign"
_PIPELINE_RETURN_STAGE = "return"


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds and tuning knobs shared by all detectors."""

    min_duplicate_statements: int = 3
    min_shared_pipeline_stages: int = 5
    min_nested_builder_forwarded_params: int = 4
    min_string_cases: int = 3
    min_attribute_probes: int = 2
    min_builder_keywords: int = 3
    min_export_keys: int = 3
    min_registration_sites: int = 2
    min_prefixed_role_shared_fields: int = 2
    min_prefixed_role_bundle_fields: int = 3
    min_reflective_selector_values: int = 2
    min_hardcoded_string_sites: int = 3
    min_orchestration_function_lines: int = 150
    min_orchestration_branches: int = 15
    min_orchestration_calls: int = 50
    min_shared_parameters: int = 5
    min_parameter_family_function_lines: int = 40
    excluded_pattern_ids: tuple = ()

    @classmethod
    def from_namespace(cls, namespace: Any) -> "DetectorConfig":
        namespace_values = vars(namespace)
        excluded = tuple(namespace_values.get("excluded_pattern_ids", []) or [])
        return cls(
            min_duplicate_statements=int(namespace.min_duplicate_statements),
            min_shared_pipeline_stages=int(
                namespace_values.get("min_shared_pipeline_stages", 5)
            ),
            min_nested_builder_forwarded_params=int(
                namespace_values.get("min_nested_builder_forwarded_params", 4)
            ),
            min_string_cases=int(namespace.min_string_cases),
            min_attribute_probes=int(namespace.min_attribute_probes),
            min_builder_keywords=int(namespace.min_builder_keywords),
            min_export_keys=int(namespace.min_export_keys),
            min_registration_sites=int(namespace.min_registration_sites),
            min_reflective_selector_values=int(
                namespace_values.get("min_reflective_selector_values", 2)
            ),
            min_hardcoded_string_sites=int(namespace.min_hardcoded_string_sites),
            min_orchestration_function_lines=int(
                namespace_values.get("min_orchestration_function_lines", 150)
            ),
            min_orchestration_branches=int(
                namespace_values.get("min_orchestration_branches", 15)
            ),
            min_orchestration_calls=int(
                namespace_values.get("min_orchestration_calls", 50)
            ),
            min_shared_parameters=int(namespace_values.get("min_shared_parameters", 5)),
            min_parameter_family_function_lines=int(
                namespace_values.get("min_parameter_family_function_lines", 40)
            ),
            excluded_pattern_ids=excluded,
        )


class IssueDetector(ABC, metaclass=AutoRegisterMeta):
    """Metaclass-registered detector base class."""

    __registry_key__ = "detector_id"
    __skip_if_no_key__ = True
    detector_id: ClassVar[str | None] = None
    genericity: ClassVar[str] = "generic"
    detector_priority: ClassVar[int] = 0

    @classmethod
    def registered_detector_types(cls) -> tuple[type["IssueDetector"], ...]:
        detector_registry = cast("dict[str, type[IssueDetector]]", cls.__registry__)
        return tuple(
            sorted(
                detector_registry.values(),
                key=lambda item: (
                    item.detector_priority,
                    item.__module__,
                    getattr(item, "__firstlineno__", 0),
                    item.__qualname__,
                ),
            )
        )

    def detect(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings = self._collect_findings(modules, config)
        if config.excluded_pattern_ids:
            findings = [
                f for f in findings if f.pattern_id not in config.excluded_pattern_ids
            ]
        return sorted(
            findings,
            key=lambda finding: (finding.pattern_id, finding.title, finding.summary),
        )

    @abstractmethod
    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        raise NotImplementedError


class PerModuleIssueDetector(IssueDetector):
    """Detector base that evaluates one parsed module at a time."""

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for module in modules:
            findings.extend(self._findings_for_module(module, config))
        return findings

    @abstractmethod
    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        raise NotImplementedError


class CandidateFindingDetector(PerModuleIssueDetector, ABC):
    """Detector base for candidate-to-finding pipelines."""

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        return [
            self._finding_for_candidate(candidate)
            for candidate in self._candidate_items(module, config)
        ]

    @abstractmethod
    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        raise NotImplementedError

    @abstractmethod
    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        raise NotImplementedError


class CrossModuleCandidateDetector(IssueDetector, ABC):
    """Detector base for repository-wide candidate-to-finding pipelines."""

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        return [
            self._finding_for_candidate(candidate)
            for candidate in self._candidate_items(modules, config)
        ]

    @abstractmethod
    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        raise NotImplementedError

    @abstractmethod
    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        raise NotImplementedError


class EvidenceOnlyPerModuleDetector(PerModuleIssueDetector):
    """Per-module detector that first collects evidence and then builds one finding."""

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        evidence = self._module_evidence(module, config)
        if len(evidence) < self._minimum_evidence(config):
            return []
        return [self._build_finding(module, evidence, config)]

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        return 1

    @abstractmethod
    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        raise NotImplementedError

    @abstractmethod
    def _build_finding(
        self,
        module: ParsedModule,
        evidence: tuple[SourceLocation, ...],
        config: DetectorConfig,
    ) -> RefactorFinding:
        raise NotImplementedError


class StaticModulePatternDetector(EvidenceOnlyPerModuleDetector):
    """Evidence-only detector that emits one finding from a fixed spec."""

    finding_spec: FindingSpec

    def _build_finding(
        self,
        module: ParsedModule,
        evidence: tuple[SourceLocation, ...],
        config: DetectorConfig,
    ) -> RefactorFinding:
        return self.finding_spec.build(
            self.detector_id,
            self._summary(module, evidence),
            self._evidence_slice(evidence),
        )

    def _evidence_slice(
        self, evidence: tuple[SourceLocation, ...]
    ) -> tuple[SourceLocation, ...]:
        return evidence[:6]

    @abstractmethod
    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        raise NotImplementedError


class GroupedShapeIssueDetector(IssueDetector):
    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        groups: dict[object, list[object]] = defaultdict(list)
        for shape in self._collect_shapes(modules, config):
            groups[self._group_key(shape)].append(shape)

        findings: list[RefactorFinding] = []
        for shapes in groups.values():
            finding = self._finding_from_group(tuple(shapes), config)
            if finding is not None:
                findings.append(finding)
        return findings

    @abstractmethod
    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        raise NotImplementedError

    @abstractmethod
    def _group_key(self, shape: object) -> object:
        raise NotImplementedError

    @abstractmethod
    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        raise NotImplementedError


class FiberCollectedShapeIssueDetector(GroupedShapeIssueDetector, ABC):
    observation_kind: ObservationKind
    execution_level: StructuralExecutionLevel = StructuralExecutionLevel.FUNCTION_BODY

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        shapes = tuple(
            shape
            for module in modules
            for shape in self._module_shapes(module)
            if self._include_shape(shape, config)
        )
        groups = _fiber_grouped_shapes(
            modules,
            shapes,
            self.observation_kind,
            self.execution_level,
        )
        return [shape for group in groups for shape in group]

    @abstractmethod
    def _module_shapes(self, module: ParsedModule) -> tuple[object, ...]:
        raise NotImplementedError

    @abstractmethod
    def _include_shape(self, shape: object, config: DetectorConfig) -> bool:
        raise NotImplementedError


CollectedItemT = TypeVar("CollectedItemT")


def _collect_typed_family_items(
    module: ParsedModule,
    family: type[CollectedFamily],
    item_type: type[CollectedItemT],
) -> tuple[CollectedItemT, ...]:
    items = tuple(collect_family_items(module, family))
    if not all(isinstance(item, item_type) for item in items):
        raise TypeError(
            f"Collected items for {family.__name__} did not match {item_type.__name__}"
        )
    return cast(tuple[CollectedItemT, ...], items)


_GENERIC_PARAMETER_NAMES = frozenset(
    {
        "args",
        "cls",
        "config",
        "configs",
        "evidence",
        "finding",
        "findings",
        "group",
        "groups",
        "item",
        "items",
        "kwargs",
        "module",
        "modules",
        "node",
        "nodes",
        "observation",
        "observations",
        "parsed_module",
        "path",
        "paths",
        "root",
        "self",
        "shape",
        "shapes",
        "tmp_path",
    }
)


def _parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    return tuple(
        item.arg
        for item in (
            tuple(node.args.posonlyargs)
            + tuple(node.args.args)
            + tuple(node.args.kwonlyargs)
        )
        if item.arg not in {"self", "cls"}
    )


def _callee_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        value_name = ast.unparse(node.func.value)
        return f"{value_name}.{node.func.attr}"
    return None


@lru_cache(maxsize=None)
def _function_profiles(module: ParsedModule) -> tuple[FunctionProfile, ...]:
    profiles: list[FunctionProfile] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record(node)

        def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
            callee_name_set: set[str] = set()
            branch_count = 0
            call_count = 0
            for subnode in _walk_nodes(node):
                if isinstance(subnode, ast.If):
                    branch_count += 1
                if not isinstance(subnode, ast.Call):
                    continue
                call_count += 1
                callee_name = _callee_name(subnode)
                if callee_name is not None:
                    callee_name_set.add(callee_name)
            profiles.append(
                FunctionProfile(
                    file_path=str(module.path),
                    qualname=".".join((*self.class_stack, node.name)),
                    lineno=node.lineno,
                    line_count=end_lineno - node.lineno + 1,
                    branch_count=branch_count,
                    call_count=call_count,
                    callee_names=tuple(sorted(callee_name_set)),
                    parameter_names=_parameter_names(node),
                )
            )
            self.generic_visit(node)

    Visitor().visit(module.module)
    return tuple(sorted(profiles, key=lambda item: (item.lineno, item.qualname)))


_PRIVATE_SUBSYSTEM_TOKEN_STOPWORDS = frozenset(
    {
        "active",
        "base",
        "build",
        "builder",
        "certified",
        "collect",
        "compute",
        "context",
        "create",
        "data",
        "derive",
        "detect",
        "exact",
        "final",
        "families",
        "family",
        "for",
        "from",
        "get",
        "has",
        "helper",
        "inactive",
        "iter",
        "keyed",
        "load",
        "make",
        "manager",
        "module",
        "candidate",
        "candidates",
        "parallel",
        "prepare",
        "refresh",
        "resolve",
        "result",
        "run",
        "selection",
        "select",
        "state",
        "support",
        "update",
        "value",
        "values",
        "with",
    }
)


def _private_subsystem_name_tokens(symbol_name: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _ordered_class_name_tokens(symbol_name)
        if len(token) >= 3
        and not token.isdigit()
        and token not in _PRIVATE_SUBSYSTEM_TOKEN_STOPWORDS
    )


def _module_line_count(module: ParsedModule) -> int:
    return module.source.count("\n") + 1


def _top_level_private_symbol_references(
    node: ast.AST,
    *,
    top_level_names: frozenset[str],
    symbol_name: str,
) -> tuple[str, ...]:
    referenced: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if node.id in top_level_names and node.id != symbol_name:
                referenced.add(node.id)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            chain = _ast_attribute_chain(node)
            if chain is not None and chain[0] in top_level_names and chain[0] != symbol_name:
                referenced.add(chain[0])
            self.generic_visit(node)

    Visitor().visit(node)
    return tuple(sorted(referenced))


@lru_cache(maxsize=None)
def _top_level_private_symbol_profiles(
    module: ParsedModule,
) -> tuple[PrivateTopLevelSymbolProfile, ...]:
    private_defs = tuple(
        statement
        for statement in _trim_docstring_body(module.module.body)
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and _is_private_symbol_name(statement.name)
    )
    top_level_names = frozenset(statement.name for statement in private_defs)
    profiles: list[PrivateTopLevelSymbolProfile] = []
    for statement in private_defs:
        end_lineno = (
            statement.end_lineno if statement.end_lineno is not None else statement.lineno
        )
        profiles.append(
            PrivateTopLevelSymbolProfile(
                file_path=str(module.path),
                module_name=module.module_name,
                symbol=statement.name,
                kind="class" if isinstance(statement, ast.ClassDef) else "function",
                line=statement.lineno,
                line_count=end_lineno - statement.lineno + 1,
                name_tokens=_private_subsystem_name_tokens(statement.name),
                referenced_private_symbols=_top_level_private_symbol_references(
                    statement,
                    top_level_names=top_level_names,
                    symbol_name=statement.name,
                ),
            )
        )
    return tuple(sorted(profiles, key=lambda item: (item.line, item.symbol)))


def _suggest_private_cohort_module_name(
    candidate: PrivateCohortShouldBeModuleCandidate,
) -> str:
    module_tail = candidate.module_name.rsplit(".", 1)[-1]
    suffix_tokens = tuple(
        token
        for token in candidate.shared_tokens
        if token not in set(module_tail.split("_"))
    )
    suffix = "_".join(suffix_tokens[:3]) or "subsystem"
    return f"{module_tail}_{suffix}"


def _build_private_cohort_candidate(
    *,
    module: ParsedModule,
    module_line_count: int,
    members: tuple[PrivateTopLevelSymbolProfile, ...],
    shared_tokens: tuple[str, ...] | None,
    reference_edges: set[tuple[str, str]],
    lexical_edges: set[tuple[str, str]],
    config: DetectorConfig,
) -> PrivateCohortShouldBeModuleCandidate | None:
    min_symbol_count = max(4, config.min_registration_sites + 2)
    if len(members) < min_symbol_count:
        return None
    member_names = {member.symbol for member in members}
    total_cohort_lines = sum(member.line_count for member in members)
    if total_cohort_lines < max(60, config.min_orchestration_function_lines * 3):
        return None
    component_reference_edges = sum(
        1
        for left, right in reference_edges
        if left in member_names and right in member_names
    )
    component_lexical_edges = sum(
        1
        for left, right in lexical_edges
        if left in member_names and right in member_names
    )
    if component_reference_edges + component_lexical_edges < len(member_names) - 1:
        return None
    token_counts = Counter(token for member in members for token in member.name_tokens)
    discovered_tokens = tuple(
        token
        for token, count in sorted(
            token_counts.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )
        if count >= 2
    )
    ordered_shared_tokens = tuple(
        dict.fromkeys((*(shared_tokens or ()), *discovered_tokens))
    )
    if len(ordered_shared_tokens) < 2 and component_reference_edges < max(
        2, len(member_names) // 2
    ):
        return None
    return PrivateCohortShouldBeModuleCandidate(
        file_path=str(module.path),
        module_name=module.module_name,
        module_line_count=module_line_count,
        total_cohort_lines=total_cohort_lines,
        shared_tokens=ordered_shared_tokens[:4],
        reference_edge_count=component_reference_edges,
        lexical_edge_count=component_lexical_edges,
        symbols=members,
    )


def _dedupe_private_cohort_candidates(
    candidates: Sequence[PrivateCohortShouldBeModuleCandidate],
) -> tuple[PrivateCohortShouldBeModuleCandidate, ...]:
    accepted: list[PrivateCohortShouldBeModuleCandidate] = []
    accepted_symbol_sets: list[frozenset[str]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -item.total_cohort_lines,
            -len(item.symbols),
            item.symbols[0].line,
            item.file_path,
        ),
    ):
        symbol_names = frozenset(symbol.symbol for symbol in candidate.symbols)
        if any(
            len(symbol_names & existing) / min(len(symbol_names), len(existing)) >= 0.85
            for existing in accepted_symbol_sets
        ):
            continue
        accepted.append(candidate)
        accepted_symbol_sets.append(symbol_names)
    return tuple(
        sorted(
            accepted,
            key=lambda item: (
                item.file_path,
                item.symbols[0].line,
                -item.total_cohort_lines,
            ),
        )
    )


def _private_cohort_should_be_module_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[PrivateCohortShouldBeModuleCandidate, ...]:
    min_module_lines = max(240, config.min_orchestration_function_lines * 4)
    module_line_count = _module_line_count(module)
    if module_line_count < min_module_lines:
        return ()
    profiles = _top_level_private_symbol_profiles(module)
    min_symbol_count = max(4, config.min_registration_sites + 2)
    if len(profiles) < min_symbol_count:
        return ()
    profile_by_name = {profile.symbol: profile for profile in profiles}
    adjacency: dict[str, set[str]] = {profile.symbol: set() for profile in profiles}
    reference_edges: set[tuple[str, str]] = set()
    lexical_edges: set[tuple[str, str]] = set()
    for profile in profiles:
        for referenced_name in profile.referenced_private_symbols:
            if referenced_name not in profile_by_name:
                continue
            edge = tuple(sorted((profile.symbol, referenced_name)))
            reference_edges.add(edge)
            adjacency[edge[0]].add(edge[1])
            adjacency[edge[1]].add(edge[0])
    for left, right in combinations(profiles, 2):
        if len(set(left.name_tokens) & set(right.name_tokens)) < 2:
            continue
        edge = tuple(sorted((left.symbol, right.symbol)))
        lexical_edges.add(edge)
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])

    token_pair_candidates: list[PrivateCohortShouldBeModuleCandidate] = []
    token_pair_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for profile in profiles:
        ordered_tokens = tuple(dict.fromkeys(profile.name_tokens))
        for token_pair in combinations(ordered_tokens, 2):
            token_pair_groups[token_pair].add(profile.symbol)
    for token_pair, symbol_names in token_pair_groups.items():
        if len(symbol_names) < min_symbol_count:
            continue
        members = tuple(
            sorted(
                (profile_by_name[name] for name in symbol_names),
                key=lambda item: (item.line, item.symbol),
            )
        )
        candidate = _build_private_cohort_candidate(
            module=module,
            module_line_count=module_line_count,
            members=members,
            shared_tokens=token_pair,
            reference_edges=reference_edges,
            lexical_edges=lexical_edges,
            config=config,
        )
        if candidate is not None:
            token_pair_candidates.append(candidate)
    if token_pair_candidates:
        return _dedupe_private_cohort_candidates(token_pair_candidates)

    candidates: list[PrivateCohortShouldBeModuleCandidate] = []
    seen: set[str] = set()
    for symbol_name in sorted(adjacency):
        if symbol_name in seen or not adjacency[symbol_name]:
            continue
        stack = [symbol_name]
        component_names: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component_names:
                continue
            component_names.add(current)
            stack.extend(
                neighbor
                for neighbor in adjacency[current]
                if neighbor not in component_names
            )
        seen.update(component_names)
        if len(component_names) < min_symbol_count:
            continue
        members = tuple(
            sorted(
                (profile_by_name[name] for name in component_names),
                key=lambda item: (item.line, item.symbol),
            )
        )
        candidate = _build_private_cohort_candidate(
            module=module,
            module_line_count=module_line_count,
            members=members,
            shared_tokens=None,
            reference_edges=reference_edges,
            lexical_edges=lexical_edges,
            config=config,
        )
        if candidate is None:
            continue
        if len(candidate.symbols) > max(24, len(profiles) // 2):
            continue
        candidates.append(candidate)
    return _dedupe_private_cohort_candidates(candidates)


def _parameter_thread_family_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[ParameterThreadFamilyCandidate, ...]:
    profiles = tuple(
        profile
        for profile in _function_profiles(module)
        if len(profile.semantic_parameter_names) >= config.min_shared_parameters
    )
    candidate_map: dict[
        tuple[str, ...],
        tuple[FunctionProfile, ...],
    ] = {}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in combinations(profiles, 2):
        shared_parameter_names = tuple(
            sorted(
                set(left.semantic_parameter_names) & set(right.semantic_parameter_names)
            )
        )
        if len(shared_parameter_names) < config.min_shared_parameters:
            continue
        functions = tuple(
            profile
            for profile in profiles
            if set(shared_parameter_names) <= set(profile.semantic_parameter_names)
        )
        if len(functions) < 2:
            continue
        if not any(
            profile.line_count >= config.min_parameter_family_function_lines
            for profile in functions
        ):
            continue
        adjacency[left.qualname].add(right.qualname)
        adjacency[right.qualname].add(left.qualname)
        existing = candidate_map.get(shared_parameter_names)
        if existing is None or len(functions) > len(existing):
            candidate_map[shared_parameter_names] = functions

    candidates = [
        ParameterThreadFamilyCandidate(
            shared_parameter_names=shared_parameter_names,
            functions=functions,
        )
        for shared_parameter_names, functions in candidate_map.items()
    ]
    if not candidates:
        return ()

    profile_lookup = {profile.qualname: profile for profile in profiles}
    component_candidates: list[ParameterThreadFamilyCandidate] = []
    visited: set[str] = set()
    for profile in profiles:
        if profile.qualname in visited or profile.qualname not in adjacency:
            continue
        stack = [profile.qualname]
        component_names: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_names.add(current)
            stack.extend(sorted(adjacency[current] - visited))
        best_candidate = max(
            (
                candidate
                for candidate in candidates
                if {item.qualname for item in candidate.functions} <= component_names
            ),
            key=lambda item: (
                len(item.shared_parameter_names) * len(item.functions),
                len(item.functions),
                len(item.shared_parameter_names),
                max(profile_lookup[name].line_count for name in component_names),
            ),
        )
        component_candidates.append(best_candidate)
    return tuple(
        sorted(
            component_candidates,
            key=lambda item: (
                -len(item.shared_parameter_names),
                -len(item.functions),
                item.functions[0].qualname,
            ),
        )
    )


@lru_cache(maxsize=None)
def _iter_named_functions(
    module: ParsedModule,
) -> tuple[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef], ...]:
    functions: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            functions.append((".".join((*self.class_stack, node.name)), node))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            functions.append((".".join((*self.class_stack, node.name)), node))
            self.generic_visit(node)

    Visitor().visit(module.module)
    return tuple(functions)


def _comparison_dispatch_case(test: ast.AST) -> tuple[str, str] | None:
    if not isinstance(test, ast.Compare):
        return None
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return None
    if not isinstance(test.ops[0], (ast.Eq, ast.Is)):
        return None
    return (ast.unparse(test.left), ast.unparse(test.comparators[0]))


def _enum_dispatch_from_if(node: ast.If) -> tuple[str, tuple[str, ...]] | None:
    axis_name: str | None = None
    cases: list[str] = []
    current: ast.If | None = node
    while current is not None:
        dispatch_case = _comparison_dispatch_case(current.test)
        if dispatch_case is None:
            return None
        current_axis, case_name = dispatch_case
        if axis_name is None:
            axis_name = current_axis
        elif current_axis != axis_name:
            return None
        cases.append(case_name)
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
            continue
        current = None
    if axis_name is None or len(cases) < 2:
        return None
    return (axis_name, tuple(cases))


def _enum_dispatch_from_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, tuple[str, ...]] | None:
    body = _trim_docstring_body(function.body)
    if len(body) < 2:
        return None
    best_family: tuple[str, tuple[str, ...]] | None = None
    for start in range(len(body)):
        if not isinstance(body[start], ast.If):
            continue
        axis_name: str | None = None
        cases: list[str] = []
        for statement in body[start:]:
            if not isinstance(statement, ast.If) or statement.orelse:
                break
            dispatch_case = _comparison_dispatch_case(statement.test)
            if dispatch_case is None:
                break
            current_axis, case_name = dispatch_case
            if axis_name is None:
                axis_name = current_axis
            elif current_axis != axis_name:
                break
            cases.append(case_name)
        if axis_name is None or len(cases) < 2:
            continue
        current_family = (axis_name, tuple(cases))
        if best_family is None or len(current_family[1]) > len(best_family[1]):
            best_family = current_family
    return best_family


def _enum_dispatch_from_match(node: ast.Match) -> tuple[str, tuple[str, ...]] | None:
    cases = []
    for case in node.cases:
        if isinstance(case.pattern, ast.MatchAs) and case.pattern.name is None:
            continue
        if not isinstance(case.pattern, ast.MatchValue):
            return None
        cases.append(ast.unparse(case.pattern.value))
    if len(cases) < 2:
        return None
    return (ast.unparse(node.subject), tuple(cases))


def _enum_strategy_dispatch_candidates(
    module: ParsedModule,
) -> tuple[EnumStrategyDispatchCandidate, ...]:
    candidate_map: dict[tuple[str, str], EnumStrategyDispatchCandidate] = {}
    for qualname, function in _iter_named_functions(module):
        top_level_dispatch = _enum_dispatch_from_body(function)
        if top_level_dispatch is not None:
            axis_name, case_names = top_level_dispatch
            if any("." in case_name for case_name in case_names):
                candidate = EnumStrategyDispatchCandidate(
                    file_path=str(module.path),
                    qualname=qualname,
                    lineno=function.lineno,
                    dispatch_axis=axis_name,
                    case_names=case_names,
                )
                key = (qualname, axis_name)
                existing = candidate_map.get(key)
                if existing is None or len(candidate.case_names) > len(existing.case_names):
                    candidate_map[key] = candidate
        for subnode in _walk_nodes(function):
            dispatch_family: tuple[str, tuple[str, ...]] | None = None
            if isinstance(subnode, ast.If):
                dispatch_family = _enum_dispatch_from_if(subnode)
            elif isinstance(subnode, ast.Match):
                dispatch_family = _enum_dispatch_from_match(subnode)
            if dispatch_family is None:
                continue
            axis_name, case_names = dispatch_family
            if not any("." in case_name for case_name in case_names):
                continue
            lineno = int(getattr(subnode, "lineno", 0))
            candidate = EnumStrategyDispatchCandidate(
                file_path=str(module.path),
                qualname=qualname,
                lineno=lineno,
                dispatch_axis=axis_name,
                case_names=case_names,
            )
            key = (qualname, axis_name)
            existing = candidate_map.get(key)
            if existing is None or len(candidate.case_names) > len(existing.case_names):
                candidate_map[key] = candidate
    return tuple(
        sorted(
            candidate_map.values(),
            key=lambda item: (item.file_path, item.lineno, item.qualname),
        )
    )


def _enum_family_name(case_names: tuple[str, ...]) -> str | None:
    family_names = {case_name.split(".", 1)[0] for case_name in case_names if "." in case_name}
    if len(family_names) != 1:
        return None
    return next(iter(family_names))


def _repeated_enum_strategy_dispatch_candidates(
    module: ParsedModule,
) -> tuple["RepeatedEnumStrategyDispatchCandidate", ...]:
    candidates = _enum_strategy_dispatch_candidates(module)
    grouped: dict[tuple[str, tuple[str, ...]], tuple[EnumStrategyDispatchCandidate, ...]] = {}
    for left, right in combinations(candidates, 2):
        if left.qualname == right.qualname:
            continue
        left_family = _enum_family_name(left.case_names)
        right_family = _enum_family_name(right.case_names)
        if left_family is None or left_family != right_family:
            continue
        shared_cases = tuple(sorted(set(left.case_names) & set(right.case_names)))
        if len(shared_cases) < 2:
            continue
        functions = tuple(
            candidate
            for candidate in candidates
            if _enum_family_name(candidate.case_names) == left_family
            and set(shared_cases) <= set(candidate.case_names)
        )
        if len(functions) < 2:
            continue
        key = (left_family, shared_cases)
        existing = grouped.get(key)
        if existing is None or len(functions) > len(existing):
            grouped[key] = functions
    repeated = [
        RepeatedEnumStrategyDispatchCandidate(
            file_path=str(module.path),
            enum_family=enum_family,
            shared_case_names=shared_cases,
            functions=tuple(
                sorted(items, key=lambda item: (item.file_path, item.lineno, item.qualname))
            ),
        )
        for (enum_family, shared_cases), items in grouped.items()
    ]
    return tuple(
        sorted(
            repeated,
            key=lambda item: (-len(item.shared_case_names), -len(item.functions), item.functions[0].qualname),
        )
    )


@dataclass(frozen=True)
class _LineCaseSpec(ABC):
    line: int
    case_names: tuple[str, ...]


@dataclass(frozen=True)
class _SelectorCaseSpec(_LineCaseSpec):
    selector_method_name: str


@dataclass(frozen=True)
class _StrategySelectorSpec(_SelectorCaseSpec):
    root_name: str
    mapping_name: str


@dataclass(frozen=True)
class _GenericDispatchSpec(_LineCaseSpec):
    function_name: str


@dataclass(frozen=True)
class _AxisExpressionSite(ABC):
    axis_expression: str
    line: int


@dataclass(frozen=True)
class _SelectorAssignment(_AxisExpressionSite):
    variable_name: str
    selector_spec: _StrategySelectorSpec


@dataclass(frozen=True)
class _NestedGenericUsage(_AxisExpressionSite):
    callback_name: str
    generic_spec: _GenericDispatchSpec


@dataclass(frozen=True)
class _GuardedReturnCase:
    guard_expression: str | None
    return_value: ast.AST
    line: int

    @classmethod
    def from_returned(
        cls, guard_expression: str | None, returned: tuple[ast.AST, int]
    ) -> "_GuardedReturnCase":
        return_value, line = returned
        return cls(
            guard_expression=guard_expression,
            return_value=return_value,
            line=line,
        )


@dataclass(frozen=True)
class _SelectedConstantReturnShape:
    constant_name: str
    wrapper_name: str | None
    template_key: tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class _ModuleConstantBinding:
    line: int
    constructor_name: str | None


@dataclass(frozen=True)
class _SelectionHelperShape:
    function_name: str
    selected_field_name: str
    line: int


@dataclass(frozen=True)
class _SelectionLookupShape:
    function_name: str
    line: int


def _module_level_dict_literals(
    module: ParsedModule,
) -> dict[str, tuple[int, ast.Dict]]:
    dicts: dict[str, tuple[int, ast.Dict]] = {}
    for statement in module.module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Dict)
        ):
            dicts[statement.targets[0].id] = (statement.lineno, statement.value)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and isinstance(statement.value, ast.Dict)
        ):
            dicts[statement.target.id] = (statement.lineno, statement.value)
    return dicts


def _dict_case_names(node: ast.Dict) -> tuple[str, ...]:
    return tuple(ast.unparse(key) for key in node.keys if key is not None)


def _mapping_selector_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    known_mapping_names: frozenset[str],
) -> tuple[str, str] | None:
    parameter_names = set(_parameter_names(method))
    if not parameter_names:
        return None
    for subnode in _walk_nodes(method):
        if not isinstance(subnode, ast.Subscript):
            continue
        if not isinstance(subnode.value, ast.Name):
            continue
        mapping_name = subnode.value.id
        if mapping_name not in known_mapping_names:
            continue
        axis_expression = ast.unparse(subnode.slice)
        if axis_expression not in parameter_names:
            continue
        return (mapping_name, axis_expression)
    return None


def _strategy_selector_specs(
    module: ParsedModule,
) -> tuple[_StrategySelectorSpec, ...]:
    dict_literals = _module_level_dict_literals(module)
    known_mapping_names = frozenset(
        name
        for name, (_, node) in dict_literals.items()
        if len(_dict_case_names(node)) >= 2
    )
    specs: list[_StrategySelectorSpec] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for method in _iter_class_methods(node):
            if not _is_classmethod(method) or not method.name.startswith("for_"):
                continue
            selector_shape = _mapping_selector_shape(
                method,
                known_mapping_names=known_mapping_names,
            )
            if selector_shape is None:
                continue
            mapping_name, _ = selector_shape
            _, mapping_node = dict_literals[mapping_name]
            specs.append(
                _StrategySelectorSpec(
                    root_name=node.name,
                    selector_method_name=method.name,
                    mapping_name=mapping_name,
                    case_names=_dict_case_names(mapping_node),
                    line=method.lineno,
                )
            )
    return tuple(specs)


def _first_parameter_annotation_name(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    parameters = (
        tuple(function.args.posonlyargs)
        + tuple(function.args.args)
        + tuple(function.args.kwonlyargs)
    )
    for parameter in parameters:
        if parameter.arg in {"self", "cls"}:
            continue
        annotation_names = _annotation_type_names(parameter.annotation)
        if annotation_names:
            return annotation_names[0]
        return None
    return None


def _generic_dispatch_specs(
    module: ParsedModule,
) -> tuple[_GenericDispatchSpec, ...]:
    root_lines: dict[str, int] = {}
    case_names_by_root: dict[str, list[str]] = defaultdict(list)
    for statement in module.module.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in statement.decorator_list:
            decorator_name = _ast_terminal_name(decorator)
            if decorator_name == "singledispatch":
                root_lines[statement.name] = statement.lineno
                continue
            generic_name: str | None = None
            explicit_case_name: str | None = None
            if (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "register"
                and isinstance(decorator.value, ast.Name)
            ):
                generic_name = decorator.value.id
            elif (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "register"
                and isinstance(decorator.func.value, ast.Name)
            ):
                generic_name = decorator.func.value.id
                if decorator.args:
                    explicit_case_name = ast.unparse(decorator.args[0])
            if generic_name is None:
                continue
            case_name = explicit_case_name or _first_parameter_annotation_name(statement)
            if case_name is None:
                continue
            case_names_by_root[generic_name].append(case_name)
    return tuple(
        _GenericDispatchSpec(
            function_name=function_name,
            case_names=tuple(sorted(set(case_names_by_root[function_name]))),
            line=root_lines[function_name],
        )
        for function_name in sorted(root_lines)
        if len(set(case_names_by_root[function_name])) >= 2
    )


def _non_nested_subnodes(
    statements: Sequence[ast.stmt],
) -> tuple[ast.AST, ...]:
    nodes: list[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def generic_visit(self, node: ast.AST) -> None:
            nodes.append(node)
            super().generic_visit(node)

    visitor = Visitor()
    for statement in statements:
        visitor.visit(statement)
    return tuple(nodes)


def _selector_assignments_for_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    selector_specs: tuple[_StrategySelectorSpec, ...],
) -> tuple[_SelectorAssignment, ...]:
    selector_specs_by_name = {
        (spec.root_name, spec.selector_method_name): spec for spec in selector_specs
    }
    assignments: list[_SelectorAssignment] = []
    for subnode in _non_nested_subnodes(function.body):
        if isinstance(subnode, ast.Assign) and len(subnode.targets) == 1:
            target = subnode.targets[0]
            value = subnode.value
            if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
                continue
        elif isinstance(subnode, ast.AnnAssign):
            target = subnode.target
            value = subnode.value
            if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
                continue
        else:
            continue
        if (
            not isinstance(value.func, ast.Attribute)
            or not isinstance(value.func.value, ast.Name)
        ):
            continue
        selector_spec = selector_specs_by_name.get(
            (value.func.value.id, value.func.attr)
        )
        if selector_spec is None:
            continue
        axis_expression = None
        if value.args:
            axis_expression = ast.unparse(value.args[0])
        elif value.keywords:
            for keyword in value.keywords:
                if keyword.arg is None:
                    continue
                axis_expression = ast.unparse(keyword.value)
                break
        if axis_expression is None:
            continue
        assignments.append(
            _SelectorAssignment(
                variable_name=target.id,
                selector_spec=selector_spec,
                axis_expression=axis_expression,
                line=value.lineno,
            )
        )
    return tuple(assignments)


def _nested_generic_usages_for_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    generic_specs: tuple[_GenericDispatchSpec, ...],
) -> tuple[_NestedGenericUsage, ...]:
    generics_by_name = {spec.function_name: spec for spec in generic_specs}
    usages: list[_NestedGenericUsage] = []
    for statement in function.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for subnode in _walk_nodes(statement):
            if not isinstance(subnode, ast.Call) or not isinstance(subnode.func, ast.Name):
                continue
            generic_spec = generics_by_name.get(subnode.func.id)
            if generic_spec is None or not subnode.args:
                continue
            usages.append(
                _NestedGenericUsage(
                    callback_name=statement.name,
                    generic_spec=generic_spec,
                    axis_expression=ast.unparse(subnode.args[0]),
                    line=subnode.lineno,
                )
            )
            break
    return tuple(usages)


def _strategy_bridge_calls(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    strategy_variable_name: str,
) -> tuple[ast.Call, ...]:
    calls: list[ast.Call] = []
    for subnode in _non_nested_subnodes(function.body):
        if not isinstance(subnode, ast.Call):
            continue
        if (
            isinstance(subnode.func, ast.Attribute)
            and isinstance(subnode.func.value, ast.Name)
            and subnode.func.value.id == strategy_variable_name
        ):
            calls.append(subnode)
    return tuple(calls)


def _callback_names_referenced(call: ast.Call) -> tuple[str, ...]:
    referenced_names: set[str] = set()
    for arg in call.args:
        if isinstance(arg, ast.Name):
            referenced_names.add(arg.id)
    for keyword in call.keywords:
        if isinstance(keyword.value, ast.Name):
            referenced_names.add(keyword.value.id)
    return tuple(sorted(referenced_names))


def _split_dispatch_authority_candidates(
    module: ParsedModule,
) -> tuple[SplitDispatchAuthorityCandidate, ...]:
    selector_specs = _strategy_selector_specs(module)
    generic_specs = _generic_dispatch_specs(module)
    if not selector_specs or not generic_specs:
        return ()
    candidates: list[SplitDispatchAuthorityCandidate] = []
    candidate_keys: set[tuple[str, str, str, str]] = set()
    for qualname, function in _iter_named_functions(module):
        selector_assignments = _selector_assignments_for_function(function, selector_specs)
        if not selector_assignments:
            continue
        nested_generic_usages = _nested_generic_usages_for_function(function, generic_specs)
        if not nested_generic_usages:
            continue
        usage_by_callback = {
            usage.callback_name: usage for usage in nested_generic_usages
        }
        for selector_assignment in selector_assignments:
            strategy_calls = _strategy_bridge_calls(
                function,
                strategy_variable_name=selector_assignment.variable_name,
            )
            if not strategy_calls:
                continue
            for strategy_call in strategy_calls:
                callback_names = _callback_names_referenced(strategy_call)
                for callback_name in callback_names:
                    generic_usage = usage_by_callback.get(callback_name)
                    if generic_usage is None:
                        continue
                    key = (
                        qualname,
                        selector_assignment.selector_spec.root_name,
                        generic_usage.generic_spec.function_name,
                        callback_name,
                    )
                    if key in candidate_keys:
                        continue
                    candidate_keys.add(key)
                    strategy_call_method_name = (
                        strategy_call.func.attr
                        if isinstance(strategy_call.func, ast.Attribute)
                        else "<call>"
                    )
                    candidates.append(
                        SplitDispatchAuthorityCandidate(
                            file_path=str(module.path),
                            qualname=qualname,
                            line=function.lineno,
                            strategy_root_name=selector_assignment.selector_spec.root_name,
                            selector_method_name=selector_assignment.selector_spec.selector_method_name,
                            strategy_axis_expression=selector_assignment.axis_expression,
                            strategy_case_names=selector_assignment.selector_spec.case_names,
                            strategy_call_method_name=strategy_call_method_name,
                            generic_function_name=generic_usage.generic_spec.function_name,
                            generic_axis_expression=generic_usage.axis_expression,
                            generic_case_names=generic_usage.generic_spec.case_names,
                            bridge_callback_name=callback_name,
                            selector_line=selector_assignment.line,
                            generic_line=generic_usage.line,
                        )
                    )
    return tuple(candidates)


def _is_trivial_empty_class(node: ast.ClassDef) -> bool:
    body = _trim_docstring_body(list(node.body))
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Pass):
        return True
    return bool(
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    )


def _is_reusable_axis_base(
    class_defs_by_name: dict[str, ast.ClassDef],
    base_name: str,
) -> bool:
    if base_name.endswith("Mixin"):
        return True
    base_node = class_defs_by_name.get(base_name)
    return base_node is not None and _is_abstract_class(base_node)


def _bipartition_product_axes(
    edges: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left_name, right_name in edges:
        adjacency[left_name].add(right_name)
        adjacency[right_name].add(left_name)
    colors: dict[str, int] = {}
    for node_name in sorted(adjacency):
        if node_name in colors:
            continue
        colors[node_name] = 0
        queue = [node_name]
        while queue:
            current = queue.pop(0)
            for neighbor in sorted(adjacency[current]):
                expected = 1 - colors[current]
                if neighbor in colors:
                    if colors[neighbor] != expected:
                        return None
                    continue
                colors[neighbor] = expected
                queue.append(neighbor)
    left_axis = tuple(sorted(name for name, color in colors.items() if color == 0))
    right_axis = tuple(sorted(name for name, color in colors.items() if color == 1))
    if len(left_axis) < 2 or len(right_axis) < 2:
        return None
    return (left_axis, right_axis)


def _empty_leaf_product_family_candidates(
    module: ParsedModule,
) -> tuple[EmptyLeafProductFamilyCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    leaves: list[tuple[str, int, tuple[str, str]]] = []
    for node in _walk_nodes(module.module):
        if (
            not isinstance(node, ast.ClassDef)
            or _is_abstract_class(node)
            or not _is_trivial_empty_class(node)
        ):
            continue
        base_names = tuple(
            name
            for name in _declared_base_names(node)
            if name not in _IGNORED_BASE_NAMES
        )
        if len(base_names) != 2:
            continue
        if not all(_is_reusable_axis_base(class_defs_by_name, name) for name in base_names):
            continue
        leaves.append((node.name, node.lineno, cast(tuple[str, str], base_names)))
    if len(leaves) < 4:
        return ()
    base_graph_edges = tuple(sorted({leaf[2] for leaf in leaves}))
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left_name, right_name in base_graph_edges:
        adjacency[left_name].add(right_name)
        adjacency[right_name].add(left_name)
    visited: set[str] = set()
    candidates: list[EmptyLeafProductFamilyCandidate] = []
    for start_name in sorted(adjacency):
        if start_name in visited:
            continue
        component_nodes: set[str] = set()
        queue = [start_name]
        while queue:
            current = queue.pop(0)
            if current in component_nodes:
                continue
            component_nodes.add(current)
            visited.add(current)
            queue.extend(sorted(adjacency[current] - component_nodes))
        component_edges = tuple(
            sorted(
                edge
                for edge in base_graph_edges
                if edge[0] in component_nodes and edge[1] in component_nodes
            )
        )
        if len(component_edges) < 4:
            continue
        axes = _bipartition_product_axes(component_edges)
        if axes is None:
            continue
        left_axis, right_axis = axes
        if len(component_edges) != len(left_axis) * len(right_axis):
            continue
        leaf_map: dict[tuple[str, str], tuple[str, int]] = {}
        for class_name, line, base_names in leaves:
            if set(base_names) - component_nodes:
                continue
            left_name, right_name = base_names
            if left_name in right_axis and right_name in left_axis:
                left_name, right_name = right_name, left_name
            if left_name not in left_axis or right_name not in right_axis:
                break
            key = (left_name, right_name)
            if key in leaf_map:
                break
            leaf_map[key] = (class_name, line)
        else:
            if len(leaf_map) != len(left_axis) * len(right_axis):
                continue
            ordered_leaves = tuple(
                leaf_map[(left_name, right_name)]
                for left_name in left_axis
                for right_name in right_axis
            )
            candidates.append(
                EmptyLeafProductFamilyCandidate(
                    file_path=str(module.path),
                    left_axis_base_names=left_axis,
                    right_axis_base_names=right_axis,
                    leaf_class_names=tuple(
                        class_name for class_name, _ in ordered_leaves
                    ),
                    leaf_lines=tuple(line for _, line in ordered_leaves),
                )
            )
    return tuple(candidates)


def _self_method_call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute):
        return None
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "self":
        return None
    return node.func.attr


def _transport_shell_template_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str, str, str, str | None] | None:
    body = _trim_docstring_body(list(method.body))
    if len(body) != 2:
        return None
    assign, tail = body
    if not (
        isinstance(assign, ast.Assign)
        and len(assign.targets) == 1
        and isinstance(assign.targets[0], ast.Name)
        and isinstance(assign.value, ast.Call)
        and len(assign.value.args) >= 2
    ):
        return None
    intermediate_var_name = assign.targets[0].id
    constructor_name = _call_name(assign.value.func)
    if constructor_name is None:
        return None
    selector_attr_name: str | None = None
    for arg in assign.value.args:
        selector_attr_name = _selector_attribute_name(arg)
        if selector_attr_name is not None:
            break
    if selector_attr_name is None:
        for keyword in assign.value.keywords:
            selector_attr_name = _selector_attribute_name(keyword.value)
            if selector_attr_name is not None:
                break
    source_param_name: str | None = None
    for arg in assign.value.args:
        if isinstance(arg, ast.Name) and arg.id in _parameter_names(method):
            source_param_name = arg.id
            break
    if selector_attr_name is None or source_param_name is None:
        return None
    kwargs_helper_name: str | None = None
    for keyword in assign.value.keywords:
        if keyword.arg is not None:
            continue
        if not isinstance(keyword.value, ast.Call):
            return None
        helper_name = _self_method_call_name(keyword.value)
        if helper_name is None:
            return None
        if (
            len(keyword.value.args) != 1
            or not isinstance(keyword.value.args[0], ast.Name)
            or keyword.value.args[0].id != source_param_name
            or keyword.value.keywords
        ):
            return None
        kwargs_helper_name = helper_name
    if not isinstance(tail, ast.Return) or tail.value is None:
        return None
    outcome_method_name = _self_method_call_name(tail.value)
    if (
        outcome_method_name is None
        or not isinstance(tail.value, ast.Call)
        or len(tail.value.args) != 1
        or tail.value.keywords
    ):
        return None
    inner_call = tail.value.args[0]
    inner_hook_name = _self_method_call_name(inner_call)
    if (
        inner_hook_name is None
        or not isinstance(inner_call, ast.Call)
        or len(inner_call.args) != 1
        or inner_call.keywords
        or not isinstance(inner_call.args[0], ast.Name)
        or inner_call.args[0].id != intermediate_var_name
    ):
        return None
    return (
        selector_attr_name,
        source_param_name,
        constructor_name,
        inner_hook_name,
        outcome_method_name,
        kwargs_helper_name,
    )


def _class_direct_name_like_assignment(
    node: ast.ClassDef, attr_name: str
) -> str | None:
    value = _class_direct_assignments(node).get(attr_name)
    if value is None or not isinstance(value, (ast.Name, ast.Attribute)):
        return None
    return ast.unparse(value)


def _transport_shell_template_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[TransportShellTemplateCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    candidates: list[TransportShellTemplateCandidate] = []
    for class_name, node in sorted(class_defs_by_name.items()):
        if not _is_abstract_class(node):
            continue
        driver_method = next(
            (
                method
                for method in _iter_class_methods(node)
                if not _is_abstract_method(method)
                and (
                    shape := _transport_shell_template_shape(method)
                )
                is not None
            ),
            None,
        )
        if driver_method is None:
            continue
        shape = _transport_shell_template_shape(driver_method)
        if shape is None:
            continue
        (
            selector_attr_name,
            source_param_name,
            constructor_name,
            inner_hook_name,
            outer_hook_name,
            kwargs_helper_name,
        ) = shape
        inner_hook = _class_method_named(node, inner_hook_name)
        outer_hook = _class_method_named(node, outer_hook_name)
        if inner_hook is None or outer_hook is None:
            continue
        if not (_is_abstract_method(inner_hook) and _is_abstract_method(outer_hook)):
            continue
        descendants = tuple(
            descendant
            for descendant in _descendant_class_names(class_defs_by_name, class_name)
            if not _is_abstract_class(class_defs_by_name[descendant])
        )
        if len(descendants) < config.min_registration_sites:
            continue
        selector_value_by_class = {
            descendant: _class_direct_name_like_assignment(
                class_defs_by_name[descendant], selector_attr_name
            )
            for descendant in descendants
        }
        concrete_selector_values = tuple(
            sorted(
                {
                    selector_value_name
                    for selector_value_name in selector_value_by_class.values()
                    if selector_value_name is not None
                }
            )
        )
        if len(concrete_selector_values) < config.min_registration_sites:
            continue
        concrete_class_names = tuple(
            descendant
            for descendant in descendants
            if selector_value_by_class[descendant] is not None
        )
        candidates.append(
            TransportShellTemplateCandidate(
                file_path=str(module.path),
                line=driver_method.lineno,
                class_name=class_name,
                driver_method_name=driver_method.name,
                selector_attr_name=selector_attr_name,
                selector_value_names=concrete_selector_values,
                concrete_class_names=concrete_class_names,
                source_param_name=source_param_name,
                constructor_name=constructor_name,
                kwargs_helper_name=kwargs_helper_name,
                inner_hook_name=inner_hook_name,
                outer_hook_name=outer_hook_name,
            )
        )
    return tuple(
        sorted(candidates, key=lambda item: (item.file_path, item.line, item.class_name))
    )


_TYPE_NAME_LITERAL = "type"
_SUBJECT_NAME_FIELD = "subject_name"
_NAME_FAMILY_FIELD = "name_family"


_IDENTITY_AXIS_KEYWORDS = frozenset(
    {
        "artifact",
        "artifact_cls",
        "backend",
        "cls",
        "class",
        "component",
        "family",
        "kind",
        "key",
        "mode",
        "name",
        "request_type",
        "role",
        "stage",
        "strategy",
        _TYPE_NAME_LITERAL,
    }
)
_IDENTITY_AXIS_SUFFIXES = (
    "_cls",
    "_class",
    "_family",
    "_kind",
    "_key",
    "_mode",
    "_name",
    "_role",
    "_stage",
    "_strategy",
    "_type",
)
_EXECUTABLE_AXIS_KEYWORDS = frozenset(
    {
        "builder",
        "callback",
        "callable",
        "executor",
        "factory",
        "func",
        "function",
        "handler",
        "hook",
        "operation",
        "packager",
        "processor",
        "runner",
    }
)
_EXECUTABLE_AXIS_SUFFIXES = (
    "_builder",
    "_callback",
    "_executor",
    "_factory",
    "_func",
    "_function",
    "_handler",
    "_hook",
    "_operation",
    "_packager",
    "_processor",
    "_runner",
)


def _looks_like_type_or_nominal_key(value: str) -> bool:
    tail = value.rsplit(".", 1)[-1]
    return bool(tail) and (tail[0].isupper() or "." in value)


def _looks_like_callable_value(value: str) -> bool:
    tail = value.rsplit(".", 1)[-1]
    return bool(tail) and (
        tail.startswith(("build_", "create_", "derive_", "execute_", "handle_", "make_", "run_"))
        or tail.endswith(("_builder", "_factory", "_handler", "_runner"))
        or (tail.islower() and "_" in tail)
    )


def _identity_axis_keyword_names(keyword_map: dict[str, ast.AST]) -> tuple[str, ...]:
    names = []
    for name, value in keyword_map.items():
        normalized = name.lower()
        if (
            normalized in _IDENTITY_AXIS_KEYWORDS
            or normalized.endswith(_IDENTITY_AXIS_SUFFIXES)
            or _looks_like_type_or_nominal_key(ast.unparse(value))
        ):
            names.append(name)
    return tuple(sorted(names))


def _executable_axis_keyword_names(keyword_map: dict[str, ast.AST]) -> tuple[str, ...]:
    names = []
    for name, value in keyword_map.items():
        normalized = name.lower()
        if (
            normalized in _EXECUTABLE_AXIS_KEYWORDS
            or normalized.endswith(_EXECUTABLE_AXIS_SUFFIXES)
            or _looks_like_callable_value(ast.unparse(value))
        ):
            names.append(name)
    return tuple(sorted(names))


def _spec_axis_families(
    module: ParsedModule,
) -> tuple[SpecAxisFamily, ...]:
    def parse_entry_call(
        element: ast.AST,
    ) -> tuple[str, tuple[tuple[str, str], tuple[str, str]], tuple[str, ...]] | None:
        if not isinstance(element, ast.Call) or element.args:
            return None
        current_constructor_name = _call_name(element.func)
        if current_constructor_name is None:
            return None
        keyword_map = {
            keyword.arg: keyword.value
            for keyword in element.keywords
            if keyword.arg is not None and keyword.value is not None
        }
        if len(keyword_map) < 2:
            return None
        identity_names = _identity_axis_keyword_names(keyword_map)
        executable_names = _executable_axis_keyword_names(keyword_map)
        axis_pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
        for identity_name in identity_names:
            for executable_name in executable_names:
                if identity_name == executable_name:
                    continue
                axis_pairs.append(
                    (
                        (identity_name, executable_name),
                        (
                            ast.unparse(keyword_map[identity_name]),
                            ast.unparse(keyword_map[executable_name]),
                        ),
                    )
                )
        if not axis_pairs:
            return None
        extra_keyword_names = tuple(
            sorted(
                name
                for name in keyword_map
                if name not in set(identity_names) | set(executable_names)
            )
        )
        return (
            current_constructor_name,
            tuple(axis_pairs),
            extra_keyword_names,
        )

    families: list[SpecAxisFamily] = []
    standalone_specs_by_constructor: dict[
        str, list[tuple[str, int, tuple[tuple[tuple[str, str], tuple[str, str]], ...], tuple[str, ...]]]
    ] = defaultdict(list)
    for statement in _trim_docstring_body(module.module.body):
        family_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            if isinstance(statement.targets[0], ast.Name):
                family_name = statement.targets[0].id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            family_name = statement.target.id
            value = statement.value
        if family_name is None or value is None:
            continue
        if isinstance(value, ast.Call):
            parsed_entry = parse_entry_call(value)
            if parsed_entry is None:
                continue
            constructor_name, axis_pairs, extra_keyword_names = parsed_entry
            standalone_specs_by_constructor[constructor_name].append(
                (
                    family_name,
                    statement.lineno,
                    axis_pairs,
                    extra_keyword_names,
                )
            )
            continue
        if not isinstance(value, (ast.Tuple, ast.List)) or len(value.elts) < 2:
            continue
        entries_by_axis: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        extra_keyword_names: set[str] = set()
        constructor_name: str | None = None
        line = statement.lineno
        for element in value.elts:
            parsed_entry = parse_entry_call(element)
            if parsed_entry is None:
                entries_by_axis = {}
                break
            current_constructor_name, axis_pairs, entry_extra_keyword_names = parsed_entry
            if constructor_name is None:
                constructor_name = current_constructor_name
            elif current_constructor_name != constructor_name:
                entries_by_axis = {}
                break
            for axis_field_names, axis_pair in axis_pairs:
                entries_by_axis[axis_field_names].append(axis_pair)
            extra_keyword_names.update(entry_extra_keyword_names)
        if constructor_name is None:
            continue
        for axis_field_names, entries in entries_by_axis.items():
            if len(entries) < 2:
                continue
            families.append(
                SpecAxisFamily(
                    file_path=str(module.path),
                    line=line,
                    family_name=family_name,
                    constructor_name=constructor_name,
                    axis_field_names=axis_field_names,
                    axis_pairs=tuple(entries),
                    extra_keyword_names=tuple(sorted(extra_keyword_names)),
                )
            )
    for constructor_name, items in sorted(standalone_specs_by_constructor.items()):
        if len(items) < 2:
            continue
        ordered_items = tuple(sorted(items, key=lambda item: (item[1], item[0])))
        entries_by_axis: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for _, _, axis_pairs, _ in ordered_items:
            for axis_field_names, axis_pair in axis_pairs:
                entries_by_axis[axis_field_names].append(axis_pair)
        for axis_field_names, entries in entries_by_axis.items():
            if len(entries) < 2:
                continue
            families.append(
                SpecAxisFamily(
                    file_path=str(module.path),
                    line=ordered_items[0][1],
                    family_name=" + ".join(item[0] for item in ordered_items),
                    constructor_name=constructor_name,
                    axis_field_names=axis_field_names,
                    axis_pairs=tuple(entries),
                    extra_keyword_names=tuple(
                        sorted({name for _, _, _, names in ordered_items for name in names})
                    ),
                )
            )
    return tuple(
        sorted(families, key=lambda item: (item.file_path, item.line, item.family_name))
    )


def _cross_module_spec_axis_authority_candidates(
    modules: Sequence[ParsedModule], config: DetectorConfig
) -> tuple[CrossModuleSpecAxisAuthorityCandidate, ...]:
    del config
    families = tuple(
        family
        for module in modules
        for family in _spec_axis_families(module)
    )
    candidates: list[CrossModuleSpecAxisAuthorityCandidate] = []
    for left, right in combinations(families, 2):
        if left.file_path == right.file_path:
            continue
        if left.axis_field_names != right.axis_field_names:
            continue
        shared_pairs = tuple(
            sorted(set(left.axis_pairs) & set(right.axis_pairs))
        )
        if len(shared_pairs) < 2:
            continue
        if (
            left.constructor_name == right.constructor_name
            and left.extra_keyword_names == right.extra_keyword_names
            and left.axis_pairs == right.axis_pairs
        ):
            continue
        candidates.append(
            CrossModuleSpecAxisAuthorityCandidate(
                axis_field_names=left.axis_field_names,
                shared_axis_pairs=shared_pairs,
                families=tuple(
                    sorted(
                        (left, right),
                        key=lambda item: (item.file_path, item.line, item.family_name),
                    )
                ),
            )
        )
    deduped: dict[
        tuple[tuple[str, str], tuple[str, str], tuple[str, ...]], CrossModuleSpecAxisAuthorityCandidate
    ] = {}
    for candidate in candidates:
        family_names = tuple(family.family_name for family in candidate.families)
        key = (candidate.axis_field_names, candidate.shared_axis_pairs, family_names)
        deduped[key] = candidate
    return tuple(
        sorted(
            deduped.values(),
            key=lambda item: (
                -len(item.shared_axis_pairs),
                item.families[0].file_path,
                item.families[0].family_name,
            ),
        )
    )


def _registered_catalog_projection_candidates(
    module: ParsedModule,
) -> tuple[RegisteredCatalogProjectionCandidate, ...]:
    candidates: list[RegisteredCatalogProjectionCandidate] = []
    for qualname, function in _iter_named_functions(module):
        body = _trim_docstring_body(list(function.body))
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            continue
        returned = body[0].value
        if not isinstance(returned, ast.Call) or returned.args:
            continue
        if len(returned.keywords) != 1:
            continue
        keyword = returned.keywords[0]
        if keyword.arg is None or keyword.value is None:
            continue
        if not isinstance(keyword.value, ast.Call) or keyword.value.keywords:
            continue
        collector_name = ast.unparse(keyword.value.func)
        if len(keyword.value.args) != 2 or not isinstance(keyword.value.args[0], ast.Name):
            continue
        structure_param_name = keyword.value.args[0].id
        registry_call = keyword.value.args[1]
        if not (
            isinstance(registry_call, ast.Call)
            and not registry_call.args
            and not registry_call.keywords
            and isinstance(registry_call.func, ast.Attribute)
        ):
            continue
        extractor_base_name = ast.unparse(registry_call.func.value)
        candidates.append(
            RegisteredCatalogProjectionCandidate(
                file_path=str(module.path),
                line=function.lineno,
                qualname=qualname,
                catalog_type_name=ast.unparse(returned.func),
                collector_name=collector_name,
                structure_param_name=structure_param_name,
                extractor_base_name=extractor_base_name,
                registry_accessor_name=registry_call.func.attr,
                return_keyword_names=tuple(
                    keyword_item.arg
                    for keyword_item in returned.keywords
                    if keyword_item.arg is not None
                ),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.line, item.qualname),
        )
    )


def _is_upper_snake_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]*", name))


def _module_constant_bindings(
    module: ParsedModule,
) -> dict[str, _ModuleConstantBinding]:
    bindings: dict[str, _ModuleConstantBinding] = {}
    for statement in module.module.body:
        target_name: str | None = None
        value: ast.AST | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target_name = statement.targets[0].id
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is None or value is None or not _is_upper_snake_identifier(
            target_name
        ):
            continue
        constructor_name = (
            ast.unparse(value.func) if isinstance(value, ast.Call) else None
        )
        bindings[target_name] = _ModuleConstantBinding(
            line=statement.lineno,
            constructor_name=constructor_name,
        )
    return bindings


def _module_level_named_sequences(
    module: ParsedModule,
) -> dict[str, tuple[int, tuple[ast.AST, ...]]]:
    sequences: dict[str, tuple[int, tuple[ast.AST, ...]]] = {}
    for statement in _trim_docstring_body(module.module.body):
        target_name: str | None = None
        value: ast.AST | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target_name = statement.targets[0].id
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is None or not isinstance(value, (ast.Tuple, ast.List)):
            continue
        sequences[target_name] = (statement.lineno, tuple(value.elts))
    return sequences


def _module_level_named_calls(
    module: ParsedModule,
) -> dict[str, tuple[int, ast.Call]]:
    calls: dict[str, tuple[int, ast.Call]] = {}
    for statement in _trim_docstring_body(module.module.body):
        target_name: str | None = None
        value: ast.AST | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target_name = statement.targets[0].id
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is None or not isinstance(value, ast.Call):
            continue
        calls[target_name] = (statement.lineno, value)
    return calls


def _module_level_named_dicts(
    module: ParsedModule,
) -> dict[str, tuple[int, ast.Dict]]:
    dicts: dict[str, tuple[int, ast.Dict]] = {}
    for statement in _trim_docstring_body(module.module.body):
        target_name: str | None = None
        value: ast.AST | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target_name = statement.targets[0].id
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is None or not isinstance(value, ast.Dict):
            continue
        dicts[target_name] = (statement.lineno, value)
    return dicts


def _single_return_case(
    statements: Sequence[ast.stmt],
) -> tuple[ast.AST, int] | None:
    trimmed = _trim_docstring_body(list(statements))
    if len(trimmed) != 1 or not isinstance(trimmed[0], ast.Return):
        return None
    return_value = trimmed[0].value
    if return_value is None:
        return None
    return (return_value, trimmed[0].lineno)


def _guarded_return_cases_from_if(node: ast.If) -> tuple[_GuardedReturnCase, ...] | None:
    cases: list[_GuardedReturnCase] = []
    current: ast.If | None = node
    while current is not None:
        returned = _single_return_case(current.body)
        if returned is None:
            return None
        cases.append(
            _GuardedReturnCase.from_returned(ast.unparse(current.test), returned)
        )
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
            continue
        if current.orelse:
            fallback = _single_return_case(current.orelse)
            if fallback is None:
                return None
            cases.append(_GuardedReturnCase.from_returned(None, fallback))
        current = None
    return tuple(cases) if len(cases) >= 2 else None


def _guarded_return_cases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[_GuardedReturnCase, ...]:
    body = _trim_docstring_body(function.body)
    if not body:
        return ()
    if len(body) == 1 and isinstance(body[0], ast.If):
        return _guarded_return_cases_from_if(body[0]) or ()

    cases: list[_GuardedReturnCase] = []
    for index, statement in enumerate(body):
        if isinstance(statement, ast.If):
            if statement.orelse:
                return ()
            returned = _single_return_case(statement.body)
            if returned is None:
                return ()
            cases.append(
                _GuardedReturnCase.from_returned(ast.unparse(statement.test), returned)
            )
            continue
        if (
            isinstance(statement, ast.Return)
            and statement.value is not None
            and index == len(body) - 1
            and cases
        ):
            cases.append(
                _GuardedReturnCase.from_returned(
                    None,
                    (statement.value, statement.lineno),
                )
            )
            return tuple(cases)
        return ()
    return ()


def _selected_constant_return_shape(
    node: ast.AST,
) -> _SelectedConstantReturnShape | None:
    if isinstance(node, ast.Name) and _is_upper_snake_identifier(node.id):
        return _SelectedConstantReturnShape(
            constant_name=node.id,
            wrapper_name=None,
            template_key=("<direct>", ("__SELECTED__",), ()),
        )
    if not isinstance(node, ast.Call):
        return None

    positional_template: list[str] = []
    keyword_template: list[tuple[str, str]] = []
    constant_name: str | None = None
    constant_slot_count = 0

    for argument in node.args:
        if isinstance(argument, ast.Name) and _is_upper_snake_identifier(argument.id):
            constant_name = argument.id
            constant_slot_count += 1
            positional_template.append("__SELECTED__")
            continue
        positional_template.append(ast.unparse(argument))

    for keyword in node.keywords:
        if keyword.arg is None:
            return None
        if isinstance(keyword.value, ast.Name) and _is_upper_snake_identifier(
            keyword.value.id
        ):
            constant_name = keyword.value.id
            constant_slot_count += 1
            keyword_template.append((keyword.arg, "__SELECTED__"))
            continue
        keyword_template.append((keyword.arg, ast.unparse(keyword.value)))

    if constant_slot_count != 1 or constant_name is None:
        return None
    return _SelectedConstantReturnShape(
        constant_name=constant_name,
        wrapper_name=ast.unparse(node.func),
        template_key=(
            ast.unparse(node.func),
            tuple(positional_template),
            tuple(keyword_template),
        ),
    )


def _shared_constant_suffix(names: tuple[str, ...]) -> str | None:
    if len(names) < 2:
        return None
    token_lists = [name.split("_") for name in names]
    suffix: list[str] = []
    for shared_tokens in zip(*(reversed(tokens) for tokens in token_lists), strict=False):
        if len(set(shared_tokens)) != 1:
            break
        suffix.append(shared_tokens[0])
    if not suffix:
        return None
    return "_".join(reversed(suffix))


def _closed_constant_selector_candidates(
    module: ParsedModule,
) -> tuple[ClosedConstantSelectorCandidate, ...]:
    constant_bindings = _module_constant_bindings(module)
    candidates: list[ClosedConstantSelectorCandidate] = []
    for qualname, function in _iter_named_functions(module):
        guarded_cases = _guarded_return_cases(function)
        if len(guarded_cases) < 2:
            continue
        return_shapes = tuple(
            _selected_constant_return_shape(case.return_value) for case in guarded_cases
        )
        if any(shape is None for shape in return_shapes):
            continue
        concrete_shapes = cast(tuple[_SelectedConstantReturnShape, ...], return_shapes)
        constant_names = tuple(shape.constant_name for shape in concrete_shapes)
        if len(set(constant_names)) < 2:
            continue
        template_keys = {shape.template_key for shape in concrete_shapes}
        if len(template_keys) != 1:
            continue
        family_suffix = _shared_constant_suffix(constant_names)
        constructor_names = {
            binding.constructor_name
            for name in constant_names
            if (binding := constant_bindings.get(name)) is not None
            and binding.constructor_name is not None
        }
        common_constructor_name = (
            next(iter(constructor_names)) if len(constructor_names) == 1 else None
        )
        if family_suffix is None and common_constructor_name is None:
            continue
        evidence: list[SourceLocation] = [
            SourceLocation(str(module.path), function.lineno, qualname)
        ]
        for constant_name in constant_names:
            binding = constant_bindings.get(constant_name)
            if binding is None:
                continue
            evidence.append(
                SourceLocation(str(module.path), binding.line, constant_name)
            )
        candidates.append(
            ClosedConstantSelectorCandidate(
                file_path=str(module.path),
                qualname=qualname,
                line=function.lineno,
                guard_expressions=tuple(
                    case.guard_expression
                    for case in guarded_cases
                    if case.guard_expression is not None
                ),
                constant_names=tuple(dict.fromkeys(constant_names)),
                wrapper_name=concrete_shapes[0].wrapper_name,
                family_suffix=family_suffix,
                common_constructor_name=common_constructor_name,
                evidence_locations=tuple(evidence[:6]),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.line, item.qualname),
        )
    )


def _call_uses_iteration_variable(
    node: ast.AST,
    iteration_variable_name: str,
) -> bool:
    return any(
        isinstance(subnode, ast.Name) and subnode.id == iteration_variable_name
        for subnode in _walk_nodes(node)
    )


def _comprehension_builder_names(
    module: ParsedModule,
    family_name: str,
) -> tuple[str, ...]:
    builder_names: set[str] = set()
    for subnode in _walk_nodes(module.module):
        if not isinstance(
            subnode, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            continue
        if len(subnode.generators) != 1:
            continue
        generator = subnode.generators[0]
        if generator.ifs or not isinstance(generator.iter, ast.Name):
            continue
        if generator.iter.id != family_name or not isinstance(generator.target, ast.Name):
            continue
        iteration_variable_name = generator.target.id
        candidate_calls: list[ast.Call] = []
        if isinstance(subnode, ast.DictComp):
            candidate_nodes = (subnode.key, subnode.value)
        else:
            candidate_nodes = (subnode.elt,)
        for candidate_node in candidate_nodes:
            if candidate_node is None:
                continue
            for nested in _walk_nodes(candidate_node):
                if isinstance(nested, ast.Call) and _call_uses_iteration_variable(
                    nested, iteration_variable_name
                ):
                    candidate_calls.append(nested)
        for call in candidate_calls:
            call_name = _call_name(call.func)
            if call_name is not None:
                builder_names.add(call_name)
    return tuple(sorted(builder_names))


def _named_family_for_constants(
    named_sequences: dict[str, tuple[int, tuple[ast.AST, ...]]],
    constant_names: tuple[str, ...],
) -> str | None:
    constant_set = set(constant_names)
    for family_name, (_, elements) in sorted(named_sequences.items()):
        element_names = tuple(
            element.id for element in elements if isinstance(element, ast.Name)
        )
        if len(element_names) != len(elements):
            continue
        if constant_set <= set(element_names):
            return family_name
    return None


def _derived_wrapper_spec_shadow_candidates(
    module: ParsedModule,
) -> tuple[DerivedWrapperSpecShadowCandidate, ...]:
    constant_bindings = _module_constant_bindings(module)
    named_sequences = _module_level_named_sequences(module)
    candidates: list[DerivedWrapperSpecShadowCandidate] = []
    for family_name, (family_line, elements) in sorted(named_sequences.items()):
        if len(elements) < 2 or not all(isinstance(element, ast.Call) for element in elements):
            continue
        entry_calls = cast(tuple[ast.Call, ...], elements)
        constructor_names = {_call_name(element.func) for element in entry_calls}
        if len(constructor_names) != 1 or None in constructor_names:
            continue
        keyword_maps: list[dict[str, ast.AST]] = []
        for element in entry_calls:
            keyword_map = {
                keyword.arg: keyword.value
                for keyword in element.keywords
                if keyword.arg is not None and keyword.value is not None
            }
            if not keyword_map:
                keyword_maps = []
                break
            keyword_maps.append(keyword_map)
        if not keyword_maps:
            continue
        common_keyword_names = set(keyword_maps[0])
        for keyword_map in keyword_maps[1:]:
            common_keyword_names &= set(keyword_map)
        if not common_keyword_names:
            continue
        builder_names = _comprehension_builder_names(module, family_name)
        if not builder_names:
            continue
        for link_field_name in sorted(common_keyword_names):
            referenced_constant_names: list[str] = []
            for keyword_map in keyword_maps:
                referenced = keyword_map[link_field_name]
                if not isinstance(referenced, ast.Name) or not _is_upper_snake_identifier(
                    referenced.id
                ):
                    referenced_constant_names = []
                    break
                referenced_constant_names.append(referenced.id)
            if len(set(referenced_constant_names)) < 2:
                continue
            primary_constructor_names = {
                binding.constructor_name
                for constant_name in referenced_constant_names
                if (binding := constant_bindings.get(constant_name)) is not None
                and binding.constructor_name is not None
            }
            if len(primary_constructor_names) != 1:
                continue
            primary_constant_names = tuple(dict.fromkeys(referenced_constant_names))
            primary_family_name = _named_family_for_constants(
                named_sequences, primary_constant_names
            )
            extra_field_names = tuple(
                sorted(name for name in common_keyword_names if name != link_field_name)
            )
            evidence: list[SourceLocation] = [
                SourceLocation(str(module.path), family_line, family_name)
            ]
            evidence.extend(
                SourceLocation(
                    str(module.path),
                    constant_bindings[name].line,
                    name,
                )
                for name in primary_constant_names[:3]
                if name in constant_bindings
            )
            candidates.append(
                DerivedWrapperSpecShadowCandidate(
                    file_path=str(module.path),
                    line=family_line,
                    derived_family_name=family_name,
                    derived_constructor_name=next(iter(constructor_names)),
                    primary_family_name=primary_family_name,
                    primary_constructor_name=next(iter(primary_constructor_names)),
                    link_field_name=link_field_name,
                    primary_constant_names=primary_constant_names,
                    extra_field_names=extra_field_names,
                    builder_names=builder_names,
                    evidence_locations=tuple(evidence[:6]),
                )
            )
            break
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.line, item.derived_family_name),
        )
    )


def _dataclass_field_names(node: ast.ClassDef) -> tuple[str, ...]:
    field_names: list[str] = []
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            field_names.append(statement.target.id)
    return tuple(field_names)


def _selection_helper_shape(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _SelectionHelperShape | None:
    body = _trim_docstring_body(function.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    returned = body[0].value
    if not isinstance(returned, ast.DictComp) or len(returned.generators) != 1:
        return None
    generator = returned.generators[0]
    if generator.ifs or not isinstance(generator.target, ast.Name):
        return None
    target_name = generator.target.id
    key = returned.key
    value = returned.value
    if not (
        isinstance(key, ast.Attribute)
        and isinstance(key.value, ast.Name)
        and key.value.id == target_name
        and key.attr == "key"
    ):
        return None
    if not (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == target_name
    ):
        return None
    return _SelectionHelperShape(
        function_name=function.name,
        selected_field_name=value.attr,
        line=function.lineno,
    )


def _selection_lookup_shape(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _SelectionLookupShape | None:
    body = _trim_docstring_body(function.body)
    if len(body) != 1 or not isinstance(body[0], ast.Try):
        return None
    try_node = body[0]
    if len(try_node.body) != 1 or len(try_node.handlers) != 1:
        return None
    return_stmt = try_node.body[0]
    if not isinstance(return_stmt, ast.Return) or return_stmt.value is None:
        return None
    returned = return_stmt.value
    if not isinstance(returned, ast.Subscript):
        return None
    if not isinstance(returned.value, ast.Name) or not isinstance(returned.slice, ast.Name):
        return None
    handler = try_node.handlers[0]
    if not isinstance(handler.type, ast.Name) or handler.type.id != "KeyError":
        return None
    if not handler.body or not isinstance(handler.body[0], ast.Raise):
        return None
    return _SelectionLookupShape(function_name=function.name, line=function.lineno)


def _module_keyed_selection_helper_candidates(
    module: ParsedModule,
) -> tuple[ModuleKeyedSelectionHelperCandidate, ...]:
    helper_shapes = tuple(
        helper
        for _, function in _iter_named_functions(module)
        if "." not in _
        and (helper := _selection_helper_shape(function)) is not None
    )
    lookup_shapes = tuple(
        lookup
        for _, function in _iter_named_functions(module)
        if "." not in _
        and (lookup := _selection_lookup_shape(function)) is not None
    )
    if not helper_shapes or not lookup_shapes:
        return ()
    named_sequences = _module_level_named_sequences(module)
    named_calls = _module_level_named_calls(module)
    candidates: list[ModuleKeyedSelectionHelperCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or not _is_dataclass_class(node):
            continue
        field_names = _dataclass_field_names(node)
        if len(field_names) != 2 or field_names[0] != "key":
            continue
        selected_field_name = field_names[1]
        matching_helpers = tuple(
            helper for helper in helper_shapes if helper.selected_field_name == selected_field_name
        )
        if not matching_helpers:
            continue
        rule_table_names: list[str] = []
        indexed_table_names: list[str] = []
        evidence: list[SourceLocation] = [
            SourceLocation(str(module.path), node.lineno, node.name)
        ]
        for family_name, (line, elements) in sorted(named_sequences.items()):
            if len(elements) < 2:
                continue
            if not all(
                isinstance(element, ast.Call) and _call_name(element.func) == node.name
                for element in elements
            ):
                continue
            keyword_maps = [
                {
                    keyword.arg: keyword.value
                    for keyword in element.keywords
                    if keyword.arg is not None and keyword.value is not None
                }
                for element in cast(tuple[ast.Call, ...], elements)
            ]
            if not all(
                "key" in keyword_map and selected_field_name in keyword_map
                for keyword_map in keyword_maps
            ):
                continue
            rule_table_names.append(family_name)
            evidence.append(SourceLocation(str(module.path), line, family_name))
        if len(rule_table_names) < 2:
            continue
        helper_names = {helper.function_name for helper in matching_helpers}
        for call_name, (line, call) in sorted(named_calls.items()):
            if _call_name(call.func) not in helper_names or not call.args:
                continue
            argument = call.args[0]
            if isinstance(argument, ast.Name) and argument.id in rule_table_names:
                indexed_table_names.append(call_name)
                evidence.append(SourceLocation(str(module.path), line, call_name))
        if len(indexed_table_names) < 2:
            continue
        candidates.append(
            ModuleKeyedSelectionHelperCandidate(
                file_path=str(module.path),
                line=node.lineno,
                rule_class_name=node.name,
                selected_field_name=selected_field_name,
                helper_function_name=matching_helpers[0].function_name,
                lookup_function_name=lookup_shapes[0].function_name,
                rule_table_names=tuple(rule_table_names),
                index_table_names=tuple(indexed_table_names),
                evidence_locations=tuple(evidence[:6]),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.line, item.rule_class_name),
        )
    )


@dataclass(frozen=True)
class _FileAxisCaseSpec(_LineCaseSpec):
    file_path: str
    key_type_name: str


@dataclass(frozen=True)
class _FamilyAxisSpec(_FileAxisCaseSpec):
    family_name: str


@dataclass(frozen=True)
class _KeyedFamilyAxisSpec(_FamilyAxisSpec):
    family_label: str | None
    registry_key_attr_name: str


@dataclass(frozen=True)
class _ManualSelectorAxisSpec(_FamilyAxisSpec):
    selector_method_name: str


@dataclass(frozen=True)
class _KeyedTableAxisSpec(_FileAxisCaseSpec):
    table_name: str
    value_shape_name: str | None


@dataclass(frozen=True)
class _ClassAssignedEnumAxisSpec:
    file_path: str
    line: int
    class_name: str
    key_attr_name: str
    key_type_name: str
    case_name: str


def _keyed_family_key_type_name(node: ast.ClassDef) -> str | None:
    for base in node.bases:
        if not isinstance(base, ast.Subscript):
            continue
        if _call_name(base.value) != "KeyedNominalFamily":
            continue
        type_names = _annotation_type_names(base.slice)
        if type_names:
            return type_names[0]
    return None


def _keyed_family_axis_specs(
    modules: Sequence[ParsedModule],
) -> tuple[_KeyedFamilyAxisSpec, ...]:
    class_index = build_class_family_index(list(modules))
    specs: list[_KeyedFamilyAxisSpec] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        node = indexed_class.node
        key_type_name = _keyed_family_key_type_name(node)
        if key_type_name is None:
            continue
        registry_key_attr_name = _constant_string(
            _class_direct_assignments(node).get("registry_key_attr")
        )
        if registry_key_attr_name is None:
            continue
        case_names = tuple(
            sorted(
                {
                    ast.unparse(assignment)
                    for descendant in _indexed_descendant_classes(
                        class_index, indexed_class.symbol
                    )
                    if (
                        assignment := _class_direct_assignments(descendant.node).get(
                            registry_key_attr_name
                        )
                    )
                    is not None
                }
            )
        )
        if len(case_names) < 2:
            continue
        specs.append(
            _KeyedFamilyAxisSpec(
                file_path=indexed_class.file_path,
                line=indexed_class.line,
                family_name=_indexed_class_display_name(indexed_class, class_index),
                key_type_name=key_type_name,
                family_label=_constant_string(
                    _class_direct_assignments(node).get("family_label")
                ),
                registry_key_attr_name=registry_key_attr_name,
                case_names=case_names,
            )
        )
    return tuple(specs)


def _case_overlap_ratio(
    left_case_names: tuple[str, ...],
    right_case_names: tuple[str, ...],
) -> float:
    if not left_case_names or not right_case_names:
        return 0.0
    shared_case_count = len(set(left_case_names) & set(right_case_names))
    return shared_case_count / float(min(len(left_case_names), len(right_case_names)))


def _parallel_keyed_family_name_overlap(
    left_family_name: str,
    right_family_name: str,
) -> float:
    left_tokens = _class_name_tokens(left_family_name)
    right_tokens = _class_name_tokens(right_family_name)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / float(min(len(left_tokens), len(right_tokens)))


def _identifier_name_overlap(left_name: str, right_name: str) -> float:
    left_tokens = _class_name_tokens(left_name)
    right_tokens = _class_name_tokens(right_name)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / float(min(len(left_tokens), len(right_tokens)))


def _module_keyed_table_axis_specs(
    module: ParsedModule,
) -> tuple[_KeyedTableAxisSpec, ...]:
    specs: list[_KeyedTableAxisSpec] = []
    for table_name, (line, mapping) in sorted(_module_level_named_dicts(module).items()):
        if len(mapping.keys) < 2 or any(key is None for key in mapping.keys):
            continue
        case_names = tuple(ast.unparse(key) for key in mapping.keys if key is not None)
        key_type_name = _enum_family_name(case_names)
        if key_type_name is None:
            continue
        value_shape_name: str | None = None
        all_values_are_calls = all(isinstance(value, ast.Call) for value in mapping.values)
        value_constructor_names = {
            ast.unparse(value.func)
            for value in mapping.values
            if isinstance(value, ast.Call)
        }
        if all_values_are_calls and len(value_constructor_names) == 1:
            value_shape_name = next(iter(value_constructor_names))
        specs.append(
            _KeyedTableAxisSpec(
                file_path=str(module.path),
                line=line,
                table_name=table_name,
                key_type_name=key_type_name,
                case_names=tuple(sorted(case_names)),
                value_shape_name=value_shape_name,
            )
        )
    return tuple(specs)


def _module_class_assigned_enum_axis_specs(
    module: ParsedModule,
) -> tuple[_ClassAssignedEnumAxisSpec, ...]:
    specs: list[_ClassAssignedEnumAxisSpec] = []
    for statement in _trim_docstring_body(module.module.body):
        if not isinstance(statement, ast.ClassDef):
            continue
        assignments = _class_direct_assignments(statement)
        for key_attr_name, value in assignments.items():
            if value is None:
                continue
            case_name = ast.unparse(value)
            key_type_name = _enum_family_name((case_name,))
            if key_type_name is None:
                continue
            specs.append(
                _ClassAssignedEnumAxisSpec(
                    file_path=str(module.path),
                    line=statement.lineno,
                    class_name=statement.name,
                    key_attr_name=key_attr_name,
                    key_type_name=key_type_name,
                    case_name=case_name,
                )
            )
    return tuple(specs)


def _enum_keyed_table_class_axis_shadow_candidates(
    module: ParsedModule,
) -> tuple["EnumKeyedTableClassAxisShadowCandidate", ...]:
    class_axis_specs = _module_class_assigned_enum_axis_specs(module)
    if not class_axis_specs:
        return ()
    axis_specs_by_key: dict[tuple[str, str], list[_ClassAssignedEnumAxisSpec]] = (
        defaultdict(list)
    )
    for axis_spec in class_axis_specs:
        axis_specs_by_key[(axis_spec.key_type_name, axis_spec.key_attr_name)].append(
            axis_spec
        )
    candidates: list[EnumKeyedTableClassAxisShadowCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for table_name, (line, mapping) in sorted(_module_level_named_dicts(module).items()):
        if len(mapping.keys) < 2 or any(key is None for key in mapping.keys):
            continue
        table_case_names = tuple(ast.unparse(key) for key in mapping.keys if key is not None)
        key_type_name = _enum_family_name(table_case_names)
        if key_type_name is None:
            continue
        if not all(isinstance(value, (ast.Name, ast.Attribute)) for value in mapping.values):
            continue
        value_type_names = tuple(ast.unparse(value) for value in mapping.values)
        if not value_type_names or not all(
            _looks_like_type_or_nominal_key(value_name)
            for value_name in value_type_names
        ):
            continue
        for (axis_key_type_name, key_attr_name), axis_specs in sorted(
            axis_specs_by_key.items()
        ):
            if axis_key_type_name != key_type_name:
                continue
            class_sites = tuple(
                sorted(
                    {(axis_spec.class_name, axis_spec.line) for axis_spec in axis_specs},
                    key=lambda item: (item[1], item[0]),
                )
            )
            if len(class_sites) < 2:
                continue
            class_case_names = tuple(sorted({axis_spec.case_name for axis_spec in axis_specs}))
            shared_case_names = tuple(
                sorted(set(class_case_names) & set(table_case_names))
            )
            if len(shared_case_names) < 2:
                continue
            case_overlap_ratio = _case_overlap_ratio(
                tuple(sorted(table_case_names)),
                class_case_names,
            )
            if case_overlap_ratio < 0.8:
                continue
            key = (str(module.path), table_name, key_attr_name)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                EnumKeyedTableClassAxisShadowCandidate(
                    file_path=str(module.path),
                    line=line,
                    table_name=table_name,
                    key_type_name=key_type_name,
                    key_attr_name=key_attr_name,
                    class_sites=class_sites,
                    shared_case_names=shared_case_names,
                    value_type_names=tuple(sorted(set(value_type_names))),
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.key_type_name,
                item.table_name,
                item.key_attr_name,
            ),
        )
    )


def _parallel_keyed_table_and_family_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[ParallelKeyedTableAndFamilyCandidate, ...]:
    family_specs_by_file: dict[str, list[_KeyedFamilyAxisSpec]] = {}
    for family_spec in _keyed_family_axis_specs(modules):
        family_specs_by_file.setdefault(family_spec.file_path, []).append(family_spec)
    candidates: list[ParallelKeyedTableAndFamilyCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for module in modules:
        table_specs = _module_keyed_table_axis_specs(module)
        family_specs = family_specs_by_file.get(str(module.path), ())
        for table_spec in table_specs:
            for family_spec in family_specs:
                if table_spec.key_type_name != family_spec.key_type_name:
                    continue
                shared_case_names = tuple(
                    sorted(set(table_spec.case_names) & set(family_spec.case_names))
                )
                if len(shared_case_names) < 2:
                    continue
                case_overlap_ratio = _case_overlap_ratio(
                    table_spec.case_names,
                    family_spec.case_names,
                )
                if case_overlap_ratio < 0.8:
                    continue
                table_overlap = _identifier_name_overlap(
                    table_spec.table_name,
                    family_spec.family_name,
                )
                value_overlap = (
                    0.0
                    if table_spec.value_shape_name is None
                    else _identifier_name_overlap(
                        table_spec.value_shape_name,
                        family_spec.family_name,
                    )
                )
                if max(table_overlap, value_overlap) < 0.5:
                    continue
                key = (
                    table_spec.file_path,
                    table_spec.table_name,
                    family_spec.family_name,
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    ParallelKeyedTableAndFamilyCandidate(
                        table=table_spec,
                        family_name=family_spec.family_name,
                        family_line=family_spec.line,
                        shared_case_names=shared_case_names,
                    )
                )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.key_type_name,
                item.table_name,
                item.family_name,
            ),
        )
    )


def _parallel_keyed_axis_family_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[ParallelKeyedAxisFamilyCandidate, ...]:
    specs = _keyed_family_axis_specs(modules)
    candidates: list[ParallelKeyedAxisFamilyCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for index, left_spec in enumerate(specs):
        for right_spec in specs[index + 1 :]:
            if left_spec.file_path == right_spec.file_path:
                continue
            if left_spec.key_type_name != right_spec.key_type_name:
                continue
            if left_spec.registry_key_attr_name != right_spec.registry_key_attr_name:
                continue
            shared_case_names = tuple(
                sorted(set(left_spec.case_names) & set(right_spec.case_names))
            )
            if len(shared_case_names) < 2:
                continue
            family_label_match = (
                left_spec.family_label is not None
                and left_spec.family_label == right_spec.family_label
            )
            case_overlap_ratio = _case_overlap_ratio(
                left_spec.case_names,
                right_spec.case_names,
            )
            name_overlap_ratio = _parallel_keyed_family_name_overlap(
                left_spec.family_name,
                right_spec.family_name,
            )
            if not family_label_match and (
                case_overlap_ratio < 0.8 or name_overlap_ratio < 0.6
            ):
                continue
            key = tuple(
                sorted(
                    (left_spec.family_name, right_spec.family_name),
                )
            ) + (left_spec.key_type_name,)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                ParallelKeyedAxisFamilyCandidate(
                    key_type_name=left_spec.key_type_name,
                    left=KeyedAxisFamilySite(
                        file_path=left_spec.file_path,
                        line=left_spec.line,
                        family_name=left_spec.family_name,
                        family_label=left_spec.family_label,
                    ),
                    right=KeyedAxisFamilySite(
                        file_path=right_spec.file_path,
                        line=right_spec.line,
                        family_name=right_spec.family_name,
                        family_label=right_spec.family_label,
                    ),
                    shared_case_names=shared_case_names,
                    case_overlap_ratio=case_overlap_ratio,
                    name_overlap_ratio=name_overlap_ratio,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.key_type_name,
                item.left.file_path,
                item.left.family_name,
                item.right.file_path,
                item.right.family_name,
            ),
        )
    )


def _manual_selector_axis_specs(
    modules: Sequence[ParsedModule],
) -> tuple[_ManualSelectorAxisSpec, ...]:
    specs: list[_ManualSelectorAxisSpec] = []
    for module in modules:
        for selector_spec in _strategy_selector_specs(module):
            key_type_name = _enum_family_name(selector_spec.case_names)
            if key_type_name is None:
                continue
            specs.append(
                _ManualSelectorAxisSpec(
                    file_path=str(module.path),
                    line=selector_spec.line,
                    family_name=selector_spec.root_name,
                    selector_method_name=selector_spec.selector_method_name,
                    key_type_name=key_type_name,
                    case_names=selector_spec.case_names,
                )
            )
    return tuple(specs)


def _cross_module_axis_shadow_family_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[CrossModuleAxisShadowFamilyCandidate, ...]:
    authoritative_specs = _keyed_family_axis_specs(modules)
    shadow_specs = _manual_selector_axis_specs(modules)
    candidates: list[CrossModuleAxisShadowFamilyCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for authoritative_spec in authoritative_specs:
        for shadow_spec in shadow_specs:
            if authoritative_spec.file_path == shadow_spec.file_path:
                continue
            if authoritative_spec.key_type_name != shadow_spec.key_type_name:
                continue
            shared_case_names = tuple(
                sorted(set(authoritative_spec.case_names) & set(shadow_spec.case_names))
            )
            if len(shared_case_names) < 2:
                continue
            key = (
                authoritative_spec.family_name,
                shadow_spec.family_name,
                authoritative_spec.key_type_name,
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                CrossModuleAxisShadowFamilyCandidate(
                    key_type_name=authoritative_spec.key_type_name,
                    authoritative=AxisFamilySite(
                        file_path=authoritative_spec.file_path,
                        line=authoritative_spec.line,
                        family_name=authoritative_spec.family_name,
                    ),
                    shadow=AxisFamilySite(
                        file_path=shadow_spec.file_path,
                        line=shadow_spec.line,
                        family_name=shadow_spec.family_name,
                    ),
                    selector_method_name=shadow_spec.selector_method_name,
                    shared_case_names=shared_case_names,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.key_type_name,
                item.authoritative.file_path,
                item.shadow.file_path,
            ),
        )
    )


def _enum_member_refs_for_known_key_types(
    node: ast.AST,
    *,
    key_type_names: frozenset[str],
) -> dict[str, tuple[str, ...]]:
    refs: dict[str, set[str]] = defaultdict(set)
    for subnode in _walk_nodes(node):
        parts = _ast_attribute_chain(subnode)
        if parts is None or len(parts) < 2:
            continue
        key_type_name = parts[-2]
        if key_type_name not in key_type_names:
            continue
        refs[key_type_name].add(f"{key_type_name}.{parts[-1]}")
    return {
        key_type_name: tuple(sorted(case_names))
        for key_type_name, case_names in refs.items()
    }


def _residual_closed_axis_branching_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[ResidualClosedAxisBranchingCandidate, ...]:
    authoritative_specs_by_key: dict[str, list[_KeyedFamilyAxisSpec]] = defaultdict(list)
    for spec in _keyed_family_axis_specs(modules):
        authoritative_specs_by_key[spec.key_type_name].append(spec)
    if not authoritative_specs_by_key:
        return ()
    key_type_names = frozenset(authoritative_specs_by_key)
    candidates: list[ResidualClosedAxisBranchingCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for module in modules:
        file_path = str(module.path)
        if "/tests/" in file_path:
            continue
        for qualname, function in _iter_named_functions(module):
            branch_site_count: Counter[str] = Counter()
            case_names_by_key: dict[str, set[str]] = defaultdict(set)
            for subnode in _non_nested_subnodes(function.body):
                if isinstance(subnode, ast.If):
                    refs = _enum_member_refs_for_known_key_types(
                        subnode.test, key_type_names=key_type_names
                    )
                    for key_type_name, case_names in refs.items():
                        branch_site_count[key_type_name] += 1
                        case_names_by_key[key_type_name].update(case_names)
                    continue
                if isinstance(subnode, ast.Match):
                    refs_by_key: dict[str, set[str]] = defaultdict(set)
                    for case in subnode.cases:
                        pattern_refs = _enum_member_refs_for_known_key_types(
                            case.pattern, key_type_names=key_type_names
                        )
                        for key_type_name, case_names in pattern_refs.items():
                            refs_by_key[key_type_name].update(case_names)
                        if case.guard is not None:
                            guard_refs = _enum_member_refs_for_known_key_types(
                                case.guard, key_type_names=key_type_names
                            )
                            for key_type_name, case_names in guard_refs.items():
                                refs_by_key[key_type_name].update(case_names)
                    for key_type_name, case_names in refs_by_key.items():
                        branch_site_count[key_type_name] += 1
                        case_names_by_key[key_type_name].update(case_names)
            for key_type_name, branch_count in sorted(branch_site_count.items()):
                if branch_count <= 0:
                    continue
                specs = authoritative_specs_by_key.get(key_type_name, ())
                if not specs:
                    continue
                if any(spec.file_path == file_path for spec in specs):
                    continue
                authoritative_case_names = {
                    case_name
                    for spec in specs
                    for case_name in spec.case_names
                }
                shared_case_names = tuple(
                    sorted(case_names_by_key[key_type_name] & authoritative_case_names)
                )
                if not shared_case_names:
                    continue
                key = (file_path, qualname, key_type_name)
                if key in seen:
                    continue
                seen.add(key)
                authoritative_families = tuple(
                    sorted(
                        (
                            spec.family_name,
                            spec.file_path,
                            spec.line,
                        )
                        for spec in specs
                    )
                )
                candidates.append(
                    ResidualClosedAxisBranchingCandidate(
                        key_type_name=key_type_name,
                        file_path=file_path,
                        line=function.lineno,
                        qualname=qualname,
                        branch_site_count=branch_count,
                        case_names=shared_case_names,
                        authoritative_families=authoritative_families,
                    )
                )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.key_type_name, item.file_path, item.line, item.qualname),
        )
    )


def _parallel_registry_projection_family_candidates(
    module: ParsedModule,
) -> tuple[ParallelRegistryProjectionFamilyCandidate, ...]:
    candidates = _registered_catalog_projection_candidates(module)
    grouped: dict[
        tuple[str, str, tuple[str, ...]],
        list[RegisteredCatalogProjectionCandidate],
    ] = defaultdict(list)
    for candidate in candidates:
        grouped[
            (
                candidate.collector_name,
                candidate.registry_accessor_name,
                candidate.return_keyword_names,
            )
        ].append(candidate)
    return tuple(
        ParallelRegistryProjectionFamilyCandidate(
            file_path=str(module.path),
            collector_name=collector_name,
            registry_accessor_name=registry_accessor_name,
            return_keyword_names=return_keyword_names,
            functions=tuple(
                sorted(functions, key=lambda item: (item.line, item.qualname))
            ),
        )
        for (collector_name, registry_accessor_name, return_keyword_names), functions in sorted(
            grouped.items()
        )
        if len(functions) >= 2
        and len({item.catalog_type_name for item in functions}) >= 2
        and len({item.extractor_base_name for item in functions}) >= 2
    )


def _is_classmethod(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return any(
        _ast_terminal_name(decorator) == "classmethod"
        for decorator in node.decorator_list
    )


def _cls_registry_key_expr(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if not (
        isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "cls"
        and node.value.attr == "_registry"
    ):
        return None
    return ast.unparse(node.slice)


def _cls_registry_membership_test(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Compare):
        return None
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    comparator = node.comparators[0]
    if not (
        isinstance(comparator, ast.Attribute)
        and isinstance(comparator.value, ast.Name)
        and comparator.value.id == "cls"
        and comparator.attr == "_registry"
    ):
        return None
    operator = node.ops[0]
    if isinstance(operator, ast.In):
        return ("in", ast.unparse(node.left))
    if isinstance(operator, ast.NotIn):
        return ("not_in", ast.unparse(node.left))
    return None


def _raise_exception_type_name(node: ast.Raise) -> str | None:
    if node.exc is None:
        return None
    if isinstance(node.exc, ast.Call):
        return _call_name(node.exc.func)
    return _call_name(node.exc)


def _registry_lookup_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> RegistryLookupShape | None:
    body = _trim_docstring_body(list(method.body))
    if len(body) == 1 and isinstance(body[0], ast.Try):
        try_node = body[0]
        if (
            len(try_node.body) != 1
            or len(try_node.handlers) != 1
            or try_node.orelse
            or try_node.finalbody
        ):
            return None
        try_stmt = try_node.body[0]
        if not isinstance(try_stmt, ast.Return) or try_stmt.value is None:
            return None
        key_expr = _cls_registry_key_expr(try_stmt.value)
        if key_expr is None:
            return None
        handler = try_node.handlers[0]
        if _ast_terminal_name(handler.type) != "KeyError":
            return None
        raise_stmt = next(
            (stmt for stmt in handler.body if isinstance(stmt, ast.Raise)),
            None,
        )
        return RegistryLookupShape(
            key_expr=key_expr,
            error_type_name=None
            if raise_stmt is None
            else _raise_exception_type_name(raise_stmt),
            style="try_except",
        )
    if len(body) >= 2 and isinstance(body[0], ast.If):
        membership = _cls_registry_membership_test(body[0].test)
        if membership is None or membership[0] != "not_in":
            return None
        raise_stmt = next(
            (stmt for stmt in body[0].body if isinstance(stmt, ast.Raise)),
            None,
        )
        tail = body[-1]
        if not isinstance(tail, ast.Return) or tail.value is None:
            return None
        returned_key = _cls_registry_key_expr(tail.value)
        if returned_key != membership[1]:
            return None
        return RegistryLookupShape(
            key_expr=membership[1],
            error_type_name=None
            if raise_stmt is None
            else _raise_exception_type_name(raise_stmt),
            style="membership_guard",
        )
    return None


def _repeated_keyed_family_candidates(
    modules: Sequence[ParsedModule], config: DetectorConfig
) -> tuple[RepeatedKeyedFamilyCandidate, ...]:
    roots: list[KeyedFamilyRootCandidate] = []
    for module in modules:
        for node in (
            class_node
            for class_node in module.module.body
            if isinstance(class_node, ast.ClassDef)
        ):
            base_names = _declared_base_names(node)
            if "AutoRegisterByClassVar" not in base_names:
                continue
            assignments = _class_direct_assignments(node)
            registry_key_attr_name = _constant_string(
                assignments.get("registry_key_attr")
            )
            if registry_key_attr_name is None:
                continue
            if not _is_empty_dict_expr(assignments.get("_registry")):
                continue
            lookup_methods = [
                (method, shape)
                for method in _iter_class_methods(node)
                if _is_classmethod(method)
                and method.name.startswith("for_")
                and (shape := _registry_lookup_shape(method)) is not None
            ]
            if len(lookup_methods) != 1:
                continue
            lookup_method, lookup_shape = lookup_methods[0]
            roots.append(
                KeyedFamilyRootCandidate(
                    file_path=str(module.path),
                    line=node.lineno,
                    class_name=node.name,
                    family_base_name="AutoRegisterByClassVar",
                    registry_key_attr_name=registry_key_attr_name,
                    lookup_method_name=lookup_method.name,
                    lookup_style=lookup_shape.style,
                    error_type_name=lookup_shape.error_type_name,
                    abstract_hook_names=tuple(
                        method.name
                        for method in _iter_class_methods(node)
                        if _is_abstract_method(method)
                    ),
                )
            )
    min_roots = max(3, config.min_registration_sites)
    grouped: dict[tuple[str, str], list[KeyedFamilyRootCandidate]] = defaultdict(list)
    for root in roots:
        grouped[(root.family_base_name, root.lookup_style)].append(root)
    return tuple(
        RepeatedKeyedFamilyCandidate(
            family_base_name=family_base_name,
            lookup_style=lookup_style,
            roots=tuple(
                sorted(
                    items,
                    key=lambda item: (item.file_path, item.line, item.class_name),
                )
            ),
        )
        for (family_base_name, lookup_style), items in sorted(grouped.items())
        if len(items) >= min_roots
    )


def _manual_record_registration_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ManualRecordRegistrationShape | None:
    if not _is_classmethod(method):
        return None
    body = _trim_docstring_body(list(method.body))
    if len(body) < 2 or not isinstance(body[0], ast.If):
        return None
    membership = _cls_registry_membership_test(body[0].test)
    if membership is None or membership[0] != "in":
        return None
    key_expr = membership[1]
    assignment = next(
        (
            statement
            for statement in body[1:]
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and _cls_registry_key_expr(statement.targets[0]) == key_expr
        ),
        None,
    )
    if assignment is None or not isinstance(assignment.value, ast.Call):
        return None
    if _call_name(assignment.value.func) != "cls":
        return None
    constructor_field_names = tuple(
        keyword.arg
        for keyword in assignment.value.keywords
        if keyword.arg is not None
    )
    key_field_names = tuple(
        keyword.arg
        for keyword in assignment.value.keywords
        if keyword.arg is not None and ast.unparse(keyword.value) == key_expr
    )
    if len(key_field_names) != 1:
        return None
    return ManualRecordRegistrationShape(
        key_expr=key_expr,
        key_field_name=key_field_names[0],
        constructor_field_names=constructor_field_names,
    )


def _manual_keyed_record_table_group_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[ManualKeyedRecordTableGroupCandidate, ...]:
    classes: list[ManualKeyedRecordTableClassCandidate] = []
    for node in (
        class_node
        for class_node in module.module.body
        if isinstance(class_node, ast.ClassDef)
    ):
        if not _is_dataclass_class(node):
            continue
        if not _is_empty_dict_expr(_class_direct_assignments(node).get("_registry")):
            continue
        register_method = _class_method_named(node, "register")
        if register_method is None:
            continue
        registration_shape = _manual_record_registration_shape(register_method)
        if registration_shape is None:
            continue
        lookup_methods = [
            (method, shape)
            for method in _iter_class_methods(node)
            if _is_classmethod(method)
            and method.name.startswith("for_")
            and (shape := _registry_lookup_shape(method)) is not None
        ]
        if len(lookup_methods) != 1:
            continue
        lookup_method, lookup_shape = lookup_methods[0]
        classes.append(
            ManualKeyedRecordTableClassCandidate(
                file_path=str(module.path),
                line=node.lineno,
                class_name=node.name,
                register_method_name="register",
                lookup_method_name=lookup_method.name,
                lookup_style=lookup_shape.style,
                key_field_name=registration_shape.key_field_name,
                key_expr=registration_shape.key_expr,
                constructor_field_names=registration_shape.constructor_field_names,
            )
        )
    if len(classes) < config.min_registration_sites:
        return ()
    grouped: dict[tuple[str, str], list[ManualKeyedRecordTableClassCandidate]] = (
        defaultdict(list)
    )
    for candidate in classes:
        grouped[(candidate.register_method_name, candidate.lookup_style)].append(
            candidate
        )
    return tuple(
        ManualKeyedRecordTableGroupCandidate(
            file_path=str(module.path),
            classes=tuple(
                sorted(items, key=lambda item: (item.line, item.class_name))
            ),
        )
        for _, items in sorted(grouped.items())
        if len(items) >= config.min_registration_sites
    )


def _returns_tuple_of_self_attributes(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    returned = _single_return_case(method.body)
    if returned is None:
        return False
    return_value, _ = returned
    return isinstance(return_value, ast.Tuple) and all(
        isinstance(item, ast.Attribute)
        and isinstance(item.value, ast.Name)
        and item.value.id == "self"
        for item in return_value.elts
    )


def _returns_constructor_call(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    accepted_names: tuple[str, ...],
) -> bool:
    returned = _single_return_case(method.body)
    if returned is None:
        return False
    return_value, _ = returned
    if not isinstance(return_value, ast.Call):
        return False
    call_name = _call_name(return_value.func)
    return call_name in accepted_names


def _validation_guard_count(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    count = 0
    for node in _walk_nodes(method):
        if isinstance(node, ast.Attribute) and node.attr in {"ndim", "shape"}:
            count += 1
        if isinstance(node, ast.Compare) and any(
            isinstance(operator, (ast.Lt, ast.LtE, ast.NotEq))
            for operator in node.ops
        ):
            count += 1
    return count


def _same_type_constructor_method_names(
    node: ast.ClassDef,
    *,
    include_classmethods: bool,
    include_instance_methods: bool,
) -> tuple[str, ...]:
    accepted_instance_names = (node.name,)
    accepted_class_names = ("cls", node.name)
    names: list[str] = []
    for method in _iter_class_methods(node):
        if _is_classmethod(method):
            if (
                include_classmethods
                and _returns_constructor_call(
                    method, accepted_names=accepted_class_names
                )
            ):
                names.append(method.name)
            continue
        if (
            include_instance_methods
            and _returns_constructor_call(method, accepted_names=accepted_instance_names)
        ):
            names.append(method.name)
    return tuple(sorted(set(names)))


def _shared_record_base_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        name
        for name in _declared_base_names(node)
        if name not in _IGNORED_ANCESTOR_NAMES
    )


def _shared_record_mechanics_method_names(
    candidates: Sequence["ManualStructuralRecordMechanicsClassCandidate"],
) -> tuple[str, ...]:
    shared_projection_method_names = set.intersection(
        *(set(candidate.projection_method_names) for candidate in candidates)
    )
    shared_roundtrip_method_names = set.intersection(
        *(set(candidate.roundtrip_method_names) for candidate in candidates)
    )
    return tuple(
        sorted(
            {"validate"}
            | shared_projection_method_names
            | shared_roundtrip_method_names
        )
    )


def _manual_structural_record_mechanics_group_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[ManualStructuralRecordMechanicsGroupCandidate, ...]:
    threshold = max(3, config.min_registration_sites)
    classes: list[ManualStructuralRecordMechanicsClassCandidate] = []
    for node in (
        class_node
        for class_node in module.module.body
        if isinstance(class_node, ast.ClassDef)
    ):
        if not _is_dataclass_class(node) or _is_abstract_class(node):
            continue
        base_names = _shared_record_base_names(node)
        if not base_names:
            continue
        validate_method = _class_method_named(node, "validate")
        if validate_method is None or _validation_guard_count(validate_method) < 3:
            continue
        projection_method_names = tuple(
            sorted(
                method.name
                for method in _iter_class_methods(node)
                if _returns_tuple_of_self_attributes(method)
            )
        )
        if not projection_method_names:
            continue
        roundtrip_method_names = _same_type_constructor_method_names(
            node,
            include_classmethods=True,
            include_instance_methods=False,
        )
        if not roundtrip_method_names:
            continue
        transform_method_names = tuple(
            method_name
            for method_name in _same_type_constructor_method_names(
                node,
                include_classmethods=False,
                include_instance_methods=True,
            )
            if method_name != "validate"
        )
        if not transform_method_names:
            continue
        classes.append(
            ManualStructuralRecordMechanicsClassCandidate(
                file_path=str(module.path),
                line=node.lineno,
                class_name=node.name,
                base_names=base_names,
                validation_method_name=validate_method.name,
                projection_method_names=projection_method_names,
                roundtrip_method_names=roundtrip_method_names,
                transform_method_names=transform_method_names,
            )
        )
    if len(classes) < threshold:
        return ()
    grouped: dict[
        tuple[str, ...], list[ManualStructuralRecordMechanicsClassCandidate]
    ] = defaultdict(list)
    for candidate in classes:
        grouped[candidate.base_names].append(candidate)
    return tuple(
        ManualStructuralRecordMechanicsGroupCandidate(
            file_path=str(module.path),
            base_names=base_names,
            classes=tuple(
                sorted(items, key=lambda item: (item.line, item.class_name))
            ),
        )
        for base_names, items in sorted(grouped.items())
        if len(items) >= threshold
        if set.intersection(*(set(item.projection_method_names) for item in items))
        if set.intersection(*(set(item.roundtrip_method_names) for item in items))
    )


def _simple_param_alias_from_attr(
    statement: ast.stmt,
    *,
    param_name: str,
) -> tuple[str, str] | None:
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
        or not isinstance(statement.value, ast.Attribute)
        or not isinstance(statement.value.value, ast.Name)
        or statement.value.value.id != param_name
    ):
        return None
    return (statement.targets[0].id, statement.value.attr)


def _simple_name_or_attr_expression(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _simple_name_or_attr_expression(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def _top_level_attribute_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for statement in _trim_docstring_body(list(function.body)):
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            continue
        value_expression = _simple_name_or_attr_expression(statement.value)
        if value_expression is None or "." not in value_expression:
            continue
        aliases[statement.targets[0].id] = value_expression
    return aliases


def _attribute_family_subject_expression(
    node: ast.AST,
    *,
    alias_sources: dict[str, str],
) -> str | None:
    if isinstance(node, ast.Name):
        aliased = alias_sources.get(node.id)
        if aliased is None or "." not in aliased:
            return None
        return aliased
    subject_expression = _simple_name_or_attr_expression(node)
    if subject_expression is None or "." not in subject_expression:
        return None
    return subject_expression


def _flatten_union_member_type_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return (
            _flatten_union_member_type_names(node.left)
            + _flatten_union_member_type_names(node.right)
        )
    type_name = _ast_terminal_name(node)
    if type_name in {None, "None", "NoneType"}:
        return ()
    return (type_name,)


def _module_union_type_aliases(
    module: ParsedModule,
) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for statement in module.module.body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            continue
        member_names = tuple(
            sorted(set(_flatten_union_member_type_names(statement.value)))
        )
        if len(member_names) < 2:
            continue
        aliases[statement.targets[0].id] = member_names
    return aliases


def _indexed_class_for_simple_name(
    module: ParsedModule,
    class_index: ClassFamilyIndex,
    class_name: str,
) -> IndexedClass | None:
    module_local_symbol = f"{module.module_name}.{class_name}"
    indexed_class = class_index.class_for(module_local_symbol)
    if indexed_class is not None:
        return indexed_class
    symbols = class_index.symbols_by_simple_name.get(class_name, ())
    if len(symbols) != 1:
        return None
    return class_index.class_for(symbols[0])


def _resolved_isinstance_type_names(
    node: ast.AST,
    *,
    module: ParsedModule,
    class_index: ClassFamilyIndex,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(node, ast.Tuple):
        items = node.elts
    else:
        items = (node,)
    concrete_names: list[str] = []
    abstract_names: list[str] = []
    for item in items:
        type_name = _ast_terminal_name(item)
        if type_name in {None, "None", "NoneType"}:
            continue
        indexed_class = _indexed_class_for_simple_name(
            module, class_index, type_name
        )
        if indexed_class is None:
            continue
        display_name = _indexed_class_display_name(indexed_class, class_index)
        if _is_abstract_class(indexed_class.node):
            abstract_names.append(display_name)
        else:
            concrete_names.append(display_name)
    return (
        tuple(sorted(set(concrete_names))),
        tuple(sorted(set(abstract_names))),
    )


def _indexed_ancestor_symbols(
    class_index: ClassFamilyIndex,
    symbol: str,
) -> tuple[str, ...]:
    ancestors: list[str] = []
    seen: set[str] = set()
    queue = list(
        class_index.class_for(symbol).resolved_base_symbols
        if class_index.class_for(symbol) is not None
        else ()
    )
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ancestors.append(current)
        indexed_class = class_index.class_for(current)
        if indexed_class is None:
            continue
        queue.extend(indexed_class.resolved_base_symbols)
    return tuple(ancestors)


def _common_abstract_base_names(
    module: ParsedModule,
    class_index: ClassFamilyIndex,
    class_names: tuple[str, ...],
) -> tuple[str, ...]:
    indexed_classes = tuple(
        indexed_class
        for class_name in class_names
        if (
            indexed_class := _indexed_class_for_simple_name(
                module, class_index, class_name
            )
        )
        is not None
    )
    if len(indexed_classes) < 2:
        return ()
    common_symbols = set(_indexed_ancestor_symbols(class_index, indexed_classes[0].symbol))
    for indexed_class in indexed_classes[1:]:
        common_symbols &= set(_indexed_ancestor_symbols(class_index, indexed_class.symbol))
    abstract_bases = tuple(
        sorted(
            (
                indexed_class
                for symbol in common_symbols
                if (indexed_class := class_index.class_for(symbol)) is not None
                and _is_abstract_class(indexed_class.node)
            ),
            key=lambda item: item.symbol,
        )
    )
    return _indexed_class_display_names(abstract_bases, class_index)


def _concrete_type_case_function_candidates(
    module: ParsedModule,
    *,
    class_index: ClassFamilyIndex,
) -> tuple[ConcreteTypeCaseFunctionCandidate, ...]:
    union_aliases = _module_union_type_aliases(module)
    candidates: list[ConcreteTypeCaseFunctionCandidate] = []
    for qualname, function in _iter_named_functions(module):
        alias_sources = _top_level_attribute_aliases(function)
        grouped_checks: dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]] = (
            defaultdict(list)
        )
        for subnode in _walk_nodes(function):
            if not (
                isinstance(subnode, ast.Call)
                and len(subnode.args) == 2
                and not subnode.keywords
                and _ast_terminal_name(subnode.func) == "isinstance"
            ):
                continue
            subject_expression = _attribute_family_subject_expression(
                subnode.args[0],
                alias_sources=alias_sources,
            )
            if subject_expression is None:
                continue
            concrete_names, abstract_names = _resolved_isinstance_type_names(
                subnode.args[1],
                module=module,
                class_index=class_index,
            )
            if not concrete_names:
                continue
            grouped_checks[subject_expression].append((concrete_names, abstract_names))
        for subject_expression, checks in sorted(grouped_checks.items()):
            concrete_class_names = tuple(
                sorted({name for concrete_names, _ in checks for name in concrete_names})
            )
            if len(concrete_class_names) < 2:
                continue
            subject_role = subject_expression.rsplit(".", 1)[-1]
            union_alias_names = tuple(
                sorted(
                    alias_name
                    for alias_name, member_names in union_aliases.items()
                    if set(concrete_class_names) <= set(member_names)
                )
            )
            candidates.append(
                ConcreteTypeCaseFunctionCandidate(
                    file_path=str(module.path),
                    line=function.lineno,
                    function_name=qualname,
                    subject_expression=subject_expression,
                    subject_role=subject_role,
                    concrete_class_names=concrete_class_names,
                    abstract_class_names=tuple(
                        sorted(
                            {
                                name
                                for _, abstract_names in checks
                                for name in abstract_names
                            }
                        )
                    ),
                    union_alias_names=union_alias_names,
                    case_site_count=len(checks),
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.subject_role, item.line),
        )
    )


def _repeated_concrete_type_case_analysis_candidates(
    modules: list[ParsedModule],
    config: DetectorConfig,
) -> tuple[RepeatedConcreteTypeCaseAnalysisCandidate, ...]:
    class_index = build_class_family_index(modules)
    min_function_count = max(3, config.min_registration_sites)
    min_class_count = max(2, config.min_reflective_selector_values)
    candidates: list[RepeatedConcreteTypeCaseAnalysisCandidate] = []
    for module in modules:
        grouped: dict[str, list[ConcreteTypeCaseFunctionCandidate]] = defaultdict(list)
        for function_candidate in _concrete_type_case_function_candidates(
            module, class_index=class_index
        ):
            grouped[function_candidate.subject_role].append(function_candidate)
        for subject_role, functions in sorted(grouped.items()):
            if len(functions) < min_function_count:
                continue
            concrete_class_names = tuple(
                sorted(
                    {
                        class_name
                        for function in functions
                        for class_name in function.concrete_class_names
                    }
                )
            )
            if len(concrete_class_names) < min_class_count:
                continue
            abstract_base_names = _common_abstract_base_names(
                module,
                class_index,
                concrete_class_names,
            )
            union_alias_names = tuple(
                sorted(
                    {
                        alias_name
                        for function in functions
                        for alias_name in function.union_alias_names
                    }
                )
            )
            shared_suffix = _longest_common_suffix(concrete_class_names)
            shared_prefix = _longest_common_prefix(concrete_class_names)
            if (
                not abstract_base_names
                and not union_alias_names
                and max(len(shared_suffix), len(shared_prefix)) < 6
            ):
                continue
            candidates.append(
                RepeatedConcreteTypeCaseAnalysisCandidate(
                    file_path=str(module.path),
                    functions=tuple(
                        sorted(
                            functions,
                            key=lambda item: (item.line, item.function_name),
                        )
                    ),
                    abstract_base_names=abstract_base_names,
                )
            )
    return tuple(candidates)


def _self_cast_type_name(node: ast.AST) -> str | None:
    if not (
        isinstance(node, ast.Call)
        and _ast_terminal_name(node.func) == "cast"
        and len(node.args) == 2
        and not node.keywords
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "self"
    ):
        return None
    type_name = ast.unparse(node.args[0])
    if not type_name:
        return None
    return type_name


def _self_cast_alias_names(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    aliases: set[str] = set()
    cast_type_names: set[str] = set()
    for statement in _walk_nodes(method):
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            continue
        cast_type_name = _self_cast_type_name(statement.value)
        if cast_type_name is None:
            continue
        aliases.add(statement.targets[0].id)
        cast_type_names.add(cast_type_name)
    return (tuple(sorted(aliases)), tuple(sorted(cast_type_names)))


def _implicit_self_contract_mixin_candidates(
    modules: list[ParsedModule],
    config: DetectorConfig,
) -> tuple[ImplicitSelfContractMixinCandidate, ...]:
    class_index = build_class_family_index(modules)
    min_consumer_count = max(2, config.min_registration_sites)
    candidates: list[ImplicitSelfContractMixinCandidate] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        if not indexed_class.simple_name.endswith("Mixin"):
            continue
        if _is_abstract_class(indexed_class.node):
            continue
        consumer_classes = tuple(
            descendant
            for descendant in _indexed_descendant_classes(
                class_index, indexed_class.symbol
            )
            if not _is_abstract_class(descendant.node)
        )
        if len(consumer_classes) < min_consumer_count:
            continue
        method_names: list[str] = []
        method_lines: list[int] = []
        cast_type_names: set[str] = set()
        accessed_attr_names: set[str] = set()
        for method in _iter_class_methods(indexed_class.node):
            if _is_abstract_method(method):
                continue
            alias_names, method_cast_type_names = _self_cast_alias_names(method)
            if not alias_names:
                continue
            method_names.append(method.name)
            method_lines.append(method.lineno)
            cast_type_names.update(method_cast_type_names)
            accessed_attr_names.update(
                _attribute_names_for_roots(method, root_names=set(alias_names))
            )
        if not method_names:
            continue
        candidates.append(
            ImplicitSelfContractMixinCandidate(
                file_path=indexed_class.file_path,
                line=indexed_class.line,
                mixin_name=_indexed_class_display_name(indexed_class, class_index),
                method_names=tuple(method_names),
                method_lines=tuple(method_lines),
                cast_type_names=tuple(sorted(cast_type_names)),
                consumer_class_names=_indexed_class_display_names(
                    consumer_classes,
                    class_index,
                ),
                consumer_lines=tuple(
                    consumer_class.line for consumer_class in consumer_classes
                ),
                accessed_attribute_names=tuple(sorted(accessed_attr_names)),
            )
        )
    return tuple(candidates)


def _returns_false_only(statements: Sequence[ast.stmt]) -> bool:
    returned = _single_return_case(statements)
    if returned is None:
        return False
    return_value, _ = returned
    return isinstance(return_value, ast.Constant) and return_value.value is False


def _contains_nonfalse_return(node: ast.AST) -> bool:
    for subnode in _walk_nodes(node):
        if not isinstance(subnode, ast.Return) or subnode.value is None:
            continue
        if isinstance(subnode.value, ast.Constant) and subnode.value.value is False:
            continue
        return True
    return False


def _attribute_names_for_roots(
    node: ast.AST,
    *,
    root_names: set[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                subnode.attr
                for subnode in _walk_nodes(node)
                if isinstance(subnode, ast.Attribute)
                and isinstance(subnode.value, ast.Name)
                and subnode.value.id in root_names
            }
        )
    )


def _guard_validator_function_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    min_guard_count: int,
) -> GuardValidatorFunctionCandidate | None:
    if "." in qualname:
        return None
    parameter_names = _parameter_names(function)
    if len(parameter_names) != 1:
        return None
    subject_param_name = parameter_names[0]
    body = _trim_docstring_body(list(function.body))
    if len(body) < min_guard_count + 1:
        return None
    alias_name: str | None = None
    alias_source_attr: str | None = None
    if body:
        alias = _simple_param_alias_from_attr(body[0], param_name=subject_param_name)
        if alias is not None:
            alias_name, alias_source_attr = alias
            body = body[1:]
    guard_count = sum(
        1
        for statement in body
        if isinstance(statement, ast.If)
        and not statement.orelse
        and _returns_false_only(statement.body)
    )
    if guard_count < min_guard_count:
        return None
    if not any(_contains_nonfalse_return(statement) for statement in body):
        return None
    root_names = {subject_param_name}
    if alias_name is not None:
        root_names.add(alias_name)
    accessed_attr_names = _attribute_names_for_roots(function, root_names=root_names)
    if len(accessed_attr_names) < min_guard_count:
        return None
    helper_call_names = tuple(
        sorted(
            {
                call_name
                for subnode in _walk_nodes(function)
                if isinstance(subnode, ast.Call)
                for call_name in (_call_name(subnode.func),)
                if call_name is not None
            }
        )
    )
    return GuardValidatorFunctionCandidate(
        file_path=str(module.path),
        line=function.lineno,
        function_name=qualname,
        subject_param_name=subject_param_name,
        alias_source_attr=alias_source_attr,
        guard_count=guard_count,
        accessed_attr_names=accessed_attr_names,
        helper_call_names=helper_call_names,
    )


def _repeated_guard_validator_family_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[RepeatedGuardValidatorFamilyCandidate, ...]:
    min_guard_count = max(3, config.min_duplicate_statements)
    min_family_size = max(3, config.min_registration_sites)
    functions = [
        candidate
        for qualname, function in _iter_named_functions(module)
        if (
            candidate := _guard_validator_function_candidate(
                module,
                qualname,
                function,
                min_guard_count=min_guard_count,
            )
        )
        is not None
    ]
    grouped: dict[tuple[str, str | None], list[GuardValidatorFunctionCandidate]] = (
        defaultdict(list)
    )
    for candidate in functions:
        grouped[(candidate.subject_param_name, candidate.alias_source_attr)].append(
            candidate
        )
    families: list[RepeatedGuardValidatorFamilyCandidate] = []
    for (subject_param_name, alias_source_attr), items in sorted(grouped.items()):
        if len(items) < min_family_size:
            continue
        shared_attr_names = tuple(
            sorted(
                set.intersection(
                    *(set(item.accessed_attr_names) for item in items)
                )
            )
        )
        if len(shared_attr_names) < min_guard_count:
            continue
        shared_helper_call_names = tuple(
            sorted(
                set.intersection(
                    *(set(item.helper_call_names) for item in items)
                )
            )
        )
        ordered = tuple(sorted(items, key=lambda item: (item.line, item.function_name)))
        families.append(
            RepeatedGuardValidatorFamilyCandidate(
                file_path=str(module.path),
                subject_param_name=subject_param_name,
                alias_source_attr=alias_source_attr,
                functions=ordered,
                shared_attr_names=shared_attr_names,
                shared_helper_call_names=shared_helper_call_names,
            )
        )
    return tuple(families)


def _is_fail_loud_guard_raise(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Raise) or statement.exc is None:
        return False
    exc = statement.exc
    if isinstance(exc, ast.Call):
        error_name = _call_name(exc.func)
    elif isinstance(exc, ast.Name):
        error_name = exc.id
    else:
        return False
    return error_name in {"ValueError", "TypeError", "AssertionError"}


def _normalized_shape_guard_signature(test: ast.AST) -> str:
    mapping: dict[str, str] = {}

    class SelfAttrNormalizer(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                placeholder = mapping.setdefault(node.attr, f"_S{len(mapping)}")
                return ast.copy_location(ast.Name(id=placeholder, ctx=ast.Load()), node)
            return self.generic_visit(node)

    normalized_test = ast.parse(ast.unparse(test), mode="eval").body
    normalized_test = ast.copy_location(normalized_test, test)
    normalized_test = ast.fix_missing_locations(normalized_test)
    normalized = cast(ast.AST, SelfAttrNormalizer().visit(normalized_test))
    signature = ast.unparse(normalized)
    return re.sub(r"_S\\d+", "_S", signature)


def _is_shape_guard_signature(signature: str) -> bool:
    return any(token in signature for token in (".shape", ".ndim", "len("))


def _shape_guard_signatures(test: ast.AST) -> tuple[str, ...]:
    if isinstance(test, ast.BoolOp):
        return tuple(
            signature
            for value in test.values
            for signature in _shape_guard_signatures(value)
        )
    signature = _normalized_shape_guard_signature(test)
    if not _is_shape_guard_signature(signature):
        return ()
    return (signature,)


def _validate_shape_guard_method_candidate(
    module: ParsedModule,
    class_node: ast.ClassDef,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    min_guard_count: int,
) -> ValidateShapeGuardMethodCandidate | None:
    if method.name != "validate":
        return None
    if not method.args.args or method.args.args[0].arg != "self":
        return None
    body = _trim_docstring_body(list(method.body))
    guard_statements = tuple(
        statement
        for statement in body
        if isinstance(statement, ast.If)
        and not statement.orelse
        and statement.body
        and all(_is_fail_loud_guard_raise(item) for item in statement.body)
    )
    if len(guard_statements) < min_guard_count:
        return None
    shape_guard_signatures = tuple(
        sorted(
            signature
            for statement in guard_statements
            for signature in _shape_guard_signatures(statement.test)
        )
    )
    if len(set(shape_guard_signatures)) < min_guard_count:
        return None
    return ValidateShapeGuardMethodCandidate(
        file_path=str(module.path),
        line=method.lineno,
        class_name=class_node.name,
        method_name=method.name,
        guard_count=len(guard_statements),
        shape_guard_count=len(set(shape_guard_signatures)),
        shape_guard_signatures=shape_guard_signatures,
    )


def _shared_shape_guard_signature_count(
    left: ValidateShapeGuardMethodCandidate,
    right: ValidateShapeGuardMethodCandidate,
) -> int:
    return len(set(left.shape_guard_signatures) & set(right.shape_guard_signatures))


def _validate_shape_guard_method_candidates(
    modules: Sequence[ParsedModule],
    *,
    min_guard_count: int,
) -> tuple[ValidateShapeGuardMethodCandidate, ...]:
    return tuple(
        candidate
        for module in modules
        for class_node in _walk_nodes(module.module)
        if isinstance(class_node, ast.ClassDef)
        for statement in class_node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        for candidate in (
            _validate_shape_guard_method_candidate(
                module,
                class_node,
                statement,
                min_guard_count=min_guard_count,
            ),
        )
        if candidate is not None
    )


def _group_repeated_validate_shape_guard_candidates(
    method_candidates: Sequence[ValidateShapeGuardMethodCandidate],
    config: DetectorConfig,
) -> tuple[RepeatedValidateShapeGuardFamilyCandidate, ...]:
    min_guard_count = max(2, config.min_duplicate_statements - 1)
    min_family_size = max(2, config.min_registration_sites)
    min_shared_shape_guards = max(2, min_guard_count)
    if len(method_candidates) < min_family_size:
        return ()
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left_index, left in enumerate(method_candidates):
        for right_index in range(left_index + 1, len(method_candidates)):
            right = method_candidates[right_index]
            if (
                _shared_shape_guard_signature_count(left, right)
                < min_shared_shape_guards
            ):
                continue
            adjacency[left_index].add(right_index)
            adjacency[right_index].add(left_index)
    groups: list[RepeatedValidateShapeGuardFamilyCandidate] = []
    maximal_cliques: list[tuple[int, ...]] = []
    clique_keys: set[tuple[int, ...]] = set()
    vertices = set(adjacency)

    def bron_kerbosch(
        current: set[int], prospective: set[int], excluded: set[int]
    ) -> None:
        if not prospective and not excluded:
            if len(current) >= min_family_size:
                clique = tuple(sorted(current))
                if clique not in clique_keys:
                    clique_keys.add(clique)
                    maximal_cliques.append(clique)
            return
        for vertex in tuple(sorted(prospective)):
            neighbors = adjacency.get(vertex, set())
            bron_kerbosch(
                current | {vertex},
                prospective & neighbors,
                excluded & neighbors,
            )
            prospective.remove(vertex)
            excluded.add(vertex)

    bron_kerbosch(set(), set(vertices), set())
    for clique in maximal_cliques:
        ordered_methods = tuple(
            sorted(
                (method_candidates[item] for item in clique),
                key=lambda candidate: (
                    candidate.file_path,
                    candidate.line,
                    candidate.symbol,
                ),
            )
        )
        signature_support = Counter(
            signature
            for method in ordered_methods
            for signature in set(method.shape_guard_signatures)
        )
        shared_shape_guard_signatures = tuple(
            sorted(
                signature
                for signature, count in signature_support.items()
                if count >= 2
            )
        )
        if len(shared_shape_guard_signatures) < min_shared_shape_guards:
            continue
        groups.append(
            RepeatedValidateShapeGuardFamilyCandidate(
                file_path=ordered_methods[0].file_path,
                methods=ordered_methods,
                shared_shape_guard_signatures=shared_shape_guard_signatures,
            )
        )
    return tuple(
        sorted(
            groups,
            key=lambda candidate: (
                candidate.methods[0].file_path,
                candidate.methods[0].line,
                candidate.methods[0].symbol,
            ),
        )
    )


def _repeated_validate_shape_guard_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[RepeatedValidateShapeGuardFamilyCandidate, ...]:
    min_guard_count = max(2, config.min_duplicate_statements - 1)
    method_candidates = _validate_shape_guard_method_candidates(
        (module,),
        min_guard_count=min_guard_count,
    )
    return _group_repeated_validate_shape_guard_candidates(method_candidates, config)


def _repeated_validate_shape_guard_candidates_for_modules(
    modules: Sequence[ParsedModule],
    config: DetectorConfig,
) -> tuple[RepeatedValidateShapeGuardFamilyCandidate, ...]:
    min_guard_count = max(2, config.min_duplicate_statements - 1)
    method_candidates = _validate_shape_guard_method_candidates(
        modules,
        min_guard_count=min_guard_count,
    )
    return _group_repeated_validate_shape_guard_candidates(method_candidates, config)


def _nominal_strategy_scaffold(candidate: EnumStrategyDispatchCandidate) -> str:
    axis_tail = (
        candidate.dispatch_axis.split(".")[-1]
        .replace("_", " ")
        .title()
        .replace(" ", "")
    )
    axis_attr_name = candidate.dispatch_axis.split(".")[-1]
    root_name = f"{axis_tail}Runner"
    lines = [
        "from metaclass_registry import AutoRegisterMeta",
        "",
        f"class {root_name}(ABC, metaclass=AutoRegisterMeta):",
        f"    __registry_key__ = \"{axis_attr_name}\"",
        "    __skip_if_no_key__ = True",
        f"    {axis_attr_name} = None",
        "",
        "    @classmethod",
        f"    def for_{axis_attr_name}(cls, key):",
        "        return cls.__registry__[key]()",
        "",
        "    @abstractmethod",
        "    def run(self, ctx): ...",
        "",
    ]
    for case_name in candidate.case_names:
        case_tail = case_name.split(".")[-1].replace("_", " ").title().replace(" ", "")
        lines.extend(
            (
                f"class {case_tail}{root_name}({root_name}):",
                f"    {axis_attr_name} = {case_name}",
                "    ...",
                "",
            )
        )
    return "\n".join(lines)


def _nominal_strategy_patch(candidate: EnumStrategyDispatchCandidate) -> str:
    axis_tail = (
        candidate.dispatch_axis.split(".")[-1]
        .replace("_", " ")
        .title()
        .replace(" ", "")
    )
    axis_attr_name = candidate.dispatch_axis.split(".")[-1]
    root_name = f"{axis_tail}Runner"
    return (
        f"# Replace `{candidate.dispatch_axis}` branching with a metaclass-registry-backed nominal runner family\n"
        f"runner = {root_name}.for_{axis_attr_name}({candidate.dispatch_axis})\n"
        f"return runner.run(ctx)"
    )


def _self_attr_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        if target.value.id == "self":
            return target.attr
    return None


def _assigned_self_attrs(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    assigned: list[str] = []
    for subnode in _walk_nodes(node):
        if isinstance(subnode, ast.Assign):
            for target in subnode.targets:
                attr_name = _self_attr_name(target)
                if attr_name is not None:
                    assigned.append(attr_name)
        elif isinstance(subnode, ast.AnnAssign):
            attr_name = _self_attr_name(subnode.target)
            if attr_name is not None:
                assigned.append(attr_name)
    return tuple(dict.fromkeys(assigned))


def _assigned_self_attr_from_param(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    param_names = {
        item.arg for item in tuple(node.args.posonlyargs) + tuple(node.args.args)
    }
    assigned: dict[str, str] = {}
    for subnode in _walk_nodes(node):
        if not isinstance(subnode, ast.Assign):
            continue
        if len(subnode.targets) != 1:
            continue
        attr_name = _self_attr_name(subnode.targets[0])
        if attr_name is None:
            continue
        if isinstance(subnode.value, ast.Name) and subnode.value.id in param_names:
            assigned[attr_name] = subnode.value.id
    return assigned


def _string_dispatch_cases_from_body(
    body: list[ast.stmt],
    axis_expression: str,
) -> tuple[str, ...]:
    cases: list[str] = []
    if not body:
        return ()
    current = body[0]
    while isinstance(current, ast.If):
        dispatch_case = _comparison_dispatch_case(current.test)
        if dispatch_case is None:
            return ()
        current_axis, case_name = dispatch_case
        if current_axis != axis_expression:
            return ()
        if _constant_string(ast.parse(case_name, mode="eval").body) is None:
            return ()
        cases.append(case_name)
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
            continue
        break
    return tuple(cases)


_TAG_PARAM_NAMES = frozenset({"kind", "mode", _TYPE_NAME_LITERAL, "tag", "backend"})


def _manual_fiber_tag_candidates(
    module: ParsedModule,
) -> tuple[ManualFiberTagCandidate, ...]:
    candidates: list[ManualFiberTagCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {
            item.name: item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        init_method = methods.get("__init__")
        if init_method is None:
            continue
        assigned_from_param = _assigned_self_attr_from_param(init_method)
        tag_names = tuple(
            attr_name
            for attr_name, param_name in assigned_from_param.items()
            if param_name in _TAG_PARAM_NAMES
        )
        if not tag_names:
            continue
        assigned_field_names = _assigned_self_attrs(init_method)
        for method_name, method in methods.items():
            if method_name == "__init__":
                continue
            if not method.body:
                continue
            for tag_name in tag_names:
                case_names = _string_dispatch_cases_from_body(
                    method.body,
                    f"self.{tag_name}",
                )
                if len(case_names) < 2:
                    continue
                if len(assigned_field_names) <= len(case_names) + 1:
                    continue
                candidates.append(
                    ManualFiberTagCandidate(
                        file_path=str(module.path),
                        line=method.lineno,
                        subject_name=node.name,
                        name_family=case_names,
                        init_line=init_method.lineno,
                        method_name=method_name,
                        tag_name=tag_name,
                        assigned_field_names=assigned_field_names,
                    )
                )
    return tuple(candidates)


def _expr_mentions_self_attr(expr: ast.AST, attr_name: str) -> bool:
    for subnode in _walk_nodes(expr):
        if isinstance(subnode, ast.Attribute) and isinstance(subnode.value, ast.Name):
            if subnode.value.id == "self" and subnode.attr == attr_name:
                return True
        if isinstance(subnode, ast.Name) and subnode.id == attr_name:
            return True
    return False


def _descriptor_derived_view_candidates(
    module: ParsedModule,
) -> tuple[DescriptorDerivedViewCandidate, ...]:
    candidates: list[DescriptorDerivedViewCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = [
            item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        init_method = next((item for item in methods if item.name == "__init__"), None)
        if init_method is None:
            continue
        source_assignments = _assigned_self_attr_from_param(init_method)
        for source_attr in source_assignments:
            derived_field_names = []
            for subnode in _walk_nodes(init_method):
                if not isinstance(subnode, ast.Assign) or len(subnode.targets) != 1:
                    continue
                target_name = _self_attr_name(subnode.targets[0])
                if target_name is None or target_name == source_attr:
                    continue
                if _expr_mentions_self_attr(subnode.value, source_attr):
                    derived_field_names.append(target_name)
            derived_field_names = cast(
                tuple[str, ...],
                tuple(dict.fromkeys(derived_field_names)),
            )
            if len(derived_field_names) < 2:
                continue
            for method in methods:
                if method.name == "__init__":
                    continue
                updated_field_names = []
                rewrites_source = False
                for subnode in _walk_nodes(method):
                    if not isinstance(subnode, ast.Assign) or len(subnode.targets) != 1:
                        continue
                    target_name = _self_attr_name(subnode.targets[0])
                    if target_name is None:
                        continue
                    if target_name == source_attr:
                        rewrites_source = True
                    if target_name in derived_field_names:
                        updated_field_names.append(target_name)
                updated_field_names = cast(
                    tuple[str, ...],
                    tuple(dict.fromkeys(updated_field_names)),
                )
                if not rewrites_source:
                    continue
                if not updated_field_names or set(updated_field_names) >= set(
                    derived_field_names
                ):
                    continue
                candidate_derived_field_names: tuple[str, ...] = tuple(
                    derived_field_names
                )
                candidate_updated_field_names: tuple[str, ...] = tuple(
                    updated_field_names
                )
                candidates.append(
                    DescriptorDerivedViewCandidate(
                        file_path=str(module.path),
                        line=method.lineno,
                        subject_name=node.name,
                        name_family=candidate_derived_field_names,
                        source_attr=source_attr,
                        init_line=init_method.lineno,
                        mutator_name=method.name,
                        updated_field_names=candidate_updated_field_names,
                    )
                )
    return tuple(candidates)


def _is_empty_dict_expr(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Dict):
        return not node.keys and not node.values
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
    )


def _module_registry_names(module: ParsedModule) -> tuple[str, ...]:
    names: list[str] = []
    for node in module.module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and _is_empty_dict_expr(node.value):
                names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and _is_empty_dict_expr(node.value):
                names.append(node.target.id)
    return tuple(names)


def _manual_registry_candidates(
    module: ParsedModule,
) -> tuple[ManualRegistryCandidate, ...]:
    registry_names = set(_module_registry_names(module))
    if not registry_names:
        return ()
    candidates: list[ManualRegistryCandidate] = []
    module_classes = [
        node for node in module.module.body if isinstance(node, ast.ClassDef)
    ]
    handler_classes = tuple(
        node.name
        for node in module_classes
        if node.name.endswith("Handler")
        or any(
            isinstance(item, ast.FunctionDef) and item.name == "handle"
            for item in node.body
        )
    )
    for node in module.module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for subnode in node.body:
            if not isinstance(subnode, ast.FunctionDef):
                continue
            registry_name: str | None = None
            for inner_node in _walk_nodes(subnode):
                if isinstance(inner_node, ast.Assign):
                    for target in inner_node.targets:
                        if isinstance(target, ast.Subscript) and isinstance(
                            target.value, ast.Name
                        ):
                            if target.value.id in registry_names:
                                registry_name = target.value.id
                elif isinstance(inner_node, ast.Return) and isinstance(
                    inner_node.value, ast.Name
                ):
                    if (
                        inner_node.value.id == subnode.args.args[0].arg
                        if subnode.args.args
                        else False
                    ):
                        continue
            if registry_name is None:
                continue
            decorated_class_names = tuple(
                class_node.name
                for class_node in module_classes
                if any(
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == node.name
                    for decorator in class_node.decorator_list
                )
            )
            if len(decorated_class_names) < 2:
                continue
            unregistered_class_names = tuple(
                sorted(set(handler_classes) - set(decorated_class_names))
            )
            candidates.append(
                ManualRegistryCandidate(
                    file_path=str(module.path),
                    line=node.lineno,
                    subject_name=registry_name,
                    name_family=decorated_class_names,
                    decorator_name=node.name,
                    unregistered_class_names=unregistered_class_names,
                )
            )
    return tuple(candidates)


def _method_names(node: ast.ClassDef) -> frozenset[str]:
    return frozenset(
        item.name
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _shared_abstract_nominal_authority(
    classes: tuple[ast.ClassDef, ...],
    *,
    class_lookup: dict[str, ast.ClassDef],
) -> bool:
    def abstract_lineage_names(node: ast.ClassDef) -> set[str]:
        lineage: set[str] = set()
        seen: set[str] = set()
        stack = [node.name]
        while stack:
            current_name = stack.pop()
            if current_name in seen or current_name in _IGNORED_ANCESTOR_NAMES:
                continue
            seen.add(current_name)
            current_node = class_lookup.get(current_name)
            if current_node is None:
                continue
            if _is_abstract_class(current_node):
                lineage.add(current_name)
            stack.extend(
                base_name
                for base_name in _declared_base_names(current_node)
                if base_name not in seen
            )
        return lineage

    lineage_sets = [abstract_lineage_names(node) for node in classes]
    if not lineage_sets or any(not lineage for lineage in lineage_sets):
        return False
    return bool(set.intersection(*lineage_sets))


def _structural_confusability_candidates(
    module: ParsedModule,
) -> tuple[StructuralConfusabilityCandidate, ...]:
    class_nodes = [
        node for node in module.module.body if isinstance(node, ast.ClassDef)
    ]
    class_lookup = {node.name: node for node in class_nodes}
    candidates: list[StructuralConfusabilityCandidate] = []
    for qualname, function in _iter_named_functions(module):
        for parameter_name in _parameter_names(function):
            observed_method_names = tuple(
                sorted(
                    {
                        subnode.func.attr
                        for subnode in _walk_nodes(function)
                        if isinstance(subnode, ast.Call)
                        and isinstance(subnode.func, ast.Attribute)
                        and isinstance(subnode.func.value, ast.Name)
                        and subnode.func.value.id == parameter_name
                    }
                )
            )
            if len(observed_method_names) < 2:
                continue
            confusable_classes = tuple(
                node
                for node in class_nodes
                if set(observed_method_names) <= _method_names(node)
            )
            if len(confusable_classes) < 2:
                continue
            if _shared_abstract_nominal_authority(
                confusable_classes,
                class_lookup=class_lookup,
            ):
                continue
            candidates.append(
                StructuralConfusabilityCandidate(
                    file_path=str(module.path),
                    line=function.lineno,
                    subject_name=qualname,
                    name_family=tuple(node.name for node in confusable_classes),
                    parameter_name=parameter_name,
                    observed_method_names=observed_method_names,
                )
            )
    return tuple(candidates)


def _is_dataclass_decorator(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "dataclass"
    if isinstance(node, ast.Call):
        return _is_dataclass_decorator(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr == "dataclass"
    return False


def _is_frozen_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and _is_dataclass_decorator(decorator.func):
            for keyword in decorator.keywords:
                if keyword.arg == "frozen":
                    return isinstance(keyword.value, ast.Constant) and bool(
                        keyword.value.value
                    )
            return False
        if _is_dataclass_decorator(decorator):
            return False
    return False


def _annassign_field_names(node: ast.ClassDef) -> tuple[str, ...]:
    field_names: list[str] = []
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            field_names.append(statement.target.id)
    return tuple(field_names)


def _normalize_witness_field_roles(field_name: str) -> tuple[str, ...]:
    roles: list[str] = []
    if field_name == "file_path":
        roles.append("witness_file_path")
    if field_name in {"line", "init_line", "method_line", "mutator_line"}:
        roles.append("witness_line")
    if field_name in {"class_name", "function_name", "registry_name", _SUBJECT_NAME_FIELD}:
        roles.extend(("witness_subject", "witness_name_payload"))
    if field_name == _NAME_FAMILY_FIELD or field_name.endswith("_names"):
        roles.extend(("witness_name_family", "witness_name_payload"))
    return tuple(dict.fromkeys(roles))


def _normalize_semantic_field_roles(field_name: str) -> tuple[str, ...]:
    roles: list[str] = []
    if field_name == "file_path" or field_name.endswith("_path"):
        roles.append("source_path")
    if field_name in {"line", "lineno"} or field_name.endswith("_line"):
        roles.append("source_line")
    if field_name in {_SUBJECT_NAME_FIELD, "class_name", "function_name"}:
        roles.append(_SUBJECT_NAME_FIELD)
    if field_name in {"observed_name", "method_name", "builder_name", "export_name"}:
        roles.append("observed_name")
    if field_name == "name" or field_name == _SUBJECT_NAME_FIELD or field_name.endswith(
        "_name"
    ):
        roles.append("name_payload")
    if field_name == _NAME_FAMILY_FIELD or field_name.endswith("_names"):
        roles.append(_NAME_FAMILY_FIELD)
    if field_name in {"owner_symbol", "symbol"} or field_name.endswith("_symbol"):
        roles.append("owner_symbol")
    return tuple(dict.fromkeys(roles))


def _normalized_semantic_role_fields(
    field_names: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    role_to_fields: dict[str, set[str]] = defaultdict(set)
    for field_name in field_names:
        for role_name in _normalize_semantic_field_roles(field_name):
            role_to_fields[role_name].add(field_name)
    return tuple(
        (role_name, tuple(sorted(field_names)))
        for role_name, field_names in sorted(role_to_fields.items())
    )


_GENERIC_FAMILY_CLASS_TOKENS = frozenset(
    {
        "candidate",
        "data",
        "entry",
        "group",
        "item",
        "profile",
        "record",
        "result",
        "shape",
        "spec",
    }
)


def _carrier_family_tokens(class_name: str) -> tuple[str, ...]:
    tokens = tuple(
        token.lower()
        for token in re.findall(
            r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+",
            class_name,
        )
        if token.lower() not in _GENERIC_FAMILY_CLASS_TOKENS
    )
    if not tokens:
        return ()
    return (tokens[-1],)


def _normalized_witness_role_fields(
    field_names: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    role_to_fields: dict[str, set[str]] = defaultdict(set)
    for field_name in field_names:
        for role_name in _normalize_witness_field_roles(field_name):
            role_to_fields[role_name].add(field_name)
    return tuple(
        (role_name, tuple(sorted(field_names)))
        for role_name, field_names in sorted(role_to_fields.items())
    )


def _witness_carrier_class_candidates(
    module: ParsedModule,
) -> tuple[WitnessCarrierClassCandidate, ...]:
    candidates: list[WitnessCarrierClassCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not _is_frozen_dataclass(node):
            continue
        if _is_abstract_class(node):
            continue
        field_names = _annassign_field_names(node)
        normalized_role_fields = _normalized_semantic_role_fields(field_names)
        normalized_roles = tuple(role_name for role_name, _ in normalized_role_fields)
        family_tokens = _carrier_family_tokens(node.name)
        if not family_tokens:
            continue
        if len(normalized_roles) < 3:
            continue
        if {"source_path", "source_line"} - set(normalized_roles):
            continue
        if not {
            "name_payload",
            _NAME_FAMILY_FIELD,
            _SUBJECT_NAME_FIELD,
            "observed_name",
        } & set(normalized_roles):
            continue
        candidates.append(
            WitnessCarrierClassCandidate(
                file_path=str(module.path),
                line=node.lineno,
                subject_name=node.name,
                name_family=field_names,
                base_names=_shared_record_base_names(node),
                family_tokens=family_tokens,
                normalized_roles=normalized_roles,
                normalized_role_fields=normalized_role_fields,
            )
        )
    return tuple(candidates)


def _witness_carrier_family_candidates(
    module: ParsedModule,
) -> tuple[WitnessCarrierFamilyCandidate, ...]:
    classes = _witness_carrier_class_candidates(module)
    if len(classes) < 2:
        return ()
    grouped: dict[str, list[WitnessCarrierClassCandidate]] = defaultdict(list)
    for candidate in classes:
        for token in candidate.family_tokens:
            grouped[token].append(candidate)
    seen_class_names: set[tuple[str, ...]] = set()
    findings: list[WitnessCarrierFamilyCandidate] = []
    for token, items in sorted(grouped.items()):
        if len(items) < 3:
            continue
        ordered_items = tuple(sorted(items, key=lambda item: (item.line, item.class_name)))
        class_names = tuple(item.class_name for item in ordered_items)
        if class_names in seen_class_names:
            continue
        shared_role_names = cast(
            tuple[str, ...],
            tuple(
                sorted(
                    set.intersection(
                        *(set(candidate.normalized_roles) for candidate in ordered_items)
                    )
                )
            ),
        )
        if len(shared_role_names) < 3:
            continue
        if set.intersection(*(set(candidate.base_names) for candidate in ordered_items)):
            continue
        seen_class_names.add(class_names)
        findings.append(
            WitnessCarrierFamilyCandidate(
                file_path=str(module.path),
                class_names=class_names,
                line_numbers=tuple(candidate.line for candidate in ordered_items),
                shared_role_names=shared_role_names,
            )
        )
    return tuple(findings)


def _manual_fiber_tag_scaffold(candidate: ManualFiberTagCandidate) -> str:
    root_name = candidate.class_name
    first_case = _camel_case(candidate.case_names[0].strip("'\""))
    second_case = _camel_case(candidate.case_names[1].strip("'\""))
    return (
        f"class {root_name}(ABC):\n"
        f"    @abstractmethod\n    def {candidate.method_name}(self): ...\n\n"
        f"class {first_case}{root_name}({root_name}): ...\n"
        f"class {second_case}{root_name}({root_name}): ..."
    )


def _manual_fiber_tag_patch(candidate: ManualFiberTagCandidate) -> str:
    return (
        f"# Remove the manual fiber tag `{candidate.tag_name}` from `{candidate.class_name}`\n"
        f"# Split `{candidate.class_name}` into one ABC root plus one subclass per fiber case.\n"
        f"# Keep only case-relevant fields in each subclass constructor."
    )


def _descriptor_derived_view_scaffold(candidate: DescriptorDerivedViewCandidate) -> str:
    return (
        "class DerivedField:\n"
        "    def __init__(self, template):\n"
        "        self.template = template\n"
        "    def __set_name__(self, owner, name): ...\n"
        "    def __get__(self, obj, objtype=None): ..."
    )


def _descriptor_derived_view_patch(candidate: DescriptorDerivedViewCandidate) -> str:
    return (
        f"# Treat `{candidate.source_attr}` as the sole authoritative source.\n"
        f"# Replace stored derived fields {candidate.derived_field_names} with descriptor-backed views.\n"
        f"# Remove partial resynchronization from `{candidate.mutator_name}`."
    )


def _manual_registry_scaffold(candidate: ManualRegistryCandidate) -> str:
    return (
        "from metaclass_registry import AutoRegisterMeta\n\n"
        "class EventHandler(ABC, metaclass=AutoRegisterMeta):\n"
        "    __registry_key__ = \"event_type\"\n"
        "    __skip_if_no_key__ = True\n"
        "    event_type = None\n\n"
        "    @classmethod\n"
        "    def for_event_type(cls, event_type):\n"
        "        return cls.__registry__[event_type]"
    )


def _manual_registry_patch(candidate: ManualRegistryCandidate) -> str:
    return (
        f"# Replace decorator `{candidate.decorator_name}` and registry `{candidate.registry_name}`\n"
        "# with `from metaclass_registry import AutoRegisterMeta`, a declarative class key, and\n"
        "# `cls.__registry__` so class creation and registration are one event."
    )


_AXIS_POLICY_ROOT_NAME = "AxisPolicy"
_AXIS_POLICY_KEY_TYPE_NAME = "AxisEnum"
_AXIS_POLICY_KEY_ATTR_NAME = "axis_key"


def _metaclass_registry_keyed_family_scaffold(
    *,
    root_name: str,
    key_type_name: str,
    key_attr_name: str,
    method_defs: tuple[str, ...],
) -> str:
    lines = [
        "from abc import ABC, abstractmethod",
        "from metaclass_registry import AutoRegisterMeta",
        "from typing import ClassVar",
        "",
        f"class {root_name}(ABC, metaclass=AutoRegisterMeta):",
        f"    __registry_key__ = \"{key_attr_name}\"",
        "    __skip_if_no_key__ = True",
        f"    {key_attr_name}: ClassVar[{key_type_name} | None] = None",
        "",
        "    @classmethod",
        f"    def for_key(cls, key: {key_type_name}):",
        "        return cls.__registry__[key]()",
    ]
    for method_def in method_defs:
        lines.extend(
            (
                "",
                "    @abstractmethod",
                f"    def {method_def}: ...",
            )
        )
    return "\n".join(lines)


def _axis_policy_registry_scaffold(*method_defs: str) -> str:
    return _metaclass_registry_keyed_family_scaffold(
        root_name=_AXIS_POLICY_ROOT_NAME,
        key_type_name=_AXIS_POLICY_KEY_TYPE_NAME,
        key_attr_name=_AXIS_POLICY_KEY_ATTR_NAME,
        method_defs=method_defs,
    )


def _structural_confusability_scaffold(
    candidate: StructuralConfusabilityCandidate,
) -> str:
    root_name = f"{_camel_case(candidate.parameter_name)}Interface"
    method_block = "\n".join(
        f"    @abstractmethod\n    def {name}(self, *args, **kwargs): ..."
        for name in candidate.observed_method_names
    )
    return f"class {root_name}(ABC):\n{method_block}"


def _structural_confusability_patch(candidate: StructuralConfusabilityCandidate) -> str:
    return (
        f"# The consumer `{candidate.function_name}` only observes `{candidate.parameter_name}` through methods {candidate.observed_method_names}.\n"
        f"# Introduce an ABC witness for that view and type the consumer against it instead of duck-typed coincidence."
    )


def _witness_carrier_family_scaffold(
    candidate: WitnessCarrierFamilyCandidate,
) -> str:
    lines = [
        "@dataclass(frozen=True)",
        "class SemanticCarrier(ABC):",
        "    source_path: str",
        "    source_line: int",
        "    primary_name: str | None",
        "",
        "@dataclass(frozen=True)",
        f"class {candidate.class_names[0]}(SemanticCarrier): ...",
    ]
    return "\n".join(lines)


def _witness_carrier_family_patch(
    candidate: WitnessCarrierFamilyCandidate,
) -> str:
    return (
        f"# Introduce one nominal carrier root for {candidate.class_names}.\n"
        f"# Move shared semantic roles {candidate.shared_role_names} into the base class and keep only fiber-specific payload in each leaf carrier."
    )


_WITNESS_NAME_PAYLOAD_ROLE = "name_payload"
_WITNESS_NAME_FAMILY_ROLE = _NAME_FAMILY_FIELD
_WITNESS_LINE_ROLE = "source_line"
_WITNESS_PATH_ROLE = "source_path"
_WITNESS_MIXIN_ROLE_NAMES = (
    _WITNESS_NAME_PAYLOAD_ROLE,
    _WITNESS_NAME_FAMILY_ROLE,
    _WITNESS_LINE_ROLE,
    _WITNESS_PATH_ROLE,
)


@dataclass(frozen=True)
class WitnessMixinRoleSpec:
    mixin_name: str
    scaffold: str


_WITNESS_MIXIN_ROLE_SPECS = {
    _WITNESS_NAME_PAYLOAD_ROLE: WitnessMixinRoleSpec(
        mixin_name="PrimaryNameMixin",
        scaffold=(
            "class PrimaryNameMixin(ABC):\n"
            "    @property\n"
            "    @abstractmethod\n"
            "    def primary_name(self) -> str | None: ..."
        ),
    ),
    _WITNESS_NAME_FAMILY_ROLE: WitnessMixinRoleSpec(
        mixin_name="NameFamilyMixin",
        scaffold=(
            "class NameFamilyMixin(ABC):\n"
            "    @property\n"
            "    @abstractmethod\n"
            f"    def {_WITNESS_NAME_FAMILY_ROLE}(self) -> tuple[str, ...]: ...\n\n"
            "    @property\n"
            "    def primary_name(self) -> str | None:\n"
            f"        return self.{_WITNESS_NAME_FAMILY_ROLE}[0] if self.{_WITNESS_NAME_FAMILY_ROLE} else None"
        ),
    ),
    _WITNESS_LINE_ROLE: WitnessMixinRoleSpec(
        mixin_name="SourceLineMixin",
        scaffold=(
            "class SourceLineMixin(ABC):\n"
            "    @property\n"
            "    @abstractmethod\n"
            "    def source_line(self) -> int: ..."
        ),
    ),
    _WITNESS_PATH_ROLE: WitnessMixinRoleSpec(
        mixin_name="SourcePathMixin",
        scaffold=(
            "class SourcePathMixin(ABC):\n"
            "    @property\n"
            "    @abstractmethod\n"
            "    def source_path(self) -> str: ..."
        ),
    ),
}


def _witness_mixin_role_spec(role_name: str) -> WitnessMixinRoleSpec:
    try:
        return _WITNESS_MIXIN_ROLE_SPECS[role_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported semantic mixin role: {role_name}") from exc


def _witness_role_mixin_name(role_name: str) -> str:
    return _witness_mixin_role_spec(role_name).mixin_name


def _witness_role_mixin_scaffold(role_name: str) -> str:
    return _witness_mixin_role_spec(role_name).scaffold


def _witness_mixin_enforcement_scaffold(
    candidate: WitnessMixinEnforcementCandidate,
) -> str:
    role_names = tuple(role_name for role_name, _ in candidate.role_field_names)
    blocks = [_witness_role_mixin_scaffold(role_name) for role_name in role_names]
    mixin_names = ", ".join(
        _witness_role_mixin_name(role_name) for role_name in role_names
    )
    blocks.append(
        "\n".join(
            (
                "@dataclass(frozen=True)",
                f"class {candidate.class_names[0]}(SemanticCarrier, {mixin_names}): ...",
            )
        )
    )
    return "\n\n".join(blocks)


def _witness_mixin_enforcement_patch(
    candidate: WitnessMixinEnforcementCandidate,
) -> str:
    role_summary = "; ".join(
        f"{_witness_role_mixin_name(role_name)} <- {field_names}"
        for role_name, field_names in candidate.role_field_names
    )
    return (
        f"# Collapse renamed semantic role slices {role_summary} into reusable mixins.\n"
        "# Normalize the leaf carriers onto the shared semantic base plus those mixins.\n"
        "# Use multiple inheritance when one carrier needs several orthogonal witness roles."
    )


def _orchestration_stage_scaffold(profile: FunctionProfile) -> str:
    stage_context_name = (
        f"{profile.qualname.split('.')[-1].title().replace('_', '')}StageContext"
    )
    return (
        f"@dataclass(frozen=True)\n"
        f"class {stage_context_name}:\n"
        f"    ...\n\n"
        f"def prepare_{profile.qualname.split('.')[-1]}_stage(ctx: {stage_context_name}): ...\n"
        f"def execute_{profile.qualname.split('.')[-1]}_stage(ctx: {stage_context_name}): ...\n"
        f"def finalize_{profile.qualname.split('.')[-1]}_stage(ctx: {stage_context_name}): ..."
    )


def _orchestration_stage_patch(profile: FunctionProfile) -> str:
    function_name = profile.qualname.split(".")[-1]
    stage_context_name = f"{function_name.title().replace('_', '')}StageContext"
    return (
        f"# Extract a nominal stage context from `{function_name}`\n"
        f"ctx = {stage_context_name}(...)\n"
        f"prepared = prepare_{function_name}_stage(ctx)\n"
        f"executed = execute_{function_name}_stage(prepared)\n"
        f"return finalize_{function_name}_stage(executed)"
    )


def _authoritative_context_scaffold(
    candidate: ParameterThreadFamilyCandidate,
) -> str:
    shared_names = candidate.shared_parameter_names
    context_name = "SharedContext"
    lines = ["@dataclass(frozen=True)", f"class {context_name}:"]
    lines.extend(f"    {name}: object" for name in shared_names)
    if not shared_names:
        lines.append("    ...")
    lines.append("")
    lines.append(f"def helper(ctx: {context_name}, ...): ...")
    return "\n".join(lines)


def _authoritative_context_patch(
    candidate: ParameterThreadFamilyCandidate,
) -> str:
    shared_names = ", ".join(candidate.shared_parameter_names)
    return (
        f"# Collapse the shared parameter family into one nominal record\n"
        f"ctx = SharedContext({shared_names})\n"
        f"first_result = first_helper(ctx, ...)\n"
        f"second_result = second_helper(ctx, ...)"
    )


def _as_method_shape(shape: object) -> MethodShape:
    if not isinstance(shape, MethodShape):
        raise TypeError(f"Expected MethodShape, got {type(shape)!r}")
    return shape


def _as_builder_shape(shape: object) -> BuilderCallShape:
    if not isinstance(shape, BuilderCallShape):
        raise TypeError(f"Expected BuilderCallShape, got {type(shape)!r}")
    return shape


def _as_registration_shape(shape: object) -> RegistrationShape:
    if not isinstance(shape, RegistrationShape):
        raise TypeError(f"Expected RegistrationShape, got {type(shape)!r}")
    return shape


def _as_export_shape(shape: object) -> ExportDictShape:
    if not isinstance(shape, ExportDictShape):
        raise TypeError(f"Expected ExportDictShape, got {type(shape)!r}")
    return shape


def _as_projection_helper_shape(shape: object) -> ProjectionHelperShape:
    if not isinstance(shape, ProjectionHelperShape):
        raise TypeError(f"Expected ProjectionHelperShape, got {type(shape)!r}")
    return shape


def _as_accessor_wrapper_candidate(shape: object) -> AccessorWrapperCandidate:
    if not isinstance(shape, AccessorWrapperCandidate):
        raise TypeError(f"Expected AccessorWrapperCandidate, got {type(shape)!r}")
    return shape


def _carrier_identity(carrier: object) -> tuple[str, int, str]:
    if not isinstance(carrier, StructuralObservationCarrier):
        raise TypeError(f"Unsupported structural carrier: {type(carrier)!r}")
    return carrier.structural_observation.structural_identity


def _carrier_lookup(items: tuple[object, ...]) -> dict[tuple[str, int, str], object]:
    return {_carrier_identity(item): item for item in items}


def _materialize_observations(
    observations: tuple[StructuralObservation, ...],
    lookup: dict[tuple[str, int, str], object],
) -> tuple[object, ...]:
    return tuple(
        sorted(
            (
                lookup[item.structural_identity]
                for item in observations
                if item.structural_identity in lookup
            ),
            key=_carrier_identity,
        )
    )


def _fiber_grouped_shapes(
    modules: list[ParsedModule],
    shapes: tuple[object, ...],
    observation_kind: ObservationKind,
    execution_level: StructuralExecutionLevel,
) -> list[tuple[object, ...]]:
    del modules
    lookup = _carrier_lookup(shapes)
    groups: list[tuple[object, ...]] = []
    graph = ObservationGraph(
        tuple(
            shape.structural_observation
            for shape in shapes
            if isinstance(shape, (MethodShape, BuilderCallShape, ExportDictShape))
        )
    )
    for fiber in graph.fibers_for(observation_kind, execution_level):
        grouped_items = _materialize_observations(fiber.observations, lookup)
        if len(grouped_items) < 2:
            continue
        groups.append(grouped_items)
    return groups


@dataclass(frozen=True)
class SemanticDataclassRecommendation:
    class_name: str
    base_class_name: str
    matched_schema_name: str | None
    rationale: str
    scaffold: str
    certification: CertificationLevel

    @classmethod
    def existing_schema(
        cls,
        class_name: str,
        base_class_name: str,
        rationale: str,
        scaffold: str,
    ) -> "SemanticDataclassRecommendation":
        return cls(
            class_name,
            base_class_name,
            class_name,
            rationale,
            scaffold,
            CERTIFIED,
        )

    @classmethod
    def proposed_schema(
        cls,
        class_name: str,
        base_class_name: str,
        matched_schema_name: str | None,
        rationale: str,
        scaffold: str,
    ) -> "SemanticDataclassRecommendation":
        return cls(
            class_name,
            base_class_name,
            matched_schema_name,
            rationale,
            scaffold,
            STRONG_HEURISTIC,
        )


@dataclass(frozen=True)
class SemanticDictBagCandidate:
    line: int
    symbol: str
    key_names: tuple[str, ...]
    context_kind: str
    recommendation: SemanticDataclassRecommendation


@dataclass(frozen=True)
class FieldFamilyCandidate:
    class_names: tuple[str, ...]
    field_names: tuple[str, ...]
    execution_level: StructuralExecutionLevel
    observations: tuple[FieldObservation, ...]
    dataclass_count: int
    field_type_map: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LineWitnessCandidate(ABC):
    file_path: str
    line: int

    @property
    def witness_name(self) -> str:
        return type(self).__name__

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.witness_name)


@dataclass(frozen=True)
class EvidenceLocationsWitnessCandidate(LineWitnessCandidate):
    evidence_locations: tuple[SourceLocation, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return self.evidence_locations


@dataclass(frozen=True)
class ClassLineWitnessCandidate(LineWitnessCandidate):
    class_name: str

    @property
    def witness_name(self) -> str:
        return self.class_name


@dataclass(frozen=True)
class FunctionLineWitnessCandidate(LineWitnessCandidate):
    function_name: str

    @property
    def witness_name(self) -> str:
        return self.function_name


@dataclass(frozen=True)
class ClassMethodLineWitnessCandidate(LineWitnessCandidate):
    class_name: str
    method_name: str

    @property
    def symbol(self) -> str:
        return f"{self.class_name}.{self.method_name}"

    @property
    def witness_name(self) -> str:
        return self.symbol


@dataclass(frozen=True)
class PrefixedRoleFieldBundleCandidate(ClassLineWitnessCandidate):
    role_names: tuple[str, ...]
    shared_member_names: tuple[str, ...]
    role_field_map: tuple[tuple[str, tuple[str, ...]], ...]
    manual_transport_methods: tuple[str, ...]
    pytree_base_names: tuple[str, ...]
    is_dataclass_family: bool
    observations: tuple[FieldObservation, ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for _, field_names in self.role_field_map
            for field_name in field_names
        )

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            super().evidence,
            *tuple(
                SourceLocation(item.file_path, item.lineno, item.symbol)
                for item in self.observations[:7]
            ),
        )


@dataclass(frozen=True)
class NominalAuthorityShape:
    file_path: str
    class_name: str
    line: int
    declared_base_names: tuple[str, ...]
    ancestor_names: tuple[str, ...]
    field_names: tuple[str, ...]
    field_type_map: tuple[tuple[str, str], ...]
    method_names: tuple[str, ...]
    is_abstract: bool
    is_dataclass_family: bool


@dataclass(frozen=True)
class ManualFamilyRosterCandidate(LineWitnessCandidate):
    owner_name: str
    member_names: tuple[str, ...]
    family_base_name: str
    constructor_style: str


@dataclass(frozen=True)
class ManualConcreteSubclassRosterCandidate(ClassLineWitnessCandidate):
    registration_site: "_ManualSubclassRegistrationSite"
    consumer_locations: tuple[SourceLocation, ...]
    concrete_class_names: tuple[str, ...]

    @property
    def registry_name(self) -> str:
        return self.registration_site.registry_name

    @property
    def guard_summary(self) -> str | None:
        return self.registration_site.guard_summary

    @property
    def consumer_names(self) -> tuple[str, ...]:
        return tuple(location.symbol for location in self.consumer_locations)


@dataclass(frozen=True)
class PredicateSelectedConcreteFamilyCandidate(ClassLineWitnessCandidate):
    selector_method_name: str
    predicate_method_name: str
    context_param_name: str
    concrete_class_names: tuple[str, ...]


@dataclass(frozen=True)
class MirroredLeafFamilySide(LineWitnessCandidate):
    root_name: str
    leaf_evidence: tuple[SourceLocation, ...]

    @property
    def witness_name(self) -> str:
        return self.root_name


@dataclass(frozen=True)
class ParallelMirroredLeafFamilyCandidate:
    left: MirroredLeafFamilySide
    right: MirroredLeafFamilySide
    contract_method_names: tuple[str, ...]
    shared_leaf_family_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            self.left.evidence,
            self.right.evidence,
            *self.left.leaf_evidence[:2],
            *self.right.leaf_evidence[:2],
        )


@dataclass(frozen=True)
class FragmentedFamilyAuthorityCandidate:
    file_path: str
    mapping_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    key_family_name: str
    shared_keys: tuple[str, ...]
    total_keys: tuple[str, ...]


@dataclass(frozen=True)
class WitnessCarrierCandidate(LineWitnessCandidate):
    subject_name: str
    name_family: tuple[str, ...]

    @property
    def witness_name(self) -> str:
        return self.subject_name

    @property
    def class_name(self) -> str:
        return self.subject_name


class NameFamilyClassNamesMixin(ABC):
    name_family: tuple[str, ...]

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.name_family


class SubjectNameFunctionNameMixin(ABC):
    subject_name: str

    @property
    def function_name(self) -> str:
        return self.subject_name


@dataclass(frozen=True)
class ExistingNominalAuthorityReuseCandidate(WitnessCarrierCandidate):
    compatible_authority_file_path: str
    compatible_authority_name: str
    compatible_authority_line: int
    reuse_kind: str
    shared_role_names: tuple[str, ...]

    @property
    def shared_field_names(self) -> tuple[str, ...]:
        return self.name_family


@dataclass(frozen=True)
class PassThroughNominalWrapperCandidate(WitnessCarrierCandidate):
    delegate_field_name: str
    delegate_authority_file_path: str
    delegate_authority_name: str
    delegate_authority_line: int

    @property
    def forwarded_member_names(self) -> tuple[str, ...]:
        return self.name_family


@dataclass(frozen=True)
class FindingAssemblyPipelineCandidate(WitnessCarrierCandidate):
    method_name: str
    candidate_source_name: str
    metrics_type_name: str | None
    scaffold_helper_name: str | None
    patch_helper_name: str | None


@dataclass(frozen=True)
class GuardedDelegatorCandidate(WitnessCarrierCandidate):
    method_name: str
    guard_role: str
    delegate_name: str
    scope_role: str


@dataclass(frozen=True)
class StructuralObservationPropertyCandidate(WitnessCarrierCandidate):
    property_name: str
    constructor_name: str

    @property
    def keyword_names(self) -> tuple[str, ...]:
        return self.name_family


@dataclass(frozen=True)
class ClassLineNumbersGroup(ABC):
    file_path: str
    class_names: tuple[str, ...]
    line_numbers: tuple[int, ...]


@dataclass(frozen=True)
class PropertyHookGroup(ClassLineNumbersGroup):
    base_name: str
    property_name: str


@dataclass(frozen=True)
class PropertyAliasHookGroup(PropertyHookGroup):
    returned_attribute: str


@dataclass(frozen=True)
class ConstantPropertyHookGroup(PropertyHookGroup):
    return_expressions: tuple[str, ...]


@dataclass(frozen=True)
class HelperBackedObservationSpecCandidate(WitnessCarrierCandidate):
    base_names: tuple[str, ...]
    method_name: str
    helper_name: str
    wrapper_kind: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class HelperBackedObservationSpecGroup(ClassLineNumbersGroup):
    base_names: tuple[str, ...]
    method_names: tuple[str, ...]
    helper_names: tuple[str, ...]
    wrapper_kinds: tuple[str, ...]


@dataclass(frozen=True)
class GuardedWrapperSpecPair:
    file_path: str
    spec_name: str
    spec_line: int
    function_name: str
    function_line: int
    constructor_name: str
    node_types: tuple[str, ...]


@dataclass(frozen=True)
class DeclarativeFamilyLeafCandidate(WitnessCarrierCandidate):
    base_names: tuple[str, ...]
    assigned_names: tuple[str, ...]


@dataclass(frozen=True)
class DeclarativeFamilyBoilerplateGroup(ClassLineNumbersGroup):
    base_names: tuple[str, ...]
    assigned_names: tuple[str, ...]


@dataclass(frozen=True)
class TypeIndexedDefinitionBoilerplateGroup:
    file_path: str
    base_names: tuple[str, ...]
    definition_class_names: tuple[str, ...]
    alias_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    assigned_names: tuple[str, ...]


@dataclass(frozen=True)
class ExportSurfaceCandidate(LineWitnessCandidate):
    export_symbol: str
    exported_names: tuple[str, ...]


@dataclass(frozen=True)
class DerivedExportSurfaceCandidate(ExportSurfaceCandidate):
    derivable_root_names: tuple[str, ...]


@dataclass(frozen=True)
class ManualPublicApiSurfaceCandidate(ExportSurfaceCandidate):
    source_name_count: int


@dataclass(frozen=True)
class DerivedIndexedSurfaceCandidate(LineWitnessCandidate):
    surface_name: str
    key_kind: str
    value_names: tuple[str, ...]
    derivable_root_names: tuple[str, ...]


@dataclass(frozen=True)
class RegisteredUnionSurfaceCandidate(LineWitnessCandidate):
    owner_name: str
    accessor_name: str
    root_names: tuple[str, ...]


@dataclass(frozen=True)
class ExportPolicyPredicateCandidate(
    WitnessCarrierCandidate, SubjectNameFunctionNameMixin
):
    role_names: tuple[str, ...]
    root_type_names: tuple[str, ...]


@dataclass(frozen=True)
class RegistryTraversalGroup(ClassLineNumbersGroup):
    method_names: tuple[str, ...]
    materialization_kinds: tuple[str, ...]
    registry_attribute_names: tuple[str, ...]


@dataclass(frozen=True)
class AlternateConstructorFamilyGroup:
    file_path: str
    class_name: str
    method_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    keyword_names: tuple[str, ...]
    source_type_names: tuple[str, ...]


@dataclass(frozen=True)
class SelfReflectiveBuiltinCandidate(WitnessCarrierCandidate):
    method_name: str
    reflective_builtin: str


@dataclass(frozen=True)
class ReflectiveSelfAttributeCandidate(SelfReflectiveBuiltinCandidate):
    attribute_name: str


@dataclass(frozen=True)
class DynamicSelfFieldSelectionCandidate(SelfReflectiveBuiltinCandidate):
    selector_expression: str


@dataclass(frozen=True)
class StringBackedReflectiveNominalLookupCandidate(ClassLineWitnessCandidate):
    method_name: str
    selector_attr_name: str
    lookup_kind: str
    receiver_expression: str
    concrete_class_names: tuple[str, ...]
    selector_values: tuple[str, ...]


@dataclass(frozen=True)
class ConcreteConfigFieldProbeCandidate(ClassLineWitnessCandidate):
    method_name: str
    config_attr_name: str
    config_type_name: str
    missing_field_names: tuple[str, ...]
    probe_builtin_names: tuple[str, ...]


@dataclass(frozen=True)
class _ManualSubclassRegistrationSite:
    registry_name: str
    guard_summary: str | None
    selector_attr_name: str | None = None
    requires_concrete_subclass: bool = False


@dataclass(frozen=True)
class IndexedFamilyWrapperCandidate:
    function_name: str
    lineno: int
    collector_name: str
    spec_root_name: str
    item_type_name: str


@dataclass(frozen=True)
class FunctionProfile:
    file_path: str
    qualname: str
    lineno: int
    line_count: int
    branch_count: int
    call_count: int
    callee_names: tuple[str, ...]
    parameter_names: tuple[str, ...]

    @property
    def callee_family_count(self) -> int:
        return len(self.callee_names)

    @property
    def semantic_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.parameter_names
            if name not in _GENERIC_PARAMETER_NAMES and not name.startswith("_")
        )

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.lineno, self.qualname)


@dataclass(frozen=True)
class ParameterThreadFamilyCandidate:
    shared_parameter_names: tuple[str, ...]
    functions: tuple[FunctionProfile, ...]


@dataclass(frozen=True)
class PrivateTopLevelSymbolProfile:
    file_path: str
    module_name: str
    symbol: str
    kind: str
    line: int
    line_count: int
    name_tokens: tuple[str, ...]
    referenced_private_symbols: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.symbol)


@dataclass(frozen=True)
class PrivateCohortShouldBeModuleCandidate:
    file_path: str
    module_name: str
    module_line_count: int
    total_cohort_lines: int
    shared_tokens: tuple[str, ...]
    reference_edge_count: int
    lexical_edge_count: int
    symbols: tuple[PrivateTopLevelSymbolProfile, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(symbol.evidence for symbol in self.symbols[:6])


@dataclass(frozen=True)
class EnumStrategyDispatchCandidate:
    file_path: str
    qualname: str
    lineno: int
    dispatch_axis: str
    case_names: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.lineno, self.qualname)


@dataclass(frozen=True)
class RepeatedEnumStrategyDispatchCandidate:
    file_path: str
    enum_family: str
    shared_case_names: tuple[str, ...]
    functions: tuple[EnumStrategyDispatchCandidate, ...]


@dataclass(frozen=True)
class SplitDispatchAuthorityCandidate(LineWitnessCandidate):
    qualname: str
    strategy_root_name: str
    selector_method_name: str
    strategy_axis_expression: str
    strategy_case_names: tuple[str, ...]
    strategy_call_method_name: str
    generic_function_name: str
    generic_axis_expression: str
    generic_case_names: tuple[str, ...]
    bridge_callback_name: str
    selector_line: int
    generic_line: int

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.qualname)


@dataclass(frozen=True)
class ClosedConstantSelectorCandidate(EvidenceLocationsWitnessCandidate):
    qualname: str
    guard_expressions: tuple[str, ...]
    constant_names: tuple[str, ...]
    wrapper_name: str | None
    family_suffix: str | None
    common_constructor_name: str | None

    @property
    def witness_name(self) -> str:
        return self.qualname


@dataclass(frozen=True)
class DerivedWrapperSpecShadowCandidate(EvidenceLocationsWitnessCandidate):
    derived_family_name: str
    derived_constructor_name: str
    primary_family_name: str | None
    primary_constructor_name: str
    link_field_name: str
    primary_constant_names: tuple[str, ...]
    extra_field_names: tuple[str, ...]
    builder_names: tuple[str, ...]

    @property
    def witness_name(self) -> str:
        return self.derived_family_name


@dataclass(frozen=True)
class ModuleKeyedSelectionHelperCandidate(EvidenceLocationsWitnessCandidate):
    rule_class_name: str
    selected_field_name: str
    helper_function_name: str
    lookup_function_name: str
    rule_table_names: tuple[str, ...]
    index_table_names: tuple[str, ...]

    @property
    def witness_name(self) -> str:
        return self.rule_class_name


@dataclass(frozen=True)
class AxisFamilySite(LineWitnessCandidate):
    family_name: str

    @property
    def witness_name(self) -> str:
        return self.family_name


@dataclass(frozen=True)
class KeyedAxisFamilySite(AxisFamilySite):
    family_label: str | None


@dataclass(frozen=True)
class CrossModuleAxisShadowFamilyCandidate:
    key_type_name: str
    authoritative: AxisFamilySite
    shadow: AxisFamilySite
    selector_method_name: str
    shared_case_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (self.authoritative.evidence, self.shadow.evidence)


@dataclass(frozen=True)
class ResidualClosedAxisBranchingCandidate(LineWitnessCandidate):
    key_type_name: str
    qualname: str
    branch_site_count: int
    case_names: tuple[str, ...]
    authoritative_families: tuple[tuple[str, str, int], ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence = [SourceLocation(self.file_path, self.line, self.qualname)]
        evidence.extend(
            SourceLocation(file_path, line, family_name)
            for family_name, file_path, line in self.authoritative_families
        )
        return tuple(evidence[:6])


@dataclass(frozen=True)
class ParallelKeyedAxisFamilyCandidate:
    key_type_name: str
    left: KeyedAxisFamilySite
    right: KeyedAxisFamilySite
    shared_case_names: tuple[str, ...]
    case_overlap_ratio: float
    name_overlap_ratio: float

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (self.left.evidence, self.right.evidence)


@dataclass(frozen=True)
class ParallelKeyedTableAndFamilyCandidate:
    table: _KeyedTableAxisSpec
    family_name: str
    family_line: int
    shared_case_names: tuple[str, ...]

    @property
    def file_path(self) -> str:
        return self.table.file_path

    @property
    def key_type_name(self) -> str:
        return self.table.key_type_name

    @property
    def table_name(self) -> str:
        return self.table.table_name

    @property
    def table_line(self) -> int:
        return self.table.line

    @property
    def value_shape_name(self) -> str | None:
        return self.table.value_shape_name

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(self.file_path, self.table_line, self.table_name),
            SourceLocation(self.file_path, self.family_line, self.family_name),
        )


@dataclass(frozen=True)
class EnumKeyedTableClassAxisShadowCandidate(LineWitnessCandidate):
    table_name: str
    key_type_name: str
    key_attr_name: str
    class_sites: tuple[tuple[str, int], ...]
    shared_case_names: tuple[str, ...]
    value_type_names: tuple[str, ...]

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(class_name for class_name, _ in self.class_sites)

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence = [SourceLocation(self.file_path, self.line, self.table_name)]
        evidence.extend(
            SourceLocation(self.file_path, line, class_name)
            for class_name, line in self.class_sites
        )
        return tuple(evidence[:6])


@dataclass(frozen=True)
class TransportShellTemplateCandidate(ClassLineWitnessCandidate):
    driver_method_name: str
    selector_attr_name: str
    selector_value_names: tuple[str, ...]
    concrete_class_names: tuple[str, ...]
    source_param_name: str
    constructor_name: str
    kwargs_helper_name: str | None
    inner_hook_name: str
    outer_hook_name: str


@dataclass(frozen=True)
class SpecAxisFamily:
    file_path: str
    line: int
    family_name: str
    constructor_name: str
    axis_field_names: tuple[str, str]
    axis_pairs: tuple[tuple[str, str], ...]
    extra_keyword_names: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.family_name)


@dataclass(frozen=True)
class CrossModuleSpecAxisAuthorityCandidate:
    axis_field_names: tuple[str, str]
    shared_axis_pairs: tuple[tuple[str, str], ...]
    families: tuple[SpecAxisFamily, ...]


@dataclass(frozen=True)
class RegisteredCatalogProjectionCandidate(LineWitnessCandidate):
    qualname: str
    catalog_type_name: str
    collector_name: str
    structure_param_name: str
    extractor_base_name: str
    registry_accessor_name: str
    return_keyword_names: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.qualname)


@dataclass(frozen=True)
class ParallelRegistryProjectionFamilyCandidate:
    file_path: str
    collector_name: str
    registry_accessor_name: str
    return_keyword_names: tuple[str, ...]
    functions: tuple[RegisteredCatalogProjectionCandidate, ...]


@dataclass(frozen=True)
class RegistryLookupShape:
    key_expr: str
    error_type_name: str | None
    style: str


@dataclass(frozen=True)
class KeyedFamilyRootCandidate(ClassLineWitnessCandidate):
    family_base_name: str
    registry_key_attr_name: str
    lookup_method_name: str
    lookup_style: str
    error_type_name: str | None
    abstract_hook_names: tuple[str, ...]


@dataclass(frozen=True)
class RepeatedKeyedFamilyCandidate:
    family_base_name: str
    lookup_style: str
    roots: tuple[KeyedFamilyRootCandidate, ...]


@dataclass(frozen=True)
class ManualRecordRegistrationShape:
    key_expr: str
    key_field_name: str
    constructor_field_names: tuple[str, ...]


@dataclass(frozen=True)
class ManualKeyedRecordTableClassCandidate(ClassLineWitnessCandidate):
    register_method_name: str
    lookup_method_name: str
    lookup_style: str
    key_field_name: str
    key_expr: str
    constructor_field_names: tuple[str, ...]


@dataclass(frozen=True)
class ManualKeyedRecordTableGroupCandidate:
    file_path: str
    classes: tuple[ManualKeyedRecordTableClassCandidate, ...]


@dataclass(frozen=True)
class ManualStructuralRecordMechanicsClassCandidate(ClassLineWitnessCandidate):
    base_names: tuple[str, ...]
    validation_method_name: str
    projection_method_names: tuple[str, ...]
    roundtrip_method_names: tuple[str, ...]
    transform_method_names: tuple[str, ...]

    @property
    def method_names(self) -> tuple[str, ...]:
        return (
            self.validation_method_name,
            *self.projection_method_names,
            *self.roundtrip_method_names,
            *self.transform_method_names,
        )


@dataclass(frozen=True)
class ManualStructuralRecordMechanicsGroupCandidate:
    file_path: str
    base_names: tuple[str, ...]
    classes: tuple[ManualStructuralRecordMechanicsClassCandidate, ...]

    @property
    def shared_method_names(self) -> tuple[str, ...]:
        return _shared_record_mechanics_method_names(self.classes)

    @property
    def transform_method_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    method_name
                    for candidate in self.classes
                    for method_name in candidate.transform_method_names
                }
            )
        )


@dataclass(frozen=True)
class ConcreteTypeCaseFunctionCandidate(FunctionLineWitnessCandidate):
    subject_expression: str
    subject_role: str
    concrete_class_names: tuple[str, ...]
    abstract_class_names: tuple[str, ...]
    union_alias_names: tuple[str, ...]
    case_site_count: int


@dataclass(frozen=True)
class RepeatedConcreteTypeCaseAnalysisCandidate:
    file_path: str
    functions: tuple[ConcreteTypeCaseFunctionCandidate, ...]
    abstract_base_names: tuple[str, ...]

    @property
    def subject_role(self) -> str:
        return self.functions[0].subject_role

    @property
    def concrete_class_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    class_name
                    for function in self.functions
                    for class_name in function.concrete_class_names
                }
            )
        )

    @property
    def union_alias_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    alias_name
                    for function in self.functions
                    for alias_name in function.union_alias_names
                }
            )
        )

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(function.evidence for function in self.functions[:6])


@dataclass(frozen=True)
class GuardValidatorFunctionCandidate(FunctionLineWitnessCandidate):
    subject_param_name: str
    alias_source_attr: str | None
    guard_count: int
    accessed_attr_names: tuple[str, ...]
    helper_call_names: tuple[str, ...]


@dataclass(frozen=True)
class RepeatedGuardValidatorFamilyCandidate:
    file_path: str
    subject_param_name: str
    alias_source_attr: str | None
    functions: tuple[GuardValidatorFunctionCandidate, ...]
    shared_attr_names: tuple[str, ...]
    shared_helper_call_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(function.evidence for function in self.functions[:6])


@dataclass(frozen=True)
class ValidateShapeGuardMethodCandidate(ClassMethodLineWitnessCandidate):
    guard_count: int
    shape_guard_count: int
    shape_guard_signatures: tuple[str, ...]


@dataclass(frozen=True)
class RepeatedValidateShapeGuardFamilyCandidate:
    file_path: str
    methods: tuple[ValidateShapeGuardMethodCandidate, ...]
    shared_shape_guard_signatures: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(method.evidence for method in self.methods[:6])


@dataclass(frozen=True)
class ImplicitSelfContractMixinCandidate(LineWitnessCandidate):
    mixin_name: str
    method_names: tuple[str, ...]
    method_lines: tuple[int, ...]
    cast_type_names: tuple[str, ...]
    consumer_class_names: tuple[str, ...]
    consumer_lines: tuple[int, ...]
    accessed_attribute_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence = [
            SourceLocation(self.file_path, self.line, self.mixin_name),
            *(
                SourceLocation(self.file_path, line, f"{self.mixin_name}.{name}")
                for name, line in zip(self.method_names, self.method_lines, strict=True)
            ),
            *(
                SourceLocation(self.file_path, line, class_name)
                for class_name, line in zip(
                    self.consumer_class_names,
                    self.consumer_lines,
                    strict=True,
                )
            ),
        ]
        return tuple(evidence[:6])


@dataclass(frozen=True)
class EmptyLeafProductFamilyCandidate:
    file_path: str
    left_axis_base_names: tuple[str, ...]
    right_axis_base_names: tuple[str, ...]
    leaf_class_names: tuple[str, ...]
    leaf_lines: tuple[int, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(self.file_path, line, class_name)
            for class_name, line in zip(
                self.leaf_class_names,
                self.leaf_lines,
                strict=True,
            )
        )


@dataclass(frozen=True)
class FunctionWrapperCandidate:
    file_path: str
    qualname: str
    lineno: int
    delegate_symbol: str
    wrapper_kind: str
    statement_count: int
    projected_attributes: tuple[str, ...] = ()

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.lineno, self.qualname)


@dataclass(frozen=True)
class TrivialForwardingWrapperCandidate(LineWitnessCandidate):
    qualname: str
    delegate_symbol: str
    call_depth: int
    forwarded_parameter_names: tuple[str, ...]
    transported_value_sources: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.qualname)


@dataclass(frozen=True)
class ResolvedExternalCallsite:
    module_name: str
    location: SourceLocation


@dataclass(frozen=True)
class PublicApiPrivateDelegateSurface(ABC):
    module_name: str
    delegate_root_symbol: str
    delegate_root_line: int | None
    external_callsites: tuple[ResolvedExternalCallsite, ...]

    @property
    def external_module_names(self) -> tuple[str, ...]:
        return tuple(sorted({site.module_name for site in self.external_callsites}))


@dataclass(frozen=True)
class PublicApiPrivateDelegateShellCandidate(PublicApiPrivateDelegateSurface):
    wrapper: TrivialForwardingWrapperCandidate

    @property
    def wrapper_symbol(self) -> str:
        return f"{self.module_name}.{self.wrapper.qualname}"

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence = [self.wrapper.evidence]
        if self.delegate_root_line is not None:
            evidence.append(
                SourceLocation(
                    self.wrapper.file_path,
                    self.delegate_root_line,
                    self.delegate_root_symbol,
                )
            )
        evidence.extend(site.location for site in self.external_callsites[:4])
        return tuple(evidence[:6])


@dataclass(frozen=True)
class PublicApiPrivateDelegateFamilyCandidate(PublicApiPrivateDelegateSurface):
    file_path: str
    wrappers: tuple[TrivialForwardingWrapperCandidate, ...]

    @property
    def wrapper_names(self) -> tuple[str, ...]:
        return tuple(wrapper.qualname for wrapper in self.wrappers)

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence = [wrapper.evidence for wrapper in self.wrappers[:3]]
        if self.delegate_root_line is not None:
            evidence.append(
                SourceLocation(
                    self.file_path,
                    self.delegate_root_line,
                    self.delegate_root_symbol,
                )
            )
        evidence.extend(site.location for site in self.external_callsites[:2])
        return tuple(evidence[:6])


@dataclass(frozen=True)
class NominalPolicySurfaceMethodCandidate(LineWitnessCandidate):
    qualname: str
    owner_class_name: str
    method_name: str
    policy_root_symbol: str
    selector_method_name: str
    policy_member_name: str
    selector_source_exprs: tuple[str, ...]
    transported_value_sources: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.qualname)


@dataclass(frozen=True)
class NominalPolicySurfaceFamilyCandidate:
    methods: tuple[NominalPolicySurfaceMethodCandidate, ...]

    @property
    def file_path(self) -> str:
        return self.methods[0].file_path

    @property
    def owner_class_name(self) -> str:
        return self.methods[0].owner_class_name

    @property
    def policy_root_symbol(self) -> str:
        return self.methods[0].policy_root_symbol

    @property
    def selector_method_name(self) -> str:
        return self.methods[0].selector_method_name

    @property
    def selector_source_exprs(self) -> tuple[str, ...]:
        return self.methods[0].selector_source_exprs

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(method.evidence for method in self.methods[:6])


@dataclass(frozen=True)
class WrapperChainCandidate:
    file_path: str
    wrappers: tuple[FunctionWrapperCandidate, ...]
    leaf_delegate_symbol: str


@dataclass(frozen=True)
class PipelineAssemblyStage:
    kind: str
    callee_name: str
    output_arity: int
    arg_count: int
    keyword_names: tuple[str, ...] = ()

    @property
    def shape_key(self) -> tuple[object, ...]:
        return (
            self.kind,
            self.callee_name,
            self.output_arity,
            self.arg_count,
            self.keyword_names,
        )


@dataclass(frozen=True)
class ResultAssemblyPipelineFunction:
    file_path: str
    qualname: str
    lineno: int
    stages: tuple[PipelineAssemblyStage, ...]

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.lineno, self.qualname)


@dataclass(frozen=True)
class RepeatedResultAssemblyPipelineCandidate:
    file_path: str
    shared_tail: tuple[PipelineAssemblyStage, ...]
    functions: tuple[ResultAssemblyPipelineFunction, ...]


@dataclass(frozen=True)
class NestedBuilderShellCandidate:
    file_path: str
    qualname: str
    lineno: int
    outer_callee_name: str
    nested_field_name: str
    nested_callee_name: str
    forwarded_parameter_names: tuple[str, ...]
    residue_field_names: tuple[str, ...]
    residue_source_names: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.lineno, self.qualname)


@dataclass(frozen=True)
class ManualFiberTagCandidate(WitnessCarrierCandidate):
    init_line: int
    method_name: str
    tag_name: str
    assigned_field_names: tuple[str, ...]

    @property
    def method_line(self) -> int:
        return self.line

    @property
    def case_names(self) -> tuple[str, ...]:
        return self.name_family


@dataclass(frozen=True)
class DescriptorDerivedViewCandidate(WitnessCarrierCandidate):
    source_attr: str
    init_line: int
    mutator_name: str
    updated_field_names: tuple[str, ...]

    @property
    def mutator_line(self) -> int:
        return self.line

    @property
    def derived_field_names(self) -> tuple[str, ...]:
        return self.name_family


@dataclass(frozen=True)
class ManualRegistryCandidate(WitnessCarrierCandidate, NameFamilyClassNamesMixin):
    decorator_name: str
    unregistered_class_names: tuple[str, ...]

    @property
    def registry_name(self) -> str:
        return self.subject_name


@dataclass(frozen=True)
class StructuralConfusabilityCandidate(
    WitnessCarrierCandidate,
    NameFamilyClassNamesMixin,
    SubjectNameFunctionNameMixin,
):
    parameter_name: str
    observed_method_names: tuple[str, ...]


@dataclass(frozen=True)
class WitnessCarrierClassCandidate(WitnessCarrierCandidate):
    base_names: tuple[str, ...]
    family_tokens: tuple[str, ...]
    normalized_roles: tuple[str, ...]
    normalized_role_fields: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.name_family


@dataclass(frozen=True)
class WitnessCarrierFamilyCandidate(ClassLineNumbersGroup):
    shared_role_names: tuple[str, ...]


@dataclass(frozen=True)
class WitnessMixinEnforcementCandidate(ClassLineNumbersGroup):
    role_field_names: tuple[tuple[str, tuple[str, ...]], ...]


def _axis_dispatch_metrics(
    literal_cases: tuple[str, ...],
    dispatch_axis: str,
    dispatch_site_count: int | None = None,
) -> DispatchCountMetrics:
    if dispatch_site_count is None:
        dispatch_site_count = len(literal_cases)
    return DispatchCountMetrics(
        dispatch_site_count=dispatch_site_count,
        dispatch_axis=dispatch_axis,
        literal_cases=literal_cases,
    )


class RepeatedPrivateMethodDetector(FiberCollectedShapeIssueDetector):
    detector_id = "repeated_private_methods"
    observation_kind = ObservationKind.METHOD_SHAPE
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated non-orthogonal method skeleton across classes",
        why=(
            "Shared orchestration logic is duplicated across a behavior family. The docs say this shared "
            "non-orthogonal logic should move into an ABC with a concrete template method, leaving only "
            "orthogonal hooks in subclasses."
        ),
        capability_gap="single authoritative algorithm for a nominal behavior family",
        relation_context="same method role across sibling classes",
        confidence=HIGH_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.NORMALIZED_AST,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.METHOD_ROLE,
        ),
    )

    def _module_shapes(self, module: ParsedModule) -> tuple[object, ...]:
        return tuple(
            _collect_typed_family_items(module, MethodShapeFamily, MethodShape)
        )

    def _include_shape(self, shape: object, config: DetectorConfig) -> bool:
        method = _as_method_shape(shape)
        return bool(
            method.class_name
            and method.statement_count >= config.min_duplicate_statements
        )

    def _group_key(self, shape: object) -> object:
        method = _as_method_shape(shape)
        return (method.is_private, method.param_count, method.fingerprint)

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        methods = tuple(
            sorted(
                (_as_method_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        class_names = {method.class_name for method in methods}
        if len(methods) < 2 or len(class_names) < 2:
            return None
        evidence = tuple(
            SourceLocation(method.file_path, method.lineno, method.symbol)
            for method in methods[:6]
        )
        relation = (
            "same private helper role across sibling classes"
            if methods[0].is_private
            else "same method role across sibling classes"
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"{len(methods)} methods across {len(class_names)} classes share the same normalized AST shape."
            ),
            evidence,
            relation_context=relation,
            scaffold=_abc_scaffold_for_methods(methods),
            codemod_patch=_abc_patch_for_methods(methods),
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(methods),
                statement_count=methods[0].statement_count,
                class_count=len(class_names),
                method_symbols=tuple(method.symbol for method in methods),
                shared_statement_texts=methods[0].statement_texts,
            ),
        )


class InheritanceHierarchyCandidateDetector(IssueDetector):
    detector_id = "inheritance_hierarchy_candidate"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Classes cluster into an ABC hierarchy candidate",
        why=(
            "The same set of classes repeats multiple non-orthogonal method skeletons. The docs say this is a "
            "strong signal that the family should be factored into an ABC with one concrete template method "
            "layer; orthogonal reusable concerns can then live in mixins so MRO preserves declared precedence."
        ),
        capability_gap="single authoritative inheritance hierarchy for a duplicated behavior family",
        relation_context="same class set repeats several method roles across the same family boundary",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.REPEATED_METHOD_ROLES,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        repeated_methods = tuple(
            method
            for module in modules
            for method in _collect_typed_family_items(
                module, MethodShapeFamily, MethodShape
            )
            if method.class_name
            and method.statement_count >= config.min_duplicate_statements
        )
        graph = ObservationGraph(
            tuple(method.structural_observation for method in repeated_methods)
        )
        lookup = _carrier_lookup(tuple(repeated_methods))

        findings: list[RefactorFinding] = []
        for cohort in graph.coherence_cohorts_for(
            ObservationKind.METHOD_SHAPE,
            StructuralExecutionLevel.FUNCTION_BODY,
            minimum_witnesses=2,
            minimum_fibers=1,
        ):
            groups = [
                tuple(
                    _as_method_shape(item)
                    for item in _materialize_observations(fiber.observations, lookup)
                )
                for fiber in cohort.fibers
            ]
            if not groups:
                continue
            class_names = frozenset(cohort.nominal_witnesses)
            method_count_by_class: dict[str, int] = defaultdict(int)
            for methods in groups:
                for method in methods:
                    if method.class_name is not None:
                        method_count_by_class[method.class_name] += 1
            supports_family = (
                len(groups) >= 2
                or sum(1 for count in method_count_by_class.values() if count >= 2) >= 2
            )
            if not supports_family:
                continue
            evidence = [
                SourceLocation(method.file_path, method.lineno, method.symbol)
                for methods in groups
                for method in methods
            ]
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"Classes {', '.join(sorted(class_names))} share {len(groups)} repeated method-shape groups and repeated method roles that likely want one ABC family."
                    ),
                    tuple(evidence[:8]),
                    scaffold=_abc_family_scaffold(class_names, groups),
                    codemod_patch=_abc_family_patch(class_names, groups),
                    metrics=HierarchyCandidateMetrics(
                        duplicate_group_count=len(groups),
                        class_count=len(class_names),
                    ),
                )
            )
        return findings


class OrchestrationHubDetector(CandidateFindingDetector):
    detector_id = "orchestration_hub"
    finding_spec = FindingSpec(
        pattern_id=PatternId.STAGED_ORCHESTRATION,
        title="Oversized orchestration hub",
        why=(
            "One function is owning too many control branches, helper calls, and phase transitions at once. "
            "The architecture wants explicit staged boundaries so the orchestration surface remains nominal and legible."
        ),
        capability_gap="explicit staged orchestration boundaries with named phase contracts",
        relation_context="one owner centralizes many operational phases and helper families",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return tuple(
            profile
            for profile in _function_profiles(module)
            if profile.line_count >= config.min_orchestration_function_lines
            and profile.branch_count >= config.min_orchestration_branches
            and profile.call_count >= config.min_orchestration_calls
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        profile = cast(FunctionProfile, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{profile.qualname}` concentrates {profile.line_count} lines, {profile.branch_count} branches, and {profile.call_count} calls across {profile.callee_family_count} callee families in one owner."
            ),
            (profile.evidence,),
            scaffold=_orchestration_stage_scaffold(profile),
            codemod_patch=_orchestration_stage_patch(profile),
            metrics=OrchestrationMetrics(
                function_line_count=profile.line_count,
                branch_site_count=profile.branch_count,
                call_site_count=profile.call_count,
                parameter_count=len(profile.parameter_names),
                callee_family_count=profile.callee_family_count,
            ),
        )


class PrivateCohortShouldBeModuleDetector(CandidateFindingDetector):
    detector_id = "private_cohort_should_be_module"
    finding_spec = FindingSpec(
        pattern_id=PatternId.STAGED_ORCHESTRATION,
        title="Private subsystem cohort wants its own module",
        why=(
            "One module is carrying a tightly-coupled private subsystem cohort as if it were a whole package. "
            "The architecture wants a dedicated module for that bounded context, with the original file reduced to orchestration or public entry points."
        ),
        capability_gap="explicit module-level subsystem boundaries with extracted private cohorts",
        relation_context="one file contains a dense private context/result/helper family that should move together",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _private_cohort_should_be_module_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        cohort = cast(PrivateCohortShouldBeModuleCandidate, candidate)
        shared_tokens = ", ".join(cohort.shared_tokens[:3]) or "subsystem"
        sample_symbols = ", ".join(
            symbol.symbol
            for symbol in sorted(
                cohort.symbols,
                key=lambda item: (-item.line_count, item.line, item.symbol),
            )[:3]
        )
        target_module = _suggest_private_cohort_module_name(cohort)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{cohort.module_name}` carries a private {shared_tokens} cohort across "
                f"{len(cohort.symbols)} top-level symbols / {cohort.total_cohort_lines} lines "
                f"inside a {cohort.module_line_count}-line module; extract `{sample_symbols}` "
                f"into a dedicated `{target_module}.py` module."
            ),
            cohort.evidence,
            scaffold=(
                f"# {target_module}.py\n"
                "@dataclass(frozen=True)\n"
                f"class {_camel_case('_'.join(cohort.shared_tokens[:2]) or 'subsystem')}Context:\n"
                "    ...\n\n"
                f"def run_{'_'.join(cohort.shared_tokens[:2]) or 'subsystem'}(...):\n"
                "    ...\n\n"
                "# Move the private context/result carriers and worker helpers here.\n"
                "# Leave only public orchestration entry points in the original module."
            ),
            codemod_patch=(
                f"# Extract the private {shared_tokens} cohort into `{target_module}.py`.\n"
                "# Move the cohort's private dataclasses, helper functions, and result carriers together.\n"
                "# Import the extracted helpers back into the original module only where public entry points still need them.\n"
                "# Keep sequencing, public APIs, and thin phase boundaries in the original file."
            ),
        )


class ParameterThreadFamilyDetector(CandidateFindingDetector):
    detector_id = "parameter_thread_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_CONTEXT,
        title="Repeated threaded semantic parameter family",
        why=(
            "Several helpers keep re-threading the same semantic parameter bundle instead of carrying one nominal context. "
            "That weakens provenance and makes each helper signature a partially duplicated view of the same authority."
        ),
        capability_gap="one authoritative context/request record for a shared semantic parameter family",
        relation_context="the same semantic parameter bundle is threaded through several sibling helpers",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _parameter_thread_family_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        parameter_family = cast(ParameterThreadFamilyCandidate, candidate)
        function_names = tuple(item.qualname for item in parameter_family.functions)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Functions {', '.join(function_names[:4])} thread the same semantic parameter family `{', '.join(parameter_family.shared_parameter_names)}` across {len(parameter_family.functions)} helpers."
            ),
            tuple(item.evidence for item in parameter_family.functions[:6]),
            scaffold=_authoritative_context_scaffold(parameter_family),
            codemod_patch=_authoritative_context_patch(parameter_family),
            metrics=ParameterThreadMetrics(
                function_count=len(parameter_family.functions),
                shared_parameter_count=len(parameter_family.shared_parameter_names),
                shared_parameter_names=parameter_family.shared_parameter_names,
            ),
        )


class EnumStrategyDispatchDetector(CandidateFindingDetector):
    detector_id = "enum_strategy_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_STRATEGY_FAMILY,
        title="Enum strategy ladder wants nominal family",
        why=(
            "A closed enum/member dispatch ladder is choosing among behavior implementations inline. "
            "That wants an ABC-backed strategy family so each implementation guarantees one common method and the caller stops branching."
        ),
        capability_gap="nominal strategy family with one guaranteed call surface",
        relation_context="one owner branches over a closed enum/member family instead of delegating to implementation classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _enum_strategy_dispatch_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        dispatch_candidate = cast(EnumStrategyDispatchCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{dispatch_candidate.qualname}` branches on `{dispatch_candidate.dispatch_axis}` across closed cases {', '.join(dispatch_candidate.case_names)} and should delegate to a nominal strategy family."
            ),
            (dispatch_candidate.evidence,),
            scaffold=_nominal_strategy_scaffold(dispatch_candidate),
            codemod_patch=_nominal_strategy_patch(dispatch_candidate),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(dispatch_candidate.case_names),
                dispatch_axis=dispatch_candidate.dispatch_axis,
                literal_cases=dispatch_candidate.case_names,
            ),
        )


class RepeatedEnumStrategyDispatchDetector(CandidateFindingDetector):
    detector_id = "repeated_enum_strategy_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_STRATEGY_FAMILY,
        title="Repeated closed-strategy dispatch should centralize in one nominal strategy family",
        why=(
            "Several owners re-dispatch the same closed enum family inline. The docs treat that as duplicated "
            "strategy orchestration: dispatch should happen once through one authoritative nominal strategy family "
            "or one shared strategy substrate."
        ),
        capability_gap="single authoritative nominal strategy family for a repeated closed dispatch axis",
        relation_context="same closed enum family is re-dispatched across sibling functions or methods",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _repeated_enum_strategy_dispatch_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        dispatch_candidate = cast(RepeatedEnumStrategyDispatchCandidate, candidate)
        evidence = tuple(
            item.evidence for item in dispatch_candidate.functions[:6]
        )
        representative = dispatch_candidate.functions[0]
        function_names = ", ".join(
            item.qualname for item in dispatch_candidate.functions[:4]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Functions {function_names} each re-dispatch `{dispatch_candidate.enum_family}` cases "
                f"{', '.join(dispatch_candidate.shared_case_names)} inline."
            ),
            evidence,
            scaffold=_nominal_strategy_scaffold(representative),
            codemod_patch=_nominal_strategy_patch(representative),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(dispatch_candidate.functions),
                dispatch_axis=dispatch_candidate.enum_family,
                literal_cases=dispatch_candidate.shared_case_names,
            ),
        )


class SplitDispatchAuthorityDetector(CandidateFindingDetector):
    detector_id = "split_dispatch_authority"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_STRATEGY_FAMILY,
        title="Cooperating dispatch layers should collapse into one product-family authority",
        why=(
            "The docs treat repeated cooperating dispatch layers as split authority. When one orchestration function "
            "selects a strategy-family implementation and separately routes another axis through `singledispatch`, "
            "the operation usually wants one authoritative product-family policy or one request-dispatched plan."
        ),
        capability_gap="single authoritative product-family or request-dispatched policy for cooperating dispatch axes",
        relation_context="one orchestrator combines a strategy-family selector with a separate singledispatch generic",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.FACTORY_DISPATCH,
            ObservationTag.REPEATED_METHOD_ROLES,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _split_dispatch_authority_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        dispatch_candidate = cast(SplitDispatchAuthorityCandidate, candidate)
        evidence = (
            dispatch_candidate.evidence,
            SourceLocation(
                dispatch_candidate.file_path,
                dispatch_candidate.selector_line,
                f"{dispatch_candidate.strategy_root_name}.{dispatch_candidate.selector_method_name}",
            ),
            SourceLocation(
                dispatch_candidate.file_path,
                dispatch_candidate.generic_line,
                dispatch_candidate.generic_function_name,
            ),
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{dispatch_candidate.qualname}` combines strategy selector "
                f"`{dispatch_candidate.strategy_root_name}.{dispatch_candidate.selector_method_name}({dispatch_candidate.strategy_axis_expression})` "
                f"with singledispatch `{dispatch_candidate.generic_function_name}({dispatch_candidate.generic_axis_expression})` "
                f"through callback `{dispatch_candidate.bridge_callback_name}`, splitting one operation across two dispatch authorities."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class DispatchPlan:\n"
                "    strategy: object\n"
                "    source_type: type[object]\n\n"
                "class ProductPolicy(ABC):\n"
                "    plan_key: ClassVar[DispatchPlan]\n"
                "    def run(self, request): ...\n"
            ),
            codemod_patch=(
                f"# Collapse `{dispatch_candidate.strategy_root_name}` and `{dispatch_candidate.generic_function_name}` under one product-family authority.\n"
                "# Let one nominal plan/policy own both `{dispatch_candidate.strategy_axis_expression}` and `{dispatch_candidate.generic_axis_expression}` so the orchestrator dispatches once."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=2,
                dispatch_axis=(
                    f"{dispatch_candidate.strategy_axis_expression} x "
                    f"{dispatch_candidate.generic_axis_expression}"
                ),
                literal_cases=(
                    *dispatch_candidate.strategy_case_names[:3],
                    *dispatch_candidate.generic_case_names[:3],
                ),
            ),
        )


class EmptyLeafProductFamilyDetector(CandidateFindingDetector):
    detector_id = "empty_leaf_product_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CLOSED_FAMILY_DISPATCH,
        title="Empty multiple-inheritance leaves should collapse into one product-family authority",
        why=(
            "The docs allow mixins for orthogonal reusable concerns, but empty leaf classes that merely enumerate "
            "all combinations of two reusable axes are usually a handwritten product table in inheritance form. "
            "That product should become one keyed authority or one product-family selector."
        ),
        capability_gap="single authoritative keyed product family instead of empty inheritance combinations",
        relation_context="empty leaf classes encode the full Cartesian product of two reusable inheritance axes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REPEATED_METHOD_ROLES,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _empty_leaf_product_family_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        product_candidate = cast(EmptyLeafProductFamilyCandidate, candidate)
        left_axis = ", ".join(product_candidate.left_axis_base_names)
        right_axis = ", ".join(product_candidate.right_axis_base_names)
        leaf_preview = ", ".join(product_candidate.leaf_class_names[:6])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Empty leaf classes {leaf_preview} encode `{left_axis}` x `{right_axis}` through multiple inheritance instead of one product-family authority."
            ),
            product_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class ProductRule:\n"
                "    axis_left: object\n"
                "    axis_right: object\n"
                "    policy_type: type[object]\n\n"
                "PRODUCT_RULES = (...)\n"
            ),
            codemod_patch=(
                "# Replace the empty Cartesian-product leaf classes with one keyed product table or one nominal selector family.\n"
                "# Keep only irreducible axis-local behavior on the reusable bases; do not encode the cross product as `pass` subclasses."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(product_candidate.leaf_class_names),
                dispatch_axis=(
                    f"{' | '.join(product_candidate.left_axis_base_names)} x "
                    f"{' | '.join(product_candidate.right_axis_base_names)}"
                ),
                literal_cases=product_candidate.leaf_class_names,
            ),
        )


class ClosedConstantSelectorDetector(CandidateFindingDetector):
    detector_id = "closed_constant_selector"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Closed selector over sibling constants should derive from one selector table",
        why=(
            "The docs treat branch ladders that choose among sibling specs, plans, contracts, or other immutable "
            "constants as duplicated selector logic once the constant family already exists. The selector should "
            "collapse into one authoritative keyed table or selector record so wrappers and downstream views are derived."
        ),
        capability_gap="single authoritative selector table for a closed constant family",
        relation_context="one function branches over a small predicate family and returns sibling constants or one shared wrapper around them",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.PREDICATE_CHAIN,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _closed_constant_selector_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        selector_candidate = cast(ClosedConstantSelectorCandidate, candidate)
        constants_preview = ", ".join(selector_candidate.constant_names[:4])
        guard_preview = ", ".join(selector_candidate.guard_expressions[:2])
        family_label = (
            selector_candidate.common_constructor_name
            or selector_candidate.family_suffix
            or "selected constant family"
        )
        wrapper_summary = (
            f"`{selector_candidate.wrapper_name}(...)` around "
            if selector_candidate.wrapper_name is not None
            else ""
        )
        guard_summary = (
            f"guards `{guard_preview}` and default fallback"
            if selector_candidate.guard_expressions
            else "a closed fallback ladder"
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{selector_candidate.qualname}` branches over {guard_summary}, returning {wrapper_summary}"
                f"sibling constants {constants_preview} from `{family_label}`."
            ),
            selector_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class SelectorRule:\n"
                "    key: object\n"
                "    selected: object\n\n"
                "SELECTOR_RULES = (\n"
                "    SelectorRule(key=..., selected=...),\n"
                ")\n"
                "_SELECTED_BY_KEY = {rule.key: rule.selected for rule in SELECTOR_RULES}\n"
            ),
            codemod_patch=(
                f"# Replace manual branches in `{selector_candidate.qualname}` with one authoritative selector table.\n"
                "# Select the sibling constant once, then apply any shared wrapper outside the selector."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(selector_candidate.constant_names),
                field_count=max(len(selector_candidate.guard_expressions), 1),
                mapping_name=selector_candidate.wrapper_name or family_label,
                field_names=selector_candidate.constant_names,
                source_name=selector_candidate.qualname,
            ),
        )


class DerivedWrapperSpecShadowDetector(CandidateFindingDetector):
    detector_id = "derived_wrapper_spec_shadow"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Generated wrapper spec family should collapse into the authoritative spec family",
        why=(
            "The docs treat writable wrapper-spec tables as secondary authorities when they just point back at an "
            "existing spec family and feed code generation. Wrapper metadata should live on the authoritative spec "
            "records so generated wrappers are derived from one source rather than synchronized across parallel tables."
        ),
        capability_gap="single authoritative spec family carrying wrapper-generation metadata",
        relation_context="secondary spec table references an authoritative spec family entry-by-entry and is only consumed by wrapper generation",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.SCOPED_SHAPE_WRAPPER,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _derived_wrapper_spec_shadow_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        shadow_candidate = cast(DerivedWrapperSpecShadowCandidate, candidate)
        primary_family_label = (
            shadow_candidate.primary_family_name or shadow_candidate.primary_constructor_name
        )
        constant_preview = ", ".join(shadow_candidate.primary_constant_names[:4])
        builder_preview = ", ".join(shadow_candidate.builder_names[:3])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{shadow_candidate.derived_family_name}` re-encodes wrapper metadata over authoritative family "
                f"`{primary_family_label}` through link field `{shadow_candidate.link_field_name}` for {constant_preview}, "
                f"then feeds generated wrappers via {builder_preview}."
            ),
            shadow_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class ExecutionSpec:\n"
                "    key: object\n"
                "    runner: object\n"
                "    wrapper_name: str | None = None\n"
                "    wrapper_defaults: dict[str, object] = field(default_factory=dict)\n\n"
                "def build_wrapper(spec: ExecutionSpec): ...\n"
            ),
            codemod_patch=(
                f"# Remove parallel family `{shadow_candidate.derived_family_name}`.\n"
                f"# Move `{', '.join(shadow_candidate.extra_field_names) or 'wrapper metadata'}` onto the authoritative "
                f"`{shadow_candidate.primary_constructor_name}` records and derive wrappers directly from that family."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(shadow_candidate.primary_constant_names),
                field_count=max(len(shadow_candidate.extra_field_names), 1),
                mapping_name=shadow_candidate.derived_family_name,
                field_names=shadow_candidate.extra_field_names,
                source_name=primary_family_label,
                identity_field_names=(shadow_candidate.link_field_name,),
            ),
        )


class ModuleKeyedSelectionHelperDetector(CandidateFindingDetector):
    detector_id = "module_keyed_selection_helper"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Local keyed-selection helper should collapse into the generic keyed-record table",
        why=(
            "The docs push reusable table/index machinery into one authoritative substrate. When a module defines a "
            "local selection-rule dataclass, a dict-index builder, and a keyed lookup helper that power multiple rule "
            "tables, it is reintroducing a second keyed-table framework instead of reusing the generic keyed-record helper."
        ),
        capability_gap="single authoritative keyed-record table substrate reused across module-level selector tables",
        relation_context="module-local selection helper framework powers multiple keyed rule tables",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _module_keyed_selection_helper_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        helper_candidate = cast(ModuleKeyedSelectionHelperCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{helper_candidate.rule_class_name}`, `{helper_candidate.helper_function_name}`, and "
                f"`{helper_candidate.lookup_function_name}` implement a local keyed-selection substrate for "
                f"{', '.join(helper_candidate.rule_table_names[:4])} and indexes {', '.join(helper_candidate.index_table_names[:4])}."
            ),
            helper_candidate.evidence,
            scaffold=(
                "KeyT = TypeVar(\"KeyT\")\n"
                "RecordT = TypeVar(\"RecordT\")\n\n"
                "@dataclass(frozen=True)\n"
                "class KeyedRecordTable(Generic[KeyT, RecordT]):\n"
                "    records: tuple[RecordT, ...]\n"
                "    key_of: Callable[[RecordT], KeyT]\n"
                "    def require(self, key: KeyT, *, missing_error=None) -> RecordT: ...\n"
            ),
            codemod_patch=(
                f"# Remove local keyed-selection helper `{helper_candidate.rule_class_name}` / "
                f"`{helper_candidate.helper_function_name}` / `{helper_candidate.lookup_function_name}`.\n"
                "# Re-express these rule tables through the shared KeyedRecordTable substrate."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(helper_candidate.rule_table_names),
                field_count=1,
                mapping_name=helper_candidate.rule_class_name,
                field_names=(helper_candidate.selected_field_name,),
                source_name=helper_candidate.helper_function_name,
                identity_field_names=("key",),
            ),
        )


class CrossModuleAxisShadowFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "cross_module_axis_shadow_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_STRATEGY_FAMILY,
        title="Cross-module shadow family should collapse into one axis authority",
        why=(
            "The docs require one authoritative owner per closed semantic axis. When one module already owns an enum/keyed "
            "family nominally and another module reintroduces a second family over the same cases, the axis has split "
            "authority and local behavior should derive from the authoritative family instead."
        ),
        capability_gap="single authoritative closed-axis family reused across modules",
        relation_context="same keyed enum axis is modeled by an authoritative family in one module and a shadow selector family in another",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.FACTORY_DISPATCH,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _cross_module_axis_shadow_family_candidates(modules)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        shadow_candidate = cast(CrossModuleAxisShadowFamilyCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Axis `{shadow_candidate.key_type_name}` is already owned by "
                f"`{shadow_candidate.authoritative.family_name}` but re-encoded by "
                f"`{shadow_candidate.shadow.family_name}.{shadow_candidate.selector_method_name}` "
                f"across cases {', '.join(shadow_candidate.shared_case_names[:4])}."
            ),
            shadow_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("invariant(self)")
                + "\n\n"
                f"def run_with_axis(axis: {_AXIS_POLICY_KEY_TYPE_NAME}, ...):\n"
                f"    policy = {_AXIS_POLICY_ROOT_NAME}.for_key(axis)\n"
                "    # derive local execution from authoritative policy facts\n"
            ),
            codemod_patch=(
                f"# Remove shadow family `{shadow_candidate.shadow.family_name}`.\n"
                f"# Derive local behavior from authoritative family `{shadow_candidate.authoritative.family_name}` instead of re-owning axis `{shadow_candidate.key_type_name}`."
            ),
            metrics=_axis_dispatch_metrics(
                shadow_candidate.shared_case_names,
                shadow_candidate.key_type_name,
            ),
        )


class ResidualClosedAxisBranchingDetector(CrossModuleCandidateDetector):
    detector_id = "residual_closed_axis_branching"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CLOSED_FAMILY_DISPATCH,
        title="Manual closed-axis branching should derive from existing keyed authority",
        why=(
            "The docs require one authoritative owner per closed enum/key axis. When a keyed nominal family already "
            "owns that axis, downstream `if`/`match` ladders over the same cases become residual shadow dispatch."
        ),
        capability_gap="behavior derived from authoritative keyed family rather than downstream enum branching",
        relation_context="function branches on an enum axis already owned by a keyed nominal family in another module",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _residual_closed_axis_branching_candidates(modules)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        residual_candidate = cast(ResidualClosedAxisBranchingCandidate, candidate)
        authoritative_family_names = ", ".join(
            family_name
            for family_name, _, _ in residual_candidate.authoritative_families[:4]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{residual_candidate.qualname}` branches {residual_candidate.branch_site_count} time(s) on axis "
                f"`{residual_candidate.key_type_name}` across cases {', '.join(residual_candidate.case_names)}, "
                f"even though authoritative family `{authoritative_family_names}` already owns that axis."
            ),
            residual_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("apply(self, context)")
                + "\n\n"
                "def run(context):\n"
                f"    policy = {_AXIS_POLICY_ROOT_NAME}.for_key(context.axis)\n"
                "    return policy.apply(context)\n"
            ),
            codemod_patch=(
                f"# Remove residual `{residual_candidate.key_type_name}` branch ladder in `{residual_candidate.qualname}`.\n"
                "# Delegate through the existing keyed family authority and keep only case-local residue on the policy leaves."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=residual_candidate.branch_site_count,
                dispatch_axis=residual_candidate.key_type_name,
                literal_cases=residual_candidate.case_names,
            ),
        )


class ParallelKeyedAxisFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "parallel_keyed_axis_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_STRATEGY_FAMILY,
        title="Parallel keyed families should collapse into one axis authority",
        why=(
            "The docs require one authoritative nominal owner per closed semantic axis. When two modules each define a "
            "keyed family over the same enum/key cases, the axis has split ownership even if both sides are nominal."
        ),
        capability_gap="single cross-module keyed-axis authority with module-local adapters derived from it",
        relation_context="same keyed enum axis is modeled by multiple nominal families across modules",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.FACTORY_DISPATCH,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _parallel_keyed_axis_family_candidates(modules)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(ParallelKeyedAxisFamilyCandidate, candidate)
        shared_cases = ", ".join(family_candidate.shared_case_names[:4])
        label_clause = ""
        if (
            family_candidate.left.family_label is not None
            and family_candidate.left.family_label == family_candidate.right.family_label
        ):
            label_clause = (
                f" Both declare family label `{family_candidate.left.family_label}`."
            )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Axis `{family_candidate.key_type_name}` is owned in parallel by "
                f"`{family_candidate.left.family_name}` and `{family_candidate.right.family_name}` "
                f"across cases {shared_cases}.{label_clause}"
            ),
            family_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold(
                    "invariant(self)",
                    "runtime_adapter(self, context)",
                )
                + "\n\n"
                "# Keep one authoritative keyed family and let secondary modules derive local adapters/specs from it."
            ),
            codemod_patch=(
                f"# Collapse `{family_candidate.left.family_name}` and `{family_candidate.right.family_name}` onto one authoritative keyed family.\n"
                "# Move the irreducible case-specific hooks to that family or to a single derived adapter table, not two parallel nominal roots."
            ),
            metrics=_axis_dispatch_metrics(
                family_candidate.shared_case_names,
                family_candidate.key_type_name,
            ),
        )


class ParallelKeyedTableAndFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "parallel_keyed_table_and_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Keyed table and keyed family should collapse into one auto-registered axis family",
        why=(
            "The docs require one authoritative owner per closed semantic axis. When a module keeps one keyed table of "
            "per-case records and a second keyed nominal family over the same cases, the axis is split across data and behavior. "
            "If the family already carries the runtime behavior boundary, the table should derive from that family instead of competing with it."
        ),
        capability_gap="single authoritative metaclass-registry axis family with derived table/view projections",
        relation_context="same enum/key axis is encoded by both a keyed table and a keyed nominal family",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _parallel_keyed_table_and_family_candidates(modules)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        table_candidate = cast(ParallelKeyedTableAndFamilyCandidate, candidate)
        shape_clause = (
            ""
            if table_candidate.value_shape_name is None
            else f" of `{table_candidate.value_shape_name}` records"
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Axis `{table_candidate.key_type_name}` is split between keyed table `{table_candidate.table_name}`"
                f"{shape_clause} and keyed family `{table_candidate.family_name}` across cases "
                f"{', '.join(table_candidate.shared_case_names[:4])}."
            ),
            table_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("build(self)")
                + "\n\n"
                "@dataclass(frozen=True)\n"
                "class DerivedAxisRow:\n"
                f"    key: {_AXIS_POLICY_KEY_TYPE_NAME}\n"
                f"    policy_type: type[{_AXIS_POLICY_ROOT_NAME}]\n"
                "    config: object\n\n"
                "def build_axis_rows() -> tuple[DerivedAxisRow, ...]:\n"
                "    return tuple(\n"
                "        DerivedAxisRow(key=key, policy_type=policy_type, config=...)\n"
                f"        for key, policy_type in {_AXIS_POLICY_ROOT_NAME}.__registry__.items()\n"
                "    )"
            ),
            codemod_patch=(
                f"# Collapse `{table_candidate.table_name}` and `{table_candidate.family_name}` onto one authoritative metaclass-registry family.\n"
                "# Keep the runtime boundary on the auto-registered family and derive any keyed rows/views from `AxisPolicy.__registry__`."
            ),
            metrics=_axis_dispatch_metrics(
                table_candidate.shared_case_names,
                table_candidate.key_type_name,
            ),
        )


class EnumKeyedTableClassAxisShadowDetector(CandidateFindingDetector):
    detector_id = "enum_keyed_table_class_axis_shadow"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Enum-keyed table should derive from auto-registered class-declared axis keys",
        why=(
            "The docs require a single writable owner per closed semantic axis. If a module already declares "
            "that axis through class-level enum assignments, adding a writable enum-keyed table over the same "
            "cases creates duplicate authority and a synchronization surface. The class-declared axis should be the "
            "primary owner and any enum-keyed lookup should be derived from the family registry."
        ),
        capability_gap="one authoritative metaclass-registry closed-axis owner with derived table/view projections",
        relation_context="module-level enum-keyed table overlaps a class family that already declares the same enum axis",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.PROJECTION_DICT,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _enum_keyed_table_class_axis_shadow_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        axis_candidate = cast(EnumKeyedTableClassAxisShadowCandidate, candidate)
        class_names = ", ".join(axis_candidate.class_names[:4])
        shared_cases = ", ".join(axis_candidate.shared_case_names[:4])
        value_names = ", ".join(axis_candidate.value_type_names[:4])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{axis_candidate.table_name}` maps `{axis_candidate.key_type_name}` cases {shared_cases} "
                f"to {value_names}, while classes {class_names} already declare the same axis via "
                f"`{axis_candidate.key_attr_name}`."
            ),
            axis_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("route_type(self)")
                + "\n\n"
                "AXIS_BY_KEY = {\n"
                "    key: policy_type\n"
                f"    for key, policy_type in {_AXIS_POLICY_ROOT_NAME}.__registry__.items()\n"
                "}\n"
            ),
            codemod_patch=(
                f"# Remove `{axis_candidate.table_name}` as a second writable authority.\n"
                f"# Derive `{axis_candidate.key_type_name}` lookup views from the auto-registered family keyed by `{axis_candidate.key_attr_name}` instead of hardcoding a parallel table."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(axis_candidate.shared_case_names),
                field_count=1,
                mapping_name=axis_candidate.table_name,
                field_names=(axis_candidate.key_attr_name,),
                source_name=axis_candidate.key_type_name,
                identity_field_names=(axis_candidate.key_attr_name,),
            ),
        )


class TransportShellTemplateMethodDetector(CandidateFindingDetector):
    detector_id = "transport_shell_template_method"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Template-method family is a transport shell over a downstream authority",
        why=(
            "The docs say nominal families should have one authoritative owner. When an ABC template method only "
            "materializes an intermediate object from a class-level selector, delegates through one hook, and "
            "repackages through another hook, the extra family is usually a transport shell around an already "
            "authoritative boundary."
        ),
        capability_gap="single authoritative materialization/execution family instead of a parallel transport shell",
        relation_context="template family varies mostly by class-level selector and result adapter",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _transport_shell_template_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        shell_candidate = cast(TransportShellTemplateCandidate, candidate)
        selector_values = ", ".join(shell_candidate.selector_value_names)
        kwargs_clause = (
            f" plus `{shell_candidate.kwargs_helper_name}({shell_candidate.source_param_name})`"
            if shell_candidate.kwargs_helper_name is not None
            else ""
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{shell_candidate.class_name}.{shell_candidate.driver_method_name}` materializes selector values "
                f"{selector_values} from `{shell_candidate.selector_attr_name}` via `{shell_candidate.constructor_name}`"
                f"{kwargs_clause} across {len(shell_candidate.concrete_class_names)} concrete leaves, then only delegates "
                f"through `{shell_candidate.inner_hook_name}` and `{shell_candidate.outer_hook_name}`."
            ),
            (shell_candidate.evidence,),
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class MaterializationSpec:\n"
                "    selector: object\n"
                "    materializer: object\n"
                "    executor: object\n"
                "    packager: object\n"
                "# Dispatch once on the authoritative selector/spec family."
            ),
            codemod_patch=(
                f"# Collapse `{shell_candidate.class_name}` onto the downstream selector/spec family.\n"
                "# Keep one selection boundary and let that boundary own materialization, execution, and result packaging."
            ),
        )


class CrossModuleSpecAxisAuthorityDetector(CrossModuleCandidateDetector):
    detector_id = "cross_module_spec_axis_authority"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Cross-module spec axis should have one authority",
        why=(
            "The docs say one semantic family should have one authoritative owner. When two modules encode the same "
            "identity-axis -> executable-axis spec pairs, one table is a duplicate authority unless it is explicitly derived."
        ),
        capability_gap="one repository-wide authoritative spec-axis family",
        relation_context="same identity/executable spec axis is re-encoded across modules",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _cross_module_spec_axis_authority_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        authority_candidate = cast(CrossModuleSpecAxisAuthorityCandidate, candidate)
        family_names = ", ".join(
            f"{Path(family.file_path).name}:{family.family_name}"
            for family in authority_candidate.families
        )
        pair_names = ", ".join(
            f"{identity}->{executable}"
            for identity, executable in authority_candidate.shared_axis_pairs
        )
        axis_fields = " -> ".join(authority_candidate.axis_field_names)
        evidence = tuple(
            family.evidence for family in authority_candidate.families[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Families {family_names} each encode the same `{axis_fields}` pairs {pair_names} across module boundaries."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class AxisExecutionSpec:\n"
                "    identity: object\n"
                "    executable: object\n"
                "# Keep one exported authority and let downstream modules compose from it."
            ),
            codemod_patch=(
                "# Extract one repository-wide spec-axis family.\n"
                "# Make downstream wrappers, benchmarks, or adapters reference that authority instead of restating identity/executable pairs."
            ),
        )


class ParallelRegistryProjectionFamilyDetector(CandidateFindingDetector):
    detector_id = "parallel_registry_projection_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Parallel registry projection builders should collapse into one family spec",
        why=(
            "The docs say one semantic family should have one authoritative owner. When several functions differ only in "
            "which registry authority feeds which target constructor, the projection-axis mapping should become one declared "
            "spec or family authority instead of several hand-wired wrappers."
        ),
        capability_gap="single authoritative registry-projection family",
        relation_context="same registry-authority-to-target projection shape repeated across sibling functions",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _parallel_registry_projection_family_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        catalog_candidate = cast(ParallelRegistryProjectionFamilyCandidate, candidate)
        function_names = ", ".join(
            function.qualname for function in catalog_candidate.functions[:4]
        )
        extractor_bases = ", ".join(
            function.extractor_base_name for function in catalog_candidate.functions[:4]
        )
        catalog_types = ", ".join(
            function.catalog_type_name for function in catalog_candidate.functions[:4]
        )
        evidence = tuple(function.evidence for function in catalog_candidate.functions[:6])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Functions {function_names} each build {catalog_types} through "
                f"`{catalog_candidate.collector_name}(structure, ExtractorBase.{catalog_candidate.registry_accessor_name}())` "
                f"over parallel extractor bases {extractor_bases}."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class RegistryProjectionSpec:\n"
                "    registry_authority: type\n"
                "    target_type: type\n"
                "# One helper should own the registry-authority to target mapping."
            ),
            codemod_patch=(
                "# Extract one registry-projection family spec and one authoritative projection builder.\n"
                "# Make per-axis public helpers delegate to that authority instead of reconstructing collector(...registry_accessor())."
            ),
        )


class RepeatedKeyedFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "repeated_keyed_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Repeated keyed family scaffolding should collapse into one typed metaclass-registry base",
        why=(
            "The docs encourage aggressive metaprogramming when several nominal families repeat the same "
            "class-level registration and lookup shell. When many roots restate `registry_key_attr`, "
            "`_registry`, and `for_*` lookup methods, the family algorithm should live in one typed "
            "`metaclass-registry` base."
        ),
        capability_gap="single typed metaclass-registry substrate for keyed nominal registries",
        relation_context="same keyed family registration and lookup shell repeated across nominal family roots",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _repeated_keyed_family_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(RepeatedKeyedFamilyCandidate, candidate)
        class_names = ", ".join(
            root.class_name for root in family_candidate.roots[:8]
        )
        lookup_names = ", ".join(
            sorted({root.lookup_method_name for root in family_candidate.roots[:8]})
        )
        registry_keys = ", ".join(
            sorted({root.registry_key_attr_name for root in family_candidate.roots[:8]})
        )
        evidence = tuple(root.evidence for root in family_candidate.roots[:8])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Registry roots {class_names} each repeat `{registry_keys}` + `_registry` + "
                f"`{lookup_names}` over `{family_candidate.family_base_name}`."
            ),
            evidence,
            scaffold=(
                "from metaclass_registry import AutoRegisterMeta\n\n"
                "KeyT = TypeVar(\"KeyT\")\n\n"
                "class KeyedNominalFamily(ABC, Generic[KeyT], metaclass=AutoRegisterMeta):\n"
                "    __registry_key__ = \"registry_key\"\n"
                "    __skip_if_no_key__ = True\n"
                "    registry_key: ClassVar[KeyT | None] = None\n"
                "    family_label: ClassVar[str] = \"family\"\n"
                "    @classmethod\n"
                "    def for_key(cls, key: KeyT):\n"
                "        try:\n"
                "            return cls.__registry__[key]\n"
                "        except KeyError as error:\n"
                "            raise ValueError(f\"Unknown {cls.family_label}: {key}\") from error"
            ),
            codemod_patch=(
                "# Extract one typed metaclass-registry base that owns registration lookup, duplicate handling, and error shaping.\n"
                "# Leave only declarative key attributes and irreducible hook methods on each family root, and read the registered classes from `cls.__registry__`."
            ),
        )


class ManualKeyedRecordTableDetector(CandidateFindingDetector):
    detector_id = "manual_keyed_record_table"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Manual keyed record tables should collapse into one authoritative spec table",
        why=(
            "When several frozen record classes repeat `_registry`, `register`, and `for_*` lookup around closed keys, "
            "the code is hand-maintaining multiple writable tables. The docs prefer one authoritative spec tuple or "
            "generic keyed-record table with derived indexes."
        ),
        capability_gap="single authoritative keyed-record table or derived index",
        relation_context="same manual record registration and keyed lookup shell repeated across data classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _manual_keyed_record_table_group_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group_candidate = cast(ManualKeyedRecordTableGroupCandidate, candidate)
        class_names = ", ".join(
            item.class_name for item in group_candidate.classes[:6]
        )
        key_fields = ", ".join(
            sorted({item.key_field_name for item in group_candidate.classes[:6]})
        )
        lookup_names = ", ".join(
            sorted({item.lookup_method_name for item in group_candidate.classes[:6]})
        )
        evidence = tuple(item.evidence for item in group_candidate.classes[:6])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Record tables {class_names} each repeat `_registry`, `{group_candidate.classes[0].register_method_name}`, "
                f"and `{lookup_names}` around key fields {key_fields}."
            ),
            evidence,
            scaffold=(
                "KeyT = TypeVar(\"KeyT\")\n"
                "RecordT = TypeVar(\"RecordT\")\n\n"
                "@dataclass(frozen=True)\n"
                "class KeyedRecordTable(Generic[KeyT, RecordT]):\n"
                "    records: tuple[RecordT, ...]\n"
                "    key_of: Callable[[RecordT], KeyT]\n\n"
                "    def by_key(self) -> dict[KeyT, RecordT]:\n"
                "        return {self.key_of(record): record for record in self.records}"
            ),
            codemod_patch=(
                "# Replace per-class mutable `_registry` + `register` shells with one authoritative tuple of record specs.\n"
                "# Derive the keyed lookup dict once, or factor the pattern into a generic keyed-record table helper."
            ),
        )


class ManualStructuralRecordMechanicsDetector(CandidateFindingDetector):
    detector_id = "manual_structural_record_mechanics"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated structural record mechanics should derive from field metadata",
        why=(
            "When several frozen dataclass records hand-write validation, tuple-style field projection, "
            "round-trip reconstruction, and fieldwise transform logic, those mechanics have become a second "
            "authority beside the field declarations. The docs prefer one metadata-driven record substrate "
            "that derives those mechanics from typed fields."
        ),
        capability_gap="single typed structural-record substrate with derived validation, projection, and transform mechanics",
        relation_context="same dataclass record lifecycle mechanics repeated across sibling structural record classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
            CapabilityTag.TYPE_LINEAGE,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.BUILDER_CALL,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _manual_structural_record_mechanics_group_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group_candidate = cast(ManualStructuralRecordMechanicsGroupCandidate, candidate)
        class_names = ", ".join(
            item.class_name for item in group_candidate.classes[:6]
        )
        shared_methods = ", ".join(group_candidate.shared_method_names)
        transform_methods = ", ".join(group_candidate.transform_method_names[:6])
        base_names = ", ".join(group_candidate.base_names)
        evidence = tuple(item.evidence for item in group_candidate.classes[:6])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Dataclass records {class_names} each hand-roll `{shared_methods}` plus fieldwise transforms "
                f"{transform_methods} on top of base family `{base_names}`."
            ),
            evidence,
            scaffold=(
                "@dataclass_transform(field_specifiers=(field, record_field))\n"
                "class StructuralRecordBase:\n"
                "    def validate(self): ...\n"
                "    def project_fields(self): ...\n"
                "    @classmethod\n"
                "    def from_projected(cls, projected, metadata): ...\n"
                "    def transformed(self, **changes): ...\n"
            ),
            codemod_patch=(
                "# Move validation constraints, projected-field partitions, and transform semantics into typed field metadata.\n"
                "# Derive projection, round-trip reconstruction, and fieldwise transforms from one structural-record base instead of re-encoding them per class."
            ),
        )


class RepeatedConcreteTypeCaseAnalysisDetector(CrossModuleCandidateDetector):
    detector_id = "repeated_concrete_type_case_analysis"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_INTERFACE_WITNESS,
        title="Repeated concrete-type recovery should become nominal family behavior",
        why=(
            "When several functions repeatedly recover the same semantic family through concrete `isinstance` "
            "checks on one carried attribute, the family boundary is still latent. The docs want one nominal "
            "ABC and concrete leaf behavior exposed through typed properties or hooks instead of repeated leaf decoding."
        ),
        capability_gap="single ABC-backed family for the carried subject, with repeated case recovery moved into nominal properties or hooks",
        relation_context="same attribute-carried family is re-decoded through repeated concrete runtime type checks across several functions",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _repeated_concrete_type_case_analysis_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        case_candidate = cast(RepeatedConcreteTypeCaseAnalysisCandidate, candidate)
        function_names = ", ".join(
            function.function_name for function in case_candidate.functions[:6]
        )
        class_names = ", ".join(case_candidate.concrete_class_names[:6])
        alias_summary = (
            f" Union alias(es): {', '.join(case_candidate.union_alias_names)}."
            if case_candidate.union_alias_names
            else ""
        )
        existing_base_summary = (
            f" Existing abstract base(s): {', '.join(case_candidate.abstract_base_names)}."
            if case_candidate.abstract_base_names
            else ""
        )
        suggested_family_name = _camel_case(case_candidate.subject_role)
        shared_suffix = _longest_common_suffix(case_candidate.concrete_class_names)
        if (
            shared_suffix
            and len(shared_suffix) >= 6
            and not suggested_family_name.endswith(shared_suffix)
        ):
            suggested_family_name = f"{suggested_family_name}{shared_suffix}"
        elif not suggested_family_name.endswith(("Family", "Witness", "Variant")):
            suggested_family_name = f"{suggested_family_name}Family"
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Functions {function_names} repeatedly recover `{case_candidate.subject_role}` across concrete classes {class_names}.{alias_summary}{existing_base_summary}"
            ),
            case_candidate.evidence,
            scaffold=(
                f"class {suggested_family_name}(ABC):\n"
                "    @property\n"
                "    @abstractmethod\n"
                "    def case_label(self) -> str: ...\n\n"
                "    def explain_case(self, context):\n"
                "        return None\n"
            ),
            codemod_patch=(
                f"# Type `{case_candidate.subject_role}` against one nominal ABC family instead of a concrete union surface.\n"
                "# Move repeated concrete `isinstance` recovery into abstract properties or case hooks on that family.\n"
                "# Keep only irreducible case-local residue in the concrete subclasses."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(case_candidate.functions),
                dispatch_axis=case_candidate.subject_role,
                literal_cases=case_candidate.concrete_class_names,
            ),
        )


class ImplicitSelfContractMixinDetector(CrossModuleCandidateDetector):
    detector_id = "implicit_self_contract_mixin"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Concrete mixins should not hide consumer contracts behind `self`-casts",
        why=(
            "The docs reserve mixins for orthogonal reusable concerns that participate in nominal MRO cleanly. "
            "When a concrete mixin erases `self` through `cast(..., self)` to reach consumer-owned fields, the "
            "mixin is carrying non-orthogonal family logic through a hidden contract instead of a declared base or policy."
        ),
        capability_gap="declared nominal base or policy row for the shared algorithm instead of a hidden mixin self-contract",
        relation_context="concrete mixin methods erase `self` through casts and depend on consumer-owned attributes across several subclasses",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REPEATED_METHOD_ROLES,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _implicit_self_contract_mixin_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        mixin_candidate = cast(ImplicitSelfContractMixinCandidate, candidate)
        methods = ", ".join(mixin_candidate.method_names)
        consumers = ", ".join(mixin_candidate.consumer_class_names[:6])
        accessed_attributes = ", ".join(mixin_candidate.accessed_attribute_names[:6])
        cast_types = ", ".join(mixin_candidate.cast_type_names[:6])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{mixin_candidate.mixin_name}` uses `cast(..., self)` ({cast_types}) in `{methods}` to reach consumer-owned attributes ({accessed_attributes}) across subclasses {consumers}."
            ),
            mixin_candidate.evidence,
            scaffold=(
                "class FamilyBase(ABC):\n"
                "    def run_shared_step(self): ...\n\n"
                "class CasePolicy(ABC):\n"
                "    def run(self, request): ...\n"
            ),
            codemod_patch=(
                f"# `{mixin_candidate.mixin_name}` is not an orthogonal mixin; it hides a consumer contract behind `cast(..., self)`.\n"
                "# Move the shared behavior to a declared nominal base or a keyed policy/spec family, and leave only true orthogonal residue in mixins."
            ),
            metrics=HierarchyCandidateMetrics(
                duplicate_group_count=len(mixin_candidate.method_names),
                class_count=len(mixin_candidate.consumer_class_names) + 1,
            ),
        )


class RepeatedGuardValidatorFamilyDetector(CandidateFindingDetector):
    detector_id = "repeated_guard_validator_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated guard validators should collapse into one case-policy authority",
        why=(
            "When several sibling boolean helpers walk the same subject through fail-fast guards and case-local final "
            "checks, the algorithm skeleton is split across helper names instead of being owned by one nominal case "
            "policy or declarative rule family."
        ),
        capability_gap="single authoritative case-policy or rule-table validator",
        relation_context="same subject and subordinate view validated through repeated fail-fast sibling helpers",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.PARTIAL_VIEW,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _repeated_guard_validator_family_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(RepeatedGuardValidatorFamilyCandidate, candidate)
        function_names = ", ".join(
            function.function_name for function in family_candidate.functions[:6]
        )
        shared_attrs = ", ".join(family_candidate.shared_attr_names[:6])
        alias_summary = (
            f" through `{family_candidate.alias_source_attr}`"
            if family_candidate.alias_source_attr is not None
            else ""
        )
        shared_helpers = ", ".join(family_candidate.shared_helper_call_names[:3])
        helper_summary = (
            f" Shared helper calls: {shared_helpers}."
            if shared_helpers
            else ""
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Boolean validators {function_names} each guard `{family_candidate.subject_param_name}`{alias_summary} "
                f"with the same fail-fast attribute checks over {shared_attrs}.{helper_summary}"
            ),
            family_candidate.evidence,
            scaffold=(
                "class ValidationCasePolicy(ABC):\n"
                "    def validation_error(self, subject):\n"
                "        child = self._subject_child(subject)\n"
                "        if not self._shared_preconditions(subject, child):\n"
                "            return self._shared_failure_message()\n"
                "        return self._case_specific_error(subject, child)\n\n"
                "    @abstractmethod\n"
                "    def _case_specific_error(self, subject, child): ..."
            ),
            codemod_patch=(
                "# Collapse these sibling boolean helpers into one authoritative case-policy family or one declarative rule table.\n"
                "# Keep shared fail-fast guards in one concrete validator method, and leave only case-specific predicates or handle sets per case."
            ),
        )


class RepeatedValidateShapeGuardFamilyDetector(IssueDetector):
    detector_id = "repeated_validate_shape_guard_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated validate() shape guards should collapse into one validated-record authority",
        why=(
            "Sibling nominal records repeat the same fail-fast shape and dimensional guards in `validate()` while "
            "differing only in field names or a small residue check. The docs treat that as duplicated contract "
            "authority that should move into one shared validated-record base, field-spec table, or mixin hook."
        ),
        capability_gap="single authoritative validated-record contract for repeated shape/ndim guards",
        relation_context="same nominal record family repeats fail-loud shape validation scaffolding",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.METHOD_ROLE,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        return [
            self._finding_for_candidate(candidate)
            for candidate in _repeated_validate_shape_guard_candidates_for_modules(
                modules, config
            )
        ]

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(RepeatedValidateShapeGuardFamilyCandidate, candidate)
        method_symbols = tuple(method.symbol for method in family_candidate.methods)
        method_summary = ", ".join(method_symbols[:6])
        shared_guard_count = len(family_candidate.shared_shape_guard_signatures)
        shared_guard_preview = ", ".join(
            family_candidate.shared_shape_guard_signatures[:3]
        )
        preview_suffix = (
            f" Shared normalized guards include {shared_guard_preview}."
            if shared_guard_preview
            else ""
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Validate methods {method_summary} repeat {shared_guard_count} shared shape/ndim guard forms."
            ),
            family_candidate.evidence,
            scaffold=(
                "class ShapeValidatedRecord(ABC):\n"
                "    def validate(self):\n"
                "        for predicate, message in self._shape_guard_rules():\n"
                "            if predicate(self):\n"
                "                raise ValueError(message)\n"
                "        self._validate_residue()\n\n"
                "    @classmethod\n"
                "    @abstractmethod\n"
                "    def _shape_guard_rules(cls): ...\n\n"
                "    def _validate_residue(self):\n"
                "        return None"
                f"{preview_suffix}"
            ),
            codemod_patch=(
                "# Collapse repeated `validate()` shape guards into one authoritative validated-record base or field-spec table.\n"
                "# Keep only the truly variable residue checks, messages, or field roster on each concrete record."
            ),
        )


class RepeatedResultAssemblyPipelineDetector(CandidateFindingDetector):
    detector_id = "repeated_result_assembly_pipeline"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated result-assembly pipeline should collapse into one authoritative assembler",
        why=(
            "Several owners repeat the same downstream result-assembly stages and differ only in the "
            "upstream source or projection that feeds the pipeline. The docs treat that as shared "
            "algorithm authority that should move into one template method or authoritative helper with "
            "one orthogonal source hook."
        ),
        capability_gap="single authoritative result-assembly pipeline with one source hook",
        relation_context="same staged assembly tail is repeated across sibling functions or methods",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _repeated_result_assembly_pipeline_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        pipeline_candidate = cast(RepeatedResultAssemblyPipelineCandidate, candidate)
        function_names = ", ".join(
            function.qualname for function in pipeline_candidate.functions[:4]
        )
        stage_names = ", ".join(
            stage.callee_name for stage in pipeline_candidate.shared_tail
        )
        evidence = tuple(
            function.evidence for function in pipeline_candidate.functions[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Functions {function_names} share the same result-assembly tail "
                f"{stage_names} and differ only in their leading source stages."
            ),
            evidence,
            scaffold=(
                "class ResultAssembler(ABC):\n"
                "    @abstractmethod\n"
                "    def supply_inputs(self, request): ...\n\n"
                "    def assemble(self, request):\n"
                "        supplied = self.supply_inputs(request)\n"
                "        # run the shared downstream assembly stages here\n"
                "        return result"
            ),
            codemod_patch=(
                "# Extract the shared assignment/return tail into one authoritative helper.\n"
                "# Leave only the source-supplier stage variant-specific."
            ),
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(pipeline_candidate.functions),
                statement_count=len(pipeline_candidate.shared_tail),
                class_count=len(
                    {
                        function.qualname.split(".", 1)[0]
                        for function in pipeline_candidate.functions
                        if "." in function.qualname
                    }
                    or {pipeline_candidate.functions[0].qualname}
                ),
                method_symbols=tuple(
                    function.qualname for function in pipeline_candidate.functions
                ),
                shared_statement_texts=tuple(
                    stage.callee_name for stage in pipeline_candidate.shared_tail
                ),
            ),
        )


class NestedBuilderShellDetector(CandidateFindingDetector):
    detector_id = "nested_builder_shell"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_CONTEXT,
        title="Nested builder shell should collapse into one authoritative request boundary",
        why=(
            "A builder forwards a substantial semantic parameter family unchanged into a subordinate "
            "nominal builder and only adds a small residue locally. The docs treat that as split request "
            "authority: one layer should own the forwarded family instead of rebuilding it inside another shell."
        ),
        capability_gap="single authoritative request/context builder boundary",
        relation_context="one builder nests a forwarded subordinate request builder inside a second nominal shell",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _nested_builder_shell_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        shell_candidate = cast(NestedBuilderShellCandidate, candidate)
        forwarded = ", ".join(shell_candidate.forwarded_parameter_names)
        residue_fields = ", ".join(shell_candidate.residue_field_names)
        residue_sources = ", ".join(shell_candidate.residue_source_names)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{shell_candidate.qualname}` forwards `{forwarded}` into "
                f"`{shell_candidate.nested_callee_name}` under `{shell_candidate.nested_field_name}` "
                f"while separately deriving `{residue_fields}` from `{residue_sources}`."
            ),
            (shell_candidate.evidence,),
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class OuterRequest:\n"
                "    child_request: ChildRequest\n\n"
                "    @classmethod\n"
                "    def from_source(cls, source, *, child_request: ChildRequest):\n"
                "        return cls(child_request=child_request, ...)\n"
            ),
            codemod_patch=(
                f"# Stop rebuilding `{shell_candidate.nested_callee_name}` inside `{shell_candidate.qualname}`.\n"
                "# Accept the subordinate request/context directly, or move both layers into one authoritative builder."
            ),
            metrics=ParameterThreadMetrics(
                function_count=1,
                shared_parameter_count=len(shell_candidate.forwarded_parameter_names),
                shared_parameter_names=shell_candidate.forwarded_parameter_names,
            ),
        )


class ManualFiberTagDetector(CandidateFindingDetector):
    detector_id = "manual_fiber_tag"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Manual fiber tag should become nominal family",
        why=(
            "A string-valued instance tag is manually selecting behavior while the same instance still carries fields from several incompatible fibers. "
            "That leaves the family above the zero-incoherence threshold and admits disagreement states the host type system could rule out."
        ),
        capability_gap="host-native nominal fiber decomposition with one subclass per behavior fiber",
        relation_context="manual instance tag drives behavior while irrelevant coordinates remain constructible on every fiber",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _manual_fiber_tag_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        fiber_candidate = cast(ManualFiberTagCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{fiber_candidate.class_name}` branches on manual fiber tag `self.{fiber_candidate.tag_name}` across {fiber_candidate.case_names} while still carrying cross-fiber fields {fiber_candidate.assigned_field_names}."
            ),
            (
                SourceLocation(
                    fiber_candidate.file_path,
                    fiber_candidate.init_line,
                    f"{fiber_candidate.class_name}.__init__",
                ),
                SourceLocation(
                    fiber_candidate.file_path,
                    fiber_candidate.method_line,
                    f"{fiber_candidate.class_name}.{fiber_candidate.method_name}",
                ),
            ),
            scaffold=_manual_fiber_tag_scaffold(fiber_candidate),
            codemod_patch=_manual_fiber_tag_patch(fiber_candidate),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(fiber_candidate.case_names),
                dispatch_axis=f"self.{fiber_candidate.tag_name}",
                literal_cases=fiber_candidate.case_names,
            ),
        )


class DescriptorDerivedViewDetector(CandidateFindingDetector):
    detector_id = "descriptor_derived_view"
    finding_spec = FindingSpec(
        pattern_id=PatternId.DESCRIPTOR_DERIVED_VIEW,
        title="Derived views stored independently of their source",
        why=(
            "Several stored fields are derived from one authoritative source field, but mutators resynchronize them manually and incompletely. "
            "That raises the degree of freedom above one and makes view disagreement reachable."
        ),
        capability_gap="descriptor- or property-mediated derived views rooted in one authoritative source",
        relation_context="stored derived views must be manually kept coherent with a single source field",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _descriptor_derived_view_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        view_candidate = cast(DescriptorDerivedViewCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{view_candidate.class_name}` stores derived views {view_candidate.derived_field_names} from `{view_candidate.source_attr}`, but `{view_candidate.mutator_name}` only updates {view_candidate.updated_field_names}."
            ),
            (
                SourceLocation(
                    view_candidate.file_path,
                    view_candidate.init_line,
                    f"{view_candidate.class_name}.__init__",
                ),
                SourceLocation(
                    view_candidate.file_path,
                    view_candidate.mutator_line,
                    f"{view_candidate.class_name}.{view_candidate.mutator_name}",
                ),
            ),
            scaffold=_descriptor_derived_view_scaffold(view_candidate),
            codemod_patch=_descriptor_derived_view_patch(view_candidate),
        )


class DeferredClassRegistrationDetector(CandidateFindingDetector):
    detector_id = "deferred_class_registration"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Class registration is decoupled from class existence",
        why=(
            "Manual decorator- or helper-based registration leaves a reachable state where a class exists but the registry has not been updated. "
            "The host already provides zero-delay registration via `metaclass-registry` or another class-time hook."
        ),
        capability_gap="zero-delay metaclass-registry class registration with collision checks and runtime provenance",
        relation_context="class registration is performed as a separate auxiliary step rather than at class creation time",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _manual_registry_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        registry_candidate = cast(ManualRegistryCandidate, candidate)
        evidence = [
            SourceLocation(
                registry_candidate.file_path,
                registry_candidate.line,
                registry_candidate.decorator_name,
            ),
        ]
        evidence.extend(
            SourceLocation(
                registry_candidate.file_path,
                registry_candidate.line,
                class_name,
            )
            for class_name in registry_candidate.class_names[:5]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Registry `{registry_candidate.registry_name}` is updated through manual decorator `{registry_candidate.decorator_name}` for classes {registry_candidate.class_names}, leaving registration structurally decoupled from class creation."
            ),
            tuple(evidence),
            scaffold=_manual_registry_scaffold(registry_candidate),
            codemod_patch=_manual_registry_patch(registry_candidate),
            metrics=RegistrationMetrics(
                registration_site_count=len(registry_candidate.class_names),
                registry_name=registry_candidate.registry_name,
            ),
        )


class StructuralConfusabilityDetector(CandidateFindingDetector):
    detector_id = "structural_confusability"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_INTERFACE_WITNESS,
        title="Consumer observes a confusable duck-typed family",
        why=(
            "A consumer only observes a partial structural view, and several unrelated classes are confusable under that view. "
            "Without a nominal witness, the distortion floor stays above zero and the family boundary remains implicit."
        ),
        capability_gap="ABC-backed nominal witness for a structurally confusable implementation family",
        relation_context="consumer depends on a partial structural view shared by several unrelated classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _structural_confusability_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        confusability_candidate = cast(StructuralConfusabilityCandidate, candidate)
        evidence = (
            SourceLocation(
                confusability_candidate.file_path,
                confusability_candidate.line,
                confusability_candidate.function_name,
            ),
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{confusability_candidate.function_name}` observes `{confusability_candidate.parameter_name}` only through methods {confusability_candidate.observed_method_names}, but classes {confusability_candidate.class_names} are confusable under that view."
            ),
            evidence,
            scaffold=_structural_confusability_scaffold(confusability_candidate),
            codemod_patch=_structural_confusability_patch(confusability_candidate),
        )


class SemanticWitnessFamilyDetector(CandidateFindingDetector):
    detector_id = "semantic_witness_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_WITNESS_CARRIER,
        title="Semantic carrier family should share one nominal base",
        why=(
            "Several frozen dataclass carriers repeat the same location and naming roles under different field names. "
            "That leaves one semantic family structurally expanded instead of giving it one nominal carrier root."
        ),
        capability_gap="one authoritative nominal base for a semantic metadata carrier family",
        relation_context="same carrier family repeats a renamed semantic-role spine across sibling frozen dataclasses",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _witness_carrier_family_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        witness_candidate = cast(WitnessCarrierFamilyCandidate, candidate)
        evidence = tuple(
            SourceLocation(witness_candidate.file_path, line, class_name)
            for class_name, line in zip(
                witness_candidate.class_names,
                witness_candidate.line_numbers,
                strict=True,
            )
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Frozen carrier classes {', '.join(witness_candidate.class_names)} repeat semantic roles {witness_candidate.shared_role_names} under renamed fields and should inherit one nominal base carrier."
            ),
            evidence,
            scaffold=_witness_carrier_family_scaffold(witness_candidate),
            codemod_patch=_witness_carrier_family_patch(witness_candidate),
            metrics=WitnessCarrierMetrics(
                class_count=len(witness_candidate.class_names),
                shared_role_count=len(witness_candidate.shared_role_names),
                class_names=witness_candidate.class_names,
                shared_role_names=witness_candidate.shared_role_names,
            ),
        )


def _witness_mixin_enforcement_candidate(
    module: ParsedModule,
) -> WitnessMixinEnforcementCandidate | None:
    all_classes = _witness_carrier_class_candidates(module)
    grouped: dict[str, list[WitnessCarrierClassCandidate]] = defaultdict(list)
    for candidate in all_classes:
        for token in candidate.family_tokens:
            grouped[token].append(candidate)
    classes = max(
        (
            tuple(sorted(items, key=lambda item: (item.line, item.class_name)))
            for items in grouped.values()
            if len(items) >= 3
        ),
        key=len,
        default=(),
    )
    if len(classes) < 2:
        return None
    role_to_classes: dict[str, dict[str, WitnessCarrierClassCandidate]] = defaultdict(
        dict
    )
    role_to_fields: dict[str, set[str]] = defaultdict(set)
    line_by_class: dict[str, int] = {}
    for candidate in classes:
        line_by_class[candidate.class_name] = candidate.line
        for role_name, field_names in candidate.normalized_role_fields:
            if role_name not in _WITNESS_MIXIN_ROLE_NAMES:
                continue
            role_to_classes[role_name][candidate.class_name] = candidate
            role_to_fields[role_name].update(field_names)
    role_field_names = tuple(
        (role_name, tuple(sorted(role_to_fields[role_name])))
        for role_name in _WITNESS_MIXIN_ROLE_NAMES
        if len(role_to_classes[role_name]) >= 2 and len(role_to_fields[role_name]) >= 2
    )
    if not role_field_names:
        return None
    class_names = tuple(
        sorted(
            {
                class_name
                for role_name, _ in role_field_names
                for class_name in role_to_classes[role_name]
            }
        )
    )
    return WitnessMixinEnforcementCandidate(
        file_path=str(module.path),
        class_names=class_names,
        line_numbers=tuple(line_by_class[class_name] for class_name in class_names),
        role_field_names=role_field_names,
    )


class MixinEnforcementDetector(PerModuleIssueDetector):
    detector_id = "mixin_enforcement"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_WITNESS_CARRIER,
        title="Renamed orthogonal semantic slices should become mixins",
        why=(
            "Several carrier classes repeat the same semantic slice under renamed fields such as `line` vs `method_line` or `name_family` vs `class_names`. "
            "One shared base is not enough when those slices are orthogonal; the architecture wants reusable mixins composed through multiple inheritance."
        ),
        capability_gap="one authoritative semantic carrier spine plus reusable semantic-role mixins",
        relation_context="same carrier family repeats renamed semantic slices that overlap orthogonally across sibling carriers",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidate = _witness_mixin_enforcement_candidate(module)
        if candidate is None:
            return []
        evidence = tuple(
            SourceLocation(candidate.file_path, line, class_name)
            for class_name, line in zip(
                candidate.class_names, candidate.line_numbers, strict=True
            )
        )
        role_summary = "; ".join(
            f"{role_name} via {field_names}"
            for role_name, field_names in candidate.role_field_names
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Carrier classes {', '.join(candidate.class_names)} repeat renamed semantic slices {role_summary}; enforce reusable mixins and compose them through multiple inheritance."
                ),
                evidence,
                scaffold=_witness_mixin_enforcement_scaffold(candidate),
                codemod_patch=_witness_mixin_enforcement_patch(candidate),
                metrics=WitnessCarrierMetrics(
                    class_count=len(candidate.class_names),
                    shared_role_count=len(candidate.role_field_names),
                    class_names=candidate.class_names,
                    shared_role_names=tuple(
                        role_name for role_name, _ in candidate.role_field_names
                    ),
                ),
            )
        ]


def _shared_field_type_map(
    observations: tuple[FieldObservation, ...], field_names: tuple[str, ...]
) -> tuple[tuple[str, str], ...] | None:
    typed_fields: list[tuple[str, str]] = []
    for field_name in field_names:
        annotations = {
            (item.annotation_fingerprint, item.annotation_text)
            for item in observations
            if item.field_name == field_name and item.annotation_fingerprint is not None
        }
        if len({fingerprint for fingerprint, _ in annotations}) > 1:
            return None
        if annotations:
            _, annotation_text = next(iter(annotations))
            if annotation_text is not None:
                typed_fields.append((field_name, annotation_text))
    return tuple(typed_fields)


def _field_family_candidates(module: ParsedModule) -> tuple[FieldFamilyCandidate, ...]:
    observations: tuple[FieldObservation, ...] = _collect_typed_family_items(
        module, FieldObservationFamily, FieldObservation
    )
    graph = ObservationGraph(
        observations=tuple(item.structural_observation for item in observations)
    )
    candidates: list[FieldFamilyCandidate] = []
    for execution_level in (
        StructuralExecutionLevel.CLASS_BODY,
        StructuralExecutionLevel.INIT_BODY,
    ):
        grouped_by_level = {
            group.nominal_witness: set(group.observed_names)
            for group in graph.witness_groups_for(
                ObservationKind.FIELD, execution_level
            )
        }
        for cohort in graph.coherence_cohorts_for(
            ObservationKind.FIELD,
            execution_level,
            minimum_witnesses=2,
            minimum_fibers=2,
        ):
            field_names = tuple(sorted(cohort.observed_names))
            supporting_classes = cohort.nominal_witnesses
            shared_field_set = set(field_names)
            if any(
                len(shared_field_set) / max(len(grouped_by_level[class_name]), 1) < 0.5
                for class_name in supporting_classes
            ):
                continue
            if any(
                not (grouped_by_level[class_name] - shared_field_set)
                for class_name in supporting_classes
            ):
                continue
            supporting_observations: tuple[FieldObservation, ...] = tuple(
                sorted(
                    (
                        item
                        for item in observations
                        if item.execution_level == execution_level
                        and item.class_name in supporting_classes
                        and item.field_name in field_names
                    ),
                    key=lambda item: (item.file_path, item.lineno, item.symbol),
                )
            )
            field_type_map = _shared_field_type_map(
                supporting_observations,
                field_names,
            )
            if field_type_map is None:
                continue
            candidates.append(
                FieldFamilyCandidate(
                    class_names=supporting_classes,
                    field_names=field_names,
                    execution_level=execution_level,
                    observations=supporting_observations,
                    dataclass_count=sum(
                        1
                        for class_name in supporting_classes
                        if any(
                            item.class_name == class_name and item.is_dataclass_family
                            for item in supporting_observations
                        )
                    ),
                    field_type_map=field_type_map,
                )
            )

    maximal_candidates: list[FieldFamilyCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.execution_level,
            len(item.class_names),
            len(item.field_names),
        ),
        reverse=True,
    ):
        if any(
            candidate.execution_level == other.execution_level
            and set(candidate.class_names) == set(other.class_names)
            and set(candidate.field_names) < set(other.field_names)
            for other in maximal_candidates
        ):
            continue
        maximal_candidates.append(candidate)
    return tuple(
        sorted(
            maximal_candidates,
            key=lambda item: (
                item.execution_level,
                item.class_names,
                item.field_names,
            ),
        )
    )


def _field_family_scaffold(candidate: FieldFamilyCandidate) -> str:
    base_name = _shared_field_base_name(candidate.class_names)
    field_type_lookup = dict(candidate.field_type_map)
    field_block = "\n".join(
        f"    {field}: {field_type_lookup.get(field, 'object')}"
        for field in candidate.field_names
    )
    if candidate.dataclass_count == len(candidate.class_names):
        return (
            "@dataclass(frozen=True)\n"
            f"class {base_name}(ABC):\n"
            f"{field_block}\n\n"
            f"# Move shared dataclass fields from {', '.join(candidate.class_names)} into {base_name}."
        )
    init_params = ", ".join(candidate.field_names)
    assignments = "\n".join(
        f"        self.{field} = {field}" for field in candidate.field_names
    )
    return (
        f"class {base_name}(ABC):\n"
        f"    def __init__(self, {init_params}):\n"
        f"{assignments}\n\n"
        f"# Move shared fields from {', '.join(candidate.class_names)} at {candidate.execution_level} into {base_name}."
    )


_PYTREE_TRANSPORT_METHOD_NAMES = frozenset(
    {
        "_tree_children",
        "_tree_aux_data",
        "tree_flatten",
        "tree_unflatten",
    }
)


def _role_member_name(tokens: tuple[str, ...]) -> str:
    return "_".join(tokens)


def _is_numeric_role_member_name(name: str) -> bool:
    return all(token.isdigit() for token in name.split("_"))


def _prefixed_role_field_groups(
    observations: tuple[FieldObservation, ...],
    *,
    prefix_token_count: int,
) -> dict[str, dict[str, FieldObservation]]:
    groups: dict[str, dict[str, FieldObservation]] = defaultdict(dict)
    for observation in observations:
        tokens = _ordered_class_name_tokens(observation.field_name)
        if len(tokens) <= prefix_token_count:
            continue
        role_name = _role_member_name(tokens[:prefix_token_count])
        member_name = _role_member_name(tokens[prefix_token_count:])
        if not role_name or not member_name:
            continue
        groups[role_name].setdefault(member_name, observation)
    return groups


def _class_pytree_base_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        base_name
        for base_name in _declared_base_names(node)
        if "pytree" in base_name.lower()
    )


def _class_manual_transport_methods(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(sorted(_method_names(node) & _PYTREE_TRANSPORT_METHOD_NAMES))


def _connected_role_components(
    role_to_members: dict[str, dict[str, FieldObservation]],
    *,
    min_shared_members: int,
) -> tuple[tuple[str, ...], ...]:
    roles = sorted(role_to_members)
    adjacency: dict[str, set[str]] = {role: set() for role in roles}
    for left_index, left_role in enumerate(roles):
        left_members = set(role_to_members[left_role])
        for right_role in roles[left_index + 1 :]:
            shared_members = left_members & set(role_to_members[right_role])
            if len(shared_members) < min_shared_members:
                continue
            adjacency[left_role].add(right_role)
            adjacency[right_role].add(left_role)

    components: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for role in roles:
        if role in seen or not adjacency[role]:
            continue
        stack = [role]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component))
        seen.update(component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _prefixed_role_bundle_candidate_for_class(
    module: ParsedModule,
    class_node: ast.ClassDef,
    observations: tuple[FieldObservation, ...],
    config: DetectorConfig,
) -> PrefixedRoleFieldBundleCandidate | None:
    if len(observations) < config.min_prefixed_role_shared_fields * 2:
        return None
    manual_transport_methods = _class_manual_transport_methods(class_node)
    pytree_base_names = _class_pytree_base_names(class_node)
    is_dataclass_family = any(item.is_dataclass_family for item in observations)
    if not (is_dataclass_family or manual_transport_methods or pytree_base_names):
        return None

    candidates: list[PrefixedRoleFieldBundleCandidate] = []
    for prefix_token_count in (1, 2):
        role_to_members = _prefixed_role_field_groups(
            observations,
            prefix_token_count=prefix_token_count,
        )
        role_to_members = {
            role: members
            for role, members in role_to_members.items()
            if len(members) >= config.min_prefixed_role_shared_fields
        }
        for role_names in _connected_role_components(
            role_to_members,
            min_shared_members=config.min_prefixed_role_shared_fields,
        ):
            shared_member_names = tuple(
                sorted(
                    member_name
                    for member_name in {
                        member_name
                        for role_name in role_names
                        for member_name in role_to_members[role_name]
                    }
                    if sum(
                        member_name in role_to_members[role_name]
                        for role_name in role_names
                    )
                    >= 2
                )
            )
            if len(shared_member_names) < config.min_prefixed_role_shared_fields:
                continue
            if all(
                _is_numeric_role_member_name(member_name)
                for member_name in shared_member_names
            ):
                continue
            if (
                len(shared_member_names) < config.min_prefixed_role_bundle_fields
                and not (manual_transport_methods or pytree_base_names)
            ):
                continue
            role_field_map = tuple(
                (
                    role_name,
                    tuple(
                        role_to_members[role_name][member_name].field_name
                        for member_name in shared_member_names
                        if member_name in role_to_members[role_name]
                    ),
                )
                for role_name in role_names
            )
            candidate_field_names = {
                field_name
                for _, field_names in role_field_map
                for field_name in field_names
            }
            candidate_observations = tuple(
                sorted(
                    (
                        observation
                        for observation in observations
                        if observation.field_name in candidate_field_names
                    ),
                    key=lambda item: (item.lineno, item.field_name),
                )
            )
            candidates.append(
                PrefixedRoleFieldBundleCandidate(
                    file_path=str(module.path),
                    class_name=class_node.name,
                    line=class_node.lineno,
                    role_names=role_names,
                    shared_member_names=shared_member_names,
                    role_field_map=role_field_map,
                    manual_transport_methods=manual_transport_methods,
                    pytree_base_names=pytree_base_names,
                    is_dataclass_family=is_dataclass_family,
                    observations=candidate_observations,
                )
            )

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            len(item.shared_member_names),
            len(item.role_names),
            sum(len(field_names) for _, field_names in item.role_field_map),
        ),
    )


def _prefixed_role_field_bundle_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[PrefixedRoleFieldBundleCandidate, ...]:
    observations: tuple[FieldObservation, ...] = _collect_typed_family_items(
        module, FieldObservationFamily, FieldObservation
    )
    observations_by_class: dict[str, list[FieldObservation]] = defaultdict(list)
    for observation in observations:
        if observation.execution_level not in {
            StructuralExecutionLevel.CLASS_BODY,
            StructuralExecutionLevel.INIT_BODY,
        }:
            continue
        observations_by_class[observation.class_name].append(observation)

    candidates: list[PrefixedRoleFieldBundleCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        class_observations = tuple(observations_by_class.get(class_node.name, ()))
        candidate = _prefixed_role_bundle_candidate_for_class(
            module,
            class_node,
            class_observations,
            config,
        )
        if candidate is not None:
            candidates.append(candidate)
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.line, item.class_name),
        )
    )


def _prefixed_role_bundle_scaffold(
    candidate: PrefixedRoleFieldBundleCandidate,
) -> str:
    base_name = f"{candidate.class_name}Role"
    member_block = "\n".join(
        f"    {member_name}: object" for member_name in candidate.shared_member_names
    )
    role_classes = "\n\n".join(
        f"@dataclass(frozen=True)\nclass {_public_class_name(role_name)}{base_name}({base_name}):\n    pass"
        for role_name in candidate.role_names
    )
    return (
        "from abc import ABC\n\n"
        "@dataclass(frozen=True)\n"
        f"class {base_name}(ABC):\n"
        f"{member_block}\n\n"
        f"{role_classes}\n\n"
        f"# Replace role-prefixed fields on `{candidate.class_name}` with explicit role records."
    )


def _public_class_name(name: str) -> str:
    return "".join(token.capitalize() for token in _ordered_class_name_tokens(name))


def _longest_common_prefix(values: tuple[str, ...]) -> str:
    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        while prefix and not value.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def _longest_common_suffix(values: tuple[str, ...]) -> str:
    if not values:
        return ""
    reversed_values = tuple(value[::-1] for value in values)
    return _longest_common_prefix(reversed_values)[::-1]


def _shared_field_base_name(class_names: tuple[str, ...]) -> str:
    suffix = _longest_common_suffix(class_names)
    if suffix:
        return suffix if suffix.endswith("Base") else f"{suffix}Base"
    prefix = _longest_common_prefix(class_names)
    if prefix:
        return prefix if prefix.endswith("Base") else f"{prefix}Base"
    return "SharedFieldsBase"


class RepeatedFieldFamilyDetector(CandidateFindingDetector):
    detector_id = "repeated_field_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Shared field family across sibling classes should move to an ABC base",
        why=(
            "The docs treat repeated shared state components the same way as repeated shared algorithms: when the "
            "same field family is declared across sibling classes at the same structural execution level, the shared "
            "component should move to one authoritative base rather than being duplicated in each leaf class."
        ),
        capability_gap="single authoritative state component for a nominal class family",
        relation_context="same field family repeats across sibling classes at one structural execution level",
        confidence=HIGH_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return tuple(
            candidate
            for candidate in _field_family_candidates(module)
            if len(candidate.class_names) >= 2 and len(candidate.field_names) >= 2
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        field_candidate = cast(FieldFamilyCandidate, candidate)
        evidence = tuple(
            SourceLocation(
                item.file_path,
                item.lineno,
                item.symbol,
            )
            for item in field_candidate.observations[:8]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Classes {', '.join(field_candidate.class_names)} repeat fields {field_candidate.field_names} at `{field_candidate.execution_level}`."
            ),
            evidence,
            relation_context=(
                f"same field family repeats across sibling classes at `{field_candidate.execution_level}`"
            ),
            scaffold=_field_family_scaffold(field_candidate),
            metrics=FieldFamilyMetrics(
                class_count=len(field_candidate.class_names),
                field_count=len(field_candidate.field_names),
                class_names=field_candidate.class_names,
                field_names=field_candidate.field_names,
                execution_level=field_candidate.execution_level,
                dataclass_count=field_candidate.dataclass_count,
            ),
        )


class PrefixedRoleFieldBundleDetector(CandidateFindingDetector):
    detector_id = "prefixed_role_field_bundle"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Role-prefixed field bundle should become nominal subrecords",
        why=(
            "A record that repeats the same member family behind role prefixes is encoding nominal role identity "
            "in string-shaped field names. The docs prefer explicit role records or ABC/dataclass side objects so "
            "the schema, PyTree behavior, and type-level role identity have one authoritative boundary."
        ),
        capability_gap="explicit nominal role records instead of parallel role-prefixed fields",
        relation_context="same semantic member family repeats under several leading role prefixes in one record",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _prefixed_role_field_bundle_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        bundle_candidate = cast(PrefixedRoleFieldBundleCandidate, candidate)
        role_summary = ", ".join(bundle_candidate.role_names)
        member_summary = ", ".join(bundle_candidate.shared_member_names)
        transport_summary = ""
        if bundle_candidate.manual_transport_methods:
            transport_summary = (
                " Manual transport methods also repeat the shape: "
                f"{', '.join(bundle_candidate.manual_transport_methods)}."
            )
        elif bundle_candidate.pytree_base_names:
            transport_summary = (
                " The record also participates in PyTree transport via "
                f"{', '.join(bundle_candidate.pytree_base_names)}."
            )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{bundle_candidate.class_name}` repeats role-prefixed fields for roles "
                f"{role_summary} over shared members {member_summary}.{transport_summary}"
            ),
            bundle_candidate.evidence,
            scaffold=_prefixed_role_bundle_scaffold(bundle_candidate),
            codemod_patch=(
                f"# Extract role records for {bundle_candidate.role_names} from `{bundle_candidate.class_name}`.\n"
                f"# Replace prefixed fields {bundle_candidate.field_names} with typed role subrecords and derive PyTree children from those records."
            ),
            metrics=FieldFamilyMetrics(
                class_count=1,
                field_count=len(bundle_candidate.field_names),
                class_names=(bundle_candidate.class_name,),
                field_names=bundle_candidate.field_names,
                execution_level=StructuralExecutionLevel.CLASS_BODY,
                dataclass_count=1 if bundle_candidate.is_dataclass_family else 0,
            ),
        )


class RepeatedPropertyAliasHookDetector(CandidateFindingDetector):
    detector_id = "repeated_property_alias_hooks"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated property hook aliases should move into a shared base or mixin",
        why=(
            "Several subclasses re-declare the same one-line property hook over the same backing attribute. "
            "That is non-orthogonal hook duplication and should live once in a shared base or mixin."
        ),
        capability_gap="single authoritative hook property implementation for a nominal subclass family",
        relation_context="same property hook alias repeats across siblings of one base family",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _property_alias_hook_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        hook_group = cast(PropertyAliasHookGroup, candidate)
        evidence = tuple(
            SourceLocation(
                hook_group.file_path, line, f"{class_name}.{hook_group.property_name}"
            )
            for class_name, line in zip(
                hook_group.class_names,
                hook_group.line_numbers,
                strict=True,
            )
        )
        mixin_name = f"{_camel_case(hook_group.returned_attribute)}{_camel_case(hook_group.property_name)}Mixin"
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Subclasses {', '.join(hook_group.class_names)} of `{hook_group.base_name}` all implement `{hook_group.property_name}` as `return self.{hook_group.returned_attribute}`."
            ),
            evidence,
            scaffold=(
                f"class {mixin_name}(ABC):\n"
                "    @property\n"
                f"    def {hook_group.property_name}(self):\n"
                f"        return self.{hook_group.returned_attribute}"
            ),
            codemod_patch=(
                f"# Move `{hook_group.property_name}` <- `self.{hook_group.returned_attribute}` into one shared mixin or intermediate base for `{hook_group.base_name}`."
            ),
            metrics=_repeated_property_hook_metrics(
                hook_group.class_names, hook_group.property_name
            ),
        )


class ConstantPropertyHookDetector(CandidateFindingDetector):
    detector_id = "constant_property_hooks"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Constant property hooks should move into classvars or fixed mixins",
        why=(
            "Several subclasses implement the same property as a one-line constant return. "
            "That is nominal hook boilerplate and should collapse into one classvar-backed base or one fixed-value mixin."
        ),
        capability_gap="single authoritative constant hook implementation for a nominal subclass family",
        relation_context="same property hook is re-declared as a constant return across one subclass family",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _constant_property_hook_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        hook_group = cast(ConstantPropertyHookGroup, candidate)
        evidence = tuple(
            SourceLocation(
                hook_group.file_path,
                line,
                f"{class_name}.{hook_group.property_name}",
            )
            for class_name, line in zip(
                hook_group.class_names,
                hook_group.line_numbers,
                strict=True,
            )
        )
        unique_returns = tuple(dict.fromkeys(hook_group.return_expressions))
        constant_name = hook_group.property_name.upper()
        if len(unique_returns) == 1:
            scaffold = (
                f"class {_camel_case(unique_returns[0].replace('.', '_'))}{_camel_case(hook_group.property_name)}Mixin(ABC):\n"
                "    @property\n"
                f"    def {hook_group.property_name}(self):\n"
                f"        return {unique_returns[0]}"
            )
            patch = f"# Move `{hook_group.property_name}` <- `{unique_returns[0]}` into one fixed-value mixin for `{hook_group.base_name}`."
        else:
            scaffold = (
                f"class {hook_group.base_name}{_camel_case(hook_group.property_name)}Base(ABC):\n"
                f"    {constant_name}: ClassVar[object]\n\n"
                "    @property\n"
                f"    def {hook_group.property_name}(self):\n"
                f"        return type(self).{constant_name}"
            )
            patch = f"# Replace repeated constant `{hook_group.property_name}` hooks with one classvar-backed base for `{hook_group.base_name}`."
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Subclasses {', '.join(hook_group.class_names)} of `{hook_group.base_name}` all implement `{hook_group.property_name}` as constant returns {unique_returns}."
            ),
            evidence,
            scaffold=scaffold,
            codemod_patch=patch,
            metrics=_repeated_property_hook_metrics(
                hook_group.class_names, hook_group.property_name
            ),
        )


class ReflectiveSelfAttributeEscapeDetector(CandidateFindingDetector):
    detector_id = "reflective_self_attribute_escape"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CONFIG_CONTRACTS,
        title="Reflective self-attribute access hides a nominal contract",
        why=(
            "A class uses reflective self-attribute access with a hardcoded string instead of declaring the field or property on the nominal carrier. "
            "That keeps the contract partial, stringly, and fail-soft."
        ),
        capability_gap="declared fail-loud nominal attribute contract on the carrier family",
        relation_context="class template probes its own required state through reflective string access",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.PARTIAL_VIEW,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _reflective_self_attribute_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        reflective_candidate = cast(ReflectiveSelfAttributeCandidate, candidate)
        carrier_name = f"{reflective_candidate.class_name}Carrier"
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{reflective_candidate.class_name}.{reflective_candidate.method_name}` uses `{reflective_candidate.reflective_builtin}(self, '{reflective_candidate.attribute_name}')` instead of declaring `{reflective_candidate.attribute_name}` on the nominal carrier."
            ),
            (
                SourceLocation(
                    reflective_candidate.file_path,
                    reflective_candidate.line,
                    f"{reflective_candidate.class_name}.{reflective_candidate.method_name}",
                ),
            ),
            scaffold=(
                "@dataclass(frozen=True)\n"
                f"class {carrier_name}(ABC):\n"
                f"    {reflective_candidate.attribute_name}: str"
            ),
            codemod_patch=(
                f"# Delete `{reflective_candidate.reflective_builtin}(self, '{reflective_candidate.attribute_name}')`.\n"
                f"# Declare `{reflective_candidate.attribute_name}` once on the shared nominal carrier or abstract base instead of probing it by string."
            ),
        )


class HelperBackedObservationSpecDetector(PerModuleIssueDetector):
    detector_id = "helper_backed_observation_spec"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Helper-backed wrapper classes should use a declarative substrate",
        why=(
            "Several sibling wrapper classes do nothing except forward one entrypoint to one helper. "
            "That helper metadata should live in classvars on a shared substrate rather than in repeated wrapper methods."
        ),
        capability_gap="one declarative helper-backed wrapper family with class-level registration",
        relation_context="same helper-backed wrapper shape repeats across sibling classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        group = _helper_backed_observation_spec_group(module)
        if group is None:
            return []
        evidence = tuple(
            SourceLocation(group.file_path, line, class_name)
            for class_name, line in zip(
                group.class_names,
                group.line_numbers,
                strict=True,
            )
        )
        helper_names = tuple(dict.fromkeys(group.helper_names))
        wrapper_kinds = tuple(dict.fromkeys(group.wrapper_kinds))
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Classes {', '.join(group.class_names[:6])} under base family {group.base_names} are helper-backed wrappers over {', '.join(helper_names[:6])} via wrapper kinds {', '.join(wrapper_kinds)}."
                ),
                evidence[:8],
                scaffold=(
                    "class HelperBackedTemplate(ABC):\n"
                    "    helper: ClassVar[Callable[..., object | None]]\n\n"
                    "    def build(self, *args, **kwargs):\n"
                    "        return type(self).helper(*args, **kwargs)\n\n"
                    "class TupleResultMixin(ABC):\n"
                    "    @staticmethod\n"
                    "    def wrap_result(value):\n"
                        "        return tuple(value) if value is not None else None"
                ),
                codemod_patch=(
                    "# Collapse helper-backed wrappers into declarative helper classes.\n"
                    "# Put helper identity, result wrapping, and guard policy on classvars/mixins, and let class creation discover the family."
                ),
                metrics=RepeatedMethodMetrics.from_duplicate_family(
                    duplicate_site_count=len(group.class_names),
                    statement_count=1,
                    class_count=len(group.class_names),
                    method_symbols=tuple(
                        f"{class_name}.{method_name}"
                        for class_name, method_name in zip(
                            group.class_names,
                            group.method_names,
                            strict=True,
                        )
                    ),
                ),
            )
        ]


class ClassvarOnlySiblingLeafDetector(CandidateFindingDetector):
    detector_id = "classvar_only_sibling_leaf"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Classvar-only sibling leaves should come from one metaprogrammed family table",
        why=(
            "Several sibling classes differ only by simple classvar declarations. That is class-level boilerplate and should "
            "collapse into one declarative family table plus metaprogrammed class generation or registration."
        ),
        capability_gap="one authoritative declarative family-definition table with class-generation or metaclass support",
        relation_context="same class-level family declaration boilerplate repeats across sibling family leaves",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _classvar_only_sibling_leaf_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group = cast(DeclarativeFamilyBoilerplateGroup, candidate)
        evidence = tuple(
            SourceLocation(group.file_path, line, class_name)
            for class_name, line in zip(
                group.class_names,
                group.line_numbers,
                strict=True,
            )
        )
        spec_name = _camel_case(group.base_names[0]) + "Declaration"
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Family classes {', '.join(group.class_names[:6])} all repeat declarative classvars {group.assigned_names} under bases {group.base_names}."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                f"class {spec_name}:\n"
                "    family_name: str\n"
                "    item_type: type[object]\n"
                "    spec_root: type[object] | None = None\n"
                "    spec: object | None = None\n\n"
                f"def declare_{group.base_names[0].lower()}(spec: {spec_name}) -> type[CollectedFamily]:\n"
                "    return type(spec.family_name, (...,), {...})"
            ),
            codemod_patch=(
                f"# Replace repeated family leaf classes for bases {group.base_names} with one declarative family-definition table.\n"
                "# Generate or register the concrete family classes from that table instead of re-spelling the same classvars in each class."
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(group.class_names),
                class_count=len(group.class_names),
                registry_name=group.base_names[0],
                class_names=group.class_names,
                class_key_pairs=group.assigned_names,
            ),
        )


class TypeIndexedDefinitionBoilerplateDetector(CandidateFindingDetector):
    detector_id = "type_indexed_definition_boilerplate"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Type-indexed family definitions should derive from one typed declaration table",
        why=(
            "Several `*Definition` classes plus `family_type` aliases restate the same type-indexed family metadata. "
            "That metadata should live once in a typed declaration table and definition-time materializer."
        ),
        capability_gap="one authoritative typed declaration table for family generation and export derivation",
        relation_context="same type-indexed family definition and alias boilerplate repeats across sibling declarations",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _type_indexed_definition_boilerplate_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group = cast(TypeIndexedDefinitionBoilerplateGroup, candidate)
        evidence = tuple(
            SourceLocation(group.file_path, line, class_name)
            for class_name, line in zip(
                group.definition_class_names,
                group.line_numbers,
                strict=True,
            )
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Definition classes {', '.join(group.definition_class_names[:6])} plus aliases {', '.join(group.alias_names[:6])} all repeat typed family metadata {group.assigned_names} under bases {group.base_names}."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class FamilyDeclaration(Generic[TItem]):\n"
                "    export_name: str\n"
                "    item_type: type[TItem]\n"
                "    spec_root: type[object] | None = None\n"
                "    spec: object | None = None\n"
                "    literal_kind: object | None = None\n\n"
                "def materialize_family(decl: FamilyDeclaration[object]) -> type[CollectedFamily]:\n"
                "    return AutoRegisterMeta(...)"
            ),
            codemod_patch=(
                f"# Replace repeated definition classes under {group.base_names} with one typed declaration table.\n"
                "# Derive runtime family classes, registry indexes, exported aliases, and `__all__` from the same declarations instead of restating them in classes plus assignments."
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(group.definition_class_names),
                class_count=len(group.definition_class_names),
                registry_name=group.base_names[0],
                class_names=group.definition_class_names,
                class_key_pairs=group.assigned_names,
            ),
        )


class DerivedExportSurfaceDetector(CandidateFindingDetector):
    detector_id = "derived_export_surface"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Manual export surfaces should derive from the authoritative type family",
        why=(
            "A module manually enumerates export names even though those exports are derivable from one local nominal class family. "
            "That creates a second authority for the public surface."
        ),
        capability_gap="one derived export surface projected from the authoritative class family",
        relation_context="manual export tuple/list repeats names already implied by local type families",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _derived_export_surface_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        export_candidate = cast(DerivedExportSurfaceCandidate, candidate)
        root_names = ", ".join(export_candidate.derivable_root_names)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{export_candidate.export_symbol}` manually enumerates {len(export_candidate.exported_names)} exported names that are derivable from local `{root_names}` families."
            ),
            (
                SourceLocation(
                    export_candidate.file_path,
                    export_candidate.line,
                    export_candidate.export_symbol,
                ),
            ),
            scaffold=(
                "def public_exports() -> tuple[str, ...]:\n"
                "    return tuple(\n"
                "        sorted(\n"
                "            name\n"
                "            for name, value in globals().items()\n"
                "            if is_public_export(name, value)\n"
                "        )\n"
                "    )"
            ),
            codemod_patch=(
                f"# Delete `{export_candidate.export_symbol}` as a handwritten export list.\n"
                "# Derive the public export surface from the authoritative local type family or generated-family registry instead."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(export_candidate.exported_names),
                field_count=len(export_candidate.derivable_root_names),
                mapping_name=export_candidate.export_symbol,
                field_names=export_candidate.derivable_root_names,
            ),
        )


class ManualPublicApiSurfaceDetector(CandidateFindingDetector):
    detector_id = "manual_public_api_surface"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Manual public API surfaces should derive from the module authority",
        why=(
            "A module hand-maintains `__all__` even though the exported names are derivable from the module's own public declarations. "
            "That creates a second authority for the public surface."
        ),
        capability_gap="one derived public API surface projected from the module's authoritative declarations",
        relation_context="manual public export list repeats names already present in module bindings",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _manual_public_api_surface_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        api_candidate = cast(ManualPublicApiSurfaceCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{api_candidate.export_symbol}` manually enumerates {len(api_candidate.exported_names)} public names that are already derivable from {api_candidate.source_name_count} module bindings."
            ),
            (
                SourceLocation(
                    api_candidate.file_path,
                    api_candidate.line,
                    api_candidate.export_symbol,
                ),
            ),
            scaffold=(
                "def is_public_api_export(name: str, value: object) -> bool:\n"
                "    return not name.startswith('_') and is_public_binding(value)\n\n"
                "__all__ = sorted(\n"
                "    name for name, value in globals().items() if is_public_api_export(name, value)\n"
                ")"
            ),
            codemod_patch=(
                f"# Delete `{api_candidate.export_symbol}` as a handwritten public API list.\n"
                "# Derive the public export surface from module bindings instead of restating names in a second manual surface."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(api_candidate.exported_names),
                field_count=api_candidate.source_name_count,
                mapping_name=api_candidate.export_symbol,
                field_names=("module_public_bindings",),
            ),
        )


class ExportPolicyPredicateDetector(IssueDetector):
    detector_id = "export_policy_predicate"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated derived-surface policy predicates should collapse into one declarative policy",
        why=(
            "Several modules hand-code derived-surface policy predicates instead of routing those surfaces through one declarative policy helper."
        ),
        capability_gap="one declarative policy substrate for derived module surfaces",
        relation_context="surface-policy helper logic repeats across multiple modules with only orthogonal policy residue",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidates = tuple(
            candidate
            for module in modules
            if (candidate := _module_export_policy_predicate_candidate(module))
            is not None
        )
        if len(candidates) < 2:
            return []
        evidence = tuple(
            SourceLocation(candidate.file_path, candidate.line, candidate.function_name)
            for candidate in candidates[:6]
        )
        all_roles = tuple(
            sorted({role for candidate in candidates for role in candidate.role_names})
        )
        root_type_names = tuple(
            sorted(
                {
                    type_name
                    for candidate in candidates
                    for type_name in candidate.root_type_names
                }
            )
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Derived-surface predicates {', '.join(candidate.function_name for candidate in candidates[:6])} repeat policy roles {all_roles} over root types {root_type_names or ('<unconstrained>',)}."
                ),
                evidence,
                scaffold=(
                    "@dataclass(frozen=True)\n"
                    "class DerivedSurfacePolicy:\n"
                    "    include_callables: bool = False\n"
                    "    include_types: bool = True\n"
                    "    exclude_abstract: bool = False\n"
                    "    include_enums: bool = False\n"
                    "    root_types: tuple[type[object], ...] = ()\n\n"
                    "def derive_surface_names(namespace: dict[str, object], policy: DerivedSurfacePolicy) -> tuple[str, ...]:\n"
                    "    return tuple(sorted(name for name, value in namespace.items() if matches_surface_policy(name, value, policy)))"
                ),
                codemod_patch=(
                    "# Replace repeated `_is_public_*_export` helpers with one declarative `DerivedSurfacePolicy`.\n"
                    "# Derive the exported name surface from the policy instead of open-coding the predicate in each module."
                ),
                metrics=RepeatedMethodMetrics.from_duplicate_family(
                    duplicate_site_count=len(candidates),
                    statement_count=1,
                    class_count=len(candidates),
                    method_symbols=tuple(
                        candidate.function_name for candidate in candidates
                    ),
                ),
            )
        ]


class DerivedIndexedSurfaceDetector(CandidateFindingDetector):
    detector_id = "derived_indexed_surface"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Manual indexed module surfaces should derive from the authoritative type family",
        why=(
            "A module hand-builds an index surface over local types even though that index is derivable from the same nominal family. "
            "That splits authority between the family and a second registry projection."
        ),
        capability_gap="one derived index projected from the authoritative local type family",
        relation_context="manual dict index repeats keys and values already implied by local type families",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _derived_indexed_surface_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        index_candidate = cast(DerivedIndexedSurfaceCandidate, candidate)
        root_names = ", ".join(index_candidate.derivable_root_names)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{index_candidate.surface_name}` manually indexes {len(index_candidate.value_names)} local types by `{index_candidate.key_kind}` even though that surface is derivable from local `{root_names}` families."
            ),
            (
                SourceLocation(
                    index_candidate.file_path,
                    index_candidate.line,
                    index_candidate.surface_name,
                ),
            ),
            scaffold=(
                "def derived_index() -> dict[object, type[object]]:\n"
                "    return {project_key(item): item for item in authoritative_family()}"
            ),
            codemod_patch=(
                f"# Delete `{index_candidate.surface_name}` as a handwritten index.\n"
                "# Derive the key-to-type map from the authoritative local family instead of maintaining a second module-level registry."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(index_candidate.value_names),
                field_count=len(index_candidate.derivable_root_names),
                mapping_name=index_candidate.surface_name,
                field_names=index_candidate.derivable_root_names,
            ),
        )


class RegisteredUnionSurfaceDetector(CandidateFindingDetector):
    detector_id = "registered_union_surface"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Manual sibling-registry unions should derive from one authoritative query",
        why=(
            "A module manually unions sibling class-level registry queries even though one authoritative query or shared root can derive the full family set."
        ),
        capability_gap="one derived registry-union query on an authoritative metaclass-registry root or traversal helper",
        relation_context="manual union of sibling registry queries repeats information already present in class-time registration",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _registered_union_surface_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        union_candidate = cast(RegisteredUnionSurfaceCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{union_candidate.owner_name}` manually unions `{union_candidate.accessor_name}` across roots {union_candidate.root_names}."
            ),
            (
                SourceLocation(
                    union_candidate.file_path,
                    union_candidate.line,
                    union_candidate.owner_name,
                ),
            ),
            scaffold=(
                "from metaclass_registry import AutoRegisterMeta\n\n"
                "class UnifiedRegistryRoot(ABC, metaclass=AutoRegisterMeta):\n"
                "    __registry_key__ = \"kind\"\n"
                "    __skip_if_no_key__ = True\n"
                "    kind = None\n\n"
                "    @classmethod\n"
                f"    def {union_candidate.accessor_name}(cls):\n"
                "        return tuple(cls.__registry__.values())\n\n"
                f"def {union_candidate.owner_name}(...):\n"
                f"    return UnifiedRegistryRoot.{union_candidate.accessor_name}()"
            ),
            codemod_patch=(
                f"# Replace the manual union over {union_candidate.root_names} with one authoritative `{union_candidate.accessor_name}` query.\n"
                "# Let one shared metaclass-registry root derive the full set from `__registry__` instead of concatenating sibling roots by hand."
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(union_candidate.root_names),
                class_count=len(union_candidate.root_names),
                registry_name=union_candidate.accessor_name,
                class_names=union_candidate.root_names,
            ),
        )


class RegistryTraversalSubstrateDetector(PerModuleIssueDetector):
    detector_id = "registry_traversal_substrate"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Repeated descendant traversal helpers should collapse into one metaclass-registry root",
        why=(
            "Several roots re-implement the same descendant traversal over class-time registration state instead of sharing one authoritative metaclass-registry root."
        ),
        capability_gap="one authoritative metaclass-registry root for descendant registries",
        relation_context="same descendant traversal algorithm repeats across sibling roots with only materialization residue differing",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        group = _registry_traversal_group(module)
        if group is None:
            return []
        evidence = tuple(
            SourceLocation(group.file_path, line, f"{class_name}.{method_name}")
            for class_name, method_name, line in zip(
                group.class_names,
                group.method_names,
                group.line_numbers,
                strict=True,
            )
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Roots {', '.join(group.class_names)} repeat descendant traversal helpers {', '.join(group.method_names)} over registry attributes {group.registry_attribute_names} with materialization modes {group.materialization_kinds}."
                ),
                evidence,
                scaffold=(
                    "from metaclass_registry import AutoRegisterMeta\n\n"
                    "class RegisteredRoot(ABC, metaclass=AutoRegisterMeta):\n"
                    "    __registry_key__ = \"kind\"\n"
                    "    __skip_if_no_key__ = True\n"
                    "    kind = None\n\n"
                    "    @classmethod\n"
                    "    def registered_items(cls):\n"
                    "        return tuple(cls.__registry__.values())"
                ),
                codemod_patch=(
                    "# Replace repeated descendant traversal helpers with one metaclass-registry root.\n"
                    "# Read registered classes from `cls.__registry__` and keep only the materialization choice (`registered_type()` vs `registered_type`) at the public surface."
                ),
                metrics=RepeatedMethodMetrics.from_duplicate_family(
                    duplicate_site_count=len(group.method_names),
                    statement_count=6,
                    class_count=len(group.class_names),
                    method_symbols=tuple(
                        f"{class_name}.{method_name}"
                        for class_name, method_name in zip(
                            group.class_names,
                            group.method_names,
                            strict=True,
                        )
                    ),
                ),
            )
        ]


class AlternateConstructorFamilyDetector(CandidateFindingDetector):
    detector_id = "alternate_constructor_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Alternate constructors should collapse into one provenance-dispatched builder",
        why=(
            "Several classmethods on one record class rebuild the same keyword schema from different source node types. "
            "That provenance family should collapse into one authoritative constructor with dispatch over source kind."
        ),
        capability_gap="single provenance-aware builder for one record schema",
        relation_context="same record schema is rebuilt across sibling alternate constructors for different source types",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _alternate_constructor_family_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group = cast(AlternateConstructorFamilyGroup, candidate)
        evidence = tuple(
            SourceLocation(group.file_path, line, f"{group.class_name}.{method_name}")
            for method_name, line in zip(
                group.method_names,
                group.line_numbers,
                strict=True,
            )
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{group.class_name}` repeats schema keywords {group.keyword_names} across alternate constructors {group.method_names} for source types {group.source_type_names}."
            ),
            evidence,
            scaffold=(
                "@singledispatchmethod\n"
                "@classmethod\n"
                f"def from_source(cls, source, **context) -> {group.class_name}:\n"
                "    raise TypeError\n\n"
                "@from_source.register\n"
                "@classmethod\n"
                "def _(cls, source: SomeSource, **context):\n"
                "    return cls(...)"
            ),
            codemod_patch=(
                f"# Collapse {group.method_names} into one provenance-dispatched constructor for `{group.class_name}`.\n"
                "# Keep source-kind differences in dispatch handlers and keep the shared record schema in one authoritative builder."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(group.method_names),
                field_count=len(group.keyword_names),
                mapping_name=group.class_name,
                field_names=group.keyword_names,
            ),
        )


class DynamicSelfFieldSelectionDetector(CandidateFindingDetector):
    detector_id = "dynamic_self_field_selection"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CONFIG_CONTRACTS,
        title="Dynamic self-field selection hides a nominal contract",
        why=(
            "A class selects one of its own fields through reflective indirection instead of declaring one fail-loud hook or one canonical field."
        ),
        capability_gap="declared nominal count/value hook instead of selector-driven reflective lookup",
        relation_context="class template selects its own state through dynamic reflective field names",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _dynamic_self_field_selection_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        dynamic_candidate = cast(DynamicSelfFieldSelectionCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{dynamic_candidate.class_name}.{dynamic_candidate.method_name}` uses `{dynamic_candidate.reflective_builtin}(self, {dynamic_candidate.selector_expression})` instead of one declared nominal hook."
            ),
            (dynamic_candidate.evidence,),
            scaffold=(
                "class DeclaredCountHook(ABC):\n"
                "    @property\n"
                "    @abstractmethod\n    def count_value(self) -> int: ..."
            ),
            codemod_patch=(
                f"# Delete `{dynamic_candidate.reflective_builtin}(self, {dynamic_candidate.selector_expression})`.\n"
                "# Replace selector-driven reflection with one declared property or one canonical field on the nominal carrier."
            ),
        )


class StringBackedReflectiveNominalLookupDetector(CandidateFindingDetector):
    detector_id = "string_backed_reflective_nominal_lookup"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="String-backed reflective lookup is simulating nominal identity",
        why=(
            "The docs say a class family should not smuggle behavior through string selectors and reflection. "
            "When subclasses only supply constant names that are resolved through globals, getattr, or __dict__, "
            "the boundary should become one declared nominal hook or typed handle."
        ),
        capability_gap="declared nominal hook or typed family handle instead of string selector plus reflection",
        relation_context="class family encodes behavior with constant selector strings and resolves it reflectively",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.STRING_DISPATCH,
            ObservationTag.SEMANTIC_STRING_LITERAL,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _string_backed_reflective_nominal_lookup_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        reflective_candidate = cast(
            StringBackedReflectiveNominalLookupCandidate, candidate
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{reflective_candidate.class_name}.{reflective_candidate.method_name}` resolves `{reflective_candidate.selector_attr_name}` through `{reflective_candidate.lookup_kind}` over {len(reflective_candidate.concrete_class_names)} concrete classes."
            ),
            (reflective_candidate.evidence,),
            scaffold=(
                "class DeclaredNominalRole(ABC):\n"
                "    @classmethod\n"
                "    @abstractmethod\n"
                "    def declared_handle(cls) -> object: ..."
            ),
            codemod_patch=(
                f"# Delete the reflective `{reflective_candidate.lookup_kind}` lookup keyed by `{reflective_candidate.selector_attr_name}`.\n"
                "# Move the family boundary to one declared hook, typed handle, or polymorphic method."
            ),
            metrics=SentinelSimulationMetrics(
                class_count=len(reflective_candidate.concrete_class_names),
                branch_site_count=1,
            ),
        )


class RepeatedBuilderCallDetector(FiberCollectedShapeIssueDetector):
    detector_id = "repeated_builder_calls"
    observation_kind = ObservationKind.BUILDER_CALL
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated field assignment should become an authoritative builder",
        why=(
            "The docs say repeated manual field assignment is an SSOT violation: the mapping should be declared once "
            "in an authoritative constructor, classmethod, or shared builder rather than copied across call sites."
        ),
        capability_gap="single authoritative record-builder mapping for a repeated constructor family",
        relation_context="same builder role repeated across sibling functions or methods",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _module_shapes(self, module: ParsedModule) -> tuple[object, ...]:
        return tuple(
            _collect_typed_family_items(
                module, BuilderCallShapeFamily, BuilderCallShape
            )
        )

    def _include_shape(self, shape: object, config: DetectorConfig) -> bool:
        builder = _as_builder_shape(shape)
        return len(builder.keyword_names) >= config.min_builder_keywords

    def _group_key(self, shape: object) -> object:
        builder = _as_builder_shape(shape)
        return (builder.callee_name, builder.keyword_names, builder.value_fingerprint)

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        builders = tuple(
            sorted(
                (_as_builder_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        if len(builders) < 2:
            return None
        owner_symbols = {builder.symbol for builder in builders}
        if len(owner_symbols) < 2:
            return None
        evidence = tuple(
            SourceLocation(builder.file_path, builder.lineno, builder.symbol)
            for builder in builders[:6]
        )
        same_source = all(builder.source_arity == 1 for builder in builders)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Call `{builders[0].callee_name}` repeats the same keyword-mapping shape across {len(builders)} sites."
            ),
            evidence,
            capability_gap=(
                "single authoritative data-to-record mapping"
                if same_source
                else self.finding_spec.capability_gap
            ),
            scaffold=_builder_scaffold(builders),
            codemod_patch=_builder_patch(builders),
            metrics=MappingMetrics(
                mapping_site_count=len(builders),
                field_count=len(builders[0].keyword_names),
                mapping_name=builders[0].callee_name,
                field_names=builders[0].keyword_names,
                source_name=builders[0].source_name,
                identity_field_names=builders[0].identity_field_names,
            ),
        )


class RepeatedExportDictDetector(FiberCollectedShapeIssueDetector):
    detector_id = "repeated_export_dicts"
    observation_kind = ObservationKind.EXPORT_DICT
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated projection dict should become an authoritative schema",
        why=(
            "The docs say repeated JSON/CSV/export dicts and kwargs/source-value bags should become one authoritative "
            "row schema or projection builder instead of many hand-maintained dict literals."
        ),
        capability_gap="single authoritative projection schema for a repeated record or kwargs family",
        relation_context="same string-key projection role repeated across sibling functions or methods",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.PROJECTION_DICT,
            ObservationTag.EXPORT_MAPPING,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _module_shapes(self, module: ParsedModule) -> tuple[object, ...]:
        return tuple(
            _collect_typed_family_items(module, ExportDictShapeFamily, ExportDictShape)
        )

    def _include_shape(self, shape: object, config: DetectorConfig) -> bool:
        export_shape = _as_export_shape(shape)
        return len(export_shape.key_names) >= config.min_export_keys

    def _group_key(self, shape: object) -> object:
        export_shape = _as_export_shape(shape)
        return (export_shape.key_names, export_shape.value_fingerprint)

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        export_shapes = tuple(
            sorted(
                (_as_export_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        if len(export_shapes) < 2:
            return None
        owner_symbols = {shape.symbol for shape in export_shapes}
        if len(owner_symbols) < 2:
            return None
        evidence = tuple(
            SourceLocation(shape.file_path, shape.lineno, shape.symbol)
            for shape in export_shapes[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"String-key projection dict with keys {export_shapes[0].key_names} repeats across {len(export_shapes)} sites."
            ),
            evidence,
            scaffold=_projection_schema_scaffold(export_shapes),
            codemod_patch=_projection_schema_patch(export_shapes),
            metrics=MappingMetrics(
                mapping_site_count=len(export_shapes),
                field_count=len(export_shapes[0].key_names),
                field_names=export_shapes[0].key_names,
                source_name=export_shapes[0].source_name,
                identity_field_names=export_shapes[0].identity_field_names,
            ),
        )


class ManualClassRegistrationDetector(GroupedShapeIssueDetector):
    detector_id = "manual_class_registration"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Manual class registration should become metaclass-registry AutoRegisterMeta",
        why=(
            "The docs say repeated class-level registration boilerplate is a class-level non-orthogonal algorithm. "
            "It should move into one authoritative `metaclass-registry` base so abstract-class skipping, uniqueness, "
            "and inheritance behavior are enforced in one place."
        ),
        capability_gap="single authoritative metaclass-registry class-registration algorithm with nominal class identity",
        relation_context="same registry key family repeated through manual class-level registration assignments",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        observation_tags=(
            ObservationTag.REGISTRY_POPULATION,
            ObservationTag.CLASS_LEVEL_POSITION,
            ObservationTag.MANUAL_REGISTRATION,
        ),
    )

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        return [
            shape
            for module in modules
            for shape in _collect_typed_family_items(
                module, RegistrationShapeFamily, RegistrationShape
            )
        ]

    def _group_key(self, shape: object) -> object:
        registration = _as_registration_shape(shape)
        return registration.registry_name

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        registrations = tuple(
            sorted(
                (_as_registration_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        if len(registrations) < config.min_registration_sites:
            return None
        class_names = {item.registered_class for item in registrations}
        if len(class_names) < config.min_registration_sites:
            return None
        evidence = tuple(
            SourceLocation(item.file_path, item.lineno, item.symbol)
            for item in registrations[:6]
        )
        registry_name = registrations[0].registry_name
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Registry `{registry_name}` is populated manually for {len(class_names)} classes across {len(registrations)} sites."
            ),
            evidence,
            scaffold=_autoregister_scaffold(registry_name, class_names),
            codemod_patch=_autoregister_patch(
                registry_name, class_names, registrations
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(registrations),
                class_count=len(class_names),
                registry_name=registry_name,
                class_names=tuple(sorted(class_names)),
                class_key_pairs=tuple(
                    f"{item.registered_class}={item.key_expression}"
                    for item in registrations
                ),
            ),
        )


class ManualConcreteSubclassRosterDetector(CrossModuleCandidateDetector):
    detector_id = "manual_concrete_subclass_roster"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Manual concrete-subclass roster should become a metaclass-registry base",
        why=(
            "The docs treat mutable subclass rosters maintained through __init_subclass__ as framework logic. "
            "Abstract filtering, subclass discovery, and family access should live in one reusable `metaclass-registry` base "
            "instead of being reimplemented inside each domain family."
        ),
        capability_gap="single authoritative metaclass-registry concrete-subclass registration hook with reusable family discovery",
        relation_context="class family maintains a mutable subclass roster through __init_subclass__ and then queries it manually",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.REGISTRY_POPULATION,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_REGISTRATION,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _manual_concrete_subclass_roster_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        roster_candidate = cast(ManualConcreteSubclassRosterCandidate, candidate)
        evidence = [roster_candidate.evidence]
        evidence.extend(
            SourceLocation(
                roster_candidate.file_path,
                roster_candidate.line,
                f"{roster_candidate.class_name}.{consumer_name}",
            )
            for consumer_name in roster_candidate.consumer_names[:3]
        )
        evidence.extend(
            SourceLocation(
                roster_candidate.file_path,
                roster_candidate.line,
                class_name,
            )
            for class_name in roster_candidate.concrete_class_names[:2]
        )
        guard_summary = (
            f" guarded by `{roster_candidate.guard_summary}`"
            if roster_candidate.guard_summary
            else ""
        )
        concrete_preview = ", ".join(roster_candidate.concrete_class_names[:3])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{roster_candidate.class_name}` maintains roster `{roster_candidate.registry_name}` for {len(roster_candidate.concrete_class_names)} concrete subclasses ({concrete_preview}){guard_summary} and consumes it via {roster_candidate.consumer_names}."
            ),
            tuple(evidence[:6]),
            scaffold=(
                "from metaclass_registry import AutoRegisterMeta\n\n"
                "class AutoRegisteredFamily(ABC, metaclass=AutoRegisterMeta):\n"
                "    __registry_key__ = \"family_key\"\n"
                "    __skip_if_no_key__ = True\n"
                "    family_key = None\n\n"
                "    @classmethod\n"
                "    def registered_types(cls) -> tuple[type[Self], ...]:\n"
                "        return tuple(cls.__registry__.values())"
            ),
            codemod_patch=(
                f"# Remove manual roster `{roster_candidate.registry_name}` from `{roster_candidate.class_name}`.\n"
                "# Reuse one metaclass-registry base so descendant discovery and abstract filtering are not rewritten per family."
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(roster_candidate.concrete_class_names),
                class_count=len(roster_candidate.concrete_class_names),
                registry_name=roster_candidate.registry_name,
                class_names=roster_candidate.concrete_class_names,
            ),
        )


class PredicateSelectedConcreteFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "predicate_selected_concrete_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Predicate-selected concrete family should collapse into one metaclass-registry selector base",
        why=(
            "The docs treat repeated scans over `registered_types()` plus `matches_*` predicates as family-selection "
            "framework logic. When a root class manually filters registered concrete descendants, enforces exactly one "
            "match, and then consumes the chosen subclass, the selection algorithm should live in one reusable "
            "`metaclass-registry` family base."
        ),
        capability_gap="single authoritative metaclass-registry predicate-selected concrete-family substrate",
        relation_context="registered concrete subclasses are manually scanned and cardinality-checked inside a family root",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.PREDICATE_CHAIN,
            ObservationTag.REGISTRY_POPULATION,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _predicate_selected_concrete_family_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(PredicateSelectedConcreteFamilyCandidate, candidate)
        concrete_preview = ", ".join(family_candidate.concrete_class_names[:4])
        evidence = [family_candidate.evidence]
        evidence.extend(
            SourceLocation(
                family_candidate.file_path,
                family_candidate.line,
                class_name,
            )
            for class_name in family_candidate.concrete_class_names[:3]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{family_candidate.class_name}.{family_candidate.selector_method_name}` scans `registered_types()` and "
                f"predicate `{family_candidate.predicate_method_name}({family_candidate.context_param_name})` across "
                f"{len(family_candidate.concrete_class_names)} concrete leaves ({concrete_preview}) before manually choosing one match."
            ),
            tuple(evidence[:6]),
            scaffold=(
                "from metaclass_registry import AutoRegisterMeta\n\n"
                "ContextT = TypeVar(\"ContextT\")\n\n"
                "class PredicateSelectedConcreteFamily(ABC, Generic[ContextT], metaclass=AutoRegisterMeta):\n"
                "    __registry_key__ = \"family_key\"\n"
                "    __skip_if_no_key__ = True\n"
                "    family_key = None\n\n"
                "    @classmethod\n"
                "    def matches_context(cls, context: ContextT) -> bool:\n"
                "        return True\n\n"
                "    @classmethod\n"
                "    def select_matching_type(cls, context: ContextT) -> type[Self]:\n"
                "        matches = tuple(\n"
                "            candidate\n"
                "            for candidate in cls.__registry__.values()\n"
                "            if candidate.matches_context(context)\n"
                "        )\n"
                "        ...\n"
            ),
            codemod_patch=(
                f"# Move `{family_candidate.class_name}` selection logic into a reusable predicate-selected family base.\n"
                "# Leave only `matches_context(...)` and family-specific error shaping on the root, and stop reimplementing `cls.__registry__.values()` scans."
            ),
        )


class ParallelMirroredLeafFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "parallel_mirrored_leaf_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Parallel mirrored leaf families should derive from one axis-declared family substrate",
        why=(
            "The docs treat mirrored registered leaf catalogs as framework duplication when the same contract is repeated "
            "across two family roots and only one nominal axis really varies. The axis and role table should be "
            "authoritative so registration and leaf generation are derived instead of hand-expanded twice."
        ),
        capability_gap="single authoritative axis-declared family or role-spec table that derives mirrored registered leaves",
        relation_context="two registered abstract roots own mirrored concrete leaf catalogs over the same contract method family",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REGISTRY_POPULATION,
            ObservationTag.REPEATED_METHOD_ROLES,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _parallel_mirrored_leaf_family_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        mirrored_candidate = cast(ParallelMirroredLeafFamilyCandidate, candidate)
        shared_preview = ", ".join(mirrored_candidate.shared_leaf_family_names[:4])
        contract_preview = ", ".join(mirrored_candidate.contract_method_names)
        class_names = (
            mirrored_candidate.left.root_name,
            mirrored_candidate.right.root_name,
            *(item.symbol for item in mirrored_candidate.left.leaf_evidence),
            *(item.symbol for item in mirrored_candidate.right.leaf_evidence),
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{mirrored_candidate.left.root_name}` and `{mirrored_candidate.right.root_name}` expose mirrored `{contract_preview}` leaf catalogs "
                f"across {len(mirrored_candidate.shared_leaf_family_names)} shared role families ({shared_preview})."
            ),
            mirrored_candidate.evidence[:6],
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class FamilyRoleSpec:\n"
                "    role_name: str\n"
                "    axis_impls: tuple[callable, ...]\n\n"
                "class GeneratedLeafFamily(ABC): ...\n"
                "# Declare the varying axis once, declare roles once, and derive leaf registration from the spec table."
            ),
            codemod_patch=(
                f"# Replace mirrored roots `{mirrored_candidate.left.root_name}` and `{mirrored_candidate.right.root_name}` with one axis-declared family substrate.\n"
                "# Move shared role names into one spec table and derive concrete leaf registration from that authority."
            ),
            metrics=RegistrationMetrics(
                registration_site_count=(
                    len(mirrored_candidate.left.leaf_evidence)
                    + len(mirrored_candidate.right.leaf_evidence)
                ),
                class_count=len(class_names),
                registry_name=(
                    f"{mirrored_candidate.left.root_name}/{mirrored_candidate.right.root_name}"
                ),
                class_names=class_names,
            ),
        )


class SentinelAttributeSimulationDetector(CandidateFindingDetector):
    detector_id = "sentinel_attribute_simulation"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Sentinel attribute is simulating nominal identity",
        why=(
            "The docs say sentinel attributes only simulate identity by convention. When they drive behavior across "
            "multiple classes, the boundary should become a nominal family or another explicit identity handle."
        ),
        capability_gap="enumerable and enforceable nominal role identity",
        relation_context="same class-level sentinel attribute reused as a fake identity boundary",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.SENTINEL_ATTRIBUTE,
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        sentinel_attrs = _collect_class_sentinel_attrs(module.module)
        candidates: list[object] = []
        for attr_name, evidence in sentinel_attrs.items():
            if len(evidence) < 2:
                continue
            branch_evidence = _attribute_branch_evidence(module, attr_name)
            if not branch_evidence:
                continue
            generic_name = attr_name.lower() in {"name", "label", "title"}
            if generic_name and len(branch_evidence) < 2:
                continue
            candidates.append((attr_name, tuple(evidence), tuple(branch_evidence)))
        return tuple(candidates)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        attr_name, evidence, branch_evidence = cast(
            tuple[str, tuple[SourceLocation, ...], tuple[SourceLocation, ...]],
            candidate,
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Attribute `{attr_name}` is declared across {len(evidence)} classes and also drives {len(branch_evidence)} branch sites."
            ),
            tuple((evidence + branch_evidence)[:6]),
            metrics=SentinelSimulationMetrics(
                class_count=len(evidence),
                branch_site_count=len(branch_evidence),
            ),
        )


class PredicateFactoryChainDetector(CandidateFindingDetector):
    detector_id = "predicate_factory_chain"
    finding_spec = FindingSpec(
        pattern_id=PatternId.DISCRIMINATED_UNION,
        title="Predicate chain should become a discriminated union family",
        why=(
            "The docs say repeated predicate-driven variant selection should become an explicit subclass family with "
            "enumeration rather than an open-ended if/elif chain."
        ),
        capability_gap="exhaustive nominal variant discovery and extension",
        relation_context="same factory role repeated as predicate branches inside one function",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.ENUMERATION,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.PREDICATE_CHAIN,
            ObservationTag.FACTORY_DISPATCH,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return tuple(
            (str(module.path), function, branch_count)
            for function in _iter_functions(module.module)
            if (branch_count := _predicate_factory_chain_branch_count(function))
            is not None
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, function, branch_count = cast(
            tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, int],
            candidate,
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"{function.name} contains a {branch_count}-branch predicate factory chain returning variant constructors."
            ),
            (SourceLocation(file_path, function.lineno, function.name),),
            metrics=BranchCountMetrics(branch_site_count=branch_count),
        )


class ConfigAttributeDispatchDetector(StaticModulePatternDetector):
    detector_id = "config_attribute_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CONFIG_CONTRACTS,
        title="Config dispatch is encoded through fragile attribute probing",
        why=(
            "The docs say polymorphic configuration should dispatch on declared config family identity, not on field-name "
            "probing or ad hoc attribute comparisons."
        ),
        capability_gap="fail-loud polymorphic configuration contracts",
        relation_context="same config-family choice expressed through attribute-level probing",
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.ATTRIBUTE_PROBE,
            ObservationTag.CONFIG_DISPATCH,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        observations: tuple[ConfigDispatchObservation, ...] = (
            _collect_typed_family_items(
                module,
                ConfigDispatchObservationFamily,
                ConfigDispatchObservation,
            )
        )
        return tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations
        )

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        return 2

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} contains {len(evidence)} config-specific attribute probes or comparisons."


class ConcreteConfigFieldProbeDetector(CandidateFindingDetector):
    detector_id = "concrete_config_field_probe"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CONFIG_CONTRACTS,
        title="Concrete config backend is probing fields outside its declared contract",
        why=(
            "The docs say concrete config-backed implementations should rely on declared config fields, not reflective "
            "probing of attributes that are absent from the concrete config type. That usually means the backend is "
            "borrowing another family's contract instead of owning its own configuration boundary."
        ),
        capability_gap="fail-loud concrete config contract for one backend family",
        relation_context="one concrete backend probes fields that are not declared by its concrete config type",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.ATTRIBUTE_PROBE,
            ObservationTag.CONFIG_DISPATCH,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _concrete_config_field_probe_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        probe_candidate = cast(ConcreteConfigFieldProbeCandidate, candidate)
        missing_fields = ", ".join(probe_candidate.missing_field_names)
        reflective_builtins = "/".join(probe_candidate.probe_builtin_names)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{probe_candidate.class_name}.{probe_candidate.method_name}` probes undeclared `{probe_candidate.config_type_name}` "
                f"fields {missing_fields} through `{reflective_builtins}` on `{probe_candidate.config_attr_name}`."
            ),
            (probe_candidate.evidence,),
            scaffold=(
                "class BackendConfig(ABC):\n"
                "    @property\n"
                "    @abstractmethod\n"
                "    def declared_parameter(self) -> object: ..."
            ),
            codemod_patch=(
                f"# Delete reflective field probes against `{probe_candidate.config_type_name}`.\n"
                "# Either move this backend onto its own declared config contract or use fields that the concrete config type actually owns."
            ),
        )


class GeneratedTypeLineageDetector(StaticModulePatternDetector):
    detector_id = "generated_type_lineage"
    finding_spec = FindingSpec(
        pattern_id=PatternId.TYPE_LINEAGE,
        title="Generated types need explicit lineage tracking",
        why=(
            "The docs say generated and rebuilt types need explicit nominal lineage so normalization, reverse lookup, and "
            "provenance remain exact."
        ),
        capability_gap="exact generated-type lineage and normalization",
        relation_context="same module combines runtime type generation with lineage-sensitive registries",
        certification=SPECULATIVE,
        capability_tags=(
            CapabilityTag.TYPE_LINEAGE,
            CapabilityTag.PROVENANCE,
            CapabilityTag.BIDIRECTIONAL_NORMALIZATION,
        ),
        observation_tags=(
            ObservationTag.RUNTIME_TYPE_GENERATION,
            ObservationTag.LINEAGE_MAPPING,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        generation_observations: tuple[RuntimeTypeGenerationObservation, ...] = (
            _collect_typed_family_items(
                module,
                RuntimeTypeGenerationObservationFamily,
                RuntimeTypeGenerationObservation,
            )
        )
        generation_sites = [
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in generation_observations
            if not _is_framework_lineage_symbol(item.symbol)
        ]
        lineage_observations: tuple[LineageMappingObservation, ...] = (
            _collect_typed_family_items(
                module,
                LineageMappingObservationFamily,
                LineageMappingObservation,
            )
        )
        lineage_sites = [
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in lineage_observations
            if not _is_framework_lineage_symbol(item.symbol)
        ]
        if not generation_sites or not lineage_sites:
            return ()
        return tuple((generation_sites + lineage_sites)[:6])

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} generates runtime types and also maintains type-lineage state."


class DualAxisResolutionDetector(PerModuleIssueDetector):
    detector_id = "dual_axis_resolution"
    finding_spec = FindingSpec(
        pattern_id=PatternId.DUAL_AXIS_RESOLUTION,
        title="Nested precedence walk should be a dual-axis resolution primitive",
        why=(
            "The docs say scope x type precedence should be modeled explicitly when both context and inheritance order "
            "contribute to resolution and provenance."
        ),
        capability_gap="explicit dual-axis precedence with provenance",
        relation_context="same function combines context hierarchy and type/MRO hierarchy",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.DUAL_AXIS_RESOLUTION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.NESTED_PRECEDENCE_WALK,
            ObservationTag.SCOPE_HIERARCHY,
            ObservationTag.MRO_HIERARCHY,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        observations: tuple[DualAxisResolutionObservation, ...] = (
            _collect_typed_family_items(
                module,
                DualAxisResolutionObservationFamily,
                DualAxisResolutionObservation,
            )
        )
        for observation in observations:
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{observation.symbol} nests scope-like axis `{observation.outer_axis_name}` with MRO/type-like axis `{observation.inner_axis_name}`."
                    ),
                    (
                        SourceLocation(
                            observation.file_path,
                            observation.line,
                            observation.symbol,
                        ),
                    ),
                    metrics=ResolutionAxisMetrics(resolution_axis_count=2),
                )
            )
        return findings


class ManualVirtualMembershipDetector(StaticModulePatternDetector):
    detector_id = "manual_virtual_membership"
    finding_spec = FindingSpec(
        pattern_id=PatternId.VIRTUAL_MEMBERSHIP,
        title="Manual class-marker membership should become custom isinstance semantics",
        why=(
            "The docs say explicit runtime interface membership should be class-level and inspectable. Repeated marker checks "
            "suggest a custom isinstance/subclass boundary rather than scattered manual probing."
        ),
        capability_gap="runtime-checkable virtual membership on nominal class identity",
        relation_context="same membership question repeated through class-marker probing",
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_MARKER_PROBE,
            ObservationTag.RUNTIME_MEMBERSHIP,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        observations: tuple[ClassMarkerObservation, ...] = _collect_typed_family_items(
            module,
            ClassMarkerObservationFamily,
            ClassMarkerObservation,
        )
        return tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations
        )

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        return 2

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} performs {len(evidence)} class-level marker checks on instances."


class DynamicInterfaceGenerationDetector(StaticModulePatternDetector):
    detector_id = "dynamic_interface_generation"
    finding_spec = FindingSpec(
        pattern_id=PatternId.DYNAMIC_INTERFACE,
        title="Dynamic interface generation is present or required",
        why=(
            "The docs treat dynamically generated empty or near-empty interface types as explicit nominal identity handles "
            "when structure alone cannot express membership."
        ),
        capability_gap="explicit runtime-generated nominal interface identity",
        relation_context="same module generates interface-like nominal types at runtime",
        certification=SPECULATIVE,
        capability_tags=(
            CapabilityTag.GENERATED_INTERFACE_IDENTITY,
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.RUNTIME_TYPE_GENERATION,
            ObservationTag.INTERFACE_IDENTITY,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        observations: tuple[InterfaceGenerationObservation, ...] = (
            _collect_typed_family_items(
                module,
                InterfaceGenerationObservationFamily,
                InterfaceGenerationObservation,
            )
        )
        return tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations[:6]
        )

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return (
            f"{module.path} contains {len(evidence)} runtime-generated interface sites."
        )


class SentinelTypeMarkerDetector(StaticModulePatternDetector):
    detector_id = "sentinel_type_marker"
    finding_spec = FindingSpec(
        pattern_id=PatternId.SENTINEL_TYPE_MARKER,
        title="Unique sentinel type marker is present or should be used",
        why=(
            "The docs distinguish sentinel types from sentinel attributes: unique nominal marker objects are appropriate when "
            "exact capability identity matters more than payload."
        ),
        capability_gap="exact capability-marker identity independent of structure",
        relation_context="same module creates or uses unique nominal sentinel markers",
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CAPABILITY_MARKER_IDENTITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.SENTINEL_TYPE,
            ObservationTag.CAPABILITY_MARKER,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        observations: tuple[SentinelTypeObservation, ...] = _collect_typed_family_items(
            module,
            SentinelTypeObservationFamily,
            SentinelTypeObservation,
        )
        return tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations[:6]
        )

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} contains {len(evidence)} sentinel-type capability marker sites."


class DynamicMethodInjectionDetector(StaticModulePatternDetector):
    detector_id = "dynamic_method_injection"
    finding_spec = FindingSpec(
        pattern_id=PatternId.TYPE_NAMESPACE_INJECTION,
        title="Dynamic method injection belongs in a type-namespace pattern",
        why=(
            "The docs say behavior that must affect all current and future instances belongs in a class namespace pattern, "
            "not in repeated instance-level patching."
        ),
        capability_gap="shared type-namespace mutation for a nominal family",
        relation_context="same module mutates class behavior through runtime namespace injection",
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_TYPE_NAMESPACE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.DYNAMIC_METHOD_INJECTION,
            ObservationTag.TYPE_NAMESPACE,
        ),
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        observations: tuple[DynamicMethodInjectionObservation, ...] = (
            _collect_typed_family_items(
                module,
                DynamicMethodInjectionObservationFamily,
                DynamicMethodInjectionObservation,
            )
        )
        return tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations[:6]
        )

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} contains {len(evidence)} dynamic type-namespace injection sites."


class AttributeProbeDetector(PerModuleIssueDetector):
    detector_id = "attribute_probes"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Semantic role recovered from attribute probing",
        why=(
            "Repeated hasattr/getattr/AttributeError logic means the code is recovering identity from a "
            "partial structural view. The documented fix is to migrate this region toward an ABC contract "
            "with direct method calls and fail-loud guarantees."
        ),
        capability_gap="declared semantic role identity and import-time enforcement",
        relation_context="same module-level probing layer across multiple call sites",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
        observation_tags=(
            ObservationTag.ATTRIBUTE_PROBE,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        observations: tuple[AttributeProbeObservation, ...] = (
            _collect_typed_family_items(
                module,
                AttributeProbeObservationFamily,
                AttributeProbeObservation,
            )
        )
        observations = tuple(
            item for item in observations if not _is_framework_attribute_probe(item)
        )
        total = len(observations)
        if total < config.min_attribute_probes:
            return []
        evidence = tuple(
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in observations[:6]
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                f"{module.path} contains {total} attribute-probe sites.",
                evidence,
                metrics=ProbeCountMetrics(probe_site_count=total),
            )
        ]


class InlineLiteralDispatchDetector(PerModuleIssueDetector):
    detector_id = "inline_literal_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CLOSED_FAMILY_DISPATCH,
        title="Inline literal dispatch should be a registry",
        why=(
            "When the same observed value is split across several sibling literal branches, the docs "
            "say the local rule family should be moved into one authoritative dispatch object instead of "
            "repeating inline branch logic. When the cases select behavior, prefer an auto-registered class family "
            "over a handwritten enum table."
        ),
        capability_gap="single authoritative dispatch representation for a closed local rule family, preferably an auto-registered behavior family when the cases are behavioral",
        relation_context="same branch role repeated inline inside a module block",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
        observation_tags=(
            ObservationTag.LITERAL_BRANCH_DISPATCH,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        observations: tuple[LiteralDispatchObservation, ...] = (
            _collect_typed_family_items(
                module,
                InlineStringLiteralDispatchObservationFamily,
                LiteralDispatchObservation,
            )
        )
        for observation in observations:
            branch_count = len(observation.branch_lines)
            if branch_count < config.min_attribute_probes:
                continue
            evidence = tuple(
                SourceLocation(observation.file_path, line, observation.symbol)
                for line in observation.branch_lines[:6]
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{module.path} repeats literal-case dispatch over `{observation.axis_expression}` across {branch_count} sibling branches with cases {observation.literal_cases}."
                    ),
                    evidence,
                    relation_context=(
                        f"same branch role repeated inline inside {observation.scope_owner or 'module block'}"
                    ),
                    metrics=DispatchCountMetrics.from_literal_family(
                        observation.axis_expression,
                        observation.literal_cases,
                    ),
                )
            )
        return findings


class StringDispatchDetector(PerModuleIssueDetector):
    detector_id = "string_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CLOSED_FAMILY_DISPATCH,
        title="Closed-family dispatch expressed through strings",
        why=(
            "The docs prefer enum- or type-keyed O(1) dispatch for closed families. Repeated string branches "
            "suggest the code is using a weaker representation than the domain requires. If those strings select implementations, "
            "the stronger form is an auto-registered family keyed by the stable nominal axis."
        ),
        capability_gap="closed-family dispatch with stable nominal keys and auto-registered type authority for behavioral cases",
        relation_context="same dispatch role repeated through string comparisons or string-key registries",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
        observation_tags=(
            ObservationTag.STRING_DISPATCH,
            ObservationTag.CLOSED_FAMILY_CASES,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        observations: tuple[LiteralDispatchObservation, ...] = (
            _collect_typed_family_items(
                module,
                StringLiteralDispatchObservationFamily,
                LiteralDispatchObservation,
            )
        )
        for observation in observations:
            if len(observation.literal_cases) < config.min_string_cases:
                continue
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{module.path} dispatches on `{observation.axis_expression}` through cases {observation.literal_cases}."
                    ),
                    (
                        SourceLocation(
                            observation.file_path,
                            observation.line,
                            observation.symbol,
                        ),
                    ),
                    relation_context=(
                        f"same observed axis `{observation.axis_expression}` is split across literal string cases {observation.literal_cases}"
                    ),
                    codemod_patch=(
                        "# Promote the closed string axis to a nominal key. If the cases select behavior, define an "
                        "auto-registered family keyed by that axis and dispatch through `cls.__registry__`."
                    ),
                    metrics=DispatchCountMetrics.from_literal_family(
                        observation.axis_expression,
                        observation.literal_cases,
                    ),
                )
            )
        dict_evidence = _dispatch_dict_locations(module, config.min_string_cases)
        if dict_evidence:
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{module.path} contains {len(dict_evidence)} string-key dispatch table site(s) that encode a closed family."
                    ),
                    tuple(dict_evidence[:6]),
                    certification=STRONG_HEURISTIC,
                    relation_context=(
                        "same closed family encoded in string-key dispatch tables rather than one nominal dispatch boundary"
                    ),
                    codemod_patch=(
                        "# Replace handwritten string-key dispatch tables with one authoritative nominal family. "
                        "# Keep any string-key projection as a derived view of the auto-registered family."
                    ),
                    metrics=DispatchCountMetrics(
                        dispatch_site_count=len(dict_evidence)
                    ),
                )
            )
        return findings


class NumericLiteralDispatchDetector(PerModuleIssueDetector):
    detector_id = "numeric_literal_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CLOSED_FAMILY_DISPATCH,
        title="Closed-family dispatch expressed through numeric IDs",
        why=(
            "The docs treat repeated numeric pattern or mode IDs the same way as magic strings: the "
            "domain axis is real but undeclared. Replace the literal-ID branches with a nominal "
            "family keyed by a stable axis; if the cases select behavior, prefer an auto-registered family over a handwritten lookup table."
        ),
        capability_gap="closed-family dispatch with stable nominal keys and auto-registered type authority for behavioral cases",
        relation_context="same dispatch role repeated through numeric literal comparisons",
        confidence=MEDIUM_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
        observation_tags=(
            ObservationTag.LITERAL_ID_DISPATCH,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        observations: tuple[LiteralDispatchObservation, ...] = (
            _collect_typed_family_items(
                module,
                NumericLiteralDispatchObservationFamily,
                LiteralDispatchObservation,
            )
        )
        for observation in observations:
            if len(observation.literal_cases) < config.min_string_cases:
                continue
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{module.path} dispatches on `{observation.axis_expression}` through numeric cases {observation.literal_cases}."
                    ),
                    (
                        SourceLocation(
                            observation.file_path,
                            observation.line,
                            observation.symbol,
                        ),
                    ),
                    relation_context=(
                        f"same observed axis `{observation.axis_expression}` is split across numeric literal cases {observation.literal_cases}"
                    ),
                    metrics=DispatchCountMetrics.from_literal_family(
                        observation.axis_expression,
                        observation.literal_cases,
                    ),
                )
            )
        return findings


class RepeatedHardcodedStringDetector(CandidateFindingDetector):
    detector_id = "repeated_hardcoded_strings"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated hardcoded semantic string should become authoritative",
        why=(
            "The docs treat repeated hardcoded semantic keys as a coherence failure: the key should "
            "be declared once as an authoritative constant, enum member, or nominal handle instead "
            "of being copied across sites."
        ),
        capability_gap="single authoritative semantic-key declaration",
        relation_context="same semantic key duplicated across decision-bearing or declarative sites",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.SEMANTIC_STRING_LITERAL,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return tuple(
            (str(module.path), literal, tuple(sites))
            for literal, sites in _semantic_string_literal_sites(module).items()
            if len(sites) >= config.min_hardcoded_string_sites
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, literal, sites = cast(
            tuple[str, str, tuple[SourceLocation, ...]],
            candidate,
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"String literal `{literal}` repeats across {len(sites)} semantic sites in {file_path}."
            ),
            tuple(sites[:6]),
            metrics=MappingMetrics(
                mapping_site_count=len(sites),
                field_count=1,
                mapping_name=literal,
                field_names=(literal,),
            ),
        )


class RepeatedProjectionHelperDetector(CandidateFindingDetector):
    detector_id = "repeated_projection_helpers"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated projection helper wrappers should become one projector",
        why=(
            "The docs treat parallel projection helpers as a coherence failure: once several helpers differ only in "
            "which semantic attribute they project, the wrapper structure should be centralized in one authoritative "
            "projector and the varying projection should become a parameter."
        ),
        capability_gap="single authoritative projection helper for a repeated semantic wrapper family",
        relation_context="same helper wrapper shape repeated across sibling module functions",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.PROJECTION_HELPER,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _projection_helper_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        ordered = cast(tuple[ProjectionHelperShape, ...], candidate)
        attributes = {shape.projected_attribute for shape in ordered}
        evidence = tuple(
            SourceLocation(shape.file_path, shape.lineno, shape.symbol)
            for shape in ordered[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Projection helper wrappers {', '.join(shape.function_name for shape in ordered[:4])} repeat the same wrapper shape while only projecting different attributes."
            ),
            evidence,
            scaffold=_projection_helper_scaffold(list(ordered)),
            metrics=MappingMetrics(
                mapping_site_count=len(ordered),
                field_count=len(attributes),
            ),
        )


class ScopedShapeWrapperDetector(PerModuleIssueDetector):
    detector_id = "scoped_shape_wrapper"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Parallel guarded wrappers and specs should become a polymorphic family",
        why=(
            "Parallel wrapper functions plus parallel spec declarations mean the code already has a hidden "
            "strategy family, but it is encoded as duplicated procedural glue. The docs prefer moving the shared "
            "algorithm into an ABC and letting polymorphic spec classes own the node family differences."
        ),
        capability_gap="single authoritative polymorphic wrapper/spec family",
        relation_context="same node-guarded wrapper skeleton repeated across multiple wrapper/spec pairs",
        confidence=HIGH_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.SCOPED_SHAPE_WRAPPER,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        wrapper_pairs = _guarded_wrapper_spec_pairs(module)
        if len(wrapper_pairs) < 2:
            return []
        evidence_items = [
            SourceLocation(str(module.path), pair.spec_line, pair.spec_name)
            for pair in wrapper_pairs[:6]
        ]
        evidence_items.extend(
            SourceLocation(
                str(module.path),
                pair.function_line,
                pair.function_name,
            )
            for pair in wrapper_pairs[:6]
        )
        evidence = tuple(
            sorted(
                evidence_items,
                key=lambda item: (item.line, item.symbol),
            )[:8]
        )
        function_names = ", ".join(pair.function_name for pair in wrapper_pairs)
        spec_names = ", ".join(pair.spec_name for pair in wrapper_pairs)
        node_families = ", ".join(
            sorted({"/".join(pair.node_types) for pair in wrapper_pairs})
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"{module.path} encodes guarded wrapper functions {function_names} and specs {spec_names} as parallel wrapper/spec pairs over node families {node_families}."
                ),
                evidence,
                scaffold=(
                    "class NodeFamilySpec(ABC):\n"
                    "    node_types: ClassVar[tuple[type[ast.AST], ...]]\n\n"
                    "    @classmethod\n"
                    "    def build(cls, parsed_module, observation):\n"
                    "        node = observation.node\n"
                    "        if not isinstance(node, cls.node_types):\n"
                    "            return None\n"
                    "        return cls.build_for_node(parsed_module, node, observation)"
                ),
            )
        ]


class ManualIndexedFamilyExpansionDetector(PerModuleIssueDetector):
    detector_id = "manual_indexed_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Manually expanded indexed family should become one nominal family abstraction",
        why=(
            "The same collection scaffold is being hand-expanded over a latent family index. The docs prefer one "
            "authoritative nominal family abstraction whose members provide only the varying family metadata."
        ),
        capability_gap="single authoritative indexed family abstraction",
        relation_context="same normalized family scaffold repeated across sibling top-level functions",
        confidence=HIGH_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        observation_tags=(
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        groups: dict[str, list[IndexedFamilyWrapperCandidate]] = defaultdict(list)
        for candidate in _indexed_family_wrapper_candidates(module):
            groups[candidate.collector_name].append(candidate)
        findings: list[RefactorFinding] = []
        for candidates in groups.values():
            if len(candidates) < 2:
                continue
            ordered = sorted(candidates, key=lambda item: item.lineno)
            evidence = tuple(
                SourceLocation(str(module.path), item.lineno, item.function_name)
                for item in ordered[:6]
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"{module.path} hand-expands indexed family members {', '.join(item.function_name for item in ordered[:4])} over `{ordered[0].collector_name}`."
                    ),
                    evidence,
                    scaffold=(
                        "Introduce one nominal family abstraction that owns the shared collection scaffold and encode only the varying family index metadata in subclasses or descriptors."
                    ),
                )
            )
        return findings


class AccessorWrapperDetector(CandidateFindingDetector):
    detector_id = "accessor_wrapper"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Trivial structural accessor wrapper should collapse to attribute/property access",
        why=(
            "The docs treat one-step observation wrappers as redundant structure: if a method only transports an "
            "already-owned attribute or a one-step computed view of it, the authority should remain the attribute "
            "itself, with `@property` reserved for genuine computed access."
        ),
        capability_gap="direct authoritative attribute/property access instead of transport wrappers",
        relation_context="same class exposes owned facts through one-step transport wrappers",
        confidence=HIGH_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _accessor_wrapper_groups(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        ordered = cast(tuple[AccessorWrapperCandidate, ...], candidate)
        class_name = ordered[0].class_name
        evidence = tuple(
            SourceLocation(
                ordered_item.file_path, ordered_item.lineno, ordered_item.symbol
            )
            for ordered_item in ordered[:6]
        )
        replacement_examples = "\n".join(
            _accessor_replacement_example(ordered_item) for ordered_item in ordered[:3]
        )
        observed_attrs = ", ".join(
            sorted({ordered_item.observed_attribute for ordered_item in ordered})
        )
        wrapper_shapes = ", ".join(
            sorted(
                {
                    ordered_item.wrapper_shape.replace("_", " ")
                    for ordered_item in ordered
                }
            )
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Class {class_name} exposes {len(ordered)} structural accessor wrapper(s) over {observed_attrs}."
            ),
            evidence,
            relation_context=(
                f"same class repeats {wrapper_shapes} around owned attributes instead of exposing one authoritative access path"
            ),
            scaffold=(
                "Collapse these transport wrappers to direct dot access when they only expose owned state. "
                "If a one-step computed view must remain public, express it as an `@property`.\n\n"
                "Example replacements:\n"
                f"{replacement_examples}"
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(ordered),
                field_count=len(
                    {ordered_item.observed_attribute for ordered_item in ordered}
                ),
                mapping_name=f"{class_name} property",
                field_names=tuple(
                    sorted(
                        {ordered_item.observed_attribute for ordered_item in ordered}
                    )
                ),
            ),
        )


class WrapperChainDetector(CandidateFindingDetector):
    detector_id = "wrapper_chain"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Transport wrapper chain should collapse to one authoritative view",
        why=(
            "The docs treat stacked pass-through helpers and projection wrappers as a coherence failure: once the "
            "same facts are rewrapped across multiple helper layers, the code should keep one authoritative carrier "
            "and derive smaller views directly from it."
        ),
        capability_gap="direct authoritative projection/view instead of a stacked transport wrapper chain",
        relation_context="same fact family is transported through multiple wrapper layers before reaching the real owner",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _wrapper_chain_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        chain_candidate = cast(WrapperChainCandidate, candidate)
        wrapper_symbols = tuple(item.qualname for item in chain_candidate.wrappers)
        evidence = tuple(item.evidence for item in chain_candidate.wrappers[:6])
        projected_attributes = tuple(
            sorted(
                {
                    attr
                    for item in chain_candidate.wrappers
                    for attr in item.projected_attributes
                }
            )
        )
        scaffold = (
            "Keep one authoritative view/carrier and derive the smaller wrapper views directly from it.\n\n"
            f"Wrapper chain: {' -> '.join(wrapper_symbols)} -> {chain_candidate.leaf_delegate_symbol}"
        )
        if projected_attributes:
            scaffold += (
                "\n"
                f"Projected attributes observed in the chain: {', '.join(projected_attributes)}"
            )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Wrappers {', '.join(wrapper_symbols)} form a stacked transport chain over `{chain_candidate.leaf_delegate_symbol}`."
            ),
            evidence,
            scaffold=scaffold,
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(chain_candidate.wrappers),
                statement_count=max(
                    item.statement_count for item in chain_candidate.wrappers
                ),
                class_count=len(
                    {
                        item.qualname.split(".", 1)[0]
                        if "." in item.qualname
                        else "<module>"
                        for item in chain_candidate.wrappers
                    }
                ),
                method_symbols=wrapper_symbols,
            ),
        )


class TrivialForwardingWrapperDetector(CandidateFindingDetector):
    detector_id = "trivial_forwarding_wrapper"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Trivial forwarding wrapper should be deleted in favor of the delegate authority",
        why=(
            "A one-line wrapper that only transports inputs into `for_*().method()` or a similar nested delegate call "
            "adds no stable semantics. The docs treat that as zero-information indirection: call the authority "
            "directly at the use site instead of naming a transport shell."
        ),
        capability_gap="direct delegate authority call instead of a trivial forwarding shell",
        relation_context="wrapper symbol only transports existing inputs into a nested delegate call chain",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _trivial_forwarding_wrapper_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        wrapper_candidate = cast(TrivialForwardingWrapperCandidate, candidate)
        transported_inputs = ", ".join(wrapper_candidate.transported_value_sources[:4])
        input_summary = (
            f" It only transports {transported_inputs}."
            if transported_inputs
            else ""
        )
        private_delegate_root = _delegate_root_symbol(wrapper_candidate.delegate_symbol)
        private_delegate_summary = _is_private_symbol_name(private_delegate_root)
        scaffold = (
            f"# Delete `{wrapper_candidate.qualname}` and call `{wrapper_candidate.delegate_symbol}` directly at the use site.\n"
            "# Keep the wrapper only if it owns a new invariant, provenance boundary, or semantic rename."
        )
        codemod_patch = (
            f"# Inline `{wrapper_candidate.qualname}` into its callers.\n"
            f"# Replace the wrapper with direct calls to `{wrapper_candidate.delegate_symbol}`."
        )
        if private_delegate_summary:
            scaffold = (
                f"# `{wrapper_candidate.qualname}` is trivial, but its delegate root `{private_delegate_root}` is private.\n"
                "# Promote a public facade/ABC/policy authority instead of routing callers directly to the private delegate."
            )
            codemod_patch = (
                f"# Do not inline callers of `{wrapper_candidate.qualname}` directly onto private `{private_delegate_root}`.\n"
                "# Promote one public authority that owns the delegate contract, then route callers through that authority."
            )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{wrapper_candidate.qualname}` is a {wrapper_candidate.call_depth}-step forwarding wrapper over "
                f"`{wrapper_candidate.delegate_symbol}`.{input_summary}"
            ),
            (wrapper_candidate.evidence,),
            scaffold=scaffold,
            codemod_patch=codemod_patch,
        )


class PublicApiPrivateDelegateShellDetector(IssueDetector):
    detector_id = "public_api_private_delegate_shell"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Public API shell over a private delegate should promote a public authority",
        why=(
            "A public module-level wrapper is carrying an external API contract only because the real implementation "
            "authority is hidden behind a private `_X` root. When multiple external call sites depend on that shell, "
            "the docs prefer promoting one public facade/ABC/policy authority instead of inlining callers onto the "
            "private delegate."
        ),
        capability_gap="public authoritative facade over a private delegate family",
        relation_context="external modules depend on a public forwarding shell because the true authority is private",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.INTERFACE_IDENTITY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _public_api_private_delegate_shell_candidates(modules, config):
            external_module_summary = ", ".join(candidate.external_module_names[:3])
            external_module_suffix = (
                f" External dependents include {external_module_summary}."
                if external_module_summary
                else ""
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"`{candidate.wrapper.qualname}` is a public forwarding shell over private "
                        f"`{candidate.delegate_root_symbol}`, and {len(candidate.external_callsites)} external "
                        f"call site(s) across {len(candidate.external_module_names)} module(s) depend on it."
                        f"{external_module_suffix}"
                    ),
                    candidate.evidence,
                    scaffold=(
                        "class PublicDelegatePolicy(ABC):\n"
                        "    @classmethod\n"
                        "    @abstractmethod\n"
                        "    def for_key(cls, key): ...\n\n"
                        "    @abstractmethod\n"
                        "    def execute(self, *args, **kwargs): ...\n\n"
                        "# Keep the concrete private delegate hidden behind this public authority."
                    ),
                    codemod_patch=(
                        f"# Do not inline callers of `{candidate.wrapper.qualname}` onto private `{candidate.delegate_root_symbol}`.\n"
                        "# Promote one public facade/ABC/policy authority that owns the contract, then route external call sites through it."
                    ),
                )
            )
        return findings


class PublicApiPrivateDelegateFamilyDetector(IssueDetector):
    detector_id = "public_api_private_delegate_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Multiple public shells over one private delegate should collapse into a public facade family",
        why=(
            "When several public wrappers expose one private delegate root, the external API is fragmented across "
            "transport shells instead of owned by one public authority. The docs prefer promoting a public facade, "
            "ABC, or policy surface rather than keeping multiple pass-through exports over private machinery."
        ),
        capability_gap="single public facade family over one private delegate root",
        relation_context="multiple public wrappers expose one private delegate family to external modules",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.INTERFACE_IDENTITY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _public_api_private_delegate_family_candidates(modules, config):
            wrapper_summary = ", ".join(candidate.wrapper_names[:4])
            external_module_summary = ", ".join(candidate.external_module_names[:3])
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"Public wrappers {wrapper_summary} expose private `{candidate.delegate_root_symbol}` "
                        f"through {len(candidate.external_callsites)} external call site(s) across "
                        f"{len(candidate.external_module_names)} module(s). External dependents include "
                        f"{external_module_summary}."
                    ),
                    candidate.evidence,
                    scaffold=(
                        "class PublicFacadePolicy(ABC):\n"
                        "    @classmethod\n"
                        "    @abstractmethod\n"
                        "    def for_key(cls, key): ...\n\n"
                        "    @abstractmethod\n"
                        "    def route(self, *args, **kwargs): ...\n\n"
                        "# Re-export the contract through this public authority instead of multiple module-level shells."
                    ),
                    codemod_patch=(
                        f"# Collapse wrappers {candidate.wrapper_names} into one public facade over `{candidate.delegate_root_symbol}`.\n"
                        "# Keep the private delegate hidden and route external modules through the promoted public authority."
                    ),
                )
            )
        return findings


class NominalPolicySurfaceDetector(CandidateFindingDetector):
    detector_id = "nominal_policy_surface"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Nominal surface methods should not be thin shells over a policy family",
        why=(
            "A nominal owner exposes public methods or properties that do nothing except resolve a policy family and "
            "forward into it. The docs treat that as split authority: the owner surface should either own the contract "
            "directly or expose one explicit policy hook instead of scattering zero-information shells."
        ),
        capability_gap="single authoritative owner surface or one explicit policy accessor",
        relation_context="public owner surface delegates member-for-member into a policy family",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.INTERFACE_IDENTITY,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _nominal_policy_surface_family_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(NominalPolicySurfaceFamilyCandidate, candidate)
        method_summary = ", ".join(method.method_name for method in family_candidate.methods[:4])
        selector_summary = ", ".join(family_candidate.selector_source_exprs[:2])
        method_count = len(family_candidate.methods)
        method_phrase = (
            f"surface methods {method_summary}"
            if method_count > 1
            else f"surface method `{family_candidate.methods[0].method_name}`"
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{family_candidate.owner_class_name}` exposes {method_phrase} by resolving "
                f"`{family_candidate.policy_root_symbol}.{family_candidate.selector_method_name}` from {selector_summary}."
            ),
            family_candidate.evidence,
            scaffold=(
                "class PolicyBackedSurface(ABC):\n"
                "    @property\n"
                "    @abstractmethod\n"
                "    def _policy(self): ...\n\n"
                "    def _resolve_policy(self):\n"
                "        return self._policy\n\n"
                "# Keep one explicit policy accessor and move repeated surface forwarding behind it."
            ),
            codemod_patch=(
                f"# Collapse `{family_candidate.owner_class_name}` surface shells into one explicit policy accessor or owner-owned contract.\n"
                f"# Do not keep separate pass-through methods over `{family_candidate.policy_root_symbol}` for {method_summary}."
            ),
        )


class SemanticDictBagDetector(PerModuleIssueDetector):
    detector_id = "semantic_dict_bag"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Semantic dict bag should become a nominal dataclass",
        why=(
            "The docs treat semantic field bags as coherence failures: once a dict carries named semantic "
            "fields rather than serialization payload, the data should move into a nominal dataclass family "
            "with one authoritative schema and explicit inheritance."
        ),
        capability_gap="single authoritative nominal schema for semantic field bags",
        relation_context="same semantic field family is carried through an ad hoc dict bag instead of a nominal record",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.SEMANTIC_DICT_BAG,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _semantic_dict_bag_candidates(module):
            recommendation = candidate.recommendation
            key_list = ", ".join(candidate.key_names)
            summary = f"Semantic dict bag with keys {candidate.key_names} appears at {module.path}:{candidate.line}."
            if recommendation.matched_schema_name is not None:
                summary = (
                    f"Semantic dict bag with keys {candidate.key_names} should use `{recommendation.class_name}` "
                    f"instead of an untyped dict at {module.path}:{candidate.line}."
                )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    summary,
                    (
                        SourceLocation(
                            str(module.path), candidate.line, candidate.symbol
                        ),
                    ),
                    confidence=(
                        HIGH_CONFIDENCE
                        if recommendation.certification == CERTIFIED
                        else MEDIUM_CONFIDENCE
                    ),
                    relation_context=(
                        f"same semantic field family is carried through a {candidate.context_kind.replace('_', ' ')} "
                        "instead of a nominal record"
                    ),
                    scaffold=(
                        f"{recommendation.rationale}\n"
                        f"Base: {recommendation.base_class_name}\n"
                        f"Fields: {key_list}\n\n"
                        f"{recommendation.scaffold}"
                    ),
                    certification=recommendation.certification,
                )
            )
        return findings


class BidirectionalRegistryDetector(CandidateFindingDetector):
    detector_id = "bidirectional_registry"
    finding_spec = FindingSpec(
        pattern_id=PatternId.BIDIRECTIONAL_LOOKUP,
        title="Bidirectional registry maintained manually",
        why=(
            "The docs prescribe a single authoritative bidirectional type registry when exact companion "
            "normalization and reverse lookup matter. Manual mirrored assignments are drift-prone and "
            "should be centralized."
        ),
        capability_gap="exact bijection and O(1) reverse lookup on nominal keys",
        relation_context="same class maintains forward and reverse registry state",
        confidence=MEDIUM_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.BIDIRECTIONAL_NORMALIZATION,
            CapabilityTag.EXACT_LOOKUP,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.MIRRORED_REGISTRY,
            ObservationTag.CLASS_LEVEL_POSITION,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _mirrored_registry_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, class_name, mirrored_pairs = cast(
            tuple[str, str, tuple[tuple[int, str], ...]],
            candidate,
        )
        evidence = tuple(
            SourceLocation(file_path, lineno, f"{class_name}.{label}")
            for lineno, label in mirrored_pairs[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Class {class_name} appears to maintain mirrored forward/reverse registry assignments."
            ),
            evidence,
            observation_tags=(
                ObservationTag.MIRRORED_REGISTRY,
                ObservationTag.CLASS_LEVEL_POSITION,
                ObservationTag.MANUAL_SYNCHRONIZATION,
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(mirrored_pairs),
                registry_name=class_name,
                class_key_pairs=tuple(
                    f"{class_name}.{label}" for _, label in mirrored_pairs
                ),
            ),
        )


_METRIC_BAG_SCHEMAS = metric_semantic_bag_descriptors()

_IMPACT_BAG_SCHEMA = impact_delta_semantic_bag_descriptor()


def _semantic_dict_bag_candidates(
    module: ParsedModule,
) -> list[SemanticDictBagCandidate]:
    candidates: list[SemanticDictBagCandidate] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.function_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            for stmt in _trim_docstring_body(node.body):
                self.visit(stmt)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            candidates.extend(
                _function_local_semantic_dict_bag_candidates(
                    module, node, tuple(self.class_stack)
                )
            )
            self.function_stack.append(node.name)
            for stmt in _trim_docstring_body(node.body):
                self.visit(stmt)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            candidates.extend(
                _function_local_semantic_dict_bag_candidates(
                    module, node, tuple(self.class_stack)
                )
            )
            self.function_stack.append(node.name)
            for stmt in _trim_docstring_body(node.body):
                self.visit(stmt)
            self.function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            for keyword in node.keywords:
                if keyword.arg != "metrics" or not isinstance(keyword.value, ast.Dict):
                    continue
                items = _string_dict_items(keyword.value)
                if items is None:
                    continue
                owner_symbol = _owner_symbol(
                    tuple(self.class_stack), tuple(self.function_stack), "metrics"
                )
                recommendation = _recommend_metrics_dataclass(
                    items,
                    owner_symbol=owner_symbol,
                )
                candidates.append(
                    SemanticDictBagCandidate(
                        line=keyword.value.lineno,
                        symbol=owner_symbol,
                        key_names=tuple(items),
                        context_kind="metrics_keyword",
                        recommendation=recommendation,
                    )
                )
            self.generic_visit(node)

    Visitor().visit(module.module)
    return candidates


def _function_local_semantic_dict_bag_candidates(
    module: ParsedModule,
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    class_stack: tuple[str, ...],
) -> list[SemanticDictBagCandidate]:
    assignments: dict[str, tuple[int, dict[str, ast.AST]]] = {}
    accessed_keys: dict[str, set[str]] = defaultdict(set)
    serialization_boundary_names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.target_node = function_node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is self.target_node:
                for stmt in _trim_docstring_body(node.body):
                    self.visit(stmt)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is self.target_node:
                for stmt in _trim_docstring_body(node.body):
                    self.visit(stmt)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return None

        def visit_Assign(self, node: ast.Assign) -> None:
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                items = _string_dict_items(node.value)
                if items is not None:
                    assignments[node.targets[0].id] = (node.lineno, items)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if (
                isinstance(node.target, ast.Name)
                and node.value is not None
                and (items := _string_dict_items(node.value)) is not None
            ):
                assignments[node.target.id] = (node.lineno, items)
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name):
                key_name = _string_slice_name(node.slice)
                if key_name is not None:
                    accessed_keys[node.value.id].add(key_name)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if _is_json_boundary_call(node):
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        serialization_boundary_names.add(arg.id)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.attr in {"get", "pop", "setdefault"}
                and node.args
            ):
                key_name = _constant_string(node.args[0])
                if key_name is not None:
                    accessed_keys[node.func.value.id].add(key_name)
            self.generic_visit(node)

        def visit_Return(self, node: ast.Return) -> None:
            if self.target_node.name == "to_dict" and isinstance(node.value, ast.Name):
                serialization_boundary_names.add(node.value.id)
            self.generic_visit(node)

    Visitor().visit(function_node)

    candidates: list[SemanticDictBagCandidate] = []
    owner_symbol = _owner_symbol(class_stack, (function_node.name,), "record")
    for name, (lineno, items) in assignments.items():
        if name in serialization_boundary_names:
            continue
        touched_keys = set(items) | accessed_keys.get(name, set())
        if not touched_keys:
            continue
        recommendation = _recommend_local_semantic_record(
            tuple(sorted(touched_keys)),
            owner_symbol=owner_symbol,
            variable_name=name,
            value_nodes=items,
        )
        if recommendation is None:
            continue
        candidates.append(
            SemanticDictBagCandidate(
                line=lineno,
                symbol=f"{owner_symbol}:{name}",
                key_names=tuple(sorted(touched_keys)),
                context_kind="local_string_key_bag",
                recommendation=recommendation,
            )
        )
    return candidates


def _recommend_metrics_dataclass(
    items: dict[str, ast.AST], owner_symbol: str
) -> SemanticDataclassRecommendation:
    key_names = tuple(sorted(items))
    exact_schema = _exact_schema_match(key_names, _METRIC_BAG_SCHEMAS)
    if exact_schema is not None:
        class_name = exact_schema.class_name
        base_class_name = exact_schema.base_class_name
        rationale = f"Use existing `{class_name}`, which already inherits `{base_class_name}` for this semantic field family."
        scaffold = _instantiation_scaffold(
            class_name,
            key_names,
            items,
            prefix="metrics=",
        )
        return SemanticDataclassRecommendation.existing_schema(
            class_name,
            base_class_name,
            rationale,
            scaffold,
        )

    closest_schema = _closest_schema_match(key_names, _METRIC_BAG_SCHEMAS)
    base_class_name = (
        closest_schema.base_class_name
        if closest_schema is not None
        else FindingMetrics.__name__
    )
    class_name = _suggest_dataclass_name(owner_symbol, "Metrics")
    rationale = (
        f"Create `{class_name}` inheriting from `{base_class_name}` because this key family is closest to "
        f"existing `{closest_schema.class_name}`."
        if closest_schema is not None
        else f"Create `{class_name}` inheriting from `{FindingMetrics.__name__}` to give this metrics bag a nominal schema."
    )
    scaffold = _declaration_scaffold(
        class_name,
        base_class_name,
        key_names,
        items,
        instantiation_prefix="metrics=",
    )
    return SemanticDataclassRecommendation.proposed_schema(
        class_name,
        base_class_name,
        closest_schema.class_name if closest_schema else None,
        rationale,
        scaffold,
    )


def _recommend_local_semantic_record(
    key_names: tuple[str, ...],
    owner_symbol: str,
    variable_name: str,
    value_nodes: dict[str, ast.AST],
) -> SemanticDataclassRecommendation | None:
    exact_schema = _exact_schema_match(key_names, (_IMPACT_BAG_SCHEMA,))
    if exact_schema is not None:
        class_name = exact_schema.class_name
        rationale = f"Use `{class_name}` directly instead of a string-key impact bag."
        scaffold = _instantiation_scaffold(class_name, key_names, value_nodes)
        return SemanticDataclassRecommendation.existing_schema(
            class_name,
            exact_schema.base_class_name,
            rationale,
            scaffold,
        )

    closest_schema = _closest_schema_match(key_names, (_IMPACT_BAG_SCHEMA,))
    if closest_schema is None:
        if not (variable_name.endswith("metrics") or variable_name in {"metrics"}):
            return None
        return _recommend_metrics_dataclass(value_nodes, owner_symbol=owner_symbol)

    class_name = _suggest_dataclass_name(owner_symbol, "ImpactDelta")
    rationale = f"Create `{class_name}` inheriting from `{closest_schema.class_name}` because the local bag carries the same quantified impact fields nominally modeled there."
    scaffold = _declaration_scaffold(
        class_name,
        closest_schema.class_name,
        key_names,
        value_nodes,
    )
    return SemanticDataclassRecommendation.proposed_schema(
        class_name,
        closest_schema.class_name,
        closest_schema.class_name,
        rationale,
        scaffold,
    )


def _exact_schema_match(
    key_names: tuple[str, ...], schemas: tuple[SemanticBagDescriptor, ...]
) -> SemanticBagDescriptor | None:
    key_set = frozenset(key_names)
    for schema in schemas:
        if key_set in schema.accepted_key_sets:
            return schema
    return None


def _closest_schema_match(
    key_names: tuple[str, ...], schemas: tuple[SemanticBagDescriptor, ...]
) -> SemanticBagDescriptor | None:
    key_set = frozenset(key_names)
    best_schema: SemanticBagDescriptor | None = None
    best_score = 0.0
    for schema in schemas:
        for accepted in schema.accepted_key_sets:
            score = _set_similarity(key_set, accepted)
            if score > best_score:
                best_schema = schema
                best_score = score
    if best_score < 0.4:
        return None
    return best_schema


def _set_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _declaration_scaffold(
    class_name: str,
    base_class_name: str,
    key_names: tuple[str, ...],
    value_nodes: dict[str, ast.AST],
    instantiation_prefix: str = "",
) -> str:
    field_lines = "\n".join(
        f"    {key}: {_infer_field_type_name(key, value_nodes.get(key))}"
        for key in key_names
    )
    return (
        "@dataclass(frozen=True)\n"
        f"class {class_name}({base_class_name}):\n"
        f"{field_lines}\n\n"
        f"{_instantiation_scaffold(class_name, key_names, value_nodes, prefix=instantiation_prefix)}"
    )


def _instantiation_scaffold(
    class_name: str,
    key_names: tuple[str, ...],
    value_nodes: dict[str, ast.AST],
    prefix: str = "",
) -> str:
    rendered_args = ",\n    ".join(
        f"{key}={_render_value_expression(key, value_nodes.get(key))}"
        for key in key_names
    )
    return f"{prefix}{class_name}(\n    {rendered_args}\n)"


def _infer_field_type_name(key_name: str, node: ast.AST | None) -> str:
    if (
        key_name.endswith("_count")
        or "bound" in key_name
        or key_name.startswith("loci_")
    ):
        return "int"
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "bool"
        if isinstance(node.value, int):
            return "int"
        if isinstance(node.value, str):
            return "str"
        if node.value is None:
            return "object | None"
    if isinstance(node, ast.Compare):
        return "bool"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"len", "sum", "max", "min"}:
            return "int"
    if isinstance(node, ast.Tuple):
        return "tuple[object, ...]"
    if isinstance(node, ast.List):
        return "list[object]"
    return "object"


def _render_value_expression(key_name: str, node: ast.AST | None) -> str:
    if node is None:
        if (
            key_name.endswith("_count")
            or "bound" in key_name
            or key_name.startswith("loci_")
        ):
            return "0"
        return "..."
    return ast.unparse(node)


def _suggest_dataclass_name(owner_symbol: str, suffix: str) -> str:
    parts = [
        _camel_case(part)
        for part in re.split(r"[^A-Za-z0-9]+", owner_symbol)
        if part and part not in {"module", "record", "metrics"}
    ]
    prefix = parts[-1] if parts else "Semantic"
    if prefix.endswith(suffix):
        return prefix
    return f"{prefix}{suffix}"


def _camel_case(value: str) -> str:
    if not value:
        return ""
    if value.isupper():
        return value.title().replace("_", "")
    chunks = [chunk for chunk in re.split(r"_+", value) if chunk]
    return "".join(chunk[:1].upper() + chunk[1:] for chunk in chunks)


def _owner_symbol(
    class_stack: tuple[str, ...], function_stack: tuple[str, ...], label: str
) -> str:
    owner = function_stack[-1] if function_stack else "<module>"
    if class_stack:
        owner = f"{class_stack[-1]}.{owner}"
    return f"{owner}:{label}"


def _string_dict_items(node: ast.AST) -> dict[str, ast.AST] | None:
    if not isinstance(node, ast.Dict) or not node.keys:
        return None
    items: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        key_name = _constant_string(key)
        if key_name is None or value is None:
            return None
        items[key_name] = value
    return items


def _string_slice_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _trim_docstring_body(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and _is_docstring_expr(body[0]):
        return body[1:]
    return body


def _is_json_boundary_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name) and node.func.id == "asdict":
        return True
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "json"
        and node.func.attr in {"dump", "dumps"}
    )


def _ast_terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _ast_terminal_name(node.value)
    return None


def _ast_attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _ast_attribute_chain(node.value)
        if parent is None:
            return None
        return (*parent, node.attr)
    return None


def _declared_base_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                base_name
                for base_name in (_ast_terminal_name(base) for base in node.bases)
                if base_name is not None
            }
        )
    )


def _class_direct_assignments(node: ast.ClassDef) -> dict[str, ast.AST | None]:
    assignments: dict[str, ast.AST | None] = {}
    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                assignments[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            assignments[statement.target.id] = statement.value
    return assignments


def _class_direct_constant_string_assignments(node: ast.ClassDef) -> dict[str, str]:
    return {
        name: string_value
        for name, value in _class_direct_assignments(node).items()
        if (string_value := _constant_string(value)) is not None
    }


def _class_direct_non_none_assignment_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, value in _class_direct_assignments(node).items()
            if not (isinstance(value, ast.Constant) and value.value is None)
        )
    )


def _iter_class_methods(
    node: ast.ClassDef,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    return tuple(
        statement
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _class_method_named(
    node: ast.ClassDef, method_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for statement in _iter_class_methods(node):
        if statement.name == method_name:
            return statement
    return None


def _is_abstract_class(node: ast.ClassDef) -> bool:
    if {"ABC", "ABCMeta"} & set(_declared_base_names(node)):
        return True
    for statement in node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in statement.decorator_list:
            if _ast_terminal_name(decorator) == "abstractmethod":
                return True
    return False


def _is_abstract_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _ast_terminal_name(decorator) == "abstractmethod"
        for decorator in node.decorator_list
    )


def _abstract_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        sorted(
            method.name
            for method in _iter_class_methods(node)
            if _is_abstract_method(method)
        )
    )


def _is_dataclass_class(node: ast.ClassDef) -> bool:
    return any(_is_dataclass_decorator(decorator) for decorator in node.decorator_list)


def _is_classvar_annotation(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "ClassVar"
    if isinstance(node, ast.Attribute):
        return node.attr == "ClassVar"
    if isinstance(node, ast.Subscript):
        return _is_classvar_annotation(node.value)
    return False


def _selector_attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
            return node.attr
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == _TYPE_NAME_LITERAL
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == "self"
        ):
            return node.attr
    return None


def _function_parameter_annotation_map(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for arg in (
        tuple(node.args.posonlyargs)
        + tuple(node.args.args)
        + tuple(node.args.kwonlyargs)
    ):
        if arg.annotation is None:
            continue
        annotations[arg.arg] = ast.unparse(arg.annotation)
    return annotations


def _annotation_type_names(node: ast.AST | None) -> tuple[str, ...]:
    if node is None:
        return ()
    if isinstance(node, ast.Constant) and node.value is None:
        return ()
    if isinstance(node, ast.Name):
        return () if node.id == "None" else (node.id,)
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    if isinstance(node, ast.Tuple):
        names = {
            name for element in node.elts for name in _annotation_type_names(element)
        }
        return tuple(sorted(names))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return tuple(
            sorted(
                {
                    *_annotation_type_names(node.left),
                    *_annotation_type_names(node.right),
                }
            )
        )
    if isinstance(node, ast.Subscript):
        base_name = _ast_terminal_name(node.value)
        if base_name in {"Optional", "Required", "NotRequired", "Type", _TYPE_NAME_LITERAL}:
            return _annotation_type_names(node.slice)
        if base_name == "Annotated":
            if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                return _annotation_type_names(node.slice.elts[0])
            return _annotation_type_names(node.slice)
    return ()


def _typed_field_map(node: ast.ClassDef) -> tuple[tuple[str, str], ...]:
    typed_fields: dict[str, str] = {}
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            if _is_classvar_annotation(statement.annotation):
                continue
            typed_fields.setdefault(
                statement.target.id,
                ast.unparse(statement.annotation),
            )
            continue
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if statement.name != "__init__":
            continue
        parameter_annotations = _function_parameter_annotation_map(statement)
        for inner in statement.body:
            target: ast.AST | None = None
            value: ast.AST | None = None
            if isinstance(inner, ast.Assign) and len(inner.targets) == 1:
                target = inner.targets[0]
                value = inner.value
            elif isinstance(inner, ast.AnnAssign):
                target = inner.target
                value = inner.value
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and isinstance(value, ast.Name)
                and value.id in parameter_annotations
            ):
                continue
            typed_fields.setdefault(target.attr, parameter_annotations[value.id])
    return tuple(sorted(typed_fields.items()))


def _semantic_role_names_for_fields(field_names: tuple[str, ...]) -> tuple[str, ...]:
    role_names: set[str] = set()
    for field_name in field_names:
        normalized_roles = _normalize_semantic_field_roles(field_name)
        if normalized_roles:
            role_names.update(normalized_roles)
            continue
        role_names.add(field_name)
    return tuple(sorted(role_names))


def _class_name_tokens(name: str) -> frozenset[str]:
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+", name)
    ]
    return frozenset(
        token
        for token in tokens
        if token not in {"abc", "abstract", "base", "mixin", "spec"}
    )


def _ordered_class_name_tokens(name: str) -> tuple[str, ...]:
    return tuple(
        token.lower()
        for token in re.findall(
            r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+", name.lstrip("_")
        )
        if token.lower() not in {"abc", "abstract", "base", "mixin", "spec"}
    )


def _shared_ordered_suffix(
    left_tokens: tuple[str, ...],
    right_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    shared_reversed: list[str] = []
    for left_token, right_token in zip(reversed(left_tokens), reversed(right_tokens)):
        if left_token != right_token:
            break
        shared_reversed.append(left_token)
    return tuple(reversed(shared_reversed))


def _nominal_authority_shapes(
    modules: Sequence[ParsedModule],
) -> tuple[NominalAuthorityShape, ...]:
    shapes_without_ancestors: list[NominalAuthorityShape] = []
    for module in modules:
        for node in _walk_nodes(module.module):
            if not isinstance(node, ast.ClassDef):
                continue
            field_type_map = _typed_field_map(node)
            shapes_without_ancestors.append(
                NominalAuthorityShape(
                    file_path=str(module.path),
                    class_name=node.name,
                    line=node.lineno,
                    declared_base_names=_declared_base_names(node),
                    ancestor_names=(),
                    field_names=tuple(name for name, _ in field_type_map),
                    field_type_map=field_type_map,
                    method_names=tuple(sorted(_method_names(node))),
                    is_abstract=_is_abstract_class(node),
                    is_dataclass_family=_is_dataclass_class(node),
                )
            )

    base_lookup: dict[str, set[str]] = defaultdict(set)
    for shape in shapes_without_ancestors:
        base_lookup[shape.class_name].update(shape.declared_base_names)

    def ancestors_for(class_name: str) -> tuple[str, ...]:
        seen: set[str] = set()
        stack = list(base_lookup.get(class_name, set()))
        while stack:
            base_name = stack.pop()
            if base_name in seen or base_name == class_name:
                continue
            seen.add(base_name)
            stack.extend(sorted(base_lookup.get(base_name, set()) - seen))
        return tuple(sorted(seen))

    return tuple(
        NominalAuthorityShape(
            file_path=shape.file_path,
            class_name=shape.class_name,
            line=shape.line,
            declared_base_names=shape.declared_base_names,
            ancestor_names=ancestors_for(shape.class_name),
            field_names=shape.field_names,
            field_type_map=shape.field_type_map,
            method_names=shape.method_names,
            is_abstract=shape.is_abstract,
            is_dataclass_family=shape.is_dataclass_family,
        )
        for shape in shapes_without_ancestors
    )


class NominalAuthorityIndex:
    def __init__(self, modules: Sequence[ParsedModule]) -> None:
        self._shapes = _nominal_authority_shapes(modules)
        self._shapes_by_name: dict[str, list[NominalAuthorityShape]] = defaultdict(list)
        for shape in self._shapes:
            self._shapes_by_name[shape.class_name].append(shape)

    def all_shapes(self) -> tuple[NominalAuthorityShape, ...]:
        return self._shapes

    def shapes_named(self, class_name: str) -> tuple[NominalAuthorityShape, ...]:
        return tuple(self._shapes_by_name.get(class_name, ()))

    def compatible_authorities_for(
        self, shape: NominalAuthorityShape
    ) -> tuple[NominalAuthorityShape, ...]:
        compatible: list[NominalAuthorityShape] = []
        for authority in self._shapes:
            if authority.class_name == shape.class_name:
                continue
            if authority.class_name in set(shape.ancestor_names):
                continue
            if not _is_reusable_nominal_authority(authority):
                continue
            shared_field_names = _shared_typed_field_names(shape, authority)
            if len(shared_field_names) < 2:
                continue
            if set(shared_field_names) != set(authority.field_names):
                continue
            compatible.append(authority)
        return tuple(
            sorted(
                compatible,
                key=lambda authority: (
                    -len(authority.field_names),
                    not authority.is_abstract,
                    authority.class_name,
                ),
            )
        )


def _is_reusable_nominal_authority(shape: NominalAuthorityShape) -> bool:
    if shape.class_name.endswith("Detector"):
        return False
    return bool(
        shape.is_abstract or shape.class_name.endswith(("Base", "Mixin", "Carrier"))
    )


def _shared_typed_field_names(
    concrete: NominalAuthorityShape,
    authority: NominalAuthorityShape,
) -> tuple[str, ...]:
    concrete_types = dict(concrete.field_type_map)
    return tuple(
        name
        for name, annotation_text in authority.field_type_map
        if concrete_types.get(name) == annotation_text
    )


def _extract_family_roster_members(
    node: ast.AST,
    known_class_names: set[str],
) -> tuple[tuple[str, ...], str] | None:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return None
    member_names: list[str] = []
    constructor_styles: set[str] = set()
    for element in node.elts:
        if isinstance(element, ast.Name) and element.id in known_class_names:
            member_names.append(element.id)
            constructor_styles.add("class_reference")
            continue
        if (
            isinstance(element, ast.Call)
            and not element.args
            and not element.keywords
            and isinstance(element.func, ast.Name)
            and element.func.id in known_class_names
        ):
            member_names.append(element.func.id)
            constructor_styles.add("constructor_call")
            continue
        return None
    if len(member_names) < 2:
        return None
    return (tuple(member_names), "+".join(sorted(constructor_styles)))


def _best_shared_family_base_name(
    member_names: tuple[str, ...], index: NominalAuthorityIndex
) -> str | None:
    candidate_sets: list[set[str]] = []
    for member_name in member_names:
        shapes = index.shapes_named(member_name)
        if not shapes:
            return None
        ancestor_names = {
            name
            for shape in shapes
            for name in (*shape.declared_base_names, *shape.ancestor_names)
            if name not in _IGNORED_ANCESTOR_NAMES
        }
        if not ancestor_names:
            return None
        candidate_sets.append(ancestor_names)
    shared = set.intersection(*candidate_sets)
    if not shared:
        return None
    return sorted(shared, key=lambda item: (item.startswith("Issue"), len(item), item))[
        0
    ]


def _manual_family_roster_candidates(
    module: ParsedModule,
    index: NominalAuthorityIndex,
) -> tuple[ManualFamilyRosterCandidate, ...]:
    known_class_names = {
        shape.class_name
        for shape in index.all_shapes()
        if shape.file_path == str(module.path)
    }
    candidates: list[ManualFamilyRosterCandidate] = []
    module_body = _trim_docstring_body(module.module.body)
    for statement in module_body:
        owner_name: str | None = None
        line = statement.lineno
        source_node: ast.AST | None = None
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = _trim_docstring_body(statement.body)
            if (
                len(body) != 1
                or not isinstance(body[0], ast.Return)
                or body[0].value is None
            ):
                continue
            owner_name = statement.name
            source_node = body[0].value
            line = statement.lineno
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                continue
            owner_name = target.id
            source_node = statement.value
            line = statement.lineno
        if owner_name is None or source_node is None:
            continue
        extracted = _extract_family_roster_members(source_node, known_class_names)
        if extracted is None:
            continue
        member_names, constructor_style = extracted
        family_base_name = _best_shared_family_base_name(member_names, index)
        if family_base_name is None:
            continue
        candidates.append(
            ManualFamilyRosterCandidate(
                file_path=str(module.path),
                line=line,
                owner_name=owner_name,
                member_names=member_names,
                family_base_name=family_base_name,
                constructor_style=constructor_style,
            )
        )
    return tuple(candidates)


def _enum_key_family(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Attribute):
        return None
    if not isinstance(node.value, ast.Name):
        return None
    return (node.value.id, node.attr)


def _fragmented_family_authority_candidates(
    module: ParsedModule,
) -> tuple[FragmentedFamilyAuthorityCandidate, ...]:
    family_maps: dict[str, list[tuple[str, int, tuple[str, ...]]]] = defaultdict(list)
    for statement in _trim_docstring_body(module.module.body):
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is None or value is None or not isinstance(value, ast.Dict):
            continue
        key_pairs = tuple(
            key_pair
            for key_pair in (
                _enum_key_family(key) for key in value.keys if key is not None
            )
            if key_pair is not None
        )
        if len(key_pairs) < 2 or len(key_pairs) != len(value.keys):
            continue
        family_names = {family_name for family_name, _ in key_pairs}
        if len(family_names) != 1:
            continue
        family_name = next(iter(family_names))
        key_names = tuple(sorted(member_name for _, member_name in key_pairs))
        family_maps[family_name].append((target_name, statement.lineno, key_names))

    candidates: list[FragmentedFamilyAuthorityCandidate] = []
    for family_name, entries in family_maps.items():
        if len(entries) < 2:
            continue
        key_counter: Counter[str] = Counter(
            key_name for _, _, key_names in entries for key_name in set(key_names)
        )
        shared_keys = tuple(
            sorted(key for key, count in key_counter.items() if count >= 2)
        )
        if len(shared_keys) < 3:
            continue
        total_keys = tuple(sorted(key_counter))
        ordered_entries = sorted(entries, key=lambda item: item[1])
        candidates.append(
            FragmentedFamilyAuthorityCandidate(
                file_path=str(module.path),
                mapping_names=tuple(item[0] for item in ordered_entries),
                line_numbers=tuple(item[1] for item in ordered_entries),
                key_family_name=family_name,
                shared_keys=shared_keys,
                total_keys=total_keys,
            )
        )
    return tuple(candidates)


_DETECTOR_BASE_NAMES = {
    "IssueDetector",
    "PerModuleIssueDetector",
    "EvidenceOnlyPerModuleDetector",
    "StaticModulePatternDetector",
    "GroupedShapeIssueDetector",
    "FiberCollectedShapeIssueDetector",
}

_IGNORED_BASE_NAMES = frozenset({"ABC", "object"})
_IGNORED_ANCESTOR_NAMES = frozenset({"ABC", "ABCMeta", "object"})


def _is_detectorish_class(node: ast.ClassDef) -> bool:
    if node.name.endswith("Detector"):
        return True
    return bool(_DETECTOR_BASE_NAMES & set(_declared_base_names(node)))


def _finding_build_call(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Call | None:
    for node in _walk_nodes(method):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "build":
            continue
        value = node.func.value
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "finding_spec"
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
        ):
            continue
        return node
    return None


def _call_display_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _build_call_keyword_helper_name(
    build_call: ast.Call, keyword_name: str
) -> str | None:
    for keyword in build_call.keywords:
        if keyword.arg != keyword_name or keyword.value is None:
            continue
        if isinstance(keyword.value, ast.Call):
            return _call_display_name(keyword.value)
    return None


def _candidate_source_name_from_method(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    assigned_calls: dict[str, str] = {}
    for statement in _trim_docstring_body(method.body):
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and isinstance(statement.value, ast.Call):
                call_name = _call_display_name(statement.value)
                if call_name is not None:
                    assigned_calls[target.id] = call_name
        if isinstance(statement, ast.For):
            iterator = statement.iter
            if isinstance(iterator, ast.Call):
                return _call_display_name(iterator)
            if isinstance(iterator, ast.Name):
                return assigned_calls.get(iterator.id)
    return None


def _finding_assembly_pipeline_candidates(
    module: ParsedModule,
) -> tuple[FindingAssemblyPipelineCandidate, ...]:
    candidates: list[FindingAssemblyPipelineCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef) or not _is_detectorish_class(node):
            continue
        method = next(
            (
                statement
                for statement in node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == "_findings_for_module"
            ),
            None,
        )
        if method is None:
            continue
        build_call = _finding_build_call(method)
        if build_call is None:
            continue
        candidate_source_name = _candidate_source_name_from_method(method)
        if candidate_source_name is None:
            continue
        metrics_type_name = _build_call_keyword_helper_name(build_call, "metrics")
        scaffold_helper_name = _build_call_keyword_helper_name(build_call, "scaffold")
        patch_helper_name = _build_call_keyword_helper_name(build_call, "codemod_patch")
        if not any(
            helper_name is not None
            for helper_name in (
                metrics_type_name,
                scaffold_helper_name,
                patch_helper_name,
            )
        ):
            continue
        candidates.append(
            FindingAssemblyPipelineCandidate(
                file_path=str(module.path),
                line=method.lineno,
                subject_name=node.name,
                name_family=tuple(
                    item
                    for item in (
                        candidate_source_name,
                        metrics_type_name,
                        scaffold_helper_name,
                        patch_helper_name,
                    )
                    if item is not None
                ),
                method_name=method.name,
                candidate_source_name=candidate_source_name,
                metrics_type_name=metrics_type_name,
                scaffold_helper_name=scaffold_helper_name,
                patch_helper_name=patch_helper_name,
            )
        )
    return tuple(candidates)


def _is_observation_spec_class(node: ast.ClassDef) -> bool:
    if node.name.endswith("ObservationSpec"):
        return True
    return bool(
        {
            "ObservationShapeSpec",
            "FunctionObservationSpec",
            "AssignObservationSpec",
            "ContextForwardingShapeSpec",
        }
        & set(_declared_base_names(node))
    )


def _if_returns_none_only(node: ast.If) -> bool:
    return bool(
        len(node.body) == 1
        and isinstance(node.body[0], ast.Return)
        and isinstance(node.body[0].value, ast.Constant)
        and node.body[0].value.value is None
        and not node.orelse
    )


def _delegate_name_from_return(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        outer_name = _call_display_name(node)
        if outer_name in {"tuple", "list", "set"} and len(node.args) == 1:
            inner = node.args[0]
            if isinstance(inner, ast.Call):
                return _call_display_name(inner)
        return outer_name
    return None


def _guard_role_name(node: ast.AST) -> str:
    text = ast.unparse(node)
    if "observation.class_name is not None" in text:
        return "module_only_guard"
    if "observation.class_name is None" in text:
        return "class_only_guard"
    if "observation.function_name is None" in text:
        return "module_scope_guard"
    if "observation.function_name is not None" in text:
        return "function_scope_guard"
    if "isinstance" in text:
        return "node_type_guard"
    return "guarded_delegate"


def _scope_role_name(node: ast.AST) -> str:
    text = ast.unparse(node)
    if "class_name" in text and "function_name" in text:
        return "scope_filtered"
    if "class_name" in text:
        return "class_scope"
    if "function_name" in text:
        return "function_scope"
    if "isinstance" in text:
        return "node_type"
    return "generic_scope"


def _guarded_delegator_candidates(
    module: ParsedModule,
) -> tuple[GuardedDelegatorCandidate, ...]:
    candidates: list[GuardedDelegatorCandidate] = []
    for node in _walk_nodes(module.module):
        if (
            not isinstance(node, ast.ClassDef)
            or not _is_observation_spec_class(node)
            or _is_abstract_class(node)
        ):
            continue
        for statement in node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if statement.name not in {
                "build_from_function",
                "build_from_assign",
                "build_from_observation",
                "build_from_context",
            }:
                continue
            body = _trim_docstring_body(statement.body)
            while body and isinstance(body[0], ast.Assign):
                body = body[1:]
            if len(body) != 2:
                continue
            guard, return_stmt = body
            if not isinstance(guard, ast.If) or not _if_returns_none_only(guard):
                continue
            if not isinstance(return_stmt, ast.Return) or return_stmt.value is None:
                continue
            delegate_name = _delegate_name_from_return(return_stmt.value)
            if delegate_name is None:
                continue
            candidates.append(
                GuardedDelegatorCandidate(
                    file_path=str(module.path),
                    line=statement.lineno,
                    subject_name=node.name,
                    name_family=(
                        guard.test.__class__.__name__,
                        delegate_name,
                        _scope_role_name(guard.test),
                    ),
                    method_name=statement.name,
                    guard_role=_guard_role_name(guard.test),
                    delegate_name=delegate_name,
                    scope_role=_scope_role_name(guard.test),
                )
            )
    return tuple(candidates)


def _structural_observation_property_candidates(
    module: ParsedModule,
) -> tuple[StructuralObservationPropertyCandidate, ...]:
    candidates: list[StructuralObservationPropertyCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not any(
                _ast_terminal_name(decorator) == "property"
                for decorator in statement.decorator_list
            ):
                continue
            body = _trim_docstring_body(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            returned = body[0].value
            if not isinstance(returned, ast.Call):
                continue
            constructor_name = _call_name(returned.func)
            if constructor_name is None:
                continue
            keyword_names = tuple(
                sorted(
                    keyword.arg
                    for keyword in returned.keywords
                    if keyword.arg is not None
                )
            )
            if len(keyword_names) < 6:
                continue
            candidates.append(
                StructuralObservationPropertyCandidate(
                    file_path=str(module.path),
                    line=statement.lineno,
                    subject_name=node.name,
                    name_family=keyword_names,
                    property_name=statement.name,
                    constructor_name=constructor_name,
                )
            )
    return tuple(candidates)


def _reuse_kind_for_authority(shape: NominalAuthorityShape) -> str:
    return "compose_mixin" if shape.class_name.endswith("Mixin") else "inherit_base"


def _existing_nominal_authority_reuse_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[ExistingNominalAuthorityReuseCandidate, ...]:
    index = NominalAuthorityIndex(modules)
    candidates: list[ExistingNominalAuthorityReuseCandidate] = []
    for shape in index.all_shapes():
        if shape.is_abstract or len(shape.field_type_map) < 2:
            continue
        compatible = index.compatible_authorities_for(shape)
        if not compatible:
            continue
        authority = next(
            (
                item
                for item in compatible
                if _class_name_tokens(item.class_name)
                & _class_name_tokens(shape.class_name)
            ),
            None,
        )
        if authority is None:
            continue
        shared_field_names = _shared_typed_field_names(shape, authority)
        if len(shared_field_names) < 2:
            continue
        candidates.append(
            ExistingNominalAuthorityReuseCandidate(
                file_path=shape.file_path,
                line=shape.line,
                subject_name=shape.class_name,
                name_family=shared_field_names,
                compatible_authority_file_path=authority.file_path,
                compatible_authority_name=authority.class_name,
                compatible_authority_line=authority.line,
                reuse_kind=_reuse_kind_for_authority(authority),
                shared_role_names=_semantic_role_names_for_fields(shared_field_names),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.line,
                item.class_name,
                item.compatible_authority_name,
            ),
        )
    )


def _normalized_authority_name(annotation_text: str) -> str:
    text = annotation_text.strip("\"'")
    text = re.split(r"\s*\|\s*", text, maxsplit=1)[0]
    text = re.split(r"[\[,]", text, maxsplit=1)[0]
    return text.rsplit(".", 1)[-1].strip()


def _is_self_delegate_attribute(node: ast.AST, delegate_field_name: str) -> bool:
    return bool(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == delegate_field_name
    )


def _is_forwarded_parameter_reference(
    node: ast.AST,
    parameter_names: tuple[str, ...],
) -> bool:
    return (
        isinstance(node, ast.Name) and node.id in set(parameter_names)
    ) or (
        isinstance(node, ast.Starred)
        and isinstance(node.value, ast.Name)
        and node.value.id in set(parameter_names)
    )


def _forwarded_delegate_member_name(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    delegate_field_name: str,
) -> str | None:
    body = _trim_docstring_body(method.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    returned = body[0].value
    if any(_ast_terminal_name(decorator) == "property" for decorator in method.decorator_list):
        if (
            isinstance(returned, ast.Attribute)
            and _is_self_delegate_attribute(returned.value, delegate_field_name)
            and method.name == returned.attr
        ):
            return returned.attr
        return None
    if not (
        isinstance(returned, ast.Call)
        and isinstance(returned.func, ast.Attribute)
        and _is_self_delegate_attribute(returned.func.value, delegate_field_name)
        and method.name == returned.func.attr
    ):
        return None
    parameter_names = tuple(
        arg.arg
        for arg in (
            *method.args.posonlyargs,
            *method.args.args[1:],
            *method.args.kwonlyargs,
        )
    )
    if not all(
        _is_forwarded_parameter_reference(argument, parameter_names)
        for argument in returned.args
    ):
        return None
    if not all(
        keyword.arg is None
        or (
            keyword.arg in set(parameter_names)
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == keyword.arg
        )
        for keyword in returned.keywords
    ):
        return None
    return returned.func.attr


def _pass_through_nominal_wrapper_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[PassThroughNominalWrapperCandidate, ...]:
    index = NominalAuthorityIndex(modules)
    candidates: list[PassThroughNominalWrapperCandidate] = []
    for module in modules:
        for node in _walk_nodes(module.module):
            if not isinstance(node, ast.ClassDef) or _is_abstract_class(node):
                continue
            typed_fields = _typed_field_map(node)
            if len(typed_fields) != 1:
                continue
            delegate_field_name, annotation_text = typed_fields[0]
            delegate_authority_name = _normalized_authority_name(annotation_text)
            if not delegate_authority_name:
                continue
            if delegate_authority_name in set(_declared_base_names(node)):
                continue
            authorities = tuple(
                authority
                for authority in index.shapes_named(delegate_authority_name)
                if _is_reusable_nominal_authority(authority)
            )
            if not authorities:
                continue
            authority = authorities[0]
            forwarded_member_names: list[str] = []
            unsupported_residue = False
            for statement in _trim_docstring_body(node.body):
                if isinstance(statement, ast.AnnAssign):
                    continue
                if isinstance(statement, ast.Assign):
                    continue
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if statement.name == "__init__":
                        continue
                    if statement.name.startswith("__") and statement.name.endswith("__"):
                        unsupported_residue = True
                        break
                    forwarded_member_name = _forwarded_delegate_member_name(
                        statement, delegate_field_name
                    )
                    if forwarded_member_name is None:
                        unsupported_residue = True
                        break
                    forwarded_member_names.append(forwarded_member_name)
                    continue
                unsupported_residue = True
                break
            if unsupported_residue or len(forwarded_member_names) < 2:
                continue
            if not set(forwarded_member_names) <= set(authority.method_names):
                continue
            candidates.append(
                PassThroughNominalWrapperCandidate(
                    file_path=str(module.path),
                    line=node.lineno,
                    subject_name=node.name,
                    name_family=tuple(sorted(set(forwarded_member_names))),
                    delegate_field_name=delegate_field_name,
                    delegate_authority_file_path=authority.file_path,
                    delegate_authority_name=authority.class_name,
                    delegate_authority_line=authority.line,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.line,
                item.class_name,
                item.delegate_authority_name,
            ),
        )
    )


def _is_projection_like_builder_value(value_fingerprint: str) -> bool:
    return value_fingerprint.startswith(
        (
            "Name(",
            "Attribute(",
            "IfExp(",
            "Constant(",
        )
    )


def _projection_builder_groups(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[tuple[BuilderCallShape, ...], ...]:
    grouped: dict[tuple[str, tuple[str, ...]], list[BuilderCallShape]] = defaultdict(list)
    for builder in _collect_typed_family_items(module, BuilderCallShapeFamily, BuilderCallShape):
        if len(builder.keyword_names) < max(config.min_builder_keywords, 6):
            continue
        if not all(
            _is_projection_like_builder_value(value)
            for value in builder.value_fingerprint
        ):
            continue
        grouped[(builder.callee_name, builder.keyword_names)].append(builder)
    candidates: list[tuple[BuilderCallShape, ...]] = []
    for builders in grouped.values():
        if len(builders) < 3:
            continue
        if len({builder.value_fingerprint for builder in builders}) < 2:
            continue
        if len({builder.symbol for builder in builders}) < 2:
            continue
        candidates.append(
            tuple(sorted(builders, key=lambda item: (item.file_path, item.lineno)))
        )
    return tuple(
        sorted(
            candidates,
            key=lambda group: (
                group[0].file_path,
                group[0].lineno,
                group[0].callee_name,
            ),
        )
    )


def _projection_helper_groups(
    module: ParsedModule,
) -> tuple[tuple[ProjectionHelperShape, ...], ...]:
    shapes: tuple[ProjectionHelperShape, ...] = _collect_typed_family_items(
        module,
        ProjectionHelperObservationFamily,
        ProjectionHelperShape,
    )
    graph = ObservationGraph(tuple(shape.structural_observation for shape in shapes))
    lookup = _carrier_lookup(tuple(shapes))
    groups: list[tuple[ProjectionHelperShape, ...]] = []
    for fiber in graph.fibers_with_min_observations(
        ObservationKind.PROJECTION_HELPER,
        StructuralExecutionLevel.FUNCTION_BODY,
        minimum_observations=2,
    ):
        ordered = tuple(
            _as_projection_helper_shape(item)
            for item in _materialize_observations(fiber.observations, lookup)
        )
        attributes = {shape.projected_attribute for shape in ordered}
        if len(attributes) < 2:
            continue
        groups.append(ordered)
    return tuple(groups)


def _accessor_wrapper_groups(
    module: ParsedModule,
) -> tuple[tuple[AccessorWrapperCandidate, ...], ...]:
    candidates: tuple[AccessorWrapperCandidate, ...] = _collect_typed_family_items(
        module,
        AccessorWrapperObservationFamily,
        AccessorWrapperCandidate,
    )
    graph = ObservationGraph(
        tuple(candidate.structural_observation for candidate in candidates)
    )
    lookup = _carrier_lookup(tuple(candidates))
    groups: list[tuple[AccessorWrapperCandidate, ...]] = []
    for witness_group in graph.witness_groups_for(
        ObservationKind.ACCESSOR_WRAPPER,
        StructuralExecutionLevel.FUNCTION_BODY,
    ):
        ordered = tuple(
            _as_accessor_wrapper_candidate(item)
            for item in _materialize_observations(
                witness_group.observations,
                lookup,
            )
        )
        if not _supports_accessor_wrapper_finding(list(ordered)):
            continue
        groups.append(ordered)
    return tuple(groups)


def _mirrored_registry_candidates(
    module: ParsedModule,
) -> tuple[tuple[str, str, tuple[tuple[int, str], ...]], ...]:
    candidates: list[tuple[str, str, tuple[tuple[int, str], ...]]] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        dict_attrs = _collect_dict_attrs(node)
        mirrored_pairs = _collect_mirrored_assignments(node)
        if len(dict_attrs) < 2 or not mirrored_pairs:
            continue
        candidates.append((str(module.path), node.name, tuple(mirrored_pairs)))
    return tuple(candidates)


def _property_alias_hook_groups(
    module: ParsedModule,
) -> tuple[PropertyAliasHookGroup, ...]:
    grouped: dict[tuple[str, str, str], list[tuple[str, int]]] = defaultdict(list)
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = tuple(
            name
            for name in _declared_base_names(node)
            if name not in _IGNORED_BASE_NAMES
        )
        if not base_names:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not any(
                _ast_terminal_name(decorator) == "property"
                for decorator in statement.decorator_list
            ):
                continue
            if len(statement.args.args) != 1:
                continue
            body = _trim_docstring_body(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            returned = body[0].value
            if not (
                isinstance(returned, ast.Attribute)
                and isinstance(returned.value, ast.Name)
                and returned.value.id == "self"
            ):
                continue
            for base_name in base_names:
                grouped[(base_name, statement.name, returned.attr)].append(
                    (node.name, statement.lineno)
                )
    return tuple(
        PropertyAliasHookGroup(
            file_path=str(module.path),
            base_name=base_name,
            property_name=property_name,
            returned_attribute=returned_attribute,
            class_names=tuple(class_name for class_name, _ in ordered),
            line_numbers=tuple(line for _, line in ordered),
        )
        for (base_name, property_name, returned_attribute), items in sorted(
            grouped.items()
        )
        if len(items) >= 2
        for ordered in [tuple(sorted(items, key=lambda item: (item[1], item[0])))]
    )


def _is_constant_hook_expression(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) or (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id != "self"
    )


def _module_class_defs_by_name(module: ParsedModule) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in _walk_nodes(module.module)
        if isinstance(node, ast.ClassDef)
    }


def _descendant_class_names(
    class_defs_by_name: dict[str, ast.ClassDef], base_name: str
) -> tuple[str, ...]:
    children_by_base: dict[str, set[str]] = defaultdict(set)
    for class_name, node in class_defs_by_name.items():
        for declared_base_name in _declared_base_names(node):
            children_by_base[declared_base_name].add(class_name)
    descendants: list[str] = []
    queue = sorted(children_by_base.get(base_name, ()))
    seen: set[str] = set()
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        descendants.append(current)
        queue.extend(
            child
            for child in sorted(children_by_base.get(current, ()))
            if child not in seen
        )
    return tuple(descendants)


def _indexed_class_display_name(
    indexed_class: IndexedClass,
    class_index: ClassFamilyIndex,
) -> str:
    simple_name = indexed_class.simple_name
    if len(class_index.symbols_by_simple_name.get(simple_name, ())) <= 1:
        return simple_name
    return indexed_class.symbol


def _indexed_class_display_names(
    indexed_classes: tuple[IndexedClass, ...],
    class_index: ClassFamilyIndex,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            _indexed_class_display_name(indexed_class, class_index)
            for indexed_class in indexed_classes
        )
    )


def _indexed_descendant_classes(
    class_index: ClassFamilyIndex,
    base_symbol: str,
) -> tuple[IndexedClass, ...]:
    return tuple(
        indexed_class
        for descendant_symbol in class_index.descendant_symbols(base_symbol)
        if (indexed_class := class_index.class_for(descendant_symbol)) is not None
    )


def _class_defines_property(node: ast.ClassDef, property_name: str) -> bool:
    return any(
        isinstance(statement, ast.FunctionDef)
        and statement.name == property_name
        and any(
            _ast_terminal_name(decorator) == "property"
            for decorator in statement.decorator_list
        )
        for statement in node.body
    )


def _constant_property_hook_groups(
    module: ParsedModule,
) -> tuple[ConstantPropertyHookGroup, ...]:
    grouped: dict[tuple[str, str], list[tuple[str, int, str]]] = defaultdict(list)
    class_defs_by_name = _module_class_defs_by_name(module)
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = tuple(
            name
            for name in _declared_base_names(node)
            if name not in _IGNORED_BASE_NAMES
        )
        if not base_names:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not any(
                _ast_terminal_name(decorator) == "property"
                for decorator in statement.decorator_list
            ):
                continue
            body = _trim_docstring_body(statement.body)
            if (
                len(body) != 1
                or not isinstance(body[0], ast.Return)
                or body[0].value is None
            ):
                continue
            returned = body[0].value
            if not _is_constant_hook_expression(returned):
                continue
            return_expression = ast.unparse(returned)
            for base_name in base_names:
                base_node = class_defs_by_name.get(base_name)
                if base_node is None or not _class_defines_property(
                    base_node, statement.name
                ):
                    continue
                grouped[(base_name, statement.name)].append(
                    (node.name, statement.lineno, return_expression)
                )
    return tuple(
        ConstantPropertyHookGroup(
            file_path=str(module.path),
            base_name=base_name,
            property_name=property_name,
            class_names=tuple(class_name for class_name, _, _ in ordered),
            line_numbers=tuple(line for _, line, _ in ordered),
            return_expressions=tuple(expression for _, _, expression in ordered),
        )
        for (base_name, property_name), items in sorted(grouped.items())
        if len(items) >= 2
        for ordered in [tuple(sorted(items, key=lambda item: (item[1], item[0])))]
    )


def _reflective_self_attribute_candidates(
    module: ParsedModule,
) -> tuple[ReflectiveSelfAttributeCandidate, ...]:
    candidates: list[ReflectiveSelfAttributeCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for subnode in _walk_nodes(statement):
                if not isinstance(subnode, ast.Call):
                    continue
                builtin_name = _ast_terminal_name(subnode.func)
                if builtin_name not in _REFLECTIVE_SELF_BUILTINS:
                    continue
                if len(subnode.args) < 2:
                    continue
                receiver, attribute_name_node = subnode.args[0], subnode.args[1]
                attribute_name = _constant_string(attribute_name_node)
                if not (
                    isinstance(receiver, ast.Name)
                    and receiver.id == "self"
                    and attribute_name is not None
                ):
                    continue
                candidates.append(
                    ReflectiveSelfAttributeCandidate(
                        file_path=str(module.path),
                        line=subnode.lineno,
                        subject_name=node.name,
                        name_family=(attribute_name,),
                        method_name=statement.name,
                        reflective_builtin=builtin_name,
                        attribute_name=attribute_name,
                    )
                )
    return tuple(candidates)


_HELPER_BACKED_METHOD_NAMES = frozenset(
    {
        "build_from_function",
        "build_scoped_function",
        "build_from_assign",
        "build_scoped_assign",
        "build_from_context",
    }
)


_NON_HELPER_CALL_NAMES = frozenset(
    {
        "all",
        "any",
        "bool",
        "dict",
        "frozenset",
        "int",
        "len",
        "list",
        "max",
        "min",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
    }
)


def _is_observation_spec_wrapper_class(node: ast.ClassDef) -> bool:
    if not node.name.endswith("ObservationSpec"):
        return False
    return any(
        base_name.endswith("ObservationSpec")
        for base_name in _declared_base_names(node)
    )


def _looks_like_helper_call_name(helper_name: str) -> bool:
    terminal = helper_name.rsplit(".", 1)[-1]
    return bool(
        terminal
        and terminal[0].islower()
        and terminal not in _NON_HELPER_CALL_NAMES
    )


def _helper_call_from_returned_value(node: ast.AST) -> tuple[str, bool] | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tuple"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
    ):
        helper_name = _call_display_name(node.args[0])
        if helper_name is None or not _looks_like_helper_call_name(helper_name):
            return None
        return (helper_name, True)
    if isinstance(node, ast.Call):
        helper_name = _call_display_name(node)
        if helper_name is None or not _looks_like_helper_call_name(helper_name):
            return None
        return (helper_name, False)
    return None


def _helper_backed_wrapper_kind(
    returned_value: ast.AST,
) -> str | None:
    helper_call = returned_value
    tuple_wrapped = False
    if (
        isinstance(returned_value, ast.Call)
        and isinstance(returned_value.func, ast.Name)
        and returned_value.func.id == "tuple"
        and len(returned_value.args) == 1
        and isinstance(returned_value.args[0], ast.Call)
    ):
        helper_call = returned_value.args[0]
        tuple_wrapped = True
    if not isinstance(helper_call, ast.Call):
        return None
    arguments = [ast.unparse(arg) for arg in helper_call.args]
    arguments.extend(
        f"{keyword.arg}={ast.unparse(keyword.value)}"
        for keyword in helper_call.keywords
        if keyword.arg is not None
    )
    wrapper_prefix = "tuple_wrapped" if tuple_wrapped else "direct"
    if not arguments:
        return wrapper_prefix
    return f"{wrapper_prefix}({', '.join(arguments)})"


def _is_helper_wrapper_prelude(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Assert):
        return True
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        return bool(
            isinstance(target, ast.Name)
            and isinstance(statement.value, ast.Attribute)
            and isinstance(statement.value.value, ast.Name)
            and statement.value.value.id == "observation"
        )
    if isinstance(statement, ast.If):
        return _if_returns_none_only(statement)
    return False


def _helper_backed_observation_spec_candidates(
    module: ParsedModule,
) -> tuple[HelperBackedObservationSpecCandidate, ...]:
    candidates: list[HelperBackedObservationSpecCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = _shared_record_base_names(node)
        if not base_names:
            continue
        for method in node.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if method.name.startswith("_"):
                continue
            body = _trim_docstring_body(method.body)
            if not body or len(body) > 4:
                continue
            if not all(
                _is_helper_wrapper_prelude(statement) for statement in body[:-1]
            ):
                continue
            tail = body[-1]
            if not isinstance(tail, ast.Return) or tail.value is None:
                continue
            helper_result = _helper_call_from_returned_value(tail.value)
            if helper_result is None:
                continue
            helper_name, _ = helper_result
            wrapper_kind = _helper_backed_wrapper_kind(tail.value)
            if wrapper_kind is None:
                continue
            candidates.append(
                HelperBackedObservationSpecCandidate(
                    file_path=str(module.path),
                    line=method.lineno,
                    subject_name=node.name,
                    name_family=(method.name, helper_name, wrapper_kind),
                    base_names=base_names,
                    method_name=method.name,
                    helper_name=helper_name,
                    wrapper_kind=wrapper_kind,
                    parameter_names=_parameter_names(method),
                )
            )
    return tuple(candidates)


def _helper_backed_observation_spec_group(
    module: ParsedModule,
) -> HelperBackedObservationSpecGroup | None:
    candidates = _helper_backed_observation_spec_candidates(module)
    grouped: dict[tuple[str, ...], list[HelperBackedObservationSpecCandidate]] = (
        defaultdict(list)
    )
    for candidate in candidates:
        grouped[candidate.base_names].append(candidate)
    items = max(
        (items for items in grouped.values() if len(items) >= 3),
        key=len,
        default=None,
    )
    if items is None:
        return None
    ordered = tuple(sorted(items, key=lambda item: (item.line, item.class_name)))
    return HelperBackedObservationSpecGroup(
        file_path=str(module.path),
        base_names=ordered[0].base_names,
        class_names=tuple(item.class_name for item in ordered),
        line_numbers=tuple(item.line for item in ordered),
        method_names=tuple(item.method_name for item in ordered),
        helper_names=tuple(item.helper_name for item in ordered),
        wrapper_kinds=tuple(item.wrapper_kind for item in ordered),
    )


def _guarded_wrapper_node_types(node: ast.If) -> tuple[str, ...] | None:
    test = node.test
    if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
        return None
    operand = test.operand
    if (
        not isinstance(operand, ast.Call)
        or _ast_terminal_name(operand.func) != "isinstance"
        or len(operand.args) != 2
    ):
        return None
    type_node = operand.args[1]
    if isinstance(type_node, ast.Tuple):
        node_types = tuple(ast.unparse(item) for item in type_node.elts)
    else:
        node_types = (ast.unparse(type_node),)
    return tuple(item for item in node_types if item)


def _guarded_wrapper_function_candidates(
    module: ParsedModule,
) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    candidates: list[tuple[str, int, tuple[str, ...]]] = []
    for statement in module.module.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        body = _trim_docstring_body(statement.body)
        while (
            body
            and isinstance(body[0], ast.Assign)
            and len(body[0].targets) == 1
            and isinstance(body[0].targets[0], ast.Name)
        ):
            body = body[1:]
        if len(body) != 2:
            continue
        guard, return_stmt = body
        if not isinstance(guard, ast.If) or not _if_returns_none_only(guard):
            continue
        if not isinstance(return_stmt, ast.Return) or return_stmt.value is None:
            continue
        node_types = _guarded_wrapper_node_types(guard)
        if not node_types:
            continue
        candidates.append((statement.name, statement.lineno, node_types))
    return tuple(candidates)


def _guarded_wrapper_spec_pairs(
    module: ParsedModule,
) -> tuple[GuardedWrapperSpecPair, ...]:
    wrapper_functions = {
        function_name: (lineno, node_types)
        for function_name, lineno, node_types in _guarded_wrapper_function_candidates(
            module
        )
    }
    pairs: list[GuardedWrapperSpecPair] = []
    for statement in module.module.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
            lineno = statement.lineno
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
            lineno = statement.lineno
        else:
            continue
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        constructor_name = _call_name(value.func)
        if constructor_name is None:
            continue
        referenced_functions = [
            keyword.value.id
            for keyword in value.keywords
            if keyword.arg is not None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id in wrapper_functions
        ]
        if len(referenced_functions) != 1:
            continue
        node_types_node = next(
            (
                keyword.value
                for keyword in value.keywords
                if keyword.arg == "node_types"
            ),
            None,
        )
        if node_types_node is None:
            continue
        if isinstance(node_types_node, ast.Tuple):
            node_types = tuple(ast.unparse(item) for item in node_types_node.elts)
        else:
            node_types = (ast.unparse(node_types_node),)
        function_name = referenced_functions[0]
        function_line, function_node_types = wrapper_functions[function_name]
        if tuple(node_types) != function_node_types:
            continue
        pairs.append(
            GuardedWrapperSpecPair(
                file_path=str(module.path),
                spec_name=target.id,
                spec_line=lineno,
                function_name=function_name,
                function_line=function_line,
                constructor_name=constructor_name,
                node_types=function_node_types,
            )
        )
    return tuple(pairs)


def _dynamic_self_field_selection_candidates(
    module: ParsedModule,
) -> tuple[DynamicSelfFieldSelectionCandidate, ...]:
    candidates: list[DynamicSelfFieldSelectionCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for subnode in _walk_nodes(statement):
                if not isinstance(subnode, ast.Call):
                    continue
                builtin_name = _ast_terminal_name(subnode.func)
                if builtin_name not in _REFLECTIVE_SELF_BUILTINS:
                    continue
                if len(subnode.args) < 2:
                    continue
                receiver, selector_node = subnode.args[0], subnode.args[1]
                if not isinstance(receiver, ast.Name) or receiver.id != "self":
                    continue
                if _constant_string(selector_node) is not None:
                    continue
                selector_expression = ast.unparse(selector_node)
                if not any(
                    token in selector_expression
                    for token in ("self.", "type(self).", "cls.")
                ):
                    continue
                candidates.append(
                    DynamicSelfFieldSelectionCandidate(
                        file_path=str(module.path),
                        line=subnode.lineno,
                        subject_name=node.name,
                        name_family=(selector_expression,),
                        method_name=statement.name,
                        reflective_builtin=builtin_name,
                        selector_expression=selector_expression,
                    )
                )
    return tuple(candidates)


def _class_list_registry_names(node: ast.ClassDef) -> tuple[str, ...]:
    registry_names: list[str] = []
    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and isinstance(statement.value, ast.List):
                registry_names.append(target.id)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and isinstance(statement.value, ast.List)
        ):
            registry_names.append(statement.target.id)
    return tuple(sorted(set(registry_names)))


def _registration_append_registry_name(
    node: ast.AST, registry_names: tuple[str, ...], owner_name: str
) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "append":
        return None
    if len(node.args) != 1 or not _looks_like_cls_registration_value(node.args[0]):
        return None
    target = node.func.value
    if not isinstance(target, ast.Attribute):
        return None
    if target.attr not in registry_names:
        return None
    if isinstance(target.value, ast.Name) and target.value.id in {"cls", _TYPE_NAME_LITERAL}:
        return target.attr
    if (
        isinstance(target.value, ast.Name)
        and target.value.id == owner_name
    ):
        return target.attr
    return None


def _looks_like_cls_registration_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "cls"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cast"
        and node.args
    ):
        return _looks_like_cls_registration_value(node.args[-1])
    return False


def _class_dict_get_attr_name(node: ast.AST) -> str | None:
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Attribute)
        or node.func.attr != "get"
        or len(node.args) != 1
    ):
        return None
    if not isinstance(node.func.value, ast.Attribute) or node.func.value.attr != "__dict__":
        return None
    if (
        not isinstance(node.func.value.value, ast.Name)
        or node.func.value.value.id != "cls"
    ):
        return None
    return _constant_string(node.args[0])


def _guarded_defined_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _class_dict_get_attr_name(node)
    if not isinstance(node, ast.Compare):
        return None
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    if not isinstance(node.ops[0], (ast.IsNot, ast.NotEq)):
        return None
    comparator = node.comparators[0]
    if not isinstance(comparator, ast.Constant) or comparator.value is not None:
        return None
    return _class_dict_get_attr_name(node.left)


def _guard_requires_concrete_subclass(node: ast.AST) -> bool:
    if not isinstance(node, ast.UnaryOp) or not isinstance(node.op, ast.Not):
        return False
    operand = node.operand
    return (
        isinstance(operand, ast.Call)
        and isinstance(operand.func, ast.Attribute)
        and isinstance(operand.func.value, ast.Name)
        and operand.func.value.id == "inspect"
        and operand.func.attr == "isabstract"
        and len(operand.args) == 1
        and isinstance(operand.args[0], ast.Name)
        and operand.args[0].id == "cls"
    )


def _manual_subclass_registration_sites(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    registry_names: tuple[str, ...],
    *,
    owner_name: str,
) -> tuple[_ManualSubclassRegistrationSite, ...]:
    sites: dict[str, _ManualSubclassRegistrationSite] = {}

    def walk_statements(
        statements: Sequence[ast.stmt], guard_stack: tuple[ast.AST, ...]
    ) -> None:
        for statement in statements:
            if isinstance(statement, ast.If):
                walk_statements(statement.body, (*guard_stack, statement.test))
                walk_statements(statement.orelse, guard_stack)
                continue
            for subnode in _walk_nodes(statement):
                registry_name = _registration_append_registry_name(
                    subnode, registry_names, owner_name
                )
                if registry_name is None:
                    continue
                guard_summary = (
                    " and ".join(ast.unparse(guard) for guard in guard_stack)
                    if guard_stack
                    else None
                )
                selector_attr_name = next(
                    (
                        attr_name
                        for guard in guard_stack
                        if (attr_name := _guarded_defined_attr_name(guard)) is not None
                    ),
                    None,
                )
                requires_concrete_subclass = any(
                    _guard_requires_concrete_subclass(guard) for guard in guard_stack
                )
                sites[registry_name] = _ManualSubclassRegistrationSite(
                    registry_name=registry_name,
                    guard_summary=guard_summary,
                    selector_attr_name=selector_attr_name,
                    requires_concrete_subclass=requires_concrete_subclass,
                )

    walk_statements(_trim_docstring_body(method.body), ())
    return tuple(sites[name] for name in sorted(sites))


def _uses_named_registry(
    node: ast.AST,
    *,
    registry_name: str,
    owner_names: frozenset[str],
) -> bool:
    if not isinstance(node, ast.Attribute) or node.attr != registry_name:
        return False
    if not isinstance(node.value, ast.Name):
        return False
    return node.value.id in owner_names


def _registry_consumer_locations(
    module: ParsedModule,
    node: ast.ClassDef,
    registry_name: str,
) -> tuple[SourceLocation, ...]:
    consumer_locations: list[SourceLocation] = []
    for method in _iter_class_methods(node):
        if method.name == "__init_subclass__":
            continue
        if any(
            _uses_named_registry(
                subnode,
                registry_name=registry_name,
                owner_names=frozenset({"cls", _TYPE_NAME_LITERAL, node.name}),
            )
            for subnode in _walk_nodes(method)
        ):
            consumer_locations.append(
                SourceLocation(str(module.path), method.lineno, f"{node.name}.{method.name}")
            )
    for qualname, function in _iter_named_functions(module):
        if "." in qualname:
            continue
        if any(
            _uses_named_registry(
                subnode,
                registry_name=registry_name,
                owner_names=frozenset({node.name}),
            )
            for subnode in _walk_nodes(function)
        ):
            consumer_locations.append(
                SourceLocation(str(module.path), function.lineno, qualname)
            )
    unique_locations = {
        (location.file_path, location.line, location.symbol): location
        for location in consumer_locations
    }
    return tuple(
        sorted(
            unique_locations.values(),
            key=lambda location: (location.line, location.symbol),
        )
    )


def _registered_descendant_classes(
    descendants: tuple[IndexedClass, ...],
    site: _ManualSubclassRegistrationSite,
) -> tuple[IndexedClass, ...]:
    if site.selector_attr_name is not None:
        return tuple(
            descendant
            for descendant in descendants
            if site.selector_attr_name
            in _class_direct_non_none_assignment_names(descendant.node)
        )
    if site.requires_concrete_subclass:
        return tuple(
            descendant
            for descendant in descendants
            if not _is_abstract_class(descendant.node)
        )
    return descendants


def _manual_concrete_subclass_roster_candidates(
    modules: list[ParsedModule], config: DetectorConfig
) -> tuple[ManualConcreteSubclassRosterCandidate, ...]:
    class_index = build_class_family_index(modules)
    modules_by_path = {str(module.path): module for module in modules}
    candidates: list[ManualConcreteSubclassRosterCandidate] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        node = indexed_class.node
        module = modules_by_path.get(indexed_class.file_path)
        if module is None:
            continue
        registry_names = _class_list_registry_names(node)
        if not registry_names:
            continue
        init_subclass = _class_method_named(node, "__init_subclass__")
        if init_subclass is None:
            continue
        descendants = _indexed_descendant_classes(class_index, indexed_class.symbol)
        if len(descendants) < config.min_registration_sites:
            continue
        consumer_locations_by_registry = {
            registry_name: _registry_consumer_locations(module, node, registry_name)
            for registry_name in registry_names
        }
        for site in _manual_subclass_registration_sites(
            init_subclass, registry_names, owner_name=node.name
        ):
            consumer_locations = consumer_locations_by_registry.get(
                site.registry_name, ()
            )
            if not consumer_locations:
                continue
            concrete_descendants = _registered_descendant_classes(
                descendants, site
            )
            if len(concrete_descendants) < config.min_registration_sites:
                continue
            candidates.append(
                ManualConcreteSubclassRosterCandidate(
                    file_path=indexed_class.file_path,
                    line=init_subclass.lineno,
                    class_name=_indexed_class_display_name(indexed_class, class_index),
                    registration_site=site,
                    consumer_locations=consumer_locations,
                    concrete_class_names=_indexed_class_display_names(
                        concrete_descendants,
                        class_index,
                    ),
                )
            )
    return tuple(candidates)


def _registered_type_match_assignment_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str] | None:
    body = _trim_docstring_body(list(method.body))
    assignment = next(
        (
            statement
            for statement in body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.ListComp)
        ),
        None,
    )
    if assignment is None:
        return None
    list_comp = assignment.value
    if len(list_comp.generators) != 1:
        return None
    generator = list_comp.generators[0]
    if (
        generator.is_async
        or not isinstance(generator.target, ast.Name)
        or not isinstance(list_comp.elt, ast.Name)
        or list_comp.elt.id != generator.target.id
    ):
        return None
    iter_call = generator.iter
    if not (
        isinstance(iter_call, ast.Call)
        and not iter_call.args
        and not iter_call.keywords
        and isinstance(iter_call.func, ast.Attribute)
        and isinstance(iter_call.func.value, ast.Name)
        and iter_call.func.value.id == "cls"
        and iter_call.func.attr == "registered_types"
    ):
        return None
    if len(generator.ifs) != 1:
        return None
    predicate = generator.ifs[0]
    if not (
        isinstance(predicate, ast.Call)
        and len(predicate.args) == 1
        and not predicate.keywords
        and isinstance(predicate.func, ast.Attribute)
        and isinstance(predicate.func.value, ast.Name)
        and predicate.func.value.id == generator.target.id
        and isinstance(predicate.args[0], ast.Name)
        and predicate.args[0].id in _parameter_names(method)
    ):
        return None
    return (
        assignment.targets[0].id,
        predicate.func.attr,
        predicate.args[0].id,
    )


def _is_selected_match_subscript(node: ast.AST, match_var_name: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == match_var_name
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == 0
    )


def _selection_guard_kind(node: ast.AST, match_var_name: str) -> str | None:
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name)
        and node.operand.id == match_var_name
    ):
        return "empty"
    if not isinstance(node, ast.Compare):
        return None
    if (
        not isinstance(node.left, ast.Call)
        or _ast_terminal_name(node.left.func) != "len"
        or len(node.left.args) != 1
        or not isinstance(node.left.args[0], ast.Name)
        or node.left.args[0].id != match_var_name
        or len(node.ops) != 1
        or len(node.comparators) != 1
        or not isinstance(node.comparators[0], ast.Constant)
        or not isinstance(node.comparators[0].value, int)
    ):
        return None
    comparator_value = node.comparators[0].value
    operator = node.ops[0]
    if isinstance(operator, ast.NotEq) and comparator_value == 1:
        return "not_exactly_one"
    if isinstance(operator, ast.Gt) and comparator_value == 1:
        return "ambiguous"
    if isinstance(operator, ast.Eq) and comparator_value == 0:
        return "empty"
    return None


def _predicate_selected_concrete_family_candidates(
    modules: list[ParsedModule], config: DetectorConfig
) -> tuple[PredicateSelectedConcreteFamilyCandidate, ...]:
    class_index = build_class_family_index(modules)
    candidates: list[PredicateSelectedConcreteFamilyCandidate] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        node = indexed_class.node
        assignments = _class_direct_assignments(node)
        if "_registered_types" not in assignments:
            continue
        descendants = tuple(
            descendant
            for descendant in _indexed_descendant_classes(
                class_index, indexed_class.symbol
            )
            if not _is_abstract_class(descendant.node)
        )
        if len(descendants) < config.min_registration_sites:
            continue
        for method in _iter_class_methods(node):
            if not _is_classmethod(method):
                continue
            selection_shape = _registered_type_match_assignment_shape(method)
            if selection_shape is None:
                continue
            match_var_name, predicate_method_name, context_param_name = selection_shape
            guard_kinds = {
                _selection_guard_kind(statement.test, match_var_name)
                for statement in _trim_docstring_body(list(method.body))
                if isinstance(statement, ast.If)
            }
            has_exact_guard = "not_exactly_one" in guard_kinds or (
                "empty" in guard_kinds and "ambiguous" in guard_kinds
            )
            if not has_exact_guard:
                continue
            if not any(
                _is_selected_match_subscript(subnode, match_var_name)
                for subnode in _walk_nodes(method)
            ):
                continue
            candidates.append(
                PredicateSelectedConcreteFamilyCandidate(
                    file_path=indexed_class.file_path,
                    line=method.lineno,
                    class_name=_indexed_class_display_name(indexed_class, class_index),
                    selector_method_name=method.name,
                    predicate_method_name=predicate_method_name,
                    context_param_name=context_param_name,
                    concrete_class_names=_indexed_class_display_names(
                        descendants,
                        class_index,
                    ),
                )
            )
    return tuple(candidates)


def _mirrored_leaf_family_map(
    descendants: tuple[IndexedClass, ...],
    *,
    axis_prefix_tokens: tuple[str, ...],
) -> dict[str, IndexedClass]:
    leaf_map: dict[str, IndexedClass] = {}
    for descendant in descendants:
        tokens = _ordered_class_name_tokens(descendant.simple_name)
        if (
            len(tokens) <= len(axis_prefix_tokens)
            or tokens[: len(axis_prefix_tokens)] != axis_prefix_tokens
        ):
            continue
        family_tokens = tokens[len(axis_prefix_tokens) :]
        if not family_tokens:
            continue
        family_name = " ".join(family_tokens)
        leaf_map.setdefault(family_name, descendant)
    return leaf_map


def _parallel_mirrored_leaf_family_candidates(
    modules: list[ParsedModule], config: DetectorConfig
) -> tuple[ParallelMirroredLeafFamilyCandidate, ...]:
    class_index = build_class_family_index(modules)
    min_shared_families = max(3, config.min_registration_sites)
    root_candidates: list[tuple[IndexedClass, tuple[str, ...], tuple[IndexedClass, ...]]] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        assignments = _class_direct_assignments(indexed_class.node)
        if "_registered_types" not in assignments:
            continue
        abstract_methods = _abstract_method_names(indexed_class.node)
        if not abstract_methods:
            continue
        concrete_descendants = tuple(
            descendant
            for descendant in _indexed_descendant_classes(
                class_index, indexed_class.symbol
            )
            if not _is_abstract_class(descendant.node)
        )
        if len(concrete_descendants) < min_shared_families:
            continue
        root_candidates.append(
            (indexed_class, abstract_methods, concrete_descendants)
        )

    candidates: list[ParallelMirroredLeafFamilyCandidate] = []
    for (
        left_root,
        left_contract_methods,
        left_descendants,
    ), (
        right_root,
        right_contract_methods,
        right_descendants,
    ) in combinations(root_candidates, 2):
        shared_contract_methods = tuple(
            sorted(set(left_contract_methods) & set(right_contract_methods))
        )
        if not shared_contract_methods:
            continue
        left_tokens = _ordered_class_name_tokens(left_root.simple_name)
        right_tokens = _ordered_class_name_tokens(right_root.simple_name)
        shared_root_suffix = _shared_ordered_suffix(left_tokens, right_tokens)
        if not shared_root_suffix:
            continue
        left_axis_prefix = left_tokens[: len(left_tokens) - len(shared_root_suffix)]
        right_axis_prefix = right_tokens[: len(right_tokens) - len(shared_root_suffix)]
        if (
            not left_axis_prefix
            or not right_axis_prefix
            or left_axis_prefix == right_axis_prefix
        ):
            continue
        left_leaf_map = _mirrored_leaf_family_map(
            left_descendants,
            axis_prefix_tokens=left_axis_prefix,
        )
        right_leaf_map = _mirrored_leaf_family_map(
            right_descendants,
            axis_prefix_tokens=right_axis_prefix,
        )
        if not left_leaf_map or not right_leaf_map:
            continue
        shared_leaf_families = tuple(
            sorted(set(left_leaf_map) & set(right_leaf_map))
        )
        if len(shared_leaf_families) < max(
            min_shared_families,
            min(len(left_leaf_map), len(right_leaf_map)) // 2,
        ):
            continue
        left_leaf_evidence = tuple(
            SourceLocation(
                left_leaf_map[family_name].file_path,
                left_leaf_map[family_name].line,
                _indexed_class_display_name(left_leaf_map[family_name], class_index),
            )
            for family_name in shared_leaf_families
        )
        right_leaf_evidence = tuple(
            SourceLocation(
                right_leaf_map[family_name].file_path,
                right_leaf_map[family_name].line,
                _indexed_class_display_name(right_leaf_map[family_name], class_index),
            )
            for family_name in shared_leaf_families
        )
        candidates.append(
            ParallelMirroredLeafFamilyCandidate(
                left=MirroredLeafFamilySide(
                    file_path=left_root.file_path,
                    line=left_root.line,
                    root_name=_indexed_class_display_name(left_root, class_index),
                    leaf_evidence=left_leaf_evidence,
                ),
                right=MirroredLeafFamilySide(
                    file_path=right_root.file_path,
                    line=right_root.line,
                    root_name=_indexed_class_display_name(right_root, class_index),
                    leaf_evidence=right_leaf_evidence,
                ),
                contract_method_names=shared_contract_methods,
                shared_leaf_family_names=shared_leaf_families,
            )
        )
    return tuple(candidates)


def _reflective_lookup_shape(
    node: ast.AST,
) -> tuple[str, str, ast.AST] | None:
    if isinstance(node, ast.Call):
        builtin_name = _ast_terminal_name(node.func)
        if builtin_name == _GETATTR_BUILTIN and len(node.args) >= 2:
            selector_node = node.args[1]
            if _constant_string(selector_node) is None:
                return (_GETATTR_BUILTIN, ast.unparse(node.args[0]), selector_node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) >= 1
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "__dict__"
        ):
            selector_node = node.args[0]
            if _constant_string(selector_node) is None:
                return ("dict.get", ast.unparse(node.func.value.value), selector_node)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id in {"globals", "locals"}
        and not node.value.args
        and not node.value.keywords
        and _constant_string(node.slice) is None
    ):
        return (f"{node.value.func.id}[]", f"{node.value.func.id}()", node.slice)
    return None


def _string_backed_reflective_nominal_lookup_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[StringBackedReflectiveNominalLookupCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    class_string_assignments = {
        class_name: _class_direct_constant_string_assignments(node)
        for class_name, node in class_defs_by_name.items()
    }
    candidate_map: dict[
        tuple[str, str, str, str, str], StringBackedReflectiveNominalLookupCandidate
    ] = {}
    for class_name, node in sorted(class_defs_by_name.items()):
        descendants = _descendant_class_names(class_defs_by_name, class_name)
        if len(descendants) < config.min_reflective_selector_values:
            continue
        for method in _iter_class_methods(node):
            for subnode in _walk_nodes(method):
                lookup_shape = _reflective_lookup_shape(subnode)
                if lookup_shape is None:
                    continue
                lookup_kind, receiver_expression, selector_node = lookup_shape
                selector_attr_name = _selector_attribute_name(selector_node)
                if selector_attr_name is None:
                    continue
                concrete_class_names = tuple(
                    descendant
                    for descendant in descendants
                    if selector_attr_name in class_string_assignments[descendant]
                )
                if (
                    len(concrete_class_names)
                    < config.min_reflective_selector_values
                ):
                    continue
                selector_values = tuple(
                    sorted(
                        {
                            class_string_assignments[descendant][selector_attr_name]
                            for descendant in concrete_class_names
                        }
                    )
                )
                if len(selector_values) < config.min_reflective_selector_values:
                    continue
                candidate = StringBackedReflectiveNominalLookupCandidate(
                    file_path=str(module.path),
                    line=subnode.lineno,
                    class_name=class_name,
                    method_name=method.name,
                    selector_attr_name=selector_attr_name,
                    lookup_kind=lookup_kind,
                    receiver_expression=receiver_expression,
                    concrete_class_names=concrete_class_names,
                    selector_values=selector_values,
                )
                candidate_map[
                    (
                        class_name,
                        method.name,
                        selector_attr_name,
                        lookup_kind,
                        receiver_expression,
                    )
                ] = candidate
    return tuple(
        sorted(
            candidate_map.values(),
            key=lambda item: (item.file_path, item.line, item.class_name, item.method_name),
        )
    )


def _param_backed_name(expr: ast.AST, parameter_names: set[str]) -> str | None:
    if isinstance(expr, ast.Name) and expr.id in parameter_names:
        return expr.id
    if isinstance(expr, ast.IfExp):
        body_name = _param_backed_name(expr.body, parameter_names)
        orelse_name = _param_backed_name(expr.orelse, parameter_names)
        if body_name is not None and orelse_name is None:
            return body_name
        if orelse_name is not None and body_name is None:
            return orelse_name
        if body_name == orelse_name:
            return body_name
    if isinstance(expr, ast.BoolOp):
        names = {
            name
            for value in expr.values
            for name in (_param_backed_name(value, parameter_names),)
            if name is not None
        }
        if len(names) == 1:
            return next(iter(names))
    return None


def _class_init_concrete_param_backed_attrs(node: ast.ClassDef) -> dict[str, str]:
    init_method = _class_method_named(node, "__init__")
    if init_method is None:
        return {}
    parameter_type_names = {
        argument.arg: _annotation_type_names(argument.annotation)
        for argument in (
            tuple(init_method.args.posonlyargs)
            + tuple(init_method.args.args)
            + tuple(init_method.args.kwonlyargs)
        )
        if argument.annotation is not None
    }
    parameter_names = set(parameter_type_names)
    attr_type_names: dict[str, str] = {}
    for subnode in _walk_nodes(init_method):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(subnode, ast.Assign) and len(subnode.targets) == 1:
            target = subnode.targets[0]
            value = subnode.value
        elif isinstance(subnode, ast.AnnAssign):
            target = subnode.target
            value = subnode.value
        attr_name = None if target is None else _self_attr_name(target)
        if attr_name is None or value is None:
            continue
        param_name = _param_backed_name(value, parameter_names)
        if param_name is None:
            continue
        type_names = parameter_type_names.get(param_name, ())
        if len(type_names) != 1:
            continue
        attr_type_names.setdefault(attr_name, type_names[0])
    return attr_type_names


def _method_aliases_to_self_attrs(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for subnode in _walk_nodes(method):
            target: ast.AST | None = None
            value: ast.AST | None = None
            if isinstance(subnode, ast.Assign) and len(subnode.targets) == 1:
                target = subnode.targets[0]
                value = subnode.value
            elif isinstance(subnode, ast.AnnAssign):
                target = subnode.target
                value = subnode.value
            if not (isinstance(target, ast.Name) and value is not None):
                continue
            attr_name = None
            if isinstance(value, ast.Attribute):
                attr_name = _self_attr_name(value)
            elif isinstance(value, ast.Name):
                attr_name = aliases.get(value.id)
            if attr_name is None or aliases.get(target.id) == attr_name:
                continue
            aliases[target.id] = attr_name
            changed = True
    return aliases


def _receiver_self_attr_name(
    node: ast.AST, aliases: dict[str, str]
) -> str | None:
    if isinstance(node, ast.Attribute):
        return _self_attr_name(node)
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    return None


def _concrete_config_field_probe_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[ConcreteConfigFieldProbeCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    config_field_names = {
        class_name: {
            field_name
            for field_name, _ in _typed_field_map(node)
        }
        for class_name, node in class_defs_by_name.items()
    }
    candidates: list[ConcreteConfigFieldProbeCandidate] = []
    for class_name, node in sorted(class_defs_by_name.items()):
        concrete_config_attrs = _class_init_concrete_param_backed_attrs(node)
        if not concrete_config_attrs:
            continue
        for method in _iter_class_methods(node):
            aliases = _method_aliases_to_self_attrs(method)
            grouped_missing_fields: dict[
                tuple[str, str], set[str]
            ] = defaultdict(set)
            grouped_probe_builtins: dict[
                tuple[str, str], set[str]
            ] = defaultdict(set)
            grouped_lines: dict[tuple[str, str], int] = {}
            for subnode in _walk_nodes(method):
                if not isinstance(subnode, ast.Call):
                    continue
                builtin_name = _ast_terminal_name(subnode.func)
                if (
                    builtin_name not in {_GETATTR_BUILTIN, _HASATTR_BUILTIN}
                    or len(subnode.args) < 2
                ):
                    continue
                probed_field_name = _constant_string(subnode.args[1])
                if probed_field_name is None:
                    continue
                config_attr_name = _receiver_self_attr_name(subnode.args[0], aliases)
                if config_attr_name is None:
                    continue
                config_type_name = concrete_config_attrs.get(config_attr_name)
                if config_type_name is None:
                    continue
                config_node = class_defs_by_name.get(config_type_name)
                if config_node is None or _class_method_named(config_node, "__getattr__") is not None:
                    continue
                declared_field_names = config_field_names.get(config_type_name, set())
                if not declared_field_names or probed_field_name in declared_field_names:
                    continue
                key = (config_attr_name, config_type_name)
                grouped_missing_fields[key].add(probed_field_name)
                grouped_probe_builtins[key].add(builtin_name)
                grouped_lines.setdefault(key, subnode.lineno)
            for (config_attr_name, config_type_name), missing_fields in sorted(
                grouped_missing_fields.items()
            ):
                if len(missing_fields) < config.min_attribute_probes:
                    continue
                candidates.append(
                    ConcreteConfigFieldProbeCandidate(
                        file_path=str(module.path),
                        line=grouped_lines[(config_attr_name, config_type_name)],
                        class_name=class_name,
                        method_name=method.name,
                        config_attr_name=config_attr_name,
                        config_type_name=config_type_name,
                        missing_field_names=tuple(sorted(missing_fields)),
                        probe_builtin_names=tuple(
                            sorted(grouped_probe_builtins[(config_attr_name, config_type_name)])
                        ),
                    )
                )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.line, item.class_name, item.method_name),
        )
    )


_DECLARATIVE_FAMILY_ASSIGNMENT_NAMES = frozenset(
    {"item_type", "spec", "spec_root", "literal_kind"}
)
_DECLARATIVE_FAMILY_DEFINITION_BASE_NAMES = frozenset(
    {
        "SingleShapeFamilyDefinition",
        "RegisteredShapeFamilyDefinition",
        "RegisteredObservationFamilyDefinition",
        "TypedLiteralObservationFamilyDefinition",
    }
)


def _module_alias_assignments(module: ParsedModule) -> dict[str, tuple[str, int, str]]:
    aliases: dict[str, tuple[str, int, str]] = {}
    for statement in _trim_docstring_body(module.module.body):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = statement.value
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "family_type"
            and isinstance(value.value, ast.Name)
        ):
            continue
        aliases[target.id] = (value.value.id, statement.lineno, value.attr)
    return aliases


def _module_string_sequence_assignments(
    module: ParsedModule,
) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    assignments: list[tuple[str, int, tuple[str, ...]]] = []
    for statement in _trim_docstring_body(module.module.body):
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is None or value is None:
            continue
        if not isinstance(value, (ast.Tuple, ast.List)):
            continue
        string_items = tuple(
            item.value
            for item in value.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )
        if len(string_items) != len(value.elts) or len(string_items) < 3:
            continue
        assignments.append((target_name, statement.lineno, string_items))
    return tuple(assignments)


def _is_simple_classvar_value(node: ast.AST) -> bool:
    if isinstance(node, (ast.Name, ast.Attribute, ast.Constant)):
        return True
    if isinstance(node, ast.Tuple):
        return all(_is_simple_classvar_value(item) for item in node.elts)
    return False


def _classvar_assignment_names(node: ast.ClassDef) -> tuple[str, ...] | None:
    assigned_names: list[str] = []
    for statement in _trim_docstring_body(node.body):
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if not isinstance(target, ast.Name) or not _is_simple_classvar_value(
                statement.value
            ):
                return None
            assigned_names.append(target.id)
            continue
        if isinstance(statement, ast.AnnAssign):
            if not isinstance(statement.target, ast.Name) or statement.value is None:
                return None
            if not _is_simple_classvar_value(statement.value):
                return None
            assigned_names.append(statement.target.id)
            continue
        return None
    return tuple(assigned_names)


def _classvar_only_sibling_leaf_candidates(
    module: ParsedModule,
) -> tuple[DeclarativeFamilyLeafCandidate, ...]:
    candidates: list[DeclarativeFamilyLeafCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = tuple(
            name
            for name in _declared_base_names(node)
            if name not in _IGNORED_ANCESTOR_NAMES
        )
        if not base_names:
            continue
        assigned_names = _classvar_assignment_names(node)
        if assigned_names is None:
            continue
        if len(assigned_names) < 1 or len(assigned_names) > 4:
            continue
        if len(_trim_docstring_body(node.body)) != len(assigned_names):
            continue
        candidates.append(
            DeclarativeFamilyLeafCandidate(
                file_path=str(module.path),
                line=node.lineno,
                subject_name=node.name,
                name_family=assigned_names,
                base_names=base_names,
                assigned_names=assigned_names,
            )
        )
    return tuple(candidates)


def _classvar_only_sibling_leaf_groups(
    module: ParsedModule,
) -> tuple[DeclarativeFamilyBoilerplateGroup, ...]:
    grouped: dict[
        tuple[tuple[str, ...], tuple[str, ...]],
        list[DeclarativeFamilyLeafCandidate],
    ] = defaultdict(list)
    for candidate in _classvar_only_sibling_leaf_candidates(module):
        grouped[(candidate.base_names, candidate.assigned_names)].append(candidate)
    return tuple(
        DeclarativeFamilyBoilerplateGroup(
            file_path=str(module.path),
            base_names=base_names,
            assigned_names=assigned_names,
            class_names=tuple(item.subject_name for item in items),
            line_numbers=tuple(item.line for item in items),
        )
        for (base_names, assigned_names), items in sorted(grouped.items())
        if len(items) >= 3
    )


def _type_indexed_definition_boilerplate_groups(
    module: ParsedModule,
) -> tuple[TypeIndexedDefinitionBoilerplateGroup, ...]:
    alias_assignments = _module_alias_assignments(module)
    grouped: dict[
        tuple[tuple[str, ...], tuple[str, ...]],
        list[tuple[str, str, int]],
    ] = defaultdict(list)
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef) or not node.name.endswith("Definition"):
            continue
        base_names = tuple(
            name
            for name in _declared_base_names(node)
            if name not in _IGNORED_ANCESTOR_NAMES
        )
        if not base_names or not any(
            name.endswith("Definition") for name in base_names
        ):
            continue
        assigned_names = _classvar_assignment_names(node)
        if assigned_names is None:
            continue
        if not set(assigned_names) & _DECLARATIVE_FAMILY_ASSIGNMENT_NAMES:
            continue
        alias_name = next(
            (
                alias_name
                for alias_name, (
                    definition_name,
                    _,
                    attr_name,
                ) in alias_assignments.items()
                if definition_name == node.name and attr_name == "family_type"
            ),
            None,
        )
        if alias_name is None:
            continue
        grouped[(base_names, assigned_names)].append(
            (node.name, alias_name, node.lineno)
        )
    return tuple(
        TypeIndexedDefinitionBoilerplateGroup(
            file_path=str(module.path),
            base_names=base_names,
            definition_class_names=tuple(item[0] for item in ordered),
            alias_names=tuple(item[1] for item in ordered),
            line_numbers=tuple(item[2] for item in ordered),
            assigned_names=assigned_names,
        )
        for (base_names, assigned_names), items in sorted(grouped.items())
        if len(items) >= 3
        for ordered in [tuple(sorted(items, key=lambda item: (item[2], item[0])))]
    )


def _derivable_nominal_root_names(
    shapes: Sequence[NominalAuthorityShape],
) -> tuple[str, ...]:
    root_counts: Counter[str] = Counter()
    for shape in shapes:
        root_counts.update(
            name
            for name in {*shape.declared_base_names, *shape.ancestor_names}
            if name not in _IGNORED_ANCESTOR_NAMES and name != shape.class_name
        )
    return tuple(
        sorted(root_name for root_name, count in root_counts.items() if count >= 3)
    )


def _derived_export_surface_candidates(
    module: ParsedModule,
) -> tuple[DerivedExportSurfaceCandidate, ...]:
    index = NominalAuthorityIndex((module,))
    candidates: list[DerivedExportSurfaceCandidate] = []
    for export_symbol, line, exported_names in _module_string_sequence_assignments(
        module
    ):
        local_shapes = [
            shapes[0]
            for exported_name in exported_names
            if (shapes := index.shapes_named(exported_name))
            and shapes[0].file_path == str(module.path)
        ]
        if len(local_shapes) < 6 or len(local_shapes) * 5 < len(exported_names) * 4:
            continue
        root_names = _derivable_nominal_root_names(local_shapes)
        if not root_names:
            continue
        candidates.append(
            DerivedExportSurfaceCandidate(
                file_path=str(module.path),
                export_symbol=export_symbol,
                line=line,
                exported_names=exported_names,
                derivable_root_names=root_names,
            )
        )
    return tuple(candidates)


def _module_public_source_names(module: ParsedModule) -> tuple[str, ...]:
    names: set[str] = set()
    for statement in _trim_docstring_body(module.module.body):
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not statement.name.startswith("_"):
                names.add(statement.name)
            continue
        if isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                exported_name = alias.asname or alias.name
                if not exported_name.startswith("_"):
                    names.add(exported_name)
            continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                names.add(target.id)
    return tuple(sorted(names))


def _manual_public_api_surface_candidates(
    module: ParsedModule,
) -> tuple[ManualPublicApiSurfaceCandidate, ...]:
    public_source_names = set(_module_public_source_names(module))
    candidates: list[ManualPublicApiSurfaceCandidate] = []
    for export_symbol, line, exported_names in _module_string_sequence_assignments(
        module
    ):
        if export_symbol != "__all__":
            continue
        if len(exported_names) < 4:
            continue
        if not set(exported_names) <= public_source_names:
            continue
        candidates.append(
            ManualPublicApiSurfaceCandidate(
                file_path=str(module.path),
                export_symbol=export_symbol,
                line=line,
                exported_names=exported_names,
                source_name_count=len(public_source_names),
            )
        )
    return tuple(candidates)


def _dict_key_kind(value: ast.AST) -> str | None:
    if isinstance(value, ast.Name):
        return "type_name"
    if isinstance(value, ast.Attribute):
        return "enum_member"
    if isinstance(value, ast.Constant):
        return type(value.value).__name__
    return None


def _derived_indexed_surface_candidates(
    module: ParsedModule,
) -> tuple[DerivedIndexedSurfaceCandidate, ...]:
    index = NominalAuthorityIndex((module,))
    candidates: list[DerivedIndexedSurfaceCandidate] = []
    for statement in _trim_docstring_body(module.module.body):
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is None or not isinstance(value, ast.Dict):
            continue
        if len(value.keys) < 3 or len(value.keys) != len(value.values):
            continue
        key_kinds = {
            key_kind
            for key_kind in (
                _dict_key_kind(key) for key in value.keys if key is not None
            )
            if key_kind is not None
        }
        if len(key_kinds) != 1:
            continue
        value_names = tuple(
            item.id
            for item in value.values
            if isinstance(item, ast.Name)
            and (shapes := index.shapes_named(item.id))
            and shapes[0].file_path == str(module.path)
        )
        if len(value_names) != len(value.values):
            continue
        local_shapes = [index.shapes_named(value_name)[0] for value_name in value_names]
        shared_roots = _derivable_nominal_root_names(local_shapes)
        if not shared_roots:
            continue
        candidates.append(
            DerivedIndexedSurfaceCandidate(
                file_path=str(module.path),
                surface_name=target_name,
                line=statement.lineno,
                key_kind=next(iter(key_kinds)),
                value_names=value_names,
                derivable_root_names=shared_roots,
            )
        )
    return tuple(candidates)


def _registered_surface_roots(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    calls: list[ast.Call] = []

    def collect_calls(current: ast.AST) -> bool:
        if isinstance(current, ast.BinOp) and isinstance(current.op, ast.Add):
            return collect_calls(current.left) and collect_calls(current.right)
        if isinstance(current, ast.Call):
            calls.append(current)
            return True
        return False

    if not collect_calls(node) or len(calls) < 2:
        return None
    accessor_names = {
        call.func.attr
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and not call.args
        and not call.keywords
    }
    if len(accessor_names) != 1:
        return None
    accessor_name = next(iter(accessor_names))
    root_names = tuple(
        sorted(
            call.func.value.id
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
        )
    )
    return (accessor_name, root_names)


def _registered_union_surface_candidates(
    module: ParsedModule,
) -> tuple[RegisteredUnionSurfaceCandidate, ...]:
    candidates: list[RegisteredUnionSurfaceCandidate] = []
    class_defs_by_name = {
        node.name: node
        for node in module.module.body
        if isinstance(node, ast.ClassDef)
    }
    for node in _walk_nodes(module.module):
        owner_name = "<module>"
        value: ast.AST | None = None
        line = getattr(node, "lineno", 1)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner_name = node.name
            for statement in _trim_docstring_body(node.body):
                if isinstance(statement, ast.For):
                    value = statement.iter
                    line = statement.lineno
                    break
                if isinstance(statement, ast.Assign):
                    value = statement.value
                    line = statement.lineno
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                owner_name = target.id
                value = node.value
                line = node.lineno
        if value is None:
            continue
        registered_surface = _registered_surface_roots(value)
        if registered_surface is None:
            continue
        accessor_name, root_names = registered_surface
        if len(root_names) < 2:
            continue
        root_nodes = [class_defs_by_name.get(root_name) for root_name in root_names]
        if any(root_node is None for root_node in root_nodes):
            continue
        if any(
            (method := _class_method_named(cast(ast.ClassDef, root_node), accessor_name))
            is None
            or not _is_classmethod(method)
            for root_node in root_nodes
        ):
            continue
        candidates.append(
            RegisteredUnionSurfaceCandidate(
                file_path=str(module.path),
                line=line,
                owner_name=owner_name,
                accessor_name=accessor_name,
                root_names=root_names,
            )
        )
    return tuple(candidates)


def _type_name_set(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (ast.unparse(node),)
    if isinstance(node, ast.Tuple):
        return tuple(
            sorted(
                {
                    type_name
                    for element in node.elts
                    for type_name in _type_name_set(element)
                }
            )
        )
    return ()


def _export_policy_role_names(node: ast.FunctionDef) -> tuple[str, ...]:
    body_text = "\n".join(ast.unparse(statement) for statement in node.body)
    roles: set[str] = set()
    if "name.startswith('_')" in body_text:
        roles.add("exclude_private")
    if (
        "__module__ != __name__" in body_text
        or "getattr(value, '__module__', None) == __name__" in body_text
    ):
        roles.add("module_local")
    for current in _walk_nodes(node):
        if not isinstance(current, ast.Call):
            continue
        call_name = _ast_terminal_name(current.func)
        if call_name == "isinstance" and len(current.args) == 2:
            type_names = set(_type_name_set(current.args[1]))
            if _TYPE_NAME_LITERAL in type_names:
                roles.add("type_only")
                type_names.discard(_TYPE_NAME_LITERAL)
            elif type_names:
                roles.add("value_type_filter")
            if any(type_name.endswith("Enum") for type_name in type_names):
                roles.add("enum_ok")
        elif call_name == "callable" and len(current.args) == 1:
            roles.add("callable_ok")
        elif call_name == "issubclass" and len(current.args) == 2:
            roles.add("subclass_constraint")
            type_names = set(_type_name_set(current.args[1]))
            if any(type_name.endswith("Enum") for type_name in type_names):
                roles.add("enum_ok")
        elif call_name == "isabstract":
            roles.add("exclude_abstract")
    return tuple(sorted(roles))


def _export_policy_root_type_names(node: ast.FunctionDef) -> tuple[str, ...]:
    root_type_names: set[str] = set()
    for current in _walk_nodes(node):
        if not isinstance(current, ast.Call):
            continue
        if _ast_terminal_name(current.func) != "issubclass" or len(current.args) != 2:
            continue
        root_type_names.update(
            type_name
            for type_name in _type_name_set(current.args[1])
            if type_name != _TYPE_NAME_LITERAL
        )
    return tuple(sorted(root_type_names))


def _module_export_policy_predicate_candidate(
    module: ParsedModule,
) -> ExportPolicyPredicateCandidate | None:
    exported_predicate_names: set[str] = set()
    for statement in _trim_docstring_body(module.module.body):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not (isinstance(target, ast.Name) and target.id == "__all__"):
            continue
        value = statement.value
        if (
            not isinstance(value, ast.Call)
            or _ast_terminal_name(value.func) != "sorted"
        ):
            continue
        if len(value.args) != 1 or not isinstance(value.args[0], ast.GeneratorExp):
            continue
        generator = value.args[0]
        if not generator.generators or len(generator.generators[0].ifs) != 1:
            continue
        condition = generator.generators[0].ifs[0]
        if not isinstance(condition, ast.Call) or not isinstance(
            condition.func, ast.Name
        ):
            continue
        exported_predicate_names.add(condition.func.id)
    if len(exported_predicate_names) != 1:
        return None
    predicate_name = next(iter(exported_predicate_names))
    predicate_node = next(
        (
            statement
            for statement in _trim_docstring_body(module.module.body)
            if isinstance(statement, ast.FunctionDef)
            and statement.name == predicate_name
        ),
        None,
    )
    if predicate_node is None or len(predicate_node.args.args) != 2:
        return None
    role_names = _export_policy_role_names(predicate_node)
    if len(role_names) < 2:
        return None
    root_type_names = _export_policy_root_type_names(predicate_node)
    return ExportPolicyPredicateCandidate(
        file_path=str(module.path),
        line=predicate_node.lineno,
        subject_name=predicate_name,
        name_family=role_names,
        role_names=role_names,
        root_type_names=root_type_names,
    )


def _registry_materialization_kind(node: ast.FunctionDef) -> str | None:
    body = _trim_docstring_body(node.body)
    append_calls = [
        call
        for call in _walk_nodes(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "ordered"
        and len(call.args) == 1
    ]
    if len(append_calls) != 1:
        return None
    arg = append_calls[0].args[0]
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
        return "instantiate"
    if isinstance(arg, ast.Name):
        return _TYPE_NAME_LITERAL
    return None


def _registry_attribute_name(node: ast.FunctionDef) -> str | None:
    attribute_names = {
        attribute_name
        for current in _walk_nodes(node)
        if isinstance(current, ast.Call)
        and isinstance(current.func, ast.Attribute)
        and current.func.attr == "get"
        and isinstance(current.func.value, ast.Attribute)
        and current.func.value.attr == "__dict__"
        and isinstance(current.func.value.value, ast.Name)
        and current.func.value.value.id == "current"
        and len(current.args) == 1
        and (attribute_name := _constant_string(current.args[0])) is not None
    }
    if len(attribute_names) != 1:
        return None
    return next(iter(attribute_names))


def _is_registry_traversal_method(node: ast.FunctionDef) -> bool:
    if not _is_classmethod(node):
        return False
    if _registry_attribute_name(node) is None:
        return False
    body_text = "\n".join(ast.unparse(statement) for statement in node.body)
    required_fragments = (
        "cls.__subclasses__()",
        "current.__subclasses__()",
        "seen.add",
        "ordered.append",
        "return tuple(ordered)",
    )
    return all(fragment in body_text for fragment in required_fragments)


def _registry_traversal_group(module: ParsedModule) -> RegistryTraversalGroup | None:
    methods: list[tuple[str, str, int, str]] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not _is_registry_traversal_method(statement):
                continue
            materialization_kind = _registry_materialization_kind(statement)
            registry_attribute_name = _registry_attribute_name(statement)
            if materialization_kind is None or registry_attribute_name is None:
                continue
            methods.append(
                (
                    node.name,
                    statement.name,
                    statement.lineno,
                    materialization_kind,
                    registry_attribute_name,
                )
            )
    if len(methods) < 2:
        return None
    ordered = tuple(sorted(methods, key=lambda item: (item[2], item[0], item[1])))
    return RegistryTraversalGroup(
        file_path=str(module.path),
        class_names=tuple(item[0] for item in ordered),
        method_names=tuple(item[1] for item in ordered),
        line_numbers=tuple(item[2] for item in ordered),
        materialization_kinds=tuple(item[3] for item in ordered),
        registry_attribute_names=tuple(item[4] for item in ordered),
    )


def _declarative_family_boilerplate_groups(
    module: ParsedModule,
) -> tuple[DeclarativeFamilyBoilerplateGroup, ...]:
    return _classvar_only_sibling_leaf_groups(module)


def _constructor_return_call(node: ast.FunctionDef) -> ast.Call | None:
    body = _trim_docstring_body(node.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    returned = body[0].value
    if not isinstance(returned, ast.Call):
        return None
    if not isinstance(returned.func, ast.Name) or returned.func.id != "cls":
        return None
    return returned


def _source_type_name_for_constructor(node: ast.FunctionDef) -> str | None:
    if len(node.args.args) < 3:
        return None
    source_arg = node.args.args[2]
    if source_arg.annotation is None:
        return source_arg.arg
    return ast.unparse(source_arg.annotation)


def _alternate_constructor_family_groups(
    module: ParsedModule,
) -> tuple[AlternateConstructorFamilyGroup, ...]:
    groups: list[AlternateConstructorFamilyGroup] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        constructor_methods: list[tuple[ast.FunctionDef, ast.Call, str]] = []
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not statement.name.startswith("from_") or not _is_classmethod(statement):
                continue
            return_call = _constructor_return_call(statement)
            if return_call is None:
                continue
            source_type_name = _source_type_name_for_constructor(statement)
            if source_type_name is None:
                continue
            constructor_methods.append((statement, return_call, source_type_name))
        if len(constructor_methods) < 3:
            continue
        keyword_sets = [
            {keyword.arg for keyword in call.keywords if keyword.arg is not None}
            for _, call, _ in constructor_methods
        ]
        shared_keyword_names = tuple(
            sorted(str(item) for item in set.intersection(*keyword_sets))
        )
        if len(shared_keyword_names) < 4:
            continue
        groups.append(
            AlternateConstructorFamilyGroup(
                file_path=str(module.path),
                class_name=node.name,
                method_names=tuple(method.name for method, _, _ in constructor_methods),
                line_numbers=tuple(
                    method.lineno for method, _, _ in constructor_methods
                ),
                keyword_names=shared_keyword_names,
                source_type_names=tuple(
                    source_type_name for _, _, source_type_name in constructor_methods
                ),
            )
        )
    return tuple(groups)


def _repeated_property_hook_metrics(
    class_names: tuple[str, ...], property_name: str
) -> RepeatedMethodMetrics:
    return RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(class_names),
        statement_count=1,
        class_count=len(class_names),
        method_symbols=tuple(
            f"{class_name}.{property_name}" for class_name in class_names
        ),
    )


class ManualFamilyRosterDetector(IssueDetector):
    detector_id = "manual_family_roster"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Manual subclass roster should become metaclass-registry auto-registration",
        why=(
            "One helper manually enumerates a class family instead of deriving membership from class existence. "
            "The docs treat that as class-level registration logic that should live in one authoritative `metaclass-registry` hook."
        ),
        capability_gap="zero-delay metaclass-registry class-family discovery with declarative ordering",
        relation_context="family membership is maintained by a manual roster function or constant",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        index = NominalAuthorityIndex(modules)
        findings: list[RefactorFinding] = []
        for module in modules:
            for candidate in _manual_family_roster_candidates(module, index):
                evidence = [
                    SourceLocation(
                        candidate.file_path, candidate.line, candidate.owner_name
                    )
                ]
                evidence.extend(
                    SourceLocation(shape.file_path, shape.line, shape.class_name)
                    for member_name in candidate.member_names[:4]
                    for shape in index.shapes_named(member_name)[:1]
                )
                findings.append(
                    self.finding_spec.build(
                        self.detector_id,
                        (
                            f"`{candidate.owner_name}` manually enumerates {len(candidate.member_names)} members of the `{candidate.family_base_name}` family."
                        ),
                        tuple(evidence[:6]),
                        scaffold=(
                            "from metaclass_registry import AutoRegisterMeta\n\n"
                            f"class Registered{candidate.family_base_name}({candidate.family_base_name}, metaclass=AutoRegisterMeta):\n"
                            "    __registry_key__ = \"registration_key\"\n"
                            "    __skip_if_no_key__ = True\n"
                            "    registration_key = None\n"
                            "    registration_order: ClassVar[int] = 0\n\n"
                            "    @classmethod\n"
                            "    def registered_types(cls):\n"
                            "        ordered = sorted(\n"
                            "            cls.__registry__.values(),\n"
                            "            key=lambda registered_type: registered_type.registration_order,\n"
                            "        )\n"
                            "        return tuple(ordered)"
                        ),
                        codemod_patch=(
                            f"# Replace `{candidate.owner_name}` with metaclass-registry class-time registration for the `{candidate.family_base_name}` family.\n"
                            f"# Delete the manual {candidate.constructor_style} roster once subclasses are discoverable through `cls.__registry__.values()`."
                        ),
                        metrics=RegistrationMetrics(
                            registration_site_count=len(candidate.member_names),
                            class_count=len(candidate.member_names),
                            registry_name=candidate.owner_name,
                            class_names=candidate.member_names,
                        ),
                    )
                )
        return findings


class FragmentedFamilyAuthorityDetector(CandidateFindingDetector):
    detector_id = "fragmented_family_authority"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Parallel key-family tables should become one authoritative record",
        why=(
            "Several dicts keyed by the same nominal family collectively encode one semantic record. "
            "The docs treat that as fragmented authority that should collapse into one authoritative schema."
        ),
        capability_gap="single authoritative enum-keyed planning record",
        relation_context="one key family is split across parallel metadata tables",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _fragmented_family_authority_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        authority_candidate = cast(FragmentedFamilyAuthorityCandidate, candidate)
        evidence = tuple(
            SourceLocation(authority_candidate.file_path, line, name)
            for name, line in zip(
                authority_candidate.mapping_names,
                authority_candidate.line_numbers,
                strict=True,
            )
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Tables {', '.join(authority_candidate.mapping_names)} split one `{authority_candidate.key_family_name}` metadata family across {len(authority_candidate.mapping_names)} authorities."
            ),
            evidence[:6],
            scaffold=(
                "@dataclass(frozen=True)\n"
                f"class {authority_candidate.key_family_name}Spec:\n"
                f"    key: {authority_candidate.key_family_name}\n"
                "    priority: int\n"
                "    dependencies: tuple[object, ...] = ()\n"
                "    synergy_with: tuple[object, ...] = ()\n"
                "    builder: object | None = None"
            ),
            codemod_patch=(
                f"# Collapse {authority_candidate.mapping_names} into one `{authority_candidate.key_family_name}`-keyed spec table.\n"
                f"# Move shared keys {authority_candidate.shared_keys} into one authoritative record instead of parallel dicts."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(authority_candidate.mapping_names),
                field_count=len(authority_candidate.shared_keys),
                mapping_name=f"{authority_candidate.key_family_name} spec",
                field_names=authority_candidate.shared_keys,
            ),
        )


class ExistingNominalAuthorityReuseDetector(IssueDetector):
    detector_id = "existing_nominal_authority_reuse"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Existing nominal authority should be reused",
        why=(
            "A compatible nominal authority already exists, but another class repeats the same semantic field family outside that hierarchy. "
            "The docs prefer reusing the existing authority before synthesizing a new one."
        ),
        capability_gap="reuse of an existing authoritative base or mixin instead of duplicating the family",
        relation_context="a concrete class repeats a semantic family already declared by an existing nominal authority",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _existing_nominal_authority_reuse_candidates(modules):
            evidence = (
                SourceLocation(
                    candidate.file_path,
                    candidate.line,
                    candidate.class_name,
                ),
                SourceLocation(
                    candidate.compatible_authority_file_path,
                    candidate.compatible_authority_line,
                    candidate.compatible_authority_name,
                ),
            )
            inheritance_clause = (
                f"{candidate.compatible_authority_name}, ExistingResidueMixin"
                if candidate.reuse_kind == "compose_mixin"
                else candidate.compatible_authority_name
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"`{candidate.class_name}` repeats semantic fields {candidate.shared_field_names} already owned by `{candidate.compatible_authority_name}`."
                    ),
                    evidence,
                    scaffold=(
                        f"class {candidate.class_name}({inheritance_clause}):\n"
                        "    ...\n\n"
                        f"# Reuse `{candidate.compatible_authority_name}` for roles {candidate.shared_role_names}."
                    ),
                    codemod_patch=(
                        f"# Route `{candidate.class_name}` through existing authority `{candidate.compatible_authority_name}`.\n"
                        f"# Do not synthesize a fresh base for shared fields {candidate.shared_field_names}."
                    ),
                    metrics=FieldFamilyMetrics(
                        class_count=2,
                        field_count=len(candidate.shared_field_names),
                        class_names=(
                            candidate.compatible_authority_name,
                            candidate.class_name,
                        ),
                        field_names=candidate.shared_field_names,
                        execution_level="existing_nominal_authority",
                    ),
                )
            )
        return findings


class PassThroughNominalWrapperDetector(IssueDetector):
    detector_id = "pass_through_nominal_wrapper"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Pass-through wrapper should reuse the existing nominal authority directly",
        why=(
            "A wrapper re-exposes an existing nominal contract through pure forwarding without adding any new invariant, "
            "provenance boundary, or semantic residue. The docs treat that as zero-information duplication: consumers "
            "should use the existing authority directly."
        ),
        capability_gap="direct reuse of the existing nominal authority instead of a zero-information forwarding wrapper",
        relation_context="a concrete class forwards an existing nominal contract member-for-member without adding new semantics",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _pass_through_nominal_wrapper_candidates(modules):
            evidence = (
                SourceLocation(candidate.file_path, candidate.line, candidate.class_name),
                SourceLocation(
                    candidate.delegate_authority_file_path,
                    candidate.delegate_authority_line,
                    candidate.delegate_authority_name,
                ),
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"`{candidate.class_name}` forwards members {candidate.forwarded_member_names} to "
                        f"`{candidate.delegate_authority_name}` through `{candidate.delegate_field_name}` without "
                        "adding any new invariant."
                    ),
                    evidence,
                    scaffold=(
                        f"# Delete `{candidate.class_name}` and type consumers against `{candidate.delegate_authority_name}` directly.\n"
                        f"{candidate.delegate_field_name}: {candidate.delegate_authority_name}"
                    ),
                    codemod_patch=(
                        f"# Remove `{candidate.class_name}` as a pass-through wrapper.\n"
                        f"# Accept `{candidate.delegate_authority_name}` directly anywhere the wrapper is only forwarding "
                        f"{candidate.forwarded_member_names}."
                    ),
                )
            )
        return findings


class FindingAssemblyPipelineDetector(PerModuleIssueDetector):
    detector_id = "finding_assembly_pipeline"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated finding-assembly pipeline should move into a detector base",
        why=(
            "Several detectors repeat the same candidate-to-finding pipeline with only orthogonal hooks varying. "
            "The docs prefer one template-method substrate plus mixins for residue."
        ),
        capability_gap="candidate-driven detector template with abstract hooks and mixins",
        relation_context="same finding assembly stages repeat across sibling detector classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidates = _finding_assembly_pipeline_candidates(module)
        if len(candidates) < 3:
            return []
        evidence = tuple(
            SourceLocation(
                candidate.file_path,
                candidate.line,
                f"{candidate.class_name}.{candidate.method_name}",
            )
            for candidate in candidates[:6]
        )
        collector_names = tuple(
            sorted({candidate.candidate_source_name for candidate in candidates})
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Detectors {', '.join(candidate.class_name for candidate in candidates[:5])} repeat the same candidate-to-finding pipeline over collectors {', '.join(collector_names[:4])}."
                ),
                evidence,
                scaffold=(
                    "class CandidateFindingDetector(PerModuleIssueDetector, ABC):\n"
                    "    @abstractmethod\n"
                    "    def iter_candidates(self, module, config): ...\n\n"
                    "    @abstractmethod\n"
                    "    def build_finding(self, candidate): ...\n\n"
                    "    def _findings_for_module(self, module, config):\n"
                    "        return [self.build_finding(candidate) for candidate in self.iter_candidates(module, config)]"
                ),
                codemod_patch=(
                    "# Extract one candidate-driven detector base for `_findings_for_module`.\n"
                    "# Leave only candidate collection, evidence shaping, metrics, and scaffold/patch helpers on the leaves."
                ),
                metrics=RepeatedMethodMetrics.from_duplicate_family(
                    duplicate_site_count=len(candidates),
                    statement_count=3,
                    class_count=len(candidates),
                    method_symbols=tuple(
                        f"{candidate.class_name}.{candidate.method_name}"
                        for candidate in candidates
                    ),
                ),
            )
        ]


def _keyword_mapping_metrics(
    mapping_site_count: int,
    field_names: tuple[str, ...],
    mapping_name: str,
) -> MappingMetrics:
    return MappingMetrics(
        mapping_site_count=mapping_site_count,
        field_count=len(field_names),
        mapping_name=mapping_name,
        field_names=field_names,
    )


class ProjectionBuilderAuthorityDetector(PerModuleIssueDetector):
    detector_id = "projection_builder_authority"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Projection-style record rebuild should collapse into one authoritative builder",
        why=(
            "Several call sites rebuild the same nominal record by projecting overlapping source authorities field-by-field, "
            "often with guard/default residue mixed into the call. The docs treat that as fragmented builder authority: "
            "the projection belongs in one authoritative constructor, classmethod, or helper."
        ),
        capability_gap="one authoritative projection builder for a repeated record family",
        relation_context="same nominal record is re-projected from overlapping sources at several call sites",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        observation_tags=(
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for builders in _projection_builder_groups(module, config):
            callee_name = builders[0].callee_name
            keyword_names = builders[0].keyword_names
            evidence = tuple(
                SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                for builder in builders[:6]
            )
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"`{callee_name}` is rebuilt across {len(builders)} projection sites over keyword family {keyword_names}, "
                        "with guards/defaults varying per site."
                    ),
                    evidence,
                    scaffold=(
                        "@dataclass(frozen=True)\n"
                        f"class {callee_name}Builder:\n"
                        "    @classmethod\n"
                        "    def from_sources(cls, ...):\n"
                        f"        return {callee_name}(...)"
                    ),
                    codemod_patch=(
                        f"# Move `{callee_name}` projection logic into one authoritative builder/classmethod.\n"
                        "# Leave call sites responsible only for naming the source authorities, not reassigning every field."
                    ),
                    metrics=_keyword_mapping_metrics(
                        len(builders), keyword_names, callee_name
                    ),
                )
            )
        return findings


class GuardedDelegatorSpecDetector(PerModuleIssueDetector):
    detector_id = "guarded_delegator_spec"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated guarded spec wrappers should collapse into mixins",
        why=(
            "Several observation-spec methods differ only by a scope guard and one delegate helper call. "
            "The docs prefer one shared wrapper substrate with orthogonal scope mixins."
        ),
        capability_gap="shared wrapper substrate with orthogonal scope mixins",
        relation_context="guard-and-delegate wrapper logic repeats across sibling observation specs",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidates = _guarded_delegator_candidates(module)
        if len(candidates) < 2:
            return []
        evidence = tuple(
            SourceLocation(
                candidate.file_path,
                candidate.line,
                f"{candidate.class_name}.{candidate.method_name}",
            )
            for candidate in candidates[:6]
        )
        scope_roles = tuple(sorted({candidate.scope_role for candidate in candidates}))
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"Observation specs {', '.join(candidate.class_name for candidate in candidates[:5])} repeat guarded delegation over scope roles {', '.join(scope_roles)}."
                ),
                evidence,
                scaffold=(
                    "class ScopeFilteredSpec(ObservationShapeSpec, ABC):\n"
                    "    @abstractmethod\n"
                    "    def accepts_scope(self, observation): ...\n\n"
                    "    @abstractmethod\n"
                    "    def delegate(self, parsed_module, node, observation): ...\n\n"
                    "    def build_shape(self, parsed_module, observation):\n"
                    "        if not self.accepts_scope(observation):\n"
                    "            return None\n"
                    "        return self.delegate(parsed_module, observation.node, observation)"
                ),
                codemod_patch=(
                    "# Collapse repeated guard-and-delegate wrappers into one shared spec base.\n"
                    "# Encode module-only, class-only, function-only, or node-type residue as mixins or tiny hooks."
                ),
                metrics=RepeatedMethodMetrics.from_duplicate_family(
                    duplicate_site_count=len(candidates),
                    statement_count=2,
                    class_count=len({candidate.class_name for candidate in candidates}),
                    method_symbols=tuple(
                        f"{candidate.class_name}.{candidate.method_name}"
                        for candidate in candidates
                    ),
                ),
            )
        ]


class StructuralObservationProjectionDetector(CandidateFindingDetector):
    detector_id = "structural_observation_projection"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated property projection builders should share one projection substrate",
        why=(
            "Several classes repeat the same property-backed constructor projection schema with only role hooks varying. "
            "The docs prefer one authoritative projection template."
        ),
        capability_gap="single authoritative projection builder with role hooks",
        relation_context="same property-backed constructor schema is manually rebuilt across many classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        grouped: dict[
            tuple[str, str, tuple[str, ...]],
            list[StructuralObservationPropertyCandidate],
        ] = defaultdict(list)
        for candidate in _structural_observation_property_candidates(module):
            grouped[
                (
                    candidate.property_name,
                    candidate.constructor_name,
                    candidate.keyword_names,
                )
            ].append(candidate)
        return tuple(
            (group_key, tuple(candidates))
            for group_key, candidates in grouped.items()
            if len(candidates) >= 3
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group_key, grouped_candidates = cast(
            tuple[
                tuple[str, str, tuple[str, ...]],
                tuple[StructuralObservationPropertyCandidate, ...],
            ],
            candidate,
        )
        property_name, constructor_name, keyword_names = group_key
        evidence = tuple(
            SourceLocation(item.file_path, item.line, item.class_name)
            for item in grouped_candidates[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Classes {', '.join(item.class_name for item in grouped_candidates[:5])} rebuild property `{property_name}` with the same `{constructor_name}` schema over roles {keyword_names}."
            ),
            evidence,
            scaffold=(
                "class ProjectionTemplate(ABC):\n"
                "    @property\n"
                f"    def {property_name}(self) -> {constructor_name}:\n"
                f"        return {constructor_name}(...)"
            ),
            codemod_patch=(
                f"# Introduce one projection template for `{property_name}` over roles {keyword_names}.\n"
                "# Leave only the role-specific hooks on the concrete carriers."
            ),
            metrics=_keyword_mapping_metrics(
                len(grouped_candidates), keyword_names, constructor_name
            ),
        )


def default_detectors() -> tuple[IssueDetector, ...]:
    """Instantiate all registered detectors in deterministic priority order."""
    return tuple(
        detector_type() for detector_type in IssueDetector.registered_detector_types()
    )


_SEMANTIC_STRING_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_SEMANTIC_KEYWORD_NAMES = {
    "backend",
    "capability_gap",
    "capability_tags",
    "certification",
    "confidence",
    "key",
    "kind",
    "label",
    "mode",
    "name",
    "observation_tags",
    "pattern_id",
    "registry_key",
    "relation_context",
    "status",
    "title",
    _TYPE_NAME_LITERAL,
}
_SEMANTIC_NAME_SUFFIXES = (
    "_backend",
    "_certification",
    "_family",
    "_id",
    "_key",
    "_kind",
    "_label",
    "_mode",
    "_name",
    "_pattern",
    "_role",
    "_status",
    "_type",
)


def _semantic_string_literal_sites(
    module: ParsedModule,
) -> dict[str, list[SourceLocation]]:
    groups: dict[str, set[SourceLocation]] = defaultdict(set)

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.function_stack: list[str] = []

        def visit_Module(self, node: ast.Module) -> None:
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_stack.append(node.name)
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.function_stack.append(node.name)
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)
            self.function_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            self._record_literals(node.value, node.lineno, "assign")
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is not None:
                self._record_literals(node.value, node.lineno, "assign")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                if not _is_semantic_keyword_name(keyword.arg):
                    continue
                self._record_literals(keyword.value, node.lineno, keyword.arg)
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            if not _compare_subject_is_semantic(node):
                self.generic_visit(node)
                return
            self._record_literals(node.left, node.lineno, "compare")
            for comparator in node.comparators:
                self._record_literals(comparator, node.lineno, "compare")
            self.generic_visit(node)

        def _record_literals(self, node: ast.AST, lineno: int, kind: str) -> None:
            for literal in _literal_strings(node):
                groups[literal].add(
                    SourceLocation(str(module.path), lineno, self._symbol(kind))
                )

        def _symbol(self, kind: str) -> str:
            owner = self.function_stack[-1] if self.function_stack else "<module>"
            if self.class_stack:
                owner = f"{self.class_stack[-1]}.{owner}"
            return f"{owner}:{kind}"

    Visitor().visit(module.module)
    return {
        literal: sorted(
            sites, key=lambda item: (item.file_path, item.line, item.symbol)
        )
        for literal, sites in groups.items()
        if len(sites) >= 2
    }


def _literal_strings(node: ast.AST) -> tuple[str, ...]:
    literals: list[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if _is_semantic_string(node.value):
            literals.append(node.value)
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for item in node.elts:
            literals.extend(_literal_strings(item))
    return tuple(literals)


def _is_semantic_string(value: str) -> bool:
    return bool(_SEMANTIC_STRING_RE.fullmatch(value))


def _is_semantic_keyword_name(name: str) -> bool:
    return name in _SEMANTIC_KEYWORD_NAMES or name.endswith(_SEMANTIC_NAME_SUFFIXES)


def _compare_subject_is_semantic(node: ast.Compare) -> bool:
    candidates = [node.left] + list(node.comparators)
    return any(_looks_like_semantic_subject(candidate) for candidate in candidates)


def _looks_like_semantic_subject(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return _is_semantic_keyword_name(node.id)
    if isinstance(node, ast.Attribute):
        return _is_semantic_keyword_name(node.attr)
    return False


def _is_docstring_expr(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _collect_dict_attrs(node: ast.ClassDef) -> set[str]:
    dict_attrs: set[str] = set()
    for child in _walk_nodes(node):
        if not isinstance(child, ast.Assign):
            continue
        if not isinstance(child.value, ast.Dict):
            continue
        for target in child.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                dict_attrs.add(target.attr)
    return dict_attrs


def _collect_mirrored_assignments(node: ast.ClassDef) -> list[tuple[int, str]]:
    mirrored: list[tuple[int, str]] = []
    for child in _walk_nodes(node):
        if not isinstance(child, ast.Assign):
            continue
        for target in child.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Attribute):
                continue
            if (
                not isinstance(target.value.value, ast.Name)
                or target.value.value.id != "self"
            ):
                continue
            if isinstance(child.value, ast.Name):
                mirrored.append((child.lineno, target.value.attr))
    return mirrored


def _collect_repeated_method_shapes(
    modules: list[ParsedModule], config: DetectorConfig
) -> tuple[MethodShape, ...]:
    groups = _group_repeated_methods(modules, config)
    return tuple(method for group in groups for method in group)


def _group_repeated_methods(
    modules: list[ParsedModule], config: DetectorConfig
) -> list[tuple[MethodShape, ...]]:
    methods = tuple(
        method
        for module in modules
        for method in _collect_typed_family_items(
            module, MethodShapeFamily, MethodShape
        )
        if method.class_name
        and method.statement_count >= config.min_duplicate_statements
    )
    groups = _fiber_grouped_shapes(
        modules,
        tuple(methods),
        ObservationKind.METHOD_SHAPE,
        StructuralExecutionLevel.FUNCTION_BODY,
    )
    return [
        tuple(_as_method_shape(method) for method in group)
        for group in groups
        if len(group) >= 2
        and len({_as_method_shape(method).class_name for method in group}) >= 2
    ]


def _abc_patch_for_methods(methods: tuple[MethodShape, ...]) -> str:
    target_file = methods[0].file_path
    base_name = (
        _shared_family_name(
            sorted(
                {
                    method.class_name
                    for method in methods
                    if method.class_name is not None
                }
            )
        )
        or "ExtractedBase"
    )
    hook_name = methods[0].method_name
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        f"@@\n"
        f"+class {base_name}(ABC):\n"
        f"+    def run(self, request):\n"
        f"+        normalized = self._normalize(request)\n"
        f"+        return self.{hook_name}(normalized)\n"
        f"+\n"
        f"+    @abstractmethod\n"
        f"+    def {hook_name}(self, normalized): ...\n"
        "*** End Patch"
    )


def _abc_family_patch(
    class_names: frozenset[str], groups: list[tuple[MethodShape, ...]]
) -> str:
    ordered = sorted(class_names)
    target_file = groups[0][0].file_path
    base_name = _shared_family_name(ordered) or "FamilyBase"
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        f"+class {base_name}(ABC):\n"
        "+    def run(self, request): ...\n"
        "+\n"
        "+    @abstractmethod\n"
        "+    def hook(self, request): ...\n"
        "*** End Patch"
    )


def _builder_patch(builders: tuple[BuilderCallShape, ...]) -> str:
    target_file = builders[0].file_path
    callee_name = builders[0].callee_name
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        f"+@classmethod\n"
        f"+def from_source(cls, source):\n"
        f"+    return {callee_name}(...)\n"
        "*** End Patch"
    )


def _projection_schema_patch(export_shapes: tuple[ExportDictShape, ...]) -> str:
    target_file = export_shapes[0].file_path
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        "+@dataclass(frozen=True)\n"
        "+class ProjectionSchema:\n"
        "+    ...\n"
        "+\n"
        "+    @classmethod\n"
        "+    def from_source(cls, source): ...\n"
        "*** End Patch"
    )


def _autoregister_patch(
    registry_name: str,
    class_names: set[str],
    registrations: tuple[RegistrationShape, ...],
) -> str:
    target_file = registrations[0].file_path
    base_name = _shared_family_name(sorted(class_names)) or "RegisteredBase"
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        "+from metaclass_registry import AutoRegisterMeta\n"
        "+\n"
        f"+class {base_name}(ABC, metaclass=AutoRegisterMeta):\n"
        "+    __registry_key__ = \"registry_key\"\n"
        "+    __skip_if_no_key__ = True\n"
        "+    registry_key = None\n"
        "+\n"
        f"+# Replace `{registry_name}` with `{base_name}.__registry__`.\n"
        "*** End Patch"
    )


def _abc_scaffold_for_methods(methods: tuple[MethodShape, ...]) -> str:
    class_names = sorted(
        {method.class_name for method in methods if method.class_name is not None}
    )
    hook_names = sorted({method.method_name for method in methods})
    base_name = _shared_family_name(class_names) or "ExtractedBase"
    hook_name = hook_names[0] if hook_names else "hook"
    return (
        f"class {base_name}(ABC):\n"
        f"    def run(self, request):\n"
        f"        normalized = self._normalize(request)\n"
        f"        return self.{hook_name}(normalized)\n\n"
        f"    @abstractmethod\n"
        f"    def {hook_name}(self, normalized): ..."
    )


def _abc_family_scaffold(
    class_names: frozenset[str], groups: list[tuple[MethodShape, ...]]
) -> str:
    ordered = sorted(class_names)
    base_name = _shared_family_name(ordered) or "FamilyBase"
    hook_methods = sorted(
        {
            method.method_name
            for group in groups
            for method in group
            if method.class_name in class_names
        }
    )
    hook_block = "\n".join(
        f"    @abstractmethod\n    def {name}(self, request): ..."
        for name in hook_methods[:3]
    )
    subclass_block = "\n".join(
        f"class {name}({base_name}):\n    ..." for name in ordered[:3]
    )
    return f"class {base_name}(ABC):\n    def run(self, request): ...\n{hook_block}\n\n{subclass_block}"


def _builder_scaffold(builders: tuple[BuilderCallShape, ...]) -> str:
    callee_name = builders[0].callee_name
    keywords = builders[0].keyword_names
    row_name = callee_name if callee_name[:1].isupper() else "ProjectedRow"
    args_block = "\n".join(
        f"            {name}=source.{name}," for name in keywords[:4]
    )
    return (
        f"@dataclass(frozen=True)\n"
        f"class {row_name}:\n"
        f"    ...\n\n"
        f"    @classmethod\n"
        f"    def from_source(cls, source):\n"
        f"        return cls(\n{args_block}\n        )"
    )


def _projection_schema_scaffold(export_shapes: tuple[ExportDictShape, ...]) -> str:
    keys = export_shapes[0].key_names
    field_block = "\n".join(f"    {key}: object" for key in keys[:4])
    mapping_block = "\n".join(f"            {key}=source.{key}," for key in keys[:4])
    return (
        "@dataclass(frozen=True)\n"
        "class ProjectionSchema:\n"
        f"{field_block}\n\n"
        "    @classmethod\n"
        "    def from_source(cls, source):\n"
        f"        return cls(\n{mapping_block}\n        )"
    )


def _autoregister_scaffold(registry_name: str, class_names: set[str]) -> str:
    base_name = _shared_family_name(sorted(class_names)) or "RegisteredBase"
    sample = sorted(class_names)[:2]
    subclass_block = "\n".join(
        f'class {name}({base_name}):\n    registry_key = "{name.lower()}"'
        for name in sample
    )
    return (
        "from metaclass_registry import AutoRegisterMeta\n\n"
        f"class {base_name}(ABC, metaclass=AutoRegisterMeta):\n"
        "    __registry_key__ = \"registry_key\"\n"
        "    __skip_if_no_key__ = True\n"
        "    registry_key = None\n\n"
        f"{subclass_block}"
    )


def _shared_family_name(class_names: list[str]) -> str | None:
    if not class_names:
        return None
    prefix = class_names[0]
    for name in class_names[1:]:
        while prefix and not name.startswith(prefix):
            prefix = prefix[:-1]
    return prefix or None


@lru_cache(maxsize=None)
def _dispatch_dict_locations(
    module: ParsedModule, min_string_cases: int
) -> list[SourceLocation]:
    locations: list[SourceLocation] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_depth += 1
            self.generic_visit(node)
            self.function_depth -= 1

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.function_depth += 1
            self.generic_visit(node)
            self.function_depth -= 1

        def visit_Assign(self, node: ast.Assign) -> None:
            if self.function_depth > 0:
                return
            if not isinstance(node.value, ast.Dict):
                return
            if _looks_like_dispatch_dict(node.value, min_string_cases):
                locations.append(
                    SourceLocation(
                        str(module.path), node.lineno, "dict-string-dispatch"
                    )
                )

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if self.function_depth > 0:
                return
            if not isinstance(node.value, ast.Dict):
                return
            if _looks_like_dispatch_dict(node.value, min_string_cases):
                locations.append(
                    SourceLocation(
                        str(module.path), node.lineno, "dict-string-dispatch"
                    )
                )

    Visitor().visit(module.module)
    return locations


def _looks_like_dispatch_dict(node: ast.Dict, min_string_cases: int) -> bool:
    string_keys = [
        key
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]
    if len(string_keys) < min_string_cases or len(string_keys) != len(node.keys):
        return False
    if not node.values:
        return False
    if all(isinstance(value, ast.Constant) for value in node.values):
        return False
    return any(
        isinstance(value, (ast.Name, ast.Attribute, ast.Lambda, ast.Call))
        for value in node.values
    )


@lru_cache(maxsize=None)
def _attribute_branch_evidence(
    module: ParsedModule, attr_name: str
) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in _walk_nodes(module.module):
        if isinstance(node, ast.If):
            if _test_compares_attribute(node.test, attr_name):
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, f"if-{attr_name}")
                )
        if isinstance(node, ast.Match):
            subject = node.subject
            if isinstance(subject, ast.Attribute) and subject.attr == attr_name:
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, f"match-{attr_name}")
                )
    return evidence


def _test_compares_attribute(test: ast.AST, attr_name: str) -> bool:
    for node in _walk_nodes(test):
        if isinstance(node, ast.Compare):
            values = [node.left] + list(node.comparators)
            attr_match = any(
                isinstance(value, ast.Attribute) and value.attr == attr_name
                for value in values
            )
            literal_match = any(
                isinstance(value, ast.Constant)
                and isinstance(value.value, (str, int, bool))
                for value in values
            )
            if attr_match and literal_match:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == _GETATTR_BUILTIN and len(node.args) >= 2:
                arg = node.args[1]
                if isinstance(arg, ast.Constant) and arg.value == attr_name:
                    return True
    return False


def _iter_functions(module: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in _walk_nodes(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _projection_helper_scaffold(shapes: Sequence[ProjectionHelperShape]) -> str:
    function_names = ", ".join(shape.function_name for shape in shapes)
    attributes = ", ".join(sorted({shape.projected_attribute for shape in shapes}))
    return (
        "def _render_projection(items, projector):\n"
        "    return tuple(_dedupe_preserve_order(projector(item) for item in items))\n\n"
        f"# Replace {function_names} with `_render_projection(..., lambda item: item.<field>)`.\n"
        f"# Projected fields: {attributes}"
    )


def _supports_accessor_wrapper_finding(
    candidates: Sequence[AccessorWrapperCandidate],
) -> bool:
    if not candidates:
        return False
    if any(candidate.wrapper_shape.startswith("computed_") for candidate in candidates):
        return True
    if len(candidates) >= 2:
        return True
    return False


def _is_framework_adapter_symbol(symbol: str) -> bool:
    return symbol.startswith(("build_from_", "build_scoped_", "accepts_"))


def _is_framework_lineage_symbol(symbol: str) -> bool:
    return _is_framework_adapter_symbol(symbol) or symbol in {
        "__new__",
        "collect",
        "registered_specs_for_literal_type",
    }


def _is_framework_attribute_probe(observation: AttributeProbeObservation) -> bool:
    return observation.observed_attribute in {
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
        # Standard array protocol compatibility checks are not semantic-role recovery.
        "shape",
        "ndim",
        "dtype",
        "size",
    }


def _accessor_replacement_example(candidate: AccessorWrapperCandidate) -> str:
    if candidate.accessor_kind == "setter":
        return f"- replace `{candidate.symbol}(value)` with `{candidate.observed_attribute} = value`"
    if candidate.wrapper_shape == "read_through":
        return f"- replace `{candidate.symbol}()` with `{candidate.observed_attribute}`"
    return f"- replace `{candidate.symbol}()` with an `@property` exposing `{candidate.target_expression}`"


def _expression_root_names(node: ast.AST) -> set[str]:
    roots: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                current = current.value
            if isinstance(current, ast.Name):
                roots.add(current.id)
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            roots.add(node.id)

    Visitor().visit(node)
    return roots


def _function_param_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    names = {arg.arg for arg in function.args.args}
    names.update(arg.arg for arg in function.args.kwonlyargs)
    if function.args.vararg is not None:
        names.add(function.args.vararg.arg)
    if function.args.kwarg is not None:
        names.add(function.args.kwarg.arg)
    return names


def _is_transport_expression(
    node: ast.AST,
    *,
    allowed_roots: set[str],
) -> bool:
    return _expression_root_names(node) <= allowed_roots


def _wrapper_delegate_symbol(
    node: ast.AST,
    *,
    class_name: str | None,
) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"self", "cls"}
        and class_name is not None
    ):
        return f"{class_name}.{node.attr}"
    return None


def _projected_attribute_names(
    node: ast.AST,
    *,
    bound_name: str,
) -> tuple[str, ...] | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == bound_name:
            return (node.attr,)
        return None
    if isinstance(node, ast.Tuple):
        projected: list[str] = []
        for item in node.elts:
            if not isinstance(item, ast.Attribute) or not isinstance(item.value, ast.Name):
                return None
            if item.value.id != bound_name:
                return None
            projected.append(item.attr)
        return tuple(projected)
    return None


def _call_chain_from_outer_call(call: ast.Call) -> tuple[ast.Call, ...]:
    chain = [call]
    current = call
    while (
        isinstance(current.func, ast.Attribute)
        and isinstance(current.func.value, ast.Call)
    ):
        current = current.func.value
        chain.append(current)
    return tuple(chain)


def _call_chain_transport_values(chain: tuple[ast.Call, ...]) -> tuple[ast.AST, ...]:
    values: list[ast.AST] = []
    for call in chain:
        values.extend(call.args)
        values.extend(keyword.value for keyword in call.keywords)
    return tuple(values)


def _call_chain_delegate_symbol(
    chain: tuple[ast.Call, ...],
    *,
    class_name: str | None,
) -> str:
    inner = chain[-1]
    symbol = _wrapper_delegate_symbol(inner.func, class_name=class_name)
    if symbol is None:
        symbol = ast.unparse(inner.func)
    for call in reversed(chain[:-1]):
        method_name = _call_name(call.func)
        if method_name is None:
            method_name = ast.unparse(call.func)
        symbol = f"{symbol}.{method_name}"
    return symbol


def _delegate_root_symbol(delegate_symbol: str) -> str:
    return delegate_symbol.split(".", 1)[0]


def _is_private_symbol_name(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _is_public_module_api_qualname(qualname: str) -> bool:
    return "." not in qualname and not qualname.startswith("_")


@lru_cache(maxsize=None)
def _top_level_symbol_lines(module: ParsedModule) -> dict[str, int]:
    lines: dict[str, int] = {}
    for statement in _trim_docstring_body(module.module.body):
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.setdefault(statement.name, statement.lineno)
    return lines


def _resolved_import_call_target_symbols(
    module: ParsedModule,
    node: ast.AST,
    *,
    import_aliases: dict[str, str],
) -> tuple[str, ...]:
    del module
    parts = _ast_attribute_chain(node)
    if parts is None:
        return ()
    first, *rest = parts
    alias_target = import_aliases.get(first)
    if alias_target is None:
        return ()
    return (".".join((alias_target, *rest)) if rest else alias_target,)


def _external_callsites_by_target(
    modules: Sequence[ParsedModule],
) -> dict[str, tuple[ResolvedExternalCallsite, ...]]:
    return _external_callsites_by_target_cached(tuple(modules))


@lru_cache(maxsize=None)
def _external_callsites_by_target_cached(
    modules: tuple[ParsedModule, ...],
) -> dict[str, tuple[ResolvedExternalCallsite, ...]]:
    callsites_by_target: dict[str, set[ResolvedExternalCallsite]] = defaultdict(set)
    for module in modules:
        import_aliases = _module_import_aliases(module)

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.class_stack: list[str] = []
                self.function_stack: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_Call(self, node: ast.Call) -> None:
                for target in _resolved_import_call_target_symbols(
                    module,
                    node.func,
                    import_aliases=import_aliases,
                ):
                    callsites_by_target[target].add(
                        ResolvedExternalCallsite(
                            module_name=module.module_name,
                            location=SourceLocation(
                                str(module.path),
                                node.lineno,
                                self._symbol("call"),
                            ),
                        )
                    )
                self.generic_visit(node)

            def _symbol(self, kind: str) -> str:
                owner = self.function_stack[-1] if self.function_stack else "<module>"
                if self.class_stack:
                    owner = f"{self.class_stack[-1]}.{owner}"
                return f"{owner}:{kind}"

        Visitor().visit(module.module)
    return {
        target: tuple(
            sorted(
                callsites,
                key=lambda item: (
                    item.location.file_path,
                    item.location.line,
                    item.location.symbol,
                    item.module_name,
                ),
            )
        )
        for target, callsites in callsites_by_target.items()
    }


def _matching_external_callsites(
    callsites_by_target: dict[str, tuple[ResolvedExternalCallsite, ...]],
    *,
    target_symbol: str,
) -> tuple[ResolvedExternalCallsite, ...]:
    matched: set[ResolvedExternalCallsite] = set()
    for observed_target, callsites in callsites_by_target.items():
        if (
            observed_target == target_symbol
            or observed_target.endswith(f".{target_symbol}")
            or target_symbol.endswith(f".{observed_target}")
        ):
            matched.update(callsites)
    return tuple(
        sorted(
            matched,
            key=lambda item: (
                item.location.file_path,
                item.location.line,
                item.location.symbol,
                item.module_name,
            ),
        )
    )


def _trivial_forwarding_wrapper_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> TrivialForwardingWrapperCandidate | None:
    if function.name.startswith("__") and function.name.endswith("__"):
        return None
    body = _trim_docstring_body(function.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    returned = body[0].value
    if not isinstance(returned, ast.Call):
        return None
    chain = _call_chain_from_outer_call(returned)
    if len(chain) < 2:
        return None
    class_name = qualname.rsplit(".", 1)[0] if "." in qualname else None
    allowed_roots = _function_param_names(function) | {"self", "cls"}
    values = _call_chain_transport_values(chain)
    if not values:
        return None
    if not all(_is_transport_expression(value, allowed_roots=allowed_roots) for value in values):
        return None
    transported_value_sources = tuple(sorted({ast.unparse(value) for value in values}))
    parameter_names = _function_param_names(function) - {"self", "cls"}
    forwarded_parameter_names = tuple(
        sorted(
            {
                node.id
                for value in values
                for node in _walk_nodes(value)
                if isinstance(node, ast.Name) and node.id in parameter_names
            }
        )
    )
    if not transported_value_sources:
        return None
    delegate_symbol = _call_chain_delegate_symbol(chain, class_name=class_name)
    return TrivialForwardingWrapperCandidate(
        file_path=str(module.path),
        line=function.lineno,
        qualname=qualname,
        delegate_symbol=delegate_symbol,
        call_depth=len(chain),
        forwarded_parameter_names=forwarded_parameter_names,
        transported_value_sources=transported_value_sources,
    )


@lru_cache(maxsize=None)
def _trivial_forwarding_wrapper_candidates(
    module: ParsedModule,
) -> tuple[TrivialForwardingWrapperCandidate, ...]:
    candidates = [
        candidate
        for qualname, function in _iter_named_functions(module)
        for candidate in (
            _trivial_forwarding_wrapper_candidate(module, qualname, function),
        )
        if candidate is not None
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (candidate.file_path, candidate.line, candidate.qualname),
        )
    )


def _public_api_private_delegate_shell_candidates(
    modules: Sequence[ParsedModule],
    config: DetectorConfig,
) -> tuple[PublicApiPrivateDelegateShellCandidate, ...]:
    min_external_callsites = max(2, config.min_registration_sites)
    callsites_by_target = _external_callsites_by_target(modules)
    candidates: list[PublicApiPrivateDelegateShellCandidate] = []
    for module in modules:
        top_level_lines = _top_level_symbol_lines(module)
        for wrapper_candidate in _trivial_forwarding_wrapper_candidates(module):
            if not _is_public_module_api_qualname(wrapper_candidate.qualname):
                continue
            delegate_root_symbol = _delegate_root_symbol(
                wrapper_candidate.delegate_symbol
            )
            if not _is_private_symbol_name(delegate_root_symbol):
                continue
            wrapper_symbol = f"{module.module_name}.{wrapper_candidate.qualname}"
            external_callsites = tuple(
                site
                for site in _matching_external_callsites(
                    callsites_by_target,
                    target_symbol=wrapper_symbol,
                )
                if site.module_name != module.module_name
            )
            if len(external_callsites) < min_external_callsites:
                continue
            candidates.append(
                PublicApiPrivateDelegateShellCandidate(
                    wrapper=wrapper_candidate,
                    module_name=module.module_name,
                    delegate_root_symbol=delegate_root_symbol,
                    delegate_root_line=top_level_lines.get(delegate_root_symbol),
                    external_callsites=external_callsites,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.wrapper.file_path,
                item.wrapper.line,
                item.wrapper.qualname,
            ),
        )
    )


def _public_api_private_delegate_family_candidates(
    modules: Sequence[ParsedModule],
    config: DetectorConfig,
) -> tuple[PublicApiPrivateDelegateFamilyCandidate, ...]:
    min_wrapper_count = max(2, config.min_registration_sites)
    min_external_callsites = max(2, config.min_registration_sites)
    callsites_by_target = _external_callsites_by_target(modules)
    grouped_wrappers: dict[
        tuple[str, str, str], list[TrivialForwardingWrapperCandidate]
    ] = defaultdict(list)
    delegate_lines: dict[tuple[str, str, str], int | None] = {}
    for module in modules:
        top_level_lines = _top_level_symbol_lines(module)
        for wrapper_candidate in _trivial_forwarding_wrapper_candidates(module):
            if not _is_public_module_api_qualname(wrapper_candidate.qualname):
                continue
            delegate_root_symbol = _delegate_root_symbol(
                wrapper_candidate.delegate_symbol
            )
            if not _is_private_symbol_name(delegate_root_symbol):
                continue
            key = (str(module.path), module.module_name, delegate_root_symbol)
            grouped_wrappers[key].append(wrapper_candidate)
            delegate_lines.setdefault(key, top_level_lines.get(delegate_root_symbol))
    candidates: list[PublicApiPrivateDelegateFamilyCandidate] = []
    for (file_path, module_name, delegate_root_symbol), wrappers in grouped_wrappers.items():
        if len(wrappers) < min_wrapper_count:
            continue
        external_callsites = tuple(
            sorted(
                {
                    site
                    for wrapper in wrappers
                    for site in _matching_external_callsites(
                        callsites_by_target,
                        target_symbol=f"{module_name}.{wrapper.qualname}",
                    )
                    if site.module_name != module_name
                },
                key=lambda item: (
                    item.location.file_path,
                    item.location.line,
                    item.location.symbol,
                    item.module_name,
                ),
            )
        )
        if len(external_callsites) < min_external_callsites:
            continue
        candidates.append(
            PublicApiPrivateDelegateFamilyCandidate(
                file_path=file_path,
                module_name=module_name,
                delegate_root_symbol=delegate_root_symbol,
                delegate_root_line=delegate_lines[(file_path, module_name, delegate_root_symbol)],
                wrappers=tuple(sorted(wrappers, key=lambda item: (item.line, item.qualname))),
                external_callsites=external_callsites,
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.delegate_root_symbol,
                item.wrappers[0].line,
            ),
        )
    )


def _policy_selector_source_exprs(
    selector_call: ast.Call,
) -> tuple[str, ...]:
    return tuple(
        ast.unparse(value)
        for value in (
            *selector_call.args,
            *(keyword.value for keyword in selector_call.keywords if keyword.arg is not None),
        )
    )


def _looks_like_self_selector_source(expr: str) -> bool:
    return expr == "self" or expr.startswith("self.") or expr == "cls" or expr.startswith("cls.")


def _nominal_policy_surface_method_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> NominalPolicySurfaceMethodCandidate | None:
    if "." not in qualname:
        return None
    owner_class_name, method_name = qualname.rsplit(".", 1)
    if method_name.startswith("_") or (method_name.startswith("__") and method_name.endswith("__")):
        return None
    body = _trim_docstring_body(function.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    returned = body[0].value
    if not isinstance(returned, ast.Call):
        return None
    chain = _call_chain_from_outer_call(returned)
    if len(chain) != 2:
        return None
    outer_call, selector_call = chain
    if not isinstance(selector_call.func, ast.Attribute):
        return None
    selector_method_name = selector_call.func.attr
    if not selector_method_name.startswith("for_"):
        return None
    policy_root_parts = _ast_attribute_chain(selector_call.func.value)
    if policy_root_parts is None:
        return None
    selector_source_exprs = _policy_selector_source_exprs(selector_call)
    if not selector_source_exprs or not any(
        _looks_like_self_selector_source(expr) for expr in selector_source_exprs
    ):
        return None
    allowed_roots = _function_param_names(function) | {"self", "cls"}
    transported_values = _call_chain_transport_values(chain)
    if not transported_values:
        return None
    if not all(
        _is_transport_expression(value, allowed_roots=allowed_roots)
        for value in transported_values
    ):
        return None
    policy_member_name = _call_name(outer_call.func) or ast.unparse(outer_call.func)
    return NominalPolicySurfaceMethodCandidate(
        file_path=str(module.path),
        line=function.lineno,
        qualname=qualname,
        owner_class_name=owner_class_name,
        method_name=method_name,
        policy_root_symbol=".".join(policy_root_parts),
        selector_method_name=selector_method_name,
        policy_member_name=policy_member_name,
        selector_source_exprs=selector_source_exprs,
        transported_value_sources=tuple(sorted({ast.unparse(value) for value in transported_values})),
    )


def _nominal_policy_surface_family_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[NominalPolicySurfaceFamilyCandidate, ...]:
    min_family_size = max(2, config.min_registration_sites)
    method_candidates = tuple(
        candidate
        for qualname, function in _iter_named_functions(module)
        for candidate in (
            _nominal_policy_surface_method_candidate(module, qualname, function),
        )
        if candidate is not None
    )
    grouped: dict[
        tuple[str, str, str, tuple[str, ...]], list[NominalPolicySurfaceMethodCandidate]
    ] = defaultdict(list)
    for candidate in method_candidates:
        grouped[
            (
                candidate.owner_class_name,
                candidate.policy_root_symbol,
                candidate.selector_method_name,
                candidate.selector_source_exprs,
            )
        ].append(candidate)
    return tuple(
        sorted(
            (
                NominalPolicySurfaceFamilyCandidate(
                    methods=tuple(
                        sorted(candidates, key=lambda item: (item.line, item.qualname))
                    ),
                )
                for (
                    owner_class_name,
                    policy_root_symbol,
                    selector_method_name,
                    selector_source_exprs,
                ), candidates in grouped.items()
                if len(candidates) >= min_family_size
            ),
            key=lambda item: (
                item.file_path,
                item.owner_class_name,
                item.policy_root_symbol,
                item.methods[0].line,
            ),
        )
    )


def _function_wrapper_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> FunctionWrapperCandidate | None:
    body = _trim_docstring_body(function.body)
    if not body:
        return None
    class_name = qualname.rsplit(".", 1)[0] if "." in qualname else None
    allowed_roots = _function_param_names(function) | {"self", "cls"}

    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
        returned = body[0].value
        if not isinstance(returned, ast.Call):
            return None
        delegate_symbol = _wrapper_delegate_symbol(
            returned.func,
            class_name=class_name,
        )
        if delegate_symbol is None:
            return None
        values = list(returned.args) + [
            keyword.value for keyword in returned.keywords if keyword.arg is not None
        ]
        if not all(
            _is_transport_expression(value, allowed_roots=allowed_roots)
            for value in values
        ):
            return None
        return FunctionWrapperCandidate(
            file_path=str(module.path),
            qualname=qualname,
            lineno=function.lineno,
            delegate_symbol=delegate_symbol,
            wrapper_kind="direct",
            statement_count=len(body),
        )

    if (
        len(body) == 2
        and isinstance(body[0], ast.Assign)
        and len(body[0].targets) == 1
        and isinstance(body[0].targets[0], ast.Name)
        and isinstance(body[1], ast.Return)
        and body[1].value is not None
        and isinstance(body[0].value, ast.Call)
    ):
        bound_name = body[0].targets[0].id
        delegate_symbol = _wrapper_delegate_symbol(
            body[0].value.func,
            class_name=class_name,
        )
        if delegate_symbol is None:
            return None
        values = list(body[0].value.args) + [
            keyword.value
            for keyword in body[0].value.keywords
            if keyword.arg is not None
        ]
        if not all(
            _is_transport_expression(value, allowed_roots=allowed_roots)
            for value in values
        ):
            return None
        projected_attributes = _projected_attribute_names(
            body[1].value,
            bound_name=bound_name,
        )
        if projected_attributes is None:
            return None
        return FunctionWrapperCandidate(
            file_path=str(module.path),
            qualname=qualname,
            lineno=function.lineno,
            delegate_symbol=delegate_symbol,
            wrapper_kind="projection",
            statement_count=len(body),
            projected_attributes=projected_attributes,
        )

    return None


def _function_wrapper_candidates(
    module: ParsedModule,
) -> tuple[FunctionWrapperCandidate, ...]:
    candidates = [
        candidate
        for qualname, function in _iter_named_functions(module)
        for candidate in (_function_wrapper_candidate(module, qualname, function),)
        if candidate is not None
    ]
    return tuple(sorted(candidates, key=lambda item: (item.file_path, item.lineno, item.qualname)))


def _wrapper_chain_candidates(
    module: ParsedModule,
) -> tuple[WrapperChainCandidate, ...]:
    candidates = _function_wrapper_candidates(module)
    if len(candidates) < 2:
        return ()
    by_symbol = {candidate.qualname: candidate for candidate in candidates}
    inbound = Counter(
        candidate.delegate_symbol
        for candidate in candidates
        if candidate.delegate_symbol in by_symbol
    )
    chains: list[WrapperChainCandidate] = []
    for candidate in candidates:
        if inbound[candidate.qualname] > 0:
            continue
        current = candidate
        chain = [candidate]
        seen = {candidate.qualname}
        while current.delegate_symbol in by_symbol:
            next_candidate = by_symbol[current.delegate_symbol]
            if next_candidate.qualname in seen:
                break
            chain.append(next_candidate)
            seen.add(next_candidate.qualname)
            current = next_candidate
        if len(chain) < 2:
            continue
        chains.append(
            WrapperChainCandidate(
                file_path=str(module.path),
                wrappers=tuple(chain),
                leaf_delegate_symbol=current.delegate_symbol,
            )
        )
    return tuple(
        sorted(
            chains,
            key=lambda item: (-len(item.wrappers), item.wrappers[0].lineno),
        )
    )


def _pipeline_body_stages(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[PipelineAssemblyStage, ...] | None:
    body = list(function.body)
    if body and _is_docstring_expr(body[0]):
        body = body[1:]
    if len(body) < 2:
        return None
    stages: list[PipelineAssemblyStage] = []
    for statement in body:
        stage = _pipeline_stage(statement)
        if stage is None:
            return None
        stages.append(stage)
    if not stages or stages[-1].kind != _PIPELINE_RETURN_STAGE:
        return None
    return tuple(stages)


def _pipeline_stage(statement: ast.stmt) -> PipelineAssemblyStage | None:
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(statement.value, ast.Call):
            return None
        output_arity = _assignment_target_arity(statement.targets[0])
        if output_arity is None:
            return None
        callee_name = _call_name(statement.value.func)
        if callee_name is None:
            return None
        keyword_names = tuple(
            keyword.arg for keyword in statement.value.keywords if keyword.arg is not None
        )
        return PipelineAssemblyStage(
            kind=_PIPELINE_ASSIGN_STAGE,
            callee_name=callee_name,
            output_arity=output_arity,
            arg_count=len(statement.value.args) + len(keyword_names),
            keyword_names=keyword_names,
        )
    if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call):
        callee_name = _call_name(statement.value.func)
        if callee_name is None:
            return None
        keyword_names = tuple(
            keyword.arg for keyword in statement.value.keywords if keyword.arg is not None
        )
        return PipelineAssemblyStage(
            kind=_PIPELINE_RETURN_STAGE,
            callee_name=callee_name,
            output_arity=0,
            arg_count=len(statement.value.args) + len(keyword_names),
            keyword_names=keyword_names,
        )
    return None


def _assignment_target_arity(target: ast.AST) -> int | None:
    if isinstance(target, ast.Name):
        return 1
    if isinstance(target, (ast.Tuple, ast.List)):
        if not target.elts or not all(isinstance(item, ast.Name) for item in target.elts):
            return None
        return len(target.elts)
    return None


def _result_assembly_pipeline_functions(
    module: ParsedModule,
) -> tuple[ResultAssemblyPipelineFunction, ...]:
    functions: list[ResultAssemblyPipelineFunction] = []
    for qualname, function in _iter_named_functions(module):
        stages = _pipeline_body_stages(function)
        if stages is None:
            continue
        functions.append(
            ResultAssemblyPipelineFunction(
                file_path=str(module.path),
                qualname=qualname,
                lineno=function.lineno,
                stages=stages,
            )
        )
    return tuple(sorted(functions, key=lambda item: (item.lineno, item.qualname)))


def _shared_pipeline_tail(
    left: ResultAssemblyPipelineFunction,
    right: ResultAssemblyPipelineFunction,
) -> tuple[PipelineAssemblyStage, ...]:
    shared: list[PipelineAssemblyStage] = []
    left_index = len(left.stages) - 1
    right_index = len(right.stages) - 1
    while left_index >= 0 and right_index >= 0:
        left_stage = left.stages[left_index]
        right_stage = right.stages[right_index]
        if left_stage.shape_key != right_stage.shape_key:
            break
        shared.append(left_stage)
        left_index -= 1
        right_index -= 1
    return tuple(reversed(shared))


def _repeated_result_assembly_pipeline_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[RepeatedResultAssemblyPipelineCandidate, ...]:
    functions = _result_assembly_pipeline_functions(module)
    if len(functions) < 2:
        return ()
    grouped_functions: dict[
        tuple[tuple[object, ...], ...],
        tuple[tuple[PipelineAssemblyStage, ...], set[ResultAssemblyPipelineFunction]],
    ] = {}
    for left, right in combinations(functions, 2):
        shared_tail = _shared_pipeline_tail(left, right)
        if len(shared_tail) < config.min_shared_pipeline_stages:
            continue
        if len(shared_tail) >= len(left.stages) or len(shared_tail) >= len(right.stages):
            continue
        if shared_tail[-1].kind != _PIPELINE_RETURN_STAGE:
            continue
        distinct_stage_names = {stage.callee_name for stage in shared_tail}
        if len(distinct_stage_names) < config.min_shared_pipeline_stages - 1:
            continue
        key = tuple(stage.shape_key for stage in shared_tail)
        if key not in grouped_functions:
            grouped_functions[key] = (shared_tail, set())
        grouped_functions[key][1].update((left, right))

    candidates = [
        RepeatedResultAssemblyPipelineCandidate(
            file_path=str(module.path),
            shared_tail=shared_tail,
            functions=tuple(
                sorted(grouped, key=lambda item: (item.lineno, item.qualname))
            ),
        )
        for shared_tail, grouped in grouped_functions.values()
        if len(grouped) >= 2
    ]
    filtered_candidates: list[RepeatedResultAssemblyPipelineCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -len(item.shared_tail),
            -len(item.functions),
            item.functions[0].qualname,
        ),
    ):
        candidate_function_names = tuple(
            function.qualname for function in candidate.functions
        )
        if any(
            len(existing.shared_tail) >= len(candidate.shared_tail)
            and candidate_function_names
            == tuple(function.qualname for function in existing.functions)
            for existing in filtered_candidates
        ):
            continue
        filtered_candidates.append(candidate)
    return tuple(filtered_candidates)


def _direct_forwarded_parameter_names(
    call: ast.Call,
    *,
    parameter_names: set[str],
) -> tuple[str, ...] | None:
    forwarded: list[str] = []
    seen: set[str] = set()
    for argument in call.args:
        if isinstance(argument, ast.Name) and argument.id in parameter_names:
            if argument.id not in seen:
                seen.add(argument.id)
                forwarded.append(argument.id)
            continue
        return None
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        if isinstance(keyword.value, ast.Name) and keyword.value.id in parameter_names:
            if keyword.value.id not in seen:
                seen.add(keyword.value.id)
                forwarded.append(keyword.value.id)
            continue
        return None
    return tuple(forwarded)


def _qualified_call_display_name(node: ast.Call) -> str:
    return ast.unparse(node.func)


def _nested_builder_shell_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[NestedBuilderShellCandidate, ...]:
    candidates: list[NestedBuilderShellCandidate] = []
    for qualname, function in _iter_named_functions(module):
        body = _trim_docstring_body(list(function.body))
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            continue
        returned = body[0].value
        if not isinstance(returned, ast.Call) or returned.args:
            continue
        outer_callee_name = _call_name(returned.func)
        if outer_callee_name is None:
            continue
        parameter_names = _function_param_names(function) - {"self", "cls"}
        if len(parameter_names) < config.min_nested_builder_forwarded_params:
            continue
        nested_matches: list[
            tuple[str, str, tuple[str, ...]]
        ] = []
        for keyword in returned.keywords:
            if keyword.arg is None or not isinstance(keyword.value, ast.Call):
                continue
            nested_callee_name = _qualified_call_display_name(keyword.value)
            if (
                not nested_callee_name
                or _call_name(keyword.value.func) == outer_callee_name
            ):
                continue
            forwarded = _direct_forwarded_parameter_names(
                keyword.value,
                parameter_names=parameter_names,
            )
            if forwarded is None:
                continue
            if len(forwarded) < config.min_nested_builder_forwarded_params:
                continue
            nested_matches.append((keyword.arg, nested_callee_name, forwarded))
        if len(nested_matches) != 1:
            continue
        nested_field_name, nested_callee_name, forwarded_parameter_names = (
            nested_matches[0]
        )
        residue_keywords = tuple(
            keyword
            for keyword in returned.keywords
            if keyword.arg is not None and keyword.arg != nested_field_name
        )
        if not residue_keywords:
            continue
        residue_source_names = tuple(
            sorted(
                {
                    root_name
                    for keyword in residue_keywords
                    for root_name in _expression_root_names(keyword.value)
                    if root_name in (parameter_names - set(forwarded_parameter_names))
                }
            )
        )
        if not residue_source_names:
            continue
        candidates.append(
            NestedBuilderShellCandidate(
                file_path=str(module.path),
                qualname=qualname,
                lineno=function.lineno,
                outer_callee_name=outer_callee_name,
                nested_field_name=nested_field_name,
                nested_callee_name=nested_callee_name,
                forwarded_parameter_names=forwarded_parameter_names,
                residue_field_names=tuple(
                    keyword.arg for keyword in residue_keywords if keyword.arg is not None
                ),
                residue_source_names=residue_source_names,
            )
        )
    return tuple(sorted(candidates, key=lambda item: (item.file_path, item.lineno)))


def _indexed_family_wrapper_candidates(
    module: ParsedModule,
) -> tuple[IndexedFamilyWrapperCandidate, ...]:
    candidates: list[IndexedFamilyWrapperCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
            continue
        value = node.body[0].value
        if not isinstance(value, ast.ListComp) or len(value.generators) != 1:
            continue
        generator = value.generators[0]
        if not isinstance(generator.target, ast.Name) or generator.target.id != "item":
            continue
        if not isinstance(generator.iter, ast.Call):
            continue
        collector_name = _call_name(generator.iter.func)
        if collector_name not in {
            "_collect_items_from_spec_root",
            "collect_family_items",
        }:
            continue
        if collector_name == "_collect_items_from_spec_root":
            if len(generator.iter.args) < 3:
                continue
            spec_root_name = _call_name(generator.iter.args[0])
            item_type_name = _call_name(generator.iter.args[2])
        else:
            if len(generator.iter.args) < 2:
                continue
            spec_root_name = _call_name(generator.iter.args[1])
            item_type_name = _call_name(generator.iter.args[1])
        if spec_root_name is None or item_type_name is None:
            continue
        if not _is_instance_filter(generator.ifs, item_type_name):
            continue
        candidates.append(
            IndexedFamilyWrapperCandidate(
                function_name=node.name,
                lineno=node.lineno,
                collector_name=collector_name,
                spec_root_name=spec_root_name,
                item_type_name=item_type_name,
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.lineno))


def _is_instance_filter(filters: list[ast.expr], item_type_name: str) -> bool:
    for condition in filters:
        if not isinstance(condition, ast.Call):
            continue
        if _call_name(condition.func) != "isinstance":
            continue
        if len(condition.args) != 2:
            continue
        if (
            not isinstance(condition.args[0], ast.Name)
            or condition.args[0].id != "item"
        ):
            continue
        if _call_name(condition.args[1]) == item_type_name:
            return True
    return False


def _function_has_param(
    function: ast.FunctionDef | ast.AsyncFunctionDef, param_name: str
) -> bool:
    return any(arg.arg == param_name for arg in function.args.args)


def _collect_class_sentinel_attrs(
    module: ast.Module,
) -> dict[str, list[SourceLocation]]:
    grouped: dict[str, list[SourceLocation]] = defaultdict(list)
    for node in _walk_nodes(module):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if not isinstance(stmt.value, ast.Constant):
                continue
            if not isinstance(stmt.value.value, (str, int, bool)):
                continue
            grouped[target.id].append(
                SourceLocation("<module>", stmt.lineno, f"{node.name}.{target.id}")
            )
    return grouped


def _module_compares_attribute(module: ast.Module, attr_name: str) -> bool:
    for node in _walk_nodes(module):
        if isinstance(node, ast.Compare):
            values = [node.left] + list(node.comparators)
            if any(
                isinstance(value, ast.Attribute) and value.attr == attr_name
                for value in values
            ):
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == _GETATTR_BUILTIN and len(node.args) >= 2:
                attr = node.args[1]
                if isinstance(attr, ast.Constant) and attr.value == attr_name:
                    return True
    return False


def _predicate_factory_chain_branch_count(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int | None:
    if not function.body or not isinstance(function.body[0], ast.If):
        return None
    branch_count = 0
    current: ast.stmt | None = function.body[0]
    while isinstance(current, ast.If):
        if not _test_has_call(current.test):
            return None
        if not any(
            isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call)
            for stmt in current.body
        ):
            return None
        branch_count += 1
        current = current.orelse[0] if len(current.orelse) == 1 else None
    if branch_count < 2:
        return None
    return branch_count


def _test_has_call(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Call) for child in _walk_nodes(node))


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
