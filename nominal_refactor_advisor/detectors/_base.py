"""Detector substrate and shared helper machinery.

This module contains the shared detector registry, common base classes, candidate
records, helper functions, and patch/scaffold utilities used by the concrete
detector implementations.
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

from ..ast_tools import (
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
from ..class_index import (
    ClassFamilyIndex,
    IndexedClass,
    _module_import_aliases,
    build_class_family_index,
)
from ..models import (
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
from ..observation_graph import (
    ObservationGraph,
    ObservationKind,
    StructuralExecutionLevel,
    StructuralObservation,
    StructuralObservationCarrier,
)
from ..patterns import PatternId
from ..taxonomy import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    CapabilityTag,
    CertificationLevel,
    ObservationTag,
)
from ._substrate_support import *


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
_NAME_LITERAL = "name"
_EVAL_PARSE_MODE = "eval"


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
        _NAME_LITERAL,
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


def _parallel_keyed_table_axis_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[ParallelKeyedTableAxisCandidate, ...]:
    specs = tuple(
        sorted(
            (
                table_spec
                for module in modules
                for table_spec in _module_keyed_table_axis_specs(module)
            ),
            key=lambda item: (item.file_path, item.line, item.table_name),
        )
    )
    candidates: list[ParallelKeyedTableAxisCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for index, left_spec in enumerate(specs):
        for right_spec in specs[index + 1 :]:
            if left_spec.file_path == right_spec.file_path:
                continue
            if left_spec.key_type_name != right_spec.key_type_name:
                continue
            shared_case_names = tuple(
                sorted(set(left_spec.case_names) & set(right_spec.case_names))
            )
            if len(shared_case_names) < 2:
                continue
            case_overlap_ratio = _case_overlap_ratio(
                left_spec.case_names,
                right_spec.case_names,
            )
            if case_overlap_ratio < 0.8:
                continue
            table_overlap = _identifier_name_overlap(
                left_spec.table_name,
                right_spec.table_name,
            )
            value_overlap = 0.0
            if left_spec.value_shape_name is not None and right_spec.value_shape_name is not None:
                value_overlap = _identifier_name_overlap(
                    left_spec.value_shape_name,
                    right_spec.value_shape_name,
                )
            name_overlap_ratio = max(table_overlap, value_overlap)
            if name_overlap_ratio < 0.5:
                continue
            key = tuple(
                sorted((left_spec.table_name, right_spec.table_name))
            ) + (left_spec.key_type_name,)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                ParallelKeyedTableAxisCandidate(
                    key_type_name=left_spec.key_type_name,
                    left=left_spec,
                    right=right_spec,
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
                item.left.table_name,
                item.right.file_path,
                item.right.table_name,
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

    normalized_test = ast.parse(ast.unparse(test), mode=_EVAL_PARSE_MODE).body
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
        if _constant_string(ast.parse(case_name, mode=_EVAL_PARSE_MODE).body) is None:
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
    if field_name == _NAME_LITERAL or field_name == _SUBJECT_NAME_FIELD or field_name.endswith(
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
        "from abc import ABC\n"
        "from metaclass_registry import AutoRegisterMeta\n\n"
        "class EventHandler(ABC, metaclass=AutoRegisterMeta):\n"
        "    __registry_key__ = \"event_type\"\n"
        "    __skip_if_no_key__ = True\n"
        "    event_type = None\n\n"
        "    @classmethod\n"
        "    def type_for_event_type(cls, event_type):\n"
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
_CLASS_NAME_TOKEN_PATTERN = (
    r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+"
)


def _string_constant_expression(expression: str) -> str | None:
    try:
        node = ast.parse(expression, mode=_EVAL_PARSE_MODE).body
    except SyntaxError:
        return None
    return _constant_string(node)


def _normalized_registry_key_from_class_name(
    class_name: str,
    *,
    stripped_suffix: str | None = None,
) -> str:
    source_name = (
        class_name.removesuffix(stripped_suffix)
        if stripped_suffix
        else class_name
    )
    tokens = _ordered_class_name_tokens(source_name)
    if tokens:
        return "_".join(tokens)
    return source_name.lower()


def _raw_class_name_tokens(name: str) -> tuple[str, ...]:
    return tuple(re.findall(_CLASS_NAME_TOKEN_PATTERN, name.lstrip("_")))


def _shared_registry_key_suffix(class_names: Sequence[str]) -> str | None:
    if len(class_names) < 2:
        return None
    raw_token_lists = tuple(_raw_class_name_tokens(name) for name in class_names)
    lower_token_lists = tuple(
        tuple(token.lower() for token in token_list) for token_list in raw_token_lists
    )
    if not all(token_list for token_list in lower_token_lists):
        return None
    shared_reversed: list[str] = []
    for shared_tokens in zip(
        *(reversed(tokens) for tokens in lower_token_lists),
        strict=False,
    ):
        if len(set(shared_tokens)) != 1:
            break
        shared_reversed.append(shared_tokens[0])
    if not shared_reversed:
        return None
    shared_count = len(shared_reversed)
    if len(lower_token_lists[0]) <= shared_count:
        return None
    return "".join(raw_token_lists[0][-shared_count:])


def _derivable_registry_key_suffix(
    class_names: Sequence[str],
    explicit_key_values: Sequence[str] | None = None,
) -> str | None:
    if not class_names:
        return None
    normalized_names = tuple(class_names)
    suffix_candidates = []
    shared_suffix = _shared_registry_key_suffix(normalized_names)
    if shared_suffix and all(
        name.removesuffix(shared_suffix) for name in normalized_names
    ):
        suffix_candidates.append(shared_suffix)
    suffix_candidates.append("")
    if explicit_key_values is None:
        return suffix_candidates[0]
    for suffix in suffix_candidates:
        stripped_suffix = suffix or None
        derived_values = tuple(
            _normalized_registry_key_from_class_name(
                class_name,
                stripped_suffix=stripped_suffix,
            )
            for class_name in normalized_names
        )
        if tuple(explicit_key_values) == derived_values:
            return stripped_suffix
    return None


def _derived_registry_key_block(
    class_names: Sequence[str],
    *,
    registry_key_attr_name: str = "registry_key",
) -> str:
    stripped_suffix = _derivable_registry_key_suffix(class_names)
    source_name = _NAME_LITERAL
    if stripped_suffix:
        source_name = f'name.removesuffix("{stripped_suffix}")'
    return "\n".join(
        (
            f"    __registry_key__ = \"{registry_key_attr_name}\"",
            "    __skip_if_no_key__ = True",
            "",
            "    @staticmethod",
            "    def _registry_key(name: str, cls):",
            "        del cls",
            f"        tokens = re.findall(r\"{_CLASS_NAME_TOKEN_PATTERN}\", {source_name})",
            "        return \"_\".join(token.lower() for token in tokens)",
            "",
            "    __key_extractor__ = _registry_key",
        )
    )


def _declared_registry_key_block(
    key_attr_name: str,
    *,
    key_type_name: str | None = None,
) -> str:
    type_suffix = f": ClassVar[{key_type_name} | None]" if key_type_name else ""
    return "\n".join(
        (
            f"    __registry_key__ = \"{key_attr_name}\"",
            "    __skip_if_no_key__ = True",
            f"    {key_attr_name}{type_suffix} = None",
        )
    )


def _metaclass_registry_keyed_family_scaffold(
    *,
    root_name: str,
    key_type_name: str,
    key_attr_name: str,
    method_defs: tuple[str, ...],
    returns_instance: bool = True,
) -> str:
    registry_lookup = "cls.__registry__[key]()"
    if not returns_instance:
        registry_lookup = "cls.__registry__[key]"
    lines = [
        "from abc import ABC, abstractmethod",
        "from metaclass_registry import AutoRegisterMeta",
        "from typing import ClassVar",
        "",
        f"class {root_name}(ABC, metaclass=AutoRegisterMeta):",
        _declared_registry_key_block(
            key_attr_name,
            key_type_name=key_type_name,
        ),
        "",
        "    @classmethod",
        f"    def for_key(cls, key: {key_type_name}):",
        f"        return {registry_lookup}",
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
class SubclassTraversalSite:
    file_path: str
    line: int
    symbol: str
    root_expression: str
    materialization_kind: str
    registry_attribute_names: tuple[str, ...]
    filter_names: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.symbol)


@dataclass(frozen=True)
class SubclassTraversalGroup:
    symbols: tuple[str, ...]
    file_paths: tuple[str, ...]
    line_numbers: tuple[int, ...]
    root_expressions: tuple[str, ...]
    materialization_kinds: tuple[str, ...]
    registry_attribute_names: tuple[str, ...]
    filter_names: tuple[str, ...]


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
class ParallelKeyedTableAxisCandidate:
    key_type_name: str
    left: _KeyedTableAxisSpec
    right: _KeyedTableAxisSpec
    shared_case_names: tuple[str, ...]
    case_overlap_ratio: float
    name_overlap_ratio: float

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(self.left.file_path, self.left.line, self.left.table_name),
            SourceLocation(self.right.file_path, self.right.line, self.right.table_name),
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
class DerivedQueryIndexCandidate:
    file_path: str
    line_numbers: tuple[int, ...]
    function_names: tuple[str, ...]
    source_expression: str
    query_key_names: tuple[str, ...]
    return_expressions: tuple[str, ...]
    exception_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(self.file_path, line, function_name)
            for function_name, line in zip(
                self.function_names,
                self.line_numbers,
                strict=True,
            )
        )


@dataclass(frozen=True)
class RuntimeAdapterShellCandidate(FunctionLineWitnessCandidate):
    adapter_class_name: str
    source_name: str
    copied_field_names: tuple[str, ...]
    resolver_field_names: tuple[str, ...]
    resolver_table_names: tuple[str, ...]
    selector_field_names: tuple[str, ...]
    evidence_locations: tuple[SourceLocation, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return self.evidence_locations


@dataclass(frozen=True)
class KeywordBagAdapterCandidate(FunctionLineWitnessCandidate):
    source_name: str
    key_names: tuple[str, ...]
    source_field_names: tuple[str, ...]


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

__all__ = tuple(name for name in globals() if not name.startswith("__"))
