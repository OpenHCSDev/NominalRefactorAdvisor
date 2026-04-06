from __future__ import annotations

import ast
import inspect
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, ClassVar, Sequence, TypeVar, cast

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


@dataclass(frozen=True)
class DetectorConfig:
    min_duplicate_statements: int = 3
    min_string_cases: int = 3
    min_attribute_probes: int = 2
    min_builder_keywords: int = 3
    min_export_keys: int = 3
    min_registration_sites: int = 2
    min_hardcoded_string_sites: int = 3
    min_orchestration_function_lines: int = 150
    min_orchestration_branches: int = 15
    min_orchestration_calls: int = 50
    min_shared_parameters: int = 5
    min_parameter_family_function_lines: int = 40

    @classmethod
    def from_namespace(cls, namespace: Any) -> "DetectorConfig":
        namespace_values = vars(namespace)
        return cls(
            min_duplicate_statements=int(namespace.min_duplicate_statements),
            min_string_cases=int(namespace.min_string_cases),
            min_attribute_probes=int(namespace.min_attribute_probes),
            min_builder_keywords=int(namespace.min_builder_keywords),
            min_export_keys=int(namespace.min_export_keys),
            min_registration_sites=int(namespace.min_registration_sites),
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
        )


class IssueDetector(ABC):
    detector_id: str
    detector_priority: ClassVar[int] = 0
    _registered_detector_types: ClassVar[list[type["IssueDetector"]]] = []
    _definition_order: ClassVar[int] = 0
    _detector_registration_index: ClassVar[int]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        cls._detector_registration_index = IssueDetector._definition_order
        IssueDetector._definition_order += 1
        IssueDetector._registered_detector_types.append(cls)

    @classmethod
    def registered_detector_types(cls) -> tuple[type["IssueDetector"], ...]:
        return tuple(
            sorted(
                cls._registered_detector_types,
                key=lambda item: (
                    item.detector_priority,
                    item._detector_registration_index,
                ),
            )
        )

    def detect(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings = self._collect_findings(modules, config)
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


class EvidenceOnlyPerModuleDetector(PerModuleIssueDetector):
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
            callee_names = tuple(
                sorted(
                    {
                        name
                        for subnode in ast.walk(node)
                        if isinstance(subnode, ast.Call)
                        for name in (_callee_name(subnode),)
                        if name is not None
                    }
                )
            )
            profiles.append(
                FunctionProfile(
                    file_path=str(module.path),
                    qualname=".".join((*self.class_stack, node.name)),
                    lineno=node.lineno,
                    line_count=end_lineno - node.lineno + 1,
                    branch_count=sum(
                        isinstance(subnode, ast.If) for subnode in ast.walk(node)
                    ),
                    call_count=sum(
                        isinstance(subnode, ast.Call) for subnode in ast.walk(node)
                    ),
                    callee_names=callee_names,
                    parameter_names=_parameter_names(node),
                )
            )
            self.generic_visit(node)

    Visitor().visit(module.module)
    return tuple(sorted(profiles, key=lambda item: (item.lineno, item.qualname)))


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


def _enum_dispatch_from_match(node: ast.Match) -> tuple[str, tuple[str, ...]] | None:
    cases = []
    for case in node.cases:
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
        for subnode in ast.walk(function):
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


def _nominal_strategy_scaffold(candidate: EnumStrategyDispatchCandidate) -> str:
    axis_tail = (
        candidate.dispatch_axis.split(".")[-1]
        .replace("_", " ")
        .title()
        .replace(" ", "")
    )
    root_name = f"{axis_tail}Runner"
    lines = [
        f"class {root_name}(ABC):",
        "    @abstractmethod",
        "    def run(self, ctx): ...",
        "",
    ]
    for case_name in candidate.case_names:
        case_tail = case_name.split(".")[-1].replace("_", " ").title().replace(" ", "")
        lines.append(f"class {case_tail}{root_name}({root_name}): ...")
    return "\n".join(lines)


def _nominal_strategy_patch(candidate: EnumStrategyDispatchCandidate) -> str:
    axis_tail = (
        candidate.dispatch_axis.split(".")[-1]
        .replace("_", " ")
        .title()
        .replace(" ", "")
    )
    root_name = f"{axis_tail}Runner"
    return (
        f"# Replace `{candidate.dispatch_axis}` branching with a nominal runner family\n"
        f"runner = {root_name}.for_mode({candidate.dispatch_axis})\n"
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
    for subnode in ast.walk(node):
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
    for subnode in ast.walk(node):
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


_TAG_PARAM_NAMES = frozenset({"kind", "mode", "type", "tag", "backend"})


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
    for subnode in ast.walk(expr):
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
            for subnode in ast.walk(init_method):
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
                for subnode in ast.walk(method):
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
            for inner_node in ast.walk(subnode):
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


def _shared_nonobject_bases(classes: tuple[ast.ClassDef, ...]) -> bool:
    base_sets = []
    for node in classes:
        base_sets.append(
            {
                ast.unparse(base)
                for base in node.bases
                if ast.unparse(base) not in {"object"}
            }
        )
    if not base_sets:
        return False
    shared = set.intersection(*base_sets)
    return bool(shared)


def _structural_confusability_candidates(
    module: ParsedModule,
) -> tuple[StructuralConfusabilityCandidate, ...]:
    class_nodes = [
        node for node in module.module.body if isinstance(node, ast.ClassDef)
    ]
    candidates: list[StructuralConfusabilityCandidate] = []
    for qualname, function in _iter_named_functions(module):
        for parameter_name in _parameter_names(function):
            observed_method_names = tuple(
                sorted(
                    {
                        subnode.func.attr
                        for subnode in ast.walk(function)
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
            if _shared_nonobject_bases(confusable_classes):
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
    if field_name in {"class_name", "function_name", "registry_name", "subject_name"}:
        roles.extend(("witness_subject", "witness_name_payload"))
    if field_name == "name_family" or field_name.endswith("_names"):
        roles.extend(("witness_name_family", "witness_name_payload"))
    return tuple(dict.fromkeys(roles))


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
        if not node.name.endswith("Candidate"):
            continue
        field_names = _annassign_field_names(node)
        normalized_role_fields = _normalized_witness_role_fields(field_names)
        normalized_roles = tuple(role_name for role_name, _ in normalized_role_fields)
        if len(normalized_roles) < 3:
            continue
        if {
            "witness_file_path",
            "witness_line",
            "witness_subject",
        } - set(normalized_roles):
            continue
        candidates.append(
            WitnessCarrierClassCandidate(
                file_path=str(module.path),
                line=node.lineno,
                subject_name=node.name,
                name_family=field_names,
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
    shared_role_names = cast(
        tuple[str, ...],
        tuple(
            sorted(
                set.intersection(
                    *(set(candidate.normalized_roles) for candidate in classes)
                )
            )
        ),
    )
    if len(shared_role_names) < 3:
        return ()
    candidate_shared_role_names: tuple[str, ...] = tuple(shared_role_names)
    return (
        WitnessCarrierFamilyCandidate(
            file_path=str(module.path),
            class_names=tuple(candidate.class_name for candidate in classes),
            line_numbers=tuple(candidate.line for candidate in classes),
            shared_role_names=candidate_shared_role_names,
        ),
    )


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
        "class EventHandler(ABC):\n"
        f"    _registry = {{}}\n"
        f"    def __init_subclass__(cls, registry_key=None, **kwargs): ...\n"
        f"    @classmethod\n    def registered_types(cls): ..."
    )


def _manual_registry_patch(candidate: ManualRegistryCandidate) -> str:
    return (
        f"# Replace decorator `{candidate.decorator_name}` and registry `{candidate.registry_name}`\n"
        "# with `__init_subclass__` or a metaclass so class creation and registration are one event."
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
        "class WitnessCarrier(ABC):",
        "    file_path: str",
        "    line: int",
        "    subject_name: str",
        "",
        "@dataclass(frozen=True)",
        f"class {candidate.class_names[0]}(WitnessCarrier): ...",
    ]
    return "\n".join(lines)


def _witness_carrier_family_patch(
    candidate: WitnessCarrierFamilyCandidate,
) -> str:
    return (
        f"# Introduce one nominal witness carrier root for {candidate.class_names}.\n"
        f"# Move shared witness roles {candidate.shared_role_names} into the base class and keep only fiber-specific payload in each leaf candidate."
    )


_WITNESS_NAME_PAYLOAD_ROLE = "witness_name_payload"
_WITNESS_LINE_ROLE = "witness_line"
_WITNESS_MIXIN_ROLE_NAMES = (
    _WITNESS_NAME_PAYLOAD_ROLE,
    _WITNESS_LINE_ROLE,
)


@dataclass(frozen=True)
class WitnessMixinRoleSpec:
    mixin_name: str
    scaffold: str


_WITNESS_MIXIN_ROLE_SPECS = {
    _WITNESS_NAME_PAYLOAD_ROLE: WitnessMixinRoleSpec(
        mixin_name="NameBearingMixin",
        scaffold=(
            "class NameBearingMixin(ABC):\n"
            "    @property\n"
            "    @abstractmethod\n"
            "    def name_family(self) -> tuple[str, ...]: ...\n\n"
            "    @property\n"
            "    def subject_name(self) -> str | None:\n"
            "        return self.name_family[0] if self.name_family else None"
        ),
    ),
    _WITNESS_LINE_ROLE: WitnessMixinRoleSpec(
        mixin_name="SourceLocusMixin",
        scaffold=(
            "class SourceLocusMixin(ABC):\n"
            "    @property\n"
            "    @abstractmethod\n"
            "    def line(self) -> int: ..."
        ),
    ),
}


def _witness_mixin_role_spec(role_name: str) -> WitnessMixinRoleSpec:
    try:
        return _WITNESS_MIXIN_ROLE_SPECS[role_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported witness mixin role: {role_name}") from exc


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
                f"class {candidate.class_names[0]}(WitnessCarrier, {mixin_names}): ...",
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
        "# Normalize the leaf carriers onto the shared witness base plus those mixins.\n"
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
class ManualFamilyRosterCandidate:
    file_path: str
    line: int
    owner_name: str
    member_names: tuple[str, ...]
    family_base_name: str
    constructor_style: str


@dataclass(frozen=True)
class FragmentedFamilyAuthorityCandidate:
    file_path: str
    mapping_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    key_family_name: str
    shared_keys: tuple[str, ...]
    total_keys: tuple[str, ...]


@dataclass(frozen=True)
class WitnessCarrierCandidate(ABC):
    file_path: str
    line: int
    subject_name: str
    name_family: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.subject_name)

    @property
    def class_name(self) -> str:
        return self.subject_name


class NameFamilyClassNamesMixin(ABC):
    name_family: tuple[str, ...]

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.name_family


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
    @property
    def keyword_names(self) -> tuple[str, ...]:
        return self.name_family


@dataclass(frozen=True)
class PropertyAliasHookGroup:
    file_path: str
    base_name: str
    property_name: str
    returned_attribute: str
    class_names: tuple[str, ...]
    line_numbers: tuple[int, ...]


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
    WitnessCarrierCandidate, NameFamilyClassNamesMixin
):
    parameter_name: str
    observed_method_names: tuple[str, ...]

    @property
    def function_name(self) -> str:
        return self.subject_name


@dataclass(frozen=True)
class WitnessCarrierClassCandidate(WitnessCarrierCandidate):
    normalized_roles: tuple[str, ...]
    normalized_role_fields: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.name_family


@dataclass(frozen=True)
class WitnessCarrierFamilyCandidate:
    file_path: str
    class_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    shared_role_names: tuple[str, ...]


@dataclass(frozen=True)
class WitnessMixinEnforcementCandidate:
    file_path: str
    class_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    role_field_names: tuple[tuple[str, tuple[str, ...]], ...]


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
            metrics=RepeatedMethodMetrics(
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
            "The host already provides zero-delay registration via `__init_subclass__` or a metaclass."
        ),
        capability_gap="zero-delay class registration with collision checks and runtime provenance",
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
        title="Detector witness carriers should share one nominal base",
        why=(
            "Several frozen dataclass witness carriers repeat the same provenance and focal-subject roles under different field names. "
            "That leaves one witness family structurally expanded instead of giving it one nominal carrier root."
        ),
        capability_gap="one authoritative witness carrier base for a detector-local witness family",
        relation_context="same witness family repeats a renamed provenance spine across sibling carrier classes",
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
                f"Carrier classes {', '.join(witness_candidate.class_names)} repeat the same witness roles {witness_candidate.shared_role_names} under renamed fields and should inherit one nominal base carrier."
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
    classes = _witness_carrier_class_candidates(module)
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
        title="Renamed semantic witness slices should become mixins",
        why=(
            "Several carrier classes repeat the same semantic slice under renamed fields such as `class_name` vs `class_names`. "
            "One shared base is not enough when those slices are orthogonal; the architecture wants reusable mixins composed through multiple inheritance."
        ),
        capability_gap="one authoritative witness spine plus reusable semantic-role mixins",
        relation_context="same witness family repeats renamed semantic slices that overlap orthogonally across sibling carriers",
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
            metrics=RepeatedMethodMetrics(
                duplicate_site_count=len(hook_group.class_names),
                statement_count=1,
                class_count=len(hook_group.class_names),
                method_symbols=tuple(
                    f"{class_name}.{hook_group.property_name}"
                    for class_name in hook_group.class_names
                ),
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
        title="Manual class registration should become AutoRegisterMeta",
        why=(
            "The docs say repeated class-level registration boilerplate is a class-level non-orthogonal algorithm. "
            "It should move into one authoritative metaclass or registry base so abstract-class skipping, uniqueness, "
            "and inheritance behavior are enforced in one place."
        ),
        capability_gap="single authoritative class-registration algorithm with nominal class identity",
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
            "say the local rule family should be moved into an authoritative registry, dataclass table, "
            "or another closed dispatch object instead of repeating inline branch logic."
        ),
        capability_gap="single authoritative dispatch representation for a closed local rule family",
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
            "suggest the code is using a weaker representation than the domain requires."
        ),
        capability_gap="closed-family dispatch with stable nominal keys",
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
            "domain axis is real but undeclared. Replace the literal-ID branches with an enum, nominal "
            "registry, or polymorphic family and dispatch once."
        ),
        capability_gap="closed-family dispatch with stable nominal keys",
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
        title="Parallel scoped-shape wrappers should become a polymorphic spec family",
        why=(
            "Parallel wrapper functions plus parallel spec declarations mean the code already has a hidden "
            "strategy family, but it is encoded as duplicated procedural glue. The docs prefer moving the shared "
            "algorithm into an ABC and letting polymorphic spec classes own the node family differences."
        ),
        capability_gap="single authoritative polymorphic observation-spec family",
        relation_context="same scoped-observation wrapper skeleton repeated across multiple shape builders",
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
        wrapper_function_items: tuple[ScopedShapeWrapperFunction, ...] = (
            _collect_typed_family_items(
                module,
                ScopedShapeWrapperFunctionFamily,
                ScopedShapeWrapperFunction,
            )
        )
        wrapper_functions = {
            item.function_name: item for item in wrapper_function_items
        }
        wrapper_spec_items: tuple[ScopedShapeWrapperSpec, ...] = (
            _collect_typed_family_items(
                module,
                ScopedShapeWrapperSpecFamily,
                ScopedShapeWrapperSpec,
            )
        )
        wrapper_specs = [
            spec
            for spec in wrapper_spec_items
            if spec.function_name in wrapper_functions
            and spec.node_types == wrapper_functions[spec.function_name].node_types
        ]
        if len(wrapper_specs) < 2:
            return []
        evidence_items = [
            SourceLocation(str(module.path), spec.lineno, spec.spec_name)
            for spec in wrapper_specs[:6]
        ]
        evidence_items.extend(
            SourceLocation(
                str(module.path),
                wrapper_functions[spec.function_name].lineno,
                wrapper_functions[spec.function_name].function_name,
            )
            for spec in wrapper_specs[:6]
        )
        evidence = tuple(
            sorted(
                evidence_items,
                key=lambda item: (item.line, item.symbol),
            )[:8]
        )
        function_names = ", ".join(spec.function_name for spec in wrapper_specs)
        spec_names = ", ".join(spec.spec_name for spec in wrapper_specs)
        node_families = ", ".join(
            sorted({"/".join(spec.node_types) for spec in wrapper_specs})
        )
        return [
            self.finding_spec.build(
                self.detector_id,
                (
                    f"{module.path} encodes scoped shape builders {function_names} and specs {spec_names} as parallel wrappers over node families {node_families}."
                ),
                evidence,
                scaffold=(
                    "Introduce one `ScopedShapeSpec` ABC with a concrete collect path and polymorphic subclasses such as\n"
                    "`MethodShapeSpec`, `BuilderCallShapeSpec`, and `ExportDictShapeSpec`, then delete the wrapper functions."
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
        return SemanticDataclassRecommendation(
            class_name=class_name,
            base_class_name=base_class_name,
            matched_schema_name=class_name,
            rationale=rationale,
            scaffold=scaffold,
            certification=CERTIFIED,
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
    return SemanticDataclassRecommendation(
        class_name=class_name,
        base_class_name=base_class_name,
        matched_schema_name=closest_schema.class_name if closest_schema else None,
        rationale=rationale,
        scaffold=scaffold,
        certification=STRONG_HEURISTIC,
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
        return SemanticDataclassRecommendation(
            class_name=class_name,
            base_class_name=exact_schema.base_class_name,
            matched_schema_name=class_name,
            rationale=rationale,
            scaffold=scaffold,
            certification=CERTIFIED,
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
    return SemanticDataclassRecommendation(
        class_name=class_name,
        base_class_name=closest_schema.class_name,
        matched_schema_name=closest_schema.class_name,
        rationale=rationale,
        scaffold=scaffold,
        certification=STRONG_HEURISTIC,
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
        normalized_roles = _normalize_witness_field_roles(field_name)
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


def _nominal_authority_shapes(
    modules: Sequence[ParsedModule],
) -> tuple[NominalAuthorityShape, ...]:
    shapes_without_ancestors: list[NominalAuthorityShape] = []
    for module in modules:
        for node in ast.walk(module.module):
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
            if name not in {"ABC", "ABCMeta", "object"}
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


def _is_detectorish_class(node: ast.ClassDef) -> bool:
    if node.name.endswith("Detector"):
        return True
    return bool(_DETECTOR_BASE_NAMES & set(_declared_base_names(node)))


def _finding_build_call(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Call | None:
    for node in ast.walk(method):
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
    for node in ast.walk(module.module):
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
    for node in ast.walk(module.module):
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
    for node in ast.walk(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if statement.name != "structural_observation":
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
            if (
                not isinstance(returned, ast.Call)
                or _ast_terminal_name(returned.func) != "StructuralObservation"
            ):
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
    for node in ast.walk(module.module):
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
    for node in ast.walk(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = tuple(
            name for name in _declared_base_names(node) if name not in {"object"}
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


class ManualFamilyRosterDetector(IssueDetector):
    detector_id = "manual_family_roster"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Manual subclass roster should become auto-registration",
        why=(
            "One helper manually enumerates a class family instead of deriving membership from class existence. "
            "The docs treat that as class-level registration logic that should live in one authoritative hook."
        ),
        capability_gap="zero-delay class-family discovery with declarative ordering",
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
                            f"class Registered{candidate.family_base_name}({candidate.family_base_name}):\n"
                            "    registration_order: ClassVar[int] = 0\n\n"
                            "    def __init_subclass__(cls, **kwargs):\n"
                            "        super().__init_subclass__(**kwargs)\n"
                            "        if not inspect.isabstract(cls):\n"
                            "            FAMILY_REGISTRY.register(cls, priority=cls.registration_order)"
                        ),
                        codemod_patch=(
                            f"# Replace `{candidate.owner_name}` with class-time registration for the `{candidate.family_base_name}` family.\n"
                            f"# Delete the manual {candidate.constructor_style} roster once subclasses self-register."
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
                metrics=RepeatedMethodMetrics(
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
                metrics=RepeatedMethodMetrics(
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
        title="Repeated StructuralObservation builders should share one projection substrate",
        why=(
            "Several carriers repeat the same StructuralObservation projection schema with only role hooks varying. "
            "The docs prefer one authoritative projection template."
        ),
        capability_gap="single authoritative projection builder with role hooks",
        relation_context="same StructuralObservation schema is manually rebuilt across many carriers",
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
        grouped: dict[tuple[str, ...], list[StructuralObservationPropertyCandidate]] = (
            defaultdict(list)
        )
        for candidate in _structural_observation_property_candidates(module):
            grouped[candidate.keyword_names].append(candidate)
        return tuple(
            (keyword_names, tuple(candidates))
            for keyword_names, candidates in grouped.items()
            if len(candidates) >= 3
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        keyword_names, grouped_candidates = cast(
            tuple[
                tuple[str, ...],
                tuple[StructuralObservationPropertyCandidate, ...],
            ],
            candidate,
        )
        evidence = tuple(
            SourceLocation(item.file_path, item.line, item.class_name)
            for item in grouped_candidates[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Classes {', '.join(item.class_name for item in grouped_candidates[:5])} rebuild the same StructuralObservation schema over roles {keyword_names}."
            ),
            evidence,
            scaffold=(
                "class StructuralObservationTemplate(StructuralObservationCarrier, ABC):\n"
                "    observation_kind: ClassVar[ObservationKind]\n"
                "    execution_level: ClassVar[StructuralExecutionLevel]\n\n"
                "    @property\n"
                "    def structural_observation(self) -> StructuralObservation:\n"
                "        return StructuralObservation(...)"
            ),
            codemod_patch=(
                f"# Introduce one projection template for roles {keyword_names}.\n"
                "# Leave only owner_symbol, nominal_witness, observed_name, and fiber_key hooks on the concrete carriers."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(grouped_candidates),
                field_count=len(keyword_names),
                mapping_name="StructuralObservation",
                field_names=keyword_names,
            ),
        )


def default_detectors() -> tuple[IssueDetector, ...]:
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
    "type",
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
    for child in ast.walk(node):
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
    for child in ast.walk(node):
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
        "+class AutoRegisterMeta(ABCMeta):\n"
        f"+    registry = {registry_name}\n"
        "+\n"
        f"+class {base_name}(ABC, metaclass=AutoRegisterMeta):\n"
        "+    registry_key: str\n"
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
        f'class {name}({base_name}, metaclass=AutoRegisterMeta):\n    registry_key = "{name.lower()}"'
        for name in sample
    )
    return (
        f"class AutoRegisterMeta(ABCMeta):\n"
        f"    registry = {registry_name}\n\n"
        f"class {base_name}(ABC):\n"
        f"    registry_key: str\n\n"
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


def _attribute_branch_evidence(
    module: ParsedModule, attr_name: str
) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in ast.walk(module.module):
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
    for node in ast.walk(test):
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
            if node.func.id == "getattr" and len(node.args) >= 2:
                arg = node.args[1]
                if isinstance(arg, ast.Constant) and arg.value == attr_name:
                    return True
    return False


def _iter_functions(module: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(module)
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


def _accessor_replacement_example(candidate: AccessorWrapperCandidate) -> str:
    if candidate.accessor_kind == "setter":
        return f"- replace `{candidate.symbol}(value)` with `{candidate.observed_attribute} = value`"
    if candidate.wrapper_shape == "read_through":
        return f"- replace `{candidate.symbol}()` with `{candidate.observed_attribute}`"
    return f"- replace `{candidate.symbol}()` with an `@property` exposing `{candidate.target_expression}`"


def _indexed_family_wrapper_candidates(
    module: ParsedModule,
) -> tuple[IndexedFamilyWrapperCandidate, ...]:
    candidates: list[IndexedFamilyWrapperCandidate] = []
    for node in ast.walk(module.module):
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
    for node in ast.walk(module):
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
    for node in ast.walk(module):
        if isinstance(node, ast.Compare):
            values = [node.left] + list(node.comparators)
            if any(
                isinstance(value, ast.Attribute) and value.attr == attr_name
                for value in values
            ):
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "getattr" and len(node.args) >= 2:
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
    return any(isinstance(child, ast.Call) for child in ast.walk(node))


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
