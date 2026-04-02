from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .ast_tools import (
    BuilderCallShape,
    ExportDictShape,
    MethodShape,
    ParsedModule,
    RegistrationShape,
    collect_builder_call_shapes,
    collect_export_dict_shapes,
    collect_method_shapes,
    collect_registration_shapes,
)
from .models import FindingSpec, RefactorFinding, SourceLocation


@dataclass(frozen=True)
class DetectorConfig:
    min_duplicate_statements: int = 3
    min_string_cases: int = 3
    min_attribute_probes: int = 2
    min_builder_keywords: int = 3
    min_export_keys: int = 3
    min_registration_sites: int = 2

    @classmethod
    def from_namespace(cls, namespace: Any) -> "DetectorConfig":
        return cls(
            min_duplicate_statements=int(namespace.min_duplicate_statements),
            min_string_cases=int(namespace.min_string_cases),
            min_attribute_probes=int(namespace.min_attribute_probes),
            min_builder_keywords=int(namespace.min_builder_keywords),
            min_export_keys=int(namespace.min_export_keys),
            min_registration_sites=int(namespace.min_registration_sites),
        )


class IssueDetector(ABC):
    detector_id: str

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

    def _collect_rule_locations(
        self,
        module: ParsedModule,
        rules: tuple["NodeRule", ...],
    ) -> list[SourceLocation]:
        locations: list[SourceLocation] = []
        for node in ast.walk(module.module):
            for rule in rules:
                location = rule.maybe_collect(module, node)
                if location is not None:
                    locations.append(location)
        return locations


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
            detector_id=self.detector_id,
            summary=self._summary(module, evidence),
            evidence=self._evidence_slice(evidence),
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


class NodeRule(ABC):
    @abstractmethod
    def maybe_collect(
        self, module: ParsedModule, node: ast.AST
    ) -> SourceLocation | None:
        raise NotImplementedError


@dataclass(frozen=True)
class NamedCallProbeRule(NodeRule):
    function_name: str
    min_args: int
    symbol: str

    def maybe_collect(
        self, module: ParsedModule, node: ast.AST
    ) -> SourceLocation | None:
        if not isinstance(node, ast.Call):
            return None
        if not isinstance(node.func, ast.Name):
            return None
        if node.func.id != self.function_name:
            return None
        if len(node.args) < self.min_args:
            return None
        return SourceLocation(str(module.path), node.lineno, self.symbol)


@dataclass(frozen=True)
class AttributeErrorTryRule(NodeRule):
    symbol: str = "AttributeError"

    def maybe_collect(
        self, module: ParsedModule, node: ast.AST
    ) -> SourceLocation | None:
        if not isinstance(node, ast.Try):
            return None
        for handler in node.handlers:
            if (
                isinstance(handler.type, ast.Name)
                and handler.type.id == "AttributeError"
            ):
                return SourceLocation(str(module.path), handler.lineno, self.symbol)
        return None


class RepeatedPrivateMethodDetector(GroupedShapeIssueDetector):
    detector_id = "repeated_private_methods"

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        return list(_collect_repeated_method_shapes(modules, config))

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
        return RefactorFinding(
            detector_id=self.detector_id,
            pattern_id=5,
            title="Repeated non-orthogonal method skeleton across classes",
            summary=(
                f"{len(methods)} methods across {len(class_names)} classes share the same normalized AST shape."
            ),
            why=(
                "Shared orchestration logic is duplicated across a behavior family. The docs say this shared "
                "non-orthogonal logic should move into an ABC with a concrete template method, leaving only "
                "orthogonal hooks in subclasses."
            ),
            capability_gap="single authoritative algorithm for a nominal behavior family",
            confidence="high",
            relation_context=relation,
            evidence=evidence,
        )


class InheritanceHierarchyCandidateDetector(IssueDetector):
    detector_id = "inheritance_hierarchy_candidate"

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        repeated_groups = _group_repeated_methods(modules, config)
        family_groups: dict[frozenset[str], list[tuple[MethodShape, ...]]] = (
            defaultdict(list)
        )
        for methods in repeated_groups:
            class_names = frozenset(
                method.class_name for method in methods if method.class_name is not None
            )
            if len(class_names) < 2:
                continue
            family_groups[class_names].append(methods)

        findings: list[RefactorFinding] = []
        for class_names, groups in family_groups.items():
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
            evidence: list[SourceLocation] = []
            for methods in groups:
                for method in methods:
                    evidence.append(
                        SourceLocation(method.file_path, method.lineno, method.symbol)
                    )
            findings.append(
                RefactorFinding(
                    detector_id=self.detector_id,
                    pattern_id=5,
                    title="Classes cluster into an ABC hierarchy candidate",
                    summary=(
                        f"Classes {', '.join(sorted(class_names))} share {len(groups)} repeated method-shape groups and repeated method roles that likely want one ABC family."
                    ),
                    why=(
                        "The same set of classes repeats multiple non-orthogonal method skeletons. The docs say this is a "
                        "strong signal that the family should be factored into an ABC with one concrete template method "
                        "layer; orthogonal reusable concerns can then live in mixins so MRO preserves declared precedence."
                    ),
                    capability_gap="single authoritative inheritance hierarchy for a duplicated behavior family",
                    confidence="high",
                    relation_context="same class set repeats several method roles across the same family boundary",
                    evidence=tuple(evidence[:8]),
                )
            )
        return findings


class RepeatedBuilderCallDetector(GroupedShapeIssueDetector):
    detector_id = "repeated_builder_calls"

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        shapes: list[object] = []
        for module in modules:
            for shape in collect_builder_call_shapes(module):
                if len(shape.keyword_names) < config.min_builder_keywords:
                    continue
                shapes.append(shape)
        return shapes

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
        return RefactorFinding(
            detector_id=self.detector_id,
            pattern_id=14,
            title="Repeated field assignment should become an authoritative builder",
            summary=(
                f"Call `{builders[0].callee_name}` repeats the same keyword-mapping shape across {len(builders)} sites."
            ),
            why=(
                "The docs say repeated manual field assignment is an SSOT violation: the mapping should be declared once "
                "in an authoritative constructor, classmethod, or shared builder rather than copied across call sites."
            ),
            capability_gap=(
                "single authoritative data-to-record mapping"
                if same_source
                else "single authoritative record-builder mapping for a repeated constructor family"
            ),
            confidence="medium",
            relation_context=(
                "same builder role repeated across sibling functions or methods"
            ),
            evidence=evidence,
        )


class RepeatedExportDictDetector(GroupedShapeIssueDetector):
    detector_id = "repeated_export_dicts"

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        return [
            shape
            for module in modules
            for shape in collect_export_dict_shapes(module)
            if len(shape.key_names) >= config.min_export_keys
        ]

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
        return RefactorFinding(
            detector_id=self.detector_id,
            pattern_id=14,
            title="Repeated projection dict should become an authoritative schema",
            summary=(
                f"String-key projection dict with keys {export_shapes[0].key_names} repeats across {len(export_shapes)} sites."
            ),
            why=(
                "The docs say repeated JSON/CSV/export dicts and kwargs/source-value bags should become one authoritative "
                "row schema or projection builder instead of many hand-maintained dict literals."
            ),
            capability_gap="single authoritative projection schema for a repeated record or kwargs family",
            confidence="medium",
            relation_context="same string-key projection role repeated across sibling functions or methods",
            evidence=evidence,
        )


class ManualClassRegistrationDetector(GroupedShapeIssueDetector):
    detector_id = "manual_class_registration"

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        return [
            shape for module in modules for shape in collect_registration_shapes(module)
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
        return RefactorFinding(
            detector_id=self.detector_id,
            pattern_id=6,
            title="Manual class registration should become AutoRegisterMeta",
            summary=(
                f"Registry `{registry_name}` is populated manually for {len(class_names)} classes across {len(registrations)} sites."
            ),
            why=(
                "The docs say repeated class-level registration boilerplate is a class-level non-orthogonal algorithm. "
                "It should move into one authoritative metaclass or registry base so abstract-class skipping, uniqueness, "
                "and inheritance behavior are enforced in one place."
            ),
            capability_gap="single authoritative class-registration algorithm with nominal class identity",
            confidence="medium",
            relation_context="same registry key family repeated through manual class-level registration assignments",
            evidence=evidence,
        )


class SentinelAttributeSimulationDetector(PerModuleIssueDetector):
    detector_id = "sentinel_attribute_simulation"

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        sentinel_attrs = _collect_class_sentinel_attrs(module.module)
        findings: list[RefactorFinding] = []
        for attr_name, evidence in sentinel_attrs.items():
            if len(evidence) < 2:
                continue
            if not _module_compares_attribute(module.module, attr_name):
                continue
            findings.append(
                RefactorFinding(
                    detector_id=self.detector_id,
                    pattern_id=1,
                    title="Sentinel attribute is simulating nominal identity",
                    summary=(
                        f"Attribute `{attr_name}` is declared across {len(evidence)} classes and also used for behavioral branching."
                    ),
                    why=(
                        "The docs say sentinel attributes only simulate identity by convention. When they drive behavior across "
                        "multiple classes, the boundary should become a nominal family or another explicit identity handle."
                    ),
                    capability_gap="enumerable and enforceable nominal role identity",
                    confidence="medium",
                    relation_context="same class-level sentinel attribute reused as a fake identity boundary",
                    evidence=tuple(evidence[:6]),
                )
            )
        return findings


class PredicateFactoryChainDetector(PerModuleIssueDetector):
    detector_id = "predicate_factory_chain"

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for function in _iter_functions(module.module):
            branch_count = _predicate_factory_chain_branch_count(function)
            if branch_count is None:
                continue
            findings.append(
                RefactorFinding(
                    detector_id=self.detector_id,
                    pattern_id=2,
                    title="Predicate chain should become a discriminated union family",
                    summary=(
                        f"{function.name} contains a {branch_count}-branch predicate factory chain returning variant constructors."
                    ),
                    why=(
                        "The docs say repeated predicate-driven variant selection should become an explicit subclass family with "
                        "enumeration rather than an open-ended if/elif chain."
                    ),
                    capability_gap="exhaustive nominal variant discovery and extension",
                    confidence="medium",
                    relation_context="same factory role repeated as predicate branches inside one function",
                    evidence=(
                        SourceLocation(
                            str(module.path), function.lineno, function.name
                        ),
                    ),
                )
            )
        return findings


class ConfigAttributeDispatchDetector(StaticModulePatternDetector):
    detector_id = "config_attribute_dispatch"
    finding_spec = FindingSpec(
        pattern_id=4,
        title="Config dispatch is encoded through fragile attribute probing",
        why=(
            "The docs say polymorphic configuration should dispatch on declared config family identity, not on field-name "
            "probing or ad hoc attribute comparisons."
        ),
        capability_gap="fail-loud polymorphic configuration contracts",
        relation_context="same config-family choice expressed through attribute-level probing",
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        evidence: list[SourceLocation] = []
        for function in _iter_functions(module.module):
            if not _function_has_param(function, "config"):
                continue
            evidence.extend(_config_dispatch_evidence(module, function))
        return tuple(evidence)

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        return 2

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} contains {len(evidence)} config-specific attribute probes or comparisons."


class GeneratedTypeLineageDetector(StaticModulePatternDetector):
    detector_id = "generated_type_lineage"
    finding_spec = FindingSpec(
        pattern_id=7,
        title="Generated types need explicit lineage tracking",
        why=(
            "The docs say generated and rebuilt types need explicit nominal lineage so normalization, reverse lookup, and "
            "provenance remain exact."
        ),
        capability_gap="exact generated-type lineage and normalization",
        relation_context="same module combines runtime type generation with lineage-sensitive registries",
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        generation_sites = _generated_type_sites(module)
        lineage_sites = _type_lineage_sites(module)
        if not generation_sites or not lineage_sites:
            return ()
        return tuple((generation_sites + lineage_sites)[:6])

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} generates runtime types and also maintains type-lineage state."


class DualAxisResolutionDetector(PerModuleIssueDetector):
    detector_id = "dual_axis_resolution"

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for function in _iter_functions(module.module):
            evidence = _dual_axis_resolution_evidence(module, function)
            if evidence is None:
                continue
            findings.append(
                RefactorFinding(
                    detector_id=self.detector_id,
                    pattern_id=8,
                    title="Nested precedence walk should be a dual-axis resolution primitive",
                    summary=(
                        f"{function.name} nests scope-like and MRO/type-like iteration in one precedence walk."
                    ),
                    why=(
                        "The docs say scope x type precedence should be modeled explicitly when both context and inheritance order "
                        "contribute to resolution and provenance."
                    ),
                    capability_gap="explicit dual-axis precedence with provenance",
                    confidence="medium",
                    relation_context="same function combines context hierarchy and type/MRO hierarchy",
                    evidence=evidence,
                )
            )
        return findings


class ManualVirtualMembershipDetector(StaticModulePatternDetector):
    detector_id = "manual_virtual_membership"
    finding_spec = FindingSpec(
        pattern_id=9,
        title="Manual class-marker membership should become custom isinstance semantics",
        why=(
            "The docs say explicit runtime interface membership should be class-level and inspectable. Repeated marker checks "
            "suggest a custom isinstance/subclass boundary rather than scattered manual probing."
        ),
        capability_gap="runtime-checkable virtual membership on nominal class identity",
        relation_context="same membership question repeated through class-marker probing",
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        evidence: list[SourceLocation] = []
        for function in _iter_functions(module.module):
            evidence.extend(_manual_class_marker_checks(module, function))
        return tuple(evidence)

    def _minimum_evidence(self, config: DetectorConfig) -> int:
        return 2

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} performs {len(evidence)} class-level marker checks on instances."


class DynamicInterfaceGenerationDetector(StaticModulePatternDetector):
    detector_id = "dynamic_interface_generation"
    finding_spec = FindingSpec(
        pattern_id=10,
        title="Dynamic interface generation is present or required",
        why=(
            "The docs treat dynamically generated empty or near-empty interface types as explicit nominal identity handles "
            "when structure alone cannot express membership."
        ),
        capability_gap="explicit runtime-generated nominal interface identity",
        relation_context="same module generates interface-like nominal types at runtime",
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        return tuple(_dynamic_interface_sites(module)[:6])

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return (
            f"{module.path} contains {len(evidence)} runtime-generated interface sites."
        )


class SentinelTypeMarkerDetector(StaticModulePatternDetector):
    detector_id = "sentinel_type_marker"
    finding_spec = FindingSpec(
        pattern_id=11,
        title="Unique sentinel type marker is present or should be used",
        why=(
            "The docs distinguish sentinel types from sentinel attributes: unique nominal marker objects are appropriate when "
            "exact capability identity matters more than payload."
        ),
        capability_gap="exact capability-marker identity independent of structure",
        relation_context="same module creates or uses unique nominal sentinel markers",
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        return tuple(_sentinel_type_sites(module)[:6])

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} contains {len(evidence)} sentinel-type capability marker sites."


class DynamicMethodInjectionDetector(StaticModulePatternDetector):
    detector_id = "dynamic_method_injection"
    finding_spec = FindingSpec(
        pattern_id=12,
        title="Dynamic method injection belongs in a type-namespace pattern",
        why=(
            "The docs say behavior that must affect all current and future instances belongs in a class namespace pattern, "
            "not in repeated instance-level patching."
        ),
        capability_gap="shared type-namespace mutation for a nominal family",
        relation_context="same module mutates class behavior through runtime namespace injection",
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        return tuple(_dynamic_method_injection_sites(module)[:6])

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} contains {len(evidence)} dynamic type-namespace injection sites."


class AttributeProbeDetector(PerModuleIssueDetector):
    detector_id = "attribute_probes"
    probe_rules = (
        NamedCallProbeRule("hasattr", 2, "hasattr"),
        NamedCallProbeRule("getattr", 3, "getattr"),
        AttributeErrorTryRule(),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        evidence = tuple(self._collect_rule_locations(module, self.probe_rules)[:6])
        total = len(evidence)
        if total < config.min_attribute_probes:
            return []
        return [
            RefactorFinding(
                detector_id=self.detector_id,
                pattern_id=5,
                title="Semantic role recovered from attribute probing",
                summary=f"{module.path} contains {total} attribute-probe sites.",
                why=(
                    "Repeated hasattr/getattr/AttributeError logic means the code is recovering identity from a "
                    "partial structural view. The documented fix is to migrate this region toward an ABC contract "
                    "with direct method calls and fail-loud guarantees."
                ),
                capability_gap="declared semantic role identity and import-time enforcement",
                confidence="medium",
                relation_context="same module-level probing layer across multiple call sites",
                evidence=evidence,
            )
        ]


class InlineLiteralDispatchDetector(PerModuleIssueDetector):
    detector_id = "inline_literal_dispatch"

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for owner_name, block in _iter_statement_blocks(module.module):
            for relation_key, evidence in _group_inline_literal_dispatch(
                module, block
            ).items():
                if len(evidence) < config.min_attribute_probes:
                    continue
                findings.append(
                    RefactorFinding(
                        detector_id=self.detector_id,
                        pattern_id=3,
                        title="Inline literal dispatch should be a registry",
                        summary=(
                            f"{module.path} repeats literal-case dispatch over `{relation_key}` in {len(evidence)} sibling branches."
                        ),
                        why=(
                            "When the same observed value is split across several sibling literal branches, the docs "
                            "say the local rule family should be moved into an authoritative registry, dataclass table, "
                            "or another closed dispatch object instead of repeating inline branch logic."
                        ),
                        capability_gap="single authoritative dispatch representation for a closed local rule family",
                        confidence="medium",
                        relation_context=(
                            f"same branch role repeated inline inside {owner_name or 'module block'}"
                        ),
                        evidence=tuple(evidence[:6]),
                    )
                )
        return findings


class StringDispatchDetector(PerModuleIssueDetector):
    detector_id = "string_dispatch"

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        evidence: list[SourceLocation] = []
        for node in ast.walk(module.module):
            if isinstance(node, ast.If):
                string_tests = _count_string_comparisons(node)
                if string_tests >= config.min_string_cases:
                    evidence.append(
                        SourceLocation(
                            str(module.path), node.lineno, "if-string-dispatch"
                        )
                    )
            if isinstance(node, ast.Dict):
                string_keys = sum(
                    1
                    for key in node.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                )
                if string_keys >= config.min_string_cases:
                    evidence.append(
                        SourceLocation(
                            str(module.path), node.lineno, "dict-string-dispatch"
                        )
                    )

        if not evidence:
            return []
        return [
            RefactorFinding(
                detector_id=self.detector_id,
                pattern_id=3,
                title="Closed-family dispatch expressed through strings",
                summary=(
                    f"{module.path} contains {len(evidence)} string-dispatch sites that look like closed variant logic."
                ),
                why=(
                    "The docs prefer enum- or type-keyed O(1) dispatch for closed families. Repeated string branches "
                    "suggest the code is using a weaker representation than the domain requires."
                ),
                capability_gap="closed-family dispatch with stable nominal keys",
                confidence="medium",
                relation_context="same dispatch role repeated through string comparisons or string-key registries",
                evidence=tuple(evidence[:6]),
            )
        ]


class BidirectionalRegistryDetector(PerModuleIssueDetector):
    detector_id = "bidirectional_registry"

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for node in ast.walk(module.module):
            if not isinstance(node, ast.ClassDef):
                continue
            dict_attrs = _collect_dict_attrs(node)
            mirrored_pairs = _collect_mirrored_assignments(node)
            if len(dict_attrs) < 2 or not mirrored_pairs:
                continue
            evidence = tuple(
                SourceLocation(str(module.path), lineno, f"{node.name}.{label}")
                for lineno, label in mirrored_pairs[:6]
            )
            findings.append(
                RefactorFinding(
                    detector_id=self.detector_id,
                    pattern_id=13,
                    title="Bidirectional registry maintained manually",
                    summary=(
                        f"Class {node.name} appears to maintain mirrored forward/reverse registry assignments."
                    ),
                    why=(
                        "The docs prescribe a single authoritative bidirectional type registry when exact companion "
                        "normalization and reverse lookup matter. Manual mirrored assignments are drift-prone and "
                        "should be centralized."
                    ),
                    capability_gap="exact bijection and O(1) reverse lookup on nominal keys",
                    confidence="medium",
                    relation_context="same class maintains forward and reverse registry state",
                    evidence=evidence,
                )
            )
        return findings


def default_detectors() -> tuple[IssueDetector, ...]:
    return (
        SentinelAttributeSimulationDetector(),
        PredicateFactoryChainDetector(),
        ConfigAttributeDispatchDetector(),
        GeneratedTypeLineageDetector(),
        DualAxisResolutionDetector(),
        ManualVirtualMembershipDetector(),
        DynamicInterfaceGenerationDetector(),
        SentinelTypeMarkerDetector(),
        DynamicMethodInjectionDetector(),
        RepeatedPrivateMethodDetector(),
        InheritanceHierarchyCandidateDetector(),
        RepeatedBuilderCallDetector(),
        RepeatedExportDictDetector(),
        ManualClassRegistrationDetector(),
        AttributeProbeDetector(),
        InlineLiteralDispatchDetector(),
        StringDispatchDetector(),
        BidirectionalRegistryDetector(),
    )


def _count_string_comparisons(node: ast.If) -> int:
    count = 0
    current: ast.stmt | None = node
    while isinstance(current, ast.If):
        count += _string_comparisons_in_test(current.test)
        current = current.orelse[0] if len(current.orelse) == 1 else None
    return count


def _string_comparisons_in_test(node: ast.AST) -> int:
    if isinstance(node, ast.Compare):
        if not all(isinstance(op, ast.Eq) for op in node.ops):
            return 0
        values = [node.left] + list(node.comparators)
        if any(
            isinstance(value, ast.Constant) and isinstance(value.value, str)
            for value in values
        ):
            return 1
    if isinstance(node, ast.BoolOp):
        return sum(_string_comparisons_in_test(value) for value in node.values)
    return 0


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


def _iter_statement_blocks(
    module: ast.Module,
) -> list[tuple[str | None, list[ast.stmt]]]:
    blocks: list[tuple[str | None, list[ast.stmt]]] = [(None, module.body)]

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            blocks.append((node.name, node.body))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            blocks.append((node.name, node.body))
            self.generic_visit(node)

    Visitor().visit(module)
    return blocks


def _group_inline_literal_dispatch(
    module: ParsedModule,
    block: list[ast.stmt],
) -> dict[str, list[SourceLocation]]:
    groups: dict[str, list[SourceLocation]] = defaultdict(list)
    for stmt in block:
        if not isinstance(stmt, ast.If):
            continue
        key = _literal_dispatch_key(stmt.test)
        if key is None:
            continue
        groups[key].append(
            SourceLocation(str(module.path), stmt.lineno, "inline-literal-dispatch")
        )
    return {key: items for key, items in groups.items() if len(items) >= 2}


def _literal_dispatch_key(test: ast.AST) -> str | None:
    if not isinstance(test, ast.Compare):
        return None
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    if len(test.comparators) != 1:
        return None
    comparator = test.comparators[0]
    if not (isinstance(comparator, ast.Constant) and isinstance(comparator.value, str)):
        return None
    return ast.dump(test.left, include_attributes=False)


def _collect_repeated_method_shapes(
    modules: list[ParsedModule], config: DetectorConfig
) -> tuple[MethodShape, ...]:
    return tuple(
        method
        for module in modules
        for method in collect_method_shapes(module)
        if method.class_name
        and method.statement_count >= config.min_duplicate_statements
    )


def _group_repeated_methods(
    modules: list[ParsedModule], config: DetectorConfig
) -> list[tuple[MethodShape, ...]]:
    groups: dict[tuple[bool, int, str], list[MethodShape]] = defaultdict(list)
    for method in _collect_repeated_method_shapes(modules, config):
        key = (method.is_private, method.param_count, method.fingerprint)
        groups[key].append(method)
    return [
        tuple(sorted(methods, key=lambda item: (item.file_path, item.lineno)))
        for methods in groups.values()
        if len(methods) >= 2 and len({method.class_name for method in methods}) >= 2
    ]


def _iter_functions(module: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


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


def _config_dispatch_evidence(
    module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if (
                node.func.id == "hasattr"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "config"
            ):
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, function.name)
                )
            if (
                node.func.id == "getattr"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "config"
            ):
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, function.name)
                )
        if isinstance(node, ast.Compare):
            if (
                isinstance(node.left, ast.Attribute)
                and isinstance(node.left.value, ast.Name)
                and node.left.value.id == "config"
            ):
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, function.name)
                )
    return evidence


def _generated_type_sites(module: ParsedModule) -> list[SourceLocation]:
    sites: list[SourceLocation] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if call_name in {"type", "make_dataclass", "new_class"}:
            sites.append(
                SourceLocation(str(module.path), node.lineno, call_name or "type")
            )
    return sites


def _type_lineage_sites(module: ParsedModule) -> list[SourceLocation]:
    sites: list[SourceLocation] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            name = _call_name(target.value)
            if name and any(
                token in name.lower()
                for token in ("lazy", "base", "type", "mapping", "registry")
            ):
                sites.append(SourceLocation(str(module.path), node.lineno, name))
    return sites


def _dual_axis_resolution_evidence(
    module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[SourceLocation, ...] | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.For):
            continue
        inner_loops = [child for child in node.body if isinstance(child, ast.For)]
        if not inner_loops:
            continue
        outer_name = _loop_target_name(node.target)
        inner_name = _loop_target_name(inner_loops[0].target)
        text = ast.dump(inner_loops[0].iter, include_attributes=False)
        if "__mro__" in text or "mro" in text.lower() or "type" in text.lower():
            if outer_name and any(
                token in outer_name.lower() for token in ("scope", "context", "level")
            ):
                return (
                    SourceLocation(str(module.path), node.lineno, function.name),
                    SourceLocation(
                        str(module.path), inner_loops[0].lineno, function.name
                    ),
                )
    return None


def _loop_target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _manual_class_marker_checks(
    module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hasattr"
        ):
            target = node.args[0] if node.args else None
            if isinstance(target, ast.Attribute) and target.attr == "__class__":
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, function.name)
                )
            elif isinstance(target, ast.Call) and _call_name(target.func) == "type":
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, function.name)
                )
        if isinstance(node, ast.Attribute) and node.attr.startswith("_is_"):
            evidence.append(
                SourceLocation(str(module.path), node.lineno, function.name)
            )
    return evidence


def _dynamic_interface_sites(module: ParsedModule) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) != "type":
            continue
        if len(node.args) < 3:
            continue
        bases = node.args[1]
        if isinstance(bases, ast.Tuple) and any(
            isinstance(elt, ast.Name) and elt.id == "ABC" for elt in bases.elts
        ):
            namespace = node.args[2]
            if isinstance(namespace, ast.Dict) and len(namespace.keys) == 0:
                evidence.append(SourceLocation(str(module.path), node.lineno, "type"))
    return evidence


def _sentinel_type_sites(module: ParsedModule) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    sentinel_names: set[str] = set()
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Call):
            continue
        if _call_name(node.value.func.func) != "type":
            continue
        sentinel_names.add(target.id)
        evidence.append(SourceLocation(str(module.path), node.lineno, target.id))
    for node in ast.walk(module.module):
        if isinstance(node, ast.Compare):
            names = {
                subnode.id
                for subnode in ast.walk(node)
                if isinstance(subnode, ast.Name)
            }
            if names & sentinel_names:
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, "sentinel-compare")
                )
        if isinstance(node, ast.Subscript):
            names = {
                subnode.id
                for subnode in ast.walk(node)
                if isinstance(subnode, ast.Name)
            }
            if names & sentinel_names:
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, "sentinel-subscript")
                )
    return evidence


def _dynamic_method_injection_sites(module: ParsedModule) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "setattr":
            continue
        if len(node.args) < 3:
            continue
        target = node.args[0]
        if isinstance(target, ast.Name) and target.id.endswith("type"):
            evidence.append(SourceLocation(str(module.path), node.lineno, "setattr"))
    return evidence
