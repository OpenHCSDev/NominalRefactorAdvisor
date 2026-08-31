"""Structural detector implementations.

This module groups detector families and helper logic centered on repeated
field families, wrapper surfaces, exports, and structural record mechanics.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from metaclass_registry import AutoRegisterMeta

from ..semantic_algebra import ObjectFamilyShape
from ..semantic_description_length import CompressionCertificate
from ..class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from ..ast_tools import SourceModule
from ..native_syntax import NativePythonSyntaxIndex
from ..registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from ..source_identity import source_path_text
from ..semantic_match import (
    Maybe,
    attribute_call_match,
)
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import *
from ._helpers import *
from ._helpers import _property_alias_hook_groups
from ._structural_step_regex_extractor import *

_REFLECTIVE_ATTRIBUTE_CONTRACT_REPLACEMENT_SHAPE = ObjectFamilyShape(
    shared_objects=("nominal_attribute_contract",)
)
def _reflective_self_attribute_compression_certificate(
    candidate: ReflectiveSelfAttributeCandidate,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=4,
        replacement_shape=_REFLECTIVE_ATTRIBUTE_CONTRACT_REPLACEMENT_SHAPE,
        semantic_axes=(candidate.attribute_name, candidate.reflective_builtin),
    )


def _semantic_overlap_abc_scaffold(
    candidate: SemanticOverlapABCOptimizationCandidate,
) -> str:
    base_name = f"{candidate.base_name}{_camel_case(candidate.method_name)}Template"
    classvar_block = "\n".join(
        (f"    {name}: ClassVar[object]" for name in candidate.classvar_names)
    )
    property_block = "\n".join(
        (
            f"    @property\n    @abstractmethod\n    def {name}(self): ..."
            for name in candidate.property_hook_names
        )
    )
    behavior_block = "\n".join(
        (
            f"    @abstractmethod\n    def {name}(self, *args, **kwargs): ..."
            for name in candidate.behavior_hook_names
        )
    )
    residue_block = "\n\n".join(
        block for block in (classvar_block, property_block, behavior_block) if block
    )
    if residue_block:
        residue_block = f"\n{residue_block}\n"
    return (
        f"class {base_name}({candidate.base_name}, ABC):\n"
        f"    def {candidate.method_name}(self, *args, **kwargs):\n"
        "        # Move the shared statement skeleton here.\n"
        "        # Route only irreducible differences through the declarations/hooks below.\n"
        "        ...\n"
        f"{residue_block}"
    )


class _SemanticOverlapPatchRenderer(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    @abstractmethod
    def __call__(
        self,
        candidate: (
            SemanticOverlapABCOptimizationCandidate
            | SemanticOverlapABCFamilyOptimizationCandidate
            | SemanticOverlapABCResidueAxisCatalogCandidate
        ),
    ) -> str:
        raise NotImplementedError


class _SemanticOverlapABCPatchRenderer(_SemanticOverlapPatchRenderer):
    def __call__(self, candidate: SemanticOverlapABCOptimizationCandidate) -> str:
        residue = (
            *candidate.classvar_names,
            *candidate.property_hook_names,
            *candidate.behavior_hook_names,
        )
        residue_summary = ", ".join(residue) if residue else "no hooks"
        family_summary = ", ".join(candidate.family_method_names)
        mixin_summary = (
            ", ".join(candidate.mixin_axis_specs)
            if candidate.mixin_axis_specs
            else "no subset mixins"
        )
        overlap_summary = (
            ", ".join(candidate.overlap_axis_specs)
            if candidate.overlap_axis_specs
            else "no partial overlaps"
        )
        return (
            f"# Extract `{candidate.method_name}` from {candidate.class_names} into an intermediate ABC over `{candidate.base_name}`.\n"
            f"# Hierarchy normal form: {candidate.hierarchy_normal_form}.\n"
            f"# Candidate hierarchy layer owns methods: {family_summary}; concrete ABC methods: {candidate.abc_concrete_method_names}; subset mixin axes: {mixin_summary}.\n"
            f"# Partial-overlap axes needing explicit precedence/layering: {overlap_summary}.\n"
            f"# Keep only residue declarations/hooks on leaves: {residue_summary}; leaf residue basis: {candidate.leaf_residue_names}."
        )


class _SemanticOverlapABCFamilyPatchRenderer(_SemanticOverlapPatchRenderer):
    def __call__(self, candidate: SemanticOverlapABCFamilyOptimizationCandidate) -> str:
        return (
            f"# Extract methods {candidate.method_names} from {candidate.class_names} into one ABC family over `{candidate.base_name}`.\n"
            f"# Hierarchy normal form: {candidate.hierarchy_normal_form}.\n"
            f"# Move concrete template methods {candidate.abc_concrete_method_names} to the ABC.\n"
            f"# Keep classvars {candidate.classvar_names}, properties {candidate.property_hook_names}, and behavior hooks {candidate.behavior_hook_names} as leaf residue.\n"
            f"# The family removes {candidate.shared_statement_count} shared statement objects with {candidate.residue_count} residue declarations."
        )


def _global_inheritance_optimization_patch(
    candidate: GlobalInheritanceOptimizationCandidate,
) -> str:
    mixins = (
        ", ".join(candidate.mixin_axis_specs)
        if candidate.mixin_axis_specs
        else "no clean subset mixins"
    )
    overlaps = (
        ", ".join(candidate.overlap_axis_specs)
        if candidate.overlap_axis_specs
        else "no partial-overlap layers"
    )
    return (
        f"# Treat `{candidate.base_name}` as one inheritance lattice across families {candidate.family_specs}.\n"
        f"# Move shared method skeletons {candidate.method_names} into the highest valid ABC/layer in the lattice.\n"
        f"# Use subset mixins for {mixins}; introduce explicit precedence layers for {overlaps}.\n"
        f"# Leaves keep only residue declarations/hooks {candidate.leaf_residue_names}."
    )


class _SemanticOverlapABCResidueAxisPatchRenderer(_SemanticOverlapPatchRenderer):
    def __call__(self, candidate: SemanticOverlapABCResidueAxisCatalogCandidate) -> str:
        return (
            f"# Replace per-method residue declarations for {candidate.method_names} over `{candidate.base_name}` "
            f"with one residue-axis catalog keyed by {candidate.residue_kind_names}.\n"
            "# Derive hook/classvar names from the residue axis rows instead of declaring each method's residue surface independently."
        )


_semantic_overlap_abc_patch = _SemanticOverlapABCPatchRenderer()
_semantic_overlap_abc_family_patch = _SemanticOverlapABCFamilyPatchRenderer()
_semantic_overlap_abc_residue_axis_patch = _SemanticOverlapABCResidueAxisPatchRenderer()


def _semantic_overlap_abc_family_scaffold(
    candidate: SemanticOverlapABCFamilyOptimizationCandidate,
) -> str:
    base_name = f"{candidate.base_name}TemplateFamily"
    method_block = "\n\n".join(
        (
            f"    def {method_name}(self, *args, **kwargs):\n"
            "        # Move the shared method skeleton here.\n"
            "        ..."
            for method_name in candidate.method_names
        )
    )
    return f"class {base_name}({candidate.base_name}, ABC):\n{method_block}"


def _semantic_overlap_abc_residue_axis_scaffold(
    candidate: SemanticOverlapABCResidueAxisCatalogCandidate,
) -> str:
    rows = "\n".join(
        (f"    ResidueAxisRow(kind={kind!r})," for kind in candidate.residue_kind_names)
    )
    return f"ResidueAxisCatalog(\n{rows}\n)"


from ._substrate_support import *


def _witness_mixin_enforcement_candidate(
    module: ParsedModule,
) -> WitnessMixinEnforcementCandidate | None:
    all_classes = witness_carrier_class_candidates(module)
    grouped: dict[str, list[WitnessCarrierClassCandidate]] = defaultdict(list)
    for candidate in all_classes:
        for token in candidate.family_tokens:
            grouped[token].append(candidate)
    classes = max(
        (
            sorted_tuple(items, key=lambda item: (item.line, item.class_name))
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
        (
            (role_name, sorted_tuple(role_to_fields[role_name]))
            for role_name in _WITNESS_MIXIN_ROLE_NAMES
            if len(role_to_classes[role_name]) >= 2
            and len(role_to_fields[role_name]) >= 2
        )
    )
    if not role_field_names:
        return None
    class_names = sorted_tuple(
        {
            class_name
            for role_name, _ in role_field_names
            for class_name in role_to_classes[role_name]
        }
    )
    return WitnessMixinEnforcementCandidate(
        file_path=module.file_path,
        class_names=class_names,
        line_numbers=tuple((line_by_class[class_name] for class_name in class_names)),
        role_field_names=role_field_names,
    )


class MixinEnforcementDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_WITNESS_CARRIER,
        "Renamed orthogonal semantic slices should become mixins",
        "Several carrier classes repeat the same semantic slice under renamed fields such as `line` vs `method_line` or `name_family` vs `class_names`. One shared base is not enough when those slices are orthogonal; the architecture wants reusable mixins composed through multiple inheritance.",
        "one authoritative semantic carrier spine plus reusable semantic-role mixins",
        "same carrier family repeats renamed semantic slices that overlap orthogonally across sibling carriers",
        (
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
            (
                SourceLocation(candidate.file_path, line, class_name)
                for class_name, line in zip(
                    candidate.class_names, candidate.line_numbers, strict=True
                )
            )
        )
        role_summary = "; ".join(
            (
                f"{role_name} via {field_names}"
                for role_name, field_names in candidate.role_field_names
            )
        )
        return [
            self.build_finding(
                f"Carrier classes {', '.join(candidate.class_names)} repeat renamed semantic slices {role_summary}; enforce reusable mixins and compose them through multiple inheritance.",
                evidence,
                FindingBuildContext(
                    scaffold=_witness_mixin_enforcement_scaffold(candidate),
                    codemod_patch=_witness_mixin_enforcement_patch(candidate),
                    metrics=WitnessCarrierMetrics(
                        class_count=len(candidate.class_names),
                        shared_role_count=len(candidate.role_field_names),
                        class_names=candidate.class_names,
                        shared_role_names=tuple(
                            (role_name for role_name, _ in candidate.role_field_names)
                        ),
                    ),
                ),
            )
        ]


class RepeatedPropertyAliasHookDetector(
    ModuleCollectorCandidateDetector[PropertyAliasHookGroup]
):
    detector_id = "repeated_property_alias_hooks"
    candidate_collector = _property_alias_hook_groups
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated property hook aliases should move into a shared base or mixin",
        "Several subclasses re-declare the same one-line property hook over the same backing attribute. That is non-orthogonal hook duplication and should live once in a shared base or mixin.",
        "single authoritative hook property implementation for a nominal subclass family",
        "same property hook alias repeats across siblings of one base family",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _finding_for_candidate(
        self, hook_group: PropertyAliasHookGroup
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(
                    hook_group.file_path,
                    line,
                    f"{class_name}.{hook_group.property_name}",
                )
                for class_name, line in zip(
                    hook_group.class_names, hook_group.line_numbers, strict=True
                )
            )
        )
        mixin_name = f"{_camel_case(hook_group.returned_attribute)}{_camel_case(hook_group.property_name)}Mixin"
        return self.build_finding(
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
            metrics=hook_group.repeated_method_metrics,
        )


ABCOptimizerCandidateT = TypeVar("ABCOptimizerCandidateT")


class _CompactABCOptimizerDetectorBase(
    CompactContextCandidateDetector[
        CompactModuleClassProjection,
        CompactABCOptimizerContext,
        ABCOptimizerCandidateT,
    ],
    Generic[ABCOptimizerCandidateT],
):
    module_projection_family = CompactModuleClassProjectionFamily
    compact_shared_context_builder = staticmethod(
        CompactClassRepositoryContext.from_projections
    )

    @classmethod
    def _compact_context_from_projections(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> CompactABCOptimizerContext:
        del config
        return CompactABCOptimizerContext.from_projections(projections)

    @classmethod
    def _compact_context_from_shared(
        cls,
        context: object | None,
    ) -> CompactABCOptimizerContext:
        if isinstance(context, CompactABCOptimizerContext):
            return context
        repository = CompactClassRepositoryContext.require(context)
        return repository.cached(
            CompactABCOptimizerContext,
            lambda: CompactABCOptimizerContext.from_projections(
                repository.projections,
                class_index=repository.class_index,
            ),
        )


class _CompactSemanticOverlapABCOptimizationDetectorBase(
    _CompactABCOptimizerDetectorBase[SemanticOverlapABCOptimizationCandidate]
):
    def _candidates_from_compact_context(
        self,
        context: CompactABCOptimizerContext,
        config: DetectorConfig,
    ) -> Sequence[SemanticOverlapABCOptimizationCandidate]:
        del config
        return context.method_candidates


class _CompactSemanticOverlapABCFamilyOptimizationDetectorBase(
    _CompactABCOptimizerDetectorBase[SemanticOverlapABCFamilyOptimizationCandidate]
):
    def _candidates_from_compact_context(
        self,
        context: CompactABCOptimizerContext,
        config: DetectorConfig,
    ) -> Sequence[SemanticOverlapABCFamilyOptimizationCandidate]:
        del config
        return context.family_candidates


class _CompactGlobalInheritanceOptimizationDetectorBase(
    _CompactABCOptimizerDetectorBase[GlobalInheritanceOptimizationCandidate]
):
    def _candidates_from_compact_context(
        self,
        context: CompactABCOptimizerContext,
        config: DetectorConfig,
    ) -> Sequence[GlobalInheritanceOptimizationCandidate]:
        del config
        return context.global_candidates


class _CompactSemanticOverlapABCResidueAxisCatalogDetectorBase(
    _CompactABCOptimizerDetectorBase[SemanticOverlapABCResidueAxisCatalogCandidate]
):
    def _candidates_from_compact_context(
        self,
        context: CompactABCOptimizerContext,
        config: DetectorConfig,
    ) -> Sequence[SemanticOverlapABCResidueAxisCatalogCandidate]:
        del config
        return context.residue_axis_candidates


declare_candidate_rule_detector(
    SemanticOverlapABCOptimizationCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Sibling implementations should anti-unify into an ABC template",
        "Sibling classes that share a base and implement the same method with the same statement skeleton are paying for one algorithm multiple times. When the differences are a small set of expression coordinates, the base should own the concrete algorithm and leaves should expose only classvars, properties, or abstract hooks for the irreducible residue.",
        "one intermediate ABC owns the shared method skeleton and leaves keep only minimal hooks/declarations",
        "same method across sibling classes has an anti-unifiable statement skeleton with small residue",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.method_name}` in siblings {candidate.class_names} over `{candidate.base_name}` shares "
        f"{candidate.shared_statement_count} statements with {candidate.varying_coordinate_count} residue coordinate(s): "
        f"classvars {candidate.classvar_names}, properties {candidate.property_hook_names}, hooks {candidate.behavior_hook_names}. "
        f"Move concrete methods {candidate.abc_concrete_method_names} to the ABC and leave leaf residue basis "
        f"{candidate.leaf_residue_names} on leaves ({candidate.subclass_residue_count} residue declaration(s), "
        f"shared/residue ratio {candidate.shared_to_residue_ratio:.2f}). "
        f"The derived hierarchy plan scores {candidate.optimizer_score} with {candidate.abc_layer_count} ABC layer(s), "
        f"{candidate.lattice_node_count} lattice node(s), {candidate.lattice_edge_count} lattice edge(s), "
        f"family methods {candidate.family_method_names}, mixin axes {candidate.mixin_axis_specs}, "
        f"overlap axes {candidate.overlap_axis_specs}, and normal form `{candidate.hierarchy_normal_form}`."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    scaffold=_semantic_overlap_abc_scaffold,
    codemod_patch=_semantic_overlap_abc_patch,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.class_names),
        statement_count=candidate.shared_statement_count,
        class_count=len(candidate.class_names),
        method_symbols=tuple(
            (
                f"{class_name}.{candidate.method_name}"
                for class_name in candidate.class_names
            )
        ),
    ),
    detector_priority=-10,
    detector_name="SemanticOverlapAbcOptimizationDetector",
    detector_base=_CompactSemanticOverlapABCOptimizationDetectorBase,
)


declare_candidate_rule_detector(
    SemanticOverlapABCFamilyOptimizationCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Class-family algorithms should collapse as one ABC hierarchy",
        "A class family has several methods with compatible anti-unifiable bodies over the same base and subclass set. Treating each method independently misses the larger normal form: the base hierarchy should own the full algorithm family while leaves expose only the combined residue.",
        "one ABC family owns all shared method skeletons and leaf classes keep only residue declarations",
        "multiple semantic-overlap ABC method candidates share the same base and subclass family",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.base_name}` subclasses {candidate.class_names} repeat family methods {candidate.method_names} "
        f"with {candidate.shared_statement_count} shared statements, {candidate.residue_count} residue declaration(s), "
        f"concrete ABC methods {candidate.abc_concrete_method_names}, leaf residue basis {candidate.leaf_residue_names}, "
        f"shared/residue ratio {candidate.shared_to_residue_ratio:.2f}, "
        f"{candidate.abc_layer_count} ABC layer(s), {candidate.lattice_node_count} lattice node(s), "
        f"{candidate.lattice_edge_count} lattice edge(s), and normal form `{candidate.hierarchy_normal_form}`."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    scaffold=_semantic_overlap_abc_family_scaffold,
    codemod_patch=_semantic_overlap_abc_family_patch,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.method_symbols),
        statement_count=candidate.shared_statement_count,
        class_count=len(candidate.class_names),
        method_symbols=candidate.method_symbols,
    ),
    detector_priority=-11,
    detector_name="SemanticOverlapAbcFamilyOptimizationDetector",
    detector_base=_CompactSemanticOverlapABCFamilyOptimizationDetectorBase,
)


declare_candidate_rule_detector(
    GlobalInheritanceOptimizationCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Inheritance root should optimize the whole overlap lattice",
        "A base class has several overlapping subclass method families. Optimizing each repeated override independently can trap the hierarchy in a local minimum; the base should solve the full class-set lattice and place shared algorithms, subset mixins, and partial-overlap layers globally.",
        "one inheritance-lattice cover assigns shared methods to ABCs or mixins while leaves keep only residue declarations",
        "multiple semantic-overlap ABC families under one root have intersecting but non-identical subclass sets",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.base_name}` has a global inheritance lattice over classes {candidate.class_names}: "
        f"families {candidate.family_specs}, methods {candidate.method_names}, "
        f"{candidate.lattice_node_count} lattice node(s), {candidate.lattice_edge_count} edge(s), "
        f"subset mixins {candidate.mixin_axis_specs}, partial overlaps {candidate.overlap_axis_specs}, "
        f"{candidate.shared_statement_count} shared statements, {candidate.residue_count} residue declarations, "
        f"leaf residue basis {candidate.leaf_residue_names}, optimizer score {candidate.optimizer_score}."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    scaffold=lambda candidate: (
        f"class {candidate.base_name}GlobalTemplate({candidate.base_name}, ABC):\n"
        "    # One lattice owner derives concrete ABC methods, subset mixins, and overlap layers.\n"
        "    ..."
    ),
    codemod_patch=_global_inheritance_optimization_patch,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.method_symbols),
        statement_count=candidate.shared_statement_count,
        class_count=len(candidate.class_names),
        method_symbols=candidate.method_symbols,
    ),
    detector_priority=-12,
    detector_name="GlobalInheritanceOptimizationDetector",
    detector_base=_CompactGlobalInheritanceOptimizationDetectorBase,
)


declare_candidate_rule_detector(
    SemanticOverlapABCResidueAxisCatalogCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "ABC residue axes should derive from one catalog",
        "A semantic-overlap ABC family has several methods whose varying coordinates share the same residue kind signature. Naming classvars and hooks independently per method keeps a second manual axis beside the template hierarchy.",
        "one residue-axis catalog derives classvar and hook declarations for the ABC family",
        "multiple ABC family methods share the same residue coordinate kinds",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.base_name}` family methods {candidate.method_names} share residue kinds "
        f"{candidate.residue_kind_names} across {candidate.residue_site_count} residue site(s); "
        "derive the hook/classvar surface from one residue-axis catalog."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    scaffold=_semantic_overlap_abc_residue_axis_scaffold,
    codemod_patch=_semantic_overlap_abc_residue_axis_patch,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.residue_site_count,
        mapping_name=candidate.base_name,
        field_names=candidate.residue_kind_names,
    ),
    detector_priority=-13,
    detector_name="SemanticOverlapAbcResidueAxisCatalogDetector",
    detector_base=_CompactSemanticOverlapABCResidueAxisCatalogDetectorBase,
)


declare_candidate_rule_detector(
    ConstantPropertyDefaultBundleCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Constant property defaults should derive from descriptors",
        "A class that repeats many one-line properties returning literal defaults is using method syntax for data. The default surface should be declared as typed descriptors or a property-default table while real override behavior stays in subclasses.",
        "typed constant-property descriptor defaults on the nominal base",
        "same class repeats constant-return property methods for default hook values",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` repeats {len(candidate.property_names)} constant property defaults over {candidate.return_expressions}."
    ),
    scaffold=lambda candidate: (
        "from descriptor_algebra import ConstantProperty\n\nclass Base:\n    property_name = ConstantProperty(default_value)"
    ),
    codemod_patch=lambda candidate: (
        f"# Replace constant-return property methods on `{candidate.class_name}` with `ConstantProperty[...]` descriptors.\n# Keep method syntax only for defaults that allocate or compute."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.property_names),
        mapping_name=candidate.class_name,
        field_names=candidate.property_names,
    ),
    detector_priority=-4,
    detector_name="ConstantPropertyDefaultBundleDetector",
    candidate_collector=_constant_property_default_bundle_candidates,
)


class ReflectiveSelfAttributeEscapeDetector(
    ModuleCollectorCandidateDetector[ReflectiveSelfAttributeCandidate]
):
    candidate_collector = _reflective_self_attribute_candidates
    finding_spec = high_confidence_spec(
        PatternId.CONFIG_CONTRACTS,
        "Reflective self-attribute access hides a nominal contract",
        "A class uses reflective self-attribute access with a hardcoded string instead of declaring the field or property on the nominal carrier. That keeps the contract partial, stringly, and fail-soft.",
        "declared fail-loud nominal attribute contract on the carrier family",
        "class template probes its own required state through reflective string access",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.PARTIAL_VIEW,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _finding_for_candidate(
        self, reflective_candidate: ReflectiveSelfAttributeCandidate
    ) -> RefactorFinding:
        carrier_name = f"{reflective_candidate.class_name}Carrier"
        return self.build_finding(
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
            compression_certificate=_reflective_self_attribute_compression_certificate(
                reflective_candidate
            ),
        )


declare_candidate_rule_detector(
    RepeatedBaseBundleCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated MRO base bundle should become a named ABC mixin",
        "Several classes repeat the same contiguous base bundle. That bundle is already a semantic composition unit, so it should have a nominal name and be reused as one ABC/mixin rather than respelled across implementation classes.",
        "named ABC/mixin for one repeated semantic MRO bundle",
        "class family repeats the same composable base sequence in each class declaration",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"Classes {candidate.class_names} repeat MRO bundle {candidate.base_names} across {candidate.class_count} declarations."
    ),
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: (
        f"class SharedSemanticMixin({', '.join(candidate.base_names)}, ABC):\n    pass"
    ),
    codemod_patch=lambda candidate: (
        "# Extract the repeated contiguous base bundle into one named ABC/mixin.\n# Replace the repeated base sequence in each class with that nominal bundle and keep only class-specific orthogonal bases explicit."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.class_count,
        mapping_name="mro-base-bundle",
        field_names=candidate.base_names,
    ),
    detector_priority=-5,
    detector_name="RepeatedBaseBundleDetector",
    candidate_collector=_repeated_base_bundle_candidates,
)


class TypeIndexedDefinitionBoilerplateDetector(
    ModuleCollectorCandidateDetector[TypeIndexedDefinitionBoilerplateGroup]
):
    candidate_collector = _type_indexed_definition_boilerplate_groups
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Type-indexed family definitions should derive from one typed declaration table",
        "Several `*Definition` classes plus `family_type` aliases restate the same type-indexed family metadata. That metadata should live once in a typed declaration table and definition-time materializer.",
        "one authoritative typed declaration table for family generation and export derivation",
        "same type-indexed family definition and alias boilerplate repeats across sibling declarations",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _finding_for_candidate(
        self, group: TypeIndexedDefinitionBoilerplateGroup
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(group.file_path, line, class_name)
                for class_name, line in zip(
                    group.definition_class_names, group.line_numbers, strict=True
                )
            )
        )
        return self.build_finding(
            (
                f"Definition classes {', '.join(group.definition_class_names[:6])} plus aliases {', '.join(group.alias_names[:6])} all repeat typed family metadata {group.assigned_names} under bases {group.base_names}."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass FamilyDeclaration(Generic[TItem]):\n    export_name: str\n    item_type: type[TItem]\n    spec_root: type[object] | None = None\n    spec: object | None = None\n    literal_kind: object | None = None\n\ndef materialize_family(decl: FamilyDeclaration[object]) -> type[CollectedFamily]:\n    return type(...)"
            ),
            codemod_patch=(
                f"# Replace repeated definition classes under {group.base_names} with one typed declaration table.\n"
                "# Derive runtime family classes, registry indexes, exported aliases, and `__all__` from the same declarations instead of restating them in classes plus assignments."
            ),
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(group.definition_class_names),
                registry_name=group.base_names[0],
                class_names=group.definition_class_names,
                class_key_pairs=group.assigned_names,
            ),
        )


def _native_export_policy_predicate_candidates(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[ExportPolicyPredicateCandidate] | None:
    """Project one export-policy predicate from native-selected declarations."""

    if not syntax_index.is_complete:
        return None
    try:
        export_assignments = tuple(
            syntax_index.statement_for(node)
            for node in syntax_index.top_level_assignment_statements()
            if b"__all__" in syntax_index.source_for(node)
        )
        predicate_names = frozenset(
            predicate_name
            for statement in export_assignments
            if (predicate_name := _export_all_predicate_name(statement)) is not None
        )
        if len(predicate_names) != 1:
            return []
        predicate_name = next(iter(predicate_names))
        functions = tuple(
            syntax_index.function_for(node)
            for node in syntax_index.top_level_declarations("function")
            if syntax_index.declared_name(node) == predicate_name
        )
        parsed_module = source_module.parsed_module(
            ast.Module(
                body=[*export_assignments, *functions],
                type_ignores=[],
            ),
        )
        candidate = _module_export_policy_predicate_candidate(parsed_module)
        return [] if candidate is None else [candidate]
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


class ExportPolicyPredicateCandidateFamily(
    CollectedFamily[ExportPolicyPredicateCandidate]
):
    """Persist one compact derived-export policy projection per module."""

    item_type = ExportPolicyPredicateCandidate
    report_presence_predicate = staticmethod(lambda items, config: bool(items))
    source_collector = staticmethod(_native_export_policy_predicate_candidates)

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[ExportPolicyPredicateCandidate]:
        del cls
        candidate = _module_export_policy_predicate_candidate(parsed_module)
        return [] if candidate is None else [candidate]


class ExportPolicyPredicateDetector(
    CompactModuleProjectionDetectorMixin[ExportPolicyPredicateCandidate],
    IssueDetector,
):
    module_projection_family = ExportPolicyPredicateCandidateFamily
    compact_report_context_requires_target_projection = True
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated derived-surface policy predicates should collapse into one declarative policy",
        "Several modules hand-code derived-surface policy predicates instead of routing those surfaces through one declarative policy helper.",
        "one declarative policy substrate for derived module surfaces",
        "surface-policy helper logic repeats across multiple modules with only orthogonal policy residue",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[ExportPolicyPredicateCandidate, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        candidates = projections
        if len(candidates) < 2:
            return []
        evidence = tuple(
            (
                SourceLocation(
                    candidate.file_path, candidate.line, candidate.function_name
                )
                for candidate in candidates[:6]
            )
        )
        all_roles = sorted_tuple(
            {role for candidate in candidates for role in candidate.role_names}
        )
        root_type_names = sorted_tuple(
            {
                type_name
                for candidate in candidates
                for type_name in candidate.root_type_names
            }
        )
        return [
            self.build_finding(
                (
                    f"Derived-surface predicates {', '.join(candidate.function_name for candidate in candidates[:6])} repeat policy roles {all_roles} over root types {root_type_names or ('<unconstrained>',)}."
                ),
                evidence,
                FindingBuildContext(
                    scaffold=(
                        "@dataclass(frozen=True)\nclass DerivedSurfacePolicy:\n    include_callables: bool = False\n    include_types: bool = True\n    exclude_abstract: bool = False\n    include_enums: bool = False\n    root_types: tuple[type[object], ...] = ()\n\ndef derive_surface_names(namespace: dict[str, object], policy: DerivedSurfacePolicy) -> tuple[str, ...]:\n    return tuple(sorted(name for name, value in namespace.items() if matches_surface_policy(name, value, policy)))"
                    ),
                    codemod_patch=(
                        "# Replace repeated `_is_public_*_export` helpers with one declarative `DerivedSurfacePolicy`.\n# Derive the exported name surface from the policy instead of open-coding the predicate in each module."
                    ),
                    metrics=RepeatedMethodMetrics.from_duplicate_family(
                        duplicate_site_count=len(candidates),
                        statement_count=1,
                        class_count=len(candidates),
                        method_symbols=tuple(
                            candidate.function_name for candidate in candidates
                        ),
                    ),
                ),
            )
        ]


class DerivedIndexedSurfaceDetector(
    ModuleCollectorCandidateDetector[DerivedIndexedSurfaceCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual indexed module surfaces should derive from the authoritative type family",
        "A module hand-builds an index surface over local types even though that index is derivable from the same nominal family. That splits authority between the family and a second registry projection.",
        "one derived index projected from the authoritative local type family",
        "manual dict index repeats keys and values already implied by local type families",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
    )

    def _finding_for_candidate(
        self, index_candidate: DerivedIndexedSurfaceCandidate
    ) -> RefactorFinding:
        root_names = ", ".join(index_candidate.derivable_root_names)
        return self.build_finding(
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
                "def derived_index() -> dict[object, type[object]]:\n    return {project_key(item): item for item in authoritative_family()}"
            ),
            codemod_patch=(
                f"# Delete `{index_candidate.surface_name}` as a handwritten index.\n"
                "# Derive the key-to-type map from the authoritative local family instead of maintaining a second module-level registry."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(index_candidate.value_names),
                mapping_name=index_candidate.surface_name,
                field_names=index_candidate.derivable_root_names,
            ),
        )


declare_candidate_rule_detector(
    RegisteredUnionSurfaceCandidate,
    high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual sibling-registry unions should derive from one authoritative query",
        "A module manually unions sibling class-level registry queries even though one authoritative query or shared root can derive the full family set.",
        "one derived registry-union query on an authoritative metaclass-registry root or traversal helper",
        "manual union of sibling registry queries repeats information already present in class-time registration",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.ENUMERATION,
        ),
    ),
    summary=lambda union_candidate: (
        f"`{union_candidate.owner_name}` manually unions `{union_candidate.accessor_name}` across roots {union_candidate.root_names}."
    ),
    evidence=lambda union_candidate: (
        SourceLocation(
            union_candidate.file_path,
            union_candidate.line,
            union_candidate.owner_name,
        ),
    ),
    scaffold=lambda union_candidate: (
        f"from abc import ABC\nimport re\nfrom metaclass_registry import AutoRegisterMeta\n\nclass UnifiedRegistryRoot(ABC, metaclass=AutoRegisterMeta):\n{DISPATCH_ALGEBRA_AUTHORITY.derived_registry_key_block(union_candidate.root_names)}\n\ndef {union_candidate.owner_name}(...):\n    return tuple(UnifiedRegistryRoot.__registry__.values())"
    ),
    codemod_patch=lambda union_candidate: (
        f"# Replace the manual union over {union_candidate.root_names} with one authoritative `{union_candidate.accessor_name}` query.\n# Let one shared metaclass-registry root derive the full set from `__registry__` instead of concatenating sibling roots by hand."
    ),
    metrics=lambda union_candidate: RegistrationMetrics.from_class_names(
        registration_site_count=len(union_candidate.root_names),
        registry_name=union_candidate.accessor_name,
        class_names=union_candidate.root_names,
    ),
    candidate_collector=_registered_union_surface_candidates,
)


def _concrete_type_union_contract_scaffold(
    candidate: ConcreteTypeUnionContractCandidate,
) -> str:
    method_block = "\n".join(
        (
            f"    @classmethod\n    @abstractmethod\n    def {attribute_name}(cls, context): ..."
            for attribute_name in candidate.observed_attribute_names
        )
    )
    member_block = "\n".join(
        (
            f"class {member_type_name}({candidate.suggested_contract_name}, ...): ..."
            for member_type_name in candidate.member_type_names
        )
    )
    return (
        "from abc import ABC, abstractmethod\n\n"
        f"class {candidate.suggested_contract_name}(ABC):\n"
        f"{method_block}\n\n"
        f"{member_block}\n\n"
        f"def {candidate.function_name}({candidate.parameter_name}: type[{candidate.suggested_contract_name}], ...): ..."
    )


def _concrete_type_union_contract_patch(
    candidate: ConcreteTypeUnionContractCandidate,
) -> str:
    base_action = (
        f"`{candidate.suggested_contract_name}` already declares the observed contract; use it directly."
        if candidate.common_base_names
        else f"Introduce `{candidate.suggested_contract_name}` as the shared constructor contract and make {candidate.member_type_names} inherit it."
    )
    return (
        f"# Replace the concrete class-object union on `{candidate.function_name}.{candidate.parameter_name}` "
        f"with `type[{candidate.suggested_contract_name}]` or a TypeVar bound to `{candidate.suggested_contract_name}`.\n"
        f"# {base_action}\n"
        "# Do not hide this behind a TypeAlias for the same concrete union; the consumer is depending on the shared nominal behavior."
    )


declare_candidate_rule_detector(
    ConcreteTypeUnionContractCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Concrete class-object union should be a shared nominal contract",
        "A function accepts a union of concrete class objects, then treats the parameter as one constructor or class-level capability. That concrete roster is a local re-encoding of a nominal contract.",
        "one shared ABC/protocol/base type used as type[SharedContract] or a TypeVar bound to it",
        "function parameter annotation unions concrete class objects while the body calls common class-level behavior on that parameter",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.function_name}.{candidate.parameter_name}` is annotated as a concrete class-object union "
        f"{candidate.member_type_names}, but the function only uses class-level operations "
        f"{candidate.observed_attribute_names}. Type it as `type[{candidate.suggested_contract_name}]` instead."
    ),
    evidence=lambda candidate: (
        SourceLocation(
            candidate.file_path,
            candidate.line,
            f"{candidate.function_name}.{candidate.parameter_name}",
        ),
    ),
    scaffold=_concrete_type_union_contract_scaffold,
    codemod_patch=_concrete_type_union_contract_patch,
    candidate_collector=_concrete_type_union_contract_candidates,
)


class RegistryTraversalSubstrateDetector(
    CompactModuleProjectionDetectorMixin[SubclassTraversalSite],
    IssueDetector,
):
    module_projection_family = SubclassTraversalSiteFamily
    compact_report_context_requires_target_projection = True
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Repeated subclass-family traversal should collapse into one discovery substrate",
        "Several helpers re-implement the same subclass traversal and materialization algorithm instead of sharing one authoritative family-discovery substrate.",
        "one authoritative subclass-family discovery substrate with declarative materialization hooks",
        "same subclass traversal algorithm repeats across roots, helpers, or modules with only filter/materialization residue differing",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[SubclassTraversalSite, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        group = _registry_traversal_group_from_sites(projections)
        if group is None:
            return []
        evidence = tuple(
            (
                SourceLocation(file_path, line, symbol)
                for file_path, line, symbol in zip(
                    group.file_paths, group.line_numbers, group.symbols, strict=True
                )
            )
        )
        registry_clause = (
            ""
            if not group.registry_attribute_names
            else f" over registry attributes {group.registry_attribute_names}"
        )
        filter_clause = (
            "" if not group.filter_names else f" with filter hooks {group.filter_names}"
        )
        materialization_modes = tuple(
            kind.value for kind in group.materialization_kinds
        )
        scaffold = (
            f"import re\nfrom abc import ABC\nfrom metaclass_registry import AutoRegisterMeta\n\nclass RegisteredFamily(ABC, metaclass=AutoRegisterMeta):\n{DISPATCH_ALGEBRA_AUTHORITY.derived_registry_key_block(group.symbols or ('RegisteredFamily',))}\n\ndef materialize_family(root, *, include=lambda item: True, materialize=lambda item: item):\n    return tuple(\n        materialize(item)\n        for item in root.__registry__.values()\n        if include(item)\n    )"
            if group.registry_attribute_names
            else (
                "from metaclass_registry import AutoRegisterMeta\n\ndef walk_family(root, *, include=lambda item: True, materialize=lambda item: item):\n    seen = set()\n    ordered = []\n    queue = list(root.__subclasses__())\n    while queue:\n        current = queue.pop(0)\n        queue.extend(current.__subclasses__())\n        if not include(current) or current in seen:\n            continue\n        seen.add(current)\n        ordered.append(materialize(current))\n    return tuple(ordered)\n\n# If this family is really registry-shaped, make the root an AutoRegisterMeta family and\n# read registered classes from cls.__registry__.values() instead of maintaining a second walker."
            )
        )
        return [
            self.build_finding(
                (
                    f"Helpers {', '.join(group.symbols[:6])} repeat subclass-family traversal from roots {group.root_expressions[:6]}"
                    f"{registry_clause}{filter_clause} with materialization modes {materialization_modes}."
                ),
                evidence,
                scaffold=scaffold,
                codemod_patch=(
                    "# Replace repeated subclass walkers with one shared discovery helper or one metaclass-registry root.\n# Keep only declarative include/materialize residue at each callsite instead of copying the queue/seen/append algorithm."
                ),
                metrics=RepeatedMethodMetrics.from_duplicate_family(
                    duplicate_site_count=len(group.symbols),
                    statement_count=6,
                    class_count=len(group.symbols),
                    method_symbols=group.symbols,
                ),
            )
        ]


@dataclass(frozen=True)
class CatalogInstallingMixinFamilyCandidate(ClassLineNumbersGroup):
    catalog_attribute_names: tuple[str, ...]


@dataclass(frozen=True)
class SupportPreludeModuleFamilyCandidate(MultiFileClassLineNumbersGroup):
    support_module_name: str

    @classmethod
    def from_facts(
        cls,
        facts: tuple["SupportPreludeModuleFact", ...],
    ) -> tuple["SupportPreludeModuleFamilyCandidate", ...]:
        grouped: dict[tuple[str, str], list[SupportPreludeModuleFact]] = defaultdict(
            list
        )
        for fact in facts:
            grouped[fact.parent_path, fact.support_module_name].append(fact)
        candidates: list[SupportPreludeModuleFamilyCandidate] = []
        for (_, support_import), items in grouped.items():
            if len(items) < 3:
                continue
            ordered = sorted_tuple(items, key=lambda item: item.file_path)
            candidates.append(
                cls(
                    support_module_name=support_import,
                    file_paths=tuple(item.file_path for item in ordered),
                    class_names=tuple(item.class_name for item in ordered),
                    line_numbers=tuple(item.line for item in ordered),
                )
            )
        return tuple(candidates)


@dataclass(frozen=True)
class SupportPreludeModuleFact:
    parent_path: str
    support_module_name: str
    file_path: str
    class_name: str
    line: int

    @classmethod
    def from_declaration(
        cls,
        module_path: Path,
        support_module_name: str,
        class_name: str,
        line: int,
    ) -> "SupportPreludeModuleFact | None":
        module_name = support_module_name.lstrip(".")
        if not module_name:
            return None
        support_path = module_path.parent / f"{module_name.split('.')[-1]}.py"
        if _module_has_family_catalog(support_path):
            return None
        return cls(
            parent_path=source_path_text(module_path.parent),
            support_module_name=support_module_name,
            file_path=source_path_text(module_path),
            class_name=class_name,
            line=line,
        )

    @classmethod
    def from_source_module(
        cls,
        source_module: SourceModule,
        syntax_index: NativePythonSyntaxIndex,
    ) -> list["SupportPreludeModuleFact"] | None:
        """Project one-class support-prelude modules without a module Python AST."""

        if not syntax_index.is_complete:
            return None
        top_level_classes = syntax_index.top_level_declarations("class")
        if len(top_level_classes) != 1 or syntax_index.top_level_declarations(
            "function"
        ):
            return []
        try:
            imports = tuple(
                syntax_index.statement_for(node)
                for node in syntax_index.tree.root_node.named_children
                if node.type == "import_from_statement"
            )
            support_import = _support_prelude_import_name(
                ast.Module(body=list(imports), type_ignores=[])
            )
            if support_import is None:
                return []
            class_node = top_level_classes[0]
            fact = cls.from_declaration(
                source_module.path,
                support_import,
                syntax_index.declared_name(class_node),
                class_node.start_point.row + 1,
            )
            return [] if fact is None else [fact]
        except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
            return None

    @classmethod
    def from_parsed_module(
        cls,
        parsed_module: ParsedModule,
    ) -> list["SupportPreludeModuleFact"]:
        module_node = parsed_module.module
        top_level_classes = [
            node for node in module_node.body if isinstance(node, ast.ClassDef)
        ]
        if len(top_level_classes) != 1 or any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in module_node.body
        ):
            return []
        support_import = _support_prelude_import_name(module_node)
        if support_import is None:
            return []
        class_node = top_level_classes[0]
        fact = cls.from_declaration(
            parsed_module.path,
            support_import,
            class_node.name,
            class_node.lineno,
        )
        return [] if fact is None else [fact]


@dataclass(frozen=True)
class ModuleConstructorPolicyFamilyCandidate:
    file_path: str
    constructor_name: str
    row_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    field_names: tuple[str, ...]

    evidence = ZippedSourceLocationEvidenceProperty("line_numbers", "row_names")


def _catalog_installing_mixin_candidate(method: ast.FunctionDef) -> str | None:
    return (
        Maybe.of(_CatalogInstallingMixinShape.from_method(method))
        .filter(lambda shape: shape.calls_super_init_subclass)
        .project(lambda shape: shape.catalog_attribute())
        .unwrap_or_none()
    )


@dataclass(frozen=True)
class _CatalogInstallingMixinShape:
    first_call: ast.Call
    second_call: ast.Call

    @classmethod
    def from_method(
        cls,
        method: ast.FunctionDef,
    ) -> "_CatalogInstallingMixinShape | None":
        return (
            Maybe.of(method)
            .filter(lambda function: function.name == "__init_subclass__")
            .project(
                lambda function: ast_sequence(
                    _trim_docstring_body(function.body), ast.Expr, ast.Expr
                )
            )
            .project(
                lambda statements: (
                    Maybe.of(as_ast(statements[0].value, ast.Call))
                    .with_projection(
                        lambda _first_call: as_ast(statements[1].value, ast.Call)
                    )
                    .map(lambda calls: cls(*calls))
                    .unwrap_or_none()
                )
            )
            .unwrap_or_none()
        )

    @property
    def calls_super_init_subclass(self) -> bool:
        match = attribute_call_match(
            self.first_call,
            method_name="__init_subclass__",
            owner_type=ast.Call,
            argument_count=0,
            allow_keywords=False,
        )
        return match is not None and name_id(match.owner.func) == "super"

    def catalog_attribute(self) -> str | None:
        match = attribute_call_match(
            self.second_call,
            method_name="install",
            owner_type=ast.Attribute,
            owner_name="cls",
            single_argument_name="cls",
        )
        return Maybe.of(match).map(lambda item: item.owner.attr).unwrap_or_none()


def _catalog_installing_mixin_family_candidates(
    module: ParsedModule,
) -> tuple[CatalogInstallingMixinFamilyCandidate, ...]:
    items: list[tuple[str, str, int]] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        for statement in class_node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            catalog_attribute = _catalog_installing_mixin_candidate(statement)
            if catalog_attribute is not None:
                items.append((class_node.name, catalog_attribute, statement.lineno))
    if len(items) < 2:
        return ()
    ordered = sorted_tuple(items, key=lambda item: (item[2], item[0]))
    return (
        CatalogInstallingMixinFamilyCandidate(
            file_path=module.file_path,
            class_names=tuple((item[0] for item in ordered)),
            catalog_attribute_names=tuple((item[1] for item in ordered)),
            line_numbers=tuple((item[2] for item in ordered)),
        ),
    )


def _support_prelude_import_name(module_node: ast.Module) -> str | None:
    for statement in module_node.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        if len(statement.names) != 1 or statement.names[0].name != "*":
            continue
        imported_module = statement.module or ""
        if "support" not in imported_module.lower():
            continue
        return "." * statement.level + imported_module
    return None


def _module_has_family_catalog(module_path: Path) -> bool:
    if not module_path.exists():
        return False
    try:
        source = module_path.read_text(encoding="utf-8")
    except OSError:
        return False
    syntax_index = NativePythonSyntaxIndex.from_source(source)
    if syntax_index.is_complete:
        try:
            statements = tuple(
                syntax_index.statement_for(node)
                for node in syntax_index.top_level_assignment_statements()
                if b"MODULE" in syntax_index.source_for(node)
                and (
                    b"CATALOG" in syntax_index.source_for(node)
                    or b"MANIFEST" in syntax_index.source_for(node)
                )
            )
        except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
            return False
    else:
        try:
            statements = tuple(ast.parse(source).body)
        except SyntaxError:
            return False
    return _statements_declare_family_catalog(statements)


def _statements_declare_family_catalog(
    statements: tuple[ast.stmt, ...],
) -> bool:
    for node in statements:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            (
                isinstance(target, ast.Name)
                and "MODULE" in target.id
                and ("CATALOG" in target.id or "MANIFEST" in target.id)
                for target in node.targets
            )
        ):
            continue
        if isinstance(node.value, (ast.Tuple, ast.List, ast.Set, ast.Call)):
            return True
    return False


class SupportPreludeModuleFactFamily(CollectedFamily[SupportPreludeModuleFact]):
    item_type = SupportPreludeModuleFact
    report_presence_predicate = staticmethod(lambda items, config: bool(items))
    source_collector = staticmethod(SupportPreludeModuleFact.from_source_module)
    collect = staticmethod(SupportPreludeModuleFact.from_parsed_module)


def _is_module_policy_row_name(name: str) -> bool:
    return name.isupper() and "_" in name


def _constructor_call_schema(call: ast.Call) -> tuple[str, ...]:
    return (
        *(f"arg{index}" for index, _arg in enumerate(call.args)),
        *(keyword.arg or "**" for keyword in call.keywords),
    )


def _module_constructor_policy_family_candidates(
    module: ParsedModule,
) -> tuple[ModuleConstructorPolicyFamilyCandidate, ...]:
    grouped: dict[tuple[str, tuple[str, ...]], list[tuple[str, int]]] = defaultdict(
        list
    )
    for row_name, (line, call) in _module_level_named_calls(module).items():
        if not _is_module_policy_row_name(row_name):
            continue
        schema = _constructor_call_schema(call)
        if len(schema) < 2:
            continue
        grouped[(ast.unparse(call.func), schema)].append((row_name, line))

    candidates: list[ModuleConstructorPolicyFamilyCandidate] = []
    for (constructor_name, field_names), rows in grouped.items():
        if len(rows) < 4:
            continue
        ordered = sorted_tuple(rows, key=lambda item: (item[1], item[0]))
        candidates.append(
            ModuleConstructorPolicyFamilyCandidate(
                file_path=module.file_path,
                constructor_name=constructor_name,
                row_names=tuple((row_name for row_name, _line in ordered)),
                line_numbers=tuple((line for _row_name, line in ordered)),
                field_names=field_names,
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line_numbers, item.constructor_name),
    )


class AlternateConstructorFamilyDetector(
    ModuleCollectorCandidateDetector[AlternateConstructorFamilyGroup]
):
    candidate_collector = _alternate_constructor_family_groups
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Alternate constructors should collapse into one provenance-dispatched builder",
        "Several classmethods on one record class rebuild the same keyword schema from different source node types. That provenance family should collapse into one authoritative constructor with dispatch over source kind.",
        "single provenance-aware builder for one record schema",
        "same record schema is rebuilt across sibling alternate constructors for different source types",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _finding_for_candidate(
        self, group: AlternateConstructorFamilyGroup
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(
                    group.file_path, line, f"{group.class_name}.{method_name}"
                )
                for method_name, line in zip(
                    group.method_names, group.line_numbers, strict=True
                )
            )
        )
        return self.build_finding(
            (
                f"`{group.class_name}` repeats schema keywords {group.keyword_names} across alternate constructors {group.method_names} for source types {group.source_type_names}."
            ),
            evidence,
            scaffold=(
                f"@singledispatchmethod\n@classmethod\ndef from_source(cls, source, **context) -> {group.class_name}:\n    raise TypeError\n\n@from_source.register\n@classmethod\ndef _(cls, source: SomeSource, **context):\n    return cls(...)"
            ),
            codemod_patch=(
                f"# Collapse {group.method_names} into one provenance-dispatched constructor for `{group.class_name}`.\n"
                "# Keep source-kind differences in dispatch handlers and keep the shared record schema in one authoritative builder."
            ),
            metrics=group.mapping_metrics,
        )


declare_candidate_rule_detector(
    AccumulatorFoldFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Accumulator folds should derive from one fold algebra",
        "Several methods instantiate the same accumulator, stream one source iterable through different accumulator step hooks, and return the same projection. The loop skeleton is an algebraic fold and should be one reusable composition primitive.",
        "single accumulator-fold substrate with declarative step hooks",
        "same owner class repeats accumulator initialization, loop, and result projection with only the step hook varying",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda fold_candidate: (
        f"`{fold_candidate.class_name}` repeats `{fold_candidate.accumulator_type_name}` folds across methods {fold_candidate.method_names}; step hooks are {fold_candidate.step_method_names} and result hook is `{fold_candidate.result_method_name}`."
    ),
    evidence=lambda fold_candidate: fold_candidate.evidence,
    scaffold=lambda fold_candidate: (
        "@dataclass(frozen=True)\nclass AccumulatorFoldSpec:\n    name: str\n    step_method_name: str\n\nclass AccumulatorFoldMixin:\n    __accumulator_folds__: ClassVar[AccumulatorFoldCatalog]\n    def __init_subclass__(cls):\n        cls.__accumulator_folds__.install(cls)"
    ),
    codemod_patch=lambda fold_candidate: (
        f"# Replace fold methods {fold_candidate.method_names} on `{fold_candidate.class_name}` with one accumulator-fold catalog.\n# Keep accumulator type and result projection in one authority; each source method only declares its step hook."
    ),
    metrics=lambda fold_candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(fold_candidate.method_names),
        statement_count=3,
        class_count=1,
        method_symbols=tuple(
            (
                f"{fold_candidate.class_name}.{name}"
                for name in fold_candidate.method_names
            )
        ),
    ),
    candidate_collector=_accumulator_fold_family_candidates,
)


declare_candidate_rule_detector(
    CatalogInstallingMixinFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Catalog-installing mixins should share one subclass hook",
        "Several mixins repeat the same `__init_subclass__` template: delegate to `super()` and install one classvar-held catalog. Only the catalog attribute is orthogonal; the subclass hook is one shared algorithm.",
        "one reusable catalog-installing subclass hook with declarative catalog attribute residue",
        "sibling mixins repeat an identical class-creation hook over different catalog classvars",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.MRO_ORDERING,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda catalog_candidate: (
        f"Mixins {catalog_candidate.class_names} repeat catalog installation over attributes {catalog_candidate.catalog_attribute_names}."
    ),
    evidence=lambda catalog_candidate: catalog_candidate.evidence,
    scaffold=lambda catalog_candidate: (
        "class CatalogInstallingMixin:\n    __catalog_attribute__: ClassVar[str]\n    def __init_subclass__(cls):\n        super().__init_subclass__()\n        getattr(cls, cls.__catalog_attribute__).install(cls)"
    ),
    codemod_patch=lambda catalog_candidate: (
        "# Delete the repeated `__init_subclass__` bodies after moving the lifecycle code into one catalog-installing mixin.\n# Leave only `__catalog_attribute__` on each concrete catalog mixin."
    ),
    metrics=lambda catalog_candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(catalog_candidate.class_names),
        statement_count=2,
        class_count=len(catalog_candidate.class_names),
        method_symbols=tuple(
            (
                f"{class_name}.__init_subclass__"
                for class_name in catalog_candidate.class_names
            )
        ),
    ),
    candidate_collector=_catalog_installing_mixin_family_candidates,
)


declare_candidate_rule_detector(
    RegexGroupExtractorFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Regex group extractor methods should derive from descriptors",
        "Several methods repeat `match = pattern.<mode>(text); return match.group(n) if match else None`. The pattern field and matcher mode are data; the extractor algorithm should be one descriptor or helper substrate.",
        "one regex group extraction descriptor with declared pattern and matcher coordinates",
        "same class repeats regex group extractor methods over different pattern fields",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda regex_candidate: (
        f"`{regex_candidate.class_name}` repeats regex group-{regex_candidate.group_index} extractors {regex_candidate.method_names} over patterns {regex_candidate.pattern_attribute_names}."
    ),
    evidence=lambda regex_candidate: regex_candidate.evidence,
    scaffold=lambda regex_candidate: (
        "@dataclass(frozen=True)\nclass RegexGroupExtractor:\n    pattern_attr: str\n    matcher_name: str = 'search'\n    group_index: int = 1\n    def __get__(self, instance, owner): ..."
    ),
    codemod_patch=lambda regex_candidate: (
        "# Replace repeated regex extractor methods with descriptor rows.\n# Each method name becomes a descriptor assignment declaring pattern attribute, matcher mode, and group index."
    ),
    metrics=lambda regex_candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(regex_candidate.method_names),
        mapping_name=regex_candidate.class_name,
        field_names=regex_candidate.pattern_attribute_names,
    ),
    candidate_collector=_regex_group_extractor_family_candidates,
)


class SupportPreludeModuleFamilyDetector(
    CompactModuleProjectionDetectorMixin[SupportPreludeModuleFact],
    IssueDetector,
):
    module_projection_family = SupportPreludeModuleFactFamily
    compact_report_context_requires_target_projection = True
    finding_spec = finding_spec_template(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Support-prelude module families should have a manifest authority",
        "Many one-class modules importing the same support prelude form an implicit module family. The family boundary should be derived from one manifest/catalog rather than remaining visible only as repeated import shape.",
        "one manifest authority for a support-prelude module family",
        "several one-class modules share the same star-import support prelude without a module-family catalog",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[SupportPreludeModuleFact, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in SupportPreludeModuleFamilyCandidate.from_facts(projections):
            findings.append(
                self.build_finding(
                    (
                        f"{len(candidate.class_names)} one-class modules share support prelude `{candidate.support_module_name}` without a manifest authority."
                    ),
                    candidate.evidence[:8],
                    scaffold=(
                        "@dataclass(frozen=True)\nclass ModuleFamilyCatalog:\n    members: tuple[ModuleFamilyMember, ...]\n    @classmethod\n    def from_package(cls, package_dir, support_module): ..."
                    ),
                    codemod_patch=(
                        "# Add one module-family catalog beside the shared support prelude.\n# Derive member rows from package structure instead of relying only on repeated star-import shape."
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(candidate.class_names),
                        mapping_name=candidate.support_module_name,
                        field_names=candidate.class_names,
                    ),
                )
            )
        return findings


declare_candidate_rule_detector(
    ModuleConstructorPolicyFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Module constructor policy rows should derive from a semantic catalog",
        "Several module-level constant rows instantiate the same policy constructor with the same argument schema. Those rows are semantic data, so the architecture should derive them from one role/catalog authority rather than spell each constructor call by hand.",
        "one constructor-row catalog keyed by semantic policy role",
        "same module has multiple constant rows assigned from the same constructor shape",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    ),
    summary=lambda policy_candidate: (
        f"Module constants {', '.join(policy_candidate.row_names)} repeat `{policy_candidate.constructor_name}` constructor rows with schema {policy_candidate.field_names}."
    ),
    evidence=lambda policy_candidate: policy_candidate.evidence,
    scaffold=lambda policy_candidate: (
        "@dataclass(frozen=True)\nclass PolicyRowSpec:\n    role_name: str\n    constructor_args: tuple[object, ...]\n\nclass PolicyCatalog:\n    def materialize(self) -> dict[str, object]: ..."
    ),
    codemod_patch=lambda policy_candidate: (
        "# Replace repeated module-level constructor rows with one semantic policy catalog.\n# Keep role names and constructor coordinates as data, then derive the module constants from the catalog."
    ),
    metrics=lambda policy_candidate: MappingMetrics(
        mapping_site_count=len(policy_candidate.row_names),
        field_count=len(policy_candidate.field_names),
        mapping_name=policy_candidate.constructor_name,
        field_names=policy_candidate.row_names,
    ),
    candidate_collector=_module_constructor_policy_family_candidates,
)


declare_candidate_rule_detector(
    DynamicSelfFieldSelectionCandidate,
    high_confidence_spec(
        PatternId.CONFIG_CONTRACTS,
        "Dynamic self-field selection hides a nominal contract",
        "A class selects one of its own fields through reflective indirection instead of declaring one fail-loud hook or one canonical field.",
        "declared nominal count/value hook instead of selector-driven reflective lookup",
        "class template selects its own state through dynamic reflective field names",
        (
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    ),
    summary=lambda dynamic_candidate: (
        f"`{dynamic_candidate.class_name}.{dynamic_candidate.method_name}` uses `{dynamic_candidate.reflective_builtin}(self, {dynamic_candidate.selector_expression})` instead of one declared nominal value."
    ),
    scaffold=lambda dynamic_candidate: (
        "class DeclaredCountValue(ABC):\n    @property\n    @abstractmethod\n    def count_value(self) -> int: ..."
    ),
    codemod_patch=lambda dynamic_candidate: (
        f"# Delete `{dynamic_candidate.reflective_builtin}(self, {dynamic_candidate.selector_expression})`.\n# Replace selector-driven reflection with one declared property or one canonical field on the nominal carrier."
    ),
    candidate_collector=_dynamic_self_field_selection_candidates,
)

declare_candidate_rule_detector(
    StringBackedReflectiveNominalLookupCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "String-backed reflective lookup is simulating nominal identity",
        "The docs say a class family should not smuggle behavior through string selectors and reflection. When subclasses only supply constant names that are resolved through globals, getattr, or __dict__, the boundary should become one declared nominal hook or typed handle.",
        "declared nominal hook or typed family handle instead of string selector plus reflection",
        "class family encodes behavior with constant selector strings and resolves it reflectively",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.STRING_DISPATCH,
            ObservationTag.SEMANTIC_STRING_LITERAL,
            ObservationTag.CLASS_FAMILY,
        ),
    ),
    summary=lambda reflective_candidate: (
        f"`{reflective_candidate.class_name}.{reflective_candidate.method_name}` resolves `{reflective_candidate.selector_attr_name}` through `{reflective_candidate.lookup_kind}` over {len(reflective_candidate.concrete_class_names)} concrete classes."
    ),
    scaffold=lambda reflective_candidate: (
        "class DeclaredNominalRole(ABC):\n    @classmethod\n    @abstractmethod\n    def declared_handle(cls) -> object: ..."
    ),
    codemod_patch=lambda reflective_candidate: (
        f"# Delete the reflective `{reflective_candidate.lookup_kind}` lookup keyed by `{reflective_candidate.selector_attr_name}`.\n# Move the family boundary to one declared hook, typed handle, or polymorphic method."
    ),
    metrics=lambda reflective_candidate: SentinelSimulationMetrics(
        class_count=len(reflective_candidate.concrete_class_names),
        branch_site_count=1,
    ),
    detector_base=ConfiguredModuleCollectorCandidateDetector,
    candidate_collector=_string_backed_reflective_nominal_lookup_candidates,
)
