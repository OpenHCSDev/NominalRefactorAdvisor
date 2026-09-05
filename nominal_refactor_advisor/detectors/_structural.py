"""Structural detector implementations.

This module groups detector families and helper logic centered on repeated
field families, wrapper surfaces, exports, and structural record mechanics.
"""

from __future__ import annotations

import ast

from ..semantic_algebra import ObjectFamilyShape
from ..semantic_description_length import CompressionCertificate
from ..class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from ..codemod import (
    ExactLeafMethodAncestorPromotionFindingRecipeSynthesizer,
    ExactMethodRoleFindingRecipeSynthesizer,
)
from ..exact_method_authority import (
    ExactLeafMethodAncestorPromotionComponent,
    ExactMethodRoleComponent,
)
from ..semantic_match import (
    Maybe,
    attribute_call_match,
)
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import *
from ._helpers import *
from ._helpers import _property_alias_hook_groups
from ._structural_step_regex_extractor import *
from ._substrate_support import *

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
            if not WitnessMixinRole.recognizes(role_name):
                continue
            role_to_classes[role_name][candidate.class_name] = candidate
            role_to_fields[role_name].update(field_names)
    role_field_names = tuple(
        (
            (role.value, sorted_tuple(role_to_fields[role.value]))
            for role in WitnessMixinRole
            if len(role_to_classes[role.value]) >= 2
            and len(role_to_fields[role.value]) >= 2
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
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
        return self.build_finding(
            (
                f"Subclasses {', '.join(hook_group.class_names)} of `{hook_group.base_name}` all implement `{hook_group.property_name}` as `return self.{hook_group.returned_attribute}`."
            ),
            evidence,
            metrics=hook_group.repeated_method_metrics,
        )


MethodFamilyCandidateT = TypeVar("MethodFamilyCandidateT")


class _CompactMethodFamilyDetectorBase(
    CompactContextCandidateDetector[
        CompactModuleClassProjection,
        CompactMethodFamilyContext,
        MethodFamilyCandidateT,
    ],
    Generic[MethodFamilyCandidateT],
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
    ) -> CompactMethodFamilyContext:
        del config
        return CompactMethodFamilyContext.from_projections(projections)

    @classmethod
    def _compact_context_from_shared(
        cls,
        context: object | None,
    ) -> CompactMethodFamilyContext:
        if isinstance(context, CompactMethodFamilyContext):
            return context
        repository = CompactClassRepositoryContext.require(context)
        return repository.cached(
            CompactMethodFamilyContext,
            lambda: CompactMethodFamilyContext.from_projections(
                repository.projections,
                class_index=repository.class_index,
            ),
        )

    def _candidates_from_compact_context(
        self,
        context: CompactMethodFamilyContext,
        config: DetectorConfig,
    ) -> Sequence[MethodFamilyCandidateT]:
        del config
        return context.candidates_for(type(self).required_candidate_type())


class _CompactExactTinyMethodRoleDetectorBase(
    ExactMethodRoleFindingRecipeSynthesizer,
    _CompactMethodFamilyDetectorBase[ExactMethodRoleComponent],
):
    """Compose compact exact-role detection with its executable refactor."""


class _CompactExactLeafMethodAncestorPromotionDetectorBase(
    ExactLeafMethodAncestorPromotionFindingRecipeSynthesizer,
    _CompactMethodFamilyDetectorBase[ExactLeafMethodAncestorPromotionComponent],
):
    """Compose closed-leaf detection with its executable promotion refactor."""


declare_candidate_rule_detector(
    ExactMethodRoleComponent,
    high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Exact tiny methods expose one repeated nominal role",
        "Several classes without one nominal ancestor own the same complete tiny method declaration. The exact method role is one maintenance object, but the current snapshot does not prove whether an existing authority or a new inheritance boundary should own it.",
        "one proved nominal authority derives the exact method role for every participant",
        "unrelated classes repeat the same promotion-safe method declaration and the exact-role compression certificate pays rent",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.METHOD_ROLE,
            ObservationTag.REPEATED_METHOD_ROLES,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"{candidate.method_names} are repeated exactly across "
        f"{candidate.participant_class_names} without one nominal ancestor and contain "
        f"{candidate.statement_count} shared statement(s); no ownership placement is "
        "selected from this snapshot."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.participant_class_names),
        statement_count=candidate.statement_count,
        class_count=len(candidate.participant_class_names),
        method_symbols=candidate.method_symbols,
    ),
    detector_name="ExactTinyMethodRoleDetector",
    detector_base=_CompactExactTinyMethodRoleDetectorBase,
)


declare_candidate_rule_detector(
    ExactLeafMethodAncestorPromotionComponent,
    high_confidence_certified_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Closed leaf family repeats methods owned by its direct authority",
        "Every direct leaf of one resolved nominal authority owns the same complete promotion-safe method set. The complete direct-child relation, exact source identity, receiver contract, and absence of competing ancestor definitions prove the existing authority as the unique promotion target.",
        "the existing direct authority owns each exact shared method once",
        "all direct children are leaves, the authority is unique, and the complete exact method batch is closed over authority-declared receiver members",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.METHOD_ROLE,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"{candidate.participant_class_names} repeat exact methods "
        f"{candidate.method_names} across every direct leaf of "
        f"`{candidate.authority_name}`; the closed inheritance relation proves "
        "that existing class as the unique owner."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    authority_evidence=lambda candidate: candidate.evidence_locations[0],
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.participant_class_symbols),
        statement_count=candidate.statement_count,
        class_count=len(candidate.participant_class_symbols),
        method_symbols=candidate.method_symbols,
    ),
    detector_name="ExactLeafMethodAncestorPromotionDetector",
    detector_base=_CompactExactLeafMethodAncestorPromotionDetectorBase,
)


declare_candidate_rule_detector(
    SemanticOverlapMethodCandidate,
    high_confidence_certified_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Sibling implementations expose one repeated algorithm",
        "Sibling classes that share a base implement the same statement skeleton at several maintenance sites. The common statements and varying expression coordinates prove semantic overlap, but do not by themselves prove where that algorithm belongs in the reachable inheritance design.",
        "one proved nominal authority derives the shared algorithm without duplicating its residue semantics",
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
        f"The observed leaf residue basis is {candidate.leaf_residue_names} across "
        f"{candidate.residue_declaration_count} residue declaration(s), with a "
        f"shared/residue ratio of {candidate.shared_to_residue_ratio:.2f}. "
        f"Its enclosing class-set evidence contains {candidate.lattice_node_count} lattice node(s), "
        f"{candidate.lattice_edge_count} lattice edge(s), family methods {candidate.family_method_names}, "
        f"strict-subset families {candidate.strict_subset_family_specs}, and partial-overlap families "
        f"{candidate.partial_overlap_family_specs}; no hierarchy placement is selected from this snapshot."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
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
    detector_name="SemanticOverlapMethodDetector",
    detector_base=_CompactMethodFamilyDetectorBase,
)


declare_candidate_rule_detector(
    SemanticOverlapMethodFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Class-family methods expose one repeated algorithm family",
        "A class family has several methods with compatible anti-unifiable bodies over the same base and subclass set. Their combined evidence establishes a shared maintenance object while leaving its correct inheritance placement unresolved.",
        "one proved nominal authority derives the complete shared method family and its irreducible residue",
        "multiple semantic-overlap method observations share the same base and subclass family",
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
        f"with {candidate.shared_statement_count} shared statements and {candidate.residue_declaration_count} "
        f"residue declaration(s). The observed leaf residue basis is {candidate.leaf_residue_names}, "
        f"with a shared/residue ratio of {candidate.shared_to_residue_ratio:.2f}. The surrounding class sets "
        f"contain {candidate.lattice_node_count} lattice node(s), {candidate.lattice_edge_count} lattice edge(s), "
        f"strict-subset families {candidate.strict_subset_family_specs}, and partial-overlap families "
        f"{candidate.partial_overlap_family_specs}; no hierarchy placement is selected from this snapshot."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.method_symbols),
        statement_count=candidate.shared_statement_count,
        class_count=len(candidate.class_names),
        method_symbols=candidate.method_symbols,
    ),
    detector_name="SemanticOverlapMethodFamilyDetector",
    detector_base=_CompactMethodFamilyDetectorBase,
)


declare_candidate_rule_detector(
    OverlappingInheritanceFamiliesCandidate,
    high_confidence_certified_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Overlapping inheritance families require one authority proof",
        "A base class has several intersecting subclass method families. Each family is only a local view; the complete class-set lattice is the evidence boundary that any safe authority assignment must cover.",
        "one proved authority assignment covers the complete inheritance lattice without duplicating algorithms or residue semantics",
        "multiple semantic-overlap method families under one root have intersecting but non-identical subclass sets",
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
        f"`{candidate.base_name}` has an inheritance lattice over classes {candidate.class_names}: "
        f"families {candidate.family_specs}, methods {candidate.method_names}, "
        f"{candidate.lattice_node_count} lattice node(s), {candidate.lattice_edge_count} edge(s), "
        f"strict-subset families {candidate.strict_subset_family_specs}, partial-overlap families "
        f"{candidate.partial_overlap_family_specs}, {candidate.shared_statement_count} shared statements, "
        f"{candidate.residue_declaration_count} residue declarations, and observed residue basis "
        f"{candidate.leaf_residue_names}. No hierarchy placement is selected from this snapshot."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    compression_certificate=lambda candidate: candidate.compression_certificate,
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.method_symbols),
        statement_count=candidate.shared_statement_count,
        class_count=len(candidate.class_names),
        method_symbols=candidate.method_symbols,
    ),
    detector_name="OverlappingInheritanceFamiliesDetector",
    detector_base=_CompactMethodFamilyDetectorBase,
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
            compression_certificate=_reflective_self_attribute_compression_certificate(
                reflective_candidate
            ),
        )


declare_candidate_rule_detector(
    RepeatedBaseBundleCandidate,
    high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.class_count,
        mapping_name="mro-base-bundle",
        field_names=candidate.base_names,
    ),
    detector_name="RepeatedBaseBundleDetector",
    candidate_collector=_repeated_base_bundle_candidates,
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
    metrics=lambda union_candidate: RegistrationMetrics.from_class_names(
        registration_site_count=len(union_candidate.root_names),
        registry_name=union_candidate.accessor_name,
        class_names=union_candidate.root_names,
    ),
    candidate_collector=_registered_union_surface_candidates,
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
        "one shared ABC used as type[SharedContract] or a TypeVar bound to it",
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
    candidate_collector=_concrete_type_union_contract_candidates,
)


@dataclass(frozen=True)
class CatalogInstallingMixinFamilyCandidate(ClassLineNumbersGroup):
    catalog_attribute_names: tuple[str, ...]


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
                    statements_without_docstring(function.body), ast.Expr, ast.Expr
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


declare_candidate_rule_detector(
    AccumulatorFoldFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
    metrics=lambda regex_candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(regex_candidate.method_names),
        mapping_name=regex_candidate.class_name,
        field_names=regex_candidate.pattern_attribute_names,
    ),
    candidate_collector=_regex_group_extractor_family_candidates,
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
    metrics=lambda reflective_candidate: SentinelSimulationMetrics(
        class_count=len(reflective_candidate.concrete_class_names),
        branch_site_count=1,
    ),
    detector_base=ConfiguredModuleCollectorCandidateDetector,
    candidate_collector=_string_backed_reflective_nominal_lookup_candidates,
)
