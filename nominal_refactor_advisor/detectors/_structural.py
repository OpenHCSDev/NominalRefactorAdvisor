"""Structural detector implementations.

This module groups detector families and helper logic centered on repeated
field families, wrapper surfaces, exports, and structural record mechanics.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

from nominal_refactor_advisor.semantic_algebra import ObjectFamilyShape
from nominal_refactor_advisor.semantic_description_length import CompressionCertificate
from nominal_refactor_advisor.class_index import CompactModuleClassProjection
from nominal_refactor_advisor.refactors.finding_recipes import (
    ExactLeafMethodAncestorPromotionFindingRecipeSynthesizer,
    ExactMethodRoleFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.exact_method_authority import (
    ExactLeafMethodAncestorPromotionComponent,
    ExactMethodRoleComponent,
)
from nominal_refactor_advisor.semantic_match import (
    Maybe,
    attribute_call_match,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag
from dataclasses import dataclass
from typing import Generic, Self, TypeVar, Iterable, cast
from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    statements_without_docstring,
    AttributeAccessCall,
    BuiltinCall,
    Getattr,
    NamespaceCall,
)
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.models import (
    MappingMetrics,
    RefactorFinding,
    RegistrationMetrics,
    RepeatedMethodMetrics,
    SentinelSimulationMetrics,
    SourceLocation,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.semantic_match import as_ast, ast_sequence, name_id
from nominal_refactor_advisor.detectors._base import (
    CollectsItself,
    ClassLineNumbersGroup,
    CompactClassRepositoryContext,
    CompactClassRepositoryProjectionMixin,
    CompactContextCandidateDetector,
    ConfiguredModuleCollectorCandidateDetector,
    DetectorConfig,
    ModuleCollectorCandidateDetector,
    OverlappingInheritanceFamiliesCandidate,
    SemanticOverlapMethodCandidate,
    SemanticOverlapMethodFamilyCandidate,
    declare_candidate_rule_detector,
    high_confidence_certified_spec,
    high_confidence_spec,
    CANDIDATE_COLLECTION_AUTHORITY,
    SelfReflectiveBuiltinCandidate,
    ClassLineWitnessCandidate,
    CollectsItselfUnderConfig,
    PropertyHookGroup,
)
from nominal_refactor_advisor.detectors._helpers import (
    CompactMethodFamilyContext,
    BaseBundleClassGroups,
    _IMPLICIT_METHOD_PARAMETER_NAMES,
    _RegisteredUnionSurfaceSource,
    _annotation_union_members,
    HELPER_SYNTAX_PROJECTION_AUTHORITY,
)
from nominal_refactor_advisor.detectors._structural_step_regex_extractor import (
    AccumulatorFoldFamilyCandidate,
    RegexGroupExtractorFamilyCandidate,
)
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.class_families.parsed_index import ClassStatement, ModuleClassStatements
from nominal_refactor_advisor.detectors._candidates import LineWitnessCandidate
from nominal_refactor_advisor.detectors._substrate_support import _class_ancestor_name_map, _constant_string, _selector_attribute_name
from collections import defaultdict
from nominal_refactor_advisor.class_symbols import ClassSymbolResolutionAuthority
from nominal_refactor_advisor.name_algebra import CLASS_NAME_ALGEBRA
from nominal_refactor_advisor.annotation_semantics import TypeWrapper

@dataclass(frozen=True)
class ReflectiveSelfAttributeCandidate(CollectsItself, SelfReflectiveBuiltinCandidate):
    attribute_name: str

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
            module,
            ast.ClassDef,
            cls._for_class,
        )

    @classmethod
    def _for_class(cls, module: ParsedModule, node: ast.ClassDef) -> Iterable[Self]:
        for statement in node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for subnode in ast.walk(statement):
                if not isinstance(subnode, ast.Call):
                    continue
                match BuiltinCall.decoded(subnode):
                    case Getattr(carries_default=False) as call:
                        receiver, attribute_name_node = (
                            call.object_argument,
                            call.name_argument,
                        )
                    case _:
                        continue
                attribute_name = _constant_string(attribute_name_node)
                if not (
                    isinstance(receiver, ast.Name)
                    and receiver.id == "self"
                    and (attribute_name is not None)
                ):
                    continue
                yield cls(
                    file_path=module.file_path,
                    line=subnode.lineno,
                    subject_name=node.name,
                    name_family=(attribute_name,),
                    method_name=statement.name,
                    reflective_builtin=call.builtin,
                    attribute_name=attribute_name,
                )


@dataclass(frozen=True)
class PropertyAliasHookGroup(CollectsItself, PropertyHookGroup):
    returned_attribute: str

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        grouped: dict[tuple[str, str, str], list[tuple[str, int]]] = defaultdict(list)
        for declared in module.fact(ModuleClassStatements).statements:
            node = declared.node
            base_names = declared.family_base_names
            if not base_names:
                continue
            for method in declared.methods:
                if not method.spells("property"):
                    continue
                statement = method.node
                if len(statement.args.args) != 1:
                    continue
                body = statements_without_docstring(statement.body)
                if len(body) != 1 or not isinstance(body[0], ast.Return):
                    continue
                returned = body[0].value
                if not (
                    isinstance(returned, ast.Attribute)
                    and isinstance(returned.value, ast.Name)
                    and (returned.value.id == "self")
                ):
                    continue
                for base_name in base_names:
                    grouped[base_name, statement.name, returned.attr].append(
                        (node.name, statement.lineno)
                    )
        return tuple(
            (
                cls(
                    file_path=module.file_path,
                    base_name=base_name,
                    property_name=property_name,
                    returned_attribute=returned_attribute,
                    class_names=tuple((class_name for class_name, _ in ordered)),
                    line_numbers=tuple((line for _, line in ordered)),
                )
                for (base_name, property_name, returned_attribute), items in sorted(
                    grouped.items()
                )
                if len(items) >= 2
                for ordered in [sorted_tuple(items, key=lambda item: (item[1], item[0]))]
            )
        )


class RepeatedPropertyAliasHookDetector(
    ModuleCollectorCandidateDetector[PropertyAliasHookGroup]
):
    detector_id = "repeated_property_alias_hooks"
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Repeated property hook aliases should move into a shared base or mixin",
        "Several subclasses redeclare the same one-line property over the same backing attribute. Declare it once in a shared base or mixin.",
        "one property implementation on the shared base",
        "the same property alias repeats across siblings of one base",
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
    CompactClassRepositoryProjectionMixin,
    CompactContextCandidateDetector[
        CompactModuleClassProjection,
        CompactMethodFamilyContext,
        MethodFamilyCandidateT,
    ],
    Generic[MethodFamilyCandidateT],
):
    @classmethod
    def _compact_context_from_projections(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        config: DetectorConfig,
    ) -> CompactMethodFamilyContext:
        del config
        return CompactMethodFamilyContext.from_projections(projections)

    @classmethod
    def _compact_context_from_shared(
        cls,
        context: object | None,
    ) -> CompactMethodFamilyContext:
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
        "Several classes with no common base each own the same complete tiny method. One owner should hold it, but the scan can't tell whether that's an existing class or a new base.",
        "one proven owner provides the method for every participant",
        "unrelated classes repeat the same method, and the receiver is closed; you decide where it goes",
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


ExactLeafMethodAncestorPromotionDetector = declare_candidate_rule_detector(
    ExactLeafMethodAncestorPromotionComponent,
    high_confidence_certified_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Closed leaf family repeats methods owned by its direct authority",
        "Every direct leaf of one resolved base owns the same complete set of safe-to-promote methods. The child list, source identity, receiver contract and lack of competing ancestor definitions show the existing base is the only place to promote them.",
        "the existing base owns each shared method once",
        "all direct children are leaves, the base is unique, and the whole method batch only uses members the base declares",
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
    authority_evidence=lambda candidate: candidate.evidence[0],
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
        "Sibling classes under one base implement the same statement skeleton in several places. The shared statements and varying expressions show the overlap, but not where in the inheritance tree the algorithm belongs.",
        "one proven owner provides the shared algorithm without duplicating its varying parts",
        "the same method across sibling classes has one statement skeleton with a small varying part",
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
        "A class family has several methods whose bodies line up over the same base and subclass set. Together they show one shared maintenance target, but not where in the inheritance tree it belongs.",
        "one proven owner provides the whole shared method family and its varying parts",
        "several overlapping-method findings share the same base and subclass family",
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
        "A base class has several overlapping subclass method families. Each is only a partial view. Any safe owner assignment has to cover the full set of overlapping subclass sets.",
        "one proven owner assignment covers the whole inheritance structure without duplicating algorithms or varying parts",
        "several overlapping-method families under one root have intersecting but different subclass sets",
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
    _REFLECTIVE_ATTRIBUTE_CONTRACT_REPLACEMENT_SHAPE = ObjectFamilyShape(
        shared_objects=("nominal_attribute_contract",)
    )
    finding_spec = high_confidence_spec(
        PatternId.CONFIG_CONTRACTS,
        "Reflective self-attribute access hides a nominal contract",
        "A class reads its own attributes through getattr or similar with a hardcoded string, instead of declaring the field or property on the base. The contract stays partial, stringly and fails quietly.",
        "declared attribute on the base that fails loudly when missing",
        "class template reads its own required state through a string attribute lookup",
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
            compression_certificate=self._reflective_self_attribute_compression_certificate(
                reflective_candidate
            ),
        )

    @staticmethod
    def _reflective_self_attribute_compression_certificate(
        candidate: ReflectiveSelfAttributeCandidate,
    ) -> CompressionCertificate:
        return CompressionCertificate.from_object_family(
            manual_object_count=4,
            replacement_shape=ReflectiveSelfAttributeEscapeDetector._REFLECTIVE_ATTRIBUTE_CONTRACT_REPLACEMENT_SHAPE,
            semantic_axes=(candidate.attribute_name, candidate.reflective_builtin),
        )


@dataclass(frozen=True)
class RepeatedBaseBundleCandidate(CollectsItself, ClassLineNumbersGroup):
    base_names: tuple[str, ...]
    bundle_width: int
    class_count: int
    line_count: int

    _ABC_BASE_NAME = "ABC"

    _COMPOSABLE_BASE_NAME_SUFFIXES = (
        _ABC_BASE_NAME,
        "Base",
        "Carrier",
        "Contract",
        "Mixin",
        "Template",
    )

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        grouped: BaseBundleClassGroups = defaultdict(list)
        for declared in module.fact(ModuleClassStatements).top_level:
            node = declared.node
            if node.end_lineno is None:
                continue
            for bundle in cls._contiguous_base_bundles(declared.spelled_base_names):
                grouped[bundle].append(node)
        candidates: list[Self] = []
        for bundle, nodes in cls._maximal_repeated_base_bundle_items(grouped):
            line_numbers = tuple((node.lineno for node in nodes))
            candidates.append(
                cls(
                    file_path=module.file_path,
                    class_names=tuple((node.name for node in nodes)),
                    line_numbers=line_numbers,
                    base_names=bundle,
                    bundle_width=len(bundle),
                    class_count=len(nodes),
                    line_count=sum(
                        (
                            (node.end_lineno or node.lineno) - node.lineno + 1
                            for node in nodes
                        )
                    ),
                )
            )
        return tuple(candidates)

    @staticmethod
    def _contiguous_base_bundles(
        base_names: tuple[str, ...],
        *,
        minimum_width: int = 3,
    ) -> tuple[tuple[str, ...], ...]:
        return tuple(
            (
                base_names[start:end]
                for start in range(len(base_names))
                for end in range(start + minimum_width, len(base_names) + 1)
                if all(RepeatedBaseBundleCandidate._is_composable_base_name(name) for name in base_names[start:end])
            )
        )

    @staticmethod
    def _maximal_repeated_base_bundle_items(
        grouped: BaseBundleClassGroups,
    ) -> tuple[tuple[tuple[str, ...], tuple[ast.ClassDef, ...]], ...]:
        qualified = tuple(
            (
                (bundle, sorted_tuple(nodes, key=lambda node: node.lineno))
                for bundle, nodes in grouped.items()
                if len(nodes) >= 3
            )
        )
        maximal: list[tuple[tuple[str, ...], tuple[ast.ClassDef, ...]]] = []
        for bundle, nodes in sorted(
            qualified,
            key=lambda item: (-len(item[1]), -len(item[0]), item[0]),
        ):
            class_names = {node.name for node in nodes}
            if any(
                (
                    class_names <= {node.name for node in existing_nodes}
                    and RepeatedBaseBundleCandidate._is_contiguous_subtuple(bundle, existing_bundle)
                    for existing_bundle, existing_nodes in maximal
                )
            ):
                continue
            maximal.append((bundle, nodes))
        return tuple(maximal)

    @staticmethod
    def _is_composable_base_name(base_name: str) -> bool:
        return base_name == RepeatedBaseBundleCandidate._ABC_BASE_NAME or base_name.endswith(
            RepeatedBaseBundleCandidate._COMPOSABLE_BASE_NAME_SUFFIXES
        )

    @staticmethod
    def _is_contiguous_subtuple(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
        if len(needle) > len(haystack):
            return False
        return any(
            haystack[start : start + len(needle)] == needle
            for start in range(len(haystack) - len(needle) + 1)
        )


declare_candidate_rule_detector(
    RepeatedBaseBundleCandidate,
    high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Repeated MRO base bundle should become a named ABC mixin",
        "Several classes repeat the same run of base classes. That run is already a unit, so give it a name and reuse it as one ABC or mixin.",
        "one named ABC or mixin for the repeated base run",
        "class family repeats the same base sequence in each declaration",
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
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.class_count,
        mapping_name="mro-base-bundle",
        field_names=candidate.base_names,
    ),
    detector_name="RepeatedBaseBundleDetector",
)


@dataclass(frozen=True)
class RegisteredUnionSurfaceCandidate(CollectsItself, LineWitnessCandidate):
    owner_name: str
    accessor_name: str
    root_names: tuple[str, ...]
    witness_name = AliasProperty[str]("owner_name")

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        classes_by_name = module.fact(ModuleClassStatements).top_level_by_name
        source_nodes: tuple[ast.AST, ...] = (
            *(function for _, function in module.syntax_index.named_functions),
            *module.syntax_index.instances_of(ast.Assign),
        )
        return tuple(
            candidate
            for node in source_nodes
            for candidate in cls._for_node(
                module, node, classes_by_name
            )
        )

    @classmethod
    def _for_node(
        cls,
        module: ParsedModule,
        node: ast.AST,
        classes_by_name: dict[str, ClassStatement],
    ) -> Iterable[Self]:
        source = _RegisteredUnionSurfaceSource.from_node(node)
        if source is None:
            return
        registered_surface = cls._registered_surface_roots(source.value)
        if registered_surface is None:
            return
        accessor_name, root_names = registered_surface
        if len(root_names) < 2:
            return
        root_nodes = [classes_by_name.get(root_name) for root_name in root_names]
        if any((root_node is None for root_node in root_nodes)):
            return
        if any(
            (
                (
                    method := cast(ClassStatement, root_node).method_named(
                        accessor_name
                    )
                )
                is None
                or not method.spells("classmethod")
                for root_node in root_nodes
            )
        ):
            return
        yield cls(
            file_path=module.file_path,
            line=source.line,
            owner_name=source.owner_name,
            accessor_name=accessor_name,
            root_names=root_names,
        )

    @staticmethod
    def _registered_surface_roots(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
        calls: list[ast.Call] = []

        def collect_calls(current: ast.AST) -> bool:
            binary = as_ast(current, ast.BinOp)
            if binary is not None and isinstance(binary.op, ast.Add):
                return collect_calls(binary.left) and collect_calls(binary.right)
            call = as_ast(current, ast.Call)
            if call is None:
                return False
            calls.append(call)
            return True

        if not collect_calls(node) or len(calls) < 2:
            return None
        accessors = tuple(
            (
                accessor
                for call in calls
                if (
                    accessor := attribute_call_match(
                        call, owner_type=ast.Name, argument_count=0, allow_keywords=False
                    )
                )
                is not None
            )
        )
        accessor_names = {accessor.attribute.attr for accessor in accessors}
        if len(accessor_names) != 1:
            return None
        accessor_name = next(iter(accessor_names))
        root_names = sorted_tuple((accessor.owner.id for accessor in accessors))
        return (accessor_name, root_names)


declare_candidate_rule_detector(
    RegisteredUnionSurfaceCandidate,
    high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual sibling-registry unions should derive from one authoritative query",
        "A module unions several sibling registry queries by hand, though one query or shared root can produce the whole set.",
        "one derived registry-union query on a registry root or traversal helper",
        "the manual union repeats what class-time registration already knows",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.ENUMERATION,
        ),
    ),
    summary=lambda union_candidate: (
        f"`{union_candidate.owner_name}` manually unions `{union_candidate.accessor_name}` across roots {union_candidate.root_names}."
    ),
    metrics=lambda union_candidate: RegistrationMetrics.from_class_names(
        registration_site_count=len(union_candidate.root_names),
        registry_name=union_candidate.accessor_name,
        class_names=union_candidate.root_names,
    ),
)


@dataclass(frozen=True)
class ConcreteTypeUnionContractCandidate(CollectsItself, LineWitnessCandidate):
    function_name: str
    parameter_name: str
    member_type_names: tuple[str, ...]
    observed_attribute_names: tuple[str, ...]
    suggested_contract_name: str
    common_base_names: tuple[str, ...]

    _CONCRETE_UNION_FACTORY_ATTRIBUTE_PREFIXES = (
        "build",
        "create",
        "from_",
        "make",
    )

    @property
    def witness_name(self) -> str:
        return f"{self.function_name}.{self.parameter_name}"

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        classes_by_name = module.fact(ModuleClassStatements).by_name
        base_lookup = {
            class_name: set(declared.base_terminal_names)
            for class_name, declared in classes_by_name.items()
        }
        ancestor_names_by_class = _class_ancestor_name_map(base_lookup)
        return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
            module,
            cls._for_function,
            classes_by_name,
            ancestor_names_by_class,
            sort_key=lambda item: (item.file_path, item.line, item.function_name),
        )

    @classmethod
    def _for_function(
        cls,
        module: ParsedModule,
        function_name: str,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        classes_by_name: dict[str, ClassStatement],
        ancestor_names_by_class: dict[str, tuple[str, ...]],
    ) -> Iterable[Self]:
        arguments = (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
        for argument in arguments:
            if argument.arg in _IMPLICIT_METHOD_PARAMETER_NAMES:
                continue
            member_type_names = cls._concrete_class_object_union_member_names(
                argument.annotation
            )
            if len(member_type_names) < 2 or any(
                member_type_name not in classes_by_name
                for member_type_name in member_type_names
            ):
                continue
            observed_attribute_names = cls._parameter_class_call_attribute_names(
                function, argument.arg
            )
            if not observed_attribute_names:
                continue
            common_base_names = cls._common_contract_base_names(
                classes_by_name,
                ancestor_names_by_class,
                member_type_names,
                observed_attribute_names,
            )
            yield cls(
                file_path=module.file_path,
                line=argument.lineno,
                function_name=function_name,
                parameter_name=argument.arg,
                member_type_names=member_type_names,
                observed_attribute_names=observed_attribute_names,
                suggested_contract_name=(
                    common_base_names[0]
                    if common_base_names
                    else cls._concrete_union_contract_name(
                        member_type_names, observed_attribute_names
                    )
                ),
                common_base_names=common_base_names,
            )

    @staticmethod
    def _concrete_class_object_union_member_names(
        annotation: ast.AST | None,
    ) -> tuple[str, ...]:
        if annotation is None:
            return ()
        union_members = _annotation_union_members(annotation)
        member_names = tuple(
            (
                member_name
                for member in union_members
                if (member_name := ConcreteTypeUnionContractCandidate._class_object_type_argument_name(member)) is not None
            )
        )
        if len(member_names) != len(union_members):
            return ()
        return tuple(dict.fromkeys(member_names))

    @staticmethod
    def _parameter_class_call_attribute_names(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        parameter_name: str,
    ) -> tuple[str, ...]:
        attribute_names: set[str] = set()

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == parameter_name
                ):
                    attribute_names.add(node.func.attr)
                self.generic_visit(node)

        Visitor().visit(
            ast.Module(body=statements_without_docstring(function.body), type_ignores=[])
        )
        return sorted_tuple(attribute_names)

    @staticmethod
    def _common_contract_base_names(
        classes_by_name: dict[str, ClassStatement],
        ancestor_names_by_class: dict[str, tuple[str, ...]],
        member_type_names: tuple[str, ...],
        observed_attribute_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        base_sets = tuple(
            (
                ConcreteTypeUnionContractCandidate._class_def_declared_and_ancestor_names(
                    classes_by_name, ancestor_names_by_class, member_type_name
                )
                for member_type_name in member_type_names
            )
        )
        common_base_names = {
            name
            for name in set.intersection(*base_sets)
            if ClassSymbolResolutionAuthority.establishes_nominal_family(name)
        }
        return sorted_tuple(
            (
                base_name
                for base_name in common_base_names
                if ConcreteTypeUnionContractCandidate._base_defines_contract_attributes(
                    classes_by_name, base_name, observed_attribute_names
                )
            )
        )

    @staticmethod
    def _concrete_union_contract_name(
        member_type_names: tuple[str, ...],
        observed_attribute_names: tuple[str, ...],
    ) -> str:
        edge_tokens = ConcreteTypeUnionContractCandidate._shared_edge_tokens(member_type_names)
        base_name = (
            "".join((token.title() for token in edge_tokens))
            if edge_tokens
            else f"{member_type_names[0]}Family"
        )
        role_suffix = (
            "Factory"
            if any(
                (
                    attribute_name.startswith(ConcreteTypeUnionContractCandidate._CONCRETE_UNION_FACTORY_ATTRIBUTE_PREFIXES)
                    for attribute_name in observed_attribute_names
                )
            )
            else "Contract"
        )
        return base_name if base_name.endswith(role_suffix) else f"{base_name}{role_suffix}"

    @staticmethod
    def _class_object_type_argument_name(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Subscript):
            return None
        root_name = HELPER_SYNTAX_PROJECTION_AUTHORITY.annotation_root_name(node.value)
        if not TypeWrapper.CLASS_OBJECT.spelled_by(root_name):
            return None
        return HELPER_SYNTAX_PROJECTION_AUTHORITY.annotation_root_name(node.slice)

    @staticmethod
    def _class_def_declared_and_ancestor_names(
        classes_by_name: dict[str, ClassStatement],
        ancestor_names_by_class: dict[str, tuple[str, ...]],
        class_name: str,
    ) -> set[str]:
        return set(classes_by_name[class_name].base_terminal_names) | set(
            ancestor_names_by_class[class_name]
        )

    @staticmethod
    def _base_defines_contract_attributes(
        classes_by_name: dict[str, ClassStatement],
        base_name: str,
        observed_attribute_names: tuple[str, ...],
    ) -> bool:
        base_class = classes_by_name.get(base_name)
        if base_class is None:
            return False
        return all(
            (
                base_class.method_named(attribute_name) is not None
                for attribute_name in observed_attribute_names
            )
        )

    @staticmethod
    def _shared_edge_tokens(member_type_names: tuple[str, ...]) -> tuple[str, ...]:
        token_rows = tuple(
            (
                CLASS_NAME_ALGEBRA.ordered_tokens(member_type_name)
                for member_type_name in member_type_names
            )
        )
        if not token_rows or not all(token_rows):
            return ()
        prefix_tokens: list[str] = []
        for column in zip(*token_rows, strict=False):
            if len(set(column)) != 1:
                break
            prefix_tokens.append(column[0])
        suffix_tokens: list[str] = []
        reversed_rows = tuple(tuple(reversed(row)) for row in token_rows)
        for column in zip(*reversed_rows, strict=False):
            if len(set(column)) != 1:
                break
            suffix_tokens.append(column[0])
        return (*prefix_tokens, *tuple(reversed(suffix_tokens)))


declare_candidate_rule_detector(
    ConcreteTypeUnionContractCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Concrete class-object union should be a shared nominal contract",
        "A function accepts a union of concrete classes, then uses the parameter as one constructor or class-level capability. That list of classes restates a shared contract.",
        "one shared ABC used as type[SharedContract], or a TypeVar bound to it",
        "parameter is annotated as a union of concrete classes while the body calls common class-level behavior on it",
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
)


@dataclass(frozen=True)
class CatalogInstallingMixinFamilyCandidate(CollectsItself, ClassLineNumbersGroup):
    catalog_attribute_names: tuple[str, ...]

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        items: list[tuple[str, str, int]] = []
        for class_node in (
            node for node in module.syntax_index.breadth_first_nodes if isinstance(node, ast.ClassDef)
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
            cls(
                file_path=module.file_path,
                class_names=tuple((item[0] for item in ordered)),
                catalog_attribute_names=tuple((item[1] for item in ordered)),
                line_numbers=tuple((item[2] for item in ordered)),
            ),
        )


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
    ) -> _CatalogInstallingMixinShape | None:
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


declare_candidate_rule_detector(
    AccumulatorFoldFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Accumulator folds should derive from one fold algebra",
        "Several methods create the same accumulator, feed one iterable through different step hooks, and return the same projection. That loop is a fold. Write it once.",
        "one accumulator-fold base with declared step hooks",
        "the same class repeats accumulator setup, loop and result projection and differs only in the step hook",
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
)


declare_candidate_rule_detector(
    CatalogInstallingMixinFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Catalog-installing mixins should share one subclass hook",
        "Several mixins repeat the same `__init_subclass__`: call `super()` and install one classvar catalog. Only the catalog attribute differs. Share the hook.",
        "one reusable catalog-installing subclass hook, with the catalog attribute as data",
        "sibling mixins repeat the same class-creation hook over different catalog classvars",
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
)


declare_candidate_rule_detector(
    RegexGroupExtractorFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Regex group extractor methods should derive from descriptors",
        "Several methods repeat `match = pattern.<mode>(text); return match.group(n) if match else None`. The pattern field and match mode are data. Write the extractor once as a descriptor or helper.",
        "one regex group descriptor with declared pattern and match mode",
        "the same class repeats regex group extractor methods over different pattern fields",
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
    metrics=lambda regex_candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(regex_candidate.method_names),
        mapping_name=regex_candidate.class_name,
        field_names=regex_candidate.pattern_attribute_names,
    ),
)


@dataclass(frozen=True)
class DynamicSelfFieldSelectionCandidate(CollectsItself, SelfReflectiveBuiltinCandidate):
    selector_expression: str

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
            module,
            ast.ClassDef,
            cls._for_class,
        )

    @classmethod
    def _for_class(cls, module: ParsedModule, node: ast.ClassDef) -> Iterable[Self]:
        for statement in node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for subnode in module.syntax_index.function_walks[statement]:
                if not isinstance(subnode, ast.Call):
                    continue
                match BuiltinCall.decoded(subnode):
                    case AttributeAccessCall() as call:
                        receiver, selector_node = call.object_argument, call.name_argument
                    case _:
                        continue
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
                yield cls(
                    file_path=module.file_path,
                    line=subnode.lineno,
                    subject_name=node.name,
                    name_family=(selector_expression,),
                    method_name=statement.name,
                    reflective_builtin=call.builtin,
                    selector_expression=selector_expression,
                )


declare_candidate_rule_detector(
    DynamicSelfFieldSelectionCandidate,
    high_confidence_spec(
        PatternId.CONFIG_CONTRACTS,
        "Dynamic self-field selection hides a nominal contract",
        "A class picks one of its own fields by name at runtime instead of declaring one fail-loud hook or one canonical field.",
        "declared count or value hook instead of lookup by field name",
        "class template picks its own state by dynamic field name",
        (
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    ),
    summary=lambda dynamic_candidate: (
        f"`{dynamic_candidate.class_name}.{dynamic_candidate.method_name}` uses `{dynamic_candidate.reflective_builtin}(self, {dynamic_candidate.selector_expression})` instead of one declared nominal value."
    ),
)


@dataclass(frozen=True)
class StringBackedReflectiveNominalLookupCandidate(CollectsItselfUnderConfig, ClassLineWitnessCandidate):
    method_name: str
    selector_attr_name: str
    lookup_kind: str
    receiver_expression: str
    concrete_class_names: tuple[str, ...]
    selector_values: tuple[str, ...]

    @classmethod
    def from_module(cls, module: ParsedModule, config: DetectorConfig) -> tuple[Self, ...]:
        statements = module.fact(ModuleClassStatements)
        classes_by_name = statements.by_name
        class_string_assignments = {
            class_name: cls._class_direct_constant_string_assignments(declared)
            for class_name, declared in classes_by_name.items()
        }
        candidate_map: dict[
            tuple[str, str, str, str, str], Self
        ] = {}
        for class_name, declared in sorted(classes_by_name.items()):
            descendants = statements.descendant_names(class_name)
            if len(descendants) < config.min_reflective_selector_values:
                continue
            for method in declared.methods:
                for subnode in ast.walk(method.node):
                    lookup_shape = cls._reflective_lookup_shape(subnode)
                    if lookup_shape is None:
                        continue
                    lookup_kind, receiver_expression, selector_node = lookup_shape
                    selector_attr_name = _selector_attribute_name(selector_node)
                    if selector_attr_name is None:
                        continue
                    concrete_class_names = tuple(
                        (
                            descendant
                            for descendant in descendants
                            if selector_attr_name in class_string_assignments[descendant]
                        )
                    )
                    if len(concrete_class_names) < config.min_reflective_selector_values:
                        continue
                    selector_values = sorted_tuple(
                        {
                            class_string_assignments[descendant][selector_attr_name]
                            for descendant in concrete_class_names
                        }
                    )
                    if len(selector_values) < config.min_reflective_selector_values:
                        continue
                    candidate = cls(
                        file_path=module.file_path,
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
        return sorted_tuple(
            candidate_map.values(),
            key=lambda item: (item.file_path, item.line, item.class_name, item.method_name),
        )

    @staticmethod
    def _class_direct_constant_string_assignments(
        declared: ClassStatement,
    ) -> dict[str, str]:
        return {
            name: string_value
            for name, value in declared.direct_assignments.items()
            if (string_value := _constant_string(value)) is not None
        }

    @staticmethod
    def _reflective_lookup_shape(
        node: ast.AST,
    ) -> tuple[str, str, ast.AST] | None:
        if isinstance(node, ast.Call):
            match BuiltinCall.decoded(node):
                case Getattr(name_argument=selector_node) as call if (
                    _constant_string(selector_node) is None
                ):
                    return (call.builtin, ast.unparse(call.object_argument), selector_node)
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
            and isinstance(namespace := BuiltinCall.decoded(node.value), NamespaceCall)
            and (_constant_string(node.slice) is None)
        ):
            return (f"{namespace.builtin}[]", f"{namespace.builtin}()", node.slice)
        return None


declare_candidate_rule_detector(
    StringBackedReflectiveNominalLookupCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "String-backed reflective lookup is simulating nominal identity",
        "A class family shouldn't pass behavior through string selectors and reflection. When subclasses only supply constant names resolved through globals, getattr or __dict__, use one declared hook or typed handle.",
        "declared hook or typed family handle instead of a string selector plus reflection",
        "class family encodes behavior as constant selector strings and resolves them by reflection",
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
)
