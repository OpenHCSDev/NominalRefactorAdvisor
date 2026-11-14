"""Systemic detector implementations.

This module holds the earlier detector families that focus on orchestration,
axis authority, registration, and other repo-wide architectural smells.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast, ClassVar, Generic, Self, TypeVar

from nominal_refactor_advisor.ast_projection import ExpressionReference, CallKeywords
from nominal_refactor_advisor.class_syntax_facets import (
    CompactClosedAxisBranchFunction,
    CompactNamedProjectionSurface,
    CompactSortedKeyCall,
    ClosedAxis,
    CompactClosedAxisBranchFact,
    CompactKeyedTableAxis,
)
from nominal_refactor_advisor.detectors._candidates import (
    CollectsItselfFromRegistryFacts,
    KeyedRegistryAxisFact,
    LineWitnessCandidate,
    WitnessCarrierCandidate,
)
from nominal_refactor_advisor.refactors.autoregister import (
    AutoRegisterExplicitPriorityOrderingFindingRecipeSynthesizer,
    InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.semantic_algebra import ObjectFamilyShape, DispatchAxisExpression
from nominal_refactor_advisor.semantic_description_length import CompressionCertificate
from nominal_refactor_advisor.ast_tools import (
    BuiltinCallName,
    CompactModuleIdentity,
    PythonSourcePathPolicy,
    SourceModule,
)
from nominal_refactor_advisor.class_index import (
    CompactClassFamilyIndex,
    CompactClassMappingKeyAuthority,
    CompactIndexedClass,
)
from nominal_refactor_advisor.refactors.candidate_collector import (
    CandidateCollectorBoilerplateFindingRecipeSynthesizer,
    DeclarativeDetectorClassFindingRecipeSynthesizer,
    DirectBuildFindingRendererFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    CasesFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.native_syntax import NativePythonSyntaxIndex
from nominal_refactor_advisor.registry_identity import (
    INHERITABLE_AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag

from nominal_refactor_advisor.ast_tools import (
    CollectedFamily,
    ParsedModule,
    statements_without_docstring,
)
from nominal_refactor_advisor.class_index import (
    build_compact_class_family_index,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.models import (
    DispatchCountMetrics,
    EvidenceLocated,
    LineWitness,
    HierarchyCandidateMetrics,
    MappingMetrics,
    OrchestrationMetrics,
    RefactorFinding,
    RegistrationMetrics,
    RepeatedMethodMetrics,
    SourceLineReference,
    SourceLocation,
    WitnessCarrierMetrics,
    SemanticFieldRole,
)
from nominal_refactor_advisor.name_algebra import CLASS_NAME_ALGEBRA, overlap_coefficient
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.semantic_match import as_ast, Maybe, name_id
from nominal_refactor_advisor.semantic_shape_algebra import InjectiveTypeRegistryProof
from nominal_refactor_advisor.detectors._base import (
    KeyedRecordInfrastructureCandidate,
    CollectsItself,
    SubscriptedCandidateType,
    ModuleCollectedLineWitnessCandidate,
    TargetClassEvidenceAnchored,
    TargetProjectionRequired,
    SYNTAX_PROJECTION_AUTHORITY,
    CandidateCollectorBoilerplateCandidate,
    ClassLineWitnessCandidate,
    ClassMethodLineWitnessCandidate,
    CompactClassIndexMultiProjectionDetector,
    CompactClassRepositoryCandidateDetector,
    CompactClassRepositoryContext,
    CompactClassRepositoryProjectionMixin,
    CompactMultiProjectionCandidateDetector,
    CompactProjectionCandidateDetector,
    CompactProjectionGroups,
    ConfiguredModuleCollectorCandidateDetector,
    DeclarativeDetectorClassCandidate,
    DetectorConfig,
    DirectBuildFindingRendererCandidate,
    FindingSpecConstructionCandidate,
    InjectiveTypeRegistryCandidate,
    IssueDetector,
    ManualFiberTagCandidate,
    ModuleCollectorCandidateDetector,
    NonInjectiveTypeRegistryCandidate,
    RegistryProjectionSurfaceEvidence,
    RegistryProjectionSurfaceKind,
    RepeatedKeyedFamilyCandidate,
    SemanticMirrorIssueDetector,
    SourceModuleCollectorCandidateDetector,
    TupleIndexSemanticOpacityCandidate,
    _KeyedFamilyAxisSpec,
    declare_candidate_rule_detector,
    high_confidence_certified_spec,
    high_confidence_spec,
    ConcreteCandidateDetectorShape,
    ZippedSourceLocationEvidenceProperty,
    CollectsItselfUnderConfig,
    PipelineAssemblyStage,
    ResultAssemblyPipelineFunction,
    _PIPELINE_RETURN_STAGE,
    CollectsItselfFromCompactContext,
    RegistryProjectionMaterialization,
    RegistryProjectionRole,
    KeyedFamilyAxisSpecsByKey,
    ResidualClosedAxisBranchingIdentity,
    CANDIDATE_COLLECTION_AUTHORITY,
    EnumProjectionTableCandidate,
    EvidenceLocationsWitnessCandidate,
    NameFamilyClassNamesMixin,
    NamedFunctionNode,
    SUPPORT_PROJECTION_AUTHORITY,
    SubjectNameFunctionNameMixin,
    ClassLineNumbersGroup,
    FieldFamilyCompressionSurface,
    ManualStructuralRecordMechanicsClassCandidate,
    WitnessCarrierClassCandidate,
    witness_carrier_class_candidates,
    DISPATCH_ALGEBRA_AUTHORITY,
    SelfCastAliasPartition,
    _call_uses_iteration_variable,
    _dict_expr_from_table_value,
    _enum_member_names_by_class,
    _flatten_union_member_type_names,
    _is_enum_projection_table_value,
    _returns_constructor_call,
    _self_cast_type_name,
)
from nominal_refactor_advisor.detectors._helpers import (
    _cli_argument_spec_fields,
    _dataclass_config_field_names,
    _finding_spec_build_boilerplate_candidates,
    _from_namespace_keyword_names,
    TupleIndexSemanticOpacityCandidateCollector,
    _enum_metadata_property_table,
    _enum_metadata_table_cases,
    _typed_candidate_cast_assignment,
    _result_assembly_pipeline_functions,
    _shared_pipeline_tail,
    _SCHEMA_ACCESSOR_MIN_METHODS,
    _STATIC_OBSERVATION_DETECTOR_BASE_NAMES,
    _schema_accessor_axis_system,
    _schema_accessor_family_certificate,
    _schema_accessor_method_row,
    _source_location_from_line_symbol_call,
    _static_minimum_evidence_count,
    _static_summary_expression,
    _typed_observation_collection_call,
)
from nominal_refactor_advisor.class_families.parsed_index import (
    ModuleClassStatements,
    ModuleIndexedClasses,
    UnindexedClassStatement,
    ClassStatement,
)
from nominal_refactor_advisor.detectors._substrate_support import (
    _camel_case,
    ModuleNamedValues,
    _is_upper_snake_identifier,
    ModuleNamedSequenceMap,
)
from nominal_refactor_advisor.detectors._finding_spec_defaults import (
    FindingSpecConstructionCandidateCollector,
)
from itertools import combinations
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.registration_facets import CompactManualSelectorAxis
from nominal_refactor_advisor.class_symbols import ClassSymbolResolutionAuthority


@dataclass(frozen=True)
class CompactConcreteTypeCaseFunctionFact(CompactModuleIdentity, LineWitness):
    """One function's `isinstance` checks of one subject, with the type names as written."""

    function_name: str
    subject_expression: str
    subject_role: str
    type_names_by_check: tuple[tuple[str, ...], ...]
    union_aliases: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def witness_name(self) -> str:
        return self.function_name


@dataclass(frozen=True)
class ConcreteTypeCaseFunctionCandidate:
    """One function's case analysis with its type names resolved against the class index."""

    fact: CompactConcreteTypeCaseFunctionFact
    concrete_class_names: tuple[str, ...]
    union_alias_names: tuple[str, ...]


@dataclass(frozen=True)
class RepeatedConcreteTypeCaseAnalysisCandidate(EvidenceLocated):
    """Functions of one module that each recover the same subject's concrete class."""

    functions: tuple[ConcreteTypeCaseFunctionCandidate, ...]
    concrete_class_names: tuple[str, ...]
    union_alias_names: tuple[str, ...]
    abstract_base_names: tuple[str, ...]

    @property
    def subject_role(self) -> str:
        return self.functions[0].fact.subject_role

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(function.fact.location for function in self.functions[:6])


@dataclass(frozen=True)
class CompactImplicitSelfMixinFact(SourceLineReference):
    qualname: str
    method_names: tuple[str, ...]
    method_lines: tuple[int, ...]
    cast_type_names: tuple[str, ...]
    accessed_attribute_names: tuple[str, ...]


@dataclass(frozen=True)
class ImplicitSelfContractMixinCandidate(EvidenceLocated):
    """A mixin whose methods cast `self` to reach attributes its concrete subclasses own."""

    fact: CompactImplicitSelfMixinFact
    mixin_name: str
    consumers: tuple[SourceLocation, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        fact = self.fact
        return (
            SourceLocation(fact.file_path, fact.line, self.mixin_name),
            *(
                SourceLocation(fact.file_path, line, f"{self.mixin_name}.{name}")
                for name, line in zip(fact.method_names, fact.method_lines, strict=True)
            ),
            *self.consumers,
        )[:6]


@dataclass(frozen=True)
class CompactRemainingSystemicModuleProjection(CompactModuleIdentity):
    concrete_type_functions: tuple[CompactConcreteTypeCaseFunctionFact, ...]
    implicit_self_mixins: tuple[CompactImplicitSelfMixinFact, ...]


def _top_level_attribute_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for statement in statements_without_docstring(list(function.body)):
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or (not isinstance(statement.targets[0], ast.Name))
        ):
            continue
        value_expression = SUPPORT_PROJECTION_AUTHORITY.simple_name_or_attr_expression(
            statement.value
        )
        if value_expression is None or "." not in value_expression:
            continue
        aliases[statement.targets[0].id] = value_expression
    return aliases


def _module_union_type_aliases(
    module: ParsedModule,
) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for statement in module.module.body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or (not isinstance(statement.targets[0], ast.Name))
        ):
            continue
        member_names = sorted_tuple(
            set(_flatten_union_member_type_names(statement.value))
        )
        if len(member_names) < 2:
            continue
        aliases[statement.targets[0].id] = member_names
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
    subject_expression = SUPPORT_PROJECTION_AUTHORITY.simple_name_or_attr_expression(
        node
    )
    if subject_expression is None or "." not in subject_expression:
        return None
    return subject_expression


def _compact_concrete_type_case_function_facts(
    module: ParsedModule,
) -> tuple[CompactConcreteTypeCaseFunctionFact, ...]:
    union_aliases = tuple(sorted(_module_union_type_aliases(module).items()))
    facts: list[CompactConcreteTypeCaseFunctionFact] = []
    for qualname, function in module.syntax_index.named_functions:
        alias_sources = _top_level_attribute_aliases(function)
        grouped_checks: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for subnode in module.syntax_index.function_walks[function]:
            isinstance_arguments = BuiltinCallName.isinstance_arguments(subnode)
            if isinstance_arguments is None:
                continue
            subject, type_node = isinstance_arguments
            subject_expression = _attribute_family_subject_expression(
                subject, alias_sources=alias_sources
            )
            if subject_expression is None:
                continue
            type_items = (
                type_node.elts if isinstance(type_node, ast.Tuple) else (type_node,)
            )
            type_names = sorted_tuple(
                {
                    type_name
                    for item in type_items
                    if (type_name := ExpressionReference.decoded(item).terminal_name)
                    not in {None, "None", "NoneType"}
                }
            )
            if type_names:
                grouped_checks[subject_expression].append(type_names)
        for subject_expression, checks in sorted(grouped_checks.items()):
            facts.append(
                CompactConcreteTypeCaseFunctionFact(
                    file_path=module.file_path,
                    module_name=module.module_name,
                    line=function.lineno,
                    function_name=qualname,
                    subject_expression=subject_expression,
                    subject_role=subject_expression.rsplit(".", 1)[-1],
                    type_names_by_check=tuple(checks),
                    union_aliases=union_aliases,
                )
            )
    return sorted_tuple(
        facts,
        key=lambda item: (item.file_path, item.subject_role, item.line),
    )


def _self_cast_alias_names(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> SelfCastAliasPartition:
    aliases: set[str] = set()
    cast_type_names: set[str] = set()
    for statement in ast.walk(method):
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or (not isinstance(statement.targets[0], ast.Name))
        ):
            continue
        cast_type_name = _self_cast_type_name(statement.value)
        if cast_type_name is None:
            continue
        aliases.add(statement.targets[0].id)
        cast_type_names.add(cast_type_name)
    return (sorted_tuple(aliases), sorted_tuple(cast_type_names))


def _compact_implicit_self_mixin_facts(
    module: ParsedModule,
) -> tuple[CompactImplicitSelfMixinFact, ...]:
    facts: list[CompactImplicitSelfMixinFact] = []
    for indexed_class in module.fact(ModuleIndexedClasses).records:
        qualname, class_node = indexed_class.qualname, indexed_class.node
        if not class_node.name.endswith("Mixin") or indexed_class.is_abstract:
            continue
        method_names: list[str] = []
        method_lines: list[int] = []
        cast_type_names: set[str] = set()
        accessed_attribute_names: set[str] = set()
        for method in indexed_class.methods:
            if method.spells("abstractmethod"):
                continue
            alias_names, method_cast_types = _self_cast_alias_names(method.node)
            if not alias_names:
                continue
            method_names.append(method.name)
            method_lines.append(method.node.lineno)
            cast_type_names.update(method_cast_types)
            accessed_attribute_names.update(
                SYNTAX_PROJECTION_AUTHORITY.attribute_names_for_roots(
                    method.node, root_names=set(alias_names)
                )
            )
        if method_names:
            facts.append(
                CompactImplicitSelfMixinFact(
                    file_path=module.file_path,
                    qualname=qualname,
                    line=class_node.lineno,
                    method_names=tuple(method_names),
                    method_lines=tuple(method_lines),
                    cast_type_names=sorted_tuple(cast_type_names),
                    accessed_attribute_names=sorted_tuple(accessed_attribute_names),
                )
            )
    return tuple(facts)


class CompactRemainingSystemicModuleProjectionFamily(
    CollectedFamily[CompactRemainingSystemicModuleProjection]
):
    cache_payload_max_bytes = 1_000_000

    @classmethod
    def collect(
        cls, parsed_module: ParsedModule
    ) -> list[CompactRemainingSystemicModuleProjection]:
        del cls
        return [
            CompactRemainingSystemicModuleProjection(
                file_path=parsed_module.file_path,
                module_name=parsed_module.module_name,
                concrete_type_functions=_compact_concrete_type_case_function_facts(
                    parsed_module
                ),
                implicit_self_mixins=_compact_implicit_self_mixin_facts(parsed_module),
            )
        ]


def _compact_class_for_detector_name(
    class_index: CompactClassFamilyIndex,
    *,
    module_name: str,
    class_name: str,
) -> CompactIndexedClass | None:
    local_class = class_index.class_for(f"{module_name}.{class_name}")
    if local_class is not None:
        return local_class
    symbols = class_index.symbols_by_simple_name.get(class_name, ())
    if len(symbols) != 1:
        return None
    return class_index.class_for(symbols[0])


def _compact_concrete_type_function_candidates(
    projection: CompactRemainingSystemicModuleProjection,
    class_index: CompactClassFamilyIndex,
) -> tuple[ConcreteTypeCaseFunctionCandidate, ...]:
    candidates: list[ConcreteTypeCaseFunctionCandidate] = []
    for fact in projection.concrete_type_functions:
        concrete_class_names = sorted_tuple(
            {
                class_index.display_name(indexed_class)
                for type_names in fact.type_names_by_check
                for type_name in type_names
                if (
                    indexed_class := _compact_class_for_detector_name(
                        class_index,
                        module_name=fact.module_name,
                        class_name=type_name,
                    )
                )
                is not None
                and not indexed_class.is_abstract
            }
        )
        if len(concrete_class_names) < 2:
            continue
        union_alias_names = sorted_tuple(
            alias_name
            for alias_name, member_names in fact.union_aliases
            if set(concrete_class_names) <= set(member_names)
        )
        candidates.append(
            ConcreteTypeCaseFunctionCandidate(
                fact=fact,
                concrete_class_names=concrete_class_names,
                union_alias_names=union_alias_names,
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.fact.file_path, item.fact.subject_role, item.fact.line),
    )


def _compact_common_abstract_base_names(
    projection: CompactRemainingSystemicModuleProjection,
    class_index: CompactClassFamilyIndex,
    class_names: tuple[str, ...],
) -> tuple[str, ...]:
    indexed_classes = tuple(
        indexed_class
        for class_name in class_names
        if (
            indexed_class := _compact_class_for_detector_name(
                class_index,
                module_name=projection.module_name,
                class_name=class_name,
            )
        )
        is not None
    )
    if len(indexed_classes) < 2:
        return ()
    common_symbols = set(class_index.ancestor_symbols(indexed_classes[0].symbol))
    for indexed_class in indexed_classes[1:]:
        common_symbols &= set(class_index.ancestor_symbols(indexed_class.symbol))
    return sorted_tuple(
        class_index.display_name(indexed_class)
        for symbol in sorted(common_symbols)
        if (indexed_class := class_index.class_for(symbol)) is not None
        and indexed_class.is_abstract
    )


def _compact_repeated_concrete_type_case_candidates(
    projections: Sequence[CompactRemainingSystemicModuleProjection],
    class_projections: Sequence[CompactModuleClassProjection],
    config: DetectorConfig,
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> tuple[RepeatedConcreteTypeCaseAnalysisCandidate, ...]:
    if class_index is None:
        class_index = build_compact_class_family_index(class_projections)
    min_function_count = max(3, config.min_registration_sites)
    min_class_count = max(2, config.min_reflective_selector_values)
    candidates: list[RepeatedConcreteTypeCaseAnalysisCandidate] = []
    for projection in projections:
        grouped: dict[str, list[ConcreteTypeCaseFunctionCandidate]] = defaultdict(list)
        for function_candidate in _compact_concrete_type_function_candidates(
            projection, class_index
        ):
            grouped[function_candidate.fact.subject_role].append(function_candidate)
        for subject_role, functions in sorted(grouped.items()):
            if len(functions) < min_function_count:
                continue
            concrete_class_names = sorted_tuple(
                class_name
                for function in functions
                for class_name in function.concrete_class_names
            )
            concrete_class_names = sorted_tuple(set(concrete_class_names))
            if len(concrete_class_names) < min_class_count:
                continue
            abstract_base_names = _compact_common_abstract_base_names(
                projection, class_index, concrete_class_names
            )
            union_alias_names = sorted_tuple(
                {
                    alias_name
                    for function in functions
                    for alias_name in function.union_alias_names
                }
            )
            shared_suffix = CLASS_NAME_ALGEBRA.longest_common_suffix(
                concrete_class_names
            )
            shared_prefix = CLASS_NAME_ALGEBRA.longest_common_prefix(
                concrete_class_names
            )
            if (
                not abstract_base_names
                and not union_alias_names
                and max(len(shared_suffix), len(shared_prefix)) < 6
            ):
                continue
            candidates.append(
                RepeatedConcreteTypeCaseAnalysisCandidate(
                    functions=sorted_tuple(
                        functions,
                        key=lambda item: (item.fact.line, item.fact.function_name),
                    ),
                    concrete_class_names=concrete_class_names,
                    union_alias_names=union_alias_names,
                    abstract_base_names=abstract_base_names,
                )
            )
    return tuple(candidates)


def _compact_implicit_self_contract_mixin_candidates(
    projections: Sequence[CompactRemainingSystemicModuleProjection],
    class_projections: Sequence[CompactModuleClassProjection],
    config: DetectorConfig,
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> tuple[ImplicitSelfContractMixinCandidate, ...]:
    if class_index is None:
        class_index = build_compact_class_family_index(class_projections)
    min_consumer_count = max(2, config.min_registration_sites)
    facts_by_class_symbol = {
        class_symbol: fact
        for projection in projections
        for fact in projection.implicit_self_mixins
        if (
            class_symbol := class_index.symbol_for(
                file_path=fact.file_path, qualname=fact.qualname
            )
        )
        is not None
    }
    candidates: list[ImplicitSelfContractMixinCandidate] = []
    for class_symbol, fact in sorted(facts_by_class_symbol.items()):
        indexed_class = class_index.class_for(class_symbol)
        if indexed_class is None:
            continue
        consumers = tuple(
            descendant
            for descendant_symbol in class_index.descendant_symbols(class_symbol)
            if (descendant := class_index.class_for(descendant_symbol)) is not None
            and not descendant.is_abstract
        )
        if len(consumers) < min_consumer_count:
            continue
        candidates.append(
            ImplicitSelfContractMixinCandidate(
                fact=fact,
                mixin_name=class_index.display_name(indexed_class),
                consumers=sorted_tuple(
                    (
                        SourceLocation(
                            consumer.file_path,
                            consumer.line,
                            class_index.display_name(consumer),
                        )
                        for consumer in consumers
                    ),
                    key=lambda consumer: consumer.symbol,
                ),
            )
        )
    return tuple(candidates)


@dataclass(frozen=True)
class _DataclassNamespaceProjection(LineWitness):
    """A dataclass and the fields its `from_namespace` constructor reads back by name."""

    class_name: str
    field_names: tuple[str, ...]
    namespace_field_names: tuple[str, ...]
    from_namespace_line: int

    @property
    def witness_name(self) -> str:
        return self.class_name

    @property
    def from_namespace_location(self) -> SourceLocation:
        return SourceLocation(
            self.file_path, self.from_namespace_line, f"{self.class_name}.from_namespace"
        )


@dataclass(frozen=True)
class _CliArgumentSpecProjection(LineWitness):
    """A table of CLI argument specs and the fields its flags name."""

    name: str
    field_names: tuple[str, ...]

    @property
    def witness_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class DataclassNamespaceCliMirrorCandidate(EvidenceLocated):
    """A dataclass whose fields a `from_namespace` constructor and a CLI argument table
    both list again.  ``field_names`` are the ones the constructor reads, and
    ``cli_field_names`` the ones of those the table lists."""

    config_class: _DataclassNamespaceProjection
    argument_spec: _CliArgumentSpecProjection
    field_names: tuple[str, ...]
    cli_field_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            self.config_class.location,
            self.config_class.from_namespace_location,
            self.argument_spec.location,
        )


@dataclass(frozen=True)
class _DataclassNamespaceCliModuleProjection:
    dataclasses: tuple[_DataclassNamespaceProjection, ...]
    cli_specs: tuple[_CliArgumentSpecProjection, ...]


class _DataclassNamespaceCliModuleProjectionFamily(
    CollectedFamily[_DataclassNamespaceCliModuleProjection]
):
    report_presence_predicate = staticmethod(
        lambda items, config: any(
            item.dataclasses or item.cli_specs
            for item in items
            if isinstance(item, _DataclassNamespaceCliModuleProjection)
        )
    )
    source_collector = staticmethod(
        lambda source_module, syntax_index: _native_dataclass_namespace_cli_projections(
            source_module,
            syntax_index,
        )
    )

    @classmethod
    def collect(
        cls, parsed_module: ParsedModule
    ) -> list[_DataclassNamespaceCliModuleProjection]:
        del cls
        file_path = parsed_module.file_path
        dataclasses: list[_DataclassNamespaceProjection] = []
        for declared in parsed_module.fact(ModuleClassStatements).top_level:
            node = declared.node
            field_names = _dataclass_config_field_names(declared)
            if not field_names:
                continue
            namespace_assignment = _from_namespace_keyword_names(node)
            if namespace_assignment is None:
                continue
            from_namespace_line, namespace_field_names = namespace_assignment
            dataclasses.append(
                _DataclassNamespaceProjection(
                    file_path=file_path,
                    line=node.lineno,
                    class_name=node.name,
                    field_names=field_names,
                    namespace_field_names=namespace_field_names,
                    from_namespace_line=from_namespace_line,
                )
            )
        cli_specs = tuple(
            _CliArgumentSpecProjection(
                file_path=file_path,
                name=name,
                line=line,
                field_names=field_names,
            )
            for name, line, field_names in _cli_argument_spec_fields(parsed_module)
        )
        if not dataclasses and not cli_specs:
            return []
        return [
            _DataclassNamespaceCliModuleProjection(
                dataclasses=tuple(dataclasses),
                cli_specs=cli_specs,
            )
        ]


def _native_dataclass_namespace_cli_projections(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[_DataclassNamespaceCliModuleProjection] | None:
    """Project sparse dataclass/CLI mirrors from selected declarations."""

    if not syntax_index.is_complete:
        return None
    file_path = source_module.file_path
    try:
        dataclasses: list[_DataclassNamespaceProjection] = []
        for class_node in syntax_index.top_level_declarations("class"):
            class_source = syntax_index.source_for(class_node)
            decorated_source = (
                syntax_index.source_for(class_node.parent)
                if class_node.parent is not None
                and class_node.parent.type == "decorated_definition"
                else class_source
            )
            if b"dataclass" not in decorated_source or b"from_namespace" not in (
                class_source
            ):
                continue
            parsed_class = syntax_index.class_for(class_node)
            field_names = _dataclass_config_field_names(
                UnindexedClassStatement(parsed_class)
            )
            if not field_names:
                continue
            namespace_assignment = _from_namespace_keyword_names(parsed_class)
            if namespace_assignment is None:
                continue
            from_namespace_line, namespace_field_names = namespace_assignment
            dataclasses.append(
                _DataclassNamespaceProjection(
                    file_path=file_path,
                    line=parsed_class.lineno,
                    class_name=parsed_class.name,
                    field_names=field_names,
                    namespace_field_names=namespace_field_names,
                    from_namespace_line=from_namespace_line,
                )
            )
        cli_statements = tuple(
            syntax_index.statement_for(node)
            for node in syntax_index.top_level_assignment_statements()
            if b"ArgumentSpec" in syntax_index.source_for(node)
        )
        cli_module = source_module.parsed_module(
            ast.Module(body=list(cli_statements), type_ignores=[]),
        )
        cli_specs = tuple(
            _CliArgumentSpecProjection(
                file_path=file_path,
                name=name,
                line=line,
                field_names=field_names,
            )
            for name, line, field_names in _cli_argument_spec_fields(cli_module)
        )
        if not dataclasses and not cli_specs:
            return []
        return [
            _DataclassNamespaceCliModuleProjection(
                dataclasses=tuple(dataclasses),
                cli_specs=cli_specs,
            )
        ]
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


def _dataclass_namespace_cli_mirror_candidates_from_projections(
    projections: Sequence[_DataclassNamespaceCliModuleProjection],
) -> tuple[DataclassNamespaceCliMirrorCandidate, ...]:
    cli_specs = tuple(
        cli_spec for projection in projections for cli_spec in projection.cli_specs
    )
    candidates: list[DataclassNamespaceCliMirrorCandidate] = []
    for projection in projections:
        for dataclass_projection in projection.dataclasses:
            mirrored_fields = tuple(
                name
                for name in dataclass_projection.namespace_field_names
                if name in dataclass_projection.field_names
            )
            if len(mirrored_fields) < 4:
                continue
            for cli_spec in cli_specs:
                mirrored_cli_fields = tuple(
                    name
                    for name in cli_spec.field_names
                    if name in dataclass_projection.field_names
                )
                shared_fields = tuple(
                    name for name in mirrored_fields if name in mirrored_cli_fields
                )
                if len(shared_fields) < 4:
                    continue
                candidates.append(
                    DataclassNamespaceCliMirrorCandidate(
                        config_class=dataclass_projection,
                        argument_spec=cli_spec,
                        field_names=mirrored_fields,
                        cli_field_names=mirrored_cli_fields,
                    )
                )
    return tuple(candidates)


_SINGLE_TEMPLATE_CALL_METRICS = OrchestrationMetrics(
    function_line_count=0,
    branch_site_count=0,
    call_site_count=1,
    parameter_count=1,
    callee_family_count=1,
)


@dataclass(frozen=True)
class ProjectionPropertyFamilyCandidate(CollectsItself, ClassLineWitnessCandidate):
    property_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    base_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(
                    self.file_path, line, f"{self.class_name}.{property_name}"
                )
                for line, property_name in zip(
                    self.line_numbers, self.property_names, strict=True
                )
            )
        )

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        candidates: list[Self] = []
        for declared in module.fact(ModuleClassStatements).statements:
            class_node = declared.node
            properties: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]] = []
            for method in declared.methods:
                if not method.spells("property"):
                    continue
                statement = method.node
                body = statements_without_docstring(statement.body)
                if len(body) != 1 or not isinstance(body[0], ast.Return):
                    continue
                base_name = _path_projection_base(body[0].value)
                if base_name is None:
                    continue
                properties.append((statement, base_name))
            if len(properties) < 3:
                continue
            ordered = sorted_tuple(properties, key=lambda item: item[0].lineno)
            candidates.append(
                cls(
                    file_path=module.file_path,
                    line=class_node.lineno,
                    class_name=class_node.name,
                    property_names=tuple((function.name for function, _ in ordered)),
                    line_numbers=tuple((function.lineno for function, _ in ordered)),
                    base_names=sorted_tuple({base_name for _, base_name in ordered}),
                )
            )
        return sorted_tuple(
            candidates, key=lambda item: (item.file_path, item.line, item.class_name)
        )


@dataclass(frozen=True)
class SelfAttributeAuthority:
    def attr_name(self, node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (node.value.id == "self")
        ):
            return node.attr
        return None


SELF_ATTRIBUTE_AUTHORITY = SelfAttributeAuthority()


def _is_path_projection_part(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):
        return all(
            (
                isinstance(value, ast.Constant)
                or (
                    isinstance(value, ast.FormattedValue)
                    and SELF_ATTRIBUTE_AUTHORITY.attr_name(value.value) is not None
                )
                for value in node.values
            )
        )
    return False


def _path_projection_base(returned: ast.AST) -> str | None:
    node = returned
    saw_path_part = False
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if not _is_path_projection_part(node.right):
            return None
        saw_path_part = True
        node = node.left
    if not saw_path_part:
        return None
    return SELF_ATTRIBUTE_AUTHORITY.attr_name(node)


declare_candidate_rule_detector(
    ProjectionPropertyFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.DESCRIPTOR_DERIVED_VIEW,
        "Path projection properties should be derived descriptors",
        "Several properties build Path views from owned base fields with the same `/` expression. Only the suffixes differ. Make the suffixes data and write the projection once as a descriptor.",
        "one descriptor for repeated Path projection properties",
        "the same class repeats Path projection properties over owned base fields",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.PROJECTION_HELPER,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    ),
    summary=lambda projection_candidate: (
        f"`{projection_candidate.class_name}` repeats Path projection properties {', '.join(projection_candidate.property_names)} over bases {', '.join(projection_candidate.base_names)}."
    ),
    metrics=lambda projection_candidate: MappingMetrics(
        mapping_site_count=len(projection_candidate.property_names),
        field_count=len(projection_candidate.base_names),
        mapping_name=f"{projection_candidate.class_name} path projection",
        field_names=projection_candidate.property_names,
        source_name=", ".join(projection_candidate.base_names),
    ),
)


def _collection_projection_property_shape(
    returned: ast.AST,
) -> tuple[str, str] | None:
    return (
        Maybe.of(as_ast(returned, ast.Call))
        .filter(
            lambda call: (
                name_id(call.func) in {"tuple", "list", "set", "frozenset"}
                and len(call.args) == 1
                and not call.keywords
            )
        )
        .project(lambda call: as_ast(call.args[0], ast.GeneratorExp))
        .filter(lambda generator: len(generator.generators) == 1)
        .map(lambda generator: (generator, generator.generators[0]))
        .filter(lambda context: not context[1].ifs)
        .combine(
            lambda context: SELF_ATTRIBUTE_AUTHORITY.attr_name(context[1].iter),
            lambda context, collection_name: (
                collection_name,
                as_ast(context[0].elt, ast.Attribute),
                name_id(context[1].target),
            ),
        )
        .project(
            lambda context: (
                (
                    context[0],
                    context[1].attr,
                )
                if context[1] is not None
                and context[2] is not None
                and name_id(context[1].value) == context[2]
                else None
            )
        )
        .unwrap_or_none()
    )


@dataclass(frozen=True)
class CollectionProjectionPropertyFamilyCandidate(CollectsItself, ClassLineWitnessCandidate):
    property_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    collection_name: str
    projected_attribute_names: tuple[str, ...]
    line_count: int
    evidence: ClassVar[ZippedSourceLocationEvidenceProperty] = (
        ZippedSourceLocationEvidenceProperty("line_numbers", "property_names")
    )

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        candidates: list[Self] = []
        for declared in module.fact(ModuleClassStatements).statements:
            class_node = declared.node
            properties: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str, str]] = []
            for method in declared.methods:
                if not method.spells("property"):
                    continue
                statement = method.node
                body = statements_without_docstring(statement.body)
                if len(body) != 1 or not isinstance(body[0], ast.Return):
                    continue
                shape = _collection_projection_property_shape(body[0].value)
                if shape is None:
                    continue
                collection_name, projected_attribute_name = shape
                properties.append((statement, collection_name, projected_attribute_name))
            grouped: dict[str, list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]]] = (
                defaultdict(list)
            )
            for statement, collection_name, projected_attribute_name in properties:
                grouped[collection_name].append((statement, projected_attribute_name))
            for collection_name, grouped_properties in grouped.items():
                if len(grouped_properties) < 2:
                    continue
                ordered = sorted_tuple(grouped_properties, key=lambda item: item[0].lineno)
                candidates.append(
                    cls(
                        file_path=module.file_path,
                        line=class_node.lineno,
                        class_name=class_node.name,
                        property_names=tuple((statement.name for statement, _ in ordered)),
                        line_numbers=tuple((statement.lineno for statement, _ in ordered)),
                        collection_name=collection_name,
                        projected_attribute_names=tuple(
                            (attribute_name for _, attribute_name in ordered)
                        ),
                        line_count=(class_node.end_lineno or class_node.lineno)
                        - class_node.lineno
                        + 1,
                    )
                )
        return sorted_tuple(
            candidates, key=lambda item: (item.file_path, item.line, item.class_name)
        )


declare_candidate_rule_detector(
    CollectionProjectionPropertyFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.DESCRIPTOR_DERIVED_VIEW,
        "Collection projection properties should be derived descriptors",
        "Sibling properties that only map one owned collection to member attributes are descriptor-shaped. Repeating `tuple(item.attr for item in self.collection)` per property makes each attribute look like behavior. The real thing is the collection projection.",
        "one collection-projection descriptor parameterized by collection and member attribute",
        "the same class repeats collection projection properties over one owned collection",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.PROJECTION_HELPER,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` repeats collection projection properties "
        f"{candidate.property_names} over `self.{candidate.collection_name}` "
        f"for member attributes {candidate.projected_attribute_names}."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.property_names),
        mapping_name=f"{candidate.class_name}.{candidate.collection_name}",
        field_names=candidate.projected_attribute_names,
    ),
)


@dataclass(frozen=True)
class ResidualClosedAxisIndirectionCandidate(CollectsItself, 
    DispatchAxisExpression, LineWitnessCandidate
):
    qualname: str
    table_name: str
    table_line: int
    enum_name: str
    table_case_names: tuple[str, ...]
    residual_case_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(self.file_path, self.table_line, self.table_name),
            self.named(self.qualname),
        )

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        tables = cls._enum_projection_tables(module)
        if not tables:
            return ()
        table_by_name = {table.table_name: table for table in tables}
        return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
            module,
            cls._for_function,
            table_by_name,
            sort_key=lambda item: (
                item.file_path,
                item.line,
                item.qualname,
                item.table_name,
            ),
        )

    @classmethod
    def _for_function(
        cls,
        module: ParsedModule,
        qualname: str,
        function: NamedFunctionNode,
        table_by_name: dict[str, EnumProjectionTableCandidate],
    ) -> Iterable[Self]:
        axis_expressions_by_table: dict[str, set[str]] = defaultdict(set)
        for node in ast.walk(function):
            for table_name in table_by_name:
                dispatch_axis_expression = cls._subscript_axis_expr_for_table(node, table_name)
                if dispatch_axis_expression is not None:
                    axis_expressions_by_table[table_name].add(dispatch_axis_expression)
        for table_name, axis_expressions in axis_expressions_by_table.items():
            table = table_by_name[table_name]
            for dispatch_axis_expression in sorted(axis_expressions):
                residual_cases = cls._residual_enum_branch_cases(
                    function,
                    enum_name=table.enum_name,
                    dispatch_axis_expression=dispatch_axis_expression,
                )
                shared_cases = tuple(
                    case_name
                    for case_name in table.case_names
                    if case_name in set(residual_cases)
                )
                if not shared_cases:
                    continue
                yield cls(
                    file_path=module.file_path,
                    qualname=qualname,
                    line=function.lineno,
                    table_name=table.table_name,
                    table_line=table.line,
                    enum_name=table.enum_name,
                    dispatch_axis_expression=dispatch_axis_expression,
                    table_case_names=table.case_names,
                    residual_case_names=shared_cases,
                )

    @staticmethod
    def _enum_projection_tables(
        module: ParsedModule,
    ) -> tuple[EnumProjectionTableCandidate, ...]:
        enum_members = _enum_member_names_by_class(module)
        tables: list[EnumProjectionTableCandidate] = []
        for statement in statements_without_docstring(module.module.body):
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
            if target_name is None:
                continue
            dict_value = _dict_expr_from_table_value(value)
            if dict_value is None or len(dict_value.keys) < 2:
                continue
            key_pairs: list[tuple[str, str]] = []
            for key, item_value in zip(dict_value.keys, dict_value.values, strict=False):
                if key is None or item_value is None:
                    break
                key_chain = ExpressionReference.decoded(key).attribute_chain
                if key_chain is None or len(key_chain) != 2:
                    break
                if not _is_enum_projection_table_value(item_value):
                    break
                key_pairs.append((key_chain[0], key_chain[1]))
            else:
                enum_names = {enum_name for enum_name, _ in key_pairs}
                if len(enum_names) != 1:
                    continue
                enum_name = next(iter(enum_names))
                if enum_name not in enum_members:
                    continue
                case_names = tuple(member_name for _, member_name in key_pairs)
                if len(set(case_names)) < 2:
                    continue
                tables.append(
                    EnumProjectionTableCandidate(
                        file_path=module.file_path,
                        table_name=target_name,
                        line=statement.lineno,
                        enum_name=enum_name,
                        case_names=case_names,
                    )
                )
        return sorted_tuple(
            tables, key=lambda item: (item.file_path, item.line, item.table_name)
        )

    @staticmethod
    def _residual_enum_branch_cases(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        enum_name: str,
        dispatch_axis_expression: str,
    ) -> tuple[str, ...]:
        case_names: set[str] = set()
        for node in ast.walk(function):
            if not isinstance(node, ast.Compare):
                continue
            left_expr = ast.unparse(node.left)
            comparators = tuple(ast.unparse(comparator) for comparator in node.comparators)
            operands = (left_expr, *comparators)
            if dispatch_axis_expression not in operands:
                continue
            for operand in operands:
                if operand.startswith(f"{enum_name}."):
                    case_names.add(operand.split(".", 1)[1])
        return sorted_tuple(case_names)

    @staticmethod
    def _subscript_axis_expr_for_table(node: ast.AST, table_name: str) -> str | None:
        if not isinstance(node, ast.Subscript):
            return None
        if not isinstance(node.value, ast.Name) or node.value.id != table_name:
            return None
        return ast.unparse(node.slice)


class ResidualClosedAxisIndirectionDetector(
    ModuleCollectorCandidateDetector[ResidualClosedAxisIndirectionCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Enum-keyed table plus leftover branches suggest scattered case ownership",
        "A function indexes an enum-keyed table and also branches on the same enum. If the cases carry behavior, prefer a public ABC: shared algorithms on the parent, per-case declarations, data and hooks on concrete subclasses. Derive consumers and any registration from that family instead of keeping another enum-to-handler list. Compose independent capabilities through inheritance only after checking MRO, construction and substitutability. If the sites only encode values or an outside vocabulary, show that role is distinct before proposing a hierarchy. This observation does not prove case binding, callable effects, full coverage or rewrite equivalence. Migrate in stages and keep unmatched cases, branch priority and unknown-input behavior.",
        "class-owned polymorphic behavior instead of separately maintained tables and case branches, if the task rules allow it",
        "the same function indexes an enum-keyed table and branches on that enum",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.PROJECTION_DICT,
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.CLOSED_FAMILY_CASES,
        ),
    )

    def _finding_for_candidate(
        self, axis_candidate: ResidualClosedAxisIndirectionCandidate
    ) -> RefactorFinding:
        residual_cases = ", ".join(axis_candidate.residual_case_names)
        table_cases = ", ".join(axis_candidate.table_case_names)
        return self.build_finding(
            (
                f"`{axis_candidate.qualname}` indexes `{axis_candidate.table_name}` by "
                f"`{axis_candidate.dispatch_axis_expression}` for `{axis_candidate.enum_name}` cases {table_cases}, "
                f"but still branches on residual cases {residual_cases}."
            ),
            axis_candidate.evidence,
            metrics=DispatchCountMetrics.from_literal_family(
                dispatch_axis=axis_candidate.enum_name,
                literal_cases=axis_candidate.table_case_names,
            ),
        )


@dataclass(frozen=True)
class DerivedWrapperSpecShadowCandidate(CollectsItself, EvidenceLocationsWitnessCandidate):
    derived_family_name: str
    derived_constructor_name: str
    primary_family_name: str | None
    primary_constructor_name: str
    link_field_name: str
    primary_constant_names: tuple[str, ...]
    extra_field_names: tuple[str, ...]
    builder_names: tuple[str, ...]
    witness_name = AliasProperty[str]("derived_family_name")

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        named_values = module.fact(ModuleNamedValues)
        constant_bindings = named_values.constants
        named_sequences = named_values.sequences
        candidates: list[Self] = []
        for family_name, (family_line, elements) in sorted(named_sequences.items()):
            if len(elements) < 2 or not all(
                (isinstance(element, ast.Call) for element in elements)
            ):
                continue
            entry_calls = cast(tuple[ast.Call, ...], elements)
            constructor_names = {
                ExpressionReference.decoded(element.func).terminal_name
                for element in entry_calls
            }
            if len(constructor_names) != 1 or None in constructor_names:
                continue
            keyword_maps: list[dict[str, ast.AST]] = []
            for element in entry_calls:
                keyword_map = dict(CallKeywords.of(element.keywords).by_name)
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
            builder_names = cls._comprehension_builder_names(module, family_name)
            if not builder_names:
                continue
            for link_field_name in sorted(common_keyword_names):
                referenced_constant_names: list[str] = []
                for keyword_map in keyword_maps:
                    referenced = keyword_map[link_field_name]
                    if not isinstance(
                        referenced, ast.Name
                    ) or not _is_upper_snake_identifier(referenced.id):
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
                primary_family_name = cls._named_family_for_constants(
                    named_sequences, primary_constant_names
                )
                extra_field_names = sorted_tuple(
                    (name for name in common_keyword_names if name != link_field_name)
                )
                evidence: list[SourceLocation] = [
                    SourceLocation(module.file_path, family_line, family_name)
                ]
                evidence.extend(
                    (
                        SourceLocation(module.file_path, constant_bindings[name].line, name)
                        for name in primary_constant_names[:3]
                        if name in constant_bindings
                    )
                )
                candidates.append(
                    cls(
                        file_path=module.file_path,
                        line=family_line,
                        derived_family_name=family_name,
                        derived_constructor_name=next(iter(constructor_names)),
                        primary_family_name=primary_family_name,
                        primary_constructor_name=next(iter(primary_constructor_names)),
                        link_field_name=link_field_name,
                        primary_constant_names=primary_constant_names,
                        extra_field_names=extra_field_names,
                        builder_names=builder_names,
                        evidence=tuple(evidence[:6]),
                    )
                )
                break
        return sorted_tuple(
            candidates,
            key=lambda item: (item.file_path, item.line, item.derived_family_name),
        )

    @staticmethod
    def _comprehension_builder_names(
        module: ParsedModule,
        family_name: str,
    ) -> tuple[str, ...]:
        builder_names: set[str] = set()
        for subnode in module.syntax_index.breadth_first_nodes:
            if not isinstance(
                subnode, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
            ):
                continue
            if len(subnode.generators) != 1:
                continue
            generator = subnode.generators[0]
            if generator.ifs or not isinstance(generator.iter, ast.Name):
                continue
            if generator.iter.id != family_name or not isinstance(
                generator.target, ast.Name
            ):
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
                for nested in ast.walk(candidate_node):
                    if isinstance(nested, ast.Call) and _call_uses_iteration_variable(
                        nested, iteration_variable_name
                    ):
                        candidate_calls.append(nested)
            for call in candidate_calls:
                call_name = ExpressionReference.decoded(call.func).terminal_name
                if call_name is not None:
                    builder_names.add(call_name)
        return sorted_tuple(builder_names)

    @staticmethod
    def _named_family_for_constants(
        named_sequences: ModuleNamedSequenceMap,
        constant_names: tuple[str, ...],
    ) -> str | None:
        constant_set = set(constant_names)
        for family_name, (_, elements) in sorted(named_sequences.items()):
            element_names = tuple(
                (element.id for element in elements if isinstance(element, ast.Name))
            )
            if len(element_names) != len(elements):
                continue
            if constant_set <= set(element_names):
                return family_name
        return None


class DerivedWrapperSpecShadowDetector(
    ModuleCollectorCandidateDetector[DerivedWrapperSpecShadowCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Generated wrapper spec family should fold into the main spec family",
        "A writable wrapper-spec table is a second source of truth when it only points back at an existing spec family and feeds code generation. Put the wrapper metadata on the main spec records and generate wrappers from them.",
        "one spec family carrying the wrapper-generation metadata",
        "a second spec table references the main spec family entry by entry and is only used for wrapper generation",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.SCOPED_SHAPE_WRAPPER,
        ),
    )

    def _finding_for_candidate(
        self, shadow_candidate: DerivedWrapperSpecShadowCandidate
    ) -> RefactorFinding:
        primary_family_label = (
            shadow_candidate.primary_family_name
            or shadow_candidate.primary_constructor_name
        )
        constant_preview = ", ".join(shadow_candidate.primary_constant_names[:4])
        builder_preview = ", ".join(shadow_candidate.builder_names[:3])
        return self.build_finding(
            (
                f"`{shadow_candidate.derived_family_name}` re-encodes wrapper metadata over authoritative family "
                f"`{primary_family_label}` through link field `{shadow_candidate.link_field_name}` for {constant_preview}, "
                f"then feeds generated wrappers via {builder_preview}."
            ),
            shadow_candidate.evidence,
            metrics=MappingMetrics(
                mapping_site_count=len(shadow_candidate.primary_constant_names),
                field_count=max(len(shadow_candidate.extra_field_names), 1),
                mapping_name=shadow_candidate.derived_family_name,
                field_names=shadow_candidate.extra_field_names,
                source_name=primary_family_label,
                identity_field_names=(shadow_candidate.link_field_name,),
            ),
        )


def _compact_keyed_family_axis_specs_from_index(
    class_index: CompactClassFamilyIndex,
) -> tuple[_KeyedFamilyAxisSpec, ...]:
    specs: list[_KeyedFamilyAxisSpec] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        key_type_name = indexed_class.keyed_family_key_type_name
        if key_type_name is None:
            continue
        registry_key_attr_name = indexed_class.constant_string_assignment(
            "registry_key_attr"
        )
        if registry_key_attr_name is None:
            continue
        case_names = sorted_tuple(
            {
                assignment
                for descendant_symbol in class_index.descendant_symbols(
                    indexed_class.symbol
                )
                if (descendant := class_index.class_for(descendant_symbol)) is not None
                if (
                    assignment := descendant.assignments_by_name.get(
                        registry_key_attr_name
                    )
                )
                is not None
            }
        )
        if len(case_names) < 2:
            continue
        family_name = class_index.display_name(indexed_class)
        specs.append(
            _KeyedFamilyAxisSpec(
                file_path=indexed_class.file_path,
                line=indexed_class.line,
                family_name=family_name,
                key_type_name=key_type_name,
                family_label=indexed_class.constant_string_assignment("family_label"),
                registry_key_attr_name=registry_key_attr_name,
                case_names=case_names,
            )
        )
    return tuple(specs)


def _compact_keyed_family_axis_specs_from_context(
    context: object | None,
) -> tuple[_KeyedFamilyAxisSpec, ...]:
    repository = CompactClassRepositoryContext.require(context)
    return repository.cached(
        _compact_keyed_family_axis_specs_from_index,
        lambda: _compact_keyed_family_axis_specs_from_index(repository.class_index),
    )


class KeyedFamilyAxisRootAnchored(TargetClassEvidenceAnchored):
    """A keyed-family report names one of the roots that owns the axis."""

    @classmethod
    def anchors_report(cls, projection: CompactModuleClassProjection) -> bool:
        return any(
            indexed_class.declares_keyed_base
            and "registry_key_attr" in indexed_class.direct_declared_member_names
            for indexed_class in projection.classes
        )


@dataclass(frozen=True)
class CrossModuleAxisShadowFamilyCandidate(CollectsItselfFromCompactContext[CompactClassRepositoryContext], ClosedAxis, EvidenceLocated):
    """A closed axis one family owns and a selector in another module enumerates again;
    its cases are the ones both name."""

    authoritative: _KeyedFamilyAxisSpec
    shadow: CompactManualSelectorAxis

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (self.authoritative.location, self.shadow.location)

    @classmethod
    def from_compact_context(
        cls, context: CompactClassRepositoryContext, config: DetectorConfig
    ) -> tuple[Self, ...]:
        del config
        return cls._from_specs(
            _compact_keyed_family_axis_specs_from_context(context),
            tuple(
                axis
                for projection in context.projections
                for axis in projection.manual_selector_axes
            ),
        )

    @classmethod
    def _from_specs(
        cls,
        authoritative_specs: Sequence[_KeyedFamilyAxisSpec],
        shadow_specs: Sequence[CompactManualSelectorAxis],
    ) -> tuple[Self, ...]:
        candidates: list[Self] = []
        seen: set[tuple[str, str, str]] = set()
        for authoritative_spec in authoritative_specs:
            for shadow_spec in shadow_specs:
                if authoritative_spec.file_path == shadow_spec.file_path:
                    continue
                if authoritative_spec.key_type_name != shadow_spec.key_type_name:
                    continue
                shared_case_names = sorted_tuple(
                    set(authoritative_spec.case_names) & set(shadow_spec.case_names)
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
                    cls(
                        key_type_name=authoritative_spec.key_type_name,
                        case_names=shared_case_names,
                        authoritative=authoritative_spec,
                        shadow=shadow_spec,
                    )
                )
        return sorted_tuple(
            candidates,
            key=lambda item: (
                item.key_type_name,
                item.authoritative.file_path,
                item.shadow.file_path,
            ),
        )


class CompactCrossModuleAxisShadowFamilyCandidateBase(
    KeyedFamilyAxisRootAnchored,
    CompactClassRepositoryCandidateDetector[CrossModuleAxisShadowFamilyCandidate],
):
    @classmethod
    def anchors_report(cls, projection: CompactModuleClassProjection) -> bool:
        return super().anchors_report(projection) or bool(
            projection.manual_selector_axes
        )

declare_candidate_rule_detector(
    CrossModuleAxisShadowFamilyCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Shadow family in another module should fold into one axis owner",
        "A closed axis should have one owner. One module already owns an enum or keyed family, and another module builds a second family over the same cases. Derive the local behavior from the owner.",
        "one closed-axis family reused across modules",
        "the same keyed enum axis is modeled by an owning family in one module and a shadow selector family in another",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.FACTORY_DISPATCH,
            ObservationTag.DATAFLOW_ROOT,
        ),
    ),
    summary=lambda shadow_candidate: (
        f"Axis `{shadow_candidate.key_type_name}` is already owned by `{shadow_candidate.authoritative.family_name}` but re-encoded by `{shadow_candidate.shadow.family_name}.{shadow_candidate.shadow.selector_method_name}` across cases {', '.join(shadow_candidate.case_names[:4])}."
    ),
    metrics=lambda shadow_candidate: shadow_candidate.dispatch_metrics,
    detector_base=CompactCrossModuleAxisShadowFamilyCandidateBase,
)


@dataclass(frozen=True)
class ResidualClosedAxisBranchingCandidate(CollectsItselfFromCompactContext[CompactClassRepositoryContext], ClosedAxis, EvidenceLocated):
    """A closed axis keyed families own that one function elsewhere still branches on;
    its cases are the ones the function and those families both name."""

    function: CompactClosedAxisBranchFunction
    branch: CompactClosedAxisBranchFact
    authoritative_families: tuple[_KeyedFamilyAxisSpec, ...]

    @property
    def dispatch_site_count(self) -> int:
        return self.branch.branch_site_count

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            self.function.location,
            *(family.location for family in self.authoritative_families),
        )[:6]

    @classmethod
    def from_compact_context(
        cls, context: CompactClassRepositoryContext, config: DetectorConfig
    ) -> tuple[Self, ...]:
        del config
        return cls._from_compact_specs(
            context.projections,
            _compact_keyed_family_axis_specs_from_context(context),
        )

    @classmethod
    def _from_compact_specs(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        keyed_family_specs: tuple[_KeyedFamilyAxisSpec, ...],
    ) -> tuple[Self, ...]:
        authoritative_specs_by_key: KeyedFamilyAxisSpecsByKey = defaultdict(list)
        for spec in keyed_family_specs:
            authoritative_specs_by_key[spec.key_type_name].append(spec)
        if not authoritative_specs_by_key:
            return ()
        candidates: list[Self] = []
        seen: set[ResidualClosedAxisBranchingIdentity] = set()
        for projection in projections:
            for function in projection.closed_axis_branch_functions:
                if "/tests/" in function.file_path:
                    continue
                for axis in function.axes:
                    specs = authoritative_specs_by_key.get(axis.key_type_name, ())
                    if not specs or any(
                        spec.file_path == function.file_path for spec in specs
                    ):
                        continue
                    authoritative_case_names = {
                        case_name for spec in specs for case_name in spec.case_names
                    }
                    shared_case_names = sorted_tuple(
                        set(axis.case_names) & authoritative_case_names
                    )
                    if not shared_case_names:
                        continue
                    identity = ResidualClosedAxisBranchingIdentity(
                        file_path=function.file_path,
                        qualname=function.qualname,
                        key_type_name=axis.key_type_name,
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    candidates.append(
                        cls(
                            key_type_name=axis.key_type_name,
                            case_names=shared_case_names,
                            function=function,
                            branch=axis,
                            authoritative_families=sorted_tuple(
                                specs,
                                key=lambda spec: (
                                    spec.family_name,
                                    spec.file_path,
                                    spec.line,
                                ),
                            ),
                        )
                    )
        return sorted_tuple(
            candidates,
            key=lambda item: (
                item.key_type_name,
                item.function.file_path,
                item.function.line,
                item.function.qualname,
            ),
        )


class ResidualClosedAxisBranchingDetector(
    KeyedFamilyAxisRootAnchored,
    CompactClassRepositoryCandidateDetector[ResidualClosedAxisBranchingCandidate],
):
    @classmethod
    def anchors_report(cls, projection: CompactModuleClassProjection) -> bool:
        return super().anchors_report(projection) or bool(
            projection.closed_axis_branch_functions
        )

    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Manual closed-axis branching should derive from the existing keyed family",
        "A closed enum or key axis should have one owner. A keyed family already owns it, so `if`/`match` ladders over the same cases downstream are leftover shadow dispatch.",
        "behavior derived from the keyed family instead of downstream enum branching",
        "function branches on an enum axis already owned by a keyed family in another module",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _finding_for_candidate(
        self, residual_candidate: ResidualClosedAxisBranchingCandidate
    ) -> RefactorFinding:
        authoritative_family_names = ", ".join(
            (
                family.family_name
                for family in residual_candidate.authoritative_families[:4]
            )
        )
        return self.build_finding(
            (
                f"`{residual_candidate.function.qualname}` branches {residual_candidate.branch.branch_site_count} time(s) on axis "
                f"`{residual_candidate.key_type_name}` across cases {', '.join(residual_candidate.case_names)}, "
                f"even though authoritative family `{authoritative_family_names}` already owns that axis."
            ),
            residual_candidate.evidence,
            metrics=residual_candidate.dispatch_metrics,
        )


@dataclass(frozen=True)
class ExternalEnumCaseRecoverySite:
    """One external function recovering cases from an enum declaration."""

    function: CompactClosedAxisBranchFunction
    branch_site_count: int = 0
    case_names: tuple[str, ...] = ()

    def with_branches(
        self, branch_site_count: int, case_names: Iterable[str]
    ) -> ExternalEnumCaseRecoverySite:
        return replace(
            self,
            branch_site_count=self.branch_site_count + branch_site_count,
            case_names=sorted_tuple({*self.case_names, *case_names}),
        )


def unique_direct_enum_classes(
    class_index: CompactClassFamilyIndex,
) -> dict[str, CompactIndexedClass]:
    return {
        simple_name: enum_class
        for simple_name, symbols in class_index.symbols_by_simple_name.items()
        if len(symbols) == 1
        if (
            enum_class := class_index.classes_by_symbol[symbols[0]]
        ).direct_enum_member_names
    }


def declared_cases(enum_class: CompactIndexedClass) -> set[str]:
    return {
        f"{enum_class.simple_name}.{member_name}"
        for member_name in enum_class.direct_enum_member_names
    }


def _receiver_field_enum(
    class_index: CompactClassFamilyIndex,
    enum_classes: dict[str, CompactIndexedClass],
    function: CompactClosedAxisBranchFunction,
    field_name: str,
) -> CompactIndexedClass | None:
    """The enum the method's own class (or an ancestor) annotates ``field_name`` with."""

    owner_qualname, _dot, _method = function.qualname.rpartition(".")
    owner = class_index.symbol_for(
        file_path=function.file_path, qualname=owner_qualname
    )
    if owner is None:
        return None
    annotated = {
        declaration.annotation_expression
        for symbol in (owner, *class_index.ancestors_by_symbol.get(owner, ()))
        if (indexed_class := class_index.class_for(symbol)) is not None
        if (
            declaration := indexed_class.direct_annotation_writes_by_name.get(
                field_name
            )
        )
        is not None
    }
    if len(annotated) != 1:
        return None
    return enum_classes.get(next(iter(annotated)))


def recovered_enum_sites(
    class_index: CompactClassFamilyIndex,
    enum_classes: dict[str, CompactIndexedClass],
    function: CompactClosedAxisBranchFunction,
) -> dict[str, ExternalEnumCaseRecoverySite]:
    """Enums one function's branches read, by name, with the cases they decide.

    ``Enum.MEMBER`` references decide the members named. ``self.field.data``, where
    the method's class annotates ``field`` with the enum, reads member data and so
    decides every member: the switch moved into the members' data, not out of the
    reader.
    """

    sites: dict[str, ExternalEnumCaseRecoverySite] = {}
    for axis in function.axes:
        compared = enum_classes.get(axis.key_type_name)
        if compared is not None:
            sites[compared.simple_name] = ExternalEnumCaseRecoverySite(
                function
            ).with_branches(
                axis.branch_site_count,
                set(axis.case_names) & declared_cases(compared),
            )
    for read in function.receiver_field_reads:
        held = _receiver_field_enum(
            class_index, enum_classes, function, read.field_name
        )
        if held is None or set(read.attribute_names) <= set(
            held.direct_enum_member_names
        ):
            continue
        sites[held.simple_name] = sites.get(
            held.simple_name, ExternalEnumCaseRecoverySite(function)
        ).with_branches(read.branch_site_count, declared_cases(held))
    return sites


@dataclass(frozen=True)
class ExternalEnumCaseRecoveryCandidate(CollectsItselfFromCompactContext[CompactClassRepositoryContext], EvidenceLocated):
    """One enum whose cases are recovered outside its declaration."""

    enum_class: CompactIndexedClass
    sites: tuple[ExternalEnumCaseRecoverySite, ...]
    case_names: tuple[str, ...]

    @property
    def branch_site_count(self) -> int:
        return sum(site.branch_site_count for site in self.sites)

    @property
    def authority_evidence(self) -> SourceLocation:
        return self.enum_class.location

    @property
    def projection_evidence(self) -> SourceLocation:
        if not self.sites:
            raise ValueError("external enum recovery requires one projection site")
        return self.sites[0].function.location

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            self.authority_evidence,
            *(site.function.location for site in self.sites[:5]),
        )

    @classmethod
    def from_compact_context(
        cls, context: CompactClassRepositoryContext, config: DetectorConfig
    ) -> tuple[Self, ...]:
        enum_classes = unique_direct_enum_classes(context.class_index)
        keyed_axis_names = {
            spec.key_type_name
            for spec in _compact_keyed_family_axis_specs_from_context(context)
        }
        sites_by_enum_name: dict[str, list[ExternalEnumCaseRecoverySite]] = defaultdict(
            list
        )
        for projection in context.projections:
            for function in projection.closed_axis_branch_functions:
                if PythonSourcePathPolicy.is_test_path(Path(function.file_path)):
                    continue
                for enum_name, site in recovered_enum_sites(
                    context.class_index, enum_classes, function
                ).items():
                    enum_class = enum_classes[enum_name]
                    if enum_name in keyed_axis_names or not site.case_names:
                        continue
                    if (
                        function.file_path == enum_class.file_path
                        and function.qualname.startswith(f"{enum_class.qualname}.")
                    ):
                        continue
                    sites_by_enum_name[enum_name].append(site)
        candidates: list[Self] = []
        for enum_name, sites in sites_by_enum_name.items():
            case_names = sorted_tuple(
                {case_name for site in sites for case_name in site.case_names}
            )
            if len(case_names) < config.min_string_cases:
                continue
            candidates.append(
                cls(
                    enum_class=enum_classes[enum_name],
                    sites=tuple(sites),
                    case_names=case_names,
                )
            )
        return sorted_tuple(
            candidates,
            key=lambda candidate: candidate.enum_class.symbol,
        )


class ExternalEnumCaseRecoveryDetector(
    SemanticMirrorIssueDetector,
    CompactClassRepositoryCandidateDetector[ExternalEnumCaseRecoveryCandidate],
    CasesFindingRecipeSynthesizer,
):
    """Find closed enum case semantics recovered outside their declaration."""

    @classmethod
    def compact_report_context_can_promote(
        cls,
        target_projections_by_family: CompactProjectionGroups,
        config: DetectorConfig,
    ) -> bool:
        """The target holds a closed-axis branch and an enum, in any of its modules."""

        del config
        projections = CompactModuleClassProjectionFamily.present_in(
            target_projections_by_family
        )
        return any(
            projection.closed_axis_branch_functions for projection in projections
        ) and any(
            indexed_class.direct_enum_member_names
            for projection in projections
            for indexed_class in projection.classes
        )

    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "External enum case recovery suggests class-owned polymorphic behavior",
        "Functions outside an enum declaration inspect several of its cases. If the cases carry behavior, prefer a public ABC whose concrete subclasses own their declarations, data and behavior, with shared algorithms on the parent. Consumers should call that contract instead of checking tags or keeping parallel tables. Moving the same switch into an Enum method does not get you there. A value-only enum or outside wire vocabulary needs its own justification, not an automatic conversion to subclasses. Trace existing classes and other possible owners, then plan the full consumer migration. This observation alone does not prove runtime binding, a required hierarchy, full coverage or behavioral equivalence.",
        "a shared public contract with case-owned behavior instead of repeated member checks, if the task rules allow it",
        "several enum members are inspected outside the enum that owns them",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.CLOSED_FAMILY_CASES,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _finding_for_candidate(
        self,
        candidate: ExternalEnumCaseRecoveryCandidate,
    ) -> RefactorFinding:
        return self.build_finding(
            (
                f"Enum `{candidate.enum_class.simple_name}` has "
                f"{candidate.branch_site_count} external branch site(s) across "
                f"{len(candidate.sites)} function(s) for cases "
                f"{', '.join(candidate.case_names)}."
            ),
            candidate.evidence,
            projection_evidence=candidate.projection_evidence,
            authority_evidence=candidate.authority_evidence,
            metrics=DispatchCountMetrics(
                dispatch_site_count=candidate.branch_site_count,
                dispatch_axis=candidate.enum_class.simple_name,
                literal_cases=candidate.case_names,
            ),
        )


@dataclass(frozen=True)
class ParallelKeyedAxisFamilyCandidate(CollectsItselfFromCompactContext[CompactClassRepositoryContext], ClosedAxis, EvidenceLocated):
    """A closed axis two keyed families in different modules both own;
    its cases are the ones both name."""

    left: _KeyedFamilyAxisSpec
    right: _KeyedFamilyAxisSpec
    case_overlap_ratio: float
    name_overlap_ratio: float

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (self.left.location, self.right.location)

    @classmethod
    def from_compact_context(
        cls, context: CompactClassRepositoryContext, config: DetectorConfig
    ) -> tuple[Self, ...]:
        del config
        return cls._from_specs(
            _compact_keyed_family_axis_specs_from_context(context)
        )

    @classmethod
    def _from_specs(cls, specs: Sequence[_KeyedFamilyAxisSpec]) -> tuple[Self, ...]:
        candidates: list[Self] = []
        seen: set[tuple[str, str, str]] = set()
        for index, left_spec in enumerate(specs):
            for right_spec in specs[index + 1 :]:
                if left_spec.file_path == right_spec.file_path:
                    continue
                if left_spec.key_type_name != right_spec.key_type_name:
                    continue
                if left_spec.registry_key_attr_name != right_spec.registry_key_attr_name:
                    continue
                shared_case_names = sorted_tuple(
                    set(left_spec.case_names) & set(right_spec.case_names)
                )
                if len(shared_case_names) < 2:
                    continue
                family_label_match = (
                    left_spec.family_label is not None
                    and left_spec.family_label == right_spec.family_label
                )
                case_overlap_score = overlap_coefficient(
                    left_spec.case_names, right_spec.case_names
                )
                name_overlap_ratio = CLASS_NAME_ALGEBRA.token_overlap(
                    left_spec.family_name, right_spec.family_name
                )
                if not family_label_match and (
                    case_overlap_score < 0.8 or name_overlap_ratio < 0.6
                ):
                    continue
                key = sorted_tuple((left_spec.family_name, right_spec.family_name)) + (
                    left_spec.key_type_name,
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    cls(
                        key_type_name=left_spec.key_type_name,
                        case_names=shared_case_names,
                        left=left_spec,
                        right=right_spec,
                        case_overlap_ratio=case_overlap_score,
                        name_overlap_ratio=name_overlap_ratio,
                    )
                )
        return sorted_tuple(
            candidates,
            key=lambda item: (
                item.key_type_name,
                item.left.file_path,
                item.left.family_name,
                item.right.file_path,
                item.right.family_name,
            ),
        )


class ParallelKeyedAxisFamilyDetector(
    KeyedFamilyAxisRootAnchored,
    CompactClassRepositoryCandidateDetector[ParallelKeyedAxisFamilyCandidate],
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Parallel keyed families should fold into one axis owner",
        "A closed axis should have one owner. Two modules each define a keyed family over the same enum or key cases, so ownership is split even though both sides are proper classes.",
        "one cross-module keyed-axis owner with module-local adapters derived from it",
        "the same keyed enum axis is modeled by several families across modules",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.FACTORY_DISPATCH,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _finding_for_candidate(
        self, family_candidate: ParallelKeyedAxisFamilyCandidate
    ) -> RefactorFinding:
        shared_cases = ", ".join(family_candidate.case_names[:4])
        label_clause = ""
        if (
            family_candidate.left.family_label is not None
            and family_candidate.left.family_label
            == family_candidate.right.family_label
        ):
            label_clause = (
                f" Both declare family label `{family_candidate.left.family_label}`."
            )
        return self.build_finding(
            (
                f"Axis `{family_candidate.key_type_name}` is owned in parallel by "
                f"`{family_candidate.left.family_name}` and `{family_candidate.right.family_name}` "
                f"across cases {shared_cases}.{label_clause}"
            ),
            family_candidate.evidence,
            metrics=family_candidate.dispatch_metrics,
        )


@dataclass(frozen=True)
class ParallelKeyedTableAndFamilyCandidate(CollectsItselfFromCompactContext[CompactClassRepositoryContext], ClosedAxis, EvidenceLocated):
    """A closed axis one module spells as a keyed table and as a keyed family;
    its cases are the ones both name."""

    table: CompactKeyedTableAxis
    family: _KeyedFamilyAxisSpec

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (self.table.location, self.family.location)

    @classmethod
    def from_compact_context(
        cls, context: CompactClassRepositoryContext, config: DetectorConfig
    ) -> tuple[Self, ...]:
        del config
        return cls._from_specs(
            _compact_keyed_family_axis_specs_from_context(context),
            tuple(
                axis
                for projection in context.projections
                for axis in projection.keyed_table_axes
            ),
        )

    @classmethod
    def _from_specs(
        cls,
        family_specs: Sequence[_KeyedFamilyAxisSpec],
        table_specs: Sequence[CompactKeyedTableAxis],
    ) -> tuple[Self, ...]:
        family_specs_by_file: KeyedFamilyAxisSpecsByKey = {}
        for family_spec in family_specs:
            family_specs_by_file.setdefault(family_spec.file_path, []).append(family_spec)
        candidates: list[Self] = []
        seen: set[tuple[str, str, str]] = set()
        for table_spec in table_specs:
            local_family_specs = family_specs_by_file.get(table_spec.file_path, ())
            for family_spec in local_family_specs:
                if table_spec.key_type_name != family_spec.key_type_name:
                    continue
                shared_case_names = sorted_tuple(
                    set(table_spec.case_names) & set(family_spec.case_names)
                )
                if len(shared_case_names) < 2:
                    continue
                case_overlap_score = overlap_coefficient(
                    table_spec.case_names, family_spec.case_names
                )
                if case_overlap_score < 0.8:
                    continue
                table_overlap = CLASS_NAME_ALGEBRA.token_overlap(
                    table_spec.table_name, family_spec.family_name
                )
                value_overlap = (
                    0.0
                    if table_spec.value_shape_name is None
                    else CLASS_NAME_ALGEBRA.token_overlap(
                        table_spec.value_shape_name, family_spec.family_name
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
                    cls(
                        key_type_name=table_spec.key_type_name,
                        case_names=shared_case_names,
                        table=table_spec,
                        family=family_spec,
                    )
                )
        return sorted_tuple(
            candidates,
            key=lambda item: (
                item.table.file_path,
                item.key_type_name,
                item.table.table_name,
                item.family.family_name,
            ),
        )


class ParallelKeyedTableAndFamilyDetector(
    TargetClassEvidenceAnchored,
    CompactClassRepositoryCandidateDetector[ParallelKeyedTableAndFamilyCandidate],
):
    @classmethod
    def anchors_report(cls, projection: CompactModuleClassProjection) -> bool:
        return bool(projection.keyed_table_axes)

    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Keyed table and keyed family should fold into one auto-registered axis family",
        "A closed axis should have one owner. A module keeps a keyed table of per-case records and also a keyed family over the same cases, so the axis is split between data and behavior. If the family already holds the runtime behavior, derive the table from it.",
        "one auto-registered axis family with table and view projections derived from it",
        "the same enum or key axis is encoded by both a keyed table and a keyed family",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _finding_for_candidate(
        self, table_candidate: ParallelKeyedTableAndFamilyCandidate
    ) -> RefactorFinding:
        shape_clause = (
            ""
            if table_candidate.table.value_shape_name is None
            else f" of `{table_candidate.table.value_shape_name}` records"
        )
        return self.build_finding(
            (
                f"Axis `{table_candidate.key_type_name}` is split between keyed table `{table_candidate.table.table_name}`"
                f"{shape_clause} and keyed family `{table_candidate.family.family_name}` across cases "
                f"{', '.join(table_candidate.case_names[:4])}."
            ),
            table_candidate.evidence,
            metrics=table_candidate.dispatch_metrics,
        )


class _CompactClassIndexDetectorBase(
    CompactClassRepositoryProjectionMixin,
    IssueDetector,
    ABC,
):
    """A detector whose findings come from the class projections and their class index."""

    def _findings_from_compact_projections(
        self,
        projections: Sequence[CompactModuleClassProjection],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        return self._findings_from_class_index(
            projections,
            build_compact_class_family_index(projections),
        )

    def _findings_from_compact_context(
        self,
        projections: Sequence[CompactModuleClassProjection],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        return self._findings_from_class_index(
            projections,
            CompactClassRepositoryContext.require(context).class_index,
        )

    @abstractmethod
    def _findings_from_class_index(
        self,
        projections: Sequence[CompactModuleClassProjection],
        class_index: CompactClassFamilyIndex,
    ) -> list[RefactorFinding]:
        raise NotImplementedError


class InheritedAutoRegisterConfigBoilerplateDetector(
    _CompactClassIndexDetectorBase,
    InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer,
):
    detector_id = "inherited_autoregister_config_boilerplate"
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "AutoRegister root repeats inherited registry configuration",
        "An AutoRegisterMeta root repeats registry settings its base already declares. Inherit the registry key, skip policy and related fields from the shared base. If AutoRegisterMeta can't honor inherited config, fix the metaclass package instead of repeating fields on every root.",
        "inherited AutoRegister registry configuration",
        "AutoRegisterMeta class repeats registry assignments from an inherited base",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _findings_from_class_index(
        self,
        projections: Sequence[CompactModuleClassProjection],
        class_index: CompactClassFamilyIndex,
    ) -> list[RefactorFinding]:
        del projections
        findings: list[RefactorFinding] = []
        for indexed_class in sorted(
            class_index.classes_by_symbol.values(), key=lambda item: item.symbol
        ):
            if not indexed_class.declares_autoregister_meta:
                continue
            repeated_fields = class_index.assignments_repeated_from_ancestors(
                indexed_class.symbol,
                INHERITABLE_AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
            )
            if not repeated_fields:
                continue
            field_list = ", ".join(repeated_fields)
            findings.append(
                self.build_finding(
                    (
                        f"`{indexed_class.simple_name}` repeats inherited AutoRegister "
                        f"registry field(s) {field_list}."
                    ),
                    (
                        indexed_class.named(indexed_class.simple_name),
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(repeated_fields),
                        mapping_name=indexed_class.simple_name,
                        field_names=repeated_fields,
                    ),
                )
            )
        return findings


_EXPLICIT_CLASS_ORDER_AXIS_NAMES = ("priority", "precedence", "rank", "order")


class AutoRegisterExplicitPriorityOrderingDetector(
    _CompactClassIndexDetectorBase,
    AutoRegisterExplicitPriorityOrderingFindingRecipeSynthesizer,
):
    detector_id = "autoregister_explicit_priority_ordering"
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "AutoRegister family uses explicit priority ordering instead of MRO",
        "An AutoRegisterMeta family whose leaves carry a `priority`, `precedence`, `rank` or `order` class attribute keeps a second ordering next to the inheritance graph. If the order matters, let the hierarchy and MRO carry it. If it's only for display, derive it in a view outside the family.",
        "MRO-owned ordering for registered families",
        "AutoRegisterMeta family declares or reads a class-level priority to sort registered implementations",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _findings_from_class_index(
        self,
        projections: Sequence[CompactModuleClassProjection],
        class_index: CompactClassFamilyIndex,
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for indexed_class in sorted(
            class_index.classes_by_symbol.values(), key=lambda item: item.symbol
        ):
            if not indexed_class.declares_autoregister_meta:
                continue
            order_axis_sites = self._order_axis_sites(class_index, indexed_class)
            order_axis_names = tuple(
                axis_name
                for axis_name, locations in order_axis_sites.items()
                if locations
            )
            if not order_axis_names:
                continue
            if not self._sorts_registry_by_order_axis(
                indexed_class.simple_name,
                projections,
                order_axis_names,
            ):
                continue
            evidence_sites = tuple(
                location
                for locations in order_axis_sites.values()
                for location in locations
            )
            axis_label = _class_order_axis_label(order_axis_names)
            findings.append(
                self.build_finding(
                    (
                        f"`{indexed_class.simple_name}` orders registered implementations "
                        f"through explicit class-level {axis_label} values."
                    ),
                    (
                        indexed_class.named(indexed_class.simple_name),
                        *evidence_sites,
                    ),
                    metrics=MappingMetrics(
                        mapping_site_count=len(evidence_sites),
                        field_count=len(evidence_sites),
                        mapping_name=indexed_class.simple_name,
                        field_names=order_axis_names,
                    ),
                )
            )
        return findings

    def _order_axis_sites(
        self,
        class_index: CompactClassFamilyIndex,
        indexed_class: CompactIndexedClass,
    ) -> dict[str, tuple[SourceLocation, ...]]:
        symbols = (
            indexed_class.symbol,
            *class_index.descendant_symbols(indexed_class.symbol),
        )
        sites_by_axis: dict[str, list[SourceLocation]] = {
            axis_name: [] for axis_name in _EXPLICIT_CLASS_ORDER_AXIS_NAMES
        }
        for symbol in symbols:
            candidate = class_index.class_for(symbol)
            if candidate is None:
                continue
            assignment_lines = candidate.declared_member_lines_by_name
            for axis_name in _EXPLICIT_CLASS_ORDER_AXIS_NAMES:
                line = assignment_lines.get(axis_name)
                if line is None:
                    continue
                sites_by_axis[axis_name].append(
                    SourceLocation(candidate.file_path, line, candidate.simple_name)
                )
        return {
            axis_name: tuple(locations)
            for axis_name, locations in sites_by_axis.items()
            if locations
        }

    @staticmethod
    def _sorts_registry_by_order_axis(
        root_name: str,
        projections: Sequence[CompactModuleClassProjection],
        axis_names: tuple[str, ...],
    ) -> bool:
        axis_name_set = frozenset(axis_names)
        return any(
            ({"cls", root_name} & frozenset(call.registry_owner_names))
            and (axis_name_set & frozenset(call.key_attribute_names))
            for projection in projections
            for call in projection.sorted_key_calls
        )


def _class_order_axis_label(axis_names: tuple[str, ...]) -> str:
    if len(axis_names) == 1:
        return f"`{axis_names[0]}`"
    return " / ".join(f"`{axis_name}`" for axis_name in axis_names)


class NominalInstanceExplicitOrderingDetector(_CompactClassIndexDetectorBase):
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Nominal declaration catalog uses explicit ordering instead of MRO",
        "An abstract value family has a `priority`, `precedence`, `rank` or `order` field. Class-owned declarations fill it in and a consumer sorts by it. That is a second ordering next to the inheritance graph. Give each declaration one class node and derive the sequence from its MRO.",
        "MRO-owned sequence for declaration catalogs",
        "class-owned instances of an abstract family supply an ordering field that is read by sorted(..., key=...)",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _findings_from_class_index(
        self,
        projections: Sequence[CompactModuleClassProjection],
        class_index: CompactClassFamilyIndex,
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        indexed_classes = tuple(class_index.classes_by_symbol.values())
        sorted_key_calls = tuple(
            call for projection in projections for call in projection.sorted_key_calls
        )
        for family_root in sorted(indexed_classes, key=lambda item: item.symbol):
            if not family_root.is_abstract or family_root.declares_autoregister_meta:
                continue
            root_axis_lines = {
                axis_name: line
                for axis_name, line in family_root.declared_member_lines_by_name.items()
                if axis_name in _EXPLICIT_CLASS_ORDER_AXIS_NAMES
            }
            if not root_axis_lines:
                continue
            descendant_names = frozenset(
                descendant.simple_name
                for symbol in class_index.descendant_symbols(family_root.symbol)
                if (descendant := class_index.class_for(symbol)) is not None
            )
            if len(descendant_names) < 2:
                continue
            axis_evidence = {
                axis_name: self._axis_evidence(
                    axis_name,
                    descendant_names,
                    indexed_classes,
                    sorted_key_calls,
                )
                for axis_name in root_axis_lines
            }
            active_axes = tuple(
                axis_name
                for axis_name, evidence in axis_evidence.items()
                if evidence is not None
            )
            if not active_axes:
                continue
            evidence_sites = tuple(
                location
                for axis_name in active_axes
                for location in axis_evidence[axis_name] or ()
            )
            axis_label = _class_order_axis_label(active_axes)
            findings.append(
                self.build_finding(
                    (
                        f"`{family_root.simple_name}` declarations are sequenced "
                        f"through explicit {axis_label} instance values."
                    ),
                    (
                        family_root.named(family_root.simple_name),
                        *(
                            SourceLocation(
                                family_root.file_path,
                                root_axis_lines[axis_name],
                                axis_name,
                            )
                            for axis_name in active_axes
                        ),
                        *evidence_sites[:6],
                    ),
                    metrics=MappingMetrics(
                        mapping_site_count=len(evidence_sites),
                        field_count=len(evidence_sites),
                        mapping_name=family_root.simple_name,
                        field_names=active_axes,
                    ),
                )
            )
        return findings

    @staticmethod
    def _axis_evidence(
        axis_name: str,
        descendant_names: frozenset[str],
        indexed_classes: tuple[CompactIndexedClass, ...],
        sorted_key_calls: tuple[CompactSortedKeyCall, ...],
    ) -> tuple[SourceLocation, ...] | None:
        constructions = tuple(
            (owner, construction)
            for owner in indexed_classes
            for construction in owner.direct_value_constructions
            if construction.constructor_name in descendant_names
            and axis_name in construction.keyword_names
        )
        if len(constructions) < 2:
            return None
        declaration_files = frozenset(owner.file_path for owner, _ in constructions)
        matching_calls = tuple(
            call
            for call in sorted_key_calls
            if axis_name in call.key_attribute_names
            and call.file_path in declaration_files
        )
        if not matching_calls:
            return None
        return (
            *(
                SourceLocation(
                    owner.file_path,
                    construction.line,
                    f"{owner.simple_name}.{construction.assigned_name}",
                )
                for owner, construction in constructions[:4]
            ),
            SourceLocation(
                matching_calls[0].file_path,
                matching_calls[0].line,
                f"sorted(..., key=...{axis_name})",
            ),
        )


class RepeatedKeyedFamilyDetector(
    TargetClassEvidenceAnchored,
    CompactProjectionCandidateDetector[
        CompactModuleClassProjection,
        RepeatedKeyedFamilyCandidate,
    ],
):
    module_projection_family = CompactModuleClassProjectionFamily

    @classmethod
    def anchors_report(cls, projection: CompactModuleClassProjection) -> bool:
        """Repeated-keyed-family evidence consists exclusively of its roots."""

        return bool(projection.repeated_keyed_family_roots)

    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Repeated keyed family scaffolding should collapse into one typed metaclass-registry base",
        "Several families repeat the same class-level registration and lookup shell: `registry_key_attr`, `_registry` and `for_*` lookup methods on each root. Put that algorithm in one typed `metaclass-registry` base.",
        "one typed metaclass-registry base for keyed registries",
        "the same registration and lookup shell is repeated across family roots",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    @staticmethod
    def _candidates_from_compact_projections(
        projections: Sequence[CompactModuleClassProjection],
        config: DetectorConfig,
    ) -> tuple[RepeatedKeyedFamilyCandidate, ...]:
        return RepeatedKeyedFamilyCandidate.from_roots(
            tuple(
                root
                for projection in projections
                for root in projection.repeated_keyed_family_roots
            ),
            scanned_roots=frozenset(
                projection.module_name.partition(".")[0] for projection in projections
            ),
            minimum_root_count=max(3, config.min_registration_sites),
        )

    def _finding_for_candidate(
        self, family_candidate: RepeatedKeyedFamilyCandidate
    ) -> RefactorFinding:
        class_names = ", ".join(
            (root.class_name for root in family_candidate.roots[:8])
        )
        lookup_names = ", ".join(
            sorted({root.lookup_method_name for root in family_candidate.roots[:8]})
        )
        registry_keys = ", ".join(
            sorted({root.registry_key_attr_name for root in family_candidate.roots[:8]})
        )
        evidence = tuple(
            root.named(root.class_name)
            for root in family_candidate.roots[:8]
        )
        return self.build_finding(
            (
                f"Registry roots {class_names} each repeat `{registry_keys}` + `_registry` + "
                f"`{lookup_names}` "
                + (
                    f"over `{family_candidate.family_base_name}`."
                    if family_candidate.family_base_name
                    else "with no shared base."
                )
            ),
            evidence,
        )


def _compact_registry_reference_edges(
    encoded_edges: str,
) -> Iterator[tuple[int, int, int]]:
    for encoded_edge in encoded_edges.split(";"):
        if not encoded_edge:
            continue
        function_index, receiver_index, attribute_index = encoded_edge.split(",")
        yield int(function_index), int(receiver_index), int(attribute_index)


def _compact_registry_consumer_index(
    projections: Sequence[CompactModuleClassProjection],
    relevant_keys: frozenset[tuple[str, str]],
) -> dict[tuple[str, str], frozenset[str]]:
    consumers: dict[tuple[str, str], set[str]] = {}
    for projection in projections:
        reference_index = projection.autoregister_reference_index
        if reference_index is None:
            continue
        for (
            function_index,
            receiver_index,
            attribute_index,
        ) in _compact_registry_reference_edges(reference_index.encoded_edges):
            key = (
                reference_index.receiver_names[receiver_index],
                reference_index.attribute_names[attribute_index],
            )
            if key not in relevant_keys:
                continue
            consumers.setdefault(key, set()).add(
                reference_index.function_qualnames[function_index]
            )
    return {key: frozenset(symbols) for key, symbols in consumers.items()}


def _compact_registry_consumer_symbols(
    consumer_index: dict[tuple[str, str], frozenset[str]],
    *,
    family_name: str,
    lookup_method_names: tuple[str, ...],
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            qualname
            for method_name in lookup_method_names
            for qualname in consumer_index.get((family_name, method_name), ())
            if not qualname.startswith(f"{family_name}.")
        }
    )


def _registry_maturity_missing_signals(
    *,
    registered_case_count: int,
    lookup_method_names: tuple[str, ...],
    consumer_count: int,
    min_case_count: int,
    min_consumer_count: int,
) -> tuple[str, ...]:
    missing: list[str] = []
    if registered_case_count < min_case_count:
        missing.append("registered_case_axis")
    if not lookup_method_names:
        missing.append("lookup_lifecycle")
    if consumer_count < min_consumer_count:
        missing.append("consumer_fanout")
    return tuple(missing)


def _compact_keyed_registry_axis_facts(
    projections: Sequence[CompactModuleClassProjection],
    config: DetectorConfig,
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> tuple[KeyedRegistryAxisFact, ...]:
    if class_index is None:
        class_index = build_compact_class_family_index(projections)
    mapping_key_authority = CompactClassMappingKeyAuthority(
        projections,
        class_index,
    )
    registry_classes = tuple(
        indexed_class
        for indexed_class in class_index.classes_by_symbol.values()
        if indexed_class.keyed_family_key_type_name is not None
        and indexed_class.constant_string_assignment("registry_key_attr") is not None
    )
    relevant_consumer_keys = frozenset(
        (family_name, method_name)
        for indexed_class in registry_classes
        for family_name in (
            class_index.display_name(indexed_class),
        )
        for method_name in indexed_class.keyed_registry_lookup_method_names
    )
    consumer_index = _compact_registry_consumer_index(
        projections, relevant_consumer_keys
    )
    min_case_count = max(2, config.min_registration_sites)
    min_consumer_count = max(2, config.min_registration_sites)
    facts: list[KeyedRegistryAxisFact] = []
    for indexed_class in sorted(registry_classes, key=lambda item: item.symbol):
        if PythonSourcePathPolicy.is_test_path(Path(indexed_class.file_path)):
            continue
        key_type_name = indexed_class.keyed_family_key_type_name
        registry_key_attr_name = indexed_class.constant_string_assignment(
            "registry_key_attr"
        )
        if key_type_name is None or registry_key_attr_name is None:
            continue
        family_name = class_index.display_name(indexed_class)
        lookup_method_names = indexed_class.keyed_registry_lookup_method_names
        consumer_symbols = _compact_registry_consumer_symbols(
            consumer_index,
            family_name=family_name,
            lookup_method_names=lookup_method_names,
        )
        descendants = tuple(
            descendant
            for symbol in class_index.descendant_symbols(indexed_class.symbol)
            if (descendant := class_index.class_for(symbol)) is not None
        )
        concrete_descendants = tuple(
            descendant for descendant in descendants if not descendant.is_abstract
        )
        registered_case_names = sorted_tuple(
            {
                expression
                for descendant in descendants
                if (
                    expression := descendant.assignments_by_name.get(
                        registry_key_attr_name
                    )
                )
                is not None
            }
        )
        key_entries = []
        unresolved_type_names = set()
        for descendant in concrete_descendants:
            descendant_name = class_index.display_name(descendant)
            try:
                member = class_index.direct_or_absent_value_declaration(
                    descendant.symbol, registry_key_attr_name
                )
                if member is None:
                    continue
                key_entries.append(
                    (
                        member.value.resolve_mapping_key(
                            mapping_key_authority,
                            descendant,
                        ),
                        member.expression,
                        (descendant_name,),
                    )
                )
            except (KeyError, ValueError):
                unresolved_type_names.add(descendant_name)
        injectivity_proof = InjectiveTypeRegistryProof.from_partially_resolved_key_entries(
            key_axis_name=key_type_name,
            key_entries=key_entries,
            registered_type_names=tuple(
                class_index.display_name(descendant)
                for descendant in concrete_descendants
            ),
            unresolved_type_names=unresolved_type_names,
            reverse_lookup_names=indexed_class.keyed_registry_reverse_lookup_method_names,
            consumer_symbols=consumer_symbols,
        )
        facts.append(
            KeyedRegistryAxisFact(
                file_path=indexed_class.file_path,
                line=indexed_class.line,
                class_name=family_name,
                key_type_name=key_type_name,
                registry_key_attr_name=registry_key_attr_name,
                lookup_method_names=lookup_method_names,
                registered_case_names=registered_case_names,
                consumer_symbols=consumer_symbols,
                missing_maturity_signals=_registry_maturity_missing_signals(
                    registered_case_count=len(registered_case_names),
                    lookup_method_names=lookup_method_names,
                    consumer_count=len(consumer_symbols),
                    min_case_count=min_case_count,
                    min_consumer_count=min_consumer_count,
                ),
                injectivity_proof=injectivity_proof,
            )
        )
    return tuple(facts)


def _compact_keyed_registry_axis_facts_from_context(
    context: object | None,
) -> tuple[KeyedRegistryAxisFact, ...]:
    repository = CompactClassRepositoryContext.require(context)
    return repository.cached(
        _compact_keyed_registry_axis_facts,
        lambda: _compact_keyed_registry_axis_facts(
            repository.projections,
            repository.config,
            class_index=repository.class_index,
        ),
    )


def _compact_registry_projection_import_aliases(
    projection: CompactModuleClassProjection,
    *,
    registry_projection: CompactModuleClassProjection,
    fact: KeyedRegistryAxisFact,
) -> dict[str, str]:
    canonical_names = frozenset(
        (
            fact.class_name,
            fact.key_type_name,
            *fact.require_proof().registered_type_names,
        )
    )
    return {
        local_name: canonical_name
        for local_name, qualified_name in projection.import_aliases
        for canonical_name in canonical_names
        if qualified_name == f"{registry_projection.module_name}.{canonical_name}"
    }


def _compact_registry_projection_reference_name(
    reference: tuple[str, str | None],
    import_aliases: dict[str, str],
) -> str:
    value, alias_head = reference
    if alias_head is None or alias_head not in import_aliases:
        return value
    canonical_head = import_aliases[alias_head]
    if value == alias_head:
        return canonical_head
    return f"{canonical_head}{value[len(alias_head) :]}"


@dataclass(frozen=True)
class RegistryProjectionSurfaceCandidate(CollectsItselfFromRegistryFacts, LineWitnessCandidate):
    registry_class_name: str
    key_type_name: str
    surface_evidence: RegistryProjectionSurfaceEvidence
    surface_kind: RegistryProjectionSurfaceKind
    projection_role: RegistryProjectionRole
    projected_names: tuple[str, ...]
    subset_policy_hint: str | None
    injectivity_proof: InjectiveTypeRegistryProof
    witness_name = AliasProperty[str]("surface_name")

    @property
    def surface_name(self) -> str:
        return self.surface_evidence.surface_name

    @property
    def shared_key_names(self) -> tuple[str, ...]:
        return self.surface_evidence.shared_key_names

    @property
    def shared_type_names(self) -> tuple[str, ...]:
        return self.surface_evidence.shared_type_names

    @property
    def projection_coverage_ratio(self) -> float:
        denominator, numerator = self.surface_kind.coverage_coordinates(
            self.injectivity_proof,
            self.surface_evidence,
        )
        return numerator / denominator

    @property
    def missing_key_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            frozenset(self.injectivity_proof.key_names)
            - frozenset(self.shared_key_names)
        )

    @property
    def missing_type_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            frozenset(self.injectivity_proof.registered_type_names)
            - frozenset(self.shared_type_names)
        )

    @property
    def projection_policy_name(self) -> str:
        return self.subset_policy_hint or "full"

    @property
    def projection_target_name(self) -> str:
        return f"{self.projection_role.value}:{self.surface_kind.value}"

    @property
    def materialization_rule(self) -> RegistryProjectionMaterialization:
        return self.surface_kind.materialization_for(self.projection_role)

    @property
    def decompression_key(self) -> str:
        return "|".join(
            (
                self.registry_class_name,
                self.key_type_name,
                self.projection_policy_name,
                self.projection_target_name,
                self.materialization_rule.value,
            )
        )

    @classmethod
    def from_registry_facts(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        facts: tuple[KeyedRegistryAxisFact, ...],
    ) -> tuple[Self, ...]:
        projections_by_path = {
            projection.file_path: projection for projection in projections
        }
        candidates: list[Self] = []
        for fact in facts:
            if not fact.is_mature_injective:
                continue
            registry_projection = projections_by_path.get(fact.file_path)
            if registry_projection is None:
                continue
            for projection in projections:
                if projection.file_path == fact.file_path:
                    import_aliases: dict[str, str] = {}
                else:
                    import_aliases = _compact_registry_projection_import_aliases(
                        projection,
                        registry_projection=registry_projection,
                        fact=fact,
                    )
                    if not (
                        fact.key_type_name in import_aliases.values()
                        or fact.class_name in import_aliases.values()
                        or frozenset(import_aliases.values())
                        & frozenset(fact.require_proof().registered_type_names)
                    ):
                        continue
                for surface in projection.named_projection_surfaces:
                    candidate = cls._from_named_surface(
                        surface,
                        fact=fact,
                        import_aliases=import_aliases,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
        return sorted_tuple(
            candidates,
            key=lambda item: (
                item.file_path,
                item.line,
                item.registry_class_name,
                item.surface_name,
            ),
        )

    policy_hint_terms: ClassVar[frozenset[str]] = frozenset(
        {
            "allow",
            "allowed",
            "deploy",
            "enabled",
            "experimental",
            "persisted",
            "public",
            "smoke",
            "stable",
            "supported",
        }
    )
    """Words in a surface's name that say it projects a policy-chosen subset."""

    @classmethod
    def policy_hint_for(cls, surface_name: str) -> str | None:
        lowered_name = surface_name.lower()
        return next(
            (
                term
                for term in sorted(cls.policy_hint_terms)
                if term in lowered_name
            ),
            None,
        )

    @classmethod
    def for_surface(
        cls,
        *,
        file_path: str,
        fact: KeyedRegistryAxisFact,
        evidence: RegistryProjectionSurfaceEvidence,
        line: int,
        surface_kind: RegistryProjectionSurfaceKind,
        projected_names: tuple[str, ...],
    ) -> Self | None:
        projection_role = RegistryProjectionRole.for_surface(
            evidence,
            file_path=file_path,
            default=surface_kind.default_role,
        )
        if projection_role is None:
            return None
        return cls(
            file_path=file_path,
            line=line,
            registry_class_name=fact.class_name,
            key_type_name=fact.key_type_name,
            surface_evidence=evidence,
            surface_kind=surface_kind,
            projection_role=projection_role,
            projected_names=projected_names,
            subset_policy_hint=cls.policy_hint_for(evidence.surface_name),
            injectivity_proof=fact.require_proof(),
        )

    @classmethod
    def _from_named_surface(
        cls,
        surface: CompactNamedProjectionSurface,
        *,
        fact: KeyedRegistryAxisFact,
        import_aliases: dict[str, str],
    ) -> Self | None:
        proof = fact.require_proof()
        if surface.sequence_references:
            reference_names = tuple(
                _compact_registry_projection_reference_name(reference, import_aliases)
                for reference in surface.sequence_references
            )
            shared_key_names = sorted_tuple(
                frozenset(reference_names) & frozenset(proof.key_names)
            )
            shared_type_names = sorted_tuple(
                frozenset(reference_names) & frozenset(proof.registered_type_names)
            )
            evidence = RegistryProjectionSurfaceEvidence(
                surface_name=surface.surface_name,
                shared_key_names=shared_key_names,
                shared_type_names=shared_type_names,
                has_key_to_type_pairs=False,
                has_type_to_key_pairs=False,
            )
            surface_kind = RegistryProjectionSurfaceKind.for_evidence(evidence)
            if surface_kind is None or len(shared_key_names) + len(shared_type_names) < 2:
                return None
            return cls.for_surface(
                file_path=surface.file_path,
                fact=fact,
                evidence=evidence,
                line=surface.line,
                surface_kind=surface_kind,
                projected_names=reference_names,
            )

        key_names = tuple(
            _compact_registry_projection_reference_name(reference, import_aliases)
            for reference in surface.dict_key_references
        )
        value_names = tuple(
            _compact_registry_projection_reference_name(reference, import_aliases)
            for reference in surface.dict_value_references
        )
        proof_key_names = frozenset(proof.key_names)
        proof_type_names = frozenset(proof.registered_type_names)
        shared_key_names = sorted_tuple(
            (frozenset(key_names) | frozenset(value_names)) & proof_key_names
        )
        shared_type_names = sorted_tuple(
            (frozenset(key_names) | frozenset(value_names)) & proof_type_names
        )
        has_key_to_type_pairs = bool(
            len(frozenset(key_names) & proof_key_names) >= 2
            and len(frozenset(value_names) & proof_type_names) >= 2
        )
        has_type_to_key_pairs = bool(
            len(frozenset(key_names) & proof_type_names) >= 2
            and len(frozenset(value_names) & proof_key_names) >= 2
        )
        evidence = RegistryProjectionSurfaceEvidence(
            surface_name=surface.surface_name,
            shared_key_names=shared_key_names,
            shared_type_names=shared_type_names,
            has_key_to_type_pairs=has_key_to_type_pairs,
            has_type_to_key_pairs=has_type_to_key_pairs,
        )
        surface_kind = RegistryProjectionSurfaceKind.for_evidence(evidence)
        if surface_kind is None or len(shared_key_names) + len(shared_type_names) < 3:
            return None
        return cls.for_surface(
            file_path=surface.file_path,
            fact=fact,
            evidence=evidence,
            line=surface.line,
            surface_kind=surface_kind,
            projected_names=(*key_names, *value_names),
        )


class KeyedRegistryAxisRootAnchored(TargetClassEvidenceAnchored):
    """Registry-axis findings are anchored at the class that owns the axis."""

    @classmethod
    def anchors_report(cls, projection: CompactModuleClassProjection) -> bool:
        return any(
            indexed_class.declares_keyed_base
            and indexed_class.constant_string_assignment("registry_key_attr") is not None
            for indexed_class in projection.classes
        )


class NamedRegistryProjectionSurfaceAnchored(TargetClassEvidenceAnchored):
    """Projection findings report their manual surface, not the joined registry."""

    @classmethod
    def anchors_report(cls, projection: CompactModuleClassProjection) -> bool:
        return bool(projection.named_projection_surfaces)


CompactKeyedRegistryCandidateT = TypeVar("CompactKeyedRegistryCandidateT")


class _CompactKeyedRegistryCandidateDetectorBase(
    SubscriptedCandidateType,
    CompactClassRepositoryProjectionMixin,
    CompactProjectionCandidateDetector[
        CompactModuleClassProjection,
        CompactKeyedRegistryCandidateT,
    ],
    Generic[CompactKeyedRegistryCandidateT],
    ABC,
):
    """Candidate detectors over the keyed registries the class repository indexes.

    The registries' facts come from the projections or from the shared context; a
    detector says what candidates those facts hold.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._require_candidate_source(
            cls._candidates_from_registry_facts,
            _CompactKeyedRegistryCandidateDetectorBase._candidates_from_registry_facts,
            CollectsItselfFromRegistryFacts.subscripted_in(cls),
        )

    def _candidates_from_registry_facts(
        self,
        projections: Sequence[CompactModuleClassProjection],
        facts: tuple[KeyedRegistryAxisFact, ...],
    ) -> Sequence[CompactKeyedRegistryCandidateT]:
        """The subscripted candidate type's own candidates in the registry facts."""

        candidate_type = CollectsItselfFromRegistryFacts.subscripted_in(type(self))
        assert candidate_type is not None
        return cast(
            Sequence[CompactKeyedRegistryCandidateT],
            candidate_type.from_registry_facts(projections, facts),
        )

    def _candidates_from_compact_projections(
        self,
        projections: Sequence[CompactModuleClassProjection],
        config: DetectorConfig,
    ) -> Sequence[CompactKeyedRegistryCandidateT]:
        return self._candidates_from_registry_facts(
            projections, _compact_keyed_registry_axis_facts(projections, config)
        )

    def _findings_from_compact_context(
        self,
        projections: Sequence[CompactModuleClassProjection],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        facts = _compact_keyed_registry_axis_facts_from_context(context)
        return self._findings_for_candidates(
            self._candidates_from_registry_facts(projections, facts), config
        )


class _CompactNonInjectiveTypeRegistryDetectorBase(
    KeyedRegistryAxisRootAnchored,
    _CompactKeyedRegistryCandidateDetectorBase[NonInjectiveTypeRegistryCandidate],
):
    """Registry-axis detectors over type registries whose keys are not proved injective."""


class _CompactInjectiveTypeRegistryDetectorBase(
    KeyedRegistryAxisRootAnchored,
    _CompactKeyedRegistryCandidateDetectorBase[InjectiveTypeRegistryCandidate],
):
    """Registry-axis detectors over type registries whose keys are proved injective."""


class _CompactRegistryProjectionSurfaceDetectorBase(
    NamedRegistryProjectionSurfaceAnchored,
    _CompactKeyedRegistryCandidateDetectorBase[RegistryProjectionSurfaceCandidate],
):
    """Registry detectors whose candidates are registry projection surface findings."""


@dataclass(frozen=True)
class RegistryProjectionPolicyAuthorityCandidate(CollectsItselfFromRegistryFacts, LineWitnessCandidate):
    registry_class_name: str
    key_type_name: str
    policy_hint: str
    surface_names: tuple[str, ...]
    surface_roles: tuple[RegistryProjectionRole, ...]
    projection_target_names: tuple[str, ...]
    materialization_rules: tuple[RegistryProjectionMaterialization, ...]
    decompression_keys: tuple[str, ...]
    file_paths: tuple[str, ...]
    line_numbers: tuple[int, ...]
    missing_key_names: tuple[str, ...]
    missing_type_names: tuple[str, ...]

    @classmethod
    def from_registry_facts(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        facts: tuple[KeyedRegistryAxisFact, ...],
    ) -> tuple[Self, ...]:
        return (
            cls.from_surfaces(
                RegistryProjectionSurfaceCandidate.from_registry_facts(
                    projections, facts
                )
            )
        )

    @classmethod
    def from_surfaces(
        cls,
        surface_candidates: Sequence[RegistryProjectionSurfaceCandidate],
    ) -> tuple[Self, ...]:
        grouped: dict[
            tuple[str, str, str], list[RegistryProjectionSurfaceCandidate]
        ] = defaultdict(list)
        for candidate in surface_candidates:
            if (
                candidate.projection_coverage_ratio >= 1.0
                or candidate.subset_policy_hint is None
            ):
                continue
            grouped[
                candidate.registry_class_name,
                candidate.key_type_name,
                candidate.subset_policy_hint,
            ].append(candidate)
        candidates: list[Self] = []
        for (
            registry_class_name,
            key_type_name,
            policy_hint,
        ), surfaces in sorted(grouped.items()):
            if len(surfaces) < 2:
                continue
            ordered = sorted_tuple(
                surfaces,
                key=lambda item: (item.file_path, item.line, item.surface_name),
            )
            candidates.append(
                cls(
                    file_path=ordered[0].file_path,
                    line=ordered[0].line,
                    registry_class_name=registry_class_name,
                    key_type_name=key_type_name,
                    policy_hint=policy_hint,
                    surface_names=tuple((surface.surface_name for surface in ordered)),
                    surface_roles=sorted_tuple(
                        {surface.projection_role for surface in ordered}
                    ),
                    projection_target_names=tuple(
                        (surface.projection_target_name for surface in ordered)
                    ),
                    materialization_rules=tuple(
                        (surface.materialization_rule for surface in ordered)
                    ),
                    decompression_keys=tuple(
                        (surface.decompression_key for surface in ordered)
                    ),
                    file_paths=tuple((surface.file_path for surface in ordered)),
                    line_numbers=tuple((surface.line for surface in ordered)),
                    missing_key_names=sorted_tuple(
                        {
                            key_name
                            for surface in ordered
                            for key_name in surface.missing_key_names
                        }
                    ),
                    missing_type_names=sorted_tuple(
                        {
                            type_name
                            for surface in ordered
                            for type_name in surface.missing_type_names
                        }
                    ),
                )
            )
        return tuple(candidates)


class _CompactRegistryProjectionPolicyAuthorityDetectorBase(
    NamedRegistryProjectionSurfaceAnchored,
    _CompactKeyedRegistryCandidateDetectorBase[
        RegistryProjectionPolicyAuthorityCandidate
    ],
):
    """Registry detectors whose candidates are registry projection policy authority findings."""

declare_candidate_rule_detector(
    NonInjectiveTypeRegistryCandidate,
    high_confidence_certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Type registry must map keys to types one-to-one",
        "A registry is only type-safe when each concrete implementation has one key and each key resolves to one implementation. Duplicate keys, duplicate types, or concrete subclasses with no key mean the registry can't be trusted as a one-to-one map.",
        "one-to-one type registry with one stable key per concrete implementation",
        "registry keys alias several implementation types or miss concrete subclasses",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` registry axis `{candidate.key_type_name}` is not injective: "
        f"duplicate keys {candidate.injectivity_proof.duplicate_key_names}, duplicate types "
        f"{candidate.injectivity_proof.duplicate_type_names}, missing keyed types {candidate.injectivity_proof.missing_type_names}."
    ),
    metrics=lambda candidate: RegistrationMetrics(
        registration_site_count=len(candidate.registered_case_names),
        registry_name=candidate.class_name,
    ),
    detector_base=_CompactNonInjectiveTypeRegistryDetectorBase,
)


declare_candidate_rule_detector(
    InjectiveTypeRegistryCandidate,
    high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Mature injective registry is a metaclass-registration candidate",
        "A registry with a stable key axis, a lookup lifecycle, many consumers and a proven one-to-one type-to-key map is ready for declaration-owned registration. Before switching to AutoRegisterMeta, check that dynamic registration timing and any outside plugin lifecycle still work.",
        "AutoRegisterMeta-backed ABC with a one-to-one type-key map",
        "the registry proves one key per implementation type, with mature lookup and many consumers, but the migration lifecycle is unproven",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` is a mature injective registry over `{candidate.key_type_name}`: "
        f"keys {candidate.registered_case_names}, lookup {candidate.lookup_method_names}, "
        f"consumers {candidate.consumer_symbols}; evaluate AutoRegisterMeta against its registration lifecycle."
    ),
    metrics=lambda candidate: RegistrationMetrics(
        registration_site_count=len(candidate.registered_case_names),
        registry_name=candidate.class_name,
    ),
    detector_base=_CompactInjectiveTypeRegistryDetectorBase,
)


declare_candidate_rule_detector(
    RegistryProjectionSurfaceCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual registry projections should derive from the registry",
        "Once a registry has one key per implementation type, export lists, key/type maps and option lists are projections of it. Maintaining them by hand creates shadow copies that can drift from the registry.",
        "generated projection derived from a one-to-one registry",
        "manual list or dict repeats keys and types the registry already proves",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.surface_name}` is a manual `{candidate.projection_role}` "
        f"`{candidate.surface_kind}` projection "
        f"of injective registry `{candidate.registry_class_name}` over `{candidate.key_type_name}`: "
        f"keys {candidate.shared_key_names}, types {candidate.shared_type_names}, "
        f"coverage {candidate.projection_coverage_ratio:.2f}; "
        f"target `{candidate.projection_target_name}`, "
        f"materialization `{candidate.materialization_rule}`, "
        f"decompression key `{candidate.decompression_key}`."
        + (
            f" Subset policy hint `{candidate.subset_policy_hint}` names the quotient; repeated use should be owned by a projection policy authority."
            if candidate.subset_policy_hint is not None
            and candidate.projection_coverage_ratio < 1.0
            else (
                f" Missing keys {candidate.missing_key_names} and types {candidate.missing_type_names} need a named projection policy."
                if candidate.projection_coverage_ratio < 1.0
                else ""
            )
        )
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.projected_names),
        mapping_name=candidate.surface_name,
        field_names=(
            candidate.registry_class_name,
            candidate.key_type_name,
            candidate.projection_policy_name,
            candidate.projection_target_name,
            candidate.materialization_rule.value,
        ),
    ),
    detector_base=_CompactRegistryProjectionSurfaceDetectorBase,
)


declare_candidate_rule_detector(
    RegistryProjectionPolicyAuthorityCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated registry subset projections should share one policy",
        "A partial projection of a one-to-one registry is a subset of its key axis. When several surfaces repeat the same subset hint, make it a real projection policy instead of separate allowlists.",
        "registry projection policy reused by generated subset surfaces",
        "several registry projections repeat the same subset hint with no single owner",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.registry_class_name}` has repeated `{candidate.policy_hint}` subset projections "
        f"{candidate.surface_names} across roles {tuple(role.value for role in candidate.surface_roles)}; move the quotient into one policy authority "
        f"and materialize targets {candidate.projection_target_names} from specs."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.surface_names),
        mapping_name=f"{candidate.policy_hint}_projection_policy",
        field_names=(
            candidate.registry_class_name,
            candidate.key_type_name,
            *(role.value for role in candidate.surface_roles),
            *(rule.value for rule in candidate.materialization_rules),
        ),
    ),
    detector_base=_CompactRegistryProjectionPolicyAuthorityDetectorBase,
)


class KeyedRecordInfrastructureDetector(
    ConfiguredModuleCollectorCandidateDetector[KeyedRecordInfrastructureCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual keyed-record infrastructure should derive from one substrate",
        "Repeated keyed records, indexes and lookup code create writable mappings next to the record declarations. Use one typed keyed-record base that derives indexes and lookup from the declared records.",
        "one typed keyed-record base with derived indexes",
        "keyed-record construction or registration and lookup code repeats across tables",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _finding_for_candidate(
        self, candidate: KeyedRecordInfrastructureCandidate
    ) -> RefactorFinding:
        return self.build_finding(
            candidate.finding_summary,
            candidate.evidence,
            metrics=candidate.finding_metrics,
        )


@dataclass(frozen=True)
class ManualStructuralRecordMechanicsGroupCandidate(CollectsItselfUnderConfig, EvidenceLocated):
    file_path: str
    base_names: tuple[str, ...]
    classes: tuple[ManualStructuralRecordMechanicsClassCandidate, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(item.location for item in self.classes[:6])

    @property
    def shared_method_names(self) -> tuple[str, ...]:
        return self._shared_record_mechanics_method_names(self.classes)

    @property
    def transform_method_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            {
                method_name
                for candidate in self.classes
                for method_name in candidate.transform_method_names
            }
        )

    @classmethod
    def from_module(cls, module: ParsedModule, config: DetectorConfig) -> tuple[Self, ...]:
        threshold = max(3, config.min_registration_sites)
        classes: list[ManualStructuralRecordMechanicsClassCandidate] = []
        for declared in module.fact(ModuleClassStatements).top_level:
            node = declared.node
            if not declared.is_dataclass or declared.is_abstract:
                continue
            base_names = declared.family_base_names
            if not base_names:
                continue
            validate_method = declared.method_named("validate")
            if validate_method is None or cls._validation_guard_count(validate_method.node) < 3:
                continue
            projection_method_names = sorted_tuple(
                (
                    method.name
                    for method in declared.methods
                    if cls._returns_tuple_of_self_attributes(method.node)
                )
            )
            if not projection_method_names:
                continue
            roundtrip_method_names = cls._same_type_constructor_method_names(
                declared, include_classmethods=True, include_instance_methods=False
            )
            if not roundtrip_method_names:
                continue
            transform_method_names = tuple(
                (
                    method_name
                    for method_name in cls._same_type_constructor_method_names(
                        declared, include_classmethods=False, include_instance_methods=True
                    )
                    if method_name != "validate"
                )
            )
            if not transform_method_names:
                continue
            classes.append(
                ManualStructuralRecordMechanicsClassCandidate(
                    file_path=module.file_path,
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
            (
                cls(
                    file_path=module.file_path,
                    base_names=base_names,
                    classes=sorted_tuple(
                        items, key=lambda item: (item.line, item.class_name)
                    ),
                )
                for base_names, items in sorted(grouped.items())
                if len(items) >= threshold
                if set.intersection(*(set(item.projection_method_names) for item in items))
                if set.intersection(*(set(item.roundtrip_method_names) for item in items))
            )
        )

    @staticmethod
    def _same_type_constructor_method_names(
        declared: ClassStatement,
        *,
        include_classmethods: bool,
        include_instance_methods: bool,
    ) -> tuple[str, ...]:
        class_name = declared.node.name
        accepted_instance_names = (class_name,)
        accepted_class_names = ("cls", class_name)
        names: list[str] = []
        for method in declared.methods:
            if method.spells("classmethod"):
                if include_classmethods and _returns_constructor_call(
                    method.node, accepted_names=accepted_class_names
                ):
                    names.append(method.name)
                continue
            if include_instance_methods and _returns_constructor_call(
                method.node, accepted_names=accepted_instance_names
            ):
                names.append(method.name)
        return sorted_tuple(set(names))

    @staticmethod
    def _returns_tuple_of_self_attributes(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        returned = DISPATCH_ALGEBRA_AUTHORITY.single_return_case(method.body)
        if returned is None:
            return False
        return_value, _ = returned
        return isinstance(return_value, ast.Tuple) and all(
            (
                isinstance(item, ast.Attribute)
                and isinstance(item.value, ast.Name)
                and (item.value.id == "self")
                for item in return_value.elts
            )
        )

    @staticmethod
    def _validation_guard_count(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int:
        count = 0
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute) and node.attr in {"ndim", "shape"}:
                count += 1
            if isinstance(node, ast.Compare) and any(
                (
                    isinstance(operator, (ast.Lt, ast.LtE, ast.NotEq))
                    for operator in node.ops
                )
            ):
                count += 1
        return count

    @staticmethod
    def _shared_record_mechanics_method_names(
        candidates: Sequence[ManualStructuralRecordMechanicsClassCandidate],
    ) -> tuple[str, ...]:
        shared_projection_method_names = set.intersection(
            *(set(candidate.projection_method_names) for candidate in candidates)
        )
        shared_roundtrip_method_names = set.intersection(
            *(set(candidate.roundtrip_method_names) for candidate in candidates)
        )
        return sorted_tuple(
            {"validate"} | shared_projection_method_names | shared_roundtrip_method_names
        )


class ManualStructuralRecordMechanicsDetector(
    ConfiguredModuleCollectorCandidateDetector[
        ManualStructuralRecordMechanicsGroupCandidate
    ]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated structural record mechanics should derive from field metadata",
        "Several frozen dataclass records hand-write validation, tuple-style field projection, round-trip reconstruction and per-field transforms. That code is now a second source of truth next to the field declarations. Derive it from the typed fields with one metadata-driven record base.",
        "one typed record base that derives validation, projection and transforms",
        "the same dataclass record lifecycle code is repeated across sibling record classes",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
            CapabilityTag.TYPE_LINEAGE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.BUILDER_CALL,
        ),
    )

    def _finding_for_candidate(
        self, group_candidate: ManualStructuralRecordMechanicsGroupCandidate
    ) -> RefactorFinding:
        class_names = ", ".join(
            (item.class_name for item in group_candidate.classes[:6])
        )
        shared_methods = ", ".join(group_candidate.shared_method_names)
        transform_methods = ", ".join(group_candidate.transform_method_names[:6])
        base_names = ", ".join(group_candidate.base_names)
        evidence = group_candidate.evidence
        return self.build_finding(
            (
                f"Dataclass records {class_names} each hand-roll `{shared_methods}` plus fieldwise transforms "
                f"{transform_methods} on top of base family `{base_names}`."
            ),
            evidence,
        )


class RepeatedConcreteTypeCaseAnalysisDetector(
    CompactClassIndexMultiProjectionDetector,
    CompactMultiProjectionCandidateDetector[
        RepeatedConcreteTypeCaseAnalysisCandidate
    ],
):
    module_projection_families = (
        CompactRemainingSystemicModuleProjectionFamily,
        CompactModuleClassProjectionFamily,
    )
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Repeated concrete-type recovery should become class-family behavior",
        "Several functions recover the same family with concrete `isinstance` checks on one carried attribute, so the family boundary is still implicit. Use one ABC and expose leaf behavior through typed properties or hooks instead of decoding leaves repeatedly.",
        "one ABC-backed family for the carried subject, with case recovery moved into properties or hooks",
        "the same attribute-carried family is decoded again with concrete runtime type checks across several functions",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _candidates_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: object | None,
        config: DetectorConfig,
    ) -> Sequence[RepeatedConcreteTypeCaseAnalysisCandidate]:
        return _compact_repeated_concrete_type_case_candidates(
            CompactRemainingSystemicModuleProjectionFamily.modules_in(projections_by_family),
            CompactModuleClassProjectionFamily.modules_in(projections_by_family),
            config,
            class_index=CompactClassFamilyIndex.require(context),
        )

    def _finding_for_candidate(
        self, case_candidate: RepeatedConcreteTypeCaseAnalysisCandidate
    ) -> RefactorFinding:
        function_names = ", ".join(
            (function.fact.function_name for function in case_candidate.functions[:6])
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
        shared_suffix = CLASS_NAME_ALGEBRA.longest_common_suffix(
            case_candidate.concrete_class_names
        )
        if (
            shared_suffix
            and len(shared_suffix) >= 6
            and not suggested_family_name.endswith(shared_suffix)
        ):
            suggested_family_name = f"{suggested_family_name}{shared_suffix}"
        elif not suggested_family_name.endswith(("Family", "Witness", "Variant")):
            suggested_family_name = f"{suggested_family_name}Family"
        return self.build_finding(
            (
                f"Functions {function_names} repeatedly recover `{case_candidate.subject_role}` across concrete classes {class_names}.{alias_summary}{existing_base_summary}"
            ),
            case_candidate.evidence,
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(case_candidate.functions),
                dispatch_axis=case_candidate.subject_role,
                literal_cases=case_candidate.concrete_class_names,
            ),
        )


class ImplicitSelfContractMixinDetector(
    CompactClassIndexMultiProjectionDetector,
    CompactMultiProjectionCandidateDetector[ImplicitSelfContractMixinCandidate],
):
    module_projection_families = (
        CompactRemainingSystemicModuleProjectionFamily,
        CompactModuleClassProjectionFamily,
    )
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Concrete mixins should not hide consumer contracts behind `self`-casts",
        "Mixins are for independent reusable concerns that fit cleanly in the MRO. A concrete mixin that uses `cast(..., self)` to reach consumer-owned fields is carrying family logic through a hidden contract. Declare a base or policy instead.",
        "declared base or policy row for the shared algorithm instead of a hidden mixin self-contract",
        "concrete mixin methods cast `self` and depend on consumer-owned attributes across several subclasses",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REPEATED_METHOD_ROLES,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _candidates_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: object | None,
        config: DetectorConfig,
    ) -> Sequence[ImplicitSelfContractMixinCandidate]:
        return _compact_implicit_self_contract_mixin_candidates(
            CompactRemainingSystemicModuleProjectionFamily.modules_in(projections_by_family),
            CompactModuleClassProjectionFamily.modules_in(projections_by_family),
            config,
            class_index=CompactClassFamilyIndex.require(context),
        )

    def _finding_for_candidate(
        self, mixin_candidate: ImplicitSelfContractMixinCandidate
    ) -> RefactorFinding:
        fact = mixin_candidate.fact
        methods = ", ".join(fact.method_names)
        consumers = ", ".join(
            consumer.symbol for consumer in mixin_candidate.consumers[:6]
        )
        accessed_attributes = ", ".join(fact.accessed_attribute_names[:6])
        cast_types = ", ".join(fact.cast_type_names[:6])
        return self.build_finding(
            (
                f"`{mixin_candidate.mixin_name}` uses `cast(..., self)` ({cast_types}) in `{methods}` to reach consumer-owned attributes ({accessed_attributes}) across subclasses {consumers}."
            ),
            mixin_candidate.evidence,
            metrics=HierarchyCandidateMetrics(
                duplicate_group_count=len(fact.method_names),
                class_count=len(mixin_candidate.consumers) + 1,
            ),
        )


@dataclass(frozen=True)
class RepeatedResultAssemblyPipelineCandidate(CollectsItselfUnderConfig, EvidenceLocated):
    file_path: str
    shared_tail: tuple[PipelineAssemblyStage, ...]
    functions: tuple[ResultAssemblyPipelineFunction, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(function.location for function in self.functions[:6])

    @classmethod
    def from_module(cls, module: ParsedModule, config: DetectorConfig) -> tuple[Self, ...]:
        functions = _result_assembly_pipeline_functions(module)
        if len(functions) < 2:
            return ()
        grouped_functions: dict[
            (
                tuple[tuple[object, ...], ...],
                tuple[
                    tuple[PipelineAssemblyStage, ...], set[ResultAssemblyPipelineFunction]
                ],
            )
        ] = {}
        for left, right in combinations(functions, 2):
            shared_tail = _shared_pipeline_tail(left, right)
            if len(shared_tail) < config.min_shared_pipeline_stages:
                continue
            if len(shared_tail) >= len(left.stages) or len(shared_tail) >= len(
                right.stages
            ):
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
            cls(
                file_path=module.file_path,
                shared_tail=shared_tail,
                functions=sorted_tuple(
                    grouped, key=lambda item: (item.lineno, item.qualname)
                ),
            )
            for shared_tail, grouped in grouped_functions.values()
            if len(grouped) >= 2
        ]
        filtered_candidates: list[Self] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                -len(item.shared_tail),
                -len(item.functions),
                item.functions[0].qualname,
            ),
        ):
            candidate_function_names = tuple(
                (function.qualname for function in candidate.functions)
            )
            if any(
                (
                    len(existing.shared_tail) >= len(candidate.shared_tail)
                    and candidate_function_names
                    == tuple((function.qualname for function in existing.functions))
                    for existing in filtered_candidates
                )
            ):
                continue
            filtered_candidates.append(candidate)
        return tuple(filtered_candidates)


class RepeatedResultAssemblyPipelineDetector(
    ConfiguredModuleCollectorCandidateDetector[RepeatedResultAssemblyPipelineCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Repeated result-assembly pipeline should collapse into one authoritative assembler",
        "Several owners repeat the same downstream result-assembly steps and differ only in the upstream source or projection. Move the steps into one template method or helper with one source hook.",
        "one result-assembly pipeline with one source hook",
        "the same assembly tail is repeated across sibling functions or methods",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _finding_for_candidate(
        self, pipeline_candidate: RepeatedResultAssemblyPipelineCandidate
    ) -> RefactorFinding:
        function_names = ", ".join(
            (function.qualname for function in pipeline_candidate.functions[:4])
        )
        stage_names = ", ".join(
            (stage.callee_name for stage in pipeline_candidate.shared_tail)
        )
        evidence = pipeline_candidate.evidence
        return self.build_finding(
            (
                f"Functions {function_names} share the same result-assembly tail "
                f"{stage_names} and differ only in their leading source stages."
            ),
            evidence,
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


class CandidateCollectorBoilerplateDetectorBase(
    ModuleCollectorCandidateDetector[CandidateCollectorBoilerplateCandidate],
    CandidateCollectorBoilerplateFindingRecipeSynthesizer,
):
    """Compose collector-boilerplate detection with its proved refactor."""


declare_candidate_rule_detector(
    CandidateCollectorBoilerplateCandidate,
    high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Candidate detector should declare collector strategy",
        "Detector classes repeat `_candidate_items()` as a one-line forwarder. That is boilerplate. The detector's identity and finding rendering matter, and candidate collection is a typed class-level strategy that can be inherited.",
        "typed detector base that derives candidate collection from a declared strategy",
        "detector class repeats a collector forwarding method instead of declaring a collector",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda collector: (
        f"`{collector.class_name}.{collector.method_name}` only forwards to `{collector.collector_name}`; inherit `{collector.recommended_base_name}` and declare `candidate_collector` instead."
    ),
    metrics=lambda collector: OrchestrationMetrics(
        function_line_count=0,
        branch_site_count=1,
        call_site_count=1,
        parameter_count=2 if collector.uses_config else 1,
        callee_family_count=1,
    ),
    detector_base=CandidateCollectorBoilerplateDetectorBase,
)


@dataclass(frozen=True)
class TypedCandidateCastBoilerplateCandidate(
    ModuleCollectedLineWitnessCandidate, ClassMethodLineWitnessCandidate
):
    parameter_name: str
    local_name: str
    candidate_type_name: str
    detector_base_name: str

    @classmethod
    def from_class(cls, module: ParsedModule, node: ast.ClassDef) -> tuple[Self, ...]:
        detector_shape = ConcreteCandidateDetectorShape.from_class(module, node)
        if detector_shape is None:
            return ()
        candidates: list[Self] = []
        for statement in node.body:
            if not (
                isinstance(statement, ast.FunctionDef)
                and statement.name == "_finding_for_candidate"
            ):
                continue
            cast_assignment = _typed_candidate_cast_assignment(statement)
            if cast_assignment is None:
                continue
            parameter_name, local_name, candidate_type_name = cast_assignment
            candidates.append(
                cls(
                    file_path=module.file_path,
                    line=statement.lineno,
                    class_name=node.name,
                    method_name=statement.name,
                    parameter_name=parameter_name,
                    local_name=local_name,
                    candidate_type_name=candidate_type_name,
                    detector_base_name=detector_shape.base_name,
                )
            )
        return tuple(candidates)


declare_candidate_rule_detector(
    TypedCandidateCastBoilerplateCandidate,
    high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Candidate template method should receive typed candidates directly",
        "Detector classes take `candidate: object`, cast it to a candidate type right away, and never use the object-typed parameter again. Put that cast in the generic detector base so the hook receives the typed candidate.",
        "generic typed candidate detector base with no per-detector cast",
        "candidate-rendering method starts by casting its only parameter",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}.{candidate.method_name}` casts `{candidate.parameter_name}` to `{candidate.candidate_type_name}` before doing real work; parameterize `{candidate.detector_base_name}` and receive `{candidate.local_name}` as that type."
    ),
    metrics=lambda candidate: _SINGLE_TEMPLATE_CALL_METRICS,
)


class DeclarativeDetectorClassDetectorBase(
    ModuleCollectorCandidateDetector[DeclarativeDetectorClassCandidate],
    DeclarativeDetectorClassFindingRecipeSynthesizer,
):
    """Compose declarative-shell detection with its executable replacement."""


DeclarativeDetectorClassDetector = declare_candidate_rule_detector(
    DeclarativeDetectorClassCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Metadata-only detector class should be declared through the detector helper",
        "A detector class whose body only assigns finding metadata and a renderer has no behavior of its own. Its class shell can be derived from the candidate type, detector base, registry key and declaration line.",
        "one detector declaration helper that derives metadata-only detector classes",
        "detector class repeats a class shell around only declarative assignments",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` is a {candidate.line_count}-line metadata-only detector over `{candidate.candidate_type_name}` with assignments {candidate.assignment_names}."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.line_count,
        mapping_name=candidate.class_name,
        field_names=candidate.assignment_names,
        source_name=candidate.base_name,
    ),
    detector_base=DeclarativeDetectorClassDetectorBase,
)


@dataclass(frozen=True)
class StaticTypedObservationDetectorCandidate(CollectsItself, ClassLineWitnessCandidate):
    observation_family_name: str
    observation_type_name: str
    minimum_evidence_count: int
    summary_expression: str
    line_count: int

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        candidates: list[Self] = []
        for declared in module.fact(ModuleClassStatements).top_level:
            node = declared.node
            if node.end_lineno is None:
                continue
            if not (
                _STATIC_OBSERVATION_DETECTOR_BASE_NAMES
                & set(declared.base_terminal_names)
            ):
                continue
            evidence_statement = declared.method_named("_module_evidence")
            if evidence_statement is None:
                continue
            evidence_method = evidence_statement.node
            collection = _typed_observation_collection_call(evidence_method)
            if collection is None:
                continue
            if not any(
                _source_location_from_line_symbol_call(child)
                for child in ast.walk(evidence_method)
            ):
                continue
            minimum_evidence = _static_minimum_evidence_count(declared)
            summary_expression = _static_summary_expression(module, declared)
            if minimum_evidence is None or summary_expression is None:
                continue
            family_name, observation_type_name = collection
            candidates.append(
                cls(
                    file_path=module.file_path,
                    line=node.lineno,
                    class_name=node.name,
                    observation_family_name=family_name,
                    observation_type_name=observation_type_name,
                    minimum_evidence_count=minimum_evidence,
                    summary_expression=summary_expression,
                    line_count=node.end_lineno - node.lineno + 1,
                )
            )
        return tuple(candidates)


declare_candidate_rule_detector(
    StaticTypedObservationDetectorCandidate,
    high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Static observation detector should derive from typed observation base",
        "A static detector whose evidence method only collects one typed observation family and maps its line and symbol into `SourceLocation` repeats the same module-observation code. Have the detector declare the observation family, item type, evidence threshold and summary template, and let the ABC do collection and evidence projection.",
        "typed observation detector base with one shared collection and projection algorithm",
        "detector class repeats collect, map and summary code for one observation family",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` repeats a {candidate.line_count}-line static observation shell over `{candidate.observation_family_name}` / `{candidate.observation_type_name}`."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.line_count,
        mapping_name=candidate.class_name,
        field_names=(
            "finding_spec",
            "observation_family",
            "observation_type",
            "minimum_evidence_count",
            "summary_template",
        ),
        source_name=candidate.observation_family_name,
    ),
)


@dataclass(frozen=True)
class SchemaAccessorFamilyCandidate(CollectsItself, 
    FieldFamilyCompressionSurface, ClassLineWitnessCandidate
):
    enum_name: str
    method_names: tuple[str, ...]
    requirement_modes: tuple[str, ...]
    coercion_kinds: tuple[str, ...]
    evidence: ClassVar[ZippedSourceLocationEvidenceProperty] = (
        ZippedSourceLocationEvidenceProperty("line_numbers", "method_names")
    )

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        candidates: list[Self] = []
        for declared in module.fact(ModuleClassStatements).statements:
            class_node = declared.node
            grouped_rows: dict[str, list[tuple[str, str, str, str, str, int, int]]] = (
                defaultdict(list)
            )
            for method in declared.methods:
                row = _schema_accessor_method_row(method.node)
                if row is not None:
                    grouped_rows[row[1]].append(row)
            for enum_name, rows in grouped_rows.items():
                if len(rows) < _SCHEMA_ACCESSOR_MIN_METHODS:
                    continue
                ordered_rows = tuple(sorted(rows, key=lambda item: (item[5], item[0])))
                field_names = tuple((row[2] for row in ordered_rows))
                if len(frozenset(field_names)) < _SCHEMA_ACCESSOR_MIN_METHODS:
                    continue
                axis_system = _schema_accessor_axis_system(ordered_rows)
                if not axis_system.determines(("field",), "method"):
                    continue
                line_count = sum((row[6] for row in ordered_rows))
                coercion_kinds = tuple((row[4] for row in ordered_rows))
                field_axes = tuple(
                    (f"{enum_name}.{field_name}" for field_name in field_names)
                )
                certificate = _schema_accessor_family_certificate(
                    method_count=len(ordered_rows),
                    line_count=line_count,
                    field_axes=field_axes,
                    coercion_kind_count=len(frozenset(coercion_kinds)),
                )
                if not certificate.pays_rent:
                    continue
                candidates.append(
                    cls(
                        file_path=module.file_path,
                        line=class_node.lineno,
                        class_name=class_node.name,
                        enum_name=enum_name,
                        method_names=tuple((row[0] for row in ordered_rows)),
                        field_names=field_names,
                        requirement_modes=tuple((row[3] for row in ordered_rows)),
                        coercion_kinds=coercion_kinds,
                        line_numbers=tuple((row[5] for row in ordered_rows)),
                        line_count=line_count,
                        compression_certificate=certificate,
                    )
                )
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.file_path,
                    item.line,
                    item.class_name,
                    item.enum_name,
                ),
            )
        )


declare_candidate_rule_detector(
    SchemaAccessorFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Accessor family should derive from one projection schema",
        "Several public methods on one class fetch one enum-keyed payload field with `self.required(...)` or `self.optional(...)`, then repeat local guards or coercions before returning. That is a field schema spread across methods. Declare the enum member, requiredness, accepted type, coercion and error policy once and project through a typed accessor.",
        "one projection schema for enum-keyed payload fields",
        "the same class repeats payload accessor methods over one closed enum or key axis",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda family: (
        f"`{family.class_name}` repeats {len(family.method_names)} accessor methods "
        f"over `{family.enum_name}` fields {family.field_names}; requiredness "
        f"{family.requirement_modes} and coercions {family.coercion_kinds} are schema rows."
    ),
    compression_certificate=lambda family: family.compression_certificate,
    metrics=lambda family: MappingMetrics.from_field_names(
        mapping_site_count=len(family.method_names),
        mapping_name=family.class_name,
        field_names=family.field_names,
        source_name=family.enum_name,
    ),
)


declare_candidate_rule_detector(
    TupleIndexSemanticOpacityCandidate,
    high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Positional tuple context should become a named record",
        "A typed carrier pipeline that reads context as `pair[0]`, `pair[1]` or nested numeric tuple paths fixed the control-flow smell but added a positional-data smell. Use a named record, an owner-held context object or a step result type whose field names carry the invariant.",
        "named effect context record instead of positional tuple plumbing",
        "effect pipeline keeps its context in numeric tuple indexes",
        (
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.function_name}` uses positional tuple paths "
        f"{candidate.index_expressions} inside carrier pipeline calls "
        f"{candidate.carrier_call_names}."
    ),
    compression_certificate=lambda candidate: CompressionCertificate.from_object_family(
        manual_object_count=candidate.nested_index_count
        + len(candidate.index_expressions),
        replacement_shape=ObjectFamilyShape(shared_objects=("named_effect_context",)),
        semantic_axes=candidate.index_expressions,
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.nested_index_count,
        mapping_name=candidate.function_name,
        field_names=candidate.index_expressions,
        source_name="carrier_tuple_context",
    ),
    candidate_collector=TupleIndexSemanticOpacityCandidateCollector.collect,
    source_candidate_collector=(
        TupleIndexSemanticOpacityCandidateCollector.collect_source
    ),
    detector_base=SourceModuleCollectorCandidateDetector,
)


class FindingSpecConstructionBoilerplateDetector(
    ModuleCollectorCandidateDetector[FindingSpecConstructionCandidate]
):
    candidate_collector = FindingSpecConstructionCandidateCollector.collect
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "FindingSpec construction should use its typed factory",
        "FindingSpec factories own the tier defaults and the canonical argument order. Calling the constructor directly repeats that schema, those defaults, or both, at every detector declaration.",
        "one typed FindingSpec factory per tier",
        "FindingSpec is constructed directly instead of through its tier factory",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _finding_for_candidate(
        self, field_candidate: FindingSpecConstructionCandidate
    ) -> RefactorFinding:
        redundant_defaults = (
            ", ".join(
                (
                    f"{name}={value}"
                    for name, value in zip(
                        field_candidate.redundant_keyword_names,
                        field_candidate.redundant_keyword_values,
                        strict=True,
                    )
                )
            )
            or "no explicit semantic defaults"
        )
        return self.build_finding(
            (
                f"`{field_candidate.constructor_name}` constructs FindingSpec directly "
                f"with {redundant_defaults}; use the semantic-tier factory "
                f"`{field_candidate.recommended_builder_name}`."
            ),
            field_candidate.evidence,
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(field_candidate.redundant_keyword_names),
                mapping_name=field_candidate.constructor_name,
                field_names=field_candidate.redundant_keyword_names,
            ),
        )


declare_candidate_rule_detector(
    ClassMethodLineWitnessCandidate,
    high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Detector finding builder should derive detector_id",
        "Concrete detectors repeatedly call `self.finding_spec.build(self.detector_id, ...)`. The detector id belongs to the detector instance, not to each finding. A shared `build_finding(...)` hook should inject it once.",
        "typed detector template method that injects the detector id when building findings",
        "finding renderer passes the detector's own id into its spec builder by hand",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.symbol}` calls `self.finding_spec.build(self.detector_id, ...)`; `build_finding(...)` can derive the detector id from the instance."
    ),
    metrics=lambda candidate: _SINGLE_TEMPLATE_CALL_METRICS,
    detector_name="FindingSpecBuildBoilerplateDetector",
    candidate_collector=_finding_spec_build_boilerplate_candidates,
)


class DirectBuildFindingRendererDetectorBase(
    ModuleCollectorCandidateDetector[DirectBuildFindingRendererCandidate],
    DirectBuildFindingRendererFindingRecipeSynthesizer,
):
    """Compose direct-renderer detection with its executable replacement."""


DirectBuildFindingRendererDetector = declare_candidate_rule_detector(
    DirectBuildFindingRendererCandidate,
    high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Direct build_finding renderer should be a typed renderer value",
        "A `_finding_for_candidate` method whose whole body is `return self.build_finding(...)` has no control flow. It is a data renderer for one candidate type. Put the candidate-to-finding algorithm once in the ABC and have the detector supply a typed renderer.",
        "typed candidate finding renderer reused by the detector ABC",
        "detector method is only a build_finding payload declaration",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.METHOD_ROLE,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    ),
    summary=lambda renderer: (
        f"`{renderer.class_name}.{renderer.method_name}` is a direct "
        f"`build_finding(...)` renderer with {renderer.positional_arg_count} "
        f"positional payloads and "
        f"{', '.join(renderer.keyword_names) or 'no keywords'}."
    ),
    metrics=lambda renderer: MappingMetrics.from_field_names(
        mapping_site_count=1,
        mapping_name=renderer.class_name,
        field_names=("summary", "evidence", *renderer.keyword_names),
    ),
    detector_base=DirectBuildFindingRendererDetectorBase,
)


@dataclass(frozen=True)
class EnumMetadataTableCandidate(CollectsItself, ClassLineWitnessCandidate):
    table_name: str
    property_names: tuple[str, ...]
    case_count: int

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        table_cases = _enum_metadata_table_cases(module)
        candidates: list[Self] = []
        for declared in module.fact(ModuleClassStatements).top_level:
            statement = declared.node
            property_tables: dict[str, list[str]] = defaultdict(list)
            for item in statement.body:
                if isinstance(item, ast.FunctionDef):
                    table_name = _enum_metadata_property_table(item)
                    if table_name is not None:
                        property_tables[table_name].append(item.name)
            for table_name, property_names in property_tables.items():
                enum_name, case_count = table_cases.get(table_name, (None, 0))
                if enum_name == statement.name:
                    candidates.append(
                        cls(
                            file_path=module.file_path,
                            line=statement.lineno,
                            class_name=statement.name,
                            table_name=table_name,
                            property_names=tuple(property_names),
                            case_count=case_count,
                        )
                    )
        return tuple(candidates)


declare_candidate_rule_detector(
    EnumMetadataTableCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Enum metadata table should live on the enum members",
        "An enum whose properties only index a module-level table by `self` splits member identity from member metadata. Move the metadata into enum construction so each member carries its own typed record.",
        "enum member construction owns the member metadata",
        "enum properties read a parallel metadata table keyed by the same enum",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.EXPORT_MAPPING,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` reads {candidate.property_names} from `{candidate.table_name}` across {candidate.case_count} enum cases."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.case_count,
        mapping_name=candidate.table_name,
        field_names=candidate.property_names,
        source_name=candidate.class_name,
    ),
)


class CompactDataclassNamespaceCliMirrorCandidateBase(
    TargetProjectionRequired[_DataclassNamespaceCliModuleProjection],
    CompactProjectionCandidateDetector[
        _DataclassNamespaceCliModuleProjection,
        DataclassNamespaceCliMirrorCandidate,
    ],
):
    module_projection_family = _DataclassNamespaceCliModuleProjectionFamily

    def _candidates_from_compact_projections(
        self,
        projections: Sequence[_DataclassNamespaceCliModuleProjection],
        config: DetectorConfig,
    ) -> Sequence[DataclassNamespaceCliMirrorCandidate]:
        del config
        return _dataclass_namespace_cli_mirror_candidates_from_projections(projections)


declare_candidate_rule_detector(
    DataclassNamespaceCliMirrorCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Dataclass config should derive its namespace and CLI adapters",
        "A dataclass already owns its field names and defaults. Listing them again in a namespace constructor and an argparse table makes parallel config that can drift from the record.",
        "one dataclass field list that derives namespace construction and CLI argument rows",
        "dataclass fields are mirrored by hand in both from-namespace construction and CLI argument specs",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.config_class.class_name}` mirrors {len(candidate.field_names)} namespace fields and {len(candidate.cli_field_names)} CLI fields through `{candidate.argument_spec.name}` instead of deriving adapters from the dataclass."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.field_names) + len(candidate.cli_field_names),
        mapping_name=candidate.config_class.class_name,
        field_names=candidate.field_names,
        source_name=candidate.argument_spec.name,
    ),
    detector_base=CompactDataclassNamespaceCliMirrorCandidateBase,
)


declare_candidate_rule_detector(
    ManualFiberTagCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Manual variant tag should become a class family",
        "A string tag on an instance picks behavior by hand, while the same instance still carries fields from several incompatible variants. The class family stays inconsistent and allows states the type system could rule out.",
        "native class decomposition with one subclass per behavior variant",
        "a manual instance tag drives behavior while fields for the other variants stay constructible on every variant",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
    ),
    summary=lambda fiber_candidate: (
        f"`{fiber_candidate.class_name}` branches on manual fiber tag `self.{fiber_candidate.tag_name}` across {fiber_candidate.case_names} while still carrying cross-fiber fields {fiber_candidate.assigned_field_names}."
    ),
    metrics=lambda fiber_candidate: DispatchCountMetrics.from_literal_family(
        dispatch_axis=f"self.{fiber_candidate.tag_name}",
        literal_cases=fiber_candidate.case_names,
    ),
)


@dataclass(frozen=True)
class StructuralConfusabilityCandidate(CollectsItself, 
    WitnessCarrierCandidate, NameFamilyClassNamesMixin, SubjectNameFunctionNameMixin
):
    parameter_name: str
    observed_method_names: tuple[str, ...]

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        statements = module.fact(ModuleClassStatements)
        return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
            module,
            cls._for_function,
            statements.top_level,
            statements.top_level_by_name,
        )

    @classmethod
    def _for_function(
        cls,
        module: ParsedModule,
        qualname: str,
        function: NamedFunctionNode,
        classes: Sequence[ClassStatement],
        class_lookup: dict[str, ClassStatement],
    ) -> Iterable[Self]:
        annotated_parameter_names = SUPPORT_PROJECTION_AUTHORITY.annotated_parameter_names(
            function
        )
        for parameter_name in SUPPORT_PROJECTION_AUTHORITY.parameter_names(function):
            if parameter_name in annotated_parameter_names:
                continue
            observed_method_names = sorted_tuple(
                {
                    subnode.func.attr
                    for subnode in ast.walk(function)
                    if isinstance(subnode, ast.Call)
                    and isinstance(subnode.func, ast.Attribute)
                    and isinstance(subnode.func.value, ast.Name)
                    and (subnode.func.value.id == parameter_name)
                }
            )
            if len(observed_method_names) < 2:
                continue
            confusable_classes = tuple(
                declared
                for declared in classes
                if set(observed_method_names)
                <= declared.first_methods_by_name.keys()
            )
            if len(confusable_classes) < 2:
                continue
            if cls._shared_abstract_nominal_authority(
                confusable_classes, class_lookup=class_lookup
            ):
                continue
            yield cls(
                file_path=module.file_path,
                line=function.lineno,
                subject_name=qualname,
                name_family=tuple((declared.node.name for declared in confusable_classes)),
                parameter_name=parameter_name,
                observed_method_names=observed_method_names,
            )

    @staticmethod
    def _shared_abstract_nominal_authority(
        classes: tuple[ClassStatement, ...],
        *,
        class_lookup: dict[str, ClassStatement],
    ) -> bool:
        def abstract_lineage_names(declared: ClassStatement) -> set[str]:
            lineage: set[str] = set()
            seen: set[str] = set()
            stack = [declared.node.name]
            while stack:
                current_name = stack.pop()
                if current_name in seen or not (
                    ClassSymbolResolutionAuthority.establishes_nominal_family(current_name)
                ):
                    continue
                seen.add(current_name)
                current_class = class_lookup.get(current_name)
                if current_class is None:
                    continue
                if current_class.is_abstract:
                    lineage.add(current_name)
                stack.extend(
                    (
                        base_name
                        for base_name in current_class.base_terminal_names
                        if base_name not in seen
                    )
                )
            return lineage

        lineage_sets = [abstract_lineage_names(declared) for declared in classes]
        if not lineage_sets or any((not lineage for lineage in lineage_sets)):
            return False
        return bool(set.intersection(*lineage_sets))


class StructuralConfusabilityDetector(
    ModuleCollectorCandidateDetector[StructuralConfusabilityCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Consumer sees a look-alike duck-typed family",
        "A consumer sees only part of the structure, and several unrelated classes look the same through that view. Without a nominal base to tell them apart, the family boundary stays implicit.",
        "ABC-backed nominal base for a family of look-alike implementations",
        "consumer depends on a partial structural view shared by several unrelated classes",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _finding_for_candidate(
        self, confusability_candidate: StructuralConfusabilityCandidate
    ) -> RefactorFinding:
        evidence = (
            confusability_candidate.named(confusability_candidate.function_name),
        )
        return self.build_finding(
            f"`{confusability_candidate.function_name}` observes `{confusability_candidate.parameter_name}` only through methods {confusability_candidate.observed_method_names}, but classes {confusability_candidate.class_names} are confusable under that view.",
            evidence,
        )


@dataclass(frozen=True)
class WitnessCarrierFamilyCandidate(CollectsItself, ClassLineNumbersGroup):
    shared_role_names: tuple[SemanticFieldRole, ...]
    role_field_names: tuple[tuple[SemanticFieldRole, tuple[str, ...]], ...]

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        classes = witness_carrier_class_candidates(module)
        if len(classes) < 2:
            return ()
        grouped: dict[str, list[WitnessCarrierClassCandidate]] = defaultdict(list)
        for candidate in classes:
            for token in candidate.family_tokens:
                grouped[token].append(candidate)
        seen_class_names: set[tuple[str, ...]] = set()
        findings: list[Self] = []
        for token, items in sorted(grouped.items()):
            if len(items) < 3:
                continue
            ordered_items = sorted_tuple(
                items, key=lambda item: (item.line, item.class_name)
            )
            class_names = tuple(item.class_name for item in ordered_items)
            if class_names in seen_class_names:
                continue
            shared_role_names = cast(
                tuple[str, ...],
                sorted_tuple(
                    set.intersection(
                        *(set(candidate.normalized_roles) for candidate in ordered_items)
                    )
                ),
            )
            if len(shared_role_names) < 3:
                continue
            if set.intersection(
                *(set(candidate.base_names) for candidate in ordered_items)
            ):
                continue
            role_field_names = tuple(
                (
                    (role, aliased_field_names)
                    for role in SemanticFieldRole.mixin_roles()
                    for aliased_field_names in (
                        SUPPORT_PROJECTION_AUTHORITY.aliased_field_names_for_role(
                            role,
                            ordered_items,
                        ),
                    )
                    if aliased_field_names
                )
            )
            if not role_field_names:
                continue
            seen_class_names.add(class_names)
            findings.append(
                cls(
                    file_path=module.file_path,
                    class_names=class_names,
                    line_numbers=tuple((candidate.line for candidate in ordered_items)),
                    shared_role_names=shared_role_names,
                    role_field_names=role_field_names,
                )
            )
        return tuple(findings)


class SemanticWitnessFamilyDetector(
    ModuleCollectorCandidateDetector[WitnessCarrierFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_WITNESS_CARRIER,
        "Renamed carrier roles need one inheritance spine",
        "Several frozen dataclass carriers repeat location and naming roles under different field names. One root should own the family identity and reusable role mixins should own the independent slices, composed through MRO.",
        "one nominal carrier root plus reusable role mixins",
        "the same carrier family repeats a renamed role spine across sibling frozen dataclasses",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _finding_for_candidate(
        self, witness_candidate: WitnessCarrierFamilyCandidate
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(witness_candidate.file_path, line, class_name)
                for class_name, line in zip(
                    witness_candidate.class_names,
                    witness_candidate.line_numbers,
                    strict=True,
                )
            )
        )
        role_summary = "; ".join(
            (
                f"{role.require_mixin_name()} for {field_names}"
                for role, field_names in witness_candidate.role_field_names
            )
        )
        shared_role_names = tuple(
            role.value for role in witness_candidate.shared_role_names
        )
        return self.build_finding(
            f"Frozen carrier classes {', '.join(witness_candidate.class_names)} repeat semantic roles {shared_role_names} under renamed fields; establish one nominal carrier root and compose {role_summary} through MRO.",
            evidence,
            metrics=WitnessCarrierMetrics(
                class_count=len(witness_candidate.class_names),
                shared_role_count=len(witness_candidate.shared_role_names),
                class_names=witness_candidate.class_names,
                shared_role_names=shared_role_names,
            ),
        )
