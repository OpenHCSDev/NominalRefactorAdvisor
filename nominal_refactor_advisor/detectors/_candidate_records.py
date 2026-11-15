"""Candidate and witness records the detectors build and read.

Moved out of the _base god file. They ride on the detector and collector
base machinery that stays in _base, imported here by name.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar, Self, cast

from nominal_refactor_advisor.ast_projection import CallKeywords, ExpressionReference
from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    statements_without_docstring,
)
from nominal_refactor_advisor.class_families.parsed_index import (
    FunctionStatement,
    ModuleClassStatements,
)
from nominal_refactor_advisor.class_index import (
    ClsRegistryMembership,
    RegistryLookupShape,
    RegistryLookupStyle,
)
from nominal_refactor_advisor.class_method_facets import (
    CompactMethodSemanticCoordinateKind,
)
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.detectors._base import (
    _FILE_PATHS_ATTRIBUTE,
    _LINE_NUMBERS_ATTRIBUTE,
    ClassFamilyWitnessCarrier,
    CollectsItselfUnderConfig,
    DetectorConfig,
    ManualRecordConstructorFieldPartition,
    MethodFamilyLatticeMetricsCarrier,
    MethodFamilyRelationSpecCarrier,
    MethodFamilyResidueEvidenceCarrier,
    MultiFileZippedSourceLocationEvidenceProperty,
    SupportProjectionAuthority,
    ZippedSourceLocationEvidenceProperty,
    _ManualRecordRegistrationConstructorContext,
    _ManualRecordRegistrationKeyContext,
    _SelectionDictCompContext,
)
from nominal_refactor_advisor.detectors._candidates import (
    ClassNameLineNumbersGroup,
    LineWitnessCandidate,
    WitnessCarrierCandidate,
)
from nominal_refactor_advisor.detectors._registry_candidates import (
    ClassLineWitnessCandidate,
)
from nominal_refactor_advisor.detectors._substrate_support import ModuleNamedValues
from nominal_refactor_advisor.factorization import ResidueHookNamesCarrier
from nominal_refactor_advisor.models import (
    EvidenceLocated,
    MappingMetrics,
    RepeatedMethodMetrics,
    SourceLineReference,
    SourceLocation,
)
from nominal_refactor_advisor.semantic_description_length import CompressionCertificate
from nominal_refactor_advisor.semantic_match import (
    Maybe,
    as_ast,
    name_id,
    single_assign_target,
    single_ast,
    single_item,
    single_return_value,
)


@dataclass(frozen=True)
class ClassLineNumbersGroup(ClassNameLineNumbersGroup):
    file_path: str

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(self.file_path, line, class_name)
            for class_name, line in zip(
                self.class_names, self.line_numbers, strict=True
            )
        )


@dataclass(frozen=True)
class ClassMethodLineWitnessCandidate(LineWitnessCandidate):
    class_name: str
    method_name: str

    @property
    def symbol(self) -> str:
        return f"{self.class_name}.{self.method_name}"

    witness_name: ClassVar[AliasProperty[str]] = AliasProperty("symbol")


@dataclass(frozen=True)
class EvidenceLocationsWitnessCandidate(LineWitnessCandidate):
    """A line witness whose collector hands it every site."""

    # field(): the inherited one-site answer must not become this field's default.
    evidence: tuple[SourceLocation, ...] = field()


@dataclass(frozen=True)
class FieldFamilyLineSurface:
    field_names: tuple[str, ...]
    line_count: int


@dataclass(frozen=True)
class FindingSpecConstructionCandidate(LineWitnessCandidate):
    constructor_name: str
    recommended_builder_name: str
    redundant_keyword_names: tuple[str, ...]
    redundant_keyword_values: tuple[str, ...]
    witness_name = AliasProperty[str]("constructor_name")


@dataclass(frozen=True)
class FunctionFamilyEvidenceSurface:
    function_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    evidence: ClassVar[ZippedSourceLocationEvidenceProperty] = (
        ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")
    )


@dataclass(frozen=True)
class FunctionEvidenceLocationsCandidate(
    FunctionFamilyEvidenceSurface,
    LineWitnessCandidate,
):
    pass


@dataclass(frozen=True)
class FunctionLineWitnessCandidate(LineWitnessCandidate):
    function_name: str
    witness_name = AliasProperty[str]("function_name")


class KeyedRecordInfrastructureCandidate(
    CollectsItselfUnderConfig, EvidenceLocated, ABC
):
    """Candidate that owns its keyed-record surface presentation. Each member finds its
    own candidates in a module; the family's candidates are all of its members'."""

    @property
    @abstractmethod
    def finding_summary(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def finding_metrics(self) -> MappingMetrics:
        raise NotImplementedError

    @classmethod
    def from_module(
        cls, module: ParsedModule, config: DetectorConfig
    ) -> tuple[Self, ...]:
        return tuple(
            candidate
            for member in cls.__subclasses__()
            for candidate in member.from_module(module, config)
        )


@dataclass(frozen=True, kw_only=True)
class ManualRecordRegistrationShape:
    key_expr: str
    key_field_name: str
    constructor_field_names: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ManualKeyedRecordTableClassCandidate(
    ManualRecordRegistrationShape,
    ClassLineWitnessCandidate,
):
    register_method_name: str
    lookup_method_name: str
    lookup_style: RegistryLookupStyle


@dataclass(frozen=True)
class ManualKeyedRecordTableGroupCandidate(KeyedRecordInfrastructureCandidate):
    file_path: str
    classes: tuple[ManualKeyedRecordTableClassCandidate, ...]

    @property
    def finding_summary(self) -> str:
        class_names = ", ".join(item.class_name for item in self.classes[:6])
        key_fields = ", ".join(
            sorted({item.key_field_name for item in self.classes[:6]})
        )
        lookup_names = ", ".join(
            sorted({item.lookup_method_name for item in self.classes[:6]})
        )
        return (
            f"Record tables {class_names} each repeat `_registry`, "
            f"`{self.classes[0].register_method_name}`, and `{lookup_names}` "
            f"around key fields {key_fields}."
        )

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(item.location for item in self.classes[:6])

    @property
    def finding_metrics(self) -> MappingMetrics:
        field_names = sorted_tuple({item.key_field_name for item in self.classes})
        return MappingMetrics.from_field_names(
            mapping_site_count=len(self.classes),
            mapping_name="keyed_record_table",
            field_names=field_names,
            identity_field_names=field_names,
        )

    @classmethod
    def from_module(
        cls, module: ParsedModule, config: DetectorConfig
    ) -> tuple[Self, ...]:
        classes: list[ManualKeyedRecordTableClassCandidate] = []
        for declared in module.fact(ModuleClassStatements).top_level:
            node = declared.node
            if not declared.is_dataclass:
                continue
            if not SupportProjectionAuthority.is_empty_dict_expr(
                declared.direct_assignments.get("_registry")
            ):
                continue
            register_method = declared.method_named("register")
            if register_method is None:
                continue
            registration_shape = cls._manual_record_registration_shape(register_method)
            if registration_shape is None:
                continue
            lookup_methods = [
                (method, shape)
                for method in declared.methods
                if method.spells("classmethod")
                and method.name.startswith("for_")
                and (shape := RegistryLookupShape.from_method(method.node)) is not None
            ]
            if len(lookup_methods) != 1:
                continue
            lookup_method, lookup_shape = lookup_methods[0]
            classes.append(
                ManualKeyedRecordTableClassCandidate(
                    file_path=module.file_path,
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
            grouped[candidate.register_method_name, candidate.lookup_style].append(
                candidate
            )
        return tuple(
            (
                cls(
                    file_path=module.file_path,
                    classes=sorted_tuple(
                        items, key=lambda item: (item.line, item.class_name)
                    ),
                )
                for _, items in sorted(grouped.items())
                if len(items) >= config.min_registration_sites
            )
        )

    @staticmethod
    def _manual_record_registration_shape(
        method: FunctionStatement,
    ) -> ManualRecordRegistrationShape | None:
        body = statements_without_docstring(list(method.node.body))
        return (
            Maybe.of(body)
            .filter(lambda _body: method.spells("classmethod"))
            .combine(
                ManualKeyedRecordTableGroupCandidate._manual_record_registration_key_expr,
                lambda _body, key_expr: _ManualRecordRegistrationKeyContext(
                    body=_body,
                    key_expr=key_expr,
                ),
            )
            .combine(
                lambda context: ManualKeyedRecordTableGroupCandidate._manual_record_registration_constructor(
                    context.body[1:],
                    context.key_expr,
                ),
                lambda context, constructor: (
                    ManualRecordRegistrationShape(
                        key_expr=context.key_expr,
                        key_field_name=constructor_context.key_field_names[0],
                        constructor_field_names=constructor_context.constructor_field_names,
                    )
                    if (
                        constructor_context := _ManualRecordRegistrationConstructorContext(
                            constructor_field_names=constructor[0],
                            key_field_names=constructor[1],
                        )
                    )
                    and len(constructor_context.key_field_names) == 1
                    else None
                ),
            )
            .unwrap_or_none()
        )

    @staticmethod
    def _manual_record_registration_constructor(
        body: list[ast.stmt], key_expr: str
    ) -> ManualRecordConstructorFieldPartition | None:
        assignment = next(
            (
                statement
                for statement in body
                if RegistryLookupShape.key_expr_from_subscript(
                    single_assign_target(statement)
                )
                == key_expr
            ),
            None,
        )
        assignment_call = as_ast(assignment.value if assignment else None, ast.Call)
        if assignment_call is None:
            return None
        if ExpressionReference.decoded(assignment_call.func).terminal_name != "cls":
            return None
        return (
            tuple(CallKeywords.of(assignment_call.keywords).names),
            tuple(
                (
                    keyword.name
                    for keyword in CallKeywords.of(assignment_call.keywords).named
                    if ast.unparse(keyword.value) == key_expr
                )
            ),
        )

    @staticmethod
    def _manual_record_registration_key_expr(body: list[ast.stmt]) -> str | None:
        first_statement = body[0] if len(body) >= 2 else None
        if not isinstance(first_statement, ast.If):
            return None
        membership = ClsRegistryMembership.from_node(first_statement.test)
        if membership is None or membership.operator_type is not ast.In:
            return None
        return membership.key_expr


@dataclass(frozen=True)
class MethodFamilyLineCompressionSurface:
    line_numbers: tuple[int, ...]
    line_count: int
    compression_certificate: CompressionCertificate


@dataclass(frozen=True)
class MethodFamilyEvidenceSurface(MethodFamilyLineCompressionSurface):
    _METHOD_SYMBOLS_ATTRIBUTE = "method_symbols"
    method_names: tuple[str, ...]
    method_symbols: tuple[str, ...]
    evidence: ClassVar[MultiFileZippedSourceLocationEvidenceProperty] = (
        MultiFileZippedSourceLocationEvidenceProperty(
            file_paths_attribute_name=_FILE_PATHS_ATTRIBUTE,
            line_numbers_attribute_name=_LINE_NUMBERS_ATTRIBUTE,
            symbol_names_attribute_name=_METHOD_SYMBOLS_ATTRIBUTE,
        )
    )


@dataclass(frozen=True)
class MethodFamilyAggregateEvidenceSurface(MethodFamilyEvidenceSurface):
    shared_statement_count: int
    residue_declaration_count: int
    leaf_residue_names: tuple[str, ...]


@dataclass(frozen=True)
class ModuleKeyedSelectionHelperCandidate(
    EvidenceLocationsWitnessCandidate,
    KeyedRecordInfrastructureCandidate,
):
    rule_class_name: str
    selected_field_name: str
    helper_function_name: str
    lookup_function_name: str
    rule_table_names: tuple[str, ...]
    index_table_names: tuple[str, ...]
    witness_name = AliasProperty[str]("rule_class_name")

    @property
    def finding_summary(self) -> str:
        return (
            f"`{self.rule_class_name}`, `{self.helper_function_name}`, and "
            f"`{self.lookup_function_name}` implement a local keyed-selection "
            f"substrate for {', '.join(self.rule_table_names[:4])} and indexes "
            f"{', '.join(self.index_table_names[:4])}."
        )

    @property
    def finding_metrics(self) -> MappingMetrics:
        return MappingMetrics(
            mapping_site_count=len(self.rule_table_names),
            field_count=1,
            mapping_name=self.rule_class_name,
            field_names=(self.selected_field_name,),
            source_name=self.helper_function_name,
            identity_field_names=("key",),
        )

    @classmethod
    def from_module(
        cls, module: ParsedModule, config: DetectorConfig
    ) -> tuple[Self, ...]:
        del config
        helper_shapes = tuple(
            (
                helper
                for _, function in module.syntax_index.named_functions
                if "." not in _
                and (helper := cls._selection_helper_shape(function)) is not None
            )
        )
        lookup_shapes = tuple(
            (
                lookup
                for _, function in module.syntax_index.named_functions
                if "." not in _
                and (lookup := cls._selection_lookup_shape(function)) is not None
            )
        )
        if not helper_shapes or not lookup_shapes:
            return ()
        named_values = module.fact(ModuleNamedValues)
        named_sequences = named_values.sequences
        named_calls = named_values.calls
        candidates: list[Self] = []
        for declared in module.fact(ModuleClassStatements).top_level:
            node = declared.node
            if not declared.is_dataclass:
                continue
            field_names = declared.annotated_names
            if len(field_names) != 2 or field_names[0] != "key":
                continue
            selected_field_name = field_names[1]
            matching_helpers = tuple(
                (
                    helper
                    for helper in helper_shapes
                    if helper.selected_field_name == selected_field_name
                )
            )
            if not matching_helpers:
                continue
            rule_table_names: list[str] = []
            indexed_table_names: list[str] = []
            evidence: list[SourceLocation] = [
                SourceLocation(module.file_path, node.lineno, node.name)
            ]
            for family_name, (line, elements) in sorted(named_sequences.items()):
                if len(elements) < 2:
                    continue
                if not all(
                    (
                        isinstance(element, ast.Call)
                        and ExpressionReference.decoded(element.func).terminal_name
                        == node.name
                        for element in elements
                    )
                ):
                    continue
                keyword_maps = [
                    dict(CallKeywords.of(element.keywords).by_name)
                    for element in cast(tuple[ast.Call, ...], elements)
                ]
                if not all(
                    (
                        "key" in keyword_map and selected_field_name in keyword_map
                        for keyword_map in keyword_maps
                    )
                ):
                    continue
                rule_table_names.append(family_name)
                evidence.append(SourceLocation(module.file_path, line, family_name))
            if len(rule_table_names) < 2:
                continue
            helper_names = {helper.function_name for helper in matching_helpers}
            for call_name, (line, call) in sorted(named_calls.items()):
                if (
                    ExpressionReference.decoded(call.func).terminal_name
                    not in helper_names
                    or not call.args
                ):
                    continue
                argument = call.args[0]
                if isinstance(argument, ast.Name) and argument.id in rule_table_names:
                    indexed_table_names.append(call_name)
                    evidence.append(SourceLocation(module.file_path, line, call_name))
            if len(indexed_table_names) < 2:
                continue
            candidates.append(
                cls(
                    file_path=module.file_path,
                    line=node.lineno,
                    rule_class_name=node.name,
                    selected_field_name=selected_field_name,
                    helper_function_name=matching_helpers[0].function_name,
                    lookup_function_name=lookup_shapes[0].function_name,
                    rule_table_names=tuple(rule_table_names),
                    index_table_names=tuple(indexed_table_names),
                    evidence=tuple(evidence[:6]),
                )
            )
        return sorted_tuple(
            candidates,
            key=lambda item: (item.file_path, item.line, item.rule_class_name),
        )

    @staticmethod
    def _selection_helper_shape(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> _SelectionHelperShape | None:
        return (
            Maybe.of(
                as_ast(
                    single_return_value(statements_without_docstring(function.body)),
                    ast.DictComp,
                )
            )
            .combine(
                lambda returned: single_item(returned.generators),
                lambda returned, generator: _SelectionDictCompContext(
                    returned, generator
                ),
            )
            .filter(
                lambda context: (
                    not context.generator.ifs
                    and isinstance(context.generator.target, ast.Name)
                )
            )
            .combine(
                lambda context: ModuleKeyedSelectionHelperCandidate._selection_dict_value_field(
                    context.returned, cast(ast.Name, context.generator.target).id
                ),
                lambda context, selected_field_name: _SelectionHelperShape(
                    function_name=function.name,
                    selected_field_name=selected_field_name,
                    line=function.lineno,
                ),
            )
            .unwrap_or_none()
        )

    @staticmethod
    def _selection_lookup_shape(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> _SelectionLookupShape | None:
        try_node = ModuleKeyedSelectionHelperCandidate._single_try_statement(function)
        if try_node is None:
            return None
        if not ModuleKeyedSelectionHelperCandidate._selection_lookup_returns_subscript(
            try_node
        ):
            return None
        if not ModuleKeyedSelectionHelperCandidate._selection_lookup_raises_key_error(
            try_node
        ):
            return None
        return _SelectionLookupShape(function_name=function.name, line=function.lineno)

    @staticmethod
    def _selection_dict_value_field(
        returned: ast.DictComp, target_name: str
    ) -> str | None:
        key = returned.key
        value = returned.value
        if not (
            isinstance(key, ast.Attribute)
            and isinstance(key.value, ast.Name)
            and (key.value.id == target_name)
            and (key.attr == "key")
        ):
            return None
        if not (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and (value.value.id == target_name)
        ):
            return None
        return value.attr

    @staticmethod
    def _selection_lookup_raises_key_error(try_node: ast.Try) -> bool:
        handler = single_item(try_node.handlers)
        handler_type_name = name_id(handler.type) if handler is not None else None
        raised = single_item(handler.body) if handler is not None else None
        return bool(
            isinstance(handler, ast.ExceptHandler)
            and handler_type_name == "KeyError"
            and isinstance(raised, ast.Raise)
        )

    @staticmethod
    def _selection_lookup_returns_subscript(try_node: ast.Try) -> bool:
        returned = as_ast(single_return_value(try_node.body), ast.Subscript)
        return bool(
            returned is not None
            and name_id(returned.value) is not None
            and (name_id(returned.slice) is not None)
        )

    @staticmethod
    def _single_try_statement(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ast.Try | None:
        return single_ast(statements_without_docstring(function.body), ast.Try)


@dataclass(frozen=True)
class NominalAuthorityShape(SourceLineReference):
    class_name: str
    declared_base_names: tuple[str, ...]
    ancestor_names: tuple[str, ...]
    field_names: tuple[str, ...]
    field_type_map: tuple[tuple[str, str], ...]
    method_names: tuple[str, ...]
    is_abstract: bool
    is_dataclass_family: bool


@dataclass(frozen=True)
class OverlappingInheritanceFamiliesCandidate(
    MethodFamilyAggregateEvidenceSurface,
    LineWitnessCandidate,
    ClassFamilyWitnessCarrier,
    MethodFamilyLatticeMetricsCarrier,
    MethodFamilyRelationSpecCarrier,
):
    family_specs: tuple[str, ...]


@dataclass(frozen=True)
class PositionalKeywordCallSurface:
    positional_arg_count: int
    keyword_names: tuple[str, ...]


@dataclass(frozen=True)
class PropertyHookGroup(ClassLineNumbersGroup):
    base_name: str
    property_name: str

    @property
    def repeated_method_metrics(self) -> RepeatedMethodMetrics:
        return RepeatedMethodMetrics.from_duplicate_family(
            duplicate_site_count=len(self.class_names),
            statement_count=1,
            class_count=len(self.class_names),
            method_symbols=tuple(
                f"{class_name}.{self.property_name}" for class_name in self.class_names
            ),
        )


@dataclass(frozen=True)
class QualnameLineWitnessCandidate(LineWitnessCandidate):
    qualname: str
    witness_name = AliasProperty[str]("qualname")


@dataclass(frozen=True)
class SelfReflectiveBuiltinCandidate(WitnessCarrierCandidate):
    method_name: str
    reflective_builtin: str


@dataclass(frozen=True)
class SemanticOverlapMethodCandidate(
    MethodFamilyLineCompressionSurface,
    LineWitnessCandidate,
    ClassFamilyWitnessCarrier,
    ResidueHookNamesCarrier,
    MethodFamilyRelationSpecCarrier,
    MethodFamilyLatticeMetricsCarrier,
    MethodFamilyResidueEvidenceCarrier,
):
    _CLASS_NAMES_ATTRIBUTE = "class_names"
    method_name: str
    shared_statement_count: int
    varying_coordinate_count: int
    family_method_names: tuple[str, ...]
    evidence: ClassVar[MultiFileZippedSourceLocationEvidenceProperty] = (
        MultiFileZippedSourceLocationEvidenceProperty(
            file_paths_attribute_name=_FILE_PATHS_ATTRIBUTE,
            line_numbers_attribute_name=_LINE_NUMBERS_ATTRIBUTE,
            symbol_names_attribute_name=_CLASS_NAMES_ATTRIBUTE,
        )
    )


@dataclass(frozen=True)
class SemanticOverlapMethodFamilyCandidate(
    MethodFamilyAggregateEvidenceSurface,
    LineWitnessCandidate,
    ClassFamilyWitnessCarrier,
    ResidueHookNamesCarrier,
    MethodFamilyLatticeMetricsCarrier,
    MethodFamilyRelationSpecCarrier,
):
    shared_to_residue_ratio: float


@dataclass(frozen=True)
class SemanticOverlapResidueAxisCandidate(
    MethodFamilyEvidenceSurface,
    LineWitnessCandidate,
    ClassFamilyWitnessCarrier,
):
    residue_kinds: tuple[CompactMethodSemanticCoordinateKind, ...]
    residue_site_count: int

    @property
    def residue_kind_names(self) -> tuple[str, ...]:
        return tuple(residue_kind.value for residue_kind in self.residue_kinds)


@dataclass(frozen=True)
class TupleIndexSemanticOpacityCandidate(FunctionLineWitnessCandidate):
    index_expressions: tuple[str, ...]
    nested_index_count: int
    carrier_call_names: tuple[str, ...]


@dataclass(frozen=True)
class _SelectionHelperShape:
    function_name: str
    selected_field_name: str
    line: int


@dataclass(frozen=True)
class _SelectionLookupShape:
    function_name: str
    line: int
