from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, cast

from .observation_graph import (
    ObservationKind,
    StructuralExecutionLevel,
    StructuralObservation,
    StructuralObservationCarrier,
)

if TYPE_CHECKING:
    from .ast_tools import ParsedModule


class LiteralKind(StrEnum):
    STRING = "string"
    NUMERIC = "numeric"


class FieldOriginKind(StrEnum):
    CLASS_ASSIGNMENT = "class_assignment"
    CLASS_ANNOTATION = "class_annotation"
    DATACLASS_FIELD = "dataclass_field"
    INIT_ASSIGNMENT = "init_assignment"


@dataclass(frozen=True)
class StructuralObservationTemplate(StructuralObservationCarrier, ABC):
    file_path: str
    OBSERVATION_KIND: ClassVar[ObservationKind]

    @property
    def observation_kind(self) -> ObservationKind:
        return type(self).OBSERVATION_KIND

    @property
    @abstractmethod
    def observation_execution_level(self) -> StructuralExecutionLevel:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_line(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def owner_symbol(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def nominal_witness(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def observed_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def fiber_key(self) -> str:
        raise NotImplementedError

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.owner_symbol,
            nominal_witness=self.nominal_witness,
            line=self.observation_line,
            observation_kind=self.observation_kind,
            execution_level=self.observation_execution_level,
            observed_name=self.observed_name,
            fiber_key=self.fiber_key,
        )


class ExecutionLevelObservationMixin(ABC):
    execution_level: StructuralExecutionLevel

    @property
    def observation_execution_level(self) -> StructuralExecutionLevel:
        return self.execution_level


class StaticExecutionLevelMixin(ABC):
    OBSERVATION_EXECUTION_LEVEL: ClassVar[StructuralExecutionLevel]

    @property
    def observation_execution_level(self) -> StructuralExecutionLevel:
        return type(self).OBSERVATION_EXECUTION_LEVEL


class FunctionBodyExecutionMixin(StaticExecutionLevelMixin):
    OBSERVATION_EXECUTION_LEVEL = StructuralExecutionLevel.FUNCTION_BODY


class ModuleBodyExecutionMixin(StaticExecutionLevelMixin):
    OBSERVATION_EXECUTION_LEVEL = StructuralExecutionLevel.MODULE_BODY


class LineObservationMixin(ABC):
    line: int

    @property
    def observation_line(self) -> int:
        return self.line


class LinenoObservationMixin(ABC):
    lineno: int

    @property
    def observation_line(self) -> int:
        return self.lineno


class SymbolOwnerMixin(ABC):
    @property
    def owner_symbol(self) -> str:
        return cast(Any, self).symbol


class FunctionNameOwnerMixin(ABC):
    function_name: str

    @property
    def owner_symbol(self) -> str:
        return self.function_name


class SymbolNominalWitnessMixin(ABC):
    @property
    def nominal_witness(self) -> str:
        return cast(Any, self).symbol


class ClassNameNominalWitnessMixin(ABC):
    class_name: str

    @property
    def nominal_witness(self) -> str:
        return self.class_name


class FunctionNameNominalWitnessMixin(ABC):
    function_name: str

    @property
    def nominal_witness(self) -> str:
        return self.function_name


class FunctionNameObservedNameMixin(ABC):
    function_name: str

    @property
    def observed_name(self) -> str:
        return self.function_name


class ObservedAttributeObservedNameMixin(ABC):
    observed_attribute: str

    @property
    def observed_name(self) -> str:
        return self.observed_attribute


class GeneratorNameObservedNameMixin(ABC):
    generator_name: str

    @property
    def observed_name(self) -> str:
        return self.generator_name


class GeneratorNameFiberKeyMixin(ABC):
    generator_name: str

    @property
    def fiber_key(self) -> str:
        return self.generator_name


@dataclass(frozen=True)
class FieldObservation(
    ExecutionLevelObservationMixin,
    LinenoObservationMixin,
    SymbolOwnerMixin,
    ClassNameNominalWitnessMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.FIELD
    class_name: str
    field_name: str
    lineno: int
    execution_level: StructuralExecutionLevel
    origin_kind: FieldOriginKind
    is_dataclass_family: bool
    value_fingerprint: str | None = None
    annotation_text: str | None = None
    annotation_fingerprint: str | None = None

    @property
    def symbol(self) -> str:
        return f"{self.class_name}.{self.field_name}"

    @property
    def observed_name(self) -> str:
        return self.field_name

    @property
    def fiber_key(self) -> str:
        return self.field_name


@dataclass(frozen=True)
class AttributeProbeObservation(
    ExecutionLevelObservationMixin,
    LineObservationMixin,
    SymbolOwnerMixin,
    SymbolNominalWitnessMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.ATTRIBUTE_PROBE
    line: int
    symbol: str
    probe_kind: str
    observed_attribute: str | None
    execution_level: StructuralExecutionLevel

    @property
    def observed_name(self) -> str:
        return self.observed_attribute or self.probe_kind

    @property
    def fiber_key(self) -> str:
        return f"{self.probe_kind}:{self.observed_name}"


@dataclass(frozen=True)
class LiteralDispatchObservation(
    ExecutionLevelObservationMixin,
    LineObservationMixin,
    SymbolOwnerMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.LITERAL_DISPATCH
    line: int
    symbol: str
    axis_fingerprint: str
    axis_expression: str
    literal_cases: tuple[str, ...]
    literal_kind: LiteralKind
    execution_level: StructuralExecutionLevel
    branch_lines: tuple[int, ...] = ()
    scope_owner: str | None = None

    @property
    def nominal_witness(self) -> str:
        return self.scope_owner or self.symbol

    @property
    def observed_name(self) -> str:
        return self.axis_expression

    @property
    def fiber_key(self) -> str:
        return f"{self.literal_kind}:{self.axis_fingerprint}"


@dataclass(frozen=True)
class ProjectionHelperShape(
    FunctionBodyExecutionMixin,
    LinenoObservationMixin,
    SymbolOwnerMixin,
    FunctionNameNominalWitnessMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.PROJECTION_HELPER
    function_name: str
    lineno: int
    outer_call_name: str
    aggregator_name: str
    iterable_fingerprint: str
    projected_attribute: str

    @property
    def symbol(self) -> str:
        return self.function_name

    @property
    def observed_name(self) -> str:
        return self.projected_attribute

    @property
    def fiber_key(self) -> str:
        return (
            f"{self.outer_call_name}:{self.aggregator_name}:{self.iterable_fingerprint}"
        )


@dataclass(frozen=True)
class AccessorWrapperCandidate(
    FunctionBodyExecutionMixin,
    LinenoObservationMixin,
    SymbolOwnerMixin,
    ClassNameNominalWitnessMixin,
    ObservedAttributeObservedNameMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.ACCESSOR_WRAPPER
    class_name: str
    method_name: str
    lineno: int
    target_expression: str
    observed_attribute: str
    accessor_kind: str
    wrapper_shape: str

    @property
    def symbol(self) -> str:
        return f"{self.class_name}.{self.method_name}"

    @property
    def fiber_key(self) -> str:
        return f"{self.accessor_kind}:{self.wrapper_shape}:{self.observed_attribute}"


@dataclass(frozen=True)
class ScopedShapeWrapperFunction(
    ModuleBodyExecutionMixin,
    LinenoObservationMixin,
    FunctionNameOwnerMixin,
    FunctionNameNominalWitnessMixin,
    FunctionNameObservedNameMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.SCOPED_SHAPE_WRAPPER
    function_name: str
    lineno: int
    node_types: tuple[str, ...]

    @property
    def fiber_key(self) -> str:
        return f"function:{'/'.join(self.node_types)}"


@dataclass(frozen=True)
class ScopedShapeWrapperSpec(
    ModuleBodyExecutionMixin,
    LinenoObservationMixin,
    FunctionNameNominalWitnessMixin,
    FunctionNameObservedNameMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.SCOPED_SHAPE_WRAPPER
    spec_name: str
    lineno: int
    function_name: str
    node_types: tuple[str, ...]

    @property
    def owner_symbol(self) -> str:
        return self.spec_name

    @property
    def fiber_key(self) -> str:
        return f"spec:{'/'.join(self.node_types)}:{self.function_name}"


@dataclass(frozen=True)
class ConfigDispatchObservation(
    FunctionBodyExecutionMixin,
    LineObservationMixin,
    SymbolOwnerMixin,
    SymbolNominalWitnessMixin,
    ObservedAttributeObservedNameMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.CONFIG_DISPATCH
    line: int
    symbol: str
    observed_attribute: str

    @property
    def fiber_key(self) -> str:
        return self.observed_attribute


@dataclass(frozen=True)
class ClassMarkerObservation(
    FunctionBodyExecutionMixin,
    LineObservationMixin,
    SymbolOwnerMixin,
    SymbolNominalWitnessMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.CLASS_MARKER
    line: int
    symbol: str
    marker_name: str

    @property
    def observed_name(self) -> str:
        return self.marker_name

    @property
    def fiber_key(self) -> str:
        return self.marker_name


@dataclass(frozen=True)
class InterfaceGenerationObservation(
    FunctionBodyExecutionMixin,
    LineObservationMixin,
    SymbolOwnerMixin,
    SymbolNominalWitnessMixin,
    GeneratorNameObservedNameMixin,
    GeneratorNameFiberKeyMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.INTERFACE_GENERATION
    line: int
    symbol: str
    generator_name: str


@dataclass(frozen=True)
class SentinelTypeObservation(
    ModuleBodyExecutionMixin,
    LineObservationMixin,
    SymbolOwnerMixin,
    SymbolNominalWitnessMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.SENTINEL_TYPE
    line: int
    symbol: str
    sentinel_name: str

    @property
    def observed_name(self) -> str:
        return self.sentinel_name

    @property
    def fiber_key(self) -> str:
        return self.sentinel_name


@dataclass(frozen=True)
class DynamicMethodInjectionObservation(
    FunctionBodyExecutionMixin,
    LineObservationMixin,
    SymbolOwnerMixin,
    SymbolNominalWitnessMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.DYNAMIC_METHOD_INJECTION
    line: int
    symbol: str
    mutator_name: str

    @property
    def observed_name(self) -> str:
        return self.mutator_name

    @property
    def fiber_key(self) -> str:
        return self.mutator_name


@dataclass(frozen=True)
class RuntimeTypeGenerationObservation(
    ModuleBodyExecutionMixin,
    LineObservationMixin,
    SymbolOwnerMixin,
    SymbolNominalWitnessMixin,
    GeneratorNameObservedNameMixin,
    GeneratorNameFiberKeyMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.RUNTIME_TYPE_GENERATION
    line: int
    symbol: str
    generator_name: str


@dataclass(frozen=True)
class LineageMappingObservation(
    ModuleBodyExecutionMixin,
    LineObservationMixin,
    SymbolOwnerMixin,
    SymbolNominalWitnessMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.LINEAGE_MAPPING
    line: int
    symbol: str
    mapping_name: str

    @property
    def observed_name(self) -> str:
        return self.mapping_name

    @property
    def fiber_key(self) -> str:
        return self.mapping_name


@dataclass(frozen=True)
class DualAxisResolutionObservation(
    FunctionBodyExecutionMixin,
    LineObservationMixin,
    SymbolOwnerMixin,
    SymbolNominalWitnessMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.DUAL_AXIS_RESOLUTION
    line: int
    symbol: str
    outer_axis_name: str
    inner_axis_name: str

    @property
    def observed_name(self) -> str:
        return f"{self.outer_axis_name}:{self.inner_axis_name}"

    @property
    def fiber_key(self) -> str:
        return self.observed_name


@dataclass(frozen=True)
class MethodShape(
    FunctionBodyExecutionMixin,
    LinenoObservationMixin,
    SymbolOwnerMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.METHOD_SHAPE
    class_name: str | None
    method_name: str
    lineno: int
    statement_count: int
    is_private: bool
    param_count: int
    decorators: tuple[str, ...]
    fingerprint: str
    statement_texts: tuple[str, ...]

    @property
    def symbol(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.method_name}"
        return self.method_name

    @property
    def nominal_witness(self) -> str:
        return self.class_name or self.symbol

    @property
    def observed_name(self) -> str:
        return self.method_name

    @property
    def fiber_key(self) -> str:
        return f"{self.is_private}:{self.param_count}:{self.fingerprint}"


@dataclass(frozen=True)
class BuilderCallShape(
    FunctionBodyExecutionMixin,
    LinenoObservationMixin,
    SymbolOwnerMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.BUILDER_CALL
    class_name: str | None
    function_name: str | None
    lineno: int
    callee_name: str
    keyword_names: tuple[str, ...]
    value_fingerprint: tuple[str, ...]
    source_arity: int
    source_name: str | None
    identity_field_names: tuple[str, ...]

    @property
    def symbol(self) -> str:
        owner = self.function_name or "<module>"
        if self.class_name:
            owner = f"{self.class_name}.{owner}"
        return f"{owner}:{self.callee_name}"

    @property
    def nominal_witness(self) -> str:
        return self.class_name or self.function_name or self.symbol

    @property
    def observed_name(self) -> str:
        return self.callee_name

    @property
    def fiber_key(self) -> str:
        return f"{self.callee_name}:{self.keyword_names}:{self.value_fingerprint}"


@dataclass(frozen=True)
class RegistrationShape:
    file_path: str
    lineno: int
    registry_name: str
    registered_class: str
    key_fingerprint: str
    key_expression: str
    registration_style: str

    @classmethod
    def from_assignment(
        cls,
        parsed_module: ParsedModule,
        node: ast.Assign,
        registry_name: str,
        key_fingerprint: str,
    ) -> RegistrationShape:
        if not isinstance(node.value, ast.Name):
            raise TypeError("Registration assignment value must be a class name")
        return cls(
            file_path=str(parsed_module.path),
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=node.value.id,
            key_fingerprint=key_fingerprint,
            key_expression=ast.unparse(node.targets[0].slice)
            if isinstance(node.targets[0], ast.Subscript)
            else "...",
            registration_style="subscript_assignment",
        )

    @classmethod
    def from_registration_call(
        cls,
        parsed_module: ParsedModule,
        node: ast.Call,
        registry_name: str,
        registered_class: str,
        key_fingerprint: str,
    ) -> RegistrationShape:
        return cls(
            file_path=str(parsed_module.path),
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=registered_class,
            key_fingerprint=key_fingerprint,
            key_expression=ast.unparse(
                node.args[1] if len(node.args) >= 2 else node.args[0]
            ),
            registration_style="registration_call",
        )

    @classmethod
    def from_decorator(
        cls,
        parsed_module: ParsedModule,
        node: ast.ClassDef,
        registry_name: str,
        key_fingerprint: str,
    ) -> RegistrationShape:
        return cls(
            file_path=str(parsed_module.path),
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=node.name,
            key_fingerprint=key_fingerprint,
            key_expression=node.name,
            registration_style="decorator_registration",
        )

    @property
    def symbol(self) -> str:
        return f"{self.registry_name}[...] = {self.registered_class}"


@dataclass(frozen=True)
class ExportDictShape(
    FunctionBodyExecutionMixin,
    LinenoObservationMixin,
    SymbolOwnerMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.EXPORT_DICT
    class_name: str | None
    function_name: str | None
    lineno: int
    key_names: tuple[str, ...]
    value_fingerprint: tuple[str, ...]
    source_arity: int
    source_name: str | None
    identity_field_names: tuple[str, ...]

    @property
    def symbol(self) -> str:
        owner = self.function_name or "<module>"
        if self.class_name:
            owner = f"{self.class_name}.{owner}"
        return f"{owner}:export-dict"

    @property
    def nominal_witness(self) -> str:
        return self.class_name or self.function_name or self.symbol

    @property
    def observed_name(self) -> str:
        return ",".join(self.key_names)

    @property
    def fiber_key(self) -> str:
        return f"{self.key_names}:{self.value_fingerprint}"


__all__ = [
    "AccessorWrapperCandidate",
    "AttributeProbeObservation",
    "BuilderCallShape",
    "ClassMarkerObservation",
    "ConfigDispatchObservation",
    "DualAxisResolutionObservation",
    "DynamicMethodInjectionObservation",
    "ExportDictShape",
    "FieldObservation",
    "FieldOriginKind",
    "InterfaceGenerationObservation",
    "LineageMappingObservation",
    "LiteralDispatchObservation",
    "LiteralKind",
    "MethodShape",
    "ProjectionHelperShape",
    "RegistrationShape",
    "RuntimeTypeGenerationObservation",
    "ScopedShapeWrapperFunction",
    "ScopedShapeWrapperSpec",
    "SentinelTypeObservation",
]
