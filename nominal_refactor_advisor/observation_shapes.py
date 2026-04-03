from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

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
class FieldObservation(StructuralObservationCarrier):
    file_path: str
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
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.class_name,
            line=self.lineno,
            observation_kind=ObservationKind.FIELD,
            execution_level=self.execution_level,
            observed_name=self.field_name,
            fiber_key=self.field_name,
        )


@dataclass(frozen=True)
class AttributeProbeObservation(StructuralObservationCarrier):
    file_path: str
    line: int
    symbol: str
    probe_kind: str
    observed_attribute: str | None
    execution_level: StructuralExecutionLevel

    @property
    def structural_observation(self) -> StructuralObservation:
        observed_name = self.observed_attribute or self.probe_kind
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.symbol,
            line=self.line,
            observation_kind=ObservationKind.ATTRIBUTE_PROBE,
            execution_level=self.execution_level,
            observed_name=observed_name,
            fiber_key=f"{self.probe_kind}:{observed_name}",
        )


@dataclass(frozen=True)
class LiteralDispatchObservation(StructuralObservationCarrier):
    file_path: str
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
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.scope_owner or self.symbol,
            line=self.line,
            observation_kind=ObservationKind.LITERAL_DISPATCH,
            execution_level=self.execution_level,
            observed_name=self.axis_expression,
            fiber_key=f"{self.literal_kind}:{self.axis_fingerprint}",
        )


@dataclass(frozen=True)
class ProjectionHelperShape(StructuralObservationCarrier):
    file_path: str
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
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.function_name,
            line=self.lineno,
            observation_kind=ObservationKind.PROJECTION_HELPER,
            execution_level=StructuralExecutionLevel.FUNCTION_BODY,
            observed_name=self.projected_attribute,
            fiber_key=(
                f"{self.outer_call_name}:{self.aggregator_name}:{self.iterable_fingerprint}"
            ),
        )


@dataclass(frozen=True)
class AccessorWrapperCandidate(StructuralObservationCarrier):
    file_path: str
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
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.class_name,
            line=self.lineno,
            observation_kind=ObservationKind.ACCESSOR_WRAPPER,
            execution_level=StructuralExecutionLevel.FUNCTION_BODY,
            observed_name=self.observed_attribute,
            fiber_key=f"{self.accessor_kind}:{self.wrapper_shape}:{self.observed_attribute}",
        )


@dataclass(frozen=True)
class ScopedShapeWrapperFunction(StructuralObservationCarrier):
    file_path: str
    function_name: str
    lineno: int
    node_types: tuple[str, ...]

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.function_name,
            nominal_witness=self.function_name,
            line=self.lineno,
            observation_kind=ObservationKind.SCOPED_SHAPE_WRAPPER,
            execution_level=StructuralExecutionLevel.MODULE_BODY,
            observed_name=self.function_name,
            fiber_key=f"function:{'/'.join(self.node_types)}",
        )


@dataclass(frozen=True)
class ScopedShapeWrapperSpec(StructuralObservationCarrier):
    file_path: str
    spec_name: str
    lineno: int
    function_name: str
    node_types: tuple[str, ...]

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.spec_name,
            nominal_witness=self.function_name,
            line=self.lineno,
            observation_kind=ObservationKind.SCOPED_SHAPE_WRAPPER,
            execution_level=StructuralExecutionLevel.MODULE_BODY,
            observed_name=self.function_name,
            fiber_key=f"spec:{'/'.join(self.node_types)}:{self.function_name}",
        )


@dataclass(frozen=True)
class ConfigDispatchObservation(StructuralObservationCarrier):
    file_path: str
    line: int
    symbol: str
    observed_attribute: str

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.symbol,
            line=self.line,
            observation_kind=ObservationKind.CONFIG_DISPATCH,
            execution_level=StructuralExecutionLevel.FUNCTION_BODY,
            observed_name=self.observed_attribute,
            fiber_key=self.observed_attribute,
        )


@dataclass(frozen=True)
class ClassMarkerObservation(StructuralObservationCarrier):
    file_path: str
    line: int
    symbol: str
    marker_name: str

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.symbol,
            line=self.line,
            observation_kind=ObservationKind.CLASS_MARKER,
            execution_level=StructuralExecutionLevel.FUNCTION_BODY,
            observed_name=self.marker_name,
            fiber_key=self.marker_name,
        )


@dataclass(frozen=True)
class InterfaceGenerationObservation(StructuralObservationCarrier):
    file_path: str
    line: int
    symbol: str
    generator_name: str

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.symbol,
            line=self.line,
            observation_kind=ObservationKind.INTERFACE_GENERATION,
            execution_level=StructuralExecutionLevel.FUNCTION_BODY,
            observed_name=self.generator_name,
            fiber_key=self.generator_name,
        )


@dataclass(frozen=True)
class SentinelTypeObservation(StructuralObservationCarrier):
    file_path: str
    line: int
    symbol: str
    sentinel_name: str

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.symbol,
            line=self.line,
            observation_kind=ObservationKind.SENTINEL_TYPE,
            execution_level=StructuralExecutionLevel.MODULE_BODY,
            observed_name=self.sentinel_name,
            fiber_key=self.sentinel_name,
        )


@dataclass(frozen=True)
class DynamicMethodInjectionObservation(StructuralObservationCarrier):
    file_path: str
    line: int
    symbol: str
    mutator_name: str

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.symbol,
            line=self.line,
            observation_kind=ObservationKind.DYNAMIC_METHOD_INJECTION,
            execution_level=StructuralExecutionLevel.FUNCTION_BODY,
            observed_name=self.mutator_name,
            fiber_key=self.mutator_name,
        )


@dataclass(frozen=True)
class RuntimeTypeGenerationObservation(StructuralObservationCarrier):
    file_path: str
    line: int
    symbol: str
    generator_name: str

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.symbol,
            line=self.line,
            observation_kind=ObservationKind.RUNTIME_TYPE_GENERATION,
            execution_level=StructuralExecutionLevel.MODULE_BODY,
            observed_name=self.generator_name,
            fiber_key=self.generator_name,
        )


@dataclass(frozen=True)
class LineageMappingObservation(StructuralObservationCarrier):
    file_path: str
    line: int
    symbol: str
    mapping_name: str

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.symbol,
            line=self.line,
            observation_kind=ObservationKind.LINEAGE_MAPPING,
            execution_level=StructuralExecutionLevel.MODULE_BODY,
            observed_name=self.mapping_name,
            fiber_key=self.mapping_name,
        )


@dataclass(frozen=True)
class DualAxisResolutionObservation(StructuralObservationCarrier):
    file_path: str
    line: int
    symbol: str
    outer_axis_name: str
    inner_axis_name: str

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.symbol,
            line=self.line,
            observation_kind=ObservationKind.DUAL_AXIS_RESOLUTION,
            execution_level=StructuralExecutionLevel.FUNCTION_BODY,
            observed_name=f"{self.outer_axis_name}:{self.inner_axis_name}",
            fiber_key=f"{self.outer_axis_name}:{self.inner_axis_name}",
        )


@dataclass(frozen=True)
class MethodShape(StructuralObservationCarrier):
    file_path: str
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
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.class_name or self.symbol,
            line=self.lineno,
            observation_kind=ObservationKind.METHOD_SHAPE,
            execution_level=StructuralExecutionLevel.FUNCTION_BODY,
            observed_name=self.method_name,
            fiber_key=f"{self.is_private}:{self.param_count}:{self.fingerprint}",
        )


@dataclass(frozen=True)
class BuilderCallShape(StructuralObservationCarrier):
    file_path: str
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
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.class_name or self.function_name or self.symbol,
            line=self.lineno,
            observation_kind=ObservationKind.BUILDER_CALL,
            execution_level=StructuralExecutionLevel.FUNCTION_BODY,
            observed_name=self.callee_name,
            fiber_key=(
                f"{self.callee_name}:{self.keyword_names}:{self.value_fingerprint}"
            ),
        )


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
class ExportDictShape(StructuralObservationCarrier):
    file_path: str
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
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.class_name or self.function_name or self.symbol,
            line=self.lineno,
            observation_kind=ObservationKind.EXPORT_DICT,
            execution_level=StructuralExecutionLevel.FUNCTION_BODY,
            observed_name=",".join(self.key_names),
            fiber_key=f"{self.key_names}:{self.value_fingerprint}",
        )


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
