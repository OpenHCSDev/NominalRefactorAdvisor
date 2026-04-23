"""Observation spec families and spec-derived family generation.

This module declares the public observation specs used by the advisor and derives
their exported collected families from the spec definitions themselves. The goal is
to keep one nominal authority per family and derive runtime family surfaces from it.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, cast

from .export_tools import PublicExportPolicy, derive_public_exports

from .observation_shapes import (
    AccessorWrapperCandidate,
    AttributeProbeObservation,
    BuilderCallShape,
    ClassMarkerObservation,
    ConfigDispatchObservation,
    DualAxisResolutionObservation,
    DynamicMethodInjectionObservation,
    ExportDictShape,
    FieldObservation,
    InterfaceGenerationObservation,
    LineageMappingObservation,
    LiteralDispatchObservation,
    LiteralKind,
    MethodShape,
    ProjectionHelperShape,
    RegistrationShape,
    RuntimeTypeGenerationObservation,
    ScopedShapeWrapperFunction,
    ScopedShapeWrapperSpec,
    SentinelTypeObservation,
)

from .ast_tools import (
    AstNameFamily,
    AssignObservationSpec,
    AutoRegisterMeta,
    AutoRegisteredModuleShapeSpec,
    ClassAstObservation,
    CollectedFamily,
    ContextForwardingShapeSpec,
    ContextHelperShapeSpec,
    FunctionObservationSpec,
    ObservationShapeSpec,
    ParsedModule,
    RegisteredSpecCollectedFamily,
    ScopedAstObservation,
    SingleSpecCollectedFamily,
    _ATTRIBUTE_ERROR_FAMILY,
    _GETATTR_CALL_FAMILY,
    _HASATTR_CALL_FAMILY,
    _REGISTRATION_CALL_FAMILY,
    _REGISTRATION_DECORATOR_FAMILY,
    _accessor_wrapper_candidate_from_function,
    _builder_call_shape,
    _class_body_field_observation,
    _class_marker_observations,
    _class_name_from_expr,
    _class_observations,
    _collect_items_from_spec_root,
    _config_dispatch_observations,
    _dual_axis_resolution_observation,
    _dynamic_method_injection_observations,
    _execution_level_for_scope,
    _export_dict_shape,
    _fingerprint_builder_value,
    _init_field_observations,
    _inline_literal_dispatch_observations_for_kind,
    _interface_generation_observation,
    _iter_attribute_family_calls,
    _iter_class_decorator_family_calls,
    _known_class_family,
    _lineage_mapping_observation,
    _literal_dispatch_observations_for_kind,
    _node_display_name,
    _node_matches_family,
    _projection_helper_shape_from_function,
    _registration_key_fingerprint,
    _runtime_type_generation_observation,
    _scoped_shape_wrapper_function_from_function,
    _scoped_shape_wrapper_spec_from_assign,
    _sentinel_type_observation,
    _sentinel_type_usage_observations,
    _subscript_base_name,
    _terminal_name,
    _terminal_name_in_family,
)


@dataclass(frozen=True)
class GeneratedFamilySpec:
    """Declarative recipe for one generated collected family export."""

    item_type: type[object]
    family_root: type[CollectedFamily]
    export_name: str | None = None


class FamilyGeneratingSpec(ABC):
    """Spec mixin that declares which collected families derive from the spec."""

    family_specs: ClassVar[tuple[GeneratedFamilySpec, ...]] = ()
    _declaring_spec_types: ClassVar[list[type["FamilyGeneratingSpec"]]] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("family_specs"):
            FamilyGeneratingSpec._declaring_spec_types.append(
                cast(type[FamilyGeneratingSpec], cls)
            )


class ObservationFamily(CollectedFamily, ABC):
    """Registry root for observation families derived from observation specs."""

    _registry_root = True


class ShapeFamily(CollectedFamily, ABC):
    """Registry root for structural shape families derived from shape specs."""

    _registry_root = True


class TypedLiteralObservationFamily(ObservationFamily, ABC):
    """Observation family root specialized by a literal-kind discriminator."""

    _registry_skip = True
    item_type = LiteralDispatchObservation
    spec_root: ClassVar[type[AutoRegisteredModuleShapeSpec]]
    literal_kind: ClassVar[LiteralKind]

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[object]:
        if issubclass(cls.spec_root, TypedLiteralObservationSpec):
            return [
                item
                for item in cls.spec_root().collect(parsed_module)
                if isinstance(item, LiteralDispatchObservation)
                if item.literal_kind == cls.literal_kind
            ]
        return [
            item
            for item in _collect_items_from_spec_root(
                cls.spec_root,
                parsed_module,
                LiteralDispatchObservation,
            )
            if isinstance(item, LiteralDispatchObservation)
            if item.literal_kind == cls.literal_kind
        ]


class MethodShapeSpec(FamilyGeneratingSpec, FunctionObservationSpec):
    family_specs = (GeneratedFamilySpec(MethodShape, ShapeFamily),)

    def build_from_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> MethodShape | None:
        return MethodShape(
            file_path=str(parsed_module.path),
            class_name=observation.class_name,
            method_name=function.name,
            lineno=function.lineno,
            statement_count=len(function.body),
            is_private=function.name.startswith("_")
            and not function.name.startswith("__"),
            param_count=len(function.args.args),
            decorators=tuple(
                _node_display_name(dec) for dec in function.decorator_list
            ),
            function_node=function,
        )


class BuilderCallShapeSpec(FamilyGeneratingSpec, ContextHelperShapeSpec):
    family_specs = (GeneratedFamilySpec(BuilderCallShape, ShapeFamily),)
    node_type = ast.Call


class ExportDictShapeSpec(FamilyGeneratingSpec, ContextHelperShapeSpec):
    family_specs = (GeneratedFamilySpec(ExportDictShape, ShapeFamily),)
    node_type = ast.Dict


BuilderCallShapeSpec.shape_helper = _builder_call_shape
ExportDictShapeSpec.shape_helper = _export_dict_shape

_METHOD_SHAPE_SPEC = MethodShapeSpec()
_BUILDER_CALL_SHAPE_SPEC = BuilderCallShapeSpec()
_EXPORT_DICT_SHAPE_SPEC = ExportDictShapeSpec()


class ConfigDispatchObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, FunctionObservationSpec
):
    _registry_root = True
    family_specs = (GeneratedFamilySpec(ConfigDispatchObservation, ObservationFamily),)


class ScopeFilteredFunctionObservationSpec(FunctionObservationSpec, ABC):
    @abstractmethod
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_scoped_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> object | None:
        raise NotImplementedError

    def build_from_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> object | None:
        if not self.accepts_scope(observation):
            return None
        return self.build_scoped_function(parsed_module, function, observation)


class ModuleOnlyFunctionObservationSpec(ScopeFilteredFunctionObservationSpec, ABC):
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        return observation.class_name is None


class ClassOnlyFunctionObservationSpec(ScopeFilteredFunctionObservationSpec, ABC):
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        return observation.class_name is not None


class ScopeFilteredAssignObservationSpec(AssignObservationSpec, ABC):
    @abstractmethod
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_scoped_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> object | None:
        raise NotImplementedError

    def build_from_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> object | None:
        if not self.accepts_scope(observation):
            return None
        return self.build_scoped_assign(parsed_module, node, observation)


class ModuleOnlyAssignObservationSpec(ScopeFilteredAssignObservationSpec, ABC):
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        return observation.class_name is None and observation.function_name is None


class TupleResultMixin(ABC):
    @staticmethod
    def wrap_helper_result(value: object | None) -> object | None:
        return None if value is None else tuple(cast(Any, value))


class FunctionAcceptanceMixin(ABC):
    def accepts_function(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> bool:
        del function, observation
        return True


class RequiredFunctionParameterMixin(FunctionAcceptanceMixin):
    required_parameter_name: ClassVar[str]

    def accepts_function(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> bool:
        return super().accepts_function(function, observation) and any(
            arg.arg == type(self).required_parameter_name for arg in function.args.args
        )


class SyncFunctionOnlyMixin(FunctionAcceptanceMixin):
    def accepts_function(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> bool:
        return super().accepts_function(function, observation) and isinstance(
            function, ast.FunctionDef
        )


class HelperBackedFunctionObservationSpec(
    FunctionAcceptanceMixin, FunctionObservationSpec, ABC
):
    shape_helper: ClassVar[Callable[..., object | None]]

    @staticmethod
    def wrap_helper_result(value: object | None) -> object | None:
        return value

    def accepts_function(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> bool:
        del function, observation
        return True

    def build_from_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> object | None:
        if not self.accepts_function(function, observation):
            return None
        return type(self).wrap_helper_result(
            type(self).shape_helper(parsed_module, function)
        )


class HelperBackedScopedFunctionObservationSpec(
    FunctionAcceptanceMixin, ScopeFilteredFunctionObservationSpec, ABC
):
    shape_helper: ClassVar[Callable[..., object | None]]

    @staticmethod
    def wrap_helper_result(value: object | None) -> object | None:
        return value

    def accepts_function(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> bool:
        del function, observation
        return True

    def build_scoped_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> object | None:
        if not self.accepts_function(function, observation):
            return None
        return type(self).wrap_helper_result(
            type(self).shape_helper(parsed_module, function)
        )


class ClassNamedFunctionHelperObservationSpec(ClassOnlyFunctionObservationSpec, ABC):
    shape_helper: ClassVar[Callable[..., object | None]]

    @staticmethod
    def wrap_helper_result(value: object | None) -> object | None:
        return value

    def build_scoped_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> object | None:
        class_name = observation.class_name
        if class_name is None:
            return None
        return type(self).wrap_helper_result(
            type(self).shape_helper(parsed_module, class_name, function)
        )


class HelperBackedAssignObservationSpec(AssignObservationSpec, ABC):
    shape_helper: ClassVar[Callable[..., object | None]]

    def build_from_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> object | None:
        del observation
        return type(self).shape_helper(parsed_module, node)


class HelperBackedScopedAssignObservationSpec(ScopeFilteredAssignObservationSpec, ABC):
    shape_helper: ClassVar[Callable[..., object | None]]

    def build_scoped_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> object | None:
        del observation
        return type(self).shape_helper(parsed_module, node)


class ObservationContextHelperShapeSpec(ContextForwardingShapeSpec, ABC):
    shape_helper: ClassVar[Callable[..., object | None]]

    def build_from_context(
        self,
        parsed_module: ParsedModule,
        node: ast.AST,
        observation: ScopedAstObservation,
    ) -> object | None:
        return type(self).shape_helper(parsed_module, node, observation)


class StandardConfigDispatchObservationSpec(
    ConfigDispatchObservationSpec,
    ModuleOnlyFunctionObservationSpec,
    RequiredFunctionParameterMixin,
    TupleResultMixin,
    HelperBackedScopedFunctionObservationSpec,
):
    required_parameter_name = "config"
    shape_helper = _config_dispatch_observations


class ClassMarkerObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, FunctionObservationSpec
):
    _registry_root = True
    family_specs = (GeneratedFamilySpec(ClassMarkerObservation, ObservationFamily),)


class StandardClassMarkerObservationSpec(
    ClassMarkerObservationSpec,
    TupleResultMixin,
    HelperBackedFunctionObservationSpec,
):
    shape_helper = _class_marker_observations


class InterfaceGenerationObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, FunctionObservationSpec
):
    _registry_root = True
    family_specs = (
        GeneratedFamilySpec(InterfaceGenerationObservation, ObservationFamily),
    )


class StandardInterfaceGenerationObservationSpec(
    InterfaceGenerationObservationSpec,
    HelperBackedFunctionObservationSpec,
):
    shape_helper = _interface_generation_observation


class SentinelTypeObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, ABC
):
    _registry_root = True
    family_specs = (GeneratedFamilySpec(SentinelTypeObservation, ObservationFamily),)


class SentinelTypeAssignmentObservationSpec(
    SentinelTypeObservationSpec,
    ModuleOnlyAssignObservationSpec,
    HelperBackedScopedAssignObservationSpec,
):
    shape_helper = _sentinel_type_observation


class SentinelTypeUsageObservationSpec(SentinelTypeObservationSpec):
    def collect(self, parsed_module: ParsedModule) -> list[object]:
        return list(_sentinel_type_usage_observations(parsed_module))


class DynamicMethodInjectionObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, FunctionObservationSpec
):
    _registry_root = True
    family_specs = (
        GeneratedFamilySpec(DynamicMethodInjectionObservation, ObservationFamily),
    )


class StandardDynamicMethodInjectionObservationSpec(
    DynamicMethodInjectionObservationSpec,
    TupleResultMixin,
    HelperBackedFunctionObservationSpec,
):
    shape_helper = _dynamic_method_injection_observations


class RuntimeTypeGenerationObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, ObservationShapeSpec, ABC
):
    _registry_root = True
    family_specs = (
        GeneratedFamilySpec(RuntimeTypeGenerationObservation, ObservationFamily),
    )


class TypeCallGenerationObservationSpec(
    RuntimeTypeGenerationObservationSpec, ObservationContextHelperShapeSpec
):
    node_type = ast.Call
    shape_helper = _runtime_type_generation_observation


class LineageMappingObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, AssignObservationSpec, ABC
):
    _registry_root = True
    family_specs = (GeneratedFamilySpec(LineageMappingObservation, ObservationFamily),)


class StandardLineageMappingObservationSpec(
    LineageMappingObservationSpec,
    HelperBackedAssignObservationSpec,
):
    shape_helper = _lineage_mapping_observation


class DualAxisResolutionObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, FunctionObservationSpec, ABC
):
    _registry_root = True
    family_specs = (
        GeneratedFamilySpec(DualAxisResolutionObservation, ObservationFamily),
    )


class StandardDualAxisResolutionObservationSpec(
    DualAxisResolutionObservationSpec,
    HelperBackedFunctionObservationSpec,
):
    shape_helper = _dual_axis_resolution_observation


class AttributeProbeObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, ABC
):
    _registry_root = True
    family_specs = (GeneratedFamilySpec(AttributeProbeObservation, ObservationFamily),)


class CallAttributeProbeObservationSpec(
    AttributeProbeObservationSpec, ContextForwardingShapeSpec, ABC
):
    node_type = ast.Call
    _registry_skip = True
    call_family: ClassVar[AstNameFamily]
    probe_kind: ClassVar[str]
    minimum_args: ClassVar[int]
    attribute_arg_index: ClassVar[int | None] = None

    def build_from_context(
        self,
        parsed_module: ParsedModule,
        node: ast.AST,
        observation: ScopedAstObservation,
    ) -> AttributeProbeObservation | None:
        if not isinstance(node, ast.Call):
            return None
        if _terminal_name_in_family(node.func, type(self).call_family) is None:
            return None
        if len(node.args) < type(self).minimum_args:
            return None
        observed_attribute = None
        attribute_arg_index = type(self).attribute_arg_index
        if attribute_arg_index is not None and len(node.args) > attribute_arg_index:
            arg = node.args[attribute_arg_index]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                observed_attribute = arg.value
        return AttributeProbeObservation(
            file_path=str(parsed_module.path),
            line=node.lineno,
            symbol=type(self).probe_kind,
            probe_kind=type(self).probe_kind,
            observed_attribute=observed_attribute,
            execution_level=_execution_level_for_scope(observation.function_name),
        )


class HasAttrProbeObservationSpec(CallAttributeProbeObservationSpec):
    call_family = _HASATTR_CALL_FAMILY
    probe_kind = "hasattr"
    minimum_args = 2
    attribute_arg_index = 1


class GetAttrProbeObservationSpec(CallAttributeProbeObservationSpec):
    call_family = _GETATTR_CALL_FAMILY
    probe_kind = "getattr"
    minimum_args = 3
    attribute_arg_index = 1


class AttributeErrorProbeObservationSpec(
    AttributeProbeObservationSpec, ObservationShapeSpec
):
    @property
    def node_types(self) -> tuple[type[ast.AST], ...]:
        return (ast.Try,)

    def build_from_observation(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> AttributeProbeObservation | None:
        node = observation.node
        if not isinstance(node, ast.Try):
            return None
        for handler in node.handlers:
            if handler.type is None:
                continue
            if not _node_matches_family(handler.type, _ATTRIBUTE_ERROR_FAMILY):
                continue
            return AttributeProbeObservation(
                file_path=str(parsed_module.path),
                line=handler.lineno,
                symbol="attribute-error-fallback",
                probe_kind="attribute_error",
                observed_attribute=None,
                execution_level=_execution_level_for_scope(observation.function_name),
            )
        return None


class TypedLiteralObservationSpec(AutoRegisteredModuleShapeSpec, ABC):
    literal_type: ClassVar[type[object]]

    @classmethod
    def registered_specs_for_literal_type(
        cls, literal_type: type[object] | None = None
    ) -> tuple[TypedLiteralObservationSpec, ...]:
        specs = tuple(
            spec
            for spec in cls.registered_specs()
            if isinstance(spec, TypedLiteralObservationSpec)
        )
        if literal_type is None:
            return specs
        return tuple(spec for spec in specs if type(spec).literal_type is literal_type)


class LiteralDispatchObservationSpec(TypedLiteralObservationSpec, ABC):
    _registry_root = True
    literal_kind: ClassVar[LiteralKind]

    def collect(self, parsed_module: ParsedModule) -> list[object]:
        return list(
            _literal_dispatch_observations_for_kind(
                parsed_module,
                type(self).literal_kind,
            )
        )


class StringLiteralDispatchObservationSpec(
    FamilyGeneratingSpec, LiteralDispatchObservationSpec
):
    family_specs = (
        GeneratedFamilySpec(LiteralDispatchObservation, TypedLiteralObservationFamily),
    )
    literal_type = str
    literal_kind = LiteralKind.STRING


class NumericLiteralDispatchObservationSpec(
    FamilyGeneratingSpec, LiteralDispatchObservationSpec
):
    family_specs = (
        GeneratedFamilySpec(LiteralDispatchObservation, TypedLiteralObservationFamily),
    )
    literal_type = int
    literal_kind = LiteralKind.NUMERIC


class InlineLiteralDispatchObservationSpec(TypedLiteralObservationSpec, ABC):
    _registry_root = True
    literal_kind: ClassVar[LiteralKind]

    def collect(self, parsed_module: ParsedModule) -> list[object]:
        return list(
            _inline_literal_dispatch_observations_for_kind(
                parsed_module,
                type(self).literal_kind,
            )
        )


class InlineStringLiteralDispatchObservationSpec(
    FamilyGeneratingSpec, InlineLiteralDispatchObservationSpec
):
    family_specs = (
        GeneratedFamilySpec(LiteralDispatchObservation, TypedLiteralObservationFamily),
    )
    literal_type = str
    literal_kind = LiteralKind.STRING


class RegistrationShapeSpec(FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, ABC):
    _registry_root = True
    family_specs = (GeneratedFamilySpec(RegistrationShape, ShapeFamily),)


class KnownClassFamilyShapeSpec(RegistrationShapeSpec, ABC):
    def collect(self, parsed_module: ParsedModule) -> list[object]:
        return self.collect_with_known_class_family(
            parsed_module,
            _known_class_family(parsed_module),
        )

    @abstractmethod
    def collect_with_known_class_family(
        self,
        parsed_module: ParsedModule,
        known_class_family: AstNameFamily,
    ) -> list[object]:
        raise NotImplementedError


class AssignmentRegistrationShapeSpec(KnownClassFamilyShapeSpec):
    def collect_with_known_class_family(
        self,
        parsed_module: ParsedModule,
        known_class_family: AstNameFamily,
    ) -> list[object]:
        shapes: list[object] = []
        for node in ast.walk(parsed_module.module):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Name):
                continue
            if _terminal_name_in_family(node.value, known_class_family) is None:
                continue
            for target in node.targets:
                registry_name = _subscript_base_name(target)
                if registry_name is None:
                    continue
                key_fingerprint = _registration_key_fingerprint(target)
                if key_fingerprint is None:
                    continue
                shapes.append(
                    RegistrationShape.from_assignment(
                        parsed_module,
                        node,
                        registry_name,
                        key_fingerprint,
                    )
                )
        return shapes


class CallRegistrationShapeSpec(KnownClassFamilyShapeSpec):
    def collect_with_known_class_family(
        self,
        parsed_module: ParsedModule,
        known_class_family: AstNameFamily,
    ) -> list[object]:
        shapes: list[object] = []
        for observation in _iter_attribute_family_calls(
            parsed_module, _REGISTRATION_CALL_FAMILY
        ):
            node = observation.call
            assert isinstance(node.func, ast.Attribute)
            registry_name = _terminal_name(node.func.value)
            if registry_name is None:
                continue
            if not node.args:
                continue
            class_name = _class_name_from_expr(node.args[0], known_class_family)
            if class_name is None:
                continue
            key_source = node.args[1] if len(node.args) >= 2 else node.args[0]
            key_fingerprint = _fingerprint_builder_value(key_source)
            shapes.append(
                RegistrationShape.from_registration_call(
                    parsed_module,
                    node,
                    registry_name,
                    class_name,
                    key_fingerprint,
                )
            )
        return shapes


class DecoratorRegistrationShapeSpec(RegistrationShapeSpec):
    def collect(self, parsed_module: ParsedModule) -> list[object]:
        shapes: list[object] = []
        for node, decorator, _matched_name in _iter_class_decorator_family_calls(
            parsed_module, _REGISTRATION_DECORATOR_FAMILY
        ):
            if not decorator.args:
                continue
            registry_name = _terminal_name(decorator.args[0])
            if registry_name is None:
                continue
            key_expr = (
                decorator.args[1]
                if len(decorator.args) >= 2
                else ast.Constant(value=node.name)
            )
            shapes.append(
                RegistrationShape.from_decorator(
                    parsed_module,
                    node,
                    registry_name,
                    _fingerprint_builder_value(key_expr),
                )
            )
        return shapes


class FieldObservationSpec(FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, ABC):
    _registry_root = True
    family_specs = (GeneratedFamilySpec(FieldObservation, ObservationFamily),)


class ClassObservationSpec(FieldObservationSpec, ABC):
    def collect(self, parsed_module: ParsedModule) -> list[object]:
        observations: list[object] = []
        for class_observation in _class_observations(parsed_module):
            observations.extend(
                self.collect_for_class(parsed_module, class_observation)
            )
        return observations

    @abstractmethod
    def collect_for_class(
        self,
        parsed_module: ParsedModule,
        class_observation: ClassAstObservation,
    ) -> list[object]:
        raise NotImplementedError


class DataclassBodyFieldObservationSpec(ClassObservationSpec):
    def collect_for_class(
        self,
        parsed_module: ParsedModule,
        class_observation: ClassAstObservation,
    ) -> list[object]:
        observations: list[object] = []
        for stmt in class_observation.node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                continue
            field_observation = _class_body_field_observation(
                parsed_module,
                class_observation.node.name,
                class_observation.is_dataclass_family,
                stmt,
            )
            if field_observation is not None:
                observations.append(field_observation)
        return observations


class InitAssignmentFieldObservationSpec(ClassObservationSpec):
    def collect_for_class(
        self,
        parsed_module: ParsedModule,
        class_observation: ClassAstObservation,
    ) -> list[object]:
        observations: list[object] = []
        for stmt in class_observation.node.body:
            if not isinstance(stmt, ast.FunctionDef) or stmt.name != "__init__":
                continue
            observations.extend(
                _init_field_observations(
                    parsed_module,
                    class_observation.node.name,
                    class_observation.is_dataclass_family,
                    stmt,
                )
            )
        return observations


class ProjectionHelperObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, FunctionObservationSpec, ABC
):
    _registry_root = True
    family_specs = (GeneratedFamilySpec(ProjectionHelperShape, ObservationFamily),)


class AccessorWrapperObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, FunctionObservationSpec, ABC
):
    _registry_root = True
    family_specs = (GeneratedFamilySpec(AccessorWrapperCandidate, ObservationFamily),)


class StandardProjectionHelperObservationSpec(
    ProjectionHelperObservationSpec,
    ModuleOnlyFunctionObservationSpec,
    HelperBackedScopedFunctionObservationSpec,
):
    shape_helper = _projection_helper_shape_from_function


class StandardAccessorWrapperObservationSpec(
    AccessorWrapperObservationSpec,
    ClassNamedFunctionHelperObservationSpec,
):
    shape_helper = _accessor_wrapper_candidate_from_function


class ScopedShapeWrapperObservationSpec(
    FamilyGeneratingSpec, AutoRegisteredModuleShapeSpec, ABC
):
    _registry_root = True
    family_specs = (
        GeneratedFamilySpec(
            ScopedShapeWrapperFunction,
            ObservationFamily,
            "ScopedShapeWrapperFunctionFamily",
        ),
        GeneratedFamilySpec(
            ScopedShapeWrapperSpec,
            ObservationFamily,
            "ScopedShapeWrapperSpecFamily",
        ),
    )


class ScopedShapeWrapperFunctionObservationSpec(
    ScopedShapeWrapperObservationSpec,
    ModuleOnlyFunctionObservationSpec,
    SyncFunctionOnlyMixin,
    HelperBackedScopedFunctionObservationSpec,
):
    shape_helper = _scoped_shape_wrapper_function_from_function


class ScopedShapeWrapperSpecObservationSpec(
    ScopedShapeWrapperObservationSpec,
    ModuleOnlyAssignObservationSpec,
    HelperBackedScopedAssignObservationSpec,
):
    shape_helper = _scoped_shape_wrapper_spec_from_assign


def _registered_family_types() -> tuple[type[CollectedFamily], ...]:
    return CollectedFamily.all_registered_families()


def family_for_item_type(item_type: type[object]) -> type[CollectedFamily]:
    """Return the generated family that owns one collected item type."""
    for family in _registered_family_types():
        if family.item_type is item_type:
            return family
    raise KeyError(item_type)


def family_for_literal_kind(literal_kind: LiteralKind) -> type[CollectedFamily]:
    """Return the generated family that owns one literal-dispatch kind."""
    for family in _registered_family_types():
        if (
            issubclass(family, TypedLiteralObservationFamily)
            and family.literal_kind is literal_kind
        ):
            return family
    raise KeyError(literal_kind)


def _materialize_generated_family(
    spec_type: type[FamilyGeneratingSpec],
    family_spec: GeneratedFamilySpec,
) -> type[CollectedFamily]:
    module_globals = cast(dict[str, object], globals())
    family_root = family_spec.family_root
    export_name = (
        family_spec.export_name or spec_type.__name__.removesuffix("Spec") + "Family"
    )
    attributes: dict[str, object] = {
        "__module__": __name__,
        "item_type": family_spec.item_type,
    }
    if family_root is TypedLiteralObservationFamily:
        attributes["spec_root"] = cast(type[AutoRegisteredModuleShapeSpec], spec_type)
        attributes["literal_kind"] = cast(Any, spec_type).literal_kind
        family_bases = (TypedLiteralObservationFamily,)
    elif issubclass(spec_type, AutoRegisteredModuleShapeSpec):
        attributes["spec_root"] = cast(type[AutoRegisteredModuleShapeSpec], spec_type)
        family_bases = (RegisteredSpecCollectedFamily, family_root)
    else:
        attributes["spec"] = spec_type()
        family_bases = (SingleSpecCollectedFamily, family_root)
    family_type = cast(
        type[CollectedFamily],
        AutoRegisterMeta(export_name, family_bases, attributes),
    )
    module_globals[export_name] = family_type
    return family_type


def _materialize_declared_families() -> dict[str, type[CollectedFamily]]:
    families: dict[str, type[CollectedFamily]] = {}
    for spec_type in FamilyGeneratingSpec._declaring_spec_types:
        for family_spec in spec_type.family_specs:
            family_type = _materialize_generated_family(spec_type, family_spec)
            families[family_type.__name__] = family_type
    return families


_FAMILY_EXPORTS = _materialize_declared_families()
_FAMILY_EXPORT_NAMES = tuple(_FAMILY_EXPORTS)


_PUBLIC_EXPORT_POLICY = PublicExportPolicy(
    module_name=__name__,
    root_types=(CollectedFamily, AutoRegisteredModuleShapeSpec),
    explicit_names=frozenset({"AutoRegisteredModuleShapeSpec"}),
)


__all__ = derive_public_exports(globals(), _PUBLIC_EXPORT_POLICY)
