from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import ClassVar

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
    _inline_literal_dispatch_groups,
    _interface_generation_observation,
    _iter_attribute_family_calls,
    _iter_class_decorator_family_calls,
    _iter_statement_blocks,
    _known_class_family,
    _lineage_mapping_observation,
    _literal_dispatch_observation_from_if,
    _node_display_name,
    _node_matches_family,
    _parent_map,
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
    fingerprint_function,
)


class MethodShapeSpec(FunctionObservationSpec):
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
            fingerprint=fingerprint_function(function),
            statement_texts=tuple(ast.unparse(stmt) for stmt in function.body),
        )


class BuilderCallShapeSpec(ContextHelperShapeSpec):
    node_type = ast.Call


class ExportDictShapeSpec(ContextHelperShapeSpec):
    node_type = ast.Dict


BuilderCallShapeSpec.shape_helper = _builder_call_shape
ExportDictShapeSpec.shape_helper = _export_dict_shape

_METHOD_SHAPE_SPEC = MethodShapeSpec()
_BUILDER_CALL_SHAPE_SPEC = BuilderCallShapeSpec()
_EXPORT_DICT_SHAPE_SPEC = ExportDictShapeSpec()


class ConfigDispatchObservationSpec(
    AutoRegisteredModuleShapeSpec, FunctionObservationSpec
):
    _registry_root = True


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


class StandardConfigDispatchObservationSpec(
    ConfigDispatchObservationSpec, ModuleOnlyFunctionObservationSpec
):
    def build_scoped_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> object | None:
        if not any(arg.arg == "config" for arg in function.args.args):
            return None
        return tuple(_config_dispatch_observations(parsed_module, function))


class ClassMarkerObservationSpec(
    AutoRegisteredModuleShapeSpec, FunctionObservationSpec
):
    _registry_root = True


class StandardClassMarkerObservationSpec(ClassMarkerObservationSpec):
    def build_from_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> object | None:
        return tuple(_class_marker_observations(parsed_module, function))


class InterfaceGenerationObservationSpec(
    AutoRegisteredModuleShapeSpec, FunctionObservationSpec
):
    _registry_root = True


class StandardInterfaceGenerationObservationSpec(InterfaceGenerationObservationSpec):
    def build_from_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> object | None:
        return _interface_generation_observation(parsed_module, function)


class SentinelTypeObservationSpec(AutoRegisteredModuleShapeSpec, ABC):
    _registry_root = True


class SentinelTypeAssignmentObservationSpec(
    SentinelTypeObservationSpec, ModuleOnlyAssignObservationSpec
):
    def build_scoped_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> object | None:
        return _sentinel_type_observation(parsed_module, node)


class SentinelTypeUsageObservationSpec(SentinelTypeObservationSpec):
    def collect(self, parsed_module: ParsedModule) -> list[object]:
        return list(_sentinel_type_usage_observations(parsed_module))


class DynamicMethodInjectionObservationSpec(
    AutoRegisteredModuleShapeSpec, FunctionObservationSpec
):
    _registry_root = True


class StandardDynamicMethodInjectionObservationSpec(
    DynamicMethodInjectionObservationSpec
):
    def build_from_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> object | None:
        return tuple(_dynamic_method_injection_observations(parsed_module, function))


class RuntimeTypeGenerationObservationSpec(
    AutoRegisteredModuleShapeSpec, ObservationShapeSpec, ABC
):
    _registry_root = True


class TypeCallGenerationObservationSpec(
    RuntimeTypeGenerationObservationSpec, ContextForwardingShapeSpec
):
    node_type = ast.Call

    def build_from_context(
        self,
        parsed_module: ParsedModule,
        node: ast.AST,
        observation: ScopedAstObservation,
    ) -> RuntimeTypeGenerationObservation | None:
        assert isinstance(node, ast.Call)
        return _runtime_type_generation_observation(parsed_module, node, observation)


class LineageMappingObservationSpec(
    AutoRegisteredModuleShapeSpec, AssignObservationSpec, ABC
):
    _registry_root = True


class StandardLineageMappingObservationSpec(LineageMappingObservationSpec):
    def build_from_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> object | None:
        return _lineage_mapping_observation(parsed_module, node)


class DualAxisResolutionObservationSpec(
    AutoRegisteredModuleShapeSpec, FunctionObservationSpec, ABC
):
    _registry_root = True


class StandardDualAxisResolutionObservationSpec(DualAxisResolutionObservationSpec):
    def build_from_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> object | None:
        return _dual_axis_resolution_observation(parsed_module, function)


class AttributeProbeObservationSpec(AutoRegisteredModuleShapeSpec, ABC):
    _registry_root = True


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
        parent_map = _parent_map(parsed_module.module)
        observations: list[object] = []
        for node in ast.walk(parsed_module.module):
            if not isinstance(node, ast.If):
                continue
            parent = parent_map.get(node)
            if (
                isinstance(parent, ast.If)
                and len(parent.orelse) == 1
                and parent.orelse[0] is node
            ):
                continue
            observation = _literal_dispatch_observation_from_if(
                parsed_module,
                node,
                type(self).literal_type,
                type(self).literal_kind,
                parent_map,
            )
            if observation is not None:
                observations.append(observation)
        return observations


class StringLiteralDispatchObservationSpec(LiteralDispatchObservationSpec):
    literal_type = str
    literal_kind = LiteralKind.STRING


class NumericLiteralDispatchObservationSpec(LiteralDispatchObservationSpec):
    literal_type = int
    literal_kind = LiteralKind.NUMERIC


class InlineLiteralDispatchObservationSpec(TypedLiteralObservationSpec, ABC):
    _registry_root = True
    literal_kind: ClassVar[LiteralKind]

    def collect(self, parsed_module: ParsedModule) -> list[object]:
        observations: list[object] = []
        for owner_name, block in _iter_statement_blocks(parsed_module.module):
            observations.extend(
                _inline_literal_dispatch_groups(
                    parsed_module,
                    owner_name,
                    block,
                    type(self).literal_type,
                    type(self).literal_kind,
                )
            )
        return observations


class InlineStringLiteralDispatchObservationSpec(InlineLiteralDispatchObservationSpec):
    literal_type = str
    literal_kind = LiteralKind.STRING


class RegistrationShapeSpec(AutoRegisteredModuleShapeSpec, ABC):
    _registry_root = True


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


class FieldObservationSpec(AutoRegisteredModuleShapeSpec, ABC):
    _registry_root = True


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
    AutoRegisteredModuleShapeSpec, FunctionObservationSpec, ABC
):
    _registry_root = True


class AccessorWrapperObservationSpec(
    AutoRegisteredModuleShapeSpec, FunctionObservationSpec, ABC
):
    _registry_root = True


class StandardProjectionHelperObservationSpec(
    ProjectionHelperObservationSpec, ModuleOnlyFunctionObservationSpec
):
    def build_scoped_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> ProjectionHelperShape | None:
        return _projection_helper_shape_from_function(parsed_module, function)


class StandardAccessorWrapperObservationSpec(
    AccessorWrapperObservationSpec, ClassOnlyFunctionObservationSpec
):
    def build_scoped_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> AccessorWrapperCandidate | None:
        class_name = observation.class_name
        if class_name is None:
            return None
        return _accessor_wrapper_candidate_from_function(
            parsed_module,
            class_name,
            function,
        )


class ScopedShapeWrapperObservationSpec(AutoRegisteredModuleShapeSpec, ABC):
    _registry_root = True


class ScopedShapeWrapperFunctionObservationSpec(
    ScopedShapeWrapperObservationSpec, ModuleOnlyFunctionObservationSpec
):
    def build_scoped_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> ScopedShapeWrapperFunction | None:
        if not isinstance(function, ast.FunctionDef):
            return None
        return _scoped_shape_wrapper_function_from_function(parsed_module, function)


class ScopedShapeWrapperSpecObservationSpec(
    ScopedShapeWrapperObservationSpec, ModuleOnlyAssignObservationSpec
):
    def build_scoped_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> ScopedShapeWrapperSpec | None:
        return _scoped_shape_wrapper_spec_from_assign(parsed_module, node)


class ObservationFamily(CollectedFamily, ABC):
    _registry_root = True


class ShapeFamily(CollectedFamily, ABC):
    _registry_root = True


class MethodShapeFamily(SingleSpecCollectedFamily, ShapeFamily):
    item_type = MethodShape
    spec = _METHOD_SHAPE_SPEC


class BuilderCallShapeFamily(SingleSpecCollectedFamily, ShapeFamily):
    item_type = BuilderCallShape
    spec = _BUILDER_CALL_SHAPE_SPEC


class RegistrationShapeFamily(RegisteredSpecCollectedFamily, ShapeFamily):
    item_type = RegistrationShape
    spec_root = RegistrationShapeSpec


class ExportDictShapeFamily(SingleSpecCollectedFamily, ShapeFamily):
    item_type = ExportDictShape
    spec = _EXPORT_DICT_SHAPE_SPEC


class FieldObservationFamily(RegisteredSpecCollectedFamily, ObservationFamily):
    item_type = FieldObservation
    spec_root = FieldObservationSpec


class ProjectionHelperObservationFamily(
    RegisteredSpecCollectedFamily, ObservationFamily
):
    item_type = ProjectionHelperShape
    spec_root = ProjectionHelperObservationSpec


class AccessorWrapperObservationFamily(
    RegisteredSpecCollectedFamily, ObservationFamily
):
    item_type = AccessorWrapperCandidate
    spec_root = AccessorWrapperObservationSpec


class ScopedShapeWrapperFunctionFamily(
    RegisteredSpecCollectedFamily, ObservationFamily
):
    item_type = ScopedShapeWrapperFunction
    spec_root = ScopedShapeWrapperObservationSpec


class ScopedShapeWrapperSpecFamily(RegisteredSpecCollectedFamily, ObservationFamily):
    item_type = ScopedShapeWrapperSpec
    spec_root = ScopedShapeWrapperObservationSpec


class ConfigDispatchObservationFamily(RegisteredSpecCollectedFamily, ObservationFamily):
    item_type = ConfigDispatchObservation
    spec_root = ConfigDispatchObservationSpec


class ClassMarkerObservationFamily(RegisteredSpecCollectedFamily, ObservationFamily):
    item_type = ClassMarkerObservation
    spec_root = ClassMarkerObservationSpec


class InterfaceGenerationObservationFamily(
    RegisteredSpecCollectedFamily, ObservationFamily
):
    item_type = InterfaceGenerationObservation
    spec_root = InterfaceGenerationObservationSpec


class SentinelTypeObservationFamily(RegisteredSpecCollectedFamily, ObservationFamily):
    item_type = SentinelTypeObservation
    spec_root = SentinelTypeObservationSpec


class DynamicMethodInjectionObservationFamily(
    RegisteredSpecCollectedFamily, ObservationFamily
):
    item_type = DynamicMethodInjectionObservation
    spec_root = DynamicMethodInjectionObservationSpec


class RuntimeTypeGenerationObservationFamily(
    RegisteredSpecCollectedFamily, ObservationFamily
):
    item_type = RuntimeTypeGenerationObservation
    spec_root = RuntimeTypeGenerationObservationSpec


class LineageMappingObservationFamily(RegisteredSpecCollectedFamily, ObservationFamily):
    item_type = LineageMappingObservation
    spec_root = LineageMappingObservationSpec


class DualAxisResolutionObservationFamily(
    RegisteredSpecCollectedFamily, ObservationFamily
):
    item_type = DualAxisResolutionObservation
    spec_root = DualAxisResolutionObservationSpec


class AttributeProbeObservationFamily(RegisteredSpecCollectedFamily, ObservationFamily):
    item_type = AttributeProbeObservation
    spec_root = AttributeProbeObservationSpec


class TypedLiteralObservationFamily(ObservationFamily, ABC):
    _registry_skip = True
    item_type = LiteralDispatchObservation
    spec_root: ClassVar[type[AutoRegisteredModuleShapeSpec]]
    literal_kind: ClassVar[LiteralKind]

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[object]:
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


class StringLiteralDispatchObservationFamily(TypedLiteralObservationFamily):
    spec_root = LiteralDispatchObservationSpec
    literal_kind = LiteralKind.STRING


class NumericLiteralDispatchObservationFamily(TypedLiteralObservationFamily):
    spec_root = LiteralDispatchObservationSpec
    literal_kind = LiteralKind.NUMERIC


class InlineStringLiteralDispatchObservationFamily(TypedLiteralObservationFamily):
    spec_root = InlineLiteralDispatchObservationSpec
    literal_kind = LiteralKind.STRING


__all__ = [
    "AccessorWrapperObservationFamily",
    "AccessorWrapperObservationSpec",
    "AssignmentRegistrationShapeSpec",
    "AttributeErrorProbeObservationSpec",
    "AttributeProbeObservationFamily",
    "AttributeProbeObservationSpec",
    "AutoRegisteredModuleShapeSpec",
    "BuilderCallShapeFamily",
    "BuilderCallShapeSpec",
    "CallAttributeProbeObservationSpec",
    "CallRegistrationShapeSpec",
    "ClassMarkerObservationFamily",
    "ClassMarkerObservationSpec",
    "ClassObservationSpec",
    "ConfigDispatchObservationFamily",
    "ConfigDispatchObservationSpec",
    "DataclassBodyFieldObservationSpec",
    "DecoratorRegistrationShapeSpec",
    "DualAxisResolutionObservationFamily",
    "DualAxisResolutionObservationSpec",
    "DynamicMethodInjectionObservationFamily",
    "DynamicMethodInjectionObservationSpec",
    "ExportDictShapeFamily",
    "ExportDictShapeSpec",
    "FieldObservationFamily",
    "FieldObservationSpec",
    "GetAttrProbeObservationSpec",
    "HasAttrProbeObservationSpec",
    "InitAssignmentFieldObservationSpec",
    "InlineLiteralDispatchObservationSpec",
    "InlineStringLiteralDispatchObservationFamily",
    "InlineStringLiteralDispatchObservationSpec",
    "InterfaceGenerationObservationFamily",
    "InterfaceGenerationObservationSpec",
    "KnownClassFamilyShapeSpec",
    "LineageMappingObservationFamily",
    "LineageMappingObservationSpec",
    "LiteralDispatchObservationSpec",
    "MethodShapeFamily",
    "MethodShapeSpec",
    "NumericLiteralDispatchObservationFamily",
    "NumericLiteralDispatchObservationSpec",
    "ObservationFamily",
    "ProjectionHelperObservationFamily",
    "ProjectionHelperObservationSpec",
    "RegistrationShapeFamily",
    "RegistrationShapeSpec",
    "RuntimeTypeGenerationObservationFamily",
    "RuntimeTypeGenerationObservationSpec",
    "ScopedShapeWrapperFunctionFamily",
    "ScopedShapeWrapperFunctionObservationSpec",
    "ScopedShapeWrapperObservationSpec",
    "ScopedShapeWrapperSpecFamily",
    "ScopedShapeWrapperSpecObservationSpec",
    "SentinelTypeAssignmentObservationSpec",
    "SentinelTypeObservationFamily",
    "SentinelTypeObservationSpec",
    "SentinelTypeUsageObservationSpec",
    "ShapeFamily",
    "StandardAccessorWrapperObservationSpec",
    "StandardClassMarkerObservationSpec",
    "StandardConfigDispatchObservationSpec",
    "StandardDualAxisResolutionObservationSpec",
    "StandardDynamicMethodInjectionObservationSpec",
    "StandardInterfaceGenerationObservationSpec",
    "StandardLineageMappingObservationSpec",
    "StandardProjectionHelperObservationSpec",
    "StringLiteralDispatchObservationFamily",
    "StringLiteralDispatchObservationSpec",
    "TypeCallGenerationObservationSpec",
    "TypedLiteralObservationFamily",
    "TypedLiteralObservationSpec",
]
