"""Observation spec families and spec-derived family generation.

This module declares the public observation specs used by the advisor and derives
their exported collected families from the spec definitions themselves. The goal is
to keep one nominal authority per family and derive runtime family surfaces from it.
"""

from __future__ import annotations

import ast
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, ClassVar, Generic, ParamSpec, TypeAlias, TypeVar, cast

from .export_tools import PublicExportPolicy, derive_public_exports
from .collection_algebra import sorted_tuple
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key

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
    CLASS_OBSERVATION_PROJECTION,
    CollectedFamily,
    COLLECTED_ITEM_PROJECTION,
    ContextForwardingShapeSpec,
    ContextHelperShapeSpec,
    FunctionObservationSpec,
    ObservationShapeSpec,
    ParsedModule,
    REGISTERED_TYPE_LINEAGE,
    RegisteredSpecCollectedFamily,
    ScopedAstObservation,
    SharedRegistryRootBase,
    ShapeEmission,
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

GeneratedItemT = TypeVar("GeneratedItemT")
GeneratedHelperParams = ParamSpec("GeneratedHelperParams")


@dataclass(frozen=True)
class GeneratedShapeHelper(Generic[GeneratedHelperParams, GeneratedItemT]):
    """Nominal wrapper for generated observation and shape helper callables."""

    function: Callable[GeneratedHelperParams, ShapeEmission[GeneratedItemT] | None]

    def __call__(
        self,
        *args: GeneratedHelperParams.args,
        **kwargs: GeneratedHelperParams.kwargs,
    ) -> ShapeEmission[GeneratedItemT] | None:
        return self.function(*args, **kwargs)


@dataclass(frozen=True)
class GeneratedFamilySpec(Generic[GeneratedItemT]):
    """Declarative recipe for one generated collected family export."""

    item_type: type[GeneratedItemT]
    family_root: type[CollectedFamily[GeneratedItemT]]
    export_name: str | None = None


GeneratedFamilySpecSet: TypeAlias = tuple[GeneratedFamilySpec, ...]
GeneratedClassTypeValue: TypeAlias = type[ast.AST] | type[str] | type[int]
GeneratedClassAttributeValue: TypeAlias = (
    bool
    | int
    | str
    | LiteralKind
    | AstNameFamily
    | GeneratedClassTypeValue
    | GeneratedFamilySpecSet
    | GeneratedShapeHelper[..., GeneratedItemT]
)
GeneratedClassAttribute: TypeAlias = tuple[str, GeneratedClassAttributeValue]
GeneratedNamespaceValue: TypeAlias = (
    str | int | GeneratedClassAttributeValue | type[CollectedFamily]
)


@dataclass(frozen=True)
class _GeneratedClassDeclaration:
    """Declarative recipe for one metadata-only generated class."""

    class_name: str
    base_names: tuple[str, ...]
    attributes: tuple[GeneratedClassAttribute, ...] = ()


@dataclass(frozen=True)
class GeneratedClassDeclarationFactory:
    def class_declaration(
        self,
        class_name: str,
        *base_names: str,
        **attributes: GeneratedClassAttributeValue,
    ) -> _GeneratedClassDeclaration:
        return _GeneratedClassDeclaration(
            class_name,
            tuple((name for group in base_names for name in group.split())),
            tuple(attributes.items()),
        )

    def root_spec_declaration(
        self,
        class_name: str,
        item_type: type[GeneratedItemT],
        family_root: type[CollectedFamily[GeneratedItemT]],
        *base_names: str,
        export_name: str | None = None,
    ) -> _GeneratedClassDeclaration:
        return self.class_declaration(
            class_name,
            "FamilyGeneratingSpec",
            *base_names,
            _registry_root=True,
            family_specs=_family_specs(item_type, family_root, export_name),
        )


_GENERATED_CLASS_DECLARATIONS = GeneratedClassDeclarationFactory()


def _family_specs(
    item_type: type[GeneratedItemT],
    family_root: type[CollectedFamily[GeneratedItemT]],
    export_name: str | None = None,
) -> tuple[GeneratedFamilySpec[GeneratedItemT], ...]:
    return (GeneratedFamilySpec(item_type, family_root, export_name),)


def _materialize_class_declarations(
    declarations: tuple[_GeneratedClassDeclaration, ...],
) -> None:
    frame = inspect.currentframe()
    caller = None if frame is None else frame.f_back
    base_lineno = 0 if caller is None else caller.f_lineno
    module_globals = cast(dict[str, GeneratedNamespaceValue], globals())
    for offset, declaration in enumerate(declarations):
        namespace: dict[str, GeneratedNamespaceValue] = {
            "__module__": __name__,
            "__qualname__": declaration.class_name,
            "__firstlineno__": base_lineno + offset,
        }
        namespace.update(dict(declaration.attributes))
        bases = tuple(
            (
                cast(type, module_globals[base_name])
                for base_name in declaration.base_names
            )
        )
        module_globals[declaration.class_name] = AutoRegisterMeta(
            declaration.class_name, bases, namespace
        )


class FamilyGeneratingSpec(ABC):
    """Spec mixin that declares which collected families derive from the spec."""

    family_specs: ClassVar[GeneratedFamilySpecSet] = ()


def _declared_family_spec_types() -> tuple[type[FamilyGeneratingSpec], ...]:
    ordered = [
        cast(type[FamilyGeneratingSpec], current)
        for current in REGISTERED_TYPE_LINEAGE.descendant_types(FamilyGeneratingSpec)
        if current.__dict__.get("family_specs")
    ]
    return sorted_tuple(
        ordered,
        key=lambda spec_type: (
            spec_type.__module__,
            int(vars(spec_type).get("__firstlineno__", 0)),
            spec_type.__qualname__,
        ),
    )


class ObservationFamily(CollectedFamily[GeneratedItemT], Generic[GeneratedItemT], ABC):
    """Registry root for observation families derived from observation specs."""

    _registry_root = True


class ShapeFamily(CollectedFamily[GeneratedItemT], Generic[GeneratedItemT], ABC):
    """Registry root for structural shape families derived from shape specs."""

    _registry_root = True


def _observation_root_spec(
    class_name: str,
    item_type: type[GeneratedItemT],
    *base_names: str,
    export_name: str | None = None,
) -> _GeneratedClassDeclaration:
    return _GENERATED_CLASS_DECLARATIONS.root_spec_declaration(
        class_name,
        item_type,
        ObservationFamily,
        *base_names,
        export_name=export_name,
    )


def _derived_type(stem: str, suffix: str) -> type:
    return cast(type, globals()[f"{stem}{suffix}"])


def _obs_root(
    stem: str,
    *base_names: str,
    item_type: type[GeneratedItemT] | None = None,
) -> _GeneratedClassDeclaration:
    return _observation_root_spec(
        f"{stem}ObservationSpec",
        item_type or _derived_type(stem, "Observation"),
        "AutoRegisteredModuleShapeSpec",
        *base_names,
    )


def _std_obs(
    stem: str,
    helper: GeneratedShapeHelper[..., GeneratedItemT],
    *base_names: str,
    **attributes: GeneratedClassAttributeValue,
) -> _GeneratedClassDeclaration:
    return _GENERATED_CLASS_DECLARATIONS.class_declaration(
        f"Standard{stem}ObservationSpec",
        f"{stem}ObservationSpec",
        *base_names,
        shape_helper=helper,
        **attributes,
    )


def _multi_obs_root(
    class_name: str,
    *base_names: str,
    family_specs: GeneratedFamilySpecSet,
) -> _GeneratedClassDeclaration:
    return _GENERATED_CLASS_DECLARATIONS.class_declaration(
        class_name,
        "FamilyGeneratingSpec",
        *base_names,
        _registry_root=True,
        family_specs=family_specs,
    )


def _shape_root(
    stem: str,
    *base_names: str,
    item_type: type[GeneratedItemT] | None = None,
    export_name: str | None = None,
) -> _GeneratedClassDeclaration:
    return _GENERATED_CLASS_DECLARATIONS.root_spec_declaration(
        f"{stem}ShapeSpec",
        item_type or _derived_type(stem, "Shape"),
        ShapeFamily,
        *base_names,
        export_name=export_name,
    )


def _ctx_shape(
    stem: str,
    node_type: type[ast.AST],
    *,
    item_type: type[GeneratedItemT] | None = None,
) -> _GeneratedClassDeclaration:
    return _GENERATED_CLASS_DECLARATIONS.class_declaration(
        f"{stem}ShapeSpec",
        "FamilyGeneratingSpec",
        "ContextHelperShapeSpec",
        family_specs=_family_specs(
            item_type or _derived_type(stem, "Shape"), ShapeFamily
        ),
        node_type=node_type,
    )


def _helper_decl(
    class_name: str,
    helper: GeneratedShapeHelper[..., GeneratedItemT],
    *base_names: str,
    **attributes: GeneratedClassAttributeValue,
) -> _GeneratedClassDeclaration:
    return _GENERATED_CLASS_DECLARATIONS.class_declaration(
        class_name,
        *base_names,
        shape_helper=helper,
        **attributes,
    )


_FUNCTION_HELPER = "HelperBackedFunctionObservationSpec"
_TUPLE_FUNCTION_HELPER = "TupleResultMixin HelperBackedFunctionObservationSpec"
_MODULE_SCOPED_FUNCTION_HELPER = (
    "ModuleOnlyFunctionObservationSpec HelperBackedScopedFunctionObservationSpec"
)
_MODULE_SYNC_SCOPED_FUNCTION_HELPER = (
    "ModuleOnlyFunctionObservationSpec SyncFunctionOnlyMixin "
    "HelperBackedScopedFunctionObservationSpec"
)
_MODULE_SCOPED_ASSIGN_HELPER = (
    "ModuleOnlyAssignObservationSpec HelperBackedScopedAssignObservationSpec"
)


class TypedLiteralObservationFamily(ObservationFamily[LiteralDispatchObservation], ABC):
    """Observation family root specialized by a literal-kind discriminator."""

    _registry_skip = True
    item_type = LiteralDispatchObservation
    spec_root: ClassVar[type[AutoRegisteredModuleShapeSpec[LiteralDispatchObservation]]]
    literal_kind: ClassVar[LiteralKind]

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[LiteralDispatchObservation]:
        if issubclass(cls.spec_root, TypedLiteralObservationSpec):
            return [
                item
                for item in cls.spec_root().collect(parsed_module)
                if isinstance(item, LiteralDispatchObservation)
                if item.literal_kind == cls.literal_kind
            ]
        return [
            item
            for item in COLLECTED_ITEM_PROJECTION.from_spec_root(
                cls.spec_root, parsed_module, LiteralDispatchObservation
            )
            if isinstance(item, LiteralDispatchObservation)
            if item.literal_kind == cls.literal_kind
        ]


def _literal_spec(
    stem: str,
    base_name: str,
    literal_type: type[str] | type[int],
    literal_kind: LiteralKind,
) -> _GeneratedClassDeclaration:
    return _GENERATED_CLASS_DECLARATIONS.class_declaration(
        f"{stem}LiteralDispatchObservationSpec",
        "FamilyGeneratingSpec",
        base_name,
        family_specs=_family_specs(
            LiteralDispatchObservation, TypedLiteralObservationFamily
        ),
        literal_type=literal_type,
        literal_kind=literal_kind,
    )


def _probe_spec(
    stem: str,
    call_family: AstNameFamily,
    probe_kind: str,
    minimum_args: int,
    *,
    attribute_arg_index: int | None = 1,
) -> _GeneratedClassDeclaration:
    return _GENERATED_CLASS_DECLARATIONS.class_declaration(
        f"{stem}ProbeObservationSpec",
        "CallAttributeProbeObservationSpec",
        call_family=call_family,
        probe_kind=probe_kind,
        minimum_args=minimum_args,
        attribute_arg_index=attribute_arg_index,
    )


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
            and (not function.name.startswith("__")),
            param_count=len(function.args.args),
            decorators=tuple(
                (_node_display_name(dec) for dec in function.decorator_list)
            ),
            function_node=function,
        )


_materialize_class_declarations(
    (
        _ctx_shape("BuilderCall", ast.Call),
        _ctx_shape("ExportDict", ast.Dict),
    )
)


BuilderCallShapeSpec.shape_helper = _builder_call_shape
ExportDictShapeSpec.shape_helper = _export_dict_shape

_METHOD_SHAPE_SPEC = MethodShapeSpec()
_BUILDER_CALL_SHAPE_SPEC = BuilderCallShapeSpec()
_EXPORT_DICT_SHAPE_SPEC = ExportDictShapeSpec()


class ScopeFilteredFunctionObservationSpec(
    FunctionObservationSpec[GeneratedItemT], Generic[GeneratedItemT], ABC
):
    @abstractmethod
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_scoped_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[GeneratedItemT] | None:
        raise NotImplementedError

    def build_from_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[GeneratedItemT] | None:
        if not self.accepts_scope(observation):
            return None
        return self.build_scoped_function(parsed_module, function, observation)


class ModuleOnlyFunctionObservationSpec(
    ScopeFilteredFunctionObservationSpec[GeneratedItemT],
    Generic[GeneratedItemT],
    ABC,
):
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        return observation.class_name is None


class ClassOnlyFunctionObservationSpec(
    ScopeFilteredFunctionObservationSpec[GeneratedItemT],
    Generic[GeneratedItemT],
    ABC,
):
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        return observation.class_name is not None


class ScopeFilteredAssignObservationSpec(
    AssignObservationSpec[GeneratedItemT], Generic[GeneratedItemT], ABC
):
    @abstractmethod
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_scoped_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[GeneratedItemT] | None:
        raise NotImplementedError

    def build_from_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[GeneratedItemT] | None:
        if not self.accepts_scope(observation):
            return None
        return self.build_scoped_assign(parsed_module, node, observation)


class ModuleOnlyAssignObservationSpec(
    ScopeFilteredAssignObservationSpec[GeneratedItemT],
    Generic[GeneratedItemT],
    ABC,
):
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        return observation.class_name is None and observation.function_name is None


class TupleResultMixin(Generic[GeneratedItemT], ABC):
    @staticmethod
    def wrap_helper_result(
        value: tuple[GeneratedItemT, ...] | None,
    ) -> tuple[GeneratedItemT, ...] | None:
        return value


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
            (
                arg.arg == type(self).required_parameter_name
                for arg in function.args.args
            )
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


class ShapeHelperBackedSpec(Generic[GeneratedItemT], ABC):
    shape_helper: ClassVar[GeneratedShapeHelper[..., GeneratedItemT]]

    @staticmethod
    def wrap_helper_result(
        value: ShapeEmission[GeneratedItemT] | None,
    ) -> ShapeEmission[GeneratedItemT] | None:
        return value


class HelperBackedFunctionObservationSpec(
    FunctionAcceptanceMixin,
    ShapeHelperBackedSpec[GeneratedItemT],
    FunctionObservationSpec[GeneratedItemT],
    Generic[GeneratedItemT],
    ABC,
):
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
    ) -> ShapeEmission[GeneratedItemT] | None:
        if not self.accepts_function(function, observation):
            return None
        return type(self).wrap_helper_result(
            type(self).shape_helper(parsed_module, function)
        )


class HelperBackedScopedFunctionObservationSpec(
    FunctionAcceptanceMixin,
    ShapeHelperBackedSpec[GeneratedItemT],
    ScopeFilteredFunctionObservationSpec[GeneratedItemT],
    Generic[GeneratedItemT],
    ABC,
):
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
    ) -> ShapeEmission[GeneratedItemT] | None:
        if not self.accepts_function(function, observation):
            return None
        return type(self).wrap_helper_result(
            type(self).shape_helper(parsed_module, function)
        )


class ClassNamedFunctionHelperObservationSpec(
    ShapeHelperBackedSpec[GeneratedItemT],
    ClassOnlyFunctionObservationSpec[GeneratedItemT],
    Generic[GeneratedItemT],
    ABC,
):
    def build_scoped_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[GeneratedItemT] | None:
        class_name = observation.class_name
        if class_name is None:
            return None
        return type(self).wrap_helper_result(
            type(self).shape_helper(parsed_module, class_name, function)
        )


class HelperBackedAssignObservationSpec(
    ShapeHelperBackedSpec[GeneratedItemT],
    AssignObservationSpec[GeneratedItemT],
    Generic[GeneratedItemT],
    ABC,
):
    def build_from_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[GeneratedItemT] | None:
        del observation
        return type(self).shape_helper(parsed_module, node)


class HelperBackedScopedAssignObservationSpec(
    ShapeHelperBackedSpec[GeneratedItemT],
    ScopeFilteredAssignObservationSpec[GeneratedItemT],
    Generic[GeneratedItemT],
    ABC,
):
    def build_scoped_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[GeneratedItemT] | None:
        del observation
        return type(self).shape_helper(parsed_module, node)


class ObservationContextHelperShapeSpec(
    ShapeHelperBackedSpec[GeneratedItemT],
    ContextHelperShapeSpec[GeneratedItemT],
    Generic[GeneratedItemT],
    ABC,
):
    def shape_helper_args(
        self, node: ast.AST, observation: ScopedAstObservation
    ) -> tuple[ast.AST, ScopedAstObservation]:
        return (node, observation)


_materialize_class_declarations(
    (
        _obs_root("ConfigDispatch", "FunctionObservationSpec"),
        _std_obs(
            "ConfigDispatch",
            GeneratedShapeHelper(_config_dispatch_observations),
            "ModuleOnlyFunctionObservationSpec RequiredFunctionParameterMixin TupleResultMixin HelperBackedScopedFunctionObservationSpec",
            required_parameter_name="config",
        ),
        _obs_root("ClassMarker", "FunctionObservationSpec"),
        _std_obs(
            "ClassMarker",
            GeneratedShapeHelper(_class_marker_observations),
            _TUPLE_FUNCTION_HELPER,
        ),
        _obs_root("InterfaceGeneration", "FunctionObservationSpec"),
        _std_obs(
            "InterfaceGeneration",
            GeneratedShapeHelper(_interface_generation_observation),
            _FUNCTION_HELPER,
        ),
        _obs_root("SentinelType", "ABC"),
        _helper_decl(
            "SentinelTypeAssignmentObservationSpec",
            GeneratedShapeHelper(_sentinel_type_observation),
            f"SentinelTypeObservationSpec {_MODULE_SCOPED_ASSIGN_HELPER}",
        ),
        _obs_root(
            "DynamicMethodInjection",
            "FunctionObservationSpec",
        ),
        _std_obs(
            "DynamicMethodInjection",
            GeneratedShapeHelper(_dynamic_method_injection_observations),
            _TUPLE_FUNCTION_HELPER,
        ),
        _obs_root(
            "RuntimeTypeGeneration",
            "ObservationContextHelperShapeSpec ABC",
        ),
        _helper_decl(
            "TypeCallGenerationObservationSpec",
            GeneratedShapeHelper(_runtime_type_generation_observation),
            "RuntimeTypeGenerationObservationSpec",
            node_type=ast.Call,
        ),
        _obs_root("LineageMapping", "AssignObservationSpec ABC"),
        _std_obs(
            "LineageMapping",
            GeneratedShapeHelper(_lineage_mapping_observation),
            "HelperBackedAssignObservationSpec",
        ),
        _obs_root(
            "DualAxisResolution",
            "FunctionObservationSpec ABC",
        ),
        _std_obs(
            "DualAxisResolution",
            GeneratedShapeHelper(_dual_axis_resolution_observation),
            _FUNCTION_HELPER,
        ),
        _obs_root("AttributeProbe", "ABC"),
    )
)


class SentinelTypeUsageObservationSpec(SentinelTypeObservationSpec):
    def collect(self, parsed_module: ParsedModule) -> list[SentinelTypeObservation]:
        return list(_sentinel_type_usage_observations(parsed_module))


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


_materialize_class_declarations(
    (
        _probe_spec(
            "HasAttr",
            _HASATTR_CALL_FAMILY,
            "hasattr",
            2,
        ),
        _probe_spec(
            "GetAttr",
            _GETATTR_CALL_FAMILY,
            "getattr",
            3,
        ),
    )
)


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


class TypedLiteralObservationSpec(
    AutoRegisteredModuleShapeSpec[LiteralDispatchObservation], ABC
):
    literal_type: ClassVar[type[str] | type[int]]

    @classmethod
    def registered_specs_for_literal_type(
        cls, literal_type: type[str] | type[int] | None = None
    ) -> tuple[TypedLiteralObservationSpec, ...]:
        specs = tuple(
            (
                spec
                for spec in cls.registered_specs()
                if isinstance(spec, TypedLiteralObservationSpec)
            )
        )
        if literal_type is None:
            return specs
        return tuple(spec for spec in specs if type(spec).literal_type is literal_type)


class LiteralDispatchObservationSpec(TypedLiteralObservationSpec, ABC):
    _registry_root = True
    _registry_skip = True
    literal_kind: ClassVar[LiteralKind]

    def collect(self, parsed_module: ParsedModule) -> list[LiteralDispatchObservation]:
        return list(
            _literal_dispatch_observations_for_kind(
                parsed_module, type(self).literal_kind
            )
        )


class InlineLiteralDispatchObservationSpec(TypedLiteralObservationSpec, ABC):
    _registry_root = True
    _registry_skip = True
    literal_kind: ClassVar[LiteralKind]

    def collect(self, parsed_module: ParsedModule) -> list[LiteralDispatchObservation]:
        return list(
            _inline_literal_dispatch_observations_for_kind(
                parsed_module, type(self).literal_kind
            )
        )


_materialize_class_declarations(
    (
        _literal_spec(
            "String",
            "LiteralDispatchObservationSpec",
            str,
            LiteralKind.STRING,
        ),
        _literal_spec(
            "Numeric",
            "LiteralDispatchObservationSpec",
            int,
            LiteralKind.NUMERIC,
        ),
        _literal_spec(
            "InlineString",
            "InlineLiteralDispatchObservationSpec",
            str,
            LiteralKind.STRING,
        ),
        _shape_root(
            "Registration",
            "AutoRegisteredModuleShapeSpec",
            "ABC",
        ),
    )
)


class KnownClassFamilyShapeSpec(RegistrationShapeSpec, ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    def collect(self, parsed_module: ParsedModule) -> list[RegistrationShape]:
        return self.collect_with_known_class_family(
            parsed_module, _known_class_family(parsed_module)
        )

    @abstractmethod
    def collect_with_known_class_family(
        self, parsed_module: ParsedModule, known_class_family: AstNameFamily
    ) -> list[RegistrationShape]:
        raise NotImplementedError


class AssignmentRegistrationShapeSpec(KnownClassFamilyShapeSpec):
    def collect_with_known_class_family(
        self,
        parsed_module: ParsedModule,
        known_class_family: AstNameFamily,
    ) -> list[RegistrationShape]:
        shapes: list[RegistrationShape] = []
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
                        parsed_module, node, registry_name, key_fingerprint
                    )
                )
        return shapes


class CallRegistrationShapeSpec(KnownClassFamilyShapeSpec):
    def collect_with_known_class_family(
        self,
        parsed_module: ParsedModule,
        known_class_family: AstNameFamily,
    ) -> list[RegistrationShape]:
        shapes: list[RegistrationShape] = []
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
                    parsed_module, node, registry_name, class_name, key_fingerprint
                )
            )
        return shapes


class DecoratorRegistrationShapeSpec(RegistrationShapeSpec):
    def collect(self, parsed_module: ParsedModule) -> list[RegistrationShape]:
        shapes: list[RegistrationShape] = []
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


_materialize_class_declarations((_obs_root("Field", "ABC"),))


class ClassObservationSpec(FieldObservationSpec, ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    def collect(self, parsed_module: ParsedModule) -> list[FieldObservation]:
        observations: list[FieldObservation] = []
        for class_observation in CLASS_OBSERVATION_PROJECTION.project(parsed_module):
            observations.extend(
                self.collect_for_class(parsed_module, class_observation)
            )
        return observations

    @abstractmethod
    def collect_for_class(
        self, parsed_module: ParsedModule, class_observation: ClassAstObservation
    ) -> list[FieldObservation]:
        raise NotImplementedError


class DataclassBodyFieldObservationSpec(ClassObservationSpec):
    def collect_for_class(
        self,
        parsed_module: ParsedModule,
        class_observation: ClassAstObservation,
    ) -> list[FieldObservation]:
        observations: list[FieldObservation] = []
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
    ) -> list[FieldObservation]:
        observations: list[FieldObservation] = []
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


_materialize_class_declarations(
    (
        _obs_root(
            "ProjectionHelper",
            "FunctionObservationSpec ABC",
            item_type=ProjectionHelperShape,
        ),
        _obs_root(
            "AccessorWrapper",
            "FunctionObservationSpec ABC",
            item_type=AccessorWrapperCandidate,
        ),
        _std_obs(
            "ProjectionHelper",
            GeneratedShapeHelper(_projection_helper_shape_from_function),
            _MODULE_SCOPED_FUNCTION_HELPER,
        ),
        _std_obs(
            "AccessorWrapper",
            GeneratedShapeHelper(_accessor_wrapper_candidate_from_function),
            "ClassNamedFunctionHelperObservationSpec",
        ),
        _multi_obs_root(
            "ScopedShapeWrapperObservationSpec",
            "AutoRegisteredModuleShapeSpec ABC",
            family_specs=(
                *_family_specs(
                    ScopedShapeWrapperFunction,
                    ObservationFamily,
                    "ScopedShapeWrapperFunctionFamily",
                ),
                *_family_specs(
                    ScopedShapeWrapperSpec,
                    ObservationFamily,
                    "ScopedShapeWrapperSpecFamily",
                ),
            ),
        ),
        _helper_decl(
            "ScopedShapeWrapperFunctionObservationSpec",
            GeneratedShapeHelper(_scoped_shape_wrapper_function_from_function),
            f"ScopedShapeWrapperObservationSpec {_MODULE_SYNC_SCOPED_FUNCTION_HELPER}",
        ),
        _helper_decl(
            "ScopedShapeWrapperSpecObservationSpec",
            GeneratedShapeHelper(_scoped_shape_wrapper_spec_from_assign),
            f"ScopedShapeWrapperObservationSpec {_MODULE_SCOPED_ASSIGN_HELPER}",
        ),
    )
)


def _registered_family_types() -> tuple[type[CollectedFamily], ...]:
    return CollectedFamily.all_registered_families()


@lru_cache(maxsize=1)
def _family_types_by_item_type() -> dict[type, type[CollectedFamily]]:
    return {family.item_type: family for family in _registered_family_types()}


@lru_cache(maxsize=1)
def _literal_family_types_by_kind() -> dict[LiteralKind, type[CollectedFamily]]:
    return {
        family.literal_kind: family
        for family in _registered_family_types()
        if issubclass(family, TypedLiteralObservationFamily)
    }


def family_for_item_type(
    item_type: type[GeneratedItemT],
) -> type[CollectedFamily[GeneratedItemT]]:
    """Return the generated family that owns one collected item type."""
    return cast(
        type[CollectedFamily[GeneratedItemT]], _family_types_by_item_type()[item_type]
    )


def family_for_literal_kind(literal_kind: LiteralKind) -> type[CollectedFamily]:
    """Return the generated family that owns one literal-dispatch kind."""
    return _literal_family_types_by_kind()[literal_kind]


GeneratedFamilyNamespaceValue: TypeAlias = (
    str
    | LiteralKind
    | FamilyGeneratingSpec
    | type[GeneratedItemT]
    | type[TypedLiteralObservationSpec]
    | type[AutoRegisteredModuleShapeSpec]
    | type[CollectedFamily]
)


def _materialize_generated_family(
    spec_type: type[FamilyGeneratingSpec],
    family_spec: GeneratedFamilySpec,
) -> type[CollectedFamily]:
    module_globals = cast(dict[str, GeneratedFamilyNamespaceValue], globals())
    family_root = family_spec.family_root
    export_name = (
        family_spec.export_name or spec_type.__name__.removesuffix("Spec") + "Family"
    )
    attributes: dict[str, GeneratedFamilyNamespaceValue] = {
        "__module__": __name__,
        "item_type": family_spec.item_type,
    }
    if family_root is TypedLiteralObservationFamily:
        literal_spec_type = cast(type[TypedLiteralObservationSpec], spec_type)
        attributes["spec_root"] = literal_spec_type
        attributes["literal_kind"] = literal_spec_type.literal_kind
        family_bases = (TypedLiteralObservationFamily,)
    elif issubclass(spec_type, AutoRegisteredModuleShapeSpec):
        attributes["spec_root"] = cast(type[AutoRegisteredModuleShapeSpec], spec_type)
        family_bases = (RegisteredSpecCollectedFamily, family_root)
    else:
        attributes["spec"] = spec_type()
        family_bases = (SingleSpecCollectedFamily, family_root)
    family_type = cast(
        type[CollectedFamily], AutoRegisterMeta(export_name, family_bases, attributes)
    )
    module_globals[export_name] = family_type
    return family_type


def _materialize_declared_families() -> dict[str, type[CollectedFamily]]:
    families: dict[str, type[CollectedFamily]] = {}
    for spec_type in _declared_family_spec_types():
        for family_spec in spec_type.family_specs:
            family_type = _materialize_generated_family(spec_type, family_spec)
            families[family_type.__name__] = family_type
    return families


_FAMILY_EXPORTS = _materialize_declared_families()
_FAMILY_EXPORT_NAMES = tuple(_FAMILY_EXPORTS)


_PUBLIC_EXPORT_POLICY = PublicExportPolicy(
    module_name=__name__,
    root_types=tuple(SharedRegistryRootBase.__subclasses__()),
    explicit_names=frozenset({"AutoRegisteredModuleShapeSpec"}),
)


__all__ = derive_public_exports(globals(), _PUBLIC_EXPORT_POLICY)
