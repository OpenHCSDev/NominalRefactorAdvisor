"""Source-visible observation specs and their collected families."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, ClassVar, Generic, TypeVar

from .export_tools import PublicExportPolicy, derive_public_exports
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from .native_syntax import NativePythonSyntaxIndex

from .observation_shapes import (
    BuilderCallShape,
    DynamicMethodInjectionObservation,
    FieldObservation,
    LiteralDispatchObservation,
    LiteralKind,
    RegistrationShape,
    SentinelTypeObservation,
)

from .ast_tools import (
    AstExpressionProjection,
    AstNameFamily,
    AssignObservationSpec,
    AutoRegisterMeta,
    AutoRegisteredModuleShapeSpec,
    ClassAstObservation,
    CLASS_OBSERVATION_PROJECTION,
    CollectedFamily,
    ContextHelperShapeSpec,
    FunctionObservationSpec,
    ParsedModule,
    RegisteredSpecCollectedFamily,
    ScopedAstObservation,
    SharedRegistryRootBase,
    ShapeEmission,
    SingleSpecCollectedFamily,
    SourceModule,
    module_syntax_index,
    REGISTRATION_CALL_FAMILY,
    REGISTRATION_DECORATOR_FAMILY,
    _builder_call_shape,
    _class_body_field_observation,
    _class_name_from_expr,
    _dynamic_method_injection_observations,
    root_agnostic_expression_fingerprint,
    _init_field_observations,
    _inline_literal_dispatch_observations_for_kind,
    _iter_attribute_family_calls,
    _iter_class_decorator_family_calls,
    _known_class_family,
    _literal_dispatch_observations_for_kind,
    _registration_key_fingerprint,
    _sentinel_type_observation,
    _sentinel_type_usage_observations,
)

FamilyItemT = TypeVar("FamilyItemT")


class ObservationFamily(CollectedFamily[FamilyItemT], Generic[FamilyItemT], ABC):
    """Registry root for observation families derived from observation specs."""

    _registry_root = True


class ShapeFamily(CollectedFamily[FamilyItemT], Generic[FamilyItemT], ABC):
    """Registry root for structural shape families derived from shape specs."""

    _registry_root = True


class TypedLiteralObservationFamily(ObservationFamily[LiteralDispatchObservation], ABC):
    """Observation family root specialized by a literal-kind discriminator."""

    _registry_skip = True
    item_type = LiteralDispatchObservation
    spec_root: ClassVar[type[AutoRegisteredModuleShapeSpec[LiteralDispatchObservation]]]
    literal_kind: ClassVar[LiteralKind]

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[LiteralDispatchObservation]:
        return cls.spec_root().collect(parsed_module)


class BuilderCallShapeSpec(ContextHelperShapeSpec[BuilderCallShape]):
    """Collect builder-call shapes through the canonical AST helper."""

    node_type = ast.Call
    shape_helper = staticmethod(_builder_call_shape)


_BUILDER_CALL_SHAPE_SPEC = BuilderCallShapeSpec()


class ScopeFilteredFunctionObservationSpec(
    FunctionObservationSpec[FamilyItemT], Generic[FamilyItemT], ABC
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
    ) -> ShapeEmission[FamilyItemT] | None:
        raise NotImplementedError

    def build_from_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[FamilyItemT] | None:
        if not self.accepts_scope(observation):
            return None
        return self.build_scoped_function(parsed_module, function, observation)


class ModuleOnlyFunctionObservationSpec(
    ScopeFilteredFunctionObservationSpec[FamilyItemT],
    Generic[FamilyItemT],
    ABC,
):
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        return observation.class_name is None


class ClassOnlyFunctionObservationSpec(
    ScopeFilteredFunctionObservationSpec[FamilyItemT],
    Generic[FamilyItemT],
    ABC,
):
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        return observation.class_name is not None


class ScopeFilteredAssignObservationSpec(
    AssignObservationSpec[FamilyItemT], Generic[FamilyItemT], ABC
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
    ) -> ShapeEmission[FamilyItemT] | None:
        raise NotImplementedError

    def build_from_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[FamilyItemT] | None:
        if not self.accepts_scope(observation):
            return None
        return self.build_scoped_assign(parsed_module, node, observation)


class ModuleOnlyAssignObservationSpec(
    ScopeFilteredAssignObservationSpec[FamilyItemT],
    Generic[FamilyItemT],
    ABC,
):
    def accepts_scope(self, observation: ScopedAstObservation) -> bool:
        return observation.class_name is None and observation.function_name is None


class TupleResultMixin(Generic[FamilyItemT], ABC):
    @staticmethod
    def wrap_helper_result(
        value: tuple[FamilyItemT, ...] | None,
    ) -> tuple[FamilyItemT, ...] | None:
        return value


class FunctionAcceptanceMixin(ABC):
    def accepts_function(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> bool:
        del function, observation
        return True


class SyncFunctionOnlyMixin(FunctionAcceptanceMixin):
    def accepts_function(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> bool:
        return super().accepts_function(function, observation) and isinstance(
            function, ast.FunctionDef
        )


class ShapeHelperBackedSpec(Generic[FamilyItemT], ABC):
    shape_helper: ClassVar[Callable[..., ShapeEmission[FamilyItemT] | None]]

    @staticmethod
    def wrap_helper_result(
        value: ShapeEmission[FamilyItemT] | None,
    ) -> ShapeEmission[FamilyItemT] | None:
        return value


class HelperBackedFunctionObservationSpec(
    FunctionAcceptanceMixin,
    ShapeHelperBackedSpec[FamilyItemT],
    FunctionObservationSpec[FamilyItemT],
    Generic[FamilyItemT],
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
    ) -> ShapeEmission[FamilyItemT] | None:
        if not self.accepts_function(function, observation):
            return None
        return type(self).wrap_helper_result(
            type(self).shape_helper(parsed_module, function)
        )


class HelperBackedScopedFunctionObservationSpec(
    FunctionAcceptanceMixin,
    ShapeHelperBackedSpec[FamilyItemT],
    ScopeFilteredFunctionObservationSpec[FamilyItemT],
    Generic[FamilyItemT],
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
    ) -> ShapeEmission[FamilyItemT] | None:
        if not self.accepts_function(function, observation):
            return None
        return type(self).wrap_helper_result(
            type(self).shape_helper(parsed_module, function)
        )


class ClassNamedFunctionHelperObservationSpec(
    ShapeHelperBackedSpec[FamilyItemT],
    ClassOnlyFunctionObservationSpec[FamilyItemT],
    Generic[FamilyItemT],
    ABC,
):
    def build_scoped_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[FamilyItemT] | None:
        class_name = observation.class_name
        if class_name is None:
            return None
        return type(self).wrap_helper_result(
            type(self).shape_helper(parsed_module, class_name, function)
        )


class HelperBackedScopedAssignObservationSpec(
    ShapeHelperBackedSpec[FamilyItemT],
    ScopeFilteredAssignObservationSpec[FamilyItemT],
    Generic[FamilyItemT],
    ABC,
):
    def build_scoped_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[FamilyItemT] | None:
        del observation
        return type(self).shape_helper(parsed_module, node)


class SentinelTypeObservationSpec(
    AutoRegisteredModuleShapeSpec[SentinelTypeObservation], ABC
):
    _registry_root = True


class SentinelTypeAssignmentObservationSpec(
    SentinelTypeObservationSpec,
    ModuleOnlyAssignObservationSpec[SentinelTypeObservation],
    HelperBackedScopedAssignObservationSpec[SentinelTypeObservation],
):
    shape_helper = staticmethod(_sentinel_type_observation)


class DynamicMethodInjectionObservationSpec(
    AutoRegisteredModuleShapeSpec[DynamicMethodInjectionObservation],
    FunctionObservationSpec[DynamicMethodInjectionObservation],
    ABC,
):
    _registry_root = True


class StandardDynamicMethodInjectionObservationSpec(
    DynamicMethodInjectionObservationSpec,
    TupleResultMixin[DynamicMethodInjectionObservation],
    HelperBackedFunctionObservationSpec[DynamicMethodInjectionObservation],
):
    shape_helper = staticmethod(_dynamic_method_injection_observations)


class SentinelTypeUsageObservationSpec(SentinelTypeObservationSpec):
    def collect(self, parsed_module: ParsedModule) -> list[SentinelTypeObservation]:
        return list(_sentinel_type_usage_observations(parsed_module))


class TypedLiteralObservationSpec(
    AutoRegisteredModuleShapeSpec[LiteralDispatchObservation], ABC
):
    literal_kind: ClassVar[LiteralKind]

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
        return tuple(
            spec
            for spec in specs
            if type(spec).literal_kind.literal_type is literal_type
        )


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


_NATIVE_REGISTRATION_QUERY = """
(class_definition name: (identifier) @class_name)
(assignment
    left: (subscript) @target
    right: (identifier) @assigned_class) @assignment
((call
    function: (attribute attribute: (identifier) @method)
    arguments: (argument_list . (identifier) @called_class)) @call
    (#any-of? @method "register" "add" "register_class" "register_type"))
((decorated_definition
    (decorator (call function: (identifier) @decorator_method))
    definition: (class_definition)) @decorated
    (#any-of? @decorator_method
        "register" "add" "register_class" "register_type" "auto_register"))
((decorated_definition
    (decorator
        (call function: (attribute attribute: (identifier) @decorator_method)))
    definition: (class_definition)) @decorated
    (#any-of? @decorator_method
        "register" "add" "register_class" "register_type" "auto_register"))
"""


def _native_registration_shapes(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[RegistrationShape] | None:
    """Recover registration shapes natively, falling back on any uncertainty."""

    if not syntax_index.is_complete:
        return None
    captures = syntax_index.captures(_NATIVE_REGISTRATION_QUERY)
    try:
        class_names = {
            syntax_index.source_for(node).decode("utf-8")
            for node in captures.get("class_name", ())
        }
        known_class_family = AstNameFamily(frozenset(class_names))
        assignments: list[RegistrationShape] = []
        for syntax_node in sorted(
            captures.get("assignment", ()),
            key=lambda node: node.start_byte,
        ):
            statement = syntax_index.statement_for(syntax_node)
            if not isinstance(statement, ast.Assign) or not isinstance(
                statement.value, ast.Name
            ):
                return None
            if statement.value.id not in class_names:
                continue
            for target in statement.targets:
                registry_name = AstExpressionProjection.terminal_name(target)
                key_fingerprint = _registration_key_fingerprint(target)
                if registry_name is None or key_fingerprint is None:
                    continue
                assignments.append(
                    RegistrationShape.from_assignment(
                        source_module,  # type: ignore[arg-type]
                        statement,
                        registry_name,
                        key_fingerprint,
                    )
                )

        calls: list[RegistrationShape] = []
        for syntax_node in sorted(
            captures.get("call", ()),
            key=lambda node: (node.start_point.row, node.start_byte),
        ):
            node = syntax_index.expression_for(syntax_node)
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in REGISTRATION_CALL_FAMILY.names
                and node.args
            ):
                return None
            class_name = _class_name_from_expr(node.args[0], known_class_family)
            registry_name = AstExpressionProjection.terminal_name(node.func.value)
            if class_name is None or registry_name is None:
                continue
            key_source = node.args[1] if len(node.args) >= 2 else node.args[0]
            calls.append(
                RegistrationShape.from_registration_call(
                    source_module,  # type: ignore[arg-type]
                    node,
                    registry_name,
                    class_name,
                    root_agnostic_expression_fingerprint(key_source),
                )
            )

        decorators: list[RegistrationShape] = []
        decorated_nodes = {
            (node.start_byte, node.end_byte): node
            for node in captures.get("decorated", ())
        }
        for syntax_node in sorted(
            decorated_nodes.values(),
            key=lambda node: (node.start_point.row, node.start_byte),
        ):
            statement = syntax_index.statement_for(syntax_node)
            if not isinstance(statement, ast.ClassDef):
                return None
            for decorator in statement.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and AstExpressionProjection.terminal_name(decorator.func)
                    in REGISTRATION_DECORATOR_FAMILY.names
                    and decorator.args
                ):
                    continue
                registry_name = AstExpressionProjection.terminal_name(decorator.args[0])
                if registry_name is None:
                    continue
                key_expression = (
                    decorator.args[1]
                    if len(decorator.args) >= 2
                    else ast.Constant(value=statement.name)
                )
                decorators.append(
                    RegistrationShape.from_decorator(
                        source_module,  # type: ignore[arg-type]
                        statement,
                        decorator,
                        registry_name,
                        root_agnostic_expression_fingerprint(key_expression),
                    )
                )
        factory_shapes = _factory_decorator_registration_shapes(
            source_module,
            (
                *(
                    syntax_index.statement_for(node)
                    for node in syntax_index.top_level_assignment_statements()
                ),
                *(
                    syntax_index.function_for(node)
                    for node in syntax_index.top_level_declarations("function")
                ),
                *(
                    syntax_index.class_for(node)
                    for node in syntax_index.top_level_declarations("class")
                ),
            ),
        )
        return [*assignments, *calls, *decorators, *factory_shapes]
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


class StringLiteralDispatchObservationSpec(LiteralDispatchObservationSpec):
    literal_kind = LiteralKind.STRING


class NumericLiteralDispatchObservationSpec(LiteralDispatchObservationSpec):
    literal_kind = LiteralKind.NUMERIC


class InlineStringLiteralDispatchObservationSpec(InlineLiteralDispatchObservationSpec):
    literal_kind = LiteralKind.STRING


class RegistrationShapeSpec(AutoRegisteredModuleShapeSpec[RegistrationShape], ABC):
    _registry_root = True


class KnownClassFamilyShapeSpec(RegistrationShapeSpec, ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key

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
        for _node_index, node in module_syntax_index(
            parsed_module.module
        ).indexed_nodes_of_type(ast.Assign):
            if not isinstance(node.value, ast.Name):
                continue
            if known_class_family.matching_name(node.value) is None:
                continue
            for target in node.targets:
                registry_name = AstExpressionProjection.terminal_name(target)
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
            parsed_module, REGISTRATION_CALL_FAMILY
        ):
            node = observation.call
            assert isinstance(node.func, ast.Attribute)
            registry_name = AstExpressionProjection.terminal_name(node.func.value)
            if registry_name is None:
                continue
            if not node.args:
                continue
            class_name = _class_name_from_expr(node.args[0], known_class_family)
            if class_name is None:
                continue
            key_source = node.args[1] if len(node.args) >= 2 else node.args[0]
            key_fingerprint = root_agnostic_expression_fingerprint(key_source)
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
            parsed_module, REGISTRATION_DECORATOR_FAMILY
        ):
            if not decorator.args:
                continue
            registry_name = AstExpressionProjection.terminal_name(decorator.args[0])
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
                    decorator,
                    registry_name,
                    root_agnostic_expression_fingerprint(key_expr),
                )
            )
        return shapes


@dataclass(frozen=True)
class RegistryMutatingDecoratorFactory:
    """Source-proved decorator factory that writes a class into a registry."""

    function_name: str
    registry_name: str
    key_parameter_index: int

    @classmethod
    def from_function(
        cls,
        function: ast.FunctionDef,
        registry_names: frozenset[str],
    ) -> "RegistryMutatingDecoratorFactory | None":
        parameters = (*function.args.posonlyargs, *function.args.args)
        parameter_names = tuple(parameter.arg for parameter in parameters)
        nested_functions = tuple(
            statement
            for statement in function.body
            if isinstance(statement, ast.FunctionDef)
        )
        returned_factory_names = {
            statement.value.id
            for statement in function.body
            if isinstance(statement, ast.Return)
            and isinstance(statement.value, ast.Name)
        }
        nested_function = next(
            (
                candidate
                for candidate in nested_functions
                if candidate.name in returned_factory_names
            ),
            None,
        )
        if nested_function is None:
            return None
        registered_parameters = (
            *nested_function.args.posonlyargs,
            *nested_function.args.args,
        )
        if not registered_parameters:
            return None
        registered_parameter_name = registered_parameters[0].arg
        returns_registered_class = any(
            isinstance(statement, ast.Return)
            and isinstance(statement.value, ast.Name)
            and statement.value.id == registered_parameter_name
            for statement in nested_function.body
        )
        if not returns_registered_class:
            return None
        for statement in nested_function.body:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Subscript)
                and isinstance(statement.targets[0].value, ast.Name)
                and statement.targets[0].value.id in registry_names
                and isinstance(statement.targets[0].slice, ast.Name)
                and statement.targets[0].slice.id in parameter_names
                and isinstance(statement.value, ast.Name)
                and statement.value.id == registered_parameter_name
            ):
                continue
            key_parameter_name = statement.targets[0].slice.id
            return cls(
                function_name=function.name,
                registry_name=statement.targets[0].value.id,
                key_parameter_index=parameter_names.index(key_parameter_name),
            )
        return None

    def shape_for_class(
        self,
        parsed_module: ParsedModule | SourceModule,
        node: ast.ClassDef,
    ) -> RegistrationShape | None:
        decorator = next(
            (
                decorator
                for decorator in node.decorator_list
                if isinstance(decorator, ast.Call)
                and AstExpressionProjection.terminal_name(decorator.func)
                == self.function_name
                and len(decorator.args) > self.key_parameter_index
            ),
            None,
        )
        if decorator is None:
            return None
        key_expression = decorator.args[self.key_parameter_index]
        return RegistrationShape.from_factory_decorator(
            parsed_module,
            node,
            self.registry_name,
            ast.unparse(key_expression),
            root_agnostic_expression_fingerprint(key_expression),
        )


def _factory_decorator_registration_shapes(
    parsed_module: ParsedModule | SourceModule,
    statements: Sequence[ast.stmt],
) -> tuple[RegistrationShape, ...]:
    registry_names = frozenset(
        registry_name
        for statement in statements
        if (
            registry_name := _empty_registry_declaration_name(statement)
        )
        is not None
    )
    factories = tuple(
        factory
        for statement in statements
        if isinstance(statement, ast.FunctionDef)
        and (
            factory := RegistryMutatingDecoratorFactory.from_function(
                statement,
                registry_names,
            )
        )
        is not None
    )
    return tuple(
        shape
        for statement in statements
        if isinstance(statement, ast.ClassDef)
        for factory in factories
        if (shape := factory.shape_for_class(parsed_module, statement)) is not None
    )


def _empty_registry_declaration_name(statement: ast.stmt) -> str | None:
    match statement:
        case ast.Assign(targets=[ast.Name(id=name)], value=ast.Dict(keys=[])):
            return name
        case ast.AnnAssign(target=ast.Name(id=name), value=ast.Dict(keys=[])):
            return name
        case _:
            return None


class FactoryDecoratorRegistrationShapeSpec(RegistrationShapeSpec):
    """Collect registry writes hidden behind a source-proved decorator factory."""

    def collect(self, parsed_module: ParsedModule) -> list[RegistrationShape]:
        return list(
            _factory_decorator_registration_shapes(
                parsed_module,
                parsed_module.module.body,
            )
        )


class FieldObservationSpec(AutoRegisteredModuleShapeSpec[FieldObservation], ABC):
    _registry_root = True


class ClassObservationSpec(FieldObservationSpec, ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key

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


class BuilderCallShapeFamily(
    SingleSpecCollectedFamily[BuilderCallShape], ShapeFamily[BuilderCallShape]
):
    item_type = BuilderCallShape
    spec = _BUILDER_CALL_SHAPE_SPEC


class SentinelTypeObservationFamily(
    RegisteredSpecCollectedFamily[SentinelTypeObservation],
    ObservationFamily[SentinelTypeObservation],
):
    item_type = SentinelTypeObservation
    spec_root = SentinelTypeObservationSpec


class DynamicMethodInjectionObservationFamily(
    RegisteredSpecCollectedFamily[DynamicMethodInjectionObservation],
    ObservationFamily[DynamicMethodInjectionObservation],
):
    item_type = DynamicMethodInjectionObservation
    spec_root = DynamicMethodInjectionObservationSpec


class StringLiteralDispatchObservationFamily(TypedLiteralObservationFamily):
    spec_root = StringLiteralDispatchObservationSpec
    literal_kind = spec_root.literal_kind


class NumericLiteralDispatchObservationFamily(TypedLiteralObservationFamily):
    spec_root = NumericLiteralDispatchObservationSpec
    literal_kind = spec_root.literal_kind


class InlineStringLiteralDispatchObservationFamily(TypedLiteralObservationFamily):
    spec_root = InlineStringLiteralDispatchObservationSpec
    literal_kind = spec_root.literal_kind


class RegistrationShapeFamily(
    RegisteredSpecCollectedFamily[RegistrationShape], ShapeFamily[RegistrationShape]
):
    item_type = RegistrationShape
    spec_root = RegistrationShapeSpec
    source_collector = staticmethod(_native_registration_shapes)

    @staticmethod
    def report_presence_predicate(items: tuple[object, ...], config: object) -> bool:
        """Only registration shapes themselves establish report presence."""

        del config
        return bool(items)


class FieldObservationFamily(
    RegisteredSpecCollectedFamily[FieldObservation],
    ObservationFamily[FieldObservation],
):
    item_type = FieldObservation
    spec_root = FieldObservationSpec


_PUBLIC_EXPORT_POLICY = PublicExportPolicy(
    module_name=__name__,
    root_types=tuple(SharedRegistryRootBase.__subclasses__()),
    explicit_names=frozenset({"AutoRegisteredModuleShapeSpec"}),
)


__all__ = derive_public_exports(globals(), _PUBLIC_EXPORT_POLICY)
