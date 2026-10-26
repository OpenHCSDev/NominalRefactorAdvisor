"""Source-visible observation specs and their collected families."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, ClassVar, Generic, TypeVar

from .ast_projection import ExpressionReference
from .class_index import (
    ModuleNominalBindingAuthority,
    ModuleNominalBindingSnapshot,
    module_import_names,
)
from .foreign_family import FunctionDefinition
from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from .export_tools import PublicExportPolicy, derive_public_exports
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from .native_syntax import NativePythonSyntaxIndex

from .observation_shapes import (
    BuilderCallShape,
    DynamicMethodInjectionObservation,
    FieldObservation,
    ForeignTypeTestModule,
    ForeignTypeTestShape,
    LiteralDispatchObservation,
    LiteralKind,
    ModuleExpressionShape,
    RegistrationShape,
    SentinelTypeObservation,
)

from .ast_tools import (
    BuiltinCallName,
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
                registry_name = ExpressionReference.decoded(target).terminal_name
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
            registry_name = ExpressionReference.decoded(node.func.value).terminal_name
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
                    and ExpressionReference.decoded(decorator.func).terminal_name
                    in REGISTRATION_DECORATOR_FAMILY.names
                    and decorator.args
                ):
                    continue
                registry_name = ExpressionReference.decoded(
                    decorator.args[0]
                ).terminal_name
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
                registry_name = ExpressionReference.decoded(target).terminal_name
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
            registry_name = ExpressionReference.decoded(node.func.value).terminal_name
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
            registry_name = ExpressionReference.decoded(decorator.args[0]).terminal_name
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
                and ExpressionReference.decoded(decorator.func).terminal_name
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


def _reads_only_modules(node: ast.AST, module_names: frozenset[str]) -> bool:
    """Whether an expression is computed from imported modules and constants only."""

    match node:
        case ast.Constant():
            return True
        case ast.Name(id=name, ctx=ast.Load()):
            return name in module_names
        case ast.Attribute(value=value, ctx=ast.Load()):
            return _reads_only_modules(value, module_names)
        case ast.Subscript(value=value, slice=index, ctx=ast.Load()):
            return _reads_only_modules(value, module_names) and _is_constant_index(
                index
            )
        case ast.Tuple(elts=elements, ctx=ast.Load()):
            return all(_reads_only_modules(item, module_names) for item in elements)
        case ast.Compare(left=left, comparators=comparators):
            return all(
                _reads_only_modules(item, module_names)
                for item in (left, *comparators)
            )
    return False


def _is_constant_index(index: ast.AST) -> bool:
    match index:
        case ast.Constant():
            return True
        case ast.Slice(lower=lower, upper=upper, step=step):
            return all(
                bound is None or isinstance(bound, ast.Constant)
                for bound in (lower, upper, step)
            )
        case ast.Tuple(elts=elements):
            return all(isinstance(item, ast.Constant) for item in elements)
    return False


def _module_expression_shapes(
    parsed_module: ParsedModule,
) -> list[ModuleExpressionShape]:
    """Collect maximal pure module-attribute expressions; a bare `mod.name` is not one.

    A lone attribute needs two reads (`sys.version_info.major`); a tuple,
    subscript or comparison needs one. A callee and a subscript's index or
    type argument (`tuple[ast.expr, ...]`) are not values.
    """

    module_names = module_import_names(parsed_module)
    if not module_names:
        return []
    index = module_syntax_index(parsed_module.module)
    shapes: list[ModuleExpressionShape] = []
    for node_index, node in sorted(
        (
            *index.indexed_nodes_of_type(ast.Tuple),
            *index.indexed_nodes_of_type(ast.Attribute),
            *index.indexed_nodes_of_type(ast.Subscript),
            *index.indexed_nodes_of_type(ast.Compare),
        )
    ):
        attribute_reads = sum(
            isinstance(item, ast.Attribute) for item in ast.walk(node)
        )
        if attribute_reads < (2 if isinstance(node, ast.Attribute) else 1):
            continue
        if not _reads_only_modules(node, module_names):
            continue
        parent = index.parent_node(node_index)
        match parent:
            case ast.Call(func=callee) if callee is node:
                continue
            case ast.Subscript(slice=argument) if argument is node:
                continue
        if parent is not None and _reads_only_modules(parent, module_names):
            continue
        shapes.append(
            ModuleExpressionShape(
                file_path=parsed_module.file_path,
                lineno=node.lineno,
                symbol=index.lexical_symbol(node_index),
                expression_key=ast.dump(node, include_attributes=False),
                expression_text=ast.unparse(node),
            )
        )
    return shapes


class ModuleExpressionShapeFamily(ShapeFamily[ModuleExpressionShape]):
    item_type = ModuleExpressionShape

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ModuleExpressionShape]:
        return _module_expression_shapes(parsed_module)


@dataclass(frozen=True)
class TypeArgumentResolver:
    """Resolve an `isinstance` type argument to qualified class names, fail closed.

    Names resolve through the module's final import and builtin bindings; a
    tuple, a `|` union, or a once-assigned module-level alias of either expands
    to its members. Anything unresolved or shadowed in an enclosing function
    makes the whole site unresolved.
    """

    parsed_module: ParsedModule

    @cached_property
    def snapshot(self) -> ModuleNominalBindingSnapshot:
        return ModuleNominalBindingAuthority(self.parsed_module).snapshot_before(None)

    @cached_property
    def aliases(self) -> dict[str, ast.expr]:
        assignments = [
            (statement.targets[0].id, statement.value)
            for statement in self.parsed_module.module.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ]
        counts = Counter(name for name, _value in assignments)
        return {
            name: value
            for name, value in assignments
            if counts[name] == 1 and self._is_member_collection(value)
        }

    @staticmethod
    def _is_member_collection(node: ast.AST) -> bool:
        match node:
            case ast.Tuple() | ast.BinOp(op=ast.BitOr()):
                return True
        return False

    def members(
        self,
        node: ast.expr,
        shadowed: frozenset[str],
        expanding: frozenset[str] = frozenset(),
    ) -> tuple[ast.expr, ...]:
        match node:
            case ast.Tuple(elts=elements):
                return tuple(
                    member
                    for element in elements
                    for member in self.members(element, shadowed, expanding)
                )
            case ast.BinOp(left=left, op=ast.BitOr(), right=right):
                return (
                    *self.members(left, shadowed, expanding),
                    *self.members(right, shadowed, expanding),
                )
            case ast.Name(id=name) if (
                name in self.aliases and name not in shadowed | expanding
            ):
                return self.members(self.aliases[name], shadowed, expanding | {name})
        return (node,)

    def qualified_name(self, member: ast.expr, shadowed: frozenset[str]) -> str | None:
        if ExpressionReference.decoded(member).qualified_name is None:
            return None
        parts = ExpressionReference.decoded(member).attribute_chain
        if parts is None or parts[0] in shadowed:
            return None
        reference = self.snapshot.reference_for(parts)
        return None if reference.root_binding is None else reference.qualified_name

    def qualified_class_at(
        self, member: ast.expr, ancestors: Sequence[ast.AST]
    ) -> str | None:
        """Qualified class a name expression denotes beneath these ancestors."""

        return self.qualified_name(member, _enclosing_function_names(ancestors))

    def qualified_names(
        self, type_argument: ast.expr, shadowed: frozenset[str]
    ) -> tuple[str, ...] | None:
        members = self.members(type_argument, shadowed)
        names = tuple(self.qualified_name(member, shadowed) for member in members)
        if not names or None in names:
            return None
        return tuple(sorted(set(names)))

    def qualified_names_of_test(self, call: ast.Call) -> tuple[str, ...] | None:
        """Qualified classes one `isinstance` call of this module tests, if resolved."""

        arguments = BuiltinCallName.isinstance_arguments(call)
        if arguments is None:
            return None
        parents = module_syntax_index(self.parsed_module.module).parent_by_node
        ancestors = tuple(self._ancestors(parents, call))
        return self.qualified_names(arguments[1], _enclosing_function_names(ancestors))

    @staticmethod
    def _ancestors(
        parents: Mapping[ast.AST, ast.AST], node: ast.AST
    ) -> Iterator[ast.AST]:
        parent = parents.get(node)
        if parent is not None:
            yield parent
            yield from TypeArgumentResolver._ancestors(parents, parent)


def _enclosing_function_names(ancestors: Sequence[ast.AST]) -> frozenset[str]:
    """Names rebound by any enclosing function, which hide module bindings."""

    return frozenset().union(
        *(
            LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(ancestor.body)
            | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(ancestor)
            for ancestor in FunctionDefinition.within(ancestors)
        )
    )


def _foreign_type_test_shapes(
    parsed_module: ParsedModule,
) -> list[ForeignTypeTestShape]:
    """Collect type tests whose classes all come from one imported owner module."""

    index = module_syntax_index(parsed_module.module)
    resolver = TypeArgumentResolver(parsed_module)
    shapes: list[ForeignTypeTestShape] = []
    for node_index, node in index.indexed_nodes_of_type(ast.Call):
        arguments = BuiltinCallName.type_test_arguments(node)
        if arguments is None:
            continue
        names = resolver.qualified_names(
            arguments[1], _enclosing_function_names(index.ancestor_nodes(node_index))
        )
        if names is None:
            continue
        owners = {name.rpartition(".")[0] for name in names}
        if len(owners) != 1 or "" in owners:
            continue
        shapes.append(
            ForeignTypeTestShape(
                file_path=parsed_module.file_path,
                lineno=node.lineno,
                symbol=index.lexical_symbol(node_index),
                owner=owners.pop(),
                type_names=names,
                subject=ast.dump(arguments[0]),
            )
        )
    return shapes


class ForeignTypeTestModuleFamily(ShapeFamily[ForeignTypeTestModule]):
    item_type = ForeignTypeTestModule

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ForeignTypeTestModule]:
        return [
            ForeignTypeTestModule(
                module_name=parsed_module.module_name,
                sites=tuple(_foreign_type_test_shapes(parsed_module)),
            )
        ]


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
