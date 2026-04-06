from __future__ import annotations

import ast
import copy
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, TypeAlias

from .observation_graph import (
    NominalWitnessGroup,
    ObservationCohort,
    ObservationFiber,
    ObservationGraph,
    ObservationKind,
    StructuralExecutionLevel,
    StructuralObservation,
    StructuralObservationCarrier,
    build_observation_graph,
    collect_structural_observations,
)
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
    FieldOriginKind,
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


@dataclass(frozen=True)
class ParsedModule:
    path: Path
    module: ast.Module
    source: str


@dataclass(frozen=True)
class AstNameFamily:
    names: frozenset[str]


@dataclass(frozen=True)
class AstCallObservation:
    call: ast.Call
    matched_name: str


AstScopedNode: TypeAlias = ast.AST


@dataclass(frozen=True)
class ScopedAstObservation:
    node: AstScopedNode
    class_name: str | None
    function_name: str | None


@dataclass(frozen=True)
class ClassAstObservation:
    node: ast.ClassDef
    is_dataclass_family: bool


class AutoRegisterMeta(ABCMeta):
    def __new__(
        mcls,
        name: str,
        bases: tuple[type[object], ...],
        namespace: dict[str, object],
        **kwargs: object,
    ):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if namespace.get("_registry_root", False):
            setattr(cls, "_registered_spec_types", [])
            return cls
        if namespace.get("_registry_skip", False):
            return cls
        if cls.__abstractmethods__:
            return cls
        for base in cls.__mro__[1:]:
            registry = base.__dict__.get("_registered_spec_types")
            if registry is None:
                continue
            registry.append(cls)
            break
        return cls


class ModuleShapeSpec(ABC):
    @abstractmethod
    def collect(self, parsed_module: ParsedModule) -> list[object]:
        raise NotImplementedError


class AutoRegisteredModuleShapeSpec(ModuleShapeSpec, ABC, metaclass=AutoRegisterMeta):
    _registry_root: ClassVar[bool] = False
    _registered_spec_types: ClassVar[list[type["AutoRegisteredModuleShapeSpec"]]]

    @classmethod
    def registered_specs(cls) -> tuple["AutoRegisteredModuleShapeSpec", ...]:
        return tuple(spec_type() for spec_type in cls._registered_spec_types)

    @classmethod
    def all_registered_specs(cls) -> tuple["AutoRegisteredModuleShapeSpec", ...]:
        seen: set[type[AutoRegisteredModuleShapeSpec]] = set()
        ordered: list[AutoRegisteredModuleShapeSpec] = []
        queue = list(cls.__subclasses__())
        while queue:
            current = queue.pop(0)
            queue.extend(current.__subclasses__())
            registry = current.__dict__.get("_registered_spec_types")
            if registry is None:
                continue
            for spec_type in registry:
                if spec_type in seen:
                    continue
                seen.add(spec_type)
                ordered.append(spec_type())
        return tuple(ordered)


class CollectedFamily(ABC, metaclass=AutoRegisterMeta):
    _registry_root: ClassVar[bool] = False
    _registered_spec_types: ClassVar[list[type["CollectedFamily"]]]
    item_type: ClassVar[type[object]]

    @classmethod
    def registered_families(cls) -> tuple[type["CollectedFamily"], ...]:
        return tuple(cls._registered_spec_types)

    @classmethod
    def all_registered_families(cls) -> tuple[type["CollectedFamily"], ...]:
        seen: set[type[CollectedFamily]] = set()
        ordered: list[type[CollectedFamily]] = []
        queue = list(cls.__subclasses__())
        while queue:
            current = queue.pop(0)
            queue.extend(current.__subclasses__())
            registry = current.__dict__.get("_registered_spec_types")
            if registry is None:
                continue
            for family_type in registry:
                if family_type in seen:
                    continue
                seen.add(family_type)
                ordered.append(family_type)
        return tuple(ordered)

    @classmethod
    @abstractmethod
    def collect(cls, parsed_module: ParsedModule) -> list[object]:
        raise NotImplementedError


def collect_family_items(
    parsed_module: ParsedModule,
    family: type[CollectedFamily],
) -> list[object]:
    return [
        item
        for item in _flatten_collected_items(family.collect(parsed_module))
        if isinstance(item, family.item_type)
    ]


class RegisteredSpecCollectedFamily(CollectedFamily, ABC):
    spec_root: ClassVar[type[AutoRegisteredModuleShapeSpec]]

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[object]:
        return _collect_items_from_spec_root(
            cls.spec_root,
            parsed_module,
            cls.item_type,
        )


class SingleSpecCollectedFamily(CollectedFamily, ABC):
    spec: ClassVar[ModuleShapeSpec]

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[object]:
        return [
            item
            for item in _flatten_collected_items(cls.spec.collect(parsed_module))
            if isinstance(item, cls.item_type)
        ]


class ScopedShapeSpec(ModuleShapeSpec, ABC):
    @property
    @abstractmethod
    def node_types(self) -> tuple[type[ast.AST], ...]:
        raise NotImplementedError

    def collect(self, parsed_module: ParsedModule) -> list[object]:
        shapes: list[object] = []
        for observation in collect_scoped_observations(parsed_module, self.node_types):
            shape = self.build_shape(parsed_module, observation)
            if shape is not None:
                shapes.append(shape)
        return shapes

    @abstractmethod
    def build_shape(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> object | None:
        raise NotImplementedError


class ObservationShapeSpec(ScopedShapeSpec, ABC):
    def build_shape(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> object | None:
        if not isinstance(observation.node, self.node_types):
            return None
        return self.build_from_observation(parsed_module, observation)

    @abstractmethod
    def build_from_observation(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> object | None:
        raise NotImplementedError


class ContextForwardingShapeSpec(ObservationShapeSpec, ABC):
    node_type: ClassVar[type[ast.AST]]

    @property
    def node_types(self) -> tuple[type[ast.AST], ...]:
        return (type(self).node_type,)

    def build_from_observation(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> object | None:
        node = observation.node
        assert isinstance(node, type(self).node_type)
        return self.build_from_context(parsed_module, node, observation)

    @abstractmethod
    def build_from_context(
        self,
        parsed_module: ParsedModule,
        node: ast.AST,
        observation: ScopedAstObservation,
    ) -> object | None:
        raise NotImplementedError


class ContextHelperShapeSpec(ContextForwardingShapeSpec, ABC):
    shape_helper: ClassVar[
        Callable[[ParsedModule, ast.AST, str | None, str | None], object | None]
    ]

    def build_from_context(
        self,
        parsed_module: ParsedModule,
        node: ast.AST,
        observation: ScopedAstObservation,
    ) -> object | None:
        return type(self).shape_helper(
            parsed_module,
            node,
            observation.class_name,
            observation.function_name,
        )


class FunctionObservationSpec(ObservationShapeSpec, ABC):
    @property
    def node_types(self) -> tuple[type[ast.AST], ...]:
        return (ast.FunctionDef, ast.AsyncFunctionDef)

    def build_from_observation(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> object | None:
        node = observation.node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        return self.build_from_function(parsed_module, node, observation)

    @abstractmethod
    def build_from_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> object | None:
        raise NotImplementedError


class AssignObservationSpec(ObservationShapeSpec, ABC):
    @property
    def node_types(self) -> tuple[type[ast.AST], ...]:
        return (ast.Assign,)

    def build_from_observation(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> object | None:
        node = observation.node
        if not isinstance(node, ast.Assign):
            return None
        return self.build_from_assign(parsed_module, node, observation)

    @abstractmethod
    def build_from_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> object | None:
        raise NotImplementedError


class SentinelTypeObservationSpecRoot(AutoRegisteredModuleShapeSpec, ABC):
    _registry_root = True


def parse_python_modules(root: Path) -> list[ParsedModule]:
    modules: list[ParsedModule] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        modules.append(ParsedModule(path=path, module=ast.parse(source), source=source))
    return modules


class _ShapeNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node = ast.arg(arg="ARG", annotation=None, type_comment=None)
        return ast.copy_location(node, node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="STR"), node)
        if isinstance(node.value, (int, float, complex)):
            return ast.copy_location(ast.Constant(value=0), node)
        if node.value is None:
            return ast.copy_location(ast.Constant(value=None), node)
        if isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value=True), node)
        return ast.copy_location(ast.Constant(value="CONST"), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        value = self.visit(node.value)
        new_node = ast.Attribute(value=value, attr="ATTR", ctx=node.ctx)
        return ast.copy_location(new_node, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = ast.FunctionDef(
            name="FUNC",
            args=self.visit(node.args),
            body=[self.visit(stmt) for stmt in node.body],
            decorator_list=[self.visit(dec) for dec in node.decorator_list],
            returns=None,
            type_comment=None,
        )
        return ast.copy_location(node, node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = ast.AsyncFunctionDef(
            name="FUNC",
            args=self.visit(node.args),
            body=[self.visit(stmt) for stmt in node.body],
            decorator_list=[self.visit(dec) for dec in node.decorator_list],
            returns=None,
            type_comment=None,
        )
        return ast.copy_location(node, node)


def fingerprint_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    normalized = _ShapeNormalizer().visit(copy.deepcopy(node))
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _subscript_base_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    return _terminal_name(node.value)


_CLASSVAR_REFERENCE_FAMILY = AstNameFamily(frozenset({"ClassVar"}))
_DATACLASS_DECORATOR_FAMILY = AstNameFamily(frozenset({"dataclass"}))
_ATTRIBUTE_ERROR_FAMILY = AstNameFamily(frozenset({"AttributeError"}))
_HASATTR_CALL_FAMILY = AstNameFamily(frozenset({"hasattr"}))
_GETATTR_CALL_FAMILY = AstNameFamily(frozenset({"getattr"}))
_REGISTRATION_CALL_FAMILY = AstNameFamily(
    frozenset({"register", "add", "register_class", "register_type"})
)
_REGISTRATION_DECORATOR_FAMILY = AstNameFamily(
    _REGISTRATION_CALL_FAMILY.names | frozenset({"auto_register"})
)


def _node_matches_family(node: ast.AST, family: AstNameFamily) -> bool:
    return (
        _terminal_name(node) in family.names
        or _subscript_base_name(node) in family.names
    )


def _terminal_name_in_family(node: ast.AST, family: AstNameFamily) -> str | None:
    terminal_name = _terminal_name(node)
    if terminal_name in family.names:
        return terminal_name
    return None


def _name_family(names: set[str] | frozenset[str]) -> AstNameFamily:
    return AstNameFamily(frozenset(names))


def _iter_attribute_family_calls(
    parsed_module: ParsedModule, family: AstNameFamily
) -> tuple[AstCallObservation, ...]:
    observations: list[AstCallObservation] = []
    for node in ast.walk(parsed_module.module):
        if not isinstance(node, ast.Call):
            continue
        matched_name = _attribute_call_family_name(node, family)
        if matched_name is None:
            continue
        observations.append(AstCallObservation(call=node, matched_name=matched_name))
    return tuple(sorted(observations, key=lambda item: item.call.lineno))


def _attribute_call_family_name(node: ast.Call, family: AstNameFamily) -> str | None:
    if not isinstance(node.func, ast.Attribute):
        return None
    return _terminal_name_in_family(node.func, family)


def _iter_class_decorator_family_calls(
    parsed_module: ParsedModule, family: AstNameFamily
) -> tuple[tuple[ast.ClassDef, ast.Call, str], ...]:
    observations: list[tuple[ast.ClassDef, ast.Call, str]] = []
    for node in ast.walk(parsed_module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            matched_name = _terminal_name_in_family(decorator.func, family)
            if matched_name is None:
                continue
            observations.append((node, decorator, matched_name))
    return tuple(sorted(observations, key=lambda item: item[0].lineno))


def _node_display_name(node: ast.AST) -> str:
    return _terminal_name(node) or node.__class__.__name__


def collect_scoped_observations(
    parsed_module: ParsedModule,
    node_types: tuple[type[ast.AST], ...],
) -> tuple[ScopedAstObservation, ...]:
    observations: list[ScopedAstObservation] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.function_stack: list[str] = []

        def _record(self, node: ast.AST) -> None:
            if not isinstance(node, node_types):
                return
            observations.append(
                ScopedAstObservation(
                    node=node,
                    class_name=self.class_stack[-1] if self.class_stack else None,
                    function_name=(
                        self.function_stack[-1] if self.function_stack else None
                    ),
                )
            )

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._record(node)
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node)
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record(node)
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def generic_visit(self, node: ast.AST) -> None:
            if not isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                self._record(node)
            super().generic_visit(node)

    Visitor().visit(parsed_module.module)
    return tuple(observations)


def collect_scoped_shapes(
    parsed_module: ParsedModule,
    spec: ScopedShapeSpec,
) -> list[object]:
    return spec.collect(parsed_module)


def _flatten_collected_items(items: list[object]) -> tuple[object, ...]:
    flattened: list[object] = []
    for item in items:
        if isinstance(item, tuple):
            flattened.extend(item)
        else:
            flattened.append(item)
    return tuple(flattened)


def _collect_items_from_spec_root(
    spec_root: type[AutoRegisteredModuleShapeSpec],
    parsed_module: ParsedModule,
    item_type: type[object],
) -> list[object]:
    items: list[object] = []
    for spec in spec_root.registered_specs():
        items.extend(
            item
            for item in _flatten_collected_items(spec.collect(parsed_module))
            if isinstance(item, item_type)
        )
    return items


def _execution_level_for_scope(function_name: str | None) -> StructuralExecutionLevel:
    if function_name is None:
        return StructuralExecutionLevel.MODULE_BODY
    return StructuralExecutionLevel.FUNCTION_BODY


def _class_observations(parsed_module: ParsedModule) -> tuple[ClassAstObservation, ...]:
    observations: list[ClassAstObservation] = []
    for observation in collect_scoped_observations(parsed_module, (ast.ClassDef,)):
        node = observation.node
        assert isinstance(node, ast.ClassDef)
        observations.append(
            ClassAstObservation(
                node=node,
                is_dataclass_family=any(
                    _node_matches_family(decorator, _DATACLASS_DECORATOR_FAMILY)
                    for decorator in node.decorator_list
                ),
            )
        )
    return tuple(observations)


def _known_class_family(parsed_module: ParsedModule) -> AstNameFamily:
    return _name_family({item.node.name for item in _class_observations(parsed_module)})


def _class_body_field_observation(
    parsed_module: ParsedModule,
    class_name: str,
    is_dataclass_family: bool,
    stmt: ast.stmt,
) -> FieldObservation | None:
    if not is_dataclass_family:
        return None
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        if _node_matches_family(stmt.annotation, _CLASSVAR_REFERENCE_FAMILY):
            return None
        return FieldObservation(
            file_path=str(parsed_module.path),
            class_name=class_name,
            field_name=stmt.target.id,
            lineno=stmt.lineno,
            execution_level=StructuralExecutionLevel.CLASS_BODY,
            origin_kind=(
                FieldOriginKind.DATACLASS_FIELD
                if is_dataclass_family
                else FieldOriginKind.CLASS_ANNOTATION
            ),
            is_dataclass_family=is_dataclass_family,
            value_fingerprint=(
                _fingerprint_builder_value(stmt.value)
                if stmt.value is not None
                else None
            ),
            annotation_text=ast.unparse(stmt.annotation),
            annotation_fingerprint=_annotation_fingerprint(stmt.annotation),
        )
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            return None
        return FieldObservation(
            file_path=str(parsed_module.path),
            class_name=class_name,
            field_name=target.id,
            lineno=stmt.lineno,
            execution_level=StructuralExecutionLevel.CLASS_BODY,
            origin_kind=FieldOriginKind.CLASS_ASSIGNMENT,
            is_dataclass_family=is_dataclass_family,
            value_fingerprint=_fingerprint_builder_value(stmt.value),
        )
    return None


def _annotation_fingerprint(node: ast.AST) -> str:
    return ast.dump(copy.deepcopy(node), include_attributes=False)


def _parameter_annotation_map(
    function: ast.FunctionDef,
) -> dict[str, tuple[str, str]]:
    annotations: dict[str, tuple[str, str]] = {}
    for arg in function.args.args:
        if arg.annotation is None:
            continue
        annotations[arg.arg] = (
            ast.unparse(arg.annotation),
            _annotation_fingerprint(arg.annotation),
        )
    return annotations


def _init_field_observations(
    parsed_module: ParsedModule,
    class_name: str,
    is_dataclass_family: bool,
    function: ast.FunctionDef,
) -> list[FieldObservation]:
    observations: list[FieldObservation] = []
    parameter_annotations = _parameter_annotation_map(function)
    for stmt in function.body:
        value: ast.AST | None = None
        target: ast.AST | None = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            value = stmt.value
        else:
            continue
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            continue
        observations.append(
            FieldObservation(
                file_path=str(parsed_module.path),
                class_name=class_name,
                field_name=target.attr,
                lineno=stmt.lineno,
                execution_level=StructuralExecutionLevel.INIT_BODY,
                origin_kind=FieldOriginKind.INIT_ASSIGNMENT,
                is_dataclass_family=is_dataclass_family,
                value_fingerprint=(
                    _fingerprint_builder_value(value) if value is not None else None
                ),
                annotation_text=(
                    parameter_annotations[value.id][0]
                    if isinstance(value, ast.Name) and value.id in parameter_annotations
                    else None
                ),
                annotation_fingerprint=(
                    parameter_annotations[value.id][1]
                    if isinstance(value, ast.Name) and value.id in parameter_annotations
                    else None
                ),
            )
        )
    return observations


def _parent_map(module: ast.Module) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(module)
        for child in ast.iter_child_nodes(parent)
    }


def _enclosing_function_name(
    node: ast.AST, parent_map: dict[ast.AST, ast.AST]
) -> str | None:
    current: ast.AST | None = node
    while current is not None:
        current = parent_map.get(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return None


def _literal_dispatch_case(
    test: ast.AST, literal_type: type[object]
) -> tuple[str, str, str] | None:
    if not isinstance(test, ast.Compare):
        return None
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    if len(test.comparators) != 1:
        return None
    left = test.left
    right = test.comparators[0]
    if isinstance(left, ast.Constant) and isinstance(left.value, literal_type):
        return (
            ast.dump(right, include_attributes=False),
            ast.unparse(right),
            repr(left.value),
        )
    if isinstance(right, ast.Constant) and isinstance(right.value, literal_type):
        return (
            ast.dump(left, include_attributes=False),
            ast.unparse(left),
            repr(right.value),
        )
    return None


def _literal_dispatch_observation_from_if(
    parsed_module: ParsedModule,
    node: ast.If,
    literal_type: type[object],
    literal_kind: LiteralKind,
    parent_map: dict[ast.AST, ast.AST],
) -> LiteralDispatchObservation | None:
    literal_cases: list[str] = []
    branch_lines: list[int] = []
    axis_fingerprint: str | None = None
    axis_expression: str | None = None
    current: ast.stmt | None = node
    while isinstance(current, ast.If):
        case = _literal_dispatch_case(current.test, literal_type)
        if case is None:
            return None
        current_fingerprint, current_expression, literal_case = case
        if axis_fingerprint is None:
            axis_fingerprint = current_fingerprint
            axis_expression = current_expression
        elif axis_fingerprint != current_fingerprint:
            return None
        literal_cases.append(literal_case)
        branch_lines.append(current.lineno)
        current = current.orelse[0] if len(current.orelse) == 1 else None
    if axis_fingerprint is None or axis_expression is None or len(literal_cases) < 2:
        return None
    function_name = _enclosing_function_name(node, parent_map)
    return LiteralDispatchObservation(
        file_path=str(parsed_module.path),
        line=node.lineno,
        symbol=(function_name or "<module>") + ":literal-dispatch",
        axis_fingerprint=axis_fingerprint,
        axis_expression=axis_expression,
        literal_cases=tuple(literal_cases),
        literal_kind=literal_kind,
        execution_level=_execution_level_for_scope(function_name),
        branch_lines=tuple(branch_lines),
        scope_owner=function_name,
    )


def _iter_statement_blocks(
    module: ast.Module,
) -> tuple[tuple[str | None, list[ast.stmt]], ...]:
    blocks: list[tuple[str | None, list[ast.stmt]]] = [(None, module.body)]
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            blocks.append((node.name, node.body))
    return tuple(blocks)


def _inline_literal_dispatch_groups(
    parsed_module: ParsedModule,
    owner_name: str | None,
    block: list[ast.stmt],
    literal_type: type[object],
    literal_kind: LiteralKind,
) -> tuple[LiteralDispatchObservation, ...]:
    groups: dict[str, list[tuple[int, str, str]]] = {}
    for stmt in block:
        if not isinstance(stmt, ast.If):
            continue
        case = _literal_dispatch_case(stmt.test, literal_type)
        if case is None:
            continue
        axis_fingerprint, axis_expression, literal_case = case
        groups.setdefault(axis_fingerprint, []).append(
            (stmt.lineno, axis_expression, literal_case)
        )
    observations: list[LiteralDispatchObservation] = []
    for axis_fingerprint, items in groups.items():
        literal_cases = tuple(
            sorted({literal_case for _, _, literal_case in items}, key=str)
        )
        if len(literal_cases) < 2:
            continue
        observations.append(
            LiteralDispatchObservation(
                file_path=str(parsed_module.path),
                line=min(line for line, _, _ in items),
                symbol=(owner_name or "<module>") + ":inline-literal-dispatch",
                axis_fingerprint=axis_fingerprint,
                axis_expression=items[0][1],
                literal_cases=literal_cases,
                literal_kind=literal_kind,
                execution_level=_execution_level_for_scope(owner_name),
                branch_lines=tuple(sorted(line for line, _, _ in items)),
                scope_owner=owner_name,
            )
        )
    return tuple(sorted(observations, key=lambda item: item.line))


def _is_docstring_expr(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _trim_docstring_body(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and _is_docstring_expr(body[0]):
        return body[1:]
    return body


def _projection_helper_shape_from_function(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ProjectionHelperShape | None:
    body = _trim_docstring_body(function.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    returned = body[0].value
    if not isinstance(returned, ast.Call) or len(returned.args) != 1:
        return None
    outer_call_name = _terminal_name(returned.func)
    if outer_call_name not in {"tuple", "list", "set"}:
        return None
    inner_call = returned.args[0]
    if not isinstance(inner_call, ast.Call) or len(inner_call.args) != 1:
        return None
    aggregator_name = _terminal_name(inner_call.func)
    if aggregator_name is None:
        return None
    generator = inner_call.args[0]
    if not isinstance(generator, ast.GeneratorExp) or len(generator.generators) != 1:
        return None
    comp = generator.generators[0]
    if comp.is_async or comp.ifs or not isinstance(comp.target, ast.Name):
        return None
    if not isinstance(generator.elt, ast.Attribute):
        return None
    if not isinstance(generator.elt.value, ast.Name):
        return None
    if generator.elt.value.id != comp.target.id:
        return None
    return ProjectionHelperShape(
        file_path=str(parsed_module.path),
        function_name=function.name,
        lineno=function.lineno,
        outer_call_name=outer_call_name,
        aggregator_name=aggregator_name,
        iterable_fingerprint=fingerprint_function(function),
        projected_attribute=generator.elt.attr,
    )


def _accessor_wrapper_candidate_from_function(
    parsed_module: ParsedModule,
    class_name: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> AccessorWrapperCandidate | None:
    if _is_dunder_method(function.name):
        return None
    if _has_property_like_decorator(function):
        return None
    body = _trim_docstring_body(function.body)
    if not body:
        return None
    getter_candidate = _getter_wrapper_candidate(function, body)
    if getter_candidate is not None:
        target_expression, observed_attribute, wrapper_shape = getter_candidate
        return AccessorWrapperCandidate(
            file_path=str(parsed_module.path),
            class_name=class_name,
            method_name=function.name,
            lineno=function.lineno,
            target_expression=target_expression,
            observed_attribute=observed_attribute,
            accessor_kind="getter",
            wrapper_shape=wrapper_shape,
        )
    setter_candidate = _setter_wrapper_candidate(function, body)
    if setter_candidate is not None:
        target_expression, observed_attribute = setter_candidate
        return AccessorWrapperCandidate(
            file_path=str(parsed_module.path),
            class_name=class_name,
            method_name=function.name,
            lineno=function.lineno,
            target_expression=target_expression,
            observed_attribute=observed_attribute,
            accessor_kind="setter",
            wrapper_shape="write_through",
        )
    return None


def _scoped_shape_wrapper_function_from_function(
    parsed_module: ParsedModule,
    function: ast.FunctionDef,
) -> ScopedShapeWrapperFunction | None:
    if len(function.args.args) != 2:
        return None
    body = _trim_docstring_body(function.body)
    if len(body) < 3:
        return None
    first_stmt = body[0]
    if not (
        isinstance(first_stmt, ast.Assign)
        and len(first_stmt.targets) == 1
        and isinstance(first_stmt.targets[0], ast.Name)
        and first_stmt.targets[0].id == "node"
        and isinstance(first_stmt.value, ast.Attribute)
        and isinstance(first_stmt.value.value, ast.Name)
        and first_stmt.value.value.id == function.args.args[1].arg
        and first_stmt.value.attr == "node"
    ):
        return None
    second_stmt = body[1]
    if not isinstance(second_stmt, ast.If):
        return None
    node_types = _guarded_node_types(second_stmt.test, "node")
    if not node_types:
        return None
    if not (
        len(second_stmt.body) == 1
        and isinstance(second_stmt.body[0], ast.Return)
        and isinstance(second_stmt.body[0].value, ast.Constant)
        and second_stmt.body[0].value.value is None
    ):
        return None
    if not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return None
    return ScopedShapeWrapperFunction(
        file_path=str(parsed_module.path),
        function_name=function.name,
        lineno=function.lineno,
        node_types=node_types,
    )


def _scoped_shape_wrapper_spec_from_assign(
    parsed_module: ParsedModule,
    node: ast.Assign,
) -> ScopedShapeWrapperSpec | None:
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return None
    if not isinstance(node.value, ast.Call):
        return None
    if _terminal_name(node.value.func) != "ScopedShapeSpec":
        return None
    node_types: tuple[str, ...] = ()
    function_name = None
    for keyword in node.value.keywords:
        if keyword.arg == "node_types":
            node_types = _type_name_tuple(keyword.value)
        if keyword.arg == "build_shape":
            function_name = _terminal_name(keyword.value)
    if not node_types or function_name is None:
        return None
    return ScopedShapeWrapperSpec(
        file_path=str(parsed_module.path),
        spec_name=target.id,
        lineno=node.lineno,
        function_name=function_name,
        node_types=node_types,
    )


def _getter_wrapper_candidate(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    body: list[ast.stmt],
) -> tuple[str, str, str] | None:
    if len(function.args.args) != 1:
        return None
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    expr = body[0].value
    if _is_self_attribute_expression(expr):
        observed_attribute = _self_attribute_name(expr)
        if observed_attribute is None:
            return None
        return ast.unparse(expr), observed_attribute, "read_through"
    wrapped = _wrapped_self_attribute_expression(expr)
    if wrapped is not None:
        wrapper_name, observed_attribute = wrapped
        return ast.unparse(expr), observed_attribute, f"computed_{wrapper_name}"
    return None


def _setter_wrapper_candidate(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    body: list[ast.stmt],
) -> tuple[str, str] | None:
    if len(function.args.args) != 2:
        return None
    if len(body) != 1 or not isinstance(body[0], ast.Assign):
        return None
    assign = body[0]
    if len(assign.targets) != 1:
        return None
    target = assign.targets[0]
    value_arg = function.args.args[1].arg
    if not (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return None
    if not (isinstance(assign.value, ast.Name) and assign.value.id == value_arg):
        return None
    observed_attribute = _self_attribute_name(target)
    if observed_attribute is None:
        return None
    return ast.unparse(target), observed_attribute


def _is_self_attribute_expression(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _wrapped_self_attribute_expression(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return None
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id not in {
        "tuple",
        "list",
        "set",
        "frozenset",
        "str",
        "int",
        "bool",
        "len",
        "sorted",
    }:
        return None
    if not _is_self_attribute_expression(node.args[0]):
        return None
    observed_attribute = _self_attribute_name(node.args[0])
    if observed_attribute is None:
        return None
    return node.func.id, observed_attribute


def _self_attribute_name(node: ast.AST) -> str | None:
    if not _is_self_attribute_expression(node):
        return None
    assert isinstance(node, ast.Attribute)
    return node.attr.lstrip("_") or node.attr


def _is_dunder_method(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _has_property_like_decorator(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "setter":
            return True
    return False


def _guarded_node_types(test: ast.AST, expected_name: str) -> tuple[str, ...]:
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _guarded_node_types(test.operand, expected_name)
    if not isinstance(test, ast.Call):
        return ()
    if not isinstance(test.func, ast.Name) or test.func.id != "isinstance":
        return ()
    if len(test.args) != 2:
        return ()
    if not isinstance(test.args[0], ast.Name) or test.args[0].id != expected_name:
        return ()
    return _type_name_tuple(test.args[1])


def _type_name_tuple(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    if isinstance(node, ast.Tuple):
        names: list[str] = []
        for item in node.elts:
            names.extend(_type_name_tuple(item))
        return tuple(names)
    return ()


def _config_dispatch_observations(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ConfigDispatchObservation, ...]:
    seen: set[tuple[int, str]] = set()
    observations: list[ConfigDispatchObservation] = []
    for node in ast.walk(function):
        if isinstance(node, ast.If):
            for attr_name in _config_dispatch_attributes(node.test):
                key = (node.lineno, attr_name)
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    ConfigDispatchObservation(
                        file_path=str(parsed_module.path),
                        line=node.lineno,
                        symbol=function.name,
                        observed_attribute=attr_name,
                    )
                )
        if isinstance(node, ast.Match):
            for attr_name in _match_config_dispatch_attributes(node.subject):
                key = (node.lineno, attr_name)
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    ConfigDispatchObservation(
                        file_path=str(parsed_module.path),
                        line=node.lineno,
                        symbol=function.name,
                        observed_attribute=attr_name,
                    )
                )
    return tuple(
        sorted(observations, key=lambda item: (item.line, item.observed_attribute))
    )


def _config_dispatch_attributes(test: ast.AST) -> tuple[str, ...]:
    attrs: set[str] = set()
    for node in ast.walk(test):
        if isinstance(node, ast.Call) and _terminal_name_in_family(
            node.func, _HASATTR_CALL_FAMILY
        ):
            if _call_targets_name(node, "config") and len(node.args) >= 2:
                if isinstance(node.args[1], ast.Constant) and isinstance(
                    node.args[1].value, str
                ):
                    attrs.add(node.args[1].value)
        if isinstance(node, ast.Call) and _terminal_name_in_family(
            node.func, _GETATTR_CALL_FAMILY
        ):
            if _call_targets_name(node, "config") and len(node.args) >= 2:
                if isinstance(node.args[1], ast.Constant) and isinstance(
                    node.args[1].value, str
                ):
                    attrs.add(node.args[1].value)
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                continue
            if not isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                continue
            left_name = _config_subject_name(node.left)
            right_name = _config_subject_name(node.comparators[0])
            left_literal = _literal_dispatch_value(node.left)
            right_literal = _literal_dispatch_value(node.comparators[0])
            if left_name is not None and right_literal is not None:
                attrs.add(left_name)
            if right_name is not None and left_literal is not None:
                attrs.add(right_name)
    return tuple(sorted(attrs))


def _match_config_dispatch_attributes(subject: ast.AST) -> tuple[str, ...]:
    attr_name = _config_subject_name(subject)
    if attr_name is not None:
        return (attr_name,)
    return ()


def _class_marker_observations(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ClassMarkerObservation, ...]:
    seen: set[tuple[int, str]] = set()
    observations: list[ClassMarkerObservation] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and _terminal_name_in_family(
            node.func, _HASATTR_CALL_FAMILY
        ):
            target = node.args[0] if node.args else None
            marker_name = None
            if _is_class_target(target):
                marker_name = (
                    _constant_string(node.args[1]) if len(node.args) >= 2 else None
                )
            if marker_name is not None:
                key = (node.lineno, marker_name)
                if key not in seen:
                    seen.add(key)
                    observations.append(
                        ClassMarkerObservation(
                            file_path=str(parsed_module.path),
                            line=node.lineno,
                            symbol=function.name,
                            marker_name=marker_name,
                        )
                    )
        if isinstance(node, ast.Attribute) and node.attr.startswith("_is_"):
            key = (node.lineno, node.attr)
            if key not in seen:
                seen.add(key)
                observations.append(
                    ClassMarkerObservation(
                        file_path=str(parsed_module.path),
                        line=node.lineno,
                        symbol=function.name,
                        marker_name=node.attr,
                    )
                )
    return tuple(sorted(observations, key=lambda item: (item.line, item.marker_name)))


def _interface_generation_observation(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> InterfaceGenerationObservation | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if _terminal_name(node.func) != "type":
            continue
        if len(node.args) < 3:
            continue
        bases = node.args[1]
        namespace = node.args[2]
        if not isinstance(namespace, ast.Dict) or namespace.keys:
            continue
        if not isinstance(bases, ast.Tuple):
            continue
        if any(_terminal_name(base) == "ABC" for base in bases.elts):
            return InterfaceGenerationObservation(
                file_path=str(parsed_module.path),
                line=node.lineno,
                symbol=function.name,
                generator_name="type",
            )
    return None


def _sentinel_type_observation(
    parsed_module: ParsedModule,
    node: ast.Assign,
) -> SentinelTypeObservation | None:
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return None
    if not isinstance(node.value, ast.Call):
        return None
    if not isinstance(node.value.func, ast.Call):
        return None
    if _terminal_name(node.value.func.func) != "type":
        return None
    return SentinelTypeObservation(
        file_path=str(parsed_module.path),
        line=node.lineno,
        symbol=target.id,
        sentinel_name=target.id,
    )


def _sentinel_type_usage_observations(
    parsed_module: ParsedModule,
) -> tuple[SentinelTypeObservation, ...]:
    sentinel_names = {
        item.sentinel_name
        for node in ast.walk(parsed_module.module)
        if isinstance(node, ast.Assign)
        if (item := _sentinel_type_observation(parsed_module, node)) is not None
    }
    observations: list[SentinelTypeObservation] = []
    seen: set[tuple[int, str]] = set()
    for node in ast.walk(parsed_module.module):
        if isinstance(node, (ast.Compare, ast.Subscript)):
            names = {
                subnode.id
                for subnode in ast.walk(node)
                if isinstance(subnode, ast.Name)
            }
            for name in sorted(names & sentinel_names):
                key = (node.lineno, name)
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    SentinelTypeObservation(
                        file_path=str(parsed_module.path),
                        line=node.lineno,
                        symbol=f"sentinel:{name}",
                        sentinel_name=name,
                    )
                )
    return tuple(sorted(observations, key=lambda item: (item.line, item.sentinel_name)))


def _dynamic_method_injection_observations(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[DynamicMethodInjectionObservation, ...]:
    observations: list[DynamicMethodInjectionObservation] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if _terminal_name(node.func) != "setattr":
            continue
        if len(node.args) < 3:
            continue
        target = node.args[0]
        if isinstance(target, ast.Name) and target.id.endswith("type"):
            observations.append(
                DynamicMethodInjectionObservation(
                    file_path=str(parsed_module.path),
                    line=node.lineno,
                    symbol=function.name,
                    mutator_name="setattr",
                )
            )
    return tuple(sorted(observations, key=lambda item: item.line))


def _runtime_type_generation_observation(
    parsed_module: ParsedModule,
    node: ast.Call,
    observation: ScopedAstObservation,
) -> RuntimeTypeGenerationObservation | None:
    generator_name = _terminal_name(node.func)
    if generator_name not in {"type", "make_dataclass", "new_class"}:
        return None
    return RuntimeTypeGenerationObservation(
        file_path=str(parsed_module.path),
        line=node.lineno,
        symbol=observation.function_name or generator_name,
        generator_name=generator_name,
    )


def _lineage_mapping_observation(
    parsed_module: ParsedModule,
    node: ast.Assign,
) -> LineageMappingObservation | None:
    for target in node.targets:
        if not isinstance(target, ast.Subscript):
            continue
        name = _terminal_name(target.value)
        if name and any(
            token in name.lower()
            for token in ("lazy", "base", "type", "mapping", "registry")
        ):
            return LineageMappingObservation(
                file_path=str(parsed_module.path),
                line=node.lineno,
                symbol=name,
                mapping_name=name,
            )
    return None


def _dual_axis_resolution_observation(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> DualAxisResolutionObservation | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.For):
            continue
        inner_loops = [child for child in node.body if isinstance(child, ast.For)]
        if not inner_loops:
            continue
        outer_name = _loop_target_name(node.target)
        inner_name = _loop_target_name(inner_loops[0].target)
        text = ast.dump(inner_loops[0].iter, include_attributes=False)
        if "__mro__" in text or "mro" in text.lower() or "type" in text.lower():
            if outer_name and any(
                token in outer_name.lower() for token in ("scope", "context", "level")
            ):
                return DualAxisResolutionObservation(
                    file_path=str(parsed_module.path),
                    line=node.lineno,
                    symbol=function.name,
                    outer_axis_name=outer_name,
                    inner_axis_name=inner_name or "mro_type",
                )
    return None


def _loop_target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _call_targets_name(node: ast.Call, expected_name: str) -> bool:
    return bool(
        node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == expected_name
    )


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _attribute_name_if_root(node: ast.AST, expected_root: str) -> str | None:
    if not isinstance(node, ast.Attribute):
        return None
    if isinstance(node.value, ast.Name) and node.value.id == expected_root:
        return node.attr
    return None


def _config_subject_name(node: ast.AST) -> str | None:
    attr_name = _attribute_name_if_root(node, "config")
    if attr_name is not None:
        return attr_name
    if isinstance(node, ast.Call) and _terminal_name_in_family(
        node.func, _GETATTR_CALL_FAMILY
    ):
        if _call_targets_name(node, "config") and len(node.args) >= 2:
            return _constant_string(node.args[1])
    return None


def _literal_dispatch_value(node: ast.AST) -> object | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, bool)):
        return node.value
    return None


def _is_class_target(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Attribute) and node.attr == "__class__":
        return True
    if isinstance(node, ast.Call) and _terminal_name(node.func) == "type":
        return True
    return False


def _builder_call_shape(
    parsed_module: ParsedModule,
    node: ast.AST,
    class_name: str | None,
    function_name: str | None,
) -> BuilderCallShape | None:
    if not isinstance(node, ast.Call):
        return None
    if function_name is None:
        return None
    keyword_pairs = [(kw.arg, kw.value) for kw in node.keywords if kw.arg is not None]
    if len(keyword_pairs) < 3:
        return None
    callee_name = _terminal_name(node.func)
    if callee_name is None:
        return None
    keyword_names = tuple(name for name, _ in keyword_pairs)
    value_fingerprint = tuple(
        _fingerprint_builder_value(value) for _, value in keyword_pairs
    )
    source_roots = set()
    for _, value in keyword_pairs:
        source_roots.update(_root_names(value))
    source_name = next(iter(source_roots)) if len(source_roots) == 1 else None
    identity_field_names = tuple(
        name for name, value in keyword_pairs if _terminal_name(value) == name
    )
    return BuilderCallShape(
        file_path=str(parsed_module.path),
        class_name=class_name,
        function_name=function_name,
        lineno=node.lineno,
        callee_name=callee_name,
        keyword_names=keyword_names,
        value_fingerprint=value_fingerprint,
        source_arity=len(source_roots),
        source_name=source_name,
        identity_field_names=identity_field_names,
    )


def _export_dict_shape(
    parsed_module: ParsedModule,
    node: ast.AST,
    class_name: str | None,
    function_name: str | None,
) -> ExportDictShape | None:
    if not isinstance(node, ast.Dict):
        return None
    if function_name is None:
        return None
    key_pairs = [
        (key.value, value)
        for key, value in zip(node.keys, node.values, strict=False)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]
    if len(key_pairs) < 3 or len(key_pairs) != len(node.keys):
        return None
    key_names = tuple(name for name, _ in key_pairs)
    value_fingerprint = tuple(
        _fingerprint_builder_value(value) for _, value in key_pairs
    )
    source_roots = set()
    for _, value in key_pairs:
        source_roots.update(_root_names(value))
    if not source_roots:
        return None
    source_name = next(iter(source_roots)) if len(source_roots) == 1 else None
    identity_field_names = tuple(
        name for name, value in key_pairs if _terminal_name(value) == name
    )
    return ExportDictShape(
        file_path=str(parsed_module.path),
        class_name=class_name,
        function_name=function_name,
        lineno=node.lineno,
        key_names=key_names,
        value_fingerprint=value_fingerprint,
        source_arity=len(source_roots),
        source_name=source_name,
        identity_field_names=identity_field_names,
    )


class _BuilderValueNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="ROOT", ctx=node.ctx), node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="STR"), node)
        if isinstance(node.value, (int, float, complex)):
            return ast.copy_location(ast.Constant(value=0), node)
        if isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value=True), node)
        if node.value is None:
            return ast.copy_location(ast.Constant(value=None), node)
        return ast.copy_location(ast.Constant(value="CONST"), node)


def _fingerprint_builder_value(node: ast.AST) -> str:
    normalized = _BuilderValueNormalizer().visit(copy.deepcopy(node))
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _root_names(node: ast.AST) -> set[str]:
    roots: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                current = current.value
            if isinstance(current, ast.Name):
                roots.add(current.id)
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            roots.add(node.id)

    Visitor().visit(node)
    return roots


def _registration_key_fingerprint(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    return _fingerprint_builder_value(node.slice)


def _class_name_from_expr(
    node: ast.AST, known_class_family: AstNameFamily
) -> str | None:
    return _terminal_name_in_family(node, known_class_family)


from .observation_families import (
    AccessorWrapperObservationFamily,
    AccessorWrapperObservationSpec,
    AssignmentRegistrationShapeSpec,
    AttributeErrorProbeObservationSpec,
    AttributeProbeObservationFamily,
    AttributeProbeObservationSpec,
    BuilderCallShapeFamily,
    BuilderCallShapeSpec,
    CallAttributeProbeObservationSpec,
    CallRegistrationShapeSpec,
    ClassMarkerObservationFamily,
    ClassMarkerObservationSpec,
    ClassObservationSpec,
    ConfigDispatchObservationFamily,
    ConfigDispatchObservationSpec,
    DataclassBodyFieldObservationSpec,
    DecoratorRegistrationShapeSpec,
    DualAxisResolutionObservationFamily,
    DualAxisResolutionObservationSpec,
    DynamicMethodInjectionObservationFamily,
    DynamicMethodInjectionObservationSpec,
    ExportDictShapeFamily,
    ExportDictShapeSpec,
    FieldObservationFamily,
    FieldObservationSpec,
    GetAttrProbeObservationSpec,
    HasAttrProbeObservationSpec,
    InitAssignmentFieldObservationSpec,
    InlineLiteralDispatchObservationSpec,
    InlineStringLiteralDispatchObservationFamily,
    InlineStringLiteralDispatchObservationSpec,
    InterfaceGenerationObservationFamily,
    InterfaceGenerationObservationSpec,
    KnownClassFamilyShapeSpec,
    LineageMappingObservationFamily,
    LineageMappingObservationSpec,
    LiteralDispatchObservationSpec,
    MethodShapeFamily,
    MethodShapeSpec,
    NumericLiteralDispatchObservationFamily,
    NumericLiteralDispatchObservationSpec,
    ObservationFamily,
    ProjectionHelperObservationFamily,
    ProjectionHelperObservationSpec,
    RegistrationShapeFamily,
    RegistrationShapeSpec,
    RuntimeTypeGenerationObservationFamily,
    RuntimeTypeGenerationObservationSpec,
    ScopedShapeWrapperFunctionFamily,
    ScopedShapeWrapperFunctionObservationSpec,
    ScopedShapeWrapperObservationSpec,
    ScopedShapeWrapperSpecFamily,
    ScopedShapeWrapperSpecObservationSpec,
    SentinelTypeAssignmentObservationSpec,
    SentinelTypeObservationFamily,
    SentinelTypeObservationSpec,
    SentinelTypeUsageObservationSpec,
    ShapeFamily,
    StandardAccessorWrapperObservationSpec,
    StandardClassMarkerObservationSpec,
    StandardConfigDispatchObservationSpec,
    StandardDualAxisResolutionObservationSpec,
    StandardDynamicMethodInjectionObservationSpec,
    StandardInterfaceGenerationObservationSpec,
    StandardLineageMappingObservationSpec,
    StandardProjectionHelperObservationSpec,
    StringLiteralDispatchObservationFamily,
    StringLiteralDispatchObservationSpec,
    TypeCallGenerationObservationSpec,
    TypedLiteralObservationFamily,
    TypedLiteralObservationSpec,
)
