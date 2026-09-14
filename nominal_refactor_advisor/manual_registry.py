"""Source-derived proof models for registry refactors."""

from __future__ import annotations

import ast
from abc import ABC
from collections.abc import Hashable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    TypeAlias,
)

from .ast_tools import (
    AstExpressionProjection,
    EagerNameLoadCollector,
    ParsedModule,
    REGISTRATION_CALL_FAMILY,
    REGISTRATION_DECORATOR_FAMILY,
)
from .class_index import ModuleNominalBindingAuthority
from .collection_algebra import UniqueIdentityIndexAuthority
from .descriptor_algebra import CollectionAttributeProjection
from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from .name_algebra import CLASS_NAME_ALGEBRA
from .captured_reference import NamespaceContentsABC
from .registry_identity import (
    DEFAULT_REGISTRY_KEY_ATTRIBUTE,
    REGISTRY_ATTRIBUTE_NAME,
    AutoRegisterClassAuthority,
)
from .value_expression import (
    CompactValueExpression,
    LexicalValueReference,
    LiteralExpressionEffects,
    MappingKeyResolverABC,
)
from .enum_semantics import (
    PYTHON_ENUM_BASE_AUTHORITY,
    PythonEnumDeclarationAuthority,
)

RegistryAssignment: TypeAlias = ast.Assign | ast.AnnAssign


def _assignment_target(statement: RegistryAssignment) -> ast.expr:
    if isinstance(statement, ast.AnnAssign):
        return statement.target
    if len(statement.targets) != 1:
        raise ValueError("Registry assignment requires exactly one target")
    return statement.targets[0]


def _assignment_value(statement: RegistryAssignment) -> ast.AST | None:
    return statement.value


if TYPE_CHECKING:
    from .native_reference import NativeReferenceEnvironment
    from .captured_reference import CapturedReferenceResolution


@dataclass(frozen=True)
class DirectModuleClassGraph:
    """Direct top-level class graph recovered from one module declaration."""

    module: ast.Module

    @cached_property
    def classes_by_name(self) -> dict[str, ast.ClassDef]:
        class_nodes = tuple(
            statement
            for statement in self.module.body
            if isinstance(statement, ast.ClassDef)
        )
        try:
            return UniqueIdentityIndexAuthority.declarations_by_handle(
                class_nodes,
                lambda node: node.name,
            )
        except ValueError as error:
            raise ValueError(
                "Registry refactors require unique top-level class names"
            ) from error

    def descends_from(self, node: ast.ClassDef, ancestor_name: str) -> bool:
        pending = [base.id for base in node.bases if isinstance(base, ast.Name)]
        visited: set[str] = set()
        while pending:
            base_name = pending.pop()
            if base_name == ancestor_name:
                return True
            if base_name in visited:
                continue
            visited.add(base_name)
            base_node = self.classes_by_name.get(base_name)
            if base_node is not None:
                pending.extend(
                    base.id for base in base_node.bases if isinstance(base, ast.Name)
                )
        return False

    def descendants_of(self, ancestor_name: str) -> tuple[ast.ClassDef, ...]:
        return tuple(
            node
            for node in self.classes_by_name.values()
            if self.descends_from(node, ancestor_name)
        )


@dataclass(frozen=True)
class SourceModuleMappingKeyAuthority(MappingKeyResolverABC[ast.Module]):
    """Resolve AST key expressions from their exact module-local declarations."""

    module: ast.Module

    @cached_property
    def parsed_module(self) -> ParsedModule:
        return ParsedModule(
            Path("__nra_registry_key__.py"),
            "__nra_registry_key__",
            False,
            self.module,
            ast.unparse(self.module),
        )

    @cached_property
    def enum_declarations_by_name(
        self,
    ) -> dict[str, PythonEnumDeclarationAuthority]:
        bindings = ModuleNominalBindingAuthority(self.parsed_module)
        declarations: dict[str, PythonEnumDeclarationAuthority] = {}
        for name, node in DirectModuleClassGraph(self.module).classes_by_name.items():
            snapshot = bindings.snapshot_before(node.lineno)

            def qualified_reference(
                reference: ast.expr,
                shadowed_names: frozenset[str],
            ) -> str | None:
                parts = AstExpressionProjection.attribute_chain(reference)
                if not parts or parts[0] in shadowed_names:
                    return None
                nominal = snapshot.reference_for(parts)
                return None if nominal.root_binding is None else nominal.qualified_name

            try:
                declarations[name] = PYTHON_ENUM_BASE_AUTHORITY.declaration_from_node(
                    node,
                    identity=f"{self.parsed_module.module_name}.{name}",
                    qualified_reference=qualified_reference,
                )
            except ValueError:
                continue
        return declarations

    def _lexical_mapping_key(
        self,
        reference: LexicalValueReference,
        context: ast.Module,
    ) -> Hashable:
        if context is not self.module or len(reference.attribute_path) != 1:
            raise ValueError("Mapping key reference has no exact module declaration")
        declaration = self.enum_declarations_by_name.get(reference.root_name)
        if declaration is None:
            raise ValueError("Mapping key declaration semantics remain unproved")
        return declaration.require_mapping_key(reference.attribute_path[0])


@dataclass(frozen=True)
class SourceClassKeyEntry:
    """One class/key semantic edge recovered from current source."""

    class_node: ast.ClassDef
    key_node: ast.expr
    value_node: ast.expr

    def require_original_value(self, environment: NativeReferenceEnvironment) -> None:
        """Authenticate the original registered operand, not the proposed replacement."""
        self.require_class_value(self.captured_class(environment), environment)

    def require_class_value(
        self,
        value: CapturedReferenceResolution,
        environment: NativeReferenceEnvironment,
    ) -> None:
        """Require this entry's actual class value in the supplied execution.

        A matching source producer in another activation is not the same class.
        This relation does not establish class behavior across rewritten programs.
        """
        expected = environment.capture_definition(self.class_node)
        expected.require_closed()
        value.require_closed()
        if not value.proves_same_object(expected):
            raise ValueError(
                "Registry value is not proved to be the selected class creation"
            )

    def captured_class(
        self,
        environment: NativeReferenceEnvironment,
    ) -> CapturedReferenceResolution:
        return environment.capture(self.value_node)

    @property
    def key_value(self) -> Hashable:
        try:
            return LiteralExpressionEffects(self.key_node).hashable_value
        except (
            ValueError,
            TypeError,
            SyntaxError,
            MemoryError,
            RecursionError,
        ) as error:
            raise ValueError(
                f"Registry key equivalence remains unproved for {self.key_source!r}"
            ) from error

    @property
    def class_name(self) -> str:
        return self.class_node.name

    @property
    def key_source(self) -> str:
        return ast.unparse(self.key_node)

    def require_relocatable_key(self, module: ast.Module) -> None:
        reference_names = _relocatable_key_reference_names(self.key_node)
        if reference_names is None:
            raise ValueError(
                f"Registry key for {self.class_name!r} is not a relocatable "
                "declaration expression"
            )
        preceding_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            statement
            for statement in module.body
            if statement.lineno < self.class_node.lineno
        )
        unresolved_names = reference_names - preceding_names
        if unresolved_names:
            raise ValueError(
                f"Registry key for {self.class_name!r} depends on names not bound "
                f"before its declaration: {tuple(sorted(unresolved_names))!r}"
            )


@dataclass(frozen=True)
class ConstructorClassKeyEntry(SourceClassKeyEntry):
    """An original instance entry consumes a class and executes its constructor."""

    value_node: ast.Call

    def captured_class(
        self,
        environment: NativeReferenceEnvironment,
    ) -> CapturedReferenceResolution:
        return environment.capture(self.value_node.func)

    def require_original_value(self, environment: NativeReferenceEnvironment) -> None:
        super().require_original_value(environment)
        environment.capture_value(self.value_node).require_closed()


class RegistryEntries(ABC):
    """Original declaration/key edges share ordering and native-key obligations."""

    module: ast.Module
    entries: tuple[SourceClassKeyEntry, ...]

    class_names = CollectionAttributeProjection[str]("entries", "class_name")
    class_nodes = CollectionAttributeProjection[ast.ClassDef]("entries", "class_node")

    @cached_property
    def mapping_key_values(self) -> tuple[Hashable, ...]:
        authority = SourceModuleMappingKeyAuthority(self.module)
        return tuple(
            CompactValueExpression.project(entry.key_node).resolve_mapping_key(
                authority,
                self.module,
            )
            for entry in self.entries
        )

    def require_original_entry_values(
        self,
        environment: NativeReferenceEnvironment,
    ) -> None:
        """Prove original operands only; new activation and observation remain separate."""
        if environment.source.module.module is not self.module:
            raise ValueError("Registry entries require their original source module")
        for entry in self.entries:
            entry.require_original_value(environment)

    def require_class_mapping(
        self, contents: NamespaceContentsABC, environment: NativeReferenceEnvironment
    ) -> None:
        """Require complete unordered keys and source class values at this observation."""
        contents.require_closed()
        if (
            contents.kernel is not environment.kernel
            or environment.source.module.module is not self.module
        ):
            raise ValueError("Class mapping requires its original source execution")
        keys = self.mapping_key_values
        if len(frozenset(keys)) != len(keys):
            raise ValueError("Class mapping requires unique declared keys")
        if contents.names != frozenset(keys):
            raise ValueError("Complete keys do not match the selected class entries")
        for key, entry in zip(keys, self.entries, strict=True):
            entry.require_class_value(contents.require_member(key), environment)

    def require_destination_key_compatibility(
        self, authority: AutoRegisterClassAuthority
    ) -> None:
        authority.require_requested_key_compatibility(
            tuple(zip(self.class_names, self.mapping_key_values, strict=True))
        )

    def require_ordered_unique_entries(self) -> None:
        if len(frozenset(self.class_names)) != len(self.class_names):
            raise ValueError("Each registered declaration must have exactly one key")
        declaration_order = tuple(
            node.name for node in sorted(self.class_nodes, key=lambda node: node.lineno)
        )
        if self.class_names != declaration_order:
            raise ValueError("Registry order must match class declaration order")
        for entry in self.entries:
            entry.require_relocatable_key(self.module)
        keys = self.mapping_key_values
        if len(frozenset(keys)) != len(keys):
            raise ValueError("Registry keys must be unique")


@dataclass(frozen=True)
class DirectManualRegistryEntry(SourceClassKeyEntry):
    """One manual class-registration edge recovered directly from source."""

    registry_name: str
    removal_node: ast.stmt


@dataclass(frozen=True)
class DirectManualRegistryComponent(RegistryEntries):
    """Complete direct-registration component anchored by one registered class."""

    module: ast.Module
    registry_assignment: RegistryAssignment
    entries: tuple[DirectManualRegistryEntry, ...]

    @property
    def reuses_registry_binding(self) -> bool:
        authority = self.existing_authority_node
        authority_line = (
            authority.lineno
            if authority is not None
            else min(node.lineno for node in self.class_nodes)
        )
        return (
            self.initializes_empty_registry
            and self.registry_assignment.lineno < authority_line
        )

    @cached_property
    def destination_authority(self) -> AutoRegisterClassAuthority:
        return AutoRegisterClassAuthority.for_registration(
            self.authority_name,
            (
                ast.Name(id=self.registry_name, ctx=ast.Load())
                if self.reuses_registry_binding
                else ast.Dict(keys=[], values=[])
            ),
        )

    @classmethod
    def from_module_anchor(
        cls,
        module: ast.Module,
        anchor_class_name: str,
    ) -> "DirectManualRegistryComponent":
        class_nodes = DirectModuleClassGraph(module).classes_by_name
        entries = cls.direct_entries(module, class_nodes)
        registry_names = frozenset(
            entry.registry_name
            for entry in entries
            if entry.class_name == anchor_class_name
        )
        if len(registry_names) != 1:
            raise ValueError(
                f"Registered class {anchor_class_name!r} must identify exactly one "
                "direct registry component"
            )
        registry_name = next(iter(registry_names))
        component_entries = tuple(
            entry for entry in entries if entry.registry_name == registry_name
        )
        assignments = cls.registry_assignments(module, registry_name)
        if len(assignments) != 1:
            raise ValueError(
                f"Registry {registry_name!r} must have exactly one module assignment"
            )
        component = cls(
            module=module,
            registry_assignment=assignments[0],
            entries=component_entries,
        )
        component.require_complete()
        return component

    @classmethod
    def direct_entries(
        cls,
        module: ast.Module,
        class_nodes: dict[str, ast.ClassDef],
    ) -> tuple[DirectManualRegistryEntry, ...]:
        entries: list[DirectManualRegistryEntry] = []
        for statement in module.body:
            entries.extend(cls.entries_from_dict_assignment(statement, class_nodes))
            entry = cls.entry_from_subscript_assignment(statement, class_nodes)
            if entry is not None:
                entries.append(entry)
        return tuple(entries)

    @staticmethod
    def entries_from_dict_assignment(
        statement: ast.stmt,
        class_nodes: dict[str, ast.ClassDef],
    ) -> tuple[DirectManualRegistryEntry, ...]:
        if not isinstance(statement, RegistryAssignment):
            return ()
        try:
            registry_name = AstExpressionProjection.identifier(
                _assignment_target(statement)
            )
        except ValueError:
            return ()
        value = _assignment_value(statement)
        if registry_name is None or not isinstance(value, ast.Dict) or not value.keys:
            return ()
        entries = []
        for key_node, value_node in zip(value.keys, value.values, strict=True):
            class_node = class_nodes.get(AstExpressionProjection.identifier(value_node))
            if class_node is None or key_node is None:
                return ()
            entries.append(
                DirectManualRegistryEntry(
                    registry_name=registry_name,
                    class_node=class_node,
                    key_node=key_node,
                    value_node=value_node,
                    removal_node=statement,
                )
            )
        return tuple(entries)

    @staticmethod
    def entry_from_subscript_assignment(
        statement: ast.stmt,
        class_nodes: dict[str, ast.ClassDef],
    ) -> DirectManualRegistryEntry | None:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        class_name = AstExpressionProjection.identifier(statement.value)
        if not isinstance(target, ast.Subscript) or class_name not in class_nodes:
            return None
        registry_name = AstExpressionProjection.identifier(target.value)
        if registry_name is None:
            return None
        return DirectManualRegistryEntry(
            registry_name=registry_name,
            class_node=class_nodes[class_name],
            key_node=target.slice,
            value_node=statement.value,
            removal_node=statement,
        )

    @staticmethod
    def registry_assignments(
        module: ast.Module,
        registry_name: str,
    ) -> tuple[RegistryAssignment, ...]:
        assignments = []
        for statement in module.body:
            if not isinstance(statement, RegistryAssignment):
                continue
            try:
                target_name = AstExpressionProjection.identifier(
                    _assignment_target(statement)
                )
            except ValueError:
                continue
            if target_name == registry_name:
                assignments.append(statement)
        return tuple(assignments)

    @property
    def registry_name(self) -> str:
        registry_name = AstExpressionProjection.identifier(
            _assignment_target(self.registry_assignment)
        )
        if registry_name is None:
            raise ValueError("Registry assignment target is not a name")
        return registry_name

    @cached_property
    def class_graph(self) -> DirectModuleClassGraph:
        return DirectModuleClassGraph(self.module)

    @property
    def classes_by_name(self) -> dict[str, ast.ClassDef]:
        return self.class_graph.classes_by_name

    @property
    def registration_statements(self) -> tuple[ast.stmt, ...]:
        statements_by_span = {
            _source_span(entry.removal_node): entry.removal_node
            for entry in self.entries
        }
        return tuple(statements_by_span[span] for span in sorted(statements_by_span))

    @property
    def registry_value(self) -> ast.AST:
        value = _assignment_value(self.registry_assignment)
        if value is None:
            raise ValueError(f"Registry {self.registry_name!r} has no value")
        return value

    @property
    def initializes_empty_registry(self) -> bool:
        value = self.registry_value
        return isinstance(value, ast.Dict) and not value.keys

    @property
    def declares_registry_entries(self) -> bool:
        assignment_span = _source_span(self.registry_assignment)
        return any(
            _source_span(statement) == assignment_span
            for statement in self.registration_statements
        )

    @cached_property
    def existing_authority_node(self) -> ast.ClassDef | None:
        class_nodes = self.classes_by_name
        direct_base_sets = tuple(
            frozenset(
                base.id
                for base in class_node.bases
                if isinstance(base, ast.Name) and base.id in class_nodes
            )
            for class_node in self.class_nodes
        )
        if not direct_base_sets:
            return None
        common_names = set.intersection(*(set(names) for names in direct_base_sets))
        if not common_names:
            return None
        if len(common_names) != 1:
            raise ValueError("Registered classes have multiple shared local bases")
        authority_name = next(iter(common_names))
        authority_node = class_nodes[authority_name]
        descendants = self.class_graph.descendants_of(authority_name)
        registered_names = frozenset(self.class_names)
        unsafe_descendants = tuple(
            node.name
            for node in descendants
            if node.name not in registered_names
            and (
                any(
                    self.class_graph.descends_from(node, registered_name)
                    for registered_name in registered_names
                )
                or _class_declares_non_null_name(
                    node,
                    DEFAULT_REGISTRY_KEY_ATTRIBUTE,
                )
            )
        )
        if unsafe_descendants:
            raise ValueError(
                f"Shared registry base {authority_name!r} has implicitly registered "
                f"descendants {unsafe_descendants!r} outside the registry component"
            )
        return authority_node

    @property
    def generated_authority_name(self) -> str:
        suffix = CLASS_NAME_ALGEBRA.shared_declared_suffix(self.class_names)
        if suffix:
            return f"Registered{suffix}"
        registry_suffix = CLASS_NAME_ALGEBRA.pascal_identifier(
            self.registry_name.lower()
        )
        return (
            f"Registered{registry_suffix}" if registry_suffix else "RegisteredRegistry"
        )

    @property
    def authority_name(self) -> str:
        authority = self.existing_authority_node
        return (
            authority.name if authority is not None else self.generated_authority_name
        )

    def require_complete(self) -> None:
        if len(self.entries) < 2:
            raise ValueError("Manual registry conversion requires at least two classes")
        self.require_ordered_unique_entries()
        if not isinstance(self.registry_value, ast.Dict):
            raise ValueError(
                f"Registry {self.registry_name!r} is not initialized as a dict"
            )
        if self.unsupported_registration_nodes:
            raise ValueError(
                f"Registry {self.registry_name!r} includes behavior-bearing "
                "registration calls or decorators"
            )
        if self.eager_observation_before_registration:
            raise ValueError(
                f"Registry {self.registry_name!r} is observed while its manual "
                "population is still in progress"
            )
        if not self.declares_registry_entries and not self.initializes_empty_registry:
            raise ValueError(
                f"Registry {self.registry_name!r} is neither empty nor a direct class map"
            )
        if self.existing_authority_node is None and any(
            class_node.keywords for class_node in self.class_nodes
        ):
            raise ValueError(
                "Generated registry authority cannot cross class keyword boundaries"
            )
        if self.existing_authority_node is None:
            unsupported_bases = tuple(
                base
                for class_node in self.class_nodes
                for base in class_node.bases
                if not (isinstance(base, ast.Name) and base.id in {"ABC", "object"})
            )
            if unsupported_bases:
                raise ValueError(
                    "Generated registry authority requires empty, object, or ABC "
                    "leaf bases"
                )
            bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.module.body)
            if self.generated_authority_name in bound_names:
                raise ValueError(
                    f"Generated authority name {self.generated_authority_name!r} is bound"
                )

    @cached_property
    def unsupported_registration_nodes(self) -> tuple[ast.AST, ...]:
        nodes: list[ast.AST] = []
        for statement in self.module.body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr in REGISTRATION_CALL_FAMILY.names
                and AstExpressionProjection.identifier(statement.value.func.value)
                == self.registry_name
            ):
                nodes.append(statement)
            if not isinstance(statement, ast.ClassDef):
                continue
            for decorator in statement.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and AstExpressionProjection.terminal_name(decorator.func)
                    in REGISTRATION_DECORATOR_FAMILY.names
                    and decorator.args
                    and AstExpressionProjection.identifier(decorator.args[0])
                    == self.registry_name
                ):
                    nodes.append(decorator)
        return tuple(nodes)

    @cached_property
    def eager_observation_before_registration(self) -> tuple[ast.Name, ...]:
        first_class_line = min(node.lineno for node in self.class_nodes)
        last_registration_line = max(
            statement.end_lineno or statement.lineno
            for statement in self.registration_statements
        )
        registration_lines = frozenset(
            line
            for statement in self.registration_statements
            for line in range(
                statement.lineno,
                (statement.end_lineno or statement.lineno) + 1,
            )
        )
        return tuple(
            node
            for node in EagerNameLoadCollector.collect(self.module, self.registry_name)
            if first_class_line <= node.lineno <= last_registration_line
            and node.lineno not in registration_lines
        )


@dataclass(frozen=True)
class AutoRegisterInstanceViewComponent(RegistryEntries):
    """One constructor-valued view proved against its AutoRegister authority."""

    module: ast.Module
    authority_node: ast.ClassDef
    assignment: RegistryAssignment
    entries: tuple[SourceClassKeyEntry, ...]

    def require_original_entry_values(
        self, environment: NativeReferenceEnvironment
    ) -> None:
        self.authority.require_native_metaclass(environment)
        super().require_original_entry_values(environment)

    @classmethod
    def from_module_authority(
        cls,
        module: ast.Module,
        authority_name: str,
    ) -> "AutoRegisterInstanceViewComponent":
        class_graph = DirectModuleClassGraph(module)
        authority_node = class_graph.classes_by_name.get(authority_name)
        if authority_node is None:
            raise ValueError(
                f"AutoRegister authority {authority_name!r} is not a top-level class"
            )
        candidates = tuple(
            candidate
            for statement in module.body
            if (
                candidate := cls.from_assignment(
                    module,
                    authority_node,
                    statement,
                    class_graph,
                )
            )
            is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                f"AutoRegister authority {authority_name!r} must identify exactly "
                "one constructor-valued instance view"
            )
        component = candidates[0]
        component.require_complete(class_graph)
        return component

    @classmethod
    def from_assignment(
        cls,
        module: ast.Module,
        authority_node: ast.ClassDef,
        statement: ast.stmt,
        class_graph: DirectModuleClassGraph,
    ) -> "AutoRegisterInstanceViewComponent | None":
        if not isinstance(statement, RegistryAssignment):
            return None
        try:
            assignment_name = AstExpressionProjection.identifier(
                _assignment_target(statement)
            )
        except ValueError:
            return None
        value = _assignment_value(statement)
        if (
            assignment_name is None
            or not isinstance(value, ast.Dict)
            or len(value.keys) < 2
            or any(key is None for key in value.keys)
        ):
            return None
        entries = []
        for key, value_node in zip(value.keys, value.values, strict=True):
            if (
                key is None
                or not isinstance(value_node, ast.Call)
                or value_node.args
                or value_node.keywords
                or not isinstance(value_node.func, ast.Name)
            ):
                return None
            class_node = class_graph.classes_by_name.get(value_node.func.id)
            if class_node is None or not class_graph.descends_from(
                class_node,
                authority_node.name,
            ):
                return None
            entries.append(ConstructorClassKeyEntry(class_node, key, value_node))
        return cls(
            module=module,
            authority_node=authority_node,
            assignment=statement,
            entries=tuple(entries),
        )

    @cached_property
    def authority(self) -> AutoRegisterClassAuthority:
        return AutoRegisterClassAuthority(self.authority_node)

    @property
    def authority_name(self) -> str:
        return self.authority_node.name

    @property
    def assignment_name(self) -> str:
        assignment_name = AstExpressionProjection.identifier(
            _assignment_target(self.assignment)
        )
        if assignment_name is None:
            raise ValueError("Instance-view assignment target is not a name")
        return assignment_name

    @property
    def registry_key_attribute(self) -> str:
        registry_key_attribute = self.authority.registry_key_attribute
        if registry_key_attribute is None:
            raise ValueError(
                f"AutoRegister authority {self.authority_name!r} has no registry key"
            )
        return registry_key_attribute

    def require_complete(self, class_graph: DirectModuleClassGraph) -> None:
        if not self.authority.runtime_autoregister_family:
            raise ValueError(
                f"{self.authority_name!r} is not an AutoRegisterMeta family"
            )
        if self.authority.declares_registry:
            registry_value = self.authority.assignment_value(REGISTRY_ATTRIBUTE_NAME)
            if not (
                isinstance(registry_value, ast.Dict)
                and not registry_value.keys
                and not registry_value.values
            ):
                raise ValueError(
                    f"AutoRegister authority {self.authority_name!r} must own an "
                    "empty direct registry"
                )
        self.require_ordered_unique_entries()
        assignment_line = self.assignment.lineno
        if any(
            (node.end_lineno or node.lineno) >= assignment_line
            for node in self.class_nodes
        ):
            raise ValueError(
                "Instance-view assignment must follow every constructed class"
            )
        registered_names = frozenset(self.class_names)
        unsafe_descendants = tuple(
            node.name
            for node in class_graph.descendants_of(self.authority_name)
            if node.lineno < assignment_line
            and node.name not in registered_names
            and (
                any(
                    class_graph.descends_from(node, registered_name)
                    for registered_name in registered_names
                )
                or _class_declares_non_null_name(
                    node,
                    self.registry_key_attribute,
                )
            )
        )
        if unsafe_descendants:
            raise ValueError(
                f"Instance view omits registered descendants {unsafe_descendants!r}"
            )


def _source_span(node: ast.AST) -> tuple[int, int]:
    return node.lineno, node.end_lineno or node.lineno


def _relocatable_key_reference_names(node: ast.expr) -> frozenset[str] | None:
    try:
        LiteralExpressionEffects(node).hashable_value
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    else:
        return frozenset()
    if isinstance(node, ast.Name):
        return frozenset((node.id,))
    if isinstance(node, ast.Attribute):
        return _relocatable_key_reference_names(node.value)
    if isinstance(node, ast.Tuple):
        child_names = tuple(
            _relocatable_key_reference_names(element) for element in node.elts
        )
        if any(names is None for names in child_names):
            return None
        return frozenset(
            name for names in child_names if names is not None for name in names
        )
    return None


def _class_declares_non_null_name(node: ast.ClassDef, name: str) -> bool:
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            if not any(
                AstExpressionProjection.identifier(target) == name
                for target in statement.targets
            ):
                continue
            value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and AstExpressionProjection.identifier(statement.target) == name
        ):
            value = statement.value
        else:
            continue
        return not (isinstance(value, ast.Constant) and value.value is None)
    return False
