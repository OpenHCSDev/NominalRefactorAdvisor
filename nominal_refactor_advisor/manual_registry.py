"""Source-derived proof model for direct manual class registries."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias

from .ast_tools import (
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    REGISTRATION_CALL_FAMILY,
    REGISTRATION_DECORATOR_FAMILY,
)
from .collection_algebra import UniqueIdentityIndexAuthority
from .descriptor_algebra import CollectionAttributeProjection
from .name_algebra import CLASS_NAME_ALGEBRA
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE

RegistryAssignment: TypeAlias = ast.Assign | ast.AnnAssign


def _name(node: ast.AST | None) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _assignment_target(statement: RegistryAssignment) -> ast.expr:
    if isinstance(statement, ast.AnnAssign):
        return statement.target
    if len(statement.targets) != 1:
        raise ValueError("Registry assignment requires exactly one target")
    return statement.targets[0]


def _assignment_value(statement: RegistryAssignment) -> ast.AST | None:
    return statement.value


@dataclass(frozen=True)
class DirectManualRegistryEntry:
    """One class/key edge recovered directly from Python syntax."""

    registry_name: str
    class_node: ast.ClassDef
    key_node: ast.expr
    removal_node: ast.stmt

    @property
    def class_name(self) -> str:
        return self.class_node.name

    @property
    def key_source(self) -> str:
        return ast.unparse(self.key_node)

    @property
    def key_identity(self) -> tuple[str, object]:
        try:
            value = ast.literal_eval(self.key_node)
            hash(value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return "syntax", ast.dump(self.key_node, include_attributes=False)
        return "literal", value


@dataclass(frozen=True)
class DirectManualRegistryComponent:
    """Complete direct-registration component anchored by one registered class."""

    module: ast.Module
    registry_assignment: RegistryAssignment
    entries: tuple[DirectManualRegistryEntry, ...]

    @classmethod
    def from_module_anchor(
        cls,
        module: ast.Module,
        anchor_class_name: str,
    ) -> "DirectManualRegistryComponent":
        class_nodes = cls.top_level_classes(module)
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

    @staticmethod
    def top_level_classes(module: ast.Module) -> dict[str, ast.ClassDef]:
        class_nodes = tuple(
            statement
            for statement in module.body
            if isinstance(statement, ast.ClassDef)
        )
        try:
            return UniqueIdentityIndexAuthority.declarations_by_handle(
                class_nodes,
                lambda node: node.name,
            )
        except ValueError as error:
            raise ValueError(
                "Manual registry classes require unique module names"
            ) from error

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
            registry_name = _name(_assignment_target(statement))
        except ValueError:
            return ()
        value = _assignment_value(statement)
        if registry_name is None or not isinstance(value, ast.Dict) or not value.keys:
            return ()
        if any(key is None for key in value.keys):
            return ()
        value_names = tuple(_name(item) for item in value.values)
        if any(name not in class_nodes for name in value_names):
            return ()
        return tuple(
            DirectManualRegistryEntry(
                registry_name=registry_name,
                class_node=class_nodes[class_name],
                key_node=key,
                removal_node=statement,
            )
            for key, class_name in zip(value.keys, value_names, strict=True)
            if key is not None and class_name is not None
        )

    @staticmethod
    def entry_from_subscript_assignment(
        statement: ast.stmt,
        class_nodes: dict[str, ast.ClassDef],
    ) -> DirectManualRegistryEntry | None:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        class_name = _name(statement.value)
        if not isinstance(target, ast.Subscript) or class_name not in class_nodes:
            return None
        registry_name = _name(target.value)
        if registry_name is None:
            return None
        return DirectManualRegistryEntry(
            registry_name=registry_name,
            class_node=class_nodes[class_name],
            key_node=target.slice,
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
                target_name = _name(_assignment_target(statement))
            except ValueError:
                continue
            if target_name == registry_name:
                assignments.append(statement)
        return tuple(assignments)

    @property
    def registry_name(self) -> str:
        registry_name = _name(_assignment_target(self.registry_assignment))
        if registry_name is None:
            raise ValueError("Registry assignment target is not a name")
        return registry_name

    class_names = CollectionAttributeProjection[str]("entries", "class_name")
    class_nodes = CollectionAttributeProjection[ast.ClassDef]("entries", "class_node")

    @cached_property
    def classes_by_name(self) -> dict[str, ast.ClassDef]:
        return self.top_level_classes(self.module)

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
        descendants = tuple(
            node
            for node in class_nodes.values()
            if self.descends_from(node, authority_name, class_nodes)
        )
        registered_names = frozenset(self.class_names)
        unsafe_descendants = tuple(
            node.name
            for node in descendants
            if node.name not in registered_names
            and (
                any(
                    self.descends_from(node, registered_name, class_nodes)
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

    @staticmethod
    def descends_from(
        node: ast.ClassDef,
        ancestor_name: str,
        class_nodes: dict[str, ast.ClassDef],
    ) -> bool:
        pending = [base.id for base in node.bases if isinstance(base, ast.Name)]
        visited: set[str] = set()
        while pending:
            base_name = pending.pop()
            if base_name == ancestor_name:
                return True
            if base_name in visited:
                continue
            visited.add(base_name)
            base_node = class_nodes.get(base_name)
            if base_node is not None:
                pending.extend(
                    base.id for base in base_node.bases if isinstance(base, ast.Name)
                )
        return False

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
        if len(frozenset(self.class_names)) != len(self.class_names):
            raise ValueError("Each registered class must have exactly one registry key")
        class_definition_order = tuple(
            node.name for node in sorted(self.class_nodes, key=lambda node: node.lineno)
        )
        if self.class_names != class_definition_order:
            raise ValueError("Manual registry order must match class declaration order")
        key_identities = tuple(entry.key_identity for entry in self.entries)
        if any(
            left == right
            for index, left in enumerate(key_identities)
            for right in key_identities[index + 1 :]
        ):
            raise ValueError("Manual registry keys must be unique")
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
                and _name(statement.value.func.value) == self.registry_name
            ):
                nodes.append(statement)
            if not isinstance(statement, ast.ClassDef):
                continue
            for decorator in statement.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and _terminal_name(decorator.func)
                    in REGISTRATION_DECORATOR_FAMILY.names
                    and decorator.args
                    and _name(decorator.args[0]) == self.registry_name
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


class EagerNameLoadCollector(ast.NodeVisitor):
    """Collect module-executed name loads without descending into call bodies."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.loads: list[ast.Name] = []

    @classmethod
    def collect(cls, module: ast.Module, name: str) -> tuple[ast.Name, ...]:
        collector = cls(name)
        collector.visit(module)
        return tuple(collector.loads)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            self.loads.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.visit_function_header(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_function_header(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit_arguments(node.args)

    def visit_function_header(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit_arguments(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for type_parameter in getattr(node, "type_params", ()):
            self.visit(type_parameter)

    def visit_arguments(self, arguments: ast.arguments) -> None:
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if arguments.vararg is not None and arguments.vararg.annotation is not None:
            self.visit(arguments.vararg.annotation)
        if arguments.kwarg is not None and arguments.kwarg.annotation is not None:
            self.visit(arguments.kwarg.annotation)
        for default in (*arguments.defaults, *arguments.kw_defaults):
            if default is not None:
                self.visit(default)


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _source_span(node: ast.AST) -> tuple[int, int]:
    return node.lineno, node.end_lineno or node.lineno


def _class_declares_non_null_name(node: ast.ClassDef, name: str) -> bool:
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            if not any(_name(target) == name for target in statement.targets):
                continue
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and _name(statement.target) == name:
            value = statement.value
        else:
            continue
        return not (isinstance(value, ast.Constant) and value.value is None)
    return False
