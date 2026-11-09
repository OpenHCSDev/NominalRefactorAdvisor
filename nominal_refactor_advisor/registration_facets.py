"""How a class takes part in a registry: its registry key, manual subclass rosters and selector axes."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_symbols import (
    _terminal_reference_name,
    iter_class_definitions,
)
from nominal_refactor_advisor.class_syntax_facets import (
    CompactSourceLocation,
    _trim_leading_docstring,
)
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.enum_semantics import PYTHON_ENUM_BASE_AUTHORITY
from nominal_refactor_advisor.foreign_family import FunctionDefinition
from nominal_refactor_advisor.registry_identity import (
    REGISTRY_ATTRIBUTE_NAME,
    REGISTRY_KEY_ATTRIBUTE_NAME,
)
from nominal_refactor_advisor.value_expression import CompactValueExpression


@dataclass(frozen=True)
class CompactManualSubclassRegistrationSite:
    registry_name: str
    guard_summary: str | None
    selector_attr_name: str | None
    requires_concrete_subclass: bool


@dataclass(frozen=True)
class CompactManualSubclassRosterRoot:
    class_symbol: str
    init_subclass_line: int
    registration_sites: tuple[CompactManualSubclassRegistrationSite, ...]
    consumer_locations: tuple[tuple[str, int, str, str], ...]


class SelectionGuardKind(StrEnum):
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"
    NOT_EXACTLY_ONE = "not_exactly_one"

    @classmethod
    def proves_exact_selection(
        cls,
        guard_kinds: set[SelectionGuardKind | None],
    ) -> bool:
        """Return whether the observed guards prove exactly one selected item."""

        return (
            cls.NOT_EXACTLY_ONE in guard_kinds
            or {
                cls.EMPTY,
                cls.AMBIGUOUS,
            }
            <= guard_kinds
        )

    @classmethod
    def from_node(
        cls,
        node: ast.AST,
        match_name: str,
    ) -> SelectionGuardKind | None:
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Name)
            and node.operand.id == match_name
        ):
            return cls.EMPTY
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "len"
            and len(node.left.args) == 1
            and isinstance(node.left.args[0], ast.Name)
            and node.left.args[0].id == match_name
            and isinstance(node.comparators[0], ast.Constant)
            and isinstance(node.comparators[0].value, int)
        ):
            return None
        operator = node.ops[0]
        comparator = node.comparators[0].value
        if isinstance(operator, ast.NotEq) and comparator == 1:
            return cls.NOT_EXACTLY_ONE
        if isinstance(operator, ast.Gt) and comparator == 1:
            return cls.AMBIGUOUS
        if isinstance(operator, ast.Eq) and comparator == 0:
            return cls.EMPTY
        return None


@dataclass(frozen=True)
class CompactManualSelectorAxis(CompactSourceLocation):
    family_name: str
    selector_method_name: str
    key_type_name: str
    case_names: tuple[str, ...]


def _compact_predicate_selected_methods(
    node: ast.ClassDef,
) -> tuple[tuple[int, str, str, str], ...]:
    """Project exact registered-types selector shapes without retaining method ASTs."""

    methods: list[tuple[int, str, str, str]] = []
    for statement in FunctionDefinition.within(node.body):
        if not any(
            _terminal_reference_name(decorator) == "classmethod"
            for decorator in statement.decorator_list
        ):
            continue
        shape = _compact_registered_type_match_shape(statement)
        if shape is None:
            continue
        match_name, predicate_name, context_name = shape
        guard_kinds = {
            SelectionGuardKind.from_node(candidate.test, match_name)
            for candidate in _trim_leading_docstring(list(statement.body))
            if isinstance(candidate, ast.If)
        }
        if not SelectionGuardKind.proves_exact_selection(guard_kinds):
            continue
        if not any(
            isinstance(candidate, ast.Subscript)
            and isinstance(candidate.value, ast.Name)
            and candidate.value.id == match_name
            and isinstance(candidate.slice, ast.Constant)
            and candidate.slice.value == 0
            for candidate in ast.walk(statement)
        ):
            continue
        methods.append((statement.lineno, statement.name, predicate_name, context_name))
    return tuple(methods)


def _compact_registered_type_match_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str] | None:
    parameter_names = {
        argument.arg
        for argument in (
            *method.args.posonlyargs,
            *method.args.args,
            *method.args.kwonlyargs,
        )
        if argument.arg not in {"self", "cls"}
    }
    for statement in _trim_leading_docstring(list(method.body)):
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.ListComp)
            and len(statement.value.generators) == 1
        ):
            continue
        match_name = statement.targets[0].id
        comprehension = statement.value
        generator = comprehension.generators[0]
        if not (
            not generator.is_async
            and isinstance(generator.target, ast.Name)
            and isinstance(comprehension.elt, ast.Name)
            and comprehension.elt.id == generator.target.id
            and isinstance(generator.iter, ast.Call)
            and not generator.iter.args
            and not generator.iter.keywords
            and isinstance(generator.iter.func, ast.Attribute)
            and generator.iter.func.attr == "registered_types"
            and isinstance(generator.iter.func.value, ast.Name)
            and generator.iter.func.value.id == "cls"
            and len(generator.ifs) == 1
            and isinstance(generator.ifs[0], ast.Call)
        ):
            continue
        predicate = generator.ifs[0]
        if not (
            not predicate.keywords
            and len(predicate.args) == 1
            and isinstance(predicate.args[0], ast.Name)
            and predicate.args[0].id in parameter_names
            and isinstance(predicate.func, ast.Attribute)
            and predicate.func.attr
            and isinstance(predicate.func.value, ast.Name)
            and predicate.func.value.id == generator.target.id
        ):
            continue
        return match_name, predicate.func.attr, predicate.args[0].id
    return None


def _compact_manual_subclass_roster_roots(
    parsed_module: ParsedModule,
) -> tuple[CompactManualSubclassRosterRoot, ...]:
    roots: list[CompactManualSubclassRosterRoot] = []
    file_path = parsed_module.file_path
    for qualname, node in iter_class_definitions(list(parsed_module.module.body)):
        registry_names = _compact_class_list_registry_names(node)
        if not registry_names:
            continue
        init_subclass = next(
            (
                statement
                for statement in node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == "__init_subclass__"
            ),
            None,
        )
        if init_subclass is None:
            continue
        sites = _compact_manual_subclass_registration_sites(
            init_subclass, registry_names, owner_name=node.name
        )
        if not sites:
            continue
        consumers: list[tuple[str, int, str, str]] = []
        for registry_name in registry_names:
            for method in FunctionDefinition.within(node.body):
                if method.name == "__init_subclass__":
                    continue
                if _compact_uses_named_registry(
                    method,
                    registry_name=registry_name,
                    owner_names=frozenset({"cls", "type", node.name}),
                ):
                    consumers.append(
                        (
                            registry_name,
                            method.lineno,
                            f"{node.name}.{method.name}",
                            file_path,
                        )
                    )
            for function_qualname, function in parsed_module.syntax_index.named_functions:
                if "." in function_qualname:
                    continue
                if _compact_uses_named_registry(
                    function,
                    registry_name=registry_name,
                    owner_names=frozenset({node.name}),
                ):
                    consumers.append(
                        (
                            registry_name,
                            function.lineno,
                            function_qualname,
                            file_path,
                        )
                    )
        roots.append(
            CompactManualSubclassRosterRoot(
                class_symbol=f"{parsed_module.module_name}.{qualname}",
                init_subclass_line=init_subclass.lineno,
                registration_sites=sites,
                consumer_locations=sorted_tuple(
                    set(consumers), key=lambda item: (item[1], item[2], item[0])
                ),
            )
        )
    return tuple(roots)


def _compact_class_list_registry_names(node: ast.ClassDef) -> tuple[str, ...]:
    return sorted_tuple(
        {
            target_name
            for statement in node.body
            for target_name in (
                (
                    statement.targets[0].id
                    if isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and isinstance(statement.value, ast.List)
                    else (
                        statement.target.id
                        if isinstance(statement, ast.AnnAssign)
                        and isinstance(statement.target, ast.Name)
                        and isinstance(statement.value, ast.List)
                        else None
                    )
                ),
            )
            if target_name is not None
        }
    )


def _compact_manual_subclass_registration_sites(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    registry_names: tuple[str, ...],
    *,
    owner_name: str,
) -> tuple[CompactManualSubclassRegistrationSite, ...]:
    sites: dict[str, CompactManualSubclassRegistrationSite] = {}

    def walk_statements(
        statements: list[ast.stmt], guard_stack: tuple[ast.AST, ...]
    ) -> None:
        for statement in statements:
            if isinstance(statement, ast.If):
                walk_statements(statement.body, (*guard_stack, statement.test))
                walk_statements(statement.orelse, guard_stack)
                continue
            for subnode in ast.walk(statement):
                registry_name = _compact_registration_append_registry_name(
                    subnode, registry_names, owner_name
                )
                if registry_name is None:
                    continue
                sites[registry_name] = CompactManualSubclassRegistrationSite(
                    registry_name=registry_name,
                    guard_summary=(
                        " and ".join(ast.unparse(guard) for guard in guard_stack)
                        if guard_stack
                        else None
                    ),
                    selector_attr_name=next(
                        (
                            attr_name
                            for guard in guard_stack
                            if (attr_name := _compact_guarded_defined_attr_name(guard))
                            is not None
                        ),
                        None,
                    ),
                    requires_concrete_subclass=any(
                        _compact_guard_requires_concrete_subclass(guard)
                        for guard in guard_stack
                    ),
                )

    walk_statements(_trim_leading_docstring(list(method.body)), ())
    return tuple(sites[name] for name in sorted(sites))


def _compact_registration_append_registry_name(
    node: ast.AST, registry_names: tuple[str, ...], owner_name: str
) -> str | None:
    if not (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr in registry_names
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id in {"cls", "type", owner_name}
        and _compact_looks_like_cls_registration_value(node.args[0])
    ):
        return None
    return node.func.value.attr


def _compact_looks_like_cls_registration_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "cls"
    return bool(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cast"
        and node.args
        and _compact_looks_like_cls_registration_value(node.args[-1])
    )


def _compact_class_dict_get_attr_name(node: ast.AST) -> str | None:
    if not (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "__dict__"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "cls"
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return None
    return node.args[0].value


def _compact_guarded_defined_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _compact_class_dict_get_attr_name(node)
    if not (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.IsNot, ast.NotEq))
        and len(node.comparators) == 1
        and CompactValueExpression.project(node.comparators[0]).value_is_none_literal
    ):
        return None
    return _compact_class_dict_get_attr_name(node.left)


def _compact_guard_requires_concrete_subclass(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Call)
        and isinstance(node.operand.func, ast.Attribute)
        and isinstance(node.operand.func.value, ast.Name)
        and node.operand.func.value.id == "inspect"
        and node.operand.func.attr == "isabstract"
        and len(node.operand.args) == 1
        and isinstance(node.operand.args[0], ast.Name)
        and node.operand.args[0].id == "cls"
    )


def _compact_uses_named_registry(
    node: ast.AST,
    *,
    registry_name: str,
    owner_names: frozenset[str],
) -> bool:
    return any(
        isinstance(subnode, ast.Attribute)
        and subnode.attr == registry_name
        and isinstance(subnode.value, ast.Name)
        and subnode.value.id in owner_names
        for subnode in ast.walk(node)
    )


def _compact_manual_selector_axes(
    parsed_module: ParsedModule,
) -> tuple[CompactManualSelectorAxis, ...]:
    dict_literals: dict[str, ast.Dict] = {}
    for statement in parsed_module.module.body:
        name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            if isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            name = statement.target.id
            value = statement.value
        if name is not None and isinstance(value, ast.Dict):
            dict_literals[name] = value
    case_names_by_mapping = {
        name: tuple(ast.unparse(key) for key in mapping.keys if key is not None)
        for name, mapping in dict_literals.items()
    }
    known_mapping_names = frozenset(
        name
        for name, case_names in case_names_by_mapping.items()
        if len(case_names) >= 2
    )
    axes: list[CompactManualSelectorAxis] = []
    for node in parsed_module.syntax_index.breadth_first_nodes:
        if not isinstance(node, ast.ClassDef):
            continue
        for method in FunctionDefinition.within(node.body):
            if not method.name.startswith("for_") or not any(
                _terminal_reference_name(decorator) == "classmethod"
                for decorator in method.decorator_list
            ):
                continue
            parameter_names = {
                item.arg
                for item in (
                    *method.args.posonlyargs,
                    *method.args.args,
                    *method.args.kwonlyargs,
                )
                if item.arg not in {"self", "cls"}
            }
            if not parameter_names:
                continue
            mapping_name: str | None = None
            for subnode in ast.walk(method):
                if not (
                    isinstance(subnode, ast.Subscript)
                    and isinstance(subnode.value, ast.Name)
                    and subnode.value.id in known_mapping_names
                    and ast.unparse(subnode.slice) in parameter_names
                ):
                    continue
                mapping_name = subnode.value.id
                break
            if mapping_name is None:
                continue
            case_names = case_names_by_mapping[mapping_name]
            key_type_names = {
                case_name.split(".", 1)[0]
                for case_name in case_names
                if "." in case_name
            }
            if len(key_type_names) != 1:
                continue
            axes.append(
                CompactManualSelectorAxis(
                    file_path=parsed_module.file_path,
                    line=method.lineno,
                    family_name=node.name,
                    selector_method_name=method.name,
                    key_type_name=next(iter(key_type_names)),
                    case_names=case_names,
                )
            )
    return tuple(axes)


def _direct_class_assignments(node: ast.ClassDef) -> dict[str, ast.AST | None]:
    assignments: dict[str, ast.AST | None] = {}
    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                assignments[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target,
            ast.Name,
        ):
            assignments[statement.target.id] = statement.value
    return assignments


def _string_constant(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _module_string_constant_assignments(
    parsed_module: ParsedModule,
) -> dict[str, str]:
    constants: dict[str, str] = {}
    for statement in parsed_module.module.body:
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is not None and (string_value := _string_constant(value)):
            constants[target_name] = string_value
    return constants


def _class_direct_string_member_assignments(
    node: ast.ClassDef,
) -> dict[str, str]:
    return {
        name: string_value
        for name, value in _direct_class_assignments(node).items()
        if (string_value := _string_constant(value)) is not None
    }


def _module_string_enum_member_assignments(
    parsed_module: ParsedModule,
) -> dict[tuple[str, str], str]:
    members: dict[tuple[str, str], str] = {}
    for statement in parsed_module.module.body:
        if not isinstance(statement, ast.ClassDef):
            continue
        if not PYTHON_ENUM_BASE_AUTHORITY.matches_any(
            _terminal_reference_name(base) for base in statement.bases
        ):
            continue
        for member_name, string_value in _class_direct_string_member_assignments(
            statement
        ).items():
            members[(statement.name, member_name)] = string_value
    return members


def _enum_member_value_reference(node: ast.AST | None) -> tuple[str, str] | None:
    if not (
        isinstance(node, ast.Attribute)
        and node.attr == "value"
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
    ):
        return None
    return node.value.value.id, node.value.attr


def _autoregister_constant_name(
    node: ast.AST | None,
    parsed_module: ParsedModule,
) -> str | None:
    if (string_value := _string_constant(node)) is not None:
        return string_value
    if (enum_ref := _enum_member_value_reference(node)) is not None:
        return _module_string_enum_member_assignments(parsed_module).get(enum_ref)
    if isinstance(node, ast.Name):
        return (
            _module_string_constant_assignments(parsed_module).get(node.id) or node.id
        )
    if isinstance(node, ast.Attribute) and node.attr == "__registry_key__":
        return node.attr
    return None


def _registry_family_key_attr_name(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if _terminal_reference_name(node.func) != "RegistryFamily" or not node.args:
        return None
    key_arg = node.args[0]
    if (key_literal := _string_constant(key_arg)) is not None:
        return key_literal
    if isinstance(key_arg, ast.Attribute):
        return key_arg.attr.lower()
    return None


def _registry_config_keyword(
    node: ast.AST | None,
    keyword_name: str,
) -> ast.AST | None:
    if not isinstance(node, ast.Call):
        return None
    if _terminal_reference_name(node.func) != "RegistryConfig":
        return None
    values = tuple(
        keyword.value for keyword in node.keywords if keyword.arg == keyword_name
    )
    return values[0] if len(values) == 1 else None


class FamilyWalkRoot(ABC):
    """Where a registry family finds its members: it imports the modules of one
    package, its whole tree when the walk is recursive, its direct modules when flat.
    A member moved where the walk does not reach registers only if something else
    imports it, so a move refuses to take one there."""

    @abstractmethod
    def leaving_reason(self, source: str, destination: str) -> str | None:
        """Why moving a member from module `source` to `destination` takes it out
        of this walk, or None when the walk still reaches it."""


@dataclass(frozen=True)
class NoWalkRoot(FamilyWalkRoot):
    """A class that declares no walk: its members register when imported."""

    def leaving_reason(self, source: str, destination: str) -> str | None:
        return None


@dataclass(frozen=True)
class PackageWalkRoot(FamilyWalkRoot):
    package: str
    recursive: bool

    def walks(self, module: str) -> bool:
        if module == self.package:
            return True
        prefix = f"{self.package}."
        if not module.startswith(prefix):
            return False
        return self.recursive or "." not in module.removeprefix(prefix)

    def leaving_reason(self, source: str, destination: str) -> str | None:
        if not self.walks(source) or self.walks(destination):
            return None
        reach = "tree" if self.recursive else "direct modules (the walk is flat)"
        return f"it walks {self.package}'s {reach}, and {destination} is outside it"


@dataclass(frozen=True)
class UnresolvedWalkRoot(FamilyWalkRoot):
    """A walk whose package is not a static value: no move can show it still
    reaches the member, so every member move refuses with the expression named."""

    expression: str

    def leaving_reason(self, source: str, destination: str) -> str | None:
        return f"it walks the package `{self.expression}`, which is not a static value"


def static_module_name(node: ast.AST | None, parsed_module: ParsedModule) -> str | None:
    """A string the module spells statically: a literal, `__package__`, `__name__`,
    or an f-string or `+` of those."""

    identity = parsed_module.module_path_identity
    match node:
        case ast.Constant(value=str(value)):
            return value
        case ast.Name(id="__package__"):
            return identity.package_name
        case ast.Name(id="__name__"):
            return identity.import_name
        case ast.FormattedValue(value=value, conversion=-1, format_spec=None):
            return static_module_name(value, parsed_module)
        case ast.JoinedStr(values=values):
            parts = [static_module_name(value, parsed_module) for value in values]
            return None if None in parts else "".join(part for part in parts if part is not None)
        case ast.BinOp(left=left, op=ast.Add(), right=right):
            pair = (static_module_name(left, parsed_module), static_module_name(right, parsed_module))
            return None if None in pair else f"{pair[0]}{pair[1]}"
    return None


def registry_config_walk_root(parsed_module: ParsedModule, node: ast.ClassDef) -> FamilyWalkRoot:
    """The walk a class's `__registry_config__ = RegistryConfig(discovery_package=...)`
    declares; flat unless `discovery_recursive=True`."""

    config = _direct_class_assignments(node).get("__registry_config__")
    package = _registry_config_keyword(config, "discovery_package")
    if package is None:
        return NoWalkRoot()
    recursive = _registry_config_keyword(config, "discovery_recursive")
    name = static_module_name(package, parsed_module)
    match recursive:
        case None | ast.Constant(value=bool()) if name is not None:
            return PackageWalkRoot(
                package=name,
                recursive=isinstance(recursive, ast.Constant) and recursive.value is True,
            )
    return UnresolvedWalkRoot(ast.unparse(package if name is None else recursive))


def _autoregister_registry_key_attr_name(
    parsed_module: ParsedModule,
    node: ast.ClassDef,
) -> str | None:
    assignments = _direct_class_assignments(node)
    explicit_key = _autoregister_constant_name(
        assignments.get("__registry_key__"), parsed_module
    )
    if explicit_key is not None:
        return explicit_key
    configured_key = _autoregister_constant_name(
        _registry_config_keyword(
            assignments.get("__registry_config__"),
            "key_attribute",
        ),
        parsed_module,
    )
    if configured_key is not None:
        return configured_key
    stable_key_axis = assignments.get("stable_key_axis")
    if (
        isinstance(stable_key_axis, ast.Name)
        and stable_key_axis.id == "__registry_key__"
    ):
        return stable_key_axis.id
    stable_key_name = _autoregister_constant_name(stable_key_axis, parsed_module)
    if stable_key_name is not None:
        return stable_key_name
    return _registry_family_key_attr_name(assignments.get("__registry_family__"))


def _is_registration_authority(node: ast.ClassDef) -> bool:
    """Whether the class declares the registry protocol itself: a registry and its key attribute.

    A registry family is an authority too: a registering metaclass creates
    it, or it or an ancestor has an `__init_subclass__` that stores the
    subclass. The index adds that from what the class does
    (`RegisteringMetaclassAuthority.is_registry_family`,
    `with_registering_metaclass`). A class's name is no evidence either way.
    """

    assignments = _direct_class_assignments(node)
    return (
        REGISTRY_ATTRIBUTE_NAME in assignments
        and REGISTRY_KEY_ATTRIBUTE_NAME in assignments
    )
