"""Shared detector helper functions.

This module contains private analysis helpers that support detector families
across the split implementation modules.
"""

from __future__ import annotations

from ..record_algebra import (
    materialize_product_record,
    materialize_product_records,
    product_record_spec,
)
from ..semantic_algebra import ObjectFamilyShape
from ..semantic_description_length import CompressionCertificate

import io
import re
import tokenize

from ._base import *
from ._substrate_support import *


class _BuiltinCollectionName(StrEnum):
    TUPLE = "tuple"
    LIST = "list"
    SET = "set"
    DICT = "dict"


_SEQUENCE_WRAPPER_CALL_NAMES = frozenset(
    {
        _BuiltinCollectionName.TUPLE,
        _BuiltinCollectionName.LIST,
        _BuiltinCollectionName.SET,
    }
)
_RETURN_COLLECTION_KIND_NAMES = frozenset(
    {
        _BuiltinCollectionName.TUPLE,
        _BuiltinCollectionName.LIST,
        _BuiltinCollectionName.DICT,
    }
)


def _semantic_dict_bag_candidates(
    module: ParsedModule,
) -> list[SemanticDictBagCandidate]:
    candidates: list[SemanticDictBagCandidate] = []

    class Visitor(ClassFunctionStackNodeVisitor):
        traverse_class_body = ClassFunctionStackNodeVisitor.traverse_trimmed_node_body

        def before_visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            candidates.extend(
                _function_local_semantic_dict_bag_candidates(
                    module, node, tuple(self.class_stack)
                )
            )

        traverse_function_body = (
            ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
        )

        def visit_Call(self, node: ast.Call) -> None:
            for keyword in node.keywords:
                if keyword.arg != "metrics" or not isinstance(keyword.value, ast.Dict):
                    continue
                items = _string_dict_items(keyword.value)
                if items is None:
                    continue
                owner_symbol = _owner_symbol(
                    tuple(self.class_stack), tuple(self.function_stack), "metrics"
                )
                recommendation = _recommend_metrics_dataclass(
                    items, owner_symbol=owner_symbol
                )
                candidates.append(
                    SemanticDictBagCandidate(
                        line=keyword.value.lineno,
                        symbol=owner_symbol,
                        key_names=tuple(items),
                        context_kind="metrics_keyword",
                        recommendation=recommendation,
                    )
                )
            self.generic_visit(node)

    Visitor().visit(module.module)
    return candidates


def _function_local_semantic_dict_bag_candidates(
    module: ParsedModule,
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    class_stack: tuple[str, ...],
) -> list[SemanticDictBagCandidate]:
    assignments: dict[str, tuple[int, dict[str, ast.AST]]] = {}
    accessed_keys: dict[str, set[str]] = defaultdict(set)
    serialization_boundary_names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.target_node = function_node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is self.target_node:
                for stmt in _trim_docstring_body(node.body):
                    self.visit(stmt)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return None

        def visit_Assign(self, node: ast.Assign) -> None:
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                items = _string_dict_items(node.value)
                if items is not None:
                    assignments[node.targets[0].id] = (node.lineno, items)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if (
                isinstance(node.target, ast.Name)
                and node.value is not None
                and ((items := _string_dict_items(node.value)) is not None)
            ):
                assignments[node.target.id] = (node.lineno, items)
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name):
                key_name = _string_slice_name(node.slice)
                if key_name is not None:
                    accessed_keys[node.value.id].add(key_name)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if _is_json_boundary_call(node):
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        serialization_boundary_names.add(arg.id)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.attr in {"get", "pop", "setdefault"}
                and node.args
            ):
                key_name = _constant_string(node.args[0])
                if key_name is not None:
                    accessed_keys[node.func.value.id].add(key_name)
            self.generic_visit(node)

        def visit_Return(self, node: ast.Return) -> None:
            if self.target_node.name == "to_dict" and isinstance(node.value, ast.Name):
                serialization_boundary_names.add(node.value.id)
            self.generic_visit(node)

    Visitor().visit(function_node)

    candidates: list[SemanticDictBagCandidate] = []
    owner_symbol = _owner_symbol(class_stack, (function_node.name,), "record")
    for name, (lineno, items) in assignments.items():
        if name in serialization_boundary_names:
            continue
        touched_keys = set(items) | accessed_keys.get(name, set())
        if not touched_keys:
            continue
        recommendation = _recommend_local_semantic_record(
            sorted_tuple(touched_keys),
            owner_symbol=owner_symbol,
            variable_name=name,
            value_nodes=items,
        )
        if recommendation is None:
            continue
        candidates.append(
            SemanticDictBagCandidate(
                line=lineno,
                symbol=f"{owner_symbol}:{name}",
                key_names=sorted_tuple(touched_keys),
                context_kind="local_string_key_bag",
                recommendation=recommendation,
            )
        )
    return candidates


def _recommend_metrics_dataclass(
    items: dict[str, ast.AST], owner_symbol: str
) -> SemanticDataclassRecommendation:
    key_names = sorted_tuple(items)
    exact_schema = _exact_schema_match(key_names, _METRIC_BAG_SCHEMAS)
    if exact_schema is not None:
        class_name = exact_schema.class_name
        base_class_name = exact_schema.base_class_name
        rationale = f"Use existing `{class_name}`, which already inherits `{base_class_name}` for this semantic field family."
        scaffold = _instantiation_scaffold(
            class_name, key_names, items, prefix="metrics="
        )
        return SemanticDataclassRecommendation.existing_schema(
            class_name, base_class_name, rationale, scaffold
        )

    closest_schema = _closest_schema_match(key_names, _METRIC_BAG_SCHEMAS)
    base_class_name = (
        closest_schema.base_class_name
        if closest_schema is not None
        else FindingMetrics.__name__
    )
    class_name = _suggest_dataclass_name(owner_symbol, "Metrics")
    rationale = (
        f"Create `{class_name}` inheriting from `{base_class_name}` because this key family is closest to "
        f"existing `{closest_schema.class_name}`."
        if closest_schema is not None
        else f"Create `{class_name}` inheriting from `{FindingMetrics.__name__}` to give this metrics bag a nominal schema."
    )
    scaffold = _declaration_scaffold(
        class_name, base_class_name, key_names, items, instantiation_prefix="metrics="
    )
    return SemanticDataclassRecommendation.proposed_schema(
        class_name,
        base_class_name,
        closest_schema.class_name if closest_schema else None,
        rationale,
        scaffold,
    )


def _recommend_local_semantic_record(
    key_names: tuple[str, ...],
    owner_symbol: str,
    variable_name: str,
    value_nodes: dict[str, ast.AST],
) -> SemanticDataclassRecommendation | None:
    exact_schema = _exact_schema_match(key_names, (_IMPACT_BAG_SCHEMA,))
    if exact_schema is not None:
        class_name = exact_schema.class_name
        rationale = f"Use `{class_name}` directly instead of a string-key impact bag."
        scaffold = _instantiation_scaffold(class_name, key_names, value_nodes)
        return SemanticDataclassRecommendation.existing_schema(
            class_name, exact_schema.base_class_name, rationale, scaffold
        )

    closest_schema = _closest_schema_match(key_names, (_IMPACT_BAG_SCHEMA,))
    if closest_schema is None:
        if not (variable_name.endswith("metrics") or variable_name in {"metrics"}):
            return None
        return _recommend_metrics_dataclass(value_nodes, owner_symbol=owner_symbol)

    class_name = _suggest_dataclass_name(owner_symbol, "ImpactDelta")
    rationale = f"Create `{class_name}` inheriting from `{closest_schema.class_name}` because the local bag carries the same quantified impact fields nominally modeled there."
    scaffold = _declaration_scaffold(
        class_name, closest_schema.class_name, key_names, value_nodes
    )
    return SemanticDataclassRecommendation.proposed_schema(
        class_name,
        closest_schema.class_name,
        closest_schema.class_name,
        rationale,
        scaffold,
    )


def _exact_schema_match(
    key_names: tuple[str, ...], schemas: tuple[SemanticBagDescriptor, ...]
) -> SemanticBagDescriptor | None:
    key_set = frozenset(key_names)
    for schema in schemas:
        if key_set in schema.accepted_key_sets:
            return schema
    return None


def _closest_schema_match(
    key_names: tuple[str, ...], schemas: tuple[SemanticBagDescriptor, ...]
) -> SemanticBagDescriptor | None:
    key_set = frozenset(key_names)
    best_schema: SemanticBagDescriptor | None = None
    best_score = 0.0
    for schema in schemas:
        for accepted in schema.accepted_key_sets:
            score = _set_similarity(key_set, accepted)
            if score > best_score:
                best_schema = schema
                best_score = score
    if best_score < 0.4:
        return None
    return best_schema


def _set_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and (not right):
        return 1.0
    return len(left & right) / len(left | right)


def _declaration_scaffold(
    class_name: str,
    base_class_name: str,
    key_names: tuple[str, ...],
    value_nodes: dict[str, ast.AST],
    instantiation_prefix: str = "",
) -> str:
    field_lines = "\n".join(
        (
            f"    {key}: {_infer_field_type_name(key, value_nodes.get(key))}"
            for key in key_names
        )
    )
    return f"@dataclass(frozen=True)\nclass {class_name}({base_class_name}):\n{field_lines}\n\n{_instantiation_scaffold(class_name, key_names, value_nodes, prefix=instantiation_prefix)}"


def _instantiation_scaffold(
    class_name: str,
    key_names: tuple[str, ...],
    value_nodes: dict[str, ast.AST],
    prefix: str = "",
) -> str:
    rendered_args = ",\n    ".join(
        (
            f"{key}={_render_value_expression(key, value_nodes.get(key))}"
            for key in key_names
        )
    )
    return f"{prefix}{class_name}(\n    {rendered_args}\n)"


def _infer_field_type_name(key_name: str, node: ast.AST | None) -> str:
    if (
        key_name.endswith("_count")
        or "bound" in key_name
        or key_name.startswith("loci_")
    ):
        return "int"
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "bool"
        if isinstance(node.value, int):
            return "int"
        if isinstance(node.value, str):
            return "str"
        if node.value is None:
            return "object | None"
    if isinstance(node, ast.Compare):
        return "bool"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"len", "sum", "max", "min"}:
            return "int"
    if isinstance(node, ast.Tuple):
        return "tuple[object, ...]"
    if isinstance(node, ast.List):
        return "list[object]"
    return "object"


def _render_value_expression(key_name: str, node: ast.AST | None) -> str:
    if node is None:
        if (
            key_name.endswith("_count")
            or "bound" in key_name
            or key_name.startswith("loci_")
        ):
            return "0"
        return "..."
    return ast.unparse(node)


def _suggest_dataclass_name(owner_symbol: str, suffix: str) -> str:
    parts = [
        _camel_case(part)
        for part in re.split(r"[^A-Za-z0-9]+", owner_symbol)
        if part and part not in {"module", "record", "metrics"}
    ]
    prefix = parts[-1] if parts else "Semantic"
    if prefix.endswith(suffix):
        return prefix
    return f"{prefix}{suffix}"


def _owner_symbol(
    class_stack: tuple[str, ...], function_stack: tuple[str, ...], label: str
) -> str:
    owner = function_stack[-1] if function_stack else "<module>"
    if class_stack:
        owner = f"{class_stack[-1]}.{owner}"
    return f"{owner}:{label}"


def _string_dict_items(node: ast.AST) -> dict[str, ast.AST] | None:
    if not isinstance(node, ast.Dict) or not node.keys:
        return None
    items: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        key_name = _constant_string(key)
        if key_name is None or value is None:
            return None
        items[key_name] = value
    return items


def _string_slice_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_json_boundary_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name) and node.func.id == "asdict":
        return True
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "json"
        and node.func.attr in {"dump", "dumps"}
    )


def _class_direct_constant_string_assignments(node: ast.ClassDef) -> dict[str, str]:
    return {
        name: string_value
        for name, value in _class_direct_assignments(node).items()
        if (string_value := _constant_string(value)) is not None
    }


def _class_direct_non_none_assignment_names(node: ast.ClassDef) -> tuple[str, ...]:
    return sorted_tuple(
        (
            name
            for name, value in _class_direct_assignments(node).items()
            if not (isinstance(value, ast.Constant) and value.value is None)
        )
    )


def _abstract_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return sorted_tuple(
        (
            method.name
            for method in _iter_class_methods(node)
            if _is_abstract_method(method)
        )
    )


def _is_classvar_annotation(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "ClassVar"
    if isinstance(node, ast.Attribute):
        return node.attr == "ClassVar"
    if isinstance(node, ast.Subscript):
        return _is_classvar_annotation(node.value)
    return False


def _function_parameter_annotation_map(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for arg in (
        tuple(node.args.posonlyargs)
        + tuple(node.args.args)
        + tuple(node.args.kwonlyargs)
    ):
        if arg.annotation is None:
            continue
        annotations[arg.arg] = ast.unparse(arg.annotation)
    return annotations


def _typed_field_map(node: ast.ClassDef) -> tuple[tuple[str, str], ...]:
    typed_fields: dict[str, str] = {}
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            if _is_classvar_annotation(statement.annotation):
                continue
            typed_fields.setdefault(
                statement.target.id, ast.unparse(statement.annotation)
            )
            continue
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if statement.name != "__init__":
            continue
        parameter_annotations = _function_parameter_annotation_map(statement)
        for inner in statement.body:
            target: ast.AST | None = None
            value: ast.AST | None = None
            if isinstance(inner, ast.Assign) and len(inner.targets) == 1:
                target = inner.targets[0]
                value = inner.value
            elif isinstance(inner, ast.AnnAssign):
                target = inner.target
                value = inner.value
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and (target.value.id == "self")
                and isinstance(value, ast.Name)
                and (value.id in parameter_annotations)
            ):
                continue
            typed_fields.setdefault(target.attr, parameter_annotations[value.id])
    return sorted_tuple(typed_fields.items())


def _semantic_role_names_for_fields(field_names: tuple[str, ...]) -> tuple[str, ...]:
    role_names: set[str] = set()
    for field_name in field_names:
        normalized_roles = _normalize_semantic_field_roles(field_name)
        if normalized_roles:
            role_names.update(normalized_roles)
            continue
        role_names.add(field_name)
    return sorted_tuple(role_names)


def _shared_ordered_suffix(
    left_tokens: tuple[str, ...],
    right_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    shared_reversed: list[str] = []
    for left_token, right_token in zip(reversed(left_tokens), reversed(right_tokens)):
        if left_token != right_token:
            break
        shared_reversed.append(left_token)
    return tuple(reversed(shared_reversed))


def _nominal_authority_shapes(
    modules: Sequence[ParsedModule],
) -> tuple[NominalAuthorityShape, ...]:
    shapes_without_ancestors: list[NominalAuthorityShape] = []
    for module in modules:
        for node in _walk_nodes(module.module):
            if not isinstance(node, ast.ClassDef):
                continue
            field_type_map = _typed_field_map(node)
            shapes_without_ancestors.append(
                NominalAuthorityShape(
                    file_path=str(module.path),
                    class_name=node.name,
                    line=node.lineno,
                    declared_base_names=_declared_base_names(node),
                    ancestor_names=(),
                    field_names=tuple((name for name, _ in field_type_map)),
                    field_type_map=field_type_map,
                    method_names=sorted_tuple(_method_names(node)),
                    is_abstract=_is_abstract_class(node),
                    is_dataclass_family=_is_dataclass_class(node),
                )
            )

    base_lookup: dict[str, set[str]] = defaultdict(set)
    for shape in shapes_without_ancestors:
        base_lookup[shape.class_name].update(shape.declared_base_names)

    def ancestors_for(class_name: str) -> tuple[str, ...]:
        seen: set[str] = set()
        stack = list(base_lookup.get(class_name, set()))
        while stack:
            base_name = stack.pop()
            if base_name in seen or base_name == class_name:
                continue
            seen.add(base_name)
            stack.extend(sorted(base_lookup.get(base_name, set()) - seen))
        return sorted_tuple(seen)

    return tuple(
        (
            NominalAuthorityShape(
                file_path=shape.file_path,
                class_name=shape.class_name,
                line=shape.line,
                declared_base_names=shape.declared_base_names,
                ancestor_names=ancestors_for(shape.class_name),
                field_names=shape.field_names,
                field_type_map=shape.field_type_map,
                method_names=shape.method_names,
                is_abstract=shape.is_abstract,
                is_dataclass_family=shape.is_dataclass_family,
            )
            for shape in shapes_without_ancestors
        )
    )


class NominalAuthorityIndex:
    def __init__(self, modules: Sequence[ParsedModule]) -> None:
        self._shapes = _nominal_authority_shapes(modules)
        self._shapes_by_name: dict[str, list[NominalAuthorityShape]] = defaultdict(list)
        for shape in self._shapes:
            self._shapes_by_name[shape.class_name].append(shape)

    def all_shapes(self) -> tuple[NominalAuthorityShape, ...]:
        return self._shapes

    def shapes_named(self, class_name: str) -> tuple[NominalAuthorityShape, ...]:
        return tuple(self._shapes_by_name.get(class_name, ()))

    def compatible_authorities_for(
        self, shape: NominalAuthorityShape
    ) -> tuple[NominalAuthorityShape, ...]:
        compatible: list[NominalAuthorityShape] = []
        for authority in self._shapes:
            if authority.class_name == shape.class_name:
                continue
            if authority.class_name in set(shape.ancestor_names):
                continue
            if not _is_reusable_nominal_authority(authority):
                continue
            shared_field_names = _shared_typed_field_names(shape, authority)
            if len(shared_field_names) < 2:
                continue
            if set(shared_field_names) != set(authority.field_names):
                continue
            compatible.append(authority)
        return sorted_tuple(
            compatible,
            key=lambda authority: (
                -len(authority.field_names),
                not authority.is_abstract,
                authority.class_name,
            ),
        )


def _is_reusable_nominal_authority(shape: NominalAuthorityShape) -> bool:
    if shape.class_name.endswith("Detector"):
        return False
    return bool(
        shape.is_abstract or shape.class_name.endswith(("Base", "Mixin", "Carrier"))
    )


def _shared_typed_field_names(
    concrete: NominalAuthorityShape, authority: NominalAuthorityShape
) -> tuple[str, ...]:
    concrete_types = dict(concrete.field_type_map)
    return tuple(
        (
            name
            for name, annotation_text in authority.field_type_map
            if concrete_types.get(name) == annotation_text
        )
    )


def _family_roster_member(
    node: ast.AST,
    known_class_names: set[str],
) -> tuple[str, str] | None:
    name = name_id(node)
    if name is not None and name in known_class_names:
        return (name, "class_reference")
    call = as_ast(node, ast.Call)
    call_name = name_id(call.func if call is not None else None)
    if (
        call is not None
        and call_name in known_class_names
        and (not call.args)
        and (not call.keywords)
    ):
        return (cast(str, call_name), "constructor_call")
    return None


def _extract_family_roster_members(
    node: ast.AST,
    known_class_names: set[str],
) -> tuple[tuple[str, ...], str] | None:
    collection = collection_literal(node)
    if collection is None:
        return None
    roster_members = tuple(
        (
            _family_roster_member(element, known_class_names)
            for element in collection.elements
        )
    )
    if len(roster_members) < 2 or any((member is None for member in roster_members)):
        return None
    member_names, constructor_styles = zip(*roster_members)
    return (tuple(member_names), "+".join(sorted(set(constructor_styles))))


def _best_shared_family_base_name(
    member_names: tuple[str, ...], index: NominalAuthorityIndex
) -> str | None:
    candidate_sets: list[set[str]] = []
    for member_name in member_names:
        shapes = index.shapes_named(member_name)
        if not shapes:
            return None
        ancestor_names = {
            name
            for shape in shapes
            for name in (*shape.declared_base_names, *shape.ancestor_names)
            if name not in _IGNORED_ANCESTOR_NAMES
        }
        if not ancestor_names:
            return None
        candidate_sets.append(ancestor_names)
    shared = set.intersection(*candidate_sets)
    if not shared:
        return None
    return sorted(shared, key=lambda item: (item.startswith("Issue"), len(item), item))[
        0
    ]


def _manual_family_roster_candidates(
    module: ParsedModule,
    index: NominalAuthorityIndex,
) -> tuple[ManualFamilyRosterCandidate, ...]:
    known_class_names = {
        shape.class_name
        for shape in index.all_shapes()
        if shape.file_path == str(module.path)
    }
    candidates: list[ManualFamilyRosterCandidate] = []
    module_body = _trim_docstring_body(module.module.body)
    for statement in module_body:
        owner_name: str | None = None
        line = statement.lineno
        source_node: ast.AST | None = None
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = _trim_docstring_body(statement.body)
            if (
                len(body) != 1
                or not isinstance(body[0], ast.Return)
                or body[0].value is None
            ):
                continue
            owner_name = statement.name
            source_node = body[0].value
            line = statement.lineno
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                continue
            owner_name = target.id
            source_node = statement.value
            line = statement.lineno
        if owner_name is None or source_node is None:
            continue
        extracted = _extract_family_roster_members(source_node, known_class_names)
        if extracted is None:
            continue
        member_names, constructor_style = extracted
        family_base_name = _best_shared_family_base_name(member_names, index)
        if family_base_name is None:
            continue
        candidates.append(
            ManualFamilyRosterCandidate(
                file_path=str(module.path),
                line=line,
                owner_name=owner_name,
                member_names=member_names,
                family_base_name=family_base_name,
                constructor_style=constructor_style,
            )
        )
    return tuple(candidates)


def _enum_key_family(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Attribute):
        return None
    if not isinstance(node.value, ast.Name):
        return None
    return (node.value.id, node.attr)


def _fragmented_family_authority_candidates(
    module: ParsedModule,
) -> tuple[FragmentedFamilyAuthorityCandidate, ...]:
    family_maps: dict[str, list[tuple[str, int, tuple[str, ...]]]] = defaultdict(list)
    for statement in _trim_docstring_body(module.module.body):
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
        if target_name is None or value is None or (not isinstance(value, ast.Dict)):
            continue
        key_pairs = tuple(
            (
                key_pair
                for key_pair in (
                    _enum_key_family(key) for key in value.keys if key is not None
                )
                if key_pair is not None
            )
        )
        if len(key_pairs) < 2 or len(key_pairs) != len(value.keys):
            continue
        family_names = {family_name for family_name, _ in key_pairs}
        if len(family_names) != 1:
            continue
        family_name = next(iter(family_names))
        key_names = sorted_tuple((member_name for _, member_name in key_pairs))
        family_maps[family_name].append((target_name, statement.lineno, key_names))

    candidates: list[FragmentedFamilyAuthorityCandidate] = []
    for family_name, entries in family_maps.items():
        if len(entries) < 2:
            continue
        key_counter: Counter[str] = Counter(
            (key_name for _, _, key_names in entries for key_name in set(key_names))
        )
        shared_keys = sorted_tuple(
            (key for key, count in key_counter.items() if count >= 2)
        )
        if len(shared_keys) < 3:
            continue
        total_keys = sorted_tuple(key_counter)
        ordered_entries = sorted(entries, key=lambda item: item[1])
        candidates.append(
            FragmentedFamilyAuthorityCandidate(
                file_path=str(module.path),
                mapping_names=tuple((item[0] for item in ordered_entries)),
                line_numbers=tuple((item[1] for item in ordered_entries)),
                key_family_name=family_name,
                shared_keys=shared_keys,
                total_keys=total_keys,
            )
        )
    return tuple(candidates)


_DETECTOR_BASE_NAMES = {
    "IssueDetector",
    "PerModuleIssueDetector",
    "EvidenceOnlyPerModuleDetector",
    "StaticModulePatternDetector",
    "GroupedShapeIssueDetector",
    "FiberCollectedShapeIssueDetector",
}


def _is_detectorish_class(node: ast.ClassDef) -> bool:
    if node.name.endswith("Detector"):
        return True
    return bool(_DETECTOR_BASE_NAMES & set(_declared_base_names(node)))


def _finding_build_call(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Call | None:
    for node in _walk_nodes(method):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "build":
            continue
        value = node.func.value
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "finding_spec"
            and isinstance(value.value, ast.Name)
            and (value.value.id == "self")
        ):
            continue
        return node
    return None


def _call_display_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _build_call_keyword_helper_name(
    build_call: ast.Call, keyword_name: str
) -> str | None:
    for keyword in build_call.keywords:
        if keyword.arg != keyword_name or keyword.value is None:
            continue
        if isinstance(keyword.value, ast.Call):
            return _call_display_name(keyword.value)
    return None


def _candidate_source_name_from_method(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    assigned_calls: dict[str, str] = {}
    for statement in _trim_docstring_body(method.body):
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and isinstance(statement.value, ast.Call):
                call_name = _call_display_name(statement.value)
                if call_name is not None:
                    assigned_calls[target.id] = call_name
        if isinstance(statement, ast.For):
            iterator = statement.iter
            if isinstance(iterator, ast.Call):
                return _call_display_name(iterator)
            if isinstance(iterator, ast.Name):
                return assigned_calls.get(iterator.id)
    return None


def _finding_assembly_pipeline_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[FindingAssemblyPipelineCandidate]:
    if not _is_detectorish_class(node):
        return
    method = next(
        (
            statement
            for statement in node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == "_findings_for_module"
        ),
        None,
    )
    if method is None:
        return
    build_call = _finding_build_call(method)
    if build_call is None:
        return
    candidate_source_name = _candidate_source_name_from_method(method)
    if candidate_source_name is None:
        return
    metrics_type_name = _build_call_keyword_helper_name(build_call, "metrics")
    scaffold_helper_name = _build_call_keyword_helper_name(build_call, "scaffold")
    patch_helper_name = _build_call_keyword_helper_name(build_call, "codemod_patch")
    if not any(
        helper_name is not None
        for helper_name in (
            metrics_type_name,
            scaffold_helper_name,
            patch_helper_name,
        )
    ):
        return
    yield FindingAssemblyPipelineCandidate(
        file_path=str(module.path),
        line=method.lineno,
        subject_name=node.name,
        name_family=tuple(
            item
            for item in (
                candidate_source_name,
                metrics_type_name,
                scaffold_helper_name,
                patch_helper_name,
            )
            if item is not None
        ),
        method_name=method.name,
        candidate_source_name=candidate_source_name,
        metrics_type_name=metrics_type_name,
        scaffold_helper_name=scaffold_helper_name,
        patch_helper_name=patch_helper_name,
    )


def _finding_assembly_pipeline_candidates(
    module: ParsedModule,
) -> tuple[FindingAssemblyPipelineCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _finding_assembly_pipeline_candidates_for_class,
    )


def _is_observation_spec_class(node: ast.ClassDef) -> bool:
    if node.name.endswith("ObservationSpec"):
        return True
    return bool(
        {
            "ObservationShapeSpec",
            "FunctionObservationSpec",
            "AssignObservationSpec",
            "ContextForwardingShapeSpec",
        }
        & set(_declared_base_names(node))
    )


def _if_returns_none_only(node: ast.If) -> bool:
    return bool(
        len(node.body) == 1
        and isinstance(node.body[0], ast.Return)
        and isinstance(node.body[0].value, ast.Constant)
        and (node.body[0].value.value is None)
        and (not node.orelse)
    )


def _delegate_name_from_return(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        outer_name = _call_display_name(node)
        if outer_name in _SEQUENCE_WRAPPER_CALL_NAMES and len(node.args) == 1:
            inner = node.args[0]
            if isinstance(inner, ast.Call):
                return _call_display_name(inner)
        return outer_name
    return None


def _guard_role_name(node: ast.AST) -> str:
    text = ast.unparse(node)
    if "observation.class_name is not None" in text:
        return "module_only_guard"
    if "observation.class_name is None" in text:
        return "class_only_guard"
    if "observation.function_name is None" in text:
        return "module_scope_guard"
    if "observation.function_name is not None" in text:
        return "function_scope_guard"
    if "isinstance" in text:
        return "node_type_guard"
    return "guarded_delegate"


def _scope_role_name(node: ast.AST) -> str:
    text = ast.unparse(node)
    if "class_name" in text and "function_name" in text:
        return "scope_filtered"
    if "class_name" in text:
        return "class_scope"
    if "function_name" in text:
        return "function_scope"
    if "isinstance" in text:
        return "node_type"
    return "generic_scope"


def _guarded_delegator_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[GuardedDelegatorCandidate]:
    if not _is_observation_spec_class(node) or _is_abstract_class(node):
        return
    for statement in node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if statement.name not in {
            "build_from_function",
            "build_from_assign",
            "build_from_observation",
            "build_from_context",
        }:
            continue
        body = _trim_docstring_body(statement.body)
        while body and isinstance(body[0], ast.Assign):
            body = body[1:]
        if len(body) != 2:
            continue
        guard, return_stmt = body
        if not isinstance(guard, ast.If) or not _if_returns_none_only(guard):
            continue
        if not isinstance(return_stmt, ast.Return) or return_stmt.value is None:
            continue
        delegate_name = _delegate_name_from_return(return_stmt.value)
        if delegate_name is None:
            continue
        yield GuardedDelegatorCandidate(
            file_path=str(module.path),
            line=statement.lineno,
            subject_name=node.name,
            name_family=(
                guard.test.__class__.__name__,
                delegate_name,
                _scope_role_name(guard.test),
            ),
            method_name=statement.name,
            guard_role=_guard_role_name(guard.test),
            delegate_name=delegate_name,
            scope_role=_scope_role_name(guard.test),
        )


def _guarded_delegator_candidates(
    module: ParsedModule,
) -> tuple[GuardedDelegatorCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _guarded_delegator_candidates_for_class,
    )


def _name_mentions(node: ast.AST, name: str) -> bool:
    return any(
        (
            isinstance(current, ast.Name) and current.id == name
            for current in _walk_nodes(node)
        )
    )


def _raised_exception_name(
    statement: ast.stmt,
) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(statement, ast.Raise) or statement.exc is None:
        return None
    exc = statement.exc
    if isinstance(exc, ast.Call):
        exc_name = _ast_terminal_name(exc.func)
        referenced_names = sorted_tuple(
            {
                current.id
                for current in _walk_nodes(exc)
                if isinstance(current, ast.Name)
            }
        )
        if exc_name is not None:
            return (exc_name, referenced_names)
    exc_name = _ast_terminal_name(exc)
    if exc_name is not None:
        return (exc_name, ())
    return None


def _linear_query_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, tuple[str, ...], str, str] | None:
    body = _trim_docstring_body(node.body)
    loop = _linear_query_loop(body)
    if loop is None:
        return None
    result_name = cast(ast.Name, loop.target).id
    return_expr = _linear_query_return_expr(loop, result_name)
    raised = _linear_query_raised_exception(body)
    if return_expr is None or raised is None:
        return None
    exception_name, exception_names = raised
    query_key_names = _linear_query_key_names(node, loop, return_expr, exception_names)
    if not query_key_names:
        return None
    return (
        ast.unparse(loop.iter),
        query_key_names,
        ast.unparse(return_expr),
        exception_name,
    )


def _linear_query_loop(body: list[ast.stmt]) -> ast.For | None:
    if len(body) < 2:
        return None
    loop = next(
        (statement for statement in body if isinstance(statement, ast.For)), None
    )
    if loop is None or not isinstance(loop.target, ast.Name):
        return None
    return loop


def _linear_query_return_expr(loop: ast.For, result_name: str) -> ast.AST | None:
    return_exprs = [
        current.value
        for current in _walk_nodes(loop)
        if isinstance(current, ast.Return) and current.value is not None
    ]
    if len(return_exprs) != 1 or not _name_mentions(return_exprs[0], result_name):
        return None
    return return_exprs[0]


def _linear_query_raised_exception(
    body: list[ast.stmt],
) -> tuple[str, tuple[str, ...]] | None:
    raised = next(
        (
            _raised_exception_name(statement)
            for statement in body
            if _raised_exception_name(statement) is not None
        ),
        None,
    )
    if raised is None:
        return None
    exception_name, exception_names = raised
    if exception_name not in {"KeyError", "LookupError", "ValueError"}:
        return None
    return exception_name, exception_names


def _linear_query_key_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    loop: ast.For,
    return_expr: ast.AST,
    exception_names: tuple[str, ...],
) -> tuple[str, ...]:
    parameter_names = tuple(
        (
            arg.arg
            for arg in tuple(node.args.posonlyargs)
            + tuple(node.args.args)
            + tuple(node.args.kwonlyargs)
            if arg.arg not in {"self", "cls"}
        )
    )
    query_key_names = sorted_tuple(
        (
            name
            for name in parameter_names
            if _name_mentions(return_expr, name)
            or name in exception_names
            or any(
                (
                    isinstance(current, ast.If) and _name_mentions(current.test, name)
                    for current in _walk_nodes(loop)
                )
            )
        )
    )
    return query_key_names


def _derived_query_index_candidates(
    module: ParsedModule,
) -> tuple[DerivedQueryIndexCandidate, ...]:
    grouped: dict[(tuple[str, str, str], list[tuple[str, int, tuple[str, ...]]])] = (
        defaultdict(list)
    )
    for qualname, function in _iter_named_functions(module):
        signature = _linear_query_signature(function)
        if signature is None:
            continue
        source_expression, query_key_names, return_expression, exception_name = (
            signature
        )
        grouped[source_expression, return_expression, exception_name].append(
            (qualname, function.lineno, query_key_names)
        )
    candidates: list[DerivedQueryIndexCandidate] = []
    for (
        source_expression,
        return_expression,
        exception_name,
    ), entries in grouped.items():
        if len(entries) < 2:
            continue
        ordered = sorted_tuple(entries, key=lambda item: (item[1], item[0]))
        query_key_names = sorted_tuple(
            {
                key_name
                for _, _, entry_query_key_names in ordered
                for key_name in entry_query_key_names
            }
        )
        candidates.append(
            DerivedQueryIndexCandidate(
                file_path=str(module.path),
                line_numbers=tuple((item[1] for item in ordered)),
                function_names=tuple((item[0] for item in ordered)),
                source_expression=source_expression,
                query_key_names=query_key_names,
                return_expressions=tuple((return_expression for _ in ordered)),
                exception_names=(exception_name,),
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.source_expression, item.function_names),
    )


def _simple_attribute_accesses(node: ast.AST) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            (current.value.id, current.attr)
            for current in _walk_nodes(node)
            if isinstance(current, ast.Attribute)
            and isinstance(current.value, ast.Name)
            and (current.value.id not in {"self", "cls"})
        )
    )


def _projection_source_name(node: ast.Call) -> str | None:
    source_counts: Counter[str] = Counter(
        (
            root_name
            for keyword in node.keywords
            if keyword.arg is not None
            for root_name, _ in _simple_attribute_accesses(keyword.value)
        )
    )
    if not source_counts:
        return None
    source_name, count = source_counts.most_common(1)[0]
    if count < 3:
        return None
    if sum((1 for value in source_counts.values() if value == count)) > 1:
        return None
    return source_name


def _direct_source_attribute_name(node: ast.AST, source_name: str) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and (node.value.id == source_name)
    ):
        return node.attr
    return None


def _resolver_lookup_metadata(
    node: ast.AST, source_name: str
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    table_names: set[str] = set()
    selector_field_names = sorted_tuple(
        {
            attr_name
            for root_name, attr_name in _simple_attribute_accesses(node)
            if root_name == source_name
        }
    )
    if not selector_field_names:
        return None
    for current in _walk_nodes(node):
        if (
            isinstance(current, ast.Call)
            and isinstance(current.func, ast.Attribute)
            and current.func.attr == "get"
            and isinstance(current.func.value, ast.Name)
        ):
            table_names.add(current.func.value.id)
            continue
        if isinstance(current, ast.Subscript) and isinstance(current.value, ast.Name):
            table_names.add(current.value.id)
    if not table_names:
        return None
    return (sorted_tuple(table_names), selector_field_names)


def _runtime_adapter_shell_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
    local_dataclass_names: set[str],
    table_lines: dict[str, int],
) -> Iterable[RuntimeAdapterShellCandidate]:
    for current in _walk_nodes(function):
        if (
            not isinstance(current, ast.Return)
            or not isinstance(current.value, ast.Call)
            or len(current.value.keywords) < 3
        ):
            continue
        adapter_class_name = _ast_terminal_name(current.value.func)
        if adapter_class_name not in local_dataclass_names:
            continue
        source_name = _projection_source_name(current.value)
        if source_name is None:
            continue
        copied_field_names: list[str] = []
        resolver_field_names: list[str] = []
        resolver_table_names: set[str] = set()
        selector_field_names: set[str] = set()
        for keyword in current.value.keywords:
            if keyword.arg is None:
                continue
            if (
                direct_attr_name := _direct_source_attribute_name(
                    keyword.value, source_name
                )
            ) is not None:
                copied_field_names.append(keyword.arg)
                selector_field_names.add(direct_attr_name)
                continue
            resolver_metadata = _resolver_lookup_metadata(keyword.value, source_name)
            if resolver_metadata is None:
                continue
            table_names, resolver_fields = resolver_metadata
            resolver_field_names.append(keyword.arg)
            resolver_table_names.update(table_names)
            selector_field_names.update(resolver_fields)
        if len(copied_field_names) < 2 or not resolver_field_names:
            continue
        evidence = [
            SourceLocation(str(module.path), function.lineno, qualname),
            SourceLocation(str(module.path), current.lineno, adapter_class_name),
        ]
        evidence.extend(
            SourceLocation(
                str(module.path),
                table_lines.get(table_name, current.lineno),
                table_name,
            )
            for table_name in sorted(resolver_table_names)
        )
        yield RuntimeAdapterShellCandidate(
            file_path=str(module.path),
            line=function.lineno,
            function_name=qualname,
            adapter_class_name=adapter_class_name,
            source_name=source_name,
            copied_field_names=sorted_tuple(copied_field_names),
            resolver_field_names=sorted_tuple(resolver_field_names),
            resolver_table_names=sorted_tuple(resolver_table_names),
            selector_field_names=sorted_tuple(selector_field_names),
            evidence_locations=tuple(evidence[:6]),
        )
        return


def _runtime_adapter_shell_candidates(
    module: ParsedModule,
) -> tuple[RuntimeAdapterShellCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    local_dataclass_names = {
        class_name
        for class_name, node in class_defs_by_name.items()
        if _is_dataclass_class(node)
    }
    if not local_dataclass_names:
        return ()
    table_lines = {
        table_name: line
        for table_name, (line, _) in _module_level_named_dicts(module).items()
    }
    return _collect_named_function_candidates(
        module,
        _runtime_adapter_shell_candidates_for_function,
        local_dataclass_names,
        table_lines,
        sort_key=lambda item: (item.file_path, item.line, item.function_name),
    )


def _is_none_guard_for_source_attr(
    node: ast.AST, source_name: str
) -> tuple[str, str] | None:
    if (
        not isinstance(node, ast.Compare)
        or len(node.ops) != 1
        or len(node.comparators) != 1
        or (not isinstance(node.ops[0], (ast.IsNot, ast.NotEq)))
    ):
        return None
    attr_name = _direct_source_attribute_name(node.left, source_name)
    comparator = node.comparators[0]
    if attr_name is None or not (
        isinstance(comparator, ast.Constant) and comparator.value is None
    ):
        return None
    return (source_name, attr_name)


def _keyword_bag_adapter_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
) -> Iterable[KeywordBagAdapterCandidate]:
    body = _trim_docstring_body(function.body)
    if len(body) < 2:
        return
    kwargs_name: str | None = None
    source_name: str | None = None
    key_names: list[str] = []
    source_field_names: list[str] = []
    invalid_shape = False
    for index, statement in enumerate(body):
        if index == 0:
            target_name: str | None = None
            value: ast.AST | None = None
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                target_name = statement.targets[0].id
                value = statement.value
            elif isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                target_name = statement.target.id
                value = statement.value
            if target_name is None or not isinstance(value, ast.Dict):
                invalid_shape = True
                break
            kwargs_name = target_name
            for key, value in zip(value.keys, value.values, strict=False):
                key_name = _constant_string(key)
                if key_name is None:
                    invalid_shape = True
                    break
                accesses = _simple_attribute_accesses(value)
                if len(accesses) != 1:
                    invalid_shape = True
                    break
                field_source_name, field_name = accesses[0]
                source_name = source_name or field_source_name
                if field_source_name != source_name:
                    invalid_shape = True
                    break
                key_names.append(key_name)
                source_field_names.append(field_name)
            if invalid_shape:
                break
            continue
        if index == len(body) - 1:
            if (
                not isinstance(statement, ast.Return)
                or not isinstance(statement.value, ast.Name)
                or statement.value.id != kwargs_name
            ):
                invalid_shape = True
            break
        if (
            not isinstance(statement, ast.If)
            or statement.orelse
            or source_name is None
            or _is_none_guard_for_source_attr(statement.test, source_name) is None
            or len(statement.body) != 1
            or not isinstance(statement.body[0], ast.Assign)
            or len(statement.body[0].targets) != 1
        ):
            invalid_shape = True
            break
        guard_source_name, guard_field_name = cast(
            tuple[str, str],
            _is_none_guard_for_source_attr(statement.test, source_name),
        )
        target = statement.body[0].targets[0]
        value = statement.body[0].value
        if (
            not isinstance(target, ast.Subscript)
            or not isinstance(target.value, ast.Name)
            or target.value.id != kwargs_name
            or (key_name := _constant_string(target.slice)) is None
        ):
            invalid_shape = True
            break
        value_attr_name = _direct_source_attribute_name(value, guard_source_name)
        if value_attr_name != guard_field_name:
            invalid_shape = True
            break
        key_names.append(key_name)
        source_field_names.append(value_attr_name)
    if invalid_shape or source_name is None or len(key_names) < 3:
        return
    yield KeywordBagAdapterCandidate(
        file_path=str(module.path),
        line=function.lineno,
        function_name=qualname,
        source_name=source_name,
        key_names=sorted_tuple(key_names),
        source_field_names=sorted_tuple(source_field_names),
    )


def _keyword_bag_adapter_candidates(
    module: ParsedModule,
) -> tuple[KeywordBagAdapterCandidate, ...]:
    return _collect_named_function_candidates(
        module,
        _keyword_bag_adapter_candidates_for_function,
        sort_key=lambda item: (item.file_path, item.line, item.function_name),
    )


def _structural_observation_property_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[StructuralObservationPropertyCandidate]:
    for statement in node.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        if not any(
            _ast_terminal_name(decorator) == "property"
            for decorator in statement.decorator_list
        ):
            continue
        body = _trim_docstring_body(statement.body)
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            continue
        returned = body[0].value
        if not isinstance(returned, ast.Call):
            continue
        constructor_name = _call_name(returned.func)
        if constructor_name is None:
            continue
        keyword_names = sorted_tuple(
            keyword.arg for keyword in returned.keywords if keyword.arg is not None
        )
        if len(keyword_names) < 6:
            continue
        yield StructuralObservationPropertyCandidate(
            file_path=str(module.path),
            line=statement.lineno,
            subject_name=node.name,
            name_family=keyword_names,
            property_name=statement.name,
            constructor_name=constructor_name,
        )


def _structural_observation_property_candidates(
    module: ParsedModule,
) -> tuple[StructuralObservationPropertyCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _structural_observation_property_candidates_for_class,
    )


def _reuse_kind_for_authority(shape: NominalAuthorityShape) -> str:
    return "compose_mixin" if shape.class_name.endswith("Mixin") else "inherit_base"


def _existing_nominal_authority_reuse_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[ExistingNominalAuthorityReuseCandidate, ...]:
    index = NominalAuthorityIndex(modules)
    candidates: list[ExistingNominalAuthorityReuseCandidate] = []
    for shape in index.all_shapes():
        if shape.is_abstract or len(shape.field_type_map) < 2:
            continue
        compatible = index.compatible_authorities_for(shape)
        if not compatible:
            continue
        authority = next(
            (
                item
                for item in compatible
                if _class_name_tokens(item.class_name)
                & _class_name_tokens(shape.class_name)
            ),
            None,
        )
        if authority is None:
            continue
        shared_field_names = _shared_typed_field_names(shape, authority)
        if len(shared_field_names) < 2:
            continue
        candidates.append(
            ExistingNominalAuthorityReuseCandidate(
                file_path=shape.file_path,
                line=shape.line,
                subject_name=shape.class_name,
                name_family=shared_field_names,
                compatible_authority_file_path=authority.file_path,
                compatible_authority_name=authority.class_name,
                compatible_authority_line=authority.line,
                reuse_kind=_reuse_kind_for_authority(authority),
                shared_role_names=_semantic_role_names_for_fields(shared_field_names),
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.file_path,
            item.line,
            item.class_name,
            item.compatible_authority_name,
        ),
    )


def _normalized_authority_name(annotation_text: str) -> str:
    text = annotation_text.strip("\"'")
    text = re.split("\\s*\\|\\s*", text, maxsplit=1)[0]
    text = re.split("[\\[,]", text, maxsplit=1)[0]
    return text.rsplit(".", 1)[-1].strip()


def _is_self_delegate_attribute(node: ast.AST, delegate_field_name: str) -> bool:
    return bool(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and (node.value.id == "self")
        and (node.attr == delegate_field_name)
    )


def _forwarded_delegate_property_name(
    returned: ast.AST,
    method_name: str,
    delegate_field_name: str,
) -> str | None:
    attribute = as_ast(returned, ast.Attribute)
    if (
        attribute is None
        or attribute.attr != method_name
        or (not _is_self_delegate_attribute(attribute.value, delegate_field_name))
    ):
        return None
    return attribute.attr


def _forwarded_delegate_call(
    returned: ast.AST,
    method_name: str,
    delegate_field_name: str,
) -> ast.Call | None:
    call = as_ast(returned, ast.Call)
    match = (
        attribute_call_match(
            call, method_name=method_name, owner_type=ast.Attribute, owner_name="self"
        )
        if call is not None
        else None
    )
    if match is None or match.owner.attr != delegate_field_name:
        return None
    return call


def _method_forwarded_parameter_names(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    return tuple(
        (
            arg.arg
            for arg in (
                *method.args.posonlyargs,
                *method.args.args[1:],
                *method.args.kwonlyargs,
            )
        )
    )


def _call_forwards_parameters(
    call: ast.Call,
    parameter_names: tuple[str, ...],
) -> bool:
    parameter_set = frozenset(parameter_names)

    def forwards_argument(node: ast.AST) -> bool:
        name = name_id(node)
        if name is not None:
            return name in parameter_set
        starred = as_ast(node, ast.Starred)
        return name_id(starred.value if starred else None) in parameter_set

    return all(forwards_argument(argument) for argument in call.args) and all(
        (
            keyword.arg is None
            or (keyword.arg in parameter_set and name_id(keyword.value) == keyword.arg)
            for keyword in call.keywords
        )
    )


def _forwarded_delegate_member_name(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    delegate_field_name: str,
) -> str | None:
    body = _trim_docstring_body(method.body)
    returned = single_return_value(body)
    if returned is None:
        return None
    if any(
        (
            _ast_terminal_name(decorator) == "property"
            for decorator in method.decorator_list
        )
    ):
        return _forwarded_delegate_property_name(
            returned, method.name, delegate_field_name
        )
    call = _forwarded_delegate_call(returned, method.name, delegate_field_name)
    if call is None or not _call_forwards_parameters(
        call, _method_forwarded_parameter_names(method)
    ):
        return None
    return method.name


def _pass_through_nominal_wrapper_candidates_for_class(
    module: ParsedModule,
    node: ast.ClassDef,
    index: NominalAuthorityIndex,
) -> Iterable[PassThroughNominalWrapperCandidate]:
    if _is_abstract_class(node):
        return
    typed_fields = _typed_field_map(node)
    if len(typed_fields) != 1:
        return
    delegate_field_name, annotation_text = typed_fields[0]
    delegate_authority_name = _normalized_authority_name(annotation_text)
    if not delegate_authority_name:
        return
    if delegate_authority_name in set(_declared_base_names(node)):
        return
    authorities = tuple(
        authority
        for authority in index.shapes_named(delegate_authority_name)
        if _is_reusable_nominal_authority(authority)
    )
    if not authorities:
        return
    authority = authorities[0]
    forwarded_member_names: list[str] = []
    unsupported_residue = False
    for statement in _trim_docstring_body(node.body):
        if isinstance(statement, ast.AnnAssign):
            continue
        if isinstance(statement, ast.Assign):
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.name == "__init__":
                continue
            if statement.name.startswith("__") and statement.name.endswith("__"):
                unsupported_residue = True
                break
            forwarded_member_name = _forwarded_delegate_member_name(
                statement, delegate_field_name
            )
            if forwarded_member_name is None:
                unsupported_residue = True
                break
            forwarded_member_names.append(forwarded_member_name)
            continue
        unsupported_residue = True
        break
    if unsupported_residue or len(forwarded_member_names) < 2:
        return
    if not set(forwarded_member_names) <= set(authority.method_names):
        return
    yield PassThroughNominalWrapperCandidate(
        file_path=str(module.path),
        line=node.lineno,
        subject_name=node.name,
        name_family=sorted_tuple(set(forwarded_member_names)),
        delegate_field_name=delegate_field_name,
        delegate_authority_file_path=authority.file_path,
        delegate_authority_name=authority.class_name,
        delegate_authority_line=authority.line,
    )


def _pass_through_nominal_wrapper_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[PassThroughNominalWrapperCandidate, ...]:
    index = NominalAuthorityIndex(modules)
    return sorted_tuple(
        (
            candidate
            for module in modules
            for candidate in _collect_ast_node_candidates(
                module,
                module.module,
                ast.ClassDef,
                _pass_through_nominal_wrapper_candidates_for_class,
                index,
            )
        ),
        key=lambda item: (
            item.file_path,
            item.line,
            item.class_name,
            item.delegate_authority_name,
        ),
    )


def _is_projection_like_builder_value(value_fingerprint: str) -> bool:
    return value_fingerprint.startswith(("Name(", "Attribute(", "IfExp(", "Constant("))


def _projection_builder_groups(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[tuple[BuilderCallShape, ...], ...]:
    grouped: dict[tuple[str, tuple[str, ...]], list[BuilderCallShape]] = defaultdict(
        list
    )
    for builder in _collect_typed_family_items(
        module, BuilderCallShapeFamily, BuilderCallShape
    ):
        if len(builder.keyword_names) < max(config.min_builder_keywords, 6):
            continue
        if not all(
            (
                _is_projection_like_builder_value(value)
                for value in builder.value_fingerprint
            )
        ):
            continue
        grouped[(builder.callee_name, builder.keyword_names)].append(builder)
    candidates: list[tuple[BuilderCallShape, ...]] = []
    for builders in grouped.values():
        if len(builders) < 3:
            continue
        if len({builder.value_fingerprint for builder in builders}) < 2:
            continue
        if len({builder.symbol for builder in builders}) < 2:
            continue
        candidates.append(
            sorted_tuple(builders, key=lambda item: (item.file_path, item.lineno))
        )
    return sorted_tuple(
        candidates,
        key=lambda group: (group[0].file_path, group[0].lineno, group[0].callee_name),
    )


def _projection_helper_groups(
    module: ParsedModule,
) -> tuple[tuple[ProjectionHelperShape, ...], ...]:
    shapes: tuple[ProjectionHelperShape, ...] = _collect_typed_family_items(
        module, ProjectionHelperObservationFamily, ProjectionHelperShape
    )
    graph = ObservationGraph(tuple(shape.structural_observation for shape in shapes))
    lookup = _carrier_lookup(tuple(shapes))
    groups: list[tuple[ProjectionHelperShape, ...]] = []
    for fiber in graph.fibers_with_min_observations(
        ObservationKind.PROJECTION_HELPER,
        StructuralExecutionLevel.FUNCTION_BODY,
        minimum_observations=2,
    ):
        ordered = tuple(
            (
                _as_projection_helper_shape(item)
                for item in _materialize_observations(fiber.observations, lookup)
            )
        )
        attributes = {shape.projected_attribute for shape in ordered}
        if len(attributes) < 2:
            continue
        groups.append(ordered)
    return tuple(groups)


def _accessor_wrapper_groups(
    module: ParsedModule,
) -> tuple[tuple[AccessorWrapperCandidate, ...], ...]:
    candidates: tuple[AccessorWrapperCandidate, ...] = _collect_typed_family_items(
        module, AccessorWrapperObservationFamily, AccessorWrapperCandidate
    )
    graph = ObservationGraph(
        tuple((candidate.structural_observation for candidate in candidates))
    )
    lookup = _carrier_lookup(tuple(candidates))
    groups: list[tuple[AccessorWrapperCandidate, ...]] = []
    for witness_group in graph.witness_groups_for(
        ObservationKind.ACCESSOR_WRAPPER, StructuralExecutionLevel.FUNCTION_BODY
    ):
        ordered = tuple(
            (
                _as_accessor_wrapper_candidate(item)
                for item in _materialize_observations(
                    witness_group.observations, lookup
                )
            )
        )
        if not _supports_accessor_wrapper_finding(list(ordered)):
            continue
        groups.append(ordered)
    return tuple(groups)


def _mirrored_registry_candidates(
    module: ParsedModule,
) -> tuple[tuple[str, str, tuple[tuple[int, str], ...]], ...]:
    candidates: list[tuple[str, str, tuple[tuple[int, str], ...]]] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        dict_attrs = _collect_dict_attrs(node)
        mirrored_pairs = _collect_mirrored_assignments(node)
        if len(dict_attrs) < 2 or not mirrored_pairs:
            continue
        candidates.append((str(module.path), node.name, tuple(mirrored_pairs)))
    return tuple(candidates)


def _property_alias_hook_groups(
    module: ParsedModule,
) -> tuple[PropertyAliasHookGroup, ...]:
    grouped: dict[tuple[str, str, str], list[tuple[str, int]]] = defaultdict(list)
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = tuple(
            (
                name
                for name in _declared_base_names(node)
                if name not in _IGNORED_BASE_NAMES
            )
        )
        if not base_names:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not any(
                (
                    _ast_terminal_name(decorator) == "property"
                    for decorator in statement.decorator_list
                )
            ):
                continue
            if len(statement.args.args) != 1:
                continue
            body = _trim_docstring_body(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            returned = body[0].value
            if not (
                isinstance(returned, ast.Attribute)
                and isinstance(returned.value, ast.Name)
                and (returned.value.id == "self")
            ):
                continue
            for base_name in base_names:
                grouped[base_name, statement.name, returned.attr].append(
                    (node.name, statement.lineno)
                )
    return tuple(
        (
            PropertyAliasHookGroup(
                file_path=str(module.path),
                base_name=base_name,
                property_name=property_name,
                returned_attribute=returned_attribute,
                class_names=tuple((class_name for class_name, _ in ordered)),
                line_numbers=tuple((line for _, line in ordered)),
            )
            for (base_name, property_name, returned_attribute), items in sorted(
                grouped.items()
            )
            if len(items) >= 2
            for ordered in [sorted_tuple(items, key=lambda item: (item[1], item[0]))]
        )
    )


def _is_constant_hook_expression(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) or (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and (node.value.id != "self")
    )


def _class_defines_property(node: ast.ClassDef, property_name: str) -> bool:
    return any(
        (
            isinstance(statement, ast.FunctionDef)
            and statement.name == property_name
            and any(
                (
                    _ast_terminal_name(decorator) == "property"
                    for decorator in statement.decorator_list
                )
            )
            for statement in node.body
        )
    )


def _constant_property_hook_groups(
    module: ParsedModule,
) -> tuple[ConstantPropertyHookGroup, ...]:
    grouped: dict[tuple[str, str], list[tuple[str, int, str]]] = defaultdict(list)
    class_defs_by_name = _module_class_defs_by_name(module)
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = tuple(
            (
                name
                for name in _declared_base_names(node)
                if name not in _IGNORED_BASE_NAMES
            )
        )
        if not base_names:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not any(
                (
                    _ast_terminal_name(decorator) == "property"
                    for decorator in statement.decorator_list
                )
            ):
                continue
            body = _trim_docstring_body(statement.body)
            if (
                len(body) != 1
                or not isinstance(body[0], ast.Return)
                or body[0].value is None
            ):
                continue
            returned = body[0].value
            if not _is_constant_hook_expression(returned):
                continue
            return_expression = ast.unparse(returned)
            for base_name in base_names:
                base_node = class_defs_by_name.get(base_name)
                if base_node is None or not _class_defines_property(
                    base_node, statement.name
                ):
                    continue
                grouped[base_name, statement.name].append(
                    (node.name, statement.lineno, return_expression)
                )
    return tuple(
        (
            ConstantPropertyHookGroup(
                file_path=str(module.path),
                base_name=base_name,
                property_name=property_name,
                class_names=tuple((class_name for class_name, _, _ in ordered)),
                line_numbers=tuple((line for _, line, _ in ordered)),
                return_expressions=tuple((expression for _, _, expression in ordered)),
            )
            for (base_name, property_name), items in sorted(grouped.items())
            if len(items) >= 2
            for ordered in [sorted_tuple(items, key=lambda item: (item[1], item[0]))]
        )
    )


def _is_literal_constant_property_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal_constant_property_value(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (
                key is not None
                and _is_literal_constant_property_value(key)
                and _is_literal_constant_property_value(value)
                for key, value in zip(node.keys, node.values, strict=True)
            )
        )
    return False


def _constant_property_default_methods(
    module: ParsedModule, node: ast.ClassDef
) -> tuple[tuple[str, int, str, int], ...]:
    defaults: list[tuple[str, int, str, int]] = []
    for statement in node.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        if not any(
            (
                _ast_terminal_name(decorator) == "property"
                for decorator in statement.decorator_list
            )
        ):
            continue
        body = _trim_docstring_body(statement.body)
        returned = single_item(body)
        if (
            not isinstance(returned, ast.Return)
            or returned.value is None
            or not _is_literal_constant_property_value(returned.value)
        ):
            continue
        defaults.append(
            (
                statement.name,
                statement.lineno,
                _source_segment(module, returned.value),
                (statement.end_lineno or statement.lineno) - statement.lineno + 1,
            )
        )
    return tuple(defaults)


def _constant_property_default_bundle_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[ConstantPropertyDefaultBundleCandidate]:
    defaults = _constant_property_default_methods(module, node)
    if len(defaults) < 4:
        return
    yield ConstantPropertyDefaultBundleCandidate(
        file_path=str(module.path),
        line=defaults[0][1],
        class_name=node.name,
        property_names=tuple((name for name, _, _, _ in defaults)),
        return_expressions=tuple((expression for _, _, expression, _ in defaults)),
        line_count=sum((line_count for _, _, _, line_count in defaults)),
    )


def _constant_property_default_bundle_candidates(
    module: ParsedModule,
) -> tuple[ConstantPropertyDefaultBundleCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _constant_property_default_bundle_candidates_for_class,
    )


def _reflective_self_attribute_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[ReflectiveSelfAttributeCandidate]:
    for statement in node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for subnode in _walk_nodes(statement):
            if not isinstance(subnode, ast.Call):
                continue
            builtin_name = _ast_terminal_name(subnode.func)
            if builtin_name not in _REFLECTIVE_SELF_BUILTINS:
                continue
            if len(subnode.args) < 2:
                continue
            receiver, attribute_name_node = subnode.args[0], subnode.args[1]
            attribute_name = _constant_string(attribute_name_node)
            if not (
                isinstance(receiver, ast.Name)
                and receiver.id == "self"
                and (attribute_name is not None)
            ):
                continue
            yield ReflectiveSelfAttributeCandidate(
                file_path=str(module.path),
                line=subnode.lineno,
                subject_name=node.name,
                name_family=(attribute_name,),
                method_name=statement.name,
                reflective_builtin=builtin_name,
                attribute_name=attribute_name,
            )


def _reflective_self_attribute_candidates(
    module: ParsedModule,
) -> tuple[ReflectiveSelfAttributeCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _reflective_self_attribute_candidates_for_class,
    )


_HELPER_BACKED_METHOD_NAMES = frozenset(
    {
        "build_from_function",
        "build_scoped_function",
        "build_from_assign",
        "build_scoped_assign",
        "build_from_context",
    }
)

_NON_HELPER_CALL_NAMES = frozenset(
    {
        "all",
        "any",
        "bool",
        "dict",
        "frozenset",
        "int",
        "len",
        _BuiltinCollectionName.LIST,
        "max",
        "min",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
    }
)


def _looks_like_helper_call_name(helper_name: str) -> bool:
    terminal = helper_name.rsplit(".", 1)[-1]
    return bool(
        terminal and terminal[0].islower() and (terminal not in _NON_HELPER_CALL_NAMES)
    )


def _helper_call_from_returned_value(node: ast.AST) -> tuple[str, bool] | None:
    tuple_wrapped_call = single_named_call_argument(
        node, call_name=_BuiltinCollectionName.TUPLE, argument_type=ast.Call
    )
    helper_call = tuple_wrapped_call or as_ast(node, ast.Call)
    helper_name = _call_display_name(helper_call) if helper_call is not None else None
    if helper_name is None or not _looks_like_helper_call_name(helper_name):
        return None
    return (helper_name, tuple_wrapped_call is not None)


def _helper_backed_wrapper_kind(
    returned_value: ast.AST,
) -> str | None:
    helper_call = returned_value
    tuple_wrapped = False
    if (
        isinstance(returned_value, ast.Call)
        and isinstance(returned_value.func, ast.Name)
        and returned_value.func.id == "tuple"
        and len(returned_value.args) == 1
        and isinstance(returned_value.args[0], ast.Call)
    ):
        helper_call = returned_value.args[0]
        tuple_wrapped = True
    if not isinstance(helper_call, ast.Call):
        return None
    arguments = [ast.unparse(arg) for arg in helper_call.args]
    arguments.extend(
        (
            f"{keyword.arg}={ast.unparse(keyword.value)}"
            for keyword in helper_call.keywords
            if keyword.arg is not None
        )
    )
    wrapper_prefix = "tuple_wrapped" if tuple_wrapped else "direct"
    if not arguments:
        return wrapper_prefix
    return f"{wrapper_prefix}({', '.join(arguments)})"


def _is_helper_wrapper_prelude(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Assert):
        return True
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        return bool(
            isinstance(target, ast.Name)
            and isinstance(statement.value, ast.Attribute)
            and isinstance(statement.value.value, ast.Name)
            and (statement.value.value.id == "observation")
        )
    if isinstance(statement, ast.If):
        return _if_returns_none_only(statement)
    return False


def _helper_backed_observation_spec_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[HelperBackedObservationSpecCandidate]:
    base_names = _shared_record_base_names(node)
    if not base_names:
        return
    for method in node.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        if method.name.startswith("_"):
            continue
        body = _trim_docstring_body(method.body)
        if not body or len(body) > 4:
            continue
        if not all((_is_helper_wrapper_prelude(statement) for statement in body[:-1])):
            continue
        tail = body[-1]
        if not isinstance(tail, ast.Return) or tail.value is None:
            continue
        helper_result = _helper_call_from_returned_value(tail.value)
        if helper_result is None:
            continue
        helper_name, _ = helper_result
        wrapper_kind = _helper_backed_wrapper_kind(tail.value)
        if wrapper_kind is None:
            continue
        yield HelperBackedObservationSpecCandidate(
            file_path=str(module.path),
            line=method.lineno,
            subject_name=node.name,
            name_family=(method.name, helper_name, wrapper_kind),
            base_names=base_names,
            method_name=method.name,
            helper_name=helper_name,
            wrapper_kind=wrapper_kind,
            parameter_names=_parameter_names(method),
        )


def _helper_backed_observation_spec_candidates(
    module: ParsedModule,
) -> tuple[HelperBackedObservationSpecCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _helper_backed_observation_spec_candidates_for_class,
    )


def _helper_backed_observation_spec_group(
    module: ParsedModule,
) -> HelperBackedObservationSpecGroup | None:
    candidates = _helper_backed_observation_spec_candidates(module)
    grouped: dict[
        tuple[tuple[str, ...], str], list[HelperBackedObservationSpecCandidate]
    ] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.base_names, candidate.method_name)].append(candidate)
    items = max(
        (items for items in grouped.values() if len(items) >= 3), key=len, default=None
    )
    if items is None:
        return None
    ordered = sorted_tuple(items, key=lambda item: (item.line, item.class_name))
    return HelperBackedObservationSpecGroup(
        file_path=str(module.path),
        base_names=ordered[0].base_names,
        class_names=tuple((item.class_name for item in ordered)),
        line_numbers=tuple((item.line for item in ordered)),
        method_names=tuple((item.method_name for item in ordered)),
        helper_names=tuple((item.helper_name for item in ordered)),
        wrapper_kinds=tuple((item.wrapper_kind for item in ordered)),
    )


def _guarded_wrapper_node_types(node: ast.If) -> tuple[str, ...] | None:
    test = node.test
    if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
        return None
    operand = test.operand
    if (
        not isinstance(operand, ast.Call)
        or _ast_terminal_name(operand.func) != "isinstance"
        or len(operand.args) != 2
    ):
        return None
    type_node = operand.args[1]
    if isinstance(type_node, ast.Tuple):
        node_types = tuple(ast.unparse(item) for item in type_node.elts)
    else:
        node_types = (ast.unparse(type_node),)
    return tuple(item for item in node_types if item)


def _guarded_wrapper_function_candidates(
    module: ParsedModule,
) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    candidates: list[tuple[str, int, tuple[str, ...]]] = []
    for statement in module.module.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        body = _trim_docstring_body(statement.body)
        while (
            body
            and isinstance(body[0], ast.Assign)
            and (len(body[0].targets) == 1)
            and isinstance(body[0].targets[0], ast.Name)
        ):
            body = body[1:]
        if len(body) != 2:
            continue
        guard, return_stmt = body
        if not isinstance(guard, ast.If) or not _if_returns_none_only(guard):
            continue
        if not isinstance(return_stmt, ast.Return) or return_stmt.value is None:
            continue
        node_types = _guarded_wrapper_node_types(guard)
        if not node_types:
            continue
        candidates.append((statement.name, statement.lineno, node_types))
    return tuple(candidates)


def _guarded_wrapper_spec_pairs(
    module: ParsedModule,
) -> tuple[GuardedWrapperSpecPair, ...]:
    wrapper_functions = {
        function_name: (lineno, node_types)
        for function_name, lineno, node_types in _guarded_wrapper_function_candidates(
            module
        )
    }
    pairs: list[GuardedWrapperSpecPair] = []
    for statement in module.module.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
            lineno = statement.lineno
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
            lineno = statement.lineno
        else:
            continue
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        constructor_name = _call_name(value.func)
        if constructor_name is None:
            continue
        referenced_functions = [
            keyword.value.id
            for keyword in value.keywords
            if keyword.arg is not None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id in wrapper_functions
        ]
        if len(referenced_functions) != 1:
            continue
        node_types_node = next(
            (
                keyword.value
                for keyword in value.keywords
                if keyword.arg == "node_types"
            ),
            None,
        )
        if node_types_node is None:
            continue
        if isinstance(node_types_node, ast.Tuple):
            node_types = tuple(ast.unparse(item) for item in node_types_node.elts)
        else:
            node_types = (ast.unparse(node_types_node),)
        function_name = referenced_functions[0]
        function_line, function_node_types = wrapper_functions[function_name]
        if tuple(node_types) != function_node_types:
            continue
        pairs.append(
            GuardedWrapperSpecPair(
                file_path=str(module.path),
                spec_name=target.id,
                spec_line=lineno,
                function_name=function_name,
                function_line=function_line,
                constructor_name=constructor_name,
                node_types=function_node_types,
            )
        )
    return tuple(pairs)


def _dynamic_self_field_selection_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[DynamicSelfFieldSelectionCandidate]:
    for statement in node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for subnode in _walk_nodes(statement):
            if not isinstance(subnode, ast.Call):
                continue
            builtin_name = _ast_terminal_name(subnode.func)
            if builtin_name not in _REFLECTIVE_SELF_BUILTINS:
                continue
            if len(subnode.args) < 2:
                continue
            receiver, selector_node = subnode.args[0], subnode.args[1]
            if not isinstance(receiver, ast.Name) or receiver.id != "self":
                continue
            if _constant_string(selector_node) is not None:
                continue
            selector_expression = ast.unparse(selector_node)
            if not any(
                token in selector_expression
                for token in ("self.", "type(self).", "cls.")
            ):
                continue
            yield DynamicSelfFieldSelectionCandidate(
                file_path=str(module.path),
                line=subnode.lineno,
                subject_name=node.name,
                name_family=(selector_expression,),
                method_name=statement.name,
                reflective_builtin=builtin_name,
                selector_expression=selector_expression,
            )


def _dynamic_self_field_selection_candidates(
    module: ParsedModule,
) -> tuple[DynamicSelfFieldSelectionCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _dynamic_self_field_selection_candidates_for_class,
    )


def _class_list_registry_names(node: ast.ClassDef) -> tuple[str, ...]:
    registry_names: list[str] = []
    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and isinstance(statement.value, ast.List):
                registry_names.append(target.id)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and isinstance(statement.value, ast.List)
        ):
            registry_names.append(statement.target.id)
    return sorted_tuple(set(registry_names))


def _registration_append_registry_name(
    node: ast.AST, registry_names: tuple[str, ...], owner_name: str
) -> str | None:
    call = as_ast(node, ast.Call)
    if call is None:
        return None
    append_call = attribute_call_match(
        call,
        method_name=_APPEND_METHOD_NAME,
        owner_type=ast.Attribute,
        single_argument_required=True,
    )
    if append_call is None or append_call.owner.attr not in registry_names:
        return None
    if not _looks_like_cls_registration_value(append_call.single_argument):
        return None
    if name_id(append_call.owner.value) in {"cls", _TYPE_NAME_LITERAL, owner_name}:
        return append_call.owner.attr
    return None


def _looks_like_cls_registration_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "cls"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and (node.func.id == "cast")
        and node.args
    ):
        return _looks_like_cls_registration_value(node.args[-1])
    return False


def _class_dict_get_attr_name(node: ast.AST) -> str | None:
    call = as_ast(node, ast.Call)
    if call is None:
        return None
    dict_get = attribute_call_match(
        call,
        method_name="get",
        owner_type=ast.Attribute,
        owner_name="cls",
        single_argument_required=True,
    )
    if dict_get is None or dict_get.owner.attr != "__dict__":
        return None
    return _constant_string(dict_get.single_argument)


def _guarded_defined_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _class_dict_get_attr_name(node)
    compare = as_ast(node, ast.Compare)
    if compare is None:
        return None
    comparison = single_compare_match(compare, ast.IsNot) or single_compare_match(
        compare, ast.NotEq
    )
    none_constant = as_ast(comparison.right if comparison else None, ast.Constant)
    if comparison is None or none_constant is None or none_constant.value is not None:
        return None
    return _class_dict_get_attr_name(comparison.left)


def _guard_requires_concrete_subclass(node: ast.AST) -> bool:
    if not isinstance(node, ast.UnaryOp) or not isinstance(node.op, ast.Not):
        return False
    operand = node.operand
    return (
        isinstance(operand, ast.Call)
        and isinstance(operand.func, ast.Attribute)
        and isinstance(operand.func.value, ast.Name)
        and operand.func.value.id == "inspect"
        and operand.func.attr == "isabstract"
        and len(operand.args) == 1
        and isinstance(operand.args[0], ast.Name)
        and operand.args[0].id == "cls"
    )


def _manual_subclass_registration_sites(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    registry_names: tuple[str, ...],
    *,
    owner_name: str,
) -> tuple[_ManualSubclassRegistrationSite, ...]:
    sites: dict[str, _ManualSubclassRegistrationSite] = {}

    def walk_statements(
        statements: Sequence[ast.stmt], guard_stack: tuple[ast.AST, ...]
    ) -> None:
        for statement in statements:
            if isinstance(statement, ast.If):
                walk_statements(statement.body, (*guard_stack, statement.test))
                walk_statements(statement.orelse, guard_stack)
                continue
            for subnode in _walk_nodes(statement):
                registry_name = _registration_append_registry_name(
                    subnode, registry_names, owner_name
                )
                if registry_name is None:
                    continue
                guard_summary = (
                    " and ".join(ast.unparse(guard) for guard in guard_stack)
                    if guard_stack
                    else None
                )
                selector_attr_name = next(
                    (
                        attr_name
                        for guard in guard_stack
                        if (attr_name := _guarded_defined_attr_name(guard)) is not None
                    ),
                    None,
                )
                requires_concrete_subclass = any(
                    (_guard_requires_concrete_subclass(guard) for guard in guard_stack)
                )
                sites[registry_name] = _ManualSubclassRegistrationSite(
                    registry_name=registry_name,
                    guard_summary=guard_summary,
                    selector_attr_name=selector_attr_name,
                    requires_concrete_subclass=requires_concrete_subclass,
                )

    walk_statements(_trim_docstring_body(method.body), ())
    return tuple(sites[name] for name in sorted(sites))


def _uses_named_registry(
    node: ast.AST,
    *,
    registry_name: str,
    owner_names: frozenset[str],
) -> bool:
    if not isinstance(node, ast.Attribute) or node.attr != registry_name:
        return False
    if not isinstance(node.value, ast.Name):
        return False
    return node.value.id in owner_names


def _registry_consumer_locations(
    module: ParsedModule,
    node: ast.ClassDef,
    registry_name: str,
) -> tuple[SourceLocation, ...]:
    consumer_locations: list[SourceLocation] = []
    for method in _iter_class_methods(node):
        if method.name == "__init_subclass__":
            continue
        if any(
            (
                _uses_named_registry(
                    subnode,
                    registry_name=registry_name,
                    owner_names=frozenset({"cls", _TYPE_NAME_LITERAL, node.name}),
                )
                for subnode in _walk_nodes(method)
            )
        ):
            consumer_locations.append(
                SourceLocation(
                    str(module.path), method.lineno, f"{node.name}.{method.name}"
                )
            )
    for qualname, function in _iter_named_functions(module):
        if "." in qualname:
            continue
        if any(
            (
                _uses_named_registry(
                    subnode,
                    registry_name=registry_name,
                    owner_names=frozenset({node.name}),
                )
                for subnode in _walk_nodes(function)
            )
        ):
            consumer_locations.append(
                SourceLocation(str(module.path), function.lineno, qualname)
            )
    unique_locations = {
        (location.file_path, location.line, location.symbol): location
        for location in consumer_locations
    }
    return sorted_tuple(
        unique_locations.values(), key=lambda location: (location.line, location.symbol)
    )


def _registered_descendant_classes(
    descendants: tuple[IndexedClass, ...],
    site: _ManualSubclassRegistrationSite,
) -> tuple[IndexedClass, ...]:
    if site.selector_attr_name is not None:
        return tuple(
            (
                descendant
                for descendant in descendants
                if site.selector_attr_name
                in _class_direct_non_none_assignment_names(descendant.node)
            )
        )
    if site.requires_concrete_subclass:
        return tuple(
            (
                descendant
                for descendant in descendants
                if not _is_abstract_class(descendant.node)
            )
        )
    return descendants


def _manual_concrete_subclass_roster_candidates(
    modules: list[ParsedModule], config: DetectorConfig
) -> tuple[ManualConcreteSubclassRosterCandidate, ...]:
    class_index = build_class_family_index(modules)
    modules_by_path = {str(module.path): module for module in modules}
    candidates: list[ManualConcreteSubclassRosterCandidate] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        node = indexed_class.node
        module = modules_by_path.get(indexed_class.file_path)
        if module is None:
            continue
        registry_names = _class_list_registry_names(node)
        if not registry_names:
            continue
        init_subclass = _class_method_named(node, "__init_subclass__")
        if init_subclass is None:
            continue
        descendants = _indexed_descendant_classes(class_index, indexed_class.symbol)
        if len(descendants) < config.min_registration_sites:
            continue
        consumer_locations_by_registry = {
            registry_name: _registry_consumer_locations(module, node, registry_name)
            for registry_name in registry_names
        }
        for site in _manual_subclass_registration_sites(
            init_subclass, registry_names, owner_name=node.name
        ):
            consumer_locations = consumer_locations_by_registry.get(
                site.registry_name, ()
            )
            if not consumer_locations:
                continue
            concrete_descendants = _registered_descendant_classes(descendants, site)
            if len(concrete_descendants) < config.min_registration_sites:
                continue
            candidates.append(
                ManualConcreteSubclassRosterCandidate(
                    file_path=indexed_class.file_path,
                    line=init_subclass.lineno,
                    class_name=_indexed_class_display_name(indexed_class, class_index),
                    registration_site=site,
                    consumer_locations=consumer_locations,
                    concrete_class_names=_indexed_class_display_names(
                        concrete_descendants, class_index
                    ),
                )
            )
    return tuple(candidates)


def _registered_type_match_assignment_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str] | None:
    body = _trim_docstring_body(list(method.body))
    assignment = _registered_type_list_assignment(body)
    if assignment is None:
        return None
    target_name, list_comp = assignment
    generator = _registered_type_list_generator(list_comp)
    if generator is None:
        return None
    predicate_shape = _registered_type_predicate_shape(generator, method)
    if predicate_shape is None:
        return None
    return (target_name, *predicate_shape)


def _registered_type_list_assignment(
    body: list[ast.stmt],
) -> tuple[str, ast.ListComp] | None:
    for statement in body:
        if not isinstance(statement, ast.Assign):
            continue
        binding = named_value_binding(statement)
        list_comp = as_ast(binding.value if binding else None, ast.ListComp)
        if binding is not None and list_comp is not None:
            return (binding.name, list_comp)
    return None


def _registered_type_list_generator(
    list_comp: ast.ListComp,
) -> ast.comprehension | None:
    generator = single_item(list_comp.generators)
    target_name = name_id(generator.target) if generator is not None else None
    if (
        generator is None
        or generator.is_async
        or target_name is None
        or (name_id(list_comp.elt) != target_name)
    ):
        return None
    iter_call = as_ast(generator.iter, ast.Call)
    if iter_call is None:
        return None
    registered_type_call = attribute_call_match(
        iter_call,
        method_name="registered_types",
        owner_type=ast.Name,
        owner_name="cls",
        argument_count=0,
        allow_keywords=False,
    )
    return generator if registered_type_call is not None else None


def _registered_type_predicate_shape(
    generator: ast.comprehension,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str] | None:
    predicate = as_ast(single_item(generator.ifs), ast.Call)
    argument = single_item(predicate.args) if predicate is not None else None
    target_name = name_id(generator.target)
    if not (
        predicate is not None
        and (not predicate.keywords)
        and (target_name is not None)
        and (call_attribute_name(predicate, owner_name=target_name) is not None)
        and (name_id(argument) in _parameter_names(method))
    ):
        return None
    return cast(ast.Attribute, predicate.func).attr, cast(ast.Name, argument).id


def _is_selected_match_subscript(node: ast.AST, match_var_name: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and (node.value.id == match_var_name)
        and isinstance(node.slice, ast.Constant)
        and (node.slice.value == 0)
    )


# fmt: off
materialize_product_record(product_record_spec('_SelectionGuardContext', 'node: ast.AST; match_var_name: str'))
# fmt: on


class _SelectionGuardKindStep(RegisteredEffectStep):
    pass


class _UnaryEmptySelectionGuardStep(
    _SelectionGuardKindStep,
    GuardedEffectStep[_SelectionGuardContext, str],
):
    step_id = "unary_empty_selection_guard"
    registration_order = 10

    def project(self, value: _SelectionGuardContext) -> str | None:
        return _unary_empty_selection_guard_kind(value)


class _LengthCompareSelectionGuardStep(
    _SelectionGuardKindStep,
    GuardedEffectStep[_SelectionGuardContext, str],
):
    step_id = "length_compare_selection_guard"
    registration_order = 20

    def project(self, value: _SelectionGuardContext) -> str | None:
        return _length_compare_selection_guard_kind(value)


def _unary_empty_selection_guard_kind(context: _SelectionGuardContext) -> str | None:
    unary = as_ast(context.node, ast.UnaryOp)
    if (
        unary is None
        or not isinstance(unary.op, ast.Not)
        or name_id(unary.operand) != context.match_var_name
    ):
        return None
    return "empty"


def _selection_len_arg_name(compare: ast.Compare) -> str | None:
    length_call = as_ast(compare.left, ast.Call)
    if length_call is None or _ast_terminal_name(length_call.func) != "len":
        return None
    return name_id(single_item(length_call.args))


def _selection_int_comparator(compare: ast.Compare) -> int | None:
    comparator = single_ast(compare.comparators, ast.Constant)
    if comparator is None:
        return None
    return comparator.value if isinstance(comparator.value, int) else None


def _selection_compare_operator(compare: ast.Compare) -> ast.cmpop | None:
    return single_item(compare.ops)


_SELECTION_LENGTH_GUARD_KINDS: tuple[(tuple[type[ast.cmpop], int, str], ...)] = (
    (ast.NotEq, 1, "not_exactly_one"),
    (ast.Gt, 1, "ambiguous"),
    (ast.Eq, 0, "empty"),
)


def _length_compare_selection_guard_kind(
    context: _SelectionGuardContext,
) -> str | None:
    compare = as_ast(context.node, ast.Compare)
    if compare is None or _selection_len_arg_name(compare) != context.match_var_name:
        return None
    operator = _selection_compare_operator(compare)
    comparator_value = _selection_int_comparator(compare)
    return next(
        (
            guard_kind
            for operator_type, expected_value, guard_kind in _SELECTION_LENGTH_GUARD_KINDS
            if isinstance(operator, operator_type)
            and comparator_value == expected_value
        ),
        None,
    )


def _selection_guard_kind(node: ast.AST, match_var_name: str) -> str | None:
    return cast(
        str | None,
        Maybe.of(_SelectionGuardContext(node, match_var_name))
        .bind(
            FirstSuccessfulEffectStep(registered_effect_steps(_SelectionGuardKindStep))
        )
        .unwrap_or_none(),
    )


def _predicate_selected_concrete_family_candidates(
    modules: list[ParsedModule], config: DetectorConfig
) -> tuple[PredicateSelectedConcreteFamilyCandidate, ...]:
    class_index = build_class_family_index(modules)
    candidates: list[PredicateSelectedConcreteFamilyCandidate] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        node = indexed_class.node
        assignments = _class_direct_assignments(node)
        if "_registered_types" not in assignments:
            continue
        descendants = tuple(
            (
                descendant
                for descendant in _indexed_descendant_classes(
                    class_index, indexed_class.symbol
                )
                if not _is_abstract_class(descendant.node)
            )
        )
        if len(descendants) < config.min_registration_sites:
            continue
        for method in _iter_class_methods(node):
            if not _is_classmethod(method):
                continue
            selection_shape = _registered_type_match_assignment_shape(method)
            if selection_shape is None:
                continue
            match_var_name, predicate_method_name, context_param_name = selection_shape
            guard_kinds = {
                _selection_guard_kind(statement.test, match_var_name)
                for statement in _trim_docstring_body(list(method.body))
                if isinstance(statement, ast.If)
            }
            has_exact_guard = "not_exactly_one" in guard_kinds or (
                "empty" in guard_kinds and "ambiguous" in guard_kinds
            )
            if not has_exact_guard:
                continue
            if not any(
                (
                    _is_selected_match_subscript(subnode, match_var_name)
                    for subnode in _walk_nodes(method)
                )
            ):
                continue
            candidates.append(
                PredicateSelectedConcreteFamilyCandidate(
                    file_path=indexed_class.file_path,
                    line=method.lineno,
                    class_name=_indexed_class_display_name(indexed_class, class_index),
                    selector_method_name=method.name,
                    predicate_method_name=predicate_method_name,
                    context_param_name=context_param_name,
                    concrete_class_names=_indexed_class_display_names(
                        descendants, class_index
                    ),
                )
            )
    return tuple(candidates)


def _mirrored_leaf_family_map(
    descendants: tuple[IndexedClass, ...],
    *,
    axis_prefix_tokens: tuple[str, ...],
) -> dict[str, IndexedClass]:
    leaf_map: dict[str, IndexedClass] = {}
    for descendant in descendants:
        tokens = _ordered_class_name_tokens(descendant.simple_name)
        if (
            len(tokens) <= len(axis_prefix_tokens)
            or tokens[: len(axis_prefix_tokens)] != axis_prefix_tokens
        ):
            continue
        family_tokens = tokens[len(axis_prefix_tokens) :]
        if not family_tokens:
            continue
        family_name = " ".join(family_tokens)
        leaf_map.setdefault(family_name, descendant)
    return leaf_map


def _parallel_mirrored_leaf_family_candidates(
    modules: list[ParsedModule], config: DetectorConfig
) -> tuple[ParallelMirroredLeafFamilyCandidate, ...]:
    class_index = build_class_family_index(modules)
    min_shared_families = max(3, config.min_registration_sites)
    root_candidates: list[
        tuple[IndexedClass, tuple[str, ...], tuple[IndexedClass, ...]]
    ] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        assignments = _class_direct_assignments(indexed_class.node)
        if "_registered_types" not in assignments:
            continue
        abstract_methods = _abstract_method_names(indexed_class.node)
        if not abstract_methods:
            continue
        concrete_descendants = tuple(
            (
                descendant
                for descendant in _indexed_descendant_classes(
                    class_index, indexed_class.symbol
                )
                if not _is_abstract_class(descendant.node)
            )
        )
        if len(concrete_descendants) < min_shared_families:
            continue
        root_candidates.append((indexed_class, abstract_methods, concrete_descendants))

    candidates: list[ParallelMirroredLeafFamilyCandidate] = []
    for (left_root, left_contract_methods, left_descendants), (
        right_root,
        right_contract_methods,
        right_descendants,
    ) in combinations(root_candidates, 2):
        shared_contract_methods = sorted_tuple(
            set(left_contract_methods) & set(right_contract_methods)
        )
        if not shared_contract_methods:
            continue
        left_tokens = _ordered_class_name_tokens(left_root.simple_name)
        right_tokens = _ordered_class_name_tokens(right_root.simple_name)
        shared_root_suffix = _shared_ordered_suffix(left_tokens, right_tokens)
        if not shared_root_suffix:
            continue
        left_axis_prefix = left_tokens[: len(left_tokens) - len(shared_root_suffix)]
        right_axis_prefix = right_tokens[: len(right_tokens) - len(shared_root_suffix)]
        if (
            not left_axis_prefix
            or not right_axis_prefix
            or left_axis_prefix == right_axis_prefix
        ):
            continue
        left_leaf_map = _mirrored_leaf_family_map(
            left_descendants, axis_prefix_tokens=left_axis_prefix
        )
        right_leaf_map = _mirrored_leaf_family_map(
            right_descendants, axis_prefix_tokens=right_axis_prefix
        )
        if not left_leaf_map or not right_leaf_map:
            continue
        shared_leaf_families = sorted_tuple(set(left_leaf_map) & set(right_leaf_map))
        if len(shared_leaf_families) < max(
            min_shared_families, min(len(left_leaf_map), len(right_leaf_map)) // 2
        ):
            continue
        left_leaf_evidence = tuple(
            (
                SourceLocation(
                    left_leaf_map[family_name].file_path,
                    left_leaf_map[family_name].line,
                    _indexed_class_display_name(
                        left_leaf_map[family_name], class_index
                    ),
                )
                for family_name in shared_leaf_families
            )
        )
        right_leaf_evidence = tuple(
            (
                SourceLocation(
                    right_leaf_map[family_name].file_path,
                    right_leaf_map[family_name].line,
                    _indexed_class_display_name(
                        right_leaf_map[family_name], class_index
                    ),
                )
                for family_name in shared_leaf_families
            )
        )
        candidates.append(
            ParallelMirroredLeafFamilyCandidate(
                left=MirroredLeafFamilySide(
                    file_path=left_root.file_path,
                    line=left_root.line,
                    root_name=_indexed_class_display_name(left_root, class_index),
                    leaf_evidence=left_leaf_evidence,
                ),
                right=MirroredLeafFamilySide(
                    file_path=right_root.file_path,
                    line=right_root.line,
                    root_name=_indexed_class_display_name(right_root, class_index),
                    leaf_evidence=right_leaf_evidence,
                ),
                contract_method_names=shared_contract_methods,
                shared_leaf_family_names=shared_leaf_families,
            )
        )
    return tuple(candidates)


def _reflective_lookup_shape(
    node: ast.AST,
) -> tuple[str, str, ast.AST] | None:
    if isinstance(node, ast.Call):
        builtin_name = _ast_terminal_name(node.func)
        if builtin_name == _GETATTR_BUILTIN and len(node.args) >= 2:
            selector_node = node.args[1]
            if _constant_string(selector_node) is None:
                return (_GETATTR_BUILTIN, ast.unparse(node.args[0]), selector_node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) >= 1
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "__dict__"
        ):
            selector_node = node.args[0]
            if _constant_string(selector_node) is None:
                return ("dict.get", ast.unparse(node.func.value.value), selector_node)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and (node.value.func.id in {"globals", "locals"})
        and (not node.value.args)
        and (not node.value.keywords)
        and (_constant_string(node.slice) is None)
    ):
        return (f"{node.value.func.id}[]", f"{node.value.func.id}()", node.slice)
    return None


def _string_backed_reflective_nominal_lookup_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[StringBackedReflectiveNominalLookupCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    class_string_assignments = {
        class_name: _class_direct_constant_string_assignments(node)
        for class_name, node in class_defs_by_name.items()
    }
    candidate_map: dict[
        tuple[str, str, str, str, str], StringBackedReflectiveNominalLookupCandidate
    ] = {}
    for class_name, node in sorted(class_defs_by_name.items()):
        descendants = _descendant_class_names(class_defs_by_name, class_name)
        if len(descendants) < config.min_reflective_selector_values:
            continue
        for method in _iter_class_methods(node):
            for subnode in _walk_nodes(method):
                lookup_shape = _reflective_lookup_shape(subnode)
                if lookup_shape is None:
                    continue
                lookup_kind, receiver_expression, selector_node = lookup_shape
                selector_attr_name = _selector_attribute_name(selector_node)
                if selector_attr_name is None:
                    continue
                concrete_class_names = tuple(
                    (
                        descendant
                        for descendant in descendants
                        if selector_attr_name in class_string_assignments[descendant]
                    )
                )
                if len(concrete_class_names) < config.min_reflective_selector_values:
                    continue
                selector_values = sorted_tuple(
                    {
                        class_string_assignments[descendant][selector_attr_name]
                        for descendant in concrete_class_names
                    }
                )
                if len(selector_values) < config.min_reflective_selector_values:
                    continue
                candidate = StringBackedReflectiveNominalLookupCandidate(
                    file_path=str(module.path),
                    line=subnode.lineno,
                    class_name=class_name,
                    method_name=method.name,
                    selector_attr_name=selector_attr_name,
                    lookup_kind=lookup_kind,
                    receiver_expression=receiver_expression,
                    concrete_class_names=concrete_class_names,
                    selector_values=selector_values,
                )
                candidate_map[
                    (
                        class_name,
                        method.name,
                        selector_attr_name,
                        lookup_kind,
                        receiver_expression,
                    )
                ] = candidate
    return sorted_tuple(
        candidate_map.values(),
        key=lambda item: (item.file_path, item.line, item.class_name, item.method_name),
    )


def _param_backed_name(expr: ast.AST, parameter_names: set[str]) -> str | None:
    if isinstance(expr, ast.Name) and expr.id in parameter_names:
        return expr.id
    if isinstance(expr, ast.IfExp):
        body_name = _param_backed_name(expr.body, parameter_names)
        orelse_name = _param_backed_name(expr.orelse, parameter_names)
        if body_name is not None and orelse_name is None:
            return body_name
        if orelse_name is not None and body_name is None:
            return orelse_name
        if body_name == orelse_name:
            return body_name
    if isinstance(expr, ast.BoolOp):
        names = {
            name
            for value in expr.values
            for name in (_param_backed_name(value, parameter_names),)
            if name is not None
        }
        if len(names) == 1:
            return next(iter(names))
    return None


def _class_init_concrete_param_backed_attrs(node: ast.ClassDef) -> dict[str, str]:
    init_method = _class_method_named(node, "__init__")
    if init_method is None:
        return {}
    parameter_type_names = {
        argument.arg: _annotation_type_names(argument.annotation)
        for argument in (
            tuple(init_method.args.posonlyargs)
            + tuple(init_method.args.args)
            + tuple(init_method.args.kwonlyargs)
        )
        if argument.annotation is not None
    }
    parameter_names = set(parameter_type_names)
    attr_type_names: dict[str, str] = {}
    for subnode in _walk_nodes(init_method):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(subnode, ast.Assign) and len(subnode.targets) == 1:
            target = subnode.targets[0]
            value = subnode.value
        elif isinstance(subnode, ast.AnnAssign):
            target = subnode.target
            value = subnode.value
        attr_name = None if target is None else _self_attr_name(target)
        if attr_name is None or value is None:
            continue
        param_name = _param_backed_name(value, parameter_names)
        if param_name is None:
            continue
        type_names = parameter_type_names.get(param_name, ())
        if len(type_names) != 1:
            continue
        attr_type_names.setdefault(attr_name, type_names[0])
    return attr_type_names


def _method_aliases_to_self_attrs(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for subnode in _walk_nodes(method):
            target: ast.AST | None = None
            value: ast.AST | None = None
            if isinstance(subnode, ast.Assign) and len(subnode.targets) == 1:
                target = subnode.targets[0]
                value = subnode.value
            elif isinstance(subnode, ast.AnnAssign):
                target = subnode.target
                value = subnode.value
            if not (isinstance(target, ast.Name) and value is not None):
                continue
            attr_name = None
            if isinstance(value, ast.Attribute):
                attr_name = _self_attr_name(value)
            elif isinstance(value, ast.Name):
                attr_name = aliases.get(value.id)
            if attr_name is None or aliases.get(target.id) == attr_name:
                continue
            aliases[target.id] = attr_name
            changed = True
    return aliases


def _receiver_self_attr_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Attribute):
        return _self_attr_name(node)
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    return None


def _concrete_config_field_probe_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[ConcreteConfigFieldProbeCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    config_field_names = {
        class_name: {field_name for field_name, _ in _typed_field_map(node)}
        for class_name, node in class_defs_by_name.items()
    }
    candidates: list[ConcreteConfigFieldProbeCandidate] = []
    for class_name, node in sorted(class_defs_by_name.items()):
        concrete_config_attrs = _class_init_concrete_param_backed_attrs(node)
        if not concrete_config_attrs:
            continue
        for method in _iter_class_methods(node):
            aliases = _method_aliases_to_self_attrs(method)
            grouped_missing_fields: dict[tuple[str, str], set[str]] = defaultdict(set)
            grouped_probe_builtins: dict[tuple[str, str], set[str]] = defaultdict(set)
            grouped_lines: dict[tuple[str, str], int] = {}
            for subnode in _walk_nodes(method):
                if not isinstance(subnode, ast.Call):
                    continue
                builtin_name = _ast_terminal_name(subnode.func)
                if (
                    builtin_name not in {_GETATTR_BUILTIN, _HASATTR_BUILTIN}
                    or len(subnode.args) < 2
                ):
                    continue
                probed_field_name = _constant_string(subnode.args[1])
                if probed_field_name is None:
                    continue
                config_attr_name = _receiver_self_attr_name(subnode.args[0], aliases)
                if config_attr_name is None:
                    continue
                config_type_name = concrete_config_attrs.get(config_attr_name)
                if config_type_name is None:
                    continue
                config_node = class_defs_by_name.get(config_type_name)
                if (
                    config_node is None
                    or _class_method_named(config_node, "__getattr__") is not None
                ):
                    continue
                declared_field_names = config_field_names.get(config_type_name, set())
                if (
                    not declared_field_names
                    or probed_field_name in declared_field_names
                ):
                    continue
                key = (config_attr_name, config_type_name)
                grouped_missing_fields[key].add(probed_field_name)
                grouped_probe_builtins[key].add(builtin_name)
                grouped_lines.setdefault(key, subnode.lineno)
            for (config_attr_name, config_type_name), missing_fields in sorted(
                grouped_missing_fields.items()
            ):
                if len(missing_fields) < config.min_attribute_probes:
                    continue
                candidates.append(
                    ConcreteConfigFieldProbeCandidate(
                        file_path=str(module.path),
                        line=grouped_lines[config_attr_name, config_type_name],
                        class_name=class_name,
                        method_name=method.name,
                        config_attr_name=config_attr_name,
                        config_type_name=config_type_name,
                        missing_field_names=sorted_tuple(missing_fields),
                        probe_builtin_names=sorted_tuple(
                            grouped_probe_builtins[config_attr_name, config_type_name]
                        ),
                    )
                )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line, item.class_name, item.method_name),
    )


_DECLARATIVE_FAMILY_ASSIGNMENT_NAMES = frozenset(
    {"item_type", "spec", "spec_root", "literal_kind"}
)
_DECLARATIVE_FAMILY_DEFINITION_BASE_NAMES = frozenset(
    {
        "SingleShapeFamilyDefinition",
        "RegisteredShapeFamilyDefinition",
        "RegisteredObservationFamilyDefinition",
        "TypedLiteralObservationFamilyDefinition",
    }
)


def _is_declarative_class_value(node: ast.AST) -> bool:
    if isinstance(node, (ast.Name, ast.Attribute, ast.Constant)):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all((_is_declarative_class_value(item) for item in node.elts))
    if isinstance(node, ast.Dict):
        return all(
            (
                key is not None
                and _is_declarative_class_value(key)
                and _is_declarative_class_value(value)
                for key, value in zip(node.keys, node.values, strict=True)
            )
        )
    if isinstance(node, ast.Call):
        return (
            _is_declarative_class_value(node.func)
            and all((_is_declarative_class_value(item) for item in node.args))
            and all(
                (
                    keyword.arg is not None
                    and _is_declarative_class_value(keyword.value)
                    for keyword in node.keywords
                )
            )
        )
    if isinstance(node, ast.Subscript):
        return _is_declarative_class_value(node.value) and _is_declarative_class_value(
            node.slice
        )
    if isinstance(node, ast.UnaryOp):
        return _is_declarative_class_value(node.operand)
    return False


def _metadata_only_class_assignment_names(node: ast.ClassDef) -> tuple[str, ...] | None:
    assigned_names: list[str] = []
    for statement in _trim_docstring_body(node.body):
        if isinstance(statement, ast.Pass):
            continue
        binding = named_value_binding(statement)
        if (
            binding is None
            or binding.value is None
            or (not _is_declarative_class_value(binding.value))
        ):
            return None
        assigned_names.append(binding.name)
    return tuple(assigned_names)


def _nominal_class_name_suffixes(class_name: str) -> tuple[str, ...]:
    tokens = re.findall(
        "[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+", class_name.lstrip("_")
    )
    return tuple(("".join(tokens[index:]) for index in range(len(tokens) - 1)))


def _metadata_only_class_family_candidates(
    module: ParsedModule,
) -> tuple[MetadataOnlyClassFamilyCandidate, ...]:
    grouped: dict[
        str,
        list[tuple[ast.ClassDef, tuple[str, ...], tuple[str, ...], int]],
    ] = defaultdict(list)
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef) or node.decorator_list:
            continue
        assigned_names = _metadata_only_class_assignment_names(node)
        if not assigned_names:
            continue
        base_names = _declared_base_names(node)
        if not base_names:
            continue
        line_count = (node.end_lineno or node.lineno) - node.lineno + 1
        for suffix in _nominal_class_name_suffixes(node.name):
            grouped[suffix].append((node, base_names, assigned_names, line_count))

    selected_class_names: set[str] = set()
    candidates: list[MetadataOnlyClassFamilyCandidate] = []
    ordered_groups = sorted_tuple(
        (
            (suffix, tuple(items))
            for suffix, items in grouped.items()
            if len(items) >= 3
        ),
        key=lambda item: (
            -len(item[1]),
            -sum((candidate[3] for candidate in item[1])),
            item[0],
        ),
    )
    for suffix, items in ordered_groups:
        ordered_items = sorted_tuple(
            items, key=lambda item: (item[0].lineno, item[0].name)
        )
        class_names = tuple((item[0].name for item in ordered_items))
        if set(class_names) <= selected_class_names:
            continue
        selected_class_names.update(class_names)
        line_numbers = tuple((item[0].lineno for item in ordered_items))
        base_name_families = sorted_tuple(
            {item[1] for item in ordered_items},
            key=lambda names: (len(names), names),
        )
        assigned_names = sorted_tuple(
            {name for _, _, names, _ in ordered_items for name in names}
        )
        candidates.append(
            MetadataOnlyClassFamilyCandidate(
                file_path=str(module.path),
                family_suffix=suffix,
                class_names=class_names,
                line_numbers=line_numbers,
                base_name_families=base_name_families,
                assigned_names=assigned_names,
                line_count=sum((item[3] for item in ordered_items)),
            )
        )
    return tuple(candidates)


def _self_naming_builder_catalog_candidates(
    module: ParsedModule,
) -> tuple[SelfNamingBuilderCatalogCandidate, ...]:
    grouped: dict[tuple[str, int, tuple[str, ...]], list[tuple[str, int, int]]] = (
        defaultdict(list)
    )
    for statement in _trim_docstring_body(module.module.body):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        call = statement.value
        if not isinstance(call, ast.Call) or len(call.args) < 2:
            continue
        builder_name = _ast_terminal_name(call.func)
        if builder_name is None:
            continue
        first_arg = call.args[0]
        if (
            not isinstance(first_arg, ast.Constant)
            or not isinstance(first_arg.value, str)
            or first_arg.value != target.id
        ):
            continue
        keyword_names = tuple(
            (keyword.arg for keyword in call.keywords if keyword.arg is not None)
        )
        if len(keyword_names) != len(call.keywords):
            continue
        line_count = (statement.end_lineno or statement.lineno) - statement.lineno + 1
        grouped[builder_name, len(call.args), keyword_names].append(
            (target.id, statement.lineno, line_count)
        )
    return tuple(
        (
            SelfNamingBuilderCatalogCandidate(
                file_path=str(module.path),
                class_names=tuple((item[0] for item in ordered)),
                line_numbers=tuple((item[1] for item in ordered)),
                builder_name=builder_name,
                positional_arg_count=positional_arg_count,
                keyword_names=keyword_names,
                line_count=sum((item[2] for item in ordered)),
            )
            for (builder_name, positional_arg_count, keyword_names), items in sorted(
                grouped.items()
            )
            if len(items) >= 3
            for ordered in [sorted_tuple(items, key=lambda item: (item[1], item[0]))]
        )
    )


_COMPOSABLE_BASE_NAME_SUFFIXES = (
    "ABC",
    "Base",
    "Carrier",
    "Contract",
    "Mixin",
    "Template",
)


def _is_composable_base_name(base_name: str) -> bool:
    return base_name == "ABC" or base_name.endswith(_COMPOSABLE_BASE_NAME_SUFFIXES)


def _declared_base_name_sequence(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        (
            base_name
            for base_name in (_ast_terminal_name(base) for base in node.bases)
            if base_name is not None
        )
    )


def _contiguous_base_bundles(
    base_names: tuple[str, ...],
    *,
    minimum_width: int = 3,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        (
            base_names[start:end]
            for start in range(len(base_names))
            for end in range(start + minimum_width, len(base_names) + 1)
            if all(_is_composable_base_name(name) for name in base_names[start:end])
        )
    )


def _is_contiguous_subtuple(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    if len(needle) > len(haystack):
        return False
    return any(
        haystack[start : start + len(needle)] == needle
        for start in range(len(haystack) - len(needle) + 1)
    )


def _maximal_repeated_base_bundle_items(
    grouped: dict[tuple[str, ...], list[ast.ClassDef]],
) -> tuple[tuple[tuple[str, ...], tuple[ast.ClassDef, ...]], ...]:
    qualified = tuple(
        (
            (bundle, sorted_tuple(nodes, key=lambda node: node.lineno))
            for bundle, nodes in grouped.items()
            if len(nodes) >= 3
        )
    )
    maximal: list[tuple[tuple[str, ...], tuple[ast.ClassDef, ...]]] = []
    for bundle, nodes in sorted(
        qualified,
        key=lambda item: (-len(item[1]), -len(item[0]), item[0]),
    ):
        class_names = {node.name for node in nodes}
        if any(
            (
                class_names <= {node.name for node in existing_nodes}
                and _is_contiguous_subtuple(bundle, existing_bundle)
                for existing_bundle, existing_nodes in maximal
            )
        ):
            continue
        maximal.append((bundle, nodes))
    return tuple(maximal)


def _repeated_base_bundle_candidates(
    module: ParsedModule,
) -> tuple[RepeatedBaseBundleCandidate, ...]:
    grouped: dict[tuple[str, ...], list[ast.ClassDef]] = defaultdict(list)
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or node.end_lineno is None:
            continue
        for bundle in _contiguous_base_bundles(_declared_base_name_sequence(node)):
            grouped[bundle].append(node)
    candidates: list[RepeatedBaseBundleCandidate] = []
    for bundle, nodes in _maximal_repeated_base_bundle_items(grouped):
        line_numbers = tuple((node.lineno for node in nodes))
        candidates.append(
            RepeatedBaseBundleCandidate(
                file_path=str(module.path),
                class_names=tuple((node.name for node in nodes)),
                line_numbers=line_numbers,
                base_names=bundle,
                bundle_width=len(bundle),
                class_count=len(nodes),
                line_count=sum(
                    (
                        (node.end_lineno or node.lineno) - node.lineno + 1
                        for node in nodes
                    )
                ),
            )
        )
    return tuple(candidates)


def _module_alias_assignments(module: ParsedModule) -> dict[str, tuple[str, int, str]]:
    aliases: dict[str, tuple[str, int, str]] = {}
    for statement in _trim_docstring_body(module.module.body):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = statement.value
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "family_type"
            and isinstance(value.value, ast.Name)
        ):
            continue
        aliases[target.id] = (value.value.id, statement.lineno, value.attr)
    return aliases


def _module_string_sequence_assignments(
    module: ParsedModule,
) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    assignments: list[tuple[str, int, tuple[str, ...]]] = []
    for statement in _trim_docstring_body(module.module.body):
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
        if target_name is None or value is None:
            continue
        if not isinstance(value, (ast.Tuple, ast.List)):
            continue
        string_items = tuple(
            (
                item.value
                for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
        )
        if len(string_items) != len(value.elts) or len(string_items) < 3:
            continue
        assignments.append((target_name, statement.lineno, string_items))
    return tuple(assignments)


def _is_simple_classvar_value(node: ast.AST) -> bool:
    if isinstance(node, (ast.Name, ast.Attribute, ast.Constant)):
        return True
    if isinstance(node, ast.Tuple):
        return all((_is_simple_classvar_value(item) for item in node.elts))
    return False


def _classvar_assignment_names(node: ast.ClassDef) -> tuple[str, ...] | None:
    assigned_names: list[str] = []
    for statement in _trim_docstring_body(node.body):
        binding = named_value_binding(statement)
        if (
            binding is None
            or binding.value is None
            or (not _is_simple_classvar_value(binding.value))
        ):
            return None
        assigned_names.append(binding.name)
    return tuple(assigned_names)


def _classvar_only_sibling_leaf_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[DeclarativeFamilyLeafCandidate]:
    base_names = tuple(
        name
        for name in _declared_base_names(node)
        if name not in _IGNORED_ANCESTOR_NAMES
    )
    if not base_names:
        return
    assigned_names = _classvar_assignment_names(node)
    if assigned_names is None:
        return
    if len(assigned_names) < 1 or len(assigned_names) > 4:
        return
    if len(_trim_docstring_body(node.body)) != len(assigned_names):
        return
    yield DeclarativeFamilyLeafCandidate(
        file_path=str(module.path),
        line=node.lineno,
        subject_name=node.name,
        name_family=assigned_names,
        base_names=base_names,
        assigned_names=assigned_names,
    )


def _classvar_only_sibling_leaf_candidates(
    module: ParsedModule,
) -> tuple[DeclarativeFamilyLeafCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _classvar_only_sibling_leaf_candidates_for_class,
    )


def _classvar_only_sibling_leaf_groups(
    module: ParsedModule,
) -> tuple[DeclarativeFamilyBoilerplateGroup, ...]:
    grouped: dict[
        (tuple[tuple[str, ...], tuple[str, ...]], list[DeclarativeFamilyLeafCandidate])
    ] = defaultdict(list)
    for candidate in _classvar_only_sibling_leaf_candidates(module):
        grouped[candidate.base_names, candidate.assigned_names].append(candidate)
    return tuple(
        (
            DeclarativeFamilyBoilerplateGroup(
                file_path=str(module.path),
                base_names=base_names,
                assigned_names=assigned_names,
                class_names=tuple((item.subject_name for item in items)),
                line_numbers=tuple((item.line for item in items)),
            )
            for (base_names, assigned_names), items in sorted(grouped.items())
            if len(items) >= 3
        )
    )


def _type_indexed_definition_boilerplate_groups(
    module: ParsedModule,
) -> tuple[TypeIndexedDefinitionBoilerplateGroup, ...]:
    alias_assignments = _module_alias_assignments(module)
    grouped: dict[
        (tuple[tuple[str, ...], tuple[str, ...]], list[tuple[str, str, int]])
    ] = defaultdict(list)
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef) or not node.name.endswith("Definition"):
            continue
        base_names = tuple(
            (
                name
                for name in _declared_base_names(node)
                if name not in _IGNORED_ANCESTOR_NAMES
            )
        )
        if not base_names or not any(
            (name.endswith("Definition") for name in base_names)
        ):
            continue
        assigned_names = _classvar_assignment_names(node)
        if assigned_names is None:
            continue
        if not set(assigned_names) & _DECLARATIVE_FAMILY_ASSIGNMENT_NAMES:
            continue
        alias_name = next(
            (
                alias_name
                for alias_name, (
                    definition_name,
                    _,
                    attr_name,
                ) in alias_assignments.items()
                if definition_name == node.name and attr_name == "family_type"
            ),
            None,
        )
        if alias_name is None:
            continue
        grouped[base_names, assigned_names].append((node.name, alias_name, node.lineno))
    return tuple(
        (
            TypeIndexedDefinitionBoilerplateGroup(
                file_path=str(module.path),
                base_names=base_names,
                definition_class_names=tuple((item[0] for item in ordered)),
                alias_names=tuple((item[1] for item in ordered)),
                line_numbers=tuple((item[2] for item in ordered)),
                assigned_names=assigned_names,
            )
            for (base_names, assigned_names), items in sorted(grouped.items())
            if len(items) >= 3
            for ordered in [sorted_tuple(items, key=lambda item: (item[2], item[0]))]
        )
    )


def _derivable_nominal_root_names(
    shapes: Sequence[NominalAuthorityShape],
) -> tuple[str, ...]:
    root_counts: Counter[str] = Counter()
    for shape in shapes:
        root_counts.update(
            (
                name
                for name in {*shape.declared_base_names, *shape.ancestor_names}
                if name not in _IGNORED_ANCESTOR_NAMES and name != shape.class_name
            )
        )
    return sorted_tuple(
        (root_name for root_name, count in root_counts.items() if count >= 3)
    )


def _derived_export_surface_candidates(
    module: ParsedModule,
) -> tuple[DerivedExportSurfaceCandidate, ...]:
    index = NominalAuthorityIndex((module,))
    candidates: list[DerivedExportSurfaceCandidate] = []
    for export_symbol, line, exported_names in _module_string_sequence_assignments(
        module
    ):
        local_shapes = [
            shapes[0]
            for exported_name in exported_names
            if (shapes := index.shapes_named(exported_name))
            and shapes[0].file_path == str(module.path)
        ]
        if len(local_shapes) < 6 or len(local_shapes) * 5 < len(exported_names) * 4:
            continue
        root_names = _derivable_nominal_root_names(local_shapes)
        if not root_names:
            continue
        candidates.append(
            DerivedExportSurfaceCandidate(
                file_path=str(module.path),
                export_symbol=export_symbol,
                line=line,
                exported_names=exported_names,
                derivable_root_names=root_names,
            )
        )
    return tuple(candidates)


def _module_public_source_names(module: ParsedModule) -> tuple[str, ...]:
    names: set[str] = set()
    for statement in _trim_docstring_body(module.module.body):
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not statement.name.startswith("_"):
                names.add(statement.name)
            continue
        if isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                exported_name = alias.asname or alias.name
                if not exported_name.startswith("_"):
                    names.add(exported_name)
            continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and (not target.id.startswith("_")):
                names.add(target.id)
    return sorted_tuple(names)


def _manual_public_api_surface_candidates(
    module: ParsedModule,
) -> tuple[ManualPublicApiSurfaceCandidate, ...]:
    public_source_names = set(_module_public_source_names(module))
    candidates: list[ManualPublicApiSurfaceCandidate] = []
    for export_symbol, line, exported_names in _module_string_sequence_assignments(
        module
    ):
        if export_symbol != "__all__":
            continue
        if len(exported_names) < 4:
            continue
        if not set(exported_names) <= public_source_names:
            continue
        candidates.append(
            ManualPublicApiSurfaceCandidate(
                file_path=str(module.path),
                export_symbol=export_symbol,
                line=line,
                exported_names=exported_names,
                source_name_count=len(public_source_names),
            )
        )
    return tuple(candidates)


def _dict_key_kind(value: ast.AST) -> str | None:
    if isinstance(value, ast.Name):
        return "type_name"
    if isinstance(value, ast.Attribute):
        return "enum_member"
    if isinstance(value, ast.Constant):
        return type(value.value).__name__
    return None


def _derived_indexed_surface_candidates(
    module: ParsedModule,
) -> tuple[DerivedIndexedSurfaceCandidate, ...]:
    index = NominalAuthorityIndex((module,))
    candidates: list[DerivedIndexedSurfaceCandidate] = []
    for statement in _trim_docstring_body(module.module.body):
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
        if target_name is None or not isinstance(value, ast.Dict):
            continue
        if len(value.keys) < 3 or len(value.keys) != len(value.values):
            continue
        key_kinds = {
            key_kind
            for key_kind in (
                _dict_key_kind(key) for key in value.keys if key is not None
            )
            if key_kind is not None
        }
        if len(key_kinds) != 1:
            continue
        value_names = tuple(
            (
                item.id
                for item in value.values
                if isinstance(item, ast.Name)
                and (shapes := index.shapes_named(item.id))
                and (shapes[0].file_path == str(module.path))
            )
        )
        if len(value_names) != len(value.values):
            continue
        local_shapes = [index.shapes_named(value_name)[0] for value_name in value_names]
        shared_roots = _derivable_nominal_root_names(local_shapes)
        if not shared_roots:
            continue
        candidates.append(
            DerivedIndexedSurfaceCandidate(
                file_path=str(module.path),
                surface_name=target_name,
                line=statement.lineno,
                key_kind=next(iter(key_kinds)),
                value_names=value_names,
                derivable_root_names=shared_roots,
            )
        )
    return tuple(candidates)


def _registered_surface_roots(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    calls: list[ast.Call] = []

    def collect_calls(current: ast.AST) -> bool:
        binary = as_ast(current, ast.BinOp)
        if binary is not None and isinstance(binary.op, ast.Add):
            return collect_calls(binary.left) and collect_calls(binary.right)
        call = as_ast(current, ast.Call)
        if call is None:
            return False
        calls.append(call)
        return True

    if not collect_calls(node) or len(calls) < 2:
        return None
    accessors = tuple(
        (
            accessor
            for call in calls
            if (
                accessor := attribute_call_match(
                    call, owner_type=ast.Name, argument_count=0, allow_keywords=False
                )
            )
            is not None
        )
    )
    accessor_names = {accessor.attribute.attr for accessor in accessors}
    if len(accessor_names) != 1:
        return None
    accessor_name = next(iter(accessor_names))
    root_names = sorted_tuple((accessor.owner.id for accessor in accessors))
    return (accessor_name, root_names)


class _RegisteredUnionSurfaceSourceStep(RegisteredEffectStep):
    pass


class _RegisteredUnionFunctionSourceStep(
    _RegisteredUnionSurfaceSourceStep,
    GuardedEffectStep[ast.AST, tuple[str, ast.AST, int]],
):
    step_id = "registered_union_function_source"
    registration_order = 10

    def accepts(self, value: ast.AST) -> bool:
        return isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef))

    def project(self, value: ast.AST) -> tuple[str, ast.AST, int] | None:
        function = cast(NamedFunctionNode, value)
        for statement in _trim_docstring_body(function.body):
            if isinstance(statement, ast.For):
                return function.name, statement.iter, statement.lineno
            if isinstance(statement, ast.Assign):
                return function.name, statement.value, statement.lineno
        return None


class _RegisteredUnionAssignmentSourceStep(
    _RegisteredUnionSurfaceSourceStep,
    AstTypedEffectStep[ast.Assign, tuple[str, ast.AST, int]],
):
    step_id = "registered_union_assignment_source"
    registration_order = 20
    node_type = ast.Assign

    def project_ast(self, value: ast.Assign) -> tuple[str, ast.AST, int] | None:
        target_name = name_id(single_item(value.targets))
        return None if target_name is None else (target_name, value.value, value.lineno)


def _registered_union_surface_source(
    node: ast.AST,
) -> tuple[str, ast.AST, int] | None:
    return cast(
        tuple[str, ast.AST, int] | None,
        Maybe.of(node)
        .bind(
            FirstSuccessfulEffectStep(
                registered_effect_steps(_RegisteredUnionSurfaceSourceStep)
            )
        )
        .unwrap_or_none(),
    )


def _registered_union_surface_candidates_for_node(
    module: ParsedModule,
    node: ast.AST,
    class_defs_by_name: dict[str, ast.ClassDef],
) -> Iterable[RegisteredUnionSurfaceCandidate]:
    source = _registered_union_surface_source(node)
    if source is None:
        return
    owner_name, value, line = source
    registered_surface = _registered_surface_roots(value)
    if registered_surface is None:
        return
    accessor_name, root_names = registered_surface
    if len(root_names) < 2:
        return
    root_nodes = [class_defs_by_name.get(root_name) for root_name in root_names]
    if any((root_node is None for root_node in root_nodes)):
        return
    if any(
        (
            (
                method := _class_method_named(
                    cast(ast.ClassDef, root_node), accessor_name
                )
            )
            is None
            or not _is_classmethod(method)
            for root_node in root_nodes
        )
    ):
        return
    yield RegisteredUnionSurfaceCandidate(
        file_path=str(module.path),
        line=line,
        owner_name=owner_name,
        accessor_name=accessor_name,
        root_names=root_names,
    )


def _registered_union_surface_candidates(
    module: ParsedModule,
) -> tuple[RegisteredUnionSurfaceCandidate, ...]:
    class_defs_by_name = {
        node.name: node for node in module.module.body if isinstance(node, ast.ClassDef)
    }
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.AST,
        _registered_union_surface_candidates_for_node,
        class_defs_by_name,
    )


def _type_name_set(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (ast.unparse(node),)
    if isinstance(node, ast.Tuple):
        return sorted_tuple(
            {
                type_name
                for element in node.elts
                for type_name in _type_name_set(element)
            }
        )
    return ()


def _export_policy_role_names(node: ast.FunctionDef) -> tuple[str, ...]:
    body_text = "\n".join(ast.unparse(statement) for statement in node.body)
    roles: set[str] = set()
    if "name.startswith('_')" in body_text:
        roles.add("exclude_private")
    if (
        "__module__ != __name__" in body_text
        or "getattr(value, '__module__', None) == __name__" in body_text
    ):
        roles.add("module_local")
    for current in _walk_nodes(node):
        if not isinstance(current, ast.Call):
            continue
        call_name = _ast_terminal_name(current.func)
        if call_name == "isinstance" and len(current.args) == 2:
            type_names = set(_type_name_set(current.args[1]))
            if _TYPE_NAME_LITERAL in type_names:
                roles.add("type_only")
                type_names.discard(_TYPE_NAME_LITERAL)
            elif type_names:
                roles.add("value_type_filter")
            if any((type_name.endswith("Enum") for type_name in type_names)):
                roles.add("enum_ok")
        elif call_name == "callable" and len(current.args) == 1:
            roles.add("callable_ok")
        elif call_name == "issubclass" and len(current.args) == 2:
            roles.add("subclass_constraint")
            type_names = set(_type_name_set(current.args[1]))
            if any((type_name.endswith("Enum") for type_name in type_names)):
                roles.add("enum_ok")
        elif call_name == "isabstract":
            roles.add("exclude_abstract")
    return sorted_tuple(roles)


def _export_policy_root_type_names(node: ast.FunctionDef) -> tuple[str, ...]:
    root_type_names: set[str] = set()
    for current in _walk_nodes(node):
        if not isinstance(current, ast.Call):
            continue
        if _ast_terminal_name(current.func) != "issubclass" or len(current.args) != 2:
            continue
        root_type_names.update(
            (
                type_name
                for type_name in _type_name_set(current.args[1])
                if type_name != _TYPE_NAME_LITERAL
            )
        )
    return sorted_tuple(root_type_names)


def _module_function_named(
    module: ParsedModule, function_name: str
) -> ast.FunctionDef | None:
    return next(
        (
            statement
            for statement in _trim_docstring_body(module.module.body)
            if isinstance(statement, ast.FunctionDef)
            and statement.name == function_name
        ),
        None,
    )


def _export_all_assignment_value(statement: ast.stmt) -> ast.AST | None:
    assignment = as_ast(statement, ast.Assign)
    if assignment is None or name_id(single_item(assignment.targets)) != "__all__":
        return None
    return assignment.value


def _sorted_generator_arg(value: ast.AST) -> ast.GeneratorExp | None:
    call = as_ast(value, ast.Call)
    if call is None or _ast_terminal_name(call.func) != "sorted":
        return None
    return single_ast(call.args, ast.GeneratorExp)


def _single_generator_filter_call(generator: ast.GeneratorExp) -> ast.Call | None:
    comprehension = single_item(generator.generators)
    if comprehension is None:
        return None
    return as_ast(single_item(comprehension.ifs), ast.Call)


def _export_all_predicate_name(statement: ast.stmt) -> str | None:
    value = _export_all_assignment_value(statement)
    if value is None:
        return None
    generator = _sorted_generator_arg(value)
    if generator is None:
        return None
    condition = _single_generator_filter_call(generator)
    return name_id(condition.func if condition else None)


def _module_exported_predicate_names(module: ParsedModule) -> frozenset[str]:
    return frozenset(
        (
            predicate_name
            for statement in _trim_docstring_body(module.module.body)
            if (predicate_name := _export_all_predicate_name(statement)) is not None
        )
    )


def _module_export_policy_predicate_candidate(
    module: ParsedModule,
) -> ExportPolicyPredicateCandidate | None:
    exported_predicate_names = _module_exported_predicate_names(module)
    predicate_name = single_item(tuple(exported_predicate_names))
    if predicate_name is None:
        return None
    predicate_node = _module_function_named(module, predicate_name)
    if predicate_node is None or len(predicate_node.args.args) != 2:
        return None
    role_names = _export_policy_role_names(predicate_node)
    if len(role_names) < 2:
        return None
    root_type_names = _export_policy_root_type_names(predicate_node)
    return ExportPolicyPredicateCandidate(
        file_path=str(module.path),
        line=predicate_node.lineno,
        subject_name=predicate_name,
        name_family=role_names,
        role_names=role_names,
        root_type_names=root_type_names,
    )


def _returned_sequence_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    for current in _walk_nodes(node):
        if not isinstance(current, ast.Return) or current.value is None:
            continue
        value = current.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "tuple"
            and len(value.args) == 1
        ):
            inner = value.args[0]
            if isinstance(inner, ast.Name):
                return inner.id
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and (inner.func.id == "sorted")
                and inner.args
                and isinstance(inner.args[0], ast.Name)
            ):
                return inner.args[0].id
    return None


def _subclasses_root_expression(node: ast.AST) -> str | None:
    subclasses_call = single_named_call_argument(
        node, call_name=_BuiltinCollectionName.LIST, argument_type=ast.Call
    ) or as_ast(node, ast.Call)
    if subclasses_call is None:
        return None
    match = attribute_call_match(
        subclasses_call,
        method_name="__subclasses__",
        owner_type=ast.AST,
        argument_count=0,
    )
    return None if match is None else ast.unparse(match.owner)


def _subclass_traversal_seed(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str] | None:
    for statement in _trim_docstring_body(node.body):
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or (not isinstance(statement.targets[0], ast.Name))
        ):
            continue
        if (root_expression := _subclasses_root_expression(statement.value)) is None:
            continue
        return statement.targets[0].id, root_expression
    return None


def _queue_pop_target_name(statement: ast.stmt, queue_name: str) -> str | None:
    assignment_node = as_ast(statement, ast.Assign)
    assignment = named_call_assignment(assignment_node) if assignment_node else None
    pop_call = (
        attribute_call_match(
            assignment.call,
            method_name="pop",
            owner_type=ast.Name,
            owner_name=queue_name,
            single_argument_required=True,
        )
        if assignment is not None
        else None
    )
    if pop_call is None or constant_value(pop_call.single_argument) != 0:
        return None
    return assignment.target_name


def _extends_subclasses_queue(
    statement: ast.stmt, queue_name: str, current_name: str
) -> bool:
    if (
        not isinstance(statement, ast.Expr)
        or not isinstance(statement.value, ast.Call)
        or (not isinstance(statement.value.func, ast.Attribute))
        or (statement.value.func.attr != "extend")
        or (not isinstance(statement.value.func.value, ast.Name))
        or (statement.value.func.value.id != queue_name)
        or (len(statement.value.args) != 1)
    ):
        return False
    return _subclasses_root_expression(statement.value.args[0]) == current_name


def _result_append_args(
    node: ast.FunctionDef | ast.AsyncFunctionDef, result_name: str
) -> tuple[ast.AST, ...]:
    return tuple(
        (
            current.args[0]
            for current in _walk_nodes(node)
            if isinstance(current, ast.Call)
            and isinstance(current.func, ast.Attribute)
            and (current.func.attr == _APPEND_METHOD_NAME)
            and isinstance(current.func.value, ast.Name)
            and (current.func.value.id == result_name)
            and (len(current.args) == 1)
        )
    )


def _registry_materialization_kind(
    node: ast.FunctionDef | ast.AsyncFunctionDef, result_name: str
) -> str | None:
    append_args = _result_append_args(node, result_name)
    if len(append_args) != 1:
        return None
    arg = append_args[0]
    if isinstance(arg, ast.Call):
        return "instantiate"
    if isinstance(arg, ast.Name):
        return _TYPE_NAME_LITERAL
    return "projection"


def _registry_attribute_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            attribute_name
            for current in _walk_nodes(node)
            if isinstance(current, ast.Call)
            and isinstance(current.func, ast.Attribute)
            and (current.func.attr == "get")
            and isinstance(current.func.value, ast.Attribute)
            and (current.func.value.attr == "__dict__")
            and isinstance(current.func.value.value, ast.Name)
            and (len(current.args) == 1)
            and ((attribute_name := _constant_string(current.args[0])) is not None)
        }
    )


def _subclass_traversal_filter_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    current_name: str,
) -> tuple[str, ...]:
    filter_names: set[str] = set()
    for current in _walk_nodes(node):
        if not isinstance(current, ast.Call):
            continue
        if isinstance(current.func, ast.Name) and any(
            (
                isinstance(subnode, ast.Name) and subnode.id == current_name
                for subnode in current.args
            )
        ):
            filter_names.add(current.func.id)
            continue
        if (
            isinstance(current.func, ast.Attribute)
            and current.func.attr == "get"
            and isinstance(current.func.value, ast.Attribute)
            and (current.func.value.attr == "__dict__")
            and isinstance(current.func.value.value, ast.Name)
            and (current.func.value.value.id == current_name)
            and (len(current.args) == 1)
            and ((attribute_name := _constant_string(current.args[0])) is not None)
        ):
            filter_names.add(attribute_name)
    return sorted_tuple(filter_names)


def _subclass_traversal_site(
    module: ParsedModule,
    qualname: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> SubclassTraversalSite | None:
    seed = _subclass_traversal_seed(node)
    if seed is None:
        return None
    queue_name, root_expression = seed
    result_name = _returned_sequence_name(node)
    loop_profile = _subclass_traversal_loop_profile(node, queue_name)
    materialization_kind = _subclass_traversal_materialization_kind(node, result_name)
    if result_name is None or loop_profile is None or materialization_kind is None:
        return None
    current_name = loop_profile
    return SubclassTraversalSite(
        file_path=str(module.path),
        line=node.lineno,
        symbol=qualname,
        root_expression=root_expression,
        materialization_kind=materialization_kind,
        registry_attribute_names=_registry_attribute_names(node),
        filter_names=_subclass_traversal_filter_names(node, current_name),
    )


def _subclass_traversal_loop_profile(
    node: ast.FunctionDef | ast.AsyncFunctionDef, queue_name: str
) -> str | None:
    current_name: str | None = None
    extends_queue = False
    for statement in _walk_nodes(node):
        if not isinstance(statement, ast.While):
            continue
        for body_statement in statement.body:
            current_name = current_name or _queue_pop_target_name(
                body_statement, queue_name
            )
            if current_name is not None and _extends_subclasses_queue(
                body_statement, queue_name, current_name
            ):
                extends_queue = True
    if current_name is None or not extends_queue:
        return None
    return current_name


def _subclass_traversal_materialization_kind(
    node: ast.FunctionDef | ast.AsyncFunctionDef, result_name: str | None
) -> str | None:
    if result_name is None:
        return None
    materialization_kind = _registry_materialization_kind(node, result_name)
    append_args = _result_append_args(node, result_name)
    if materialization_kind is None or not append_args:
        return None
    return materialization_kind


def _registry_traversal_group(
    modules: Sequence[ParsedModule],
) -> SubclassTraversalGroup | None:
    sites = sorted_tuple(
        (
            site
            for module in modules
            for qualname, function in _iter_named_functions(module)
            if (
                site := _subclass_traversal_site(
                    module,
                    qualname,
                    cast(ast.FunctionDef | ast.AsyncFunctionDef, function),
                )
            )
            is not None
        ),
        key=lambda item: (item.file_path, item.line, item.symbol),
    )
    if len(sites) < 2:
        return None
    return SubclassTraversalGroup(
        symbols=tuple((site.symbol for site in sites)),
        file_paths=tuple((site.file_path for site in sites)),
        line_numbers=tuple((site.line for site in sites)),
        root_expressions=tuple((site.root_expression for site in sites)),
        materialization_kinds=tuple((site.materialization_kind for site in sites)),
        registry_attribute_names=sorted_tuple(
            {
                attribute_name
                for site in sites
                for attribute_name in site.registry_attribute_names
            }
        ),
        filter_names=sorted_tuple(
            {filter_name for site in sites for filter_name in site.filter_names}
        ),
    )


def _declarative_family_boilerplate_groups(
    module: ParsedModule,
) -> tuple[DeclarativeFamilyBoilerplateGroup, ...]:
    return _classvar_only_sibling_leaf_groups(module)


def _constructor_return_call(node: ast.FunctionDef) -> ast.Call | None:
    body = _trim_docstring_body(node.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    returned = body[0].value
    if not isinstance(returned, ast.Call):
        return None
    if not isinstance(returned.func, ast.Name) or returned.func.id != "cls":
        return None
    return returned


def _source_type_name_for_constructor(node: ast.FunctionDef) -> str | None:
    if len(node.args.args) < 3:
        return None
    source_arg = node.args.args[2]
    if source_arg.annotation is None:
        return source_arg.arg
    return ast.unparse(source_arg.annotation)


def _alternate_constructor_family_groups(
    module: ParsedModule,
) -> tuple[AlternateConstructorFamilyGroup, ...]:
    groups: list[AlternateConstructorFamilyGroup] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        constructor_methods: list[tuple[ast.FunctionDef, ast.Call, str]] = []
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not statement.name.startswith("from_") or not _is_classmethod(statement):
                continue
            return_call = _constructor_return_call(statement)
            if return_call is None:
                continue
            source_type_name = _source_type_name_for_constructor(statement)
            if source_type_name is None:
                continue
            constructor_methods.append((statement, return_call, source_type_name))
        if len(constructor_methods) < 3:
            continue
        keyword_sets = [
            {keyword.arg for keyword in call.keywords if keyword.arg is not None}
            for _, call, _ in constructor_methods
        ]
        shared_keyword_names = sorted_tuple(
            (str(item) for item in set.intersection(*keyword_sets))
        )
        if len(shared_keyword_names) < 4:
            continue
        groups.append(
            AlternateConstructorFamilyGroup(
                file_path=str(module.path),
                class_name=node.name,
                method_names=tuple(
                    (method.name for method, _, _ in constructor_methods)
                ),
                line_numbers=tuple(
                    (method.lineno for method, _, _ in constructor_methods)
                ),
                keyword_names=shared_keyword_names,
                source_type_names=tuple(
                    (source_type_name for _, _, source_type_name in constructor_methods)
                ),
            )
        )
    return tuple(groups)


def _repeated_property_hook_metrics(
    class_names: tuple[str, ...], property_name: str
) -> RepeatedMethodMetrics:
    return RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(class_names),
        statement_count=1,
        class_count=len(class_names),
        method_symbols=tuple(
            (f"{class_name}.{property_name}" for class_name in class_names)
        ),
    )


_METRIC_BAG_SCHEMAS = metric_semantic_bag_descriptors()

_IMPACT_BAG_SCHEMA = impact_delta_semantic_bag_descriptor()

_SEMANTIC_STRING_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_SEMANTIC_KEYWORD_NAMES = {
    "backend",
    "capability_gap",
    "capability_tags",
    "certification",
    "confidence",
    "key",
    "kind",
    "label",
    "mode",
    _NAME_LITERAL,
    "observation_tags",
    "pattern_id",
    "registry_key",
    "relation_context",
    "status",
    "title",
    _TYPE_NAME_LITERAL,
}
_SEMANTIC_NAME_SUFFIXES = (
    "_backend",
    "_certification",
    "_family",
    "_id",
    "_key",
    "_kind",
    "_label",
    "_mode",
    "_name",
    "_pattern",
    "_role",
    "_status",
    "_type",
)


def _semantic_string_literal_sites(
    module: ParsedModule,
) -> dict[str, list[SourceLocation]]:
    groups: dict[str, set[SourceLocation]] = defaultdict(set)

    class Visitor(ClassFunctionStackNodeVisitor):
        def visit_Module(self, node: ast.Module) -> None:
            self.traverse_trimmed_statements(node.body)

        traverse_class_body = ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
        traverse_function_body = (
            ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
        )

        def visit_Assign(self, node: ast.Assign) -> None:
            self._record_literals(node.value, node.lineno, "assign")
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is not None:
                self._record_literals(node.value, node.lineno, "assign")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                if not _is_semantic_keyword_name(keyword.arg):
                    continue
                self._record_literals(keyword.value, node.lineno, keyword.arg)
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            if not _compare_subject_is_semantic(node):
                self.generic_visit(node)
                return
            self._record_literals(node.left, node.lineno, "compare")
            for comparator in node.comparators:
                self._record_literals(comparator, node.lineno, "compare")
            self.generic_visit(node)

        def _record_literals(self, node: ast.AST, lineno: int, kind: str) -> None:
            for literal in _literal_strings(node):
                groups[literal].add(
                    SourceLocation(str(module.path), lineno, self._symbol(kind))
                )

        def _symbol(self, kind: str) -> str:
            owner = self.function_stack[-1] if self.function_stack else "<module>"
            if self.class_stack:
                owner = f"{self.class_stack[-1]}.{owner}"
            return f"{owner}:{kind}"

    Visitor().visit(module.module)
    return {
        literal: sorted(
            sites, key=lambda item: (item.file_path, item.line, item.symbol)
        )
        for literal, sites in groups.items()
        if len(sites) >= 2
    }


def _literal_strings(node: ast.AST) -> tuple[str, ...]:
    literals: list[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if _is_semantic_string(node.value):
            literals.append(node.value)
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for item in node.elts:
            literals.extend(_literal_strings(item))
    return tuple(literals)


def _is_semantic_string(value: str) -> bool:
    return bool(_SEMANTIC_STRING_RE.fullmatch(value))


def _is_semantic_keyword_name(name: str) -> bool:
    return name in _SEMANTIC_KEYWORD_NAMES or name.endswith(_SEMANTIC_NAME_SUFFIXES)


def _compare_subject_is_semantic(node: ast.Compare) -> bool:
    candidates = [node.left] + list(node.comparators)
    return any((_looks_like_semantic_subject(candidate) for candidate in candidates))


def _looks_like_semantic_subject(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return _is_semantic_keyword_name(node.id)
    if isinstance(node, ast.Attribute):
        return _is_semantic_keyword_name(node.attr)
    return False


def _collect_dict_attrs(node: ast.ClassDef) -> set[str]:
    dict_attrs: set[str] = set()
    for child in _walk_nodes(node):
        if not isinstance(child, ast.Assign):
            continue
        if not isinstance(child.value, ast.Dict):
            continue
        for target in child.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and (target.value.id == "self")
            ):
                dict_attrs.add(target.attr)
    return dict_attrs


def _collect_mirrored_assignments(node: ast.ClassDef) -> list[tuple[int, str]]:
    mirrored: list[tuple[int, str]] = []
    for child in _walk_nodes(node):
        if not isinstance(child, ast.Assign):
            continue
        for target in child.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Attribute):
                continue
            if (
                not isinstance(target.value.value, ast.Name)
                or target.value.value.id != "self"
            ):
                continue
            if isinstance(child.value, ast.Name):
                mirrored.append((child.lineno, target.value.attr))
    return mirrored


def _collect_repeated_method_shapes(
    modules: list[ParsedModule], config: DetectorConfig
) -> tuple[MethodShape, ...]:
    groups = _group_repeated_methods(modules, config)
    return tuple((method for group in groups for method in group))


def _group_repeated_methods(
    modules: list[ParsedModule], config: DetectorConfig
) -> list[tuple[MethodShape, ...]]:
    methods = tuple(
        (
            method
            for module in modules
            for method in _collect_typed_family_items(
                module, MethodShapeFamily, MethodShape
            )
            if method.class_name
            and method.statement_count >= config.min_duplicate_statements
        )
    )
    groups = _fiber_grouped_shapes(
        modules,
        tuple(methods),
        ObservationKind.METHOD_SHAPE,
        StructuralExecutionLevel.FUNCTION_BODY,
    )
    return [
        tuple((_as_method_shape(method) for method in group))
        for group in groups
        if len(group) >= 2
        and len({_as_method_shape(method).class_name for method in group}) >= 2
    ]


def _abc_patch_for_methods(methods: tuple[MethodShape, ...]) -> str:
    target_file = methods[0].file_path
    base_name = (
        _shared_family_name(
            sorted(
                {
                    method.class_name
                    for method in methods
                    if method.class_name is not None
                }
            )
        )
        or "ExtractedBase"
    )
    hook_name = methods[0].method_name
    return f"*** Begin Patch\n*** Update File: {target_file}\n@@\n+class {base_name}(ABC):\n+    def run(self, request):\n+        normalized = self._normalize(request)\n+        return self.{hook_name}(normalized)\n+\n+    @abstractmethod\n+    def {hook_name}(self, normalized): ...\n*** End Patch"


def _abc_family_patch(
    class_names: frozenset[str], groups: list[tuple[MethodShape, ...]]
) -> str:
    ordered = sorted(class_names)
    target_file = groups[0][0].file_path
    base_name = _shared_family_name(ordered) or "FamilyBase"
    return f"*** Begin Patch\n*** Update File: {target_file}\n@@\n+class {base_name}(ABC):\n+    def run(self, request): ...\n+\n+    @abstractmethod\n+    def hook(self, request): ...\n*** End Patch"


def _builder_patch(builders: tuple[BuilderCallShape, ...]) -> str:
    target_file = builders[0].file_path
    callee_name = builders[0].callee_name
    return f"*** Begin Patch\n*** Update File: {target_file}\n@@\n+@classmethod\n+def from_source(cls, source):\n+    return {callee_name}(...)\n*** End Patch"


def _single_owner_builder_family_patch(owner_symbol: str, callee_name: str) -> str:
    return (
        f"# Replace the repeated `{callee_name}` calls inside `{owner_symbol}` with one declarative invocation table.\n"
        "# Keep the builder authority in one row family and materialize the calls in one loop."
    )


def _projection_schema_patch(export_shapes: tuple[ExportDictShape, ...]) -> str:
    target_file = export_shapes[0].file_path
    return f"*** Begin Patch\n*** Update File: {target_file}\n@@\n+@dataclass(frozen=True)\n+class ProjectionSchema:\n+    ...\n+\n+    @classmethod\n+    def from_source(cls, source): ...\n*** End Patch"


def _autoregister_patch(
    registry_name: str,
    class_names: set[str],
    registrations: tuple[RegistrationShape, ...],
) -> str:
    target_file = registrations[0].file_path
    base_name = _shared_family_name(sorted(class_names)) or "RegisteredBase"
    ordered_class_names = sorted_tuple(class_names)
    key_values = tuple(
        (
            key_value
            for class_name in ordered_class_names
            if (
                key_value := _string_constant_expression(
                    next(
                        (
                            registration.key_expression
                            for registration in registrations
                            if registration.registered_class == class_name
                        )
                    )
                )
            )
            is not None
        )
    )
    use_extractor = len(key_values) == len(ordered_class_names) and (
        _derivable_registry_key_suffix(ordered_class_names, key_values) is not None
    )
    config_block = (
        _derived_registry_key_block(ordered_class_names)
        if use_extractor
        else _declared_registry_key_block("registry_key")
    )
    return f"*** Begin Patch\n*** Update File: {target_file}\n@@\n" + (
        "+from metaclass_registry import AutoRegisterMeta\n"
        + ("+import re\n" if use_extractor else "")
        + "+\n"
        + f"+class {base_name}(ABC, metaclass=AutoRegisterMeta):\n"
        + "".join(f"+{line}\n" for line in config_block.splitlines())
        + "+\n"
        + f"+# Replace `{registry_name}` with `{base_name}.__registry__`.\n"
        + "*** End Patch"
    )


def _abc_scaffold_for_methods(methods: tuple[MethodShape, ...]) -> str:
    class_names = sorted(
        {method.class_name for method in methods if method.class_name is not None}
    )
    hook_names = sorted({method.method_name for method in methods})
    base_name = _shared_family_name(class_names) or "ExtractedBase"
    hook_name = hook_names[0] if hook_names else "hook"
    return f"class {base_name}(ABC):\n    def run(self, request):\n        normalized = self._normalize(request)\n        return self.{hook_name}(normalized)\n\n    @abstractmethod\n    def {hook_name}(self, normalized): ..."


def _abc_family_scaffold(
    class_names: frozenset[str], groups: list[tuple[MethodShape, ...]]
) -> str:
    ordered = sorted(class_names)
    base_name = _shared_family_name(ordered) or "FamilyBase"
    hook_methods = sorted(
        {
            method.method_name
            for group in groups
            for method in group
            if method.class_name in class_names
        }
    )
    hook_block = "\n".join(
        (
            f"    @abstractmethod\n    def {name}(self, request): ..."
            for name in hook_methods[:3]
        )
    )
    subclass_block = "\n".join(
        (f"class {name}({base_name}):\n    ..." for name in ordered[:3])
    )
    return f"class {base_name}(ABC):\n    def run(self, request): ...\n{hook_block}\n\n{subclass_block}"


def _builder_scaffold(builders: tuple[BuilderCallShape, ...]) -> str:
    callee_name = builders[0].callee_name
    keywords = builders[0].keyword_names
    row_name = callee_name if callee_name[:1].isupper() else "ProjectedRow"
    args_block = "\n".join(
        (f"            {name}=source.{name}," for name in keywords[:4])
    )
    return f"@dataclass(frozen=True)\nclass {row_name}:\n    ...\n\n    @classmethod\n    def from_source(cls, source):\n        return cls(\n{args_block}\n        )"


def _single_owner_builder_family_scaffold(callee_name: str) -> str:
    return f'@dataclass(frozen=True)\nclass InvocationSpec:\n    args: tuple[object, ...]\n    kwargs: dict[str, object]\n\nINVOCATION_SPECS = (\n    InvocationSpec(args=(...), kwargs={{"flag": True}}),\n)\n\nfor spec in INVOCATION_SPECS:\n    owner.{callee_name}(*spec.args, **spec.kwargs)'


def _projection_schema_scaffold(export_shapes: tuple[ExportDictShape, ...]) -> str:
    keys = export_shapes[0].key_names
    field_block = "\n".join((f"    {key}: object" for key in keys[:4]))
    mapping_block = "\n".join((f"            {key}=source.{key}," for key in keys[:4]))
    return f"@dataclass(frozen=True)\nclass ProjectionSchema:\n{field_block}\n\n    @classmethod\n    def from_source(cls, source):\n        return cls(\n{mapping_block}\n        )"


def _autoregister_scaffold(registry_name: str, class_names: set[str]) -> str:
    base_name = _shared_family_name(sorted(class_names)) or "RegisteredBase"
    sample = sorted(class_names)[:2]
    config_block = _derived_registry_key_block(sample)
    subclass_block = "\n".join(
        (f"class {name}({base_name}):\n    ..." for name in sample)
    )
    return f"from abc import ABC\nimport re\nfrom metaclass_registry import AutoRegisterMeta\n\nclass {base_name}(ABC, metaclass=AutoRegisterMeta):\n{config_block}\n\n{subclass_block}"


def _shared_family_name(class_names: list[str]) -> str | None:
    if not class_names:
        return None
    prefix = class_names[0]
    for name in class_names[1:]:
        while prefix and (not name.startswith(prefix)):
            prefix = prefix[:-1]
    return prefix or None


@lru_cache(maxsize=None)
def _dispatch_dict_locations(
    module: ParsedModule, min_string_cases: int
) -> list[SourceLocation]:
    locations: list[SourceLocation] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_depth += 1
            self.generic_visit(node)
            self.function_depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Assign(self, node: ast.Assign) -> None:
            if self.function_depth > 0:
                return
            if not isinstance(node.value, ast.Dict):
                return
            if _looks_like_dispatch_dict(node.value, min_string_cases):
                locations.append(
                    SourceLocation(
                        str(module.path), node.lineno, "dict-string-dispatch"
                    )
                )

        visit_AnnAssign = visit_Assign

    Visitor().visit(module.module)
    return locations


def _looks_like_dispatch_dict(node: ast.Dict, min_string_cases: int) -> bool:
    string_keys = [
        key
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]
    if len(string_keys) < min_string_cases or len(string_keys) != len(node.keys):
        return False
    if not node.values:
        return False
    if all((isinstance(value, ast.Constant) for value in node.values)):
        return False
    return any(
        (
            isinstance(value, (ast.Name, ast.Attribute, ast.Lambda, ast.Call))
            for value in node.values
        )
    )


@lru_cache(maxsize=None)
def _attribute_branch_evidence(
    module: ParsedModule, attr_name: str
) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in _walk_nodes(module.module):
        if isinstance(node, ast.If):
            if _test_compares_attribute(node.test, attr_name):
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, f"if-{attr_name}")
                )
        if isinstance(node, ast.Match):
            subject = node.subject
            if isinstance(subject, ast.Attribute) and subject.attr == attr_name:
                evidence.append(
                    SourceLocation(str(module.path), node.lineno, f"match-{attr_name}")
                )
    return evidence


def _test_compares_attribute(test: ast.AST, attr_name: str) -> bool:
    for node in _walk_nodes(test):
        if isinstance(node, ast.Compare):
            values = [node.left] + list(node.comparators)
            attr_match = any(
                (
                    isinstance(value, ast.Attribute) and value.attr == attr_name
                    for value in values
                )
            )
            literal_match = any(
                (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, (str, int, bool))
                    for value in values
                )
            )
            if attr_match and literal_match:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == _GETATTR_BUILTIN and len(node.args) >= 2:
                arg = node.args[1]
                if isinstance(arg, ast.Constant) and arg.value == attr_name:
                    return True
    return False


def _iter_functions(module: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in _walk_nodes(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _projection_helper_scaffold(shapes: Sequence[ProjectionHelperShape]) -> str:
    function_names = ", ".join(shape.function_name for shape in shapes)
    attributes = ", ".join(sorted({shape.projected_attribute for shape in shapes}))
    return f"def _render_projection(items, projector):\n    return tuple(_dedupe_preserve_order(projector(item) for item in items))\n\n# Replace {function_names} with `_render_projection(..., lambda item: item.<field>)`.\n# Projected fields: {attributes}"


def _supports_accessor_wrapper_finding(
    candidates: Sequence[AccessorWrapperCandidate],
) -> bool:
    if not candidates:
        return False
    if any(
        (candidate.wrapper_shape.startswith("computed_") for candidate in candidates)
    ):
        return True
    if len(candidates) >= 2:
        return True
    return False


def _is_framework_adapter_symbol(symbol: str) -> bool:
    return symbol.startswith(("build_from_", "build_scoped_", "accepts_"))


def _is_framework_lineage_symbol(symbol: str) -> bool:
    return _is_framework_adapter_symbol(symbol) or symbol in {
        "__new__",
        "collect",
        "registered_specs_for_literal_type",
    }


def _is_framework_attribute_probe(observation: AttributeProbeObservation) -> bool:
    return observation.observed_attribute in {
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
        # Standard array protocol compatibility checks are not semantic-role recovery.
        "shape",
        "ndim",
        "dtype",
        "size",
    }


def _accessor_replacement_example(candidate: AccessorWrapperCandidate) -> str:
    if candidate.accessor_kind == "setter":
        return f"- replace `{candidate.symbol}(value)` with `{candidate.observed_attribute} = value`"
    if candidate.wrapper_shape == "read_through":
        return f"- replace `{candidate.symbol}()` with `{candidate.observed_attribute}`"
    return f"- replace `{candidate.symbol}()` with an `@property` exposing `{candidate.target_expression}`"


def _expression_root_names(node: ast.AST) -> set[str]:
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


def _function_param_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    names = {arg.arg for arg in function.args.args}
    names.update(arg.arg for arg in function.args.kwonlyargs)
    if function.args.vararg is not None:
        names.add(function.args.vararg.arg)
    if function.args.kwarg is not None:
        names.add(function.args.kwarg.arg)
    return names


def _is_transport_expression(node: ast.AST, *, allowed_roots: set[str]) -> bool:
    return _expression_root_names(node) <= allowed_roots


def _wrapper_delegate_symbol(
    node: ast.AST,
    *,
    class_name: str | None,
) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and (node.value.id in {"self", "cls"})
        and (class_name is not None)
    ):
        return f"{class_name}.{node.attr}"
    return None


def _call_transport_values(call: ast.Call) -> tuple[ast.AST, ...]:
    return (*call.args, *(keyword.value for keyword in call.keywords if keyword.arg))


def _transported_delegate_symbol(
    call: ast.Call,
    *,
    class_name: str | None,
    allowed_roots: set[str],
) -> str | None:
    delegate_symbol = _wrapper_delegate_symbol(call.func, class_name=class_name)
    if delegate_symbol is None:
        return None
    if not all(
        (
            _is_transport_expression(value, allowed_roots=allowed_roots)
            for value in _call_transport_values(call)
        )
    ):
        return None
    return delegate_symbol


def _projected_attribute_names(
    node: ast.AST, *, bound_name: str
) -> tuple[str, ...] | None:
    return _single_projected_attribute_name(
        node, bound_name=bound_name
    ) or _tuple_projected_attribute_names(node, bound_name=bound_name)


def _single_projected_attribute_name(
    node: ast.AST, *, bound_name: str
) -> tuple[str, ...] | None:
    projected_name = attribute_name(node, owner_name=bound_name)
    return None if projected_name is None else (projected_name,)


def _tuple_projected_attribute_names(
    node: ast.AST,
    *,
    bound_name: str,
) -> tuple[str, ...] | None:
    tuple_node = as_ast(node, ast.Tuple)
    if tuple_node is None:
        return None
    projected_names = tuple(
        (attribute_name(item, owner_name=bound_name) for item in tuple_node.elts)
    )
    if any((projected_name is None for projected_name in projected_names)):
        return None
    return cast(tuple[str, ...], projected_names)


def _call_chain_from_outer_call(call: ast.Call) -> tuple[ast.Call, ...]:
    chain = [call]
    current = call
    while isinstance(current.func, ast.Attribute) and isinstance(
        current.func.value, ast.Call
    ):
        current = current.func.value
        chain.append(current)
    return tuple(chain)


def _call_chain_transport_values(chain: tuple[ast.Call, ...]) -> tuple[ast.AST, ...]:
    values: list[ast.AST] = []
    for call in chain:
        values.extend(call.args)
        values.extend(keyword.value for keyword in call.keywords)
    return tuple(values)


def _call_chain_delegate_symbol(
    chain: tuple[ast.Call, ...],
    *,
    class_name: str | None,
) -> str:
    inner = chain[-1]
    symbol = _wrapper_delegate_symbol(inner.func, class_name=class_name)
    if symbol is None:
        symbol = ast.unparse(inner.func)
    for call in reversed(chain[:-1]):
        method_name = _call_name(call.func)
        if method_name is None:
            method_name = ast.unparse(call.func)
        symbol = f"{symbol}.{method_name}"
    return symbol


def _delegate_root_symbol(delegate_symbol: str) -> str:
    return delegate_symbol.split(".", 1)[0]


def _is_public_module_api_qualname(qualname: str) -> bool:
    return "." not in qualname and (not qualname.startswith("_"))


@lru_cache(maxsize=None)
def _top_level_symbol_lines(module: ParsedModule) -> dict[str, int]:
    lines: dict[str, int] = {}
    for statement in _trim_docstring_body(module.module.body):
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.setdefault(statement.name, statement.lineno)
    return lines


def _resolved_import_call_target_symbols(
    module: ParsedModule,
    node: ast.AST,
    *,
    import_aliases: dict[str, str],
) -> tuple[str, ...]:
    del module
    parts = _ast_attribute_chain(node)
    if parts is None:
        return ()
    first, *rest = parts
    alias_target = import_aliases.get(first)
    if alias_target is None:
        return ()
    return (".".join((alias_target, *rest)) if rest else alias_target,)


def _external_callsites_by_target(
    modules: Sequence[ParsedModule],
) -> dict[str, tuple[ResolvedExternalCallsite, ...]]:
    return _external_callsites_by_target_cached(tuple(modules))


@lru_cache(maxsize=None)
def _external_callsites_by_target_cached(
    modules: tuple[ParsedModule, ...],
) -> dict[str, tuple[ResolvedExternalCallsite, ...]]:
    callsites_by_target: dict[str, set[ResolvedExternalCallsite]] = defaultdict(set)
    for module in modules:
        import_aliases = _module_import_aliases(module)

        class Visitor(ClassFunctionStackNodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                for target in _resolved_import_call_target_symbols(
                    module, node.func, import_aliases=import_aliases
                ):
                    callsites_by_target[target].add(
                        ResolvedExternalCallsite(
                            module_name=module.module_name,
                            location=SourceLocation(
                                str(module.path), node.lineno, self._symbol("call")
                            ),
                        )
                    )
                self.generic_visit(node)

            def _symbol(self, kind: str) -> str:
                owner = self.function_stack[-1] if self.function_stack else "<module>"
                if self.class_stack:
                    owner = f"{self.class_stack[-1]}.{owner}"
                return f"{owner}:{kind}"

        Visitor().visit(module.module)
    return {
        target: sorted_tuple(
            callsites,
            key=lambda item: (
                item.location.file_path,
                item.location.line,
                item.location.symbol,
                item.module_name,
            ),
        )
        for target, callsites in callsites_by_target.items()
    }


def _matching_external_callsites(
    callsites_by_target: dict[str, tuple[ResolvedExternalCallsite, ...]],
    *,
    target_symbol: str,
) -> tuple[ResolvedExternalCallsite, ...]:
    matched: set[ResolvedExternalCallsite] = set()
    for observed_target, callsites in callsites_by_target.items():
        if (
            observed_target == target_symbol
            or observed_target.endswith(f".{target_symbol}")
            or target_symbol.endswith(f".{observed_target}")
        ):
            matched.update(callsites)
    return sorted_tuple(
        matched,
        key=lambda item: (
            item.location.file_path,
            item.location.line,
            item.location.symbol,
            item.module_name,
        ),
    )


def _transport_call_chain_match(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    min_depth: int = 2,
    exact_depth: int | None = None,
) -> tuple[tuple[ast.Call, ...], tuple[ast.AST, ...]] | None:
    returned = single_return_call(_trim_docstring_body(function.body))
    if returned is None:
        return None
    chain = _call_chain_from_outer_call(returned)
    if len(chain) < min_depth or (
        exact_depth is not None and len(chain) != exact_depth
    ):
        return None
    allowed_roots = _function_param_names(function) | {"self", "cls"}
    values = _call_chain_transport_values(chain)
    if not values or not all(
        (
            _is_transport_expression(value, allowed_roots=allowed_roots)
            for value in values
        )
    ):
        return None
    return chain, values


def _trivial_forwarding_wrapper_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> TrivialForwardingWrapperCandidate | None:
    if function.name.startswith("__") and function.name.endswith("__"):
        return None
    if function.name == _CANDIDATE_COLLECTOR_METHOD_NAME:
        return None
    chain_match = _transport_call_chain_match(function)
    if chain_match is None:
        return None
    chain, values = chain_match
    class_name = qualname.rsplit(".", 1)[0] if "." in qualname else None
    transported_value_sources = sorted_tuple({ast.unparse(value) for value in values})
    parameter_names = _function_param_names(function) - {"self", "cls"}
    forwarded_parameter_names = sorted_tuple(
        {
            node.id
            for value in values
            for node in _walk_nodes(value)
            if isinstance(node, ast.Name) and node.id in parameter_names
        }
    )
    delegate_symbol = _call_chain_delegate_symbol(chain, class_name=class_name)
    return TrivialForwardingWrapperCandidate(
        file_path=str(module.path),
        line=function.lineno,
        qualname=qualname,
        delegate_symbol=delegate_symbol,
        call_depth=len(chain),
        forwarded_parameter_names=forwarded_parameter_names,
        transported_value_sources=transported_value_sources,
    )


@lru_cache(maxsize=None)
def _trivial_forwarding_wrapper_candidates(
    module: ParsedModule,
) -> tuple[TrivialForwardingWrapperCandidate, ...]:
    candidates = [
        candidate
        for qualname, function in _iter_named_functions(module)
        for candidate in (
            _trivial_forwarding_wrapper_candidate(module, qualname, function),
        )
        if candidate is not None
    ]
    return sorted_tuple(
        candidates,
        key=lambda candidate: (candidate.file_path, candidate.line, candidate.qualname),
    )


def _public_api_private_delegate_shell_candidates(
    modules: Sequence[ParsedModule],
    config: DetectorConfig,
) -> tuple[PublicApiPrivateDelegateShellCandidate, ...]:
    min_external_callsites = max(2, config.min_registration_sites)
    callsites_by_target = _external_callsites_by_target(modules)
    candidates: list[PublicApiPrivateDelegateShellCandidate] = []
    for module in modules:
        top_level_lines = _top_level_symbol_lines(module)
        for wrapper_candidate in _trivial_forwarding_wrapper_candidates(module):
            if not _is_public_module_api_qualname(wrapper_candidate.qualname):
                continue
            delegate_root_symbol = _delegate_root_symbol(
                wrapper_candidate.delegate_symbol
            )
            if not _is_private_symbol_name(delegate_root_symbol):
                continue
            wrapper_symbol = f"{module.module_name}.{wrapper_candidate.qualname}"
            external_callsites = tuple(
                (
                    site
                    for site in _matching_external_callsites(
                        callsites_by_target, target_symbol=wrapper_symbol
                    )
                    if site.module_name != module.module_name
                )
            )
            if len(external_callsites) < min_external_callsites:
                continue
            candidates.append(
                PublicApiPrivateDelegateShellCandidate(
                    wrapper=wrapper_candidate,
                    module_name=module.module_name,
                    delegate_root_symbol=delegate_root_symbol,
                    delegate_root_line=top_level_lines.get(delegate_root_symbol),
                    external_callsites=external_callsites,
                )
            )
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.wrapper.file_path,
            item.wrapper.line,
            item.wrapper.qualname,
        ),
    )


def _public_api_private_delegate_family_candidates(
    modules: Sequence[ParsedModule],
    config: DetectorConfig,
) -> tuple[PublicApiPrivateDelegateFamilyCandidate, ...]:
    min_wrapper_count = max(2, config.min_registration_sites)
    min_external_callsites = max(2, config.min_registration_sites)
    callsites_by_target = _external_callsites_by_target(modules)
    grouped_wrappers: dict[
        tuple[str, str, str], list[TrivialForwardingWrapperCandidate]
    ] = defaultdict(list)
    delegate_lines: dict[tuple[str, str, str], int | None] = {}
    for module in modules:
        top_level_lines = _top_level_symbol_lines(module)
        for wrapper_candidate in _trivial_forwarding_wrapper_candidates(module):
            if not _is_public_module_api_qualname(wrapper_candidate.qualname):
                continue
            delegate_root_symbol = _delegate_root_symbol(
                wrapper_candidate.delegate_symbol
            )
            if not _is_private_symbol_name(delegate_root_symbol):
                continue
            key = (str(module.path), module.module_name, delegate_root_symbol)
            grouped_wrappers[key].append(wrapper_candidate)
            delegate_lines.setdefault(key, top_level_lines.get(delegate_root_symbol))
    candidates: list[PublicApiPrivateDelegateFamilyCandidate] = []
    for (
        file_path,
        module_name,
        delegate_root_symbol,
    ), wrappers in grouped_wrappers.items():
        if len(wrappers) < min_wrapper_count:
            continue
        external_callsites = sorted_tuple(
            {
                site
                for wrapper in wrappers
                for site in _matching_external_callsites(
                    callsites_by_target,
                    target_symbol=f"{module_name}.{wrapper.qualname}",
                )
                if site.module_name != module_name
            },
            key=lambda item: (
                item.location.file_path,
                item.location.line,
                item.location.symbol,
                item.module_name,
            ),
        )
        if len(external_callsites) < min_external_callsites:
            continue
        candidates.append(
            PublicApiPrivateDelegateFamilyCandidate(
                file_path=file_path,
                module_name=module_name,
                delegate_root_symbol=delegate_root_symbol,
                delegate_root_line=delegate_lines[
                    file_path, module_name, delegate_root_symbol
                ],
                wrappers=sorted_tuple(
                    wrappers, key=lambda item: (item.line, item.qualname)
                ),
                external_callsites=external_callsites,
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.file_path,
            item.delegate_root_symbol,
            item.wrappers[0].line,
        ),
    )


def _policy_selector_source_exprs(selector_call: ast.Call) -> tuple[str, ...]:
    return tuple(
        (
            ast.unparse(value)
            for value in (
                *selector_call.args,
                *(
                    keyword.value
                    for keyword in selector_call.keywords
                    if keyword.arg is not None
                ),
            )
        )
    )


def _looks_like_self_selector_source(expr: str) -> bool:
    return (
        expr == "self"
        or expr.startswith("self.")
        or expr == "cls"
        or expr.startswith("cls.")
    )


def _nominal_policy_method_header(
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str] | None:
    if "." not in qualname:
        return None
    owner_class_name, method_name = qualname.rsplit(".", 1)
    if method_name.startswith("_") or (
        method_name.startswith("__") and method_name.endswith("__")
    ):
        return None
    return owner_class_name, method_name


def _policy_selector_match(
    selector_call: ast.Call,
) -> tuple[str, str, tuple[str, ...]] | None:
    if not isinstance(selector_call.func, ast.Attribute):
        return None
    selector_method_name = selector_call.func.attr
    if not selector_method_name.startswith("for_"):
        return None
    policy_root_parts = _ast_attribute_chain(selector_call.func.value)
    if policy_root_parts is None:
        return None
    selector_source_exprs = _policy_selector_source_exprs(selector_call)
    if not selector_source_exprs or not any(
        (_looks_like_self_selector_source(expr) for expr in selector_source_exprs)
    ):
        return None
    return ".".join(policy_root_parts), selector_method_name, selector_source_exprs


def _nominal_policy_surface_method_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> NominalPolicySurfaceMethodCandidate | None:
    method_header = _nominal_policy_method_header(qualname, function)
    if method_header is None:
        return None
    chain_match = _transport_call_chain_match(function, exact_depth=2)
    if chain_match is None:
        return None
    owner_class_name, method_name = method_header
    chain, transported_values = chain_match
    outer_call, selector_call = chain
    selector_match = _policy_selector_match(selector_call)
    if selector_match is None:
        return None
    policy_root_symbol, selector_method_name, selector_source_exprs = selector_match
    policy_member_name = _call_name(outer_call.func) or ast.unparse(outer_call.func)
    return NominalPolicySurfaceMethodCandidate(
        file_path=str(module.path),
        line=function.lineno,
        qualname=qualname,
        owner_class_name=owner_class_name,
        method_name=method_name,
        policy_root_symbol=policy_root_symbol,
        selector_method_name=selector_method_name,
        policy_member_name=policy_member_name,
        selector_source_exprs=selector_source_exprs,
        transported_value_sources=sorted_tuple(
            {ast.unparse(value) for value in transported_values}
        ),
    )


def _nominal_policy_surface_family_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[NominalPolicySurfaceFamilyCandidate, ...]:
    min_family_size = max(2, config.min_registration_sites)
    method_candidates = tuple(
        (
            candidate
            for qualname, function in _iter_named_functions(module)
            for candidate in (
                _nominal_policy_surface_method_candidate(module, qualname, function),
            )
            if candidate is not None
        )
    )
    grouped: dict[
        tuple[str, str, str, tuple[str, ...]], list[NominalPolicySurfaceMethodCandidate]
    ] = defaultdict(list)
    for candidate in method_candidates:
        grouped[
            candidate.owner_class_name,
            candidate.policy_root_symbol,
            candidate.selector_method_name,
            candidate.selector_source_exprs,
        ].append(candidate)
    return sorted_tuple(
        (
            NominalPolicySurfaceFamilyCandidate(
                methods=sorted_tuple(
                    candidates, key=lambda item: (item.line, item.qualname)
                )
            )
            for (
                owner_class_name,
                policy_root_symbol,
                selector_method_name,
                selector_source_exprs,
            ), candidates in grouped.items()
            if len(candidates) >= min_family_size
        ),
        key=lambda item: (
            item.file_path,
            item.owner_class_name,
            item.policy_root_symbol,
            item.methods[0].line,
        ),
    )


# fmt: off
materialize_product_records((
    product_record_spec('_FunctionWrapperContext', 'module: ParsedModule; qualname: str; function: ast.FunctionDef | ast.AsyncFunctionDef; body: list[ast.stmt]; class_name: str | None; allowed_roots: set[str]'),
    product_record_spec('_ProjectionWrapperCall', 'bound_name: str; delegate_call: ast.Call; returned_value: ast.AST'),
))
# fmt: on


class _FunctionWrapperStep(RegisteredEffectStep):
    pass


class _DirectFunctionWrapperStep(
    _FunctionWrapperStep,
    GuardedEffectStep[_FunctionWrapperContext, FunctionWrapperCandidate],
):
    step_id = "direct_function_wrapper"
    registration_order = 10

    def project(
        self, value: _FunctionWrapperContext
    ) -> FunctionWrapperCandidate | None:
        returned_call = _single_returned_wrapper_call(value.body)
        if returned_call is None:
            return None
        delegate_symbol = _transported_delegate_symbol(
            returned_call,
            class_name=value.class_name,
            allowed_roots=value.allowed_roots,
        )
        if delegate_symbol is None:
            return None
        return _function_wrapper_candidate_from_context(
            value, delegate_symbol=delegate_symbol, wrapper_kind="direct"
        )


class _ProjectionFunctionWrapperStep(
    _FunctionWrapperStep,
    GuardedEffectStep[_FunctionWrapperContext, FunctionWrapperCandidate],
):
    step_id = "projection_function_wrapper"
    registration_order = 20

    def project(
        self, value: _FunctionWrapperContext
    ) -> FunctionWrapperCandidate | None:
        projection = _projection_wrapper_call(value.body)
        if projection is None:
            return None
        delegate_symbol = _transported_delegate_symbol(
            projection.delegate_call,
            class_name=value.class_name,
            allowed_roots=value.allowed_roots,
        )
        if delegate_symbol is None:
            return None
        projected_attributes = _projected_attribute_names(
            projection.returned_value, bound_name=projection.bound_name
        )
        if projected_attributes is None:
            return None
        return _function_wrapper_candidate_from_context(
            value,
            delegate_symbol=delegate_symbol,
            wrapper_kind="projection",
            projected_attributes=projected_attributes,
        )


def _function_wrapper_context(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _FunctionWrapperContext | None:
    body = _trim_docstring_body(function.body)
    if not body:
        return None
    return _FunctionWrapperContext(
        module=module,
        qualname=qualname,
        function=function,
        body=body,
        class_name=qualname.rsplit(".", 1)[0] if "." in qualname else None,
        allowed_roots=_function_param_names(function) | {"self", "cls"},
    )


def _single_returned_wrapper_call(body: list[ast.stmt]) -> ast.Call | None:
    returned = single_ast(body, ast.Return)
    return as_ast(returned.value if returned else None, ast.Call)


def _projection_wrapper_call(body: list[ast.stmt]) -> _ProjectionWrapperCall | None:
    statements = ast_sequence(body, ast.Assign, ast.Return)
    if statements is None:
        return None
    assigned, returned = statements
    assignment = named_call_assignment(assigned)
    if assignment is None or returned.value is None:
        return None
    return _ProjectionWrapperCall(
        bound_name=assignment.target_name,
        delegate_call=assignment.call,
        returned_value=returned.value,
    )


def _function_wrapper_candidate_from_context(
    context: _FunctionWrapperContext,
    *,
    delegate_symbol: str,
    wrapper_kind: str,
    projected_attributes: tuple[str, ...] = (),
) -> FunctionWrapperCandidate:
    return FunctionWrapperCandidate(
        file_path=str(context.module.path),
        qualname=context.qualname,
        lineno=context.function.lineno,
        delegate_symbol=delegate_symbol,
        wrapper_kind=wrapper_kind,
        statement_count=len(context.body),
        projected_attributes=projected_attributes,
    )


def _function_wrapper_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> FunctionWrapperCandidate | None:
    context = _function_wrapper_context(module, qualname, function)
    if context is None:
        return None
    return cast(
        FunctionWrapperCandidate | None,
        Maybe.of(context)
        .bind(FirstSuccessfulEffectStep(registered_effect_steps(_FunctionWrapperStep)))
        .unwrap_or_none(),
    )


def _function_wrapper_candidates(
    module: ParsedModule,
) -> tuple[FunctionWrapperCandidate, ...]:
    candidates = [
        candidate
        for qualname, function in _iter_named_functions(module)
        for candidate in (_function_wrapper_candidate(module, qualname, function),)
        if candidate is not None
    ]
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.lineno, item.qualname)
    )


def _wrapper_chain_candidates(
    module: ParsedModule,
) -> tuple[WrapperChainCandidate, ...]:
    candidates = _function_wrapper_candidates(module)
    if len(candidates) < 2:
        return ()
    by_symbol = {candidate.qualname: candidate for candidate in candidates}
    inbound = Counter(
        (
            candidate.delegate_symbol
            for candidate in candidates
            if candidate.delegate_symbol in by_symbol
        )
    )
    chains: list[WrapperChainCandidate] = []
    for candidate in candidates:
        if inbound[candidate.qualname] > 0:
            continue
        current = candidate
        chain = [candidate]
        seen = {candidate.qualname}
        while current.delegate_symbol in by_symbol:
            next_candidate = by_symbol[current.delegate_symbol]
            if next_candidate.qualname in seen:
                break
            chain.append(next_candidate)
            seen.add(next_candidate.qualname)
            current = next_candidate
        if len(chain) < 2:
            continue
        chains.append(
            WrapperChainCandidate(
                file_path=str(module.path),
                wrappers=tuple(chain),
                leaf_delegate_symbol=current.delegate_symbol,
            )
        )
    return sorted_tuple(
        chains, key=lambda item: (-len(item.wrappers), item.wrappers[0].lineno)
    )


def _pipeline_body_stages(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[PipelineAssemblyStage, ...] | None:
    body = list(function.body)
    if body and _is_docstring_expr(body[0]):
        body = body[1:]
    if len(body) < 2:
        return None
    stages: list[PipelineAssemblyStage] = []
    for statement in body:
        stage = _pipeline_stage(statement)
        if stage is None:
            return None
        stages.append(stage)
    if not stages or stages[-1].kind != _PIPELINE_RETURN_STAGE:
        return None
    return tuple(stages)


# fmt: off
materialize_product_record(product_record_spec('_PipelineStageSource', 'kind: str; call: ast.Call; output_arity: int'))
# fmt: on


def _pipeline_stage_source(statement: ast.stmt) -> _PipelineStageSource | None:
    assignment = as_ast(statement, ast.Assign)
    if assignment is not None:
        call = as_ast(assignment.value, ast.Call)
        output_arity = _assignment_target_arity(single_assign_target(assignment))
        if call is None or output_arity is None:
            return None
        return _PipelineStageSource(_PIPELINE_ASSIGN_STAGE, call, output_arity)
    call = return_call(statement)
    if call is None:
        return None
    return _PipelineStageSource(_PIPELINE_RETURN_STAGE, call, 0)


def _pipeline_stage(statement: ast.stmt) -> PipelineAssemblyStage | None:
    source = _pipeline_stage_source(statement)
    callee_name = _call_name(source.call.func) if source is not None else None
    if source is None or callee_name is None:
        return None
    keyword_names = tuple(
        (keyword.arg for keyword in source.call.keywords if keyword.arg is not None)
    )
    return PipelineAssemblyStage(
        kind=source.kind,
        callee_name=callee_name,
        output_arity=source.output_arity,
        arg_count=len(source.call.args) + len(keyword_names),
        keyword_names=keyword_names,
    )


def _return_none_statement(statement: ast.stmt) -> bool:
    return bool(
        isinstance(statement, ast.Return)
        and isinstance(statement.value, ast.Constant)
        and (statement.value.value is None)
    )


def _success_return_statement(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Return) and (not _return_none_statement(statement))


def _none_guard_binding_names(test: ast.AST) -> tuple[str, ...]:
    if isinstance(test, ast.BoolOp):
        names: set[str] = set()
        for value in test.values:
            names.update(_none_guard_binding_names(value))
        return sorted_tuple(names)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        if isinstance(test.operand, ast.Name):
            return (test.operand.id,)
        return _none_guard_binding_names(test.operand)
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or len(test.comparators) != 1
        or (not isinstance(test.ops[0], (ast.Is, ast.Eq)))
    ):
        return ()
    left = test.left
    right = test.comparators[0]
    if (
        isinstance(left, ast.Name)
        and isinstance(right, ast.Constant)
        and (right.value is None)
    ):
        return (left.id,)
    if (
        isinstance(right, ast.Name)
        and isinstance(left, ast.Constant)
        and (left.value is None)
    ):
        return (right.id,)
    return ()


def _effect_stage_kind(statement: ast.stmt) -> str:
    if isinstance(statement, ast.If) and _if_returns_none_only(statement):
        return "fail_soft_guard"
    if isinstance(statement, ast.Assign):
        return (
            "call_assignment" if isinstance(statement.value, ast.Call) else "assignment"
        )
    if isinstance(statement, ast.AnnAssign):
        return (
            "call_assignment" if isinstance(statement.value, ast.Call) else "assignment"
        )
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        return "loop"
    if isinstance(statement, ast.Try):
        return "exception_boundary"
    if _success_return_statement(statement):
        return "success_return"
    if isinstance(statement, ast.If):
        return "branch"
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        return "effect_call"
    return statement.__class__.__name__


def _return_kind(statement: ast.Return) -> str:
    value = statement.value
    if isinstance(value, ast.Call):
        call_name = _call_display_name(value)
        return call_name or "call"
    if isinstance(value, ast.Tuple):
        return "tuple"
    if isinstance(value, ast.List):
        return _BuiltinCollectionName.LIST
    if isinstance(value, ast.Dict):
        return "dict"
    if isinstance(value, ast.Name):
        return "name"
    if isinstance(value, ast.Attribute):
        return "attribute"
    return value.__class__.__name__ if value is not None else "implicit"


def _statement_call_names(statement: ast.stmt) -> tuple[str, ...]:
    names: set[str] = set()
    for node in _walk_nodes(statement):
        if isinstance(node, ast.Call) and (call_name := _call_display_name(node)):
            names.add(call_name)
    return sorted_tuple(names)


def _effect_pipeline_normal_form(
    *,
    helper_call_names: tuple[str, ...],
    stage_kinds: tuple[str, ...],
    success_return_kind: str,
) -> str:
    helper_names = frozenset(helper_call_names)
    if helper_names & {
        "_call_chain_delegate_symbol",
        "_call_chain_from_outer_call",
        "_call_chain_transport_values",
    }:
        return "transport_call_chain_matcher"
    if helper_names & {"_enum_member_ref", "_enum_member_refs_for_known_key_types"}:
        return "comparison_guard_matcher"
    if helper_names & {
        "_extends_subclasses_queue",
        "_queue_pop_target_name",
        "_registry_materialization_kind",
    }:
        return "loop_fold_matcher"
    if helper_names & {
        "_guarded_node_types",
        "_self_attribute_name",
        "_terminal_name",
        "_type_name_tuple",
    }:
        return "ast_shape_matcher"
    if "loop" in stage_kinds or "exception_boundary" in stage_kinds:
        return "statement_sequence_matcher"
    if success_return_kind in _RETURN_COLLECTION_KIND_NAMES:
        return "statement_sequence_matcher"
    return "typed_effect_carrier"


def _fail_soft_effect_pipeline_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    config: DetectorConfig,
) -> FailSoftEffectPipelineCandidate | None:
    body = _trim_docstring_body(function.body)
    guard_statements = tuple(
        (
            statement
            for statement in body
            if isinstance(statement, ast.If) and _if_returns_none_only(statement)
        )
    )
    if len(guard_statements) < config.min_effect_guard_stages:
        return None
    success_returns = tuple(
        (
            statement
            for statement in body
            if isinstance(statement, ast.Return)
            and _success_return_statement(statement)
        )
    )
    if not success_returns:
        return None
    last_guard_index = max(
        (index for index, statement in enumerate(body) if statement in guard_statements)
    )
    last_success_index = max(
        (index for index, statement in enumerate(body) if statement in success_returns)
    )
    if last_success_index <= last_guard_index:
        return None
    guarded_binding_names = sorted_tuple(
        {
            name
            for guard_statement in guard_statements
            for name in _none_guard_binding_names(guard_statement.test)
        }
    )
    helper_call_names = sorted_tuple(
        {name for statement in body for name in _statement_call_names(statement)}
    )
    stage_kinds = tuple(_effect_stage_kind(statement) for statement in body)
    success_return_kind = _return_kind(success_returns[-1])
    return FailSoftEffectPipelineCandidate(
        file_path=str(module.path),
        line=function.lineno,
        function_name=qualname,
        line_count=(
            function.end_lineno if function.end_lineno is not None else function.lineno
        )
        - function.lineno
        + 1,
        guard_count=len(guard_statements),
        normal_form=_effect_pipeline_normal_form(
            helper_call_names=helper_call_names,
            stage_kinds=stage_kinds,
            success_return_kind=success_return_kind,
        ),
        guarded_binding_names=guarded_binding_names,
        stage_kinds=stage_kinds,
        success_return_kind=success_return_kind,
        helper_call_names=helper_call_names,
    )


def _fail_soft_effect_pipeline_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[FailSoftEffectPipelineCandidate, ...]:
    candidates = tuple(
        (
            candidate
            for qualname, function in _iter_named_functions(module)
            if (
                candidate := _fail_soft_effect_pipeline_candidate(
                    module, qualname, function, config
                )
            )
            is not None
        )
    )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.function_name)
    )


_SEMANTIC_MATCH_HELPER_NAMES = frozenset(
    {
        "as_ast",
        "ast_sequence",
        "attribute_name",
        "call_attribute_name",
        "name_id",
        "single_assign_target",
        "single_ast",
        "single_call_arg",
        "single_call_arg_name",
        "single_item",
        "single_return_call",
        "single_return_value",
    }
)


def _ast_type_name(node: ast.AST) -> str | None:
    return (
        node.attr
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and (node.value.id == "ast")
        else None
    )


def _ast_type_names(node: ast.AST) -> tuple[str, ...]:
    if ast_type_name := _ast_type_name(node):
        return (ast_type_name,)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(
            (
                ast_type_name
                for item in node.elts
                if (ast_type_name := _ast_type_name(item)) is not None
            )
        )
    return ()


def _isinstance_ast_type_names(node: ast.AST) -> tuple[str, ...]:
    call = as_ast(node, ast.Call)
    if call is None or _call_name(call.func) != "isinstance" or len(call.args) < 2:
        return ()
    return _ast_type_names(call.args[1])


def _len_cardinality_guard_count(node: ast.AST) -> int:
    return sum(
        (
            1
            for item in _walk_nodes(node)
            if isinstance(item, ast.Compare)
            and any(
                (
                    isinstance(value, ast.Call) and _call_name(value.func) == "len"
                    for value in (item.left, *item.comparators)
                )
            )
        )
    )


def _semantic_match_helper_names(node: ast.AST) -> tuple[str, ...]:
    return sorted_tuple(
        {
            call_name
            for item in _walk_nodes(node)
            if isinstance(item, ast.Call)
            for call_name in (_call_name(item.func),)
            if call_name in _SEMANTIC_MATCH_HELPER_NAMES
        }
    )


def _uses_effect_step_pipeline(node: ast.AST) -> bool:
    return any(
        (
            isinstance(item, ast.Call)
            and (
                _call_name(item.func) in {"Maybe.of", "bind_all"}
                or (
                    isinstance(item.func, ast.Attribute)
                    and item.func.attr in {"bind_all", "bind_step"}
                )
            )
            for item in _walk_nodes(node)
        )
    )


def _effect_step_normal_form(
    *,
    ast_type_names: tuple[str, ...],
    semantic_helper_names: tuple[str, ...],
    cardinality_guard_count: int,
) -> str:
    ast_type_set = frozenset(ast_type_names)
    helper_set = frozenset(semantic_helper_names)
    if helper_set & {"single_return_call", "single_return_value"}:
        return "statement_sequence_effect_steps"
    if cardinality_guard_count >= 2 or helper_set & {"single_item", "ast_sequence"}:
        return "cardinality_guard_effect_steps"
    if ast_type_set & {"Call", "Attribute", "Name", "GeneratorExp", "Compare"}:
        return "ast_shape_effect_steps"
    return "typed_effect_steps"


_EFFECT_STEP_REPLACEMENT_SHAPE = ObjectFamilyShape(
    shared_objects=("effect_carrier", "registered_step_abc", "matcher_shell"),
    per_axis_objects=(
        "step_declaration",
        "step_identity",
        "ordering_slot",
        "semantic_hook",
    ),
)


@dataclass(frozen=True)
class _EffectStepPayoffProfile:
    none_return_count: int
    ast_type_guard_count: int
    cardinality_guard_count: int
    semantic_helper_count: int
    ast_type_names: tuple[str, ...]
    semantic_helper_names: tuple[str, ...]
    has_success_return: bool
    already_uses_effect_step_pipeline: bool

    @property
    def payoff_score(self) -> int:
        return (
            self.none_return_count
            + self.ast_type_guard_count
            + self.cardinality_guard_count
            + self.semantic_helper_count
        )

    @property
    def normal_form(self) -> str:
        return _effect_step_normal_form(
            ast_type_names=self.ast_type_names,
            semantic_helper_names=self.semantic_helper_names,
            cardinality_guard_count=self.cardinality_guard_count,
        )

    @property
    def estimated_step_count(self) -> int:
        return len(self.compression_certificate.semantic_axes)

    @property
    def semantic_axes(self) -> tuple[str, ...]:
        return (
            tuple((f"ast_type:{name}" for name in self.ast_type_names))
            + (("optional_exit",) if self.none_return_count else ())
            + (("cardinality_guard",) if self.cardinality_guard_count else ())
            + (("semantic_helper",) if self.semantic_helper_count else ())
        )

    @property
    def compression_certificate(self) -> CompressionCertificate:
        return CompressionCertificate.from_object_family(
            manual_object_count=self.payoff_score,
            replacement_shape=_EFFECT_STEP_REPLACEMENT_SHAPE,
            semantic_axes=self.semantic_axes,
        )

    @property
    def generated_object_budget(self) -> int:
        return self.compression_certificate.after_description_length

    @property
    def net_object_savings(self) -> int:
        return self.compression_certificate.description_length_savings

    @property
    def description_length_before(self) -> int:
        return self.compression_certificate.before_description_length

    @property
    def description_length_after(self) -> int:
        return self.compression_certificate.after_description_length

    @property
    def description_length_savings(self) -> int:
        return self.compression_certificate.certified_description_length_savings

    def qualifies(self, config: DetectorConfig) -> bool:
        return (
            not self.already_uses_effect_step_pipeline
            and self.none_return_count >= 2
            and (self.payoff_score >= config.min_effect_step_payoff_score)
            and (self.ast_type_guard_count + self.semantic_helper_count >= 3)
            and self.compression_certificate.pays_rent
            and self.has_success_return
        )


def _effect_step_payoff_profile(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _EffectStepPayoffProfile:
    body = _trim_docstring_body(function.body)
    none_return_count = sum(
        (
            1
            for item in _walk_nodes(ast.Module(body=body, type_ignores=[]))
            if isinstance(item, ast.Return)
            and isinstance(item.value, ast.Constant)
            and (item.value.value is None)
        )
    )
    ast_type_names = sorted_tuple(
        {
            ast_type_name
            for item in _walk_nodes(ast.Module(body=body, type_ignores=[]))
            for ast_type_name in _isinstance_ast_type_names(item)
        }
    )
    body_module = ast.Module(body=body, type_ignores=[])
    semantic_helper_names = _semantic_match_helper_names(body_module)
    ast_type_guard_count = sum(
        (1 for item in _walk_nodes(body_module) if _isinstance_ast_type_names(item))
    )
    semantic_helper_count = sum(
        (
            1
            for item in _walk_nodes(body_module)
            if isinstance(item, ast.Call)
            and _call_name(item.func) in _SEMANTIC_MATCH_HELPER_NAMES
        )
    )
    return _EffectStepPayoffProfile(
        none_return_count=none_return_count,
        ast_type_guard_count=ast_type_guard_count,
        cardinality_guard_count=_len_cardinality_guard_count(body_module),
        semantic_helper_count=semantic_helper_count,
        ast_type_names=ast_type_names,
        semantic_helper_names=semantic_helper_names,
        has_success_return=any(
            (
                isinstance(item, ast.Return)
                and item.value is not None
                and (
                    not (
                        isinstance(item.value, ast.Constant)
                        and item.value.value is None
                    )
                )
                for item in _walk_nodes(body_module)
            )
        ),
        already_uses_effect_step_pipeline=_uses_effect_step_pipeline(body_module),
    )


def _effect_step_amortization_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    config: DetectorConfig,
) -> EffectStepAmortizationCandidate | None:
    profile = _effect_step_payoff_profile(function)
    if not profile.qualifies(config):
        return None
    return EffectStepAmortizationCandidate(
        file_path=str(module.path),
        line=function.lineno,
        function_name=qualname,
        line_count=(
            function.end_lineno if function.end_lineno is not None else function.lineno
        )
        - function.lineno
        + 1,
        payoff_score=profile.payoff_score,
        none_return_count=profile.none_return_count,
        ast_type_guard_count=profile.ast_type_guard_count,
        cardinality_guard_count=profile.cardinality_guard_count,
        semantic_helper_count=profile.semantic_helper_count,
        ast_type_names=profile.ast_type_names,
        semantic_helper_names=profile.semantic_helper_names,
        normal_form=profile.normal_form,
        estimated_step_count=profile.estimated_step_count,
        generated_object_budget=profile.generated_object_budget,
        net_object_savings=profile.net_object_savings,
        description_length_before=profile.description_length_before,
        description_length_after=profile.description_length_after,
        description_length_savings=profile.description_length_savings,
        compression_certificate=profile.compression_certificate,
    )


def _effect_step_amortization_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[EffectStepAmortizationCandidate, ...]:
    candidates = tuple(
        (
            candidate
            for qualname, function in _iter_named_functions(module)
            if (
                candidate := _effect_step_amortization_candidate(
                    module, qualname, function, config
                )
            )
            is not None
        )
    )
    return sorted_tuple(
        candidates, key=lambda item: (-item.payoff_score, item.file_path, item.line)
    )


def _public_top_level_declarations(
    module: ParsedModule,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    return {
        node.name: node
        for node in module.module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and (not node.name.startswith("_"))
    }


def _declares_effect_infrastructure(
    declarations: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef],
) -> bool:
    return any(
        (name.endswith(("EffectStep", "EffectCarrier")) for name in declarations)
    )


def _local_symbol_reference_sites(
    modules: Sequence[ParsedModule],
) -> dict[str, set[SourceLocation]]:
    references: dict[str, set[SourceLocation]] = defaultdict(set)

    class Visitor(ast.NodeVisitor):
        def __init__(self, module: ParsedModule) -> None:
            self.module = module
            self.stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Name(self, node: ast.Name) -> None:
            references[node.id].add(self._site(node.lineno))

        def visit_Attribute(self, node: ast.Attribute) -> None:
            references[node.attr].add(self._site(node.lineno))
            self.generic_visit(node)

        def _site(self, line: int) -> SourceLocation:
            symbol = ".".join(self.stack) if self.stack else "<module>"
            return SourceLocation(str(self.module.path), line, symbol)

    for module in modules:
        Visitor(module).visit(module.module)
    return references


def _public_declaration_reference_names(
    declaration: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    public_names: frozenset[str],
) -> frozenset[str]:
    return frozenset(
        (
            name
            for item in _walk_nodes(declaration)
            for name in (
                item.id if isinstance(item, ast.Name) else None,
                item.attr if isinstance(item, ast.Attribute) else None,
            )
            if name in public_names and name != declaration.name
        )
    )


def _under_amortized_infrastructure_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[UnderAmortizedInfrastructureCandidate, ...]:
    reference_sites = _local_symbol_reference_sites(modules)
    candidates: list[UnderAmortizedInfrastructureCandidate] = []
    for module in modules:
        declarations = _public_top_level_declarations(module)
        if not declarations or not _declares_effect_infrastructure(declarations):
            continue
        module_path = str(module.path)
        public_names = frozenset(declarations)
        external_consumers = {
            name: frozenset(
                (
                    site.symbol
                    for site in reference_sites.get(name, set())
                    if site.file_path != module_path
                )
            )
            for name in public_names
        }
        declaration_refs = {
            name: _public_declaration_reference_names(node, public_names)
            for name, node in declarations.items()
        }
        amortized_support_names = frozenset(
            (
                support_name
                for name, refs in declaration_refs.items()
                if len(external_consumers[name]) > 1
                for support_name in refs
            )
        )
        names = sorted_tuple(
            (
                name
                for name in public_names
                if len(external_consumers[name]) == 1
                and name not in amortized_support_names
            )
        )
        if not names:
            continue
        internal_consumers: dict[str, set[str]] = defaultdict(set)
        for name, refs in declaration_refs.items():
            for ref_name in refs:
                internal_consumers[ref_name].add(name)
        support_names = sorted_tuple(
            (
                ref_name
                for name in names
                for ref_name in declaration_refs[name]
                if not external_consumers[ref_name]
                and internal_consumers[ref_name] <= {name}
            )
        )
        consumers = sorted_tuple(
            {consumer for name in names for consumer in external_consumers[name]}
        )
        first_line = min(declarations[name].lineno for name in names)
        candidates.append(
            UnderAmortizedInfrastructureCandidate(
                file_path=module_path,
                line=first_line,
                declaration_names=names,
                consumer_symbols=consumers,
                support_names=support_names,
            )
        )
    return sorted_tuple(candidates, key=lambda item: (item.file_path, item.line))


_CANDIDATE_COLLECTOR_METHOD_NAME = "_candidate_items"
_CANDIDATE_COLLECTOR_BASE_BY_SHAPE = {
    ("module", False): "ModuleCollectorCandidateDetector",
    ("module", True): "ConfiguredModuleCollectorCandidateDetector",
    ("cross_module", False): "CrossModuleCollectorCandidateDetector",
    ("cross_module", True): "ConfiguredCrossModuleCollectorCandidateDetector",
}
_DECLARATIVE_DETECTOR_BASE_NAMES = frozenset(
    _CANDIDATE_COLLECTOR_BASE_BY_SHAPE.values()
)


def _subscript_base_parts(base: ast.AST) -> tuple[str, str] | None:
    if not isinstance(base, ast.Subscript):
        return None
    base_name = name_id(base.value)
    return None if base_name is None else (base_name, ast.unparse(base.slice))


def _class_assignment_names(node: ast.ClassDef) -> tuple[str, ...]:
    assignment_names: list[str] = []
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            assignment_names.extend(
                (
                    name
                    for target in statement.targets
                    for name in (name_id(target),)
                    if name is not None
                )
            )
        elif isinstance(statement, ast.AnnAssign):
            target_name = name_id(statement.target)
            if target_name is not None:
                assignment_names.append(target_name)
        elif not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            return ()
    return tuple(assignment_names)


def _declarative_detector_class_candidates(
    module: ParsedModule,
) -> tuple[DeclarativeDetectorClassCandidate, ...]:
    candidates: list[DeclarativeDetectorClassCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or node.end_lineno is None:
            continue
        base_parts = tuple(
            part
            for base in node.bases
            for part in (_subscript_base_parts(base),)
            if part is not None
        )
        base_part = single_item(
            tuple(
                part
                for part in base_parts
                if part[0] in _DECLARATIVE_DETECTOR_BASE_NAMES
            )
        )
        assignment_names = _class_assignment_names(node)
        if (
            base_part is None
            or "finding_spec" not in assignment_names
            or "finding_renderer" not in assignment_names
        ):
            continue
        candidates.append(
            DeclarativeDetectorClassCandidate(
                file_path=str(module.path),
                line=node.lineno,
                class_name=node.name,
                base_name=base_part[0],
                candidate_type_name=base_part[1],
                assignment_names=assignment_names,
                line_count=node.end_lineno - node.lineno + 1,
            )
        )
    return tuple(candidates)


_STATIC_OBSERVATION_DETECTOR_BASE_NAMES = frozenset({"StaticModulePatternDetector"})
_STATIC_OBSERVATION_CONTROL_FLOW_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.Match,
)


def _typed_observation_collection_call(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str] | None:
    if any(
        (
            isinstance(child, _STATIC_OBSERVATION_CONTROL_FLOW_NODES)
            for child in ast.walk(method)
        )
    ):
        return None
    collection_calls: list[tuple[str, str]] = []
    for child in ast.walk(method):
        if not isinstance(child, ast.Call):
            continue
        if name_id(child.func) != "_collect_typed_family_items":
            continue
        if len(child.args) < 3 or name_id(child.args[0]) != "module":
            continue
        family_name = name_id(child.args[1])
        observation_type_name = name_id(child.args[2])
        if family_name is not None and observation_type_name is not None:
            collection_calls.append((family_name, observation_type_name))
    return single_item(tuple(collection_calls))


def _source_location_from_line_symbol_call(
    node: ast.AST,
) -> bool:
    if not isinstance(node, ast.Call) or name_id(node.func) != "SourceLocation":
        return False
    if len(node.args) < 3:
        return False
    return all(
        (
            isinstance(argument, ast.Attribute) and argument.attr == expected_attribute
            for argument, expected_attribute in zip(
                node.args[:3], ("file_path", "line", "symbol"), strict=True
            )
        )
    )


def _static_minimum_evidence_count(node: ast.ClassDef) -> int | None:
    minimum_method = _class_method_named(node, "_minimum_evidence")
    if minimum_method is None:
        return 1
    returns = tuple(
        child for child in ast.walk(minimum_method) if isinstance(child, ast.Return)
    )
    returned = single_item(returns)
    if returned is None:
        return None
    value = constant_value(returned.value)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _static_summary_expression(module: ParsedModule, node: ast.ClassDef) -> str | None:
    summary_method = _class_method_named(node, "_summary")
    if summary_method is None:
        return None
    returns = tuple(
        child for child in ast.walk(summary_method) if isinstance(child, ast.Return)
    )
    returned = single_item(returns)
    if returned is None or returned.value is None:
        return None
    return _source_segment(module, returned.value)


def _static_typed_observation_detector_candidates(
    module: ParsedModule,
) -> tuple[StaticTypedObservationDetectorCandidate, ...]:
    candidates: list[StaticTypedObservationDetectorCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or node.end_lineno is None:
            continue
        if not (
            _STATIC_OBSERVATION_DETECTOR_BASE_NAMES & set(_declared_base_names(node))
        ):
            continue
        evidence_method = _class_method_named(node, "_module_evidence")
        if evidence_method is None:
            continue
        collection = _typed_observation_collection_call(evidence_method)
        if collection is None:
            continue
        if not any(
            _source_location_from_line_symbol_call(child)
            for child in ast.walk(evidence_method)
        ):
            continue
        minimum_evidence = _static_minimum_evidence_count(node)
        summary_expression = _static_summary_expression(module, node)
        if minimum_evidence is None or summary_expression is None:
            continue
        family_name, observation_type_name = collection
        candidates.append(
            StaticTypedObservationDetectorCandidate(
                file_path=str(module.path),
                line=node.lineno,
                class_name=node.name,
                observation_family_name=family_name,
                observation_type_name=observation_type_name,
                minimum_evidence_count=minimum_evidence,
                summary_expression=summary_expression,
                line_count=node.end_lineno - node.lineno + 1,
            )
        )
    return tuple(candidates)


_DECLARE_MODULE_DETECTOR_NAME = "declare_module_detector"
_DECLARE_CANDIDATE_RULE_DETECTOR_NAME = "declare_candidate_rule_detector"
_DECLARE_TYPED_OBSERVATION_DETECTOR_NAME = "declare_typed_observation_detector"
_DETECTOR_DECLARATION_CALL_NAMES = frozenset(
    {
        _DECLARE_CANDIDATE_RULE_DETECTOR_NAME,
        _DECLARE_MODULE_DETECTOR_NAME,
        _DECLARE_TYPED_OBSERVATION_DETECTOR_NAME,
    }
)
_CANDIDATE_FINDING_RENDERER_NAME = "CandidateFindingRenderer"
_ABSTRACTION_RENT_TRIGGER_TERMS = frozenset(
    {
        "algebra",
        "base",
        "collector",
        "compose",
        "composition",
        "delegate",
        "factory",
        "helper",
        "hook",
        "infrastructure",
        "metaprogram",
        "metaprogrammed",
        "mixin",
        "registry",
        "template",
        "traversal",
        "wrapper",
    }
)
_DETECTOR_ACTION_KEYWORD_NAMES = frozenset({"scaffold", "codemod_patch"})
_DETECTOR_GUARD_KEYWORD_NAMES = frozenset({"compression_certificate", "metrics"})
_NET_REDUCTION_GUARD_TERMS = frozenset(
    {
        "collapse",
        "cut",
        "delete",
        "inline",
        "less",
        "lower",
        "net",
        "reduce",
        "remove",
        "replace",
    }
)
_AMORTIZATION_GUARD_TERMS = frozenset(
    {
        "amortization",
        "amortize",
        "amortized",
        "fanout",
        "more than one",
        "multiple consumer",
        "payoff",
        "pays rent",
        "repeated external",
        "single-consumer",
    }
)


def _all_string_literals(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        (
            literal.value
            for literal in ast.walk(node)
            if isinstance(literal, ast.Constant) and isinstance(literal.value, str)
        )
    )


def _matched_terms(text: str, terms: frozenset[str]) -> tuple[str, ...]:
    lowered = text.lower()
    return sorted_tuple((term for term in terms if term in lowered))


def _keyword_value_text(call: ast.Call, keyword_names: frozenset[str]) -> str:
    return "\n".join(
        (
            "\n".join(_all_string_literals(keyword.value))
            for keyword in call.keywords
            if keyword.arg in keyword_names
        )
    )


def _keyword_source_text(
    module: ParsedModule, call: ast.Call, keyword_names: frozenset[str]
) -> str:
    return "\n".join(
        (
            _source_segment(module, keyword.value)
            for keyword in call.keywords
            if keyword.arg in keyword_names
        )
    )


def _detector_declaration_candidate_type_name(call: ast.Call) -> str | None:
    call_name = _call_name(call.func)
    if call_name == _DECLARE_TYPED_OBSERVATION_DETECTOR_NAME:
        if not call.args:
            return None
        detector_name = constant_value(call.args[0])
        return detector_name if isinstance(detector_name, str) else None
    if call_name in {
        _DECLARE_CANDIDATE_RULE_DETECTOR_NAME,
        _DECLARE_MODULE_DETECTOR_NAME,
    }:
        return _source_segment_from_node(call.args[0]) if call.args else None
    return None


def _source_segment_from_node(node: ast.AST) -> str:
    return ast.unparse(node)


def _detector_declaration_qualname(call: ast.Call) -> str | None:
    candidate_type_name = _detector_declaration_candidate_type_name(call)
    if candidate_type_name is None:
        return None
    if candidate_type_name.endswith("Detector"):
        return candidate_type_name
    return f"{candidate_type_name.removesuffix('Candidate')}Detector"


def _detector_class_candidate_type_name(node: ast.ClassDef) -> str:
    candidate_type_name = single_item(
        tuple(
            (
                base_part[1]
                for base in node.bases
                for base_part in (_subscript_base_parts(base),)
                if base_part is not None and base_part[0].endswith("CandidateDetector")
            )
        )
    )
    return (
        candidate_type_name
        if candidate_type_name is not None
        else f"{node.name.removesuffix('Detector')}Candidate"
    )


def _class_assigns_finding_spec(node: ast.ClassDef) -> bool:
    return any(
        (
            isinstance(statement, ast.Assign)
            and any((name_id(target) == "finding_spec" for target in statement.targets))
        )
        for statement in node.body
    )


def _is_payoff_guarded_detector_class(node: ast.ClassDef) -> bool:
    return node.name.endswith("Detector") and _class_assigns_finding_spec(node)


def _build_finding_action_text(node: ast.ClassDef) -> str:
    return "\n".join(
        (
            _keyword_value_text(call, _DETECTOR_ACTION_KEYWORD_NAMES)
            for call in (
                child for child in ast.walk(node) if isinstance(child, ast.Call)
            )
            if _call_name(call.func) == "self.build_finding"
        )
    )


def _build_finding_guard_text(module: ParsedModule, node: ast.ClassDef) -> str:
    return "\n".join(
        (
            _source_segment(module, keyword.value)
            for call in (
                child for child in ast.walk(node) if isinstance(child, ast.Call)
            )
            if _call_name(call.func) == "self.build_finding"
            for keyword in call.keywords
            if keyword.arg in _DETECTOR_GUARD_KEYWORD_NAMES
        )
    )


def _detector_payoff_missing_guard_names(
    action_text: str, guard_text: str
) -> tuple[str, ...]:
    has_structured_payoff = bool(guard_text.strip())
    has_compression_certificate = "compression_certificate" in guard_text
    has_backend_loc_budget = has_structured_payoff
    has_net_reduction_action = bool(
        _matched_terms(action_text, _NET_REDUCTION_GUARD_TERMS)
    )
    has_amortization_or_fanout = bool(
        has_structured_payoff and _matched_terms(guard_text, _AMORTIZATION_GUARD_TERMS)
    )
    if (
        has_compression_certificate
        or has_amortization_or_fanout
        or (has_backend_loc_budget and has_net_reduction_action)
    ):
        return ()
    missing: list[str] = []
    if not has_structured_payoff:
        missing.append("structured_payoff_metrics")
    if not has_backend_loc_budget:
        missing.append("backend_loc_budget")
    if not has_net_reduction_action:
        missing.append("net_reduction_action")
    if not has_amortization_or_fanout:
        missing.append("amortization_or_fanout_gate")
    if not has_compression_certificate:
        missing.append("compression_certificate_or_explicit_fanout")
    return tuple(missing)


def _detector_backend_payoff_guard_candidates(
    module: ParsedModule,
) -> tuple[DetectorBackendPayoffGuardCandidate, ...]:
    candidates: list[DetectorBackendPayoffGuardCandidate] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) not in _DETECTOR_DECLARATION_CALL_NAMES:
            continue
        action_text = _keyword_value_text(node, _DETECTOR_ACTION_KEYWORD_NAMES)
        abstraction_terms = _matched_terms(action_text, _ABSTRACTION_RENT_TRIGGER_TERMS)
        if not abstraction_terms:
            continue
        guard_text = "\n".join(
            (
                _keyword_value_text(node, _DETECTOR_GUARD_KEYWORD_NAMES),
                _keyword_source_text(module, node, _DETECTOR_GUARD_KEYWORD_NAMES),
            )
        )
        missing_guard_names = _detector_payoff_missing_guard_names(
            action_text, guard_text
        )
        if not missing_guard_names:
            continue
        qualname = _detector_declaration_qualname(node)
        candidate_type_name = _detector_declaration_candidate_type_name(node)
        if qualname is None or candidate_type_name is None:
            continue
        declaration_line_count = (node.end_lineno or node.lineno) - node.lineno + 1
        candidates.append(
            DetectorBackendPayoffGuardCandidate(
                file_path=str(module.path),
                line=node.lineno,
                qualname=qualname,
                candidate_type_name=candidate_type_name,
                abstraction_terms=abstraction_terms,
                missing_guard_names=missing_guard_names,
                declaration_line_count=declaration_line_count,
            )
        )
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or not _is_payoff_guarded_detector_class(
            node
        ):
            continue
        action_text = _build_finding_action_text(node)
        abstraction_terms = _matched_terms(action_text, _ABSTRACTION_RENT_TRIGGER_TERMS)
        if not abstraction_terms:
            continue
        missing_guard_names = _detector_payoff_missing_guard_names(
            action_text, _build_finding_guard_text(module, node)
        )
        if not missing_guard_names:
            continue
        declaration_line_count = (node.end_lineno or node.lineno) - node.lineno + 1
        candidates.append(
            DetectorBackendPayoffGuardCandidate(
                file_path=str(module.path),
                line=node.lineno,
                qualname=node.name,
                candidate_type_name=_detector_class_candidate_type_name(node),
                abstraction_terms=abstraction_terms,
                missing_guard_names=missing_guard_names,
                declaration_line_count=declaration_line_count,
            )
        )
    return sorted_tuple(candidates, key=lambda item: (item.file_path, item.line))


def _renderer_keyword_names(call: ast.Call) -> tuple[str, ...]:
    return tuple((keyword.arg for keyword in call.keywords if keyword.arg is not None))


def _is_candidate_finding_renderer_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (
        _call_name(node.func) == _CANDIDATE_FINDING_RENDERER_NAME
    )


def _is_single_candidate_evidence_lambda(node: ast.AST) -> bool:
    if not isinstance(node, ast.Lambda) or len(node.args.args) != 1:
        return False
    parameter_name = node.args.args[0].arg
    if not isinstance(node.body, ast.Tuple) or len(node.body.elts) != 1:
        return False
    evidence_expr = node.body.elts[0]
    return (
        isinstance(evidence_expr, ast.Attribute)
        and evidence_expr.attr == "evidence"
        and isinstance(evidence_expr.value, ast.Name)
        and evidence_expr.value.id == parameter_name
    )


def _inline_candidate_renderer_declaration_candidates(
    module: ParsedModule,
) -> tuple[InlineCandidateRendererDeclarationCandidate, ...]:
    candidates: list[InlineCandidateRendererDeclarationCandidate] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) != _DECLARE_MODULE_DETECTOR_NAME or len(node.args) < 3:
            continue
        renderer = node.args[2]
        if not _is_candidate_finding_renderer_call(renderer):
            continue
        renderer_call = cast(ast.Call, renderer)
        renderer_keywords = {
            keyword.arg: keyword.value
            for keyword in renderer_call.keywords
            if keyword.arg is not None
        }
        if "summary" not in renderer_keywords or "evidence" not in renderer_keywords:
            continue
        candidate_type_name = _source_segment(module, node.args[0])
        candidates.append(
            InlineCandidateRendererDeclarationCandidate(
                file_path=str(module.path),
                line=node.lineno,
                qualname=f"{_DECLARE_MODULE_DETECTOR_NAME}[{candidate_type_name}]",
                candidate_type_name=candidate_type_name,
                renderer_keyword_names=_renderer_keyword_names(renderer_call),
                detector_keyword_names=tuple(
                    (
                        keyword.arg
                        for keyword in node.keywords
                        if keyword.arg is not None
                    )
                ),
                has_single_candidate_evidence=_is_single_candidate_evidence_lambda(
                    renderer_keywords["evidence"]
                ),
                line_count=(node.end_lineno or node.lineno) - node.lineno + 1,
            )
        )
    return tuple(candidates)


_NAMED_FUNCTION_ITERATOR_NAME = "_iter_named_functions"
_CANDIDATE_ACCUMULATOR_NAME = "candidates"
_AST_STREAM_CALL_NAMES = frozenset({"ast.walk", "_walk_nodes"})
_APPEND_METHOD_NAME = "append"


def _assigns_candidate_accumulator(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == _CANDIDATE_ACCUMULATOR_NAME
    ) or (
        isinstance(node, ast.Assign)
        and any(
            (
                isinstance(target, ast.Name)
                and target.id == _CANDIDATE_ACCUMULATOR_NAME
                for target in node.targets
            )
        )
    )


def _is_named_function_iteration(node: ast.For) -> bool:
    return (
        isinstance(node.iter, ast.Call)
        and _call_name(node.iter.func) == _NAMED_FUNCTION_ITERATOR_NAME
    )


class _CandidateAppendConstructorNameStep(RegisteredEffectStep):
    pass


class _CandidateAccumulatorAppendArgumentStep(
    _CandidateAppendConstructorNameStep,
    AstTypedEffectStep[ast.Call, ast.AST],
):
    step_id = "candidate_accumulator_append_argument"
    registration_order = 10
    node_type = ast.Call

    def project_ast(self, value: ast.Call) -> ast.AST | None:
        match = attribute_call_match(
            value,
            method_name=_APPEND_METHOD_NAME,
            owner_type=ast.Name,
            owner_name=_CANDIDATE_ACCUMULATOR_NAME,
            single_argument_required=True,
            argument_count=1,
            allow_keywords=False,
        )
        return None if match is None else match.single_argument


class _CandidateConstructorNameStep(
    _CandidateAppendConstructorNameStep,
    AstTypedEffectStep[ast.Call, str],
):
    step_id = "candidate_constructor_name"
    registration_order = 20
    node_type = ast.Call

    def project_ast(self, value: ast.Call) -> str | None:
        candidate_name = _call_name(value.func)
        if candidate_name is None or not candidate_name.endswith("Candidate"):
            return None
        return candidate_name


def _candidate_append_constructor_name(node: ast.AST) -> str | None:
    return cast(
        str | None,
        Maybe.of(node)
        .bind_all(registered_effect_steps(_CandidateAppendConstructorNameStep))
        .unwrap_or_none(),
    )


def _candidate_append_constructor_names(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        (
            candidate_name
            for child in ast.walk(node)
            if (candidate_name := _candidate_append_constructor_name(child)) is not None
        )
    )


def _is_candidate_accumulator_return_value(node: ast.AST | None) -> bool:
    call = as_ast(node, ast.Call)
    if call is None or not call.args:
        return False
    if _call_name(call.func) not in {"tuple", "sorted_tuple"}:
        return False
    return name_id(call.args[0]) == _CANDIDATE_ACCUMULATOR_NAME


def _returns_candidate_accumulator(node: ast.FunctionDef) -> bool:
    for statement in _trim_docstring_body(node.body):
        if not isinstance(statement, ast.Return):
            continue
        if _is_candidate_accumulator_return_value(statement.value):
            return True
    return False


def _named_function_collector_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[NamedFunctionCollectorBoilerplateCandidate, ...]:
    collector_candidates: list[NamedFunctionCollectorBoilerplateCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.FunctionDef) or node.end_lineno is None:
            continue
        if not node.name.endswith("_candidates"):
            continue
        if not any(_assigns_candidate_accumulator(child) for child in ast.walk(node)):
            continue
        if not _returns_candidate_accumulator(node):
            continue
        append_type_names = tuple(
            (
                candidate_name
                for child in ast.walk(node)
                if isinstance(child, ast.For) and _is_named_function_iteration(child)
                for candidate_name in _candidate_append_constructor_names(child)
            )
        )
        if not append_type_names:
            continue
        collector_candidates.append(
            NamedFunctionCollectorBoilerplateCandidate(
                file_path=str(module.path),
                line=node.lineno,
                function_name=node.name,
                candidate_type_names=sorted_tuple(set(append_type_names)),
                append_count=len(append_type_names),
                line_count=node.end_lineno - node.lineno + 1,
            )
        )
    return tuple(collector_candidates)


class _ListAccumulatorAssignmentStep(RegisteredEffectStep):
    pass


class _EmptyListAccumulatorValueStep(RegisteredEffectStep):
    pass


class _NamedValueBindingStep(
    _ListAccumulatorAssignmentStep,
    GuardedEffectStep[ast.stmt, NamedValueBinding],
):
    step_id = "named_value_binding"
    registration_order = 10

    def project(self, value: ast.stmt) -> NamedValueBinding | None:
        return named_value_binding(value)


class _LiteralEmptyListAccumulatorValueStep(
    _EmptyListAccumulatorValueStep,
    GuardedEffectStep[NamedValueBinding, str],
):
    step_id = "literal_empty_list_accumulator_value"
    registration_order = 10

    def accepts(self, value: NamedValueBinding) -> bool:
        return isinstance(value.value, ast.List)

    def project(self, value: NamedValueBinding) -> str | None:
        list_node = cast(ast.List, value.value)
        return value.name if not list_node.elts else None


class _ConstructorEmptyListAccumulatorValueStep(
    _EmptyListAccumulatorValueStep,
    GuardedEffectStep[NamedValueBinding, str],
):
    step_id = "constructor_empty_list_accumulator_value"
    registration_order = 20

    def accepts(self, value: NamedValueBinding) -> bool:
        return isinstance(value.value, ast.Call)

    def project(self, value: NamedValueBinding) -> str | None:
        call = cast(ast.Call, value.value)
        is_empty_list = (
            _call_name(call.func) == _BuiltinCollectionName.LIST
            and not call.args
            and not call.keywords
        )
        return value.name if is_empty_list else None


class _EmptyListValueBindingStep(
    _ListAccumulatorAssignmentStep,
    GuardedEffectStep[NamedValueBinding, str],
):
    step_id = "empty_list_value_binding"
    registration_order = 20

    def project(self, value: NamedValueBinding) -> str | None:
        return cast(
            str | None,
            Maybe.of(value)
            .bind(
                FirstSuccessfulEffectStep(
                    registered_effect_steps(_EmptyListAccumulatorValueStep)
                )
            )
            .unwrap_or_none(),
        )


def _list_accumulator_name_from_assignment(statement: ast.stmt) -> str | None:
    return cast(
        str | None,
        Maybe.of(statement)
        .bind_all(registered_effect_steps(_ListAccumulatorAssignmentStep))
        .unwrap_or_none(),
    )


def _local_list_accumulator_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    return frozenset(
        accumulator_name
        for statement in _trim_docstring_body(function.body)
        if (accumulator_name := _list_accumulator_name_from_assignment(statement))
        is not None
    )


def _returned_accumulator_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    returned_names: set[str] = set()
    for statement in _trim_docstring_body(function.body):
        if not isinstance(statement, ast.Return):
            continue
        call = as_ast(statement.value, ast.Call)
        if call is None or not call.args:
            continue
        if _call_name(call.func) not in {"tuple", "sorted_tuple"}:
            continue
        if (returned_name := name_id(call.args[0])) is not None:
            returned_names.add(returned_name)
    return frozenset(returned_names)


def _append_candidate_constructor_name(
    node: ast.AST, accumulator_names: frozenset[str]
) -> tuple[str, str] | None:
    call = as_ast(node, ast.Call)
    if call is None:
        return None
    match = attribute_call_match(
        call,
        method_name=_APPEND_METHOD_NAME,
        owner_type=ast.Name,
        single_argument_required=True,
        argument_count=1,
        allow_keywords=False,
    )
    if match is None or match.owner.id not in accumulator_names:
        return None
    appended = as_ast(match.single_argument, ast.Call)
    candidate_name = _call_name(appended.func) if appended is not None else None
    if candidate_name is None or not candidate_name.endswith("Candidate"):
        return None
    return match.owner.id, candidate_name


def _ast_stream_call_name(node: ast.AST) -> str | None:
    call = as_ast(node, ast.Call)
    if call is None:
        return None
    call_name = _call_name(call.func)
    return call_name if call_name in _AST_STREAM_CALL_NAMES else None


def _ast_stream_collector_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[AstStreamCollectorBoilerplateCandidate, ...]:
    stream_collectors: list[AstStreamCollectorBoilerplateCandidate] = []
    for function in module.module.body:
        if not isinstance(function, ast.FunctionDef) or function.end_lineno is None:
            continue
        accumulator_names = _local_list_accumulator_names(
            function
        ) & _returned_accumulator_names(function)
        if not accumulator_names:
            continue
        stream_calls: list[str] = []
        appended_pairs: list[tuple[str, str]] = []
        for node in ast.walk(function):
            if not isinstance(node, ast.For):
                continue
            stream_call_name = _ast_stream_call_name(node.iter)
            if stream_call_name is None:
                continue
            loop_pairs = tuple(
                pair
                for child in ast.walk(node)
                if (
                    pair := _append_candidate_constructor_name(child, accumulator_names)
                )
                is not None
            )
            if not loop_pairs:
                continue
            stream_calls.append(stream_call_name)
            appended_pairs.extend(loop_pairs)
        if not appended_pairs:
            continue
        accumulator_name = next(
            iter(sorted({name for name, _ in appended_pairs} & accumulator_names))
        )
        stream_collectors.append(
            AstStreamCollectorBoilerplateCandidate(
                file_path=str(module.path),
                line=function.lineno,
                function_name=function.name,
                accumulator_name=accumulator_name,
                stream_call_names=sorted_tuple(set(stream_calls)),
                candidate_type_names=sorted_tuple(
                    {candidate_name for _, candidate_name in appended_pairs}
                ),
                append_count=len(appended_pairs),
                line_count=function.end_lineno - function.lineno + 1,
            )
        )
    return tuple(stream_collectors)


def _candidate_detector_scope_kind(node: ast.ClassDef) -> str | None:
    if not any(
        (
            isinstance(statement, ast.Assign)
            and any((name_id(target) == "detector_id" for target in statement.targets))
            for statement in node.body
        )
    ):
        return None
    base_names = set(_class_base_names(node))
    if "CandidateFindingDetector" in base_names:
        return "module"
    if "CrossModuleCandidateDetector" in base_names:
        return "cross_module"
    return None


def _candidate_collector_method_call(
    method: ast.FunctionDef,
    scope_kind: str,
) -> tuple[str, bool] | None:
    body = tuple(
        (
            statement
            for statement in _trim_docstring_body(method.body)
            if not (
                isinstance(statement, ast.Delete)
                and any((name_id(target) == "config" for target in statement.targets))
            )
        )
    )
    returned_call = return_call(single_item(body)) if len(body) == 1 else None
    collector_name = _call_name(returned_call.func) if returned_call else None
    if collector_name is None:
        return None
    expected_first_arg = "modules" if scope_kind == "cross_module" else "module"
    arg_names = tuple(name_id(argument) for argument in returned_call.args)
    if arg_names == (expected_first_arg,):
        return (collector_name, False)
    if arg_names == (expected_first_arg, "config"):
        return (collector_name, True)
    return None


def _candidate_collector_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[CandidateCollectorBoilerplateCandidate, ...]:
    candidates: list[CandidateCollectorBoilerplateCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        scope_kind = _candidate_detector_scope_kind(node)
        if scope_kind is None:
            continue
        method = next(
            (
                statement
                for statement in node.body
                if isinstance(statement, ast.FunctionDef)
                and statement.name == _CANDIDATE_COLLECTOR_METHOD_NAME
            ),
            None,
        )
        if method is None:
            continue
        collector_call = _candidate_collector_method_call(method, scope_kind)
        if collector_call is None:
            continue
        collector_name, uses_config = collector_call
        candidates.append(
            CandidateCollectorBoilerplateCandidate(
                file_path=str(module.path),
                line=method.lineno,
                class_name=node.name,
                method_name=method.name,
                collector_name=collector_name,
                scope_kind=scope_kind,
                uses_config=uses_config,
                recommended_base_name=_CANDIDATE_COLLECTOR_BASE_BY_SHAPE[
                    scope_kind, uses_config
                ],
            )
        )
    return tuple(candidates)


_TYPED_CANDIDATE_DETECTOR_BASE_NAMES = frozenset(
    {
        "CandidateFindingDetector",
        "ModuleCollectorCandidateDetector",
        "ConfiguredModuleCollectorCandidateDetector",
        "CrossModuleCandidateDetector",
        "CrossModuleCollectorCandidateDetector",
        "ConfiguredCrossModuleCollectorCandidateDetector",
    }
)


def _concrete_detector_base_name(node: ast.ClassDef) -> str | None:
    if not any(
        (
            isinstance(statement, ast.Assign)
            and any((name_id(target) == "detector_id" for target in statement.targets))
            for statement in node.body
        )
    ):
        return None
    base_names = tuple(
        (
            base_name
            for base_name in _class_base_names(node)
            if base_name in _TYPED_CANDIDATE_DETECTOR_BASE_NAMES
        )
    )
    return single_item(base_names) if len(base_names) == 1 else None


def _single_payload_parameter_name(method: ast.FunctionDef) -> str | None:
    positional = (*method.args.posonlyargs, *method.args.args)
    payload_names = tuple(
        (argument.arg for argument in positional if argument.arg not in {"self", "cls"})
    )
    if method.args.vararg is not None or method.args.kwarg is not None:
        return None
    return single_item(payload_names) if len(payload_names) == 1 else None


def _parameter_name_is_reused(
    statements: Sequence[ast.stmt], parameter_name: str
) -> bool:
    return any(
        (
            isinstance(node, ast.Name) and node.id == parameter_name
            for statement in statements
            for node in ast.walk(statement)
        )
    )


def _first_named_call_assignment(
    statements: Sequence[ast.stmt],
) -> NamedCallAssignment | None:
    first_statement = single_item(statements[:1])
    assignment = as_ast(first_statement, ast.Assign)
    return named_call_assignment(assignment) if assignment is not None else None


def _call_is_cast_of_parameter(call: ast.Call, parameter_name: str) -> bool:
    return (
        _call_name(call.func) == "cast"
        and len(call.args) == 2
        and (name_id(call.args[1]) == parameter_name)
    )


def _typed_candidate_cast_assignment(
    method: ast.FunctionDef,
) -> tuple[str, str, str] | None:
    parameter_name = _single_payload_parameter_name(method)
    body = _trim_docstring_body(method.body)
    call_assignment = _first_named_call_assignment(body)
    if parameter_name is None or call_assignment is None:
        return None
    cast_call = call_assignment.call
    if not _call_is_cast_of_parameter(cast_call, parameter_name):
        return None
    if _parameter_name_is_reused(body[1:], parameter_name):
        return None
    return (parameter_name, call_assignment.target_name, ast.unparse(cast_call.args[0]))


def _typed_candidate_cast_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[TypedCandidateCastBoilerplateCandidate, ...]:
    candidates: list[TypedCandidateCastBoilerplateCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        detector_base_name = _concrete_detector_base_name(node)
        if detector_base_name is None:
            continue
        for statement in node.body:
            if not (
                isinstance(statement, ast.FunctionDef)
                and statement.name == "_finding_for_candidate"
            ):
                continue
            cast_assignment = _typed_candidate_cast_assignment(statement)
            if cast_assignment is None:
                continue
            parameter_name, local_name, candidate_type_name = cast_assignment
            candidates.append(
                TypedCandidateCastBoilerplateCandidate(
                    file_path=str(module.path),
                    line=statement.lineno,
                    class_name=node.name,
                    method_name=statement.name,
                    parameter_name=parameter_name,
                    local_name=local_name,
                    candidate_type_name=candidate_type_name,
                    detector_base_name=detector_base_name,
                )
            )
    return tuple(candidates)


_FINDING_SPEC_DEFAULTS_BY_CONSTRUCTOR = {
    "FindingSpec": ("MEDIUM_CONFIDENCE", "STRONG_HEURISTIC"),
    "HighConfidenceFindingSpec": ("HIGH_CONFIDENCE", "STRONG_HEURISTIC"),
    "CertifiedFindingSpec": ("MEDIUM_CONFIDENCE", "CERTIFIED"),
    "HighConfidenceCertifiedFindingSpec": ("HIGH_CONFIDENCE", "CERTIFIED"),
}
_FINDING_SPEC_CONSTRUCTOR_BY_DEFAULTS = {
    defaults: constructor
    for constructor, defaults in _FINDING_SPEC_DEFAULTS_BY_CONSTRUCTOR.items()
}
_FINDING_SPEC_SEMANTIC_KEYWORD_INDEX = {"confidence": 0, "certification": 1}


def _keyword_value_name(keyword: ast.keyword | None) -> str | None:
    return _call_name(keyword.value) if keyword is not None else None


def _recommended_finding_spec_constructor(
    constructor_name: str, semantic_keywords: dict[str, ast.keyword]
) -> str:
    defaults = _FINDING_SPEC_DEFAULTS_BY_CONSTRUCTOR[constructor_name]
    confidence_name = _keyword_value_name(semantic_keywords.get("confidence"))
    certification_name = _keyword_value_name(semantic_keywords.get("certification"))
    target_defaults = (
        confidence_name or defaults[0],
        certification_name or defaults[1],
    )
    return _FINDING_SPEC_CONSTRUCTOR_BY_DEFAULTS.get(target_defaults, constructor_name)


def _finding_spec_default_field_candidates(
    module: ParsedModule,
) -> tuple[FindingSpecDefaultFieldCandidate, ...]:
    candidates: list[FindingSpecDefaultFieldCandidate] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Call):
            continue
        constructor_name = _call_name(node.func)
        if constructor_name not in _FINDING_SPEC_DEFAULTS_BY_CONSTRUCTOR:
            continue
        semantic_keywords = {
            keyword.arg: keyword
            for keyword in node.keywords
            if keyword.arg in _FINDING_SPEC_SEMANTIC_KEYWORD_INDEX
        }
        if not semantic_keywords:
            continue
        recommended_constructor_name = _recommended_finding_spec_constructor(
            constructor_name, semantic_keywords
        )
        recommended_defaults = _FINDING_SPEC_DEFAULTS_BY_CONSTRUCTOR[
            recommended_constructor_name
        ]
        redundant_keywords = tuple(
            (
                (name, value_name)
                for name, keyword in semantic_keywords.items()
                for value_name in (_keyword_value_name(keyword),)
                if value_name
                == recommended_defaults[_FINDING_SPEC_SEMANTIC_KEYWORD_INDEX[name]]
            )
        )
        if not redundant_keywords:
            continue
        candidates.append(
            FindingSpecDefaultFieldCandidate(
                file_path=str(module.path),
                line=node.lineno,
                constructor_name=constructor_name,
                recommended_constructor_name=recommended_constructor_name,
                redundant_keyword_names=tuple((name for name, _ in redundant_keywords)),
                redundant_keyword_values=tuple(
                    (value for _, value in redundant_keywords)
                ),
            )
        )
    return tuple(candidates)


def _self_finding_spec_build_call(node: ast.AST) -> ast.Call | None:
    call = as_ast(node, ast.Call)
    if call is None or len(call.args) < 1:
        return None
    if not _is_self_finding_spec_build_func(call.func):
        return None
    if not _is_self_detector_id_attribute(call.args[0]):
        return None
    return call


def _is_self_finding_spec_build_func(node: ast.AST) -> bool:
    build_attr = as_ast(node, ast.Attribute)
    spec_attr = as_ast(build_attr.value if build_attr else None, ast.Attribute)
    return (
        build_attr is not None
        and build_attr.attr == "build"
        and (spec_attr is not None)
        and (spec_attr.attr == "finding_spec")
        and (name_id(spec_attr.value) == "self")
    )


def _is_self_detector_id_attribute(node: ast.AST) -> bool:
    detector_id_arg = as_ast(node, ast.Attribute)
    return (
        detector_id_arg is not None
        and detector_id_arg.attr == "detector_id"
        and (name_id(detector_id_arg.value) == "self")
    )


def _finding_spec_build_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[ClassMethodLineWitnessCandidate, ...]:
    candidates: list[ClassMethodLineWitnessCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            (
                isinstance(statement, ast.Assign)
                and any(
                    (name_id(target) == "detector_id" for target in statement.targets)
                )
                for statement in node.body
            )
        ):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            for child in ast.walk(statement):
                if _self_finding_spec_build_call(child) is None:
                    continue
                candidates.append(
                    ClassMethodLineWitnessCandidate(
                        file_path=str(module.path),
                        line=child.lineno,
                        class_name=node.name,
                        method_name=statement.name,
                    )
                )
    return tuple(candidates)


def _self_build_finding_call(node: ast.AST) -> ast.Call | None:
    call = return_call(node)
    if call is None:
        return None
    function = as_ast(call.func, ast.Attribute)
    if function is None or function.attr != "build_finding":
        return None
    return call if name_id(function.value) == "self" else None


def _direct_build_finding_renderer_candidates(
    module: ParsedModule,
) -> tuple[DirectBuildFindingRendererCandidate, ...]:
    candidates: list[DirectBuildFindingRendererCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_name = _concrete_detector_base_name(node)
        if base_name is None:
            continue
        for statement in node.body:
            if not (
                isinstance(statement, ast.FunctionDef)
                and statement.name == "_finding_for_candidate"
            ):
                continue
            body = _trim_docstring_body(statement.body)
            call = (
                _self_build_finding_call(single_item(body)) if len(body) == 1 else None
            )
            if call is None:
                continue
            candidates.append(
                DirectBuildFindingRendererCandidate(
                    file_path=str(module.path),
                    line=statement.lineno,
                    class_name=node.name,
                    method_name=statement.name,
                    base_name=base_name,
                    positional_arg_count=len(call.args),
                    keyword_names=tuple(
                        (keyword.arg for keyword in call.keywords if keyword.arg)
                    ),
                )
            )
    return tuple(candidates)


def _class_detector_id_assignment(node: ast.ClassDef) -> tuple[int, str] | None:
    for statement in node.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any((name_id(target) == "detector_id" for target in statement.targets)):
            continue
        if isinstance(statement.value, ast.Constant) and isinstance(
            statement.value.value, str
        ):
            return (statement.lineno, statement.value.value)
    return None


def _class_declares_finding_spec(node: ast.ClassDef) -> bool:
    return any(
        (
            isinstance(statement, ast.Assign)
            and any((name_id(target) == "finding_spec" for target in statement.targets))
            for statement in node.body
        )
    )


def _class_candidate_collector_assignment(node: ast.ClassDef) -> tuple[int, str] | None:
    for statement in node.body:
        targets: list[ast.AST]
        value: ast.AST | None
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        else:
            continue
        if not any((name_id(target) == "candidate_collector" for target in targets)):
            continue
        collector_name = name_id(value) if value is not None else None
        if collector_name is not None:
            return (statement.lineno, collector_name)
    return None


DerivableClassCandidateT = TypeVar("DerivableClassCandidateT")
ClassAssignmentReader = Callable[[ast.ClassDef], tuple[int, str] | None]
ClassExpectedValue = Callable[[str], str | None]


def _derivable_class_assignment_candidates(
    module: ParsedModule,
    assignment_reader: ClassAssignmentReader,
    expected_value: ClassExpectedValue,
    candidate_factory: Callable[[str, int, str, str], DerivableClassCandidateT],
) -> tuple[DerivableClassCandidateT, ...]:
    candidates: list[DerivableClassCandidateT] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        assignment = assignment_reader(node)
        if assignment is None:
            continue
        line, assigned_value = assignment
        if not _class_declares_finding_spec(node):
            continue
        if assigned_value != expected_value(node.name):
            continue
        candidates.append(
            candidate_factory(str(module.path), line, node.name, assigned_value)
        )
    return tuple(candidates)


def _derivable_detector_id_candidates(
    module: ParsedModule,
) -> tuple[DerivableDetectorIdCandidate, ...]:
    return _derivable_class_assignment_candidates(
        module,
        _class_detector_id_assignment,
        _detector_id_value_from_class_name,
        DerivableDetectorIdCandidate,
    )


def _derivable_candidate_collector_candidates(
    module: ParsedModule,
) -> tuple[DerivableCandidateCollectorCandidate, ...]:
    return _derivable_class_assignment_candidates(
        module,
        _class_candidate_collector_assignment,
        _candidate_collector_name_from_class_name,
        DerivableCandidateCollectorCandidate,
    )


_FINDING_SPEC_BUILDER_BY_CONSTRUCTOR = {
    "FindingSpec": "finding_spec_template",
    "HighConfidenceFindingSpec": "high_confidence_spec",
    "CertifiedFindingSpec": "certified_spec",
    "HighConfidenceCertifiedFindingSpec": "high_confidence_certified_spec",
}
_CANONICAL_FINDING_SPEC_FIELD_NAMES = (
    "pattern_id",
    "title",
    "why",
    "capability_gap",
    "relation_context",
    "capability_tags",
    "observation_tags",
)


def _canonical_finding_spec_builder_candidates(
    module: ParsedModule,
) -> tuple[CanonicalFindingSpecBuilderCandidate, ...]:
    candidates: list[CanonicalFindingSpecBuilderCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(
                (name_id(target) == "finding_spec" for target in statement.targets)
            ):
                continue
            call = as_ast(statement.value, ast.Call)
            if call is None or call.args:
                continue
            constructor_name = name_id(call.func)
            if constructor_name not in _FINDING_SPEC_BUILDER_BY_CONSTRUCTOR:
                continue
            keyword_names = tuple(
                keyword.arg for keyword in call.keywords if keyword.arg
            )
            if not set(_CANONICAL_FINDING_SPEC_FIELD_NAMES[:5]).issubset(keyword_names):
                continue
            candidates.append(
                CanonicalFindingSpecBuilderCandidate(
                    file_path=str(module.path),
                    line=statement.lineno,
                    class_name=node.name,
                    constructor_name=constructor_name,
                    builder_name=_FINDING_SPEC_BUILDER_BY_CONSTRUCTOR[constructor_name],
                    keyword_names=keyword_names,
                )
            )
    return tuple(candidates)


def _source_segment(module: ParsedModule, node: ast.AST) -> str:
    return ast.get_source_segment(module.source, node) or ast.unparse(node)


def _local_return_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.Return, ...]:
    returns: list[ast.Return] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is function:
                self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return None

        def visit_Return(self, node: ast.Return) -> None:
            returns.append(node)
            self.generic_visit(node)

    Visitor().visit(function)
    return tuple(returns)


def _sorted_call_in_tuple_return(node: ast.Return) -> ast.Call | None:
    return _sorted_call_in_tuple_expression(node.value)


def _sorted_call_in_tuple_expression(node: ast.AST | None) -> ast.Call | None:
    return cast(
        ast.Call | None,
        Maybe.of(node)
        .bind_all(registered_effect_steps(_SortedTupleReturnStep))
        .unwrap_or_none(),
    )


class _SortedTupleReturnStep(RegisteredEffectStep):
    pass


class _TupleCallPayloadStep(
    _SortedTupleReturnStep,
    GuardedEffectStep[ast.AST, ast.Call],
):
    step_id = "tuple_call_payload"
    registration_order = 10

    def accepts(self, value: ast.AST) -> bool:
        return isinstance(value, ast.Call) and name_id(value.func) == "tuple"

    def project(self, value: ast.AST) -> ast.Call | None:
        call = cast(ast.Call, value)
        if call.keywords:
            return None
        return as_ast(single_item(call.args), ast.Call)


class _SortedCallPayloadStep(
    _SortedTupleReturnStep,
    GuardedEffectStep[ast.Call, ast.Call],
):
    step_id = "sorted_call_payload"
    registration_order = 20

    def project(self, value: ast.Call) -> ast.Call | None:
        if name_id(value.func) != "sorted" or not value.args:
            return None
        return value


def _call_keyword_expression(
    module: ParsedModule, call: ast.Call, keyword_name: str
) -> str | None:
    keyword = next((item for item in call.keywords if item.arg == keyword_name), None)
    return None if keyword is None else _source_segment(module, keyword.value)


def _manual_sorted_tuple_return_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
) -> Iterable[ManualSortedTupleReturnCandidate]:
    for return_node in _local_return_nodes(function):
        sorted_call = _sorted_call_in_tuple_return(return_node)
        if sorted_call is None:
            continue
        yield ManualSortedTupleReturnCandidate(
            file_path=str(module.path),
            line=return_node.lineno,
            qualname=qualname,
            sorted_expression=_source_segment(module, sorted_call.args[0]),
            key_expression=_call_keyword_expression(module, sorted_call, "key"),
            reverse_expression=_call_keyword_expression(module, sorted_call, "reverse"),
            line_count=(return_node.end_lineno or return_node.lineno)
            - return_node.lineno
            + 1,
        )


def _manual_sorted_tuple_return_candidates(
    module: ParsedModule,
) -> tuple[ManualSortedTupleReturnCandidate, ...]:
    return _collect_named_function_candidates(
        module,
        _manual_sorted_tuple_return_candidates_for_function,
    )


def _manual_sorted_tuple_expression_candidates(
    module: ParsedModule,
) -> tuple[ManualSortedTupleExpressionCandidate, ...]:
    candidates: list[ManualSortedTupleExpressionCandidate] = []

    class Visitor(ClassFunctionStackNodeVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.parent_stack: list[ast.AST] = []

        def visit(self, node: ast.AST) -> None:
            self.parent_stack.append(node)
            super().visit(node)
            self.parent_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            sorted_call = _sorted_call_in_tuple_expression(node)
            parent = self.parent_stack[-2] if len(self.parent_stack) > 1 else None
            if sorted_call is not None and (not isinstance(parent, ast.Return)):
                candidates.append(
                    ManualSortedTupleExpressionCandidate(
                        file_path=str(module.path),
                        line=node.lineno,
                        qualname=self.qualname,
                        sorted_expression=_source_segment(module, sorted_call.args[0]),
                        key_expression=_call_keyword_expression(
                            module, sorted_call, "key"
                        ),
                        reverse_expression=_call_keyword_expression(
                            module, sorted_call, "reverse"
                        ),
                        line_count=(node.end_lineno or node.lineno) - node.lineno + 1,
                        context_kind=type(parent).__name__,
                    )
                )
            self.generic_visit(node)

    Visitor().visit(module.module)
    return tuple(candidates)


def _decorator_terminal_names(node: ast.FunctionDef) -> tuple[str, ...]:
    return tuple(
        (
            name
            for name in (
                _ast_terminal_name(decorator) for decorator in node.decorator_list
            )
            if name is not None
        )
    )


def _single_self_parameter_name(node: ast.FunctionDef) -> str | None:
    if (
        node.args.posonlyargs
        or node.args.vararg is not None
        or node.args.kwonlyargs
        or (node.args.kwarg is not None)
        or node.args.defaults
    ):
        return None
    parameter = single_item(node.args.args)
    return None if parameter is None else parameter.arg


# fmt: off
materialize_product_record(product_record_spec('_PropertyMethodReturn', 'method_name: str; returned: ast.AST'))
# fmt: on


class _SimplePropertyAliasPairStep(RegisteredEffectStep):
    pass


class _ConcretePropertyMethodStep(
    _SimplePropertyAliasPairStep,
    GuardedEffectStep[ast.FunctionDef, ast.FunctionDef],
):
    step_id = "concrete_property_method"
    registration_order = 10

    def accepts(self, value: ast.FunctionDef) -> bool:
        decorator_names = _decorator_terminal_names(value)
        return (
            "property" in decorator_names
            and "abstractmethod" not in decorator_names
            and (_single_self_parameter_name(value) == "self")
        )

    def project(self, value: ast.FunctionDef) -> ast.FunctionDef | None:
        return value


class _SinglePropertyReturnStep(
    _SimplePropertyAliasPairStep,
    GuardedEffectStep[ast.FunctionDef, _PropertyMethodReturn],
):
    step_id = "single_property_return"
    registration_order = 20

    def project(self, value: ast.FunctionDef) -> _PropertyMethodReturn | None:
        returned = single_return_value(_trim_docstring_body(value.body))
        if returned is None:
            return None
        return _PropertyMethodReturn(value.name, returned)


class _SelfAttributeReturnStep(
    _SimplePropertyAliasPairStep,
    GuardedEffectStep[_PropertyMethodReturn, tuple[str, str]],
):
    step_id = "self_attribute_return"
    registration_order = 30

    def project(self, value: _PropertyMethodReturn) -> tuple[str, str] | None:
        returned = as_ast(value.returned, ast.Attribute)
        if returned is None or name_id(returned.value) != "self":
            return None
        return value.method_name, returned.attr


def _simple_property_alias_pair(node: ast.FunctionDef) -> tuple[str, str] | None:
    return cast(
        tuple[str, str] | None,
        Maybe.of(node)
        .bind_all(registered_effect_steps(_SimplePropertyAliasPairStep))
        .unwrap_or_none(),
    )


def _simple_property_alias_class_shape(
    node: ast.ClassDef,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]] | None:
    alias_pairs: list[tuple[str, str]] = []
    declared_field_names: list[str] = []
    for statement in _trim_docstring_body(node.body):
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            declared_field_names.append(statement.target.id)
            continue
        if isinstance(statement, ast.FunctionDef):
            alias_pair = _simple_property_alias_pair(statement)
            if alias_pair is None:
                return None
            alias_pairs.append(alias_pair)
            continue
        return None
    if not alias_pairs:
        return None
    return tuple(alias_pairs), tuple(declared_field_names)


ClassShapeT = TypeVar("ClassShapeT")
BuiltCandidateT = TypeVar("BuiltCandidateT")
ClassShapeProjector = Callable[[ast.ClassDef], ClassShapeT | None]
ClassShapeCandidateFactory = Callable[
    [ParsedModule, ast.ClassDef, ClassShapeT], BuiltCandidateT
]


def _class_shape_candidates(
    module: ParsedModule,
    shape_projector: ClassShapeProjector[ClassShapeT],
    candidate_factory: ClassShapeCandidateFactory[ClassShapeT, BuiltCandidateT],
) -> tuple[BuiltCandidateT, ...]:
    candidates: list[BuiltCandidateT] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        shape = shape_projector(node)
        if shape is None:
            continue
        candidates.append(candidate_factory(module, node, shape))
    return tuple(candidates)


def _simple_property_alias_class_candidate(
    module: ParsedModule,
    node: ast.ClassDef,
    shape: tuple[tuple[tuple[str, str], ...], tuple[str, ...]],
) -> SimplePropertyAliasClassCandidate:
    alias_pairs, declared_field_names = shape
    return SimplePropertyAliasClassCandidate(
        file_path=str(module.path),
        line=node.lineno,
        class_name=node.name,
        alias_pairs=alias_pairs,
        declared_field_names=declared_field_names,
        line_count=(node.end_lineno or node.lineno) - node.lineno + 1,
    )


def _simple_property_alias_class_candidates(
    module: ParsedModule,
) -> tuple[SimplePropertyAliasClassCandidate, ...]:
    return _class_shape_candidates(
        module,
        _simple_property_alias_class_shape,
        _simple_property_alias_class_candidate,
    )


def _simple_property_alias_method_candidates(
    module: ParsedModule,
) -> tuple[SimplePropertyAliasMethodCandidate, ...]:
    candidates: list[SimplePropertyAliasMethodCandidate] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        @property
        def class_name(self) -> str:
            return ".".join(self.class_stack)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            for statement in _trim_docstring_body(node.body):
                if not isinstance(statement, ast.FunctionDef):
                    continue
                alias_pair = _simple_property_alias_pair(statement)
                if alias_pair is None:
                    continue
                method_name, source_name = alias_pair
                candidates.append(
                    SimplePropertyAliasMethodCandidate(
                        file_path=str(module.path),
                        line=statement.lineno,
                        class_name=self.class_name,
                        method_name=method_name,
                        source_name=source_name,
                        return_annotation=(
                            ast.unparse(statement.returns)
                            if statement.returns is not None
                            else None
                        ),
                    )
                )
            self.generic_visit(node)
            self.class_stack.pop()

    Visitor().visit(module.module)
    return tuple(candidates)


def _self_attribute_name(node: ast.AST) -> str | None:
    return (
        node.attr
        if isinstance(node, ast.Attribute) and name_id(node.value) == "self"
        else None
    )


class _SourceLocationEvidenceShapeStep(RegisteredEffectStep):
    pass


class _EvidencePropertyReturnStep(
    _SourceLocationEvidenceShapeStep,
    GuardedEffectStep[ast.FunctionDef | ast.AsyncFunctionDef, ast.Return],
):
    step_id = "evidence_property_return"
    registration_order = 10

    def accepts(self, value: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        return value.name == "evidence" and any(
            (
                _ast_terminal_name(decorator) == "property"
                for decorator in value.decorator_list
            )
        )

    def project(
        self, value: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.Return | None:
        return as_ast(single_item(_trim_docstring_body(value.body)), ast.Return)


class _SourceLocationReturnCallStep(
    _SourceLocationEvidenceShapeStep,
    GuardedEffectStep[ast.Return, ast.Call],
):
    step_id = "source_location_return_call"
    registration_order = 20

    def project(self, value: ast.Return) -> ast.Call | None:
        call = as_ast(value.value, ast.Call)
        if call is None or name_id(call.func) != "SourceLocation":
            return None
        return call


class _SourceLocationSelfAttributeArgsStep(
    _SourceLocationEvidenceShapeStep,
    GuardedEffectStep[ast.Call, tuple[str, str, str]],
):
    step_id = "source_location_self_attribute_args"
    registration_order = 30

    def project(self, value: ast.Call) -> tuple[str, str, str] | None:
        if value.keywords:
            return None
        attributes = ast_sequence(
            value.args, ast.Attribute, ast.Attribute, ast.Attribute
        )
        if attributes is None:
            return None
        attribute_names = tuple((_self_attribute_name(arg) for arg in attributes))
        return (
            cast(tuple[str, str, str], attribute_names)
            if all((name is not None for name in attribute_names))
            else None
        )


def _source_location_evidence_shape(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str] | None:
    return cast(
        tuple[str, str, str] | None,
        Maybe.of(node)
        .bind_all(registered_effect_steps(_SourceLocationEvidenceShapeStep))
        .unwrap_or_none(),
    )


def _source_location_evidence_property_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[SourceLocationEvidencePropertyCandidate]:
    for statement in _trim_docstring_body(node.body):
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        shape = _source_location_evidence_shape(statement)
        if shape is None:
            continue
        file_attribute_name, line_attribute_name, symbol_attribute_name = shape
        yield SourceLocationEvidencePropertyCandidate(
            file_path=str(module.path),
            line=statement.lineno,
            class_name=node.name,
            method_name=statement.name,
            file_attribute_name=file_attribute_name,
            line_attribute_name=line_attribute_name,
            symbol_attribute_name=symbol_attribute_name,
        )


def _source_location_evidence_property_candidates(
    module: ParsedModule,
) -> tuple[SourceLocationEvidencePropertyCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _source_location_evidence_property_candidates_for_class,
    )


# fmt: off
materialize_product_records((
    product_record_spec('_ZippedSourceLocationGeneratorCall', 'source_location_call: ast.Call; generator: ast.GeneratorExp'),
    product_record_spec('_ZippedSourceLocationVariableArgs', 'file_attribute_name: str; line_variable_name: str; symbol_variable_name: str; generator: ast.GeneratorExp'),
))
# fmt: on


class _ZippedSourceLocationEvidenceShapeStep(RegisteredEffectStep):
    pass


class _ZippedEvidencePropertyReturnStep(
    _ZippedSourceLocationEvidenceShapeStep,
    GuardedEffectStep[ast.FunctionDef | ast.AsyncFunctionDef, ast.Return],
):
    step_id = "zipped_evidence_property_return"
    registration_order = 10

    def accepts(self, value: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        return value.name in {"evidence", "evidence_locations"} and any(
            (
                _ast_terminal_name(decorator) == "property"
                for decorator in value.decorator_list
            )
        )

    def project(
        self, value: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.Return | None:
        return as_ast(single_item(_trim_docstring_body(value.body)), ast.Return)


class _ZippedTupleGeneratorReturnStep(
    _ZippedSourceLocationEvidenceShapeStep,
    GuardedEffectStep[ast.Return, ast.GeneratorExp],
):
    step_id = "zipped_tuple_generator_return"
    registration_order = 20

    def project(self, value: ast.Return) -> ast.GeneratorExp | None:
        outer_call = as_ast(value.value, ast.Call)
        return (
            as_ast(outer_call.args[0], ast.GeneratorExp)
            if outer_call is not None
            and name_id(outer_call.func) == "tuple"
            and len(outer_call.args) == 1
            and not outer_call.keywords
            else None
        )


class _ZippedSourceLocationGeneratorCallStep(
    _ZippedSourceLocationEvidenceShapeStep,
    GuardedEffectStep[ast.GeneratorExp, _ZippedSourceLocationGeneratorCall],
):
    step_id = "zipped_source_location_generator_call"
    registration_order = 30

    def project(
        self, value: ast.GeneratorExp
    ) -> _ZippedSourceLocationGeneratorCall | None:
        source_location_call = as_ast(value.elt, ast.Call)
        return (
            _ZippedSourceLocationGeneratorCall(source_location_call, value)
            if source_location_call is not None
            and name_id(source_location_call.func) == "SourceLocation"
            and len(source_location_call.args) == 3
            and not source_location_call.keywords
            else None
        )


class _ZippedSourceLocationCallArgsStep(
    _ZippedSourceLocationEvidenceShapeStep,
    GuardedEffectStep[
        _ZippedSourceLocationGeneratorCall, _ZippedSourceLocationVariableArgs
    ],
):
    step_id = "zipped_source_location_call_args"
    registration_order = 40

    def project(
        self, value: _ZippedSourceLocationGeneratorCall
    ) -> _ZippedSourceLocationVariableArgs | None:
        file_attribute_name = _self_attribute_name(value.source_location_call.args[0])
        line_variable_name = name_id(value.source_location_call.args[1])
        symbol_variable_name = name_id(value.source_location_call.args[2])
        return (
            _ZippedSourceLocationVariableArgs(
                file_attribute_name,
                line_variable_name,
                symbol_variable_name,
                value.generator,
            )
            if file_attribute_name is not None
            and line_variable_name is not None
            and symbol_variable_name is not None
            else None
        )


def _zipped_source_location_self_attribute_shape(
    value: _ZippedSourceLocationVariableArgs,
) -> tuple[str, str, str] | None:
    comprehension = single_item(tuple(value.generator.generators))
    if (
        comprehension is None
        or comprehension.ifs
        or comprehension.is_async
        or not isinstance(comprehension.target, ast.Tuple)
    ):
        return None
    targets = ast_sequence(comprehension.target.elts, ast.Name, ast.Name)
    zip_call = as_ast(comprehension.iter, ast.Call)
    zipped_attributes = (
        ast_sequence(zip_call.args, ast.Attribute, ast.Attribute)
        if zip_call is not None and name_id(zip_call.func) == "zip"
        else None
    )
    attribute_names = (
        tuple(_self_attribute_name(attribute) for attribute in zipped_attributes)
        if zipped_attributes is not None
        else None
    )
    bindings = (
        {
            target.id: attribute_name
            for target, attribute_name in zip(targets, attribute_names, strict=True)
        }
        if targets is not None
        and attribute_names is not None
        and all((name is not None for name in attribute_names))
        else {}
    )
    line_numbers_attribute_name = bindings.get(value.line_variable_name)
    symbol_names_attribute_name = bindings.get(value.symbol_variable_name)
    return (
        (
            value.file_attribute_name,
            line_numbers_attribute_name,
            symbol_names_attribute_name,
        )
        if line_numbers_attribute_name is not None
        and symbol_names_attribute_name is not None
        else None
    )


class _ZippedSelfAttributeBindingsStep(
    _ZippedSourceLocationEvidenceShapeStep,
    GuardedEffectStep[_ZippedSourceLocationVariableArgs, tuple[str, str, str]],
):
    step_id = "zipped_self_attribute_bindings"
    registration_order = 50

    def project(
        self, value: _ZippedSourceLocationVariableArgs
    ) -> tuple[str, str, str] | None:
        return _zipped_source_location_self_attribute_shape(value)


def _zipped_source_location_evidence_shape(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str] | None:
    return cast(
        tuple[str, str, str] | None,
        Maybe.of(node)
        .bind_all(registered_effect_steps(_ZippedSourceLocationEvidenceShapeStep))
        .unwrap_or_none(),
    )


def _zipped_source_location_evidence_property_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[ZippedSourceLocationEvidencePropertyCandidate]:
    for statement in _trim_docstring_body(node.body):
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        shape = _zipped_source_location_evidence_shape(statement)
        if shape is None or statement.end_lineno is None:
            continue
        (
            file_attribute_name,
            line_numbers_attribute_name,
            symbol_names_attribute_name,
        ) = shape
        yield ZippedSourceLocationEvidencePropertyCandidate(
            file_path=str(module.path),
            line=statement.lineno,
            class_name=node.name,
            method_name=statement.name,
            file_attribute_name=file_attribute_name,
            line_numbers_attribute_name=line_numbers_attribute_name,
            symbol_names_attribute_name=symbol_names_attribute_name,
            line_count=statement.end_lineno - statement.lineno + 1,
        )


def _zipped_source_location_evidence_property_candidates(
    module: ParsedModule,
) -> tuple[ZippedSourceLocationEvidencePropertyCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _zipped_source_location_evidence_property_candidates_for_class,
    )


def _top_level_helper_definitions(
    module: ParsedModule,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            (node.name, node.lineno)
            for node in module.module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
    )


def _private_helper_shadow_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[PrivateHelperShadowCandidate, ...]:
    public_definitions: dict[str, list[tuple[ParsedModule, int]]] = defaultdict(list)
    private_definitions: list[tuple[ParsedModule, str, int]] = []
    for module in modules:
        for name, line in _top_level_helper_definitions(module):
            if name.startswith("__"):
                continue
            if name.startswith("_"):
                private_definitions.append((module, name, line))
            else:
                public_definitions[name].append((module, line))

    candidates: list[PrivateHelperShadowCandidate] = []
    for private_module, private_name, private_line in private_definitions:
        public_name = private_name.lstrip("_")
        public_sites = tuple(
            (
                (module, line)
                for module, line in public_definitions.get(public_name, ())
                if module.path != private_module.path
            )
        )
        public_site = single_item(public_sites)
        if public_site is None:
            continue
        public_module, public_line = public_site
        candidates.append(
            PrivateHelperShadowCandidate(
                file_path=str(private_module.path),
                line=private_line,
                private_name=private_name,
                public_name=public_name,
                public_file_path=str(public_module.path),
                public_line=public_line,
                evidence_locations=(
                    SourceLocation(
                        str(private_module.path), private_line, private_name
                    ),
                    SourceLocation(str(public_module.path), public_line, public_name),
                ),
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.private_name,
        ),
    )


def _is_frozen_dataclass_decorator(node: ast.AST) -> bool:
    call = as_ast(node, ast.Call)
    if call is None or name_id(call.func) != "dataclass":
        return False
    return any(
        (
            keyword.arg == "frozen"
            and isinstance(keyword.value, ast.Constant)
            and (keyword.value.value is True)
            for keyword in call.keywords
        )
    )


def _dataclass_keyword_bool(node: ast.ClassDef, keyword_name: str) -> bool:
    for decorator in node.decorator_list:
        call = as_ast(decorator, ast.Call)
        if call is None or name_id(call.func) != "dataclass":
            continue
        for keyword in call.keywords:
            if (
                keyword.arg == keyword_name
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, bool)
            ):
                return keyword.value.value
    return False


class _FieldOnlyFrozenDataclassShapeStep(RegisteredEffectStep):
    pass


class _FrozenDataclassClassStep(
    _FieldOnlyFrozenDataclassShapeStep,
    GuardedEffectStep[ast.ClassDef, ast.ClassDef],
):
    step_id = "frozen_dataclass_class"
    registration_order = 10

    def accepts(self, value: ast.ClassDef) -> bool:
        return any(
            (
                _is_frozen_dataclass_decorator(decorator)
                for decorator in value.decorator_list
            )
        )

    def project(self, value: ast.ClassDef) -> ast.ClassDef | None:
        return value


def _ann_assign_product_field_spec(
    statement: ast.stmt,
) -> tuple[str, str, str | None] | None:
    assignment = as_ast(statement, ast.AnnAssign)
    if assignment is None:
        return None
    target = as_ast(assignment.target, ast.Name)
    if target is None:
        return None
    return (
        target.id,
        ast.unparse(assignment.annotation),
        ast.unparse(assignment.value) if assignment.value is not None else None,
    )


class _ProductRecordAnnotatedFieldsStep(
    _FieldOnlyFrozenDataclassShapeStep,
    GuardedEffectStep[
        (ast.ClassDef, tuple[ast.ClassDef, tuple[tuple[str, str, str | None], ...]])
    ],
):
    step_id = "product_record_annotated_fields"
    registration_order = 20

    def project(
        self,
        value: ast.ClassDef,
    ) -> tuple[ast.ClassDef, tuple[tuple[str, str, str | None], ...]] | None:
        field_specs: list[tuple[str, str, str | None]] = []
        for statement in _trim_docstring_body(value.body):
            if isinstance(statement, ast.Pass):
                continue
            field_spec = _ann_assign_product_field_spec(statement)
            if field_spec is None:
                return None
            field_specs.append(field_spec)
        if not field_specs:
            return None
        return value, tuple(field_specs)


class _ProductRecordShapeStep(
    _FieldOnlyFrozenDataclassShapeStep,
    GuardedEffectStep[
        (
            tuple[ast.ClassDef, tuple[tuple[str, str, str | None], ...]],
            tuple[
                tuple[str, ...],
                tuple[tuple[str, str], ...],
                tuple[tuple[str, str], ...],
                str | None,
                bool,
            ],
        )
    ],
):
    step_id = "product_record_shape"
    registration_order = 30

    def project(
        self, value: tuple[ast.ClassDef, tuple[tuple[str, str, str | None], ...]]
    ) -> (
        tuple[
            tuple[str, ...],
            tuple[tuple[str, str], ...],
            tuple[tuple[str, str], ...],
            str | None,
            bool,
        ]
        | None
    ):
        node, product_fields = value
        return (
            tuple((ast.unparse(base) for base in node.bases)),
            tuple(((name, annotation) for name, annotation, _ in product_fields)),
            tuple(
                (
                    (name, default)
                    for name, _, default in product_fields
                    if default is not None
                )
            ),
            ast.get_docstring(node),
            _dataclass_keyword_bool(node, "kw_only"),
        )


def _field_only_frozen_dataclass_shape(
    node: ast.ClassDef,
) -> (
    tuple[
        tuple[str, ...],
        tuple[tuple[str, str], ...],
        tuple[tuple[str, str], ...],
        str | None,
        bool,
    ]
    | None
):
    return cast(
        tuple[
            tuple[str, ...],
            tuple[tuple[str, str], ...],
            tuple[tuple[str, str], ...],
            str | None,
            bool,
        ]
        | None,
        Maybe.of(node)
        .bind_all(registered_effect_steps(_FieldOnlyFrozenDataclassShapeStep))
        .unwrap_or_none(),
    )


def _field_only_frozen_dataclass_candidate(
    module: ParsedModule,
    node: ast.ClassDef,
    shape: tuple[
        tuple[str, ...],
        tuple[tuple[str, str], ...],
        tuple[tuple[str, str], ...],
        str | None,
        bool,
    ],
) -> FieldOnlyFrozenDataclassCandidate:
    base_names, field_specs, default_specs, docstring, kw_only = shape
    return FieldOnlyFrozenDataclassCandidate(
        file_path=str(module.path),
        line=node.lineno,
        class_name=node.name,
        base_names=base_names,
        field_specs=field_specs,
        default_specs=default_specs,
        docstring=docstring,
        kw_only=kw_only,
        line_count=(node.end_lineno or node.lineno) - node.lineno + 1,
    )


def _field_only_frozen_dataclass_candidates(
    module: ParsedModule,
) -> tuple[FieldOnlyFrozenDataclassCandidate, ...]:
    return _class_shape_candidates(
        module,
        _field_only_frozen_dataclass_shape,
        _field_only_frozen_dataclass_candidate,
    )


_STRUCTURAL_ALIAS_ROOTS = frozenset(
    {
        "Callable",
        "Mapping",
        "Sequence",
        "dict",
        "frozenset",
        "list",
        "set",
        "tuple",
    }
)
_STRUCTURAL_ALIAS_LEAF_NAMES = frozenset(
    {
        "Any",
        "Callable",
        "ClassVar",
        "Final",
        "Generic",
        "Literal",
        "Mapping",
        "NamedTuple",
        "Optional",
        "Protocol",
        "Sequence",
        "Self",
        "TypeAlias",
        "TypeVar",
        "Union",
        "dict",
        "frozenset",
        "list",
        "set",
        "tuple",
    }
)


def _annotation_root_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_root_name(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_root_name(node.left) or _annotation_root_name(node.right)
    return None


def _structural_annotation_complexity(node: ast.AST) -> int:
    if isinstance(node, ast.Subscript):
        return (
            1
            + _structural_annotation_complexity(node.value)
            + _structural_annotation_complexity(node.slice)
        )
    if isinstance(node, ast.Tuple):
        return sum(_structural_annotation_complexity(element) for element in node.elts)
    if isinstance(node, ast.List):
        return sum(_structural_annotation_complexity(element) for element in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return (
            1
            + _structural_annotation_complexity(node.left)
            + _structural_annotation_complexity(node.right)
        )
    return 0


def _domain_annotation_names(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        name
        for name in _annotation_type_names(node)
        if name not in _STRUCTURAL_ALIAS_LEAF_NAMES
    )


def _semantic_alias_name(annotation: ast.AST) -> str:
    domain_names = _domain_annotation_names(annotation)
    if len(domain_names) >= 2:
        return "_" + "".join(domain_names[:2])
    if len(domain_names) == 1:
        return f"_{domain_names[0]}Shape"
    root_name = _annotation_root_name(annotation) or "Semantic"
    return f"_{root_name.removesuffix('s').title()}Shape"


def _annotation_is_alias_worthy(annotation: ast.AST) -> bool:
    root_name = _annotation_root_name(annotation)
    if root_name not in _STRUCTURAL_ALIAS_ROOTS:
        return False
    complexity = _structural_annotation_complexity(annotation)
    if complexity < 2:
        return False
    annotation_text = ast.unparse(annotation)
    return len(annotation_text) >= 35 or complexity >= 4


def _semantic_type_alias_candidates(
    module: ParsedModule,
) -> tuple[SemanticTypeAliasCandidate, ...]:
    annotations_by_shape: dict[
        str,
        list[tuple[str, str, ast.AST, SourceLocation]],
    ] = {}

    class Visitor(ClassFunctionStackNodeVisitor):
        traverse_class_body = ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
        traverse_function_body = (
            ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
        )

        def _owner_symbol(
            self, fallback: str, function_name: str | None = None
        ) -> str:
            function_suffix = () if function_name is None else (function_name,)
            parts = (*self.class_stack, *self.function_stack, *function_suffix)
            return ".".join(parts) if parts else fallback

        def _record(self, annotation: ast.AST | None, line: int, symbol: str) -> None:
            if annotation is None or not _annotation_is_alias_worthy(annotation):
                return
            fingerprint = ast.dump(annotation, include_attributes=False)
            annotations_by_shape.setdefault(fingerprint, []).append(
                (
                    ast.unparse(annotation),
                    symbol,
                    annotation,
                    SourceLocation(str(module.path), line, symbol),
                )
            )

        def before_visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            arguments = (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
            for argument in arguments:
                self._record(
                    argument.annotation,
                    argument.lineno,
                    f"{self._owner_symbol(node.name, node.name)}.{argument.arg}",
                )
            if node.args.vararg is not None:
                self._record(
                    node.args.vararg.annotation,
                    node.args.vararg.lineno,
                    (
                        f"{self._owner_symbol(node.name, node.name)}."
                        f"*{node.args.vararg.arg}"
                    ),
                )
            if node.args.kwarg is not None:
                self._record(
                    node.args.kwarg.annotation,
                    node.args.kwarg.lineno,
                    (
                        f"{self._owner_symbol(node.name, node.name)}."
                        f"**{node.args.kwarg.arg}"
                    ),
                )
            self._record(
                node.returns,
                node.lineno,
                f"{self._owner_symbol(node.name, node.name)} return",
            )

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            target = ast.unparse(node.target)
            self._record(
                node.annotation,
                node.lineno,
                f"{self._owner_symbol(target)}.{target}",
            )
            self.generic_visit(node)

    Visitor().visit(module.module)
    candidates: list[SemanticTypeAliasCandidate] = []
    for occurrences in annotations_by_shape.values():
        if len(occurrences) < 2:
            continue
        annotation_text, _, annotation, first_evidence = occurrences[0]
        candidates.append(
            SemanticTypeAliasCandidate(
                file_path=str(module.path),
                line=first_evidence.line,
                evidence_locations=tuple(
                    location for _, _, _, location in occurrences[:8]
                ),
                annotation_text=annotation_text,
                occurrence_count=len(occurrences),
                owner_symbols=tuple(symbol for _, symbol, _, _ in occurrences[:8]),
                suggested_alias_name=_semantic_alias_name(annotation),
            )
        )
    return tuple(candidates)


def _method_body_fingerprint(method: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    body = _trim_docstring_body(method.body)
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


_NODE_VISITOR_BASE_NAME = "NodeVisitor"
_VISITOR_METHOD_PREFIX = "visit_"
_VISITOR_STACK_SUFFIX = "_stack"


def _iter_scoped_class_defs(
    statements: Sequence[ast.stmt], scope: tuple[str, ...] = ()
) -> tuple[tuple[str, ast.ClassDef], ...]:
    class_defs: list[tuple[str, ast.ClassDef]] = []
    for statement in statements:
        if isinstance(statement, ast.ClassDef):
            qualname = ".".join((*scope, statement.name))
            class_defs.append((qualname, statement))
            class_defs.extend(
                _iter_scoped_class_defs(statement.body, (*scope, statement.name))
            )
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            class_defs.extend(
                _iter_scoped_class_defs(statement.body, (*scope, statement.name))
            )
        else:
            class_defs.extend(
                _iter_scoped_class_defs(
                    tuple(
                        (
                            child
                            for child in ast.iter_child_nodes(statement)
                            if isinstance(child, ast.stmt)
                        )
                    ),
                    scope,
                )
            )
    return tuple(class_defs)


def _inherits_node_visitor(node: ast.ClassDef) -> bool:
    return any(
        (
            ((base_chain := _ast_attribute_chain(base)) is not None)
            and base_chain[-1] == _NODE_VISITOR_BASE_NAME
            for base in node.bases
        )
    )


def _self_attribute_name_from_target(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def _is_empty_list_value(node: ast.AST | None) -> bool:
    return isinstance(node, ast.List) and len(node.elts) == 0


def _assigned_self_stack_names(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> tuple[str, ...]:
    if method is None:
        return ()
    stack_names: set[str] = set()
    for statement in _trim_docstring_body(method.body):
        if isinstance(statement, ast.AnnAssign) and _is_empty_list_value(
            statement.value
        ):
            target_name = _self_attribute_name_from_target(statement.target)
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and _is_empty_list_value(statement.value)
        ):
            target_name = _self_attribute_name_from_target(statement.targets[0])
        else:
            target_name = None
        if target_name is not None and target_name.endswith(_VISITOR_STACK_SUFFIX):
            stack_names.add(target_name)
    return sorted_tuple(stack_names)


def _self_stack_call_name(call: ast.Call, method_name: str) -> str | None:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != method_name:
        return None
    stack_expr = call.func.value
    if not isinstance(stack_expr, ast.Attribute):
        return None
    if not isinstance(stack_expr.value, ast.Name) or stack_expr.value.id != "self":
        return None
    return stack_expr.attr


def _is_node_name_argument(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "name"
        and isinstance(node.value, ast.Name)
        and node.value.id == "node"
    )


def _visitor_stack_transition_names(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    stack_names: frozenset[str],
) -> tuple[str, ...]:
    appended: set[str] = set()
    popped: set[str] = set()
    for statement in _trim_docstring_body(method.body):
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        call = statement.value
        append_stack = _self_stack_call_name(call, _APPEND_METHOD_NAME)
        if (
            append_stack in stack_names
            and len(call.args) == 1
            and _is_node_name_argument(call.args[0])
        ):
            appended.add(append_stack)
            continue
        pop_stack = _self_stack_call_name(call, "pop")
        if pop_stack in stack_names and not call.args and not call.keywords:
            popped.add(pop_stack)
    return sorted_tuple(appended & popped)


def _node_visitor_stack_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[NodeVisitorStackBoilerplateCandidate, ...]:
    candidates: list[NodeVisitorStackBoilerplateCandidate] = []
    for qualname, node in _iter_scoped_class_defs(module.module.body):
        if (not _inherits_node_visitor(node)) or _is_abstract_class(node):
            continue
        stack_names = _assigned_self_stack_names(_class_method_named(node, "__init__"))
        if not stack_names:
            continue
        transitions_by_method = {
            method.name: _visitor_stack_transition_names(method, frozenset(stack_names))
            for method in _iter_class_methods(node)
            if method.name.startswith(_VISITOR_METHOD_PREFIX)
        }
        transition_stack_names = sorted_tuple(
            {
                stack_name
                for stack_names_for_method in transitions_by_method.values()
                for stack_name in stack_names_for_method
            }
        )
        transition_method_names = sorted_tuple(
            (
                method_name
                for method_name, method_stack_names in transitions_by_method.items()
                if method_stack_names
            )
        )
        if len(transition_stack_names) < 2:
            continue
        candidates.append(
            NodeVisitorStackBoilerplateCandidate(
                file_path=str(module.path),
                line=node.lineno,
                qualname=qualname,
                stack_names=transition_stack_names,
                transition_method_names=transition_method_names,
                line_count=(node.end_lineno or node.lineno) - node.lineno + 1,
            )
        )
    return tuple(candidates)


def _duplicate_visitor_method_body_candidates(
    module: ParsedModule,
) -> tuple[DuplicateVisitorMethodBodyCandidate, ...]:
    candidates: list[DuplicateVisitorMethodBodyCandidate] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            groups: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = (
                defaultdict(list)
            )
            for statement in node.body:
                if isinstance(
                    statement, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and statement.name.startswith("visit_"):
                    groups[_method_body_fingerprint(statement)].append(statement)
            for methods in groups.values():
                if len(methods) < 2:
                    continue
                candidates.append(
                    DuplicateVisitorMethodBodyCandidate(
                        file_path=str(module.path),
                        line=methods[0].lineno,
                        class_name=".".join(self.class_stack),
                        method_names=tuple((method.name for method in methods)),
                        statement_count=len(_trim_docstring_body(methods[0].body)),
                    )
                )
            self.generic_visit(node)
            self.class_stack.pop()

    Visitor().visit(module.module)
    return tuple(candidates)


def _enum_metadata_table_cases(module: ast.Module) -> dict[str, tuple[str, int]]:
    tables: dict[str, tuple[str, int]] = {}
    for statement in module.body:
        binding = named_value_binding(statement)
        value = as_ast(None if binding is None else binding.value, ast.Dict)
        if binding is None or value is None:
            continue
        enum_key_names = tuple(
            (
                key.value.id
                for key in value.keys
                if isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name)
            )
        )
        enum_names = set(enum_key_names)
        if len(enum_key_names) == len(value.keys) and len(enum_names) == 1:
            tables[binding.name] = (next(iter(enum_names)), len(value.keys))
    return tables


def _enum_metadata_property_table(method: ast.FunctionDef) -> str | None:
    returned = single_return_value(_trim_docstring_body(method.body))
    lookup_source = returned.value if isinstance(returned, ast.Attribute) else returned
    lookup = as_ast(lookup_source, ast.Subscript)
    table_name = name_id(None if lookup is None else lookup.value)
    property_method = any(
        (name_id(decorator) == "property" for decorator in method.decorator_list)
    )
    return (
        table_name
        if property_method and lookup is not None and (name_id(lookup.slice) == "self")
        else None
    )


def _enum_metadata_table_candidates(
    module: ParsedModule,
) -> tuple[EnumMetadataTableCandidate, ...]:
    table_cases = _enum_metadata_table_cases(module.module)
    candidates: list[EnumMetadataTableCandidate] = []
    for statement in module.module.body:
        if not isinstance(statement, ast.ClassDef):
            continue
        property_tables: dict[str, list[str]] = defaultdict(list)
        for item in statement.body:
            if isinstance(item, ast.FunctionDef):
                table_name = _enum_metadata_property_table(item)
                if table_name is not None:
                    property_tables[table_name].append(item.name)
        for table_name, property_names in property_tables.items():
            enum_name, case_count = table_cases.get(table_name, (None, 0))
            if enum_name == statement.name:
                candidates.append(
                    EnumMetadataTableCandidate(
                        file_path=str(module.path),
                        line=statement.lineno,
                        class_name=statement.name,
                        table_name=table_name,
                        property_names=tuple(property_names),
                        case_count=case_count,
                    )
                )
    return tuple(candidates)


_READABILITY_COMPRESSED_LINE_LIMIT = 140
_READABILITY_STRING_TOKEN_TYPE_NAMES = (
    "STRING",
    "FSTRING_START",
    "FSTRING_MIDDLE",
    "FSTRING_END",
    "TSTRING_START",
    "TSTRING_MIDDLE",
    "TSTRING_END",
)
_READABILITY_STRING_TOKEN_TYPES = frozenset(
    getattr(tokenize, token_type_name)
    for token_type_name in _READABILITY_STRING_TOKEN_TYPE_NAMES
    if hasattr(tokenize, token_type_name)
)
_READABILITY_INLINE_SUITE_TYPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Match,
)


def _readability_semicolon_counts(source: str) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.OP and token.string == ";":
            counts[token.start[0]] += 1
    return counts


def _readability_line_length(line: str) -> int:
    return len(line.expandtabs(4).rstrip("\n\r"))


def _readability_line_is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _readability_line_has_string_token(line: str) -> bool:
    return any(
        token.type in _READABILITY_STRING_TOKEN_TYPES
        for token in tokenize.generate_tokens(io.StringIO(f"{line}\n").readline)
    )


def _readability_compressed_line_candidates(
    module: ParsedModule,
) -> tuple[ReadabilityCompressedLineCandidate, ...]:
    lines = module.source.splitlines()
    reasons_by_line: dict[int, set[str]] = defaultdict(set)
    statement_counts_by_line: dict[int, int] = defaultdict(lambda: 1)

    for line_number, line in enumerate(lines, 1):
        if _readability_line_is_comment_or_blank(line):
            continue
        if _readability_line_length(
            line
        ) > _READABILITY_COMPRESSED_LINE_LIMIT and not _readability_line_has_string_token(
            line
        ):
            reasons_by_line[line_number].add("overlong physical line")

    for line_number, semicolon_count in _readability_semicolon_counts(
        module.source
    ).items():
        reasons_by_line[line_number].add("semicolon-separated statements")
        statement_counts_by_line[line_number] = max(
            statement_counts_by_line[line_number],
            semicolon_count + 1,
        )

    for node in ast.walk(module.module):
        if not isinstance(node, _READABILITY_INLINE_SUITE_TYPES):
            continue
        body = getattr(node, "body", ())
        if body and getattr(body[0], "lineno", None) == node.lineno:
            reasons_by_line[node.lineno].add(f"inline {type(node).__name__} suite")
            statement_counts_by_line[node.lineno] = max(
                statement_counts_by_line[node.lineno],
                len(body),
            )

    return tuple(
        ReadabilityCompressedLineCandidate(
            file_path=str(module.path),
            line=line_number,
            char_count=_readability_line_length(lines[line_number - 1]),
            reason=", ".join(sorted(reasons)),
            statement_count=statement_counts_by_line[line_number],
        )
        for line_number, reasons in sorted(reasons_by_line.items())
    )


def _dataclass_config_field_names(node: ast.ClassDef) -> tuple[str, ...]:
    if not any(
        (
            name_id(decorator) == "dataclass"
            or (
                isinstance(decorator, ast.Call)
                and name_id(decorator.func) == "dataclass"
            )
            for decorator in node.decorator_list
        )
    ):
        return ()
    return tuple(
        (
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        )
    )


def _from_namespace_keyword_names(
    node: ast.ClassDef,
) -> tuple[int, tuple[str, ...]] | None:
    for statement in node.body:
        if (
            not isinstance(statement, ast.FunctionDef)
            or statement.name != "from_namespace"
        ):
            continue
        for call in ast.walk(statement):
            if isinstance(call, ast.Call) and name_id(call.func) == "cls":
                keyword_names = tuple(
                    (
                        keyword.arg
                        for keyword in call.keywords
                        if keyword.arg is not None
                    )
                )
                if keyword_names:
                    return (statement.lineno, keyword_names)
    return None


def _argument_spec_field_name(node: ast.AST) -> str | None:
    call = as_ast(node, ast.Call)
    if call is None or not (name_id(call.func) or "").endswith("ArgumentSpec"):
        return None
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    flags = as_ast(keywords.get("flags"), ast.Tuple)
    if flags is None or not flags.elts:
        return None
    first_flag = constant_value(flags.elts[0])
    if not isinstance(first_flag, str) or not first_flag.startswith("--"):
        return None
    dest_name = constant_value(keywords.get("dest"))
    return (
        dest_name
        if isinstance(dest_name, str)
        else first_flag.removeprefix("--").replace("-", "_")
    )


def _cli_argument_spec_fields(
    module: ParsedModule,
) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    specs: list[tuple[str, int, tuple[str, ...]]] = []
    for statement in module.module.body:
        binding = named_value_binding(statement)
        if binding is None or not isinstance(binding.value, ast.Tuple):
            continue
        field_names = tuple(
            (
                field_name
                for field_name in (
                    _argument_spec_field_name(element) for element in binding.value.elts
                )
                if field_name is not None
            )
        )
        if field_names:
            specs.append((binding.target_name, binding.line, field_names))
    return tuple(specs)


def _dataclass_namespace_cli_mirror_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[DataclassNamespaceCliMirrorCandidate, ...]:
    cli_specs = tuple(
        (
            (module, spec_name, spec_line, field_names)
            for module in modules
            for spec_name, spec_line, field_names in _cli_argument_spec_fields(module)
        )
    )
    candidates: list[DataclassNamespaceCliMirrorCandidate] = []
    for module in modules:
        for node in module.module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            dataclass_fields = _dataclass_config_field_names(node)
            if not dataclass_fields:
                continue
            namespace_assignment = _from_namespace_keyword_names(node)
            if namespace_assignment is None:
                continue
            from_namespace_line, namespace_fields = namespace_assignment
            mirrored_fields = tuple(
                (name for name in namespace_fields if name in dataclass_fields)
            )
            if len(mirrored_fields) < 4:
                continue
            for spec_module, spec_name, spec_line, cli_field_names in cli_specs:
                mirrored_cli_fields = tuple(
                    (name for name in cli_field_names if name in dataclass_fields)
                )
                shared_fields = tuple(
                    (name for name in mirrored_fields if name in mirrored_cli_fields)
                )
                if len(shared_fields) < 4:
                    continue
                candidates.append(
                    DataclassNamespaceCliMirrorCandidate(
                        file_path=str(module.path),
                        line=node.lineno,
                        class_name=node.name,
                        argument_spec_name=spec_name,
                        field_names=mirrored_fields,
                        cli_field_names=mirrored_cli_fields,
                        from_namespace_line=from_namespace_line,
                        argument_spec_file_path=str(spec_module.path),
                        argument_spec_line=spec_line,
                    )
                )
    return tuple(candidates)


_SEMANTIC_TAG_KEYWORDS = frozenset({"capability_tags", "observation_tags"})
_SEMANTIC_TAG_CONSTANT_SUFFIX = {
    "capability_tags": "CAPABILITY_TAGS",
    "observation_tags": "OBSERVATION_TAGS",
}


def _semantic_tag_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    return name_id(node)


def _semantic_tag_tuple_value(
    keyword: ast.keyword,
) -> tuple[str, tuple[str, ...]] | None:
    tuple_node = as_ast(keyword.value, ast.Tuple)
    if tuple_node is None or len(tuple_node.elts) < 2:
        return None
    tag_names = tuple(_semantic_tag_name(element) for element in tuple_node.elts)
    if any((tag_name is None for tag_name in tag_names)):
        return None
    return ast.unparse(tuple_node), cast(tuple[str, ...], tag_names)


def _semantic_tag_constant_name(keyword_name: str, tag_names: tuple[str, ...]) -> str:
    role_tokens = tuple(
        (
            tag_name.removesuffix("_MAPPING").removesuffix("_TAG")
            for tag_name in tag_names
        )
    )
    return f"_{'_'.join(role_tokens)}_{_SEMANTIC_TAG_CONSTANT_SUFFIX[keyword_name]}"


def _semantic_tag_tuple_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[SemanticTagTupleBoilerplateCandidate, ...]:
    candidates: list[SemanticTagTupleBoilerplateCandidate] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.keyword) or node.arg not in _SEMANTIC_TAG_KEYWORDS:
            continue
        tuple_value = _semantic_tag_tuple_value(node)
        if tuple_value is None:
            continue
        _, tag_names = tuple_value
        location = SourceLocation(str(module.path), node.lineno, node.arg)
        candidates.append(
            SemanticTagTupleBoilerplateCandidate(
                file_path=str(module.path),
                line=node.lineno,
                evidence_locations=(location,),
                keyword_name=node.arg,
                constant_name=_semantic_tag_constant_name(node.arg, tag_names),
                tag_names=tag_names,
            )
        )
    return (*candidates, *_derivable_semantic_tag_constant_candidates(module))


_SEMANTIC_TAG_CONSTANT_SUFFIXES = tuple(_SEMANTIC_TAG_CONSTANT_SUFFIX.values())
_SEMANTIC_TAG_ASSIGNMENT_OWNER_NAMES = frozenset({"CapabilityTag", "ObservationTag"})


def _semantic_tag_constant_member_names(target_name: str | None) -> tuple[str, ...]:
    if target_name is None:
        return ()
    try:
        return tuple(
            tag.name for tag in _semantic_tag_tuple_from_constant_name(target_name)
        )
    except (StopIteration, ValueError):
        return ()


def _semantic_tag_tuple_member_names(
    tuple_node: ast.Tuple | None,
) -> tuple[str, ...]:
    if tuple_node is None or len(tuple_node.elts) < 2:
        return ()
    tag_names = tuple(
        (
            element.attr
            for element in tuple_node.elts
            if isinstance(element, ast.Attribute)
            and name_id(element.value) in _SEMANTIC_TAG_ASSIGNMENT_OWNER_NAMES
        )
    )
    return tag_names if len(tag_names) == len(tuple_node.elts) else ()


def _derivable_semantic_tag_constant_assignment(
    statement: ast.stmt,
) -> tuple[str, str] | None:
    assignment = as_ast(statement, ast.Assign)
    target_name = name_id(single_item(assignment.targets)) if assignment else None
    tuple_node = as_ast(assignment.value, ast.Tuple) if assignment else None
    assigned_tag_names = _semantic_tag_tuple_member_names(tuple_node)
    if (
        not assigned_tag_names
        or assigned_tag_names != _semantic_tag_constant_member_names(target_name)
    ):
        return None
    suffix = next(
        suffix
        for suffix in _SEMANTIC_TAG_CONSTANT_SUFFIXES
        if cast(str, target_name).endswith(f"_{suffix}")
    )
    return suffix.removesuffix("_TAGS").lower(), cast(str, target_name)


def _derivable_semantic_tag_constant_candidates(
    module: ParsedModule,
) -> tuple[SemanticTagTupleBoilerplateCandidate, ...]:
    grouped: dict[str, list[tuple[str, SourceLocation]]] = defaultdict(list)
    for statement in module.module.body:
        constant = _derivable_semantic_tag_constant_assignment(statement)
        if constant is None:
            continue
        tag_kind, constant_name = constant
        grouped[tag_kind].append(
            (
                constant_name,
                SourceLocation(str(module.path), statement.lineno, constant_name),
            )
        )
    return tuple(
        (
            SemanticTagTupleBoilerplateCandidate(
                file_path=str(module.path),
                line=min((location.line for _, location in entries)),
                evidence_locations=tuple((location for _, location in entries)),
                keyword_name=tag_kind,
                constant_name=f"{tag_kind}_tag_constants",
                tag_names=tuple((constant_name for constant_name, _ in entries)),
                source_kind="derived_constant",
            )
            for tag_kind, entries in sorted(grouped.items())
        )
    )


_DERIVED_COUNT_METRIC_SHAPES = {
    "MappingMetrics": ("from_field_names", (("field_count", "field_names"),)),
    "DispatchCountMetrics": (
        "from_literal_family",
        (("dispatch_site_count", "literal_cases"),),
    ),
    "RegistrationMetrics": ("from_class_names", (("class_count", "class_names"),)),
}


def _ast_expression_equal(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def _len_call_argument(node: ast.AST) -> ast.AST | None:
    call = as_ast(node, ast.Call)
    if call is None or _call_name(call.func) != "len" or len(call.args) != 1:
        return None
    return call.args[0]


def _derived_metric_count_pairs(
    call: ast.Call,
    pair_shapes: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    derived_pairs: list[tuple[str, str]] = []
    for count_keyword, collection_keyword in pair_shapes:
        count_value = keywords.get(count_keyword)
        collection_value = keywords.get(collection_keyword)
        if count_value is None or collection_value is None:
            continue
        counted_expression = _len_call_argument(count_value)
        if counted_expression is not None and _ast_expression_equal(
            counted_expression, collection_value
        ):
            derived_pairs.append((count_keyword, collection_keyword))
    return tuple(derived_pairs)


def _derived_metric_count_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[DerivedMetricCountBoilerplateCandidate, ...]:
    candidates: list[DerivedMetricCountBoilerplateCandidate] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Call):
            continue
        metric_class_name = _call_name(node.func)
        metric_shape = _DERIVED_COUNT_METRIC_SHAPES.get(metric_class_name or "")
        if metric_shape is None:
            continue
        constructor_name, pair_shapes = metric_shape
        derived_pairs = _derived_metric_count_pairs(node, pair_shapes)
        if not derived_pairs:
            continue
        candidates.append(
            DerivedMetricCountBoilerplateCandidate(
                file_path=str(module.path),
                line=node.lineno,
                metric_class_name=cast(str, metric_class_name),
                recommended_constructor_name=constructor_name,
                count_keyword_names=tuple((pair[0] for pair in derived_pairs)),
                collection_keyword_names=tuple((pair[1] for pair in derived_pairs)),
            )
        )
    return tuple(candidates)


_EFFECT_STEP_BASE_NAMES = frozenset(
    {
        "AstTypedEffectStep",
        "EffectStep",
        "GuardedEffectStep",
        "RegisteredEffectStep",
        "SingleCompareEffectStep",
    }
)


@dataclass(frozen=True)
class _EffectStepLeakPolicy:
    method_name: str
    minimum_raw_guard_count: int
    requires_optional_exit: bool
    ignore_when_calling_template_hook: bool = False

    def leaks(
        self,
        method: ast.FunctionDef,
        *,
        none_return_count: int,
        raw_guard_count: int,
    ) -> bool:
        if self.ignore_when_calling_template_hook and _method_calls_template_hook(
            method
        ):
            return False
        if self.requires_optional_exit and none_return_count == 0:
            return False
        return raw_guard_count >= self.minimum_raw_guard_count


_EFFECT_STEP_APPLY_LEAK_POLICY = _EffectStepLeakPolicy(
    "apply",
    minimum_raw_guard_count=1,
    requires_optional_exit=True,
    ignore_when_calling_template_hook=True,
)
_EFFECT_STEP_LEAK_POLICIES = (
    _EFFECT_STEP_APPLY_LEAK_POLICY,
    _EffectStepLeakPolicy(
        "accepts", minimum_raw_guard_count=3, requires_optional_exit=False
    ),
    _EffectStepLeakPolicy(
        "project", minimum_raw_guard_count=3, requires_optional_exit=True
    ),
    _EffectStepLeakPolicy(
        "project_ast", minimum_raw_guard_count=3, requires_optional_exit=True
    ),
)
_EFFECT_STEP_TEMPLATE_HOOK_NAMES = frozenset(
    (
        policy.method_name
        for policy in _EFFECT_STEP_LEAK_POLICIES
        if policy is not _EFFECT_STEP_APPLY_LEAK_POLICY
    )
) | frozenset(
    {
        "attribute_from",
        "call_from",
        "comprehension_from",
        "owner_name_from",
        "project_attribute_call",
        "project_call_pair",
        "project_compare",
    }
)
_EFFECT_STEP_LEAF_METHOD_NAMES = tuple(
    (policy.method_name for policy in _EFFECT_STEP_LEAK_POLICIES)
)
_EFFECT_STEP_LEAK_POLICY_BY_METHOD = {
    policy.method_name: policy for policy in _EFFECT_STEP_LEAK_POLICIES
}


def _class_base_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        (
            base_name
            for base in node.bases
            if (base_name := _call_name(base)) is not None
        )
    )


def _looks_like_effect_step_class(node: ast.ClassDef) -> bool:
    return node.name.endswith("Step") or bool(
        set(_class_base_names(node)) & _EFFECT_STEP_BASE_NAMES
    )


def _effect_step_leaf_methods(node: ast.ClassDef) -> tuple[ast.FunctionDef, ...]:
    return tuple(
        (
            statement
            for statement in node.body
            if isinstance(statement, ast.FunctionDef)
            and statement.name in _EFFECT_STEP_LEAF_METHOD_NAMES
        )
    )


def _has_abstract_effect_step_hooks(node: ast.ClassDef) -> bool:
    return any(
        (
            isinstance(statement, ast.FunctionDef)
            and any(
                (
                    _call_name(decorator) == "abstractmethod"
                    for decorator in statement.decorator_list
                )
            )
            for statement in node.body
        )
    )


def _method_calls_template_hook(method: ast.FunctionDef) -> bool:
    return any(
        (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and isinstance(item.func.value, ast.Name)
            and (item.func.value.id == "self")
            and (item.func.attr in _EFFECT_STEP_TEMPLATE_HOOK_NAMES)
            for item in _walk_nodes(method)
        )
    )


def _raw_effect_step_guard_count(method: ast.FunctionDef) -> int:
    raw_guard_count = sum(
        (
            1
            for item in _walk_nodes(method)
            if isinstance(item, ast.Call)
            and (
                _call_name(item.func) in {"as_ast", "isinstance", "single_item"}
                or (
                    isinstance(item.func, ast.Attribute)
                    and item.func.attr in {"args", "keywords"}
                )
            )
            or (
                isinstance(item, ast.Compare)
                and any(
                    (
                        isinstance(value, ast.Call) and _call_name(value.func) == "len"
                        for value in (item.left, *item.comparators)
                    )
                )
            )
        )
    )
    boolean_clause_count = sum(
        (
            max(0, len(item.values) - 1)
            for item in _walk_nodes(method)
            if isinstance(item, ast.BoolOp)
        )
    )
    return raw_guard_count + boolean_clause_count


def _effect_step_method_leaks(method: ast.FunctionDef) -> bool:
    none_return_count = _none_return_count(method)
    raw_guard_count = _raw_effect_step_guard_count(method)
    policy = _EFFECT_STEP_LEAK_POLICY_BY_METHOD.get(method.name)
    return (
        False
        if policy is None
        else policy.leaks(
            method, none_return_count=none_return_count, raw_guard_count=raw_guard_count
        )
    )


def _none_return_count(node: ast.AST) -> int:
    return sum(
        (
            1
            for item in _walk_nodes(node)
            if isinstance(item, ast.Return)
            and isinstance(item.value, ast.Constant)
            and (item.value.value is None)
        )
    )


def _suggested_effect_step_base(method: ast.FunctionDef) -> str:
    ast_type_guards = sum(
        (
            1
            for item in _walk_nodes(method)
            if _isinstance_ast_type_names(item)
            or (
                isinstance(item, ast.Call)
                and _call_name(item.func) == "as_ast"
                and (len(item.args) >= 2)
                and bool(_ast_type_names(item.args[1]))
            )
        )
    )
    return "AstTypedEffectStep" if ast_type_guards else "GuardedEffectStep"


def _effect_step_implementation_leak_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[EffectStepImplementationLeakCandidate]:
    if not _looks_like_effect_step_class(node):
        return
    if _has_abstract_effect_step_hooks(node):
        return
    for method in _effect_step_leaf_methods(node):
        if not _effect_step_method_leaks(method):
            continue
        yield EffectStepImplementationLeakCandidate(
            file_path=str(module.path),
            class_name=node.name,
            method_name=method.name,
            line=method.lineno,
            none_return_count=_none_return_count(method),
            raw_guard_count=_raw_effect_step_guard_count(method),
            suggested_base_name=_suggested_effect_step_base(method),
        )


def _effect_step_implementation_leak_candidates(
    module: ParsedModule,
) -> tuple[EffectStepImplementationLeakCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _effect_step_implementation_leak_candidates_for_class,
        sort_key=lambda item: (item.file_path, item.line),
    )


def _assignment_target_arity(target: ast.AST) -> int | None:
    if isinstance(target, ast.Name):
        return 1
    if isinstance(target, (ast.Tuple, ast.List)):
        if not target.elts or not all(
            (isinstance(item, ast.Name) for item in target.elts)
        ):
            return None
        return len(target.elts)
    return None


def _result_assembly_pipeline_functions(
    module: ParsedModule,
) -> tuple[ResultAssemblyPipelineFunction, ...]:
    functions: list[ResultAssemblyPipelineFunction] = []
    for qualname, function in _iter_named_functions(module):
        stages = _pipeline_body_stages(function)
        if stages is None:
            continue
        functions.append(
            ResultAssemblyPipelineFunction(
                file_path=str(module.path),
                qualname=qualname,
                lineno=function.lineno,
                stages=stages,
            )
        )
    return sorted_tuple(functions, key=lambda item: (item.lineno, item.qualname))


def _shared_pipeline_tail(
    left: ResultAssemblyPipelineFunction,
    right: ResultAssemblyPipelineFunction,
) -> tuple[PipelineAssemblyStage, ...]:
    shared: list[PipelineAssemblyStage] = []
    left_index = len(left.stages) - 1
    right_index = len(right.stages) - 1
    while left_index >= 0 and right_index >= 0:
        left_stage = left.stages[left_index]
        right_stage = right.stages[right_index]
        if left_stage.shape_key != right_stage.shape_key:
            break
        shared.append(left_stage)
        left_index -= 1
        right_index -= 1
    return tuple(reversed(shared))


def _repeated_result_assembly_pipeline_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[RepeatedResultAssemblyPipelineCandidate, ...]:
    functions = _result_assembly_pipeline_functions(module)
    if len(functions) < 2:
        return ()
    grouped_functions: dict[
        (
            tuple[tuple[object, ...], ...],
            tuple[
                tuple[PipelineAssemblyStage, ...], set[ResultAssemblyPipelineFunction]
            ],
        )
    ] = {}
    for left, right in combinations(functions, 2):
        shared_tail = _shared_pipeline_tail(left, right)
        if len(shared_tail) < config.min_shared_pipeline_stages:
            continue
        if len(shared_tail) >= len(left.stages) or len(shared_tail) >= len(
            right.stages
        ):
            continue
        if shared_tail[-1].kind != _PIPELINE_RETURN_STAGE:
            continue
        distinct_stage_names = {stage.callee_name for stage in shared_tail}
        if len(distinct_stage_names) < config.min_shared_pipeline_stages - 1:
            continue
        key = tuple(stage.shape_key for stage in shared_tail)
        if key not in grouped_functions:
            grouped_functions[key] = (shared_tail, set())
        grouped_functions[key][1].update((left, right))

    candidates = [
        RepeatedResultAssemblyPipelineCandidate(
            file_path=str(module.path),
            shared_tail=shared_tail,
            functions=sorted_tuple(
                grouped, key=lambda item: (item.lineno, item.qualname)
            ),
        )
        for shared_tail, grouped in grouped_functions.values()
        if len(grouped) >= 2
    ]
    filtered_candidates: list[RepeatedResultAssemblyPipelineCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -len(item.shared_tail),
            -len(item.functions),
            item.functions[0].qualname,
        ),
    ):
        candidate_function_names = tuple(
            (function.qualname for function in candidate.functions)
        )
        if any(
            (
                len(existing.shared_tail) >= len(candidate.shared_tail)
                and candidate_function_names
                == tuple((function.qualname for function in existing.functions))
                for existing in filtered_candidates
            )
        ):
            continue
        filtered_candidates.append(candidate)
    return tuple(filtered_candidates)


def _direct_forwarded_parameter_names(
    call: ast.Call,
    *,
    parameter_names: set[str],
) -> tuple[str, ...] | None:
    forwarded: list[str] = []
    seen: set[str] = set()
    for argument in call.args:
        if isinstance(argument, ast.Name) and argument.id in parameter_names:
            if argument.id not in seen:
                seen.add(argument.id)
                forwarded.append(argument.id)
            continue
        return None
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        if isinstance(keyword.value, ast.Name) and keyword.value.id in parameter_names:
            if keyword.value.id not in seen:
                seen.add(keyword.value.id)
                forwarded.append(keyword.value.id)
            continue
        return None
    return tuple(forwarded)


def _qualified_call_display_name(node: ast.Call) -> str:
    return ast.unparse(node.func)


def _nested_builder_shell_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
    config: DetectorConfig,
) -> Iterable[NestedBuilderShellCandidate]:
    body = _trim_docstring_body(list(function.body))
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return
    returned = body[0].value
    if not isinstance(returned, ast.Call) or returned.args:
        return
    outer_callee_name = _call_name(returned.func)
    if outer_callee_name is None:
        return
    parameter_names = _function_param_names(function) - {"self", "cls"}
    if len(parameter_names) < config.min_nested_builder_forwarded_params:
        return
    nested_matches: list[tuple[str, str, tuple[str, ...]]] = []
    for keyword in returned.keywords:
        if keyword.arg is None or not isinstance(keyword.value, ast.Call):
            continue
        nested_callee_name = _qualified_call_display_name(keyword.value)
        if (
            not nested_callee_name
            or _call_name(keyword.value.func) == outer_callee_name
        ):
            continue
        forwarded = _direct_forwarded_parameter_names(
            keyword.value, parameter_names=parameter_names
        )
        if forwarded is None:
            continue
        if len(forwarded) < config.min_nested_builder_forwarded_params:
            continue
        nested_matches.append((keyword.arg, nested_callee_name, forwarded))
    if len(nested_matches) != 1:
        return
    nested_field_name, nested_callee_name, forwarded_parameter_names = nested_matches[0]
    residue_keywords = tuple(
        keyword
        for keyword in returned.keywords
        if keyword.arg is not None and keyword.arg != nested_field_name
    )
    if not residue_keywords:
        return
    residue_source_names = sorted_tuple(
        {
            root_name
            for keyword in residue_keywords
            for root_name in _expression_root_names(keyword.value)
            if root_name in parameter_names - set(forwarded_parameter_names)
        }
    )
    if not residue_source_names:
        return
    yield NestedBuilderShellCandidate(
        file_path=str(module.path),
        qualname=qualname,
        lineno=function.lineno,
        outer_callee_name=outer_callee_name,
        nested_field_name=nested_field_name,
        nested_callee_name=nested_callee_name,
        forwarded_parameter_names=forwarded_parameter_names,
        residue_field_names=tuple(
            keyword.arg for keyword in residue_keywords if keyword.arg is not None
        ),
        residue_source_names=residue_source_names,
    )


def _nested_builder_shell_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[NestedBuilderShellCandidate, ...]:
    return _collect_configured_named_function_candidates(
        module,
        config,
        _nested_builder_shell_candidates_for_function,
        sort_key=lambda item: (item.file_path, item.lineno),
    )


def _indexed_family_wrapper_candidates_for_function(
    module: ParsedModule, node: ast.FunctionDef
) -> Iterable[IndexedFamilyWrapperCandidate]:
    del module
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
        return
    value = node.body[0].value
    if not isinstance(value, ast.ListComp) or len(value.generators) != 1:
        return
    generator = value.generators[0]
    if not isinstance(generator.target, ast.Name) or generator.target.id != "item":
        return
    if not isinstance(generator.iter, ast.Call):
        return
    collector_name = _call_name(generator.iter.func)
    if collector_name not in {
        "_collect_items_from_spec_root",
        "collect_family_items",
    }:
        return
    if collector_name == "_collect_items_from_spec_root":
        if len(generator.iter.args) < 3:
            return
        spec_root_name = _call_name(generator.iter.args[0])
        item_type_name = _call_name(generator.iter.args[2])
    else:
        if len(generator.iter.args) < 2:
            return
        spec_root_name = _call_name(generator.iter.args[1])
        item_type_name = _call_name(generator.iter.args[1])
    if spec_root_name is None or item_type_name is None:
        return
    if not _is_instance_filter(generator.ifs, item_type_name):
        return
    yield IndexedFamilyWrapperCandidate(
        function_name=node.name,
        lineno=node.lineno,
        collector_name=collector_name,
        spec_root_name=spec_root_name,
        item_type_name=item_type_name,
    )


def _indexed_family_wrapper_candidates(
    module: ParsedModule,
) -> tuple[IndexedFamilyWrapperCandidate, ...]:
    return _collect_ast_node_candidates(
        module,
        module.module,
        ast.FunctionDef,
        _indexed_family_wrapper_candidates_for_function,
        sort_key=lambda item: item.lineno,
    )


def _is_instance_filter(filters: list[ast.expr], item_type_name: str) -> bool:
    for condition in filters:
        if not isinstance(condition, ast.Call):
            continue
        if _call_name(condition.func) != "isinstance":
            continue
        if len(condition.args) != 2:
            continue
        if (
            not isinstance(condition.args[0], ast.Name)
            or condition.args[0].id != "item"
        ):
            continue
        if _call_name(condition.args[1]) == item_type_name:
            return True
    return False


def _function_has_param(
    function: ast.FunctionDef | ast.AsyncFunctionDef, param_name: str
) -> bool:
    return any((arg.arg == param_name for arg in function.args.args))


def _collect_class_sentinel_attrs(
    module: ast.Module,
) -> dict[str, list[SourceLocation]]:
    grouped: dict[str, list[SourceLocation]] = defaultdict(list)
    for node in _walk_nodes(module):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if not isinstance(stmt.value, ast.Constant):
                continue
            if not isinstance(stmt.value.value, (str, int, bool)):
                continue
            grouped[target.id].append(
                SourceLocation("<module>", stmt.lineno, f"{node.name}.{target.id}")
            )
    return grouped


def _predicate_factory_chain_branch_count(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int | None:
    current = as_ast(function.body[0], ast.If) if function.body else None
    branches: list[ast.If] = []
    while current is not None:
        branches.append(current)
        current = as_ast(single_item(current.orelse), ast.If)
    if len(branches) < 2:
        return None
    for branch in branches:
        if not _test_has_call(branch.test):
            return None
        if not any((return_call(statement) is not None for statement in branch.body)):
            return None
    return len(branches)


def _test_has_call(node: ast.AST) -> bool:
    return any((isinstance(child, ast.Call) for child in _walk_nodes(node)))


__all__ = tuple(name for name in globals() if not name.startswith("__"))
