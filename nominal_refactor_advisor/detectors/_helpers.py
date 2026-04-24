"""Shared detector helper functions.

This module contains private analysis helpers that support detector families
across the split implementation modules.
"""

from __future__ import annotations

from ._base import *
from ._substrate_support import *

def _semantic_dict_bag_candidates(
    module: ParsedModule,
) -> list[SemanticDictBagCandidate]:
    candidates: list[SemanticDictBagCandidate] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.function_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            for stmt in _trim_docstring_body(node.body):
                self.visit(stmt)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            candidates.extend(
                _function_local_semantic_dict_bag_candidates(
                    module, node, tuple(self.class_stack)
                )
            )
            self.function_stack.append(node.name)
            for stmt in _trim_docstring_body(node.body):
                self.visit(stmt)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            candidates.extend(
                _function_local_semantic_dict_bag_candidates(
                    module, node, tuple(self.class_stack)
                )
            )
            self.function_stack.append(node.name)
            for stmt in _trim_docstring_body(node.body):
                self.visit(stmt)
            self.function_stack.pop()

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
                    items,
                    owner_symbol=owner_symbol,
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

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is self.target_node:
                for stmt in _trim_docstring_body(node.body):
                    self.visit(stmt)

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
                and (items := _string_dict_items(node.value)) is not None
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
            tuple(sorted(touched_keys)),
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
                key_names=tuple(sorted(touched_keys)),
                context_kind="local_string_key_bag",
                recommendation=recommendation,
            )
        )
    return candidates

def _recommend_metrics_dataclass(
    items: dict[str, ast.AST], owner_symbol: str
) -> SemanticDataclassRecommendation:
    key_names = tuple(sorted(items))
    exact_schema = _exact_schema_match(key_names, _METRIC_BAG_SCHEMAS)
    if exact_schema is not None:
        class_name = exact_schema.class_name
        base_class_name = exact_schema.base_class_name
        rationale = f"Use existing `{class_name}`, which already inherits `{base_class_name}` for this semantic field family."
        scaffold = _instantiation_scaffold(
            class_name,
            key_names,
            items,
            prefix="metrics=",
        )
        return SemanticDataclassRecommendation.existing_schema(
            class_name,
            base_class_name,
            rationale,
            scaffold,
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
        class_name,
        base_class_name,
        key_names,
        items,
        instantiation_prefix="metrics=",
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
            class_name,
            exact_schema.base_class_name,
            rationale,
            scaffold,
        )

    closest_schema = _closest_schema_match(key_names, (_IMPACT_BAG_SCHEMA,))
    if closest_schema is None:
        if not (variable_name.endswith("metrics") or variable_name in {"metrics"}):
            return None
        return _recommend_metrics_dataclass(value_nodes, owner_symbol=owner_symbol)

    class_name = _suggest_dataclass_name(owner_symbol, "ImpactDelta")
    rationale = f"Create `{class_name}` inheriting from `{closest_schema.class_name}` because the local bag carries the same quantified impact fields nominally modeled there."
    scaffold = _declaration_scaffold(
        class_name,
        closest_schema.class_name,
        key_names,
        value_nodes,
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
    if not left and not right:
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
        f"    {key}: {_infer_field_type_name(key, value_nodes.get(key))}"
        for key in key_names
    )
    return (
        "@dataclass(frozen=True)\n"
        f"class {class_name}({base_class_name}):\n"
        f"{field_lines}\n\n"
        f"{_instantiation_scaffold(class_name, key_names, value_nodes, prefix=instantiation_prefix)}"
    )

def _instantiation_scaffold(
    class_name: str,
    key_names: tuple[str, ...],
    value_nodes: dict[str, ast.AST],
    prefix: str = "",
) -> str:
    rendered_args = ",\n    ".join(
        f"{key}={_render_value_expression(key, value_nodes.get(key))}"
        for key in key_names
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
    return tuple(
        sorted(
            name
            for name, value in _class_direct_assignments(node).items()
            if not (isinstance(value, ast.Constant) and value.value is None)
        )
    )

def _abstract_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        sorted(
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
                statement.target.id,
                ast.unparse(statement.annotation),
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
                and target.value.id == "self"
                and isinstance(value, ast.Name)
                and value.id in parameter_annotations
            ):
                continue
            typed_fields.setdefault(target.attr, parameter_annotations[value.id])
    return tuple(sorted(typed_fields.items()))

def _semantic_role_names_for_fields(field_names: tuple[str, ...]) -> tuple[str, ...]:
    role_names: set[str] = set()
    for field_name in field_names:
        normalized_roles = _normalize_semantic_field_roles(field_name)
        if normalized_roles:
            role_names.update(normalized_roles)
            continue
        role_names.add(field_name)
    return tuple(sorted(role_names))

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
                    field_names=tuple(name for name, _ in field_type_map),
                    field_type_map=field_type_map,
                    method_names=tuple(sorted(_method_names(node))),
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
        return tuple(sorted(seen))

    return tuple(
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
        return tuple(
            sorted(
                compatible,
                key=lambda authority: (
                    -len(authority.field_names),
                    not authority.is_abstract,
                    authority.class_name,
                ),
            )
        )

def _is_reusable_nominal_authority(shape: NominalAuthorityShape) -> bool:
    if shape.class_name.endswith("Detector"):
        return False
    return bool(
        shape.is_abstract or shape.class_name.endswith(("Base", "Mixin", "Carrier"))
    )

def _shared_typed_field_names(
    concrete: NominalAuthorityShape,
    authority: NominalAuthorityShape,
) -> tuple[str, ...]:
    concrete_types = dict(concrete.field_type_map)
    return tuple(
        name
        for name, annotation_text in authority.field_type_map
        if concrete_types.get(name) == annotation_text
    )

def _extract_family_roster_members(
    node: ast.AST,
    known_class_names: set[str],
) -> tuple[tuple[str, ...], str] | None:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return None
    member_names: list[str] = []
    constructor_styles: set[str] = set()
    for element in node.elts:
        if isinstance(element, ast.Name) and element.id in known_class_names:
            member_names.append(element.id)
            constructor_styles.add("class_reference")
            continue
        if (
            isinstance(element, ast.Call)
            and not element.args
            and not element.keywords
            and isinstance(element.func, ast.Name)
            and element.func.id in known_class_names
        ):
            member_names.append(element.func.id)
            constructor_styles.add("constructor_call")
            continue
        return None
    if len(member_names) < 2:
        return None
    return (tuple(member_names), "+".join(sorted(constructor_styles)))

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
        if target_name is None or value is None or not isinstance(value, ast.Dict):
            continue
        key_pairs = tuple(
            key_pair
            for key_pair in (
                _enum_key_family(key) for key in value.keys if key is not None
            )
            if key_pair is not None
        )
        if len(key_pairs) < 2 or len(key_pairs) != len(value.keys):
            continue
        family_names = {family_name for family_name, _ in key_pairs}
        if len(family_names) != 1:
            continue
        family_name = next(iter(family_names))
        key_names = tuple(sorted(member_name for _, member_name in key_pairs))
        family_maps[family_name].append((target_name, statement.lineno, key_names))

    candidates: list[FragmentedFamilyAuthorityCandidate] = []
    for family_name, entries in family_maps.items():
        if len(entries) < 2:
            continue
        key_counter: Counter[str] = Counter(
            key_name for _, _, key_names in entries for key_name in set(key_names)
        )
        shared_keys = tuple(
            sorted(key for key, count in key_counter.items() if count >= 2)
        )
        if len(shared_keys) < 3:
            continue
        total_keys = tuple(sorted(key_counter))
        ordered_entries = sorted(entries, key=lambda item: item[1])
        candidates.append(
            FragmentedFamilyAuthorityCandidate(
                file_path=str(module.path),
                mapping_names=tuple(item[0] for item in ordered_entries),
                line_numbers=tuple(item[1] for item in ordered_entries),
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
            and value.value.id == "self"
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

def _finding_assembly_pipeline_candidates(
    module: ParsedModule,
) -> tuple[FindingAssemblyPipelineCandidate, ...]:
    candidates: list[FindingAssemblyPipelineCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef) or not _is_detectorish_class(node):
            continue
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
            continue
        build_call = _finding_build_call(method)
        if build_call is None:
            continue
        candidate_source_name = _candidate_source_name_from_method(method)
        if candidate_source_name is None:
            continue
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
            continue
        candidates.append(
            FindingAssemblyPipelineCandidate(
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
        )
    return tuple(candidates)

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
        and node.body[0].value.value is None
        and not node.orelse
    )

def _delegate_name_from_return(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        outer_name = _call_display_name(node)
        if outer_name in {"tuple", "list", "set"} and len(node.args) == 1:
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

def _guarded_delegator_candidates(
    module: ParsedModule,
) -> tuple[GuardedDelegatorCandidate, ...]:
    candidates: list[GuardedDelegatorCandidate] = []
    for node in _walk_nodes(module.module):
        if (
            not isinstance(node, ast.ClassDef)
            or not _is_observation_spec_class(node)
            or _is_abstract_class(node)
        ):
            continue
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
            candidates.append(
                GuardedDelegatorCandidate(
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
            )
    return tuple(candidates)

def _name_mentions(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(current, ast.Name) and current.id == name for current in _walk_nodes(node)
    )

def _raised_exception_name(
    statement: ast.stmt,
) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(statement, ast.Raise) or statement.exc is None:
        return None
    exc = statement.exc
    if isinstance(exc, ast.Call):
        exc_name = _ast_terminal_name(exc.func)
        referenced_names = tuple(
            sorted(
                {
                    current.id
                    for current in _walk_nodes(exc)
                    if isinstance(current, ast.Name)
                }
            )
        )
        if exc_name is not None:
            return exc_name, referenced_names
    exc_name = _ast_terminal_name(exc)
    if exc_name is not None:
        return exc_name, ()
    return None

def _linear_query_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, tuple[str, ...], str, str] | None:
    body = _trim_docstring_body(node.body)
    if len(body) < 2:
        return None
    loop = next((statement for statement in body if isinstance(statement, ast.For)), None)
    if loop is None or not isinstance(loop.target, ast.Name):
        return None
    result_name = loop.target.id
    return_exprs = [
        current.value
        for current in _walk_nodes(loop)
        if isinstance(current, ast.Return) and current.value is not None
    ]
    if len(return_exprs) != 1 or not _name_mentions(return_exprs[0], result_name):
        return None
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
    parameter_names = tuple(
        arg.arg
        for arg in (
            tuple(node.args.posonlyargs)
            + tuple(node.args.args)
            + tuple(node.args.kwonlyargs)
        )
        if arg.arg not in {"self", "cls"}
    )
    query_key_names = tuple(
        sorted(
            name
            for name in parameter_names
            if any(_name_mentions(current, name) for current in return_exprs)
            or name in exception_names
            or any(
                isinstance(current, ast.If) and _name_mentions(current.test, name)
                for current in _walk_nodes(loop)
            )
        )
    )
    if not query_key_names:
        return None
    return (
        ast.unparse(loop.iter),
        query_key_names,
        ast.unparse(return_exprs[0]),
        exception_name,
    )

def _derived_query_index_candidates(
    module: ParsedModule,
) -> tuple[DerivedQueryIndexCandidate, ...]:
    grouped: dict[
        tuple[str, str, str],
        list[tuple[str, int, tuple[str, ...]]],
    ] = defaultdict(list)
    for qualname, function in _iter_named_functions(module):
        signature = _linear_query_signature(function)
        if signature is None:
            continue
        source_expression, query_key_names, return_expression, exception_name = signature
        grouped[(source_expression, return_expression, exception_name)].append(
            (qualname, function.lineno, query_key_names)
        )
    candidates: list[DerivedQueryIndexCandidate] = []
    for (source_expression, return_expression, exception_name), entries in grouped.items():
        if len(entries) < 2:
            continue
        ordered = tuple(sorted(entries, key=lambda item: (item[1], item[0])))
        query_key_names = tuple(
            sorted(
                {
                    key_name
                    for _, _, entry_query_key_names in ordered
                    for key_name in entry_query_key_names
                }
            )
        )
        candidates.append(
            DerivedQueryIndexCandidate(
                file_path=str(module.path),
                line_numbers=tuple(item[1] for item in ordered),
                function_names=tuple(item[0] for item in ordered),
                source_expression=source_expression,
                query_key_names=query_key_names,
                return_expressions=tuple(return_expression for _ in ordered),
                exception_names=(exception_name,),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.source_expression,
                item.function_names,
            ),
        )
    )

def _simple_attribute_accesses(node: ast.AST) -> tuple[tuple[str, str], ...]:
    return tuple(
        (current.value.id, current.attr)
        for current in _walk_nodes(node)
        if isinstance(current, ast.Attribute)
        and isinstance(current.value, ast.Name)
        and current.value.id not in {"self", "cls"}
    )

def _projection_source_name(node: ast.Call) -> str | None:
    source_counts: Counter[str] = Counter(
        root_name
        for keyword in node.keywords
        if keyword.arg is not None
        for root_name, _ in _simple_attribute_accesses(keyword.value)
    )
    if not source_counts:
        return None
    source_name, count = source_counts.most_common(1)[0]
    if count < 3:
        return None
    if sum(1 for value in source_counts.values() if value == count) > 1:
        return None
    return source_name

def _direct_source_attribute_name(node: ast.AST, source_name: str) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == source_name
    ):
        return node.attr
    return None

def _resolver_lookup_metadata(
    node: ast.AST, source_name: str
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    table_names: set[str] = set()
    selector_field_names = tuple(
        sorted(
            {
                attr_name
                for root_name, attr_name in _simple_attribute_accesses(node)
                if root_name == source_name
            }
        )
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
    return (tuple(sorted(table_names)), selector_field_names)

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
    candidates: list[RuntimeAdapterShellCandidate] = []
    for qualname, function in _iter_named_functions(module):
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
            candidates.append(
                RuntimeAdapterShellCandidate(
                    file_path=str(module.path),
                    line=function.lineno,
                    function_name=qualname,
                    adapter_class_name=adapter_class_name,
                    source_name=source_name,
                    copied_field_names=tuple(sorted(copied_field_names)),
                    resolver_field_names=tuple(sorted(resolver_field_names)),
                    resolver_table_names=tuple(sorted(resolver_table_names)),
                    selector_field_names=tuple(sorted(selector_field_names)),
                    evidence_locations=tuple(evidence[:6]),
                )
            )
            break
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.line, item.function_name),
        )
    )

def _is_none_guard_for_source_attr(
    node: ast.AST, source_name: str
) -> tuple[str, str] | None:
    if (
        not isinstance(node, ast.Compare)
        or len(node.ops) != 1
        or len(node.comparators) != 1
        or not isinstance(node.ops[0], (ast.IsNot, ast.NotEq))
    ):
        return None
    attr_name = _direct_source_attribute_name(node.left, source_name)
    comparator = node.comparators[0]
    if attr_name is None or not (
        isinstance(comparator, ast.Constant) and comparator.value is None
    ):
        return None
    return (source_name, attr_name)

def _keyword_bag_adapter_candidates(
    module: ParsedModule,
) -> tuple[KeywordBagAdapterCandidate, ...]:
    candidates: list[KeywordBagAdapterCandidate] = []
    for qualname, function in _iter_named_functions(module):
        body = _trim_docstring_body(function.body)
        if len(body) < 2:
            continue
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
                for key, value in zip(
                    value.keys, value.values, strict=False
                ):
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
            continue
        candidates.append(
            KeywordBagAdapterCandidate(
                file_path=str(module.path),
                line=function.lineno,
                function_name=qualname,
                source_name=source_name,
                key_names=tuple(sorted(key_names)),
                source_field_names=tuple(sorted(source_field_names)),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.line, item.function_name),
        )
    )

def _structural_observation_property_candidates(
    module: ParsedModule,
) -> tuple[StructuralObservationPropertyCandidate, ...]:
    candidates: list[StructuralObservationPropertyCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
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
            keyword_names = tuple(
                sorted(
                    keyword.arg
                    for keyword in returned.keywords
                    if keyword.arg is not None
                )
            )
            if len(keyword_names) < 6:
                continue
            candidates.append(
                StructuralObservationPropertyCandidate(
                    file_path=str(module.path),
                    line=statement.lineno,
                    subject_name=node.name,
                    name_family=keyword_names,
                    property_name=statement.name,
                    constructor_name=constructor_name,
                )
            )
    return tuple(candidates)

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
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.line,
                item.class_name,
                item.compatible_authority_name,
            ),
        )
    )

def _normalized_authority_name(annotation_text: str) -> str:
    text = annotation_text.strip("\"'")
    text = re.split(r"\s*\|\s*", text, maxsplit=1)[0]
    text = re.split(r"[\[,]", text, maxsplit=1)[0]
    return text.rsplit(".", 1)[-1].strip()

def _is_self_delegate_attribute(node: ast.AST, delegate_field_name: str) -> bool:
    return bool(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == delegate_field_name
    )

def _is_forwarded_parameter_reference(
    node: ast.AST,
    parameter_names: tuple[str, ...],
) -> bool:
    return (
        isinstance(node, ast.Name) and node.id in set(parameter_names)
    ) or (
        isinstance(node, ast.Starred)
        and isinstance(node.value, ast.Name)
        and node.value.id in set(parameter_names)
    )

def _forwarded_delegate_member_name(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    delegate_field_name: str,
) -> str | None:
    body = _trim_docstring_body(method.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    returned = body[0].value
    if any(_ast_terminal_name(decorator) == "property" for decorator in method.decorator_list):
        if (
            isinstance(returned, ast.Attribute)
            and _is_self_delegate_attribute(returned.value, delegate_field_name)
            and method.name == returned.attr
        ):
            return returned.attr
        return None
    if not (
        isinstance(returned, ast.Call)
        and isinstance(returned.func, ast.Attribute)
        and _is_self_delegate_attribute(returned.func.value, delegate_field_name)
        and method.name == returned.func.attr
    ):
        return None
    parameter_names = tuple(
        arg.arg
        for arg in (
            *method.args.posonlyargs,
            *method.args.args[1:],
            *method.args.kwonlyargs,
        )
    )
    if not all(
        _is_forwarded_parameter_reference(argument, parameter_names)
        for argument in returned.args
    ):
        return None
    if not all(
        keyword.arg is None
        or (
            keyword.arg in set(parameter_names)
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == keyword.arg
        )
        for keyword in returned.keywords
    ):
        return None
    return returned.func.attr

def _pass_through_nominal_wrapper_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[PassThroughNominalWrapperCandidate, ...]:
    index = NominalAuthorityIndex(modules)
    candidates: list[PassThroughNominalWrapperCandidate] = []
    for module in modules:
        for node in _walk_nodes(module.module):
            if not isinstance(node, ast.ClassDef) or _is_abstract_class(node):
                continue
            typed_fields = _typed_field_map(node)
            if len(typed_fields) != 1:
                continue
            delegate_field_name, annotation_text = typed_fields[0]
            delegate_authority_name = _normalized_authority_name(annotation_text)
            if not delegate_authority_name:
                continue
            if delegate_authority_name in set(_declared_base_names(node)):
                continue
            authorities = tuple(
                authority
                for authority in index.shapes_named(delegate_authority_name)
                if _is_reusable_nominal_authority(authority)
            )
            if not authorities:
                continue
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
                continue
            if not set(forwarded_member_names) <= set(authority.method_names):
                continue
            candidates.append(
                PassThroughNominalWrapperCandidate(
                    file_path=str(module.path),
                    line=node.lineno,
                    subject_name=node.name,
                    name_family=tuple(sorted(set(forwarded_member_names))),
                    delegate_field_name=delegate_field_name,
                    delegate_authority_file_path=authority.file_path,
                    delegate_authority_name=authority.class_name,
                    delegate_authority_line=authority.line,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.line,
                item.class_name,
                item.delegate_authority_name,
            ),
        )
    )

def _is_projection_like_builder_value(value_fingerprint: str) -> bool:
    return value_fingerprint.startswith(
        (
            "Name(",
            "Attribute(",
            "IfExp(",
            "Constant(",
        )
    )

def _projection_builder_groups(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[tuple[BuilderCallShape, ...], ...]:
    grouped: dict[tuple[str, tuple[str, ...]], list[BuilderCallShape]] = defaultdict(list)
    for builder in _collect_typed_family_items(module, BuilderCallShapeFamily, BuilderCallShape):
        if len(builder.keyword_names) < max(config.min_builder_keywords, 6):
            continue
        if not all(
            _is_projection_like_builder_value(value)
            for value in builder.value_fingerprint
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
            tuple(sorted(builders, key=lambda item: (item.file_path, item.lineno)))
        )
    return tuple(
        sorted(
            candidates,
            key=lambda group: (
                group[0].file_path,
                group[0].lineno,
                group[0].callee_name,
            ),
        )
    )

def _projection_helper_groups(
    module: ParsedModule,
) -> tuple[tuple[ProjectionHelperShape, ...], ...]:
    shapes: tuple[ProjectionHelperShape, ...] = _collect_typed_family_items(
        module,
        ProjectionHelperObservationFamily,
        ProjectionHelperShape,
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
            _as_projection_helper_shape(item)
            for item in _materialize_observations(fiber.observations, lookup)
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
        module,
        AccessorWrapperObservationFamily,
        AccessorWrapperCandidate,
    )
    graph = ObservationGraph(
        tuple(candidate.structural_observation for candidate in candidates)
    )
    lookup = _carrier_lookup(tuple(candidates))
    groups: list[tuple[AccessorWrapperCandidate, ...]] = []
    for witness_group in graph.witness_groups_for(
        ObservationKind.ACCESSOR_WRAPPER,
        StructuralExecutionLevel.FUNCTION_BODY,
    ):
        ordered = tuple(
            _as_accessor_wrapper_candidate(item)
            for item in _materialize_observations(
                witness_group.observations,
                lookup,
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
            name
            for name in _declared_base_names(node)
            if name not in _IGNORED_BASE_NAMES
        )
        if not base_names:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not any(
                _ast_terminal_name(decorator) == "property"
                for decorator in statement.decorator_list
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
                and returned.value.id == "self"
            ):
                continue
            for base_name in base_names:
                grouped[(base_name, statement.name, returned.attr)].append(
                    (node.name, statement.lineno)
                )
    return tuple(
        PropertyAliasHookGroup(
            file_path=str(module.path),
            base_name=base_name,
            property_name=property_name,
            returned_attribute=returned_attribute,
            class_names=tuple(class_name for class_name, _ in ordered),
            line_numbers=tuple(line for _, line in ordered),
        )
        for (base_name, property_name, returned_attribute), items in sorted(
            grouped.items()
        )
        if len(items) >= 2
        for ordered in [tuple(sorted(items, key=lambda item: (item[1], item[0])))]
    )

def _is_constant_hook_expression(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) or (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id != "self"
    )

def _class_defines_property(node: ast.ClassDef, property_name: str) -> bool:
    return any(
        isinstance(statement, ast.FunctionDef)
        and statement.name == property_name
        and any(
            _ast_terminal_name(decorator) == "property"
            for decorator in statement.decorator_list
        )
        for statement in node.body
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
            name
            for name in _declared_base_names(node)
            if name not in _IGNORED_BASE_NAMES
        )
        if not base_names:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not any(
                _ast_terminal_name(decorator) == "property"
                for decorator in statement.decorator_list
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
                grouped[(base_name, statement.name)].append(
                    (node.name, statement.lineno, return_expression)
                )
    return tuple(
        ConstantPropertyHookGroup(
            file_path=str(module.path),
            base_name=base_name,
            property_name=property_name,
            class_names=tuple(class_name for class_name, _, _ in ordered),
            line_numbers=tuple(line for _, line, _ in ordered),
            return_expressions=tuple(expression for _, _, expression in ordered),
        )
        for (base_name, property_name), items in sorted(grouped.items())
        if len(items) >= 2
        for ordered in [tuple(sorted(items, key=lambda item: (item[1], item[0])))]
    )

def _reflective_self_attribute_candidates(
    module: ParsedModule,
) -> tuple[ReflectiveSelfAttributeCandidate, ...]:
    candidates: list[ReflectiveSelfAttributeCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
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
                    and attribute_name is not None
                ):
                    continue
                candidates.append(
                    ReflectiveSelfAttributeCandidate(
                        file_path=str(module.path),
                        line=subnode.lineno,
                        subject_name=node.name,
                        name_family=(attribute_name,),
                        method_name=statement.name,
                        reflective_builtin=builtin_name,
                        attribute_name=attribute_name,
                    )
                )
    return tuple(candidates)

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
        "list",
        "max",
        "min",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
    }
)

def _is_observation_spec_wrapper_class(node: ast.ClassDef) -> bool:
    if not node.name.endswith("ObservationSpec"):
        return False
    return any(
        base_name.endswith("ObservationSpec")
        for base_name in _declared_base_names(node)
    )

def _looks_like_helper_call_name(helper_name: str) -> bool:
    terminal = helper_name.rsplit(".", 1)[-1]
    return bool(
        terminal
        and terminal[0].islower()
        and terminal not in _NON_HELPER_CALL_NAMES
    )

def _helper_call_from_returned_value(node: ast.AST) -> tuple[str, bool] | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tuple"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
    ):
        helper_name = _call_display_name(node.args[0])
        if helper_name is None or not _looks_like_helper_call_name(helper_name):
            return None
        return (helper_name, True)
    if isinstance(node, ast.Call):
        helper_name = _call_display_name(node)
        if helper_name is None or not _looks_like_helper_call_name(helper_name):
            return None
        return (helper_name, False)
    return None

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
        f"{keyword.arg}={ast.unparse(keyword.value)}"
        for keyword in helper_call.keywords
        if keyword.arg is not None
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
            and statement.value.value.id == "observation"
        )
    if isinstance(statement, ast.If):
        return _if_returns_none_only(statement)
    return False

def _helper_backed_observation_spec_candidates(
    module: ParsedModule,
) -> tuple[HelperBackedObservationSpecCandidate, ...]:
    candidates: list[HelperBackedObservationSpecCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = _shared_record_base_names(node)
        if not base_names:
            continue
        for method in node.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if method.name.startswith("_"):
                continue
            body = _trim_docstring_body(method.body)
            if not body or len(body) > 4:
                continue
            if not all(
                _is_helper_wrapper_prelude(statement) for statement in body[:-1]
            ):
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
            candidates.append(
                HelperBackedObservationSpecCandidate(
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
            )
    return tuple(candidates)

def _helper_backed_observation_spec_group(
    module: ParsedModule,
) -> HelperBackedObservationSpecGroup | None:
    candidates = _helper_backed_observation_spec_candidates(module)
    grouped: dict[tuple[str, ...], list[HelperBackedObservationSpecCandidate]] = (
        defaultdict(list)
    )
    for candidate in candidates:
        grouped[candidate.base_names].append(candidate)
    items = max(
        (items for items in grouped.values() if len(items) >= 3),
        key=len,
        default=None,
    )
    if items is None:
        return None
    ordered = tuple(sorted(items, key=lambda item: (item.line, item.class_name)))
    return HelperBackedObservationSpecGroup(
        file_path=str(module.path),
        base_names=ordered[0].base_names,
        class_names=tuple(item.class_name for item in ordered),
        line_numbers=tuple(item.line for item in ordered),
        method_names=tuple(item.method_name for item in ordered),
        helper_names=tuple(item.helper_name for item in ordered),
        wrapper_kinds=tuple(item.wrapper_kind for item in ordered),
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
            and len(body[0].targets) == 1
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

def _dynamic_self_field_selection_candidates(
    module: ParsedModule,
) -> tuple[DynamicSelfFieldSelectionCandidate, ...]:
    candidates: list[DynamicSelfFieldSelectionCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
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
                candidates.append(
                    DynamicSelfFieldSelectionCandidate(
                        file_path=str(module.path),
                        line=subnode.lineno,
                        subject_name=node.name,
                        name_family=(selector_expression,),
                        method_name=statement.name,
                        reflective_builtin=builtin_name,
                        selector_expression=selector_expression,
                    )
                )
    return tuple(candidates)

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
    return tuple(sorted(set(registry_names)))

def _registration_append_registry_name(
    node: ast.AST, registry_names: tuple[str, ...], owner_name: str
) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "append":
        return None
    if len(node.args) != 1 or not _looks_like_cls_registration_value(node.args[0]):
        return None
    target = node.func.value
    if not isinstance(target, ast.Attribute):
        return None
    if target.attr not in registry_names:
        return None
    if isinstance(target.value, ast.Name) and target.value.id in {"cls", _TYPE_NAME_LITERAL}:
        return target.attr
    if (
        isinstance(target.value, ast.Name)
        and target.value.id == owner_name
    ):
        return target.attr
    return None

def _looks_like_cls_registration_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "cls"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cast"
        and node.args
    ):
        return _looks_like_cls_registration_value(node.args[-1])
    return False

def _class_dict_get_attr_name(node: ast.AST) -> str | None:
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Attribute)
        or node.func.attr != "get"
        or len(node.args) != 1
    ):
        return None
    if not isinstance(node.func.value, ast.Attribute) or node.func.value.attr != "__dict__":
        return None
    if (
        not isinstance(node.func.value.value, ast.Name)
        or node.func.value.value.id != "cls"
    ):
        return None
    return _constant_string(node.args[0])

def _guarded_defined_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _class_dict_get_attr_name(node)
    if not isinstance(node, ast.Compare):
        return None
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    if not isinstance(node.ops[0], (ast.IsNot, ast.NotEq)):
        return None
    comparator = node.comparators[0]
    if not isinstance(comparator, ast.Constant) or comparator.value is not None:
        return None
    return _class_dict_get_attr_name(node.left)

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
                    _guard_requires_concrete_subclass(guard) for guard in guard_stack
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
            _uses_named_registry(
                subnode,
                registry_name=registry_name,
                owner_names=frozenset({"cls", _TYPE_NAME_LITERAL, node.name}),
            )
            for subnode in _walk_nodes(method)
        ):
            consumer_locations.append(
                SourceLocation(str(module.path), method.lineno, f"{node.name}.{method.name}")
            )
    for qualname, function in _iter_named_functions(module):
        if "." in qualname:
            continue
        if any(
            _uses_named_registry(
                subnode,
                registry_name=registry_name,
                owner_names=frozenset({node.name}),
            )
            for subnode in _walk_nodes(function)
        ):
            consumer_locations.append(
                SourceLocation(str(module.path), function.lineno, qualname)
            )
    unique_locations = {
        (location.file_path, location.line, location.symbol): location
        for location in consumer_locations
    }
    return tuple(
        sorted(
            unique_locations.values(),
            key=lambda location: (location.line, location.symbol),
        )
    )

def _registered_descendant_classes(
    descendants: tuple[IndexedClass, ...],
    site: _ManualSubclassRegistrationSite,
) -> tuple[IndexedClass, ...]:
    if site.selector_attr_name is not None:
        return tuple(
            descendant
            for descendant in descendants
            if site.selector_attr_name
            in _class_direct_non_none_assignment_names(descendant.node)
        )
    if site.requires_concrete_subclass:
        return tuple(
            descendant
            for descendant in descendants
            if not _is_abstract_class(descendant.node)
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
            concrete_descendants = _registered_descendant_classes(
                descendants, site
            )
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
                        concrete_descendants,
                        class_index,
                    ),
                )
            )
    return tuple(candidates)

def _registered_type_match_assignment_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str] | None:
    body = _trim_docstring_body(list(method.body))
    assignment = next(
        (
            statement
            for statement in body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.ListComp)
        ),
        None,
    )
    if assignment is None:
        return None
    list_comp = assignment.value
    if len(list_comp.generators) != 1:
        return None
    generator = list_comp.generators[0]
    if (
        generator.is_async
        or not isinstance(generator.target, ast.Name)
        or not isinstance(list_comp.elt, ast.Name)
        or list_comp.elt.id != generator.target.id
    ):
        return None
    iter_call = generator.iter
    if not (
        isinstance(iter_call, ast.Call)
        and not iter_call.args
        and not iter_call.keywords
        and isinstance(iter_call.func, ast.Attribute)
        and isinstance(iter_call.func.value, ast.Name)
        and iter_call.func.value.id == "cls"
        and iter_call.func.attr == "registered_types"
    ):
        return None
    if len(generator.ifs) != 1:
        return None
    predicate = generator.ifs[0]
    if not (
        isinstance(predicate, ast.Call)
        and len(predicate.args) == 1
        and not predicate.keywords
        and isinstance(predicate.func, ast.Attribute)
        and isinstance(predicate.func.value, ast.Name)
        and predicate.func.value.id == generator.target.id
        and isinstance(predicate.args[0], ast.Name)
        and predicate.args[0].id in _parameter_names(method)
    ):
        return None
    return (
        assignment.targets[0].id,
        predicate.func.attr,
        predicate.args[0].id,
    )

def _is_selected_match_subscript(node: ast.AST, match_var_name: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == match_var_name
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == 0
    )

def _selection_guard_kind(node: ast.AST, match_var_name: str) -> str | None:
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name)
        and node.operand.id == match_var_name
    ):
        return "empty"
    if not isinstance(node, ast.Compare):
        return None
    if (
        not isinstance(node.left, ast.Call)
        or _ast_terminal_name(node.left.func) != "len"
        or len(node.left.args) != 1
        or not isinstance(node.left.args[0], ast.Name)
        or node.left.args[0].id != match_var_name
        or len(node.ops) != 1
        or len(node.comparators) != 1
        or not isinstance(node.comparators[0], ast.Constant)
        or not isinstance(node.comparators[0].value, int)
    ):
        return None
    comparator_value = node.comparators[0].value
    operator = node.ops[0]
    if isinstance(operator, ast.NotEq) and comparator_value == 1:
        return "not_exactly_one"
    if isinstance(operator, ast.Gt) and comparator_value == 1:
        return "ambiguous"
    if isinstance(operator, ast.Eq) and comparator_value == 0:
        return "empty"
    return None

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
            descendant
            for descendant in _indexed_descendant_classes(
                class_index, indexed_class.symbol
            )
            if not _is_abstract_class(descendant.node)
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
                _is_selected_match_subscript(subnode, match_var_name)
                for subnode in _walk_nodes(method)
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
                        descendants,
                        class_index,
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
    root_candidates: list[tuple[IndexedClass, tuple[str, ...], tuple[IndexedClass, ...]]] = []
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
            descendant
            for descendant in _indexed_descendant_classes(
                class_index, indexed_class.symbol
            )
            if not _is_abstract_class(descendant.node)
        )
        if len(concrete_descendants) < min_shared_families:
            continue
        root_candidates.append(
            (indexed_class, abstract_methods, concrete_descendants)
        )

    candidates: list[ParallelMirroredLeafFamilyCandidate] = []
    for (
        left_root,
        left_contract_methods,
        left_descendants,
    ), (
        right_root,
        right_contract_methods,
        right_descendants,
    ) in combinations(root_candidates, 2):
        shared_contract_methods = tuple(
            sorted(set(left_contract_methods) & set(right_contract_methods))
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
            left_descendants,
            axis_prefix_tokens=left_axis_prefix,
        )
        right_leaf_map = _mirrored_leaf_family_map(
            right_descendants,
            axis_prefix_tokens=right_axis_prefix,
        )
        if not left_leaf_map or not right_leaf_map:
            continue
        shared_leaf_families = tuple(
            sorted(set(left_leaf_map) & set(right_leaf_map))
        )
        if len(shared_leaf_families) < max(
            min_shared_families,
            min(len(left_leaf_map), len(right_leaf_map)) // 2,
        ):
            continue
        left_leaf_evidence = tuple(
            SourceLocation(
                left_leaf_map[family_name].file_path,
                left_leaf_map[family_name].line,
                _indexed_class_display_name(left_leaf_map[family_name], class_index),
            )
            for family_name in shared_leaf_families
        )
        right_leaf_evidence = tuple(
            SourceLocation(
                right_leaf_map[family_name].file_path,
                right_leaf_map[family_name].line,
                _indexed_class_display_name(right_leaf_map[family_name], class_index),
            )
            for family_name in shared_leaf_families
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
        and node.value.func.id in {"globals", "locals"}
        and not node.value.args
        and not node.value.keywords
        and _constant_string(node.slice) is None
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
                    descendant
                    for descendant in descendants
                    if selector_attr_name in class_string_assignments[descendant]
                )
                if (
                    len(concrete_class_names)
                    < config.min_reflective_selector_values
                ):
                    continue
                selector_values = tuple(
                    sorted(
                        {
                            class_string_assignments[descendant][selector_attr_name]
                            for descendant in concrete_class_names
                        }
                    )
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
    return tuple(
        sorted(
            candidate_map.values(),
            key=lambda item: (item.file_path, item.line, item.class_name, item.method_name),
        )
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

def _receiver_self_attr_name(
    node: ast.AST, aliases: dict[str, str]
) -> str | None:
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
        class_name: {
            field_name
            for field_name, _ in _typed_field_map(node)
        }
        for class_name, node in class_defs_by_name.items()
    }
    candidates: list[ConcreteConfigFieldProbeCandidate] = []
    for class_name, node in sorted(class_defs_by_name.items()):
        concrete_config_attrs = _class_init_concrete_param_backed_attrs(node)
        if not concrete_config_attrs:
            continue
        for method in _iter_class_methods(node):
            aliases = _method_aliases_to_self_attrs(method)
            grouped_missing_fields: dict[
                tuple[str, str], set[str]
            ] = defaultdict(set)
            grouped_probe_builtins: dict[
                tuple[str, str], set[str]
            ] = defaultdict(set)
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
                if config_node is None or _class_method_named(config_node, "__getattr__") is not None:
                    continue
                declared_field_names = config_field_names.get(config_type_name, set())
                if not declared_field_names or probed_field_name in declared_field_names:
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
                        line=grouped_lines[(config_attr_name, config_type_name)],
                        class_name=class_name,
                        method_name=method.name,
                        config_attr_name=config_attr_name,
                        config_type_name=config_type_name,
                        missing_field_names=tuple(sorted(missing_fields)),
                        probe_builtin_names=tuple(
                            sorted(grouped_probe_builtins[(config_attr_name, config_type_name)])
                        ),
                    )
                )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.file_path, item.line, item.class_name, item.method_name),
        )
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
            item.value
            for item in value.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )
        if len(string_items) != len(value.elts) or len(string_items) < 3:
            continue
        assignments.append((target_name, statement.lineno, string_items))
    return tuple(assignments)

def _is_simple_classvar_value(node: ast.AST) -> bool:
    if isinstance(node, (ast.Name, ast.Attribute, ast.Constant)):
        return True
    if isinstance(node, ast.Tuple):
        return all(_is_simple_classvar_value(item) for item in node.elts)
    return False

def _classvar_assignment_names(node: ast.ClassDef) -> tuple[str, ...] | None:
    assigned_names: list[str] = []
    for statement in _trim_docstring_body(node.body):
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if not isinstance(target, ast.Name) or not _is_simple_classvar_value(
                statement.value
            ):
                return None
            assigned_names.append(target.id)
            continue
        if isinstance(statement, ast.AnnAssign):
            if not isinstance(statement.target, ast.Name) or statement.value is None:
                return None
            if not _is_simple_classvar_value(statement.value):
                return None
            assigned_names.append(statement.target.id)
            continue
        return None
    return tuple(assigned_names)

def _classvar_only_sibling_leaf_candidates(
    module: ParsedModule,
) -> tuple[DeclarativeFamilyLeafCandidate, ...]:
    candidates: list[DeclarativeFamilyLeafCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = tuple(
            name
            for name in _declared_base_names(node)
            if name not in _IGNORED_ANCESTOR_NAMES
        )
        if not base_names:
            continue
        assigned_names = _classvar_assignment_names(node)
        if assigned_names is None:
            continue
        if len(assigned_names) < 1 or len(assigned_names) > 4:
            continue
        if len(_trim_docstring_body(node.body)) != len(assigned_names):
            continue
        candidates.append(
            DeclarativeFamilyLeafCandidate(
                file_path=str(module.path),
                line=node.lineno,
                subject_name=node.name,
                name_family=assigned_names,
                base_names=base_names,
                assigned_names=assigned_names,
            )
        )
    return tuple(candidates)

def _classvar_only_sibling_leaf_groups(
    module: ParsedModule,
) -> tuple[DeclarativeFamilyBoilerplateGroup, ...]:
    grouped: dict[
        tuple[tuple[str, ...], tuple[str, ...]],
        list[DeclarativeFamilyLeafCandidate],
    ] = defaultdict(list)
    for candidate in _classvar_only_sibling_leaf_candidates(module):
        grouped[(candidate.base_names, candidate.assigned_names)].append(candidate)
    return tuple(
        DeclarativeFamilyBoilerplateGroup(
            file_path=str(module.path),
            base_names=base_names,
            assigned_names=assigned_names,
            class_names=tuple(item.subject_name for item in items),
            line_numbers=tuple(item.line for item in items),
        )
        for (base_names, assigned_names), items in sorted(grouped.items())
        if len(items) >= 3
    )

def _type_indexed_definition_boilerplate_groups(
    module: ParsedModule,
) -> tuple[TypeIndexedDefinitionBoilerplateGroup, ...]:
    alias_assignments = _module_alias_assignments(module)
    grouped: dict[
        tuple[tuple[str, ...], tuple[str, ...]],
        list[tuple[str, str, int]],
    ] = defaultdict(list)
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef) or not node.name.endswith("Definition"):
            continue
        base_names = tuple(
            name
            for name in _declared_base_names(node)
            if name not in _IGNORED_ANCESTOR_NAMES
        )
        if not base_names or not any(
            name.endswith("Definition") for name in base_names
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
        grouped[(base_names, assigned_names)].append(
            (node.name, alias_name, node.lineno)
        )
    return tuple(
        TypeIndexedDefinitionBoilerplateGroup(
            file_path=str(module.path),
            base_names=base_names,
            definition_class_names=tuple(item[0] for item in ordered),
            alias_names=tuple(item[1] for item in ordered),
            line_numbers=tuple(item[2] for item in ordered),
            assigned_names=assigned_names,
        )
        for (base_names, assigned_names), items in sorted(grouped.items())
        if len(items) >= 3
        for ordered in [tuple(sorted(items, key=lambda item: (item[2], item[0])))]
    )

def _derivable_nominal_root_names(
    shapes: Sequence[NominalAuthorityShape],
) -> tuple[str, ...]:
    root_counts: Counter[str] = Counter()
    for shape in shapes:
        root_counts.update(
            name
            for name in {*shape.declared_base_names, *shape.ancestor_names}
            if name not in _IGNORED_ANCESTOR_NAMES and name != shape.class_name
        )
    return tuple(
        sorted(root_name for root_name, count in root_counts.items() if count >= 3)
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
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                names.add(target.id)
    return tuple(sorted(names))

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
            item.id
            for item in value.values
            if isinstance(item, ast.Name)
            and (shapes := index.shapes_named(item.id))
            and shapes[0].file_path == str(module.path)
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
        if isinstance(current, ast.BinOp) and isinstance(current.op, ast.Add):
            return collect_calls(current.left) and collect_calls(current.right)
        if isinstance(current, ast.Call):
            calls.append(current)
            return True
        return False

    if not collect_calls(node) or len(calls) < 2:
        return None
    accessor_names = {
        call.func.attr
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and not call.args
        and not call.keywords
    }
    if len(accessor_names) != 1:
        return None
    accessor_name = next(iter(accessor_names))
    root_names = tuple(
        sorted(
            call.func.value.id
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
        )
    )
    return (accessor_name, root_names)

def _registered_union_surface_candidates(
    module: ParsedModule,
) -> tuple[RegisteredUnionSurfaceCandidate, ...]:
    candidates: list[RegisteredUnionSurfaceCandidate] = []
    class_defs_by_name = {
        node.name: node
        for node in module.module.body
        if isinstance(node, ast.ClassDef)
    }
    for node in _walk_nodes(module.module):
        owner_name = "<module>"
        value: ast.AST | None = None
        line = getattr(node, "lineno", 1)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner_name = node.name
            for statement in _trim_docstring_body(node.body):
                if isinstance(statement, ast.For):
                    value = statement.iter
                    line = statement.lineno
                    break
                if isinstance(statement, ast.Assign):
                    value = statement.value
                    line = statement.lineno
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                owner_name = target.id
                value = node.value
                line = node.lineno
        if value is None:
            continue
        registered_surface = _registered_surface_roots(value)
        if registered_surface is None:
            continue
        accessor_name, root_names = registered_surface
        if len(root_names) < 2:
            continue
        root_nodes = [class_defs_by_name.get(root_name) for root_name in root_names]
        if any(root_node is None for root_node in root_nodes):
            continue
        if any(
            (method := _class_method_named(cast(ast.ClassDef, root_node), accessor_name))
            is None
            or not _is_classmethod(method)
            for root_node in root_nodes
        ):
            continue
        candidates.append(
            RegisteredUnionSurfaceCandidate(
                file_path=str(module.path),
                line=line,
                owner_name=owner_name,
                accessor_name=accessor_name,
                root_names=root_names,
            )
        )
    return tuple(candidates)

def _type_name_set(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (ast.unparse(node),)
    if isinstance(node, ast.Tuple):
        return tuple(
            sorted(
                {
                    type_name
                    for element in node.elts
                    for type_name in _type_name_set(element)
                }
            )
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
            if any(type_name.endswith("Enum") for type_name in type_names):
                roles.add("enum_ok")
        elif call_name == "callable" and len(current.args) == 1:
            roles.add("callable_ok")
        elif call_name == "issubclass" and len(current.args) == 2:
            roles.add("subclass_constraint")
            type_names = set(_type_name_set(current.args[1]))
            if any(type_name.endswith("Enum") for type_name in type_names):
                roles.add("enum_ok")
        elif call_name == "isabstract":
            roles.add("exclude_abstract")
    return tuple(sorted(roles))

def _export_policy_root_type_names(node: ast.FunctionDef) -> tuple[str, ...]:
    root_type_names: set[str] = set()
    for current in _walk_nodes(node):
        if not isinstance(current, ast.Call):
            continue
        if _ast_terminal_name(current.func) != "issubclass" or len(current.args) != 2:
            continue
        root_type_names.update(
            type_name
            for type_name in _type_name_set(current.args[1])
            if type_name != _TYPE_NAME_LITERAL
        )
    return tuple(sorted(root_type_names))

def _module_export_policy_predicate_candidate(
    module: ParsedModule,
) -> ExportPolicyPredicateCandidate | None:
    exported_predicate_names: set[str] = set()
    for statement in _trim_docstring_body(module.module.body):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not (isinstance(target, ast.Name) and target.id == "__all__"):
            continue
        value = statement.value
        if (
            not isinstance(value, ast.Call)
            or _ast_terminal_name(value.func) != "sorted"
        ):
            continue
        if len(value.args) != 1 or not isinstance(value.args[0], ast.GeneratorExp):
            continue
        generator = value.args[0]
        if not generator.generators or len(generator.generators[0].ifs) != 1:
            continue
        condition = generator.generators[0].ifs[0]
        if not isinstance(condition, ast.Call) or not isinstance(
            condition.func, ast.Name
        ):
            continue
        exported_predicate_names.add(condition.func.id)
    if len(exported_predicate_names) != 1:
        return None
    predicate_name = next(iter(exported_predicate_names))
    predicate_node = next(
        (
            statement
            for statement in _trim_docstring_body(module.module.body)
            if isinstance(statement, ast.FunctionDef)
            and statement.name == predicate_name
        ),
        None,
    )
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
                and inner.func.id == "sorted"
                and inner.args
                and isinstance(inner.args[0], ast.Name)
            ):
                return inner.args[0].id
    return None

def _subclasses_root_expression(node: ast.AST) -> str | None:
    subclasses_call: ast.Call | None = None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
    ):
        subclasses_call = node.args[0]
    elif isinstance(node, ast.Call):
        subclasses_call = node
    if subclasses_call is None:
        return None
    if (
        not isinstance(subclasses_call.func, ast.Attribute)
        or subclasses_call.func.attr != "__subclasses__"
        or subclasses_call.args
    ):
        return None
    return ast.unparse(subclasses_call.func.value)

def _subclass_traversal_seed(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str] | None:
    for statement in _trim_docstring_body(node.body):
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            continue
        if (root_expression := _subclasses_root_expression(statement.value)) is None:
            continue
        return statement.targets[0].id, root_expression
    return None

def _queue_pop_target_name(statement: ast.stmt, queue_name: str) -> str | None:
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
        or not isinstance(statement.value, ast.Call)
        or not isinstance(statement.value.func, ast.Attribute)
        or statement.value.func.attr != "pop"
        or not isinstance(statement.value.func.value, ast.Name)
        or statement.value.func.value.id != queue_name
        or len(statement.value.args) != 1
    ):
        return None
    pop_index = statement.value.args[0]
    if not (isinstance(pop_index, ast.Constant) and pop_index.value == 0):
        return None
    return statement.targets[0].id

def _extends_subclasses_queue(
    statement: ast.stmt, queue_name: str, current_name: str
) -> bool:
    if (
        not isinstance(statement, ast.Expr)
        or not isinstance(statement.value, ast.Call)
        or not isinstance(statement.value.func, ast.Attribute)
        or statement.value.func.attr != "extend"
        or not isinstance(statement.value.func.value, ast.Name)
        or statement.value.func.value.id != queue_name
        or len(statement.value.args) != 1
    ):
        return False
    return _subclasses_root_expression(statement.value.args[0]) == current_name

def _result_append_args(
    node: ast.FunctionDef | ast.AsyncFunctionDef, result_name: str
) -> tuple[ast.AST, ...]:
    return tuple(
        current.args[0]
        for current in _walk_nodes(node)
        if isinstance(current, ast.Call)
        and isinstance(current.func, ast.Attribute)
        and current.func.attr == "append"
        and isinstance(current.func.value, ast.Name)
        and current.func.value.id == result_name
        and len(current.args) == 1
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
    return tuple(
        sorted(
            {
                attribute_name
                for current in _walk_nodes(node)
                if isinstance(current, ast.Call)
                and isinstance(current.func, ast.Attribute)
                and current.func.attr == "get"
                and isinstance(current.func.value, ast.Attribute)
                and current.func.value.attr == "__dict__"
                and isinstance(current.func.value.value, ast.Name)
                and len(current.args) == 1
                and (attribute_name := _constant_string(current.args[0])) is not None
            }
        )
    )

def _subclass_traversal_filter_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    current_name: str,
) -> tuple[str, ...]:
    filter_names: set[str] = set()
    for current in _walk_nodes(node):
        if not isinstance(current, ast.Call):
            continue
        if (
            isinstance(current.func, ast.Name)
            and any(
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
            and current.func.value.attr == "__dict__"
            and isinstance(current.func.value.value, ast.Name)
            and current.func.value.value.id == current_name
            and len(current.args) == 1
            and (attribute_name := _constant_string(current.args[0])) is not None
        ):
            filter_names.add(attribute_name)
    return tuple(sorted(filter_names))

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
    if result_name is None:
        return None
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
    materialization_kind = _registry_materialization_kind(node, result_name)
    if materialization_kind is None:
        return None
    append_args = _result_append_args(node, result_name)
    if not append_args:
        return None
    return SubclassTraversalSite(
        file_path=str(module.path),
        line=node.lineno,
        symbol=qualname,
        root_expression=root_expression,
        materialization_kind=materialization_kind,
        registry_attribute_names=_registry_attribute_names(node),
        filter_names=_subclass_traversal_filter_names(node, current_name),
    )

def _registry_traversal_group(
    modules: Sequence[ParsedModule],
) -> SubclassTraversalGroup | None:
    sites = tuple(
        sorted(
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
    )
    if len(sites) < 2:
        return None
    return SubclassTraversalGroup(
        symbols=tuple(site.symbol for site in sites),
        file_paths=tuple(site.file_path for site in sites),
        line_numbers=tuple(site.line for site in sites),
        root_expressions=tuple(site.root_expression for site in sites),
        materialization_kinds=tuple(site.materialization_kind for site in sites),
        registry_attribute_names=tuple(
            sorted(
                {
                    attribute_name
                    for site in sites
                    for attribute_name in site.registry_attribute_names
                }
            )
        ),
        filter_names=tuple(
            sorted(
                {
                    filter_name
                    for site in sites
                    for filter_name in site.filter_names
                }
            )
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
        shared_keyword_names = tuple(
            sorted(str(item) for item in set.intersection(*keyword_sets))
        )
        if len(shared_keyword_names) < 4:
            continue
        groups.append(
            AlternateConstructorFamilyGroup(
                file_path=str(module.path),
                class_name=node.name,
                method_names=tuple(method.name for method, _, _ in constructor_methods),
                line_numbers=tuple(
                    method.lineno for method, _, _ in constructor_methods
                ),
                keyword_names=shared_keyword_names,
                source_type_names=tuple(
                    source_type_name for _, _, source_type_name in constructor_methods
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
            f"{class_name}.{property_name}" for class_name in class_names
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

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.function_stack: list[str] = []

        def visit_Module(self, node: ast.Module) -> None:
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_stack.append(node.name)
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.function_stack.append(node.name)
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            for stmt in body:
                self.visit(stmt)
            self.function_stack.pop()

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
    return any(_looks_like_semantic_subject(candidate) for candidate in candidates)

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
                and target.value.id == "self"
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
    return tuple(method for group in groups for method in group)

def _group_repeated_methods(
    modules: list[ParsedModule], config: DetectorConfig
) -> list[tuple[MethodShape, ...]]:
    methods = tuple(
        method
        for module in modules
        for method in _collect_typed_family_items(
            module, MethodShapeFamily, MethodShape
        )
        if method.class_name
        and method.statement_count >= config.min_duplicate_statements
    )
    groups = _fiber_grouped_shapes(
        modules,
        tuple(methods),
        ObservationKind.METHOD_SHAPE,
        StructuralExecutionLevel.FUNCTION_BODY,
    )
    return [
        tuple(_as_method_shape(method) for method in group)
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
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        f"@@\n"
        f"+class {base_name}(ABC):\n"
        f"+    def run(self, request):\n"
        f"+        normalized = self._normalize(request)\n"
        f"+        return self.{hook_name}(normalized)\n"
        f"+\n"
        f"+    @abstractmethod\n"
        f"+    def {hook_name}(self, normalized): ...\n"
        "*** End Patch"
    )

def _abc_family_patch(
    class_names: frozenset[str], groups: list[tuple[MethodShape, ...]]
) -> str:
    ordered = sorted(class_names)
    target_file = groups[0][0].file_path
    base_name = _shared_family_name(ordered) or "FamilyBase"
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        f"+class {base_name}(ABC):\n"
        "+    def run(self, request): ...\n"
        "+\n"
        "+    @abstractmethod\n"
        "+    def hook(self, request): ...\n"
        "*** End Patch"
    )

def _builder_patch(builders: tuple[BuilderCallShape, ...]) -> str:
    target_file = builders[0].file_path
    callee_name = builders[0].callee_name
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        f"+@classmethod\n"
        f"+def from_source(cls, source):\n"
        f"+    return {callee_name}(...)\n"
        "*** End Patch"
    )

def _single_owner_builder_family_patch(owner_symbol: str, callee_name: str) -> str:
    return (
        f"# Replace the repeated `{callee_name}` calls inside `{owner_symbol}` with one declarative invocation table.\n"
        "# Keep the builder authority in one row family and materialize the calls in one loop."
    )

def _projection_schema_patch(export_shapes: tuple[ExportDictShape, ...]) -> str:
    target_file = export_shapes[0].file_path
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        "+@dataclass(frozen=True)\n"
        "+class ProjectionSchema:\n"
        "+    ...\n"
        "+\n"
        "+    @classmethod\n"
        "+    def from_source(cls, source): ...\n"
        "*** End Patch"
    )

def _autoregister_patch(
    registry_name: str,
    class_names: set[str],
    registrations: tuple[RegistrationShape, ...],
) -> str:
    target_file = registrations[0].file_path
    base_name = _shared_family_name(sorted(class_names)) or "RegisteredBase"
    ordered_class_names = tuple(sorted(class_names))
    key_values = tuple(
        key_value
        for class_name in ordered_class_names
        if (
            key_value := _string_constant_expression(
                next(
                    registration.key_expression
                    for registration in registrations
                    if registration.registered_class == class_name
                )
            )
        )
        is not None
    )
    use_extractor = len(key_values) == len(ordered_class_names) and (
        _derivable_registry_key_suffix(ordered_class_names, key_values) is not None
    )
    config_block = (
        _derived_registry_key_block(ordered_class_names)
        if use_extractor
        else _declared_registry_key_block("registry_key")
    )
    return (
        "*** Begin Patch\n"
        f"*** Update File: {target_file}\n"
        "@@\n"
        + (
            "+from metaclass_registry import AutoRegisterMeta\n"
            + ("+import re\n" if use_extractor else "")
            + "+\n"
            + f"+class {base_name}(ABC, metaclass=AutoRegisterMeta):\n"
            + "".join(f"+{line}\n" for line in config_block.splitlines())
            + "+\n"
            + f"+# Replace `{registry_name}` with `{base_name}.__registry__`.\n"
            + "*** End Patch"
        )
    )

def _abc_scaffold_for_methods(methods: tuple[MethodShape, ...]) -> str:
    class_names = sorted(
        {method.class_name for method in methods if method.class_name is not None}
    )
    hook_names = sorted({method.method_name for method in methods})
    base_name = _shared_family_name(class_names) or "ExtractedBase"
    hook_name = hook_names[0] if hook_names else "hook"
    return (
        f"class {base_name}(ABC):\n"
        f"    def run(self, request):\n"
        f"        normalized = self._normalize(request)\n"
        f"        return self.{hook_name}(normalized)\n\n"
        f"    @abstractmethod\n"
        f"    def {hook_name}(self, normalized): ..."
    )

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
        f"    @abstractmethod\n    def {name}(self, request): ..."
        for name in hook_methods[:3]
    )
    subclass_block = "\n".join(
        f"class {name}({base_name}):\n    ..." for name in ordered[:3]
    )
    return f"class {base_name}(ABC):\n    def run(self, request): ...\n{hook_block}\n\n{subclass_block}"

def _builder_scaffold(builders: tuple[BuilderCallShape, ...]) -> str:
    callee_name = builders[0].callee_name
    keywords = builders[0].keyword_names
    row_name = callee_name if callee_name[:1].isupper() else "ProjectedRow"
    args_block = "\n".join(
        f"            {name}=source.{name}," for name in keywords[:4]
    )
    return (
        f"@dataclass(frozen=True)\n"
        f"class {row_name}:\n"
        f"    ...\n\n"
        f"    @classmethod\n"
        f"    def from_source(cls, source):\n"
        f"        return cls(\n{args_block}\n        )"
    )

def _single_owner_builder_family_scaffold(callee_name: str) -> str:
    return (
        "@dataclass(frozen=True)\n"
        "class InvocationSpec:\n"
        "    args: tuple[object, ...]\n"
        "    kwargs: dict[str, object]\n\n"
        "INVOCATION_SPECS = (\n"
        "    InvocationSpec(args=(...), kwargs={\"flag\": True}),\n"
        ")\n\n"
        f"for spec in INVOCATION_SPECS:\n"
        f"    owner.{callee_name}(*spec.args, **spec.kwargs)"
    )

def _projection_schema_scaffold(export_shapes: tuple[ExportDictShape, ...]) -> str:
    keys = export_shapes[0].key_names
    field_block = "\n".join(f"    {key}: object" for key in keys[:4])
    mapping_block = "\n".join(f"            {key}=source.{key}," for key in keys[:4])
    return (
        "@dataclass(frozen=True)\n"
        "class ProjectionSchema:\n"
        f"{field_block}\n\n"
        "    @classmethod\n"
        "    def from_source(cls, source):\n"
        f"        return cls(\n{mapping_block}\n        )"
    )

def _autoregister_scaffold(registry_name: str, class_names: set[str]) -> str:
    base_name = _shared_family_name(sorted(class_names)) or "RegisteredBase"
    sample = sorted(class_names)[:2]
    config_block = _derived_registry_key_block(sample)
    subclass_block = "\n".join(f"class {name}({base_name}):\n    ..." for name in sample)
    return (
        "from abc import ABC\n"
        "import re\n"
        "from metaclass_registry import AutoRegisterMeta\n\n"
        f"class {base_name}(ABC, metaclass=AutoRegisterMeta):\n"
        f"{config_block}\n\n"
        f"{subclass_block}"
    )

def _shared_family_name(class_names: list[str]) -> str | None:
    if not class_names:
        return None
    prefix = class_names[0]
    for name in class_names[1:]:
        while prefix and not name.startswith(prefix):
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

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.function_depth += 1
            self.generic_visit(node)
            self.function_depth -= 1

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

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
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
    if all(isinstance(value, ast.Constant) for value in node.values):
        return False
    return any(
        isinstance(value, (ast.Name, ast.Attribute, ast.Lambda, ast.Call))
        for value in node.values
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
                isinstance(value, ast.Attribute) and value.attr == attr_name
                for value in values
            )
            literal_match = any(
                isinstance(value, ast.Constant)
                and isinstance(value.value, (str, int, bool))
                for value in values
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
    return (
        "def _render_projection(items, projector):\n"
        "    return tuple(_dedupe_preserve_order(projector(item) for item in items))\n\n"
        f"# Replace {function_names} with `_render_projection(..., lambda item: item.<field>)`.\n"
        f"# Projected fields: {attributes}"
    )

def _supports_accessor_wrapper_finding(
    candidates: Sequence[AccessorWrapperCandidate],
) -> bool:
    if not candidates:
        return False
    if any(candidate.wrapper_shape.startswith("computed_") for candidate in candidates):
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

def _is_transport_expression(
    node: ast.AST,
    *,
    allowed_roots: set[str],
) -> bool:
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
        and node.value.id in {"self", "cls"}
        and class_name is not None
    ):
        return f"{class_name}.{node.attr}"
    return None

def _projected_attribute_names(
    node: ast.AST,
    *,
    bound_name: str,
) -> tuple[str, ...] | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == bound_name:
            return (node.attr,)
        return None
    if isinstance(node, ast.Tuple):
        projected: list[str] = []
        for item in node.elts:
            if not isinstance(item, ast.Attribute) or not isinstance(item.value, ast.Name):
                return None
            if item.value.id != bound_name:
                return None
            projected.append(item.attr)
        return tuple(projected)
    return None

def _call_chain_from_outer_call(call: ast.Call) -> tuple[ast.Call, ...]:
    chain = [call]
    current = call
    while (
        isinstance(current.func, ast.Attribute)
        and isinstance(current.func.value, ast.Call)
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
    return "." not in qualname and not qualname.startswith("_")

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

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.class_stack: list[str] = []
                self.function_stack: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_Call(self, node: ast.Call) -> None:
                for target in _resolved_import_call_target_symbols(
                    module,
                    node.func,
                    import_aliases=import_aliases,
                ):
                    callsites_by_target[target].add(
                        ResolvedExternalCallsite(
                            module_name=module.module_name,
                            location=SourceLocation(
                                str(module.path),
                                node.lineno,
                                self._symbol("call"),
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
        target: tuple(
            sorted(
                callsites,
                key=lambda item: (
                    item.location.file_path,
                    item.location.line,
                    item.location.symbol,
                    item.module_name,
                ),
            )
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
    return tuple(
        sorted(
            matched,
            key=lambda item: (
                item.location.file_path,
                item.location.line,
                item.location.symbol,
                item.module_name,
            ),
        )
    )

def _trivial_forwarding_wrapper_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> TrivialForwardingWrapperCandidate | None:
    if function.name.startswith("__") and function.name.endswith("__"):
        return None
    body = _trim_docstring_body(function.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    returned = body[0].value
    if not isinstance(returned, ast.Call):
        return None
    chain = _call_chain_from_outer_call(returned)
    if len(chain) < 2:
        return None
    class_name = qualname.rsplit(".", 1)[0] if "." in qualname else None
    allowed_roots = _function_param_names(function) | {"self", "cls"}
    values = _call_chain_transport_values(chain)
    if not values:
        return None
    if not all(_is_transport_expression(value, allowed_roots=allowed_roots) for value in values):
        return None
    transported_value_sources = tuple(sorted({ast.unparse(value) for value in values}))
    parameter_names = _function_param_names(function) - {"self", "cls"}
    forwarded_parameter_names = tuple(
        sorted(
            {
                node.id
                for value in values
                for node in _walk_nodes(value)
                if isinstance(node, ast.Name) and node.id in parameter_names
            }
        )
    )
    if not transported_value_sources:
        return None
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
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (candidate.file_path, candidate.line, candidate.qualname),
        )
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
                site
                for site in _matching_external_callsites(
                    callsites_by_target,
                    target_symbol=wrapper_symbol,
                )
                if site.module_name != module.module_name
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
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.wrapper.file_path,
                item.wrapper.line,
                item.wrapper.qualname,
            ),
        )
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
    for (file_path, module_name, delegate_root_symbol), wrappers in grouped_wrappers.items():
        if len(wrappers) < min_wrapper_count:
            continue
        external_callsites = tuple(
            sorted(
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
        )
        if len(external_callsites) < min_external_callsites:
            continue
        candidates.append(
            PublicApiPrivateDelegateFamilyCandidate(
                file_path=file_path,
                module_name=module_name,
                delegate_root_symbol=delegate_root_symbol,
                delegate_root_line=delegate_lines[(file_path, module_name, delegate_root_symbol)],
                wrappers=tuple(sorted(wrappers, key=lambda item: (item.line, item.qualname))),
                external_callsites=external_callsites,
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.delegate_root_symbol,
                item.wrappers[0].line,
            ),
        )
    )

def _policy_selector_source_exprs(
    selector_call: ast.Call,
) -> tuple[str, ...]:
    return tuple(
        ast.unparse(value)
        for value in (
            *selector_call.args,
            *(keyword.value for keyword in selector_call.keywords if keyword.arg is not None),
        )
    )

def _looks_like_self_selector_source(expr: str) -> bool:
    return expr == "self" or expr.startswith("self.") or expr == "cls" or expr.startswith("cls.")

def _nominal_policy_surface_method_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> NominalPolicySurfaceMethodCandidate | None:
    if "." not in qualname:
        return None
    owner_class_name, method_name = qualname.rsplit(".", 1)
    if method_name.startswith("_") or (method_name.startswith("__") and method_name.endswith("__")):
        return None
    body = _trim_docstring_body(function.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    returned = body[0].value
    if not isinstance(returned, ast.Call):
        return None
    chain = _call_chain_from_outer_call(returned)
    if len(chain) != 2:
        return None
    outer_call, selector_call = chain
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
        _looks_like_self_selector_source(expr) for expr in selector_source_exprs
    ):
        return None
    allowed_roots = _function_param_names(function) | {"self", "cls"}
    transported_values = _call_chain_transport_values(chain)
    if not transported_values:
        return None
    if not all(
        _is_transport_expression(value, allowed_roots=allowed_roots)
        for value in transported_values
    ):
        return None
    policy_member_name = _call_name(outer_call.func) or ast.unparse(outer_call.func)
    return NominalPolicySurfaceMethodCandidate(
        file_path=str(module.path),
        line=function.lineno,
        qualname=qualname,
        owner_class_name=owner_class_name,
        method_name=method_name,
        policy_root_symbol=".".join(policy_root_parts),
        selector_method_name=selector_method_name,
        policy_member_name=policy_member_name,
        selector_source_exprs=selector_source_exprs,
        transported_value_sources=tuple(sorted({ast.unparse(value) for value in transported_values})),
    )

def _nominal_policy_surface_family_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[NominalPolicySurfaceFamilyCandidate, ...]:
    min_family_size = max(2, config.min_registration_sites)
    method_candidates = tuple(
        candidate
        for qualname, function in _iter_named_functions(module)
        for candidate in (
            _nominal_policy_surface_method_candidate(module, qualname, function),
        )
        if candidate is not None
    )
    grouped: dict[
        tuple[str, str, str, tuple[str, ...]], list[NominalPolicySurfaceMethodCandidate]
    ] = defaultdict(list)
    for candidate in method_candidates:
        grouped[
            (
                candidate.owner_class_name,
                candidate.policy_root_symbol,
                candidate.selector_method_name,
                candidate.selector_source_exprs,
            )
        ].append(candidate)
    return tuple(
        sorted(
            (
                NominalPolicySurfaceFamilyCandidate(
                    methods=tuple(
                        sorted(candidates, key=lambda item: (item.line, item.qualname))
                    ),
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
    )

def _function_wrapper_candidate(
    module: ParsedModule,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> FunctionWrapperCandidate | None:
    body = _trim_docstring_body(function.body)
    if not body:
        return None
    class_name = qualname.rsplit(".", 1)[0] if "." in qualname else None
    allowed_roots = _function_param_names(function) | {"self", "cls"}

    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
        returned = body[0].value
        if not isinstance(returned, ast.Call):
            return None
        delegate_symbol = _wrapper_delegate_symbol(
            returned.func,
            class_name=class_name,
        )
        if delegate_symbol is None:
            return None
        values = list(returned.args) + [
            keyword.value for keyword in returned.keywords if keyword.arg is not None
        ]
        if not all(
            _is_transport_expression(value, allowed_roots=allowed_roots)
            for value in values
        ):
            return None
        return FunctionWrapperCandidate(
            file_path=str(module.path),
            qualname=qualname,
            lineno=function.lineno,
            delegate_symbol=delegate_symbol,
            wrapper_kind="direct",
            statement_count=len(body),
        )

    if (
        len(body) == 2
        and isinstance(body[0], ast.Assign)
        and len(body[0].targets) == 1
        and isinstance(body[0].targets[0], ast.Name)
        and isinstance(body[1], ast.Return)
        and body[1].value is not None
        and isinstance(body[0].value, ast.Call)
    ):
        bound_name = body[0].targets[0].id
        delegate_symbol = _wrapper_delegate_symbol(
            body[0].value.func,
            class_name=class_name,
        )
        if delegate_symbol is None:
            return None
        values = list(body[0].value.args) + [
            keyword.value
            for keyword in body[0].value.keywords
            if keyword.arg is not None
        ]
        if not all(
            _is_transport_expression(value, allowed_roots=allowed_roots)
            for value in values
        ):
            return None
        projected_attributes = _projected_attribute_names(
            body[1].value,
            bound_name=bound_name,
        )
        if projected_attributes is None:
            return None
        return FunctionWrapperCandidate(
            file_path=str(module.path),
            qualname=qualname,
            lineno=function.lineno,
            delegate_symbol=delegate_symbol,
            wrapper_kind="projection",
            statement_count=len(body),
            projected_attributes=projected_attributes,
        )

    return None

def _function_wrapper_candidates(
    module: ParsedModule,
) -> tuple[FunctionWrapperCandidate, ...]:
    candidates = [
        candidate
        for qualname, function in _iter_named_functions(module)
        for candidate in (_function_wrapper_candidate(module, qualname, function),)
        if candidate is not None
    ]
    return tuple(sorted(candidates, key=lambda item: (item.file_path, item.lineno, item.qualname)))

def _wrapper_chain_candidates(
    module: ParsedModule,
) -> tuple[WrapperChainCandidate, ...]:
    candidates = _function_wrapper_candidates(module)
    if len(candidates) < 2:
        return ()
    by_symbol = {candidate.qualname: candidate for candidate in candidates}
    inbound = Counter(
        candidate.delegate_symbol
        for candidate in candidates
        if candidate.delegate_symbol in by_symbol
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
    return tuple(
        sorted(
            chains,
            key=lambda item: (-len(item.wrappers), item.wrappers[0].lineno),
        )
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

def _pipeline_stage(statement: ast.stmt) -> PipelineAssemblyStage | None:
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(statement.value, ast.Call):
            return None
        output_arity = _assignment_target_arity(statement.targets[0])
        if output_arity is None:
            return None
        callee_name = _call_name(statement.value.func)
        if callee_name is None:
            return None
        keyword_names = tuple(
            keyword.arg for keyword in statement.value.keywords if keyword.arg is not None
        )
        return PipelineAssemblyStage(
            kind=_PIPELINE_ASSIGN_STAGE,
            callee_name=callee_name,
            output_arity=output_arity,
            arg_count=len(statement.value.args) + len(keyword_names),
            keyword_names=keyword_names,
        )
    if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call):
        callee_name = _call_name(statement.value.func)
        if callee_name is None:
            return None
        keyword_names = tuple(
            keyword.arg for keyword in statement.value.keywords if keyword.arg is not None
        )
        return PipelineAssemblyStage(
            kind=_PIPELINE_RETURN_STAGE,
            callee_name=callee_name,
            output_arity=0,
            arg_count=len(statement.value.args) + len(keyword_names),
            keyword_names=keyword_names,
        )
    return None

def _assignment_target_arity(target: ast.AST) -> int | None:
    if isinstance(target, ast.Name):
        return 1
    if isinstance(target, (ast.Tuple, ast.List)):
        if not target.elts or not all(isinstance(item, ast.Name) for item in target.elts):
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
    return tuple(sorted(functions, key=lambda item: (item.lineno, item.qualname)))

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
        tuple[tuple[object, ...], ...],
        tuple[tuple[PipelineAssemblyStage, ...], set[ResultAssemblyPipelineFunction]],
    ] = {}
    for left, right in combinations(functions, 2):
        shared_tail = _shared_pipeline_tail(left, right)
        if len(shared_tail) < config.min_shared_pipeline_stages:
            continue
        if len(shared_tail) >= len(left.stages) or len(shared_tail) >= len(right.stages):
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
            functions=tuple(
                sorted(grouped, key=lambda item: (item.lineno, item.qualname))
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
            function.qualname for function in candidate.functions
        )
        if any(
            len(existing.shared_tail) >= len(candidate.shared_tail)
            and candidate_function_names
            == tuple(function.qualname for function in existing.functions)
            for existing in filtered_candidates
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

def _nested_builder_shell_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[NestedBuilderShellCandidate, ...]:
    candidates: list[NestedBuilderShellCandidate] = []
    for qualname, function in _iter_named_functions(module):
        body = _trim_docstring_body(list(function.body))
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            continue
        returned = body[0].value
        if not isinstance(returned, ast.Call) or returned.args:
            continue
        outer_callee_name = _call_name(returned.func)
        if outer_callee_name is None:
            continue
        parameter_names = _function_param_names(function) - {"self", "cls"}
        if len(parameter_names) < config.min_nested_builder_forwarded_params:
            continue
        nested_matches: list[
            tuple[str, str, tuple[str, ...]]
        ] = []
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
                keyword.value,
                parameter_names=parameter_names,
            )
            if forwarded is None:
                continue
            if len(forwarded) < config.min_nested_builder_forwarded_params:
                continue
            nested_matches.append((keyword.arg, nested_callee_name, forwarded))
        if len(nested_matches) != 1:
            continue
        nested_field_name, nested_callee_name, forwarded_parameter_names = (
            nested_matches[0]
        )
        residue_keywords = tuple(
            keyword
            for keyword in returned.keywords
            if keyword.arg is not None and keyword.arg != nested_field_name
        )
        if not residue_keywords:
            continue
        residue_source_names = tuple(
            sorted(
                {
                    root_name
                    for keyword in residue_keywords
                    for root_name in _expression_root_names(keyword.value)
                    if root_name in (parameter_names - set(forwarded_parameter_names))
                }
            )
        )
        if not residue_source_names:
            continue
        candidates.append(
            NestedBuilderShellCandidate(
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
        )
    return tuple(sorted(candidates, key=lambda item: (item.file_path, item.lineno)))

def _indexed_family_wrapper_candidates(
    module: ParsedModule,
) -> tuple[IndexedFamilyWrapperCandidate, ...]:
    candidates: list[IndexedFamilyWrapperCandidate] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
            continue
        value = node.body[0].value
        if not isinstance(value, ast.ListComp) or len(value.generators) != 1:
            continue
        generator = value.generators[0]
        if not isinstance(generator.target, ast.Name) or generator.target.id != "item":
            continue
        if not isinstance(generator.iter, ast.Call):
            continue
        collector_name = _call_name(generator.iter.func)
        if collector_name not in {
            "_collect_items_from_spec_root",
            "collect_family_items",
        }:
            continue
        if collector_name == "_collect_items_from_spec_root":
            if len(generator.iter.args) < 3:
                continue
            spec_root_name = _call_name(generator.iter.args[0])
            item_type_name = _call_name(generator.iter.args[2])
        else:
            if len(generator.iter.args) < 2:
                continue
            spec_root_name = _call_name(generator.iter.args[1])
            item_type_name = _call_name(generator.iter.args[1])
        if spec_root_name is None or item_type_name is None:
            continue
        if not _is_instance_filter(generator.ifs, item_type_name):
            continue
        candidates.append(
            IndexedFamilyWrapperCandidate(
                function_name=node.name,
                lineno=node.lineno,
                collector_name=collector_name,
                spec_root_name=spec_root_name,
                item_type_name=item_type_name,
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.lineno))

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
    return any(arg.arg == param_name for arg in function.args.args)

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

def _module_compares_attribute(module: ast.Module, attr_name: str) -> bool:
    for node in _walk_nodes(module):
        if isinstance(node, ast.Compare):
            values = [node.left] + list(node.comparators)
            if any(
                isinstance(value, ast.Attribute) and value.attr == attr_name
                for value in values
            ):
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == _GETATTR_BUILTIN and len(node.args) >= 2:
                attr = node.args[1]
                if isinstance(attr, ast.Constant) and attr.value == attr_name:
                    return True
    return False

def _predicate_factory_chain_branch_count(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int | None:
    if not function.body or not isinstance(function.body[0], ast.If):
        return None
    branch_count = 0
    current: ast.stmt | None = function.body[0]
    while isinstance(current, ast.If):
        if not _test_has_call(current.test):
            return None
        if not any(
            isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call)
            for stmt in current.body
        ):
            return None
        branch_count += 1
        current = current.orelse[0] if len(current.orelse) == 1 else None
    if branch_count < 2:
        return None
    return branch_count

def _test_has_call(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Call) for child in _walk_nodes(node))

__all__ = tuple(name for name in globals() if not name.startswith("__"))
