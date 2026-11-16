"""Diagnose each remaining missing_mixin candidate with its exact blocker.

Prints actionable output: which ABC parents can be promoted to @dataclass,
which children's fields can be deleted, and what blocks each one that can't.

Usage: python tools/diagnose_missing_mixins.py
"""
import ast
import sys
from pathlib import Path


def _is_dataclass_decorator(d):
    return (isinstance(d, ast.Name) and d.id == "dataclass") or (
        isinstance(d, ast.Call)
        and isinstance(d.func, ast.Name)
        and d.func.id == "dataclass"
    )


def _is_frozen(node):
    for d in node.decorator_list:
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass":
            for kw in d.keywords:
                if kw.arg == "frozen" and isinstance(kw.value, ast.Constant) and kw.value.value:
                    return True
    return False


def _field_has_init_false(value_node):
    if not isinstance(value_node, ast.Call):
        return False
    func = value_node.func
    if not (isinstance(func, ast.Name) and func.id == "field"):
        return False
    for kw in value_node.keywords:
        if kw.arg == "init" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
            return True
    return False


def _field_default_repr(item):
    if isinstance(item, ast.AnnAssign) and item.value is not None:
        return ast.unparse(item.value)
    return None


def scan():
    classes_by_file: dict[str, dict[str, dict]] = {}

    for p in sorted(Path("nominal_refactor_advisor").rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        try:
            source = p.read_text()
            tree = ast.parse(source)
        except SyntaxError:
            continue

        classes = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            is_dc = any(_is_dataclass_decorator(d) for d in node.decorator_list)
            is_frozen = _is_frozen(node)
            fields = {}
            field_defaults = {}
            field_init_false = set()
            bare_assigns = {}
            alias_props = set()
            properties = set()
            cached_properties = set()
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    ann = item.annotation
                    if (
                        isinstance(ann, ast.Subscript)
                        and isinstance(ann.value, ast.Name)
                        and ann.value.id == "ClassVar"
                    ):
                        continue
                    fields[item.target.id] = ast.unparse(ann)
                    default = _field_default_repr(item)
                    if default is not None:
                        field_defaults[item.target.id] = default
                    if item.value is not None and _field_has_init_false(item.value):
                        field_init_false.add(item.target.id)
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            val = ast.unparse(item.value) if item.value else ""
                            if "AliasProperty" in val:
                                alias_props.add(target.id)
                            else:
                                bare_assigns[target.id] = val
                elif isinstance(item, ast.FunctionDef):
                    for d in item.decorator_list:
                        if isinstance(d, ast.Name) and d.id == "property":
                            properties.add(item.name)
                        if isinstance(d, ast.Name) and d.id == "cached_property":
                            cached_properties.add(item.name)

            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            classes[node.name] = {
                "is_dc": is_dc,
                "is_frozen": is_frozen,
                "fields": fields,
                "field_defaults": field_defaults,
                "field_init_false": field_init_false,
                "bases": bases,
                "line": node.lineno,
                "bare_assigns": bare_assigns,
                "alias_props": alias_props,
                "properties": properties,
                "cached_properties": cached_properties,
            }
        classes_by_file[str(p)] = classes

    # Resolve transitive descendants within each file
    def all_descendants(classes, parent_name):
        result = []
        for child_name, child in classes.items():
            if parent_name in child["bases"]:
                result.append((child_name, child))
                result.extend(all_descendants(classes, child_name))
        return result

    for file_path, classes in classes_by_file.items():
        parents: dict[str, list[tuple[str, dict]]] = {}
        for child_name, child in classes.items():
            if not child["is_dc"]:
                continue
            for base_name in child["bases"]:
                parent = classes.get(base_name)
                if parent is None or not parent["fields"]:
                    continue
                parents.setdefault(base_name, []).append((child_name, child))

        for parent_name, direct_children in parents.items():
            parent = classes[parent_name]
            if parent["is_dc"]:
                for child_name, child in direct_children:
                    for field_name, field_type in child["fields"].items():
                        if field_name not in parent["fields"]:
                            continue
                        if field_name in child["field_init_false"]:
                            print(
                                f"COMPUTED {file_path}:{child['line']} "
                                f"{child_name}.{field_name} "
                                f"(init=False override of parent {parent_name}'s field — keep)"
                            )
                        elif parent["fields"][field_name] != field_type:
                            print(
                                f"NARROWS {file_path}:{child['line']} "
                                f"{child_name}.{field_name}: "
                                f"{parent['fields'][field_name]} → {field_type} "
                                f"(keep for type narrowing)"
                            )
                        elif (
                            field_name in child.get("field_defaults", {})
                            and field_name in parent.get("field_defaults", {})
                            and child["field_defaults"][field_name] != parent["field_defaults"][field_name]
                        ):
                            print(
                                f"DIFF_DEFAULT {file_path}:{child['line']} "
                                f"{child_name}.{field_name} "
                                f"(same type but different default: "
                                f"parent={parent['field_defaults'][field_name]}, "
                                f"child={child['field_defaults'][field_name]} — keep)"
                            )
                        elif parent["fields"][field_name] == field_type:
                            print(
                                f"DELETE {file_path}:{child['line']} "
                                f"{child_name}.{field_name} "
                                f"(parent {parent_name} already owns it)"
                            )
                continue

            # Parent is NOT a dataclass — diagnose blockers
            # Check ALL descendants (not just direct children) for blockers
            all_desc = all_descendants(classes, parent_name)
            all_dc_desc = [(n, c) for n, c in all_desc if c["is_dc"]]

            blockers = []
            for desc_name, desc in all_desc:
                for field_name in parent["fields"]:
                    if field_name in desc.get("bare_assigns", {}):
                        blockers.append(
                            f"BARE_ASSIGN: {desc_name}.{field_name} = "
                            f"{desc['bare_assigns'][field_name][:40]} "
                            f"→ convert to field(default=..., init=False)"
                        )
                    if field_name in desc.get("alias_props", set()):
                        blockers.append(
                            f"ALIAS_PROPERTY: {desc_name}.{field_name} "
                            f"→ abstract contract, not stored field"
                        )
                    if field_name in desc.get("properties", set()):
                        blockers.append(
                            f"PROPERTY: {desc_name}.{field_name} "
                            f"→ abstract contract, not stored field"
                        )
                    if field_name in desc.get("cached_properties", set()):
                        blockers.append(
                            f"CACHED_PROPERTY: {desc_name}.{field_name} "
                            f"→ abstract contract, not stored field"
                        )

            # Check for frozen/mutable conflict: if promoting parent as frozen,
            # any non-frozen dataclass descendant would break
            for desc_name, desc in all_dc_desc:
                if desc["is_dc"] and not desc["is_frozen"]:
                    blockers.append(
                        f"MUTABLE_CHILD: {desc_name} is @dataclass without frozen=True "
                        f"→ frozen parent would break __setattr__"
                    )

            # Check for same-type deletable fields (direct children only)
            deletable = []
            narrowing = []
            for child_name, child in direct_children:
                for field_name, field_type in child["fields"].items():
                    if field_name not in parent["fields"]:
                        continue
                    if field_name in child["field_init_false"]:
                        narrowing.append(
                            f"{child_name}.{field_name} "
                            f"(init=False computed override — keep)"
                        )
                    elif parent["fields"][field_name] == field_type:
                        deletable.append(f"{child_name}.{field_name}")
                    else:
                        narrowing.append(
                            f"{child_name}.{field_name}: "
                            f"{parent['fields'][field_name]} → {field_type}"
                        )

            if blockers:
                print(
                    f"\nBLOCKED {file_path}:{parent['line']} {parent_name}"
                )
                for b in blockers:
                    print(f"  {b}")
                if deletable:
                    print(f"  (would delete: {', '.join(deletable)})")
            elif deletable:
                print(
                    f"\nPROMOTE {file_path}:{parent['line']} {parent_name}"
                )
                print(f"  add @dataclass(frozen=True)")
                for d in deletable:
                    print(f"  delete {d}")
                for n in narrowing:
                    print(f"  keep (narrows): {n}")


if __name__ == "__main__":
    scan()
