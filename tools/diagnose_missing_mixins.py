"""Diagnose each remaining missing_mixin candidate with its exact blocker.

Prints actionable output: which ABC parents can be promoted to @dataclass,
which children's fields can be deleted, and what blocks each one that can't.

Usage: python tools/diagnose_missing_mixins.py
"""
import ast
import sys
from pathlib import Path


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
            is_dc = any(
                (isinstance(d, ast.Name) and d.id == "dataclass")
                or (
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name)
                    and d.func.id == "dataclass"
                )
                for d in node.decorator_list
            )
            fields = {}
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
                "fields": fields,
                "bases": bases,
                "line": node.lineno,
                "bare_assigns": bare_assigns,
                "alias_props": alias_props,
                "properties": properties,
                "cached_properties": cached_properties,
            }
        classes_by_file[str(p)] = classes

    # Build parent→children map per file
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

        for parent_name, children in parents.items():
            parent = classes[parent_name]
            if parent["is_dc"]:
                # Parent already a dataclass — check for deletable child fields
                for child_name, child in children:
                    for field_name, field_type in child["fields"].items():
                        if (
                            field_name in parent["fields"]
                            and parent["fields"][field_name] == field_type
                        ):
                            print(
                                f"DELETE {file_path}:{child['line']} "
                                f"{child_name}.{field_name} "
                                f"(parent {parent_name} already owns it)"
                            )
                        elif field_name in parent["fields"]:
                            print(
                                f"NARROWS {file_path}:{child['line']} "
                                f"{child_name}.{field_name}: "
                                f"{parent['fields'][field_name]} → {field_type} "
                                f"(keep for type narrowing)"
                            )
                continue

            # Parent is NOT a dataclass — diagnose blockers
            blockers = []
            for child_name, child in children:
                for field_name in parent["fields"]:
                    if field_name in child.get("bare_assigns", {}):
                        blockers.append(
                            f"BARE_ASSIGN: {child_name}.{field_name} = "
                            f"{child['bare_assigns'][field_name][:40]} "
                            f"→ convert to field(default=..., init=False)"
                        )
                    if field_name in child.get("alias_props", set()):
                        blockers.append(
                            f"ALIAS_PROPERTY: {child_name}.{field_name} "
                            f"→ abstract contract, not stored field"
                        )
                    if field_name in child.get("properties", set()):
                        blockers.append(
                            f"PROPERTY: {child_name}.{field_name} "
                            f"→ abstract contract, not stored field"
                        )
                    if field_name in child.get("cached_properties", set()):
                        blockers.append(
                            f"CACHED_PROPERTY: {child_name}.{field_name} "
                            f"→ abstract contract, not stored field"
                        )

            # Check for same-type deletable fields
            deletable = []
            narrowing = []
            for child_name, child in children:
                for field_name, field_type in child["fields"].items():
                    if field_name not in parent["fields"]:
                        continue
                    if parent["fields"][field_name] == field_type:
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
            # else: no shared fields — not a candidate


if __name__ == "__main__":
    scan()
