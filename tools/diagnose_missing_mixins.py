"""Diagnose each remaining missing_mixin candidate with its exact blocker.

Prints actionable output: which ABC parents can be promoted to @dataclass,
which children's fields can be deleted, and what blocks each one that can't.

Categories beyond DELETE and NARROWS:
  COMPUTED   — child uses init=False (post_init computes it, not a pure copy)
  METADATA   — child adds field() metadata the parent lacks (e.g. json_report_field)
  DEFAULT_DIFFERS — child provides a different default than parent
  FIELD_WRAP — child wraps in field() with extra args the parent doesn't use
  FROZEN_MUTABLE — mutable children prevent frozen parent promotion

Usage: python tools/diagnose_missing_mixins.py
"""
import ast
import sys
from pathlib import Path


def _field_metadata(item: ast.AnnAssign) -> dict:
    meta = {"has_default": False, "init": True, "has_metadata": False}
    if item.value is None:
        return meta
    val = item.value
    src = ast.unparse(val)
    if isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id == "field":
        for kw in val.keywords:
            if kw.arg == "init" and isinstance(kw.value, ast.Constant):
                meta["init"] = kw.value.value
            if kw.arg in ("default", "default_factory"):
                meta["has_default"] = True
            if kw.arg == "metadata":
                meta["has_metadata"] = True
            if kw.arg == "kw_only" and isinstance(kw.value, ast.Constant):
                meta["kw_only"] = kw.value.value
        meta["field_src"] = src
    else:
        meta["has_default"] = True
        meta["default_src"] = src
    return meta


def _dc_frozen(node: ast.ClassDef) -> bool | None:
    for d in node.decorator_list:
        if isinstance(d, ast.Name) and d.id == "dataclass":
            return False
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass":
            for kw in d.keywords:
                if kw.arg == "frozen" and isinstance(kw.value, ast.Constant):
                    return kw.value.value
            return False
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
            frozen = _dc_frozen(node)
            is_dc = frozen is not None
            fields = {}
            field_meta = {}
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
                    field_meta[item.target.id] = _field_metadata(item)
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
                "frozen": frozen,
                "fields": fields,
                "field_meta": field_meta,
                "bases": bases,
                "line": node.lineno,
                "bare_assigns": bare_assigns,
                "alias_props": alias_props,
                "properties": properties,
                "cached_properties": cached_properties,
            }
        classes_by_file[str(p)] = classes

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
                for child_name, child in children:
                    for field_name, field_type in child["fields"].items():
                        if field_name not in parent["fields"]:
                            continue
                        parent_type = parent["fields"][field_name]
                        child_meta = child["field_meta"].get(field_name, {})
                        parent_meta = parent["field_meta"].get(field_name, {})
                        if parent_type != field_type:
                            print(
                                f"NARROWS {file_path}:{child['line']} "
                                f"{child_name}.{field_name}: "
                                f"{parent_type} → {field_type} "
                                f"(keep for type narrowing)"
                            )
                        elif not child_meta.get("init", True):
                            print(
                                f"COMPUTED {file_path}:{child['line']} "
                                f"{child_name}.{field_name} "
                                f"(init=False — computed in child, not a pure copy)"
                            )
                        elif child_meta.get("has_metadata") and not parent_meta.get("has_metadata"):
                            print(
                                f"METADATA {file_path}:{child['line']} "
                                f"{child_name}.{field_name} "
                                f"(child adds field metadata — not a pure copy)"
                            )
                        elif child_meta.get("has_default") != parent_meta.get("has_default"):
                            print(
                                f"DEFAULT_DIFFERS {file_path}:{child['line']} "
                                f"{child_name}.{field_name} "
                                f"(different default — not a pure copy)"
                            )
                        elif child_meta.get("field_src") and not parent_meta.get("field_src"):
                            print(
                                f"FIELD_WRAP {file_path}:{child['line']} "
                                f"{child_name}.{field_name} "
                                f"(child uses field() with extra args — not a pure copy)"
                            )
                        else:
                            print(
                                f"DELETE {file_path}:{child['line']} "
                                f"{child_name}.{field_name} "
                                f"(parent {parent_name} already owns it)"
                            )
                continue

            # Parent is NOT a dataclass — diagnose blockers
            # Check ALL descendants (not just direct children) for blockers
            def _all_descendants(cls_name, visited=None):
                if visited is None:
                    visited = set()
                result = []
                for cname, cdata in classes.items():
                    if cls_name in cdata["bases"] and cname not in visited:
                        visited.add(cname)
                        result.append((cname, cdata))
                        result.extend(_all_descendants(cname, visited))
                return result

            all_desc = _all_descendants(parent_name)
            blockers = []
            for desc_name, desc in all_desc:
                if desc["is_dc"] and not desc.get("frozen"):
                    blockers.append(
                        f"MUTABLE_CHILD: {desc_name} is @dataclass without frozen=True "
                        f"→ frozen parent would break __setattr__"
                    )

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

            deletable = []
            narrowing = []
            computed = []
            metadata_diff = []
            for child_name, child in children:
                for field_name, field_type in child["fields"].items():
                    if field_name not in parent["fields"]:
                        continue
                    child_meta = child["field_meta"].get(field_name, {})
                    if parent["fields"][field_name] != field_type:
                        narrowing.append(
                            f"{child_name}.{field_name}: "
                            f"{parent['fields'][field_name]} → {field_type}"
                        )
                    elif not child_meta.get("init", True):
                        computed.append(f"{child_name}.{field_name} (init=False)")
                    elif child_meta.get("has_metadata"):
                        metadata_diff.append(f"{child_name}.{field_name} (has field metadata)")
                    else:
                        deletable.append(f"{child_name}.{field_name}")

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
                for c in computed:
                    print(f"  keep (computed): {c}")
                for m in metadata_diff:
                    print(f"  keep (metadata): {m}")


if __name__ == "__main__":
    scan()
