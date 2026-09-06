"""Source C3 projections share work only within one fixed proof scenario."""

from abc import ABC
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_namespace import ClassNamespaceExecutionEvidence
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.native_class_mro import NativeClassMroDeclaration
from nominal_refactor_advisor.source_native_mro import (
    NativeClassBaseSubstitution,
    SourceNativeClassMro,
)


def _snapshot(path: Path, source: str) -> CodemodSourceSnapshot:
    path.write_text(source, encoding="utf-8", newline="")
    return CodemodSourceSnapshot.from_modules(parse_python_modules(path))


def test_cohort_closes_each_reachable_namespace_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "class Root: pass\n"
    for index in range(20):
        parent = "Root" if index == 0 else f"Level{index - 1}"
        source += f"class Level{index}({parent}): pass\n"
    source += "".join(f"class Leaf{index}(Level19): pass\n" for index in range(30))
    snapshot = _snapshot(tmp_path / "cohort.py", source)
    hierarchy = SourceNativeClassMro(snapshot)
    classes = tuple(snapshot.required_class_family_index.classes_by_symbol.values())
    inspected = []
    original = ClassNamespaceExecutionEvidence.require_closed

    def record(self, bindings, module, owner):
        inspected.append(owner)
        return original(self, bindings, module, owner)

    monkeypatch.setattr(ClassNamespaceExecutionEvidence, "require_closed", record)
    started = perf_counter()
    projections = tuple(hierarchy.for_source_class(owner) for owner in classes)
    print(
        f"cohort namespaces={len(classes)} inspections={len(inspected)} seconds={perf_counter() - started:.6f}"
    )
    assert len(inspected) == len(classes)
    assert all(
        hierarchy.for_source_class(owner) is projection
        for owner, projection in zip(classes, projections, strict=True)
    )
    assert len(inspected) == len(classes)


def test_substitution_and_replaced_snapshot_get_independent_projections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    snapshot = _snapshot(
        path, "from abc import ABC\nclass Root(ABC): pass\nclass Leaf(Root): pass\n"
    )
    classes = snapshot.required_class_family_index.classes_by_symbol
    root, leaf = classes["probe.Root"], classes["probe.Leaf"]
    hierarchy = SourceNativeClassMro(snapshot)
    original = hierarchy.for_source_class(leaf)
    substitution = NativeClassBaseSubstitution(
        root, root.node.bases[0], NativeClassMroDeclaration(object)
    )
    changed = replace(hierarchy, substitution=substitution)
    projected = changed.for_source_class(leaf)
    assert projected is not original
    assert ABC in original.__mro__
    assert ABC not in projected.__mro__
    assert hierarchy.for_source_class(leaf) is original
    assert changed.for_source_class(leaf) is projected

    next_snapshot = _snapshot(path, "class Root: pass\nclass Leaf(Root): pass\n")
    next_hierarchy = replace(hierarchy, context=next_snapshot)
    next_leaf = next_snapshot.required_class_family_index.classes_by_symbol[
        "probe.Leaf"
    ]
    next_projection = next_hierarchy.for_source_class(next_leaf)
    assert next_projection is not original
    assert next_projection.declaration is next_leaf


def test_unproved_class_is_not_cached_as_a_success_after_partial_traversal(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path / "probe.py",
        "class Root: pass\nclass Leaf(Root):\n    field = unknown()\n",
    )
    classes = snapshot.required_class_family_index.classes_by_symbol
    hierarchy = SourceNativeClassMro(snapshot)
    for _ in range(2):
        with pytest.raises(ValueError):
            hierarchy.for_source_class(classes["probe.Leaf"])
    assert (
        hierarchy.for_source_class(classes["probe.Root"]).declaration
        is classes["probe.Root"]
    )
