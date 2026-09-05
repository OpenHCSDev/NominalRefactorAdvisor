"""Compact and syntax-backed indexes derive the same nominal graph views."""

from dataclasses import fields
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_index import (
    ClassFamilyIndex,
    CompactClassFamilyIndex,
    build_class_family_index,
)


@pytest.fixture
def indexes(tmp_path: Path):
    source = (
        "class Root: pass\n"
        "class Left(Root): pass\n"
        "class Right(Root): pass\n"
        "class Diamond(Left, Right): pass\n"
        "class Tip(Diamond, Right): pass\n"
        "class Unresolved(Unknown): pass\n"
        "class Container:\n    class Nested: pass\n"
    )
    (tmp_path / "family.py").write_text(source, encoding="utf-8")
    (tmp_path / "other.py").write_text(
        "from family import Root as Parent\nclass Imported(Parent): pass\nclass Root: pass\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(tmp_path))
    return build_class_family_index(
        list(modules)
    ), CompactClassFamilyIndex.from_modules(modules)


@pytest.mark.parametrize("index_type", (ClassFamilyIndex, CompactClassFamilyIndex))
def test_only_declarations_are_constructor_authorities(index_type) -> None:
    assert tuple(field.name for field in fields(index_type)) == ("classes_by_symbol",)


def test_both_record_forms_inherit_one_lookup_implementation(indexes) -> None:
    full, compact = indexes
    assert full.class_for.__func__ is compact.class_for.__func__
    assert full.symbol_for.__func__ is compact.symbol_for.__func__
    assert full.ancestor_symbols.__func__ is compact.ancestor_symbols.__func__
    assert full.descendant_symbols.__func__ is compact.descendant_symbols.__func__


def test_all_derived_views_agree_between_record_forms(indexes) -> None:
    full, compact = indexes
    for name in (
        "symbols_by_simple_name",
        "symbols_by_file_and_qualname",
        "children_by_symbol",
        "ancestors_by_symbol",
        "descendants_by_symbol",
    ):
        assert getattr(full, name) == getattr(compact, name)
    for index in indexes:
        assert index.class_for("family.Root") is index.classes_by_symbol["family.Root"]
        assert index.ancestor_symbols("family.Tip") == (
            "family.Diamond",
            "family.Right",
            "family.Left",
            "family.Root",
        )
        assert index.children_by_symbol["family.Root"] == (
            "family.Left",
            "family.Right",
            "other.Imported",
        )
        assert index.descendant_symbols("family.Root") == (
            "family.Left",
            "family.Right",
            "other.Imported",
            "family.Diamond",
            "family.Tip",
        )
        assert index.ancestor_symbols("family.Unresolved") == ()
        assert index.symbols_by_simple_name["Root"] == ("family.Root", "other.Root")
        assert index.class_for("missing") is None
        assert index.symbol_for(file_path="missing", qualname="Root") is None
        assert (
            index.ancestor_symbols("missing")
            == index.descendant_symbols("missing")
            == ()
        )


def test_lookup_does_not_materialize_unrequested_global_closures(indexes) -> None:
    for index in indexes:
        assert "ancestors_by_symbol" not in vars(index)
        assert "descendants_by_symbol" not in vars(index)
        assert index.ancestor_symbols("family.Tip")
        assert "ancestors_by_symbol" not in vars(index)
        assert "descendants_by_symbol" not in vars(index)


def test_reachability_order_does_not_replace_native_mro(indexes) -> None:
    full, compact = indexes
    source = Path(full.class_for("family.Root").file_path).read_text(encoding="utf-8")
    namespace = {}
    exec(source.replace("class Unresolved(Unknown): pass\n", ""), namespace)
    native_order = tuple(
        f"family.{owner.__name__}" for owner in namespace["Tip"].__mro__[1:-1]
    )
    assert native_order == (
        "family.Diamond",
        "family.Left",
        "family.Right",
        "family.Root",
    )
    assert compact.ancestor_symbols("family.Tip") != native_order


def test_repeated_class_identity_is_unproved_in_both_record_forms(
    tmp_path: Path,
) -> None:
    (tmp_path / "rebound.py").write_text(
        "class Base: first = True\nclass Base: second = True\nclass Child(Base): pass\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(tmp_path))
    full = build_class_family_index(list(modules))
    compact = CompactClassFamilyIndex.from_modules(modules)
    for index in (full, compact):
        assert index.class_for("rebound.Base") is None
        assert "Base" not in index.symbols_by_simple_name
        assert index.ancestor_symbols("rebound.Child") == ()
