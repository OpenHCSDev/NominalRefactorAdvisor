"""Class indexing consumes bindings captured at each declaration, including projections."""

from pathlib import Path
from unittest.mock import patch

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule, parse_python_modules
from nominal_refactor_advisor.class_families.parsed_index import (
    ClassFamilyIndexBuilder,
    ClassFamilyIndexPredecessor,
    build_class_family_index,
)
from nominal_refactor_advisor.class_index import (
    ModuleNominalBindingAuthority,
    ModuleNominalBindingTimeline,
    _module_nominal_binding_snapshots,
)


@pytest.fixture
def declaration_modules(tmp_path: Path) -> tuple[ParsedModule, ...]:
    (tmp_path / "parents.py").write_text(
        "class First: pass\nclass Second: pass\n", encoding="utf-8"
    )
    (tmp_path / "consumer.py").write_text(
        "from parents import First as Parent\n"
        "class Early(Parent): pass\n"
        "from parents import Second as Parent\n"
        "class Late(Parent): pass\n"
        "Parent = unknown\n"
        "class Unproved(Parent): pass\n"
        "class Container:\n"
        "    class Nested(Parent): pass\n",
        encoding="utf-8",
    )
    return tuple(parse_python_modules(tmp_path, use_parse_cache=False))


def test_class_index_uses_captured_bases_without_per_class_module_walks(
    declaration_modules: tuple[ParsedModule, ...],
) -> None:
    with patch.object(
        ModuleNominalBindingAuthority,
        "snapshot_before",
        autospec=True,
        side_effect=ModuleNominalBindingAuthority.snapshot_before,
    ) as snapshot_reads:
        index = build_class_family_index(declaration_modules)

    assert {
        symbol: record.resolved_base_symbols
        for symbol, record in index.classes_by_symbol.items()
    } == {
        "parents.First": (),
        "parents.Second": (),
        "consumer.Early": ("parents.First",),
        "consumer.Late": ("parents.Second",),
        "consumer.Unproved": (),
        "consumer.Container": (),
        "consumer.Container.Nested": (),
    }
    # Module passes already captured these references while constructing the records.
    assert snapshot_reads.call_count == 0


def test_projection_rebinds_changed_declarations_and_reuses_unchanged_ancestors(
    declaration_modules: tuple[ParsedModule, ...], tmp_path: Path
) -> None:
    before = build_class_family_index(declaration_modules)
    consumer_path = tmp_path / "consumer.py"
    consumer_path.write_text(
        "from parents import Second as Parent\nclass Early(Parent): pass\n",
        encoding="utf-8",
    )
    changed = tuple(
        module
        for module in parse_python_modules(tmp_path, use_parse_cache=False)
        if module.path == consumer_path
    )
    after = ClassFamilyIndexBuilder(
        tuple(module for module in declaration_modules if module.path != consumer_path)
        + changed,
        previous=ClassFamilyIndexPredecessor(
            before, {module.file_path: module for module in declaration_modules}
        ),
    ).build()

    assert before.ancestor_symbols("consumer.Early") == ("parents.First",)
    assert after.ancestor_symbols("consumer.Early") == ("parents.Second",)
    assert set(after.classes_by_symbol) == {
        "parents.First", "parents.Second", "consumer.Early"
    }
    for symbol in ("parents.First", "parents.Second"):
        assert after.class_for(symbol) is before.class_for(symbol)


def test_binding_timeline_matches_sequential_snapshots_including_star_imports(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("class X: pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("class Y: pass\n", encoding="utf-8")
    (tmp_path / "m.py").write_text(
        "from a import X\n"
        "class C1(X): pass\n"
        "from b import *\n"
        "class C2(X): pass\n"
        "from a import X as Z\n"
        "class C3(Z): pass\n",
        encoding="utf-8",
    )
    (module,) = (
        m
        for m in parse_python_modules(tmp_path, use_parse_cache=False)
        if m.module_name == "m"
    )
    authority = ModuleNominalBindingAuthority(module)
    lines = (1, 2, 3, 4, 5, 6, 7)
    direct = _module_nominal_binding_snapshots(authority, lines, include_final=True)
    via_timeline = authority.snapshots_before((*lines, None))
    assert via_timeline == direct
    assert module.fact(ModuleNominalBindingTimeline).final_snapshot == direct[None]
