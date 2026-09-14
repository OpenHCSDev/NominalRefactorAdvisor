"""Point queries retain global ambiguity without enumerating unrelated flows."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import RepositoryModuleBindingProof
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    ReplaceDeclaredCallArgumentsOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.product_flow_authority import (
    CompactProductFlowRepository,
    SourceProductFlowRepository,
)


def module(name, source, path=None):
    return ParsedModule(
        Path(path or name.replace(".", "/") + ".py"),
        name,
        False,
        ast.parse(source),
        source,
    )


def test_one_scope_keeps_all_modules_available_without_building_unrelated_flows():
    selected = module("selected", "def run(): return 1\n")
    unrelated = tuple(
        module(f"other{index}", "class Other:\n    def method(self): pass\n")
        for index in range(40)
    )
    repository = SourceProductFlowRepository.from_modules((selected, *unrelated))
    context = repository.flow_context_for_symbol("selected.run")
    assert context is not None
    assert set(repository._source_projections) == {id(selected)}
    assert repository.flow_context_for_symbol("selected.run") is context
    assert repository.flow_context_for_symbol("selected.missing") is None
    assert repository.flow_context_for_symbol("absent.run") is None
    assert set(repository._source_projections) == {id(selected)}
    assert repository.modules == (selected, *unrelated)
    assert context is repository.flow_contexts_by_owner_symbol["selected.run"]
    assert len(repository._source_projections) == len(repository.modules)


@pytest.mark.parametrize(
    "modules,symbol",
    (
        ((module("m", "def f(): pass\ndef f(): pass\n"),), "m.f"),
        ((module("m", "class C: pass\nclass C: pass\n"),), "m.C"),
        ((module("m", "def f(): pass\nclass f: pass\n"),), "m.f"),
        (
            (
                module("m", "class C:\n    def f(): pass\n"),
                module("m.C", "def f(): pass\n"),
            ),
            "m.C.f",
        ),
        ((module("m", "class C: pass\n"), module("m.C", "pass\n")), "m.C"),
        (
            (
                module("m", "def f(): pass\n", "left.py"),
                module("m", "def f(): pass\n", "right.py"),
            ),
            "m.f",
        ),
        (
            (
                module("m", "def f(): pass\n", "left.py"),
                module("m", "def g(): pass\n", "right.py"),
            ),
            "m.f",
        ),
        ((module("m", "class C: pass\n"), module("m.C", "def f(): pass\n")), "m.C.f"),
        (
            (
                module("m", "class C:\n    def f(): pass\n"),
                module("m.C", "class f: pass\n"),
            ),
            "m.C.f",
        ),
    ),
)
def test_point_queries_match_full_original_indexes_including_ambiguity(modules, symbol):
    repository = SourceProductFlowRepository.from_modules(modules)
    context = repository.flow_context_for_symbol(symbol)
    multiplicity = repository.function_declaration_multiplicity_for_symbol(symbol)
    resolution = repository._declared_function_resolution(symbol)
    # Enumeration runs only after the point query, over the same original owners.
    assert context is repository.flow_contexts_by_owner_symbol.get(symbol)
    assert multiplicity.unambiguous_declarations_by_handle.get(
        symbol
    ) is repository.function_declarations_by_symbol.get(symbol)
    assert (symbol in multiplicity.ambiguous_handles) == (
        symbol in repository.ambiguous_function_declaration_symbols
    )
    compact = CompactProductFlowRepository(
        repository.product_projections, repository.class_projections
    )
    assert compact.flow_context_for_symbol(symbol) is context
    assert compact._declared_function_resolution(symbol) == resolution
    for name in {source.module_name for source in modules}:
        assert repository.module_flow_context_for_name(
            name
        ) is repository.module_flow_contexts.get(name)
        assert repository.product_projection_for_module(
            name
        ) is repository.product_projections_by_module_name.get(name)


def test_prefix_candidates_respect_module_boundaries_and_retain_both_owners():
    outer = module("m", "class C:\n    def f(): pass\n")
    overlapping = module("m.C", "def f(): pass\n")
    unrelated = module("m.Cousin", "def f(): pass\n")
    repository = SourceProductFlowRepository.from_modules(
        (outer, overlapping, unrelated)
    )
    assert repository.flow_context_for_symbol("m.C.f") is None
    assert set(repository._source_projections) == {id(outer), id(overlapping)}
    assert repository.module_flow_context_for_name("m.C").module_name == "m.C"
    assert set(repository._source_projections) == {id(outer), id(overlapping)}


def test_duplicate_original_module_rows_are_not_collapsed_by_projection_caching():
    original = module("m", "def f(): pass\n")
    repository = SourceProductFlowRepository.from_modules((original, original))
    assert repository.flow_context_for_symbol("m.f") is None
    assert repository.module_flow_context_for_name("m") is None
    assert repository.source_projection_for_module("m") is None
    assert repository.product_projection_for_module("m") is None
    assert repository.function_declaration_multiplicity_for_symbol(
        "m.f"
    ).ambiguous_handles == frozenset(("m.f",))
    assert repository.flow_contexts_by_owner_symbol == {}


def test_module_query_reuses_its_entry_without_observing_parent_scopes():
    parent = module("m", "class C: pass\n")
    selected = module("m.C", "def f(): pass\n")
    repository = SourceProductFlowRepository.from_modules((parent, selected))
    context = repository.module_flow_context_for_name("m.C")
    assert context is repository.source_projection(selected).module_context
    assert set(repository._source_projections) == {id(selected)}
    assert repository.module_flow_context_for_name("missing") is None
    assert set(repository._source_projections) == {id(selected)}


def test_star_import_metadata_does_not_require_class_or_flow_projection():
    importing = module("importing", "from provider import *\ndef f(): pass\n")
    provider = module("provider", "__all__ = ['item']\nitem = 1\n")
    repository = SourceProductFlowRepository.from_modules((importing, provider))
    origins = repository.star_import_origins_for("importing")
    assert (
        SourceProductFlowRepository.star_import_origins_for
        is RepositoryModuleBindingProof.star_import_origins_for
    )
    assert len(origins) == 1
    assert origins[0].module_name == "provider"
    assert not repository._source_projections
    assert "class_projections" not in vars(repository)
    assert repository.star_import_origins_for("missing") == ()
    assert (
        origins
        == repository.class_projections_by_module_name["importing"].star_import_origins
    )


def test_ambiguous_module_does_not_select_one_star_import_source():
    repository = SourceProductFlowRepository.from_modules(
        (
            module("m", "from left import *\n", "one.py"),
            module("m", "from right import *\n", "two.py"),
        )
    )
    assert repository.star_import_origins_for("m") == ()
    assert not repository._source_projections
    assert "m" not in repository.class_projections_by_module_name


def test_new_snapshot_cannot_reuse_a_pre_edit_unique_symbol(tmp_path):
    path = str(tmp_path / "probe.py")
    original = "def render(value): return value\ndef run(): return render(1)\n"
    snapshot = CodemodSourceSnapshot.from_source_mapping({path: original})
    first = snapshot.product_flow_repository.flow_context_for_symbol("probe.render")
    assert first is not None
    changed = snapshot.with_virtual_sources(
        {path: original + "def render(value): return None\n"}
    )
    assert (
        changed.product_flow_repository.flow_context_for_symbol("probe.render") is None
    )
    assert (
        snapshot.product_flow_repository.flow_context_for_symbol("probe.render")
        is first
    )
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceDeclaredCallArgumentsOperation(
                target=SourceRewriteTarget(file_path=path, qualname="run"),
                callee=SourceRewriteTarget(file_path=path, qualname="render"),
                arguments_source="2",
            ),
        )
    )
    assert (
        plan.simulate(snapshot)
        .final_snapshot.sources_by_file_path[path]
        .endswith("render(2)\n")
    )
    with pytest.raises(ValueError):
        plan.simulate(changed)
