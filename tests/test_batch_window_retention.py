"""Both real lowering phases retain their actual source-window participants."""

from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    EnsureImportOperation,
    InsertClassMemberOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_runtime import (
    NominalEditBatch,
    RefactorRecipeOperationCompiler,
)
from nominal_refactor_advisor.codemod_declaration_source import ClassMemberInsertion
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_source_edits import (
    CoalescedSourceWindow,
    ExactSourceWindow,
    NominalSourceEdit,
    PhysicalSourceEdit,
    SourceFileCreation,
    SourceInsertion,
    SourceSpanDeletion,
    SourceSpanReplacement,
    SourceTextGeometry,
    SourceTextMutation,
    SourceTextSpan,
    SourceTextSpanReplacement,
)

PATH = Path(__file__).with_name("window_retention_fixture.py").resolve().as_posix()
SOURCE = "left = object; right = type; tail = sorted\n"


def compiler(source=SOURCE):
    return RefactorRecipeOperationCompiler.from_context(
        CodemodSourceSnapshot.from_source_mapping({PATH: source})
    )


def exact(source=SOURCE, old="left", new="renamed"):
    offset = source.index(old)
    return SourceTextGeometry(source).nominal_edit(
        file_path=PATH,
        replacements=(
            SourceTextSpanReplacement(
                offset, offset + len(old), replacement_source=new
            ),
        ),
    )


def insertion(name):
    return SourceInsertion(
        file_path=PATH, insertion_line=1, inserted_lines=(f"import {name}\n",)
    )


def rendered(batch):
    simulation = batch.compiler.simulate_rewrites(batch.planned_rewrites)
    assert simulation.parse_valid
    return simulation.rewritten_sources[PATH]


def test_two_phase_groups_keep_actual_occurrences_and_rendering_order():
    first, last = insertion("typing"), insertion("builtins")
    middle = SourceTextGeometry(SOURCE).nominal_edit(
        file_path=PATH,
        replacements=(
            SourceTextSpanReplacement(0, 0, replacement_source="import collections\n"),
        ),
    )
    batch = NominalEditBatch(compiler(), (first, middle, last, first))
    (final,) = batch.windows
    physical_group, exact_window = final.windows
    assert physical_group.windows[0] is first
    assert physical_group.windows[1] is last
    assert physical_group.windows[2] is first
    assert exact_window.resolution.mutations[0] is middle
    assert (
        rendered(batch)
        == "import typing\nimport builtins\nimport collections\n" + SOURCE
    )
    assert len(batch.physical_edits) == 1


@pytest.mark.parametrize("opaque", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_equal_output_intersects_actual_exact_and_opaque_evidence(opaque, reverse):
    class OtherExact(SourceTextMutation):
        pass

    first = exact(old="object", new="object")
    other = exact(old="type", new="type")
    second = (
        SourceSpanReplacement(
            file_path=PATH, start_line=1, end_line=1, replacement_lines=(SOURCE,)
        )
        if opaque
        else OtherExact(revision=other.revision, replacements=other.replacements)
    )
    batch = NominalEditBatch(
        compiler(), (second, first) if reverse else (first, second)
    )
    assert rendered(batch) == SOURCE
    (window,) = batch.windows
    for name in ("object", "type"):
        span = SourceTextSpan(SOURCE.index(name), SOURCE.index(name) + len(name))
        with pytest.raises(ValueError):
            window.project_retained_interiors((span,))
    tail = SourceTextSpan(SOURCE.index("sorted"), SOURCE.index("sorted") + 6)
    if opaque:
        with pytest.raises(ValueError, match="Opaque"):
            window.project_retained_interiors((tail,))
    else:
        assert window.project_retained_interiors((tail,)) == (tail,)


def test_deletion_groups_preserve_sorted_overlap_and_pairwise_metadata_fold():
    source = "".join(f"value_{index} = object\n" for index in range(1, 8))
    first, second, third, isolated = tuple(
        SourceSpanDeletion(
            file_path=PATH, start_line=start, end_line=end, rationale=rationale
        )
        for start, end, rationale in (
            (1, 2, "A"),
            (2, 3, "B"),
            (3, 4, "A"),
            (6, 6, "C"),
        )
    )
    batch = NominalEditBatch(compiler(source), (third, isolated, first, second))
    joined, separate = batch.windows
    assert joined.windows[0].windows == (first, second, third)
    assert separate.windows[0].windows == (isolated,)
    assert batch.physical_edits[0].rationale == "A B A"
    assert rendered(batch) == "value_5 = object\nvalue_7 = object\n"


@pytest.mark.parametrize("wrong_key", (False, True))
def test_window_constructor_rejects_wrong_nominal_owner_or_group(wrong_key):
    class OtherInsertion(SourceInsertion):
        pass

    first = insertion("typing")
    second = (
        replace(first, insertion_line=2)
        if wrong_key
        else OtherInsertion(
            file_path=PATH, insertion_line=1, inserted_lines=first.inserted_lines
        )
    )
    with pytest.raises(ValueError):
        CoalescedSourceWindow((first, second))


def test_imports_keep_nominal_grouping_without_becoming_window_declarations():
    operation = EnsureImportOperation(
        target=SourceRewriteTarget(file_path=PATH), import_source="import typing"
    )
    owner = compiler()
    imports = owner.edit_batch_for_recipes(
        (RefactorRecipe("imports", operations=(operation, operation)),)
    )
    assert all(
        not isinstance(edit, (CoalescedSourceWindow, ExactSourceWindow))
        for edit in imports.edits
    )
    batch = NominalEditBatch(owner, (*imports.edits, exact()))
    output = rendered(batch)
    assert output.count("import typing") == 1
    assert "renamed = object" in output
    assert batch.windows


@pytest.mark.parametrize("inline", (False, True))
def test_member_path_retains_exact_authority_after_nominal_coalescence(inline):
    source = "class Handler:" + (
        " value = object\n" if inline else "\n    value = object\n"
    )
    owner = compiler(source)
    target = SourceRewriteTarget(file_path=PATH, qualname="Handler")
    batch = owner.edit_batch_for_recipes(
        (
            RefactorRecipe(
                "members",
                operations=(
                    InsertClassMemberOperation(target=target, source="added = 1\n"),
                    InsertClassMemberOperation(target=target, source="other = 2\n"),
                ),
            ),
        )
    )
    assert len(batch.edits) == 2
    assert all(
        isinstance(window.windows[0], ExactSourceWindow) for window in batch.windows
    )
    output = rendered(batch)
    assert "added = 1" in output and "other = 2" in output
    if inline:
        span = SourceTextSpan(source.index("object"), source.index("object") + 6)
        with pytest.raises(ValueError, match="unchanged-text correspondence"):
            batch.windows[0].project_retained_interiors((span,))


def test_batch_caches_exact_resolution_for_render_and_repeated_interior_queries(
    monkeypatch,
):
    original = SourceTextGeometry.physical_edit_projections
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SourceTextGeometry, "physical_edit_projections", counted)
    source = SOURCE + SOURCE.replace("left", "second")
    first = exact(source)
    second = exact(source, old="second", new="another")
    batch = NominalEditBatch(compiler(source), (first, second))
    windows = batch.windows
    assert len(windows) == 2
    exact_window = windows[0].windows[0]
    assert exact_window.resolution.mutations == (first, second)
    assert windows[1].windows[0].resolution is exact_window.resolution
    span = SourceTextSpan(source.index("sorted"), source.index("sorted") + 6)
    expected = windows[0].project_retained_interiors((span,))
    for _ in range(3):
        assert batch.windows is windows
        assert windows[0].project_retained_interiors((span,)) == expected
        assert "renamed = object" in rendered(batch)
    assert len(calls) == 1


def test_one_abstract_window_owner_and_one_physical_projection():
    for declaration in (
        PhysicalSourceEdit,
        SourceFileCreation,
        ModuleImportMutation,
        ClassMemberInsertion,
        SourceTextMutation,
    ):
        assert declaration.resolved_edits is NominalSourceEdit.resolved_edits
    assert "resolved_windows" in NominalSourceEdit.__abstractmethods__
    assert "resolved_edits" not in NominalSourceEdit.__abstractmethods__

    class LegacyOnly(NominalSourceEdit):
        def coalesced_with_peers(self, peers, context):
            return peers

        def resolved_edits(self, context):
            return ()

    with pytest.raises(TypeError, match="resolved_windows"):
        LegacyOnly()


@pytest.mark.parametrize("deletion", (False, True))
def test_keyed_and_deletion_grouping_project_each_occurrence_once(deletion):
    peer = (
        SourceSpanDeletion(file_path=PATH, start_line=1, end_line=2)
        if deletion
        else insertion("typing")
    )
    duplicate = replace(peer)
    originals = (peer, duplicate, peer)
    calls = []

    def declaration(item):
        calls.append(item)
        return item

    groups = type(peer).projected_peer_groups(originals, declaration)
    assert len(groups) == 1
    assert len(calls) == len(originals)
    assert all(actual is expected for actual, expected in zip(calls, originals))
    assert all(actual is expected for actual, expected in zip(groups[0], originals))

