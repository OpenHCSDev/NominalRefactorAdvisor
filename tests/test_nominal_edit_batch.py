"""The compiler retains original records under the source that produced them."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    EnsureImportOperation,
    PatchTargetOperation,
    RefactorRecipe,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.codemod_runtime import RefactorRecipeOperationCompiler
from nominal_refactor_advisor.codemod_source_edits import (
    PhysicalSourceEditConflictError,
)
from nominal_refactor_advisor.json_reports import json_report_object

PATH = "/repo/edit_batch.py"
SOURCE = "class Handler:\n    value = object\n"


def compiler(source=SOURCE):
    return RefactorRecipeOperationCompiler.from_context(
        CodemodSourceSnapshot.from_source_mapping({PATH: source})
    )


def recipes():
    target = SourceRewriteTarget(file_path=PATH)
    return (
        RefactorRecipe(
            "imports",
            operations=(
                EnsureImportOperation(target=target, import_source="import typing"),
                EnsureImportOperation(target=target, import_source="import builtins"),
            ),
        ),
        RefactorRecipe(
            "body",
            operations=(
                PatchTargetOperation(
                    target=replace(target, qualname="Handler"),
                    replacements=(
                        SourceTextReplacement(
                            old_source="value = object", new_source="renamed = object"
                        ),
                    ),
                ),
            ),
        ),
    )


def test_batch_retains_origin_records_before_physical_coalescing():
    owner = compiler()
    batch = owner.edit_batch_for_recipes(recipes())
    assert batch.compiler is owner
    assert len(batch.edits) == 3
    assert tuple(
        (edit.origins[0].recipe_id, edit.origins[0].plan_item_index)
        for edit in batch.edits
    ) == (("imports", 0), ("imports", 1), ("body", 0))
    original = batch.edits
    assert batch.physical_edits is batch.physical_edits
    assert batch.planned_rewrites is batch.planned_rewrites
    assert batch.edits is original
    result = owner.simulate_rewrites(batch.planned_rewrites)
    assert result.parse_valid
    assert "renamed = object" in result.rewritten_sources[PATH]
    assert "import typing" in result.rewritten_sources[PATH]
    assert "import builtins" in result.rewritten_sources[PATH]
    assert (
        result.rewritten_sources
        == owner.simulate_rewrites(
            owner.planned_rewrites_for_recipes(recipes())
        ).rewritten_sources
    )


def test_lowering_does_not_reinvoke_the_actual_operation(monkeypatch):
    calls = []
    original = PatchTargetOperation.source_edits

    def observe(operation, context):
        calls.append((operation, context))
        return original(operation, context)

    monkeypatch.setattr(PatchTargetOperation, "source_edits", observe)
    owner = compiler()
    batch = owner.edit_batch_for_recipes(recipes())
    assert len(calls) == 1
    assert calls[0][1] is owner
    owner.simulate_rewrites(batch.planned_rewrites)
    batch.physical_edits
    assert len(calls) == 1


def test_bound_rewrite_rejects_a_different_source_at_the_same_coordinates():
    batch = compiler().edit_batch_for_recipes(recipes())
    changed = compiler(SOURCE.replace("object", "sorted"))
    with pytest.raises(ValueError):
        changed.simulate_rewrites(batch.planned_rewrites)


def test_empty_batch_is_a_cached_empty_projection_not_a_new_execution():
    batch = compiler().edit_batch_for_recipes(())
    assert batch.edits == batch.physical_edits == batch.planned_rewrites == ()


def test_document_retains_its_actual_batch_through_simulation():
    snapshot = CodemodSourceSnapshot.from_source_mapping({PATH: SOURCE})
    preflight = CodemodPlanDocument(recipes=recipes()).preflight(snapshot)
    batch = preflight.required_edit_batch
    assert batch.compiler is preflight.rewrite_snapshot
    assert preflight.rewrites is batch.planned_rewrites
    result = preflight.simulate()
    assert result.edit_batch is batch
    assert result.edit_batch.edits is batch.edits
    assert result.is_clean
    payload = json_report_object(result)
    assert "edit_batch" not in payload
    assert "renamed = object" in result.simulation.rewritten_sources[PATH]


def test_failed_preflight_cannot_become_an_empty_successful_batch():
    bad_recipe = RefactorRecipe(
        "missing",
        operations=(
            PatchTargetOperation(
                target=SourceRewriteTarget(file_path=PATH, qualname="Missing"),
                replacements=(SourceTextReplacement(old_source="x", new_source="y"),),
            ),
        ),
    )
    preflight = CodemodPlanDocument(recipes=(bad_recipe,)).preflight(compiler())
    assert not preflight.report.is_clean
    assert preflight.edit_batch is None
    with pytest.raises(ValueError):
        preflight.required_edit_batch
    with pytest.raises(ValueError):
        preflight.simulate()


def test_empty_document_has_a_real_successful_batch():
    preflight = CodemodPlanDocument(recipes=()).preflight(compiler())
    batch = preflight.required_edit_batch
    assert batch.edits == ()
    result = preflight.simulate()
    assert result.is_clean
    assert result.edit_batch is batch


def test_clean_report_without_compilation_is_rejected():
    preflight = CodemodPlanDocument(recipes=()).preflight(compiler())
    broken = replace(preflight, edit_batch=None)
    with pytest.raises(ValueError, match="compiled edit batch"):
        broken.simulate()


def test_document_simulation_does_not_reinvoke_recipe_operations(monkeypatch):
    preflight = CodemodPlanDocument(recipes=recipes()).preflight(compiler())
    calls = []
    original = PatchTargetOperation.source_edits

    def observe(operation, context):
        calls.append((operation, context))
        return original(operation, context)

    monkeypatch.setattr(PatchTargetOperation, "source_edits", observe)
    assert preflight.simulate().is_clean
    assert calls == []


def test_conflicting_edits_are_rejected_during_preflight_not_simulation():
    target = SourceRewriteTarget(file_path=PATH, qualname="Handler")
    conflicting = RefactorRecipe(
        "conflict",
        operations=tuple(
            PatchTargetOperation(
                target=target,
                replacements=(
                    SourceTextReplacement(
                        old_source="value = object",
                        new_source=replacement,
                    ),
                ),
            )
            for replacement in ("first = object", "second = object")
        ),
    )
    with pytest.raises(PhysicalSourceEditConflictError):
        CodemodPlanDocument(recipes=(conflicting,)).preflight(compiler())


def test_each_sequential_stage_retains_its_own_source_bound_batch():
    target = SourceRewriteTarget(file_path=PATH, qualname="Handler")
    plan = CodemodPlanSequence.from_operations(
        tuple(
            PatchTargetOperation(
                target=target,
                replacements=(
                    SourceTextReplacement(old_source=before, new_source=after),
                ),
            )
            for before, after in (
                ("value = object", "first = object"),
                ("first = object", "second = object"),
            )
        )
    )
    result = plan.simulate(compiler())
    first, second = result.stages
    assert first.edit_batch is not second.edit_batch
    assert first.edit_batch.compiler.sources_by_file_path[PATH] == SOURCE
    assert (
        second.edit_batch.compiler.sources_by_file_path[PATH]
        == first.simulation.rewritten_sources[PATH]
    )
    assert "second = object" in result.final_snapshot.sources_by_file_path[PATH]
