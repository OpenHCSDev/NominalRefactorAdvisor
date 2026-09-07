"""The compiler retains original records under the source that produced them."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    EnsureImportOperation,
    PatchTargetOperation,
    RefactorRecipe,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.codemod_runtime import RefactorRecipeOperationCompiler

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
