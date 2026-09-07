"""Class members retain declaration-owned exact geometry before lowering."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    InsertClassMemberOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_declaration_source import ClassMemberInsertion
from nominal_refactor_advisor.codemod_runtime import RefactorRecipeOperationCompiler
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextGeometry,
    SourceTextSpan,
)

PATH = Path(__file__).with_name("member_fixture.py").resolve().as_posix()


@pytest.mark.parametrize("newline", ("\n", "\r\n", ""))
@pytest.mark.parametrize("inline", (False, True))
def test_exact_member_geometry_agrees_with_actual_lowering(newline, inline):
    suite = " Ω = object" if inline else (newline or "\n") + "    Ω = object"
    source = "class Handler:" + suite + newline
    compiler = RefactorRecipeOperationCompiler.from_context(
        CodemodSourceSnapshot.from_source_mapping({PATH: source})
    )
    operation = InsertClassMemberOperation(
        target=SourceRewriteTarget(file_path=PATH, qualname="Handler"),
        source="added = 1\n",
    )
    batch = compiler.edit_batch_for_recipes(
        (RefactorRecipe("member", operations=(operation,)),)
    )
    (edit,) = batch.edits
    assert isinstance(edit, ClassMemberInsertion)
    mutation = edit.source_mutation(compiler)
    assert mutation.revision.matches_source(source)
    assert mutation.origins is edit.origins
    assert mutation.contributors is edit.contributors
    assert mutation.resolved_edits(compiler) == edit.resolved_edits(compiler)
    result = compiler.simulate_rewrites(batch.planned_rewrites)
    rewritten = result.rewritten_sources[PATH]
    geometry = SourceTextGeometry(source)
    assert (
        geometry.source_with_replacements_in_span(0, len(source), mutation.replacements)
        == rewritten
    )
    original_read = ast.parse(source).body[0].body[0].value
    span = SourceTextSpan(*geometry.required_node_offsets(original_read))
    if inline:
        # Expanding the suite reproduces its old statements as replacement text.
        with pytest.raises(ValueError, match="unchanged-text correspondence"):
            geometry.project_unchanged_spans((span,), mutation.replacements)
    else:
        (retained,) = geometry.project_unchanged_spans((span,), mutation.replacements)
        assert rewritten[retained.start_offset : retained.end_offset] == "object"


def test_exact_member_projection_rejects_an_existing_binding():
    source = "class Handler:\n    existing = object\n"
    compiler = RefactorRecipeOperationCompiler.from_context(
        CodemodSourceSnapshot.from_source_mapping({PATH: source})
    )
    operation = InsertClassMemberOperation(
        target=SourceRewriteTarget(file_path=PATH, qualname="Handler"),
        source="existing = 1\n",
    )
    (edit,) = operation.source_edits(compiler)
    with pytest.raises(ValueError, match="already binds members"):
        edit.source_mutation(compiler)
