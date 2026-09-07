"""Actual lowering laws which a retained-origin projection must preserve."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    CreateFileOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_runtime import (
    NominalEditBatch,
    RefactorRecipeOperationCompiler,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceFileCreation,
    SourceInsertion,
    SourceSpanReplacement,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)

PATH = Path(__file__).with_name("origins_fixture.py").resolve().as_posix()
SOURCE = "left = object; right = type\n"


def owner():
    return RefactorRecipeOperationCompiler.from_context(
        CodemodSourceSnapshot.from_source_mapping({PATH: SOURCE})
    )


def exact(*replacements):
    return SourceTextGeometry(SOURCE).nominal_edit(
        file_path=PATH, replacements=tuple(replacements)
    )


def render(*edits):
    compiler = owner()
    batch = NominalEditBatch(compiler, tuple(edits))
    result = compiler.simulate_rewrites(batch.planned_rewrites)
    assert result.parse_valid
    return batch, result.rewritten_sources[PATH]


@pytest.mark.parametrize("reverse", (False, True))
def test_distinct_mixed_insertions_share_an_anchor_in_actual_lowering(reverse):
    text = SourceTextSpanReplacement(0, 0, replacement_source="import typing\n")
    line = SourceInsertion(
        file_path=PATH, insertion_line=1, inserted_lines=("import builtins\n",)
    )
    edits = (line, exact(text)) if reverse else (exact(text), line)
    batch, output = render(*edits)
    prefixes = (
        ("import builtins\n", "import typing\n")
        if reverse
        else ("import typing\n", "import builtins\n")
    )
    assert output == "".join(prefixes) + SOURCE
    assert len(batch.physical_edits) == 1
    # Flattening peer payloads is not the physical insertion composition law.
    with pytest.raises(ValueError, match="different source to the same span"):
        SourceTextGeometry(SOURCE).replacements_in_span(
            0,
            len(SOURCE),
            (
                text,
                SourceTextSpanReplacement(0, 0, replacement_source="import builtins\n"),
            ),
        )


def test_identical_mixed_insertions_are_deduplicated_by_existing_owner():
    prefix = "import typing\n"
    batch, output = render(
        exact(SourceTextSpanReplacement(0, 0, replacement_source=prefix)),
        SourceInsertion(file_path=PATH, insertion_line=1, inserted_lines=(prefix,)),
    )
    assert output == prefix + SOURCE
    assert len(batch.edits) == 2
    assert len(batch.physical_edits) == 1


def test_two_phase_lowering_preserves_declaration_group_encounter_order():
    first = SourceInsertion(
        file_path=PATH, insertion_line=1, inserted_lines=("import typing\n",)
    )
    middle = exact(
        SourceTextSpanReplacement(0, 0, replacement_source="import collections\n")
    )
    last = SourceInsertion(
        file_path=PATH, insertion_line=1, inserted_lines=("import builtins\n",)
    )
    batch, output = render(first, middle, last)
    assert batch.edits == (first, middle, last)
    assert len(batch.physical_edits) == 1
    # Physical peers merge in the first declaration pass, before exact lowering.
    assert output == "import typing\nimport builtins\nimport collections\n" + SOURCE


def test_disjoint_exact_and_line_edits_have_a_complete_fine_projection():
    replacement = SourceTextSpanReplacement(0, 4, replacement_source="renamed")
    prefix = "import typing\n"
    _, output = render(
        exact(replacement),
        SourceInsertion(file_path=PATH, insertion_line=1, inserted_lines=(prefix,)),
    )
    geometry = SourceTextGeometry(SOURCE)
    complete = (SourceTextSpanReplacement(0, 0, replacement_source=prefix), replacement)
    assert geometry.source_with_replacements_in_span(0, len(SOURCE), complete) == output
    read = SourceTextSpan(SOURCE.index("type"), SOURCE.index("type") + 4)
    (retained,) = geometry.project_unchanged_spans((read,), complete)
    assert output[retained.start_offset : retained.end_offset] == "type"


def test_equal_opaque_replacement_does_not_supply_fine_read_origin():
    replacement = SourceTextSpanReplacement(0, 4, replacement_source="renamed")
    rewritten = SOURCE.replace("left", "renamed")
    batch, output = render(
        exact(replacement),
        SourceSpanReplacement(
            file_path=PATH,
            start_line=1,
            end_line=1,
            replacement_lines=(rewritten,),
        ),
    )
    assert output == rewritten
    assert len(batch.edits) == 2
    assert len(batch.physical_edits) == 1
    read = SourceTextSpan(SOURCE.index("type"), SOURCE.index("type") + 4)
    geometry = SourceTextGeometry(SOURCE)
    assert geometry.project_unchanged_spans((read,), (replacement,))
    opaque = SourceTextSpanReplacement(0, len(SOURCE), replacement_source=rewritten)
    with pytest.raises(ValueError, match="unchanged-text correspondence"):
        geometry.project_unchanged_spans((read,), (opaque,))


def test_virtual_creation_source_is_generated_despite_existing_at_compile_time():
    snapshot = CodemodSourceSnapshot.from_source_mapping({PATH: SOURCE})
    new_path = Path(PATH).with_name("created_fixture.py").as_posix()
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                "create",
                operations=(
                    CreateFileOperation(
                        target=SourceRewriteTarget(file_path=new_path),
                        source="value = object\n",
                    ),
                ),
            ),
        )
    )
    preflight = document.preflight(snapshot)
    result = preflight.simulate()
    assert result.is_clean
    assert new_path not in preflight.base_snapshot.sources_by_file_path
    assert (
        result.edit_batch.compiler.sources_by_file_path[new_path] == "value = object\n"
    )
    assert any(isinstance(edit, SourceFileCreation) for edit in result.edit_batch.edits)
