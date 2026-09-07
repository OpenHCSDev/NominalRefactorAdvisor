"""Document revisions distinguish original source from virtual creation text."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    CreateFileOperation,
    RefactorRecipe,
    ReplaceDeclaredCallArgumentsOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)


@pytest.mark.parametrize("source", ("", "value = object\n"))
def test_created_source_has_absent_original_revision_even_when_virtual_text_matches(
    tmp_path: Path, source: str
) -> None:
    existing = (tmp_path / "existing.py").as_posix()
    created = (tmp_path / "created.py").as_posix()
    snapshot = CodemodSourceSnapshot.from_source_mapping({existing: "value = 1\n"})
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                "create",
                operations=(
                    CreateFileOperation(
                        target=SourceRewriteTarget(file_path=created), source=source
                    ),
                ),
            ),
        )
    )
    preflight = document.preflight(snapshot)
    result = preflight.simulate()
    assert result.is_clean
    assert created not in preflight.base_snapshot.sources_by_file_path
    assert result.edit_batch.compiler.sources_by_file_path[created] == source
    assert len(result.edit_batch.windows) == 1
    assert len(result.edit_batch.planned_rewrites) == 1
    assert result.simulation.changed_file_paths == (created,)
    assert result.simulation.rewritten_sources == {created: source}
    assert result.required_after_snapshot.sources_by_file_path[created] == source
    revision = result.simulation.base_revision_by_file_path[created]
    assert revision.source_hash is None
    assert revision.matches_source(None)
    assert not revision.matches_source(source)
    assert not Path(created).exists()


def test_existing_empty_module_has_present_original_revision_when_other_file_changes(
    tmp_path: Path,
) -> None:
    existing = (tmp_path / "existing_empty.py").as_posix()
    changed = (tmp_path / "changed.py").as_posix()
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {existing: "", changed: "def run(): return 1\n"}
    )
    result = CodemodPlanSequence.from_operations(
        (
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(file_path=changed, qualname="run"),
                body_source="return 2",
            ),
        )
    ).simulate(snapshot)
    assert result.is_clean
    stage = result.stage_reports[0].document_simulation
    revision = stage.simulation.base_revision_by_file_path[existing]
    assert revision.source_hash is not None
    assert revision.matches_source("")
    assert not revision.matches_source(None)
    assert existing not in stage.simulation.rewritten_sources
    assert stage.required_after_snapshot.sources_by_file_path[existing] == ""


def test_unchanged_read_dependency_is_covered_by_original_and_after_authorities(
    tmp_path: Path,
) -> None:
    caller = (tmp_path / "caller.py").as_posix()
    library = (tmp_path / "library.py").as_posix()
    library_source = "def render(value): return value\n"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {
            caller: "from library import render\ndef run(): return render(1)\n",
            library: library_source,
        }
    )
    result = CodemodPlanSequence.from_operations(
        (
            ReplaceDeclaredCallArgumentsOperation(
                target=SourceRewriteTarget(file_path=caller, qualname="run"),
                callee=SourceRewriteTarget(file_path=library, qualname="render"),
                arguments_source="value=1",
            ),
        )
    ).simulate(snapshot)
    assert result.is_clean
    stage = result.stage_reports[0].document_simulation
    assert stage.simulation.changed_file_paths == (caller,)
    assert set(stage.simulation.base_revision_by_file_path) == {caller, library}
    revision = stage.simulation.base_revision_by_file_path[library]
    assert revision.matches_source(library_source)
    assert not revision.matches_source(library_source.replace("value", "other"))
    assert library not in stage.simulation.rewritten_sources
    assert stage.required_after_snapshot.sources_by_file_path[library] == library_source
    assert "render(value=1)" in stage.simulation.rewritten_sources[caller]
    uncovered = (tmp_path / "uncovered.py").as_posix()
    with pytest.raises(KeyError):
        stage.simulation.base_revision_by_file_path[uncovered]


@pytest.mark.parametrize("source", ("", "value = object\n"))
def test_later_document_owns_created_source_without_retroactive_sequence_origin(
    tmp_path: Path, source: str
) -> None:
    caller = (tmp_path / "caller.py").as_posix()
    created = (tmp_path / "created.py").as_posix()
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {caller: "def run(): return 1\n"}
    )
    result = CodemodPlanSequence.from_operations(
        (
            CreateFileOperation(
                target=SourceRewriteTarget(file_path=created), source=source
            ),
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(file_path=caller, qualname="run"),
                body_source="return 2",
            ),
        )
    ).simulate(snapshot)
    assert result.is_clean
    first, second = (stage.document_simulation for stage in result.stage_reports)
    first_revision = first.simulation.base_revision_by_file_path[created]
    second_revision = second.simulation.base_revision_by_file_path[created]
    assert first_revision.matches_source(None)
    assert not first_revision.matches_source(source)
    assert second_revision.matches_source(source)
    assert not second_revision.matches_source(None)
    assert created not in second.simulation.rewritten_sources
    assert second.required_after_snapshot.sources_by_file_path[created] == source
    assert result.simulation.base_revision_by_file_path[created] == first_revision
    assert result.final_snapshot.sources_by_file_path[created] == source
    assert not Path(created).exists()


def test_generated_module_can_be_a_later_read_dependency_and_then_be_edited(
    tmp_path: Path,
) -> None:
    caller = (tmp_path / "caller.py").as_posix()
    generated = (tmp_path / "generated.py").as_posix()
    source = "def render(value): return value\n"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {caller: "from generated import render\ndef run(): return render(1)\n"}
    )
    target = SourceRewriteTarget(file_path=generated, qualname="render")
    result = CodemodPlanSequence.from_operations(
        (
            CreateFileOperation(
                target=SourceRewriteTarget(file_path=generated), source=source
            ),
            ReplaceDeclaredCallArgumentsOperation(
                target=SourceRewriteTarget(file_path=caller, qualname="run"),
                callee=target,
                arguments_source="value=1",
            ),
            ReplaceFunctionBodyOperation(target=target, body_source="return value + 1"),
        )
    ).simulate(snapshot)
    assert result.is_clean
    creation, dependency, change = (
        stage.document_simulation for stage in result.stage_reports
    )
    assert creation.simulation.base_revision_by_file_path[generated].matches_source(
        None
    )
    assert dependency.simulation.base_revision_by_file_path[generated].matches_source(
        source
    )
    assert change.simulation.base_revision_by_file_path[generated].matches_source(
        source
    )
    assert generated not in dependency.simulation.rewritten_sources
    assert "value + 1" in change.simulation.rewritten_sources[generated]
    assert result.simulation.base_revision_by_file_path[generated].source_hash is None
    assert not Path(generated).exists()
