"""Applying a proved edit must also validate the source it only read."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSimulationReport,
    CodemodSourceSnapshot,
    CreateFileOperation,
    ReplaceDeclaredCallArgumentsOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_source_edits import CodemodSourceRevisionError


def test_apply_rejects_changed_read_only_callee_before_writing(tmp_path: Path) -> None:
    library = tmp_path / "library.py"
    library.write_text("def render(value): return value\n", newline="")
    caller = tmp_path / "caller.py"
    source = "from library import render\ndef run(): return render(1)\n"
    caller.write_text(source, newline="")
    operation = ReplaceDeclaredCallArgumentsOperation(
        target=SourceRewriteTarget(file_path=str(caller), qualname="run"),
        callee=SourceRewriteTarget(file_path=str(library), qualname="render"),
        arguments_source="value=1",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = CodemodPlanSequence.from_operations((operation,)).simulate(snapshot)
    assert simulation.is_clean
    library.write_text("def render(other): return other\n", newline="")
    with pytest.raises(CodemodSourceRevisionError, match="library.py"):
        simulation.apply()
    assert caller.read_text() == source


def test_composed_reports_reject_a_changed_read_dependency(tmp_path: Path) -> None:
    path = tmp_path / "caller.py"
    dependency = tmp_path / "library.py"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {
            str(path): "def run(): return 1\n",
            str(dependency): "value = 1\n",
        }
    )
    operation = ReplaceFunctionBodyOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="run"),
        body_source="return 2",
    )
    first = CodemodPlanSequence.from_operations((operation,)).simulate(snapshot)
    drifted = first.final_snapshot.with_virtual_sources(
        {str(dependency): "value = 3\n"}
    )
    second = CodemodPlanSequence.from_operations(
        (replace(operation, body_source="return 3"),)
    ).simulate(drifted)
    with pytest.raises(ValueError, match="stale source transition"):
        CodemodSimulationReport.from_sequential_reports(
            (first.simulation, second.simulation)
        )


def test_read_only_source_is_validated_but_never_written(tmp_path: Path) -> None:
    path = tmp_path / "caller.py"
    path.write_text("def run(): return 1\n", newline="")
    dependency = tmp_path / "library.py"
    dependency.write_text("value = 1\n", newline="")
    original_stat = dependency.stat()
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = ReplaceFunctionBodyOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="run"),
        body_source="return 2",
    )
    simulation = CodemodPlanSequence.from_operations((operation,)).simulate(snapshot)
    revisions = simulation.simulation.base_revision_by_file_path
    assert set(revisions) == {path.as_posix(), dependency.as_posix()}
    assert simulation.apply() == (path.as_posix(),)
    assert dependency.read_text() == "value = 1\n"
    assert dependency.stat().st_mtime_ns == original_stat.st_mtime_ns


def test_created_file_can_be_read_and_changed_in_later_stages(tmp_path: Path) -> None:
    caller = tmp_path / "caller.py"
    caller.write_text(
        "from generated import render\ndef run(): return render(1)\nprint(run())\n", newline=""
    )
    generated = tmp_path / "generated.py"
    generated_target = SourceRewriteTarget(file_path=str(generated), qualname="render")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    sequence = CodemodPlanSequence.from_operations(
        (
            CreateFileOperation(
                target=SourceRewriteTarget(file_path=str(generated)),
                source="def render(value): return value\n",
            ),
            ReplaceDeclaredCallArgumentsOperation(
                target=SourceRewriteTarget(file_path=str(caller), qualname="run"),
                callee=generated_target,
                arguments_source="value=1",
            ),
            ReplaceFunctionBodyOperation(
                target=generated_target, body_source="return value + 1"
            ),
        )
    )
    simulation = sequence.simulate(snapshot)
    assert simulation.is_clean
    assert (
        simulation.simulation.base_revision_by_file_path[
            generated.as_posix()
        ].source_hash
        is None
    )
    assert not generated.exists()
    assert set(simulation.apply()) == {caller.as_posix(), generated.as_posix()}
    assert subprocess.check_output([sys.executable, str(caller)], text=True) == "2\n"


def test_revision_record_still_requires_every_write_precondition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "caller.py"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {str(path): "def run(): return 1\n"}
    )
    operation = ReplaceFunctionBodyOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="run"),
        body_source="return 2",
    )
    report = (
        CodemodPlanSequence.from_operations((operation,)).simulate(snapshot).simulation
    )
    with pytest.raises(ValueError, match="cover every changed file"):
        replace(report, base_revisions=())
