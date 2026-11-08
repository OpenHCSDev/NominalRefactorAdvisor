"""`package_move` reads a destination table and proposes the staged move plan:
relocate every declared name, dissolve the shell, top import layer first."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    ArchitectureGuardSuite,
    CodemodSourceSnapshot,
)
from nominal_refactor_advisor.codemod_workflow import (
    CodemodRefactorGoalRunner,
    CodemodWorkflowStopReason,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.refactor_concepts import PackageMoveConcept

TABLE = """
| Package | Existing owners and initial source closure | What belongs here |
|---|---|---|
| `planning` | Goal records, `steps.py`, Plan and its phases | Persistent goals. |
| `nominal` | Nothing flat today | Declarations. |
"""


def _write(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _layered_package(root: Path) -> None:
    """goals imports steps imports plans: three layers, bottom-up would refuse."""

    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/planning/__init__.py", "")
    _write(root, "pkg/plans.py", '"""Plans."""\n\nclass Plan:\n    phases = 2\n')
    _write(
        root,
        "pkg/steps.py",
        "from .plans import Plan\n\n__all__ = ('Step',)\n\n"
        "class Step:\n    plan = Plan\n\n\ndef step_count() -> int:\n    return Plan.phases\n",
    )
    _write(
        root,
        "pkg/goals.py",
        "from .steps import Step, step_count\n\n\nclass Goal:\n    step = Step\n\n"
        "LIMIT = step_count()\n",
    )
    _write(
        root,
        "pkg/app.py",
        "from pkg.goals import Goal, LIMIT\nfrom pkg import plans\n\n"
        "VALUE = (Goal.step.plan.phases, LIMIT, plans.Plan.__module__)\n",
    )


def test_package_move_plans_top_layer_first_from_live_sources(tmp_path: Path) -> None:
    _layered_package(tmp_path)
    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(package_destination_tables=(TABLE,)),
        parse_workers=1,
        dry_run=True,
        migration_type=PackageMoveConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()

    assert report.stop_reason.completed, report.to_markdown()
    operations = [
        (operation.operation_key(), Path(operation.target.file_path).name)
        for document in report.replay_sequence.documents
        for recipe in document.recipes
        for operation in recipe.operations
    ]
    assert operations == [
        ("relocate_symbols_to_new_module", "goals.py"),
        ("dissolve_module", "goals.py"),
        ("relocate_symbols_to_new_module", "steps.py"),
        ("dissolve_module", "steps.py"),
        ("relocate_symbols_to_new_module", "plans.py"),
        ("dissolve_module", "plans.py"),
    ]
    relocate_steps = report.replay_sequence.documents[2].recipes[0].operations[0]
    assert relocate_steps.symbol_qualnames == ("__all__", "Step", "step_count")
    assert relocate_steps.destination_path.endswith("pkg/planning/steps.py")

    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    replay = report.replay_sequence.simulate(snapshot)
    assert replay.is_clean
    replay.apply()
    assert sorted(p.name for p in (tmp_path / "pkg").glob("*.py")) == ["__init__.py", "app.py"]
    imported = subprocess.run(
        [sys.executable, "-c", "from pkg.app import VALUE; print(VALUE)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == "(2, 2, 'pkg.planning.plans')"


def test_no_table_no_findings(tmp_path: Path) -> None:
    _layered_package(tmp_path)
    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        migration_type=PackageMoveConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()
    assert report.stages == ()


def _cyclic_pair(root: Path) -> None:
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/planning/__init__.py", "")
    _write(
        root,
        "pkg/session_load.py",
        "import pkg.presentation as presentation\n\n\n"
        "def load() -> str:\n    return presentation.render('loaded')\n",
    )
    _write(
        root,
        "pkg/presentation.py",
        "import pkg.session_load as session_load\n\n\n"
        "def render(text: str) -> str:\n    return f'<{text}>'\n\n\n"
        "def reload() -> str:\n    return session_load.load()\n",
    )
    _write(root, "pkg/app.py", "from pkg.presentation import reload\n\nVALUE = reload()\n")


def _goal(root: Path, table: str):
    return CodemodRefactorGoalRunner(
        roots=(root,),
        config=DetectorConfig(package_destination_tables=(table,)),
        parse_workers=1,
        dry_run=True,
        migration_type=PackageMoveConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()


def test_modules_importing_each_other_are_refused_by_name(tmp_path: Path) -> None:
    _cyclic_pair(tmp_path)
    report = _goal(
        tmp_path,
        "| `planning` | `session_load.py`, `presentation.py` | |\n",
    )

    assert report.stages == ()
    reasons = " ".join(
        record.evaluation.rejection_reason
        for terminal in report.trajectory_proof.dead_ends
        for record in terminal.class_plan_report.finding_plan.records
    )
    assert "presentation.py and" in reasons and "session_load.py import each other" in reasons
    assert "merges into" in reasons


def test_a_merge_row_orders_the_target_first_and_merges_into_its_destination(
    tmp_path: Path,
) -> None:
    _cyclic_pair(tmp_path)
    report = _goal(
        tmp_path,
        "| `planning` | `presentation.py`; `session_load.py` merges into "
        "`presentation.py` | |\n",
    )

    operations = [
        (operation.operation_key(), Path(operation.target.file_path).name)
        for document in report.replay_sequence.documents
        for recipe in document.recipes
        for operation in recipe.operations
    ]
    assert operations[:3] == [
        ("relocate_symbols_to_new_module", "presentation.py"),
        ("dissolve_module", "presentation.py"),
        ("relocate_symbols_to_module", "session_load.py"),
    ]
    merge = report.replay_sequence.documents[2].recipes[0].operations[0]
    assert merge.destination_path.endswith("pkg/planning/presentation.py")
    # The merged module still reads session_load through its old module alias,
    # so the shell's dissolve is refused rather than planned broken.
    assert report.stop_reason is CodemodWorkflowStopReason.PARTIALLY_ACHIEVED
