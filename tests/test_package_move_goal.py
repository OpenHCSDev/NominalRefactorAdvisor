"""`package_move` reads a destination table and proposes the staged move plan:
relocate every declared name, dissolve the shell, top import layer first."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

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
    markdown = report.to_markdown()
    assert "Dead end at depth 0, targets by reason:" in markdown
    assert "session_load.py import each other" in markdown


def test_a_merge_row_folds_one_module_into_the_other(tmp_path: Path) -> None:
    _cyclic_pair(tmp_path)
    report = _goal(
        tmp_path,
        "| `planning` | `presentation.py`; `session_load.py` merges into "
        "`presentation.py` | |\n",
    )

    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED, report.to_markdown()
    operations = [
        (operation.operation_key(), Path(operation.target.file_path).name)
        for document in report.replay_sequence.documents
        for recipe in document.recipes
        for operation in recipe.operations
    ]
    assert operations == [
        ("relocate_symbols_to_new_module", "presentation.py"),
        ("dissolve_module", "presentation.py"),
        ("relocate_symbols_to_module", "session_load.py"),
        ("dissolve_module", "session_load.py"),
    ]
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
    assert imported.stdout.strip() == "<loaded>"
    merged = (tmp_path / "pkg/planning/presentation.py").read_text()
    assert "session_load" not in merged


def test_a_package_shadowed_by_a_module_file_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/messaging.py", "def send():\n    return 'sent'\n")
    _write(tmp_path, "pkg/bus.py", "class Bus:\n    n = 1\n")
    _write(tmp_path, "pkg/app.py", "from pkg.bus import Bus\nVALUE = Bus.n\n")
    report = _goal(tmp_path, "| `messaging` | `bus.py` | |\n")

    assert report.stages == ()
    reasons = " ".join(
        record.evaluation.rejection_reason
        for dead_end in report.trajectory_proof.dead_ends
        for record in dead_end.class_plan_report.finding_plan.records
    )
    assert "pkg/messaging.py' is the module of that name" in reasons


def test_a_consumer_root_holding_none_of_the_tables_packages_keeps_its_modules(
    tmp_path: Path,
) -> None:
    """Toad scanned beside agent-comms: its own `thread_identity.py` shares a name with a
    row's module, but Toad holds none of the table's packages, so only its import moves."""

    _layered_package(tmp_path)
    _write(tmp_path, "consumer/__init__.py", "")
    _write(
        tmp_path,
        "consumer/steps.py",
        "from pkg.steps import Step\n\nCONSUMED = Step.plan.phases\n",
    )
    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(package_destination_tables=(TABLE,)),
        parse_workers=1,
        dry_run=True,
        migration_type=PackageMoveConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()

    assert report.stop_reason.completed, report.to_markdown()
    targets = {
        Path(operation.target.file_path).relative_to(tmp_path).as_posix()
        for document in report.replay_sequence.documents
        for recipe in document.recipes
        for operation in recipe.operations
    }
    assert targets == {"pkg/goals.py", "pkg/steps.py", "pkg/plans.py"}
    replay = report.replay_sequence.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert replay.is_clean
    replay.apply()
    imported = subprocess.run(
        [sys.executable, "-c", "from consumer.steps import CONSUMED; print(CONSUMED)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert "pkg.planning.steps" in (tmp_path / "consumer/steps.py").read_text()


def _run_table(root: Path):
    return CodemodRefactorGoalRunner(
        roots=(root,),
        config=DetectorConfig(package_destination_tables=(TABLE,)),
        parse_workers=1,
        dry_run=True,
        migration_type=PackageMoveConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()


def test_a_root_holding_fewer_of_the_tables_packages_keeps_its_modules(tmp_path: Path) -> None:
    """agent-comms holds most of its table's packages; `tools` holds one (a `planning`
    beside its own flat modules), and still keeps its `steps.py`."""

    _layered_package(tmp_path)
    _write(tmp_path, "pkg/nominal/__init__.py", "")
    _write(tmp_path, "tools/__init__.py", "")
    _write(tmp_path, "tools/planning/__init__.py", "")
    _write(tmp_path, "tools/steps.py", "from pkg.steps import Step\n\nSTEP = Step\n")

    report = _run_table(tmp_path)

    assert report.stop_reason.completed, report.to_markdown()
    assert {
        Path(operation.target.file_path).relative_to(tmp_path).as_posix()
        for document in report.replay_sequence.documents
        for recipe in document.recipes
        for operation in recipe.operations
    } == {"pkg/goals.py", "pkg/steps.py", "pkg/plans.py"}


def test_two_roots_holding_equally_many_of_the_tables_packages_are_refused(
    tmp_path: Path,
) -> None:
    _layered_package(tmp_path)
    _write(tmp_path, "tools/__init__.py", "")
    _write(tmp_path, "tools/planning/__init__.py", "")
    _write(tmp_path, "tools/steps.py", "STEP = 1\n")

    with pytest.raises(ValueError, match="split evenly between pkg and tools"):
        _run_table(tmp_path)


def test_a_table_whose_packages_no_root_holds_is_refused_by_name(tmp_path: Path) -> None:
    _layered_package(tmp_path)
    (tmp_path / "pkg/planning/__init__.py").unlink()
    (tmp_path / "pkg/planning").rmdir()

    with pytest.raises(ValueError, match="no scanned root holds any of the table's packages: nominal, planning"):
        _run_table(tmp_path)


def test_the_stage_budget_comes_from_the_table_and_a_budget_stop_lands_its_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    from nominal_refactor_advisor.codemod_workflow import (
        CodemodRefactorTrajectoryBudget,
        CodemodRefactorTrajectoryStatus,
    )

    _layered_package(tmp_path)

    def run(max_depth: int):
        return CodemodRefactorGoalRunner(
            roots=(tmp_path,),
            config=DetectorConfig(package_destination_tables=(TABLE,)),
            parse_workers=1,
            dry_run=True,
            migration_type=PackageMoveConcept,
            trajectory_budget=CodemodRefactorTrajectoryBudget(max_depth=max_depth),
            guard_suite=ArchitectureGuardSuite(),
        ).run()

    lifted = run(1)
    assert lifted.stop_reason is CodemodWorkflowStopReason.ACHIEVED
    assert lifted.trajectory_proof.budget.max_depth == 6
    assert lifted.stage_count == 6

    monkeypatch.setattr(
        PackageMoveConcept,
        "trajectory_depth",
        classmethod(lambda cls, target_count, declared: declared),
    )
    prefix = run(2)
    assert prefix.stop_reason is CodemodWorkflowStopReason.PROVED_PREFIX
    assert prefix.stop_reason.completed
    assert prefix.trajectory_proof.status is CodemodRefactorTrajectoryStatus.PREFIX_PROVED
    assert [
        Path(operation.target.file_path).name
        for document in prefix.replay_sequence.documents
        for recipe in document.recipes
        for operation in recipe.operations
    ] == ["goals.py", "goals.py"]
    assert len(prefix.final_target_finding_ids) == 2
    assert "2 targets left for the next run" in prefix.to_markdown()
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert prefix.replay_sequence.simulate(snapshot).is_clean


def test_a_module_name_glued_in_an_f_string_follows_the_move(tmp_path: Path) -> None:
    _layered_package(tmp_path)
    _write(
        tmp_path,
        "pkg/check_imports.py",
        "import pytest\n\n\n"
        '@pytest.mark.parametrize("module", ("goals", "app"))\n'
        "def test_imports(module):\n"
        '    __import__(f"pkg.{module}")\n\n\n'
        'PROGRAMS = [f"import pkg.{name}" for name in ("steps", "app")]\n',
    )
    report = _goal(tmp_path, TABLE)
    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED, report.to_markdown()
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    replay = report.replay_sequence.simulate(snapshot)
    assert replay.is_clean
    replay.apply()
    text = (tmp_path / "pkg/check_imports.py").read_text()
    assert '("planning.goals", "app")' in text
    assert '("planning.steps", "app")' in text


def test_a_glue_nra_cannot_trace_refuses_by_name(tmp_path: Path) -> None:
    _layered_package(tmp_path)
    _write(
        tmp_path,
        "pkg/check_imports.py",
        'NAMES = ("steps",)\n\n\ndef load(module):\n    return f"import pkg.{module}"\n',
    )
    report = _goal(tmp_path, TABLE)
    import json

    from nominal_refactor_advisor.json_reports import json_report_object

    assert report.stop_reason is not CodemodWorkflowStopReason.ACHIEVED
    assert "glues 'pkg.' onto 'module', which NRA cannot trace, and the file spells 'steps'" in (
        json.dumps(json_report_object(report), default=str)
    )


def test_a_cycle_only_through_a_local_import_moves_and_names_its_debt(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/planning/__init__.py", "")
    _write(
        tmp_path,
        "pkg/session_load.py",
        "import pkg.presentation as presentation\n\n\n"
        "def load() -> str:\n    return presentation.render('loaded')\n",
    )
    _write(
        tmp_path,
        "pkg/presentation.py",
        "def render(text: str) -> str:\n    return f'<{text}>'\n\n\n"
        "def reload() -> str:\n"
        "    from pkg.session_load import load\n\n"
        "    return load()\n",
    )
    _write(tmp_path, "pkg/app.py", "from pkg.presentation import reload\n\nVALUE = reload()\n")
    table = TABLE.replace("`steps.py`", "`session_load.py`, `presentation.py`")
    report = _goal(tmp_path, table)

    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED, report.to_markdown()
    reasons = [
        recipe.reason
        for document in report.replay_sequence.documents
        for recipe in document.recipes
    ]
    assert reasons[0].startswith("Relocate") and "session_load" in str(
        report.replay_sequence.documents[0].recipes[0].operations[0].target.file_path
    )
    assert any("Debt carried:" in reason and "presentation.py" in reason for reason in reasons)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    replay = report.replay_sequence.simulate(snapshot)
    assert replay.is_clean
    replay.apply()
    imported = subprocess.run(
        [sys.executable, "-c", "from pkg.app import VALUE; print(VALUE)"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == "<loaded>"


def test_a_merge_into_a_renamed_module_moves_its_target_first(tmp_path: Path) -> None:
    """The table drops a prefix, so no stem matches its destination's; the merge
    row alone says which module creates the shared destination."""

    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/execution/__init__.py", "")
    _write(tmp_path, "pkg/selected_claim.py", "class Claim:\n    held = 1\n")
    _write(tmp_path, "pkg/selected_write_plan.py", "class WritePlan:\n    steps = 2\n")
    _write(
        tmp_path,
        "pkg/selected_actions.py",
        "from pkg.selected_claim import Claim\n"
        "from pkg.selected_write_plan import WritePlan\n\n\n"
        "class Action:\n    claim = Claim\n    plan = WritePlan\n",
    )
    _write(
        tmp_path,
        "pkg/selected_write_authority.py",
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n    from pkg.selected_actions import Action\n\n\n"
        "class WriteAuthority:\n"
        "    def grant(self, action) -> int:\n        return action.claim.held\n",
    )
    _write(
        tmp_path,
        "pkg/app.py",
        "from pkg.selected_actions import Action\n"
        "from pkg.selected_write_authority import WriteAuthority\n\n"
        "VALUE = (WriteAuthority().grant(Action()), Action.plan.steps)\n",
    )
    table = """
| Package | Existing owners | What belongs here |
|---|---|---|
| `execution` | `selected_actions` moves to `execution/actions`, `selected_claim` moves to `execution/claim`, `selected_write_plan` moves to `execution/write_plan`; `selected_write_authority.py` merges into `selected_actions.py` | Selected turns. |
"""
    report = _goal(tmp_path, table)

    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED, report.to_markdown()
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    replay = report.replay_sequence.simulate(snapshot)
    assert replay.is_clean
    replay.apply()
    assert sorted(p.name for p in (tmp_path / "pkg").glob("*.py")) == ["__init__.py", "app.py"]
    imported = subprocess.run(
        [sys.executable, "-c", "from pkg.app import VALUE; print(VALUE)"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == "(1, 2)"


def _module_with_helper_data(root: Path) -> None:
    """`fork.py` reads `_helpers/fork.mjs` beside itself, its only reader."""

    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/planning/__init__.py", "")
    _write(root, "pkg/_helpers/fork.mjs", "export const fork = 'forked';\n")
    _write(
        root,
        "pkg/fork.py",
        "from pathlib import Path\n\n\nclass Fork:\n"
        "    script = Path(__file__).with_name('_helpers') / 'fork.mjs'\n\n"
        "    @classmethod\n    def text(cls) -> str:\n        return cls.script.read_text()\n",
    )
    _write(root, "pkg/app.py", "from pkg.fork import Fork\n\nVALUE = Fork.text().strip()\n")


def test_a_module_carries_the_data_it_reads_beside_itself(tmp_path: Path) -> None:
    _module_with_helper_data(tmp_path)
    report = _goal(tmp_path, "| `planning` | `fork.py` | |\n")

    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED, report.to_markdown()
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    replay = report.replay_sequence.simulate(snapshot)
    assert replay.is_clean
    replay.apply()
    assert not (tmp_path / "pkg/_helpers").exists() or not any((tmp_path / "pkg/_helpers").iterdir())
    assert (tmp_path / "pkg/planning/_helpers/fork.mjs").read_text() == "export const fork = 'forked';\n"
    imported = subprocess.run(
        [sys.executable, "-c", "from pkg.app import VALUE; print(VALUE)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == "export const fork = 'forked';"


def test_data_another_module_also_reads_stays_and_refuses(tmp_path: Path) -> None:
    _module_with_helper_data(tmp_path)
    _write(
        tmp_path,
        "pkg/other.py",
        "from pathlib import Path\n\nSCRIPT = Path(__file__).with_name('_helpers')\n",
    )
    report = _goal(tmp_path, "| `planning` | `fork.py` | |\n")

    assert report.stages == ()
    assert "__file__ (" in report.to_markdown() and "also names" in report.to_markdown()


def test_entry_points_naming_two_moved_modules_follow_both_moves(tmp_path: Path) -> None:
    """One manifest names a function in each of two moved modules: every stage
    reads the manifest as the stages before it left it, so both entries follow and
    no later stage refuses a name its module no longer binds."""

    _write(
        tmp_path,
        "pyproject.toml",
        '[project]\nname = "pkg"\n\n[project.scripts]\n'
        'pkg-a = "pkg.a:main"\npkg-b = "pkg.b:main"\n',
    )
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/planning/__init__.py", "")
    for name, value in (("a", 1), ("b", 2)):
        _write(tmp_path, f"pkg/{name}.py", f"def main() -> int:\n    return {value}\n")
    report = _goal(tmp_path, "| `planning` | `a.py`, `b.py` | |\n")

    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED, report.to_markdown()
    replay = report.replay_sequence.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert replay.is_clean
    replay.apply()
    assert (tmp_path / "pyproject.toml").read_text() == (
        '[project]\nname = "pkg"\n\n[project.scripts]\n'
        'pkg-a = "pkg.planning.a:main"\npkg-b = "pkg.planning.b:main"\n'
    )
