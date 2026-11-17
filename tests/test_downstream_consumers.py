"""A move keeps the imports of the checkouts that depend on the repository true.

Each case writes a producer project `a` and a consumer checkout `b` beside it,
moves a name out of `a`'s module, and runs `b` against the result. `b` is found
only by its own `pyproject.toml` naming `a`'s distribution: nothing in `a`
restates it. Unscanned, a consumer breaks while the move's plan stays clean,
which is how agent-comms moves broke toad's imports.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.refactors.class_overlap import relocation_to
from nominal_refactor_advisor.refactors.downstream_consumers import DownstreamConsumers

EXT = "class Update:\n    kind = 'update'\n\n\nclass Other:\n    kind = 'other'\n"
CONSUMER = "from pkg.ext import (\n    Other,\n    Update,\n)\n\nprint(Update.kind, Other.kind)\n"


def _write(root: Path, consumer: str, producer_tool: str = "", consumer_depends: str = '"a_project>=1"') -> None:
    files = {
        "a/pyproject.toml": f'[project]\nname = "a-project"\n{producer_tool}',
        "a/pkg/__init__.py": "",
        "a/pkg/ext.py": EXT,
        "b/pyproject.toml": f'[project]\nname = "b"\ndependencies = [{consumer_depends}]\n',
        "b/consumer.py": consumer,
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(root: Path) -> str:
    return subprocess.check_output(
        [sys.executable, "consumer.py"], cwd=root / "b", text=True, env={"PYTHONPATH": str(root / "a")}
    )


def _plan(root: Path):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root / "a", use_parse_cache=False))
    move = relocation_to(
        snapshot,
        SourceRewriteTarget(file_path=(root / "a/pkg/ext.py").as_posix(), qualname="Update"),
        "pkg.update",
        ("Update",),
    )
    return CodemodPlanSequence.from_operations((move,)), snapshot


def test_a_move_points_a_dependent_checkouts_import_at_the_new_home(tmp_path: Path):
    _write(tmp_path, CONSUMER)
    expected = _run(tmp_path)
    plan, snapshot = _plan(tmp_path)
    simulation = plan.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert (tmp_path / "b/consumer.py").read_text() == (
        "from pkg.ext import (\n    Other,\n)\nfrom pkg.update import (\n    Update,\n)\n\n"
        "print(Update.kind, Other.kind)\n"
    )
    assert _run(tmp_path) == expected


def test_a_consumer_reading_the_name_off_the_module_refuses_the_move(tmp_path: Path):
    consumer = "from pkg import ext\n\nprint(ext.Update.kind)\n"
    _write(tmp_path, consumer)
    plan, snapshot = _plan(tmp_path)
    with pytest.raises(ValueError, match="needs the consumer's change: .*consumer.py:3: reads `Update` off `pkg.ext` itself"):
        plan.simulate(snapshot)
    assert (tmp_path / "b/consumer.py").read_text() == consumer


def test_a_checkout_the_project_names_but_lacks_refuses_the_move(tmp_path: Path):
    _write(tmp_path, CONSUMER, producer_tool='[tool.uv.workspace]\nmembers = ["../missing"]\n')
    plan, snapshot = _plan(tmp_path)
    with pytest.raises(ValueError, match="needs the consumer's checkout: .*missing.* not checked out"):
        plan.simulate(snapshot)


def test_a_sibling_that_does_not_depend_on_the_project_is_not_a_consumer(tmp_path: Path):
    _write(tmp_path, CONSUMER, consumer_depends='"requests"')
    plan, snapshot = _plan(tmp_path)
    plan.simulate(snapshot).apply()
    assert (tmp_path / "b/consumer.py").read_text() == CONSUMER


def test_the_consumer_root_comes_only_from_its_own_pyproject_and_is_named(tmp_path: Path):
    """Nothing in `a` names `b`: `b`'s dependency on `a_project` makes it a
    consumer, and the edit says which consumer it rewrites and why."""

    _write(tmp_path, CONSUMER)
    consumers = DownstreamConsumers.of(tmp_path / "a/pkg/ext.py")
    assert (consumers.project, consumers.roots) == ("a-project", ((tmp_path / "b").resolve(),))
    [edit] = consumers.edits(object, "pkg.ext", {"Update": "pkg.update"})
    assert edit.file_path.endswith("b/consumer.py")
    assert f"Consumer `{(tmp_path / 'b').resolve()}` depends on 'a-project'" in edit.rationale
