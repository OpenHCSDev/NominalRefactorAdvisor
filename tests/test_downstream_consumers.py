"""A move keeps the imports of the checkouts a repository declares downstream true.

Each case writes a producer repository `a` and a consumer checkout `b` beside it,
moves a name out of `a`'s module, and runs `b` against the result. Without the
declaration the consumer is not scanned at all, which is how agent-comms moves
broke toad's imports while their own plans stayed clean.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.refactors.class_overlap import relocation_to

EXT = "class Update:\n    kind = 'update'\n\n\nclass Other:\n    kind = 'other'\n"
CONSUMER = "from pkg.ext import (\n    Other,\n    Update,\n)\n\nprint(Update.kind, Other.kind)\n"


def _write(root: Path, consumer: str, declared: str = '["../b"]') -> None:
    files = {
        "a/pyproject.toml": f"[tool.nominal-refactor-advisor]\ndownstream-roots = {declared}\n",
        "a/pkg/__init__.py": "",
        "a/pkg/ext.py": EXT,
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


def test_a_move_points_a_declared_consumers_import_at_the_new_home(tmp_path: Path):
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


def test_a_declared_root_that_is_not_checked_out_refuses_the_move(tmp_path: Path):
    _write(tmp_path, CONSUMER, declared='["../missing"]')
    plan, snapshot = _plan(tmp_path)
    with pytest.raises(ValueError, match="declared a downstream root but is not checked out"):
        plan.simulate(snapshot)
