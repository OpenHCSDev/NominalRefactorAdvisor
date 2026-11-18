"""A move rewrites admitted source bindings without acquiring sibling branches."""

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_module_roots, parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._hidden_import_cuts import HiddenImportCycleDetector
from nominal_refactor_advisor.refactors.class_overlap import relocation_to
from nominal_refactor_advisor.scan_cache import ScanCache

EXT = "class Update:\n    kind = 'update'\n\n\nclass Other:\n    kind = 'other'\n"
CONSUMERS = (
    "from pkg.ext import Other, Update\nprint(Update.kind, Other.kind)\n",
    "from pkg.ext import Update as U, Other\nprint(U.kind, Other.kind)\n",
    "from pkg import ext\nprint(ext.Update.kind, ext.Other.kind)\n",
)


def _write(root: Path, consumer: str, dependency: str = 'a_project>=1', producer_tool: str = '') -> None:
    files = {
        'a/pyproject.toml': f'[project]\nname = "a-project"\n{producer_tool}',
        'a/pkg/__init__.py': '',
        'a/pkg/ext.py': EXT,
        'b/pyproject.toml': f'[project]\nname = "b"\ndependencies = ["{dependency}"]\n',
        'b/consumer.py': consumer,
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(root: Path) -> str:
    return subprocess.check_output(
        [sys.executable, 'consumer.py'], cwd=root / 'b', text=True,
        env={'PYTHONPATH': str(root / 'a')},
    )


def _snapshot(root: Path, *, include_consumer: bool) -> CodemodSourceSnapshot:
    roots = (root / 'a', root / 'b') if include_consumer else (root / 'a',)
    return CodemodSourceSnapshot.from_modules(parse_python_module_roots(roots, use_parse_cache=False))


def _plan(root: Path, *, include_consumer: bool):
    snapshot = _snapshot(root, include_consumer=include_consumer)
    move = relocation_to(
        snapshot,
        SourceRewriteTarget(file_path=(root / 'a/pkg/ext.py').as_posix(), qualname='Update'),
        'pkg.update', ('Update',),
    )
    return CodemodPlanSequence.from_operations((move,)), snapshot


@pytest.mark.parametrize('consumer', CONSUMERS)
@pytest.mark.parametrize('dependency', ('a_project>=1', 'requests'))
@pytest.mark.parametrize('include_consumer', (False, True))
def test_selected_graph_owns_consumer_rewrites(tmp_path: Path, consumer: str, dependency: str, include_consumer: bool):
    _write(tmp_path, consumer, dependency)
    expected = _run(tmp_path)
    plan, snapshot = _plan(tmp_path, include_consumer=include_consumer)
    simulation = plan.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    if include_consumer:
        assert _run(tmp_path) == expected
    else:
        assert (tmp_path / 'b/consumer.py').read_text() == consumer


def test_unselected_files_and_workspace_metadata_do_not_expand_the_source_graph(tmp_path: Path):
    _write(tmp_path, 'invalid Python: pkg.ext.Update',
           producer_tool='[tool.uv.workspace]\nmembers = ["../missing"]\n')
    sibling = tmp_path / 'unrelated'
    sibling.mkdir()
    (sibling / 'pyproject.toml').write_text('[project invalid')
    (sibling / 'consumer.py').write_text('invalid Python: pkg.ext.Update')
    before = {p: p.read_bytes() for p in (*sibling.iterdir(), tmp_path / 'b/consumer.py')}
    plan, snapshot = _plan(tmp_path, include_consumer=False)
    simulation = plan.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert {p: p.read_bytes() for p in before} == before


def test_a_cut_carries_the_consumer_rewrite_of_the_move_it_runs(tmp_path: Path):
    """The cut composes the move: the consumer edit the move makes is the cut's."""

    files = {
        "a/pyproject.toml": '[project]\nname = "a-project"\n',
        "a/pkg/__init__.py": "",
        "a/pkg/base.py": "LIMIT = 3\n",
        "a/pkg/hub.py": (
            "from pkg.base import LIMIT\nfrom pkg.a import a_value\n\n\n"
            "class Record:\n    size = LIMIT\n\n\ndef total():\n    return a_value()\n"
        ),
        "a/pkg/a.py": "def a_value():\n    from pkg.hub import Record\n\n    return Record.size\n",
        "b/pyproject.toml": '[project]\nname = "b"\ndependencies = ["a-project"]\n',
        "b/consumer.py": "from pkg.hub import Record, total\n\nprint(Record.size, total())\n",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    expected = _run(tmp_path)
    detector = HiddenImportCycleDetector()
    with ScanCache.scope():
        [finding] = detector.detect(
            parse_python_modules(tmp_path / "a", use_parse_cache=False), DetectorConfig()
        )
    snapshot = _snapshot(tmp_path, include_consumer=True)
    evaluation = detector.evaluate_recipe_for_finding(finding, snapshot)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert _run(tmp_path) == expected
