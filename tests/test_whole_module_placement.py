"""Whole-module placement keeps executable import boundaries and real script callers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    RefactorRecipe,
    RelocateModuleToNewModuleOperation,
    SourceRewriteTarget,
)


@pytest.mark.parametrize("declaration", ["", "def answer():\n    return VALUE\n\n"])
def test_whole_script_keeps_cold_import_order_and_path_callers(tmp_path: Path, declaration: str) -> None:
    source = (
        'import importlib\nimport json\nimport sys\n\n'
        'importlib.import_module(sys.argv[1])\n\n'
        'from pkg.later import VALUE\nfrom pkg.state import events\n\n'
        + declaration
        + 'assert VALUE == 17\n'
        + 'print(json.dumps({"events": events, "value": VALUE, "argument": sys.argv[1]}))\n'
    )
    sources = {
        'pkg/__init__.py': '',
        'pkg/state.py': 'events = []\n',
        'pkg/first.py': 'from pkg.state import events\nevents.append("first")\n',
        'pkg/later.py': 'from pkg.state import events\nassert events == ["first"]\nevents.append("later")\nVALUE = 17\n',
        'pkg/probe.py': source,
        'caller.py': 'SCRIPT = "pkg/probe.py"\nMODULE = "pkg.probe"\n',
    }
    for relative, body in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    environment = {**os.environ, 'PYTHONPATH': str(tmp_path)}
    before = subprocess.run(
        [sys.executable, str(tmp_path / 'pkg/probe.py'), 'pkg.first'],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=10,
    )
    assert before.returncode == 0, before.stderr
    destination = tmp_path / 'pkg/support/probe.py'
    operation = RelocateModuleToNewModuleOperation(
        target=SourceRewriteTarget(file_path=str(tmp_path / 'pkg/probe.py')),
        destination_path=str(destination),
    )
    result = RefactorRecipe('place-whole-script').with_operation(operation).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert result.is_clean, result
    assert result.simulation.rewritten_sources[str(destination)] == source
    result.apply()
    assert not (tmp_path / 'pkg/probe.py').exists()
    assert destination.read_text() == source
    assert (tmp_path / 'pkg/support/__init__.py').exists()
    caller = (tmp_path / 'caller.py').read_text()
    assert '"pkg/support/probe.py"' in caller
    assert '"pkg.support.probe"' in caller
    after = subprocess.run(
        [sys.executable, str(destination), 'pkg.first'],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=10,
    )
    assert after.returncode == 0, after.stderr
    assert after.stdout == before.stdout


@pytest.mark.parametrize('source', ['print(__file__)\n', 'from . import sibling\n'])
def test_whole_script_refuses_unproved_location_changes(tmp_path: Path, source: str) -> None:
    package = tmp_path / 'pkg'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'probe.py').write_text(source)
    operation = RelocateModuleToNewModuleOperation(
        target=SourceRewriteTarget(file_path=str(package / 'probe.py')),
        destination_path=str(package / 'support/probe.py'),
    )
    result = RefactorRecipe('refuse-changed-location').with_operation(operation).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert not result.is_clean
    assert (package / 'probe.py').read_text() == source
    assert not (package / 'support/probe.py').exists()
