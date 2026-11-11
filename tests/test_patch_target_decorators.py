"""patch_target edits a declaration's decorators when its plan says to include them."""

from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodPlanRoot,
    CodemodSourceSnapshot,
)

MODULE = (
    "import pytest\n\n\n"
    "@pytest.mark.parametrize('module', ('agent_comms.goal_events', 'agent_comms.other'))\n"
    "def check_imports(module):\n"
    "    assert module\n"
)


def _simulate(tmp_path: Path, **payload: object):
    (tmp_path / "m.py").write_text(MODULE)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = {
        "operation": "patch_target",
        "file_path": (tmp_path / "m.py").as_posix(),
        "target_qualname": "check_imports",
        "replacements": [
            {
                "old_source": "'agent_comms.goal_events'",
                "new_source": "'agent_comms.planning.goal_events'",
            }
        ],
        **payload,
    }
    plan = CodemodPlanRoot.from_json_value({"recipes": [{"recipe_id": "patch", "operations": [operation]}]})
    return plan.simulate(snapshot)


def test_an_included_decorator_is_patched(tmp_path: Path) -> None:
    result = _simulate(tmp_path, decorators="include")
    assert result.is_clean
    (rewritten,) = result.required_after_snapshot.sources_by_file_path.values()
    assert "('agent_comms.planning.goal_events', 'agent_comms.other')" in rewritten
    assert rewritten.count("def check_imports(module):") == 1


def test_decorators_are_outside_the_patch_by_default(tmp_path: Path) -> None:
    with pytest.raises(CodemodOperationPreflightError, match="found 0"):
        _simulate(tmp_path)
