"""A relocated symbol never lands below a destination statement that needs it."""

from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodPlanRoot,
    CodemodSourceSnapshot,
)


def _relocate(root: Path):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    plan = CodemodPlanRoot.from_json_value({"recipes": [{"recipe_id": "move", "operations": [{
        "operation": "relocate_symbols_to_module",
        "file_path": str(root / "pkg/source.py"),
        "symbol_qualnames": ["Family"],
        "destination_path": str(root / "pkg/dest.py"),
    }]}]})
    return plan.simulate(snapshot)


def _tree(root: Path, dest: str) -> None:
    (root / "pkg").mkdir()
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/source.py").write_text("from pkg.dest import Preparation\n\n\nclass Family(Preparation):\n    pass\n")
    (root / "pkg/dest.py").write_text(dest)


def test_a_subclass_above_the_needed_base_refuses(tmp_path: Path) -> None:
    """openhcs: the moved class needs a destination base defined below a destination
    subclass of it, so no position runs (the move used to land it below, NameError)."""
    _tree(
        tmp_path,
        "from pkg.source import Family\n\n\nclass Sub(Family):\n    pass\n\n\nclass Preparation:\n    pass\n",
    )
    with pytest.raises(CodemodOperationPreflightError, match=r"Family at line 4"):
        _relocate(tmp_path)


def test_a_reader_inside_a_function_does_not_constrain_the_landing(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        "def late():\n    from pkg.source import Family\n    return Family\n\n\nclass Preparation:\n    pass\n",
    )
    result = _relocate(tmp_path)
    assert result.is_clean
    dest = result.required_after_snapshot.sources_by_file_path[str(tmp_path / "pkg/dest.py")]
    assert dest.index("class Preparation") < dest.index("class Family")


def test_a_typing_alias_and_its_class_are_one_authority(tmp_path: Path) -> None:
    """openhcs: the source binds Iterable from typing, the destination from
    collections.abc; typing.Iterable stands for collections.abc.Iterable."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/source.py").write_text(
        "from typing import Iterable\n\n\nclass Family:\n"
        "    def each(self, values: Iterable[int]) -> list[int]:\n        return list(values)\n"
    )
    (tmp_path / "pkg/dest.py").write_text(
        "from collections.abc import Iterable\n\n\ndef total(values: Iterable[int]) -> int:\n    return sum(values)\n"
    )
    result = _relocate(tmp_path)
    assert result.is_clean
    dest = result.required_after_snapshot.sources_by_file_path[str(tmp_path / "pkg/dest.py")]
    assert dest.count("import Iterable") == 1 and "class Family" in dest


def test_a_typing_name_that_is_no_alias_still_conflicts(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/source.py").write_text(
        "from typing import Protocol\n\n\nclass Family(Protocol):\n    pass\n"
    )
    (tmp_path / "pkg/dest.py").write_text("from pkg.other import Protocol\n")
    (tmp_path / "pkg/other.py").write_text("class Protocol:\n    pass\n")
    with pytest.raises(CodemodOperationPreflightError, match="different authorities"):
        _relocate(tmp_path)
