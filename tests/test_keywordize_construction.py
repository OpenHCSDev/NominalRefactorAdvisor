"""Positional dataclass constructions rewritten into keyword form, and run (#86)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_overlap_codemod import KeywordizeConstructionOperation
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

RECORDS = (
    "from dataclasses import KW_ONLY, dataclass, field\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Point:\n"
    "    x: int\n"
    "    tag: str = field(default='p', kw_only=True)\n"
    "    y: int = 0\n\n"
    "    @classmethod\n"
    "    def origin(cls, scale):\n"
    "        return cls(0 * scale, 0)\n\n"
    "    def moved(self, step):\n"
    "        return type(self)(self.x + step, (n := self.y + step), tag=self.tag)\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Located(Point):\n"
    "    label: str = ''\n"
)

USERS = (
    "from pkg.records import Point as P, Located\n\n\n"
    "def build(values):\n"
    "    first = P(  # the first point\n"
    "        values[0],\n"
    "        (values[1]),\n"
    "    )\n"
    "    return first, P(1), Located(2, 3, 'l'), P.origin(2), first.moved(1)\n"
)

PROBE = "from pkg.users import build\nprint(build([4, 5]))\n"


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _simulate(root: Path, qualname: str):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    operation = KeywordizeConstructionOperation(
        target=SourceRewriteTarget(
            file_path=(root / "pkg/records.py").as_posix(), qualname=qualname
        )
    )
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    return RefactorRecipe(recipe_id="keywordize").with_operation(replayed).simulate(snapshot)


def _run(root: Path) -> str:
    return subprocess.run(
        [sys.executable, "-c", PROBE], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def test_positional_constructions_become_keywords_and_run_the_same(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS, "pkg/users.py": USERS})
    before = _run(tmp_path)

    result = _simulate(tmp_path, "Point")
    assert result.is_clean is True
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source, encoding="utf-8")

    records = (tmp_path / "pkg/records.py").read_text()
    users = (tmp_path / "pkg/users.py").read_text()
    assert "return cls(x=0 * scale, y=0)" in records
    assert "type(self)(x=self.x + step, y=(n := self.y + step), tag=self.tag)" in records
    assert "        x=values[0],\n        y=(values[1]),\n" in users
    assert "P(x=1)" in users
    assert "Located(2, 3, 'l')" in users
    assert _run(tmp_path) == before


@pytest.mark.parametrize(
    ("users", "refusal"),
    (
        ("from pkg.records import Point\nPoint(*(1, 2))\n", "unpacks `*` arguments"),
        ("from pkg.records import Point\nPoint(1, **{})\n", "unpacks `**` arguments"),
        ("from pkg.records import Point\nPoint(1, 2, 3)\n", "takes 2"),
    ),
)
def test_keywording_refuses_a_call_it_cannot_map(
    tmp_path: Path, users: str, refusal: str
) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS, "pkg/users.py": users})
    with pytest.raises(CodemodOperationPreflightError, match=re.escape(refusal)):
        _simulate(tmp_path, "Point")


def test_keywording_refuses_a_descendant_whose_positions_differ(tmp_path: Path) -> None:
    records = RECORDS.replace(
        "class Located(Point):\n    label: str = ''\n",
        "class Located(Point):\n    y: int = field(default=0, kw_only=True)\n",
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records})
    with pytest.raises(CodemodOperationPreflightError, match="other positional parameters"):
        _simulate(tmp_path, "Point")


def test_keywording_refuses_an_unknown_keyword_only_option(tmp_path: Path) -> None:
    records = RECORDS.replace("@dataclass(frozen=True)\nclass Point", "FLAG = False\n\n\n@dataclass(frozen=True, kw_only=FLAG)\nclass Point")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records})
    with pytest.raises(CodemodOperationPreflightError, match="not a literal"):
        _simulate(tmp_path, "Point")
