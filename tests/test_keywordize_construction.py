"""Positional dataclass constructions rewritten into keyword form, and run (#86)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.refactors.class_overlap import KeywordizeConstructionOperation
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


SCRIPTS = (
    "import subprocess, sys, textwrap\n\n"
    "PROBE = (\n"
    '    "from pkg.records import Point\\n"\n'
    '    "print(Point(7, \\"8\\" and 8))\\n"\n'
    ")\n"
    "DEDENTED = textwrap.dedent(\"\"\"\n"
    "    from pkg.records import Point as P\n"
    "    print(P(9))\n"
    "\"\"\")\n\n\n"
    "def run():\n"
    "    return [\n"
    "        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True).stdout\n"
    "        for code in (PROBE, DEDENTED)\n"
    "    ]\n"
)


def test_a_script_a_string_holds_is_keyworded_where_it_stands(tmp_path: Path) -> None:
    """A child interpreter runs the string as its own module: its imports name
    the class, and the rewrite lands on the literal's own characters."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS, "pkg/scripts.py": SCRIPTS})
    probe = "from pkg.scripts import run\nprint(run())\n"

    def ran() -> str:
        return subprocess.run(
            [sys.executable, "-c", probe], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout

    before = ran()
    result = _simulate(tmp_path, "Point")
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source, encoding="utf-8")

    scripts = (tmp_path / "pkg/scripts.py").read_text()
    assert '"print(Point(x=7, y=\\"8\\" and 8))\\n"' in scripts
    assert "    print(P(x=9))\n" in scripts
    assert ran() == before


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
    with pytest.raises(CodemodOperationPreflightError, match="`pkg.records.Point`'s field `x` is keyword-only by a non-literal"):
        _simulate(tmp_path, "Point")


FAMILY = (
    "class Family(type):\n"
    "    def __new__(mcs, name, bases, namespace, *, affix=None, **options):\n"
    "        namespace['declared_name'] = name.lower()\n"
    "        cls = super().__new__(mcs, name, bases, namespace, **options)\n"
    "        cls.family_root = cls\n"
    "        return cls\n\n\n"
)

FAMILY_RECORDS = RECORDS.replace(
    "\n\n\n@dataclass(frozen=True)\nclass Point:",
    "\n\n\n" + FAMILY + "@dataclass(frozen=True)\nclass Point(metaclass=Family, affix='p'):",
    1,
)


def test_a_metaclass_that_stores_no_init_keeps_the_declared_parameters(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": FAMILY_RECORDS, "pkg/users.py": USERS})
    before = _run(tmp_path)

    result = _simulate(tmp_path, "Point")
    assert result.is_clean is True
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source, encoding="utf-8")

    assert "P(x=1)" in (tmp_path / "pkg/users.py").read_text()
    assert _run(tmp_path) == before


@pytest.mark.parametrize(
    ("change", "refusal"),
    (
        (
            ("        cls.family_root = cls\n", "        cls.__init__ = lambda self: None\n"),
            "sets what a call binds",
        ),
        (
            ("namespace['declared_name']", "namespace['__init__']"),
            "keeps the parameters a call binds",
        ),
        (
            (
                "        return cls\n\n\n",
                "        return cls\n\n    def __call__(cls, *args):\n        return super().__call__()\n\n\n",
            ),
            "defines `__call__`",
        ),
        (
            ("    x: int\n", "    x: int\n\n    def __init_subclass__(cls):\n        cls.__init__ = None\n"),
            "sets what a call binds",
        ),
    ),
)
def test_a_metaclass_that_may_change_what_a_call_binds_refuses(
    tmp_path: Path, change: tuple[str, str], refusal: str
) -> None:
    records = FAMILY_RECORDS.replace(*change, 1)
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records, "pkg/users.py": USERS})
    with pytest.raises(CodemodOperationPreflightError, match=re.escape(refusal)):
        _simulate(tmp_path, "Point")


REGISTERED = RECORDS.replace(
    "from dataclasses import KW_ONLY, dataclass, field\n\n\n",
    "from dataclasses import KW_ONLY, dataclass, field\nfrom typing import ClassVar\n\n\n"
    "class Bag:\n    def add(self, value):\n        value.__init__ = None\n\n\n"
    "class Registry:\n"
    "    _seen: ClassVar[dict[str, set[type]]] = {}\n"
    "    _bags: ClassVar[dict[str, Bag]] = {}\n\n\n",
    1,
).replace(
    "    x: int\n",
    "    x: int\n\n"
    "    def __init_subclass__(cls, **kwargs):\n"
    "        super().__init_subclass__(**kwargs)\n"
    "        Registry._seen.setdefault('points', set()).add(cls)\n",
    1,
)


def test_a_creation_hook_storing_the_class_in_a_builtin_set_keeps_the_parameters(tmp_path: Path) -> None:
    """`set.add` only keeps a reference to the class; nothing a call binds changes."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": REGISTERED, "pkg/users.py": USERS})
    before = _run(tmp_path)
    result = _simulate(tmp_path, "Point")
    assert result.is_clean is True
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source, encoding="utf-8")
    assert "P(x=1)" in (tmp_path / "pkg/users.py").read_text()
    assert _run(tmp_path) == before


@pytest.mark.parametrize(
    "stored",
    (
        "Registry._bags.setdefault('points', Bag()).add(cls)",
        "Registry.unknown().add(cls)",
    ),
)
def test_a_creation_hook_storing_the_class_in_an_unproved_container_refuses(
    tmp_path: Path, stored: str
) -> None:
    records = REGISTERED.replace("Registry._seen.setdefault('points', set()).add(cls)", stored)
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records, "pkg/users.py": USERS})
    with pytest.raises(CodemodOperationPreflightError, match="passes it to a function NRA cannot read"):
        _simulate(tmp_path, "Point")


def test_storing_a_class_whose_metaclass_hashes_it_itself_refuses(tmp_path: Path) -> None:
    records = REGISTERED.replace(
        "class Registry:\n",
        "class Hashing(type):\n    def __hash__(cls):\n        return 0\n\n\nclass Registry:\n",
        1,
    ).replace("class Point:", "class Point(metaclass=Hashing):", 1)
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records, "pkg/users.py": USERS})
    with pytest.raises(CodemodOperationPreflightError, match="own `__hash__` or `__eq__`"):
        _simulate(tmp_path, "Point")
