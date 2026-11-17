"""C3 orders NRA reads from source are the orders Python builds for the same classes."""

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput

MODULE = (
    "from abc import ABC\nfrom typing import Generic, TypeVar\n\nT = TypeVar('T')\n\n\n"
    "class Box(Generic[T]):\n    pass\n\n\n"
    "class Plain:\n    pass\n\n\n"
    "class Both(Box[T], Plain, ABC):\n    pass\n\n\n"
    "class Sub(Both[int], Box[int]):\n    pass\n"
)
NAMES = ("Box", "Plain", "Both", "Sub")


def _verdicts(root: Path):
    return CodemodSourceSnapshot.from_modules(
        parse_python_modules(root, use_parse_cache=False)
    ).class_overlap_verdicts


def test_source_orders_are_the_orders_python_builds(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/m.py").write_text(MODULE)
    runtime = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import pkg.m as m\n"
            f"for name in {NAMES!r}:\n"
            "    print(' '.join(f'{k.__module__}.{k.__qualname__}' for k in getattr(m, name).__mro__))",
        ],
        cwd=tmp_path,
        text=True,
    ).splitlines()
    verdicts = _verdicts(tmp_path)
    assert [" ".join(verdicts.linearized(f"pkg.m.{name}")) for name in NAMES] == runtime


def test_bases_python_cannot_order_need_a_proof(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/m.py").write_text("class A:\n    pass\n\n\nclass B(A):\n    pass\n")
    verdicts = _verdicts(tmp_path)
    with pytest.raises(NeedsInput) as needed:
        verdicts.linearized("pkg.m.B", ("pkg.m.A", "pkg.m.B"))
    assert NeededInput.PROOF in needed.value.needs
    assert "has no C3 order with bases `A`, `B`" in str(needed.value)
