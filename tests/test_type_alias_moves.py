"""A PEP 695 type alias moves like any other top-level declaration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    RefactorRecipe,
    RelocateSymbolsToNewModuleOperation,
    SourceRewriteTarget,
)


def _write(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_a_type_alias_relocates_with_the_declarations_it_names(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/source.py",
        "class Todo:\n    pass\n\n\ntype DirectedTodo = Todo\n\n\n"
        "class Command:\n    target: DirectedTodo\n",
    )
    _write(tmp_path, "pkg/consumer.py", "from pkg.source import Command, DirectedTodo, Todo\n")
    simulation = (
        RefactorRecipe("relocate-alias")
        .with_operation(
            RelocateSymbolsToNewModuleOperation(
                target=SourceRewriteTarget(file_path=(tmp_path / "pkg/source.py").as_posix()),
                symbol_qualnames=("Todo", "DirectedTodo", "Command"),
                destination_path=(tmp_path / "pkg/orders.py").as_posix(),
            )
        )
        .simulate(CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path)))
    )
    assert simulation.is_clean, simulation.preflight_report
    simulation.apply()
    assert "type DirectedTodo = Todo" in (tmp_path / "pkg/orders.py").read_text()
    ran = subprocess.run(
        [sys.executable, "-c",
         "from pkg.consumer import DirectedTodo, Todo; assert DirectedTodo.__value__ is Todo"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert ran.returncode == 0, ran.stderr
