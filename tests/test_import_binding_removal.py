"""remove_import_bindings drops unread bindings of every import shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    RemoveImportBindingsOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def _snapshot(root: Path, source: str) -> tuple[CodemodSourceSnapshot, Path]:
    (root / "pkg").mkdir()
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/helpers.py").write_text("VALUE = 1\n")
    consumer = root / "pkg/consumer.py"
    consumer.write_text(source)
    return CodemodSourceSnapshot.from_modules(parse_python_modules(root)), consumer


def _operation(consumer: Path, *names: str) -> RemoveImportBindingsOperation:
    return RemoveImportBindingsOperation(
        target=SourceRewriteTarget(file_path=consumer.as_posix()),
        bound_names=names,
    )


def test_removes_unread_plain_and_from_import_bindings(tmp_path: Path) -> None:
    snapshot, consumer = _snapshot(
        tmp_path,
        "import pkg.helpers as helpers_module\n"
        "from pkg import helpers\n"
        "from pkg.helpers import VALUE\n\n"
        "result = VALUE\n",
    )
    operation = _operation(consumer, "helpers_module", "helpers")

    payload = json_report_object(operation)
    simulation = RefactorRecipe("drop-unread").with_operation(operation).simulate(
        snapshot
    )

    assert payload["operation"] == "remove_import_bindings"
    assert type(RefactorRecipeOperation.from_json_value(payload)) is type(operation)
    assert simulation.is_clean
    assert simulation.simulation.rewritten_sources[consumer.as_posix()] == (
        "from pkg.helpers import VALUE\n\nresult = VALUE\n"
    )


def test_refuses_a_binding_the_module_still_reads(tmp_path: Path) -> None:
    snapshot, consumer = _snapshot(
        tmp_path,
        "import pkg.helpers as helpers_module\n\n" "result = helpers_module.VALUE\n",
    )

    with pytest.raises(ValueError, match="still reads import bindings"):
        RefactorRecipe("drop-read").with_operation(
            _operation(consumer, "helpers_module")
        ).simulate(snapshot)
