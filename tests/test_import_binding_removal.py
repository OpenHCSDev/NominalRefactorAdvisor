"""remove_import_bindings drops unread bindings of every import shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.refactors.source_insertion import RemoveImportBindingsOperation
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
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


@pytest.mark.parametrize(
    ("source", "names", "refusal"),
    (
        pytest.param(
            "import pkg.helpers as helpers_module\n\nresult = helpers_module.VALUE\n",
            ("helpers_module",),
            "still reads import bindings",
            id="still-read",
        ),
        pytest.param(
            "from pkg.helpers import VALUE\n\nresult = VALUE\n",
            ("helpers",),
            "has no import binding \\('helpers',\\)",
            id="never-bound",
        ),
    ),
)
def test_refuses_a_binding_it_cannot_remove(
    tmp_path: Path, source: str, names: tuple[str, ...], refusal: str
) -> None:
    snapshot, consumer = _snapshot(tmp_path, source)

    with pytest.raises(ValueError, match=refusal):
        RefactorRecipe("drop").with_operation(_operation(consumer, *names)).simulate(
            snapshot
        )


def test_removes_an_unread_binding_under_type_checking_and_the_guard_it_empties(
    tmp_path: Path,
) -> None:
    """agent-comms child_process.py: deleting ProcessState.owns_turn leaves its
    `if TYPE_CHECKING:` import of ActiveTurn unread. It goes, and so does the
    guard and the flag it empties."""
    snapshot, consumer = _snapshot(
        tmp_path,
        "from __future__ import annotations\n\n"
        "from typing import TYPE_CHECKING\n\n"
        "from pkg.helpers import VALUE\n\n"
        "if TYPE_CHECKING:\n"
        "    from pkg import helpers\n\n"
        "result = VALUE\n",
    )
    simulation = RefactorRecipe("drop-unread").with_operation(_operation(consumer, "helpers")).simulate(
        snapshot
    )

    assert simulation.is_clean
    rewritten = simulation.simulation.rewritten_sources[consumer.as_posix()]
    assert "TYPE_CHECKING" not in rewritten
    assert "from pkg import helpers" not in rewritten
    assert [line for line in rewritten.splitlines() if line] == [
        "from __future__ import annotations",
        "from pkg.helpers import VALUE",
        "result = VALUE",
    ]
