"""A method returning zero-or-one tuples becomes Optional across its family."""

from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.refactors.return_shape import FlattenTupleReturnOperation


BASE = (
    "class Validator:\n"
    "    def warning(self) -> tuple[str, ...]:\n"
    "        return ()\n"
)

STRICT = (
    "from pkg.base import Validator\n\n\n"
    "class StrictValidator(Validator):\n"
    "    def warning(self) -> tuple[str, ...]:\n"
    "        return ('field is required',)\n"
)

CONSUMER = (
    "from pkg.base import Validator\n\n\n"
    "def check(v: Validator) -> str | None:\n"
    "    for message in v.warning():\n"
    "        return message\n"
    "    return None\n"
)


def _snapshot(tmp_path: Path) -> CodemodSourceSnapshot:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "base.py").write_text(BASE)
    (package / "strict.py").write_text(STRICT)
    (package / "consumer.py").write_text(CONSUMER)
    return CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))


def _operation(tmp_path: Path) -> FlattenTupleReturnOperation:
    return FlattenTupleReturnOperation(
        target=SourceRewriteTarget(
            file_path=(tmp_path / "pkg/base.py").as_posix(),
            qualname="Validator",
        ),
        method_name="warning",
    )


def test_return_statements_flatten_across_family(tmp_path: Path) -> None:
    """Every override's return (value,) → value and () → None."""
    snapshot = _snapshot(tmp_path)
    operation = _operation(tmp_path)
    edits = operation.source_edits_from_snapshot(snapshot)
    recipe = RefactorRecipe(recipe_id="test", operations=(operation,))
    document_snapshot = snapshot.execution_snapshot()
    rewrites = document_snapshot._planned_rewrites_from_physical_edits(
        tuple(edit for edit in edits if hasattr(edit, "file_path"))
    )
    simulation = snapshot.simulate_rewrites(rewrites)

    base_path = (tmp_path / "pkg/base.py").as_posix()
    strict_path = (tmp_path / "pkg/strict.py").as_posix()
    consumer_path = (tmp_path / "pkg/consumer.py").as_posix()

    base_rewritten = simulation.rewritten_sources.get(base_path, "")
    strict_rewritten = simulation.rewritten_sources.get(strict_path, "")
    consumer_rewritten = simulation.rewritten_sources.get(consumer_path, "")

    assert "return None" in base_rewritten
    assert "return ()" not in base_rewritten
    assert "return 'field is required'" in strict_rewritten
    assert "return ('field is required',)" not in strict_rewritten


def test_empty_return_becomes_none(tmp_path: Path) -> None:
    """A base returning () becomes return None."""
    snapshot = _snapshot(tmp_path)
    operation = _operation(tmp_path)
    edits = operation.source_edits_from_snapshot(snapshot)

    base_path = (tmp_path / "pkg/base.py").as_posix()
    all_replacements = [
        (r.start_offset, r.end_offset, r.replacement_source)
        for edit in edits
        if hasattr(edit, "file_path") and edit.file_path == base_path
        for r in getattr(edit, "replacements", ())
    ]
    replacement_texts = [text for _, _, text in all_replacements]
    assert "None" in replacement_texts


def test_annotation_flattens(tmp_path: Path) -> None:
    """tuple[str, ...] annotation becomes str | None."""
    from nominal_refactor_advisor.refactors.return_shape import FlattenTupleReturnOperation

    assert FlattenTupleReturnOperation._flatten_annotation("tuple[str, ...]") == "str | None"
    assert FlattenTupleReturnOperation._flatten_annotation("tuple[int, ...]") == "int | None"
    assert FlattenTupleReturnOperation._flatten_annotation("str") == "str"
    assert FlattenTupleReturnOperation._flatten_annotation("list[str]") == "list[str]"
