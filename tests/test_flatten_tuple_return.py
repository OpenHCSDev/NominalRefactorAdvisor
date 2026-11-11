"""A method returning zero-or-one tuples becomes Optional across its family."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
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
    recipe = RefactorRecipe(recipe_id="test", operations=(operation,))
    document = CodemodPlanDocument(recipes=(recipe,))
    simulation = document.simulate(snapshot)

    base_path = (tmp_path / "pkg/base.py").as_posix()
    strict_path = (tmp_path / "pkg/strict.py").as_posix()

    base_rewritten = simulation.simulation.rewritten_sources.get(base_path, "")
    strict_rewritten = simulation.simulation.rewritten_sources.get(strict_path, "")

    assert "return None" in base_rewritten
    assert "return ()" not in base_rewritten
    assert "return 'field is required'" in strict_rewritten
    assert "return ('field is required',)" not in strict_rewritten


def test_empty_return_becomes_none(tmp_path: Path) -> None:
    """A base returning () becomes return None."""
    snapshot = _snapshot(tmp_path)
    operation = _operation(tmp_path)
    recipe = RefactorRecipe(recipe_id="test", operations=(operation,))
    document = CodemodPlanDocument(recipes=(recipe,))
    simulation = document.simulate(snapshot)

    base_path = (tmp_path / "pkg/base.py").as_posix()
    rewritten = simulation.simulation.rewritten_sources.get(base_path, "")
    assert "return None" in rewritten
    assert "return ()" not in rewritten


def test_for_loop_reader_becomes_if_walrus(tmp_path: Path) -> None:
    """``for message in v.warning():`` → ``if (message := v.warning()) is not None:``"""
    snapshot = _snapshot(tmp_path)
    operation = _operation(tmp_path)
    recipe = RefactorRecipe(recipe_id="test", operations=(operation,))
    document = CodemodPlanDocument(recipes=(recipe,))
    simulation = document.simulate(snapshot)

    consumer_path = (tmp_path / "pkg/consumer.py").as_posix()
    rewritten = simulation.simulation.rewritten_sources.get(consumer_path, "")
    assert "if (message := v.warning()) is not None:" in rewritten
    assert "for message in v.warning():" not in rewritten


def test_next_iter_reader_collapses(tmp_path: Path) -> None:
    """``next(iter(v.warning()), None)`` → ``v.warning()``"""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "base.py").write_text(BASE)
    (package / "strict.py").write_text(STRICT)
    (package / "consumer.py").write_text(
        "from pkg.base import Validator\n\n\n"
        "def first_warning(v: Validator) -> str | None:\n"
        "    return next(iter(v.warning()), None)\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _operation(tmp_path)
    recipe = RefactorRecipe(recipe_id="test", operations=(operation,))
    document = CodemodPlanDocument(recipes=(recipe,))
    simulation = document.simulate(snapshot)
    consumer_path = (tmp_path / "pkg/consumer.py").as_posix()
    rewritten = simulation.simulation.rewritten_sources.get(consumer_path, "")
    assert "next(iter(" not in rewritten
    assert "v.warning()" in rewritten


def test_truthiness_reader_passes_through(tmp_path: Path) -> None:
    """``if v.warning():`` stays unchanged — None is already falsy."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "base.py").write_text(BASE)
    (package / "strict.py").write_text(STRICT)
    (package / "consumer.py").write_text(
        "from pkg.base import Validator\n\n\n"
        "def has_warning(v: Validator) -> bool:\n"
        "    if v.warning():\n"
        "        return True\n"
        "    return False\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _operation(tmp_path)
    recipe = RefactorRecipe(recipe_id="test", operations=(operation,))
    document = CodemodPlanDocument(recipes=(recipe,))
    simulation = document.simulate(snapshot)
    consumer_path = (tmp_path / "pkg/consumer.py").as_posix()
    rewritten = simulation.simulation.rewritten_sources.get(
        consumer_path, snapshot.sources_by_file_path[consumer_path]
    )
    assert "if v.warning():" in rewritten


def test_annotation_flattens(tmp_path: Path) -> None:
    """tuple[str, ...] annotation becomes str | None."""
    from nominal_refactor_advisor.refactors.return_shape import FlattenTupleReturnOperation

    assert FlattenTupleReturnOperation._flatten_annotation("tuple[str, ...]") == "str | None"
    assert FlattenTupleReturnOperation._flatten_annotation("tuple[int, ...]") == "int | None"
    assert FlattenTupleReturnOperation._flatten_annotation("str") == "str"
    assert FlattenTupleReturnOperation._flatten_annotation("list[str]") == "list[str]"
