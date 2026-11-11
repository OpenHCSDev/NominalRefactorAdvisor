"""Tests for the missing_mixin unblocking: inherited declaration tolerance and
descendant-field widening."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.refactors.class_overlap import CreateMixinOperation


BASE_WITH_DESCENDANT = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Record:\n"
    "    name: str\n"
    "    value: int\n\n\n"
    "@dataclass(frozen=True)\n"
    "class SpecialRecord(Record):\n"
    "    extra: str = ''\n"
)

COPIER = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Duplicate:\n"
    "    name: str\n"
    "    value: int\n"
    "    label: str = ''\n"
)


def _carrier_snapshot(tmp_path: Path) -> CodemodSourceSnapshot:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "record.py").write_text(BASE_WITH_DESCENDANT)
    (package / "duplicate.py").write_text(COPIER)
    return CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))


def test_descendant_without_projected_fields_does_not_block_carrier(
    tmp_path: Path,
) -> None:
    """SpecialRecord inherits name and value from Record but declares neither.
    The carrier operation on Duplicate should not refuse just because Record
    has a descendant."""

    from nominal_refactor_advisor.refactors.carrier_fields import (
        CarrierFieldProjection,
        ReplaceFieldsWithCarrierOperation,
    )

    snapshot = _carrier_snapshot(tmp_path)
    operation = ReplaceFieldsWithCarrierOperation(
        target=SourceRewriteTarget(
            file_path=(tmp_path / "pkg/duplicate.py").as_posix(),
            qualname="Duplicate",
        ),
        field_projections=(
            CarrierFieldProjection(source_field="name", carrier_attribute="name"),
            CarrierFieldProjection(source_field="value", carrier_attribute="value"),
        ),
        carrier_field_declaration="record: Record",
    )
    recipe = RefactorRecipe(recipe_id="test", operations=(operation,))
    document = CodemodPlanDocument(recipes=(recipe,))
    simulation = document.simulate(snapshot)
    duplicate_path = (tmp_path / "pkg/duplicate.py").as_posix()
    rewritten = simulation.simulation.rewritten_sources.get(duplicate_path, "")
    assert "record: Record" in rewritten
    assert "name: str" not in rewritten


def test_descendant_declaring_projected_field_still_refuses(
    tmp_path: Path,
) -> None:
    """A descendant of the target that declares a projected field must refuse."""

    from nominal_refactor_advisor.refactors.carrier_fields import (
        CarrierFieldProjection,
        ReplaceFieldsWithCarrierOperation,
    )

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "record.py").write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Record:\n"
        "    name: str\n"
        "    value: int\n"
    )
    (package / "duplicate.py").write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Duplicate:\n"
        "    name: str\n"
        "    value: int\n\n\n"
        "@dataclass(frozen=True)\n"
        "class OverridingDuplicate(Duplicate):\n"
        "    name: str = 'override'\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = ReplaceFieldsWithCarrierOperation(
        target=SourceRewriteTarget(
            file_path=(tmp_path / "pkg/duplicate.py").as_posix(),
            qualname="Duplicate",
        ),
        field_projections=(
            CarrierFieldProjection(source_field="name", carrier_attribute="name"),
            CarrierFieldProjection(source_field="value", carrier_attribute="value"),
        ),
        carrier_field_declaration="record: Record",
    )
    try:
        operation.source_edits_from_snapshot(snapshot)
        assert False, "should have refused"
    except ValueError as e:
        assert "descendants declaring projected fields" in str(e)
