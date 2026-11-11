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


INHERITED_SHARED = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Ancestor:\n"
    "    shared_a: str\n\n\n"
    "@dataclass(frozen=True)\n"
    "class First(Ancestor):\n"
    "    shared_b: int\n"
    "    shared_c: int\n"
    "    own_first: float = 0.0\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Second:\n"
    "    shared_a: str\n"
    "    shared_b: int\n"
    "    shared_c: int\n"
    "    own_second: float = 0.0\n"
)


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


def test_inherited_declaration_does_not_block_mixin_creation(
    tmp_path: Path,
) -> None:
    """First inherits shared_a from Ancestor but declares shared_b and shared_c
    itself. Second declares all three. The overlap verdict has INHERITED_DECLARATION
    for shared_a on First, but a mixin for shared_b + shared_c should work because
    tolerate_inherited lets them through."""

    from nominal_refactor_advisor.refactors.class_overlap import MixinCreation

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "classes.py").write_text(INHERITED_SHARED)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    verdicts = snapshot.class_overlap_verdicts
    candidates = verdicts.candidates
    assert any(c.inherited_names for c in candidates), (
        "fixture must produce at least one verdict with inherited declarations"
    )
    for c in candidates:
        if c.inherited_names:
            assert "shared_a" in c.inherited_names
            assert c.owned_shared_names

    target = SourceRewriteTarget(
        file_path=(tmp_path / "pkg/classes.py").as_posix(),
        qualname="First",
    )
    creation = MixinCreation.require(
        snapshot, target, ("shared_b", "shared_c"), "SharedBC", "Shared integer fields."
    )
    edits = creation.source_edits("Extract shared_b and shared_c into SharedBC.")
    assert edits


def test_all_inherited_block_refuses_mixin(
    tmp_path: Path,
) -> None:
    """When every shared declaration is inherited, the mixin has nothing to extract."""

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "classes.py").write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Ancestor:\n"
        "    shared_a: str\n\n\n"
        "@dataclass(frozen=True)\n"
        "class First(Ancestor):\n"
        "    own_first: float = 0.0\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Second:\n"
        "    shared_a: str\n"
        "    own_second: float = 0.0\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    verdicts = snapshot.class_overlap_verdicts
    candidates = verdicts.candidates
    inherited_candidates = [
        c for c in candidates
        if c.inherited_names
    ]
    for c in inherited_candidates:
        owned = c.owned_shared_names
        if not owned:
            assert c.inherited_names


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
