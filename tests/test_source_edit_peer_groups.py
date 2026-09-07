"""Actual peer identity survives declaration-owned physical coalescence."""

from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    PhysicalSourceEditConflictError,
    SourceEditOrigin,
    SourceInsertion,
    SourceSpanReplacement,
)
from nominal_refactor_advisor.codemod_spacing import SourceInsertionBoundary
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot

PATH = Path(__file__).with_name("peer_fixture.py").resolve().as_posix()


@pytest.fixture(params=(SourceInsertion, SourceSpanReplacement))
def peers(request):
    if request.param is SourceInsertion:
        first = SourceInsertion(
            file_path=PATH, insertion_line=1, inserted_lines=("a = 1\n",)
        )
        other = replace(first, insertion_line=2)
    else:
        first = SourceSpanReplacement(
            file_path=PATH, start_line=1, end_line=1, replacement_lines=("a = 1\n",)
        )
        other = replace(first, start_line=2, end_line=2)
    duplicate = replace(first)
    different_file = replace(first, file_path=Path(PATH).with_name("other.py").as_posix())
    return first, other, duplicate, different_file


def test_grouping_preserves_equal_but_distinct_participants_and_encounter_order(peers):
    first, other, duplicate, different_file = peers
    assert first == duplicate and first is not duplicate
    groups = type(first).peer_groups((first, other, duplicate, first, different_file))
    assert len(groups) == 3
    assert len(groups[0]) == 3
    assert groups[0][0] is first
    assert groups[0][1] is duplicate
    assert groups[0][2] is first
    assert groups[1][0] is other
    assert groups[2][0] is different_file


def test_actual_groups_derive_the_existing_public_coalescence_result(peers):
    first, other, duplicate, _different_file = peers
    first = replace(
        first, origins=(SourceEditOrigin("r", "First", 0),), rationale="first"
    )
    duplicate = replace(
        duplicate, origins=(SourceEditOrigin("r", "Second", 1),), rationale="second"
    )
    originals = first, other, duplicate
    groups = type(first).peer_groups(originals)
    derived = tuple(type(first).coalesced_group(group) for group in groups)
    context = CodemodSourceSnapshot.from_source_mapping({PATH: "a = 0\nb = 0\n"})
    assert derived == NominalSourceEdit.coalesced_by_declaration(originals, context)
    assert derived[0].origins == first.origins + duplicate.origins
    assert "first" in derived[0].rationale
    assert "second" in derived[0].rationale
    assert originals[0] is first and originals[2] is duplicate


def test_group_contract_rejects_empty_or_cross_location_members(peers):
    first, other, _duplicate, _different_file = peers
    with pytest.raises(ValueError, match="one nonempty source peer group"):
        type(first).coalesced_group(())
    with pytest.raises(ValueError, match="one nonempty source peer group"):
        type(first).coalesced_group((first, other))


def test_duplicate_insertion_keeps_first_boundary_and_all_origins():
    first = SourceInsertion(
        file_path=PATH, insertion_line=1, inserted_lines=("a = 1\n", "\n", "\n")
    )
    second = replace(
        first,
        inserted_lines=("\n", "b = 2\n"),
        leading_boundary=SourceInsertionBoundary.ONE_BLANK_LINE,
    )
    duplicate = replace(
        second,
        leading_boundary=SourceInsertionBoundary.TWO_BLANK_LINES,
        origins=(SourceEditOrigin("r", "Third", 2),),
    )
    group = SourceInsertion.peer_groups((first, second, duplicate))[0]
    assert group[2] is duplicate
    result = SourceInsertion.coalesced_group(group)
    assert result.inserted_lines == ("a = 1\n", "\n", "b = 2\n")
    assert result.origins == duplicate.origins


def test_group_exposes_conflicting_replacement_inputs_before_merge_rejects():
    first = SourceSpanReplacement(
        file_path=PATH, start_line=1, end_line=1, replacement_lines=("a = 1\n",)
    )
    conflicting = replace(first, replacement_lines=("a = 2\n",))
    group = SourceSpanReplacement.peer_groups((first, conflicting))[0]
    assert group[0] is first and group[1] is conflicting
    with pytest.raises(
        PhysicalSourceEditConflictError, match="Conflicting source span replacements"
    ):
        SourceSpanReplacement.coalesced_group(group)


@pytest.mark.parametrize("operation", ("peer_groups", "coalesced_group"))
def test_exposed_peer_boundary_rejects_another_exact_declaration(peers, operation):
    first, _other, _duplicate, _different_file = peers

    class OtherDeclaration(type(first)):
        pass

    foreign = OtherDeclaration(**first.__dict__)
    assert foreign.coalescence_key == first.coalescence_key
    with pytest.raises(ValueError, match="exact nominal declaration"):
        getattr(type(first), operation)((foreign,))
    with pytest.raises(ValueError, match="exact nominal declaration"):
        getattr(OtherDeclaration, operation)((first,))
    with pytest.raises(ValueError, match="exact nominal declaration"):
        getattr(type(first), operation)((first, foreign))
    owned = getattr(OtherDeclaration, operation)((foreign,))
    if operation == "peer_groups":
        assert owned[0][0] is foreign
    else:
        assert type(owned) is OtherDeclaration


def test_base_cannot_substitute_its_payload_law_for_a_subclass_declaration():
    class SpecializedInsertion(SourceInsertion):
        @staticmethod
        def _coalesced_group(peers):
            return replace(peers[0], inserted_lines=("specialized = True\n",))

    peer = SpecializedInsertion(
        file_path=PATH, insertion_line=1, inserted_lines=("original = True\n",)
    )
    with pytest.raises(ValueError, match="exact nominal declaration"):
        SourceInsertion.coalesced_group((peer,))
    assert SpecializedInsertion.coalesced_group((peer,)).inserted_lines == (
        "specialized = True\n",
    )
