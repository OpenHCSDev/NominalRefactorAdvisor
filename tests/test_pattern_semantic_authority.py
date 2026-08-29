from __future__ import annotations

import nominal_refactor_advisor.patterns as patterns
from nominal_refactor_advisor.factorization import RefactorPhase
from nominal_refactor_advisor.models import RefactorAction, RefactorActionKind
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.planner import (
    PatternPlanningStrategy,
    _pattern_planning,
)


def test_pattern_members_own_complete_metadata_and_phase() -> None:
    assert not hasattr(patterns, "PatternSpec")
    assert not hasattr(patterns, "PATTERN_SPECS")
    assert not hasattr(patterns, "PlanStepBuilderId")
    assert not hasattr(patterns, "ActionBuilderId")

    for pattern_id in PatternId:
        assert pattern_id.display_name
        assert pattern_id.prescription
        assert pattern_id.canonical_shape
        assert pattern_id.first_moves
        assert isinstance(pattern_id.phase, RefactorPhase)
        assert all(isinstance(item, PatternId) for item in pattern_id.dependencies)
        assert all(isinstance(item, PatternId) for item in pattern_id.synergy_with)

    assert PatternId.NOMINAL_BOUNDARY.phase is RefactorPhase.NORMALIZE
    assert PatternId.AUTO_REGISTER_META.phase is RefactorPhase.ESTABLISH_OWNER
    assert PatternId.LOCAL_VALUE_AUTHORITY.phase is RefactorPhase.DELETE_SHADOW
    assert PatternId.ABC_TEMPLATE_METHOD.dependencies == (
        PatternId.NOMINAL_BOUNDARY,
        PatternId.CONFIG_CONTRACTS,
    )


def test_specialized_pattern_planning_has_one_nominal_authority() -> None:
    assert set(PatternPlanningStrategy.__registry__) == {
        PatternId.CLOSED_FAMILY_DISPATCH,
        PatternId.ABC_TEMPLATE_METHOD,
        PatternId.AUTO_REGISTER_META,
        PatternId.BIDIRECTIONAL_LOOKUP,
        PatternId.AUTHORITATIVE_SCHEMA,
    }
    assert all(
        PatternPlanningStrategy.for_pattern(pattern_id).pattern_id is pattern_id
        for pattern_id in PatternPlanningStrategy.__registry__
    )


def test_every_pattern_derives_a_complete_planning_projection() -> None:
    for pattern_id in PatternId:
        planning = _pattern_planning("sample", pattern_id, ())

        assert planning.step
        assert planning.actions
        assert all(action.target == "sample" for action in planning.actions)
        assert all(
            isinstance(action.kind, RefactorActionKind) for action in planning.actions
        )

    assert all(
        RefactorAction.__dataclass_fields__[field_name].init is False
        for field_name in (
            "confidence",
            "statement_operation",
            "remove_symbols",
            "statement_sites",
        )
    )


def test_generic_and_specialized_pattern_planning_outputs() -> None:
    generic_planning = _pattern_planning("sample", PatternId.NOMINAL_BOUNDARY, ())

    assert PatternId.NOMINAL_BOUNDARY not in PatternPlanningStrategy.__registry__
    assert PatternId.NOMINAL_BOUNDARY.prescription in generic_planning.step
    assert tuple(action.kind for action in generic_planning.actions) == (
        RefactorActionKind.APPLY_PATTERN,
    )
    assert tuple(
        action.kind
        for action in _pattern_planning(
            "sample", PatternId.CLOSED_FAMILY_DISPATCH, ()
        ).actions
    ) == (
        RefactorActionKind.CREATE_DISPATCH_AUTHORITY,
        RefactorActionKind.REPLACE_BRANCH_SITES,
    )
    assert tuple(
        action.kind
        for action in _pattern_planning(
            "sample", PatternId.BIDIRECTIONAL_LOOKUP, ()
        ).actions
    ) == (
        RefactorActionKind.CREATE_BIDIRECTIONAL_REGISTRY,
        RefactorActionKind.DELETE_MIRRORED_UPDATES,
    )
