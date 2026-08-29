from __future__ import annotations

import nominal_refactor_advisor.patterns as patterns
from nominal_refactor_advisor.factorization import RefactorPhase
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.planner import (
    PatternActionBuilder,
    PatternPlanStepBuilder,
    _plan_step,
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


def test_builder_registries_are_keyed_directly_by_pattern_members() -> None:
    assert set(PatternPlanStepBuilder.__registry__) == {
        PatternId.CLOSED_FAMILY_DISPATCH,
        PatternId.ABC_TEMPLATE_METHOD,
        PatternId.AUTO_REGISTER_META,
        PatternId.BIDIRECTIONAL_LOOKUP,
        PatternId.AUTHORITATIVE_SCHEMA,
    }
    assert set(PatternActionBuilder.__registry__) == {
        PatternId.CLOSED_FAMILY_DISPATCH,
        PatternId.ABC_TEMPLATE_METHOD,
        PatternId.AUTO_REGISTER_META,
        PatternId.BIDIRECTIONAL_LOOKUP,
        PatternId.AUTHORITATIVE_SCHEMA,
    }
    assert all(
        PatternPlanStepBuilder.for_pattern(pattern_id).pattern_id is pattern_id
        for pattern_id in PatternPlanStepBuilder.__registry__
    )
    assert all(
        PatternActionBuilder.for_pattern(pattern_id).pattern_id is pattern_id
        for pattern_id in PatternActionBuilder.__registry__
    )


def test_generic_behavior_is_derived_from_missing_specialization() -> None:
    generic_pattern = PatternId.NOMINAL_BOUNDARY

    assert generic_pattern not in PatternPlanStepBuilder.__registry__
    assert generic_pattern not in PatternActionBuilder.__registry__
    assert generic_pattern.prescription in _plan_step("sample", generic_pattern, ())
