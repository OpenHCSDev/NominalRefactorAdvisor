from __future__ import annotations

import nominal_refactor_advisor.patterns as patterns
from nominal_refactor_advisor.patterns import PatternId


def test_pattern_members_own_only_descriptive_required_relations() -> None:
    assert not hasattr(patterns, "PatternSpec")
    assert not hasattr(patterns, "PATTERN_SPECS")
    assert not hasattr(patterns, "PlanStepBuilderId")
    assert not hasattr(patterns, "ActionBuilderId")

    for pattern_id in PatternId:
        assert pattern_id.display_name
        assert pattern_id.required_relation
        assert pattern_id.witness_capabilities
        assert not hasattr(pattern_id, "prescription")
        assert not hasattr(pattern_id, "canonical_shape")
        assert not hasattr(pattern_id, "example_skeletons")
        assert not hasattr(pattern_id, "dependencies")
        assert not hasattr(pattern_id, "synergy_with")
        assert not hasattr(pattern_id, "phase")
        assert not hasattr(pattern_id, "first_moves")
        assert not hasattr(pattern_id, "priority")
