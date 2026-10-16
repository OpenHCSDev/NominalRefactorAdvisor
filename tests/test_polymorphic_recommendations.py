"""Recommendation policy must not upgrade observation to rewrite proof."""

from pathlib import Path

from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.detectors import (
    DetectorConfig,
    ExternalEnumCaseRecoveryDetector,
    ResidualClosedAxisIndirectionDetector,
)
from nominal_refactor_advisor.taxonomy import CertificationLevel


def findings(detector, source: str):
    parsed = SourceModule(Path("package/example.py"), "package.example", source).parse()
    return detector.detect([parsed], DetectorConfig())


def test_behavior_table_and_branch_prefers_case_owned_abc_without_mandatory_enum_registry():
    rows = findings(ResidualClosedAxisIndirectionDetector(), '''
from enum import Enum
class Direction(Enum):
    INPUT = "input"
    OUTPUT = "output"
READERS = {
    Direction.INPUT: lambda plan: plan.input_dir,
    Direction.OUTPUT: lambda plan: plan.output_dir,
}
def resolve(plan, direction, fallback):
    value = READERS[direction](plan)
    if value is not None:
        return value
    if direction is Direction.INPUT:
        return plan.initial_input
    return fallback
''')
    finding, = rows
    assert "READERS" in finding.summary
    assert "public ABC" in finding.why
    assert "case-owned declarations, data and hooks" in finding.why
    assert "AutoRegisterMeta" not in finding.why
    assert "value" in finding.why and "unknown-input" in finding.why
    assert finding.certification is CertificationLevel.STRONG_HEURISTIC


def test_external_value_queries_remain_observations_not_proved_subclass_obligations():
    rows = findings(ExternalEnumCaseRecoveryDetector(), '''
from enum import Enum
class Status(Enum):
    HIT = "hit"
    PARTIAL = "partial"
    MISS = "miss"
def reusable(status):
    if status is Status.HIT:
        return True
    return False
def incomplete(status):
    if status is Status.PARTIAL:
        return True
    return False
''')
    finding, = rows
    assert finding.metrics.plan_literal_cases == ("Status.HIT", "Status.PARTIAL")
    assert "For behavior-bearing cases" in finding.why
    assert "public ABC" in finding.why
    assert "value-only enum" in finding.why
    assert "not an automatic subclass conversion" in finding.why
    assert "shared Enum method" in finding.why
    assert "required hierarchy" in finding.why
    assert finding.certification is CertificationLevel.STRONG_HEURISTIC


def test_value_only_enum_and_derived_membership_does_not_nominate_dispatch():
    source = '''
from enum import Enum
class Format(Enum):
    CSV = "csv"
    JSON = "json"
def vocabulary():
    return tuple(member.value for member in Format)
def decode(raw):
    return Format(raw)
'''
    assert findings(ExternalEnumCaseRecoveryDetector(), source) == []
    assert findings(ResidualClosedAxisIndirectionDetector(), source) == []


def test_value_table_with_residual_query_keeps_conditional_recommendation():
    rows = findings(ResidualClosedAxisIndirectionDetector(), '''
from enum import Enum
class Mode(Enum):
    FIRST = "first"
    SECOND = "second"
FIRST_LIMIT = 3
SECOND_LIMIT = 5
LIMITS = {Mode.FIRST: FIRST_LIMIT, Mode.SECOND: SECOND_LIMIT}
def limit(mode):
    value = LIMITS[mode]
    if mode is Mode.FIRST:
        return value
    return value
''')
    finding, = rows
    assert "If the sites only encode values" in finding.why
    assert "before recommending a hierarchy" in finding.why
    assert finding.certification is CertificationLevel.STRONG_HEURISTIC
