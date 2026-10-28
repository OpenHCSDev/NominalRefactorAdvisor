"""Relocated probes: each historical relocation from #82 is flagged where it landed."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.detectors import (
    ConstantSubtypeAnswerDetector,
    DetectorConfig,
    ExceptionAsAbsenceDetector,
    LambdaFoldDetector,
    OwnStateBranchAnswerDetector,
    ThreadedAbsenceDetector,
    ZeroOneTupleDetector,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache

from relocation_excerpts import (
    DECLARED_ANNOTATIONS,
    DECLARES_DATACLASS,
    EXCEPTION_ABSENCE,
    FOLD_BINDING,
    GRAPH_PARAMETERS,
    IS_OWNER,
    KEYED_METACLASS_FAMILY,
    SPELLS_AS_CLAUSE,
)

DETECTORS = (
    ConstantSubtypeAnswerDetector,
    OwnStateBranchAnswerDetector,
    ExceptionAsAbsenceDetector,
    LambdaFoldDetector,
    ZeroOneTupleDetector,
    ThreadedAbsenceDetector,
)


def _findings(detector: type, root: Path):
    with ScanCache.scope():
        modules = parse_python_modules(root, use_parse_cache=False)
        return detector().detect(modules, DetectorConfig())


def _sources(tmp_path: Path, sources: dict[str, str]) -> Path:
    for name, source in sources.items():
        (tmp_path / f"{name}.py").write_text(source)
    return tmp_path


@pytest.mark.parametrize("detector", DETECTORS)
def test_every_finding_states_fact_owner_and_prescription(detector):
    spec = detector.finding_spec
    assert spec.certification.value == "certified"
    assert detector.restated.fact in spec.why
    assert detector.restated.owner in spec.why
    assert not detector.restated.operations


def test_relocation_patterns_continue_the_stable_sequence():
    assert [
        PatternId.CONSTANT_SUBTYPE_ANSWER,
        PatternId.OWN_STATE_BRANCH_ANSWER,
        PatternId.EXCEPTION_AS_ABSENCE,
        PatternId.LAMBDA_FOLD,
        PatternId.ZERO_ONE_TUPLE,
        PatternId.THREADED_ABSENCE,
    ] == [PatternId(number) for number in range(40, 46)]


# ---- the historical relocations, read from git at the commits that made them


@pytest.mark.parametrize(
    ("relocation", "detector", "flagged"),
    (
        (DECLARES_DATACLASS, ConstantSubtypeAnswerDetector, "`declares_dataclass` is answered by a constant in 2 members"),
        (IS_OWNER, OwnStateBranchAnswerDetector, "`NamedOwnerDeclaration.is_owner` (property)"),
        (KEYED_METACLASS_FAMILY, OwnStateBranchAnswerDetector, "`AutoRegisterClassAuthority.keyed_metaclass_family` (property) probes `self.registry_key_attribute`"),
        (SPELLS_AS_CLAUSE, OwnStateBranchAnswerDetector, "`ImportAlias.spells_as_clause` (property) probes `self.asname`"),
        (FOLD_BINDING, LambdaFoldDetector, "`fold_binding` folds over its callables"),
        (DECLARED_ANNOTATIONS, ZeroOneTupleDetector, "`declared_annotations` encodes absence as a 0/1 tuple"),
        (GRAPH_PARAMETERS, ThreadedAbsenceDetector, "takes `cached_semantic_descent_graph` as `X | None`"),
    ),
)
def test_each_historical_relocation_is_flagged(tmp_path, relocation, detector, flagged):
    summaries = [finding.summary for finding in _findings(detector, relocation.write(tmp_path))]
    assert any(flagged in summary for summary in summaries), summaries


def test_the_historical_exception_sites_resolve_to_the_declared_exceptions(tmp_path):
    """12 of the 16 sites; the 4 in `detectors/_helpers.py` follow `from ._base
    import *`, which may rebind the caught name, so they stay unresolved."""

    findings = _findings(ExceptionAsAbsenceDetector, EXCEPTION_ABSENCE.write(tmp_path))
    assert len(findings) == 12
    assert {
        caught
        for finding in findings
        for caught in (
            "nominal_refactor_advisor.ast_projection.AbsentStatementValue",
            "nominal_refactor_advisor.class_mro.UnboundReference",
        )
        if f"catches `{caught}`" in finding.summary
    } == {
        "nominal_refactor_advisor.ast_projection.AbsentStatementValue",
        "nominal_refactor_advisor.class_mro.UnboundReference",
    }
    assert not any("_helpers.py" in finding.evidence[0].file_path for finding in findings)


# ---- what each detector leaves alone


def test_a_family_member_that_computes_its_answer_is_not_a_constant_family(tmp_path):
    source = (
        "class Base:\n    def flag(self) -> bool:\n        raise NotImplementedError\n\n"
        "class A(Base):\n    def flag(self) -> bool:\n        return True\n\n"
        "class B(Base):\n    def flag(self) -> bool:\n        return self.x > 1\n\n"
        "def use(item):\n    if item.flag():\n        return 1\n"
    )
    assert _findings(ConstantSubtypeAnswerDetector, _sources(tmp_path, {"m": source})) == []


def test_a_mutable_object_or_an_unbranched_bool_is_not_an_own_state_answer(tmp_path):
    source = (
        "from dataclasses import dataclass\n\n"
        "class Live:\n    @property\n    def ready(self) -> bool:\n        return self.count > 0\n\n"
        "@dataclass(frozen=True)\nclass Fixed:\n    value: int | None\n"
        "    @property\n    def present(self) -> bool:\n        return self.value is not None\n\n"
        "def use(live, fixed):\n    if live.ready:\n        return fixed.present\n"
    )
    assert _findings(OwnStateBranchAnswerDetector, _sources(tmp_path, {"m": source})) == []


def test_a_reraise_or_a_foreign_exception_is_not_absence(tmp_path):
    source = (
        "class Missing(ValueError):\n    pass\n\n"
        "def f(cache, key):\n"
        "    try:\n        return cache.lookup(key)\n    except Missing:\n        raise\n\n"
        "def g(cache, key):\n"
        "    try:\n        return cache.lookup(key)\n    except KeyError:\n        return None\n"
    )
    assert _findings(ExceptionAsAbsenceDetector, _sources(tmp_path, {"m": source})) == []


def test_a_defaulted_optional_parameter_is_not_threaded_absence(tmp_path):
    source = (
        "def f(config: int | None = None):\n"
        "    if config is None:\n        config = 1\n    return config\n"
    )
    assert _findings(ThreadedAbsenceDetector, _sources(tmp_path, {"m": source})) == []


def test_a_function_returning_only_empty_tuples_is_not_a_0_1_tuple(tmp_path):
    source = "def f(x):\n    if x:\n        return ()\n    return ()\n"
    assert _findings(ZeroOneTupleDetector, _sources(tmp_path, {"m": source})) == []
