"""Pending native Enum-key identity obligation, without xfail or skip."""

import ast
from pathlib import Path
from types import ModuleType

import pytest

from registry_test_sources import keyed_registry_source
from nominal_refactor_advisor.analysis import analyze_detector_types
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.detectors import DetectorConfig, IssueDetector


@pytest.mark.parametrize("alias_keys", (False, True))
def test_injective_registry_claim_respects_native_key_identity(alias_keys) -> None:
    source = keyed_registry_source()
    if alias_keys:
        source = source.replace("BETA = auto()", "BETA = ALPHA")
    runtime = ModuleType("registry_key_identity")
    exec(source, runtime.__dict__)
    assert (runtime.Mode.ALPHA is runtime.Mode.BETA) is alias_keys
    native_injective = (
        len({runtime.AlphaModeRunner.mode, runtime.BetaModeRunner.mode}) == 2
    )
    parsed = ParsedModule(
        Path("/repo/family.py"), "family", False, ast.parse(source), source
    )
    detectors = tuple(
        d
        for d in IssueDetector.registered_detector_types()
        if d.effective_detector_id() == "injective_type_registry"
    )
    assert len(detectors) == 1
    findings = analyze_detector_types(
        [parsed], DetectorConfig(), detector_types=detectors
    )
    assert (
        bool(findings) is native_injective
    ), "Different key spellings do not prove different native keys"
