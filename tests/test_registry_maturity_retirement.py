"""Removing an unsupported warning preserves actual registry behavior."""

import ast
from pathlib import Path
from types import ModuleType

import pytest

from registry_test_sources import keyed_registry_source
from nominal_refactor_advisor.analysis import analyze_modules
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.detectors import DetectorConfig, IssueDetector


@pytest.mark.parametrize("use_alias", (False, True))
def test_registry_consumer_spelling_does_not_create_maturity_obligations(
    use_alias,
) -> None:
    source = keyed_registry_source()
    if use_alias:
        source = source.replace(
            "\ndef run_alpha", "\nfamily = ModeRunner\n\ndef run_alpha"
        )
        source = source.replace(
            "return ModeRunner.for_mode(", "return family.for_mode("
        )
    runtime = ModuleType("maturity_control")
    exec(source, runtime.__dict__)
    assert (runtime.run_alpha(), runtime.run_beta()) == ("alpha", "beta")
    if use_alias:
        assert runtime.family is runtime.ModeRunner
    parsed = ParsedModule(
        Path("/repo/family.py"), "family", False, ast.parse(source), source
    )
    findings = analyze_modules([parsed], DetectorConfig())
    assert all(f.detector_id != "premature_registry_infrastructure" for f in findings)
    (projection,) = CompactModuleClassProjectionFamily.collect_modules((parsed,))
    assert projection.autoregister_reference_index is not None
    receiver_name = "family" if use_alias else "ModeRunner"
    assert receiver_name in projection.autoregister_reference_index.receiver_names


def test_maturity_retirement_preserves_executable_type_keyed_detector() -> None:
    ids = {d.effective_detector_id() for d in IssueDetector.registered_detector_types()}
    assert "premature_registry_infrastructure" not in ids
    assert "type_keyed_behavior_projection" in ids
    assert "registry_projection_surface" in ids
