"""Simulation envelopes retain actual existing module-move preflight evidence."""

import json
from dataclasses import replace

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    MoveSymbolsToModuleOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodPlanPreflightReport,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.codemod_module_move_reports import (
    ModuleMoveDependencyReport,
)


def _move(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "shared.py").write_text("class Shared:\n    pass\n")
    source = package / "source.py"
    source.write_text("from .shared import Shared\n\nclass Helper(Shared):\n    pass\n")
    destination = package / "destination.py"
    destination.write_text("")
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path, use_parse_cache=False)
    )
    operation = MoveSymbolsToModuleOperation(
        target=SourceRewriteTarget(file_path=str(source)),
        symbol_qualnames=("Helper",),
        destination_path=str(destination),
    )
    return snapshot, operation, source, destination


def _assert_dependency_evidence(result):
    reports = result.preflight_report.reports
    assert reports
    dependencies = tuple(
        report.detail
        for report in reports
        if isinstance(report.detail, ModuleMoveDependencyReport)
    )
    assert len(dependencies) == 1
    assert dependencies[0].moved_symbol_names == ("Helper",)
    assert dependencies[0].imported_dependency_names == ("Shared",)
    assert result.preflight_report.is_clean
    wire = json.loads(json.dumps(json_report_object(result)))
    assert wire["preflight_report"] == json.loads(
        json.dumps(json_report_object(result.preflight_report))
    )
    assert wire["preflight_report"]["reports"]


def test_document_simulation_retains_original_preflight_receipt(tmp_path):
    snapshot, operation, _, _ = _move(tmp_path)
    document = CodemodPlanDocument(
        recipes=(RefactorRecipe("move", operations=(operation,)),)
    )
    preflight = document.preflight(snapshot)
    result = preflight.simulate()
    assert result.preflight_report is preflight.report
    _assert_dependency_evidence(result)


def test_recipe_simulation_transports_dependency_report_and_json(tmp_path):
    snapshot, operation, _, _ = _move(tmp_path)
    result = RefactorRecipe("move", operations=(operation,)).simulate(snapshot)
    _assert_dependency_evidence(result)


def test_sequence_aggregates_exact_stage_report_objects(tmp_path):
    snapshot, operation, _, _ = _move(tmp_path)
    result = CodemodPlanSequence.from_operations((operation,)).simulate(snapshot)
    stage = result.stage_reports[0].document_simulation
    _assert_dependency_evidence(stage)
    _assert_dependency_evidence(result)
    assert len(result.preflight_report.reports) == len(stage.preflight_report.reports)
    assert all(
        aggregate is original
        for aggregate, original in zip(
            result.preflight_report.reports, stage.preflight_report.reports
        )
    )
    wire = json.loads(json.dumps(json_report_object(result)))
    assert wire["stages"][0]["preflight_report"] == wire["preflight_report"]


@pytest.mark.parametrize("require_clean", (True, False))
def test_failed_source_preflight_cannot_apply_under_guard_override(
    tmp_path, require_clean
):
    snapshot, operation, source, destination = _move(tmp_path)
    result = CodemodPlanSequence.from_operations((operation,)).simulate(snapshot)
    invalid = replace(operation, destination_path=str(source))
    failure = CodemodPlanPreflightReport(invalid.preflight_reports(snapshot))
    assert not failure.is_clean
    rejected = replace(result, preflight_report=failure)
    assert not rejected.is_clean
    before = {path: path.read_bytes() for path in (source, destination)}
    with pytest.raises(CodemodOperationPreflightError):
        rejected.apply(require_clean=require_clean)
    assert {path: path.read_bytes() for path in before} == before
