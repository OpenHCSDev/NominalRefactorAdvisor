"""Rendered registry conversions retain unmet candidate-native obligations."""

import builtins
from functools import cached_property
import json
from pathlib import Path
import subprocess
import sys
import typing

import metaclass_registry
import pytest

from nominal_refactor_advisor.captured_reference import InitialNativeIsland
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    PatchTargetOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_native_requirements import (
    NativeUseProvenance,
    NativeUseResolution,
)
from nominal_refactor_advisor.codemod_runtime import (
    CurrentSnapshotRecipeBatchEvaluation,
    ExecutableRecipeEvaluation,
    FindingRecipePlanBuilder,
    FindingRecipePlanCandidate,
    FindingRecipeSetDisposition,
    FindingRecipeSynthesisRecord,
    RefactorRecipeOperationCompiler,
)
from nominal_refactor_advisor.codemod_preflight import CodemodOperationPreflightError
from nominal_refactor_advisor.codemod_reproof import SourceReproofDiagnostic
from nominal_refactor_advisor.codemod_source_edits import (
    CodemodSourceRevision,
    SourceTextReplacement,
)
from nominal_refactor_advisor.product_flow_authority import SourceProductFlowRepository
from nominal_refactor_advisor.source_entry import ImportedSourceModuleEntryPremise
from nominal_refactor_advisor.finding_recipe_actions import FindingRecipeActionKey
from nominal_refactor_advisor.models import FindingSpec, PatternId, SourceLocation

SOURCE = """padding = None
REGISTRY = {}
class Alpha:
    pass
class Beta:
    pass
REGISTRY['alpha'] = Alpha
REGISTRY['beta'] = Beta
"""


def registry_entry(source):
    """Explicit native storage premise, not an invariant on mutable behavior."""
    initial = InitialNativeIsland((builtins, typing, metaclass_registry))
    return ImportedSourceModuleEntryPremise.from_standard_source_loader(
        source, initial, initial.namespace_for_storage(vars(builtins))
    )


class RegistryEntryRepository(SourceProductFlowRepository):
    source_entry = staticmethod(registry_entry)


class RegistryEntryCompiler(RefactorRecipeOperationCompiler):
    @cached_property
    def product_flow_repository(self):
        return RegistryEntryRepository.from_modules(self.parsed_modules)


def conversion(tmp_path, source=SOURCE, *, compiler=CodemodSourceSnapshot, later=()):
    path = tmp_path / "registry.py"
    path.write_text(source)
    snapshot = compiler.from_source_mapping({str(path): source})
    operation = ConvertManualRegistryToAutoregisterOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="Alpha")
    )
    document = CodemodPlanDocument(
        recipes=(RefactorRecipe("registry", operations=(operation, *later)),)
    )
    return path, operation, document.simulate(snapshot)


def native_outcome(source):
    """Execute only this test's authored fixture in a separate interpreter."""
    program = """
import json
import sys
namespace = {}
try:
    exec(compile(sys.stdin.read(), '<registry-native-control>', 'exec'), namespace)
except Exception as error:
    print(json.dumps({'error': type(error).__name__}))
else:
    assert namespace['REGISTRY'] == {
        'alpha': namespace['Alpha'], 'beta': namespace['Beta']}
    print(json.dumps({'keys': sorted(namespace['REGISTRY'])}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        input=source,
        capture_output=True,
        text=True,
        timeout=15,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    return json.loads(completed.stdout)


def candidate_reports(simulation):
    return tuple(
        report
        for report in simulation.preflight_report.reports
        if isinstance(report.detail, NativeUseResolution)
    )


def require_preview_only(path, original, simulation):
    assert simulation.simulation.rewritten_sources[str(path)]
    assert simulation.unified_diff({str(path): original})
    assert not simulation.is_clean
    for require_clean in (True, False):
        with pytest.raises(ValueError):
            simulation.apply(require_clean=require_clean)
        assert path.read_bytes() == original.encode()


@pytest.mark.parametrize(
    "wrong_creator", (False, True), ids=("native-positive", "wrong-metaclass")
)
def test_native_candidate_controls_remain_preview_without_behavior_proof(
    tmp_path, wrong_creator
):
    source = (
        SOURCE.replace("padding = None", "AutoRegisterMeta = int")
        if wrong_creator
        else SOURCE
    )
    path, operation, simulation = conversion(tmp_path, source)
    candidate = simulation.required_after_snapshot.sources_by_file_path[str(path)]
    assert native_outcome(source) == {"keys": ["alpha", "beta"]}
    assert native_outcome(candidate) == (
        {"error": "TypeError"} if wrong_creator else {"keys": ["alpha", "beta"]}
    )
    (requirement,) = operation.candidate_native_use_requirements(simulation)
    assert requirement.inspect().provenance is NativeUseProvenance.UNRESOLVED
    (report,) = candidate_reports(simulation)
    assert report.detail == requirement.inspect()
    assert report.status.is_failed
    require_preview_only(path, source, simulation)


def test_captured_generated_creator_identity_is_not_behavior_invariance(tmp_path):
    path, operation, simulation = conversion(tmp_path, compiler=RegistryEntryCompiler)
    (requirement,) = operation.candidate_native_use_requirements(simulation)
    assert requirement.inspect().provenance is NativeUseProvenance.CAPTURED_IDENTITY
    (report,) = candidate_reports(simulation)
    assert report.detail == requirement.inspect()
    assert report.status.is_failed
    require_preview_only(path, SOURCE, simulation)


def test_candidate_gate_observes_later_edit_in_complete_document(tmp_path):
    path = tmp_path / "registry.py"
    patch = PatchTargetOperation(
        target=SourceRewriteTarget(file_path=str(path)),
        replacements=(
            SourceTextReplacement("padding = None", "AutoRegisterMeta = int"),
        ),
    )
    path, operation, simulation = conversion(tmp_path, later=(patch,))
    candidate = simulation.required_after_snapshot.sources_by_file_path[str(path)]
    assert "AutoRegisterMeta = int" in candidate
    assert native_outcome(SOURCE) == {"keys": ["alpha", "beta"]}
    assert native_outcome(candidate) == {"error": "TypeError"}
    (requirement,) = operation.candidate_native_use_requirements(simulation)
    (report,) = candidate_reports(simulation)
    assert report.detail.receipt == requirement.receipt
    assert (
        report.detail.receipt.revision.source_hash
        == CodemodSourceRevision.hash_source(candidate)
    )
    assert (
        report.detail.receipt.revision.source_hash
        != CodemodSourceRevision.hash_source(SOURCE)
    )
    require_preview_only(path, SOURCE, simulation)


def test_candidate_analysis_does_not_execute_authored_function_body(tmp_path):
    marker = tmp_path / "must_not_execute"
    source = SOURCE.replace(
        "class Alpha:\n    pass",
        "class Alpha:\n"
        "    def run(self):\n"
        f"        open({str(marker)!r}, 'w').write('executed')\n"
        "        raise RuntimeError('analysis executed source')",
    )
    path, operation, simulation = conversion(tmp_path, source)
    operation.candidate_native_use_requirements(simulation)[0].inspect()
    assert not marker.exists()
    assert candidate_reports(simulation)
    require_preview_only(path, source, simulation)
    assert not marker.exists()


def test_known_wrong_creator_retains_structured_reproof_diagnostic(tmp_path):
    source = "from metaclass_registry import AutoRegisterMeta\n" + SOURCE
    patch = PatchTargetOperation(
        target=SourceRewriteTarget(file_path=str(tmp_path / "registry.py")),
        replacements=(
            SourceTextReplacement(
                "from metaclass_registry import AutoRegisterMeta",
                "from builtins import int as AutoRegisterMeta",
            ),
        ),
    )
    path, operation, simulation = conversion(
        tmp_path, source, compiler=RegistryEntryCompiler, later=(patch,)
    )
    (requirement,) = operation.candidate_native_use_requirements(simulation)
    with pytest.raises(ValueError) as refusal:
        requirement.inspect()
    assert not candidate_reports(simulation)
    (report,) = tuple(
        report
        for report in simulation.preflight_report.reports
        if isinstance(report.detail, SourceReproofDiagnostic)
    )
    assert report.status.is_failed
    assert report.operation == operation.operation_key()
    assert report.detail.target == operation.target
    assert report.message == str(refusal.value)
    assert report.detail.causes[0].message == str(refusal.value)
    assert native_outcome(source) == {"keys": ["alpha", "beta"]}
    candidate = simulation.required_after_snapshot.sources_by_file_path[str(path)]
    assert native_outcome(candidate) == {"error": "TypeError"}
    require_preview_only(path, source, simulation)


def test_recipe_apply_and_sequence_preserve_candidate_refusal(tmp_path):
    source = "from metaclass_registry import AutoRegisterMeta\n" + SOURCE
    patch = PatchTargetOperation(
        target=SourceRewriteTarget(file_path=str(tmp_path / "registry.py")),
        replacements=(
            SourceTextReplacement(
                "from metaclass_registry import AutoRegisterMeta",
                "from builtins import int as AutoRegisterMeta",
            ),
        ),
    )
    path, operation, _ = conversion(
        tmp_path, source, compiler=RegistryEntryCompiler, later=(patch,)
    )
    snapshot = RegistryEntryCompiler.from_source_mapping({str(path): source})
    recipe = RefactorRecipe("registry", operations=(operation, patch))
    preview = recipe.simulate(snapshot)
    assert any(
        isinstance(report.detail, SourceReproofDiagnostic)
        for report in preview.preflight_report.reports
    )
    require_preview_only(path, source, preview)

    sequence = CodemodPlanSequence(documents=(CodemodPlanDocument(recipes=(recipe,)),))
    preflight = sequence.preflight_snapshot(snapshot)
    assert preflight.preflight_failed
    with pytest.raises(CodemodOperationPreflightError) as refusal:
        sequence.simulate(snapshot)
    assert isinstance(refusal.value.report.detail, SourceReproofDiagnostic)
    assert refusal.value.report.status.is_failed
    assert path.read_bytes() == source.encode()


def test_actual_candidate_planner_retains_native_refusal_cause(tmp_path):
    path, operation, preview = conversion(tmp_path)
    (native_report,) = candidate_reports(preview)
    assert native_report.detail.provenance is NativeUseProvenance.UNRESOLVED
    assert preview.architecture_guard_report.violation_count == 0
    finding = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Manual registry conversion candidate",
        why="Registration belongs to its class declaration.",
        capability_gap="A proved generated registration authority",
        relation_context="Generated creator behavior requires its own proof.",
    ).build(
        "registry_candidate_gate",
        "Evaluate the real converter's candidate-native obligation.",
        (SourceLocation(str(path), 3, "Alpha"),),
    )
    candidate = FindingRecipePlanCandidate(
        FindingRecipeSynthesisRecord(
            finding=finding,
            evaluation=ExecutableRecipeEvaluation(
                executable_recipe=RefactorRecipe("registry", operations=(operation,)),
                evaluation_declaration_type=CurrentSnapshotRecipeBatchEvaluation,
            ),
            action_keys=(
                FindingRecipeActionKey(
                    detector_id=finding.detector_id,
                    file_path=str(path),
                    subject_name="Alpha",
                ),
            ),
        )
    )
    planner = CurrentSnapshotRecipeBatchEvaluation(
        candidates=(candidate,),
        source_snapshot=CodemodSourceSnapshot.from_source_mapping({str(path): SOURCE}),
        batch_projection=FindingRecipePlanBuilder(()),
    )
    assessment = planner.simulate_recipe_set((0,)).assessment
    assert assessment.disposition is FindingRecipeSetDisposition.UNPROVED
    assert native_report.message in assessment.reason
    assert "architecture guard" not in assessment.reason
    (record,) = planner.solve().records
    assert record.reason == assessment.reason
    assert not record.candidate_recipes
    assert path.read_bytes() == SOURCE.encode()
