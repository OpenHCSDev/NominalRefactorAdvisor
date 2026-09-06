from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule, parse_python_modules
from nominal_refactor_advisor.codemod import (
    ArchitectureGuardSuite,
    CollapseClosedParameterConveyorOperation,
    CodemodSourceSnapshot,
    FindingRecipeSynthesisStatus,
    RefactorRecipeOperation,
    SemanticCarrierConcept,
    codemod_plan_from_findings,
)
from nominal_refactor_advisor.codemod_workflow import (
    CodemodRefactorGoalRunner,
    CodemodRefactorTrajectoryStatus,
    CodemodWorkflowScan,
    CodemodWorkflowStopReason,
)
from nominal_refactor_advisor.detectors import ClosedParameterConveyorDetector
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.models import ParameterThreadMetrics, SourceLocation
from nominal_refactor_advisor.parameter_conveyor import (
    ClosedParameterConveyorAuthorityViolation,
    ClosedParameterConveyorComponentBuilder,
)
from nominal_refactor_advisor.patterns import PatternId


def _module(module_name: str, source: str) -> ParsedModule:
    path = Path(*module_name.split(".")).with_suffix(".py")
    return ParsedModule(
        path=path,
        module_name=module_name,
        is_package_init=False,
        module=ast.parse(source, filename=str(path)),
        source=source,
    )


def _builder(*modules: ParsedModule) -> ClosedParameterConveyorComponentBuilder:
    return ClosedParameterConveyorComponentBuilder.from_modules(modules)


def _base_source(*, callee_body: str = "    return left, right\n") -> str:
    return (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class _CacheKey:\n"
        "    left: object\n"
        "    right: object\n"
        "\n"
        "def _build(left, right):\n"
        f"{callee_body}"
        "\n"
        "def caller(left, right):\n"
        "    key = _CacheKey(left=left, right=right)\n"
        "    return _build(left, right)\n"
    )


def _write_source(root: Path, relative_path: str, source: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8", newline="")
    return path


@pytest.mark.parametrize(
    "product_declaration",
    (
        "@dataclass\nclass _CacheKey:\n    left: object\n    right: InitVar[object]\n",
        "@dataclass\n"
        "class _CacheKey:\n"
        "    left: object\n"
        "    right: object = field(init=False)\n",
        "@dataclass(init=False)\n"
        "class _CacheKey:\n"
        "    left: object\n"
        "    right: object\n"
        "    def __init__(self, left, right):\n"
        "        self.left, self.right = right, left\n",
        "@dataclass(frozen=True)\n"
        "class _CacheKey:\n"
        "    left: object\n"
        "    right: object\n"
        "    def __post_init__(self):\n"
        "        object.__setattr__(self, 'left', normalize(self.left))\n",
        "@dataclass\n"
        "class _CacheKey:\n"
        "    left: object\n"
        "    right: object\n"
        "    def __getattribute__(self, name):\n"
        "        return transform(super().__getattribute__(name))\n",
    ),
)
def test_lifecycle_open_products_never_form_proven_parameter_conveyors(
    product_declaration: str,
) -> None:
    builder = _builder(
        _module(
            "pkg.lifecycle_open",
            "from dataclasses import InitVar, dataclass, field\n"
            "\n"
            f"{product_declaration}"
            "\n"
            "def _build(left, right):\n"
            "    return left, right\n"
            "\n"
            "def caller(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _build(left, right)\n",
        )
    )

    assert builder.proven_components() == ()


def test_observed_root_carrier_keeps_parameter_conveyor_open() -> None:
    builder = _builder(
        _module(
            "pkg.observed_carrier",
            _base_source().replace(
                "    return _build(left, right)\n",
                "    audit(key)\n    return _build(left, right)\n",
            ),
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.OBSERVED_ROOT_CARRIER in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_attribute_source_evaluation_keeps_parameter_conveyor_open() -> None:
    builder = _builder(
        _module(
            "pkg.repeated_source",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _build(left, right):\n"
            "    return left, right\n"
            "\n"
            "def caller(source):\n"
            "    key = _CacheKey(left=source.left, right=source.right)\n"
            "    return _build(source.left, source.right)\n",
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.REPEATED_SOURCE_EVALUATION in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


@pytest.mark.parametrize(
    "signature",
    (
        "def _build(left: object, right: object):\n",
        "def _build(left=None, right=None):\n",
    ),
)
def test_parameter_declaration_semantics_keep_conveyor_open(signature: str) -> None:
    builder = _builder(
        _module(
            "pkg.parameter_declaration",
            _base_source().replace("def _build(left, right):\n", signature),
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.SIGNATURE_SEMANTICS_HAZARD in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_complete_closed_parameter_conveyor_is_proven_as_one_component() -> None:
    builder = _builder(_module("pkg.complete", _base_source()))

    components = builder.proven_components()

    assert len(components) == 1
    component = components[0]
    assert component.authority.class_symbol == "pkg.complete._CacheKey"
    assert component.participant_symbols == ("pkg.complete._build",)
    assert len(component.root_edges) == 1
    assert component.forwarding_edges == ()
    assert component.proof.batch_compression_delta > 0


def test_detector_emits_one_authority_anchored_certified_finding() -> None:
    module = _module("pkg.complete", _base_source())

    findings = ClosedParameterConveyorDetector().detect(
        [module],
        DetectorConfig(),
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.detector_id == "closed_parameter_conveyor"
    assert finding.pattern_id is PatternId.AUTHORITATIVE_CONTEXT
    assert finding.certification == "certified"
    assert finding.authority_evidence == SourceLocation(
        "pkg/complete.py",
        4,
        "pkg.complete._CacheKey",
    )
    assert finding.metrics == ParameterThreadMetrics(
        function_count=1,
        shared_parameter_count=2,
        shared_parameter_names=("left", "right"),
    )


def test_detector_does_not_emit_an_open_parameter_conveyor() -> None:
    module = _module(
        "pkg.open_tail",
        _base_source()
        + "\n"
        + "def unconverted(left, right):\n"
        + "    return _build(transform(left), right)\n",
    )

    findings = ClosedParameterConveyorDetector().detect(
        [module],
        DetectorConfig(),
    )

    assert findings == []


def test_proven_finding_compiles_to_an_authority_keyed_atomic_rewrite() -> None:
    module = _module("pkg.complete", _base_source())
    findings = ClosedParameterConveyorDetector().detect(
        [module],
        DetectorConfig(),
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,), findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)

    assert len(plan.records) == 1
    assert plan.records[0].status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    assert len(plan.document.recipes) == 1
    operation = plan.document.recipes[0].operations[0]
    authority_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if snapshot.source_index.symbol_for_target(target) == "pkg.complete._CacheKey"
    )
    assert operation.target.target_id == authority_target.target_id
    assert isinstance(operation, CollapseClosedParameterConveyorOperation)
    operation_payload = json_report_object(operation)
    assert RefactorRecipeOperation.from_json_value(operation_payload) == operation
    assert "source_edits_by_state_id" not in operation_payload

    simulation = snapshot.simulate_rewrites(
        plan.document.source_rewrite_batch(snapshot)
    )
    rewritten_source = simulation.rewritten_sources[module.file_path]
    assert rewritten_source == _base_source().replace(
        "def _build(left, right):\n    return left, right\n",
        "def _build(*, cache_key: '_CacheKey'):\n"
        "    return cache_key.left, cache_key.right\n",
    ).replace(
        "    return _build(left, right)\n",
        "    return _build(cache_key=key)\n",
    )
    original_namespace = {"__name__": "pkg.complete_original"}
    rewritten_namespace = {"__name__": "pkg.complete_rewritten"}
    exec(
        compile(module.source, module.file_path, "exec", dont_inherit=True),
        original_namespace,
    )
    exec(
        compile(rewritten_source, module.file_path, "exec", dont_inherit=True),
        rewritten_namespace,
    )
    assert original_namespace["caller"]("left", "right") == rewritten_namespace[
        "caller"
    ]("left", "right")

    rewritten_snapshot = snapshot.with_virtual_sources(simulation.rewritten_sources)
    assert (
        ClosedParameterConveyorDetector().detect(
            rewritten_snapshot.parsed_modules,
            DetectorConfig(),
        )
        == []
    )


def test_document_simulation_reuses_one_current_source_reproof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("pkg.complete", _base_source())
    findings = ClosedParameterConveyorDetector().detect(
        [module],
        DetectorConfig(),
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,), findings)
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    original_reproof = CollapseClosedParameterConveyorOperation.source_edits_from_snapshot
    reproof_count = 0

    def counted_reproof(
        operation: CollapseClosedParameterConveyorOperation,
        current_snapshot: CodemodSourceSnapshot,
    ):
        nonlocal reproof_count
        reproof_count += 1
        return original_reproof(operation, current_snapshot)

    monkeypatch.setattr(
        CollapseClosedParameterConveyorOperation,
        "source_edits_from_snapshot",
        counted_reproof,
    )

    simulation = plan.document.simulate(snapshot)

    assert simulation.is_clean
    assert reproof_count == 1


def test_parameter_conveyor_goal_runner_proves_one_terminal_replay(
    tmp_path: Path,
) -> None:
    source_path = _write_source(tmp_path, "pkg/complete.py", _base_source())
    modules = parse_python_modules(tmp_path)
    findings = ClosedParameterConveyorDetector().detect(
        modules,
        DetectorConfig(),
    )
    starting_snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        migration_type=SemanticCarrierConcept,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
    ).run()

    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED
    assert report.trajectory_proof.status is CodemodRefactorTrajectoryStatus.PROVED
    assert report.trajectory_proof.visited_state_count == 2
    assert report.trajectory_proof.transition_count == 1
    assert len(report.trajectory_proof.terminals) == 1
    assert report.final_target_finding_ids == ()
    assert len(report.stages) == 1
    assert report.stages[0].progress.achieved is True
    replay_operation = report.replay_sequence.documents[0].recipes[0].operations[0]
    assert isinstance(replay_operation, CollapseClosedParameterConveyorOperation)

    replay = report.replay_sequence.simulate(starting_snapshot)

    assert replay.is_clean
    assert replay.final_snapshot.sources_by_file_path[source_path.as_posix()] == (
        _base_source()
        .replace(
            "def _build(left, right):\n    return left, right\n",
            "def _build(*, cache_key: '_CacheKey'):\n"
            "    return cache_key.left, cache_key.right\n",
        )
        .replace(
            "    return _build(left, right)\n",
            "    return _build(cache_key=key)\n",
        )
    )
    assert (
        ClosedParameterConveyorDetector().detect(
            replay.final_snapshot.parsed_modules,
            DetectorConfig(),
        )
        == []
    )


def test_parameter_conveyor_cli_proves_exports_and_applies_the_goal(
    tmp_path: Path,
) -> None:
    source_path = _write_source(tmp_path, "pkg/complete.py", _base_source())
    replay_path = tmp_path / "replay.json"
    repository_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "nominal_refactor_advisor",
        tmp_path.as_posix(),
        "--no-cache",
        "--codemod-refactor-goal",
        SemanticCarrierConcept.concept_key(),
        "--json",
    ]

    preview = subprocess.run(
        [*command, "--codemod-plan-out", replay_path.as_posix()],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    preview_payload = json.loads(preview.stdout)

    assert preview.returncode == 0, preview.stderr
    assert preview_payload["stop_reason"] == "achieved"
    assert preview_payload["trajectory_proof"]["status"] == "proved"
    assert preview_payload["stage_count"] == 1
    assert replay_path.exists()
    assert source_path.read_text(encoding="utf-8") == _base_source()

    application = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--codemod-plan",
            replay_path.as_posix(),
            "--codemod-apply",
            "--json",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    application_payload = json.loads(application.stdout)

    assert application.returncode == 0, application.stderr
    assert application_payload["applied"] is True
    assert application_payload["applied_rewrite_count"] == 2
    rewritten_source = source_path.read_text(encoding="utf-8")
    assert "def _build(*, cache_key: '_CacheKey'):" in rewritten_source
    assert "return _build(cache_key=key)" in rewritten_source
    compile(rewritten_source, source_path.as_posix(), "exec")

    second_run = subprocess.run(
        command,
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    second_run_payload = json.loads(second_run.stdout)

    assert second_run.returncode == 0, second_run.stderr
    assert second_run_payload["stop_reason"] == "achieved"
    assert second_run_payload["stage_count"] == 0
    assert second_run_payload["trajectory_proof"]["status"] == "proved"


def test_parameter_conveyor_recipe_reproves_current_source_before_rewriting() -> None:
    module = _module("pkg.stale", _base_source())
    findings = ClosedParameterConveyorDetector().detect(
        [module],
        DetectorConfig(),
    )
    original_snapshot = CodemodSourceSnapshot.from_modules((module,), findings)
    current_snapshot = original_snapshot.with_virtual_sources(
        {
            module.file_path: _base_source()
            + "\n"
            + "def unconverted(left, right):\n"
            + "    return _build(transform(left), right)\n"
        }
    )

    plan = codemod_plan_from_findings(findings, selector_context=current_snapshot)

    assert plan.document.recipes == ()
    assert len(plan.records) == 1
    assert plan.records[0].status is (
        FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    )
    assert (
        "parameter-conveyor rewrite requires a proven component"
        in plan.records[0].reason
    )


def test_parameter_conveyor_rewrite_collapses_a_multistep_chain() -> None:
    source = (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class _CacheKey:\n"
        "    left: object\n"
        "    right: object\n"
        "\n"
        "def _second(first_value, second_value):\n"
        "    return first_value, second_value\n"
        "\n"
        "def _first(left, right):\n"
        "    first_alias = left\n"
        "    second_alias = right\n"
        "    return _second(first_alias, second_alias)\n"
        "\n"
        "def caller(left, right):\n"
        "    key = _CacheKey(left=left, right=right)\n"
        "    return _first(left, right)\n"
    )
    module = _module("pkg.chain_rewrite", source)
    findings = ClosedParameterConveyorDetector().detect(
        [module],
        DetectorConfig(),
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,), findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    simulation = snapshot.simulate_rewrites(
        plan.document.source_rewrite_batch(snapshot)
    )
    rewritten_source = simulation.rewritten_sources[module.file_path]

    assert "def _second(*, cache_key: '_CacheKey'):" in rewritten_source
    assert "return cache_key.left, cache_key.right" in rewritten_source
    assert "def _first(*, cache_key: '_CacheKey'):" in rewritten_source
    assert "first_alias = cache_key.left" in rewritten_source
    assert "second_alias = cache_key.right" in rewritten_source
    assert "return _second(cache_key=cache_key)" in rewritten_source
    assert "return _first(cache_key=key)" in rewritten_source
    rewritten_snapshot = snapshot.with_virtual_sources(simulation.rewritten_sources)
    assert (
        ClosedParameterConveyorDetector().detect(
            rewritten_snapshot.parsed_modules,
            DetectorConfig(),
        )
        == []
    )


def test_parameter_conveyor_rewrite_targets_multiple_calls_on_one_line_exactly() -> (
    None
):
    source = _base_source().replace(
        "    return _build(left, right)\n",
        "    return _build(left, right), _build(left, right)\n",
    )
    module = _module("pkg.same_line", source)
    findings = ClosedParameterConveyorDetector().detect(
        [module],
        DetectorConfig(),
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,), findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    simulation = snapshot.simulate_rewrites(
        plan.document.source_rewrite_batch(snapshot)
    )

    assert (
        "return _build(cache_key=key), _build(cache_key=key)"
        in simulation.rewritten_sources[module.file_path]
    )


def test_parameter_conveyor_rewrite_preserves_a_private_method_receiver() -> None:
    source = (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class _CacheKey:\n"
        "    left: object\n"
        "    right: object\n"
        "\n"
        "class _Builder:\n"
        "    def _build(self, left, right):\n"
        "        return left, right\n"
        "\n"
        "    def caller(self, left, right):\n"
        "        key = _CacheKey(left=left, right=right)\n"
        "        return self._build(left, right)\n"
    )
    module = _module("pkg.method", source)
    findings = ClosedParameterConveyorDetector().detect(
        [module],
        DetectorConfig(),
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,), findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    simulation = snapshot.simulate_rewrites(
        plan.document.source_rewrite_batch(snapshot)
    )
    rewritten_source = simulation.rewritten_sources[module.file_path]

    assert "def _build(self, *, cache_key: '_CacheKey'):" in rewritten_source
    assert "return self._build(cache_key=key)" in rewritten_source


def test_parameter_conveyor_rewrite_preserves_cross_module_identity() -> None:
    authority_module = _module(
        "pkg.types",
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class _CacheKey:\n"
        "    left: object\n"
        "    right: object\n",
    )
    worker_module = _module(
        "pkg.worker",
        "from pkg.types import _CacheKey\n"
        "\n"
        "def _build(left, right):\n"
        "    return left, right\n"
        "\n"
        "def caller(left, right):\n"
        "    key = _CacheKey(left=left, right=right)\n"
        "    return _build(left, right)\n",
    )
    modules = (authority_module, worker_module)
    findings = ClosedParameterConveyorDetector().detect(
        modules,
        DetectorConfig(),
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    simulation = snapshot.simulate_rewrites(
        plan.document.source_rewrite_batch(snapshot)
    )

    assert tuple(simulation.rewritten_sources) == (worker_module.file_path,)
    assert (
        "def _build(*, cache_key: '_CacheKey'):"
        in simulation.rewritten_sources[worker_module.file_path]
    )
    assert (
        snapshot.source_index.symbol_for_target(
            next(
                target
                for target in snapshot.source_index.ast_targets
                if target.target_id
                == plan.document.recipes[0].operations[0].target.target_id
            )
        )
        == "pkg.types._CacheKey"
    )


def test_parameter_conveyor_rewrite_avoids_existing_carrier_name_bindings() -> None:
    module = _module(
        "pkg.collision",
        _base_source(
            callee_body=(
                "    cache_key = normalize()\n    return left, right, cache_key\n"
            )
        ),
    )
    findings = ClosedParameterConveyorDetector().detect(
        [module],
        DetectorConfig(),
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,), findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    simulation = snapshot.simulate_rewrites(
        plan.document.source_rewrite_batch(snapshot)
    )
    rewritten_source = simulation.rewritten_sources[module.file_path]

    assert "def _build(*, cache_key_2: '_CacheKey'):" in rewritten_source
    assert "return cache_key_2.left, cache_key_2.right, cache_key" in rewritten_source
    assert "return _build(cache_key_2=key)" in rewritten_source


@pytest.mark.parametrize(
    "callee_source,rejection_fragment",
    (
        (
            "def _build(\n"
            "    left,  # first field\n"
            "    right,\n"
            "):\n"
            "    return left, right\n",
            "comments inside its signature",
        ),
        (
            "def _build(left, right):\n"
            "    marker = lambda: None\n"
            "    return left, right\n",
            "contains nested lexical scopes",
        ),
    ),
)
def test_parameter_conveyor_recipe_rejects_lossy_source_reconstruction(
    callee_source: str,
    rejection_fragment: str,
) -> None:
    module = _module(
        "pkg.lossy",
        _base_source().replace(
            "def _build(left, right):\n    return left, right\n",
            callee_source,
        ),
    )
    findings = ClosedParameterConveyorDetector().detect(
        [module],
        DetectorConfig(),
    )
    assert len(findings) == 1
    snapshot = CodemodSourceSnapshot.from_modules((module,), findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)

    assert plan.document.recipes == ()
    assert plan.records[0].status is (
        FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    )
    assert rejection_fragment in plan.records[0].reason


def test_multistep_forwarding_forms_one_maximal_component_not_per_edge() -> None:
    builder = _builder(
        _module(
            "pkg.chain",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _second(left, right):\n"
            "    return left, right\n"
            "\n"
            "def _first(left, right):\n"
            "    return _second(left, right)\n"
            "\n"
            "def caller(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _first(left, right)\n",
        )
    )

    components = builder.proven_components()

    assert len(components) == 1
    assert components[0].participant_symbols == (
        "pkg.chain._first",
        "pkg.chain._second",
    )
    assert len(components[0].root_edges) == 1
    assert len(components[0].forwarding_edges) == 1


def test_attractive_root_is_not_exposed_when_its_transitive_family_is_open() -> None:
    builder = _builder(
        _module(
            "pkg.open_tail",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _second(left, right):\n"
            "    return left, right\n"
            "\n"
            "def _first(left, right):\n"
            "    return _second(left, right)\n"
            "\n"
            "def caller(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _first(left, right)\n"
            "\n"
            "def unconverted(left, right):\n"
            "    return _second(transform(left), right)\n",
        )
    )

    components = builder.assessed_components()

    assert len(components) == 1
    assert components[0].participant_symbols == (
        "pkg.open_tail._first",
        "pkg.open_tail._second",
    )
    assert ClosedParameterConveyorAuthorityViolation.INCOMPLETE_CALL_FAMILY in (
        components[0].proof.violations
    )
    assert builder.proven_components() == ()


def test_open_disconnected_island_blocks_the_whole_nominal_authority() -> None:
    builder = _builder(
        _module(
            "pkg.disconnected_open",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _closed(left, right):\n"
            "    return left, right\n"
            "\n"
            "def _open(left, right):\n"
            "    return left, right\n"
            "\n"
            "def closed_root(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _closed(left, right)\n"
            "\n"
            "def open_root(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _open(left, right)\n"
            "\n"
            "def unconverted(left, right):\n"
            "    return _open(transform(left), right)\n",
        )
    )

    components = builder.assessed_components()

    assert len(components) == 1
    assert components[0].participant_symbols == (
        "pkg.disconnected_open._closed",
        "pkg.disconnected_open._open",
    )
    assert len(components[0].root_edges) == 2
    assert ClosedParameterConveyorAuthorityViolation.INCOMPLETE_CALL_FAMILY in (
        components[0].proof.violations
    )
    assert builder.proven_components() == ()


def test_closed_disconnected_islands_form_one_authority_wide_batch() -> None:
    builder = _builder(
        _module(
            "pkg.disconnected_closed",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _first(left, right):\n"
            "    return left, right\n"
            "\n"
            "def _second(left, right):\n"
            "    return left, right\n"
            "\n"
            "def first_root(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _first(left, right)\n"
            "\n"
            "def second_root(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _second(left, right)\n",
        )
    )

    components = builder.proven_components()

    assert len(components) == 1
    assert components[0].participant_symbols == (
        "pkg.disconnected_closed._first",
        "pkg.disconnected_closed._second",
    )
    assert len(components[0].root_edges) == 2


def test_partial_product_construction_never_becomes_a_component() -> None:
    builder = _builder(
        _module(
            "pkg.partial",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object = None\n"
            "\n"
            "def _build(left, right):\n"
            "    return left, right\n"
            "\n"
            "def caller(left, right):\n"
            "    key = _CacheKey(left=left)\n"
            "    return _build(left, right)\n",
        )
    )

    assert builder.assessed_components() == ()
    assert builder.proven_components() == ()


def test_unconverted_incoming_call_blocks_the_whole_component() -> None:
    builder = _builder(
        _module(
            "pkg.incoming",
            _base_source()
            + "\n"
            + "def other(left, right):\n"
            + "    return _build(transform(left), right)\n",
        )
    )

    component = builder.assessed_components()[0]

    assert component.proof.violations == (
        ClosedParameterConveyorAuthorityViolation.INCOMPLETE_CALL_FAMILY,
    )
    assert builder.proven_components() == ()


def test_callable_escape_blocks_the_whole_component() -> None:
    builder = _builder(
        _module(
            "pkg.escape",
            _base_source() + "\n" + "escaped = _build\n",
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.ESCAPING_CALLABLE_REFERENCE in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


@pytest.mark.parametrize("callee", ("_build.__call__", "_build.__call__.__call__"))
def test_callable_used_as_attribute_receiver_blocks_signature_collapse(
    callee: str,
) -> None:
    module = _module(
        "pkg.receiver_escape",
        _base_source()
        + f"\ndef invoke(left, right):\n    return {callee}(left, right)\n",
    )
    namespace = {"__name__": module.module_name}
    exec(compile(module.source, module.file_path, "exec", dont_inherit=True), namespace)
    assert namespace["invoke"](1, 2) == (1, 2)
    builder = _builder(module)
    assert builder.repository.callable_escapes_for("pkg.receiver_escape._build")
    assert ClosedParameterConveyorAuthorityViolation.ESCAPING_CALLABLE_REFERENCE in (
        builder.assessed_components()[0].proof.violations
    )
    assert builder.proven_components() == ()
    assert ClosedParameterConveyorDetector().detect([module], DetectorConfig()) == []


def test_possible_callable_escape_blocks_signature_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        _base_source() + "\ndef escaped(flag):\n"
        "    if flag:\n"
        "        from escape import _build\n"
        "    return _build\n"
    )
    runtime_module = ModuleType("escape")
    monkeypatch.setitem(sys.modules, "escape", runtime_module)
    exec(
        compile(source, "escape.py", "exec", dont_inherit=True), runtime_module.__dict__
    )
    assert runtime_module.escaped(True)(1, 2) == (1, 2)
    with pytest.raises(UnboundLocalError):
        runtime_module.escaped(False)

    module = _module("escape", source)
    builder = _builder(module)
    escapes = builder.repository.callable_escapes_for("escape._build")
    assert len(escapes) == 1
    assert escapes[0].target_resolution.declaration is None
    assert "escape._build" in escapes[0].target_resolution.possible_symbols
    assert ClosedParameterConveyorAuthorityViolation.ESCAPING_CALLABLE_REFERENCE in (
        builder.assessed_components()[0].proof.violations
    )
    assert builder.proven_components() == ()
    assert ClosedParameterConveyorDetector().detect([module], DetectorConfig()) == []


def test_public_participant_is_not_treated_as_a_closed_repository_boundary() -> None:
    builder = _builder(
        _module(
            "pkg.public",
            _base_source()
            .replace("def _build", "def build")
            .replace("return _build", "return build"),
        )
    )

    component = builder.assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.PUBLIC_OR_EXTERNAL_BOUNDARY_NOT_CLOSED
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_explicitly_exported_private_participant_keeps_boundary_open() -> None:
    builder = _builder(
        _module(
            "pkg.explicit_export",
            "__all__ = ('_build',)\n" + _base_source(),
        )
    )

    component = builder.assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.PUBLIC_OR_EXTERNAL_BOUNDARY_NOT_CLOSED
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_public_named_reexport_keeps_private_participant_boundary_open() -> None:
    builder = _builder(
        _module("pkg.impl", _base_source()),
        _module("pkg.api", "from pkg.impl import _build as build\n"),
    )

    component = builder.assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.PUBLIC_OR_EXTERNAL_BOUNDARY_NOT_CLOSED
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_transitive_public_reexport_keeps_private_participant_boundary_open() -> None:
    builder = _builder(
        _module("pkg.impl", _base_source()),
        _module(
            "pkg.facade",
            "__all__ = ()\nfrom pkg.impl import _build as internal_build\n",
        ),
        _module("pkg.api", "from pkg.facade import internal_build as build\n"),
    )

    component = builder.assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.PUBLIC_OR_EXTERNAL_BOUNDARY_NOT_CLOSED
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_dynamic_export_contract_keeps_private_participant_boundary_open() -> None:
    builder = _builder(
        _module(
            "pkg.dynamic_export",
            "__all__ = exported_names()\n" + _base_source(),
        )
    )

    component = builder.assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.PUBLIC_OR_EXTERNAL_BOUNDARY_NOT_CLOSED
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_private_method_on_public_class_keeps_boundary_open() -> None:
    builder = _builder(
        _module(
            "pkg.public_owner",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "class Builder:\n"
            "    def _build(self, left, right):\n"
            "        return left, right\n"
            "\n"
            "    def caller(self, left, right):\n"
            "        key = _CacheKey(left=left, right=right)\n"
            "        return self._build(left, right)\n",
        )
    )

    component = builder.assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.PUBLIC_OR_EXTERNAL_BOUNDARY_NOT_CLOSED
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_rebinding_between_construction_and_call_blocks_the_component() -> None:
    builder = _builder(
        _module(
            "pkg.rebound",
            _base_source().replace(
                "    return _build(left, right)\n",
                "    left = normalize(left)\n    return _build(left, right)\n",
            ),
        )
    )

    component = builder.assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.REBINDING_OR_MUTATION_BETWEEN_BINDING_AND_USE
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_branch_local_carrier_does_not_claim_dominance() -> None:
    builder = _builder(
        _module(
            "pkg.branch",
            _base_source().replace(
                "    key = _CacheKey(left=left, right=right)\n",
                "    if left:\n        key = _CacheKey(left=left, right=right)\n",
            ),
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.NON_DOMINATING_CARRIER_BINDING in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_duplicate_dominating_carriers_block_local_root_selection() -> None:
    builder = _builder(
        _module(
            "pkg.ambiguous_carrier",
            _base_source().replace(
                "    key = _CacheKey(left=left, right=right)\n",
                "    first = _CacheKey(left=left, right=right)\n"
                "    second = _CacheKey(left=left, right=right)\n",
            ),
        )
    )

    component = builder.assessed_components()[0]

    assert len(component.root_edges) == 2
    assert ClosedParameterConveyorAuthorityViolation.AMBIGUOUS_ROOT_CARRIER in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_mutated_or_unconsumed_field_parameters_block_the_component() -> None:
    mutated = _builder(
        _module(
            "pkg.mutated",
            _base_source(
                callee_body=("    left = normalize(left)\n    return left, right\n")
            ),
        )
    ).assessed_components()[0]
    unconsumed = _builder(
        _module(
            "pkg.unconsumed",
            _base_source(callee_body="    return left\n"),
        )
    ).assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.REBINDING_OR_MUTATION_BETWEEN_BINDING_AND_USE
        in mutated.proof.violations
    )
    assert ClosedParameterConveyorAuthorityViolation.INCOMPLETE_PRODUCT_CONSUMPTION in (
        unconsumed.proof.violations
    )


def test_dynamic_full_product_forwarding_blocks_the_component() -> None:
    builder = _builder(
        _module(
            "pkg.dynamic",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _build(left, right, callbacks):\n"
            "    return callbacks[0](left, right)\n"
            "\n"
            "def caller(left, right, callbacks):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _build(left, right, callbacks)\n",
        )
    )

    component = builder.assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.UNRESOLVED_COMPLETE_PRODUCT_CALL
        in (component.proof.violations)
    )
    assert builder.proven_components() == ()


def test_named_and_qualified_open_targets_block_complete_product_flow() -> None:
    source = (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class _CacheKey:\n"
        "    left: object\n"
        "    right: object\n"
        "\n"
        "def _build(left, right, callback, handler):\n"
        "    callback(left, right)\n"
        "    return handler.run(left, right)\n"
        "\n"
        "def caller(left, right, callback, handler):\n"
        "    key = _CacheKey(left=left, right=right)\n"
        "    return _build(left, right, callback, handler)\n"
    )
    component = _builder(_module("pkg.open_targets", source)).assessed_components()[0]

    assert len(component.proof.unresolved_complete_product_call_ids) == 2
    assert (
        ClosedParameterConveyorAuthorityViolation.UNRESOLVED_COMPLETE_PRODUCT_CALL
        in component.proof.violations
    )


def test_exact_alias_forwarding_and_renamed_parameters_join_the_whole_chain() -> None:
    builder = _builder(
        _module(
            "pkg.alias_chain",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _second(first_value, second_value):\n"
            "    return first_value, second_value\n"
            "\n"
            "def _first(left, right):\n"
            "    first_alias = left\n"
            "    second_alias = right\n"
            "    return _second(first_alias, second_alias)\n"
            "\n"
            "def caller(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _first(left, right)\n",
        )
    )

    components = builder.proven_components()

    assert len(components) == 1
    assert components[0].participant_symbols == (
        "pkg.alias_chain._first",
        "pkg.alias_chain._second",
    )
    assert tuple(
        (binding.field_name, binding.parameter_name)
        for binding in components[0].forwarding_edges[0].field_bindings
    ) == (("left", "first_value"), ("right", "second_value"))


def test_open_alias_forwarding_blocks_the_attractive_prefix() -> None:
    builder = _builder(
        _module(
            "pkg.open_alias",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _second(left, right):\n"
            "    return left, right\n"
            "\n"
            "def _first(left, right, flag):\n"
            "    if flag:\n"
            "        maybe_left = left\n"
            "    return _second(maybe_left, right)\n"
            "\n"
            "def caller(left, right, flag):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _first(left, right, flag)\n",
        )
    )

    component = builder.assessed_components()[0]

    assert component.participant_symbols == ("pkg.open_alias._first",)
    assert ClosedParameterConveyorAuthorityViolation.OPEN_VALUE_ALIAS_FORWARDING in (
        component.proof.violations
    )
    assert (
        ClosedParameterConveyorAuthorityViolation.UNRESOLVED_COMPLETE_PRODUCT_CALL
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_open_disconnected_root_blocks_the_authority_wide_batch() -> None:
    builder = _builder(
        _module(
            "pkg.open_root",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _closed(left, right):\n"
            "    return left, right\n"
            "\n"
            "def closed_root(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _closed(left, right)\n"
            "\n"
            "def open_root(left, right, flag, callback):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    if flag:\n"
            "        maybe_left = left\n"
            "    return callback(maybe_left, right)\n",
        )
    )

    component = builder.assessed_components()[0]

    assert component.participant_symbols == ("pkg.open_root._closed",)
    assert ClosedParameterConveyorAuthorityViolation.OPEN_VALUE_ALIAS_FORWARDING in (
        component.proof.violations
    )
    assert (
        ClosedParameterConveyorAuthorityViolation.UNRESOLVED_COMPLETE_PRODUCT_CALL
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_carrier_alias_mutation_blocks_root_substitution() -> None:
    builder = _builder(
        _module(
            "pkg.carrier_alias_mutation",
            _base_source().replace(
                "    return _build(left, right)\n",
                "    alias = key\n"
                "    alias.left = normalize(left)\n"
                "    return _build(left, right)\n",
            ),
        )
    )

    component = builder.assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.REBINDING_OR_MUTATION_BETWEEN_BINDING_AND_USE
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_local_signature_introspection_blocks_parameter_rewrite() -> None:
    builder = _builder(
        _module(
            "pkg.locals_observed",
            _base_source(
                callee_body=(
                    "    observed = locals()\n"
                    "    return observed['left'], observed['right']\n"
                )
            ),
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.SIGNATURE_SEMANTICS_HAZARD in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_incomplete_override_family_blocks_method_signature_change() -> None:
    builder = _builder(
        _module(
            "pkg.overrides",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "class Base:\n"
            "    def _build(self, left, right):\n"
            "        return left, right\n"
            "\n"
            "class Leaf(Base):\n"
            "    def _build(self, left, right):\n"
            "        return left, right\n"
            "\n"
            "    def caller(self, left, right):\n"
            "        key = _CacheKey(left=left, right=right)\n"
            "        return self._build(left, right)\n",
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.INCOMPLETE_METHOD_FAMILY in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_competing_product_authorities_block_both_local_trajectories() -> None:
    builder = _builder(
        _module(
            "pkg.competing",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class _FirstKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "@dataclass\n"
            "class _SecondKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _build(left, right):\n"
            "    return left, right\n"
            "\n"
            "def first(left, right):\n"
            "    key = _FirstKey(left=left, right=right)\n"
            "    return _build(left, right)\n"
            "\n"
            "def second(left, right):\n"
            "    key = _SecondKey(left=left, right=right)\n"
            "    return _build(left, right)\n",
        )
    )

    components = builder.assessed_components()

    assert len(components) == 2
    assert all(
        ClosedParameterConveyorAuthorityViolation.NO_UNIQUE_NOMINAL_AUTHORITY
        in component.proof.violations
        for component in components
    )
    assert all(
        ClosedParameterConveyorAuthorityViolation.CONFLICTING_CALL_MAPPING
        in component.proof.violations
        for component in components
    )
    assert builder.proven_components() == ()


def test_self_host_coherence_cache_key_does_not_cross_its_public_owner() -> None:
    module = parse_python_modules(
        Path("nominal_refactor_advisor/observation_graph.py"),
        use_parse_cache=True,
    )[0]
    builder = _builder(module)
    components = builder.assessed_components()

    component = next(
        component
        for component in components
        if component.authority.class_symbol.endswith("._CoherenceCohortCacheKey")
    )

    assert component.participant_symbols == (
        "nominal_refactor_advisor.observation_graph."
        "ObservationGraph._build_coherence_cohorts_for",
    )
    assert (
        component.proof.callable_component.open_boundary_symbols
        == component.participant_symbols
    )
    assert builder.proven_components() == ()
