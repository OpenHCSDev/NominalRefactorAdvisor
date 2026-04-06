from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

from nominal_refactor_advisor.ast_tools import (
    AccessorWrapperObservationFamily,
    AttributeProbeObservationFamily,
    BuilderCallShapeFamily,
    ClassMarkerObservationFamily,
    ConfigDispatchObservationFamily,
    DualAxisResolutionObservationFamily,
    DynamicMethodInjectionObservationFamily,
    ExportDictShapeFamily,
    FieldObservationSpec,
    FieldObservationFamily,
    InlineStringLiteralDispatchObservationFamily,
    InterfaceGenerationObservationFamily,
    LineageMappingObservationFamily,
    MethodShapeFamily,
    ProjectionHelperObservationFamily,
    RegistrationShapeSpec,
    RegistrationShapeFamily,
    RuntimeTypeGenerationObservationFamily,
    ScopedShapeWrapperFunctionFamily,
    ScopedShapeWrapperSpecFamily,
    SentinelTypeObservationFamily,
    StringLiteralDispatchObservationFamily,
    NumericLiteralDispatchObservationFamily,
    collect_family_items,
    collect_scoped_observations,
    parse_python_modules,
)
from nominal_refactor_advisor.cli import _format_markdown
from nominal_refactor_advisor.cli import _json_payload
from nominal_refactor_advisor.cli import analyze_path
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.models import DispatchCountMetrics
from nominal_refactor_advisor.observation_graph import (
    ObservationGraph,
    ObservationKind,
    StructuralExecutionLevel,
    build_observation_graph,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.planner import build_refactor_plans


def _write_module(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_detects_repeated_private_method_shape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def _build(self, item):
        prepared = self.normalize(item)
        checked = self.validate(prepared)
        return self.finish(checked)


class Beta:
    def _assemble(self, value):
        prepared = self.normalize(value)
        checked = self.validate(prepared)
        return self.finish(checked)
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 5 for finding in findings)
    assert any(finding.pattern_id == 5 and finding.scaffold for finding in findings)
    assert any(
        finding.pattern_id == 5 and finding.codemod_patch for finding in findings
    )


def test_detects_oversized_orchestration_hub(tmp_path: Path) -> None:
    branch_body = "\n".join(
        f"""
    if branch_{index}(request):
        value = phase_{index}(value)
    else:
        value = fallback_{index}(value)
    audit_{index}(value)
""".rstrip()
        for index in range(12)
    )
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "def orchestrate(request):\n"
            "    value = start(request)\n"
            f"{branch_body}\n"
            "    finalized = finalize(value)\n"
            "    publish(finalized)\n"
            "    return finalized\n"
        ),
    )

    findings = analyze_path(
        tmp_path,
        DetectorConfig(
            min_orchestration_function_lines=40,
            min_orchestration_branches=10,
            min_orchestration_calls=24,
        ),
    )

    assert any(
        finding.pattern_id == PatternId.STAGED_ORCHESTRATION for finding in findings
    )


def test_detects_repeated_threaded_parameter_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def score_exact(
    request,
    scoring_context,
    electrostatics,
    receptor_coords,
    receptor_radii,
    quaternion,
    translation,
    candidate_coords,
):
    posed = rigid(candidate_coords, quaternion, translation)
    audited = audit_pose(posed, receptor_coords)
    return compute_exact(
        request,
        scoring_context,
        electrostatics,
        receptor_coords,
        receptor_radii,
        audited,
    )


def score_softened(
    request,
    scoring_context,
    electrostatics,
    receptor_coords,
    receptor_radii,
    quaternion,
    translation,
    candidate_coords,
):
    posed = rigid(candidate_coords, quaternion, translation)
    audited = audit_pose(posed, receptor_coords)
    return compute_softened(
        request,
        scoring_context,
        electrostatics,
        receptor_coords,
        receptor_radii,
        audited,
    )


def certify_pose(
    request,
    scoring_context,
    electrostatics,
    receptor_coords,
    receptor_radii,
    quaternion,
    translation,
    pose_index,
):
    posed = derive_pose(pose_index, quaternion, translation)
    audited = audit_pose(posed, receptor_coords)
    return certify(
        request,
        scoring_context,
        electrostatics,
        receptor_coords,
        receptor_radii,
        audited,
    )
""",
    )

    findings = analyze_path(
        tmp_path,
        DetectorConfig(
            min_shared_parameters=5,
            min_parameter_family_function_lines=8,
        ),
    )

    assert any(
        finding.pattern_id == PatternId.AUTHORITATIVE_CONTEXT for finding in findings
    )


def test_detects_enum_strategy_dispatch_with_abc_guidance(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from enum import Enum


class Mode(Enum):
    OBSERVED = "observed"
    CERTIFIED = "certified"


def run_mode(mode, inputs, steps):
    if mode == Mode.OBSERVED:
        return run_observed(inputs, steps)
    elif mode == Mode.CERTIFIED:
        return run_certified(inputs, steps)
    else:
        raise ValueError(mode)
""",
    )

    findings = analyze_path(tmp_path)

    strategy_finding = next(
        finding
        for finding in findings
        if finding.pattern_id == PatternId.NOMINAL_STRATEGY_FAMILY
    )
    assert "Mode.OBSERVED" in strategy_finding.summary
    assert strategy_finding.scaffold is not None
    assert "class ModeRunner(ABC)" in strategy_finding.scaffold
    assert strategy_finding.codemod_patch is not None
    assert "runner = ModeRunner.for_mode(mode)" in strategy_finding.codemod_patch


def test_detects_manual_fiber_tag_with_abc_fix(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Notification:
    def __init__(self, kind, recipient, subject=None, body=None, phone=None, device_token=None):
        self.kind = kind
        self.recipient = recipient
        self.subject = subject
        self.body = body
        self.phone = phone
        self.device_token = device_token

    def send(self):
        if self.kind == "email":
            return smtp_send(self.recipient, self.subject, self.body)
        elif self.kind == "sms":
            return twilio_send(self.phone, self.body)
        elif self.kind == "push":
            return apns_send(self.device_token, self.body)
        raise ValueError(self.kind)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(item for item in findings if item.detector_id == "manual_fiber_tag")
    assert "self.kind" in finding.summary
    assert finding.scaffold is not None
    assert "class Notification(ABC)" in finding.scaffold


def test_detects_descriptor_derived_view_drift(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Model:
    def __init__(self, table_name):
        self.table_name = table_name
        self.select_query = f"SELECT * FROM {self.table_name}"
        self.insert_query = f"INSERT INTO {self.table_name}"
        self.count_query = f"SELECT COUNT(*) FROM {self.table_name}"

    def rename_table(self, new_name):
        self.table_name = new_name
        self.select_query = f"SELECT * FROM {self.table_name}"
        self.insert_query = f"INSERT INTO {self.table_name}"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item for item in findings if item.detector_id == "descriptor_derived_view"
    )
    assert "count_query" in finding.summary
    assert finding.scaffold is not None
    assert "class DerivedField" in finding.scaffold


def test_detects_deferred_class_registration(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
HANDLERS = {}


def register_handler(event_type):
    def decorator(cls):
        HANDLERS[event_type] = cls
        return cls
    return decorator


@register_handler("user.created")
class UserCreatedHandler:
    def handle(self, event):
        return event


@register_handler("order.placed")
class OrderPlacedHandler:
    def handle(self, event):
        return event


class PaymentFailedHandler:
    def handle(self, event):
        return event
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item for item in findings if item.detector_id == "deferred_class_registration"
    )
    assert "HANDLERS" in finding.summary
    assert finding.scaffold is not None
    assert "__init_subclass__" in finding.scaffold


def test_detects_structural_confusability_without_abc_witness(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def process_batch(items, backend):
    for item in items:
        backend.store(item)
    backend.flush()


class DatabaseBackend:
    def store(self, item):
        return item

    def flush(self):
        return None


class CacheBackend:
    def store(self, item):
        return item

    def flush(self):
        return None

    def invalidate(self):
        return None
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item for item in findings if item.detector_id == "structural_confusability"
    )
    assert "process_batch" in finding.summary
    assert finding.scaffold is not None
    assert "class BackendInterface(ABC)" in finding.scaffold


def test_detects_semantic_witness_family_with_abc_base(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class ManualFiberTagCandidate:
    file_path: str
    class_name: str
    init_line: int
    method_name: str
    method_line: int
    tag_name: str
    case_names: tuple[str, ...]
    assigned_field_names: tuple[str, ...]


@dataclass(frozen=True)
class DescriptorDerivedViewCandidate:
    file_path: str
    class_name: str
    source_attr: str
    init_line: int
    mutator_name: str
    mutator_line: int
    derived_field_names: tuple[str, ...]
    updated_field_names: tuple[str, ...]


@dataclass(frozen=True)
class ManualRegistryCandidate:
    file_path: str
    registry_name: str
    decorator_name: str
    class_names: tuple[str, ...]
    unregistered_class_names: tuple[str, ...]
    line: int


@dataclass(frozen=True)
class StructuralConfusabilityCandidate:
    file_path: str
    function_name: str
    line: int
    parameter_name: str
    observed_method_names: tuple[str, ...]
    class_names: tuple[str, ...]
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item for item in findings if item.detector_id == "semantic_witness_family"
    )
    assert "ManualFiberTagCandidate" in finding.summary
    assert finding.scaffold is not None
    assert "class WitnessCarrier(ABC)" in finding.scaffold


def test_detects_mixin_enforcement_for_renamed_semantic_roles(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class ManualFiberTagCandidate:
    file_path: str
    class_name: str
    init_line: int
    method_name: str
    method_line: int
    tag_name: str
    case_names: tuple[str, ...]


@dataclass(frozen=True)
class ManualRegistryCandidate:
    file_path: str
    registry_name: str
    decorator_name: str
    class_names: tuple[str, ...]
    line: int


@dataclass(frozen=True)
class StructuralConfusabilityCandidate:
    file_path: str
    function_name: str
    line: int
    observed_method_names: tuple[str, ...]
    class_names: tuple[str, ...]
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item
        for item in findings
        if item.detector_id == "mixin_enforcement"
        and "class_name" in item.summary
        and "class_names" in item.summary
    )
    assert finding.scaffold is not None
    assert "class NameBearingMixin(ABC)" in finding.scaffold
    assert "(WitnessCarrier, NameBearingMixin" in finding.scaffold
    assert finding.codemod_patch is not None
    assert "multiple inheritance" in finding.codemod_patch


def test_detects_sentinel_attribute_simulation(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    sigma = "alpha"


class Beta:
    sigma = "beta"


def choose(obj):
    if obj.sigma == "alpha":
        return 1
    return 2
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 1 for finding in findings)


def test_detects_predicate_factory_chain(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def build(param_type):
    if is_optional(param_type):
        return OptionalInfo()
    elif is_dataclass(param_type):
        return DataclassInfo()
    return GenericInfo()
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 2 for finding in findings)


def test_detects_config_attribute_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(config):
    if hasattr(config, "napari_port"):
        return config.napari_port
    if getattr(config, "viewer_type", None) == "fiji":
        return 2
    return 0
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 4 for finding in findings)


def test_collects_config_dispatch_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(config):
    if hasattr(config, "napari_port"):
        return config.napari_port
    if getattr(config, "viewer_type", None) == "fiji":
        return 2
    return 0
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, ConfigDispatchObservationFamily)

    assert {item.observed_attribute for item in observations} == {
        "napari_port",
        "viewer_type",
    }


def test_ignores_single_generic_name_sentinel_branch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    name = "alpha"


class Beta:
    name = "beta"


def choose(obj):
    if obj.name == "alpha":
        return 1
    return 2
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(finding.pattern_id == 1 for finding in findings)


def test_detects_generated_type_lineage(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
BASE_TO_LAZY = {}


class Base:
    pass


LazyBase = type("LazyBase", (Base,), {})
BASE_TO_LAZY[Base] = LazyBase
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 7 for finding in findings)


def test_collects_generated_type_lineage_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
BASE_TO_LAZY = {}


class Base:
    pass


LazyBase = type("LazyBase", (Base,), {})
BASE_TO_LAZY[Base] = LazyBase
""",
    )

    module = parse_python_modules(tmp_path)[0]
    generation = collect_family_items(module, RuntimeTypeGenerationObservationFamily)
    lineage = collect_family_items(module, LineageMappingObservationFamily)

    assert [item.generator_name for item in generation] == ["type"]
    assert [item.mapping_name for item in lineage] == ["BASE_TO_LAZY"]


def test_detects_dual_axis_resolution(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(scope_stack, obj):
    for scope in scope_stack:
        for mro_type in type(obj).__mro__:
            if scope and mro_type:
                return scope, mro_type
    return None
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 8 for finding in findings)


def test_collects_dual_axis_resolution_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(scope_stack, obj):
    for scope in scope_stack:
        for mro_type in type(obj).__mro__:
            if scope and mro_type:
                return scope, mro_type
    return None
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, DualAxisResolutionObservationFamily)

    assert len(observations) == 1
    assert observations[0].outer_axis_name == "scope"
    assert observations[0].inner_axis_name == "mro_type"


def test_detects_manual_virtual_membership(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def check(instance):
    if hasattr(instance.__class__, "_is_global_config"):
        return instance.__class__._is_global_config
    return False
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 9 for finding in findings)


def test_collects_class_marker_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def check(instance):
    if hasattr(instance.__class__, "_is_global_config"):
        return instance.__class__._is_global_config
    return False
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, ClassMarkerObservationFamily)

    assert any(item.marker_name == "_is_global_config" for item in observations)


def test_detects_dynamic_interface_generation(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


def make_interface(name):
    return type(name, (ABC,), {})
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 10 for finding in findings)


def test_collects_interface_generation_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


def make_interface(name):
    return type(name, (ABC,), {})
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, InterfaceGenerationObservationFamily)

    assert [item.generator_name for item in observations] == ["type"]


def test_detects_sentinel_type_marker(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
SENTINEL = type("Sentinel", (), {})()


def present(registry):
    return SENTINEL in registry
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 11 for finding in findings)


def test_collects_sentinel_type_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
SENTINEL = type("Sentinel", (), {})()


def present(registry):
    return SENTINEL in registry
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, SentinelTypeObservationFamily)

    assert any(item.sentinel_name == "SENTINEL" for item in observations)
    assert len(observations) >= 2


def test_detects_dynamic_method_injection(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def inject(target_type, method_name, method_impl):
    setattr(target_type, method_name, method_impl)
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 12 for finding in findings)


def test_collects_dynamic_method_injection_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def inject(target_type, method_name, method_impl):
    setattr(target_type, method_name, method_impl)
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, DynamicMethodInjectionObservationFamily)

    assert [item.mutator_name for item in observations] == ["setattr"]


def test_markdown_output_includes_prescription_details(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def build(param_type):
    if is_optional(param_type):
        return OptionalInfo()
    elif is_dataclass(param_type):
        return DataclassInfo()
    return GenericInfo()
""",
    )

    findings = analyze_path(tmp_path)
    output = _format_markdown(findings)
    assert "Prescription:" in output
    assert "Canonical shape:" in output
    assert "First move:" in output
    assert "Example skeleton:" in output


def test_markdown_output_handles_multiple_example_skeletons(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def _prepare(self, item):
        ready = self.normalize(item)
        checked = self.validate(ready)
        return self.finish(checked)


class Beta:
    def _build(self, value):
        ready = self.normalize(value)
        checked = self.validate(ready)
        return self.finish(checked)
""",
    )

    findings = analyze_path(tmp_path)
    output = _format_markdown(findings)
    assert output.count("Example skeleton:") >= 2
    assert "Suggested scaffold:" in output
    assert "Suggested patch:" in output


def test_clusters_redundant_methods_into_abc_candidate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def _prepare(self, item):
        ready = self.normalize(item)
        checked = self.validate(ready)
        return self.finish(checked)

    def _score(self, item):
        scored = self.compute(item)
        bounded = self.bound(scored)
        packaged = self.package(bounded)
        return self.finish(packaged)


class Beta:
    def _build(self, value):
        ready = self.normalize(value)
        checked = self.validate(ready)
        return self.finish(checked)

    def _evaluate(self, value):
        scored = self.compute(value)
        bounded = self.bound(scored)
        packaged = self.package(bounded)
        return self.finish(packaged)
""",
    )

    findings = analyze_path(tmp_path)
    assert any(
        finding.detector_id == "inheritance_hierarchy_candidate" for finding in findings
    )


def test_observation_graph_recovers_method_coherence_cohort(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def _prepare(self, item):
        ready = self.normalize(item)
        checked = self.validate(ready)
        return self.finish(checked)

    def _score(self, item):
        scored = self.compute(item)
        bounded = self.bound(scored)
        packaged = self.package(bounded)
        return self.finish(packaged)


class Beta:
    def _build(self, value):
        ready = self.normalize(value)
        checked = self.validate(ready)
        return self.finish(checked)

    def _evaluate(self, value):
        scored = self.compute(value)
        bounded = self.bound(scored)
        packaged = self.package(bounded)
        return self.finish(packaged)


class Gamma:
    def _render(self, payload):
        ready = self.normalize(payload)
        checked = self.validate(ready)
        return self.finish(checked)
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, MethodShapeFamily)
    graph = ObservationGraph(
        tuple(item.structural_observation for item in observations)
    )

    cohorts = graph.coherence_cohorts_for(
        ObservationKind.METHOD_SHAPE,
        StructuralExecutionLevel.FUNCTION_BODY,
        minimum_witnesses=2,
        minimum_fibers=2,
    )

    cohort = next(
        item for item in cohorts if item.nominal_witnesses == ("Alpha", "Beta")
    )
    assert len(cohort.fibers) == 2


def test_detects_attribute_probe_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(widget):
    if hasattr(widget, \"isChecked\"):
        return widget.isChecked()
    return getattr(widget, \"value\", None)
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.detector_id == "attribute_probes" for finding in findings)


def test_collects_attribute_probe_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(widget):
    if hasattr(widget, "checked"):
        return widget.checked
    try:
        return getattr(widget, "value", None)
    except AttributeError:
        return None
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, AttributeProbeObservationFamily)

    assert {item.probe_kind for item in observations} == {
        "hasattr",
        "getattr",
        "attribute_error",
    }
    assert any(item.observed_attribute == "checked" for item in observations)


def test_collects_literal_dispatch_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def convert(kind, value):
    if kind == "numpy":
        return value
    elif kind == "cupy":
        return value
    return value


def walk(node):
    if node.kind == "alpha":
        return 1
    if node.kind == "beta":
        return 2
    return 0
""",
    )

    module = parse_python_modules(tmp_path)[0]
    chains = collect_family_items(module, StringLiteralDispatchObservationFamily)
    inline_groups = collect_family_items(
        module, InlineStringLiteralDispatchObservationFamily
    )

    assert any(item.axis_expression == "kind" for item in chains)
    assert any(item.axis_expression == "node.kind" for item in inline_groups)


def test_detects_string_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def convert(kind, value):
    if kind == \"numpy\":
        return value
    elif kind == \"cupy\":
        return value
    elif kind == \"torch\":
        return value
    return value
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "string_dispatch"
    )
    assert finding.pattern_id == 3
    assert "`kind`" in finding.summary
    assert "'numpy'" in finding.summary
    assert finding.certification == "certified"


def test_detects_inline_literal_dispatch_registry_smell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def walk(node):
    if node.kind == "alpha":
        return 1
    if node.kind == "beta":
        return 2
    return 0
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.detector_id == "inline_literal_dispatch" for finding in findings)


def test_detects_bidirectional_registry(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Registry:
    def __init__(self):
        self._forward = {}
        self._reverse = {}

    def register(self, left, right):
        self._forward[left] = right
        self._reverse[right] = left
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 13 for finding in findings)


def test_detects_repeated_builder_call_shape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def build(self, candidate):
        return RuntimePlan(
            pose_id=candidate.pose_id,
            score=candidate.score,
            theorem_handles=tuple(candidate.theorem_handles),
        )


class Beta:
    def build(self, entry):
        return RuntimePlan(
            pose_id=entry.pose_id,
            score=entry.score,
            theorem_handles=tuple(entry.theorem_handles),
        )
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 14 for finding in findings)
    assert any(finding.pattern_id == 14 and finding.scaffold for finding in findings)
    assert any(
        finding.pattern_id == 14 and finding.codemod_patch for finding in findings
    )


def test_detects_manual_class_registration(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
REGISTRY = {}


class AlphaHandler:
    pass


class BetaHandler:
    pass


REGISTRY["alpha"] = AlphaHandler
REGISTRY["beta"] = BetaHandler
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 6 for finding in findings)
    assert any(finding.pattern_id == 6 and finding.scaffold for finding in findings)
    assert any(
        finding.pattern_id == 6 and finding.codemod_patch for finding in findings
    )


def test_detects_helper_registration_call(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Registry:
    def register(self, cls, key):
        return cls


registry = Registry()


class Alpha:
    pass


class Beta:
    pass


registry.register(Alpha, "alpha")
registry.register(Beta, "beta")
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 6 for finding in findings)


def test_detects_decorator_registration(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def register(registry, key):
    def deco(cls):
        return cls
    return deco


REGISTRY = {}


@register(REGISTRY, "alpha")
class Alpha:
    pass


@register(REGISTRY, "beta")
class Beta:
    pass
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 6 for finding in findings)


def test_detects_auto_register_decorator_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def auto_register(registry, key):
    def deco(cls):
        return cls
    return deco


REGISTRY = {}


@auto_register(REGISTRY, "alpha")
class Alpha:
    pass


@auto_register(REGISTRY, "beta")
class Beta:
    pass
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 6 for finding in findings)


def test_collects_scoped_call_observations(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def build(self, result):
        return transform(result)
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_scoped_observations(module, (ast.Call,))
    call_observation = next(
        item
        for item in observations
        if isinstance(item.node, ast.Call)
        and getattr(item.node.func, "id", None) == "transform"
    )

    assert call_observation.class_name == "Alpha"
    assert call_observation.function_name == "build"


def test_spec_families_use_autoregistration() -> None:
    registration_specs = {
        type(spec).__name__ for spec in RegistrationShapeSpec.registered_specs()
    }
    field_specs = {
        type(spec).__name__ for spec in FieldObservationSpec.registered_specs()
    }

    assert registration_specs == {
        "AssignmentRegistrationShapeSpec",
        "CallRegistrationShapeSpec",
        "DecoratorRegistrationShapeSpec",
    }
    assert field_specs == {
        "DataclassBodyFieldObservationSpec",
        "InitAssignmentFieldObservationSpec",
    }


def test_detects_parallel_scoped_shape_wrappers(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass
import ast


@dataclass(frozen=True)
class ScopedShapeSpec:
    node_types: tuple[type[ast.AST], ...]
    build_shape: object


def _build_method_shape_from_observation(parsed_module, observation):
    node = observation.node
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return (parsed_module, node, observation.class_name)


def _build_builder_call_shape_from_observation(parsed_module, observation):
    node = observation.node
    if not isinstance(node, ast.Call):
        return None
    return (parsed_module, node, observation.function_name)


_METHOD_SHAPE_SPEC = ScopedShapeSpec(
    node_types=(ast.FunctionDef, ast.AsyncFunctionDef),
    build_shape=_build_method_shape_from_observation,
)


_BUILDER_CALL_SHAPE_SPEC = ScopedShapeSpec(
    node_types=(ast.Call,),
    build_shape=_build_builder_call_shape_from_observation,
)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "scoped_shape_wrapper"
    )

    assert "polymorphic spec family" in finding.title
    assert "ScopedShapeSpec" in (finding.scaffold or "")


def test_detects_manual_indexed_family_expansion(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class FieldObservationSpec: ...
class FieldObservation: ...
class ConfigDispatchObservationSpec: ...
class ConfigDispatchObservation: ...


def collect_field_observations(parsed_module):
    return [
        item
        for item in _collect_items_from_spec_root(
            FieldObservationSpec, parsed_module, FieldObservation
        )
        if isinstance(item, FieldObservation)
    ]


def collect_config_dispatch_observations(parsed_module):
    return [
        item
        for item in _collect_items_from_spec_root(
            ConfigDispatchObservationSpec, parsed_module, ConfigDispatchObservation
        )
        if isinstance(item, ConfigDispatchObservation)
    ]
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.detector_id == "manual_indexed_family" for finding in findings)


def test_collects_scoped_shape_wrapper_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
import ast


def _build_method_shape_from_observation(parsed_module, observation):
    node = observation.node
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return (parsed_module, node)


_METHOD_SHAPE_SPEC = ScopedShapeSpec(
    node_types=(ast.FunctionDef, ast.AsyncFunctionDef),
    build_shape=_build_method_shape_from_observation,
)
""",
    )

    module = parse_python_modules(tmp_path)[0]
    functions = collect_family_items(module, ScopedShapeWrapperFunctionFamily)
    specs = collect_family_items(module, ScopedShapeWrapperSpecFamily)

    assert [item.function_name for item in functions] == [
        "_build_method_shape_from_observation"
    ]
    assert [item.spec_name for item in specs] == ["_METHOD_SHAPE_SPEC"]


def test_detects_namespaced_auto_register_decorator_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Plugins:
    def auto_register(self, registry, key):
        def deco(cls):
            return cls
        return deco


plugins = Plugins()
REGISTRY = {}


@plugins.auto_register(REGISTRY, "alpha")
class Alpha:
    pass


@plugins.auto_register(REGISTRY, "beta")
class Beta:
    pass
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 6 for finding in findings)


def test_collects_registration_shapes_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Plugins:
    def auto_register(self, registry, key):
        def deco(cls):
            return cls
        return deco


plugins = Plugins()
REGISTRY = {}


@plugins.auto_register(REGISTRY, "alpha")
class Alpha:
    pass


REGISTRY["beta"] = Alpha
""",
    )

    module = parse_python_modules(tmp_path)[0]
    shapes = collect_family_items(module, RegistrationShapeFamily)

    assert {shape.registration_style for shape in shapes} == {
        "decorator_registration",
        "subscript_assignment",
    }


def test_detects_repeated_export_dict_shape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def export(self, result):
        return {
            "pose_id": result.pose_id,
            "score": result.score,
            "label": result.label,
        }


class Beta:
    def export(self, item):
        return {
            "pose_id": item.pose_id,
            "score": item.score,
            "label": item.label,
        }
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.detector_id == "repeated_export_dicts" for finding in findings)
    assert any("projection dict" in finding.title.lower() for finding in findings)
    assert any(
        finding.detector_id == "repeated_export_dicts" and finding.scaffold
        for finding in findings
    )
    assert any(
        finding.detector_id == "repeated_export_dicts" and finding.codemod_patch
        for finding in findings
    )


def test_collects_projection_helper_shapes_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def labels(items):
    return tuple(sorted(item.label for item in items))


def scores(items):
    return tuple(sorted(item.score for item in items))
""",
    )

    module = parse_python_modules(tmp_path)[0]
    shapes = collect_family_items(module, ProjectionHelperObservationFamily)

    assert {shape.projected_attribute for shape in shapes} == {"label", "score"}


def test_collects_accessor_wrapper_candidates_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Sample:
    def current(self):
        return self._current

    def update(self, current):
        self._current = current
""",
    )

    module = parse_python_modules(tmp_path)[0]
    candidates = collect_family_items(module, AccessorWrapperObservationFamily)

    assert {candidate.accessor_kind for candidate in candidates} == {"getter", "setter"}


def test_collects_field_observation_fibers_for_dataclass_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    score: float
    label: str


@dataclass
class BetaResult:
    pose_id: int
    score: float
    label: str
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)
    graph = ObservationGraph(
        tuple(item.structural_observation for item in observations)
    )
    fibers = graph.fibers_for(
        ObservationKind.FIELD, StructuralExecutionLevel.CLASS_BODY
    )

    pose_fiber = next(fiber for fiber in fibers if fiber.observed_name == "pose_id")
    assert len(pose_fiber.observations) == 2


def test_ignores_classvar_fields_via_generic_annotation_matcher(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class AlphaResult:
    pose_id: int
    cache: ClassVar[dict[str, int]] = {}


@dataclass
class BetaResult:
    pose_id: int
    cache: ClassVar[dict[str, int]] = {}
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)

    assert all(item.field_name != "cache" for item in observations)


def test_observation_graph_recovers_field_coherence_cohort(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    score: float
    label: str
    rank: int
    alpha_only: int


@dataclass
class BetaResult:
    pose_id: int
    score: float
    label: str
    rank: int
    beta_only: int


@dataclass
class GammaResult:
    pose_id: int
    score: float
    gamma_only: int
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)
    graph = ObservationGraph(
        tuple(item.structural_observation for item in observations)
    )

    cohorts = graph.coherence_cohorts_for(
        ObservationKind.FIELD,
        StructuralExecutionLevel.CLASS_BODY,
        minimum_witnesses=2,
        minimum_fibers=2,
    )

    cohort = next(
        item
        for item in cohorts
        if item.nominal_witnesses == ("AlphaResult", "BetaResult")
    )
    assert set(cohort.observed_names) == {"pose_id", "score", "label", "rank"}


def test_ignores_namespaced_classvar_fields_via_family_matcher(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
import typing
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    cache: typing.ClassVar[dict[str, int]] = {}
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)

    assert all(item.field_name != "cache" for item in observations)


def test_collects_namespaced_dataclass_fields_via_name_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
import dataclasses as dc


@dc.dataclass
class AlphaResult:
    pose_id: int
    score: float
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)

    assert {item.field_name for item in observations} == {"pose_id", "score"}


def test_detects_repeated_field_family_in_dataclasses(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    score: float
    label: str
    alpha_only: int


@dataclass
class BetaResult:
    pose_id: int
    score: float
    label: str
    beta_only: int
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_field_family"
    )

    assert finding.pattern_id == 5
    assert "pose_id" in finding.summary
    assert "ResultBase" in (finding.scaffold or "")
    assert "pose_id: int" in (finding.scaffold or "")


def test_does_not_merge_dataclass_fields_with_conflicting_types(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    score: float
    alpha_only: int


@dataclass
class BetaResult:
    pose_id: str
    score: float
    beta_only: int
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "repeated_field_family" for finding in findings
    )


def test_plan_extracts_shared_fields_to_abc_base(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class AlphaController:
    def __init__(self, pose_id, score, label, alpha_only):
        self.pose_id = pose_id
        self.score = score
        self.label = label
        self.alpha_only = alpha_only


class BetaController:
    def __init__(self, pose_id, score, label, beta_only):
        self.pose_id = pose_id
        self.score = score
        self.label = label
        self.beta_only = beta_only
""",
    )

    findings = analyze_path(tmp_path)
    plans = build_refactor_plans(findings, tmp_path)
    plan = next(plan for plan in plans if plan.primary_pattern_id == 5)

    assert any(action.kind == "extract_shared_fields" for action in plan.actions)
    field_action = next(
        action for action in plan.actions if action.kind == "extract_shared_fields"
    )
    assert field_action.statement_operation == "move"
    assert "pose_id" in field_action.description


def test_json_payload_exposes_observation_graph(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    score: float


def convert(kind, value):
    if kind == "numpy":
        return value
    elif kind == "cupy":
        return value
    return value
""",
    )

    modules = parse_python_modules(tmp_path)
    findings = analyze_path(tmp_path)
    payload = _json_payload(findings, [], modules)
    observations = cast(list[dict[str, object]], payload["observations"])
    fibers = cast(list[dict[str, object]], payload["fibers"])

    assert "observations" in payload
    assert "fibers" in payload
    assert any(item["observation_kind"] == "field" for item in observations)
    assert any(item["observation_kind"] == "literal_dispatch" for item in fibers)


def test_observation_graph_auto_includes_registered_observation_families(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
BASE_TO_LAZY = {}
SENTINEL = type("Sentinel", (), {})()


class Base:
    pass


LazyBase = type("LazyBase", (Base,), {})
BASE_TO_LAZY[Base] = LazyBase


def resolve(config, obj):
    if hasattr(config, "kind"):
        return config.kind
    for scope in [1]:
        for mro_type in type(obj).__mro__:
            if scope and mro_type:
                return scope, mro_type
    return SENTINEL
""",
    )

    graph = build_observation_graph(parse_python_modules(tmp_path))
    kinds = {item.observation_kind for item in graph.observations}

    assert ObservationKind.CONFIG_DISPATCH in kinds
    assert ObservationKind.RUNTIME_TYPE_GENERATION in kinds
    assert ObservationKind.LINEAGE_MAPPING in kinds
    assert ObservationKind.DUAL_AXIS_RESOLUTION in kinds
    assert ObservationKind.SENTINEL_TYPE in kinds


def test_ignores_constant_string_maps_for_pattern_three(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
LOOKUP = {
    "alpha": 1,
    "beta": 2,
    "gamma": 3,
}
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(finding.detector_id == "string_dispatch" for finding in findings)


def test_detects_module_level_dispatch_dict_with_callable_targets(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def alpha():
    return 1


def beta():
    return 2


def gamma():
    return 3


DISPATCH = {
    "alpha": alpha,
    "beta": beta,
    "gamma": gamma,
}
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.detector_id == "string_dispatch" for finding in findings)


def test_ignores_non_branch_config_reads(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(config):
    port = config.napari_port
    return port
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(finding.pattern_id == 4 for finding in findings)


def test_detects_numeric_literal_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def render(pattern_id):
    if pattern_id == 3:
        return "dispatch"
    elif pattern_id == 5:
        return "abc"
    elif pattern_id == 14:
        return "schema"
    return "other"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "numeric_literal_dispatch"
    )
    assert "`pattern_id`" in finding.summary
    assert "3" in finding.summary
    assert finding.certification == "certified"


def test_detects_repeated_hardcoded_semantic_string(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
DEFAULT_CERTIFICATION = "strong_heuristic"


def first():
    return configure(certification="strong_heuristic")


def second():
    return configure(certification="strong_heuristic")
""",
    )

    findings = analyze_path(tmp_path)
    assert any(
        finding.detector_id == "repeated_hardcoded_strings" for finding in findings
    )


def test_detects_repeated_projection_helper_wrappers(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def dedupe(items):
    return items


def capability_labels(capabilities):
    return tuple(dedupe(tag.label for tag in capabilities))


def capability_distinctions(capabilities):
    return tuple(dedupe(tag.distinction for tag in capabilities))


def observation_labels(observations):
    return tuple(dedupe(tag.label for tag in observations))
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_projection_helpers"
    )

    assert "_render_projection" in (finding.scaffold or "")


def test_detects_accessor_wrapper_smell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Sample:
    def get_status(self):
        return self.status

    def set_status(self, status):
        self.status = status
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "accessor_wrapper"
    )

    assert "structural accessor wrapper" in finding.title
    assert "replace `Sample.get_status()` with `status`" in (finding.scaffold or "")


def test_detects_structural_accessor_wrappers_without_naming_convention(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Sample:
    def status(self):
        return self._status

    def store(self, status):
        self._status = status
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "accessor_wrapper"
    )

    assert "structural accessor wrapper" in finding.summary
    assert "read through" in finding.relation_context
    assert "replace `Sample.status()` with `status`" in (finding.scaffold or "")


def test_detects_single_structural_computed_property_candidate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Sample:
    def labels(self):
        return tuple(self._labels)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "accessor_wrapper"
    )

    assert "computed tuple" in finding.relation_context
    assert "an `@property` exposing `tuple(self._labels)`" in (finding.scaffold or "")


def test_uses_nominal_metric_dataclasses(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def render(pattern_id):
    if pattern_id == 3:
        return "dispatch"
    elif pattern_id == 5:
        return "abc"
    elif pattern_id == 14:
        return "schema"
    return "other"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "numeric_literal_dispatch"
    )

    assert isinstance(finding.metrics, DispatchCountMetrics)
    assert finding.metrics.dispatch_site_count == 3
    assert finding.metrics.dispatch_axis == "pattern_id"


def test_detects_semantic_metrics_dict_bag_and_recommends_nominal_class(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class RefactorFinding:
    metrics: object


def build():
    return RefactorFinding(metrics={"dispatch_site_count": len([1, 2, 3])})
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "semantic_dict_bag"
    )

    assert "DispatchCountMetrics" in (finding.scaffold or "")
    assert "CountedDispatchMetrics" in (finding.scaffold or "")


def test_detects_local_impact_dict_bag_and_recommends_impact_delta(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def estimate():
    impact = {
        "lower_bound_removable_loc": 0,
        "upper_bound_removable_loc": 0,
        "loci_of_change_before": 0,
        "loci_of_change_after": 0,
        "repeated_mappings_centralized": 0,
        "dispatch_sites_eliminated": 0,
        "registration_sites_removed": 0,
        "shared_algorithm_sites_centralized": 0,
    }
    impact["dispatch_sites_eliminated"] = 2
    return impact
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "semantic_dict_bag"
    )

    assert "ImpactDelta" in (finding.scaffold or "")


def test_builds_composed_subsystem_plan(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
REGISTRY = {}


class RuntimePlan:
    def __init__(self, pose_id, score, label):
        self.pose_id = pose_id
        self.score = score
        self.label = label


class Alpha:
    def _prepare(self, item):
        ready = self.normalize(item)
        checked = self.validate(ready)
        return self.finish(checked)

    def build(self, candidate):
        return RuntimePlan(
            pose_id=candidate.pose_id,
            score=candidate.score,
            label=candidate.label,
        )


class Beta:
    def _assemble(self, value):
        ready = self.normalize(value)
        checked = self.validate(ready)
        return self.finish(checked)

    def build(self, entry):
        return RuntimePlan(
            pose_id=entry.pose_id,
            score=entry.score,
            label=entry.label,
        )


REGISTRY["alpha"] = Alpha
REGISTRY["beta"] = Beta
""",
    )

    findings = analyze_path(tmp_path)
    plans = build_refactor_plans(findings, tmp_path)

    assert plans
    plan = plans[0]
    assert plan.primary_pattern_id == 5
    assert 6 in plan.secondary_pattern_ids
    assert 14 in plan.secondary_pattern_ids
    assert plan.outcome.loci_of_change_before > plan.outcome.loci_of_change_after
    assert plan.outcome.registration_sites_removed == 2
    assert plan.outcome.repeated_mappings_centralized >= 3
    assert any(action.kind == "create_abc_base" for action in plan.actions)
    assert any(action.kind == "create_metaclass" for action in plan.actions)
    extract_action = next(
        action for action in plan.actions if action.kind == "extract_template_method"
    )
    assert extract_action.statement_operation == "move"
    assert extract_action.statement_sites
    assert "self.normalize" in extract_action.description
    mapping_action = next(
        action
        for action in plan.actions
        if action.kind == "create_authoritative_schema"
    )
    assert mapping_action.create_symbol == "RuntimePlan.from_source"
    assert "name-for-name boilerplate" in mapping_action.description
    replace_action = next(
        action for action in plan.actions if action.kind == "replace_mapping_sites"
    )
    assert replace_action.statement_operation == "replace"
    assert replace_action.replace_with == "RuntimePlan.from_source(candidate)"


def test_markdown_output_can_include_subsystem_plans(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def _prepare(self, item):
        ready = self.normalize(item)
        checked = self.validate(ready)
        return self.finish(checked)


class Beta:
    def _build(self, value):
        ready = self.normalize(value)
        checked = self.validate(ready)
        return self.finish(checked)
""",
    )

    findings = analyze_path(tmp_path)
    plans = build_refactor_plans(findings, tmp_path)
    output = _format_markdown(findings, plans)

    assert "Subsystem plans:" in output
    assert "Primary pattern:" in output
    assert "Outcome:" in output
    assert "Action:" in output
    assert "Action sites:" in output


def test_detects_manual_family_roster_for_detector_registry(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class IssueDetector(ABC):
    pass


class AlphaDetector(IssueDetector):
    pass


class BetaDetector(IssueDetector):
    pass


def default_detectors():
    return (AlphaDetector(), BetaDetector())
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "manual_family_roster"
    )

    assert "default_detectors" in finding.summary
    assert "IssueDetector" in finding.summary
    assert "__init_subclass__" in (finding.scaffold or "")


def test_detects_fragmented_pattern_planning_tables(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class PatternId:
    ABC_TEMPLATE_METHOD = "abc"
    AUTHORITATIVE_SCHEMA = "schema"
    AUTO_REGISTER_META = "auto"


_PATTERN_DEPENDENCIES = {
    PatternId.ABC_TEMPLATE_METHOD: {PatternId.AUTHORITATIVE_SCHEMA},
    PatternId.AUTHORITATIVE_SCHEMA: {PatternId.AUTO_REGISTER_META},
    PatternId.AUTO_REGISTER_META: set(),
}


_PATTERN_PRIORITY = {
    PatternId.ABC_TEMPLATE_METHOD: 80,
    PatternId.AUTHORITATIVE_SCHEMA: 60,
    PatternId.AUTO_REGISTER_META: 50,
}


_PATTERN_BUILDERS = {
    PatternId.ABC_TEMPLATE_METHOD: build_abc,
    PatternId.AUTHORITATIVE_SCHEMA: build_schema,
    PatternId.AUTO_REGISTER_META: build_registry,
}
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "fragmented_family_authority"
    )

    assert "_PATTERN_DEPENDENCIES" in finding.summary
    assert "PatternId" in finding.summary
    assert "class PatternIdSpec" in (finding.scaffold or "")


def test_detects_existing_nominal_authority_reuse(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class EventCarrierBase(ABC):
    file_path: str
    line: int
    subject_name: str
    payload: tuple[str, ...]


@dataclass(frozen=True)
class DetachedEventCarrier:
    file_path: str
    line: int
    subject_name: str
    payload: tuple[str, ...]
    status: str
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "existing_nominal_authority_reuse"
    )

    assert "DetachedEventCarrier" in finding.summary
    assert "EventCarrierBase" in finding.summary
    assert "EventCarrierBase" in (finding.scaffold or "")


def test_detects_repeated_finding_assembly_pipeline(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class PerModuleIssueDetector:
    pass


class AlphaDetector(PerModuleIssueDetector):
    def _findings_for_module(self, module, config):
        findings = []
        for candidate in alpha_candidates(module):
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    summarize_alpha(candidate),
                    alpha_evidence(candidate),
                    scaffold=alpha_scaffold(candidate),
                    codemod_patch=alpha_patch(candidate),
                    metrics=AlphaMetrics(site_count=1),
                )
            )
        return findings


class BetaDetector(PerModuleIssueDetector):
    def _findings_for_module(self, module, config):
        findings = []
        for entry in beta_candidates(module):
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    summarize_beta(entry),
                    beta_evidence(entry),
                    scaffold=beta_scaffold(entry),
                    codemod_patch=beta_patch(entry),
                    metrics=BetaMetrics(site_count=1),
                )
            )
        return findings


class GammaDetector(PerModuleIssueDetector):
    def _findings_for_module(self, module, config):
        findings = []
        for witness in gamma_candidates(module):
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    summarize_gamma(witness),
                    gamma_evidence(witness),
                    scaffold=gamma_scaffold(witness),
                    codemod_patch=gamma_patch(witness),
                    metrics=GammaMetrics(site_count=1),
                )
            )
        return findings
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "finding_assembly_pipeline"
    )

    assert "AlphaDetector" in finding.summary
    assert "CandidateFindingDetector" in (finding.scaffold or "")


def test_detects_guarded_delegator_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class FunctionObservationSpec:
    pass


class ProjectionObservationSpec(FunctionObservationSpec):
    def build_from_function(self, parsed_module, function, observation):
        if observation.class_name is not None:
            return None
        return _projection_helper_shape_from_function(parsed_module, function)


class AccessorObservationSpec(FunctionObservationSpec):
    def build_from_function(self, parsed_module, function, observation):
        if observation.class_name is None:
            return None
        return _accessor_wrapper_candidate_from_function(parsed_module, observation.class_name, function)


class SpecAssignmentObservationSpec(FunctionObservationSpec):
    def build_from_function(self, parsed_module, function, observation):
        if observation.function_name is None:
            return None
        return _spec_candidate_from_function(parsed_module, function)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "guarded_delegator_spec"
    )

    assert "Observation specs" in finding.summary
    assert "ScopeFilteredSpec" in (finding.scaffold or "")


def test_detects_repeated_structural_observation_projection(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class StructuralObservation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MethodShape:
    @property
    def structural_observation(self):
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.class_name,
            line=self.lineno,
            observation_kind=self.observation_kind,
            execution_level=self.execution_level,
            observed_name=self.method_name,
            fiber_key=self.method_name,
        )


class BuilderShape:
    @property
    def structural_observation(self):
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.class_name,
            line=self.lineno,
            observation_kind=self.observation_kind,
            execution_level=self.execution_level,
            observed_name=self.builder_name,
            fiber_key=self.builder_name,
        )


class ExportShape:
    @property
    def structural_observation(self):
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            nominal_witness=self.class_name,
            line=self.lineno,
            observation_kind=self.observation_kind,
            execution_level=self.execution_level,
            observed_name=self.export_name,
            fiber_key=self.export_name,
        )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "structural_observation_projection"
    )

    assert "StructuralObservation schema" in finding.summary
    assert "StructuralObservationTemplate" in (finding.scaffold or "")


def test_detects_repeated_property_alias_hooks_across_subclasses(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class ProjectionTemplate(ABC):
    @property
    def observation_kind(self):
        raise NotImplementedError


class AlphaProjection(ProjectionTemplate):
    @property
    def observation_line(self):
        return self.lineno


class BetaProjection(ProjectionTemplate):
    @property
    def observation_line(self):
        return self.lineno
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_property_alias_hooks"
    )

    assert "ProjectionTemplate" in finding.summary
    assert "observation_line" in finding.summary
    assert "self.lineno" in finding.summary


def test_detects_constant_property_hooks_across_subclasses(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class ObservationKind:
    FIELD = "field"
    METHOD = "method"


class ProjectionTemplate(ABC):
    @property
    def observation_kind(self):
        raise NotImplementedError


class AlphaProjection(ProjectionTemplate):
    @property
    def observation_kind(self):
        return ObservationKind.FIELD


class BetaProjection(ProjectionTemplate):
    @property
    def observation_kind(self):
        return ObservationKind.METHOD
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "constant_property_hooks"
    )

    assert "ProjectionTemplate" in finding.summary
    assert "observation_kind" in finding.summary
    assert "ObservationKind.FIELD" in finding.summary
    assert "ObservationKind.METHOD" in finding.summary


def test_detects_reflective_self_attribute_escape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class ProjectionTemplate(ABC):
    @property
    def path_text(self):
        return getattr(self, "file_path")
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "reflective_self_attribute_escape"
    )

    assert "getattr(self, 'file_path')" in finding.summary
    assert "file_path" in (finding.scaffold or "")


def test_detects_helper_backed_observation_spec_wrappers(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class FunctionObservationSpec(ABC):
    pass


class ScopeFilteredFunctionObservationSpec(FunctionObservationSpec, ABC):
    pass


class AssignObservationSpec(ABC):
    pass


class ModuleOnlyAssignObservationSpec(AssignObservationSpec, ABC):
    pass


class ClassMarkerObservationSpec(FunctionObservationSpec):
    pass


class InterfaceGenerationObservationSpec(FunctionObservationSpec):
    pass


class DynamicMethodInjectionObservationSpec(FunctionObservationSpec):
    pass


class ProjectionHelperObservationSpec(ScopeFilteredFunctionObservationSpec):
    pass


class ScopedShapeWrapperObservationSpec(ModuleOnlyAssignObservationSpec):
    pass


class StandardClassMarkerObservationSpec(ClassMarkerObservationSpec):
    def build_from_function(self, parsed_module, function, observation):
        return tuple(_class_marker_observations(parsed_module, function))


class StandardInterfaceGenerationObservationSpec(InterfaceGenerationObservationSpec):
    def build_from_function(self, parsed_module, function, observation):
        return _interface_generation_observation(parsed_module, function)


class StandardDynamicMethodInjectionObservationSpec(DynamicMethodInjectionObservationSpec):
    def build_from_function(self, parsed_module, function, observation):
        return tuple(_dynamic_method_injection_observations(parsed_module, function))


class StandardProjectionHelperObservationSpec(ProjectionHelperObservationSpec):
    def build_scoped_function(self, parsed_module, function, observation):
        return _projection_helper_shape_from_function(parsed_module, function)


class ScopedShapeWrapperSpecObservationSpec(ScopedShapeWrapperObservationSpec):
    def build_scoped_assign(self, parsed_module, node, observation):
        return _scoped_shape_wrapper_spec_from_assign(parsed_module, node)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "helper_backed_observation_spec"
    )

    assert "StandardClassMarkerObservationSpec" in finding.summary
    assert "_class_marker_observations" in finding.summary
    assert "HelperBackedFunctionObservationSpec" in (finding.scaffold or "")


def test_detects_dynamic_self_field_selection(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class CountedDispatchMetrics(ABC):
    count_field_name = "branch_site_count"

    def _count_value(self):
        return int(getattr(self, self.count_field_name))
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "dynamic_self_field_selection"
    )

    assert "getattr(self, self.count_field_name)" in finding.summary
    assert "count_value" in (finding.scaffold or "")


def test_detects_declarative_family_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class CollectedFamily(ABC):
    pass


class RegisteredSpecCollectedFamily(CollectedFamily):
    pass


class ObservationFamily(CollectedFamily):
    pass


class AlphaFamily(RegisteredSpecCollectedFamily, ObservationFamily):
    item_type = Alpha
    spec_root = AlphaSpec


class BetaFamily(RegisteredSpecCollectedFamily, ObservationFamily):
    item_type = Beta
    spec_root = BetaSpec


class GammaFamily(RegisteredSpecCollectedFamily, ObservationFamily):
    item_type = Gamma
    spec_root = GammaSpec
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "declarative_family_boilerplate"
    )

    assert "AlphaFamily" in finding.summary
    assert "item_type" in finding.summary
    assert "spec_root" in finding.summary
    assert "declarative family-definition table" in (finding.codemod_patch or "")


def test_detects_type_indexed_definition_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class CollectedFamily(ABC):
    pass


class RegisteredObservationFamilyDefinition(ABC):
    pass


class AlphaFamilyDefinition(RegisteredObservationFamilyDefinition):
    item_type = Alpha
    spec_root = AlphaSpec


AlphaFamily = AlphaFamilyDefinition.family_type


class BetaFamilyDefinition(RegisteredObservationFamilyDefinition):
    item_type = Beta
    spec_root = BetaSpec


BetaFamily = BetaFamilyDefinition.family_type


class GammaFamilyDefinition(RegisteredObservationFamilyDefinition):
    item_type = Gamma
    spec_root = GammaSpec


GammaFamily = GammaFamilyDefinition.family_type
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "type_indexed_definition_boilerplate"
    )

    assert "AlphaFamilyDefinition" in finding.summary
    assert "AlphaFamily" in finding.summary
    assert "typed declaration table" in (finding.codemod_patch or "")


def test_detects_manual_derived_export_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class AutoRegisteredModuleShapeSpec(ABC):
    pass


class CollectedFamily(ABC):
    pass


class AlphaSpec(AutoRegisteredModuleShapeSpec):
    pass


class BetaSpec(AutoRegisteredModuleShapeSpec):
    pass


class GammaFamily(CollectedFamily):
    pass


class DeltaFamily(CollectedFamily):
    pass


class EpsilonSpec(AutoRegisteredModuleShapeSpec):
    pass


class ZetaFamily(CollectedFamily):
    pass


_STATIC_EXPORT_NAMES = (
    "AlphaSpec",
    "BetaSpec",
    "GammaFamily",
    "DeltaFamily",
    "EpsilonSpec",
    "ZetaFamily",
)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "derived_export_surface"
    )

    assert "_STATIC_EXPORT_NAMES" in finding.summary
    assert (
        "AutoRegisteredModuleShapeSpec" in finding.summary
        or "CollectedFamily" in finding.summary
    )
    assert "public_exports" in (finding.scaffold or "")


def test_detects_manual_derived_index_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class CollectedFamily(ABC):
    pass


class AlphaFamily(CollectedFamily):
    pass


class BetaFamily(CollectedFamily):
    pass


class GammaFamily(CollectedFamily):
    pass


FAMILY_BY_NAME = {
    "alpha": AlphaFamily,
    "beta": BetaFamily,
    "gamma": GammaFamily,
}
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "derived_indexed_surface"
    )

    assert "FAMILY_BY_NAME" in finding.summary
    assert "CollectedFamily" in finding.summary
    assert "derived_index" in (finding.scaffold or "")


def test_detects_manual_public_api_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    pass


class Beta:
    pass


def gamma():
    return 1


def delta():
    return 2


__all__ = ["Alpha", "Beta", "gamma", "delta"]
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "manual_public_api_surface"
    )

    assert "__all__" in finding.summary
    assert "public API" in finding.title
    assert "is_public_api_export" in (finding.scaffold or "")


def test_detects_manual_registered_union_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class ShapeFamily:
    @classmethod
    def registered_families(cls):
        return ()


class ObservationFamily:
    @classmethod
    def registered_families(cls):
        return ()


def collect_everything():
    for family in ShapeFamily.registered_families() + ObservationFamily.registered_families():
        yield family
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "registered_union_surface"
    )

    assert "collect_everything" in finding.summary
    assert "registered_families" in finding.summary
    assert "all_registered_families" in (finding.scaffold or "")


def test_detects_alternate_constructor_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class RegistrationShape:
    @classmethod
    def from_assignment(cls, parsed_module, node: Assign, registry_name, key_fingerprint):
        return cls(
            file_path=parsed_module.path,
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=node.value.id,
            key_fingerprint=key_fingerprint,
            key_expression=node.target,
            registration_style="assignment",
        )

    @classmethod
    def from_registration_call(cls, parsed_module, node: Call, registry_name, key_fingerprint):
        return cls(
            file_path=parsed_module.path,
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=node.func.id,
            key_fingerprint=key_fingerprint,
            key_expression=node.args[0],
            registration_style="call",
        )

    @classmethod
    def from_decorator(cls, parsed_module, node: ClassDef, registry_name, key_fingerprint):
        return cls(
            file_path=parsed_module.path,
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=node.name,
            key_fingerprint=key_fingerprint,
            key_expression=node.name,
            registration_style="decorator",
        )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "alternate_constructor_family"
    )

    assert "RegistrationShape" in finding.summary
    assert "from_assignment" in finding.summary
    assert "@singledispatchmethod" in (finding.scaffold or "")
