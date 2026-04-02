from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.cli import _format_markdown
from nominal_refactor_advisor.cli import analyze_path
from nominal_refactor_advisor.models import DispatchCountMetrics
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
        return self.package(bounded)


class Beta:
    def _build(self, value):
        ready = self.normalize(value)
        checked = self.validate(ready)
        return self.finish(checked)

    def _evaluate(self, value):
        scored = self.compute(value)
        bounded = self.bound(scored)
        return self.package(bounded)
""",
    )

    findings = analyze_path(tmp_path)
    assert any(
        finding.detector_id == "inheritance_hierarchy_candidate" for finding in findings
    )


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
    assert any(finding.pattern_id == 3 for finding in findings)


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
    assert any(
        finding.detector_id == "numeric_literal_dispatch" for finding in findings
    )


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
    assert finding.metrics.dispatch_site_count == 1


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
