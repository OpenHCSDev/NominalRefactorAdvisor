from __future__ import annotations

from pathlib import Path

from dq_dock_engine.refactor_advisor.cli import _format_markdown
from dq_dock_engine.refactor_advisor.cli import analyze_path


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
