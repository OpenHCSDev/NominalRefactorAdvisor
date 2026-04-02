from __future__ import annotations

from pathlib import Path

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
