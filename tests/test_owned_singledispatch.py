"""Regression tests for `OwnedSingleDispatchDetector` (PatternId 65, IMPL-3 / IMPL-4)."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.detectors._base import DetectorConfig, IssueDetector
from nominal_refactor_advisor.detectors._owned_singledispatch import (
    OwnedSingleDispatchDetector,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _detect(root: Path):
    with ScanCache.scope():
        modules = parse_python_modules(root, use_parse_cache=False)
        return OwnedSingleDispatchDetector().detect(modules, DetectorConfig())


def test_pattern_id_and_registry_include_owned_singledispatch() -> None:
    assert PatternId.OWNED_SINGLEDISPATCH == 65
    assert OwnedSingleDispatchDetector in IssueDetector.__registry__.values()


def test_flags_singledispatch_and_singledispatchmethod_on_owned_classes(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/policies.py": (
                "class VisionPolicy:\n    pass\n\n\n"
                "class DistillVision(VisionPolicy):\n    pass\n\n\n"
                "class ConfidenceHead:\n    pass\n"
            ),
            "pkg/runner.py": (
                "from functools import singledispatch, singledispatchmethod\n"
                "from pkg.policies import ConfidenceHead, DistillVision, VisionPolicy\n\n\n"
                "@singledispatch\n"
                "def apply_policy(policy: object) -> str:\n"
                "    return 'default'\n\n\n"
                "@apply_policy.register(DistillVision)\n"
                "def _apply_distill(policy: DistillVision) -> str:\n"
                "    return 'distill'\n\n\n"
                "@apply_policy.register\n"
                "def _apply_head(policy: ConfidenceHead) -> str:\n"
                "    return 'head'\n\n\n"
                "class Executor:\n"
                "    @singledispatchmethod\n"
                "    def run(self, policy: object) -> int:\n"
                "        return 0\n\n"
                "    @run.register\n"
                "    def _(self, policy: VisionPolicy) -> int:\n"
                "        return 1\n"
            ),
        },
    )
    findings = _detect(tmp_path)
    assert len(findings) == 2
    assert all(f.pattern_id is PatternId.OWNED_SINGLEDISPATCH for f in findings)
    by_sym = {f.primary_evidence.symbol: f.summary for f in findings}
    assert sorted(by_sym) == ["Executor.run", "apply_policy"]
    assert "`DistillVision`" in by_sym["apply_policy"]
    assert "`ConfidenceHead`" in by_sym["apply_policy"]
    assert "`VisionPolicy`" in by_sym["Executor.run"]


def test_ignores_singledispatch_over_stdlib_or_external_types_only(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/serialize.py": (
                "from functools import singledispatch\n"
                "from pathlib import Path\n\n\n"
                "@singledispatch\n"
                "def to_json(value: object) -> object:\n"
                "    return str(value)\n\n\n"
                "@to_json.register(Path)\n"
                "def _(value: Path) -> str:\n"
                "    return value.as_posix()\n\n\n"
                "@to_json.register\n"
                "def _(value: int) -> int:\n"
                "    return value\n"
            ),
        },
    )
    assert _detect(tmp_path) == []
