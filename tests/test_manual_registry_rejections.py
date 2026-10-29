"""Manual-registry conversion names the premise that failed, instead of one generic sentence."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.detectors._runtime import ManualClassRegistrationDetector

REGISTRY_MODULE = (
    "REGISTRY = {}\n\n\n"
    "class AlphaHandler:\n    pass\n\n\n"
    "class BetaHandler:\n    pass\n\n\n"
    'REGISTRY["alpha"] = AlphaHandler\n'
    'REGISTRY["beta"] = BetaHandler\n'
)


def _registry_finding(root: Path):
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "mod.py").write_text(REGISTRY_MODULE)
    modules = parse_python_modules(root)
    detector = ManualClassRegistrationDetector()
    (finding,) = detector.detect(modules, DetectorConfig())
    return detector, finding


def test_a_complete_direct_registry_has_an_executable_recipe(tmp_path: Path) -> None:
    detector, finding = _registry_finding(tmp_path)
    modules = parse_python_modules(tmp_path)

    evaluation = detector.evaluate_recipe_for_finding(
        finding, CodemodSourceSnapshot.from_modules(modules, [finding])
    )

    assert evaluation.rejection_reason == ""


def test_a_finding_whose_registry_was_renamed_is_refused_by_both_names(
    tmp_path: Path,
) -> None:
    detector, finding = _registry_finding(tmp_path)
    (tmp_path / "pkg" / "mod.py").write_text(
        REGISTRY_MODULE.replace("REGISTRY", "HANDLERS")
    )
    modules = parse_python_modules(tmp_path)

    evaluation = detector.evaluate_recipe_for_finding(
        finding, CodemodSourceSnapshot.from_modules(modules, [finding])
    )

    assert evaluation.rejection_reason == (
        "manual-registry conversion: the registry at `AlphaHandler` is "
        "`HANDLERS`, not the finding's `REGISTRY`"
    )
