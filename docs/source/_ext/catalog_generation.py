from __future__ import annotations

from pathlib import Path
from typing import Iterable

from nominal_refactor_advisor.detectors import IssueDetector
from nominal_refactor_advisor.models import FindingSpec
from nominal_refactor_advisor.patterns import PatternId


def generate_api_reference_pages(source_dir: Path) -> None:
    generated_dir = source_dir / "api" / "_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    detector_types = IssueDetector.registered_detector_types()
    _write_if_changed(
        generated_dir / "pattern_catalog.rst",
        _render_pattern_catalog(list(PatternId)),
    )
    _write_if_changed(
        generated_dir / "detector_catalog.rst", _render_detector_catalog(detector_types)
    )
    _write_if_changed(
        generated_dir / "detector_reference_index.rst",
        _render_detector_reference_index(detector_types),
    )
    detector_reference_dir = source_dir / "api" / "detector_reference"
    detector_reference_dir.mkdir(parents=True, exist_ok=True)
    detector_reference_paths = {
        detector_reference_dir / f"{detector_type.detector_id}.rst"
        for detector_type in detector_types
    }
    for stale_path in (
        set(detector_reference_dir.glob("*.rst")) - detector_reference_paths
    ):
        stale_path.unlink()
    for detector_type in detector_types:
        _write_if_changed(
            detector_reference_dir / f"{detector_type.detector_id}.rst",
            _render_detector_reference_page(detector_type),
        )


def _render_pattern_catalog(patterns: list[PatternId]) -> str:
    lines = [
        ".. This file is generated from nominal_refactor_advisor.patterns.PatternId.",
        ".. Do not edit manually.",
        "",
        "This catalog is generated from ``nominal_refactor_advisor.patterns.PatternId``.",
        "The code metadata remains the authoritative source; this page is only a rendered view.",
        "",
        "Summary",
        "-------",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - ID",
        "     - Name",
        "     - Required relation",
    ]
    for pattern in patterns:
        lines.extend(
            [
                f"   * - ``{pattern.value}``",
                f"     - {pattern.display_name}",
                f"     - {pattern.required_relation}",
            ]
        )
    lines.extend(["", "Patterns", "--------", ""])
    for pattern in patterns:
        title = f"Pattern {pattern.value}: {pattern.display_name}"
        lines.extend(
            [
                title,
                "^" * len(title),
                "",
                f":Required relation: {pattern.required_relation}",
                f":Witness capabilities: {_capability_list(pattern.witness_capabilities) or 'None'}",
            ]
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_detector_catalog(detector_types: tuple[type[IssueDetector], ...]) -> str:
    lines = [
        ".. This file is generated from nominal_refactor_advisor.detectors.IssueDetector.",
        ".. Do not edit manually.",
        "",
        "This catalog is generated from the registered detector family rooted at",
        "``nominal_refactor_advisor.detectors.IssueDetector``. The registry order is the",
        "authoritative source for what the tool ships.",
        "",
        "Summary",
        "-------",
        "",
        f"- Total detectors: ``{len(detector_types)}``",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Detector ID",
        "     - Pattern",
        "     - Finding title",
        "     - Class",
        "     - Base",
    ]
    for detector_type in detector_types:
        finding_spec = _finding_spec(detector_type)
        lines.extend(
            [
                f"   * - ``{detector_type.detector_id}``",
                (
                    f"     - ``{finding_spec.pattern_id.value}``"
                    if finding_spec is not None
                    else "     - Unknown"
                ),
                (
                    f"     - {finding_spec.title}"
                    if finding_spec is not None
                    else "     - Unknown"
                ),
                f"     - ``{detector_type.__name__}``",
                f"     - ``{_detector_base_name(detector_type)}``",
            ]
        )
    lines.extend(["", "Detectors", "---------", ""])
    for detector_type in detector_types:
        title = detector_type.__name__
        finding_spec = _finding_spec(detector_type)
        lines.extend(
            [
                title,
                "^" * len(title),
                "",
                f":Detector ID: ``{detector_type.detector_id}``",
                (
                    f":Pattern: ``{finding_spec.pattern_id.value}``"
                    if finding_spec is not None
                    else ":Pattern: Unknown"
                ),
                f":Base: ``{_detector_base_name(detector_type)}``",
                f":Reference: :doc:`detector_reference/{detector_type.detector_id}`",
                f":Summary: {_detector_summary(detector_type)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_detector_reference_index(
    detector_types: tuple[type[IssueDetector], ...],
) -> str:
    lines = [
        ".. This file is generated from nominal_refactor_advisor.detectors.IssueDetector.",
        ".. Do not edit manually.",
        "",
        "These pages provide one generated reference page per registered detector.",
        "The detector registry remains the authoritative source.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    lines.extend(
        (
            f"   detector_reference/{detector_type.detector_id}"
            for detector_type in detector_types
        )
    )
    lines.append("")
    return "\n".join(lines)


def _render_detector_reference_page(detector_type: type[IssueDetector]) -> str:
    qualified_name = f"nominal_refactor_advisor.detectors.{detector_type.__name__}"
    title = detector_type.__name__
    finding_spec = _finding_spec(detector_type)
    lines = [
        ".. This file is generated from nominal_refactor_advisor.detectors.IssueDetector.",
        ".. Do not edit manually.",
        "",
        title,
        "=" * len(title),
        "",
        f":Detector ID: ``{detector_type.detector_id}``",
        f":Base: ``{_detector_base_name(detector_type)}``",
        f":Implementation module: ``{detector_type.__module__}``",
        "",
        f"{_detector_summary(detector_type)}",
        "",
    ]
    if finding_spec is not None:
        lines.extend(
            [
                "Default Finding Semantics",
                "-------------------------",
                "",
                f":Pattern: ``{finding_spec.pattern_id.value}``",
                f":Title: {_rst_field_value(finding_spec.title)}",
                f":Why: {_rst_field_value(finding_spec.why)}",
                f":Capability gap: {_rst_field_value(finding_spec.capability_gap)}",
                f":Relation context: {_rst_field_value(finding_spec.relation_context)}",
                f":Default confidence: ``{finding_spec.confidence.name}``",
                f":Default certification: ``{finding_spec.certification.name}``",
                (
                    f":Capability tags: {_enum_name_list(finding_spec.capability_tags)}"
                    if finding_spec.capability_tags
                    else ":Capability tags: None"
                ),
                (
                    f":Observation tags: {_enum_name_list(finding_spec.observation_tags)}"
                    if finding_spec.observation_tags
                    else ":Observation tags: None"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "Implementation",
            "--------------",
            "",
            f".. autoclass:: {qualified_name}",
            "   :show-inheritance:",
            "",
        ]
    )
    return "\n".join(lines)


def _rst_field_value(value: str) -> str:
    return value.replace("**", r"\*\*")


def _capability_list(capabilities: Iterable[object]) -> str:
    return ", ".join((f"``{capability.name}``" for capability in capabilities))


def _detector_base_name(detector_type: type[IssueDetector]) -> str:
    for base in detector_type.__mro__[1:]:
        if issubclass(base, IssueDetector) and base is not IssueDetector:
            return base.__name__
    return IssueDetector.__name__


def _doc_summary(detector_type: type[IssueDetector]) -> str:
    doc = detector_type.__doc__
    if not doc:
        return "Internal detector implementation; inspect the detector ID and finding output for semantics."
    first_line = doc.strip().splitlines()[0].strip()
    return first_line.rstrip(".") + "."


def _finding_spec(detector_type: type[IssueDetector]) -> FindingSpec | None:
    finding_spec = getattr(detector_type, "finding_spec", None)
    if isinstance(finding_spec, FindingSpec):
        return finding_spec
    return None


def _detector_summary(detector_type: type[IssueDetector]) -> str:
    finding_spec = _finding_spec(detector_type)
    if finding_spec is not None:
        return finding_spec.title
    return _doc_summary(detector_type)


def _enum_name_list(values: Iterable[object]) -> str:
    return ", ".join((f"``{value.name}``" for value in values))


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)
