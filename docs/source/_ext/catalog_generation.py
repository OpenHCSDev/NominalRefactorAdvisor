from __future__ import annotations

from pathlib import Path
from typing import Iterable

from nominal_refactor_advisor.detectors import IssueDetector
from nominal_refactor_advisor.patterns import PATTERN_SPECS, PatternSpec


def generate_api_reference_pages(source_dir: Path) -> None:
    generated_dir = source_dir / "api" / "_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    detector_types = IssueDetector.registered_detector_types()
    _write_if_changed(
        generated_dir / "pattern_catalog.rst",
        _render_pattern_catalog(
            sorted(PATTERN_SPECS.values(), key=lambda item: item.pattern_id.value)
        ),
    )
    _write_if_changed(
        generated_dir / "detector_catalog.rst",
        _render_detector_catalog(detector_types),
    )
    _write_if_changed(
        generated_dir / "detector_reference_index.rst",
        _render_detector_reference_index(detector_types),
    )
    detector_reference_dir = source_dir / "api" / "detector_reference"
    detector_reference_dir.mkdir(parents=True, exist_ok=True)
    for detector_type in detector_types:
        _write_if_changed(
            detector_reference_dir / f"{detector_type.detector_id}.rst",
            _render_detector_reference_page(detector_type),
        )


def _render_pattern_catalog(patterns: list[PatternSpec]) -> str:
    lines = [
        ".. This file is generated from nominal_refactor_advisor.patterns.PATTERN_SPECS.",
        ".. Do not edit manually.",
        "",
        "This catalog is generated from ``nominal_refactor_advisor.patterns.PATTERN_SPECS``.",
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
        "     - Priority",
        "     - Dependencies",
        "     - Plan Builder",
        "     - Action Builder",
    ]
    for pattern in patterns:
        dependencies = _pattern_id_list(pattern.dependencies) or "None"
        plan_builder = (
            f"``{pattern.plan_step_builder_id.value}``"
            if pattern.plan_step_builder_id is not None
            else "None"
        )
        action_builder = (
            f"``{pattern.action_builder_id.value}``"
            if pattern.action_builder_id is not None
            else "None"
        )
        lines.extend(
            [
                f"   * - ``{pattern.pattern_id.value}``",
                f"     - {pattern.name}",
                f"     - ``{pattern.priority}``",
                f"     - {dependencies}",
                f"     - {plan_builder}",
                f"     - {action_builder}",
            ]
        )
    lines.extend(["", "Patterns", "--------", ""])
    for pattern in patterns:
        title = f"Pattern {pattern.pattern_id.value}: {pattern.name}"
        lines.extend(
            [
                title,
                "^" * len(title),
                "",
                f":Prescription: {pattern.prescription}",
                f":Canonical shape: {pattern.canonical_shape}",
                f":Priority: ``{pattern.priority}``",
                f":Dependencies: {_pattern_id_list(pattern.dependencies) or 'None'}",
                f":Synergy: {_pattern_id_list(pattern.synergy_with) or 'None'}",
                f":Witness capabilities: {_capability_list(pattern.witness_capabilities) or 'None'}",
                f":Plan builder: ``{pattern.plan_step_builder_id.value}``"
                if pattern.plan_step_builder_id is not None
                else ":Plan builder: None",
                f":Action builder: ``{pattern.action_builder_id.value}``"
                if pattern.action_builder_id is not None
                else ":Action builder: None",
                "",
                "First moves:",
                "",
            ]
        )
        lines.extend(f"- {step}" for step in pattern.first_moves)
        if pattern.example_skeletons:
            lines.extend(["", "Example skeletons:", ""])
            for skeleton in pattern.example_skeletons:
                lines.extend([".. code-block:: python", ""])
                lines.extend(
                    f"   {line}" if line else "" for line in skeleton.splitlines()
                )
                lines.append("")
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
        "     - Class",
        "     - Base",
        "     - Genericity",
        "     - Priority",
    ]
    for detector_type in detector_types:
        lines.extend(
            [
                f"   * - ``{detector_type.detector_id}``",
                f"     - ``{detector_type.__name__}``",
                f"     - ``{_detector_base_name(detector_type)}``",
                f"     - ``{detector_type.genericity}``",
                f"     - ``{detector_type.detector_priority}``",
            ]
        )
    lines.extend(["", "Detectors", "---------", ""])
    for detector_type in detector_types:
        title = detector_type.__name__
        lines.extend(
            [
                title,
                "^" * len(title),
                "",
                f":Detector ID: ``{detector_type.detector_id}``",
                f":Base: ``{_detector_base_name(detector_type)}``",
                f":Genericity: ``{detector_type.genericity}``",
                f":Priority: ``{detector_type.detector_priority}``",
                f":Reference: :doc:`detector_reference/{detector_type.detector_id}`",
                f":Summary: {_doc_summary(detector_type)}",
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
        f"   detector_reference/{detector_type.detector_id}"
        for detector_type in detector_types
    )
    lines.append("")
    return "\n".join(lines)


def _render_detector_reference_page(detector_type: type[IssueDetector]) -> str:
    qualified_name = f"nominal_refactor_advisor.detectors.{detector_type.__name__}"
    title = detector_type.__name__
    lines = [
        ".. This file is generated from nominal_refactor_advisor.detectors.IssueDetector.",
        ".. Do not edit manually.",
        "",
        title,
        "=" * len(title),
        "",
        f":Detector ID: ``{detector_type.detector_id}``",
        f":Base: ``{_detector_base_name(detector_type)}``",
        f":Genericity: ``{detector_type.genericity}``",
        f":Priority: ``{detector_type.detector_priority}``",
        f":Implementation module: ``{detector_type.__module__}``",
        "",
        f"{_doc_summary(detector_type)}",
        "",
        f".. autoclass:: {qualified_name}",
        "   :show-inheritance:",
        "",
    ]
    return "\n".join(lines)


def _pattern_id_list(pattern_ids: Iterable[object]) -> str:
    return ", ".join(f"``{pattern_id.value}``" for pattern_id in pattern_ids)


def _capability_list(capabilities: Iterable[object]) -> str:
    return ", ".join(f"``{capability.name}``" for capability in capabilities)


def _detector_base_name(detector_type: type[IssueDetector]) -> str:
    for base in detector_type.__mro__[1:]:
        if issubclass(base, IssueDetector) and base is not IssueDetector:
            return base.__name__
    return IssueDetector.__name__


def _doc_summary(detector_type: type[IssueDetector]) -> str:
    doc = detector_type.__doc__
    if not doc:
        return (
            "Internal detector implementation; inspect the detector ID and finding output "
            "for semantics."
        )
    first_line = doc.strip().splitlines()[0].strip()
    return first_line.rstrip(".") + "."


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)
