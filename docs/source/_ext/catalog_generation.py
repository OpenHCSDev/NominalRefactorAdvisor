from __future__ import annotations

from pathlib import Path
from typing import Iterable

from nominal_refactor_advisor.codemod import RefactorRecipeOperation

from nominal_refactor_advisor.detector_capabilities import (
    DetectorContributionRole,
    DetectorRefactorCapability,
    DetectorRefactorCapabilityReport,
)
from nominal_refactor_advisor.detectors import IssueDetector
from nominal_refactor_advisor.models import NominalDeclarationIdentity
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.source_geometry import read_source_text


def generate_api_reference_pages(source_dir: Path) -> None:
    generated_dir = source_dir / "api" / "_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    detector_types = IssueDetector.registered_detector_types()
    _write_if_changed(generated_dir / "codemod_catalog.rst", _render_codemod_catalog())
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


def _render_codemod_catalog() -> str:
    lines = [
        ".. Generated from RefactorRecipeOperation.__registry__.",
        ".. Do not edit manually.",
        "",
    ]
    for operation in RefactorRecipeOperation.__registry__.values():
        declaration = NativeDeclaration(operation)
        title = operation.__name__
        lines.extend(
            [
                title,
                "-" * len(title),
                "",
                f":Declaration: ``{declaration.qualified_name}``",
                f":Operation key: ``{operation.operation_key()}``",
                f":Source proof scope: ``{operation.source_dependency_scope.value}``",
                "",
                f".. autoclass:: {declaration.qualified_name}",
                "   :show-inheritance:",
                "   :no-index:",
                "",
            ]
        )
    return "\n".join(lines)


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
    capability_report = DetectorRefactorCapabilityReport(
        tuple(DetectorRefactorCapability(item) for item in detector_types)
    )
    lines = [
        ".. This file is generated from nominal_refactor_advisor.detectors.IssueDetector.",
        ".. Do not edit manually.",
        "",
        "This catalog is generated from the registered detector family rooted at",
        "``nominal_refactor_advisor.detectors.IssueDetector``. The registry order is the",
        "authoritative source for what the tool ships.",
        "",
        "Contribution roles",
        "------------------",
        "",
        *(
            f"- ``{role.value}``: {role.description}"
            for role in DetectorContributionRole
        ),
        "",
        "Summary",
        "-------",
        "",
        f"- Total detectors: ``{len(detector_types)}``",
        *(
            f"- {_contribution_label(role)}: "
            f"``{capability_report.contribution_count(role)}``"
            for role in DetectorContributionRole
        ),
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Detector ID",
        "     - Pattern",
        "     - Required-relation owner",
        "     - Contributions",
        "     - Recipe synthesis concept",
    ]
    for capability in capability_report.capabilities:
        detector_type = capability.detector_type
        finding_spec = detector_type.required_relation_finding_spec()
        lines.extend(
            [
                f"   * - ``{detector_type.detector_id}``",
                f"     - ``{finding_spec.pattern_id.value}``",
                f"     - ``{capability.required_relation.qualname}``",
                f"     - {_contribution_list(capability)}",
                f"     - {_declaration_reference(capability.recipe_synthesis_concept)}",
            ]
        )
    lines.extend(["", "Detectors", "---------", ""])
    for detector_type in detector_types:
        title = detector_type.__name__
        finding_spec = detector_type.required_relation_finding_spec()
        capability = DetectorRefactorCapability(detector_type)
        lines.extend(
            [
                title,
                "^" * len(title),
                "",
                f":Detector ID: ``{detector_type.detector_id}``",
                f":Pattern: ``{finding_spec.pattern_id.value}``",
                f":Base: ``{_detector_base_name(detector_type)}``",
                f":Reference: :doc:`detector_reference/{detector_type.detector_id}`",
                f":Summary: {_detector_summary(detector_type)}",
                f":Required-relation owner: ``{capability.required_relation.qualname}``",
                f":Contributions: {_contribution_list(capability)}",
                f":Recipe synthesis concept: {_declaration_reference(capability.recipe_synthesis_concept)}",
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
    qualified_name = NativeDeclaration(detector_type).qualified_name
    title = detector_type.__name__
    finding_spec = detector_type.required_relation_finding_spec()
    capability = DetectorRefactorCapability(detector_type)
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
        "Declared Refactoring Capabilities",
        "---------------------------------",
        "",
        f":Required-relation owner: ``{capability.required_relation.qualname}``",
        f":Contributions: {_contribution_list(capability)}",
        f":Recipe synthesis concept: {_declaration_reference(capability.recipe_synthesis_concept)}",
        "",
    ]
    lines.extend(_render_contract_fulfillment(capability))
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


def _detector_summary(detector_type: type[IssueDetector]) -> str:
    return detector_type.required_relation_finding_spec().title


def _declaration_reference(
    identity: NominalDeclarationIdentity | None,
) -> str:
    return "None" if identity is None else f"``{identity.qualname}``"


def _contribution_label(role: DetectorContributionRole) -> str:
    return role.value.replace("_", " ").capitalize()


def _contribution_list(capability: DetectorRefactorCapability) -> str:
    return ", ".join(
        f"``{contribution.role.value}``" for contribution in capability.contributions
    )


def _render_contract_fulfillment(
    capability: DetectorRefactorCapability,
) -> list[str]:
    lines = [
        "Contract Fulfillment",
        "--------------------",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Contribution",
        "     - Contract",
        "     - MRO resolution path",
        "     - Contract member implementations",
    ]
    for contribution in capability.contributions:
        mro_path = " -> ".join(
            f"``{declaration.qualname}``"
            for declaration in contribution.mro_resolution_path
        )
        member_implementations = ", ".join(
            f"``{member.member_name}`` by ``{member.implementation.qualname}``"
            for member in contribution.member_evidence
        )
        lines.extend(
            [
                f"   * - ``{contribution.role.value}``",
                f"     - ``{contribution.contract.qualname}``",
                f"     - {mro_path}",
                f"     - {member_implementations or 'Nominal membership only'}",
            ]
        )
    lines.append("")
    return lines


def _enum_name_list(values: Iterable[object]) -> str:
    return ", ".join((f"``{value.name}``" for value in values))


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and read_source_text(path) == content:
        return
    path.write_text(content, encoding="utf-8", newline="")
