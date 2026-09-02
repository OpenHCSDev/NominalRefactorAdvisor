"""Adapter for Lean-side advisor exports.

Lean owns proof-environment extraction; Python owns common advisor reporting.
This module keeps that boundary explicit by converting the Lean JSON schema into
the same ``RefactorFinding`` records emitted by Python detectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta

from .codemod_payload import JsonObject
from .detectors._base import high_confidence_spec
from .models import FindingSpec, RefactorFinding, SourceLocation
from .patterns import PatternId
from .taxonomy import CapabilityTag, ObservationTag

LEAN_EXPORT_SCHEMA = "nominal_refactor_advisor.lean_export.v1"


class LeanExportError(ValueError):
    """Raised when a Lean advisor export violates the expected JSON schema."""


def _object(value: object, context: str) -> JsonObject:
    if isinstance(value, Mapping):
        return JsonObject(value)
    raise LeanExportError(f"{context} must be a JSON object")


def _object_items(value: object, context: str) -> tuple[JsonObject, ...]:
    if not isinstance(value, list):
        raise LeanExportError(f"{context} must be a JSON array")
    return tuple(_object(item, f"{context} item") for item in value)


def _string(value: object, context: str) -> str:
    if isinstance(value, str):
        return value
    raise LeanExportError(f"{context} must be a string")


def _required_string(row: JsonObject, key: str) -> str:
    if key not in row:
        raise LeanExportError(f"Lean finding is missing {key!r}")
    return _string(row[key], key)


def _source_location(row: JsonObject) -> SourceLocation:
    if "line" not in row:
        raise LeanExportError("Lean evidence is missing 'line'")
    line = row["line"]
    if not isinstance(line, int) or isinstance(line, bool):
        raise LeanExportError("evidence line must be an integer")
    return SourceLocation(
        _required_string(row, "file_path"),
        line,
        _required_string(row, "symbol"),
    )


def _evidence(row: JsonObject) -> tuple[SourceLocation, ...]:
    if "evidence" not in row:
        raise LeanExportError("Lean finding is missing 'evidence'")
    return tuple(
        _source_location(item)
        for item in _object_items(row["evidence"], "finding evidence")
    )


_LEAN_REPEATED_STRUCTURAL_SIGNATURE_SPEC = high_confidence_spec(
    PatternId.NOMINAL_INTERFACE_WITNESS,
    "Repeated Lean declaration signature should use a semantic abstraction",
    (
        "Exact Lean signature orbits indicate proof declarations are "
        "structurally confusable without a named semantic owner."
    ),
    (
        "named Lean structure, typeclass, theorem schema, or bridge object "
        "that owns the repeated signature"
    ),
    "Lean environment declaration-signature orbit",
    (
        CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        CapabilityTag.PROVENANCE,
        CapabilityTag.NOMINAL_IDENTITY,
    ),
    (ObservationTag.NORMALIZED_AST,),
)


class LeanFindingAdapter(ABC, metaclass=AutoRegisterMeta):
    """Nominal adapter root for one Lean detector family."""

    __registry__: ClassVar[dict[str, type["LeanFindingAdapter"]]] = {}
    __registry_key__ = "detector_id"
    __skip_if_no_key__ = True

    detector_id: ClassVar[str]
    finding_spec: ClassVar[FindingSpec]

    @classmethod
    @abstractmethod
    def build_finding(cls, row: JsonObject) -> RefactorFinding:
        """Convert one Lean finding object into a Python advisor finding."""


class LeanRepeatedStructuralSignatureAdapter(LeanFindingAdapter):
    """Adapter for exact Lean declaration-signature orbits."""

    detector_id = "lean_repeated_structural_signature"
    finding_spec = _LEAN_REPEATED_STRUCTURAL_SIGNATURE_SPEC

    @classmethod
    def build_finding(cls, row: JsonObject) -> RefactorFinding:
        evidence = _evidence(row)
        if len(evidence) < 2:
            raise LeanExportError(
                f"{cls.detector_id!r} requires at least two evidence declarations"
            )
        return cls.finding_spec.build(
            cls.detector_id,
            _required_string(row, "summary"),
            evidence,
        )


def _adapter_for_detector(detector_id: str) -> type[LeanFindingAdapter]:
    try:
        return LeanFindingAdapter.__registry__[detector_id]
    except KeyError as error:
        raise LeanExportError(
            f"Unknown Lean finding detector_id: {detector_id!r}"
        ) from error


def findings_from_lean_export_payload(payload: JsonObject) -> list[RefactorFinding]:
    """Convert a parsed Lean advisor export into standard advisor findings."""

    schema = _required_string(payload, "schema")
    if schema != LEAN_EXPORT_SCHEMA:
        raise LeanExportError(f"Unsupported Lean advisor export schema: {schema!r}")
    if "findings" not in payload:
        raise LeanExportError("Lean export is missing 'findings'")
    findings = []
    for row in _object_items(payload["findings"], "findings"):
        detector_id = _required_string(row, "detector_id")
        findings.append(_adapter_for_detector(detector_id).build_finding(row))
    return sorted(
        findings,
        key=lambda finding: (finding.pattern_id, finding.title, finding.summary),
    )


def findings_from_lean_export_path(path: Path) -> list[RefactorFinding]:
    """Load a Lean advisor export JSON file and return standard findings."""

    payload = _object(json.loads(path.read_text()), "Lean advisor export")
    return findings_from_lean_export_payload(payload)
