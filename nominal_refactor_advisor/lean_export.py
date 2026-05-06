"""Adapter for Lean-side advisor exports.

Lean owns proof-environment extraction; Python owns common advisor reporting.
This module keeps that boundary explicit by converting the Lean JSON schema into
the same ``RefactorFinding`` records emitted by Python detectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta

from .detectors._base import high_confidence_spec
from .models import FindingSpec, RefactorFinding, SourceLocation
from .patterns import PatternId
from .taxonomy import (
    CapabilityTag,
    ObservationTag,
)

LEAN_EXPORT_SCHEMA = "nominal_refactor_advisor.lean_export.v1"


JsonObject = Mapping[str, Any]


class LeanExportError(ValueError):
    """Raised when a Lean advisor export violates the expected JSON schema."""


def _object(value: object, context: str) -> JsonObject:
    if isinstance(value, Mapping):
        return value
    raise LeanExportError(f"{context} must be a JSON object")


def _object_items(value: object, context: str) -> tuple[JsonObject, ...]:
    if not isinstance(value, list):
        raise LeanExportError(f"{context} must be a JSON array")
    return tuple(_object(item, f"{context} item") for item in value)


def _string(value: object, context: str) -> str:
    if isinstance(value, str):
        return value
    raise LeanExportError(f"{context} must be a string")


def _optional_string(row: JsonObject, key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    return _string(value, key)


def _required_string(row: JsonObject, key: str) -> str:
    if key not in row:
        raise LeanExportError(f"Lean finding is missing {key!r}")
    return _string(row[key], key)


def _source_location(row: JsonObject) -> SourceLocation:
    line = row.get("line", 0)
    if not isinstance(line, int):
        raise LeanExportError("evidence line must be an integer")
    return SourceLocation(
        _string(row.get("file_path", "<lean-env>"), "evidence file_path"),
        line,
        _string(row.get("symbol", "<lean-symbol>"), "evidence symbol"),
    )


def _evidence(row: JsonObject) -> tuple[SourceLocation, ...]:
    return tuple(
        _source_location(item)
        for item in _object_items(row.get("evidence", []), "finding evidence")
    )


def _fallback_pattern_id(row: JsonObject) -> PatternId:
    raw_pattern_id = row.get("pattern_id")
    if raw_pattern_id is None:
        return PatternId.AUTHORITATIVE_SCHEMA
    try:
        return PatternId(int(raw_pattern_id))
    except (TypeError, ValueError) as error:
        raise LeanExportError(
            f"Unknown Lean finding pattern_id: {raw_pattern_id!r}"
        ) from error


_NOMINAL_IDENTITY_PROVENANCE_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS = (
    CapabilityTag.NOMINAL_IDENTITY,
    CapabilityTag.PROVENANCE,
    CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
)
_NORMALIZED_AST_OBSERVATION_TAGS = (ObservationTag.NORMALIZED_AST,)
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
    _NOMINAL_IDENTITY_PROVENANCE_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
    _NORMALIZED_AST_OBSERVATION_TAGS,
)


class LeanFindingAdapter(ABC, metaclass=AutoRegisterMeta):
    """Nominal adapter root for one Lean detector family."""

    __registry__: ClassVar[dict[str, type["LeanFindingAdapter"]]] = {}
    __registry_key__ = "detector_id"
    __skip_if_no_key__ = True

    detector_id: ClassVar[str | None] = None
    finding_spec: ClassVar[FindingSpec]

    @classmethod
    @abstractmethod
    def build_finding(cls, row: JsonObject) -> RefactorFinding:
        """Convert one Lean finding object into a Python advisor finding."""


class GenericLeanFindingAdapter(LeanFindingAdapter):
    """Fallback for Lean detectors whose semantics are fully carried in JSON."""

    detector_id = None

    @classmethod
    def build_finding(cls, row: JsonObject) -> RefactorFinding:
        spec = FindingSpec(
            pattern_id=_fallback_pattern_id(row),
            title=_required_string(row, "title"),
            why=_optional_string(row, "why")
            or "The Lean detector emitted a structural refactoring witness.",
            capability_gap=_optional_string(row, "capability_gap")
            or "Lean-side structural witness should be routed through one semantic authority.",
            relation_context=_optional_string(row, "relation_context")
            or "Lean advisor export",
        )
        return spec.build(
            _required_string(row, "detector_id"),
            _required_string(row, "summary"),
            _evidence(row),
            scaffold=_optional_string(row, "scaffold"),
            codemod_patch=_optional_string(row, "codemod_patch"),
        )


class LeanRepeatedStructuralSignatureAdapter(LeanFindingAdapter):
    """Adapter for exact Lean declaration-signature orbits."""

    detector_id = "lean_repeated_structural_signature"
    finding_spec = _LEAN_REPEATED_STRUCTURAL_SIGNATURE_SPEC

    @classmethod
    def build_finding(cls, row: JsonObject) -> RefactorFinding:
        return cls.finding_spec.build(
            _required_string(row, "detector_id"),
            _required_string(row, "summary"),
            _evidence(row),
            title=_optional_string(row, "title"),
            scaffold=_optional_string(row, "scaffold"),
            codemod_patch=_optional_string(row, "codemod_patch"),
        )


def _adapter_for_detector(detector_id: str) -> type[LeanFindingAdapter]:
    return LeanFindingAdapter.__registry__.get(detector_id, GenericLeanFindingAdapter)


def findings_from_lean_export_payload(payload: JsonObject) -> list[RefactorFinding]:
    """Convert a parsed Lean advisor export into standard advisor findings."""

    schema = _required_string(payload, "schema")
    if schema != LEAN_EXPORT_SCHEMA:
        raise LeanExportError(f"Unsupported Lean advisor export schema: {schema!r}")
    findings = []
    for row in _object_items(payload.get("findings", []), "findings"):
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
