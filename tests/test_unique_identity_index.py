from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.collection_algebra import (
    IdentityHandleCollisionError,
    UniqueIdentityIndexAuthority,
)
from nominal_refactor_advisor.structural_overlap import StructuralOverlapRequest
from nominal_refactor_advisor.models import FindingSpec, RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.source_index import (
    EvidenceDigest,
    SourceIndex,
    StableIdAuthority,
    build_source_index,
)


@dataclass(frozen=True)
class _Declaration:
    name: str


def test_unique_identity_index_accepts_only_equal_duplicate_declarations() -> None:
    index = UniqueIdentityIndexAuthority[str, _Declaration, str]()
    index.add("same", _Declaration("alpha"), "first projection")
    index.add("same", _Declaration("alpha"), "second projection")

    assert index.values_by_handle() == {"same": "first projection"}

    with pytest.raises(IdentityHandleCollisionError) as error:
        index.add("same", _Declaration("beta"), "redirected projection")

    assert error.value.handle == "same"
    assert error.value.existing_declaration == _Declaration("alpha")
    assert error.value.colliding_declaration == _Declaration("beta")


def test_unambiguous_identity_projection_excludes_every_repeated_handle() -> None:
    declarations = (
        _Declaration("unique"),
        _Declaration("equal"),
        _Declaration("equal"),
        _Declaration("conflict"),
        _Declaration("conflict-other"),
    )

    assert UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
        declarations,
        lambda declaration: (
            "conflict" if declaration.name.startswith("conflict") else declaration.name
        ),
    ) == {"unique": _Declaration("unique")}


def test_identity_multiplicity_projection_exposes_repeated_handles_once() -> None:
    declarations = (
        _Declaration("unique"),
        _Declaration("equal"),
        _Declaration("equal"),
        _Declaration("conflict"),
        _Declaration("conflict-other"),
    )

    projection = UniqueIdentityIndexAuthority.declaration_multiplicity_by_handle(
        declarations,
        lambda declaration: (
            "conflict" if declaration.name.startswith("conflict") else declaration.name
        ),
    )

    assert projection.unambiguous_declarations_by_handle == {
        "unique": _Declaration("unique")
    }
    assert projection.ambiguous_handles == frozenset(("conflict", "equal"))


def test_source_index_rejects_forced_target_handle_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "module.py").write_text(
        "def alpha():\n    return 1\n\ndef beta():\n    return 2\n",
        encoding="utf-8", newline="",
    )
    monkeypatch.setattr(
        StableIdAuthority,
        "ast_target_id",
        lambda self, **kwargs: "forced-collision",
    )

    with pytest.raises(IdentityHandleCollisionError) as error:
        build_source_index(parse_python_modules(tmp_path), ())

    assert error.value.handle == "forced-collision"


def test_source_index_rejects_forced_file_handle_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "alpha.py").write_text("ALPHA = 1\n", encoding="utf-8", newline="")
    (tmp_path / "beta.py").write_text("BETA = 2\n", encoding="utf-8", newline="")
    monkeypatch.setattr(
        StableIdAuthority,
        "file_id",
        lambda self, file_path: "forced-collision",
    )

    with pytest.raises(IdentityHandleCollisionError):
        build_source_index(parse_python_modules(tmp_path), ())


def test_source_index_rejects_unequal_evidence_with_same_handle() -> None:
    source_index = SourceIndex(
        evidence=(
            EvidenceDigest(
                evidence_id="forced-collision",
                file_id=None,
                file_path="alpha.py",
                line=1,
                symbol="alpha",
                finding_ids=("finding-alpha",),
            ),
            EvidenceDigest(
                evidence_id="forced-collision",
                file_id=None,
                file_path="beta.py",
                line=2,
                symbol="beta",
                finding_ids=("finding-beta",),
            ),
        )
    )

    with pytest.raises(IdentityHandleCollisionError):
        _ = source_index.evidence_by_id


def test_structural_overlap_rejects_forced_finding_handle_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = FindingSpec(
        pattern_id=PatternId.SHARED_ALGORITHM_AUTHORITY,
        title="Shared title",
        why="Shared rationale",
        capability_gap="Shared capability gap",
        relation_context="Shared relation",
    )
    findings = (
        spec.build(
            "alpha-detector",
            "alpha summary",
            (SourceLocation("alpha.py", 1, "alpha"),),
        ),
        spec.build(
            "beta-detector",
            "beta summary",
            (SourceLocation("beta.py", 2, "beta"),),
        ),
    )
    monkeypatch.setattr(
        RefactorFinding,
        "stable_id",
        property(lambda self: "forced-collision"),
    )

    with pytest.raises(IdentityHandleCollisionError):
        build_source_index((), findings)

    request = StructuralOverlapRequest(findings=findings, source_index=SourceIndex())
    with pytest.raises(IdentityHandleCollisionError):
        _ = request._findings_by_id
