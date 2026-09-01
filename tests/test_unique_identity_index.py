from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.collection_algebra import (
    IdentityHandleCollisionError,
    UniqueIdentityIndexAuthority,
)
from nominal_refactor_advisor.impact_ranking import RefactorImpactRankingRequest
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


def test_source_index_rejects_forced_target_handle_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "module.py").write_text(
        "def alpha():\n" "    return 1\n" "\n" "def beta():\n" "    return 2\n",
        encoding="utf-8",
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
    (tmp_path / "alpha.py").write_text("ALPHA = 1\n", encoding="utf-8")
    (tmp_path / "beta.py").write_text("BETA = 2\n", encoding="utf-8")
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


def test_impact_ranking_rejects_forced_finding_handle_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
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

    request = RefactorImpactRankingRequest(
        findings=findings, source_index=SourceIndex()
    )
    with pytest.raises(IdentityHandleCollisionError):
        _ = request._findings_by_id
