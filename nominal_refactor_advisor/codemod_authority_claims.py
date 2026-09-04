"""Authority-claim payload, proof-resolution, and preflight declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .codemod_payload import (
    CodemodPayloadRecord,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_preflight import CodemodOperationPreflightReport
from .json_reports import DataclassJsonReport
from .models import (
    RefactorFinding,
    SourceLocation,
)
from .patterns import PatternId
from .semantic_descent import (
    AuthorityClaim,
    AuthorityClaimResolution,
    AuthorityProofEdge,
    AuthorityProofEdgeKind,
    SemanticAuthorityKind,
)
from .source_index import (
    AstTargetDigest,
    SourceIndex,
)
from .taxonomy import (
    CertificationLevel,
    ConfidenceLevel,
)


class AuthorityClaimPayload:
    """Payload field ownership for recipe authority claims."""

    field_name: ClassVar[str] = "authority_claims"


class CodemodPlanEvidenceLocation:
    """Synthetic source location owner for codemod-plan findings."""

    file_path: ClassVar[str] = "<codemod-plan>"
    line: ClassVar[int] = 0

    @classmethod
    def authority_claim(cls, recipe_id: str, claimed_symbol: str) -> SourceLocation:
        return SourceLocation(
            cls.file_path,
            cls.line,
            f"{recipe_id}:{claimed_symbol}",
        )


class AuthorityClaimPreflightFinding:
    """Advisor finding projection for failed authority-claim preflight."""

    detector_id: ClassVar[str] = "unresolved_authority_claim"

    @classmethod
    def unresolved_resolution(
        cls,
        recipe_id: str,
        resolution: AuthorityClaimResolution,
    ) -> RefactorFinding:
        discovery = resolution.discovery_required
        reason = (
            discovery.reason
            if discovery is not None
            else "authority claim is not resolved or declared"
        )
        return cls._finding(
            recipe_id=recipe_id,
            claimed_symbol=resolution.claim.claimed_symbol,
            summary=(
                f"Recipe `{recipe_id}` claims authority "
                f"`{resolution.claim.claimed_symbol}` but preflight status is "
                f"`{resolution.status.value}`: {reason}."
            ),
            evidence_symbol=resolution.claim.claimed_symbol,
        )

    @classmethod
    def _finding(
        cls,
        *,
        recipe_id: str,
        claimed_symbol: str,
        summary: str,
        evidence_symbol: str,
    ) -> RefactorFinding:
        return RefactorFinding(
            pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
            title="Unresolved authority claim",
            why=(
                "Authority-routing refactor plans must prove the named authority "
                "against the source graph or explicitly declare the new boundary."
            ),
            capability_gap=(
                "proof-carrying authority claim resolved by source-index target or "
                "declare_authority operation"
            ),
            relation_context=(
                "codemod plan asserts an authority boundary without an actionable "
                "source proof edge"
            ),
            confidence=ConfidenceLevel.HIGH,
            certification=CertificationLevel.CERTIFIED,
            detector_id=cls.detector_id,
            summary=summary,
            evidence=(
                CodemodPlanEvidenceLocation.authority_claim(
                    recipe_id,
                    evidence_symbol,
                ),
            ),
        )


@dataclass(frozen=True)
class SourceCreationConflictPreflightDetail(CodemodPayloadRecord):
    """Conflicting source paths retained as typed failed-preflight evidence."""

    duplicate_source_paths: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )
    existing_source_paths: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )


@dataclass(frozen=True)
class AuthorityClaimContextPreflightDetail(CodemodPayloadRecord):
    """Recipe identity for an authority check lacking source context."""

    recipe_id: str = codemod_payload_field(RequiredStringPayloadValueCodec())


@dataclass(frozen=True)
class AuthorityClaimDeclarationPreflightDetail(DataclassJsonReport):
    """Typed nesting of a failed declaration-derived authority check."""

    recipe_id: str
    declaration_preflight: CodemodOperationPreflightReport


@dataclass(frozen=True)
class AuthorityClaimResolutionPreflightDetail(DataclassJsonReport):
    """Typed authority resolutions and findings retained until JSON emission."""

    recipe_id: str
    resolutions: tuple[AuthorityClaimResolution, ...]
    findings: tuple[RefactorFinding, ...]


class AstTargetAuthorityClaim:
    """Authority claim derived from a concrete source-index AST target."""

    @staticmethod
    def from_target(
        target: AstTargetDigest,
        *,
        authority_kind: SemanticAuthorityKind | None = None,
    ) -> AuthorityClaim:
        return AuthorityClaim(
            claimed_symbol=target.name,
            authority_kind=authority_kind,
            file_path=target.file_path,
            qualname=target.qualname,
            authority_id=target.target_id,
        )


@dataclass(frozen=True)
class AuthorityClaimSourceIndexResolver:
    """Resolve codemod authority claims against current source-index targets."""

    source_index: SourceIndex
    declared_claims: tuple[AuthorityClaim, ...] = ()

    def resolve(self, claim: AuthorityClaim) -> AuthorityClaimResolution:
        candidates = self._candidate_targets(claim)
        searched_symbols = claim.searched_symbols
        if candidates:
            return AuthorityClaimResolution.from_proof_edges(
                claim,
                tuple(
                    AuthorityProofEdge(
                        edge_kind=AuthorityProofEdgeKind.SOURCE_INDEX_TARGET,
                        authority_id=target.target_id,
                        authority_kind=claim.authority_kind,
                        file_path=target.file_path,
                        line=target.line,
                        symbol=target.qualname,
                        detail="claim matched source-index AST target",
                    )
                    for target in candidates
                ),
                searched_symbols=searched_symbols,
                ambiguity_reason=(
                    "multiple source-index targets match the authority claim"
                ),
            )
        if any(
            claim.matches_declared_claim(declared_claim)
            for declared_claim in self.declared_claims
        ):
            return AuthorityClaimResolution.declared(
                claim,
                detail="recipe operation declares this authority boundary",
            )
        return AuthorityClaimResolution.unresolved(
            claim,
            searched_symbols=searched_symbols,
            reason="no source-index target or declaring operation matched the claim",
        )

    def _candidate_targets(self, claim: AuthorityClaim) -> tuple[AstTargetDigest, ...]:
        if claim.authority_id:
            target = self.source_index.target_by_id.get(claim.authority_id)
            if target is None:
                return ()
            return (
                (target,)
                if claim.matches_source_identity(
                    authority_id=target.target_id,
                    name=target.name,
                    file_path=target.file_path,
                    qualname=target.qualname,
                )
                else ()
            )
        symbols = claim.searched_symbols
        indexed_candidates = {
            target.target_id: target
            for symbol in symbols
            for target in self.source_index.targets_matching_symbol(symbol)
        }
        return tuple(
            target
            for target in indexed_candidates.values()
            if not target.is_module
            and claim.matches_source_identity(
                authority_id=target.target_id,
                name=target.name,
                file_path=target.file_path,
                qualname=target.qualname,
            )
        )
