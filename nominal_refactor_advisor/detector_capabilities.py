"""Declaration-derived refactoring capabilities of registered detectors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache
import inspect

from .codemod import FindingRecipeSynthesizer
from .codemod_runtime import FindingRecipeEvaluator
from .detectors import (
    IssueDetector,
    SemanticMirrorIssueDetector,
    SsotAuthorityBoundaryDetector,
)
from .json_reports import (
    DataclassJsonReport,
    json_report_cached_property,
    json_report_field,
    json_report_property,
)
from .models import NominalDeclarationIdentity, RequiredRelationIdentity, SourceLocation
from .patterns import PatternId
from .refactor_concepts import RefactorConcept


class DetectorContributionRole(StrEnum):
    """Closed detector contributions derived from nominal execution contracts."""

    REQUIRED_RELATION_OBSERVATION = (
        "required_relation_observation",
        IssueDetector,
        "Executes one declaration-owned required-relation observation.",
    )
    AUTHORITY_BOUNDARY_EVIDENCE = (
        "authority_boundary_evidence",
        SsotAuthorityBoundaryDetector,
        "Requires projection evidence at an SSOT boundary; authority may remain unknown.",
    )
    SEMANTIC_MIRROR_EVIDENCE = (
        "semantic_mirror_evidence",
        SemanticMirrorIssueDetector,
        "Identifies a semantic mirror while preserving unknown authority.",
    )
    RECIPE_EVALUATION_CAPABILITY = (
        "recipe_evaluation_capability",
        FindingRecipeEvaluator,
        "Can prove or reject a finding-backed recipe from current evidence.",
    )
    RECIPE_SYNTHESIS_CAPABILITY = (
        "recipe_synthesis_capability",
        FindingRecipeSynthesizer,
        "Can produce an executable recipe after runtime proof succeeds; membership alone is not success evidence.",
    )

    def __new__(
        cls,
        value: str,
        contract_type: type[object],
        description: str,
    ) -> "DetectorContributionRole":
        member = str.__new__(cls, value)
        member._value_ = value
        member._contract_type = contract_type
        member._description = description
        return member

    @property
    def contract_type(self) -> type[object]:
        return self._contract_type

    @property
    def description(self) -> str:
        return self._description

    def applies_to(self, detector_type: type[IssueDetector]) -> bool:
        return issubclass(detector_type, self.contract_type)

    def evidence_for(
        self,
        detector_type: type[IssueDetector],
    ) -> "DetectorContributionEvidence | None":
        """Recover this contribution and every abstract slot it fulfills."""

        if not self.applies_to(detector_type):
            return None
        abstract_member_names = tuple(
            member_name
            for member_name, member in vars(self.contract_type).items()
            if getattr(member, "__isabstractmethod__", False)
        )
        return DetectorContributionEvidence(
            role=self,
            contract=NominalDeclarationIdentity.from_declaration(self.contract_type),
            mro_resolution_path=tuple(
                NominalDeclarationIdentity.from_declaration(candidate)
                for candidate in detector_type.__mro__[
                    : detector_type.__mro__.index(self.contract_type) + 1
                ]
            ),
            member_evidence=tuple(
                NominalContractMemberEvidence.from_mro(
                    detector_type,
                    self.contract_type,
                    member_name,
                )
                for member_name in sorted(abstract_member_names)
            ),
        )


@dataclass(frozen=True)
class NominalContractMemberEvidence(DataclassJsonReport):
    """MRO proof that one nominal contract slot has an implementation owner."""

    member_name: str
    requirement: NominalDeclarationIdentity
    implementation: NominalDeclarationIdentity
    implementation_source: SourceLocation

    @classmethod
    def from_mro(
        cls,
        declaration_type: type[object],
        requirement_type: type[object],
        member_name: str,
    ) -> "NominalContractMemberEvidence":
        implementation_type = next(
            (
                candidate
                for candidate in declaration_type.__mro__
                if member_name in vars(candidate)
                and not getattr(
                    vars(candidate)[member_name],
                    "__isabstractmethod__",
                    False,
                )
            ),
            None,
        )
        if implementation_type is None:
            raise TypeError(
                f"{declaration_type.__qualname__} does not fulfill abstract "
                f"member {member_name!r} through its MRO"
            )
        return cls(
            member_name=member_name,
            requirement=NominalDeclarationIdentity.from_declaration(requirement_type),
            implementation=NominalDeclarationIdentity.from_declaration(
                implementation_type
            ),
            implementation_source=_declaration_member_source(
                implementation_type,
                member_name,
            ),
        )


@cache
def _declaration_member_source(
    declaration_type: type[object],
    member_name: str,
) -> SourceLocation:
    """Recover one physical member source once for every composed leaf."""

    member = vars(declaration_type)[member_name]
    if isinstance(member, (classmethod, staticmethod)):
        member = member.__func__
    source_path = inspect.getsourcefile(member)
    if source_path is None:
        raise TypeError(
            f"Cannot recover source for {declaration_type.__qualname__}.{member_name}"
        )
    _source_lines, first_line = inspect.getsourcelines(member)
    return SourceLocation(
        source_path,
        first_line,
        f"{declaration_type.__qualname__}.{member_name}",
    )


@dataclass(frozen=True)
class DetectorContributionEvidence(DataclassJsonReport):
    """One MRO-derived detector contribution and its contract fulfillment."""

    role: DetectorContributionRole
    contract: NominalDeclarationIdentity
    mro_resolution_path: tuple[NominalDeclarationIdentity, ...]
    member_evidence: tuple[NominalContractMemberEvidence, ...]


@dataclass(frozen=True)
class DetectorContributionSummary(DataclassJsonReport):
    """One role declaration and its derived detector population."""

    role: DetectorContributionRole
    description: str
    contract: NominalDeclarationIdentity
    detector_count: int


@dataclass(frozen=True)
class DetectorRefactorCapability(DataclassJsonReport):
    """MRO-derived capabilities of one detector leaf, not execution evidence."""

    detector_type: type[IssueDetector] = json_report_field(included=False)

    @json_report_property()
    def detector(self) -> NominalDeclarationIdentity:
        return NominalDeclarationIdentity.from_declaration(self.detector_type)

    @json_report_property()
    def detector_id(self) -> str:
        detector_id = self.detector_type.effective_detector_id()
        if detector_id is None:
            raise TypeError(
                f"{self.detector_type.__name__} is not a concrete detector declaration"
            )
        return detector_id

    @json_report_property()
    def required_relation(self) -> RequiredRelationIdentity:
        declaration = self.detector_type.required_relation_declaration_type()
        return declaration.required_relation_identity()

    @json_report_property()
    def required_relation_source(self) -> SourceLocation:
        return self.detector_type.required_relation_source()

    @json_report_property()
    def required_relation_pattern(self) -> PatternId:
        return self.detector_type.required_relation_pattern_id()

    @json_report_cached_property()
    def contributions(self) -> tuple[DetectorContributionEvidence, ...]:
        return tuple(
            evidence
            for role in DetectorContributionRole
            for evidence in (role.evidence_for(self.detector_type),)
            if evidence is not None
        )

    def contribution_for(
        self,
        role: DetectorContributionRole,
    ) -> DetectorContributionEvidence | None:
        return next(
            (
                contribution
                for contribution in self.contributions
                if contribution.role is role
            ),
            None,
        )

    @json_report_property(omit_none=True)
    def recipe_synthesis_concept(self) -> NominalDeclarationIdentity | None:
        if not DetectorContributionRole.RECIPE_SYNTHESIS_CAPABILITY.applies_to(
            self.detector_type
        ):
            return None
        declaration = RefactorConcept.leaf_concept_for_declaration(self.detector_type)
        return NominalDeclarationIdentity.from_declaration(declaration)


@dataclass(frozen=True)
class DetectorRefactorCapabilityReport(DataclassJsonReport):
    """Complete capability inventory derived from registered detector leaves."""

    capabilities: tuple[DetectorRefactorCapability, ...]

    @classmethod
    def from_registered_detectors(cls) -> "DetectorRefactorCapabilityReport":
        return cls(
            tuple(
                DetectorRefactorCapability(detector_type)
                for detector_type in IssueDetector.registered_detector_types()
            )
        )

    def contribution_count(self, role: DetectorContributionRole) -> int:
        return sum(
            capability.contribution_for(role) is not None
            for capability in self.capabilities
        )

    @json_report_cached_property()
    def contribution_summary(self) -> tuple[DetectorContributionSummary, ...]:
        return tuple(
            DetectorContributionSummary(
                role=role,
                description=role.description,
                contract=NominalDeclarationIdentity.from_declaration(
                    role.contract_type
                ),
                detector_count=self.contribution_count(role),
            )
            for role in DetectorContributionRole
        )
