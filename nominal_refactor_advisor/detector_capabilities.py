"""Declaration-derived refactoring capabilities of registered detectors."""

from __future__ import annotations

from dataclasses import dataclass

from .codemod import FindingRecipeEvaluator, FindingRecipeSynthesizer
from .detectors import (
    IssueDetector,
    SemanticMirrorIssueDetector,
    SsotAuthorityBoundaryDetector,
)
from .json_reports import (
    DataclassJsonReport,
    json_report_field,
    json_report_property,
)
from .models import NominalDeclarationIdentity, RequiredRelationIdentity, SourceLocation
from .patterns import PatternId
from .refactor_concepts import RefactorConcept


def _inherited_declaration_identity(
    detector_type: type[IssueDetector],
    contract_type: type[object],
    declaration_type: type[object],
) -> NominalDeclarationIdentity | None:
    """Project one declared MRO relation without storing a capability mirror."""

    return (
        NominalDeclarationIdentity.from_declaration(declaration_type)
        if issubclass(detector_type, contract_type)
        else None
    )


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

    @json_report_property(omit_none=True)
    def ssot_authority_boundary(self) -> NominalDeclarationIdentity | None:
        return _inherited_declaration_identity(
            self.detector_type,
            SsotAuthorityBoundaryDetector,
            SsotAuthorityBoundaryDetector,
        )

    @json_report_property(omit_none=True)
    def semantic_mirror_contract(self) -> NominalDeclarationIdentity | None:
        return _inherited_declaration_identity(
            self.detector_type,
            SemanticMirrorIssueDetector,
            SemanticMirrorIssueDetector,
        )

    @json_report_property(omit_none=True)
    def direct_recipe_evaluator(self) -> NominalDeclarationIdentity | None:
        return _inherited_declaration_identity(
            self.detector_type,
            FindingRecipeEvaluator,
            self.detector_type,
        )

    @json_report_property(omit_none=True)
    def direct_executable_refactor(self) -> NominalDeclarationIdentity | None:
        return _inherited_declaration_identity(
            self.detector_type,
            FindingRecipeSynthesizer,
            self.detector_type,
        )

    @json_report_property(omit_none=True)
    def direct_refactor_concept(self) -> NominalDeclarationIdentity | None:
        if self.direct_executable_refactor is None:
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

    @json_report_property()
    def required_relation_count(self) -> int:
        return len(self.capabilities)

    @json_report_property()
    def authority_boundary_count(self) -> int:
        return sum(
            capability.ssot_authority_boundary is not None
            for capability in self.capabilities
        )

    @json_report_property()
    def semantic_mirror_count(self) -> int:
        return sum(
            capability.semantic_mirror_contract is not None
            for capability in self.capabilities
        )

    @json_report_property()
    def direct_recipe_evaluator_count(self) -> int:
        return sum(
            capability.direct_recipe_evaluator is not None
            for capability in self.capabilities
        )

    @json_report_property()
    def direct_executable_refactor_count(self) -> int:
        return sum(
            capability.direct_executable_refactor is not None
            for capability in self.capabilities
        )
