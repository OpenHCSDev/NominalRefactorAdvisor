from __future__ import annotations

from abc import ABC
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar

from ..cases_family import CasesFamily
from ..class_member_authority_codemod import (
    FactorParallelMirroredLeafFamilyOperation,
    PromoteExactLeafMethodsToAncestorOperation,
)
from .semantic_mirror import SemanticMirrorFindingRecipeStrategy
from ..codemod_authority_claims import AstTargetAuthorityClaim
from ..codemod_preflight import CodemodOperationPreflightError
from ..codemod_runtime import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    ExecutableRecipeEvaluation,
    FindingRecipeEvaluation,
    FindingRecipeEvaluator,
    FindingRecipePlan,
    FindingRecipeSynthesisRecord,
    RefactorRecipe,
    codemod_plan_from_findings,
)
from ..codemod_selection_context import CodemodSelectorContext
from ..codemod_selector_models import SourceRewriteTarget
from ..collection_algebra import UniqueIdentityIndexAuthority
from ..descent import (
    CasesFold,
    Descent,
    KeyedMirror,
)
from ..detectors._base import IssueDetector
from ..finding_recipe_actions import FindingRecipeActionKey
from ..json_reports import (
    DataclassJsonReport,
    json_report_cached_property,
    json_report_field,
    json_report_property,
)
from ..models import (
    EnvironmentBooleanDriftMetrics,
    RefactorFinding,
    SourceLocation,
)
from ..planner import (
    RefactorExecutionClass,
    RefactorExecutionPlanReport,
    build_refactor_execution_plan,
    build_refactor_execution_plan_from_groups,
)
from ..projection_descent_codemod import TypedMirror
from ..refactor_concepts import (
    ClassFamilyAuthorityConcept,
    RefactorConcept,
)
from ..semantic_descent import (
    FindingDescentCertificateAuthority,
    build_finding_backed_semantic_descent_graph,
)


@dataclass(frozen=True)
class FindingRecipeClassPlan(DataclassJsonReport):
    """One graph-clustered smell class with executable DSL planning context."""

    execution_class: RefactorExecutionClass = json_report_field(included=False)
    finding_plan: FindingRecipePlan = json_report_field(included=False)

    @json_report_property()
    def class_id(self) -> str:
        return self.execution_class.class_id

    @cached_property
    def synthesis_records(self) -> tuple[FindingRecipeSynthesisRecord, ...]:
        finding_ids = frozenset(self.execution_class.finding_ids)
        return tuple(
            record
            for record in self.finding_plan.records
            if record.finding_id in finding_ids
        )

    @json_report_property()
    def document(self) -> CodemodPlanDocument:
        return self.document_from_records(self.synthesis_records)

    @property
    def finding_ids(self) -> tuple[str, ...]:
        return self.execution_class.finding_ids

    @property
    def expected_removed_finding_ids(self) -> tuple[str, ...]:
        return tuple(
            record.finding_id
            for record in self.synthesis_records
            if record.candidate_recipes
        )

    @staticmethod
    def document_from_records(
        records: Iterable[FindingRecipeSynthesisRecord],
    ) -> CodemodPlanDocument:
        recipes = tuple(
            recipe for record in records for recipe in record.candidate_recipes
        )
        return CodemodPlanDocument(recipes=recipes)


@dataclass(frozen=True)
class FindingRecipeClassPlanReport(DataclassJsonReport):
    """Executable plan mode grouped by graph-derived refactor classes."""

    execution_plan: RefactorExecutionPlanReport
    finding_plan: FindingRecipePlan = json_report_field(
        field_name="finding_recipe_plan"
    )

    @json_report_cached_property()
    def classes(self) -> tuple[FindingRecipeClassPlan, ...]:
        return tuple(
            FindingRecipeClassPlan(execution_class, self.finding_plan)
            for execution_class in self.execution_plan.classes
        )

    @classmethod
    def from_findings(
        cls,
        findings: Iterable[RefactorFinding],
        *,
        root: Path,
        context: CodemodSourceSnapshot,
        detector_ids: Iterable[str] = (),
    ) -> FindingRecipeClassPlanReport:
        finding_tuple = tuple(findings)
        detector_id_set = frozenset(detector_ids)
        planning_findings = tuple(
            finding
            for finding in finding_tuple
            if not detector_id_set or finding.detector_id in detector_id_set
        )
        finding_plan = codemod_plan_from_findings(
            planning_findings,
            selector_context=context,
        )
        return cls.from_finding_plan(
            planning_findings,
            root=root,
            finding_plan=finding_plan,
        )

    @classmethod
    def from_finding_plan(
        cls,
        findings: Iterable[RefactorFinding],
        *,
        root: Path,
        finding_plan: FindingRecipePlan,
    ) -> FindingRecipeClassPlanReport:
        """Group a precomputed finding-backed recipe plan by execution class."""

        planning_findings = tuple(findings)
        execution_plan = cls.execution_plan_for_findings(planning_findings, root)
        return cls(
            execution_plan=execution_plan,
            finding_plan=finding_plan,
        )

    @classmethod
    def execution_plan_for_findings(
        cls,
        findings: tuple[RefactorFinding, ...],
        root: Path,
    ) -> RefactorExecutionPlanReport:
        semantic_groups = cls.semantic_descent_finding_groups(findings, root)
        if semantic_groups is None:
            return build_refactor_execution_plan(list(findings), root)
        return build_refactor_execution_plan_from_groups(semantic_groups, root)

    @staticmethod
    def semantic_descent_finding_groups(
        findings: tuple[RefactorFinding, ...],
        root: Path,
    ) -> tuple[tuple[RefactorFinding, ...], ...] | None:
        semantic_detector_ids = IssueDetector.semantic_mirror_detector_ids()
        semantic_findings = tuple(
            finding
            for finding in findings
            if finding.detector_id in semantic_detector_ids
        )
        if not semantic_findings:
            return None
        ordinary_findings = tuple(
            finding
            for finding in findings
            if finding.detector_id not in semantic_detector_ids
        )
        graph = build_finding_backed_semantic_descent_graph(
            semantic_findings,
        )
        certificate_authority = FindingDescentCertificateAuthority(graph)
        grouped: dict[tuple[str, str], list[RefactorFinding]] = defaultdict(list)
        for finding in semantic_findings:
            resolved = certificate_authority.resolved_certificate_for_finding(finding)
            group_key = (
                resolved.authority.name,
                resolved.certificate.missing_derivation_path,
            )
            grouped[group_key].append(finding)
        ordinary_groups = FindingRecipeClassPlanReport.ordinary_finding_groups(
            ordinary_findings,
            root,
        )
        semantic_groups = tuple(
            tuple(group_findings)
            for _group_key, group_findings in sorted(grouped.items())
        )
        return (*semantic_groups, *ordinary_groups)

    @staticmethod
    def ordinary_finding_groups(
        findings: tuple[RefactorFinding, ...],
        root: Path,
    ) -> tuple[tuple[RefactorFinding, ...], ...]:
        if not findings:
            return ()
        findings_by_id = UniqueIdentityIndexAuthority.declarations_by_handle(
            findings,
            lambda finding: finding.stable_id,
        )
        execution_plan = build_refactor_execution_plan(list(findings), root)
        return tuple(
            tuple(
                findings_by_id[finding_id] for finding_id in execution_class.finding_ids
            )
            for execution_class in execution_plan.classes
        )


def codemod_class_plan_from_findings(
    findings: Iterable[RefactorFinding],
    *,
    root: Path,
    selector_context: CodemodSourceSnapshot,
    detector_ids: Iterable[str] = (),
) -> FindingRecipeClassPlanReport:
    """Group executable finding-backed plans by graph-derived refactor class."""

    return FindingRecipeClassPlanReport.from_findings(
        findings,
        root=root,
        context=selector_context,
        detector_ids=detector_ids,
    )


class FindingRecipeSynthesizer(FindingRecipeEvaluator, ABC):
    """Proof-bearing recipe production inherited by detector declarations."""

    @classmethod
    def finding_matches_concept(
        cls,
        finding: RefactorFinding,
        concept_type: type[RefactorConcept],
        selector_context: CodemodSelectorContext,
    ) -> bool:
        """Project one finding through its executable declaration's concept MRO."""

        evaluator = FindingRecipeEvaluator.for_finding(finding)
        if evaluator is None:
            return False
        return issubclass(
            evaluator.evaluation_declaration_for_finding(finding, selector_context),
            concept_type,
        )

    @classmethod
    def findings_for_concept(
        cls,
        findings: Iterable[RefactorFinding],
        concept_type: type[RefactorConcept],
        selector_context: CodemodSelectorContext,
    ) -> tuple[RefactorFinding, ...]:
        """Return findings whose executable declarations inherit one concept."""

        return tuple(
            finding
            for finding in findings
            if cls.finding_matches_concept(
                finding,
                concept_type,
                selector_context,
            )
        )

    @classmethod
    def detector_ids_for_concept(
        cls,
        concept_type: type[RefactorConcept],
    ) -> frozenset[str]:
        """Project detector identities through executable declaration MROs."""

        return frozenset(
            detector_id
            for detector_type in IssueDetector.registered_detector_types()
            for detector_id in (detector_type.effective_detector_id(),)
            if detector_id is not None
            and issubclass(detector_type, cls)
            and issubclass(detector_type, concept_type)
        )

    def executable_evaluation(
        self,
        recipe: RefactorRecipe,
    ) -> ExecutableRecipeEvaluation:
        return ExecutableRecipeEvaluation(
            executable_recipe=recipe,
            evaluation_declaration_type=type(self),
        )


class SelfDeclaredRecipeSynthesizer(FindingRecipeSynthesizer, ABC):
    """A synthesizer whose every evaluation names its own declaration.

    It returns only `rejected_evaluation` and `executable_evaluation`, both of
    which name `type(self)`, so a finding's concept is read off the type with
    no evaluation.
    """

    def evaluation_declaration_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> type[object]:
        del finding, context
        return type(self)


class PrimaryEvidenceActionKeysMixin:
    """Derive one conflict key from a finding's primary source witness."""

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = finding.primary_evidence
        if evidence is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((evidence.file_path, evidence.subject_symbol),),
        )


class FindingEvidenceActionKeysMixin:
    """Derive conflict keys from every source subject carried by a finding."""

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            sorted(
                {
                    (evidence.file_path, evidence.subject_symbol)
                    for evidence in finding.evidence
                }
            ),
        )


class SingleSourcePathFindingMixin:
    @staticmethod
    def source_path(finding: RefactorFinding) -> str | None:
        file_paths = frozenset(evidence.file_path for evidence in finding.evidence)
        if len(file_paths) != 1:
            return None
        return next(iter(file_paths))


class EnvironmentBooleanAuthorityDriftFindingRecipeEvaluator(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeEvaluator,
):
    """Preserve the exact proof gap for environment-boolean drift findings."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        del context
        metrics = finding.metrics
        if not isinstance(metrics, EnvironmentBooleanDriftMetrics):
            return self.rejected_evaluation(
                "environment-boolean drift finding lacks typed drift evidence"
            )
        authority_location = finding.authority_evidence
        authority_symbol = (
            None if authority_location is None else authority_location.symbol
        )
        return self.rejected_evaluation(
            metrics.recipe_rejection_reason(authority_symbol)
        )


class ExactMethodRoleFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """Expose the proved operation while leaving its semantic name explicit."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        del finding, context
        return self.rejected_evaluation(
            "Exact-method role factoring requires an explicit semantic authority "
            "name; author factor_exact_method_role against any evidence method"
        )


class ExactLeafMethodAncestorPromotionFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """Promote exact methods only to a source-proven existing authority."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "closed-family method promotion requires source context"
            )
        authority_location = finding.authority_evidence
        if authority_location is None:
            return self.rejected_evaluation(
                "closed-family method promotion lacks authority evidence"
            )
        try:
            authority_target = context.required_class_target_for_authority_evidence(
                authority_location
            )
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        operation = PromoteExactLeafMethodsToAncestorOperation(
            target=SourceRewriteTarget(target_id=authority_target.target_id),
            rationale="",
        )
        recipe = (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-promote-exact-leaf-methods",
                reason=(
                    "Move the complete exact method set to its proved existing "
                    "nominal authority."
                ),
            )
            .with_authority_claim(AstTargetAuthorityClaim.from_target(authority_target))
            .with_operation(operation)
        )
        return self.executable_evaluation(recipe)


class ParallelMirroredLeafFamilyFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """Factor a currently proved parallel leaf family through MI role axes."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "parallel leaf-family factoring requires source context"
            )
        authority_location = finding.authority_evidence
        if authority_location is None:
            return self.rejected_evaluation(
                "parallel leaf-family finding lacks authority evidence"
            )
        try:
            snapshot = context.execution_snapshot()
            authority_target = snapshot.required_class_target_for_authority_evidence(
                authority_location
            )
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-factor-parallel-leaf-family",
                reason=(
                    "Move exact role behavior to one authority per role and compose "
                    "each domain leaf through MRO."
                ),
            ).with_operation(
                FactorParallelMirroredLeafFamilyOperation(
                    target=SourceRewriteTarget(target_id=authority_target.target_id),
                    rationale="",
                )
            )
        )


class DescentFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ABC,
):
    """Re-prove a descent on the class the finding's authority evidence names.

    ``descent_types`` are end states in preference order; each one's own proof
    admits or refuses the class, and the first admitted becomes the recipe. A
    refusal that lists questions is a handback: the same operation becomes
    executable once its ``names`` answer them.
    """

    descent_types: ClassVar[tuple[type[Descent], ...]]
    recipe_reason: ClassVar[str]

    def target_evidence(
        self, finding: RefactorFinding, authority: SourceLocation
    ) -> SourceLocation:
        """The class the descent rewrites; by default the authority itself."""

        return authority

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        keys = "/".join(descent.operation_key() for descent in self.descent_types)
        if context is None:
            return self.rejected_evaluation(f"{keys} requires source context")
        authority_location = finding.authority_evidence
        if authority_location is None:
            return self.rejected_evaluation(f"{keys} finding lacks authority evidence")
        try:
            authority = context.required_class_target_for_authority_evidence(
                authority_location
            )
            target = context.required_class_target_for_authority_evidence(
                self.target_evidence(finding, authority_location)
            )
        except (CodemodOperationPreflightError, ValueError) as error:
            return self.rejected_evaluation(str(error))
        refusals: list[str] = []
        for descent_type in self.descent_types:
            operation = descent_type(
                target=SourceRewriteTarget(target_id=target.target_id),
                rationale="",
            )
            try:
                operation.synthesis_proof(context.execution_snapshot())
            except (CodemodOperationPreflightError, ValueError) as error:
                refusals.append(f"{descent_type.operation_key()}: {error}")
                continue
            return self.executable_evaluation(
                RefactorRecipe(
                    recipe_id=f"{finding.stable_id}-{descent_type.operation_key()}",
                    reason=self.recipe_reason,
                )
                .with_authority_claim(AstTargetAuthorityClaim.from_target(authority))
                .with_operation(operation)
            )
        return self.rejected_evaluation("; ".join(refusals))


class TypedMirrorFindingRecipeSynthesizer(
    DescentFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """Behavior keyed by a family's own classes descends onto those classes."""

    descent_types = (TypedMirror,)
    recipe_reason = (
        "Move behavior from the external type-keyed projection onto "
        "the nominal hierarchy that already owns its dispatch."
    )

    def target_evidence(
        self, finding: RefactorFinding, authority: SourceLocation
    ) -> SourceLocation:
        """The projection root the finding's first evidence names."""

        if not finding.evidence:
            raise ValueError("typed_mirror finding lacks projection-root evidence")
        return finding.evidence[0]


class KeyedMirrorFindingRecipeSynthesizer(
    DescentFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """An enum keying one map onto a family is the family restated: delete it."""

    descent_types = (KeyedMirror,)
    recipe_reason = (
        "Delete the enum and its map; the class family they restate answers "
        "every lookup and reverse query."
    )


class CasesFindingRecipeSynthesizer(
    DescentFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """An enum branched on outside itself becomes a class family.

    Held by one class and data only, it folds into that holder's family; an
    enum that declares behaviour becomes the family itself.
    """

    descent_types = (CasesFold, CasesFamily)
    recipe_reason = (
        "Members become subclasses and each branch on them becomes one "
        "override per class it decides."
    )


class SemanticMirrorFindingRecipeEvaluator(FindingRecipeEvaluator):
    """Evaluate a declared semantic-mirror finding through its metric contract."""

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        strategy = SemanticMirrorFindingRecipeStrategy.strategy_for(finding.metrics)
        if strategy is None:
            return ()
        return strategy.action_keys_for_finding(finding)

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        strategy = SemanticMirrorFindingRecipeStrategy.strategy_for(finding.metrics)
        if strategy is None:
            return self.rejected_evaluation(
                "semantic mirror metrics have no registered recipe strategy"
            )
        return strategy.evaluate_recipe_for_finding(finding, context)
