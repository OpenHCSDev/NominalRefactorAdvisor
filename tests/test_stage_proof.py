"""Which finding classes a terminal state grew that its stages answer for."""

from nominal_refactor_advisor.refactor_concepts import (
    ClassBodyCompositionConcept,
    HiddenImportPromotionConcept,
    RefactorConcept,
)
from nominal_refactor_advisor.stage_proof import ImportScopeStages, StageProof


def test_a_stage_that_may_write_what_a_finding_cites_answers_for_every_grown_class():
    grown = ("restated_identity", "constant_subtype_answer")
    assert ClassBodyCompositionConcept.stage_proof.unjustified(grown) == grown


def test_import_scope_stages_reveal_findings_instead_of_owing_them():
    """A promotion binds the same objects earlier: what it grows was there before."""
    assert HiddenImportPromotionConcept.stage_proof is ImportScopeStages
    assert ImportScopeStages.unjustified(("restated_identity",)) == ()


def test_only_import_scope_concepts_reveal():
    revealing = {
        concept.__name__
        for concept in RefactorConcept.declaration_types()
        if issubclass(concept.stage_proof, StageProof) and not concept.stage_proof.unjustified((object(),))
    }
    assert revealing == {"HiddenImportPromotionConcept"}
