"""Capability evidence follows native ABC obligations and C3 member lookup."""

from abc import ABC, abstractmethod
import inspect

import pytest

from nominal_refactor_advisor.codemod import (
    FindingRecipeSynthesizer,
    LiteralDispatchFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.detector_capabilities import (
    DetectorContributionRole,
    NominalContractMemberEvidence,
)
from nominal_refactor_advisor.models import NominalDeclarationIdentity


class Contract(ABC):
    @abstractmethod
    def execute(self):
        raise NotImplementedError


class Implementation(Contract):
    def execute(self):
        return "implementation"


class Reabstracted(Implementation):
    @abstractmethod
    def execute(self):
        raise NotImplementedError


class Alternative(Contract):
    def execute(self):
        return "alternative"


class AbstractFirst(Reabstracted, Alternative):
    pass


class ConcreteFirst(Alternative, Reabstracted):
    pass


@pytest.mark.parametrize("declaration", (Reabstracted, AbstractFirst))
def test_abstract_override_cannot_be_skipped_to_claim_fulfillment(declaration):
    assert inspect.isabstract(declaration)
    with pytest.raises(TypeError):
        declaration()
    with pytest.raises(TypeError, match="does not fulfill"):
        NominalContractMemberEvidence.from_mro(declaration, Contract, "execute")


@pytest.mark.parametrize(
    ("declaration", "owner", "result"),
    (
        (Implementation, Implementation, "implementation"),
        (ConcreteFirst, Alternative, "alternative"),
    ),
)
def test_proved_owner_is_the_member_that_python_executes(declaration, owner, result):
    assert not inspect.isabstract(declaration)
    assert declaration().execute() == result
    evidence = NominalContractMemberEvidence.from_mro(declaration, Contract, "execute")
    assert evidence.implementation == NominalDeclarationIdentity.from_declaration(owner)
    assert declaration.execute is owner.execute


@pytest.mark.parametrize("virtual", (False, True))
def test_structural_or_virtual_membership_is_not_native_derivation(virtual):
    class Foreign(ABC):
        def execute(self):
            return "foreign"

    if virtual:
        Contract.register(Foreign)
    assert issubclass(Foreign, Contract) is virtual
    assert Contract not in Foreign.__mro__
    with pytest.raises(TypeError, match="nominal contract"):
        NominalContractMemberEvidence.from_mro(Foreign, Contract, "execute")


def test_member_must_be_an_obligation_of_the_selected_contract():
    class Extra(Implementation):
        def helper(self):
            return "helper"

    with pytest.raises(TypeError, match="nominal contract"):
        NominalContractMemberEvidence.from_mro(Extra, Contract, "helper")


def test_contribution_includes_inherited_abstract_obligations():
    assert "evaluate_recipe_for_finding" not in vars(FindingRecipeSynthesizer)
    assert "evaluate_recipe_for_finding" in FindingRecipeSynthesizer.__abstractmethods__
    evidence = DetectorContributionRole.RECIPE_SYNTHESIS_CAPABILITY.evidence_for(
        LiteralDispatchFindingRecipeSynthesizer
    )
    assert evidence is not None
    assert {member.member_name for member in evidence.member_evidence} == (
        FindingRecipeSynthesizer.__abstractmethods__
    )
