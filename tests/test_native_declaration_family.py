"""Native family selection preserves object identity and nominal refinement."""

from abc import abstractmethod
from typing import ClassVar

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    NativeTypePremise,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_call import NativeCallAuthority, NativeGlobalsCall
from nominal_refactor_advisor.native_declarations import (
    NativeDeclaration,
    NativeDeclarationFamily,
)
from nominal_refactor_advisor.semantic_match import loaded_concrete_nominal_descendants
from nominal_refactor_advisor.source_execution import SourceObjectConstruction
from test_source_function_result import execution


class LocalFamily(NativeDeclarationFamily):
    @abstractmethod
    def required_protocol(self):
        raise NotImplementedError


def test_native_call_inherits_selection_without_extending_source_object_protocols():
    assert issubclass(NativeCallAuthority, NativeDeclarationFamily)
    assert not issubclass(SourceObjectConstruction, NativeDeclarationFamily)
    assert "select_from_capture" not in vars(NativeCallAuthority)
    environment = execution("held = globals()\n")
    context, invocation = environment.source_call(
        environment.module.module.body[0].value
    )
    selected = environment.call_authority(context, invocation)
    assert type(selected) is NativeGlobalsCall
    assert selected.operation is environment.source_operation(context, invocation)
    assert (
        NativeCallAuthority.select_from_capture(CapturedNativeObject(globals))
        is NativeGlobalsCall
    )


def test_actual_initial_object_selects_its_only_declared_family_member():
    class Family(LocalFamily):
        pass

    class Sequence(Family):
        native_declarations: ClassVar = (NativeDeclaration(list),)

        def required_protocol(self):
            pass

    class Mapping(Family):
        native_declarations: ClassVar = (NativeDeclaration(dict),)

        def required_protocol(self):
            pass

    alias = list
    assert Family.select_from_capture(CapturedNativeObject(alias)) is Sequence
    assert Family.select_from_capture(CapturedNativeObject(dict)) is Mapping


def test_duplicate_native_aliases_are_one_object_not_two_protocol_matches():
    class Family(LocalFamily):
        pass

    class Error(Family):
        native_declarations: ClassVar = (
            NativeDeclaration(OSError),
            NativeDeclaration(IOError),
        )

        def required_protocol(self):
            pass

    assert IOError is OSError
    assert Family.select_from_capture(CapturedNativeObject(IOError)) is Error


def test_same_qualified_name_cannot_substitute_for_native_object_identity():
    class Family(LocalFamily):
        pass

    class Sequence(Family):
        native_declarations: ClassVar = (NativeDeclaration(list),)

        def required_protocol(self):
            pass

    counterfeit = type("list", (), {"__module__": "builtins"})
    assert (
        NativeDeclaration(counterfeit).qualified_name
        == NativeDeclaration(list).qualified_name
    )
    assert NativeDeclaration(counterfeit) != NativeDeclaration(list)
    with pytest.raises(ValueError, match="required native declaration"):
        Family.select_from_capture(CapturedNativeObject(counterfeit))


@pytest.mark.parametrize(
    "violation",
    (
        CapturedReferenceViolation.UNADMITTED_IMPORT,
        CapturedReferenceViolation.CYCLIC_BINDING,
        CapturedReferenceViolation.UNPROVED_EFFECTS,
    ),
)
def test_original_open_capture_evidence_and_cause_survive_selection(violation):
    class Family(LocalFamily):
        pass

    class Sequence(Family):
        native_declarations: ClassVar = (NativeDeclaration(list),)

        def required_protocol(self):
            pass

    cause = ValueError("actual prior execution obligation remains unproved")
    evidence = OpenCapturedReference(violation, cause=cause)
    with pytest.raises(CapturedReferenceRejection) as caught:
        Family.select_from_capture(evidence)
    assert caught.value.evidence is evidence
    assert caught.value.violation is violation
    assert caught.value.__cause__ is cause


def test_exact_native_type_without_object_identity_is_not_a_native_declaration():
    class Family(LocalFamily):
        pass

    class Sequence(Family):
        native_declarations: ClassVar = (NativeDeclaration(list),)

        def required_protocol(self):
            pass

    premise = NativeTypePremise(list)
    premise.require_closed()
    with pytest.raises(ValueError):
        Family.select_from_capture(premise)


def test_most_specific_declared_subclass_refines_the_matching_parent():
    class Family(LocalFamily):
        pass

    class General(Family):
        native_declarations: ClassVar = (NativeDeclaration(list),)

        def required_protocol(self):
            pass

    class Refined(General):
        pass

    assert loaded_concrete_nominal_descendants(Family) == (General, Refined)
    assert Family.select_from_capture(CapturedNativeObject(list)) is Refined


def test_diamond_is_selected_once_from_actual_mro_not_duplicate_paths():
    class Family(LocalFamily):
        pass

    class General(Family):
        native_declarations: ClassVar = (NativeDeclaration(list),)

        def required_protocol(self):
            pass

    class Left(General):
        pass

    class Right(General):
        pass

    class Joined(Left, Right):
        pass

    assert loaded_concrete_nominal_descendants(Family).count(Joined) == 1
    assert Family.select_from_capture(CapturedNativeObject(list)) is Joined


@pytest.mark.parametrize("reverse_bases", (False, True))
def test_incomparable_matching_protocols_remain_ambiguous(reverse_bases):
    class Family(LocalFamily):
        pass

    class First(Family):
        native_declarations: ClassVar = (NativeDeclaration(list),)

        def required_protocol(self):
            pass

    class Second(Family):
        native_declarations: ClassVar = (NativeDeclaration(list),)

        def required_protocol(self):
            pass

    class CombinedFamily(*((First, Second) if reverse_bases else (Second, First))):
        @abstractmethod
        def additional_obligation(self):
            raise NotImplementedError

    # An abstract descendant cannot resolve the two live concrete authorities.
    assert CombinedFamily not in loaded_concrete_nominal_descendants(Family)
    with pytest.raises(ValueError, match="most-specific"):
        Family.select_from_capture(CapturedNativeObject(list))


def test_abstract_matching_descendant_does_not_override_concrete_parent():
    class Family(LocalFamily):
        pass

    class General(Family):
        native_declarations: ClassVar = (NativeDeclaration(list),)

        def required_protocol(self):
            pass

    class Unfinished(General):
        @abstractmethod
        def missing_obligation(self):
            raise NotImplementedError

    assert Family.select_from_capture(CapturedNativeObject(list)) is General


def test_only_abstract_matches_do_not_admit_an_undeclared_concrete_protocol():
    class Family(LocalFamily):
        pass

    class Unfinished(Family):
        native_declarations: ClassVar = (NativeDeclaration(list),)

    assert loaded_concrete_nominal_descendants(Family) == ()
    with pytest.raises(ValueError, match="required native declaration"):
        Family.select_from_capture(CapturedNativeObject(list))


def test_unrelated_families_with_same_native_value_remain_isolated():
    class Left(LocalFamily):
        pass

    class Right(LocalFamily):
        pass

    class LeftProtocol(Left):
        native_declarations: ClassVar = (NativeDeclaration(list),)

        def required_protocol(self):
            pass

    class RightProtocol(Right):
        native_declarations: ClassVar = (NativeDeclaration(list),)

        def required_protocol(self):
            pass

    captured = CapturedNativeObject(list)
    assert Left.select_from_capture(captured) is LeftProtocol
    assert Right.select_from_capture(captured) is RightProtocol
