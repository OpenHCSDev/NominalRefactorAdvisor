"""Complete copied class mappings are proved within one actual execution."""

import ast
from copy import deepcopy
from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    OpenCapturedReference,
)
from nominal_refactor_advisor.manual_registry import DirectManualRegistryComponent
from nominal_refactor_advisor.native_call import CopiedNativeNamespace
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_registry_original_values import authored_runtime
from test_source_distinct_item_stores import execution

BASE = "REGISTRY = {}\nalias = REGISTRY\nclass Alpha: pass\nclass Beta: pass\n"
STORES = "REGISTRY['alpha'] = Alpha\nREGISTRY['beta'] = Beta\n"


def _component(environment):
    return DirectManualRegistryComponent.from_module_anchor(
        environment.module.module, "Alpha"
    )


def _copy(environment):
    (call,) = tuple(
        statement.value
        for statement in environment.module.module.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "observed"
    )
    operation = environment.source.call_operation(SourceByteSpan.require_node(call))
    return CopiedNativeNamespace.from_call(environment, operation)


@pytest.mark.parametrize(
    "source",
    (
        pytest.param(BASE + STORES + "observed = dict(REGISTRY)\n", id="item-stores"),
        pytest.param(
            BASE + STORES + "frozen = dict(alias)\nobserved = dict(frozen)\n",
            id="frozen-copy",
        ),
        pytest.param(
            BASE + STORES + "observed = dict({}, beta=Beta, alpha=Alpha)\n",
            id="unordered-reversed-copy",
        ),
        pytest.param(
            BASE + STORES + "frozen = dict(alias)\nsaved = Alpha\n"
            "Alpha = Beta\nobserved = dict(frozen)\n",
            id="frozen-copy-and-later-class-rebinding",
        ),
    ),
)
def test_complete_copy_authenticates_original_class_producers(source):
    environment = execution(source)
    component = _component(environment)
    copied = _copy(environment)
    copied.require_closed()
    component.require_class_mapping(copied, environment)
    assert copied.environment is environment
    assert copied.initial_names == frozenset(("alpha", "beta"))
    for entry in component.entries:
        value = copied.member(entry.key_value)
        assert value is not None
        entry.require_class_value(value, environment)
        context, definition = value.source_definition()
        assert (
            environment.source_operation(context, definition).node is entry.class_node
        )
        assert value.proves_same_object(entry.captured_class(environment))

    runtime = authored_runtime(source)
    assert set(runtime["observed"]) == {"alpha", "beta"}
    assert runtime["observed"]["alpha"] is runtime["REGISTRY"]["alpha"]
    assert runtime["observed"]["beta"] is runtime["REGISTRY"]["beta"]
    assert runtime["observed"] is not runtime["REGISTRY"]
    assert runtime["observed"]["alpha"].__name__ == "Alpha"
    assert runtime["observed"]["beta"].__name__ == "Beta"


@pytest.mark.parametrize("installed", (0, 1))
def test_early_copy_cannot_claim_keys_installed_after_its_actual_cut(installed):
    first = "REGISTRY['alpha'] = Alpha\n"
    second = "REGISTRY['beta'] = Beta\n"
    source = BASE + (first if installed else "") + "observed = dict(alias)\n"
    source += ("" if installed else first) + second
    environment = execution(source)
    component = _component(environment)
    copied = _copy(environment)
    copied.require_closed()
    assert copied.initial_names == frozenset(("alpha",) if installed else ())
    with pytest.raises(ValueError):
        component.require_class_mapping(copied, environment)
    runtime = authored_runtime(source)
    assert tuple(runtime["observed"]) == (("alpha",) if installed else ())
    assert tuple(runtime["REGISTRY"]) == ("alpha", "beta")


def test_extra_copied_key_is_not_silently_ignored():
    source = BASE + STORES + "observed = dict(REGISTRY, extra=object)\n"
    environment = execution(source)
    copied = _copy(environment)
    copied.require_closed()
    assert copied.initial_names == frozenset(("alpha", "beta", "extra"))
    with pytest.raises(ValueError):
        _component(environment).require_class_mapping(copied, environment)
    runtime = authored_runtime(source)
    assert runtime["observed"]["extra"] is object


def test_complete_key_inventory_does_not_admit_an_unresolved_member():
    source = BASE + STORES + "observed = dict(REGISTRY, alpha=missing)\n"
    environment = execution(source)
    component = _component(environment)
    copied = _copy(environment)
    copied.require_closed()
    assert copied.initial_names == frozenset(("alpha", "beta"))
    value = copied.member("alpha")
    assert isinstance(value, OpenCapturedReference)
    assert value.violation is CapturedReferenceViolation.UNPROVED_BINDING
    with pytest.raises(CapturedReferenceRejection) as member_refusal:
        value.require_closed()
    with pytest.raises(CapturedReferenceRejection) as relation_refusal:
        component.require_class_mapping(copied, environment)
    assert relation_refusal.value.violation is member_refusal.value.violation
    assert relation_refusal.value.__cause__ is member_refusal.value.__cause__

    # The admitted copy namespace does not certify successful argument evaluation
    # or an actual returned dict when one selected value remains unresolved.
    with pytest.raises(NameError, match="missing"):
        authored_runtime(source)


@pytest.mark.parametrize("wrong_value", ("Beta", "object"))
def test_exact_key_coverage_does_not_prove_the_selected_class_value(wrong_value):
    source = BASE + STORES + f"observed = dict(REGISTRY, alpha={wrong_value})\n"
    environment = execution(source)
    component = _component(environment)
    copied = _copy(environment)
    copied.require_closed()
    assert copied.initial_names == frozenset(("alpha", "beta"))
    value = copied.member("alpha")
    assert value is not None
    value.require_closed()
    with pytest.raises(ValueError):
        component.entries[0].require_class_value(value, environment)
    with pytest.raises(ValueError):
        component.require_class_mapping(copied, environment)
    runtime = authored_runtime(source)
    assert runtime["observed"]["alpha"] is not runtime["Alpha"]
    assert runtime["observed"]["alpha"] is (
        runtime["Beta"] if wrong_value == "Beta" else object
    )


def test_equal_source_foreign_module_cannot_supply_copied_class_evidence():
    source = BASE + STORES + "observed = dict(REGISTRY)\n"
    original = execution(source)
    foreign = execution(source)
    copied = _copy(foreign)
    _component(foreign).require_class_mapping(copied, foreign)
    assert original.module.source == foreign.module.source
    assert original.module.module is not foreign.module.module
    with pytest.raises(ValueError, match="source execution"):
        _component(original).require_class_mapping(copied, foreign)


def test_equal_geometry_foreign_class_node_is_not_the_selected_producer():
    environment = execution(BASE + STORES + "observed = dict(REGISTRY)\n")
    component = _component(environment)
    copied = _copy(environment)
    component.require_class_mapping(copied, environment)
    original = component.entries[0]
    foreign = replace(original, class_node=deepcopy(original.class_node))
    assert ast.dump(foreign.class_node, include_attributes=True) == ast.dump(
        original.class_node, include_attributes=True
    )
    with pytest.raises(ValueError, match="unique actual operation"):
        replace(
            component, entries=(foreign, *component.entries[1:])
        ).require_class_mapping(copied, environment)


def test_shared_canonical_source_does_not_merge_separate_creation_executions():
    original = execution(BASE + STORES + "observed = dict(REGISTRY)\n")
    foreign = SourceModuleExecution.from_source(original.source)
    entry = _component(original).entries[0]
    original_value = entry.captured_class(original)
    foreign_value = entry.captured_class(foreign)
    entry.require_class_value(original_value, original)
    entry.require_class_value(foreign_value, foreign)
    assert original.source is foreign.source
    assert original.entry.frame is not foreign.entry.frame
    context, definition = foreign_value.source_definition()
    assert original.source_operation(context, definition).node is entry.class_node
    assert not foreign_value.proves_same_object(original_value)
    with pytest.raises(ValueError):
        entry.require_class_value(foreign_value, original)


def test_noncanonical_copy_record_does_not_inherit_actual_call_admission():
    environment = execution(BASE + STORES + "observed = dict(REGISTRY)\n")
    copied = _copy(environment)
    component = _component(environment)
    component.require_class_mapping(copied, environment)
    forged = replace(copied)
    assert forged.environment is copied.environment
    assert forged.operation is copied.operation
    assert forged is not copied
    with pytest.raises(ValueError, match="canonical admitted call activation"):
        component.require_class_mapping(forged, environment)


def test_numeric_registry_keys_cannot_be_coerced_into_string_mapping_expectations():
    source = (
        "other = {}\nclass Alpha: pass\nclass Beta: pass\n"
        "saved_alpha = Alpha\nsaved_beta = Beta\n"
        "other['1'] = saved_alpha\nother['2'] = saved_beta\n"
        "observed = dict(other)\n"
        "REGISTRY = {1: Alpha, 2: Beta}\n"
    )
    environment = execution(source)
    component = _component(environment)
    copied = _copy(environment)
    copied.require_closed()
    assert tuple(entry.key_value for entry in component.entries) == (1, 2)
    assert copied.initial_names == frozenset(("1", "2"))
    with pytest.raises(ValueError, match="Complete keys do not match"):
        component.require_class_mapping(copied, environment)
    runtime = authored_runtime(source)
    assert set(runtime["REGISTRY"]) == {1, 2}
    assert set(runtime["observed"]) == {"1", "2"}


def test_original_literal_operands_do_not_admit_nonempty_dictionary_execution():
    source = (
        "class Alpha: pass\nclass Beta: pass\n"
        "REGISTRY = {'alpha': Alpha, 'beta': Beta}\n"
        "observed = dict(REGISTRY)\n"
    )
    environment = execution(source)
    component = _component(environment)
    component.require_original_entry_values(environment)
    with pytest.raises(CapturedReferenceRejection) as refusal:
        _copy(environment)
    assert refusal.value.violation is CapturedReferenceViolation.UNPROVED_ACCESS
    assert isinstance(refusal.value.__cause__, ValueError)
    runtime = authored_runtime(source)
    assert runtime["observed"]["alpha"] is runtime["Alpha"]
    assert runtime["observed"]["beta"] is runtime["Beta"]


def test_unproved_dictionary_alias_release_is_not_bypassed_by_a_frozen_copy():
    source = BASE + STORES
    source += "frozen = dict(alias)\nalias = None\nobserved = dict(frozen)\n"
    environment = execution(source)
    component = _component(environment)
    component.require_original_entry_values(environment)
    with pytest.raises(CapturedReferenceRejection) as refusal:
        _copy(environment)
    assert refusal.value.violation is CapturedReferenceViolation.UNPROVED_EFFECTS
    assert str(refusal.value.__cause__) == "Native instance lifetime remains unproved"
    runtime = authored_runtime(source)
    assert runtime["alias"] is None
    assert runtime["observed"]["alpha"] is runtime["Alpha"]
    assert runtime["observed"]["beta"] is runtime["Beta"]
