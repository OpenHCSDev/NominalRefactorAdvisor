"""Content relations share real slot evidence without inventing dict identities."""

import ast
from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceRejection,
    NamespaceContentsABC,
    NamespaceMemberInventory,
    OpenCapturedReference,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_registry_copied_class_mapping import BASE, STORES, _component, _copy
from test_registry_original_values import authored_runtime
from test_source_distinct_item_stores import execution


def positioned_contents(environment, assignment="tail", *, completed=False):
    (node,) = (
        statement.value
        for statement in environment.module.module.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == assignment
    )
    read = environment.source.value_reads_by_node[node]
    namespace = environment.capture_value(node).dictionary_namespace(
        environment.initial
    )
    prefix = environment.required_prefix(
        read.context, None if completed else read.use.position
    )
    return NamespaceMemberInventory(environment.kernel, namespace, prefix)


def paired_contents():
    environment = execution(
        BASE + STORES + "observed = dict(REGISTRY)\ntail = REGISTRY\n"
    )
    return environment, _copy(environment), positioned_contents(environment)


def test_complete_module_end_registry_needs_no_dictionary_copy_call():
    source = BASE + STORES + "tail = REGISTRY\n"
    environment = execution(source)
    assert not any(
        isinstance(node, ast.Call) for node in ast.walk(environment.module.module)
    )
    contents = positioned_contents(environment, completed=True)
    assert isinstance(contents, NamespaceContentsABC)
    assert contents.prefix.endpoint.position is None
    contents.require_closed()
    _component(environment).require_class_mapping(contents, environment)
    assert contents.names == frozenset(("alpha", "beta"))
    runtime = authored_runtime(source)
    assert runtime["REGISTRY"]["alpha"] is runtime["Alpha"]
    assert runtime["REGISTRY"]["beta"] is runtime["Beta"]


def test_early_and_late_cuts_observe_one_namespace_at_distinct_times():
    environment = execution(BASE + "early = alias\n" + STORES + "tail = REGISTRY\n")
    early = positioned_contents(environment, "early")
    late = positioned_contents(environment)
    assert early.namespace is late.namespace
    assert early.names == frozenset()
    assert early.member("alpha") is None
    assert late.names == frozenset(("alpha", "beta"))
    component = _component(environment)
    with pytest.raises(ValueError):
        component.require_class_mapping(early, environment)
    component.require_class_mapping(late, environment)
    with pytest.raises(ValueError):
        early.require_same_mapping(late)


def test_copy_and_positioned_mapping_share_values_without_claiming_dict_identity():
    environment, copied, positioned = paired_contents()
    assert isinstance(copied, NamespaceContentsABC)
    assert copied is not positioned.namespace
    copied.require_same_mapping(positioned)
    positioned.require_same_mapping(copied)
    _component(environment).require_class_mapping(copied, environment)
    for key in copied.names:
        assert copied.require_member(key).proves_same_object(
            positioned.require_member(key)
        )


def test_keyword_overrides_are_copied_initial_contents_not_parent_contents():
    environment = execution(
        BASE + STORES + "observed = dict({}, beta=Beta, alpha=Alpha)\ntail = observed\n"
    )
    copied = _copy(environment)
    positioned = positioned_contents(environment)
    assert copied.parent.initial_names == frozenset()
    assert copied.names == frozenset(("alpha", "beta"))
    copied.require_same_mapping(positioned)
    _component(environment).require_class_mapping(copied, environment)
    _component(environment).require_class_mapping(positioned, environment)


@pytest.mark.parametrize("receiver", ("REGISTRY", "observed"))
def test_later_writes_do_not_rewrite_the_original_copy_contents(receiver):
    source = (
        BASE
        + STORES
        + "observed = dict(REGISTRY)\n"
        + f"{receiver}['extra'] = object\ntail = {receiver}\n"
    )
    environment = execution(source)
    copied = _copy(environment)
    positioned = positioned_contents(environment)
    assert copied.names == frozenset(("alpha", "beta"))
    assert copied.member("extra") is None
    assert positioned.names == frozenset(("alpha", "beta", "extra"))
    assert positioned.require_member("extra").value is object
    _component(environment).require_class_mapping(copied, environment)
    with pytest.raises(ValueError):
        copied.require_same_mapping(positioned)
    with pytest.raises(ValueError):
        _component(environment).require_class_mapping(positioned, environment)
    runtime = authored_runtime(source)
    assert set(runtime[receiver]) == {"alpha", "beta", "extra"}
    assert ("extra" in runtime["observed"]) is (receiver == "observed")


def test_missing_member_helper_preserves_absence_and_rejects_required_absence():
    _, copied, positioned = paired_contents()
    for contents in (copied, positioned):
        assert contents.member("missing") is None
        with pytest.raises(ValueError):
            contents.require_member("missing")


@pytest.mark.parametrize("key", (1.0, Ellipsis, ("alpha",)))
def test_both_actual_content_owners_reject_nonscalar_queries(key):
    _, copied, positioned = paired_contents()
    for contents in (copied, positioned):
        with pytest.raises(TypeError):
            contents.member(key)


def test_string_subclass_query_does_not_execute_hash_or_equality():
    class HostileString(str):
        def __hash__(self):
            raise AssertionError("query hash must not run")

        def __eq__(self, other):
            raise AssertionError("query equality must not run")

    _, copied, positioned = paired_contents()
    for contents in (copied, positioned):
        with pytest.raises(TypeError):
            contents.member(HostileString("alpha"))


def test_class_relation_rejects_a_foreign_kernel_even_for_same_source_projection():
    environment, copied, positioned = paired_contents()
    foreign = SourceModuleExecution.from_source(environment.source)
    assert foreign.source is environment.source
    assert foreign.kernel is not environment.kernel
    component = _component(environment)
    for contents in (copied, positioned):
        with pytest.raises(ValueError):
            component.require_class_mapping(contents, foreign)


def test_equal_source_from_another_module_cannot_supply_class_edges():
    environment, copied, _ = paired_contents()
    foreign = execution(environment.module.source)
    with pytest.raises(ValueError):
        _component(foreign).require_class_mapping(copied, environment)


def test_same_names_from_separate_class_activations_do_not_prove_same_mapping():
    _, first, _ = paired_contents()
    _, second, _ = paired_contents()
    assert first.names == second.names
    with pytest.raises(ValueError):
        first.require_same_mapping(second)


def test_native_values_can_prove_mapping_across_independent_executions():
    source = "REGISTRY = {}\nREGISTRY['native'] = object\ntail = REGISTRY\n"
    first = positioned_contents(execution(source))
    second = positioned_contents(execution(source))
    assert first.namespace is not second.namespace
    first.require_same_mapping(second)


def test_complete_names_do_not_close_an_unresolved_copied_value():
    environment = execution(
        BASE + STORES + "observed = dict(REGISTRY, alpha=missing)\n"
    )
    copied = _copy(environment)
    copied.require_closed()
    assert copied.names == frozenset(("alpha", "beta"))
    value = copied.member("alpha")
    assert isinstance(value, OpenCapturedReference)
    with pytest.raises(CapturedReferenceRejection) as original:
        value.require_closed()
    with pytest.raises(CapturedReferenceRejection) as relation:
        _component(environment).require_class_mapping(copied, environment)
    assert relation.value.violation is original.value.violation
    with pytest.raises(ValueError):
        copied.require_same_mapping(copied)


@pytest.mark.parametrize(
    "write",
    ("REGISTRY['alpha'] = missing\n", "REGISTRY[missing] = Alpha\n"),
)
def test_unadmitted_late_write_cannot_create_positioned_content_evidence(write):
    environment = execution(BASE + STORES + write + "tail = REGISTRY\n")
    with pytest.raises(ValueError):
        positioned_contents(environment)


def test_direct_member_lookup_validates_namespace_availability_before_slots():
    environment, copied, positioned = paired_contents()
    foreign = SourceModuleExecution.from_source(environment.source)
    forged = replace(positioned, namespace=copied, kernel=foreign.kernel)
    with pytest.raises(ValueError):
        forged.member("alpha")


def test_empty_names_do_not_bypass_content_owner_admission():
    environment = execution(
        "REGISTRY = {}\nobserved = dict(REGISTRY)\ntail = observed\n"
    )
    positioned = positioned_contents(environment)
    foreign = SourceModuleExecution.from_source(environment.source)
    forged = replace(positioned, kernel=foreign.kernel)
    with pytest.raises(ValueError):
        forged.names


def test_equal_copied_record_has_no_canonical_content_admission():
    environment, copied, positioned = paired_contents()
    forged = replace(copied)
    assert forged is not copied
    assert forged.environment is copied.environment
    assert forged.operation is copied.operation
    assert forged.parent is copied.parent
    with pytest.raises(ValueError):
        forged.require_member("alpha")
    with pytest.raises(ValueError):
        forged.names
    with pytest.raises(ValueError):
        forged.require_same_mapping(positioned)
    with pytest.raises(ValueError):
        positioned.require_same_mapping(forged)
    with pytest.raises(ValueError):
        _component(environment).require_class_mapping(forged, environment)


def test_actual_earlier_prefix_cannot_admit_a_later_copy_namespace():
    environment = execution(
        BASE + STORES + "early = alias\nobserved = dict(REGISTRY)\ntail = observed\n"
    )
    early = positioned_contents(environment, "early")
    later = positioned_contents(environment)
    early.require_closed()
    later.require_closed()
    early.require_same_mapping(later)
    assert early.namespace is not later.namespace
    forged = replace(later, prefix=early.prefix)
    with pytest.raises(ValueError):
        forged.names
    with pytest.raises(ValueError):
        forged.member("alpha")
    with pytest.raises(ValueError):
        later.require_same_mapping(forged)
