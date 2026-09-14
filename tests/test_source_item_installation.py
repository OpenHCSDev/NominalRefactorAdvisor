"""Installed values require actual native storage evidence, not prefix closure."""

from dataclasses import dataclass, replace
import sys

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceEffectsABC,
    CapturedReferenceKernel,
    CapturedReferenceViolation,
    CapturedSlotQuery,
    InstalledValueSlotQuery,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_call import CopiedNativeNamespace
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import CompactBindingVisit, CompactItemTarget
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_source_distinct_item_stores import (
    BASE,
    assert_closed_stores,
    execution,
    item_statements,
)


def tail_query(environment, key="alpha"):
    """Derive the observer's actual namespace and cut from the original read."""
    node = environment.module.module.body[-1].value
    read = environment.source.value_reads_by_node[node]
    namespace = environment.capture_value(node).dictionary_namespace(
        environment.initial
    )
    prefix = environment.required_prefix(read.context, read.use.position)
    return InstalledValueSlotQuery(namespace, key, prefix, frozenset())


def installation(query, index=0):
    return tuple(
        occurrence
        for occurrence in query.prefix.mutation_occurrences()
        if isinstance(occurrence.mutation.target, CompactItemTarget)
    )[index]


@pytest.mark.parametrize("aliased", (False, True))
@pytest.mark.parametrize("distinct_suffix", (False, True))
def test_unique_item_read_retains_original_class_and_strict_query_refuses(
    aliased, distinct_suffix
):
    receiver = "alias" if aliased else "REGISTRY"
    source = BASE + ("alias = REGISTRY\n" if aliased else "")
    source += f"{receiver}['alpha'] = Alpha\n"
    source += "REGISTRY['beta'] = Beta\n" if distinct_suffix else ""
    environment = execution(source + "tail = REGISTRY\n")
    assert_closed_stores(environment)
    query = tail_query(environment)
    owner = environment.class_entry(
        environment.module.module.body[1]
    ).definition.target.owner
    value = query.resolve(environment.kernel)
    value.require_definition_identity(owner)
    original = installation(query)
    assert (
        value.source_definition()[1]
        is environment.class_entry(environment.module.module.body[1]).definition
    )
    environment.item_installation_value(original).require_definition_identity(owner)
    strict = CapturedSlotQuery(query.namespace, query.key, query.prefix, query.pending)
    assert isinstance(strict.resolve(environment.kernel), OpenCapturedReference)
    assert replace(query, key="unused").resolve(environment.kernel) is None


@pytest.mark.parametrize(
    "value, expected", (("object", object), ("property", property))
)
def test_normal_kernel_slot_lookup_exposes_original_native_value(value, expected):
    environment = execution(
        f"REGISTRY = {{}}\nREGISTRY['alpha'] = {value}\ntail = REGISTRY\n"
    )
    query = tail_query(environment)
    read = environment.source.value_reads_by_node[
        environment.module.module.body[-1].value
    ]
    result = environment.kernel._slot(
        query.namespace, query.key, read.context, read.use.position, frozenset()
    )
    result.require_native_identity(NativeDeclaration(expected))


@pytest.mark.parametrize("rebound", ("saved", "Alpha"))
def test_stored_rhs_retains_alias_before_its_name_is_rebound(rebound):
    environment = execution(
        BASE + "saved = Alpha\nREGISTRY['alpha'] = saved\n"
        f"{rebound} = None\ntail = REGISTRY\n"
    )
    query = tail_query(environment)
    result = query.resolve(environment.kernel)
    original = environment.class_entry(environment.module.module.body[1]).definition
    assert result.source_definition()[1] is original


def test_native_dictionary_copy_uses_historical_installed_slots():
    environment = execution(
        BASE + "REGISTRY['alpha'] = Alpha\ncopied = dict(REGISTRY)\n"
        "REGISTRY['beta'] = Beta\ntail = copied\n"
    )
    query = tail_query(environment)
    assert isinstance(query.namespace, CopiedNativeNamespace)
    assert query.namespace is not query.namespace.parent
    value = query.resolve(environment.kernel)
    original = environment.class_entry(environment.module.module.body[1]).definition
    assert value.source_definition()[1] is original
    # The later source store does not become part of the earlier copy.
    assert replace(query, key="beta").resolve(environment.kernel) is None
    assert query.namespace.initial_names == frozenset(("alpha",))


def test_repeated_key_release_requires_the_original_independent_class_binding():
    environment = execution(
        BASE + "REGISTRY['alpha'] = Alpha\nREGISTRY['alpha'] = Beta\ntail = REGISTRY\n"
    )
    first, second = item_statements(environment)
    environment.require_item_write(first.targets[0])
    environment.require_item_write(second.targets[0])
    query = tail_query(environment)
    beta = environment.class_entry(environment.module.module.body[2]).definition
    assert query.resolve(environment.kernel).source_definition()[1] is beta
    alpha = environment.capture_value(first.value)
    # Knowing the stored value did not grant unrestricted destruction. The real
    # dictionary overwrite succeeds because the original Alpha slot retains it.
    with pytest.raises(ValueError):
        alpha.require_release()
    with pytest.raises(ValueError, match="independent retained reference"):
        alpha.require_release_from(
            environment.kernel,
            CapturedSlotQuery(environment.entry, "Alpha", query.prefix, frozenset()),
        )


@pytest.mark.parametrize("key", ("missing", "unknown()", "1.0", "('alpha',)"))
def test_unproved_key_cannot_be_promoted_into_installed_value(key):
    environment = execution(BASE + f"REGISTRY[{key}] = Alpha\ntail = REGISTRY\n")
    (statement,) = item_statements(environment)
    with pytest.raises(ValueError):
        environment.require_item_write(statement.targets[0])
    with pytest.raises(ValueError):
        tail_query(environment)


def test_custom_setter_is_not_exact_dictionary_installation():
    source = (
        "events = []\nclass Custom(dict):\n"
        "    def __setitem__(self, key, value):\n"
        "        events.append((key, value))\n"
        "REGISTRY = Custom()\nREGISTRY['alpha'] = object\ntail = REGISTRY\n"
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["REGISTRY"] == {}
    assert namespace["events"] == [("alpha", object)]
    environment = execution(source)
    (statement,) = item_statements(environment)
    with pytest.raises(ValueError):
        environment.require_item_write(statement.targets[0])


@pytest.mark.parametrize(
    "forgery", ("event", "context", "frame", "before-installation")
)
def test_attestation_requires_actual_event_context_frame_and_cut(forgery):
    environment = execution(BASE + "REGISTRY['alpha'] = Alpha\ntail = REGISTRY\n")
    query = tail_query(environment)
    actual = installation(query)
    if forgery == "event":
        candidate = replace(actual, mutation=replace(actual.mutation))
    elif forgery == "context":
        candidate = replace(
            actual,
            source=replace(actual.source, context=replace(actual.source.context)),
        )
    elif forgery == "frame":
        foreign = execution(environment.module.source)
        candidate = replace(
            actual, source=replace(actual.source, frame=foreign.entry.frame)
        )
    else:
        candidate = replace(
            actual, source=replace(actual.source, position=actual.mutation.position)
        )
    with pytest.raises(ValueError):
        environment.item_installation_value(candidate)


@dataclass(frozen=True)
class PrefixOnlyEffects(CapturedReferenceEffectsABC):
    """Delegate original prefix evidence, deliberately not storage attestation."""

    environment: SourceModuleExecution

    def admit(self, context, position):
        return self.environment.admit(context, position)


def test_arbitrary_prefix_provider_does_not_inherit_storage_attestation():
    environment = execution(BASE + "REGISTRY['alpha'] = Alpha\ntail = REGISTRY\n")
    query = tail_query(environment)
    actual = installation(query)
    effects = PrefixOnlyEffects(environment)
    prefix = effects.admit(
        query.prefix.endpoint.context, query.prefix.endpoint.position
    )
    prefix.require_admitted(environment.initial)
    with pytest.raises(ValueError):
        effects.item_installation_value(actual)
    kernel = CapturedReferenceKernel(environment.initial, effects)
    value = query.matching_item_write(kernel, actual)
    assert isinstance(value, OpenCapturedReference)
    with pytest.raises(ValueError):
        value.require_closed()


def test_installation_transition_preserves_unrelated_pending_cycle_guard():
    environment = execution(
        BASE + "REGISTRY['alpha'] = Alpha\nREGISTRY['beta'] = Beta\ntail = REGISTRY\n"
    )
    query = tail_query(environment)
    alpha, beta = installation(query, 0), installation(query, 1)
    pending = frozenset(
        CompactBindingVisit(occurrence.source.context, occurrence.mutation)
        for occurrence in (alpha, beta)
    )
    value = replace(query, pending=pending).matching_item_write(
        environment.kernel, alpha
    )
    assert isinstance(value, OpenCapturedReference)
    assert value.violation is CapturedReferenceViolation.CYCLIC_BINDING
    assert value.mutation is beta.mutation


@pytest.mark.parametrize("count", (1, 8, 32))
def test_latest_item_query_does_not_recursively_replay_superseded_values(count):
    environment = execution(
        "REGISTRY = {}\n"
        + "".join(f"REGISTRY['alpha'] = {value}\n" for value in range(count))
        + "tail = REGISTRY\n"
    )
    query = tail_query(environment)
    calls = []
    target = InstalledValueSlotQuery.matching_item_write.__code__

    def observe(frame, event, result):
        if event == "call" and frame.f_code is target:
            calls.append(frame.f_locals["occurrence"])

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        value = query.resolve(environment.kernel)
    finally:
        sys.setprofile(previous)
    assert value.require_native_scalar() == count - 1
    assert len(calls) == 1
    assert calls[0].mutation is installation(query, -1).mutation
    namespace = {}
    exec(environment.module.source, namespace)  # Authored fixture only.
    assert namespace["REGISTRY"]["alpha"] == value.require_native_scalar()
