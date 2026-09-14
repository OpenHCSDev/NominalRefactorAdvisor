"""Native lower bounds refine effect selection without inventing assembly events."""

from copy import deepcopy
from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    OpenCapturedReference,
    SingleFlowPrefix,
)
from nominal_refactor_advisor.class_namespace import (
    CallClassNamespaceEffect,
    DictionaryConstructionEffect,
    EvaluationEffectOccurrence,
    OperationEffectOccurrence,
)
from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from test_source_dictionary_construction import (
    _execution,
    _require_dictionary_effects,
    _require_original_class,
)


def evaluation(environment, node):
    return next(bound for bound in environment.source.evaluations if bound.node is node)


def dictionary_effect(environment, dictionary):
    return next(
        effect
        for site in environment.effects.sites
        if site.trigger is dictionary
        for effect in site.effects
        if isinstance(effect, DictionaryConstructionEffect)
    )


def occurrences_at(environment, node):
    read = environment.source.reference_reads_by_node[node]
    prefix = SingleFlowPrefix(read.context, environment.entry.frame, read.use.position)
    return tuple(environment.effects.occurrences(environment.source, prefix))


@pytest.mark.parametrize(
    "left,right", (("'same'", "'same'"), ("(1,(2,))", "(True,(2.0,))"))
)
def test_duplicate_key_effect_starts_after_first_original_value_capture(left, right):
    environment, dictionary = _execution(f"{{{left}: Alpha, {right}: Beta}}")
    effect = dictionary_effect(environment, dictionary)
    whole = evaluation(environment, dictionary)
    first = evaluation(environment, dictionary.values[0])
    assert effect.earliest_application(environment.source, whole) is first.entry
    assert (
        NativeCreationBackend.current().dictionary_construction_initial_operand(
            dictionary
        )
        is dictionary.values[0]
    )
    first_occurrences = occurrences_at(environment, dictionary.values[0])
    assert not any(occurrence.effect is effect for occurrence in first_occurrences)
    (later,) = tuple(
        occurrence
        for occurrence in occurrences_at(environment, dictionary.values[1])
        if occurrence.effect is effect
    )
    assert isinstance(later, EvaluationEffectOccurrence)
    assert later.evaluation is whole
    _require_original_class(environment, dictionary.values[0])
    with pytest.raises(ValueError):
        _require_original_class(environment, dictionary.values[1])
    with pytest.raises(ValueError):
        _require_dictionary_effects(environment, dictionary)


def test_nested_key_event_count_does_not_shift_the_native_effect_before_first_value():
    environment, dictionary = _execution("{(1,(2,)):Alpha,(True,(2.0,)):Beta}")
    whole = evaluation(environment, dictionary)
    key = evaluation(environment, dictionary.keys[0])
    first = evaluation(environment, dictionary.values[0])
    assert whole.entry.event_index < key.exit.event_index
    assert key.exit == first.entry
    start = dictionary_effect(environment, dictionary).earliest_application(
        environment.source, whole
    )
    assert start is first.entry
    assert start != whole.entry


@pytest.mark.parametrize(
    "expression,prefix",
    (
        ("{key: Alpha, 'beta': Beta}", "key = 'alpha'\n"),
        ("{make_key(): Alpha, 'beta': Beta}", "def make_key(): return 'alpha'\n"),
        ("{**other, 'beta': Beta}", "other={}\n"),
    ),
)
def test_unknown_or_unpacked_first_key_has_no_stronger_bound(expression, prefix):
    environment, dictionary = _execution(expression, prefix=prefix)
    whole = evaluation(environment, dictionary)
    assert (
        NativeCreationBackend.current().dictionary_construction_initial_operand(
            dictionary
        )
        is None
    )
    assert (
        dictionary_effect(environment, dictionary).earliest_application(
            environment.source, whole
        )
        is whole.entry
    )
    with pytest.raises(ValueError):
        _require_dictionary_effects(environment, dictionary)


def test_later_unpack_still_interferes_after_first_ordinary_pair():
    environment, dictionary = _execution(
        "{'alpha': Alpha, **other}", prefix="other={}\n"
    )
    _require_original_class(environment, dictionary.values[0])
    with pytest.raises(ValueError):
        _require_dictionary_effects(environment, dictionary)


def test_prior_child_call_effect_is_not_hidden_by_dictionary_lower_bound():
    environment, dictionary = _execution("{(1,): (unknown(), Alpha), (2,): Beta}")
    first_value = dictionary.values[0]
    call, alpha = first_value.elts
    context, original_call = environment.source_call(call)
    (occurrence,) = tuple(
        occurrence
        for occurrence in occurrences_at(environment, alpha)
        if isinstance(occurrence.effect, CallClassNamespaceEffect)
        and occurrence.effect.node is call
    )
    assert isinstance(occurrence, OperationEffectOccurrence)
    assert occurrence.operation is environment.source_operation(context, original_call)
    captured = environment.capture(alpha)
    assert isinstance(captured, OpenCapturedReference)
    with pytest.raises(ValueError):
        captured.require_closed()


@pytest.mark.parametrize(
    "defect",
    (
        "copied_bound",
        "foreign_bound",
        "copied_operand",
        "foreign_operand",
        "outside_operand",
    ),
)
def test_lower_bound_requires_actual_contained_source_evaluations(defect):
    environment, dictionary = _execution(
        "{(1,):Alpha,(2,):Beta}", prefix="earlier=object\n"
    )
    whole = evaluation(environment, dictionary)
    operand = dictionary.values[0]
    other, other_dictionary = _execution(
        "{(1,):Alpha,(2,):Beta}", prefix="earlier=object\n"
    )
    if defect == "copied_bound":
        whole = replace(whole)
    elif defect == "foreign_bound":
        whole = evaluation(other, other_dictionary)
    elif defect == "copied_operand":
        operand = deepcopy(operand)
    elif defect == "foreign_operand":
        operand = other_dictionary.values[0]
    else:
        operand = environment.module.module.body[-2].value
    with pytest.raises(ValueError):
        environment.source.evaluation_operand_entry(whole, operand)


def test_unknown_native_backend_cannot_claim_a_narrower_interval(monkeypatch):
    environment, dictionary = _execution("{(1,):Alpha,(2,):Beta}")
    whole = evaluation(environment, dictionary)
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    assert (
        dictionary_effect(environment, dictionary).earliest_application(
            environment.source, whole
        )
        is whole.entry
    )
    with pytest.raises(ValueError):
        _require_dictionary_effects(environment, dictionary)


def test_refined_overlap_keeps_half_open_and_continuation_boundaries():
    environment, dictionary = _execution("{(1,):Alpha,(2,):Beta}")
    whole = evaluation(environment, dictionary)
    context = environment.context_for_owner(whole.owner)
    entry = dictionary_effect(environment, dictionary).earliest_application(
        environment.source, whole
    )
    before = SingleFlowPrefix(context, environment.entry.frame, entry)
    assert not before.may_overlap_positions(entry, whole.exit)
    during = SingleFlowPrefix(context, environment.entry.frame, whole.exit)
    assert during.may_overlap_positions(entry, whole.exit)
    completed = SingleFlowPrefix(
        context, environment.entry.frame, None, after=whole.exit
    )
    assert not completed.may_overlap_positions(entry, whole.exit)


@pytest.mark.parametrize("expression", ("{1:Alpha}", "{1:None}"))
def test_contained_operand_may_share_both_parent_boundaries(expression):
    environment, dictionary = _execution(expression)
    whole = evaluation(environment, dictionary)
    operand = evaluation(environment, dictionary.values[0])
    assert operand.entry == whole.entry
    assert operand.exit == whole.exit
    assert (
        environment.source.evaluation_operand_entry(whole, dictionary.values[0])
        is operand.entry
    )
