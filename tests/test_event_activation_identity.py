"""Static event identity does not identify the activation that created a value."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import SequentialExecutionPrefix
from nominal_refactor_advisor.source_entry import SourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_source_function_result import execution


@pytest.fixture(
    params=("value = {}\n", "import builtins\nvalue = dict(vars(builtins))\n")
)
def dictionary_activations(request):
    first = execution(request.param)
    original = first.entry
    second = SourceModuleExecution(
        SourceModuleEntryPremise(
            source=original.source,
            native_island=original.initial,
            bindings=dict(original.initial_entries),
            builtins=original.builtins,
        )
    )
    node = first.module.module.body[-1].value
    captures = tuple(owner.capture_value(node) for owner in (first, second))
    prefixes = tuple(
        owner.required_prefix(owner.entry.context, None) for owner in (first, second)
    )
    return (first, second), captures, prefixes


@pytest.mark.parametrize("owner_index", (0, 1))
def test_same_static_event_in_another_frame_does_not_prove_creation(
    dictionary_activations, owner_index
):
    owners, captures, prefixes = dictionary_activations
    owner = owners[owner_index]
    with pytest.raises(ValueError, match="unique occurrence"):
        captures[owner_index].require_available(owner.kernel, prefixes[1 - owner_index])


@pytest.mark.parametrize("owner_index", (0, 1))
def test_two_real_creations_are_disambiguated_by_their_original_frame(
    dictionary_activations, owner_index
):
    owners, captures, prefixes = dictionary_activations
    combined = SequentialExecutionPrefix(prefixes)
    combined.require_admitted(owners[0].initial)
    captures[owner_index].require_available(owners[owner_index].kernel, combined)
    assert captures[0] is not captures[1]


def test_copied_frame_cannot_supply_an_original_creation(dictionary_activations):
    owners, captures, prefixes = dictionary_activations
    first = owners[0]
    copied = replace(prefixes[0], frame=replace(first.entry.frame))
    with pytest.raises(ValueError, match="unique occurrence"):
        captures[0].require_available(first.kernel, copied)


def test_duplicating_one_creation_does_not_become_two_distinct_activations(
    dictionary_activations,
):
    owners, captures, prefixes = dictionary_activations
    duplicated = SequentialExecutionPrefix((prefixes[0], prefixes[0]))
    with pytest.raises(ValueError, match="unique occurrence"):
        captures[0].require_available(owners[0].kernel, duplicated)
