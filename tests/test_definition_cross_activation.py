"""Definition availability follows original execution history, not observer identity."""

from dataclasses import dataclass, field, replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    AdmittedExecutionPrefixABC,
    SequentialExecutionPrefix,
)
from nominal_refactor_advisor.source_entry import SourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_source_function_result import execution


@dataclass(frozen=True, eq=False)
class FollowingModuleEntry(SourceModuleEntryPremise):
    """Explicit sequential entry used only by these authored two-module controls."""

    preceding: AdmittedExecutionPrefixABC = field(kw_only=True)

    def prefix(self, position, kernel):
        return SequentialExecutionPrefix(
            (self.preceding, super().prefix(position, kernel))
        )


def handoff(source):
    first = execution(source + "\noriginal = chosen\n")
    value = first.capture_value(first.module.module.body[-1].value)
    previous = first.required_prefix(first.entry.context, None)
    template = execution("received = supplied\n")
    second = SourceModuleExecution(
        FollowingModuleEntry(
            source=template.source,
            native_island=first.initial,
            bindings={"supplied": value},
            builtins=first.entry.builtins,
            preceding=previous,
        )
    )
    return first, second, value


@pytest.mark.parametrize(
    "source", ("def chosen():\n    return None", "class chosen: pass")
)
def test_original_definition_remains_available_to_a_later_source_activation(source):
    first, second, value = handoff(source)
    native_first = {}
    exec(first.module.source, native_first)
    native_second = {"supplied": native_first["chosen"]}
    exec(second.module.source, native_second)
    assert native_second["received"] is native_first["chosen"]
    prefix = second.required_prefix(second.entry.context, None)
    assert second.capture_value(second.module.module.body[-1].value) is value
    assert (
        value.require_available_at(second.kernel, second.entry.context, None) is prefix
    )
    value.require_available_in(prefix)


def test_a_shared_source_projection_does_not_substitute_another_creation_frame():
    first, second, value = handoff("def chosen():\n    return None")
    alternate = SourceModuleExecution.from_source(first.source)
    # A different native island is independently rejected even with equal source.
    with pytest.raises(ValueError):
        value.require_available_at(alternate.kernel, alternate.entry.context, None)
    missing = SourceModuleExecution(
        SourceModuleEntryPremise(
            source=first.source,
            native_island=first.initial,
            bindings=dict(first.entry.initial_entries),
            builtins=first.entry.builtins,
        )
    )
    assert missing.source is first.source
    assert missing.entry.frame is not first.entry.frame
    with pytest.raises(ValueError):
        value.require_available_at(missing.kernel, missing.entry.context, None)
    with pytest.raises(ValueError):
        value.require_available_at(second.kernel, replace(second.entry.context), None)


def test_noncanonical_composition_is_not_admitted_as_a_definition_cut():
    _, second, value = handoff("def chosen():\n    return None")
    original = second.required_prefix(second.entry.context, None)
    with pytest.raises(ValueError):
        value.require_available_in(SequentialExecutionPrefix(original.parts))


def test_handoff_does_not_grant_call_body_or_external_interference_proofs():
    _, second, value = handoff("def chosen():\n    return None")
    prefix = second.required_prefix(second.entry.context, None)
    value.require_available_in(prefix)
    with pytest.raises(ValueError, match="External source interference"):
        value.execution.entry.require_external_noninterference(prefix)
