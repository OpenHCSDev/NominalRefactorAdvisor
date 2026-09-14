"""Composite prefix traversal preserves occurrence order at arbitrary depth."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    ChildExecutionPrefix,
    SequentialExecutionPrefix,
    SingleFlowPrefix,
)
from test_contextual_execution_prefix import fixture


def _parts():
    kernel, read = fixture(
        "import builtins\nclass Child:\n    saved = object\n"
        "result = builtins.property\n"
    )
    effects = kernel.effects
    cut = effects.definition.target.header_position
    parent = SingleFlowPrefix(effects.parent, effects.parent_frame, cut)
    child = SingleFlowPrefix(effects.child, effects.child_frame, None)
    resume = SingleFlowPrefix(
        effects.parent, effects.parent_frame, read.use.position, cut
    )
    return kernel, parent, child, resume


@pytest.mark.parametrize("nested_side", ("left", "right"))
def test_deep_prefix_traversal_preserves_actual_occurrences(nested_side):
    kernel, parent, child, resume = _parts()
    prefix = ChildExecutionPrefix(parent, kernel.effects.definition, child)
    # Repetition is intentional: flattening must retain each occurrence so
    # admission can reject a duplicated activation rather than hiding it.
    count = 1200
    for _ in range(count):
        parts = (prefix, resume) if nested_side == "left" else (resume, prefix)
        prefix = SequentialExecutionPrefix(parts)
    expected = (
        (parent, child) + (resume,) * count
        if nested_side == "left"
        else (resume,) * count + (parent, child)
    )
    assert len(prefix.intervals) == len(expected)
    assert all(actual is wanted for actual, wanted in zip(prefix.intervals, expected))
    assert prefix.endpoint is expected[-1]
    with pytest.raises(ValueError):
        prefix.require_admitted(kernel.initial)


def test_alternating_composites_preserve_cuts_and_canonical_frames():
    kernel, parent, child, resume = _parts()
    prefix = SequentialExecutionPrefix(
        (
            ChildExecutionPrefix(
                SequentialExecutionPrefix((parent,)),
                kernel.effects.definition,
                SequentialExecutionPrefix((child,)),
            ),
            SequentialExecutionPrefix((resume,)),
        )
    )
    prefix.require_admitted(kernel.initial)
    assert prefix.intervals == (parent, child, resume)
    assert prefix.endpoint is resume


def test_traversal_workspace_does_not_leak_between_repeated_reads():
    _, parent, child, resume = _parts()
    prefix = SequentialExecutionPrefix((parent, child, resume))
    first = prefix.intervals
    second = prefix.intervals
    assert first == second == (parent, child, resume)
    replacement = replace(prefix, parts=(resume, parent))
    assert replacement.intervals == (resume, parent)
    assert prefix.intervals == first
