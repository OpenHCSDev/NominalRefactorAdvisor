"""Prefix construction derives and completes its actual predecessors once."""

import cProfile
import sys

import pytest

from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.source_execution import (
    SourceClassEntry,
    SourceModuleExecution,
)
from test_registry_original_values import authored_runtime
from test_source_definition_result import execution


def calls(profile, function):
    return sum(
        entry.callcount
        for entry in profile.getstats()
        if entry.code is function.__code__
    )


@pytest.mark.parametrize(
    "text",
    (
        "class First: pass\nclass Second(First): pass\nclass Last(Second): pass\n",
        "class Outer:\n    class First: pass\n    class Second: pass\n",
    ),
)
def test_each_fresh_prefix_derives_preceding_classes_once(text):
    environment = execution(text)
    profile = cProfile.Profile()
    result = profile.runcall(environment.admit, environment.source.module_context, None)
    assert not isinstance(result, OpenCapturedReference)
    prefixes = calls(profile, SourceModuleExecution._prefix)
    assert prefixes > 0
    assert calls(profile, SourceModuleExecution._preceding_class_entries) == prefixes
    assert not environment._pending

    cached_profile = cProfile.Profile()
    assert (
        cached_profile.runcall(
            environment.admit, environment.source.module_context, None
        )
        is result
    )
    assert calls(cached_profile, SourceModuleExecution._prefix) == 0
    assert calls(cached_profile, SourceModuleExecution._preceding_class_entries) == 0


def test_nested_predecessors_complete_before_the_original_class_prefix_constructs():
    text = "class Outer:\n    class First: pass\n    class Second: pass\n"
    environment = execution(text)
    observations = []
    completed_nodes = set()

    def observe(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is SourceClassEntry.completed.fget.__code__
        ):
            entry = frame.f_locals["self"]
            if entry.execution is environment:
                completed_nodes.add(entry.node)
        if event == "call" and frame.f_code is SourceClassEntry.prefix.__code__:
            entry = frame.f_locals["self"]
            if entry.execution is environment:
                observations.append(
                    (
                        entry.context,
                        frame.f_locals["position"],
                        frozenset(completed_nodes),
                    )
                )

    original_profiler = sys.getprofile()
    sys.setprofile(observe)
    try:
        environment.require_class_creation(environment.module.module.body[0])
    finally:
        sys.setprofile(original_profiler)
    assert observations
    nonempty = 0
    for context, position, completion_state in observations:
        predecessors = environment._preceding_class_entries(context, position)
        if predecessors:
            nonempty += 1
            assert all(entry.node in completion_state for entry in predecessors)
    assert nonempty
    assert not environment._pending
    runtime = authored_runtime(text)
    assert runtime["Outer"].First is not runtime["Outer"].Second


@pytest.mark.parametrize(
    "text,runtime_error",
    (
        ("if True:\n    class Original: pass\n", None),
        ("for item in (0, 1):\n    class Original: pass\n", None),
        ("class Outer:\n    def missing_default(value=absent): pass\n", NameError),
        (
            "class Base:\n    def __init_subclass__(cls): absent()\n"
            "class Derived(Base): pass\n",
            NameError,
        ),
    ),
)
def test_refused_prefix_preserves_pending_cleanup_and_cached_failure(
    text, runtime_error
):
    if runtime_error is None:
        authored_runtime(text)
    else:
        with pytest.raises(runtime_error):
            authored_runtime(text)
    environment = execution(text)
    result = environment.admit(environment.source.module_context, None)
    assert isinstance(result, OpenCapturedReference)
    with pytest.raises(ValueError):
        result.require_closed()
    assert not environment._pending
    profile = cProfile.Profile()
    assert (
        profile.runcall(environment.admit, environment.source.module_context, None)
        is result
    )
    assert calls(profile, SourceModuleExecution._prefix) == 0
    assert calls(profile, SourceModuleExecution._preceding_class_entries) == 0
    assert not environment._pending


def test_deferred_body_does_not_become_a_construction_requirement():
    text = "class First: pass\nclass Last:\n    def later(self): return absent\n"
    environment = execution(text)
    result = environment.admit(environment.source.module_context, None)
    assert not isinstance(result, OpenCapturedReference)
    assert not environment._pending
    runtime = authored_runtime(text)
    with pytest.raises(NameError):
        runtime["Last"]().later()


def test_module_function_header_refusal_cleans_pending_without_claiming_body_execution():
    text = "class Prior: pass\ndef missing_default(value=absent): pass\n"
    with pytest.raises(NameError):
        authored_runtime(text)
    environment = execution(text)
    node = environment.module.module.body[-1]
    for _ in range(2):
        with pytest.raises(ValueError, match="unproved"):
            environment.capture_definition(node).require_closed()
        assert not environment._pending
