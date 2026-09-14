"""Fresh function custom storage at an actual admitted source execution cut.

The explicit external-interference premise does not exempt source operations or
their callbacks. Native controls execute only authored subprocess fixtures.
"""

import ast
from copy import copy
from dataclasses import replace
from types import FunctionType

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    NativeTypePremise,
    OpenCapturedReference,
    SingleFlowPrefix,
)
from nominal_refactor_advisor.source_entry import NoninterferingSourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import (
    SourceCreatedFunctionCapture,
    SourceModuleExecution,
)
from test_native_source_class_preparation import SOURCE, prepared_execution
from test_registry_native_registration_controls import native_control


def prepared_function(*, explicit_scope=True, source=SOURCE, conditions=True):
    environment, (root, *_) = prepared_execution(source, conditions=conditions)
    if explicit_scope:
        original = environment.entry
        environment = SourceModuleExecution(
            NoninterferingSourceModuleEntryPremise(
                source=original.source,
                native_island=original.initial,
                bindings=dict(original.initial_entries),
                builtins=original.builtins,
                declared_operation_conditions=original.operation_conditions.values(),
            )
        )
    entry = environment.class_entry(root)
    node = next(node for node in root.body if isinstance(node, ast.FunctionDef))
    return environment, entry, SourceCreatedFunctionCapture(environment, node)


def test_actual_prepared_root_function_has_empty_custom_storage_at_completed_cut():
    environment, entry, function = prepared_function()
    prefix = entry.completion_prefix
    actual = entry.completion_member("example")
    assert actual.proves_same_object(function)
    assert actual.native_type is FunctionType
    actual.require_fresh_function_namespace(prefix)
    function.require_fresh_function_namespace(prefix)
    assert prefix is environment.required_prefix(entry.context, None)
    assert not environment._pending
    # Function storage proof supplies no class construction result.
    with pytest.raises(
        ValueError, match="construction over prepared inputs remains unproved"
    ):
        entry.result()


def test_default_entry_does_not_infer_external_noninterference():
    _, entry, function = prepared_function(explicit_scope=False)
    with pytest.raises(
        ValueError, match="External source interference remains unproved"
    ):
        function.require_fresh_function_namespace(entry.completion_prefix)


@pytest.mark.parametrize(
    "capture",
    (
        NativeTypePremise(FunctionType),
        NativeTypePremise(str),
        CapturedNativeObject(lambda: None),
        CapturedNativeObject("literal"),
        CapturedNativeObject(1),
    ),
)
def test_other_capture_kinds_do_not_infer_fresh_function_storage(capture):
    _, entry, _ = prepared_function()
    with pytest.raises(
        ValueError, match="Fresh function storage at the source cut remains unproved"
    ):
        capture.require_fresh_function_namespace(entry.completion_prefix)


@pytest.mark.parametrize("cause", (None, ValueError("original source obligation")))
def test_open_capture_keeps_original_rejection_before_capability_query(cause):
    _, entry, _ = prepared_function()
    capture = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_ACCESS, cause=cause
    )
    with pytest.raises(
        CapturedReferenceRejection,
        match="Object capture remains open: unproved_object_access",
    ) as rejected:
        capture.require_fresh_function_namespace(entry.completion_prefix)
    assert rejected.value.evidence is capture
    assert rejected.value.violation is CapturedReferenceViolation.UNPROVED_ACCESS
    assert rejected.value.__cause__ is cause
    assert capture.cause is cause


@pytest.mark.parametrize("member", ("registry_key", "__registry__", "__qualname__"))
def test_actual_non_function_root_members_keep_capability_closed(member):
    _, entry, _ = prepared_function()
    with pytest.raises(
        ValueError, match="Fresh function storage at the source cut remains unproved"
    ):
        entry.completion_member(member).require_fresh_function_namespace(
            entry.completion_prefix
        )


def test_function_storage_rejects_its_preinstallation_header_cut():
    _, _, function = prepared_function()
    with pytest.raises(ValueError, match="no unique occurrence"):
        function.require_fresh_function_namespace(function.parent_prefix)


def test_function_storage_rejects_another_activation_prefix():
    environment, _, function = prepared_function()
    _, alien_entry, _ = prepared_function()
    with pytest.raises(ValueError):
        function.require_fresh_function_namespace(alien_entry.completion_prefix)
    assert not environment._pending


@pytest.mark.parametrize(
    "rebuild",
    (
        copy,
        lambda prefix: SingleFlowPrefix(
            prefix.endpoint.context, prefix.endpoint.frame, prefix.endpoint.position
        ),
    ),
)
def test_matching_endpoint_does_not_authenticate_a_rebuilt_or_truncated_prefix(rebuild):
    _, entry, function = prepared_function()
    actual = entry.completion_prefix
    supplied = rebuild(actual)
    assert supplied.endpoint.frame is actual.endpoint.frame
    assert supplied.endpoint.context is actual.endpoint.context
    assert supplied.endpoint.position == actual.endpoint.position
    with pytest.raises(ValueError, match="canonical closed source cut"):
        function.require_fresh_function_namespace(supplied)


def test_explicit_scope_cannot_be_consumed_for_another_activation():
    environment, _, _ = prepared_function()
    # Same parsed source, island, bindings and conditions, different entry/frame.
    other = SourceModuleExecution(
        replace(
            environment.entry,
            bindings=dict(environment.entry.initial_entries),
            declared_operation_conditions=environment.entry.operation_conditions.values(),
        )
    )
    root = next(
        node for node in other.module.module.body if isinstance(node, ast.ClassDef)
    )
    with pytest.raises(ValueError, match="different source activation"):
        environment.entry.require_external_noninterference(
            other.class_entry(root).completion_prefix
        )


@pytest.mark.parametrize(
    "statement",
    (
        "    example.__isabstractmethod__ = True\n",
        "    alias = example\n    alias.__isabstractmethod__ = True\n",
    ),
)
def test_explicit_external_scope_does_not_exempt_source_function_writes(statement):
    source = SOURCE.replace("class Alpha(Family):", statement + "class Alpha(Family):")
    environment, entry, function = prepared_function(source=source)
    function.require_closed()
    with pytest.raises(ValueError):
        function.require_fresh_function_namespace(entry.completion_prefix)
    assert not environment._pending


@pytest.mark.parametrize("decorator", ("staticmethod", "classmethod"))
def test_raw_function_birth_does_not_admit_a_decorated_installed_result(decorator):
    source = SOURCE.replace("    def example", f"    @{decorator}\n    def example")
    _, entry, function = prepared_function(source=source)
    with pytest.raises(ValueError):
        function.require_fresh_function_namespace(entry.completion_prefix)


def test_external_noninterference_does_not_supply_native_callback_behavior():
    environment, entry, function = prepared_function(conditions=False)
    with pytest.raises(ValueError, match="explicit entry condition"):
        environment.entry.require_native_behavior(entry)
    with pytest.raises(ValueError, match="unproved_execution_effects"):
        function.require_fresh_function_namespace(entry.completion_prefix)


@pytest.mark.parametrize("external_trace", (False, True))
def test_native_external_trace_can_change_abstractness_after_empty_function_birth(
    external_trace,
):
    outcome = native_control(
        """
import json
import sys
from metaclass_registry import AutoRegisterMeta
armed = bool(int(sys.argv[1]))
seen = []
def observe(frame, event, arg):
    if armed and event == 'line' and frame.f_code.co_name == 'Family' and 'example' in frame.f_locals:
        function = frame.f_locals['example']
        if not seen:
            assert function.__dict__ == {}
            seen.append('observed empty before trace write')
            function.__isabstractmethod__ = True
    return observe
sys.settrace(observe)
try:
    class Family(metaclass=AutoRegisterMeta):
        __registry__ = {}
        __registry_key__ = 'registry_key'
        def example(self):
            raise AssertionError('ordinary method body must not execute')
        after_installation = None
finally:
    sys.settrace(None)
print(json.dumps({'trace_changed_function': bool(seen),
                  'abstract_methods': sorted(Family.__abstractmethods__)}))
""",
        external_trace,
    )
    assert outcome == {
        "trace_changed_function": external_trace,
        "abstract_methods": ["example"] if external_trace else [],
    }
    # The mutating run is outside the supplied noninterfering execution domain.
    # Neither this native control nor the default entry asserts that premise.
