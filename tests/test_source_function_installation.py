"""Source-bound installation retains the actual definition and historical cut."""

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
from nominal_refactor_advisor.native_compilation import NativePrimitiveOperation
from nominal_refactor_advisor.source_execution import (
    SourceCreatedFunctionCapture,
    SourceModuleExecution,
)
from test_native_text_capture import execution
from test_source_function_storage import prepared_function


def module_function(source="def chosen():\n    pass\n"):
    environment = execution(source)
    function = SourceCreatedFunctionCapture(
        environment, environment.module.module.body[0]
    )
    return environment, function


def test_prepared_root_final_method_joins_canonical_native_installation():
    environment, entry, function = prepared_function(explicit_scope=False)
    prefix = entry.completion_prefix
    installed = function.require_native_installation(prefix)
    canonical = function.native_execution.require_installation()
    actual = entry.completion_member("example")
    assert installed is canonical
    assert actual.require_native_installation(prefix) is installed
    assert installed.operation is NativePrimitiveOperation.STORE_NAME
    assert installed.name == function.definition.target.bound_name == "example"
    assert actual.proves_same_object(function)
    assert not environment._pending
    # Historical installation does not supply continued empty custom storage.
    with pytest.raises(
        ValueError, match="External source interference remains unproved"
    ):
        function.require_fresh_function_namespace(prefix)
    # Nor does this result close the surrounding native class construction.
    with pytest.raises(
        ValueError, match="External source interference remains unproved"
    ):
        entry.result()


@pytest.mark.parametrize(
    "header", ("def chosen():", "async def chosen():", "def __chosen():")
)
def test_module_installation_uses_original_source_binding_without_name_conventions(
    header,
):
    environment, function = module_function(header + "\n    pass\n")
    prefix = environment.required_prefix(environment.entry.context, None)
    installation = function.require_native_installation(prefix)
    assert installation is function.native_execution.require_installation()
    assert (
        installation.name == function.node.name == function.definition.target.bound_name
    )
    assert function.require_native_installation(prefix) is installation


def test_preinstallation_cut_does_not_contain_the_original_definition_store():
    _, function = module_function()
    function.require_closed()
    with pytest.raises(ValueError, match="no unique occurrence"):
        function.require_native_installation(function.parent_prefix)


@pytest.mark.parametrize(
    "rebuild",
    (
        copy,
        lambda prefix: SingleFlowPrefix(
            prefix.endpoint.context, prefix.endpoint.frame, prefix.endpoint.position
        ),
    ),
)
def test_copied_or_truncated_cut_cannot_borrow_canonical_installation(rebuild):
    _, entry, function = prepared_function()
    prefix = entry.completion_prefix
    copied = rebuild(prefix)
    assert copied.endpoint.context is prefix.endpoint.context
    assert copied.endpoint.frame is prefix.endpoint.frame
    with pytest.raises(ValueError, match="canonical closed source cut"):
        function.require_native_installation(copied)


def test_same_source_and_native_island_do_not_join_foreign_activation():
    environment, function = module_function()
    foreign = SourceModuleExecution(
        replace(environment.entry, bindings=dict(environment.entry.initial_entries))
    )
    prefix = foreign.required_prefix(foreign.entry.context, None)
    assert foreign.source is environment.source
    assert foreign.initial is environment.initial
    with pytest.raises(ValueError, match="Execution event has no unique occurrence"):
        function.require_native_installation(prefix)
    assert not environment._pending and not foreign._pending


@pytest.mark.parametrize("query", ("installation", "descriptor", "native_execution"))
def test_warmed_receipt_cannot_switch_to_another_original_function_node(query):
    environment, function = module_function("def first(): pass\ndef second(): pass\n")
    prefix = environment.required_prefix(environment.entry.context, None)
    original = function.native_execution
    function.require_native_installation(prefix)
    operation = function.operation
    object.__setattr__(function, "node", environment.module.module.body[1])
    assert function.operation is operation
    with pytest.raises(ValueError, match="original canonical source operation"):
        if query == "installation":
            function.require_native_installation(prefix)
        elif query == "descriptor":
            function.require_descriptor_argument()
        else:
            _ = function.native_execution
    assert original is environment.module.native_compilation.execution_for(
        original.source_span
    )


def test_decorated_result_cannot_use_raw_function_installation():
    environment = execution("class Owner:\n    @staticmethod\n    def chosen(): pass\n")
    owner = environment.class_entry(environment.module.module.body[0])
    function = SourceCreatedFunctionCapture(environment, owner.node.body[0])
    with pytest.raises(ValueError, match="decorator result remains unproved"):
        function.require_native_installation(owner.completion_prefix)


def test_conditional_native_installation_does_not_supply_source_execution():
    environment = execution("if True:\n    def chosen(): pass\n")
    node = environment.module.module.body[0].body[0]
    function = SourceCreatedFunctionCapture(environment, node)
    assert function.native_execution.require_installation().name == "chosen"
    # A real canonical earlier cut exists, but it cannot establish that the
    # conditional function was subsequently installed.
    prefix = environment.required_prefix(
        environment.entry.context, function.definition.target.header_position
    )
    with pytest.raises(
        ValueError, match="Conditional or repeated function creation remains unproved"
    ):
        function.require_native_installation(prefix)


def test_mangled_native_key_requires_actual_source_binding_correspondence():
    environment = execution("class Owner:\n    def __chosen(self): pass\n")
    owner = environment.class_entry(environment.module.module.body[0])
    function = SourceCreatedFunctionCapture(environment, owner.node.body[0])
    assert function.native_execution.require_installation().name == "_Owner__chosen"
    assert function.definition.target.bound_name == "__chosen"
    with pytest.raises(ValueError, match="differs from its original source binding"):
        function.require_native_installation(owner.completion_prefix)


def test_later_reassignment_keeps_historical_installation_not_current_value():
    environment, function = module_function(
        "def chosen(): pass\nsaved = chosen\nchosen = 17\n"
    )
    prefix = environment.required_prefix(environment.entry.context, None)
    installation = function.require_native_installation(prefix)
    assert installation.name == "chosen"
    assert installation is function.native_execution.require_installation()
    stored = environment.kernel._slot(
        environment.entry, "chosen", environment.entry.context, None, frozenset()
    )
    retained = environment.kernel._slot(
        environment.entry, "saved", environment.entry.context, None, frozenset()
    )
    assert stored is not None and retained is not None
    assert stored.require_native_scalar() == 17
    assert not function.proves_same_object(stored)
    assert retained.proves_same_object(function)
    assert retained.require_native_installation(prefix) is installation


@pytest.mark.parametrize(
    "capture",
    (
        NativeTypePremise(FunctionType),
        NativeTypePremise(str),
        CapturedNativeObject(lambda: None),
    ),
)
def test_native_type_or_identity_does_not_fabricate_source_installation(capture):
    environment, _ = module_function()
    prefix = environment.required_prefix(environment.entry.context, None)
    with pytest.raises(
        ValueError, match="Source-bound native installation remains unproved"
    ):
        capture.require_native_installation(prefix)


@pytest.mark.parametrize("cause", (None, ValueError("original prerequisite")))
def test_open_capture_preserves_its_original_typed_cause(cause):
    capture = OpenCapturedReference(
        CapturedReferenceViolation.UNADMITTED_IMPORT, cause=cause
    )
    environment, _ = module_function()
    prefix = environment.required_prefix(environment.entry.context, None)
    with pytest.raises(
        CapturedReferenceRejection, match="unadmitted_native_import"
    ) as rejected:
        capture.require_native_installation(prefix)
    assert rejected.value.evidence is capture
    assert rejected.value.violation is capture.violation
    assert rejected.value.__cause__ is cause
