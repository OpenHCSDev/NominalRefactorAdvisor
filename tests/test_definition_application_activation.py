"""Definition applications share topology without assuming transformation identity."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    DefinitionApplicationPrefix,
    SourceDefinitionApplicationAuthorityABC,
)
from nominal_refactor_advisor.source_entry import NoninterferingSourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import (
    SourceClassDecoratorApplication,
    SourceCreatedFunctionCapture,
    SourceDefinitionDecoratorApplicationABC,
    SourceDefinitionFunctionActivation,
    SourceFunctionActivationABC,
    SourceFunctionEntry,
    SourceModuleExecution,
)
from test_source_function_result import execution


def controlled_execution(source: str) -> SourceModuleExecution:
    original = execution(source)
    return SourceModuleExecution(
        NoninterferingSourceModuleEntryPremise(
            source=original.source,
            native_island=original.initial,
            bindings=dict(original.entry.initial_entries),
            builtins=original.entry.builtins,
        )
    )


def test_class_decorator_topology_is_derived_from_the_definition_store():
    environment = controlled_execution(
        "def first(value): return value\n"
        "def second(value): return value\n"
        "@first\n@second\nclass Target: pass\n"
    )
    entry = environment.class_entry(environment.module.module.body[-1])
    raw, inner, outer = entry.creation_results
    assert isinstance(inner, SourceDefinitionDecoratorApplicationABC)
    assert isinstance(inner, SourceDefinitionApplicationAuthorityABC)
    assert isinstance(inner, SourceClassDecoratorApplication)
    assert inner.creation is outer.creation is entry
    assert inner.argument is raw
    assert outer.argument is inner
    assert inner.decorator_use is entry.definition.target.decorator_uses[1]
    assert outer.decorator_use is entry.definition.target.decorator_uses[0]
    assert inner.native_value.require_definition_argument() is raw.native_construction
    assert outer.native_value.require_definition_argument() is inner.native_value
    assert outer.native_value is outer.production.value


def test_definition_application_activates_its_exact_source_callee_and_argument():
    environment = controlled_execution(
        "def keep(value):\n    return value\n@keep\nclass Target: pass\n"
    )
    class_entry = environment.class_entry(environment.module.module.body[-1])
    raw, application = class_entry.creation_results
    activation = application.function_activation()
    assert isinstance(activation, SourceFunctionActivationABC)
    assert isinstance(activation, SourceDefinitionFunctionActivation)
    assert activation.callee.proves_same_object(
        environment.capture_definition(environment.module.module.body[0])
    )
    assert activation.initial_entries["value"] is raw
    entry = SourceFunctionEntry(activation)
    prefix = entry.prefix(None, environment.kernel)
    assert isinstance(prefix, DefinitionApplicationPrefix)
    assert prefix.application is application
    assert prefix.declaration is activation.callee.declaration
    assert prefix.endpoint.frame.locals is entry
    assert prefix.intervals[0].context is class_entry.parent_context
    assert activation.entry_continuation.frame.is_body_of(
        activation.callee.native_execution
    )
    with pytest.raises(ValueError, match="Class decorator result remains unproved"):
        class_entry.result()


def test_definition_application_proves_its_returned_argument_identity():
    environment = controlled_execution(
        "def keep(value, replace=False):\n"
        "    if replace:\n"
        "        value = object()\n"
        "    return value\n"
        "@keep\n"
        "class Target: pass\n"
    )
    raw, application = environment.class_entry(
        environment.module.module.body[-1]
    ).creation_results
    returned = application.function_activation().activation.require_returned_parameter_identity(
        "value"
    )
    assert returned is raw


def test_definition_application_rejects_a_replaced_argument_result():
    environment = controlled_execution(
        "def replace(value):\n"
        "    return object()\n"
        "@replace\n"
        "class Target: pass\n"
    )
    _, application = environment.class_entry(
        environment.module.module.body[-1]
    ).creation_results
    with pytest.raises(ValueError, match="different object"):
        application.function_activation().activation.require_returned_parameter_identity(
            "value"
        )


def test_copied_application_and_foreign_callee_cannot_create_an_activation():
    environment = controlled_execution(
        "def keep(value): return value\n"
        "def other(value): return value\n"
        "@keep\nclass Target: pass\n"
    )
    class_entry = environment.class_entry(environment.module.module.body[-1])
    application = class_entry.creation_results[-1]
    with pytest.raises(ValueError, match="original creation chain"):
        replace(application).function_activation()
    foreign = environment.capture_definition(environment.module.module.body[1])
    activation = SourceDefinitionFunctionActivation(application, foreign)
    with pytest.raises(ValueError, match="different source callee"):
        _ = activation.initial_entries


def test_nested_application_does_not_assume_the_inner_result_identity():
    environment = controlled_execution(
        "def first(value): return value\n"
        "def second(value): return value\n"
        "@first\n@second\nclass Target: pass\n"
    )
    raw, inner, outer = environment.class_entry(
        environment.module.module.body[-1]
    ).creation_results
    assert inner.function_activation().initial_entries["value"] is raw
    with pytest.raises(ValueError, match="Class decorator result remains unproved"):
        _ = outer.function_activation().initial_entries


def test_native_dataclass_application_proves_its_in_place_result():
    environment = controlled_execution(
        "from dataclasses import dataclass\n@dataclass\nclass Target: pass\n"
    )
    entry = environment.class_entry(environment.module.module.body[-1])
    application = entry.creation_results[-1]
    assert application.native_value.require_definition_argument() is (
        entry.created_result.native_construction
    )
    with pytest.raises(ValueError, match="callable remains unproved"):
        application.function_activation()
    assert entry.result() is application
    assert application.proves_same_object(entry.created_result)


@pytest.mark.parametrize(
    "decorator",
    ("@dataclass(slots=True)", "@dataclass(frozen=int)"),
)
def test_native_dataclass_application_rejects_unproved_option_semantics(decorator):
    environment = controlled_execution(
        f"from dataclasses import dataclass\n{decorator}\nclass Target: pass\n"
    )
    entry = environment.class_entry(environment.module.module.body[-1])
    with pytest.raises(ValueError, match="Class decorator result remains unproved"):
        entry.result()


@pytest.mark.parametrize(
    "body",
    (
        "return value",
        "value.__annotations__.clear()\n    return value",
    ),
    ids=("returns-input", "mutates-annotations"),
)
def test_unknown_source_class_decorator_remains_fail_closed(body):
    environment = controlled_execution(
        "def transform(value):\n"
        f"    {body}\n"
        "@transform\n"
        "class Target:\n"
        "    field: int\n"
    )
    entry = environment.class_entry(environment.module.module.body[-1])
    with pytest.raises(ValueError):
        entry.result()


def test_native_dataclass_processor_rebinding_before_application_is_rejected():
    environment = controlled_execution(
        "import dataclasses\n"
        "from dataclasses import dataclass\n"
        "def replacement(*arguments): return object\n"
        "dataclasses._process_class = replacement\n"
        "@dataclass\n"
        "class Target: pass\n"
    )
    entry = environment.class_entry(environment.module.module.body[-1])
    with pytest.raises(ValueError):
        entry.result()


def test_restoring_a_processor_after_initial_capture_does_not_retroactively_prove_it(
    monkeypatch,
):
    import dataclasses

    original = dataclasses._process_class

    def replacement(*arguments):
        raise AssertionError("The analyzer must not execute a native replacement")

    monkeypatch.setattr(dataclasses, "_process_class", replacement)
    environment = controlled_execution(
        "from dataclasses import dataclass\n@dataclass\nclass Target: pass\n"
    )
    monkeypatch.setattr(dataclasses, "_process_class", original)
    entry = environment.class_entry(environment.module.module.body[-1])
    with pytest.raises(ValueError, match="Class decorator result remains unproved"):
        entry.result()


@pytest.mark.parametrize("kind", ("class", "function"))
def test_compiler_version_independent_application_path_uses_owned_topology(kind):
    """This test executes unchanged in the CPython 3.11 and 3.14 CI jobs."""
    source = (
        "@staticmethod\n@classmethod\nclass Target: pass\n"
        if kind == "class"
        else "@staticmethod\n@classmethod\ndef Target(): pass\n"
    )
    environment = controlled_execution(source)
    node = environment.module.module.body[0]
    if kind == "class":
        results = environment.class_entry(node).creation_results
    else:
        results = environment.capture_definition(node).creation_results
    raw, inner, outer = results
    native_raw = (
        raw.native_construction
        if kind == "class"
        else inner.production.production_at(
            raw.native_execution.require_creation().instruction_offset
        )
    )
    assert inner.native_value.require_definition_argument() is native_raw
    assert outer.native_value.require_definition_argument() is inner.native_value


def test_annotated_function_activation_uses_compiler_predecessor_receipt():
    """Python 3.14 may attach annotations between creation and application."""
    environment = controlled_execution(
        "def annotation(): return object\n"
        "def keep(value): return value\n"
        "@keep\n"
        "def Target(value: annotation()): pass\n"
    )
    raw, application = SourceCreatedFunctionCapture(
        environment, environment.module.module.body[-1]
    ).creation_results
    native = raw.native_execution
    receipt = native.require_applications()[0]
    assert receipt.argument is native.require_creation()
    assert application.native_value is receipt.operand_in(application.production)
    assert application.function_activation().initial_entries["value"] is raw
