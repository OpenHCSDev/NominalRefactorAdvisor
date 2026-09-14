"""Original registry operands are prerequisites, not destination equivalence."""

import ast
import builtins
from copy import deepcopy
from dataclasses import replace

import pytest
import metaclass_registry

from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    InitialNativeIsland,
    OpenCapturedReference,
)
from nominal_refactor_advisor.manual_registry import (
    ConstructorClassKeyEntry,
    DirectManualRegistryComponent,
    SourceClassKeyEntry,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.registry_identity import AutoRegisterClassAuthority
from nominal_refactor_advisor.source_entry import ImportedSourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_source_distinct_item_stores import BASE, execution


def final_entry(environment, entry_type=SourceClassKeyEntry):
    """Use the actual original class and final assignment operand."""
    module = environment.module.module
    original = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Alpha"
    )
    return entry_type(original, ast.Constant("alpha"), module.body[-1].value)


def authored_runtime(source):
    """Execute only the small fixtures authored in this test file."""
    namespace = {}
    exec(compile(source, "<registry-original-value-fixture>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "suffix",
    (
        "tail = Alpha\n",
        "saved = Alpha\ntail = saved\n",
        "saved = Alpha\nAlpha = Beta\ntail = saved\n",
    ),
)
def test_original_direct_and_alias_values_retain_the_actual_creation(suffix):
    source = BASE + suffix
    environment = execution(source)
    entry = final_entry(environment)
    entry.require_original_value(environment)
    context, definition = entry.captured_class(environment).source_definition()
    original = environment.class_entry(entry.class_node)
    assert context is original.parent_context
    assert definition is original.definition
    assert environment.source_operation(context, definition) is original.operation
    assert original.operation.node is entry.class_node
    runtime = authored_runtime(source)
    assert runtime["tail"].__name__ == "Alpha"


def test_rebound_class_name_resolves_beta_but_cannot_claim_alpha_provenance():
    source = BASE + "saved = Alpha\nAlpha = Beta\ntail = Alpha\n"
    environment = execution(source)
    entry = final_entry(environment)
    context, definition = entry.captured_class(environment).source_definition()
    beta = environment.module.module.body[2]
    assert environment.source_operation(context, definition).node is beta
    with pytest.raises(
        ValueError, match="not proved to be the selected class creation"
    ):
        entry.require_original_value(environment)
    runtime = authored_runtime(source)
    assert runtime["tail"] is runtime["Beta"]
    assert runtime["tail"] is not runtime["saved"]


def test_same_geometry_foreign_class_cannot_claim_original_value():
    environment = execution(BASE + "tail = Alpha\n")
    entry = final_entry(environment)
    entry.require_original_value(environment)
    foreign = replace(entry, class_node=deepcopy(entry.class_node))
    assert ast.dump(foreign.class_node, include_attributes=True) == ast.dump(
        entry.class_node, include_attributes=True
    )
    with pytest.raises(ValueError, match="unique actual operation"):
        foreign.require_original_value(environment)


def test_copied_operand_ast_is_not_a_canonical_read():
    environment = execution(BASE + "tail = Alpha\n")
    entry = final_entry(environment)
    with pytest.raises(ValueError):
        replace(entry, value_node=deepcopy(entry.value_node)).require_original_value(
            environment
        )


def test_registry_component_requires_its_actual_original_module():
    source = BASE + "REGISTRY['alpha'] = Alpha\nREGISTRY['beta'] = Beta\n"
    environment = execution(source)
    component = DirectManualRegistryComponent.from_module_anchor(
        environment.module.module, "Alpha"
    )
    component.require_original_entry_values(environment)
    foreign_environment = execution(source)
    with pytest.raises(ValueError, match="original source module"):
        component.require_original_entry_values(foreign_environment)


def test_dictionary_literal_entries_use_the_original_reference_read_authority():
    source = "class Alpha: pass\nclass Beta: pass\nREGISTRY = {'alpha': Alpha, 'beta': Beta}\n"
    environment = execution(source)
    component = DirectManualRegistryComponent.from_module_anchor(
        environment.module.module, "Alpha"
    )
    for entry in component.entries:
        assert entry.value_node in environment.source.reference_reads_by_node
        assert entry.value_node not in environment.source.value_reads_by_node
        environment.capture(entry.value_node).source_definition()
    component.require_original_entry_values(environment)
    runtime = authored_runtime(source)
    assert runtime["REGISTRY"]["alpha"] is runtime["Alpha"]
    assert runtime["REGISTRY"]["beta"] is runtime["Beta"]


def test_original_value_refusal_preserves_the_existing_execution_cause():
    environment = execution(BASE + "unknown()\ntail = Alpha\n")
    entry = final_entry(environment)
    capture = entry.captured_class(environment)
    assert isinstance(capture, OpenCapturedReference)
    with pytest.raises(CapturedReferenceRejection) as direct:
        capture.require_closed()
    with pytest.raises(CapturedReferenceRejection) as original:
        entry.require_original_value(environment)
    assert original.value.violation is direct.value.violation
    assert original.value.__cause__ is direct.value.__cause__


@pytest.mark.parametrize("aliased", (False, True))
def test_original_plain_constructor_is_executed_after_class_identity(aliased):
    source = "class Alpha: pass\n"
    source += "saved = Alpha\n" if aliased else ""
    source += "tail = saved()\n" if aliased else "tail = Alpha()\n"
    environment = execution(source)
    entry = final_entry(environment, ConstructorClassKeyEntry)
    entry.require_original_value(environment)
    context, definition = entry.captured_class(environment).source_definition()
    assert environment.source_operation(context, definition).node is entry.class_node
    environment.capture_value(entry.value_node).require_closed()
    runtime = authored_runtime(source)
    assert type(runtime["tail"]) is runtime["Alpha"]


@pytest.mark.parametrize(
    "body",
    (
        "def __init__(self):\n        self.changed = True",
        "def __new__(cls):\n        return object()",
    ),
)
def test_known_constructor_class_does_not_authorize_unproved_call(body):
    source = "class Alpha:\n    " + body + "\ntail = Alpha()\n"
    environment = execution(source)
    entry = final_entry(environment, ConstructorClassKeyEntry)
    # This proves the actual callee first, so the failure cannot be a spelling
    # or unrelated import refusal.
    SourceClassKeyEntry.require_original_value(entry, environment)
    with pytest.raises(ValueError):
        environment.capture_value(entry.value_node).require_closed()
    with pytest.raises(ValueError):
        entry.require_original_value(environment)
    runtime = authored_runtime(source)
    assert runtime["tail"] is not None


def test_constructor_alias_rebound_to_another_class_does_not_claim_alpha():
    source = "class Alpha: pass\nclass Beta: pass\nsaved = Alpha\nAlpha = Beta\ntail = Alpha()\n"
    environment = execution(source)
    entry = final_entry(environment, ConstructorClassKeyEntry)
    with pytest.raises(
        ValueError, match="not proved to be the selected class creation"
    ):
        entry.require_original_value(environment)
    runtime = authored_runtime(source)
    assert type(runtime["tail"]) is runtime["Beta"]


def test_metaclass_spelling_cannot_override_known_native_mismatch():
    source = (
        "AutoRegisterMeta = type\nclass Handler(metaclass=AutoRegisterMeta): pass\n"
    )
    environment = execution(source)
    node = environment.module.module.body[-1]
    environment.capture(node.keywords[0].value).require_native_identity(
        NativeDeclaration(type)
    )
    with pytest.raises(ValueError, match="required native declaration"):
        AutoRegisterClassAuthority(node).require_native_metaclass(environment)
    assert type(authored_runtime(source)["Handler"]) is type


def test_genuine_metaclass_import_remains_an_explicit_entry_obligation():
    source = "from metaclass_registry import AutoRegisterMeta\nclass Handler(metaclass=AutoRegisterMeta): pass\n"
    environment = execution(source)
    node = environment.module.module.body[-1]
    with pytest.raises(ValueError) as caught:
        AutoRegisterClassAuthority(node).require_native_metaclass(environment)
    causes = []
    error = caught.value
    while error is not None:
        causes.append(error)
        error = error.__cause__
    assert any(
        isinstance(cause, CapturedReferenceRejection)
        and cause.violation is CapturedReferenceViolation.UNADMITTED_IMPORT
        for cause in causes
    )
    # The valid authored program does not justify assuming the analyzer's
    # default entry had already admitted this module association.
    runtime = authored_runtime(source)
    assert type(runtime["Handler"]) is runtime["AutoRegisterMeta"]


@pytest.mark.parametrize(
    "binding, imported",
    (("AutoRegisterMeta", "AutoRegisterMeta"), ("Meta", "AutoRegisterMeta as Meta")),
)
def test_original_metaclass_identity_with_an_explicit_native_module_association(
    binding, imported
):
    source = (
        f"from metaclass_registry import {imported}\n"
        f"class Handler(metaclass={binding}): pass\n"
    )
    projection = execution(source).source
    module = projection.module
    initial = InitialNativeIsland((builtins, metaclass_registry))
    entry = ImportedSourceModuleEntryPremise.from_standard_source_loader(
        projection, initial, initial.namespace_for_storage(vars(builtins))
    )
    environment = SourceModuleExecution(entry)
    original_import, original_class = module.module.body
    environment.require_import(original_import)
    operand = original_class.keywords[0].value
    assert operand in projection.reference_reads_by_node
    environment.capture(operand).require_native_identity(
        NativeDeclaration(metaclass_registry.AutoRegisterMeta)
    )
    AutoRegisterClassAuthority(original_class).require_native_metaclass(environment)
    # Proved imported identity does not close actual native class construction.
    with pytest.raises(
        ValueError,
        match="^Native operation needs an explicit entry condition$",
    ):
        environment.require_class_creation(original_class)
    runtime = authored_runtime(source)
    assert type(runtime["Handler"]) is metaclass_registry.AutoRegisterMeta


def test_missing_metaclass_operand_cannot_invent_a_registration_authority():
    environment = execution("class Handler: pass\n")
    original_class = environment.module.module.body[0]
    environment.require_class_creation(original_class)
    with pytest.raises(ValueError, match="one actual metaclass operand"):
        AutoRegisterClassAuthority(original_class).require_native_metaclass(environment)
