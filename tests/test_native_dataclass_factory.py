"""Dataclass factory creation is distinct from applying its returned decorator."""

import builtins
import dataclasses
import subprocess
import sys
from types import FunctionType

import pytest

from nominal_refactor_advisor.captured_reference import InitialNativeIsland
from nominal_refactor_advisor.native_call import (
    NativeDataclassFactoryCall,
    NativeReturnedClosureFactorySource,
)
from nominal_refactor_advisor.native_compilation import NativePythonCompilation
from nominal_refactor_advisor.lexical_bindings import FunctionParameterSource
from nominal_refactor_advisor.native_declarations import DataclassRuntimeDeclaration
from nominal_refactor_advisor.source_entry import ImportedSourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.carrier_expansion import DeclaredCarrierExpansionBuilder
from nominal_refactor_advisor.parameter_conveyor import (
    ClosedParameterConveyorComponentBuilder,
)
from test_empty_product_domain import DataclassEntryRepository
from test_carrier_expansion import _closed_expansion_source
from test_parameter_conveyor import _base_source
from test_product_flow_authority import _module
from test_source_function_result import execution


def factory_environment(arguments):
    source = f"from dataclasses import dataclass\nheld = dataclass({arguments})\n"
    original = execution(source)
    island = InitialNativeIsland((builtins, dataclasses))
    entry = ImportedSourceModuleEntryPremise.from_standard_source_loader(
        original.source, island, island.namespace_for_storage(vars(builtins))
    )
    node = original.module.module.body[-1].value
    context, call = original.source_call(node)
    operation = original.source_operation(context, call)
    return SourceModuleExecution(entry), operation


def returned_closure_proof(source):
    compilation = NativePythonCompilation(source, "native_closure_factory.py")
    namespace = {}
    exec(compilation.compile(), namespace)
    function = namespace["factory"]
    return NativeReturnedClosureFactorySource(
        function,
        compilation.function_definition(function),
    )


@pytest.mark.parametrize(
    "arguments", ("", "frozen=True", "None", "repr=False, eq=False", "frozen=int")
)
def test_factory_uses_original_call_and_declared_signature_without_invoking_python(
    arguments,
):
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            f"import dataclasses, types\nassert type(dataclasses.dataclass({arguments})) is types.FunctionType",
        ],
        check=True,
        timeout=10,
    )
    environment, operation = factory_environment(arguments)
    authority = environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    )
    assert type(authority) is NativeDataclassFactoryCall
    assert authority.operation is operation
    calls = []
    native_code = dataclasses.dataclass.__code__

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is native_code:
            calls.append(frame)

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        authority.require_closed()
        assert authority.result() is authority
    finally:
        sys.setprofile(previous)
    assert not calls
    assert authority.native_type is FunctionType
    assert environment.capture_value(operation.node).native_type is FunctionType
    with pytest.raises(ValueError):
        authority.require_native(authority.native_declarations)
    with pytest.raises(ValueError):
        authority.require_release()
    assert not environment._pending


def test_current_native_source_proves_factory_behavior_without_a_runtime_condition():
    environment, operation = factory_environment("frozen=True")
    environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    ).require_closed()
    assert not environment._pending


@pytest.mark.parametrize(
    "arguments",
    (
        "int",
        "cls=int",
        "cls=None",
        "None, None",
        "unknown=True",
        "None, cls=None",
        "**{}",
        "*()",
    ),
)
def test_factory_does_not_admit_class_application_or_invalid_binding(arguments):
    environment, operation = factory_environment(arguments)
    with pytest.raises(ValueError):
        environment.call_authority(
            environment.context_for_owner(operation.owner), operation.event
        ).require_closed()
    assert not environment._pending


def test_factory_creation_does_not_prove_returned_decorator_application():
    environment, operation = factory_environment("frozen=True")
    factory = environment.capture_value(operation.node)
    factory.require_closed()
    # A function result is not itself the native dataclass declaration, nor does
    # its type authorize calling it using the factory's original call receipt.
    with pytest.raises(ValueError):
        factory.call_authority(
            environment, environment.context_for_owner(operation.owner), operation.event
        )


@pytest.mark.parametrize("arguments", ("frozen=missing", "frozen=missing()"))
def test_unknown_or_failing_argument_evaluation_is_not_hidden_by_factory_behavior(
    arguments,
):
    environment, operation = factory_environment(arguments)
    with pytest.raises(ValueError):
        environment.call_authority(
            environment.context_for_owner(operation.owner), operation.event
        ).require_closed()
    assert not environment._pending


def test_every_runtime_keyword_is_derived_from_the_native_declaration():
    declaration = DataclassRuntimeDeclaration.DATACLASS.native_declaration.node
    keywords = tuple(
        parameter.argument.arg
        for parameter in FunctionParameterSource.from_arguments(declaration.args)
        if parameter.kind.accepts_keyword and parameter.default is not None
    )
    arguments = ", ".join(f"{name}=None" for name in keywords)
    environment, operation = factory_environment(arguments)
    authority = environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    )
    authority.require_closed()
    assert (
        tuple(
            argument.parameter_name for argument in authority.bound_arguments.arguments
        )
        == keywords
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            f"import dataclasses, types\nassert type(dataclasses.dataclass({arguments})) is types.FunctionType",
        ],
        check=True,
        timeout=10,
    )


def test_native_source_proof_applies_to_a_fresh_original_activation():
    environment, operation = factory_environment("frozen=True")
    fresh_entry = ImportedSourceModuleEntryPremise.from_standard_source_loader(
        environment.source,
        environment.entry.initial,
        environment.entry.initial.namespace_for_storage(vars(builtins)),
    )
    fresh = SourceModuleExecution(fresh_entry)
    fresh.call_authority(
        fresh.context_for_owner(operation.owner), operation.event
    ).require_closed()
    assert not fresh._pending


@pytest.mark.parametrize(
    "source",
    (
        "def factory(selector=None, *, held=None):\n"
        "    observed.append('effect')\n"
        "    def wrap(value):\n"
        "        return held\n"
        "    if selector is None:\n"
        "        return wrap\n"
        "    return wrap(selector)\n",
        "def factory(selector=None, *, held=None):\n"
        "    def wrap(value):\n"
        "        return value\n"
        "    if selector is None:\n"
        "        return wrap\n"
        "    return wrap(selector)\n",
        "def factory(selector=None, *, held=None):\n"
        "    def wrap(value=held):\n"
        "        return value\n"
        "    if selector is None:\n"
        "        return wrap\n"
        "    return wrap(selector)\n",
    ),
    ids=("preceding-effect", "released-parameter", "executable-closure-header"),
)
def test_returned_closure_proof_rejects_open_factory_effects(source):
    proof = returned_closure_proof(source)
    with pytest.raises(ValueError):
        proof.require_returned_closure("selector")


@pytest.mark.parametrize(
    "builder_type,source_factory",
    (
        (ClosedParameterConveyorComponentBuilder, _base_source),
        (DeclaredCarrierExpansionBuilder, _closed_expansion_source),
    ),
)
def test_factory_decorated_product_retains_original_body_observations(
    builder_type, source_factory
):
    module = _module("example", source_factory())
    repository = DataclassEntryRepository.from_modules((module,))
    assert len(repository.product_authorities_by_symbol) == 1
    assert len(builder_type(repository).proven_components()) == 1
    assert not repository.product_runtime_failures_by_authority_symbol
