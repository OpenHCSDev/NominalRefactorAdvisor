"""Dataclass factory creation is distinct from applying its returned decorator."""

import ast
import builtins
import dataclasses
from dataclasses import replace
import subprocess
import sys
from types import FunctionType

import pytest

from nominal_refactor_advisor.captured_reference import InitialNativeIsland
from nominal_refactor_advisor.native_call import NativeDataclassFactoryCall
from nominal_refactor_advisor.lexical_bindings import FunctionParameterSource
from nominal_refactor_advisor.native_declarations import DataclassRuntimeDeclaration
from nominal_refactor_advisor.source_entry import (
    DeclaredNativeOperationBehavior,
    DeclaredOperationCompletion,
    ImportedSourceModuleEntryPremise,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.product_flow import CompactFunctionCall
from nominal_refactor_advisor.carrier_expansion import DeclaredCarrierExpansionBuilder
from nominal_refactor_advisor.parameter_conveyor import (
    ClosedParameterConveyorComponentBuilder,
)
from test_empty_product_domain import DataclassEntryRepository
from test_carrier_expansion import _closed_expansion_source
from test_parameter_conveyor import _base_source
from test_product_flow_authority import _module
from test_source_function_result import execution


def factory_environment(arguments, condition=DeclaredNativeOperationBehavior):
    source = f"from dataclasses import dataclass\nheld = dataclass({arguments})\n"
    original = execution(source)
    island = InitialNativeIsland((builtins, dataclasses))
    entry = ImportedSourceModuleEntryPremise.from_standard_source_loader(
        original.source, island, island.namespace_for_storage(vars(builtins))
    )
    node = original.module.module.body[-1].value
    context, call = original.source_call(node)
    operation = original.source_operation(context, call)
    conditions = ()
    if condition is not None:
        options = (
            {"protocol": NativeDataclassFactoryCall}
            if condition is DeclaredNativeOperationBehavior
            else {}
        )
        conditions = (condition(operation, **options),)
    entry = replace(
        entry,
        bindings=dict(entry.initial_entries),
        declared_operation_conditions=conditions,
    )
    return SourceModuleExecution(entry), operation


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


@pytest.mark.parametrize("condition", (None, DeclaredOperationCompletion))
def test_import_identity_and_completion_do_not_imply_native_behavior(condition):
    environment, operation = factory_environment("frozen=True", condition)
    with pytest.raises(ValueError):
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


def test_behavior_condition_does_not_transfer_to_a_fresh_activation():
    environment, operation = factory_environment("frozen=True")
    fresh_entry = ImportedSourceModuleEntryPremise.from_standard_source_loader(
        environment.source,
        environment.entry.initial,
        environment.entry.initial.namespace_for_storage(vars(builtins)),
    )
    fresh = SourceModuleExecution(fresh_entry)
    with pytest.raises(ValueError):
        fresh.call_authority(
            fresh.context_for_owner(operation.owner), operation.event
        ).require_closed()
    assert not fresh._pending


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
    class FactoryEntryRepository(DataclassEntryRepository):
        @staticmethod
        def source_entry(source):
            entry = DataclassEntryRepository.source_entry(source)
            declaration = next(
                node
                for node in source.module.module.body
                if isinstance(node, ast.ClassDef)
            )
            operation = source.node_operation(
                declaration.decorator_list[0], CompactFunctionCall
            )
            return replace(
                entry,
                bindings=dict(entry.initial_entries),
                declared_operation_conditions=(
                    DeclaredNativeOperationBehavior(
                        operation, protocol=NativeDataclassFactoryCall
                    ),
                ),
            )

    module = _module("example", source_factory())
    repository = FactoryEntryRepository.from_modules((module,))
    assert len(repository.product_authorities_by_symbol) == 1
    assert len(builder_type(repository).proven_components()) == 1
    assert not repository.product_runtime_failures_by_authority_symbol
