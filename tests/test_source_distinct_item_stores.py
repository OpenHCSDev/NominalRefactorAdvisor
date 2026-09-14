"""Distinct native string-key stores share existing namespace query evidence."""

import ast
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.manual_registry import DirectManualRegistryComponent
from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import CompactItemTarget
from nominal_refactor_advisor.source_execution import SourceModuleExecution

BASE = "REGISTRY = {}\nclass Alpha: pass\nclass Beta: pass\n"


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("item_stores.py"), "item_stores", False, ast.parse(source), source
        )
    )


def item_statements(environment):
    return tuple(
        statement
        for statement in environment.module.module.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Subscript)
    )


def assert_closed_stores(environment):
    for statement in item_statements(environment):
        environment.require_item_write(statement.targets[0])


@pytest.mark.parametrize("count", (1, 2))
@pytest.mark.parametrize("aliased", (False, True))
def test_first_and_distinct_key_stores_close_actual_original_class_captures(
    count, aliased
):
    source = BASE + ("alias = REGISTRY\n" if aliased else "")
    receiver = "alias" if aliased else "REGISTRY"
    for key, value in (("alpha", "Alpha"), ("beta", "Beta"))[:count]:
        source += f"{receiver}[{key!r}] = {value}\n"
    environment = execution(source)
    assert_closed_stores(environment)
    classes = {
        node.name: node
        for node in environment.module.module.body
        if isinstance(node, ast.ClassDef)
    }
    for statement in item_statements(environment):
        owner = environment.class_entry(
            classes[statement.value.id]
        ).definition.target.owner
        environment.capture_value(statement.value).require_definition_identity(owner)
    native = subprocess.run(
        [
            sys.executable,
            "-c",
            source
            + "assert REGISTRY['alpha'] is Alpha\n"
            + ("assert REGISTRY['beta'] is Beta\n" if count == 2 else ""),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr


def test_registry_component_can_demand_each_retained_original_value_operand():
    environment = execution(
        BASE + "REGISTRY['alpha'] = Alpha\nREGISTRY['beta'] = Beta\n"
    )
    component = DirectManualRegistryComponent.from_module_anchor(
        environment.module.module, "Alpha"
    )
    for entry in component.entries:
        owner = environment.class_entry(entry.class_node).definition.target.owner
        environment.capture_value(entry.value_node).require_definition_identity(owner)


def test_same_key_overwrite_keeps_the_original_receiver_and_latest_class():
    environment = execution(
        BASE + "REGISTRY['alpha'] = Alpha\nREGISTRY['alpha'] = Beta\n"
    )
    first, second = item_statements(environment)
    environment.require_item_write(first.targets[0])
    environment.require_item_write(second.targets[0])
    namespace = environment.capture_value(first.targets[0].value).dictionary_namespace(
        environment.initial
    )
    value = environment.kernel._namespace_resolution(
        namespace,
        "alpha",
        environment.required_prefix(environment.entry.context, None),
        frozenset(),
    )
    beta = environment.class_entry(environment.module.module.body[2]).definition
    assert value.source_definition()[1] is beta


def test_existing_query_proves_unused_key_absent_and_original_installed_value():
    environment = execution(
        BASE + "REGISTRY['alpha'] = Alpha\nREGISTRY['beta'] = Beta\ntail = REGISTRY\n"
    )
    assert_closed_stores(environment)
    namespace = environment.capture_value(
        environment.module.module.body[0].value
    ).dictionary_namespace(environment.initial)
    read = environment.source.value_reads_by_node[
        environment.module.module.body[-1].value
    ]
    absent = environment.kernel._slot(
        namespace, "unused", read.context, read.use.position, frozenset()
    )
    matching = environment.kernel._slot(
        namespace, "alpha", read.context, read.use.position, frozenset()
    )
    assert absent is None
    original = environment.class_entry(environment.module.module.body[1]).definition
    assert matching.source_definition()[1] is original


@pytest.mark.parametrize("key", ("1.0", "...", "('alpha',)", "unknown()", "missing"))
def test_unknown_or_nonscalar_keys_do_not_gain_setter_admission(key):
    environment = execution(BASE + f"REGISTRY[{key}] = Alpha\n")
    (statement,) = item_statements(environment)
    # The actual RHS precedes evaluation of the unsupported target/index.
    owner = environment.class_entry(
        environment.module.module.body[1]
    ).definition.target.owner
    environment.capture_value(statement.value).require_definition_identity(owner)
    with pytest.raises(ValueError):
        environment.require_item_write(statement.targets[0])


@pytest.mark.parametrize("value", ("'value'", "Alpha"))
def test_temporary_receiver_disposal_is_not_proved_by_installed_rhs(value):
    environment = execution(BASE + f"{{}}['key'] = {value}\n")
    (statement,) = item_statements(environment)
    environment.capture_value(statement.value).require_closed()
    with pytest.raises(ValueError, match="independent retained reference"):
        environment.require_item_write(statement.targets[0])


def test_key_side_rebinding_does_not_supply_receiver_liveness():
    source = BASE + "REGISTRY[(REGISTRY := 'key')] = Alpha\n"
    environment = execution(source)
    (statement,) = item_statements(environment)
    with pytest.raises(ValueError):
        environment.require_item_write(statement.targets[0])
    native = subprocess.run(
        [sys.executable, "-c", source + "assert REGISTRY == 'key'\n"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr


def test_rhs_capture_precedes_receiver_and_key_even_if_target_is_unsupported():
    environment = execution(BASE + "REGISTRY[(key := 'alpha')] = Alpha\n")
    (statement,) = item_statements(environment)
    operation = next(
        site
        for site in environment.source.operations
        if site.node is statement.targets[0]
        and isinstance(getattr(site.event, "target", None), CompactItemTarget)
    )
    mutation = operation.event
    assert mutation.value_use.position.dominates(mutation.target.receiver_use.position)
    assert mutation.target.receiver_use.position.dominates(
        mutation.target.index_use.position
    )
    assert mutation.target.index_use.position.dominates(mutation.position)
    owner = environment.class_entry(
        environment.module.module.body[1]
    ).definition.target.owner
    environment.kernel.assignment_value(
        environment.entry.context, mutation
    ).require_definition_identity(owner)


@pytest.mark.parametrize(
    "access, expected", (("property", object), ("builtins.property", property))
)
def test_raw_globals_write_affects_global_lookup_but_not_distinct_builtin_storage(
    access,
    expected,
):
    environment = execution(
        "import builtins\nnamespace = globals()\nnamespace['property'] = object\nresult = "
        + access
        + "\n"
    )
    assert_closed_stores(environment)
    result = environment.capture_value(environment.module.module.body[-1].value)
    result.require_native_identity(NativeDeclaration(expected))


@pytest.mark.parametrize(
    "access, expected",
    (("saved", property), ("property", object), ("staticmethod", staticmethod)),
)
def test_raw_builtin_storage_mutation_shares_alias_and_unrelated_key_semantics(
    access, expected
):
    source = (
        "import builtins\nsaved = builtins.property\nnamespace = vars(builtins)\nnamespace['property'] = object\nresult = "
        + access
        + "\n"
    )
    environment = execution(source)
    assert_closed_stores(environment)
    result = environment.capture_value(environment.module.module.body[-1].value)
    result.require_native_identity(NativeDeclaration(expected))


def test_native_raw_storage_control_is_isolated_from_analyser_builtins():
    source = (
        "import builtins\nsaved = builtins.property\nnamespace = vars(builtins)\n"
        "namespace['property'] = object\n"
        "assert saved is not property\nassert property is object\n"
        "assert staticmethod is builtins.staticmethod\n"
    )
    native = subprocess.run(
        [sys.executable, "-c", source], text=True, capture_output=True, check=False
    )
    assert native.returncode == 0, native.stderr
    assert property.__name__ == "property"


@pytest.mark.parametrize(
    "access, expected", (("saved", object), ("builtins.property", property))
)
def test_raw_globals_overwrite_supersedes_selected_lexical_binding(access, expected):
    source = (
        "import builtins\nsaved = builtins.property\nnamespace = globals()\n"
        "namespace['saved'] = object\nresult = " + access + "\n"
    )
    environment = execution(source)
    assert_closed_stores(environment)
    result = environment.capture_value(environment.module.module.body[-1].value)
    result.require_native_identity(NativeDeclaration(expected))


def test_branch_dependent_creation_does_not_gain_definite_store_admission():
    environment = execution(
        "if unknown:\n    REGISTRY = {}\nREGISTRY['key'] = 'value'\n"
    )
    (statement,) = item_statements(environment)
    with pytest.raises(ValueError):
        environment.require_item_write(statement.targets[0])


def test_foreign_source_target_cannot_register_a_store():
    environment = execution(BASE + "REGISTRY['alpha'] = Alpha\n")
    foreign = ast.parse("REGISTRY['alpha'] = Alpha").body[0].targets[0]
    with pytest.raises(ValueError):
        environment.require_item_write(foreign)


def test_forged_mutation_cannot_supply_original_rhs_store():
    environment = execution(BASE + "REGISTRY['alpha'] = Alpha\n")
    statement = item_statements(environment)[0]
    operation = next(
        site
        for site in environment.source.operations
        if site.node is statement.targets[0]
        and isinstance(getattr(site.event, "target", None), CompactItemTarget)
    )
    environment.entry.__dict__["source"] = replace(
        environment.source,
        operations=tuple(
            replace(site, event=replace(site.event)) if site is operation else site
            for site in environment.source.operations
        ),
    )
    with pytest.raises(ValueError):
        environment.require_item_write(statement.targets[0])


@pytest.mark.parametrize("source", ("REGISTRY['alpha'] += 1", "del REGISTRY['alpha']"))
def test_unsupported_mutation_kind_does_not_become_plain_store(source):
    environment = execution(BASE + source + "\n")
    target = (
        environment.module.module.body[-1].target
        if isinstance(environment.module.module.body[-1], ast.AugAssign)
        else environment.module.module.body[-1].targets[0]
    )
    with pytest.raises(ValueError):
        environment.require_item_write(target)


def test_unsupported_backend_cannot_admit_native_setter(monkeypatch):
    environment = execution(BASE + "REGISTRY['alpha'] = Alpha\n")
    statement = item_statements(environment)[0]
    # Establish earlier supported creations first, not the store itself.
    environment.capture_value(environment.module.module.body[0].value).require_closed()
    environment.capture_value(statement.value).require_closed()
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    with pytest.raises(ValueError):
        environment.require_item_write(statement.targets[0])
