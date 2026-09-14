"""Import installation and cleanup require the original request and source cut."""

from copy import copy
from types import MappingProxyType

import pytest

from nominal_refactor_advisor.native_compilation import NativeDiscardValue
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import (
    SourceCompletionResolver,
    SourceMemberImportStore,
    SourceModuleImportStore,
)
from test_positioned_import_effects import execution, imports
from test_source_function_result import native


def selected(
    statement="from builtins import object as first, property as last",
    *,
    in_class=True,
    prefix="",
):
    source = (
        prefix + ("class Target:\n    " + statement if in_class else statement) + "\n"
    )
    env = execution(source)
    operation = imports(env)[-1]
    store = SourceCompletionResolver(env).resolve(operation.event)
    completed = env.required_prefix(store.native_frame_context, None)
    return env, store, completed


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize(
    "statement,kind",
    (
        ("import builtins as last", SourceModuleImportStore),
        ("import builtins as first, typing as last", SourceModuleImportStore),
        ("from builtins import property as last", SourceMemberImportStore),
        (
            "from builtins import object as first, property as last",
            SourceMemberImportStore,
        ),
    ),
)
def test_original_import_installation_and_native_completion(in_class, statement, kind):
    env, store, prefix = selected(statement, in_class=in_class)
    assert isinstance(store, kind)
    binding = store.require_native_installation(prefix)
    assert binding is store.production.binding
    returned = store.return_continuation(prefix)
    assert returned is store.production.require_return()
    assert returned.continues(binding)
    offset = store.native_completion_offset(prefix)
    if kind is SourceMemberImportStore:
        assert offset > binding.instruction_offset
        cleanup = returned.production_at(offset)
        assert isinstance(cleanup, NativeDiscardValue)
        assert cleanup.inputs[0] is store.production.value.inputs[0]
    else:
        assert offset == binding.instruction_offset
    if in_class:
        entry = env.class_entry(env.module.module.body[-1])
        assert entry.completed is None
        assert entry.native_tail.completed is None
        if kind is SourceMemberImportStore:
            entry.native_tail.require_member("last").require_native_identity(
                NativeDeclaration(property)
            )
    native(
        env.module.source
        + (
            "assert Target.last is not None\n"
            if in_class
            else "assert last is not None\n"
        )
    )


def test_module_global_importer_shadow_does_not_replace_builtin_importer():
    env, store, prefix = selected(prefix="__import__ = object\n")
    store.return_continuation(prefix)
    native(env.module.source + "assert Target.last is property\n")


def test_member_import_global_destination_retains_the_original_class_frame():
    env, store, prefix = selected(
        "global last\n    from builtins import property as last"
    )
    installed = store.require_native_installation(prefix)
    assert installed.operation.name == "STORE_GLOBAL"
    store.return_continuation(prefix)
    assert env.class_entry(env.module.module.body[-1]).completed is None
    native(
        env.module.source
        + "assert last is property\nassert 'last' not in vars(Target)\n"
    )


@pytest.mark.parametrize(
    "damage",
    (
        "module",
        "level",
        "from_list",
        "member",
        "module_operand",
        "frame",
        "cleanup_operand",
        "cleanup_inventory",
        "registration",
    ),
)
def test_warm_import_cannot_borrow_substituted_request_or_cleanup(damage):
    env, store, prefix = selected()
    end = store.native_completion_offset(prefix)
    production = store.production
    member = production.value
    (module,) = member.inputs
    returned = production.require_return()
    cleanup = returned.production_at(end)
    if damage == "module":
        object.__setattr__(module, "name", "typing")
    elif damage == "level":
        object.__setattr__(module.inputs[0], "value", True)
    elif damage == "from_list":
        object.__setattr__(module.inputs[1], "value", ("property", "object"))
    elif damage == "member":
        object.__setattr__(member, "name", "object")
    elif damage == "module_operand":
        object.__setattr__(member, "inputs", (copy(module),))
    elif damage == "frame":
        object.__setattr__(production, "frame", copy(production.frame))
    elif damage == "cleanup_operand":
        object.__setattr__(cleanup, "inputs", (copy(module),))
    elif damage == "cleanup_inventory":
        object.__setattr__(
            returned,
            "values",
            tuple(
                copy(value) if value is cleanup else value for value in returned.values
            ),
        )
    else:
        object.__setattr__(
            env.initial,
            "modules_by_name",
            MappingProxyType(
                {
                    name: value
                    for name, value in env.initial.modules_by_name.items()
                    if name != "builtins"
                }
            ),
        )
    with pytest.raises(ValueError):
        store.native_completion_offset(prefix)


def test_foreign_copied_or_pre_installation_prefix_cannot_supply_import_completion():
    env, store, prefix = selected()
    _, _, foreign = selected()
    for cut in (foreign, copy(prefix), store.native_frame_prefix):
        with pytest.raises(ValueError):
            store.return_continuation(cut)
    with pytest.raises(ValueError):
        SourceCompletionResolver(env).resolve(copy(store.binding))


def test_earlier_alias_installation_is_not_the_final_cleanup_boundary():
    env, _, prefix = selected()
    first = SourceCompletionResolver(env).resolve(imports(env)[0].event)
    first.require_native_installation(prefix)
    with pytest.raises(ValueError, match="final original alias"):
        first.return_continuation(prefix)


def test_failed_later_import_does_not_remove_an_earlier_installed_alias():
    env = execution("from builtins import property as first, missing as last\n")
    first, last = imports(env)
    store = SourceCompletionResolver(env).resolve(first.event)
    before_last = env.required_prefix(store.native_frame_context, last.position)
    store.require_native_installation(before_last)
    with pytest.raises(ValueError):
        env.required_prefix(store.native_frame_context, None)


def test_conditional_native_return_does_not_admit_unknown_source_effects():
    env = execution("from builtins import property as chosen\nunknown()\n")
    store = SourceCompletionResolver(env).resolve(imports(env)[0].event)
    # Native call/discard transfers can be observed without knowing their effects.
    store.production.require_return()
    with pytest.raises(ValueError):
        env.required_prefix(store.native_frame_context, None)


def test_registered_module_retention_is_not_available_for_unknown_module():
    env, _, prefix = selected()
    with pytest.raises(ValueError):
        env.initial.require_registered_module_retention("unregistered_module", prefix)
