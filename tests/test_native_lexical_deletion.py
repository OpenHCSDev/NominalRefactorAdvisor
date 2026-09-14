"""Native deletion closes presence, release, absence and inventory together."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import NamespaceMemberInventory
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    parsed = ParsedModule(Path("delete.py"), "delete", False, ast.parse(source), source)
    return SourceModuleExecution.from_module(parsed)


def native(source, check=""):
    subprocess.run([sys.executable, "-c", source + check], check=True)


@pytest.mark.parametrize("definition", ("class Original: pass", "def Original(): pass"))
def test_surviving_alias_permits_deletion_and_inventory_removes_the_key(definition):
    source = f"{definition}\nsaved = Original\ndel Original\nresult = saved\n"
    native(
        source,
        "assert saved.__name__ == 'Original'\nassert 'Original' not in globals()\n",
    )
    owner = execution(source)
    result = owner.capture(owner.module.module.body[-1].value)
    result.require_closed()
    _, binding = result.source_definition()
    assert binding.target.bound_name == "Original"
    prefix = owner.required_prefix(owner.entry.context, None)
    assert (
        owner.kernel._namespace_resolution(owner.entry, "Original", prefix, frozenset())
        is None
    )
    names = NamespaceMemberInventory(owner.kernel, owner.entry, prefix).names
    assert "saved" in names and "Original" not in names


def test_deleted_global_name_can_fall_through_to_its_actual_builtin():
    source = "class object: pass\nsaved = object\ndel object\nobject\n"
    native(
        source,
        "import builtins\nassert object is builtins.object\nassert saved is not object\n",
    )
    owner = execution(source)
    owner.capture(owner.module.module.body[-1].value).require_native_identity(
        NativeDeclaration(object)
    )


@pytest.mark.parametrize("name", ("missing", "object"))
def test_delete_requires_destination_presence_not_outer_lookup(name):
    source = f"del {name}\nobject\n"
    outcome = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    assert outcome.returncode != 0 and "NameError" in outcome.stderr
    owner = execution(source)
    with pytest.raises(ValueError):
        owner.required_prefix(owner.entry.context, None)


def test_class_local_delete_does_not_delete_its_global_binding():
    source = (
        "class Original: pass\nclass Container:\n"
        "    local = Original\n    del local\nContainer\n"
    )
    native(
        source,
        "assert Original.__name__ == 'Original'\nassert not hasattr(Container, 'local')\n",
    )
    owner = execution(source)
    owner.capture(owner.module.module.body[-1].value).require_closed()
    entry = owner.class_entry(owner.module.module.body[1])
    assert (
        "local"
        not in NamespaceMemberInventory(
            owner.kernel, entry, entry.completion_prefix
        ).names
    )


def test_class_global_delete_uses_global_storage_but_local_alias_retains_value():
    source = (
        "class Original: pass\nclass Container:\n    global Original\n"
        "    saved = Original\n    del Original\nresult = Container\n"
    )
    native(
        source,
        "assert 'Original' not in globals()\nassert Container.saved.__name__ == 'Original'\n",
    )
    owner = execution(source)
    owner.capture(owner.module.module.body[-1].value).require_closed()
    prefix = owner.required_prefix(owner.entry.context, None)
    assert (
        "Original"
        not in NamespaceMemberInventory(owner.kernel, owner.entry, prefix).names
    )


def test_class_delete_cannot_use_a_global_only_binding_as_its_destination():
    source = "class Original: pass\nclass Container:\n    del Original\nContainer\n"
    outcome = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    assert outcome.returncode != 0 and "NameError" in outcome.stderr
    owner = execution(source)
    with pytest.raises(ValueError):
        owner.capture(owner.module.module.body[-1].value).require_closed()


@pytest.mark.parametrize(
    "bindings",
    ("saved = Original\ndel Original, saved", "saved = Original\ndel saved, Original"),
)
def test_multi_delete_does_not_reuse_a_previously_deleted_alias(bindings):
    source = f"class Original: pass\n{bindings}\nobject\n"
    native(source)
    owner = execution(source)
    with pytest.raises(ValueError):
        owner.required_prefix(owner.entry.context, None)


def test_multi_delete_with_a_surviving_third_reference_closes():
    source = (
        "class Original: pass\nfirst = Original\nsaved = Original\n"
        "del Original, first\nsaved\n"
    )
    native(source, "assert saved.__name__ == 'Original'\n")
    owner = execution(source)
    owner.capture(owner.module.module.body[-1].value).require_closed()


def test_reinstall_after_delete_proves_the_new_binding_without_old_value_release():
    source = (
        "class Original: pass\nsaved = Original\ndel Original\n"
        "Original = object\nOriginal\n"
    )
    native(source, "assert saved is not Original\n")
    owner = execution(source)
    owner.capture(owner.module.module.body[-1].value).require_closed()
    prefix = owner.required_prefix(owner.entry.context, None)
    assert (
        "Original" in NamespaceMemberInventory(owner.kernel, owner.entry, prefix).names
    )


@pytest.mark.parametrize(
    "hook,completion",
    (
        ("__init_subclass__", "class Child(Owner): pass\nresult = Child\n"),
        ("__init__", "result = Owner()\n"),
        ("__new__", "result = Owner()\n"),
    ),
)
@pytest.mark.parametrize("deleted", (False, True))
def test_native_protocol_uses_final_namespace_not_deleted_hook(
    hook, completion, deleted
):
    source = (
        f"class Owner:\n    def {hook}(self): raise RuntimeError('hook called')\n"
        f"    saved = {hook}\n" + (f"    del {hook}\n" if deleted else "") + completion
    )
    outcome = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    owner = execution(source)
    if deleted:
        assert outcome.returncode == 0, outcome.stderr
        owner.required_prefix(owner.entry.context, None)
    else:
        assert outcome.returncode != 0 and "hook called" in outcome.stderr
        with pytest.raises(ValueError):
            owner.required_prefix(owner.entry.context, None)
