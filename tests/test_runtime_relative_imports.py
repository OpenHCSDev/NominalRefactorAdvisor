"""Runtime package reads, not nominal source paths, select relative imports."""

import ast
from pathlib import Path
import subprocess
import sys
import typing

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.lexical_bindings import ImportFromModuleName
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import (
    SourceCompletionResolver,
    SourceModuleExecution,
)
from test_positioned_import_effects import imports


def environment(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("relative_probe.py"),
            "typing.synthetic",
            False,
            ast.parse(source),
            source,
        )
    )


def test_package_change_cannot_borrow_the_source_catalogue_import_target():
    source = '__package__ = "builtins"\nfrom . import Any as chosen\n'
    env = environment(source)
    operation = imports(env)[0]
    assert operation.event.target.origin.requested_module_name == "typing"
    with pytest.raises(ValueError):
        env.require_import_operation(operation)
    with pytest.raises(ValueError):
        env.required_prefix(env.source.module_context, None)
    actual = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    assert actual.returncode != 0
    assert "ImportError" in actual.stderr


@pytest.mark.parametrize(
    "package,relative,member,expected",
    (
        ("builtins", ".", "object", object),
        ("builtins.child", "..", "object", object),
        ("typing", ".", "Any", typing.Any),
    ),
)
def test_proved_global_package_controls_the_import_and_cleanup(
    package, relative, member, expected
):
    source = f"__package__ = {package!r}\nfrom {relative} import {member} as chosen\n"
    env = environment(source)
    binding = imports(env)[0].event
    prefix = env.required_prefix(env.source.module_context, None)
    store = SourceCompletionResolver(env).resolve(binding)
    store.return_continuation(prefix)
    origin = binding.target.origin
    captured = origin.resolve(
        env.kernel, (env.source.module_context, binding, frozenset())
    )
    captured.require_native_identity(NativeDeclaration(expected))
    actual = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    assert actual.returncode == 0, actual.stderr


def test_class_local_package_shadow_does_not_change_global_import_package():
    source = '__package__ = "builtins"\nclass Target:\n    __package__ = "typing"\n    from . import object as chosen\n'
    env = environment(source)
    binding = imports(env)[0].event
    store = SourceCompletionResolver(env).resolve(binding)
    prefix = env.required_prefix(store.native_frame_context, None)
    store.return_continuation(prefix)
    env.class_entry(env.module.module.body[-1]).native_tail.require_member(
        "chosen"
    ).require_native_identity(NativeDeclaration(object))
    actual = subprocess.run(
        [sys.executable, "-c", source + "assert Target.chosen is object\n"],
        capture_output=True,
        text=True,
    )
    assert actual.returncode == 0, actual.stderr


def test_later_package_rebinding_does_not_retarget_an_earlier_import():
    source = (
        '__package__ = "builtins"\nfrom . import object as chosen\n'
        '__package__ = "typing"\nretained = chosen\n'
    )
    env = environment(source)
    env.capture_value(env.module.module.body[-1].value).require_native_identity(
        NativeDeclaration(object)
    )
    actual = subprocess.run(
        [sys.executable, "-c", source + "assert retained is object\n"],
        capture_output=True,
        text=True,
    )
    assert actual.returncode == 0, actual.stderr


@pytest.mark.parametrize("package", ("None", "''", "42"))
def test_missing_or_invalid_package_does_not_gain_a_nominal_fallback(package):
    env = environment(f"__package__ = {package}\nfrom . import Any\n")
    with pytest.raises(ValueError):
        env.require_import_operation(imports(env)[0])


def test_unknown_initial_package_contents_do_not_supply_a_runtime_package():
    env = environment("from . import Any\n")
    with pytest.raises(ValueError):
        env.require_import_operation(imports(env)[0])


def test_absolute_request_does_not_read_irrelevant_runtime_package():
    def forbidden():
        raise AssertionError("Absolute import must not read package metadata")

    assert (
        ImportFromModuleName("builtins").resolve_from_package(forbidden) == "builtins"
    )


def test_relative_level_cannot_escape_actual_package_root():
    with pytest.raises(ValueError, match="package boundary"):
        ImportFromModuleName("...sibling").resolve_from_package(lambda: "parent.child")
