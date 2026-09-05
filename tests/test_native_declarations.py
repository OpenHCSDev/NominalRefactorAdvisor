"""Native declaration identity and inspected source share one authority."""

import ast
from pathlib import Path
import sys
from types import ModuleType

import pytest

from nominal_refactor_advisor.native_declarations import NativeDeclaration


class Example:
    value = 3


class EqualDeclarations(type):
    def __eq__(cls, other):
        return True

    def __hash__(cls):
        return 1


class FirstEqualDeclaration(metaclass=EqualDeclarations):
    value = 1


class SecondEqualDeclaration(metaclass=EqualDeclarations):
    value = 2


class UnhashableDeclarationMeta(type):
    __hash__ = None


class UnhashableDeclaration(metaclass=UnhashableDeclarationMeta):
    value = 4


def example(value):
    return value


async def asynchronous_example(value):
    return value


@pytest.mark.parametrize("declaration", (Example, example, asynchronous_example))
def test_native_source_is_cached_and_compared_without_locations(declaration) -> None:
    native = NativeDeclaration(declaration)
    assert native.qualified_name == f"{__name__}.{declaration.__qualname__}"
    assert native.node is native.node
    relocated = ast.increment_lineno(ast.parse(ast.unparse(native.node)).body[0], 50)
    native.require_source_matches(relocated)
    relocated.name = "DifferentDeclaration"
    with pytest.raises(ValueError, match="Source does not match native declaration"):
        native.require_source_matches(relocated)


@pytest.mark.parametrize("declaration", (staticmethod, len))
def test_builtin_identity_does_not_require_inspectable_source(declaration) -> None:
    native = NativeDeclaration(declaration)
    assert native.qualified_name == f"builtins.{declaration.__qualname__}"
    with pytest.raises(ValueError, match="no inspectable source"):
        _ = native.node


@pytest.mark.parametrize(
    "declaration", (Example, example, asynchronous_example, UnhashableDeclaration)
)
def test_repeated_wrappers_share_one_native_projection(declaration) -> None:
    first = NativeDeclaration(declaration)
    second = NativeDeclaration(declaration)
    assert first == second
    assert hash(first) == hash(second)
    assert first.node is second.node


def test_native_identity_does_not_follow_metaclass_value_equality() -> None:
    assert FirstEqualDeclaration == SecondEqualDeclaration
    first = NativeDeclaration(FirstEqualDeclaration)
    second = NativeDeclaration(SecondEqualDeclaration)
    assert first != second
    assert first.node is not second.node
    assert first.node.name == "FirstEqualDeclaration"
    assert second.node.name == "SecondEqualDeclaration"
    assert first.__eq__(FirstEqualDeclaration) is NotImplemented


def test_shared_native_projection_rechecks_each_proposed_source() -> None:
    original = NativeDeclaration(Example)
    current_node = ast.parse(ast.unparse(original.node)).body[0]
    original.require_source_matches(current_node)
    current_node.body[0].value = ast.Constant(value=99)
    with pytest.raises(ValueError, match="Source does not match native declaration"):
        NativeDeclaration(Example).require_source_matches(current_node)
    assert NativeDeclaration(Example).node is original.node


def test_reloaded_declaration_gets_a_new_projection_and_old_source_stays_unproved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "native_reload_fixture.py"
    module = ModuleType("native_reload_fixture")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = "class Reloaded:\n    value = 1\n"
    path.write_text(source, encoding="utf-8", newline="")
    exec(compile(source, str(path), "exec"), vars(module))
    original = NativeDeclaration(module.Reloaded)
    original_node = original.node

    updated_source = source.replace("value = 1", "value = 22")
    path.write_text(updated_source, encoding="utf-8", newline="")
    current_node = ast.parse(updated_source).body[0]
    with pytest.raises(ValueError, match="Source does not match native declaration"):
        NativeDeclaration(module.Reloaded).require_source_matches(current_node)
    assert original.declaration.value == 1
    assert original.node is original_node

    exec(compile(updated_source, str(path), "exec"), vars(module))
    reloaded = NativeDeclaration(module.Reloaded)
    assert reloaded.qualified_name == original.qualified_name
    assert reloaded != original
    assert reloaded.declaration.value == 22
    assert reloaded.node is not original_node
    reloaded.require_source_matches(current_node)
    with pytest.raises(ValueError, match="Source does not match native declaration"):
        reloaded.require_source_matches(original_node)
