"""Native declaration identity and inspected source share one authority."""

import ast

import pytest

from nominal_refactor_advisor.native_declarations import NativeDeclaration


class Example:
    value = 3


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
