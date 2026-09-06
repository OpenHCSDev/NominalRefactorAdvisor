"""Source callable declarations do not certify transformed runtime objects."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow_authority import (
    CompactProductFlowRepository,
    ResolvedCompactFunctionTarget,
    UnboundedCompactFunctionTarget,
)

PRODUCT = (
    "from dataclasses import dataclass\n"
    "@dataclass\nclass Product:\n"
    "    left: object\n    right: object\n"
)


def _repository(source: str) -> CompactProductFlowRepository:
    module = ParsedModule(
        path=Path("function_object_identity.py"),
        module_name="function_object_identity",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    return CompactProductFlowRepository.from_modules((module,))


def _require_final_write_failure(repository: CompactProductFlowRepository) -> None:
    symbol = "function_object_identity.Product"
    assert symbol not in repository.product_authorities_by_symbol
    context = repository.module_flow_contexts["function_object_identity"]
    assert any(
        failure.source_event is context.flow.mutations[-1]
        for failure in repository.product_runtime_failures_by_authority_symbol[symbol]
    )


@pytest.mark.parametrize(
    "decorator", ["replace", "final", "staticmethod", "classmethod"]
)
@pytest.mark.parametrize("alias", [False, True])
def test_decorator_spelling_never_proves_bound_function_object(
    decorator: str, alias: bool
) -> None:
    source = (
        PRODUCT
        + f"def {decorator}(original): return Product\n"
        + f"@{decorator}\ndef callback(value): return value\n"
        + (
            "saved = callback\ncallback = object()\nsaved.changed = 1\n"
            if alias
            else "callback.changed = 1\n"
        )
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["Product"].changed == 1
    repository = _repository(source)
    _require_final_write_failure(repository)
    declaration = repository.function_declarations_by_symbol[
        "function_object_identity.callback"
    ]
    source_target = ResolvedCompactFunctionTarget(declaration)
    assert source_target.declaration is declaration
    object_target = source_target.for_object_mutation()
    assert isinstance(object_target, UnboundedCompactFunctionTarget)
    assert object_target.candidate_symbols_within(
        frozenset(("function_object_identity.Product",))
    ) == frozenset(("function_object_identity.Product",))


def test_source_declaration_remains_retained_after_object_projection() -> None:
    repository = _repository(
        PRODUCT + "def final(original): return Product\n"
        "@final\ndef callback(value): return value\n"
    )
    declaration = repository.function_declarations_by_symbol[
        "function_object_identity.callback"
    ]
    source_target = ResolvedCompactFunctionTarget(declaration)
    assert source_target.declaration is declaration
    assert isinstance(
        source_target.for_object_mutation(), UnboundedCompactFunctionTarget
    )
    assert source_target.declaration is declaration
    assert (
        repository.function_declarations_by_symbol["function_object_identity.callback"]
        is declaration
    )


@pytest.mark.parametrize(
    "body, symbol",
    [
        ("def callback(): pass\ncallback.changed = 1\n", "callback"),
        (
            "def callback(): pass\nsaved = callback\n"
            "callback = object()\nsaved.changed = 1\n",
            "callback",
        ),
        (
            "def outer():\n"
            "    def callback(): pass\n"
            "    callback.changed = 1\n"
            "outer()\n",
            "outer.callback",
        ),
        (
            "class Owner:\n"
            "    def outer(self):\n"
            "        def callback(): pass\n"
            "        callback.changed = 1\n"
            "Owner().outer()\n",
            "Owner.outer.callback",
        ),
    ],
)
def test_raw_free_and_local_functions_remain_distinct_controls(
    body: str, symbol: str
) -> None:
    source = PRODUCT + body
    namespace = {}
    exec(source, namespace)
    assert "changed" not in vars(namespace["Product"])
    repository = _repository(source)
    assert (
        "function_object_identity.Product" in repository.product_authorities_by_symbol
    )
    declaration = repository.function_declarations_by_symbol[
        f"function_object_identity.{symbol}"
    ]
    target = ResolvedCompactFunctionTarget(declaration)
    assert target.for_object_mutation() is target


@pytest.mark.parametrize("inherited", [False, True])
def test_raw_method_namespace_setter_can_install_a_different_object(
    inherited: bool,
) -> None:
    source = (
        PRODUCT + "class Namespace(dict):\n"
        "    def __setitem__(self, key, value):\n"
        "        dict.__setitem__(self, key, Product if key == 'callback' else value)\n"
        "class Meta(type):\n"
        "    @classmethod\n"
        "    def __prepare__(meta, name, bases): return Namespace()\n"
        + (
            "class Base(metaclass=Meta): pass\nclass Owner(Base):\n"
            if inherited
            else "class Owner(metaclass=Meta):\n"
        )
        + "    def callback(value): return value\n"
        "Owner.callback.changed = 1\n"
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["Owner"].callback is namespace["Product"]
    assert namespace["Product"].changed == 1
    repository = _repository(source)
    _require_final_write_failure(repository)
    declaration = repository.function_declarations_by_symbol[
        "function_object_identity.Owner.callback"
    ]
    assert declaration.decorators == ()
    target = ResolvedCompactFunctionTarget(declaration)
    assert isinstance(target.for_object_mutation(), UnboundedCompactFunctionTarget)


def test_identity_preserving_decorator_remains_unproved_not_known_replacement() -> None:
    source = (
        PRODUCT + "def keep(original): return original\n"
        "@keep\ndef callback(): pass\n"
        "callback.changed = 1\n"
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["callback"] is not namespace["Product"]
    assert "changed" not in vars(namespace["Product"])
    repository = _repository(source)
    declaration = repository.function_declarations_by_symbol[
        "function_object_identity.callback"
    ]
    target = ResolvedCompactFunctionTarget(declaration).for_object_mutation()
    assert isinstance(target, UnboundedCompactFunctionTarget)
    _require_final_write_failure(repository)
