"""Source callable declarations do not certify transformed runtime objects."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow_authority import (
    SourceProductFlowRepository,
    ResolvedCompactFunctionTarget,
    UnboundedCompactFunctionTarget,
)
from nominal_refactor_advisor.source_execution import SourceCreatedFunctionCapture

PRODUCT = (
    "from dataclasses import dataclass\n"
    "@dataclass\nclass Product:\n"
    "    left: object\n    right: object\n"
)


def _repository(source: str) -> SourceProductFlowRepository:
    module = ParsedModule(
        path=Path("function_object_identity.py"),
        module_name="function_object_identity",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    return SourceProductFlowRepository.from_modules((module,))


def _require_final_write_failure(repository: SourceProductFlowRepository) -> None:
    _require_product_creation(repository)
    symbol = "function_object_identity.Product"
    assert symbol not in repository.product_authorities_by_symbol
    context = repository.module_flow_contexts["function_object_identity"]
    assert any(
        failure.source_event is context.flow.mutations[-1]
        for failure in repository.product_runtime_failures_by_authority_symbol[symbol]
    )


def _require_product_creation(repository):
    module = repository.modules[0]
    environment = repository.native_reference_environment(module)
    environment.require_import(module.module.body[0])
    environment.require_class_creation(module.module.body[1])


def _require_capture_cause(environment, node, message):
    with pytest.raises(ValueError) as refusal:
        environment.capture_value(node).require_closed()
    causes = []
    error = refusal.value
    while error is not None:
        causes.append(str(error))
        error = error.__cause__
    assert any(message in cause for cause in causes), causes


def _final_write_target(repository: SourceProductFlowRepository):
    context = repository.module_flow_contexts["function_object_identity"]
    return context.flow.mutations[-1].resolve(repository, context)


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
    object_target = _final_write_target(repository)
    assert isinstance(object_target, UnboundedCompactFunctionTarget)
    assert object_target.candidate_symbols_within(
        frozenset(("function_object_identity.Product",))
    ) == frozenset(("function_object_identity.Product",))


def test_source_declaration_remains_retained_after_object_projection() -> None:
    repository = _repository(
        PRODUCT + "def final(original): return Product\n"
        "@final\ndef callback(value): return value\ncallback.changed = 1\n"
    )
    declaration = repository.function_declarations_by_symbol[
        "function_object_identity.callback"
    ]
    source_target = ResolvedCompactFunctionTarget(declaration)
    assert source_target.declaration is declaration
    assert isinstance(_final_write_target(repository), UnboundedCompactFunctionTarget)
    assert source_target.declaration is declaration
    assert (
        repository.function_declarations_by_symbol["function_object_identity.callback"]
        is declaration
    )


@pytest.mark.parametrize(
    "body, symbol, product_authorized",
    [
        ("def callback(): pass\ncallback.changed = 1\n", "callback", True),
        (
            "def callback(): pass\nsaved = callback\n"
            "callback = object()\nsaved.changed = 1\n",
            "callback",
            True,
        ),
        (
            "def outer():\n"
            "    def callback(): pass\n"
            "    callback.changed = 1\n"
            "outer()\n",
            "outer.callback",
            False,
        ),
        (
            "class Owner:\n"
            "    def outer(self):\n"
            "        def callback(): pass\n"
            "        callback.changed = 1\n"
            "Owner().outer()\n",
            "Owner.outer.callback",
            False,
        ),
    ],
)
def test_raw_free_and_local_functions_remain_distinct_controls(
    body: str, symbol: str, product_authorized: bool
) -> None:
    source = PRODUCT + body
    namespace = {}
    exec(source, namespace)
    assert "changed" not in vars(namespace["Product"])
    repository = _repository(source)
    _require_product_creation(repository)
    assert (
        "function_object_identity.Product" in repository.product_authorities_by_symbol
    ) is product_authorized
    declaration = repository.function_declarations_by_symbol[
        f"function_object_identity.{symbol}"
    ]
    target = ResolvedCompactFunctionTarget(declaration)
    assert target.declaration is declaration


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
    assert target.declaration is declaration
    assert isinstance(_final_write_target(repository), UnboundedCompactFunctionTarget)


@pytest.mark.parametrize(
    "decorator", ("replace", "final", "staticmethod", "classmethod", "keep")
)
@pytest.mark.parametrize("alias", (False, True))
def test_function_decorator_refusal_reaches_actual_creation(decorator, alias):
    returned = "original" if decorator == "keep" else "Product"
    source = (
        "class Product: pass\n"
        f"def {decorator}(original): return {returned}\n"
        f"@{decorator}\ndef callback(): pass\n"
        + (
            "saved = callback\ncallback = None\nsaved.changed = 1\n"
            if alias
            else "callback.changed = 1\n"
        )
    )
    namespace = {}
    exec(source, namespace)
    assert ("changed" in vars(namespace["Product"])) is (decorator != "keep")
    repository = _repository(source)
    module = repository.modules[0]
    environment = repository.native_reference_environment(module)
    # The preceding source definitions are admitted. This is the actual
    # callback's creation/decorator chain, not a failure at a fixture import.
    environment.require_class_creation(module.module.body[0])
    SourceCreatedFunctionCapture(environment, module.module.body[1]).require_closed()
    with pytest.raises(ValueError, match="Function decorator result remains unproved"):
        SourceCreatedFunctionCapture(environment, module.module.body[2]).result()
    target = _final_write_target(repository)
    assert isinstance(target, UnboundedCompactFunctionTarget)
    assert target.candidate_symbols_within(
        frozenset(("function_object_identity.Product",))
    ) == frozenset(("function_object_identity.Product",))
    original = repository.function_declarations_by_symbol[
        "function_object_identity.callback"
    ]
    assert (
        original
        is environment.definition_operation(module.module.body[2]).event.target.owner
    )


@pytest.mark.parametrize(
    "body, cause",
    (
        (
            "import builtins\nbuiltins.__build_class__ = lambda *args, **kwargs: None\nclass callback: pass\ncallback.changed = 1\n",
            "Native instance lifetime remains unproved",
        ),
        (
            "def outer():\n    def callback(): pass\n    callback.changed = 1\nouter()\n",
            "Source activation remains unproved",
        ),
        (
            "class Owner:\n    def outer(self):\n        def callback(): pass\n        callback.changed = 1\nOwner().outer()\n",
            "Source activation remains unproved",
        ),
    ),
    ids=(
        "builtin-builder-replacement",
        "function-activation",
        "method-activation",
    ),
)
def test_other_execution_obligations_do_not_become_definition_identity(body, cause):
    repository = _repository(body)
    module = repository.modules[0]
    environment = repository.native_reference_environment(module)
    receiver = next(
        node.value
        for node in ast.walk(module.module)
        if isinstance(node, ast.Attribute)
        and node.attr == "changed"
        and isinstance(node.ctx, ast.Store)
    )
    _require_capture_cause(environment, receiver, cause)


def test_native_object_rebinding_preserves_the_saved_original_function_identity():
    source = (
        "def callback(): pass\nsaved = callback\ncallback = object()\n"
        "saved.changed = 1\n"
    )
    repository = _repository(source)
    module = repository.modules[0]
    environment = repository.native_reference_environment(module)
    original = environment.definition_operation(
        module.module.body[0]
    ).event.target.owner
    captured = environment.capture_value(module.module.body[-1].targets[0].value)
    captured.require_definition_identity(original)
    assert isinstance(_final_write_target(repository), ResolvedCompactFunctionTarget)
    replacement = environment.capture_value(module.module.body[2].value)
    assert replacement.native_type is object
    with pytest.raises(ValueError):
        replacement.require_definition_identity(original)
    namespace = {}
    exec(source, namespace)
    assert namespace["saved"].changed == 1
    assert type(namespace["callback"]) is object


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
    assert ResolvedCompactFunctionTarget(declaration).declaration is declaration
    target = _final_write_target(repository)
    assert isinstance(target, UnboundedCompactFunctionTarget)
    _require_final_write_failure(repository)
