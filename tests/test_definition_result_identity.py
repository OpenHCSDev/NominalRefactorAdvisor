"""A source definition does not identify its post-transformation bound object."""

import ast
import builtins
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow_authority import (
    ResolvedCompactClassTarget,
    ResolvedCompactFunctionTarget,
    SourceProductFlowRepository,
    UnboundedCompactFunctionTarget,
)


def _repository(source: str) -> SourceProductFlowRepository:
    module = ParsedModule(
        path=Path("definition_identity.py"),
        module_name="definition_identity",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    return SourceProductFlowRepository.from_modules((module,))


def _require_final_write_effect(
    repository: SourceProductFlowRepository, affects_product: bool
) -> None:
    # The standard dataclass transformation is proved independently of the later
    # write whose receiver identity this helper exercises.
    environment = repository.native_reference_environment(repository.modules[0])
    environment.require_import(repository.modules[0].module.body[0])
    environment.require_class_creation(repository.modules[0].module.body[1])
    symbol = "definition_identity.Product"
    assert symbol in repository.declared_product_authorities_by_symbol
    assert (symbol not in repository.product_authorities_by_symbol) is affects_product
    if not affects_product:
        return
    context = repository.module_flow_contexts["definition_identity"]
    final_write = context.flow.mutations[-1]
    failures = repository.product_runtime_failures_by_authority_symbol[symbol]
    assert any(failure.source_event is final_write for failure in failures)


@pytest.mark.parametrize(
    "definition, binds_product",
    (
        (
            "def replace(original): return Product\n" "@replace\nclass Other: pass\n",
            True,
        ),
        (
            "def replace(original): return Product\n" "@replace\ndef Other(): pass\n",
            True,
        ),
        (
            "class Replace(type):\n"
            "    def __new__(meta, name, bases, namespace): return Product\n"
            "class Other(metaclass=Replace): pass\n",
            True,
        ),
        (
            "class Replace(type):\n"
            "    def __new__(meta, name, bases, namespace):\n"
            "        if bases: return Product\n"
            "        return type.__new__(meta, name, bases, namespace)\n"
            "class Base(metaclass=Replace): pass\n"
            "class Other(Base): pass\n",
            True,
        ),
        (
            "class Replace(type):\n"
            "    def __new__(meta, name, bases, namespace): return Product\n"
            "class InheritedReplace(Replace): pass\n"
            "class Other(metaclass=InheritedReplace): pass\n",
            True,
        ),
        (
            "class Replace(type):\n"
            "    def __new__(meta, name, bases, namespace):\n"
            "        if bases: return Product\n"
            "        return type.__new__(meta, name, bases, namespace)\n"
            "class Base(metaclass=Replace): pass\n"
            "class BaseSource:\n"
            "    def __mro_entries__(self, bases): return (Base,)\n"
            "class Other(BaseSource()): pass\n",
            True,
        ),
        (
            "class MetaFactory(type):\n"
            "    def __call__(self, *args, **kwargs): return Product\n"
            "class Replace(type, metaclass=MetaFactory): pass\n"
            "class Other(metaclass=Replace): pass\n",
            True,
        ),
        (
            "def factory():\n"
            "    return lambda original: Product\n"
            "@factory()\ndef Other(): pass\n",
            True,
        ),
        ("class Other: pass\n", False),
        ("def Other(): pass\n", False),
    ),
    ids=(
        "class-decorator-replacement",
        "function-decorator-replacement",
        "metaclass-replacement",
        "inherited-metaclass-replacement",
        "inherited-metaclass-new-replacement",
        "mro-entries-select-replacing-metaclass",
        "metaclass-call-replacement",
        "decorator-factory-replacement",
        "distinct-class-control",
        "distinct-function-control",
    ),
)
@pytest.mark.parametrize(
    "write, receiver_name",
    (
        ("Other.changed = 1\n", "Other"),
        ("alias = Other\nalias.changed = 1\n", "alias"),
        ("alias = Other\nOther = object()\nalias.changed = 1\n", "alias"),
    ),
    ids=("direct-write", "alias-write", "captured-before-rebinding"),
)
def test_mutation_safety_follows_bound_object_not_source_definition(
    definition: str, binds_product: bool, write: str, receiver_name: str
) -> None:
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Product:\n"
        "    left: object\n    right: object\n" + definition + write
    )
    namespace = {}
    exec(source, namespace)
    assert (namespace[receiver_name] is namespace["Product"]) is binds_product
    assert ("changed" in vars(namespace["Product"])) is binds_product

    repository = _repository(source)
    _require_final_write_effect(repository, binds_product)


@pytest.mark.parametrize(
    "definition",
    (
        "def replace(original): return Product\n@replace\nclass Other: pass\n",
        "def replace(original): return Product\n@replace\ndef Other(): pass\n",
        "class Replace(type):\n"
        "    def __new__(meta, name, bases, namespace): return Product\n"
        "class Other(metaclass=Replace): pass\n",
    ),
    ids=("decorated-class", "decorated-function", "metaclass"),
)
def test_transformed_object_retained_in_member_alias_opens_actual_final_write(
    definition: str,
) -> None:
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Product:\n"
        "    left: object\n    right: object\n" + definition + "class Holder: pass\n"
        "Holder.saved = Other\n"
        "Other = object()\n"
        "Holder.saved.changed = 1\n"
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["Holder"].saved is namespace["Product"]
    assert namespace["Product"].changed == 1
    _require_final_write_effect(_repository(source), True)


def test_frame_builtin_class_builder_can_replace_plain_class_result() -> None:
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Product:\n"
        "    left: object\n    right: object\n"
        "__builtins__['__build_class__'] = lambda *args, **kwargs: Product\n"
        "class Other: pass\n"
        "Other.changed = 1\n"
    )
    # Native LOAD_BUILD_CLASS consults this fixture's isolated frame builtins.
    # Never mutate the interpreter's shared builtins while running this control.
    namespace = {"__builtins__": vars(builtins).copy()}
    exec(compile(source, "<trusted-class-builder-fixture>", "exec"), namespace)
    assert namespace["Other"] is namespace["Product"]
    assert namespace["Product"].changed == 1
    _require_final_write_effect(_repository(source), True)


def test_module_global_class_builder_name_does_not_replace_native_class_creation() -> (
    None
):
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Product:\n"
        "    left: object\n    right: object\n"
        "__build_class__ = lambda *args, **kwargs: Product\n"
        "class Other: pass\n"
        "Other.changed = 1\n"
    )
    namespace = {"__builtins__": vars(builtins).copy()}
    exec(compile(source, "<trusted-class-builder-fixture>", "exec"), namespace)
    assert namespace["Other"] is not namespace["Product"]
    assert "changed" not in vars(namespace["Product"])
    repository = _repository(source)
    _require_final_write_effect(repository, False)


@pytest.mark.parametrize(
    "definition, target_type",
    (
        ("class Other: pass\n", ResolvedCompactClassTarget),
        ("def Other(): pass\n", ResolvedCompactFunctionTarget),
    ),
)
@pytest.mark.parametrize(
    "write",
    (
        "Other.changed = 1\n",
        "alias = Other\nalias.changed = 1\n",
        "alias = Other\nOther = None\nalias.changed = 1\n",
    ),
    ids=("direct", "alias", "retained-before-literal-rebinding"),
)
def test_admitted_receiver_retains_actual_distinct_creation(
    definition, target_type, write
):
    repository = _repository("class Product: pass\n" + definition + write)
    module = repository.modules[0]
    environment = repository.native_reference_environment(module)
    context = repository.module_flow_contexts[module.module_name]
    mutation = context.flow.mutations[-1]
    target = mutation.resolve(repository, context)
    assert isinstance(target, target_type)
    assert target.possible_symbols == ("definition_identity.Other",)
    assert not target.candidate_symbols_within(
        frozenset(("definition_identity.Product",))
    )
    receiver = module.module.body[-1].targets[0].value
    capture = environment.capture_value(receiver)
    capture.require_closed()
    original = environment.definition_operation(module.module.body[1])
    owner_context, binding = capture.source_definition()
    assert binding is original.event
    assert owner_context is context


@pytest.mark.parametrize(
    "definition, failure",
    (
        (
            "def replace(original): return Product\n@replace\nclass Other: pass\n",
            "Class decorator result remains unproved",
        ),
        (
            "class Replace(type):\n"
            "    def __new__(meta, name, bases, namespace): return Product\n"
            "class Other(metaclass=Replace): pass\n",
            "Native object identity remains open: unproved_execution_effects",
        ),
    ),
)
def test_class_transformation_refusal_reaches_its_creation_boundary(
    definition, failure
):
    source = "class Product: pass\n" + definition + "Other.changed = 1\n"
    namespace = {}
    exec(source, namespace)
    assert namespace["Other"] is namespace["Product"]
    repository = _repository(source)
    module = repository.modules[0]
    environment = repository.native_reference_environment(module)
    environment.require_class_creation(module.module.body[0])
    with pytest.raises(ValueError, match=failure):
        environment.require_class_creation(module.module.body[-2])
    context = repository.module_flow_contexts[module.module_name]
    target = context.flow.mutations[-1].resolve(repository, context)
    assert isinstance(target, UnboundedCompactFunctionTarget)
    assert target.candidate_symbols_within(frozenset(("definition_identity.Product",)))


def test_global_builder_function_does_not_replace_frame_builtin_builder():
    repository = _repository(
        "class Product: pass\n"
        "def __build_class__(*args, **kwargs): return Product\n"
        "class Other: pass\nOther.changed = 1\n"
    )
    module = repository.modules[0]
    environment = repository.native_reference_environment(module)
    environment.require_class_creation(module.module.body[-2])
    context = repository.module_flow_contexts[module.module_name]
    target = context.flow.mutations[-1].resolve(repository, context)
    assert isinstance(target, ResolvedCompactClassTarget)
    assert target.possible_symbols == ("definition_identity.Other",)
    capture = environment.capture_value(module.module.body[-1].targets[0].value)
    assert (
        capture.source_definition()[1]
        is environment.definition_operation(module.module.body[-2]).event
    )
