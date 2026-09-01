from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    CompactModuleClassProjectionFamily,
    CompactProductAuthorityViolation,
    OpenCompactProductAuthority,
)
from nominal_refactor_advisor.product_flow import compact_product_flow_projection
from nominal_refactor_advisor.product_flow_authority import (
    CompactFunctionTargetResolutionViolation,
    CompactOpenFunctionCall,
    CompactProductFlowRepository,
)


def _module(module_name: str, source: str) -> ParsedModule:
    path = Path(*module_name.split(".")).with_suffix(".py")
    return ParsedModule(
        path=path,
        module_name=module_name,
        is_package_init=False,
        module=ast.parse(source, filename=str(path)),
        source=source,
    )


def _repository(*modules: ParsedModule) -> CompactProductFlowRepository:
    return CompactProductFlowRepository(
        product_projections=tuple(
            compact_product_flow_projection(module) for module in modules
        ),
        class_projections=tuple(
            CompactModuleClassProjectionFamily.collect(module)[0] for module in modules
        ),
    )


def _open_product_violations(
    repository: CompactProductFlowRepository,
    class_symbol: str,
) -> frozenset[CompactProductAuthorityViolation]:
    resolution = repository.class_index.product_authority_resolutions_by_symbol[
        class_symbol
    ]
    assert isinstance(resolution, OpenCompactProductAuthority)
    return frozenset(failure.violation for failure in resolution.failures)


def test_repository_joins_local_calls_constructions_and_callable_escapes() -> None:
    repository = _repository(
        _module(
            "pkg.sample",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _consume(*, left, right):\n"
            "    return left, right\n"
            "\n"
            "def wrapper(left, right):\n"
            "    key = CacheKey(left=left, right=right)\n"
            "    return _consume(left=key.left, right=key.right)\n"
            "\n"
            "escaped = _consume\n",
        )
    )

    construction = repository.resolved_product_constructions[0]
    assert construction.authority.class_symbol == "pkg.sample.CacheKey"
    assert construction.authority.field_names == ("left", "right")
    assert construction.construction.field_names == ("left", "right")
    incoming = repository.incoming_calls_for("pkg.sample._consume")
    assert len(incoming) == 1
    assert incoming[0].context.owner_symbol == "pkg.sample.wrapper"
    escapes = repository.callable_escapes_for("pkg.sample._consume")
    assert len(escapes) == 1
    assert escapes[0].context.owner_symbol == "pkg.sample"


def test_repository_resolves_imported_and_qualified_function_authorities() -> None:
    models = _module(
        "pkg.models",
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class CacheKey:\n"
        "    left: object\n"
        "    right: object\n",
    )
    sink = _module(
        "pkg.sink",
        "def consume(*, left, right):\n    return left, right\n",
    )
    worker = _module(
        "pkg.worker",
        "from pkg.models import CacheKey\n"
        "from pkg.sink import consume as send\n"
        "import pkg.sink as sink\n"
        "\n"
        "def imported(left, right):\n"
        "    key = CacheKey(left=left, right=right)\n"
        "    return send(left=key.left, right=key.right)\n"
        "\n"
        "def qualified(left, right):\n"
        "    key = CacheKey(left=left, right=right)\n"
        "    return sink.consume(left=key.left, right=key.right)\n",
    )

    repository = _repository(models, sink, worker)

    assert len(repository.resolved_product_constructions) == 2
    assert {
        edge.context.owner_symbol
        for edge in repository.incoming_calls_for("pkg.sink.consume")
    } == {"pkg.worker.imported", "pkg.worker.qualified"}


def test_repository_does_not_resolve_a_rebound_import_as_a_product_constructor() -> (
    None
):
    models = _module(
        "pkg.models",
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class CacheKey:\n"
        "    left: object\n"
        "    right: object\n",
    )
    worker = _module(
        "pkg.worker",
        "from pkg.models import CacheKey\n"
        "CacheKey = replacement\n"
        "\n"
        "def make(left, right):\n"
        "    return CacheKey(left=left, right=right)\n",
    )

    repository = _repository(models, worker)

    assert repository.resolved_product_constructions == ()


@pytest.mark.parametrize(
    "runtime_opening_use",
    (
        "Product.extra = replacement\n",
        "register(Product)\n",
        "Alias = Product\n",
        "def mutate():\n    Product.extra = replacement\n",
        "def replace():\n    global Product\n    Product = replacement\n",
    ),
)
def test_repository_rejects_locally_opened_product_class_semantics(
    runtime_opening_use: str,
) -> None:
    repository = _repository(
        _module(
            "pkg.runtime_open",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n"
            f"{runtime_opening_use}"
            "def make(left, right):\n"
            "    return Product(left, right)\n",
        )
    )

    assert "pkg.runtime_open.Product" not in (repository.product_authorities_by_symbol)
    assert "pkg.runtime_open.Product" in (
        repository.product_runtime_failures_by_authority_symbol
    )


def test_function_local_class_names_do_not_open_the_module_product() -> None:
    repository = _repository(
        _module(
            "pkg.local_class_name",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n"
            "def local():\n"
            "    Product = replacement\n"
            "    return Product\n",
        )
    )

    assert "pkg.local_class_name.Product" in repository.product_authorities_by_symbol


def test_repository_rejects_parameter_assignment_and_module_shadowing() -> None:
    repository = _repository(
        _module(
            "pkg.shadowed",
            "def _consume(value):\n"
            "    return value\n"
            "\n"
            "def parameter(_consume, value):\n"
            "    return _consume(value)\n"
            "\n"
            "def assignment(callback, value):\n"
            "    _consume = callback\n"
            "    return _consume(value)\n"
            "\n"
            "_consume = replacement\n"
            "\n"
            "def global_lookup(value):\n"
            "    return _consume(value)\n",
        )
    )

    assert repository.incoming_calls_for("pkg.shadowed._consume") == ()


def test_repository_resolves_dominating_nested_function_not_outer_namesake() -> None:
    repository = _repository(
        _module(
            "pkg.nested",
            "def _consume(value):\n"
            "    return value\n"
            "\n"
            "def caller(value):\n"
            "    def _consume(value):\n"
            "        return value + 1\n"
            "    return _consume(value)\n",
        )
    )

    assert len(repository.incoming_calls_for("pkg.nested.caller._consume")) == 1
    assert repository.incoming_calls_for("pkg.nested._consume") == ()


def test_repository_resolves_inherited_method_by_nominal_mro() -> None:
    repository = _repository(
        _module(
            "pkg.methods",
            "class Base:\n"
            "    def _consume(self, value):\n"
            "        return value\n"
            "\n"
            "class Leaf(Base):\n"
            "    def caller(self, value):\n"
            "        return self._consume(value)\n",
        )
    )

    incoming = repository.incoming_calls_for("pkg.methods.Base._consume")
    assert len(incoming) == 1
    assert incoming[0].context.owner_symbol == "pkg.methods.Leaf.caller"


def test_repository_keeps_inherited_lookup_open_across_unprojectable_base() -> None:
    repository = _repository(
        _module(
            "pkg.unprojectable_base",
            "class Base:\n"
            "    def consume(self, value):\n"
            "        return value\n"
            "\n"
            "class Leaf(Base, base_factory()):\n"
            "    def caller(self, value):\n"
            "        return self.consume(value)\n",
        )
    )

    resolution = next(
        resolution
        for resolution in repository.function_call_resolutions
        if resolution.context.owner_symbol == "pkg.unprojectable_base.Leaf.caller"
    )
    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.INCOMPLETE_RECEIVER_FAMILY
    )
    assert repository.resolved_function_calls == ()


def test_repository_retains_missing_import_resolution_without_outer_fallback() -> None:
    repository = _repository(
        _module(
            "pkg.import_shadow",
            "def consume(value):\n"
            "    return value\n"
            "\n"
            "from missing_sink import consume\n"
            "\n"
            "def caller(value):\n"
            "    return consume(value)\n",
        )
    )

    assert repository.resolved_function_calls == ()
    assert len(repository.function_call_resolutions) == 1
    resolution = repository.function_call_resolutions[0]
    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.MISSING_DECLARATION
    )
    assert resolution.target_resolution.possible_symbols == ("missing_sink.consume",)


def test_repository_stops_at_dynamic_enclosing_function_binding() -> None:
    repository = _repository(
        _module(
            "pkg.enclosing_shadow",
            "def consume(value):\n"
            "    return value\n"
            "\n"
            "def outer(callback):\n"
            "    consume = callback\n"
            "\n"
            "    def inner(value):\n"
            "        return consume(value)\n"
            "\n"
            "    return inner\n",
        )
    )

    resolution = next(
        resolution
        for resolution in repository.function_call_resolutions
        if resolution.context.owner_symbol == "pkg.enclosing_shadow.outer.inner"
    )
    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
    )
    assert repository.incoming_calls_for("pkg.enclosing_shadow.consume") == ()


def test_repository_rejects_ambiguous_inherited_method_lookup() -> None:
    repository = _repository(
        _module(
            "pkg.ambiguous_methods",
            "class Left:\n"
            "    def consume(self, value):\n"
            "        return value\n"
            "\n"
            "class Right:\n"
            "    def consume(self, value):\n"
            "        return value\n"
            "\n"
            "class Leaf(Left, Right):\n"
            "    def caller(self, value):\n"
            "        return self.consume(value)\n",
        )
    )

    resolution = repository.function_call_resolutions[0]
    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION
    )
    assert resolution.target_resolution.possible_symbols == (
        "pkg.ambiguous_methods.Leaf.consume",
        "pkg.ambiguous_methods.Left.consume",
        "pkg.ambiguous_methods.Right.consume",
    )
    assert repository.resolved_function_calls == ()


def test_repository_rejects_duplicate_nominal_function_declaration() -> None:
    repository = _repository(
        _module(
            "pkg.duplicate",
            "def consume(value):\n"
            "    return value\n"
            "\n"
            "def consume(value):\n"
            "    return value + 1\n"
            "\n"
            "def caller(value):\n"
            "    return consume(value)\n",
        )
    )

    resolution = repository.function_call_resolutions[0]
    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION
    )
    assert resolution.target_resolution.possible_symbols == ("pkg.duplicate.consume",)
    assert repository.resolved_function_calls == ()


def test_repository_keeps_star_import_function_binding_open() -> None:
    repository = _repository(
        _module(
            "pkg.star_shadow",
            "def consume(value):\n"
            "    return value\n"
            "\n"
            "from external_sink import *\n"
            "\n"
            "def caller(value):\n"
            "    return consume(value)\n",
        )
    )

    resolution = repository.function_call_resolutions[0]
    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION
    )
    assert resolution.target_resolution.possible_symbols == (
        "external_sink.consume",
        "pkg.star_shadow.consume",
    )
    assert repository.resolved_function_calls == ()


def test_repository_derives_single_inheritance_dataclass_product_fields() -> None:
    repository = _repository(
        _module(
            "pkg.models",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class BaseKey:\n"
            "    left: object\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class CacheKey(BaseKey):\n"
            "    right: object\n"
            "\n"
            "def make(left, right):\n"
            "    return CacheKey(left=left, right=right)\n",
        )
    )

    assert repository.product_authorities_by_symbol[
        "pkg.models.CacheKey"
    ].field_names == (
        "left",
        "right",
    )
    assert tuple(
        field.declaring_class_symbol
        for field in repository.product_authorities_by_symbol[
            "pkg.models.CacheKey"
        ].fields
    ) == ("pkg.models.BaseKey", "pkg.models.CacheKey")


def test_repository_derives_exact_qualified_dataclass_roles_without_classvar_mirror() -> (
    None
):
    repository = _repository(
        _module(
            "pkg.qualified",
            "import dataclasses as dc\n"
            "from typing import ClassVar\n"
            "\n"
            "@dc.dataclass(frozen=True)\n"
            "class CacheKey:\n"
            "    cache: ClassVar[dict[str, object]] = {}\n"
            "    _: dc.KW_ONLY\n"
            "    left: object\n"
            "    right: object\n",
        )
    )

    authority = repository.product_authorities_by_symbol["pkg.qualified.CacheKey"]

    assert authority.field_names == ("left", "right")
    assert tuple(field.line for field in authority.fields) == (8, 9)


def test_repository_accepts_a_direct_nominal_dataclass_alias() -> None:
    repository = _repository(
        _module(
            "pkg.direct_alias",
            "from dataclasses import dataclass as dc\n"
            "\n"
            "@dc\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n",
        )
    )

    assert repository.product_authorities_by_symbol[
        "pkg.direct_alias.Product"
    ].field_names == ("left", "right")


def test_repository_treats_explicit_object_as_a_neutral_product_base() -> None:
    repository = _repository(
        _module(
            "pkg.object_base",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Product(object):\n"
            "    left: object\n"
            "    right: object\n",
        )
    )

    assert repository.product_authorities_by_symbol[
        "pkg.object_base.Product"
    ].field_names == ("left", "right")


def test_repository_invalidates_dataclass_binding_after_star_import() -> None:
    repository = _repository(
        _module(
            "pkg.star_dataclass",
            "from dataclasses import dataclass\n"
            "from decorators import *\n"
            "\n"
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n",
        )
    )

    assert CompactProductAuthorityViolation.UNRESOLVED_DATACLASS_DECORATOR in (
        _open_product_violations(repository, "pkg.star_dataclass.Product")
    )


def test_repository_keeps_product_construction_open_across_a_star_import() -> None:
    models = _module(
        "pkg.models",
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Product:\n"
        "    left: object\n"
        "    right: object\n",
    )
    consumer = _module(
        "pkg.star_construction",
        "from pkg.models import Product\n"
        "from external import *\n"
        "def make(left, right):\n"
        "    return Product(left=left, right=right)\n",
    )

    repository = _repository(models, consumer)

    assert repository.resolved_product_constructions == ()


@pytest.mark.parametrize(
    ("consumer_source", "expected_construction_count"),
    (
        (
            "result = Product(left=left, right=right)\n"
            "from pkg.models import Product\n",
            0,
        ),
        (
            "from pkg.models import Product\n"
            "result = Product(left=left, right=right)\n",
            1,
        ),
    ),
)
def test_repository_resolves_module_construction_from_its_execution_position(
    consumer_source: str,
    expected_construction_count: int,
) -> None:
    models = _module(
        "pkg.models",
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Product:\n"
        "    left: object\n"
        "    right: object\n",
    )
    consumer = _module("pkg.module_construction", consumer_source)

    repository = _repository(models, consumer)

    assert len(repository.resolved_product_constructions) == (
        expected_construction_count
    )


@pytest.mark.parametrize(
    "prefix",
    (
        "@dataclass\n"
        "class Product(Base):\n"
        "    left: object\n"
        "    right: object\n"
        "from pkg.base import Base\n",
        "from pkg.base import Base\n"
        "Base = replacement\n"
        "@dataclass\n"
        "class Product(Base):\n"
        "    left: object\n"
        "    right: object\n",
    ),
)
def test_repository_resolves_bases_from_the_class_declaration_position(
    prefix: str,
) -> None:
    repository = _repository(
        _module(
            "pkg.base",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Base:\n"
            "    base: object\n",
        ),
        _module(
            "pkg.source_order",
            "from dataclasses import dataclass\n" + prefix,
        ),
    )

    assert (
        repository.class_index.classes_by_symbol[
            "pkg.source_order.Product"
        ].resolved_base_symbols
        == ()
    )
    assert CompactProductAuthorityViolation.INCOMPLETE_BASE_RESOLUTION in (
        _open_product_violations(repository, "pkg.source_order.Product")
    )


def test_repository_composes_a_base_imported_before_the_product_declaration() -> None:
    repository = _repository(
        _module(
            "pkg.base",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Base:\n"
            "    inherited: object\n",
        ),
        _module(
            "pkg.imported_base",
            "from dataclasses import dataclass\n"
            "from pkg.base import Base\n"
            "@dataclass\n"
            "class Product(Base):\n"
            "    left: object\n"
            "    right: object\n",
        ),
    )

    assert repository.class_index.classes_by_symbol[
        "pkg.imported_base.Product"
    ].resolved_base_symbols == ("pkg.base.Base",)
    assert repository.product_authorities_by_symbol[
        "pkg.imported_base.Product"
    ].field_names == ("inherited", "left", "right")


@pytest.mark.parametrize(
    ("schema_change", "failure_line"),
    (
        ("if enabled:\n        conditional: object\n", 7),
        ("__annotations__.update(extra_annotations)\n", 6),
        ("exec(source)\n", 6),
    ),
)
def test_repository_keeps_dynamic_dataclass_field_schemas_open(
    schema_change: str,
    failure_line: int,
) -> None:
    repository = _repository(
        _module(
            "pkg.dynamic_schema",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n"
            f"    {schema_change}",
        )
    )

    resolution = repository.class_index.product_authority_resolutions_by_symbol[
        "pkg.dynamic_schema.Product"
    ]
    assert isinstance(resolution, OpenCompactProductAuthority)
    failures = tuple(
        failure
        for failure in resolution.failures
        if failure.violation is CompactProductAuthorityViolation.DYNAMIC_FIELD_SCHEMA
    )
    assert len(failures) == 1
    assert failures[0].line == failure_line


def test_repository_resolves_module_and_class_aliases_of_dataclass_field_roles() -> (
    None
):
    repository = _repository(
        _module(
            "pkg.role_aliases",
            "from dataclasses import KW_ONLY, dataclass\n"
            "from typing import ClassVar\n"
            "CV = ClassVar\n"
            "KO = KW_ONLY\n"
            "@dataclass\n"
            "class Product:\n"
            "    LocalCV = CV\n"
            "    LocalKO = KO\n"
            "    cache: LocalCV[object] = None\n"
            "    _: LocalKO\n"
            "    left: object\n"
            "    right: object\n",
        )
    )

    assert repository.product_authorities_by_symbol[
        "pkg.role_aliases.Product"
    ].field_names == ("left", "right")


@pytest.mark.parametrize(
    "base_field",
    (
        "right: object = field(init=False)",
        "right: InitVar[object]",
    ),
)
def test_child_field_override_repairs_an_inherited_product_role(
    base_field: str,
) -> None:
    repository = _repository(
        _module(
            "pkg.override_repair",
            "from dataclasses import InitVar, dataclass, field\n"
            "@dataclass\n"
            "class Base:\n"
            "    left: object\n"
            f"    {base_field}\n"
            "@dataclass\n"
            "class Product(Base):\n"
            "    right: object\n",
        )
    )

    assert repository.product_authorities_by_symbol[
        "pkg.override_repair.Product"
    ].field_names == ("left", "right")


@pytest.mark.parametrize(
    ("class_source", "expected_violation"),
    (
        (
            "def dataclass(cls):\n    return cls\n\n"
            "@dataclass\nclass Product:\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.UNRESOLVED_DATACLASS_DECORATOR,
        ),
        (
            "@dataclass(init=False)\n"
            "class Product:\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.GENERATED_INIT_DISABLED,
        ),
        (
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object = field(init=False)\n",
            CompactProductAuthorityViolation.NON_INIT_FIELD,
        ),
        (
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: InitVar[object]\n",
            CompactProductAuthorityViolation.INIT_ONLY_FIELD,
        ),
        (
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n"
            "    def __post_init__(self):\n"
            "        self.left = normalize(self.left)\n",
            CompactProductAuthorityViolation.CUSTOM_PRODUCT_LIFECYCLE,
        ),
        (
            "@wrap\n@dataclass\nclass Product:\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.CUSTOM_CLASS_DECORATOR,
        ),
        (
            "@dataclass\n"
            "class Product(metaclass=Meta):\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.CUSTOM_CLASS_CREATION,
        ),
        (
            "@dataclass\n"
            "class Product(External):\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.INCOMPLETE_BASE_RESOLUTION,
        ),
        (
            "@dataclass\n"
            "class Product(factory()):\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.INCOMPLETE_BASE_RESOLUTION,
        ),
    ),
)
def test_repository_keeps_unproved_dataclass_semantics_explicitly_open(
    class_source: str,
    expected_violation: CompactProductAuthorityViolation,
) -> None:
    repository = _repository(
        _module(
            "pkg.open_product",
            "from dataclasses import InitVar, dataclass, field\n\n" + class_source,
        )
    )

    assert expected_violation in _open_product_violations(
        repository,
        "pkg.open_product.Product",
    )
    assert "pkg.open_product.Product" not in repository.product_authorities_by_symbol


def test_repository_keeps_partial_product_construction_visible_for_proof_rejection() -> (
    None
):
    repository = _repository(
        _module(
            "pkg.partial",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class CacheKey:\n"
            "    left: object\n"
            "    right: object = None\n"
            "\n"
            "def make(left):\n"
            "    key = CacheKey(left=left)\n"
            "    return key\n",
        )
    )

    construction = repository.resolved_product_constructions[0]
    assert construction.authority.field_names == ("left", "right")
    assert construction.construction.field_names == ("left",)
