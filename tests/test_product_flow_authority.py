from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.product_flow import compact_product_flow_projection
from nominal_refactor_advisor.product_flow_authority import (
    CompactFunctionTargetResolutionViolation,
    CompactOpenFunctionCall,
    CompactProductFlowRepository,
)
from nominal_refactor_advisor.semantic_descent import (
    CompactSemanticModuleProjectionFamily,
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
        semantic_projections=tuple(
            CompactSemanticModuleProjectionFamily.collect(module)[0]
            for module in modules
        ),
    )


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
