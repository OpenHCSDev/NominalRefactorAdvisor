from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule, parse_python_modules
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.parameter_conveyor import (
    ClosedParameterConveyorAuthorityViolation,
    ClosedParameterConveyorComponentBuilder,
)
from nominal_refactor_advisor.product_flow import compact_product_flow_projection
from nominal_refactor_advisor.product_flow_authority import (
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


def _builder(*modules: ParsedModule) -> ClosedParameterConveyorComponentBuilder:
    repository = CompactProductFlowRepository(
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
    return ClosedParameterConveyorComponentBuilder(repository)


def _base_source(*, callee_body: str = "    return left, right\n") -> str:
    return (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class _CacheKey:\n"
        "    left: object\n"
        "    right: object\n"
        "\n"
        "def _build(left, right):\n"
        f"{callee_body}"
        "\n"
        "def caller(left, right):\n"
        "    key = _CacheKey(left=left, right=right)\n"
        "    return _build(left, right)\n"
    )


def test_complete_closed_parameter_conveyor_is_proven_as_one_component() -> None:
    builder = _builder(_module("pkg.complete", _base_source()))

    components = builder.proven_components()

    assert len(components) == 1
    component = components[0]
    assert component.authority.class_symbol == "pkg.complete._CacheKey"
    assert component.participant_symbols == ("pkg.complete._build",)
    assert len(component.root_edges) == 1
    assert component.forwarding_edges == ()
    assert component.proof.batch_compression_delta > 0


def test_multistep_forwarding_forms_one_maximal_component_not_per_edge() -> None:
    builder = _builder(
        _module(
            "pkg.chain",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _second(left, right):\n"
            "    return left, right\n"
            "\n"
            "def _first(left, right):\n"
            "    return _second(left, right)\n"
            "\n"
            "def caller(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _first(left, right)\n",
        )
    )

    components = builder.proven_components()

    assert len(components) == 1
    assert components[0].participant_symbols == (
        "pkg.chain._first",
        "pkg.chain._second",
    )
    assert len(components[0].root_edges) == 1
    assert len(components[0].forwarding_edges) == 1


def test_attractive_root_is_not_exposed_when_its_transitive_family_is_open() -> None:
    builder = _builder(
        _module(
            "pkg.open_tail",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _second(left, right):\n"
            "    return left, right\n"
            "\n"
            "def _first(left, right):\n"
            "    return _second(left, right)\n"
            "\n"
            "def caller(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _first(left, right)\n"
            "\n"
            "def unconverted(left, right):\n"
            "    return _second(transform(left), right)\n",
        )
    )

    components = builder.assessed_components()

    assert len(components) == 1
    assert components[0].participant_symbols == (
        "pkg.open_tail._first",
        "pkg.open_tail._second",
    )
    assert ClosedParameterConveyorAuthorityViolation.INCOMPLETE_CALL_FAMILY in (
        components[0].proof.violations
    )
    assert builder.proven_components() == ()


def test_open_disconnected_island_blocks_the_whole_nominal_authority() -> None:
    builder = _builder(
        _module(
            "pkg.disconnected_open",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _closed(left, right):\n"
            "    return left, right\n"
            "\n"
            "def _open(left, right):\n"
            "    return left, right\n"
            "\n"
            "def closed_root(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _closed(left, right)\n"
            "\n"
            "def open_root(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _open(left, right)\n"
            "\n"
            "def unconverted(left, right):\n"
            "    return _open(transform(left), right)\n",
        )
    )

    components = builder.assessed_components()

    assert len(components) == 1
    assert components[0].participant_symbols == (
        "pkg.disconnected_open._closed",
        "pkg.disconnected_open._open",
    )
    assert len(components[0].root_edges) == 2
    assert ClosedParameterConveyorAuthorityViolation.INCOMPLETE_CALL_FAMILY in (
        components[0].proof.violations
    )
    assert builder.proven_components() == ()


def test_closed_disconnected_islands_form_one_authority_wide_batch() -> None:
    builder = _builder(
        _module(
            "pkg.disconnected_closed",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _first(left, right):\n"
            "    return left, right\n"
            "\n"
            "def _second(left, right):\n"
            "    return left, right\n"
            "\n"
            "def first_root(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _first(left, right)\n"
            "\n"
            "def second_root(left, right):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _second(left, right)\n",
        )
    )

    components = builder.proven_components()

    assert len(components) == 1
    assert components[0].participant_symbols == (
        "pkg.disconnected_closed._first",
        "pkg.disconnected_closed._second",
    )
    assert len(components[0].root_edges) == 2


def test_partial_product_construction_never_becomes_a_component() -> None:
    builder = _builder(
        _module(
            "pkg.partial",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object = None\n"
            "\n"
            "def _build(left, right):\n"
            "    return left, right\n"
            "\n"
            "def caller(left, right):\n"
            "    key = _CacheKey(left=left)\n"
            "    return _build(left, right)\n",
        )
    )

    assert builder.assessed_components() == ()
    assert builder.proven_components() == ()


def test_unconverted_incoming_call_blocks_the_whole_component() -> None:
    builder = _builder(
        _module(
            "pkg.incoming",
            _base_source()
            + "\n"
            + "def other(left, right):\n"
            + "    return _build(transform(left), right)\n",
        )
    )

    component = builder.assessed_components()[0]

    assert component.proof.violations == (
        ClosedParameterConveyorAuthorityViolation.INCOMPLETE_CALL_FAMILY,
    )
    assert builder.proven_components() == ()


def test_callable_escape_blocks_the_whole_component() -> None:
    builder = _builder(
        _module(
            "pkg.escape",
            _base_source() + "\n" + "escaped = _build\n",
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.ESCAPING_CALLABLE_REFERENCE in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_public_participant_is_not_treated_as_a_closed_repository_boundary() -> None:
    builder = _builder(
        _module(
            "pkg.public",
            _base_source()
            .replace("def _build", "def build")
            .replace("return _build", "return build"),
        )
    )

    component = builder.assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.PUBLIC_OR_EXTERNAL_BOUNDARY_NOT_CLOSED
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_rebinding_between_construction_and_call_blocks_the_component() -> None:
    builder = _builder(
        _module(
            "pkg.rebound",
            _base_source().replace(
                "    return _build(left, right)\n",
                "    left = normalize(left)\n" "    return _build(left, right)\n",
            ),
        )
    )

    component = builder.assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.REBINDING_OR_MUTATION_BETWEEN_BINDING_AND_USE
        in component.proof.violations
    )
    assert builder.proven_components() == ()


def test_branch_local_carrier_does_not_claim_dominance() -> None:
    builder = _builder(
        _module(
            "pkg.branch",
            _base_source().replace(
                "    key = _CacheKey(left=left, right=right)\n",
                "    if left:\n" "        key = _CacheKey(left=left, right=right)\n",
            ),
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.NON_DOMINATING_CARRIER_BINDING in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_duplicate_dominating_carriers_block_local_root_selection() -> None:
    builder = _builder(
        _module(
            "pkg.ambiguous_carrier",
            _base_source().replace(
                "    key = _CacheKey(left=left, right=right)\n",
                "    first = _CacheKey(left=left, right=right)\n"
                "    second = _CacheKey(left=left, right=right)\n",
            ),
        )
    )

    component = builder.assessed_components()[0]

    assert len(component.root_edges) == 2
    assert ClosedParameterConveyorAuthorityViolation.AMBIGUOUS_ROOT_CARRIER in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_mutated_or_unconsumed_field_parameters_block_the_component() -> None:
    mutated = _builder(
        _module(
            "pkg.mutated",
            _base_source(
                callee_body=("    left = normalize(left)\n" "    return left, right\n")
            ),
        )
    ).assessed_components()[0]
    unconsumed = _builder(
        _module(
            "pkg.unconsumed",
            _base_source(callee_body="    return left\n"),
        )
    ).assessed_components()[0]

    assert (
        ClosedParameterConveyorAuthorityViolation.REBINDING_OR_MUTATION_BETWEEN_BINDING_AND_USE
        in mutated.proof.violations
    )
    assert ClosedParameterConveyorAuthorityViolation.INCOMPLETE_PRODUCT_CONSUMPTION in (
        unconsumed.proof.violations
    )


def test_dynamic_full_product_forwarding_blocks_the_component() -> None:
    builder = _builder(
        _module(
            "pkg.dynamic",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _build(left, right, callbacks):\n"
            "    return callbacks[0](left, right)\n"
            "\n"
            "def caller(left, right, callbacks):\n"
            "    key = _CacheKey(left=left, right=right)\n"
            "    return _build(left, right, callbacks)\n",
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.DYNAMIC_CALL_TARGET in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_local_signature_introspection_blocks_parameter_rewrite() -> None:
    builder = _builder(
        _module(
            "pkg.locals_observed",
            _base_source(
                callee_body=(
                    "    observed = locals()\n"
                    "    return observed['left'], observed['right']\n"
                )
            ),
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.SIGNATURE_SEMANTICS_HAZARD in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_incomplete_override_family_blocks_method_signature_change() -> None:
    builder = _builder(
        _module(
            "pkg.overrides",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class _CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "class Base:\n"
            "    def _build(self, left, right):\n"
            "        return left, right\n"
            "\n"
            "class Leaf(Base):\n"
            "    def _build(self, left, right):\n"
            "        return left, right\n"
            "\n"
            "    def caller(self, left, right):\n"
            "        key = _CacheKey(left=left, right=right)\n"
            "        return self._build(left, right)\n",
        )
    )

    component = builder.assessed_components()[0]

    assert ClosedParameterConveyorAuthorityViolation.INCOMPLETE_METHOD_FAMILY in (
        component.proof.violations
    )
    assert builder.proven_components() == ()


def test_competing_product_authorities_block_both_local_trajectories() -> None:
    builder = _builder(
        _module(
            "pkg.competing",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class _FirstKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "@dataclass\n"
            "class _SecondKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _build(left, right):\n"
            "    return left, right\n"
            "\n"
            "def first(left, right):\n"
            "    key = _FirstKey(left=left, right=right)\n"
            "    return _build(left, right)\n"
            "\n"
            "def second(left, right):\n"
            "    key = _SecondKey(left=left, right=right)\n"
            "    return _build(left, right)\n",
        )
    )

    components = builder.assessed_components()

    assert len(components) == 2
    assert all(
        ClosedParameterConveyorAuthorityViolation.NO_UNIQUE_NOMINAL_AUTHORITY
        in component.proof.violations
        for component in components
    )
    assert all(
        ClosedParameterConveyorAuthorityViolation.CONFLICTING_CALL_MAPPING
        in component.proof.violations
        for component in components
    )
    assert builder.proven_components() == ()


def test_self_host_coherence_cache_key_reaches_the_intended_final_authority() -> None:
    module = parse_python_modules(
        Path("nominal_refactor_advisor/observation_graph.py"),
        use_parse_cache=True,
    )[0]
    components = _builder(module).proven_components()

    component = next(
        component
        for component in components
        if component.authority.class_symbol.endswith("._CoherenceCohortCacheKey")
    )

    assert component.participant_symbols == (
        "nominal_refactor_advisor.observation_graph."
        "ObservationGraph._build_coherence_cohorts_for",
    )
    assert component.proof.is_proven
