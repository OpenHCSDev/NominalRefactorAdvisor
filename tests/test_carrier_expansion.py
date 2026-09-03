from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.carrier_expansion import (
    DeclaredCarrierExpansionAuthorityViolation,
    DeclaredCarrierExpansionBuilder,
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


def test_builder_derives_field_mapping_from_bound_nominal_carrier() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.models",
                "class Context:\n"
                "    title: str\n"
                "    metrics: object\n"
                "\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> 'Context':\n"
                "        return base\n",
            ),
            _module(
                "pkg.worker",
                "from pkg.models import Context\n"
                "\n"
                "def consume(value, *, title=None, metrics=None):\n"
                "    return value, title, metrics\n"
                "\n"
                "def build(value, base):\n"
                "    context = Context.merge(base)\n"
                "    return consume(\n"
                "        value,\n"
                "        title=context.title,\n"
                "        metrics=context.metrics,\n"
                "    )\n",
            ),
        )
    )

    assert len(builder.expansions) == 1
    expansion = builder.expansions[0]
    assert expansion.carrier_class_symbol == "pkg.models.Context"
    assert expansion.callee_symbol == "pkg.worker.consume"
    assert expansion.field_mapping == (
        ("title", "title"),
        ("metrics", "metrics"),
    )


def test_builder_rejects_untyped_or_rebound_carrier_values() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.open_values",
                "def make(base):\n"
                "    return base\n"
                "\n"
                "def consume(*, left=None, right=None):\n"
                "    return left, right\n"
                "\n"
                "def untyped(base):\n"
                "    context = make(base)\n"
                "    return consume(left=context.left, right=context.right)\n"
                "\n"
                "class Context:\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> 'Context':\n"
                "        return base\n"
                "\n"
                "def rebound(base, replacement):\n"
                "    context = Context.merge(base)\n"
                "    context = replacement\n"
                "    return consume(left=context.left, right=context.right)\n",
            ),
        )
    )

    assert builder.expansions == ()


def test_builder_follows_complete_flat_field_forwarding_graph() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.chain",
                "class Context:\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> 'Context':\n"
                "        return base\n"
                "\n"
                "def leaf(*, first=None, second=None):\n"
                "    return first, second\n"
                "\n"
                "def middle(*, left=None, right=None):\n"
                "    return leaf(first=left, second=right)\n"
                "\n"
                "def root(base):\n"
                "    context = Context.merge(base)\n"
                "    return middle(left=context.alpha, right=context.beta)\n",
            ),
        )
    )

    assert len(builder.components) == 1
    component = builder.components[0]
    assert component.root_edges[0].field_mapping == (
        ("alpha", "left"),
        ("beta", "right"),
    )
    assert tuple(
        (edge.caller_symbol, edge.callee_symbol, edge.field_mapping)
        for edge in component.forwarding_edges
    ) == (
        (
            "pkg.chain.middle",
            "pkg.chain.leaf",
            (("alpha", "first"), ("beta", "second")),
        ),
    )
    assert component.field_mapping_by_participant == {
        "pkg.chain.middle": (("alpha", "left"), ("beta", "right")),
        "pkg.chain.leaf": (("alpha", "first"), ("beta", "second")),
    }


def test_builder_unifies_connected_root_expansions_into_one_component() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.connected",
                "class Context:\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> 'Context':\n"
                "        return base\n"
                "\n"
                "def consume(left, right):\n"
                "    return left, right\n"
                "\n"
                "def first(base):\n"
                "    context = Context.merge(base)\n"
                "    return consume(context.left, context.right)\n"
                "\n"
                "def second(base):\n"
                "    context = Context.merge(base)\n"
                "    return consume(context.left, context.right)\n",
            ),
        )
    )

    assert len(builder.components) == 1
    component = builder.components[0]
    assert tuple(edge.caller_symbol for edge in component.root_edges) == (
        "pkg.connected.first",
        "pkg.connected.second",
    )
    assert component.participant_symbols == ("pkg.connected.consume",)
    assert (
        builder.assessed_components()[0]
        .proof.callable_component.incomplete_call_family_symbols
        == ()
    )


def test_builder_proves_one_closed_private_carrier_expansion() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.closed",
                "from dataclasses import dataclass\n"
                "\n"
                "@dataclass(frozen=True)\n"
                "class _Context:\n"
                "    left: object\n"
                "    right: object\n"
                "\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> '_Context':\n"
                "        return base\n"
                "\n"
                "def _consume(left, right):\n"
                "    return left, right\n"
                "\n"
                "def root(base):\n"
                "    context = _Context.merge(base)\n"
                "    return _consume(context.left, context.right)\n",
            ),
        )
    )

    assessments = builder.assessed_components()

    assert len(assessments) == 1
    assert assessments[0].proof.is_proven
    assert assessments[0].proof.batch_compression_delta == 2
    assert builder.proven_components() == assessments


def test_builder_rejects_calls_outside_the_atomic_component() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.open_call",
                "from dataclasses import dataclass\n"
                "\n"
                "@dataclass(frozen=True)\n"
                "class _Context:\n"
                "    left: object\n"
                "    right: object\n"
                "\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> '_Context':\n"
                "        return base\n"
                "\n"
                "def _consume(left, right):\n"
                "    return left, right\n"
                "\n"
                "def root(base):\n"
                "    context = _Context.merge(base)\n"
                "    return _consume(context.left, context.right)\n"
                "\n"
                "def other(left, right):\n"
                "    return _consume(left, right)\n",
            ),
        )
    )

    proof = builder.assessed_components()[0].proof

    assert (
        DeclaredCarrierExpansionAuthorityViolation.INCOMPLETE_CALL_FAMILY
        in proof.violations
    )
    assert builder.proven_components() == ()


def test_builder_rejects_mutated_projected_parameters() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.mutated",
                "from dataclasses import dataclass\n"
                "\n"
                "@dataclass(frozen=True)\n"
                "class _Context:\n"
                "    left: object\n"
                "    right: object\n"
                "\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> '_Context':\n"
                "        return base\n"
                "\n"
                "def _consume(left, right):\n"
                "    left = right\n"
                "    return left, right\n"
                "\n"
                "def root(base):\n"
                "    context = _Context.merge(base)\n"
                "    return _consume(context.left, context.right)\n",
            ),
        )
    )

    proof = builder.assessed_components()[0].proof

    assert (
        DeclaredCarrierExpansionAuthorityViolation.REBINDING_OR_MUTATION
        in proof.violations
    )
