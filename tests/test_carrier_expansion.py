from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.carrier_expansion import (
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
    assert component.root.field_mapping == (
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
