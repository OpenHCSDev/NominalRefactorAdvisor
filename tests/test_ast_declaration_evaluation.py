from __future__ import annotations

import ast

from nominal_refactor_advisor.ast_tools import EagerNameLoadCollector


def _target_loads(source: str) -> tuple[ast.Name, ...]:
    module = ast.parse(source)
    return EagerNameLoadCollector.collect(module, "Target")


def test_eager_name_loads_exclude_deferred_annotations_and_function_body() -> None:
    loads = _target_loads(
        "from __future__ import annotations\n"
        "\n"
        "def build(value: Target = Target()) -> Target:\n"
        "    return Target()\n"
        "\n"
        "item: Target\n"
    )

    assert tuple(node.lineno for node in loads) == (3,)


def test_eager_name_loads_include_runtime_evaluated_annotations() -> None:
    loads = _target_loads(
        "def build(value: Target = Target()) -> Target:\n"
        "    return Target()\n"
        "\n"
        "item: Target\n"
    )

    assert tuple(node.lineno for node in loads) == (1, 1, 1, 4)
