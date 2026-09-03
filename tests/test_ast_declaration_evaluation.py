from __future__ import annotations

import ast

from nominal_refactor_advisor.ast_tools import (
    DeclarationAnnotationProjection,
    EagerNameLoadCollector,
    ModuleAnnotationEvaluationMode,
)


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


def test_eager_name_loads_follow_runtime_default_annotation_policy() -> None:
    loads = _target_loads(
        "def build(value: Target = Target()) -> Target:\n"
        "    return Target()\n"
        "\n"
        "item: Target\n"
    )

    expected_lines = (
        (1, 1, 1, 4)
        if ModuleAnnotationEvaluationMode.runtime_default().annotations_execute_at_declaration
        else (1,)
    )
    assert tuple(node.lineno for node in loads) == expected_lines


def test_module_annotation_evaluation_mode_distinguishes_runtime_policies() -> None:
    assert ModuleAnnotationEvaluationMode.from_module(
        ast.parse("value: ValueType\n")
    ) is ModuleAnnotationEvaluationMode.runtime_default()
    assert ModuleAnnotationEvaluationMode.from_module(
        ast.parse("from __future__ import annotations\n\nvalue: ValueType\n")
    ) is ModuleAnnotationEvaluationMode.STRINGIZED
    assert ModuleAnnotationEvaluationMode.EAGER.annotations_execute_at_declaration
    assert not ModuleAnnotationEvaluationMode.LAZY.annotations_execute_at_declaration
    assert not ModuleAnnotationEvaluationMode.STRINGIZED.annotations_execute_at_declaration


def test_declaration_annotation_projection_excludes_function_local_annotations() -> None:
    module = ast.parse(
        "class Container:\n"
        "    field: FieldType\n"
        "\n"
        "    def build(self, value: ArgumentType) -> ReturnType:\n"
        "        local: LocalType\n"
        "\n"
        "        class Nested:\n"
        "            field: NestedType\n"
        "\n"
        "        return value\n"
    )
    declaration = module.body[0]
    assert isinstance(declaration, ast.ClassDef)

    projection = DeclarationAnnotationProjection.from_declarations((declaration,))

    assert tuple(map(ast.unparse, projection.expressions)) == (
        "FieldType",
        "ArgumentType",
        "ReturnType",
        "NestedType",
    )
