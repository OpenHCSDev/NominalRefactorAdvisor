"""Implicit annotation writes cannot disappear behind later source statements."""

import ast
import dis
from types import CodeType

import pytest

from nominal_refactor_advisor.ast_tools import ModuleAnnotationEvaluationMode
from nominal_refactor_advisor.class_namespace import (
    AnnotationStorageEffect,
    AssignmentSourceEffect,
    SourceExecutionEffectEvidence,
)
from test_source_function_result import execution


def annotation_effects(source):
    evidence = SourceExecutionEffectEvidence.from_source(ast.parse(source))
    return tuple(
        effect
        for site in evidence.sites
        for effect in site.effects
        if isinstance(effect, AnnotationStorageEffect)
    )


@pytest.mark.parametrize(
    "source,expected",
    (
        ("item: object\n", 1),
        ("class Owner:\n    item: object = None\n", 1),
        ("def local():\n    item: object = None\n", 0),
        ("class Owner:\n    (item): object = None\n", 0),
        ("container.item: object = None\n", 0),
    ),
)
def test_storage_scope_and_simple_target_come_from_existing_declarations(
    source, expected
):
    assert len(annotation_effects(source)) == expected


@pytest.mark.parametrize("value", ("", " = None"))
def test_annotation_does_not_invent_a_rhs_assignment(value):
    source = f"class Owner:\n    item: object{value}\n"
    evidence = SourceExecutionEffectEvidence.from_source(ast.parse(source))
    assignments = tuple(
        effect
        for site in evidence.sites
        for effect in site.effects
        if isinstance(effect, AssignmentSourceEffect)
    )
    assert len(assignments) == bool(value)
    assert len(annotation_effects(source)) == 1


@pytest.mark.parametrize("mode", tuple(ModuleAnnotationEvaluationMode))
def test_annotation_storage_is_separate_from_annotation_expression_evaluation(mode):
    (effect,) = annotation_effects("item: object = None\n")
    assert (
        effect.executes_at_definition(mode)
        is mode.stores_variable_annotations_at_definition
    )
    assert (
        ModuleAnnotationEvaluationMode.STRINGIZED.stores_variable_annotations_at_definition
    )
    assert (
        not ModuleAnnotationEvaluationMode.STRINGIZED.annotations_execute_at_declaration
    )
    assert (
        not ModuleAnnotationEvaluationMode.LAZY.stores_variable_annotations_at_definition
    )


@pytest.mark.parametrize(
    "trailing", ("", "    last = None\n", "    last = None\n    pass\n")
)
def test_overwritten_annotation_finalizer_cannot_be_hidden_by_a_later_store(trailing):
    if (
        not ModuleAnnotationEvaluationMode.runtime_default().annotations_execute_at_declaration
    ):
        pytest.skip("This runtime defers annotation evaluation")
    source = (
        "events = []\n"
        "class Marker:\n"
        "    def __del__(self):\n"
        "        events.append('released')\n"
        "class Owner:\n"
        "    item: Marker() = None\n"
        "    item: None = None\n" + trailing
    )
    namespace = {}
    exec(source, namespace)  # Execute only this authored finalizer control.
    assert namespace["events"] == ["released"]
    env = execution(source)
    with pytest.raises(ValueError):
        env.require_class_creation(env.module.module.body[-1])


@pytest.mark.parametrize("prefix", ("", "from __future__ import annotations\n"))
def test_annotation_storage_obligation_survives_a_later_plain_assignment(prefix):
    source = prefix + "class Owner:\n    item: 'str' = 'text'\n    last = None\n"
    env = execution(source)
    node = env.module.module.body[-1]
    effects = env.effects.definition_effects(node, env.module)
    selected = tuple(
        effect for _, effect in effects if isinstance(effect, AnnotationStorageEffect)
    )
    mode = ModuleAnnotationEvaluationMode.from_module(env.module.module)
    assert bool(selected) is mode.stores_variable_annotations_at_definition
    code = compile(source, "annotation_mode_fixture.py", "exec", dont_inherit=True)
    body = next(value for value in code.co_consts if isinstance(value, CodeType))
    assert (
        any(
            instruction.opname == "STORE_SUBSCR"
            for instruction in dis.get_instructions(body)
        )
        is mode.stores_variable_annotations_at_definition
    )
    if selected:
        selected[0].require_closed(env)
        env.require_class_creation(node)
