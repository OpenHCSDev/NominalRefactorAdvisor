"""Application sites and captured operands are distinct source evidence."""

import ast
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import (
    ModuleAnnotationEvaluationMode,
    ParsedModule,
)
from nominal_refactor_advisor.class_namespace import (
    AssignmentSourceEffect,
    BindingSourceEffect,
    ClassNamespaceExecutionEvidence,
    DescriptorClassNamespaceEffect,
    LiteralSourceEffect,
    CallClassNamespaceEffect,
    SourceExecutionEffectEvidence,
    _ClassNamespaceEffectProjection,
)
from nominal_refactor_advisor.declaration_dependencies import (
    _DeclarationDependencyCollector,
)
from nominal_refactor_advisor.product_flow import (
    CompactFunctionCall,
    CompactMutation,
    source_product_flow_projection,
)


def fixture(body):
    text = (
        "class Owner:\n" + "\n".join("    " + line for line in body.splitlines()) + "\n"
    )
    module = ParsedModule(
        Path("effect_sites.py"), "effect_sites", False, ast.parse(text), text
    )
    owner = module.module.body[0]
    return module, owner, ClassNamespaceExecutionEvidence.from_class(owner)


def test_decorator_application_site_is_definition_not_captured_operand():
    module, owner, evidence = fixture("@property\ndef item(self): return 1")
    definition = owner.body[0]
    application = next(site for site in evidence.sites if site.trigger is definition)
    (effect,) = (
        effect
        for effect in application.effects
        if isinstance(effect, DescriptorClassNamespaceEffect)
    )
    assert isinstance(effect, DescriptorClassNamespaceEffect)
    assert any(
        isinstance(effect, BindingSourceEffect) for effect in application.effects
    )
    assert effect.operand is definition.decorator_list[0]
    assert application.trigger is not effect.operand
    observed = source_product_flow_projection(module)
    read = observed.reference_reads_by_node[effect.operand]
    mutation = next(
        site
        for site in observed.operations
        if site.node is definition and isinstance(site.event, CompactMutation)
    )
    assert read.context.flow.owner is mutation.owner
    assert read.use.position.dominates(mutation.position)
    assert not mutation.position.may_precede(read.use.position)


@pytest.mark.parametrize("future", (False, True))
def test_definition_projection_reuses_original_sites_and_annotation_policy(future):
    source = (
        "from __future__ import annotations\n" if future else ""
    ) + "def chosen(value: annotation() = default()): pass\n"
    module = ParsedModule(Path("phase.py"), "phase", False, ast.parse(source), source)
    evidence = SourceExecutionEffectEvidence.from_source(module.module)
    definition = module.module.body[-1]
    annotation = definition.args.args[0].annotation
    default = definition.args.defaults[0]
    original_sites = evidence.sites
    projected = evidence.definition_effects(module.module, module)
    eager = ModuleAnnotationEvaluationMode.from_module(
        module.module
    ).annotations_execute_at_declaration
    assert any(site.trigger is annotation for site, _ in projected) is eager
    assert any(site.trigger is default for site, _ in projected)
    assert any(site.trigger is annotation for site in evidence.sites)
    assert evidence.sites is original_sites
    for site, effect in projected:
        assert any(site is original for original in original_sites)
        assert any(effect is original for original in site.effects)


def test_definition_projection_derives_module_policy_once_per_query(monkeypatch):
    module, owner, _ = fixture("first: object\nsecond: object\nthird: object")
    evidence = SourceExecutionEffectEvidence.from_source(module.module)
    original = ModuleAnnotationEvaluationMode.from_module.__func__
    seen = []

    def observed(cls, source):
        seen.append(source)
        return original(cls, source)

    monkeypatch.setattr(
        ModuleAnnotationEvaluationMode, "from_module", classmethod(observed)
    )
    evidence.definition_effects(owner, module)
    assert seen == [module.module]


def test_decorator_factory_invocation_and_returned_decorator_application_survive():
    module, owner, evidence = fixture("@factory()\ndef item(self): return 1")
    definition = owner.body[0]
    factory_call = definition.decorator_list[0]
    factory_site = next(site for site in evidence.sites if site.trigger is factory_call)
    application_site = next(
        site for site in evidence.sites if site.trigger is definition
    )
    assert isinstance(factory_site.effects[0], CallClassNamespaceEffect)
    assert factory_site.effects[0].node.func is factory_call.func
    (application_effect,) = (
        effect
        for effect in application_site.effects
        if isinstance(effect, DescriptorClassNamespaceEffect)
    )
    assert application_effect.operand is factory_call
    observed = source_product_flow_projection(module)
    invocation = next(
        site
        for site in observed.operations
        if site.node is factory_call and isinstance(site.event, CompactFunctionCall)
    )
    application = next(
        site
        for site in observed.operations
        if site.node is definition and isinstance(site.event, CompactMutation)
    )
    assert invocation.position.dominates(application.position)
    program = """events = []
def factory():
    events.append("invoke")
    def decorate(function):
        events.append("apply")
        return function
    return decorate
""" + module.source + "\nprint(events)\n"
    assert (
        subprocess.check_output(
            [sys.executable, "-I", "-c", program], text=True
        ).strip()
        == "['invoke', 'apply']"
    )


def test_call_construction_and_result_installation_retain_separate_triggers():
    _, owner, evidence = fixture("item = property(lambda: None)")
    assignment = owner.body[0]
    call = assignment.value
    installation = next(site for site in evidence.sites if site.trigger is assignment)
    invocation = next(site for site in evidence.sites if site.trigger is call)
    assert isinstance(installation.effects[0], AssignmentSourceEffect)
    assert isinstance(invocation.effects[0], CallClassNamespaceEffect)
    assert installation.effects[0].node is assignment
    assert installation.effects[0].node.value is invocation.effects[0].node is call
    assert invocation.effects[0].node.func is call.func


def test_same_operand_in_distinct_obligations_is_not_globally_deduplicated():
    _, owner, evidence = fixture("item = 3")
    assignment = owner.body[0]
    sites = tuple(
        site
        for site in evidence.sites
        if site.trigger is assignment or site.trigger is assignment.value
    )
    assert tuple(site.trigger for site in sites) == (assignment, assignment.value)
    assert isinstance(sites[0].effects[0], AssignmentSourceEffect)
    assert isinstance(sites[1].effects[0], LiteralSourceEffect)
    assert sites[0].effects[0].node.value is sites[1].effects[0].node
    assert sites[0].effects[0] is not sites[1].effects[0]


def test_node_local_projection_never_visits_children_or_keeps_prior_invocations():
    _, owner, _ = fixture("item = property(callback())")
    assignment = owner.body[0]
    scope = _DeclarationDependencyCollector()
    first = _ClassNamespaceEffectProjection.project(assignment, scope)
    second = _ClassNamespaceEffectProjection.project(assignment, scope)
    assert len(first) == len(second) == 1
    assert isinstance(first[0], AssignmentSourceEffect)
    assert first[0].node is second[0].node is assignment
    assert first[0].node.value is assignment.value
    assert first[0] is not second[0]


def test_effect_sites_retain_existing_single_ordered_traversal(monkeypatch):
    invocations = []
    original = _ClassNamespaceEffectProjection.project.__func__

    def project(cls, trigger, scope):
        invocations.append(trigger)
        return original(cls, trigger, scope)

    monkeypatch.setattr(
        _ClassNamespaceEffectProjection, "project", classmethod(project)
    )
    _, owner, evidence = fixture(
        "item = property(lambda: None)\ndef method(self): return callback()"
    )
    deferred_call = owner.body[1].body[0].value
    # The common source collector inventories the deferred scope separately;
    # the existing class execution facade must not charge it to class creation.
    assert deferred_call in invocations
    assert all(site.trigger is not deferred_call for site in evidence.sites)
    assert len(invocations) == len({id(node) for node in invocations})
    assert all(site.trigger in invocations for site in evidence.sites)


def test_flat_effects_are_derived_from_sites_and_not_separately_stored():
    _, _, evidence = fixture("item = staticmethod(lambda: None)")
    assert {field.name for field in fields(evidence)} == {"binding_names", "sites"}
    expected = tuple(effect for site in evidence.sites for effect in site.effects)
    assert all(
        actual is original
        for actual, original in zip(evidence.effects, expected, strict=True)
    )


def test_unknown_truth_operand_remains_an_obligation_on_its_control_trigger():
    _, owner, evidence = fixture("if flag:\n    pass")
    control = owner.body[0]
    (site,) = evidence.sites
    assert site.trigger is control
    assert site.effects[0].node is control.test
    with pytest.raises(ValueError):
        site.effects[0].require_closed(None)


@pytest.mark.parametrize("operand", ("1", "None"))
def test_generator_acquisition_rejects_noniterable_literals(operand):
    _, owner, evidence = fixture(f"items = (item for item in {operand})")
    generator = owner.body[0].value
    site = next(site for site in evidence.sites if site.trigger is generator)
    with pytest.raises(ValueError, match="iterator acquisition"):
        site.effects[0].require_closed(None)
