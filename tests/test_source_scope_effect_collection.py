"""Source scope ownership is retained separately from runtime activation."""

import ast
import pickle
import sys

import pytest

from nominal_refactor_advisor.class_namespace import (
    ClassNamespaceExecutionEvidence,
    SourceExecutionEffectEvidence,
    _ClassNamespaceEffectProjection,
    _SourceExecutionEffectCollector,
)
from nominal_refactor_advisor.declaration_dependencies import (
    DeclarationDependencyUse,
    ModuleBindingResolutionPhase,
    ModuleLexicalDependencyProjection,
    _DeclarationDependencyCollector,
)
from nominal_refactor_advisor.lexical_scopes import (
    ClassNamespaceScope,
    FunctionBindingProjection,
    LexicalNameResolution,
    ModuleNamespaceScope,
    ScopeBindingProjection,
    TypeParameterScope,
)


def collect(source):
    root = ast.parse(source)
    return root, SourceExecutionEffectEvidence.from_source(root)


def test_module_and_class_effects_keep_actual_owners_and_original_sites():
    root, evidence = collect("top = 1\nclass Owner:\n    item = 2\n")
    owner = root.body[1]
    module_sites = evidence.sites_by_owner[root]
    class_sites = evidence.sites_by_owner[owner]
    assert any(site.trigger is root.body[0] for site in module_sites)
    assert any(site.trigger is owner for site in module_sites)
    assert all(site.scope_path[-1].node is owner for site in class_sites)
    assert all(site.scope_path[0].node is root for site in evidence.sites)
    assert any(site.trigger is owner.body[0] for site in class_sites)
    assert tuple(site for site in evidence.sites if site in class_sites) == class_sites


def test_function_defaults_are_parent_owned_and_body_is_separately_deferred():
    root, evidence = collect("def operation(value=prepare()):\n    return execute()\n")
    function = root.body[0]
    default = function.args.defaults[0]
    invocation = function.body[0].value
    assert any(site.trigger is default for site in evidence.sites_by_owner[root])
    body_sites = evidence.sites_by_owner[function]
    assert any(site.trigger is invocation for site in body_sites)
    assert all(
        site.binding_phase is ModuleBindingResolutionPhase.FINAL_MODULE
        for site in body_sites
    )
    assert not any(site.trigger is invocation for site in evidence.sites_by_owner[root])


def test_lambda_body_and_generator_body_do_not_become_module_effects():
    root, evidence = collect(
        "callback = lambda value=prepare(): execute()\nitems = (consume(x) for x in inputs())\n"
    )
    function = root.body[0].value
    generator = root.body[1].value
    assert any(
        site.trigger is function.body for site in evidence.sites_by_owner[function]
    )
    assert any(
        site.trigger is generator.elt for site in evidence.sites_by_owner[generator]
    )
    assert all(
        site.binding_phase is ModuleBindingResolutionPhase.FINAL_MODULE
        for site in evidence.sites_by_owner[generator]
    )
    assert any(
        site.trigger is generator.generators[0].iter
        for site in evidence.sites_by_owner[root]
    )


def test_nested_class_headers_keep_enclosing_scope_and_children_keep_their_own():
    root, evidence = collect(
        "class Outer:\n    class Inner(factory()):\n        item = 3\n"
    )
    outer = root.body[0]
    inner = outer.body[0]
    assert any(
        site.trigger is inner.bases[0] for site in evidence.sites_by_owner[outer]
    )
    assert any(site.trigger is inner.body[0] for site in evidence.sites_by_owner[inner])
    assert all(
        tuple(scope.node for scope in site.scope_path) == (root, outer, inner)
        for site in evidence.sites_by_owner[inner]
    )


def test_existing_class_evidence_projects_same_order_bindings_and_deferred_exclusion():
    root, common = collect(
        "class Outer:\n    item = 3\n    class Inner:\n        value = 4\n    def operation(self):\n        return callback()\n"
    )
    outer = root.body[0]
    from_common = common.class_evidence(outer)
    direct = ClassNamespaceExecutionEvidence.from_class(outer)
    assert from_common.binding_names == direct.binding_names
    assert tuple(site.trigger for site in from_common.sites) == tuple(
        site.trigger for site in direct.sites
    )
    assert tuple(
        tuple(type(effect) for effect in site.effects) for site in from_common.sites
    ) == tuple(tuple(type(effect) for effect in site.effects) for site in direct.sites)
    assert "Inner" in direct.binding_names
    assert "operation" in direct.binding_names
    deferred = outer.body[-1].body[0].value
    assert not any(site.trigger is deferred for site in direct.sites)
    assert any(
        site.trigger is deferred for site in common.sites_by_owner[outer.body[-1]]
    )


def test_original_annotation_use_is_retained_without_execution_admission():
    root, evidence = collect("class Owner:\n    item: describe()\n")
    owner = root.body[0]
    annotation = owner.body[0].annotation
    site = next(
        site for site in evidence.sites_by_owner[owner] if site.trigger is annotation
    )
    assert all(
        effect.use is DeclarationDependencyUse.EVALUATED_ANNOTATION
        for effect in site.effects
    )


def test_common_collection_invokes_projection_once_per_existing_visit(monkeypatch):
    original = _ClassNamespaceEffectProjection.project.__func__
    nodes = []

    def project(cls, node, scope):
        nodes.append(node)
        return original(cls, node, scope)

    monkeypatch.setattr(
        _ClassNamespaceEffectProjection, "project", classmethod(project)
    )
    root, evidence = collect("class Owner:\n    item = property(lambda: None)\n")
    assert len(nodes) == len({id(node) for node in nodes})
    assert all(site.trigger in nodes for site in evidence.sites)
    assert root not in nodes


def test_scope_source_owner_is_required_not_a_synthetic_default():
    function = ast.parse("def operation(value): pass").body[0]
    scope = FunctionBindingProjection.from_function(function)
    assert scope.node is function
    assert scope.local_names == frozenset({"value"})
    assert scope.declarations is scope
    with pytest.raises(TypeError):
        ScopeBindingProjection.from_nodes(function.body)


def test_module_scope_keeps_existing_external_name_resolution():
    root = ast.parse("value = 1")
    scope = ModuleNamespaceScope(root)
    assert scope.node is root
    assert scope.declarations.node is root
    assert scope.declarations.local_names == frozenset({"value"})
    assert scope.local_resolution_for("value") is None


def test_final_class_scope_retains_branch_uncertainty_without_new_state_copy():
    root, evidence = collect("class Owner:\n    if condition:\n        value = 3\n")
    owner = root.body[0]
    scope = next(
        scope
        for scope in evidence.completed_scopes
        if isinstance(scope, ClassNamespaceScope)
    )
    assert scope.node is owner
    assert scope.bindings["value"] is LexicalNameResolution.UNPROVED
    assert "value" in evidence.class_evidence(owner).binding_names


def test_pickle_preserves_site_scope_and_owner_object_sharing():
    root, evidence = collect("class Owner:\n    item = 1\n")
    restored_root, restored = pickle.loads(pickle.dumps((root, evidence)))
    owner = restored_root.body[0]
    namespace = next(
        scope for scope in restored.completed_scopes if scope.node is owner
    )
    assert all(
        site.scope_path[-1] is namespace for site in restored.sites_by_owner[owner]
    )


def test_duplicate_or_missing_class_completion_does_not_choose_a_scope():
    root, evidence = collect("class Owner:\n    item = 1\n")
    owner = root.body[0]
    with pytest.raises(ValueError, match="one actual"):
        SourceExecutionEffectEvidence(evidence.sites, ()).class_evidence(owner)
    with pytest.raises(ValueError, match="one actual"):
        SourceExecutionEffectEvidence(
            evidence.sites, evidence.completed_scopes * 2
        ).class_evidence(owner)


def test_explicit_module_scope_preserves_variable_annotation_inventory():
    root = ast.parse(
        "value: describe()\nclass Owner:\n    item: describe()\ndef method():\n    local: describe()\n"
    )
    collector = _SourceExecutionEffectCollector()
    collector.visit(root)
    assert collector.annotation_count == 2


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Native type parameter syntax requires Python3.12",
)
def test_same_definition_owner_does_not_collapse_header_and_body_scope_receipts():
    root, evidence = collect(
        "def operation[T](value: describe(T)):\n    return execute()\n"
    )
    function = root.body[0]
    sites = evidence.sites_by_owner[function]
    header = next(
        site for site in sites if site.trigger is function.args.args[0].annotation
    )
    body = next(site for site in sites if site.trigger is function.body[0].value)
    assert isinstance(header.scope_path[-1], TypeParameterScope)
    assert type(body.scope_path[-1]) is FunctionBindingProjection
    assert header.scope_path[-1] is not body.scope_path[-1]
    assert header.binding_phase is ModuleBindingResolutionPhase.SOURCE_POSITION
    assert body.binding_phase is ModuleBindingResolutionPhase.FINAL_MODULE


def test_module_dependency_consumer_enters_the_same_actual_module_scope(monkeypatch):
    root = ast.parse("value: annotation = external\n")
    scopes = []
    original = _DeclarationDependencyCollector._record_reference

    def record(self, node, resolution):
        scopes.append(tuple(self.scopes))
        return original(self, node, resolution)

    monkeypatch.setattr(_DeclarationDependencyCollector, "_record_reference", record)
    dependencies = ModuleLexicalDependencyProjection.from_module(root)
    assert dependencies.annotation_count == 1
    assert {node.id for node in dependencies.external_name_references} == {
        "annotation",
        "external",
    }
    assert scopes and all(path[-1].node is root for path in scopes)


@pytest.mark.parametrize("body", ("pass", "@property\ndef cached(self): return 1"))
def test_whole_module_class_facade_does_not_mistake_deferred_entry_for_empty_effects(
    body,
):
    source = (
        "def factory():\n    class Local:\n"
        + "\n".join("        " + line for line in body.splitlines())
        + "\n"
    )
    root, evidence = collect(source)
    owner = root.body[0].body[0]
    entry = next(site for site in evidence.sites if site.trigger is owner)
    assert entry.binding_phase is ModuleBindingResolutionPhase.FINAL_MODULE
    with pytest.raises(ValueError, match="deferred activation"):
        evidence.class_evidence(owner)
    standalone = ClassNamespaceExecutionEvidence.from_class(owner)
    if body != "pass":
        assert standalone.effects
        assert evidence.sites_by_owner[owner]
