"""Type-keyed helper consumers obey the shared native gate and lexical scope."""

import ast
from pathlib import Path
from types import ModuleType

import pytest
from registry_test_sources import _type_keyed_behavior_projection_source

from nominal_refactor_advisor.ast_tools import ParsedModule, parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DescendTypeKeyedBehaviorProjectionOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_preflight import CodemodOperationPreflightError
from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.declaration_dependencies import (
    ModuleLexicalDependencyProjection,
)
from nominal_refactor_advisor.registry_identity import mro_registry_value
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def test_counterfeit_helper_is_not_implicitly_admitted(tmp_path, monkeypatch) -> None:
    import sys

    source = _type_keyed_behavior_projection_source()
    helper_source = "def mro_registry_value(registry, cls):\n    return None\n"
    (tmp_path / "counterfeit.py").write_text(helper_source)
    helper = ModuleType("counterfeit")
    exec(helper_source, helper.__dict__)
    monkeypatch.setitem(sys.modules, "counterfeit", helper)
    source = source.replace(
        "from nominal_refactor_advisor.registry_identity import mro_registry_value",
        "from counterfeit import mro_registry_value",
    )
    path = tmp_path / "subject.py"
    path.write_text(source)
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path, use_parse_cache=False)
    )
    plan = CodemodPlanSequence.from_operations(
        (
            DescendTypeKeyedBehaviorProjectionOperation(
                target=SourceRewriteTarget(
                    file_path=str(path), qualname="EventProjection"
                )
            ),
        )
    )
    with pytest.raises((ValueError, CodemodOperationPreflightError)):
        plan.simulate(snapshot)
    assert path.read_text() == source
    runtime = ModuleType("registry_identity_counterfeit")
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    exec(source, runtime.__dict__)
    event = runtime.NamedEvent()
    event.name = "specific"
    event.value = "fallback"
    assert runtime.EventProjection.projection_for(event) is None
    assert runtime.mro_registry_value is helper.mro_registry_value
    assert (
        mro_registry_value(runtime.EventProjection.__registry__, type(event))
        is not None
    )


def _helper_method_source(source: str):
    parsed = ParsedModule(
        Path("/repo/helper.py"), "helper", False, ast.parse(source), source
    )
    (method,) = tuple(
        node
        for node in ast.walk(parsed.module)
        if isinstance(node, ast.FunctionDef) and node.name == "projection_for"
    )
    return parsed, method, CodemodSourceSnapshot.from_modules((parsed,))


def test_requirements_retain_exact_native_reads_and_declaration_owners() -> None:
    parsed, method, snapshot = _helper_method_source(
        _type_keyed_behavior_projection_source()
    )
    operation = DescendTypeKeyedBehaviorProjectionOperation(
        target=SourceRewriteTarget(
            file_path=parsed.file_path, qualname="EventProjection"
        )
    )
    expected = (
        (method.decorator_list[0], classmethod),
        (method.body[0].value.func, mro_registry_value),
        (method.body[0].value.args[1].func, type),
    )
    environment = snapshot.module_binding_proof.native_reference_environment(parsed)
    retained_reads = []
    for _ in range(3):
        requirements = operation.native_use_requirements(snapshot)
        assert len(requirements) == len(expected)
        reads = []
        for requirement, (node, native) in zip(requirements, expected, strict=True):
            assert requirement.node is node
            assert requirement.module is parsed
            assert requirement.environment is environment
            assert len(requirement.declarations) == 1
            assert requirement.declarations[0].declaration is native
            read = environment.source.reference_reads_by_node[node]
            assert read.use.source_span == SourceByteSpan.require_node(node)
            reads.append(read)
        if retained_reads:
            assert all(
                left is right for left, right in zip(reads, retained_reads, strict=True)
            )
        retained_reads = reads


@pytest.mark.parametrize("shadow_scope", ("parameter", "closure"))
def test_helper_selection_respects_actual_function_lexical_ownership(
    shadow_scope,
) -> None:
    method_source = (
        "class Family:\n"
        "    __registry__ = {}\n"
        "    @classmethod\n"
        "    def projection_for(cls, event):\n"
        "        projection_type = mro_registry_value(cls.__registry__, type(event))\n"
        "        return projection_type() if projection_type is not None else None\n"
    )
    import_source = (
        "from nominal_refactor_advisor.registry_identity import mro_registry_value\n"
    )
    if shadow_scope == "parameter":
        source = import_source + method_source.replace(
            "projection_for(cls, event)",
            "projection_for(cls, event, mro_registry_value)",
        )
    else:
        source = (
            import_source
            + "def enclosing(mro_registry_value):\n"
            + "".join("    " + line + "\n" for line in method_source.splitlines())
            + "    return Family\n"
        )
    parsed, method, snapshot = _helper_method_source(source)
    environment = snapshot.module_binding_proof.native_reference_environment(parsed)
    reference = method.body[0].value.func
    read = environment.source.reference_reads_by_node[reference]
    flow = read.context.flow
    assert flow.owner.source_span == SourceByteSpan.require_node(method)
    assert isinstance(environment.capture(reference), OpenCapturedReference)
    if shadow_scope == "parameter":
        assert flow.owner.initial_binding_for("mro_registry_value") is not None
        assert flow.lexical_scope_qualnames == ("Family.projection_for", "")
    else:
        assert flow.owner.initial_binding_for("mro_registry_value") is None
        assert flow.lexical_scope_qualnames == (
            "enclosing.Family.projection_for",
            "enclosing",
            "",
        )
        enclosing_flow = next(
            item
            for item in environment.source.compact.flows
            if item.owner.qualname == "enclosing"
        )
        assert (
            enclosing_flow.owner.initial_binding_for("mro_registry_value") is not None
        )

    # Execute only this controlled fixture to demonstrate that the actual helper
    # is the supplied local object, not the identically named module import.
    namespace = {}
    exec(compile(source, "<trusted-helper-shadow>", "exec"), namespace)

    calls = []
    selected_result = object()

    def replacement(registry, declaration):
        calls.append((registry, declaration))
        return lambda: selected_result

    if shadow_scope == "parameter":
        family = namespace["Family"]
        assert family.projection_for(object(), replacement) is selected_result
    else:
        family = namespace["enclosing"](replacement)
        assert family.projection_for(object()) is selected_result
    assert calls == [(family.__registry__, object)]
    assert calls[0][0] is family.__registry__


def test_helper_reuses_lazy_canonical_source_dependency_projection(monkeypatch):
    calls = []
    original = ModuleLexicalDependencyProjection.from_module.__func__

    def collect(cls, module):
        calls.append(module)
        return original(cls, module)

    monkeypatch.setattr(
        ModuleLexicalDependencyProjection, "from_module", classmethod(collect)
    )
    parsed, method, snapshot = _helper_method_source(
        _type_keyed_behavior_projection_source()
    )
    assert calls == []
    for _ in range(3):
        snapshot.module_lexical_dependency_projection_for_source_path(parsed.file_path)
    actual_module = snapshot.parsed_module_for_source_path(parsed.file_path)
    assert len(calls) == 1
    assert calls[0] is actual_module.module
    projection = snapshot.module_lexical_dependency_projection_for_source_path(
        parsed.file_path
    )
    assert (
        snapshot.module_lexical_dependency_projection_for_source_path(
            "/repo/./helper.py"
        )
        is projection
    )
    assert len(calls) == 1
    assert any(
        surface.reference is method.body[0].value.func
        for surface in projection.direct_name_surfaces
    )


def test_source_overlay_rebuilds_dependency_ownership_without_stale_helper_read():
    parsed, method, snapshot = _helper_method_source(
        _type_keyed_behavior_projection_source()
    )
    original = snapshot.module_lexical_dependency_projection_for_source_path(
        parsed.file_path
    )
    changed_source = parsed.source.replace(
        "projection_for(cls, event: Event)",
        "projection_for(cls, event: Event, mro_registry_value=None)",
    )
    assert changed_source != parsed.source
    changed = snapshot.with_virtual_sources({parsed.file_path: changed_source})
    assert changed is not snapshot
    current = changed.module_lexical_dependency_projection_for_source_path(
        parsed.file_path
    )
    assert current is not original
    current_module = changed.parsed_module_for_source_path(parsed.file_path)
    current_method = next(
        node
        for node in ast.walk(current_module.module)
        if isinstance(node, ast.FunctionDef) and node.name == "projection_for"
    )
    assert current_method is not method
    before = snapshot.module_binding_proof.native_reference_environment(parsed)
    after = changed.module_binding_proof.native_reference_environment(current_module)
    original_read = method.body[0].value.func
    current_read = current_method.body[0].value.func
    assert before is not after
    assert original_read in before.source.reference_reads_by_node
    assert current_read in after.source.reference_reads_by_node
    assert original_read not in after.source.reference_reads_by_node
    assert current_read not in before.source.reference_reads_by_node
    assert (
        after.source.reference_reads_by_node[
            current_read
        ].context.flow.owner.initial_binding_for("mro_registry_value")
        is not None
    )
    assert (
        snapshot.module_lexical_dependency_projection_for_source_path(parsed.file_path)
        is original
    )
    assert not any(
        surface.reference is method.body[0].value.func
        for surface in current.direct_name_surfaces
    )


def test_unchanged_overlay_preserves_source_dependency_cache():
    parsed, _, snapshot = _helper_method_source(
        _type_keyed_behavior_projection_source()
    )
    original = snapshot.module_lexical_dependency_projection_for_source_path(
        parsed.file_path
    )
    unchanged = snapshot.with_virtual_sources({parsed.file_path: parsed.source})
    assert unchanged is snapshot
    assert (
        unchanged.module_lexical_dependency_projection_for_source_path(parsed.file_path)
        is original
    )
