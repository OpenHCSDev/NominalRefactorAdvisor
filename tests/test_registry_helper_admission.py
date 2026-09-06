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
from nominal_refactor_advisor.declaration_dependencies import (
    ModuleLexicalDependencyProjection,
)
from nominal_refactor_advisor.native_reference import ScopedNativeReference
from nominal_refactor_advisor.projection_descent_codemod import (
    _TypeKeyedBehaviorSourceDerivation,
)
from nominal_refactor_advisor.registry_identity import mro_registry_value


@pytest.mark.parametrize("counterfeit", (False, True))
def test_type_keyed_descent_preserves_selected_helper_behavior(
    tmp_path, monkeypatch, counterfeit
) -> None:
    import sys

    source = _type_keyed_behavior_projection_source()
    if counterfeit:
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
    try:
        result = plan.simulate(snapshot)
    except (ValueError, CodemodOperationPreflightError):
        assert counterfeit, "Canonical supported helper unexpectedly rejected"
        assert path.read_text() == source
        return
    if not result.is_clean:
        assert counterfeit, "Canonical supported helper unexpectedly rejected"
        return
    outputs = []
    for name, text in (
        ("before", source),
        ("after", result.final_snapshot.sources_by_file_path[str(path)]),
    ):
        runtime = ModuleType(f"registry_identity_{name}")
        monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
        exec(text, runtime.__dict__)
        event = runtime.NamedEvent()
        event.name = "specific"
        event.value = "fallback"
        outputs.append(runtime.render_event(event))
    assert (
        outputs[0] == outputs[1]
    ), "A clean refactor must preserve the actual selected helper behavior"


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


def test_helper_selection_obeys_the_shared_native_gate(monkeypatch) -> None:
    parsed, method, snapshot = _helper_method_source(
        _type_keyed_behavior_projection_source()
    )
    assert _TypeKeyedBehaviorSourceDerivation._is_mro_lookup_method(
        snapshot, parsed.file_path, method
    )
    invocations = []

    def reject(self, environment, declarations):
        invocations.append((self, environment, declarations))
        raise ValueError("Native helper admission remains unproved")

    monkeypatch.setattr(ScopedNativeReference, "require_native", reject)
    assert not _TypeKeyedBehaviorSourceDerivation._is_mro_lookup_method(
        snapshot, parsed.file_path, method
    )
    ((reference, environment, declarations),) = invocations
    assert reference.node is method.body[0].value.func
    assert environment.module is snapshot.parsed_module_for_source_path(
        parsed.file_path
    )
    assert len(declarations) == 1
    assert declarations[0].declaration is mro_registry_value


@pytest.mark.parametrize(
    "import_source, expression",
    (
        (
            "from nominal_refactor_advisor.registry_identity "
            "import mro_registry_value as selected",
            "selected",
        ),
        (
            "from nominal_refactor_advisor import registry_identity as helpers",
            "helpers.mro_registry_value",
        ),
    ),
)
def test_shared_helper_gate_accepts_declared_import_aliases(
    import_source, expression
) -> None:
    source = (
        _type_keyed_behavior_projection_source()
        .replace(
            "from nominal_refactor_advisor.registry_identity import mro_registry_value",
            import_source,
        )
        .replace(
            "mro_registry_value(cls.__registry__, type(event))",
            f"{expression}(cls.__registry__, type(event))",
        )
    )
    parsed, method, snapshot = _helper_method_source(source)
    assert _TypeKeyedBehaviorSourceDerivation._is_mro_lookup_method(
        snapshot, parsed.file_path, method
    )


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
    assert not _TypeKeyedBehaviorSourceDerivation._is_mro_lookup_method(
        snapshot, parsed.file_path, method
    )

    # Execute only this controlled fixture to demonstrate that the actual helper
    # is the supplied local object, not the identically named module import.
    namespace = {}
    exec(compile(source, "<trusted-helper-shadow>", "exec"), namespace)

    def replacement(registry, declaration):
        return None

    if shadow_scope == "parameter":
        assert namespace["Family"].projection_for(object(), replacement) is None
    else:
        assert namespace["enclosing"](replacement).projection_for(object()) is None


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
        assert _TypeKeyedBehaviorSourceDerivation._is_mro_lookup_method(
            snapshot, parsed.file_path, method
        )
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
    assert not _TypeKeyedBehaviorSourceDerivation._is_mro_lookup_method(
        changed, parsed.file_path, current_method
    )
    assert _TypeKeyedBehaviorSourceDerivation._is_mro_lookup_method(
        snapshot, parsed.file_path, method
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
