"""One proposed registry declaration drives creation and existing-base edits."""

import ast
from types import ModuleType

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.manual_registry import DirectManualRegistryComponent
from nominal_refactor_advisor.source_index import build_source_index
from test_registry_destination_admission import _manual_source
from test_registry_key_equivalence import _parsed


@pytest.mark.parametrize("existing_base", (False, True))
@pytest.mark.parametrize("dictionary", (False, True))
@pytest.mark.parametrize("compact", (False, True))
def test_registry_destination_is_explicit_and_preserves_plain_mapping(
    existing_base, dictionary, compact
):
    source = _manual_source("'alpha'", dictionary=dictionary, compact=compact)
    if existing_base:
        source = "class Handler:\n    pass\n" + source.replace(
            "class Alpha:", "class Alpha(Handler):"
        ).replace("class Beta:", "class Beta(Handler):")
    parsed = _parsed(source)
    component = DirectManualRegistryComponent.from_module_anchor(parsed.module, "Alpha")
    proposed = component.destination_authority
    assert component.destination_authority is proposed
    assert proposed.node.name == component.authority_name
    registry_node = proposed.assignment_value("__registry__")
    assert isinstance(
        registry_node, ast.Dict if dictionary or existing_base else ast.Name
    )
    snapshot = CodemodSourceSnapshot.from_indexed_sources(
        build_source_index([parsed], ()), {parsed.file_path: source}
    )
    result = (
        RefactorRecipe("one-registry-declaration")
        .with_operation(
            ConvertManualRegistryToAutoregisterOperation(
                target=SourceRewriteTarget(file_path=parsed.file_path, qualname="Alpha")
            )
        )
        .simulate(snapshot)
    )
    assert result.is_clean
    runtime = ModuleType("authored_registry_destination")
    exec(result.simulation.rewritten_sources[parsed.file_path], runtime.__dict__)
    assert type(runtime.REGISTRY) is dict
    assert runtime.REGISTRY == {"alpha": runtime.Alpha, "beta": runtime.Beta}
    authority = runtime.Alpha.__bases__[0]
    assert authority is runtime.Beta.__bases__[0]
    assert runtime.REGISTRY is authority.__registry__
    assert authority.__registry_key__ == proposed.registry_key_attribute
    assert authority.__skip_if_no_key__ is proposed.skips_missing_keys


@pytest.mark.parametrize("existing_base", (False, True))
@pytest.mark.parametrize("placement", ("before", "between", "after"))
def test_empty_registry_binding_keeps_its_original_observation_point(
    existing_base, placement
):
    source = _manual_source("'alpha'", dictionary=False, compact=False)
    if placement != "before":
        source = source.removeprefix("REGISTRY = {}\n")
        anchor = "class Beta:" if placement == "between" else "REGISTRY['alpha']"
        source = source.replace(anchor, "REGISTRY = {}\n" + anchor)
    if existing_base:
        source = "class Handler:\n    pass\n" + source.replace(
            "class Alpha:", "class Alpha(Handler):"
        ).replace("class Beta:", "class Beta(Handler):")
    source = source.replace("    pass", "    saw_registry = 'REGISTRY' in globals()")
    before = ModuleType("authored_binding_before")
    exec(source, before.__dict__)
    parsed = _parsed(source)
    snapshot = CodemodSourceSnapshot.from_indexed_sources(
        build_source_index([parsed], ()), {parsed.file_path: source}
    )
    result = (
        RefactorRecipe("preserve-binding-point")
        .with_operation(
            ConvertManualRegistryToAutoregisterOperation(
                target=SourceRewriteTarget(file_path=parsed.file_path, qualname="Alpha")
            )
        )
        .simulate(snapshot)
    )
    assert result.is_clean
    after = ModuleType("authored_binding_after")
    exec(result.simulation.rewritten_sources[parsed.file_path], after.__dict__)
    assert type(after.REGISTRY) is dict
    assert after.REGISTRY == {"alpha": after.Alpha, "beta": after.Beta}
    assert after.Alpha.saw_registry is before.Alpha.saw_registry
    assert after.Beta.saw_registry is before.Beta.saw_registry
