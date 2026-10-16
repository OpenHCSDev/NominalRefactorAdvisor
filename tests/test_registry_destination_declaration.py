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
from native_use_test_support import (
    simulate_with_rendered_registry_creator_support,
)
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
    result = simulate_with_rendered_registry_creator_support(
        ConvertManualRegistryToAutoregisterOperation(
            target=SourceRewriteTarget(file_path=parsed.file_path, qualname="Alpha")
        ),
        snapshot,
        recipe_id="one-registry-declaration",
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
def test_registry_membership_observation_requires_execution_proof(
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
    operation = ConvertManualRegistryToAutoregisterOperation(
        target=SourceRewriteTarget(file_path=parsed.file_path, qualname="Alpha")
    )
    with pytest.raises(ValueError, match="unproved_execution_effects"):
        operation.source_edits_from_snapshot(snapshot)
    assert before.Alpha.saw_registry is (placement == "before")
    assert before.Beta.saw_registry is (placement != "after")
