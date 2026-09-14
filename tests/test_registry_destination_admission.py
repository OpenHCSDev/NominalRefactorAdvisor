"""Destination registration must preserve keys accepted by the source mapping."""

from types import ModuleType

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    DeriveAutoregisterInstanceViewOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.source_index import build_source_index
from test_registry_key_equivalence import _parsed


def _manual_source(key, *, dictionary, compact):
    body = " pass\n" if compact else "\n    pass\n"
    declarations = f"class Alpha:{body}class Beta:{body}"
    if dictionary:
        return declarations + f"REGISTRY = {{{key}: Alpha, 'beta': Beta}}\n"
    return (
        "REGISTRY = {}\n"
        + declarations
        + f"REGISTRY[{key}] = Alpha\nREGISTRY['beta'] = Beta\n"
    )


def _instance_source(key, *, compact):
    body = " pass\n" if compact else "\n    pass\n"
    return (
        "from metaclass_registry import AutoRegisterMeta\n"
        "class Handler(metaclass=AutoRegisterMeta):\n"
        "    __registry__ = {}\n"
        "    __registry_key__ = 'registry_key'\n"
        "    __skip_if_no_key__ = True\n"
        f"class Alpha(Handler):{body}"
        f"class Beta(Handler):{body}"
        f"REGISTRY = {{{key}: Alpha(), 'beta': Beta()}}\n"
    )


@pytest.mark.parametrize("key", ("None", "False", "0", "''", "()", "...", "-1"))
@pytest.mark.parametrize("form", ("writes", "dictionary", "instances"))
@pytest.mark.parametrize("compact", (False, True), ids=("block", "inline"))
def test_conversion_preserves_target_keys_or_rejects_missing_registration(
    key, form, compact
):
    source = (
        _instance_source(key, compact=compact)
        if form == "instances"
        else _manual_source(key, dictionary=form == "dictionary", compact=compact)
    )
    before = ModuleType("registry_destination_before")
    exec(compile(source, "<authored-registration-fixture>", "exec"), before.__dict__)
    original_keys = tuple(before.REGISTRY)
    assert len(original_keys) == 2
    parsed = _parsed(source)
    snapshot = CodemodSourceSnapshot.from_indexed_sources(
        build_source_index([parsed], ()), {parsed.file_path: source}
    )
    operation_type, anchor = (
        (DeriveAutoregisterInstanceViewOperation, "Handler")
        if form == "instances"
        else (ConvertManualRegistryToAutoregisterOperation, "Alpha")
    )
    operation = operation_type(
        target=SourceRewriteTarget(file_path=parsed.file_path, qualname=anchor)
    )
    if key == "None":
        with pytest.raises(ValueError, match="registration"):
            operation.source_edits_from_snapshot(snapshot)
        return

    result = (
        RefactorRecipe("registration-key-preservation")
        .with_operation(operation)
        .simulate(snapshot)
    )
    assert result.is_clean
    after = ModuleType("registry_destination_after")
    exec(
        compile(
            result.simulation.rewritten_sources[parsed.file_path],
            "<authored-converted-registration-fixture>",
            "exec",
        ),
        after.__dict__,
    )
    assert type(after.REGISTRY) is dict
    assert tuple(after.REGISTRY) == original_keys
    expected = (
        (after.Alpha, after.Beta)
        if form != "instances"
        else tuple(type(instance) for instance in after.REGISTRY.values())
    )
    actual = (
        tuple(after.REGISTRY.values())
        if form != "instances"
        else (after.Alpha, after.Beta)
    )
    assert actual == expected


def test_generated_instance_method_does_not_shadow_original_body_reads():
    source = (
        _instance_source("'alpha'", compact=False)
        .replace(
            "class Handler(metaclass=AutoRegisterMeta):",
            "def instances_by_registry_key():\n"
            "    return 'global marker'\n"
            "class Handler(metaclass=AutoRegisterMeta):",
        )
        .replace(
            "    __skip_if_no_key__ = True\n",
            "    __skip_if_no_key__ = True\n"
            "    def ordinary_method(self):\n"
            "        pass\n"
            "    marker = instances_by_registry_key()\n",
        )
    )
    parsed = _parsed(source)
    before = ModuleType("instance_view_body_before")
    exec(source, before.__dict__)
    assert before.Handler.marker == "global marker"
    snapshot = CodemodSourceSnapshot.from_indexed_sources(
        build_source_index([parsed], ()), {parsed.file_path: source}
    )
    result = (
        RefactorRecipe("preserve-body-reads")
        .with_operation(
            DeriveAutoregisterInstanceViewOperation(
                target=SourceRewriteTarget(
                    file_path=parsed.file_path, qualname="Handler"
                )
            )
        )
        .simulate(snapshot)
    )
    assert result.is_clean
    after = ModuleType("instance_view_body_after")
    exec(result.simulation.rewritten_sources[parsed.file_path], after.__dict__)
    assert after.Handler.marker == before.Handler.marker
    assert tuple(after.REGISTRY) == ("alpha", "beta")
