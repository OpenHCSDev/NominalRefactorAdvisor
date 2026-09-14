"""Explicit entry premises retain source identity without inventing a frame."""

import ast
import builtins
from dataclasses import dataclass, replace
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceEffectsABC,
    CapturedReferenceKernel,
    CapturedReferenceViolation,
    InitialNativeIsland,
    NamespaceEvidenceABC,
    OpenCapturedReference,
    SingleFlowPrefix,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import (
    CompactCallableReferenceUse,
    CompactFlowRead,
    source_product_flow_projection,
)
from nominal_refactor_advisor.source_entry import SourceModuleEntryPremise
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def _source(text="result = property\n"):
    module = ParsedModule(Path("entry.py"), "entry", False, ast.parse(text), text)
    return module, source_product_flow_projection(module)


def _entry(text="result = property\n", bindings=None):
    module, source = _source(text)
    island = InitialNativeIsland((builtins,))
    facts = {"__builtins__": CapturedNativeObject(builtins)}
    if bindings is not None:
        facts.update(bindings)
    return module, SourceModuleEntryPremise(
        source, island, facts, island.namespace_for_storage(vars(builtins))
    )


def test_source_entry_itself_owns_fresh_namespace_and_canonical_frame():
    _, entry = _entry()
    assert isinstance(entry, NamespaceEvidenceABC)
    assert entry.initial is entry.native_island
    assert entry.context is entry.source.compact.flow_contexts[0]
    assert entry.frame is entry.frame
    assert entry.frame.locals is entry.frame.globals is entry
    assert entry.frame.builtins is entry.initial.namespace_for_storage(vars(builtins))
    entry.initial.require_frame(entry.frame)
    assert not entry.is_initial_storage(vars(builtins))
    assert not entry.is_initial_storage({})


def test_initial_facts_are_immutable_and_do_not_mirror_source_bindings():
    _, source = _source("alias = property\nresult = alias\n")
    island = InitialNativeIsland((builtins,))
    unknown = OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
    facts = {"known_unknown": unknown}
    entry = SourceModuleEntryPremise(
        source, island, facts, island.namespace_for_storage(vars(builtins))
    )
    facts["known_unknown"] = CapturedNativeObject(property)
    facts["alias"] = CapturedNativeObject(property)
    assert entry.member("known_unknown") is unknown
    assert entry.member("alias") is None
    with pytest.raises(TypeError):
        entry.initial_entries["alias"] = CapturedNativeObject(property)


def test_entry_rejects_equal_foreign_context_and_foreign_native_admission():
    _, entry = _entry()
    _, equal_source = _source()
    foreign = equal_source.compact.flow_contexts[0]
    assert foreign == entry.context
    assert foreign is not entry.context
    with pytest.raises(ValueError, match="actual flow context"):
        entry.require_context(foreign)
    with pytest.raises(ValueError, match="different entry premise"):
        entry.require_admitted(InitialNativeIsland((builtins,)))
    other = SourceModuleEntryPremise(
        entry.source, entry.initial, {}, entry.frame.builtins
    )
    assert entry is not other
    assert entry.frame is not other.frame


@pytest.mark.parametrize("flows", ((), "duplicate", "child_only"))
def test_entry_requires_exactly_one_module_flow(flows):
    _, entry = _entry("class Body: pass\n")
    if flows == "duplicate":
        flows = (entry.context.flow, entry.context.flow)
    elif flows == "child_only":
        flows = tuple(
            flow
            for flow in entry.source.compact.flows
            if not flow.owner.kind.is_module_scope
        )
    source = replace(entry.source, compact=replace(entry.source.compact, flows=flows))
    with pytest.raises(ValueError, match="one actual module flow"):
        SourceModuleEntryPremise(source, entry.initial, {}, entry.frame.builtins)


def test_unknown_captured_builtins_does_not_erase_known_globals():
    _, entry = _entry()
    unknown = OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
    explicit = SourceModuleEntryPremise(
        entry.source,
        entry.initial,
        {"builtins": CapturedNativeObject(builtins)},
        unknown,
    )
    explicit.initial.require_frame(explicit.frame)
    assert explicit.member("builtins").value is builtins
    assert explicit.frame.builtins is unknown


@pytest.mark.parametrize(
    "facts", ({"property": property}, {(1,): CapturedNativeObject(property)})
)
def test_initial_facts_require_typed_evidence_and_exact_scalar_keys(facts):
    _, entry = _entry()
    with pytest.raises(TypeError):
        SourceModuleEntryPremise(
            entry.source, entry.initial, facts, entry.frame.builtins
        )


def test_foreign_builtin_namespace_cannot_be_used_in_entry():
    _, entry = _entry()
    foreign = InitialNativeIsland((builtins,))
    with pytest.raises(ValueError, match="different admission"):
        SourceModuleEntryPremise(
            entry.source,
            entry.initial,
            {},
            foreign.namespace_for_storage(vars(builtins)),
        )


@pytest.mark.parametrize(
    "text",
    ("result = property\n", "result = builtins.property\n", "property(argument())\n"),
)
def test_source_node_lookup_retains_canonical_compact_read(text):
    module, source = _source(text)
    sites = tuple(
        site
        for site in source.operations
        if isinstance(site.event, CompactCallableReferenceUse)
    )
    assert sites
    for site in sites:
        canonical = source.compact.reference_reads_by_span[
            SourceByteSpan.require_node(site.node)
        ]
        assert source.reference_reads_by_node[site.node] is canonical
        copied_node = ast.copy_location(
            ast.Name(id="property", ctx=ast.Load()), site.node
        )
        assert copied_node not in source.reference_reads_by_node
    foreign_tree = ast.parse(text)
    assert all(
        node not in source.reference_reads_by_node for node in ast.walk(foreign_tree)
    )


def test_duplicate_source_read_sites_do_not_choose_one_arbitrarily():
    _, source = _source()
    site = next(
        site
        for site in source.operations
        if isinstance(site.event, CompactCallableReferenceUse)
    )
    duplicated = replace(source, operations=(*source.operations, site))
    assert site.node not in duplicated.reference_reads_by_node


@pytest.mark.parametrize("component", ("event", "owner"))
def test_equal_but_foreign_operation_identity_cannot_authenticate_source_read(
    component,
):
    _, source = _source()
    site = next(
        site
        for site in source.operations
        if isinstance(site.event, CompactCallableReferenceUse)
    )
    corrupted = replace(site, **{component: replace(getattr(site, component))})
    mismatched = replace(
        source,
        operations=tuple(
            corrupted if item is site else item for item in source.operations
        ),
    )
    assert site.node not in mismatched.reference_reads_by_node


@dataclass(frozen=True)
class _FirstReadEffects(CapturedReferenceEffectsABC):
    """A fixture proves only its initial exact-dictionary name lookup.

    The program consists of a single name read and subsequent assignment; the
    selected source position precedes that first write. There are no earlier
    expressions, statements, imports, calls, or implicit operators to admit.
    """

    entry: SourceModuleEntryPremise
    read: CompactFlowRead

    def admit(self, context, position):
        self.entry.require_context(context)
        assert position is self.read.use.position
        assert position.event_index == 0
        assert not self.entry.context.flow.calls
        assert len(self.entry.context.flow.mutations) == 1
        return SingleFlowPrefix(context, self.entry.frame, position)


@pytest.mark.parametrize("shadow", (False, True))
def test_supplied_source_entry_agrees_with_isolated_actual_native_execution(shadow):
    facts = {"property": CapturedNativeObject(object)} if shadow else {}
    module, entry = _entry(bindings=facts)
    read = entry.source.reference_reads_by_node[module.module.body[0].value]
    result = CapturedReferenceKernel(
        entry.initial, _FirstReadEffects(entry, read)
    ).read(read)
    expected = object if shadow else property
    assert (
        result.require_native_identity(NativeDeclaration(expected)).declaration
        is expected
    )
    program = (
        "import builtins\ng = {'__builtins__': builtins}\n"
        + ("g['property'] = object\n" if shadow else "")
        + f"exec({module.source!r}, g)\nprint(g['result'] is {expected.__name__})\n"
    )
    assert (
        subprocess.check_output(
            [sys.executable, "-I", "-c", program], text=True
        ).strip()
        == "True"
    )


def test_known_unknown_initial_binding_does_not_fall_through_to_native_builtin():
    unknown = OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
    module, entry = _entry(bindings={"property": unknown})
    read = entry.source.reference_reads_by_node[module.module.body[0].value]
    outcome = CapturedReferenceKernel(
        entry.initial, _FirstReadEffects(entry, read)
    ).read(read)
    assert outcome is unknown
    with pytest.raises(ValueError, match="remains open"):
        outcome.require_native_identity(NativeDeclaration(property))
