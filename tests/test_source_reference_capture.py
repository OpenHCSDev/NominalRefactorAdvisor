"""Source reads use one identity join and retain separate operation obligations."""

import ast
import builtins
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceEffectsABC,
    CapturedReferenceKernel,
    CapturedReferenceViolation,
    InitialNativeIsland,
    OpenCapturedReference,
    SingleFlowPrefix,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import source_product_flow_projection
from nominal_refactor_advisor.source_entry import SourceModuleEntryPremise


@dataclass(frozen=True)
class InitialReadEffects(CapturedReferenceEffectsABC):
    """Only the fixture's first lookup, before any assignment or other operation.

    All ancestors of the queried expression are exact module/name lookups.
    Module attribute lookup itself remains the kernel's obligation, including
    absent slots, module subclass dispatch and data descriptors.
    """

    entry: SourceModuleEntryPremise

    def admit(self, context, position):
        if context is not self.entry.context:
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_EFFECTS)
        assert not context.flow.calls
        assert len(context.flow.mutations) == 1
        mutation = context.flow.mutations[0]
        assert position != mutation.position
        assert position.dominates(mutation.position)
        return SingleFlowPrefix(context, self.entry.frame, position)


def fixture(expression="property", *, bindings=None, modules=()):
    text = f"result = {expression}\n"
    module = ParsedModule(
        Path("reference.py"), "reference", False, ast.parse(text), text
    )
    projection = source_product_flow_projection(module)
    island = InitialNativeIsland((builtins, *modules))
    entry = SourceModuleEntryPremise(
        projection, island, bindings or {}, island.namespace_for_storage(vars(builtins))
    )
    kernel = CapturedReferenceKernel(island, InitialReadEffects(entry))
    return module, entry, kernel


@pytest.mark.parametrize("expression", ("property", "namespace.property"))
def test_original_operand_reaches_the_canonical_read_path(expression):
    module, entry, kernel = fixture(
        expression, bindings={"namespace": CapturedNativeObject(builtins)}
    )
    node = module.module.body[0].value
    result = kernel.read_source(entry.source, node)
    assert result.require_closed() is None
    assert (
        result.require_native_identity(NativeDeclaration(property)).declaration
        is property
    )
    assert kernel.read(entry.source.reference_reads_by_node[node]).value is result.value


@pytest.mark.parametrize(
    "corruption", ("foreign_ast", "duplicate", "foreign_event", "foreign_owner")
)
def test_equal_source_geometry_never_substitutes_for_the_actual_read(corruption):
    module, entry, kernel = fixture()
    source = entry.source
    node = module.module.body[0].value
    if corruption == "foreign_ast":
        node = ast.parse(module.source).body[0].value
    else:
        site = next(site for site in source.operations if site.node is node)
        if corruption == "duplicate":
            source = replace(source, operations=(*source.operations, site))
        else:
            field = "event" if corruption == "foreign_event" else "owner"
            wrong = replace(site, **{field: replace(getattr(site, field))})
            source = replace(
                source,
                operations=tuple(
                    wrong if item is site else item for item in source.operations
                ),
            )
    outcome = kernel.read_source(source, node)
    assert isinstance(outcome, OpenCapturedReference)
    assert outcome.violation is CapturedReferenceViolation.UNPROVED_BINDING
    with pytest.raises(ValueError, match="capture remains open"):
        outcome.require_closed()


def test_genuine_read_from_another_activation_still_requires_effect_admission():
    _, _, kernel = fixture()
    foreign_module, foreign_entry, _ = fixture()
    result = kernel.read_source(
        foreign_entry.source, foreign_module.module.body[0].value
    )
    assert result.violation is CapturedReferenceViolation.UNPROVED_EFFECTS


def test_unknown_initial_binding_cannot_be_promoted_to_closed_access():
    unknown = OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
    module, entry, kernel = fixture(bindings={"property": unknown})
    result = kernel.read_source(entry.source, module.module.body[0].value)
    assert result is unknown
    with pytest.raises(ValueError, match="capture remains open"):
        result.require_closed()


def test_closed_attribute_capture_is_not_a_class_installation_proof():
    events = []

    class Payload:
        def __set_name__(self, owner, name):
            events.append("install")

    namespace = ModuleType("fixture_namespace")
    namespace.payload = Payload()
    module, entry, kernel = fixture(
        "namespace.payload",
        bindings={"namespace": CapturedNativeObject(namespace)},
        modules=(namespace,),
    )
    result = kernel.read_source(entry.source, module.module.body[0].value)
    assert result.require_closed() is None
    assert result.value is namespace.payload
    assert events == []
    with pytest.raises(ValueError, match="not the required native"):
        result.require_native_identity(NativeDeclaration(property))
    type("Owner", (), {"value": result.value})
    assert events == ["install"]


@pytest.mark.parametrize("existing", (False, True))
def test_native_module_getattr_is_never_executed_to_prove_a_capture(existing):
    events = []

    def missing(name):
        events.append(name)
        return property

    namespace = ModuleType("fixture_namespace")
    namespace.__getattr__ = missing
    if existing:
        namespace.value = property
    module, entry, kernel = fixture(
        "namespace.value",
        bindings={"namespace": CapturedNativeObject(namespace)},
        modules=(namespace,),
    )
    result = kernel.read_source(entry.source, module.module.body[0].value)
    assert events == []
    if existing:
        result.require_closed()
        assert result.value is property
    else:
        assert result.violation is CapturedReferenceViolation.UNPROVED_ACCESS
        with pytest.raises(ValueError, match="capture remains open"):
            result.require_closed()
    assert namespace.value is property
    assert events == ([] if existing else ["value"])


def test_read_and_install_distinction_agrees_with_isolated_native_execution():
    program = """from types import ModuleType
events = []
class Payload:
    def __set_name__(self, owner, name): events.append("install")
namespace = ModuleType("fixture_namespace")
namespace.payload = Payload()
saved = namespace.payload
print(events)
class TupleOwner:
    value = (saved,)
print(events)
class DirectOwner:
    value = saved
print(events)
"""
    assert subprocess.check_output(
        [sys.executable, "-I", "-c", program], text=True
    ).splitlines() == ["[]", "[]", "['install']"]
