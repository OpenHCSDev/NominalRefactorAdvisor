"""Authored execution assumptions retain source identity and honest provenance."""

import ast
import builtins
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import typing

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.codemod import (
    DescendTypeKeyedBehaviorProjectionOperation,
    PromoteClassMembersToAncestorOperation,
)
from nominal_refactor_advisor.codemod_native_requirements import (
    DeclaredNativeUseInvariants,
    NativeUseProvenance,
    NativeUseReceipt,
    NativeUseRequirement,
)
from nominal_refactor_advisor.codemod_source_edits import CodemodSourceRevision
from nominal_refactor_advisor.codemod_semantics import CodemodPreflightStatus
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.source_geometry import SourceByteSpan

RATIONALE = "The selected native identity and behavior hold in supported executions."


def _module(source, *, path="/repo/native_invariant_probe.py"):
    return ParsedModule(
        Path(path), "native_invariant_probe", False, ast.parse(source), source
    )


def _requirement(
    module, *, expected=property, operation=DescendTypeKeyedBehaviorProjectionOperation
):
    environment = SourceModuleExecution.from_module(module)
    statement = module.module.body[-1]
    if isinstance(statement, ast.FunctionDef):
        statement = statement.body[-1]
    return NativeUseRequirement(
        operation, statement.value, (NativeDeclaration(expected),), environment
    )


def _declare(*requirements):
    return DeclaredNativeUseInvariants.from_requirements(
        requirements, rationale=RATIONALE
    )


@pytest.mark.parametrize(
    "source,expected",
    (
        ("import builtins\nvalue = builtins.property\n", property),
        ("from builtins import property as chosen\nvalue = chosen\n", property),
        ("import typing\nvalue = typing.cast\n", typing.cast),
    ),
)
def test_real_captured_identity_does_not_prove_behavior_invariance(source, expected):
    requirement = _requirement(_module(source), expected=expected)
    assert requirement.module is requirement.environment.source.module
    assert requirement.module is requirement.environment.module
    requirement.environment.capture(requirement.node).require_native(
        requirement.declarations
    )
    inspected = requirement.inspect()
    assert inspected.provenance is NativeUseProvenance.CAPTURED_IDENTITY
    assert not inspected.provenance.is_admitted
    with pytest.raises(ValueError):
        inspected.require_admitted()

    (declared,) = _declare(requirement).resolve((requirement,))
    assert declared.provenance is NativeUseProvenance.DECLARED
    assert declared.rationale == RATIONALE
    declared.require_admitted()
    assert requirement.inspect().provenance is NativeUseProvenance.CAPTURED_IDENTITY


def test_same_native_python_function_identity_can_have_different_behavior():
    program = """
import json
import typing
original = typing.cast
before = original(int, "unchanged")
def replacement(kind, value):
    return "changed"
original.__code__ = replacement.__code__
print(json.dumps({
    "same_object": typing.cast is original,
    "before": before,
    "after": typing.cast(int, "unchanged"),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == {
        "same_object": True,
        "before": "unchanged",
        "after": "changed",
    }
    # Only the trusted child program mutated a function, never this analyzer.
    assert typing.cast(int, "unchanged") == "unchanged"


@pytest.mark.parametrize(
    "source",
    (
        "import builtins\ndef use():\n    return builtins.property\n",
        "def no_change(): pass\nno_change()\nimport builtins\nvalue = builtins.property\n",
    ),
)
def test_declared_invariant_does_not_manufacture_activation_or_effect_proof(source):
    requirement = _requirement(_module(source))
    before = requirement.environment.capture(requirement.node)
    assert isinstance(before, OpenCapturedReference)
    assert requirement.inspect().provenance is NativeUseProvenance.UNRESOLVED
    (empty,) = DeclaredNativeUseInvariants().resolve((requirement,))
    with pytest.raises(ValueError):
        empty.require_admitted()

    (declared,) = _declare(requirement).resolve((requirement,))
    assert declared.provenance is NativeUseProvenance.DECLARED
    declared.require_admitted()
    after = requirement.environment.capture(requirement.node)
    assert isinstance(after, OpenCapturedReference)
    assert after.violation is before.violation
    assert requirement.inspect().provenance is NativeUseProvenance.UNRESOLVED

    # Native positive control establishes the fixture's possible behavior, not
    # a universal claim that the static provider has proved its invocation.
    namespace = {}
    exec(compile(source, "<trusted-native-use>", "exec"), namespace)
    value = namespace["use"]() if "use" in namespace else namespace["value"]
    assert value is builtins.property


@pytest.mark.parametrize(
    "source",
    (
        "import builtins\nvalue = builtins.staticmethod\n",
        "from builtins import staticmethod as chosen\nvalue = chosen\n",
    ),
)
def test_known_contradictory_native_identity_cannot_be_declared_away(source):
    requirement = _requirement(_module(source), expected=property)
    captured = requirement.environment.capture(requirement.node)
    assert not isinstance(captured, OpenCapturedReference)
    captured.require_native((NativeDeclaration(staticmethod),))
    with pytest.raises(ValueError):
        requirement.inspect()
    with pytest.raises(ValueError):
        _declare(requirement).resolve((requirement,))


def test_subset_acceptance_does_not_leak_to_other_uses_or_environments():
    source = "import builtins\nfirst = builtins.property\nsecond = builtins.property\n"
    module = _module(source)
    environment = SourceModuleExecution.from_module(module)
    requirements = tuple(
        NativeUseRequirement(
            DescendTypeKeyedBehaviorProjectionOperation,
            statement.value,
            (NativeDeclaration(property),),
            environment,
        )
        for statement in module.module.body[1:]
    )
    first, second = requirements
    acceptance = _declare(first)
    resolutions = acceptance.resolve(requirements)
    assert tuple(item.provenance for item in resolutions) == (
        NativeUseProvenance.DECLARED,
        NativeUseProvenance.CAPTURED_IDENTITY,
    )
    with pytest.raises(ValueError):
        resolutions[1].require_admitted()
    assert all(
        item.provenance is NativeUseProvenance.CAPTURED_IDENTITY
        for item in DeclaredNativeUseInvariants().resolve(requirements)
    )
    fresh = _requirement(_module(source))
    assert fresh.environment is not environment
    assert fresh.inspect().provenance is NativeUseProvenance.CAPTURED_IDENTITY
    with pytest.raises(ValueError, match="stale|foreign"):
        acceptance.resolve((second,))


@pytest.mark.parametrize("reuse_ast", (False, True), ids=("reparsed", "same-ast"))
def test_changed_source_at_identical_span_rejects_old_acceptance(reuse_ast):
    source = "import builtins\ndef use():\n    return builtins.property\n"
    module = _module(source)
    original = _requirement(module)
    acceptance = _declare(original)
    changed_source = source.replace("property", "reversed")
    changed_module = (
        replace(module, source=changed_source) if reuse_ast else _module(changed_source)
    )
    changed = _requirement(changed_module)
    assert changed.receipt.span == original.receipt.span
    assert (changed.node is original.node) is reuse_ast
    assert changed.module is changed_module
    assert changed.receipt.revision != original.receipt.revision
    with pytest.raises(ValueError, match="stale|foreign"):
        acceptance.resolve((changed,))


@pytest.mark.parametrize("change", ("operation", "expectation", "file"))
def test_acceptance_belongs_to_exact_operation_expected_native_and_source(change):
    module = _module("import builtins\ndef use():\n    return builtins.property\n")
    original = _requirement(module)
    acceptance = _declare(original)
    if change == "operation":
        changed = replace(original, operation=PromoteClassMembersToAncestorOperation)
    elif change == "expectation":
        changed = replace(original, declarations=(NativeDeclaration(staticmethod),))
    else:
        changed = _requirement(replace(module, path=Path("/repo/other.py")))
    with pytest.raises(ValueError, match="stale|foreign"):
        acceptance.resolve((changed,))


def test_a_copied_ast_read_is_not_canonical_source_evidence():
    source = "import builtins\nvalue = builtins.property\n"
    original = _requirement(_module(source))
    foreign = ast.parse(source).body[-1].value
    assert SourceByteSpan.require_node(foreign) == original.receipt.span
    with pytest.raises(ValueError, match="canonical"):
        replace(original, node=foreign)


def test_forged_revision_from_reused_ast_cannot_relabel_actual_environment_source():
    module = _module("import builtins\ndef use():\n    return builtins.property\n")
    requirement = _requirement(module)
    reused_ast_module = replace(
        module, source=module.source.replace("property", "reversed")
    )
    assert reused_ast_module.module is module.module
    forged_receipt = replace(
        requirement.receipt,
        revision=CodemodSourceRevision(
            reused_ast_module.file_path,
            CodemodSourceRevision.hash_source(reused_ast_module.source),
        ),
    )
    declared = DeclaredNativeUseInvariants((forged_receipt,), RATIONALE)
    assert requirement.module is module
    assert requirement.module is requirement.environment.source.module
    with pytest.raises(ValueError, match="stale|foreign"):
        declared.resolve((requirement,))


@pytest.mark.parametrize("deferred", (False, True), ids=("captured", "unresolved"))
def test_preflight_report_projects_declared_status_without_promoting_source_evidence(
    deferred,
):
    source = (
        "import builtins\ndef use():\n    return builtins.property\n"
        if deferred
        else "import builtins\nvalue = builtins.property\n"
    )
    requirement = _requirement(_module(source))
    original = requirement.inspect()
    original_report = original.preflight_report()
    assert original_report.detail is original
    assert original_report.status is CodemodPreflightStatus.FAILED
    assert original_report.operation == requirement.receipt.operation
    expected_provenance = (
        NativeUseProvenance.UNRESOLVED
        if deferred
        else NativeUseProvenance.CAPTURED_IDENTITY
    )
    assert original.provenance is expected_provenance
    (declared,) = _declare(requirement).resolve((requirement,))
    declared_report = declared.preflight_report()
    assert declared_report.detail is declared
    assert declared_report.status is CodemodPreflightStatus.PASSED
    wire = json.loads(json.dumps(json_report_object(declared_report)))
    assert wire["status"] == "passed"
    assert wire["details"]["provenance"] == "declared"
    assert wire["details"]["rationale"] == RATIONALE
    assert requirement.inspect().provenance is expected_provenance


@pytest.mark.parametrize("newline", ("\n", "\r\n"), ids=("lf", "crlf"))
def test_real_requirement_acceptance_json_roundtrip_preserves_native_owner_records(
    newline,
):
    source = "import builtins\nμ = 'β'; value = builtins.property\n".replace(
        "\n", newline
    )
    requirement = _requirement(_module(source))
    acceptance = _declare(requirement)
    wire = json.loads(json.dumps(json_report_object(acceptance)))
    restored = DeclaredNativeUseInvariants.from_json_value(wire)
    assert restored == acceptance
    assert type(restored.requirements[0]) is NativeUseReceipt
    assert type(restored.requirements[0].span) is SourceByteSpan
    assert type(restored.requirements[0].revision) is CodemodSourceRevision
    assert (
        restored.requirements[0].span.segment(tuple(source.splitlines(keepends=True)))
        == "builtins.property"
    )
    (resolution,) = restored.resolve((requirement,))
    assert resolution.provenance is NativeUseProvenance.DECLARED
    report = json.loads(json.dumps(json_report_object(resolution)))
    assert report["provenance"] == "declared"
    assert report["receipt"] == wire["requirements"][0]
    assert report["rationale"] == RATIONALE


@pytest.mark.parametrize(
    "field,value",
    (
        ("operation", "invented_operation"),
        ("native_declarations", ["untrusted.module.property"]),
    ),
)
def test_authored_labels_do_not_authenticate_native_declarations(field, value):
    requirement = _requirement(_module("import builtins\nvalue = builtins.property\n"))
    wire = json.loads(json.dumps(json_report_object(_declare(requirement))))
    wire["requirements"][0][field] = value
    forged = DeclaredNativeUseInvariants.from_json_value(wire)
    with pytest.raises(ValueError, match="stale|foreign"):
        forged.resolve((requirement,))


def test_duplicate_requirements_and_unscoped_rationale_are_rejected():
    requirement = _requirement(_module("import builtins\nvalue = builtins.property\n"))
    with pytest.raises(ValueError, match="duplicate"):
        _declare(requirement, requirement)
    with pytest.raises(ValueError, match="ambiguous"):
        _declare(requirement).resolve((requirement, requirement))
    with pytest.raises(ValueError, match="rationale"):
        DeclaredNativeUseInvariants((requirement.receipt,), " ")
    with pytest.raises(ValueError, match="rationale"):
        DeclaredNativeUseInvariants((), RATIONALE)


def test_receipt_cannot_reference_absent_source_or_empty_native_expectations():
    requirement = _requirement(_module("import builtins\nvalue = builtins.property\n"))
    with pytest.raises(ValueError, match="existing source"):
        replace(
            requirement.receipt, revision=CodemodSourceRevision("/repo/use.py", None)
        )
    with pytest.raises(ValueError, match="distinct"):
        replace(requirement.receipt, native_declarations=())
