"""Original literal equality is distinct from source spelling and unknown keys.

These tests exercise original compact values and the unknown fact boundary,
not an Enum whitelist.
"""

import ast
import multiprocessing
import pickle
import subprocess
import sys
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest
from registry_test_sources import keyed_registry_source

from nominal_refactor_advisor.analysis import analyze_detector_types
from nominal_refactor_advisor.ast_tools import (
    CollectedFamilyImplementationIdentity,
    ParsedModule,
)
from nominal_refactor_advisor.class_index import (
    CompactModuleClassProjectionFamily,
    _compact_class_member_declarations,
    build_compact_class_family_index,
)
from nominal_refactor_advisor.detectors import DetectorConfig, IssueDetector, _systemic
from nominal_refactor_advisor.detectors._base import (
    _compact_keyed_family_axis_specs_from_index,
)
from nominal_refactor_advisor.semantic_shape_algebra import InjectiveTypeRegistryProof
from nominal_refactor_advisor.value_expression import CompactValueExpression


def parsed(source):
    return ParsedModule(
        Path("/repo/literal_registry.py"),
        "literal_registry",
        False,
        ast.parse(source),
        source,
    )


def findings(source):
    detectors = tuple(
        detector
        for detector in IssueDetector.registered_detector_types()
        if detector.effective_detector_id()
        in (
            "injective_type_registry",
            "non_injective_type_registry",
        )
    )
    return analyze_detector_types(
        [parsed(source)], DetectorConfig(), detector_types=detectors
    )


@pytest.mark.parametrize(
    "left,right,count",
    (
        ("'alpha'", "'beta'", 2),
        ("b'alpha'", "b'beta'", 2),
        ("1", "2", 2),
        ("(1, 'alpha')", "(2, 'beta')", 2),
        ("None", "...", 2),
        ("1", "True", 1),
        ("1", "1.0", 1),
        ("0", "-0.0", 1),
        ("(1, (2,))", "(True, (2.0,))", 1),
    ),
)
def test_actual_literal_keys_own_detector_injectivity(left, right, count):
    source = (
        keyed_registry_source().replace("Mode.ALPHA", left).replace("Mode.BETA", right)
    )
    native = subprocess.run(
        [
            sys.executable,
            "-c",
            source + f"\nassert len(ModeRunner._registry) == {count}\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr
    (finding,) = findings(source)
    if count == 2:
        assert "mature injective registry" in finding.summary
    else:
        assert "is not injective" in finding.summary


@pytest.mark.parametrize("right", ("Mode.BETA", "Mode.ALPHA", "unknown()"))
def test_unproved_keys_are_neither_injective_nor_noninjective(right):
    source = keyed_registry_source().replace("mode = Mode.BETA", f"mode = {right}")
    module = parsed(source)
    projections = _systemic.InjectiveTypeRegistryDetector.compact_module_projections(
        (module,)
    )
    (fact,) = _systemic._compact_keyed_registry_axis_facts(
        projections, DetectorConfig()
    )
    assert fact.injectivity_proof is None
    assert not fact.is_mature_injective
    with pytest.raises(ValueError, match="equality remains unproved"):
        fact.require_proof()
    assert findings(source) == []


@pytest.mark.parametrize("direct_write", ("pass", "mode: object"))
def test_inherited_key_without_direct_value_is_unknown_not_missing(direct_write):
    source = (
        keyed_registry_source()
        .replace("Mode.ALPHA", "'alpha'")
        .replace("Mode.BETA", "'beta'")
        .replace("mode: ClassVar[Mode]", "mode: ClassVar[Mode] = 'beta'")
        .replace("mode = 'beta'", direct_write)
    )
    native = subprocess.run(
        [sys.executable, "-c", source + "\nassert BetaModeRunner.mode == 'beta'\n"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr
    projections = _systemic.InjectiveTypeRegistryDetector.compact_module_projections(
        (parsed(source),)
    )
    (fact,) = _systemic._compact_keyed_registry_axis_facts(
        projections, DetectorConfig()
    )
    assert fact.injectivity_proof is None
    assert findings(source) == []


@pytest.mark.parametrize(
    "later_write,proven", (("mode: object", True), ("mode = unknown()", False))
)
def test_key_evidence_follows_latest_value_write_not_latest_statement(
    later_write, proven
):
    source = (
        keyed_registry_source()
        .replace("Mode.ALPHA", "'alpha'")
        .replace("Mode.BETA", "'beta'")
        .replace("mode = 'beta'", "mode = 'beta'\n    " + later_write)
    )
    projections = _systemic.InjectiveTypeRegistryDetector.compact_module_projections(
        (parsed(source),)
    )
    (fact,) = _systemic._compact_keyed_registry_axis_facts(
        projections, DetectorConfig()
    )
    if proven:
        assert fact.require_proof().is_injective
        assert len(findings(source)) == 1
    else:
        assert fact.injectivity_proof is None
        assert findings(source) == []


def test_original_value_is_projected_once_and_display_is_not_reparsed(monkeypatch):
    node = ast.parse(
        "class Choice:\n    first = second = 1\n    absent: object\n    empty = None\n"
    ).body[0]
    original = node.body[0].value
    visited = []
    project = CompactValueExpression.project

    def observe(original_node):
        visited.append(original_node)
        return project(original_node)

    monkeypatch.setattr(CompactValueExpression, "project", staticmethod(observe))
    first, second, absent, empty = _compact_class_member_declarations(node)
    assert sum(value is original for value in visited) == 1
    assert first.value is second.value
    assert first.value.require_mapping_key() == 1
    assert absent.value is None
    assert empty.value.require_mapping_key() is None
    assert empty.value_is_none_literal and not absent.value_is_none_literal
    assert "constant_string" not in first._fields
    assert "value_is_none_literal" not in first._fields
    changed_display = first._replace(expression="not even valid Python !")
    assert changed_display.value is first.value
    assert changed_display.value.require_mapping_key() == 1


def test_key_axis_labels_derive_from_original_values_not_display_text():
    source = keyed_registry_source().replace(
        'registry_key_attr = "mode"',
        'registry_key_attr = "mode"\n    family_label = "Runner family"',
    )
    (projection,) = CompactModuleClassProjectionFamily.collect_modules(
        (parsed(source),)
    )
    changed = replace(
        projection,
        classes=tuple(
            replace(
                indexed,
                direct_member_declarations=tuple(
                    (
                        member._replace(expression="not valid Python !")
                        if member.name in ("registry_key_attr", "family_label")
                        else member
                    )
                    for member in indexed.direct_member_declarations
                ),
            )
            for indexed in projection.classes
        ),
    )
    index = build_compact_class_family_index((changed,))
    (spec,) = _compact_keyed_family_axis_specs_from_index(index)
    assert spec.registry_key_attr_name == "mode"
    assert spec.family_label == "Runner family"


def _worker_key(blob):
    member = pickle.loads(blob)
    return member.value.require_mapping_key()


@pytest.mark.parametrize(
    "expression", ("b'key'", "1+2j", "None", "...", "(b'key', (1+2j, None, ...))")
)
def test_compact_literal_pickle_and_spawn_preserve_native_equality(expression):
    (member,) = _compact_class_member_declarations(
        ast.parse(f"class Choice:\n    key = {expression}\n").body[0]
    )
    expected = ast.literal_eval(expression)
    payload = pickle.dumps(member)
    restored = pickle.loads(payload)
    assert restored.value.require_mapping_key() == expected
    assert hash(restored.value.require_mapping_key()) == hash(expected)
    pending = [restored]
    while pending:
        item = pending.pop()
        assert not isinstance(item, ast.AST)
        if is_dataclass(item):
            pending.extend(getattr(item, field.name) for field in fields(item))
        elif isinstance(item, tuple):
            pending.extend(item)
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(_worker_key, (payload,)) == expected


def test_proof_groups_native_values_and_retains_both_diagnostic_spellings():
    proof = InjectiveTypeRegistryProof.from_key_entries(
        key_axis_name="Key",
        key_entries=((1, "1", ("One",)), (True, "True", ("Two",))),
        registered_type_names=("One", "Two"),
    )
    assert not proof.is_injective
    assert proof.duplicate_key_names == ("1", "True")
    assert proof.missing_type_names == ()


def test_string_domain_empty_rows_remain_in_existing_proof():
    proof = InjectiveTypeRegistryProof.from_type_map(
        key_axis_name="Key",
        type_names_by_key={"empty": (), "present": ("Present",)},
        registered_type_names=("Present", "Missing"),
    )
    assert proof.key_names == ("empty", "present")
    assert proof.duplicate_key_names == ()
    assert proof.duplicate_type_names == ()
    assert proof.missing_type_names == ("Missing",)


def test_native_groups_count_reverse_membership_once_per_actual_key():
    proof = InjectiveTypeRegistryProof.from_key_entries(
        key_axis_name="Key",
        key_entries=(
            (1, "1", ("Same", "Same")),
            (True, "True", ("Same",)),
            (2, "2", ("Same",)),
        ),
        registered_type_names=("Same",),
    )
    assert proof.duplicate_key_names == ()
    assert proof.duplicate_type_names == ("Same",)


def test_cache_implementation_identity_includes_original_producer_and_value_owner():
    (member,) = _compact_class_member_declarations(
        ast.parse("class Choice:\n    key = 1\n").body[0]
    )
    implementation = CollectedFamilyImplementationIdentity.from_family(
        CompactModuleClassProjectionFamily
    )
    modules = {source.module_name for source in implementation.sources}
    assert type(member).__module__ in modules
    assert type(member.value).__module__ in modules
    assert "nominal_refactor_advisor.value_expression" in modules
    assert all(source.source_signature for source in implementation.sources)
