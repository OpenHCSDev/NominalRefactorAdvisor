"""Source geometry and revisions own their wire fields without shadow records."""

import ast
from dataclasses import fields
import inspect
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from nominal_refactor_advisor.codemod_payload import PayloadRecordValueCodec
from nominal_refactor_advisor.codemod_source_edits import CodemodSourceRevision
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from nominal_refactor_advisor.source_identity import python_source_cache_signature


@pytest.mark.parametrize("newline", ("\n", "\r\n"), ids=("lf", "crlf"))
@pytest.mark.parametrize("multiline", (False, True), ids=("single", "multiple"))
def test_actual_span_and_revision_roundtrip_exact_utf8_source(newline, multiline):
    expression = "(\n    native,\n    'β',\n)" if multiline else "native"
    source = ("μ = 1; result = " + expression + "\n").replace("\n", newline)
    node = ast.parse(source).body[1].value
    span = SourceByteSpan.require_node(node)
    revision = CodemodSourceRevision.from_sources(
        "/repo/use.py", {"/repo/use.py": source}
    )
    encoded = json.loads(
        json.dumps(
            {
                "span": json_report_object(span),
                "revision": json_report_object(revision),
            }
        )
    )

    restored_span = PayloadRecordValueCodec(SourceByteSpan).read(encoded, "span")
    restored_revision = PayloadRecordValueCodec(CodemodSourceRevision).read(
        encoded, "revision"
    )
    assert type(restored_span) is SourceByteSpan
    assert type(restored_revision) is CodemodSourceRevision
    assert restored_span == span
    assert restored_revision == revision
    assert restored_span.segment(tuple(source.splitlines(keepends=True))) == (
        expression.replace("\n", newline)
    )
    assert restored_revision.matches_source(source)
    assert restored_revision.source_hash == python_source_cache_signature(source)
    assert set(encoded["span"]) == {member.name for member in fields(SourceByteSpan)}
    assert set(encoded["revision"]) == {"file_path", "source_hash"}


@pytest.mark.parametrize("newline", ("\n", "\r\n"), ids=("lf", "crlf"))
def test_same_location_changed_expression_does_not_reuse_source_revision(newline):
    source = "def invoke():\n    return native()\n".replace("\n", newline)
    changed = source.replace("native", "forged")
    original_read = ast.parse(source).body[0].body[0].value.func
    changed_read = ast.parse(changed).body[0].body[0].value.func
    assert SourceByteSpan.require_node(original_read) == SourceByteSpan.require_node(
        changed_read
    )
    revision = CodemodSourceRevision.from_sources(
        "/repo/use.py", {"/repo/use.py": source}
    )
    restored = CodemodSourceRevision.from_json_value(
        json.loads(json.dumps(json_report_object(revision)))
    )
    assert restored.matches_source(source)
    assert not restored.matches_source(changed)
    assert not restored.matches_source(None)


def test_source_revision_hash_alias_is_same_canonical_function():
    assert CodemodSourceRevision.hash_source is python_source_cache_signature
    assert CodemodSourceRevision("/repo/use.py", None).hash_source is (
        python_source_cache_signature
    )
    assert CodemodSourceRevision.hash_source("μ\r\n") != (
        CodemodSourceRevision.hash_source("μ\n")
    )


@pytest.mark.parametrize("source", (None, "", "value = 'μ'\r\n"))
def test_revision_json_shape_preserves_absent_empty_and_existing_sources(source):
    sources = {} if source is None else {"/repo/use.py": source}
    revision = CodemodSourceRevision.from_sources("/repo/use.py", sources)
    expected = {
        "file_path": "/repo/use.py",
        "source_hash": (
            None if source is None else python_source_cache_signature(source)
        ),
    }
    assert json_report_object(revision) == expected
    assert CodemodSourceRevision.from_json_value(expected) == revision
    assert revision.matches_source(source)
    assert revision.matches_source(None) is (source is None)


def test_native_owner_positional_constructor_equality_hash_and_pickle_unchanged():
    span = SourceByteSpan(0, 1, 2, 3)
    revision = CodemodSourceRevision("/repo/use.py", "hash")
    assert span == SourceByteSpan(0, 1, 2, 3)
    assert revision == CodemodSourceRevision("/repo/use.py", "hash")
    assert hash(span) == hash((0, 1, 2, 3))
    assert hash(revision) == hash(("/repo/use.py", "hash"))
    assert pickle.loads(pickle.dumps((span, revision))) == (span, revision)
    for owner in (SourceByteSpan, CodemodSourceRevision):
        parameters = tuple(inspect.signature(owner).parameters.values())
        assert tuple(parameter.name for parameter in parameters) == tuple(
            member.name for member in fields(owner)
        )
        assert all(
            parameter.default is inspect.Parameter.empty for parameter in parameters
        )
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters
        )


@pytest.mark.parametrize(
    "name", tuple(member.name for member in fields(SourceByteSpan))
)
@pytest.mark.parametrize("value", (True, -1, "1", 1.5, None))
def test_span_payload_rejects_non_native_coordinate_values(name, value):
    payload = dict(start_line_index=0, end_line_index=0, start_byte=0, end_byte=1)
    payload[name] = value
    with pytest.raises(ValueError):
        SourceByteSpan.from_json_value(payload)


@pytest.mark.parametrize(
    "name", tuple(member.name for member in fields(SourceByteSpan))
)
def test_span_payload_requires_every_owned_coordinate(name):
    payload = dict(start_line_index=0, end_line_index=0, start_byte=0, end_byte=1)
    del payload[name]
    with pytest.raises(ValueError):
        SourceByteSpan.from_json_value(payload)


@pytest.mark.parametrize(
    "owner,payload",
    (
        (
            SourceByteSpan,
            dict(start_line_index=0, end_line_index=0, start_byte=0, end_byte=1),
        ),
        (CodemodSourceRevision, dict(file_path="/repo/use.py", source_hash=None)),
    ),
)
def test_native_source_payload_rejects_unknown_fields(owner, payload):
    with pytest.raises(ValueError, match="Unsupported"):
        owner.from_json_value({**payload, "invented_authority": "wrong"})


@pytest.mark.parametrize(
    "payload",
    (
        {"file_path": False, "source_hash": None},
        {"file_path": "/repo/use.py", "source_hash": 3},
        {"source_hash": None},
    ),
)
def test_revision_payload_rejects_invalid_field_types(payload):
    with pytest.raises(ValueError):
        CodemodSourceRevision.from_json_value(payload)


def test_geometry_payload_dependency_does_not_import_codemod_operations():
    # Isolate the leaf's import graph from the package's eager public __init__.
    # This checks geometry dependencies, not total public-package startup cost.
    package_path = Path(inspect.getfile(SourceByteSpan)).parent
    program = """
import importlib
import json
import sys
from types import ModuleType
package = ModuleType("nominal_refactor_advisor")
package.__path__ = [sys.argv[1]]
sys.modules[package.__name__] = package
geometry = importlib.import_module("nominal_refactor_advisor.source_geometry")
forbidden = {
    "nominal_refactor_advisor.codemod",
    "nominal_refactor_advisor.codemod_operations",
    "nominal_refactor_advisor.codemod_runtime",
    "nominal_refactor_advisor.codemod_source_edits",
    "nominal_refactor_advisor.native_compilation",
}
assert not forbidden.intersection(sys.modules)
span = geometry.SourceByteSpan.from_json_value({
    "start_line_index": 0, "end_line_index": 0, "start_byte": 0, "end_byte": 1,
})
print(json.dumps(geometry.SourceByteSpan.project_json_object(span)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(package_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "start_line_index": 0,
        "end_line_index": 0,
        "start_byte": 0,
        "end_byte": 1,
    }
