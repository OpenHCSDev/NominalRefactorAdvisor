"""The run-time introspection term: recorded by the pytest plugin, read by the census."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest

from nominal_refactor_advisor.amortization import NoTrace, RecordedTrace
from nra_introspection_trace import recording_for
from nominal_refactor_advisor.member_census import build_member_awareness_census


@dataclass
class _Left:
    a: int
    b: int


@dataclass
class _Right:
    c: int


def _generic(values):
    # A generator expression has its own frame on every Python version: the
    # introspection still belongs to _generic.
    return list(dataclasses.fields(value) for value in values)


def _single():
    return dataclasses.asdict(_Left(1, 2))


def test_a_consumer_reading_two_classes_derives_each_of_their_fields(tmp_path: Path):
    original = dataclasses.fields
    out = tmp_path / "trace.json"
    recording = recording_for(out, (__name__,))
    try:
        _generic([_Left(1, 2), _Right(3)])
        _single()
    finally:
        recording.finish()
    assert dataclasses.fields is original
    introspected = RecordedTrace(out).amortization()
    assert [(item.consumer, item.classes, item.fields) for item in introspected.generic] == [
        (f"{__name__}._generic", 2, 3)
    ]
    assert (introspected.consumers, introspected.classes) == (2, 2)
    assert introspected.derived_field_uses == 3


def test_a_trace_needs_a_project_prefix(tmp_path: Path):
    with pytest.raises(pytest.UsageError):
        recording_for(tmp_path / "trace.json", ())


def test_the_census_adds_both_derived_terms(tmp_path: Path):
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "trace.json"
    recording = recording_for(out, (__name__,))
    try:
        _generic([_Left(1, 2), _Right(3)])
    finally:
        recording.finish()
    census = build_member_awareness_census(tmp_path, introspection=RecordedTrace(out))
    assert census.derived_total == census.inherited.derived_pairs + 3
    unmeasured = build_member_awareness_census(tmp_path, introspection=NoTrace())
    assert unmeasured.introspected.trace.startswith("not measured")
