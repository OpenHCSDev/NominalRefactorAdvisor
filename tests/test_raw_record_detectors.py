"""R1 behavior: record-read ownership, absent schemas and declared type guards."""

import ast
import inspect
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.detectors import (
    DetectorConfig,
    IssueDetector,
    RedundantTypeCheckDetector,
    SemanticMirrorWithoutDescentDetector,
    UnmodeledRecordShapeDetector,
)
from nominal_refactor_advisor.semantic_descent import (
    PresentationProjectionKind,
    build_semantic_descent_graph,
)


def modules(tmp_path: Path, source: str):
    path = tmp_path / "example.py"
    path.write_text(source)
    return parse_python_modules(path, use_parse_cache=False, parse_workers=1)


def test_mapping_reads_match_schema_but_only_actual_decode_descends(tmp_path):
    parsed = modules(
        tmp_path,
        """
from dataclasses import dataclass
@dataclass
class Record:
    alpha: str
    beta: int
    gamma: str
    @classmethod
    def decode(cls, row):
        return cls(row['alpha'], row['beta'], row.get('gamma'))

def raw(row):
    return row['alpha'], row.get('beta'), row['gamma']

def decode(row):
    return Record(alpha=row['alpha'], beta=int(row['beta']), gamma=row.get('gamma'))

def positional_decode(row):
    return Record(row['alpha'], int(row['beta']), row.get('gamma'))

def swapped_positional(row):
    return Record(row['gamma'], row['beta'], row['alpha'])

def unrelated_constructor(row):
    ignored = Record(alpha='x', beta=0, gamma='z')
    return row['alpha'], row['beta'], row['gamma']

def shadowed_constructor(row, Record):
    return Record(alpha=row['alpha'], beta=row['beta'], gamma=row['gamma'])

def swapped_decode(row):
    return Record(alpha=row['gamma'], beta=row['beta'], gamma=row['alpha'])

def distinct_subjects(left, right):
    return left['alpha'], right['beta']

def dynamic(row, key):
    return row[key], row['alpha']

def store_only(row):
    row['alpha'] = 'a'
    row['beta'] = 1
    row['gamma'] = 'g'

def owner():
    def inner(row):
        return row['alpha'], row['beta'], row['gamma']
    return inner
""",
    )
    graph = build_semantic_descent_graph(parsed)
    reads = [
        p
        for p in graph.projections
        if p.kind is PresentationProjectionKind.MAPPING_READ
    ]
    assert {p.owner_symbol for p in reads} == {
        "raw",
        "decode",
        "unrelated_constructor",
        "swapped_decode",
        "owner.inner",
        "shadowed_constructor",
        "positional_decode",
        "swapped_positional",
        "Record.decode",
    }
    findings = SemanticMirrorWithoutDescentDetector().detect(parsed, DetectorConfig())
    read_findings = [f for f in findings if "mapping read" in f.why]
    assert {f.evidence[0].symbol for f in read_findings} == {
        "raw:row",
        "unrelated_constructor:row",
        "swapped_decode:row",
        "inner:row",
        "shadowed_constructor:row",
        "swapped_positional:row",
    }
    assert not UnmodeledRecordShapeDetector().detect(parsed, DetectorConfig())


def test_unmodeled_shapes_aggregate_same_keys_and_ignore_small_dynamic_reads(tmp_path):
    parsed = modules(
        tmp_path,
        """
def first(row):
    return row['alpha'], row.get('beta'), row['gamma']
def second(payload):
    return payload.get('gamma'), payload['alpha'], payload.get('beta', 0)
def small(row):
    return row['a'], row['b']
def dynamic(row, key):
    return row[key], row['a']
def separate(left, right):
    return left['alpha'], right['beta'], right['gamma']
""",
    )
    findings = UnmodeledRecordShapeDetector().detect(parsed, DetectorConfig())
    assert len(findings) == 1
    assert findings[0].metrics.field_names == ("alpha", "beta", "gamma")
    assert len(findings[0].evidence) == 2
    assert not SemanticMirrorWithoutDescentDetector().detect(parsed, DetectorConfig())


def test_redundant_checks_resolve_declared_self_parameters_and_keep_unknowns(tmp_path):
    parsed = modules(
        tmp_path,
        """
from dataclasses import dataclass
from typing import Any
@dataclass
class Record:
    name: str
    count: int
    unknown: Any
    optional: str | None
    def check(self):
        return type(self.name) is str and isinstance(self.count, int)
class Child(Record):
    def check(self):
        return type(self.name) is not str

def typed(row: Record):
    return isinstance(row.name, str)
def forward(row: "Record"):
    return type(row.count) == int

def unknown(row):
    return isinstance(row.name, str)
def union(row: Record | None):
    return isinstance(row.name, str)
def optional(row: Record):
    return isinstance(row.optional, str) or type(row.unknown) is int
def different(row: Record):
    return isinstance(row.count, bool)
def rebound(row: Record):
    row = object()
    return isinstance(row.name, str)
def shadow(row: Record, isinstance):
    return isinstance(row.name, str)
def nested(row: Record):
    def inner(row):
        return isinstance(row.name, str)
    return inner
""",
    )
    findings = RedundantTypeCheckDetector().detect(parsed, DetectorConfig())
    assert len(findings) == 5
    assert {f.evidence[0].symbol for f in findings} == {
        "Record.check",
        "Child.check",
        "typed",
        "forward",
    }
    assert all(len(f.evidence) == 2 for f in findings)
    assert all("before removal" in f.summary for f in findings)


def test_detector_ids_are_derived_and_registered():
    for detector in (UnmodeledRecordShapeDetector, RedundantTypeCheckDetector):
        assert (
            IssueDetector.registered_detector_type_for_id(
                detector.effective_detector_id()
            )
            is detector
        )
        declaration = ast.parse(inspect.getsource(detector)).body[0]
        assert not any(
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"detector_id", "__registry__"}
                for target in statement.targets
            )
            for statement in declaration.body
        )


def test_cross_module_schema_and_annotated_owner_use_complete_context(tmp_path):
    (tmp_path / "models.py").write_text("""
from dataclasses import dataclass
@dataclass
class Record:
    alpha: str
    beta: int
    gamma: str
""")
    (tmp_path / "consumer.py").write_text("""
from models import Record as ImportedRecord
def consume(row):
    return row['alpha'], row.get('beta'), row['gamma']
def typed(row: ImportedRecord):
    return isinstance(row.alpha, str)
def decode(row):
    return ImportedRecord(alpha=row['alpha'], beta=row['beta'], gamma=row['gamma'])
""")
    parsed = parse_python_modules(tmp_path, use_parse_cache=False, parse_workers=1)
    assert not UnmodeledRecordShapeDetector().detect(parsed, DetectorConfig())
    mirrors = SemanticMirrorWithoutDescentDetector().detect(parsed, DetectorConfig())
    assert len(mirrors) == 1 and mirrors[0].authority_evidence.symbol == "Record"
    checks = RedundantTypeCheckDetector().detect(parsed, DetectorConfig())
    assert len(checks) == 1 and checks[0].evidence[1].symbol == "Record.alpha"
