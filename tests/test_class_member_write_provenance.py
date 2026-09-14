"""Value and annotation inventory views retain their independent source rows.

These are direct syntactic writes, not final runtime class-state proofs.
"""

import ast
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    CompactModuleClassProjectionFamily,
    build_compact_class_family_index,
)
from nominal_refactor_advisor.type_keyed_behavior import (
    TypeKeyedBehaviorProjectionComponentBuilder,
)


def _projections(source):
    module = ParsedModule(
        Path("/repo/member_provenance.py"),
        "member_provenance",
        False,
        ast.parse(source),
        source,
    )
    return CompactModuleClassProjectionFamily.collect_modules((module,))


def _subject(statements):
    source = "class Subject:\n" + "".join(
        f"    {statement}\n" for statement in statements
    )
    (projection,) = _projections(source)
    (subject,) = projection.classes
    return subject


@pytest.mark.parametrize(
    "statements,value_index,annotation_index",
    (
        (("x = 'a'", "x: First"), 0, 1),
        (("x: First", "x = 'b'"), 1, 0),
        (("x: First = 'a'", "x = 'b'"), 1, 0),
        (("x: First = 'a'", "x: Second"), 0, 1),
        (("x = 'a'", "x = unknown()"), 1, None),
        (("x = 'a'", "x = None"), 1, None),
        (("x: First",), None, 0),
        (("x: First = 'a'",), 0, 0),
    ),
)
def test_write_views_select_original_rows(statements, value_index, annotation_index):
    subject = _subject(statements)
    rows = subject.direct_member_declarations
    before = tuple(tuple(row) for row in rows)
    value = subject.direct_value_writes_by_name.get("x")
    annotation = subject.direct_annotation_writes_by_name.get("x")
    if value_index is None:
        assert value is None
        assert "x" not in subject.assignments_by_name
    else:
        assert value is rows[value_index]
        assert value.line == value_index + 2
        assert subject.assignments_by_name["x"] == value.expression
    if annotation_index is None:
        assert annotation is None
    else:
        assert annotation is rows[annotation_index]
        assert annotation.line == annotation_index + 2
    assert tuple(tuple(row) for row in rows) == before
    assert set(subject.direct_declared_member_names) == {"x"}
    assert subject.declared_member_lines_by_name == {"x": 2}


@pytest.mark.parametrize("replacement", ("unknown()", "other", "None"))
def test_later_non_string_value_never_revives_old_literal(replacement):
    subject = _subject(("x = 'old'", f"x = {replacement}", "x: object"))
    row = subject.direct_value_writes_by_name["x"]
    assert row is subject.direct_member_declarations[1]
    assert subject.constant_string_assignment("x") is None
    assert subject.direct_constant_string_assignments == ()
    if replacement == "None":
        assert row.value.require_mapping_key() is None
        assert row.value_is_none_literal
        assert "x" not in subject.direct_non_none_assignment_names
    else:
        with pytest.raises(ValueError, match="unproved"):
            row.value.require_mapping_key()


def test_annotation_only_is_not_a_non_none_value_assignment():
    subject = _subject(("x: object",))
    assert subject.direct_value_writes_by_name == {}
    assert subject.assignments_by_name == {}
    assert subject.direct_non_none_assignment_names == ()
    assert "x" in subject.direct_declared_member_names
    assert subject.direct_annotation_writes_by_name["x"] is (
        subject.direct_member_declarations[0]
    )


def test_constructor_and_its_line_survive_annotation_only_write():
    subject = _subject(("x = Factory(option=1)", "x: object"))
    row = subject.direct_value_writes_by_name["x"]
    (construction,) = subject.direct_value_constructions
    assert row is subject.direct_member_declarations[0]
    assert row.annotation_expression is None
    assert construction.constructor_name == row.constructor_name == "Factory"
    assert construction.keyword_names == row.constructor_keyword_names == ("option",)
    assert construction.line == row.line == 2
    annotation = subject.direct_annotation_writes_by_name["x"]
    assert annotation.constructor_name is None
    assert annotation.expression is None


def test_later_plain_value_removes_prior_constructor_projection():
    subject = _subject(("x: object = Factory(option=1)", "x = 'replacement'"))
    assert subject.direct_value_constructions == ()
    assert subject.constant_string_assignment("x") == "replacement"
    assert subject.direct_annotation_writes_by_name["x"] is (
        subject.direct_member_declarations[0]
    )
    assert subject.direct_value_writes_by_name["x"].annotation_expression is None


@pytest.mark.parametrize("materialize_before_pickle", (False, True))
def test_pickle_retains_inventory_identity_in_both_views(materialize_before_pickle):
    subject = _subject(("x: First = 'a'", "x: Second", "y: First = None"))
    if materialize_before_pickle:
        assert subject.direct_value_writes_by_name["x"] is (
            subject.direct_member_declarations[0]
        )
        assert subject.direct_annotation_writes_by_name["x"] is (
            subject.direct_member_declarations[1]
        )
    restored = pickle.loads(pickle.dumps(subject))
    rows = restored.direct_member_declarations
    assert restored.direct_value_writes_by_name["x"] is rows[0]
    assert restored.direct_annotation_writes_by_name["x"] is rows[1]
    assert restored.direct_value_writes_by_name["y"] is rows[2]
    assert restored.direct_annotation_writes_by_name["y"] is rows[2]


@pytest.mark.parametrize(
    "statements,expected_value,expected_annotation",
    (
        (("x = 'a'", "x: int"), "a", "int"),
        (("x: int", "x = 'b'"), "b", "int"),
        (("x: int = 'a'", "x = 'b'"), "b", "int"),
        (("x: int = 'a'", "x: str"), "a", "str"),
        (("x: int = 'a'", "x = None"), None, "int"),
    ),
)
def test_native_class_value_and_annotation_are_independent(
    statements, expected_value, expected_annotation
):
    # Eager annotations on 3.11; explicitly demand deferred annotations on 3.14.
    # This validates final independent dictionaries, not an evaluation-order claim.
    source = "import inspect\nclass Subject:\n" + "".join(
        f"    {statement}\n" for statement in statements
    )
    source += (
        f"assert vars(Subject)['x'] == {expected_value!r}\n"
        "annotations = inspect.get_annotations(Subject)\n"
        f"assert annotations['x'] is {expected_annotation}\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_type_keyed_target_root_uses_annotation_before_plain_reassignment():
    source = """
from typing import ClassVar
class Event: pass
class Other: pass
class Projection:
    event_type: ClassVar[type[Event]]
    event_type = Other
"""
    projections = _projections(source)
    index = build_compact_class_family_index(projections)
    builder = TypeKeyedBehaviorProjectionComponentBuilder.from_projections(
        projections, index
    )
    root = next(
        item
        for item in index.classes_by_symbol.values()
        if item.simple_name == "Projection"
    )
    target = builder._declared_target_root(root, key_attribute_name="event_type")
    assert target is not None
    assert target.simple_name == "Event"
    annotation = root.direct_annotation_writes_by_name["event_type"]
    value = root.direct_value_writes_by_name["event_type"]
    assert annotation is root.direct_member_declarations[0]
    assert value is root.direct_member_declarations[1]
    assert annotation.expression is None
    assert value.annotation_expression is None
