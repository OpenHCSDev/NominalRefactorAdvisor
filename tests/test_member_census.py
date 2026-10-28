"""Census of consumers that name a family's members, and who uses its owner."""

import json
from pathlib import Path

from nominal_refactor_advisor.cli import main
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.member_census import (
    CensusFamilyKind,
    build_member_awareness_census,
)

NAME_OR_ATTRIBUTE = ("ast.Attribute", "ast.Name")


def _write(tmp_path: Path, sources: dict[str, str]) -> None:
    for name, source in sources.items():
        (tmp_path / f"{name}.py").write_text(source)


def test_name_or_attribute_consumers_split_by_projection_use(tmp_path: Path):
    _write(
        tmp_path,
        {
            "owner_user": (
                "import ast\n"
                "from nominal_refactor_advisor.ast_projection import ExpressionReference\n"
                "def uses_owner(node):\n"
                "    if isinstance(node, (ast.Name, ast.Attribute)):\n"
                "        return ExpressionReference.decoded(node).attribute_chain\n"
            ),
            "bypass_a": (
                "import ast\n"
                "def hand_rolled(node):\n"
                "    if isinstance(node, ast.Name):\n"
                "        return node.id\n"
                "    if isinstance(node, ast.Attribute):\n"
                "        return node.attr\n"
            ),
            "bypass_b": (
                "import ast\n"
                "class Reader:\n"
                "    def read(self, node):\n"
                "        return isinstance(node, ast.Name) or isinstance(node, ast.Attribute)\n"
            ),
        },
    )
    report = build_member_awareness_census(tmp_path)
    [group] = [
        group
        for group in report.groups
        if group.kind is CensusFamilyKind.FOREIGN_MODULE and group.family == "ast"
    ]
    assert group.members == NAME_OR_ATTRIBUTE
    assert group.consumer_count == 3
    assert group.owner_using_count == 1
    assert group.bypassing_count == 2
    assert [consumer.qualname for consumer in group.owner_using] == ["uses_owner"]
    assert sorted(consumer.qualname for consumer in group.bypassing) == [
        "Reader.read",
        "hand_rolled",
    ]
    assert group.covering_owners == (
        "nominal_refactor_advisor.ast_projection.ExpressionReference",
    )


def test_declared_family_within_counts_as_owner_use(tmp_path: Path):
    _write(
        tmp_path,
        {
            "routed": (
                "import ast\n"
                "from nominal_refactor_advisor.foreign_family import FunctionDefinition\n"
                "def routed(nodes):\n"
                "    kinds = isinstance(nodes, (ast.FunctionDef, ast.AsyncFunctionDef))\n"
                "    return kinds, list(FunctionDefinition.within(nodes))\n"
            ),
            "restated": (
                "import ast\n"
                "def restated(nodes):\n"
                "    return [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]\n"
            ),
        },
    )
    report = build_member_awareness_census(tmp_path)
    [group] = [group for group in report.groups if group.family == "ast"]
    assert group.members == ("ast.AsyncFunctionDef", "ast.FunctionDef")
    assert [consumer.qualname for consumer in group.owner_using] == ["routed"]
    assert [consumer.qualname for consumer in group.bypassing] == ["restated"]
    [owner] = [
        item
        for item in report.owners
        if item.owner == "nominal_refactor_advisor.foreign_family.FunctionDefinition"
    ]
    assert (owner.consumers, owner.owner_using, owner.bypassing) == (2, 1, 1)


def test_match_class_patterns_and_single_member_consumers(tmp_path: Path):
    _write(
        tmp_path,
        {
            "matcher": (
                "import ast\n"
                "def matcher(node):\n"
                "    match node:\n"
                "        case ast.Name():\n"
                "            return 1\n"
                "        case ast.Call():\n"
                "            return 2\n"
                "def single(node):\n"
                "    return isinstance(node, ast.Name)\n"
            ),
        },
    )
    report = build_member_awareness_census(tmp_path)
    [group] = report.groups
    assert group.members == ("ast.Call", "ast.Name")
    assert group.consumer_count == 1
    assert report.single_member_consumers == 1


def test_json_output_round_trips(tmp_path: Path, capsys, monkeypatch):
    _write(
        tmp_path,
        {
            "m": (
                "import ast\n"
                "def f(node):\n"
                "    return isinstance(node, (ast.Name, ast.Attribute))\n"
            ),
        },
    )
    report = build_member_awareness_census(tmp_path)
    expected = json.loads(json.dumps(json_report_object(report)))
    monkeypatch.setattr(
        "sys.argv", ["nra", str(tmp_path), "--count-member-awareness", "--json"]
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert expected["groups"][0]["bypassing_count"] == 1


def test_owner_own_methods_are_not_consumers(tmp_path: Path):
    package = tmp_path / "nominal_refactor_advisor"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "ast_projection.py").write_text(
        "import ast\n"
        "class ExpressionReference:\n"
        "    @staticmethod\n"
        "    def identifier(node):\n"
        "        return isinstance(node, (ast.Name, ast.Attribute))\n"
    )
    (tmp_path / "outside.py").write_text(
        "import ast\n"
        "def outside(node):\n"
        "    return isinstance(node, (ast.Name, ast.Attribute))\n"
    )
    report = build_member_awareness_census(tmp_path)
    [group] = [group for group in report.groups if group.family == "ast"]
    assert group.consumer_count == 1
    assert group.bypassing_count == 1
    assert [consumer.qualname for consumer in group.bypassing] == ["outside"]


def _name_attribute_consumers(tmp_path: Path, source: str) -> list[str]:
    _write(tmp_path, {"m": "import ast\n" + source})
    report = build_member_awareness_census(tmp_path)
    return sorted(
        consumer.qualname
        for group in report.groups
        if group.members == NAME_OR_ATTRIBUTE
        for consumer in (*group.owner_using, *group.bypassing)
    )


def test_sequential_ifs_on_one_subject_are_counted(tmp_path: Path):
    assert _name_attribute_consumers(
        tmp_path,
        "def call_symbol(node):\n"
        "    if isinstance(node, ast.Name):\n"
        "        return node.id\n"
        "    if isinstance(node, ast.Attribute):\n"
        "        return node.attr\n",
    ) == ["call_symbol"]


def test_boolean_operands_on_one_subject_are_counted(tmp_path: Path):
    assert _name_attribute_consumers(
        tmp_path,
        "def is_final(decorator):\n"
        "    return (isinstance(decorator, ast.Name) and decorator.id == 'final') or (\n"
        "        isinstance(decorator, ast.Attribute) and decorator.attr == 'final'\n"
        "    )\n",
    ) == ["is_final"]


def test_tests_on_different_subjects_are_not_combined(tmp_path: Path):
    assert (
        _name_attribute_consumers(
            tmp_path,
            "def self_attr_name(target):\n"
            "    return isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)\n",
        )
        == []
    )


def test_nested_def_branching_belongs_to_the_nested_def_only(tmp_path: Path):
    assert _name_attribute_consumers(
        tmp_path,
        "def outer(node):\n"
        "    def inner(child):\n"
        "        return isinstance(child, ast.Name) or isinstance(child, ast.Attribute)\n"
        "    return inner(node)\n",
    ) == ["outer.inner"]


def test_own_class_family_consumers_cost_one_edit_each_to_add_a_member(tmp_path: Path):
    """A function that tests the project's own classes one by one enumerates their
    family: declaring one more member forces an edit there. Polymorphic dispatch
    through the family's own method costs nothing."""
    _write(
        tmp_path,
        {
            "shapes": (
                "class Shape:\n"
                "    def area(self):\n"
                "        raise NotImplementedError\n"
                "class Square(Shape):\n"
                "    def area(self):\n"
                "        return 1\n"
                "class Circle(Shape):\n"
                "    def area(self):\n"
                "        return 3\n"
            ),
            "consumers": (
                "from shapes import Circle, Shape, Square\n"
                "def hand_rolled(shape):\n"
                "    if isinstance(shape, Square):\n"
                "        return 1\n"
                "    if isinstance(shape, Circle):\n"
                "        return 3\n"
                "def derived(shape: Shape):\n"
                "    return shape.area()\n"
            ),
        },
    )
    report = build_member_awareness_census(tmp_path)
    [cost] = [cost for cost in report.add_member_costs if cost.kind is CensusFamilyKind.CLASS]
    assert cost.family == "shapes.Shape"
    assert [consumer.qualname for consumer in cost.consumers] == ["hand_rolled"]
    assert report.add_member_edits == sum(item.edits for item in report.add_member_costs)


def test_inherited_pairs_count_only_where_a_typed_call_reaches_them(tmp_path: Path):
    """Two subclasses inherit ``describe`` from Base: a call through an annotated
    Base parameter reaches both pairs. ``unused`` is inherited too, but no typed
    call reaches it, and an untyped receiver resolves nothing."""
    _write(
        tmp_path,
        {
            "family": (
                "class Base:\n"
                "    def describe(self):\n"
                "        return self.name()\n"
                "    def unused(self):\n"
                "        return 0\n"
                "    def name(self):\n"
                "        raise NotImplementedError\n"
                "class Left(Base):\n"
                "    def name(self):\n"
                "        return 'left'\n"
                "class Right(Base):\n"
                "    def name(self):\n"
                "        return 'right'\n"
            ),
            "consumers": (
                "from family import Base\n"
                "def typed(item: Base):\n"
                "    return item.describe()\n"
                "def untyped(item):\n"
                "    return item.unused()\n"
            ),
        },
    )
    inherited = build_member_awareness_census(tmp_path).inherited
    [owner] = inherited.owners
    assert (owner.owner, owner.derived_pairs, owner.members) == ("family.Base", 2, 2)
    assert owner.consumers == 1
    assert (inherited.attribute_calls, inherited.receiver_typed_calls) == (3, 2)
    assert inherited.calls_reaching_inherited == 1
