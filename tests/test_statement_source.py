"""Exact member projection must preserve Python values and statement ownership."""

import ast

import pytest

from nominal_refactor_advisor.codemod_declaration_source import (
    ClassBodySourceAuthority,
    ClassMemberInsertion,
    ClassMemberSource,
)
from nominal_refactor_advisor.codemod_statement_source import StatementSource


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("indentation", ("    ", "        ", "\t"))
@pytest.mark.parametrize(
    "decorator", ("    @staticmethod\n", "    @(\n        staticmethod\n    )\n")
)
def test_decorated_member_projection_preserves_literal_bytes(
    newline: str, indentation: str, decorator: str
) -> None:
    source = (
        "class Original:\n" + decorator + "    def text():\n"
        "        return '''first\n"
        "  literal indentation\n"
        "        last''' # retained comment\n"
    ).replace("\n", newline)
    original = ast.parse(source).body[0].body[0]
    rendered = StatementSource(source=source, node=original).member_source(indentation)
    rewritten = "class Destination:" + newline + rendered
    projected = ast.parse(rewritten).body[0].body[0]
    assert ast.dump(projected) == ast.dump(original)
    assert "# retained comment" in rendered
    original_namespace, rewritten_namespace = {}, {}
    exec(source, original_namespace)
    exec(rewritten, rewritten_namespace)
    assert (
        original_namespace["Original"].text()
        == rewritten_namespace["Destination"].text()
    )


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_nested_inline_class_insertion_retains_docs_comments_and_values(
    newline: str,
) -> None:
    source = (
        "class Outer:\n"
        "    class Inner: 'Documentation'; first = 1; second = '''start\n"
        " literal\n"
        "end''' # last comment\n"
        "    sibling = 3\n"
    ).replace("\n", newline)
    node = ast.parse(source).body[0].body[0]
    authority = ClassBodySourceAuthority(node=node, source=source)
    assert authority.indentation == "        "
    replacement = authority.member_insertion_replacement(
        ("        added = 4" + newline,)
    )
    rewritten = authority.geometry.source_with_replacements_in_span(
        0, len(source), (replacement,)
    )
    before, after = {}, {}
    exec(source, before)
    exec(rewritten, after)
    assert after["Outer"].Inner.__doc__ == before["Outer"].Inner.__doc__
    assert after["Outer"].Inner.first == 1
    assert after["Outer"].Inner.second == before["Outer"].Inner.second
    assert after["Outer"].Inner.added == 4
    assert after["Outer"].sibling == 3
    assert rewritten.count("# last comment") == 1


@pytest.mark.parametrize("indentation", ("    ", "\t"))
def test_member_insertion_precedes_complete_decorator_and_attached_comment(
    indentation: str,
) -> None:
    source = (
        "class Owner:\n"
        f"{indentation}# belongs to method\n"
        f"{indentation}@(\n"
        f"{indentation}    staticmethod\n"
        f"{indentation})\n"
        f"{indentation}def value(): return 3\n"
    )
    authority = ClassBodySourceAuthority(node=ast.parse(source).body[0], source=source)
    replacement = authority.member_insertion_replacement((indentation + "added = 4\n",))
    rewritten = authority.geometry.source_with_replacements_in_span(
        0, len(source), (replacement,)
    )
    namespace = {}
    exec(rewritten, namespace)
    assert namespace["Owner"].value() == 3
    assert namespace["Owner"].added == 4
    assert rewritten.index("added = 4") < rewritten.index("# belongs to method")


def test_coalesced_class_members_keep_declared_evaluation_order() -> None:
    first = ClassMemberInsertion(
        target_id="owner", members=(ClassMemberSource("z", "    z = 3\n"),)
    )
    second = ClassMemberInsertion(
        target_id="owner", members=(ClassMemberSource("a", "    a = z + 1\n"),)
    )
    combined = ClassMemberInsertion._coalesced_same_target((first, second))
    namespace = {}
    exec(
        "class Owner:\n" + "".join(member.source for member in combined.members),
        namespace,
    )
    assert namespace["Owner"].a == 4
