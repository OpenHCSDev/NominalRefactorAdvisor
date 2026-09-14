"""Class metaclass syntax is shared without native identity or binding claims."""

import ast

import pytest

from nominal_refactor_advisor.ast_projection import AstClassProjection
from nominal_refactor_advisor.registry_identity import AutoRegisterClassAuthority


def declaration(header):
    return ast.parse(f"class Family{header}:\n    pass\n").body[0]


@pytest.mark.parametrize("header", ("", "(Base)", "(setting=True)", "(*bases)"))
def test_unexpanded_header_can_establish_explicit_keyword_absence(header):
    node = declaration(header)
    assert AstClassProjection.explicit_metaclass(node) is None
    with pytest.raises(ValueError, match="one actual metaclass operand"):
        AstClassProjection.require_explicit_metaclass(node)
    with pytest.raises(ValueError, match="one actual metaclass operand"):
        _ = AutoRegisterClassAuthority(node).metaclass_operand


@pytest.mark.parametrize(
    "operand", ("Renamed", "package.Whatever", "factory()", "int", "None")
)
def test_projection_returns_actual_original_operand_without_interpreting_it(operand):
    node = declaration(f"(metaclass={operand})")
    original = node.keywords[0].value
    assert AstClassProjection.explicit_metaclass(node) is original
    assert AstClassProjection.require_explicit_metaclass(node) is original
    assert AutoRegisterClassAuthority(node).metaclass_operand is original
    assert ast.unparse(original) == operand


@pytest.mark.parametrize(
    "header",
    (
        "(**options)",
        "(**{})",
        "(**{'metaclass': Meta})",
        "(metaclass=Meta, **options)",
        "(**options, metaclass=Meta)",
        "(metaclass=Meta, **{})",
        "(**first, **second)",
    ),
)
def test_expanded_keywords_never_count_as_proved_absence_or_exact_operand_binding(
    header,
):
    node = declaration(header)
    for projection in (
        AstClassProjection.explicit_metaclass,
        AstClassProjection.require_explicit_metaclass,
    ):
        with pytest.raises(ValueError, match="Expanded class keywords"):
            projection(node)
    with pytest.raises(ValueError, match="Expanded class keywords"):
        _ = AutoRegisterClassAuthority(node).metaclass_operand


@pytest.mark.parametrize("second", ("Meta", "Other"))
def test_duplicate_direct_metaclass_keywords_are_not_arbitrarily_selected(second):
    node = declaration(f"(metaclass=Meta, metaclass={second})")
    for projection in (
        AstClassProjection.explicit_metaclass,
        AstClassProjection.require_explicit_metaclass,
    ):
        with pytest.raises(ValueError, match="multiple explicit metaclass"):
            projection(node)
    with pytest.raises(ValueError, match="multiple explicit metaclass"):
        _ = AutoRegisterClassAuthority(node).metaclass_operand


@pytest.mark.parametrize(
    "header",
    (
        "(Base, metaclass=Meta, option=unknown())",
        "(option=unknown(), metaclass=Meta)",
        "(*bases, metaclass=Meta)",
    ),
)
def test_other_arguments_remain_separate_binding_obligations(header):
    node = declaration(header)
    original = next(
        keyword.value for keyword in node.keywords if keyword.arg == "metaclass"
    )
    assert AstClassProjection.explicit_metaclass(node) is original
    assert AutoRegisterClassAuthority(node).metaclass_operand is original


def test_same_spelling_in_another_class_is_a_different_original_operand():
    first = declaration("(metaclass=Meta)")
    second = declaration("(metaclass=Meta)")
    left = AstClassProjection.require_explicit_metaclass(first)
    right = AstClassProjection.require_explicit_metaclass(second)
    assert left is first.keywords[0].value
    assert right is second.keywords[0].value
    assert left is not right


def test_source_projection_does_not_retain_stale_ast_keyword_state():
    node = declaration("(metaclass=Meta)")
    original = AstClassProjection.require_explicit_metaclass(node)
    node.keywords.clear()
    assert AstClassProjection.explicit_metaclass(node) is None
    node.keywords.append(ast.keyword(arg="metaclass", value=original))
    assert AstClassProjection.require_explicit_metaclass(node) is original
