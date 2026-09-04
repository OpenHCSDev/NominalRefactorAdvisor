from __future__ import annotations

import ast

import pytest

from nominal_refactor_advisor.annotation_semantics import (
    NOMINAL_ANNOTATION_SOURCE_AUTHORITY,
)
from nominal_refactor_advisor.declaration_dependencies import (
    ModuleLexicalDependencyProjection,
)


def test_nominal_annotation_authority_projects_exact_name_chains() -> None:
    authority = NOMINAL_ANNOTATION_SOURCE_AUTHORITY

    assert authority.reference_parts_from_source("pkg.models.Result") == (
        "pkg",
        "models",
        "Result",
    )
    assert authority.reference_parts_from_source("'pkg.models.Result'") == (
        "pkg",
        "models",
        "Result",
    )
    assert authority.reference_parts_from_source("factory().Result") is None
    assert authority.reference_parts_from_source("list[Result]") is None
    assert authority.reference_parts_from_source("object") is None
    assert authority.source_or_none(
        ast.parse("factory().Result", mode="eval").body
    ) is (None)


def test_stringized_annotation_surfaces_retain_class_scope() -> None:
    module = ast.parse(
        "class Result:\n"
        "    @classmethod\n"
        "    def create(cls) -> 'Result | None':\n"
        "        return None\n\n"
        "class Shadowed:\n"
        "    Result = int\n"
        "    value: 'list[Result]'\n\n"
        "SERIALIZED = 'Result'\n"
    )
    surfaces = ModuleLexicalDependencyProjection.from_module(
        module
    ).stringized_annotations
    result_class, shadowed_class = (
        statement for statement in module.body if isinstance(statement, ast.ClassDef)
    )

    assert tuple(surface.literal.value for surface in surfaces) == (
        "Result | None",
        "list[Result]",
    )
    assert tuple(surface.reference_count("Result") for surface in surfaces) == (1, 1)
    assert surfaces[0].resolves_module_name("Result", result_class) is True
    assert surfaces[1].resolves_module_name("Result", result_class) is False
    assert surfaces[1].owner_classes == (shadowed_class,)
    assert (
        surfaces[0].renamed_source(
            "'Result | None'",
            old_name="Result",
            new_name="Outcome",
        )
        == "'Outcome | None'"
    )


def test_stringized_annotation_surfaces_exclude_literal_values_and_metadata() -> None:
    module = ast.parse(
        "from typing import Annotated, Literal\n\n"
        "value: tuple['Result', Literal['Result'], Annotated['Other', 'Result']]\n"
    )

    surfaces = ModuleLexicalDependencyProjection.from_module(
        module
    ).stringized_annotations

    assert tuple(surface.literal.value for surface in surfaces) == ("Result", "Other")


def test_stringized_annotation_reference_count_descends_nested_forward_refs() -> None:
    surface = ModuleLexicalDependencyProjection.from_module(
        ast.parse("value: \"list[\\'Result\\']\"\n")
    ).stringized_annotations[0]

    assert surface.reference_count("Result") == 1
    assert (
        surface.renamed_source(
            "\"list[\\'Result\\']\"",
            old_name="Result",
            new_name="Outcome",
        )
        == "\"list[\\'Outcome\\']\""
    )


def test_stringized_annotation_rename_requires_source_parse_parity() -> None:
    surface = ModuleLexicalDependencyProjection.from_module(
        ast.parse("value: 'Result'\n")
    ).stringized_annotations[0]

    with pytest.raises(ValueError, match="does not reconstruct"):
        surface.renamed_source(
            "'Result | Result'",
            old_name="Result",
            new_name="Outcome",
        )
