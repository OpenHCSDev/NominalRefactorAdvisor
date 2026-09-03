from __future__ import annotations

import ast

from nominal_refactor_advisor.annotation_semantics import (
    NOMINAL_ANNOTATION_SOURCE_AUTHORITY,
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
    assert authority.source_or_none(ast.parse("factory().Result", mode="eval").body) is (
        None
    )
