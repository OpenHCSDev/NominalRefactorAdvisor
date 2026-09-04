from __future__ import annotations

import pytest

from nominal_refactor_advisor.semantic_identity import (
    SemanticIdentifierTokenProjection,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("POLICY_PROFILE_ID", ("policy", "profile", "id")),
        ("_runtime_handler", ("runtime", "handler")),
        ("buildHTTP2Profile", ("build", "http", "2", "profile")),
        ("# generated source", ("generated", "source")),
    ),
)
def test_semantic_identifier_token_projection_normalizes_source_names(
    source: str,
    expected: tuple[str, ...],
) -> None:
    assert SemanticIdentifierTokenProjection.project(source) == expected
