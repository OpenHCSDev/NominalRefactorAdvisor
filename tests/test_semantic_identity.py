from __future__ import annotations

import pytest

from nominal_refactor_advisor.semantic_identity import (
    InheritanceIdentityAttributeProjection,
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


def test_inheritance_identity_attribute_projection_selects_shared_leaf_axis() -> None:
    assert InheritanceIdentityAttributeProjection.common_names(
        (
            ("format", "handler_name", "timeout"),
            ("format", "handler_name", "retries"),
        )
    ) == ("format", "handler_name")
