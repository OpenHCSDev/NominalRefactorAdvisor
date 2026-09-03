import pytest

from nominal_refactor_advisor.enum_semantics import PYTHON_ENUM_BASE_AUTHORITY


@pytest.mark.parametrize(
    "base_name",
    (
        "Enum",
        "enum.Flag",
        "IntEnum",
        "enum.IntFlag",
        "StrEnum",
    ),
)
def test_python_enum_base_authority_owns_all_standard_enum_families(
    base_name: str,
) -> None:
    assert PYTHON_ENUM_BASE_AUTHORITY.matches(base_name)


def test_python_enum_base_authority_rejects_non_enum_bases() -> None:
    assert not PYTHON_ENUM_BASE_AUTHORITY.matches("str")
    assert not PYTHON_ENUM_BASE_AUTHORITY.matches(None)
    assert not PYTHON_ENUM_BASE_AUTHORITY.matches_any(("ABC", "Generic"))


@pytest.mark.parametrize(
    "qualified_name",
    (
        "enum.Enum",
        "enum.Flag",
        "enum.IntEnum",
        "enum.IntFlag",
        "enum.StrEnum",
    ),
)
def test_python_enum_base_authority_requires_standard_module_for_qualified_proof(
    qualified_name: str,
) -> None:
    assert PYTHON_ENUM_BASE_AUTHORITY.matches_qualified(qualified_name)
    assert not PYTHON_ENUM_BASE_AUTHORITY.matches_qualified(
        f"third_party.{qualified_name.rsplit('.', maxsplit=1)[-1]}"
    )


def test_python_enum_base_authority_protects_inherited_members() -> None:
    assert not PYTHON_ENUM_BASE_AUTHORITY.permits_new_member("name")
    assert not PYTHON_ENUM_BASE_AUTHORITY.permits_new_member("value")
    assert not PYTHON_ENUM_BASE_AUTHORITY.permits_new_member("__str__")
    assert PYTHON_ENUM_BASE_AUTHORITY.permits_new_member("handler_type")
