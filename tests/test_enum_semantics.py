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
