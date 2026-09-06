"""Member promotion preserves header bindings at its destination owner."""

from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    PromoteClassMembersToAncestorOperation,
    SourceRewriteTarget,
)


def _plan(path: Path) -> CodemodPlanSequence:
    return CodemodPlanSequence.from_operations(
        (
            PromoteClassMembersToAncestorOperation(
                target=SourceRewriteTarget(file_path=str(path), qualname="Owner"),
                destination=SourceRewriteTarget(file_path=str(path), qualname="Base"),
                member_names=("value",),
            ),
        )
    )


@pytest.mark.parametrize(
    "source",
    (
        "class Base:\n    int = 3\n"
        "class Owner(Base):\n    def value(self, x: int): return x\n"
        "print(Owner.value.__annotations__['x'])\n",
        "class Base:\n    staticmethod = 3\n"
        "class Owner(Base):\n    @staticmethod\n    def value(): return 3\n"
        "print(Owner.value())\n",
        "from typing import get_type_hints\n"
        "class Base:\n    int = 3\n"
        "class Owner(Base):\n    value: int = 1\n"
        "print(get_type_hints(Owner)['value'])\n",
        "from typing import get_type_hints\n"
        "class Base:\n    int = 3\n"
        "class Owner(Base):\n    value: 'int' = 1\n"
        "print(get_type_hints(Owner)['value'])\n",
    ),
    ids=("annotation", "decorator", "field_annotation", "string_field_annotation"),
)
def test_promotion_rejects_destination_namespace_capture(
    tmp_path: Path, source: str
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8", newline="")
    subprocess.run([sys.executable, str(path)], check=True, capture_output=True)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(ValueError):
        _plan(path).simulate(snapshot)
    assert path.read_bytes() == source.encode("utf-8")


@pytest.mark.parametrize(
    "member,observation",
    (
        ("def value(self): return int('4')", "Owner().value()"),
        ("value: 'str' = 'text'", "get_type_hints(Owner)['value']"),
        ("value: \"Literal['int']\" = 'int'", "get_type_hints(Owner)['value']"),
        ("value: \"Annotated[str, 'int']\" = 'int'", "get_type_hints(Owner)['value']"),
    ),
    ids=("method_body", "unrelated_forward_reference", "literal", "metadata"),
)
def test_promotion_preserves_unrelated_names_and_annotation_values(
    tmp_path: Path, member: str, observation: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from typing import Annotated, Literal, get_type_hints\n"
        "class Base:\n    int = 3\n"
        f"class Owner(Base):\n    {member}\n"
        f"print({observation})\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)])
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = _plan(path).simulate(snapshot)
    assert simulation.is_clean
    assert path.read_bytes() == source.encode("utf-8")
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(path)]) == before


@pytest.mark.parametrize("annotation", ("Literal['int']", "Annotated[str, 'int']"))
def test_subscription_guard_follows_native_annotation_evaluation(
    tmp_path: Path, annotation: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from typing import Annotated, Literal, get_type_hints\n"
        "class Base:\n    int = 3\n"
        f"class Owner(Base):\n    value: {annotation} = 'int'\n"
        "print(get_type_hints(Owner))\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)])
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    if sys.version_info < (3, 14):
        with pytest.raises(ValueError, match="no unique subscription proof"):
            _plan(path).simulate(snapshot)
        assert path.read_bytes() == source.encode("utf-8")
    else:
        simulation = _plan(path).simulate(snapshot)
        assert simulation.is_clean
        simulation.apply()
        assert subprocess.check_output([sys.executable, str(path)]) == before
