"""Member promotion preserves header bindings at its destination owner."""

from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.captured_reference import CapturedReferenceViolation
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    PromoteClassMembersToAncestorOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_authority_claims import (
    AuthorityClaimDeclarationPreflightDetail,
)
from nominal_refactor_advisor.codemod_preflight import CodemodOperationPreflightError
from nominal_refactor_advisor.codemod_reproof import SourceReproofDiagnostic


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


@pytest.mark.native_proof
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


@pytest.mark.native_proof
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
        # Unsupported origins are rejected by the shared native-reference gate.
        with pytest.raises(CodemodOperationPreflightError) as rejected:
            _plan(path).simulate(snapshot)
        detail = rejected.value.report.detail
        assert isinstance(detail, AuthorityClaimDeclarationPreflightDetail)
        diagnostic = detail.declaration_preflight.detail
        assert isinstance(diagnostic, SourceReproofDiagnostic)
        assert (
            diagnostic.causes[0].violation
            is CapturedReferenceViolation.UNPROVED_EFFECTS
        )
        assert path.read_bytes() == source.encode("utf-8")
    else:
        simulation = _plan(path).simulate(snapshot)
        assert simulation.is_clean
        simulation.apply()
        assert subprocess.check_output([sys.executable, str(path)]) == before


WRAPPED = "    {decorator}\n    def value(self):\n        yield 4\n"
OBSERVE = "with Owner().value() as seen:\n    print(seen)\n"


@pytest.mark.native_proof
@pytest.mark.parametrize(
    "imports,decorator",
    (
        ("from contextlib import contextmanager\n", "@contextmanager"),
        ("import contextlib\n", "@contextlib.contextmanager"),
    ),
    ids=("from_import", "attribute"),
)
def test_an_owner_independent_wrapper_promotes_by_its_binding(
    tmp_path: Path, imports: str, decorator: str
) -> None:
    """contextlib.contextmanager wraps the function alone and touches no owner state, so a
    method it decorates promotes; which spelling binds it does not matter (P118)."""

    path = tmp_path / "probe.py"
    source = (
        f"{imports}class Base:\n    pass\n"
        f"class Owner(Base):\n{WRAPPED.format(decorator=decorator)}{OBSERVE}"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)])
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = _plan(path).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(path)]) == before
    assert "class Base:\n" + WRAPPED.format(decorator=decorator) in path.read_text()


@pytest.mark.parametrize(
    "imports,decorator,observe",
    (
        (
            "from contextlib import contextmanager as _cm\n"
            "def contextmanager(function):\n    return _cm(function)\n",
            "@contextmanager", OBSERVE,
        ),
        (
            "from functools import singledispatchmethod\n",
            "@singledispatchmethod",
            "print(Owner.value)\n",
        ),
    ),
    ids=("shadowed_local", "singledispatchmethod"),
)
def test_a_decorator_not_bound_to_an_owner_independent_wrapper_refuses(
    tmp_path: Path, imports: str, decorator: str, observe: str
) -> None:
    """A local function spelled `contextmanager` is not contextlib's, whatever it does, and
    singledispatchmethod is a descriptor that dispatches on its owner: both stay refused."""

    path = tmp_path / "probe.py"
    source = (
        f"{imports}class Base:\n    pass\n"
        f"class Owner(Base):\n{WRAPPED.format(decorator=decorator)}{observe}"
    )
    path.write_text(source, encoding="utf-8", newline="")
    subprocess.run([sys.executable, str(path)], check=True, capture_output=True)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(ValueError, match="custom_method_decorator"):
        _plan(path).simulate(snapshot)
    assert path.read_bytes() == source.encode("utf-8")


@pytest.mark.parametrize(
    "prelude,between",
    (
        ("", "from contextlib import contextmanager\n"),
        ("from contextlib import contextmanager\n", ""),
    ),
    ids=("imported_after_the_destination", "destination_body_rebinds"),
)
def test_a_wrapper_the_destination_does_not_bind_refuses(
    tmp_path: Path, prelude: str, between: str
) -> None:
    """The promoted decorator runs where the destination class statement runs. Imported only
    after the destination, it is unbound there; bound by the destination's own body, it is
    the destination's. Either way the move would change what the method is, so it refuses."""

    base_body = "    pass\n" if not prelude else "    contextmanager = staticmethod(lambda function: function)\n"
    path = tmp_path / "probe.py"
    source = (
        f"{prelude}class Base:\n{base_body}{between}"
        f"class Owner(Base):\n{WRAPPED.format(decorator='@contextmanager')}{OBSERVE}"
    )
    path.write_text(source, encoding="utf-8", newline="")
    subprocess.run([sys.executable, str(path)], check=True, capture_output=True)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(ValueError):
        _plan(path).simulate(snapshot)
    assert path.read_bytes() == source.encode("utf-8")
