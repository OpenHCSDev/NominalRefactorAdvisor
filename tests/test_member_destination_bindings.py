"""Member promotion preserves header bindings at its destination owner."""

import ast
import functools
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


def _members(path: Path) -> dict[str, dict[str, list[str]]]:
    """Each class's method names with their decorators, as source text."""

    return {
        node.name: {
            member.name: [ast.unparse(decorator) for decorator in member.decorator_list]
            for member in node.body
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef)
    }


@pytest.mark.native_proof
@pytest.mark.parametrize(
    "imports,decorator,method,observe",
    (
        ("from contextlib import contextmanager\n", "@contextmanager", WRAPPED, OBSERVE),
        ("import contextlib\n", "@contextlib.contextmanager", WRAPPED, OBSERVE),
        (
            "from contextlib import asynccontextmanager\n",
            "@asynccontextmanager",
            "    {decorator}\n    async def value(self):\n        yield 4\n",
            "print(type(Owner().value()).__name__)\n",
        ),
    ),
    ids=("from_import", "attribute", "async"),
)
def test_an_owner_independent_wrapper_promotes_by_its_binding(
    tmp_path: Path, imports: str, decorator: str, method: str, observe: str
) -> None:
    """contextlib's context-manager decorators wrap the function alone and touch no owner
    state, so a method they decorate promotes; which spelling binds them does not matter
    (P118)."""

    path = tmp_path / "probe.py"
    source = (
        f"{imports}class Base:\n    pass\n"
        f"class Owner(Base):\n{method.format(decorator=decorator)}{observe}"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)])
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = _plan(path).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(path)]) == before
    assert _members(path) == {"Base": {"value": [decorator[1:]]}, "Owner": {}}


@pytest.mark.native_proof
def test_a_wrapper_over_another_decorator_refuses(tmp_path: Path) -> None:
    """update_wrapper reads the attributes and `__dict__` of whatever it wraps. Over a fresh
    source function those are stored slots; over another decorator's result they are that
    object's, so the wrapper's claim no longer covers the application and it refuses."""

    path = tmp_path / "probe.py"
    source = (
        "from contextlib import contextmanager\nclass Base:\n    pass\n"
        "class Owner(Base):\n    @contextmanager\n    @staticmethod\n"
        "    def value():\n        yield 4\n"
        "print(callable(Owner.value))\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    subprocess.run([sys.executable, str(path)], check=True, capture_output=True)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(ValueError, match="Function decorator result remains unproved"):
        _plan(path).simulate(snapshot)
    assert path.read_bytes() == source.encode("utf-8")


@pytest.mark.native_proof
def test_a_wrapper_refuses_when_update_wrapper_reads_other_attributes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPython before 3.14 has update_wrapper read `__annotations__`, which evaluates the
    method's annotations. The wrapper's claim names the attributes it reads, so an
    interpreter reading others refuses instead of quietly admitting user code."""

    assigned = functools.WRAPPER_ASSIGNMENTS
    monkeypatch.setattr(
        functools,
        "WRAPPER_ASSIGNMENTS",
        tuple("__annotations__" if name == "__annotate__" else name for name in assigned),
    )
    path = tmp_path / "probe.py"
    source = (
        "from contextlib import contextmanager\nclass Base:\n    pass\n"
        f"class Owner(Base):\n{WRAPPED.format(decorator='@contextmanager')}{OBSERVE}"
    )
    path.write_text(source, encoding="utf-8", newline="")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(ValueError, match="Function decorator result remains unproved"):
        _plan(path).simulate(snapshot)
    assert path.read_bytes() == source.encode("utf-8")


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
