"""Function birth storage is canonical compiler evidence, not later object state."""

import ast
from dataclasses import replace
import sys
from types import FunctionType

import pytest

from nominal_refactor_advisor.native_compilation import (
    CPython311CreationBackend,
    CPython314CreationBackend,
    CPythonFunctionConstruction,
    CreatedNativeFunctionExecution,
    ExactNativeFunctionExecution,
    NativeCreationBackend,
    NativePythonCompilation,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def compilation(source="def chosen():\n    raise AssertionError('deferred body')\n"):
    owner = NativePythonCompilation(source, "<authored-fresh-function-namespace>")
    node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "chosen"
    )
    return owner, owner.execution_for(SourceByteSpan.require_node(node))


@pytest.mark.parametrize(
    "declaration",
    (
        "def chosen():\n    raise AssertionError('deferred body')\n",
        "async def chosen():\n    raise AssertionError('deferred body')\n",
        "def chosen():\n    yield missing_body_call()\n",
        "async def chosen():\n    yield missing_body_call()\n",
        "def chosen(value: int = 1, *, option: str = 'x') -> int:\n    raise AssertionError('deferred body')\n",
    ),
)
def test_actual_fresh_functions_have_empty_custom_storage_without_body_execution(
    declaration,
):
    owner, receipt = compilation(declaration)
    assert isinstance(receipt, CreatedNativeFunctionExecution)
    owner.require_fresh_function_namespace(receipt)
    namespace = {}
    exec(
        owner.compile(), namespace
    )  # This authored fixture only, never repository code.
    function = namespace["chosen"]
    assert type(function) is FunctionType
    assert vars(function) == {}
    assert all(
        "__isabstractmethod__" not in vars(base) for base in FunctionType.__mro__
    )


def test_canonical_birth_law_reuses_the_compilation_without_rerunning_native_work(
    monkeypatch,
):
    owner, receipt = compilation()

    def forbidden(self):
        raise AssertionError("The canonical compilation is already cached")

    monkeypatch.setattr(NativePythonCompilation, "compile", forbidden)
    for _ in range(3):
        owner.require_fresh_function_namespace(receipt)


@pytest.mark.parametrize(
    "damage", ("copy", "offset", "flags", "source_identity", "span")
)
def test_copied_or_forged_receipts_do_not_authenticate(damage):
    owner, receipt = compilation()
    variants = {
        "copy": replace(receipt),
        "offset": replace(
            receipt, creation=replace(receipt.creation, instruction_offset=999999)
        ),
        "flags": replace(receipt, native_flags=receipt.native_flags ^ 1),
        "source_identity": replace(receipt, compilation=replace(receipt.compilation)),
        "span": replace(
            receipt,
            source_span=SourceByteSpan.require_node(
                ast.parse("def other(): pass").body[0]
            ),
        ),
    }
    with pytest.raises(ValueError, match="canonical compilation receipt"):
        owner.require_fresh_function_namespace(variants[damage])


def test_equal_source_compilations_are_distinct_receipt_owners():
    owner, receipt = compilation()
    other, other_receipt = compilation(owner.source)
    assert owner.identity == other.identity and owner.identity is not other.identity
    other.require_fresh_function_namespace(other_receipt)
    with pytest.raises(ValueError, match="canonical compilation receipt"):
        other.require_fresh_function_namespace(receipt)


def test_merely_exact_code_receipt_cannot_supply_a_birth_law(monkeypatch):
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    owner, receipt = compilation()
    assert type(receipt) is ExactNativeFunctionExecution
    with pytest.raises(ValueError, match="raw function creation"):
        owner.require_fresh_function_namespace(receipt)


def test_changed_to_unsupported_backend_does_not_reuse_prior_native_admission(
    monkeypatch,
):
    owner, receipt = compilation()
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    with pytest.raises(ValueError, match="fresh function namespace remains unproved"):
        owner.require_fresh_function_namespace(receipt)


def test_canonical_receipt_with_foreign_interpreter_identity_is_rejected():
    owner = NativePythonCompilation("def chosen(): pass\n", "<foreign-interpreter>")
    owner.__dict__["identity"] = replace(owner.identity, interpreter=("foreign", "0.0"))
    node = ast.parse(owner.source).body[0]
    receipt = owner.execution_for(SourceByteSpan.require_node(node))
    assert receipt.compilation is owner.identity
    with pytest.raises(ValueError, match="actual native interpreter backend"):
        owner.require_fresh_function_namespace(receipt)


def test_wrong_cpython_backend_does_not_reuse_another_versions_contract(monkeypatch):
    owner, receipt = compilation()
    other_backend = (
        CPython314CreationBackend
        if sys.version_info[:2] == (3, 11)
        else CPython311CreationBackend
    )
    wrong = object.__new__(other_backend)
    monkeypatch.setattr(
        NativeCreationBackend, "current", classmethod(lambda cls: wrong)
    )
    with pytest.raises(ValueError, match="actual native interpreter backend"):
        owner.require_fresh_function_namespace(receipt)


def test_supported_backends_share_the_nominal_birth_law():
    for backend in (CPython311CreationBackend, CPython314CreationBackend):
        assert (
            backend.require_fresh_function_namespace
            is CPythonFunctionConstruction.require_fresh_function_namespace
        )


def test_birth_storage_does_not_claim_later_abstractness_or_instance_identity():
    owner, receipt = compilation()
    namespace = {}
    exec(owner.compile(), namespace)
    function = namespace["chosen"]
    assert vars(function) == {}
    function.__isabstractmethod__ = True
    assert type(function) is FunctionType and vars(function) == {
        "__isabstractmethod__": True
    }
    # Still a true conditional birth-state claim, not the current function dictionary.
    assert owner.require_fresh_function_namespace(receipt) is None


def test_canonical_nested_creation_does_not_assert_its_enclosing_body_ran():
    owner, receipt = compilation(
        "def outer():\n    def chosen():\n        raise AssertionError('deferred')\n    return chosen\n"
    )
    owner.require_fresh_function_namespace(receipt)
    namespace = {}
    exec(owner.compile(), namespace)
    assert "chosen" not in namespace
