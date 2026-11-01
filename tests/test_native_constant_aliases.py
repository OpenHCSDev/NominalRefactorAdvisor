"""Compiler alias records survive transport without becoming target identities."""

import ast
import dis
from dataclasses import replace
import marshal
from pathlib import Path
import pickle

import pytest

from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.native_compilation import (
    NativeConstantValue,
    NativeCreationBackend,
    NativePrimitiveOperation,
    NativeValueStore,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def default_stores(source: str, path: Path) -> tuple[NativeValueStore, ...]:
    parsed = SourceModule.from_source_path(path, source).parse()
    result = []
    for node in parsed.module.body:
        assert isinstance(node, ast.FunctionDef)
        span = SourceByteSpan.require_node(node)
        result.append(parsed.native_compilation.value_store_for(span, span, node.name))
    assert not path.exists()
    return tuple(result)


@pytest.fixture
def original_stores(tmp_path: Path) -> tuple[NativeValueStore, ...]:
    return default_stores(
        "def positional(value=5000): pass\ndef keyword(*, value=5000): pass\n",
        tmp_path / "unexecuted.py",
    )


def test_original_alias_survives_compact_serialization(
    original_stores: tuple[NativeValueStore, ...],
) -> None:
    for stores in (original_stores, pickle.loads(pickle.dumps(original_stores))):
        first = stores[0].value.positional_default_operands()[0]
        second = stores[1].value.keyword_default_operands()[0]
        stores[0].require_value(first.storage)
        stores[1].require_value(second.storage)
        first.value.require_same_compiler_constant(second.value)
        first.value.require_literal_contents(5000)
        second.value.require_literal_contents(5000)
        with pytest.raises(ValueError, match="requires its original production"):
            stores[0].require_value(replace(first.storage))


def test_distinct_pool_origins_do_not_gain_shared_identity(tmp_path: Path) -> None:
    first, second = default_stores(
        "def positional(value=5000): pass\ndef keyword(*, value=5001): pass\n",
        tmp_path / "distinct.py",
    )
    left = first.value.positional_default_operands()[0].value
    right = second.value.keyword_default_operands()[0].value
    with pytest.raises(ValueError, match="shared original compiler identity"):
        left.require_same_compiler_constant(right)


def test_equal_contents_in_independent_observations_are_not_one_origin(tmp_path: Path) -> None:
    source = "def selected(value=5000): pass\n"
    (first,) = default_stores(source, tmp_path / "first.py")
    (second,) = default_stores(source, tmp_path / "second.py")
    left = first.value.positional_default_operands()[0].value
    right = second.value.positional_default_operands()[0].value
    left.require_literal_contents(5000)
    right.require_literal_contents(5000)
    with pytest.raises(ValueError, match="shared original compiler identity"):
        left.require_same_compiler_constant(right)


def test_contents_only_value_does_not_claim_its_own_alias_identity() -> None:
    value = NativeConstantValue(0, (), 5000)
    value.require_literal_contents(5000)
    with pytest.raises(ValueError, match="alias remains unproved"):
        value.require_same_compiler_constant(value)


@pytest.mark.skipif(NativePrimitiveOperation.LOAD_SMALL_INT.name not in dis.opmap, reason="Compiler has no immediate integer opcode")
def test_immediate_boxed_argval_is_not_constant_pool_identity(tmp_path: Path) -> None:
    first, second = default_stores(
        "def positional(value=0): pass\ndef keyword(*, value=0): pass\n",
        tmp_path / "immediate.py",
    )
    pooled = first.value.positional_default_operands()[0].value
    immediate = second.value.keyword_default_operands()[0].value
    pooled.require_literal_contents(0)
    immediate.require_literal_contents(0)
    with pytest.raises(ValueError, match="alias remains unproved"):
        pooled.require_same_compiler_constant(immediate)


@pytest.mark.parametrize("index", [-1, 1])
def test_constant_member_stays_inside_original_tuple(
    original_stores: tuple[NativeValueStore, ...], index: int
) -> None:
    storage = original_stores[0].value.positional_default_operands()[0].storage
    with pytest.raises(ValueError, match="member is outside its original graph"):
        storage.compiler_constant_identity().member(index)


def test_fingerprint_still_distinguishes_equal_contents_with_different_aliases() -> None:
    first, second = int("5000"), int("5000")
    assert first is not second
    template = compile("pass", "unexecuted.py", "exec")
    shared = template.replace(co_consts=((first, first),))
    distinct = template.replace(co_consts=((first, second),))
    assert marshal.dumps(shared, 2) == marshal.dumps(distinct, 2)
    backend = NativeCreationBackend.current()
    assert backend.code_contents(shared) != backend.code_contents(distinct)
