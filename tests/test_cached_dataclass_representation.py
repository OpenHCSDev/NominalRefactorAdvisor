"""Immutable representation reuse preserves native declaration-derived text."""

import copy
import gc
import pickle
import weakref
from dataclasses import dataclass, field, fields, make_dataclass, replace

import pytest

from nominal_refactor_advisor.ast_tools import (
    BuilderCallShapeFamily,
    CollectedFamilyCacheContext,
    CollectedFamilySchemaIdentity,
)
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.value_graph import CachedDataclassRepresentation


@dataclass(frozen=True, repr=False)
class Record(CachedDataclassRepresentation):
    text: str
    hidden: str = field(repr=False)


@dataclass(frozen=True, repr=False)
class ExtendedRecord(Record):
    coordinates: tuple[int, ...]


def native_repr(value):
    owner = type(value)
    native = make_dataclass(
        owner.__name__,
        [(item.name, item.type, field(repr=item.repr)) for item in fields(value)],
        frozen=True,
    )
    native.__qualname__ = owner.__qualname__
    return repr(native(*(getattr(value, item.name) for item in fields(value))))


@pytest.mark.parametrize(
    "value",
    (Record("quote'\\\n", "hidden"), ExtendedRecord("a", "b", (1, 2))),
)
def test_native_format_and_field_flags_are_preserved(value):
    expected = native_repr(value)
    assert repr(value) == expected
    assert repr(value) is repr(value)
    assert value.__getstate__() == {
        item.name: getattr(value, item.name) for item in fields(value)
    }


@pytest.mark.parametrize(
    "transport", (copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x)))
)
def test_transport_and_replacement_rederive_text(transport):
    original = ExtendedRecord("old", "hidden", (1, 2))
    expected = repr(original)
    restored = transport(original)
    assert "_representation" not in vars(restored)
    assert restored == original
    assert repr(restored) == expected
    changed = replace(original, text="new")
    assert "_representation" not in vars(changed)
    assert repr(changed) == native_repr(changed) != expected


def test_shared_family_representation_is_derived_once_and_keys_are_unchanged(
    tmp_path, monkeypatch
):
    calls = []
    original = CollectedFamilySchemaIdentity._representation.func

    def observed(value):
        calls.append(id(value))
        return original(value)

    monkeypatch.setattr(CollectedFamilySchemaIdentity._representation, "func", observed)
    with ScanCache.scope():
        first = CollectedFamilyCacheContext(
            tmp_path / "a.py", "a", "source", None
        ).identity(BuilderCallShapeFamily)
        second = CollectedFamilyCacheContext(
            tmp_path / "b.py", "b", "source", None
        ).identity(BuilderCallShapeFamily)
        schema = first.family_schema
        assert schema is second.family_schema
        expected = native_repr(schema)
        assert repr(schema) == expected
        assert repr(first) == native_repr(first)
        assert repr(second) == native_repr(second)
        assert calls == [id(schema)]
        assert hash(schema) == hash(replace(schema))
        reference = weakref.ref(schema)
        del first, second, schema
    gc.collect()
    assert reference() is None


def test_implementation_changes_and_transport_do_not_reuse_stale_schema_text(tmp_path):
    identity = CollectedFamilyCacheContext(
        tmp_path / "a.py", "a", "source", None
    ).identity(BuilderCallShapeFamily)
    schema = identity.family_schema
    original_token = identity.cache_token
    first, *remaining = schema.implementation.sources
    changed = replace(
        schema,
        implementation=replace(
            schema.implementation,
            sources=(replace(first, source_signature="new-implementation"), *remaining),
        ),
    )
    assert "_representation" not in vars(changed)
    assert repr(changed) == native_repr(changed) != repr(schema)
    assert replace(identity, family_schema=changed).cache_token != original_token
    restored = pickle.loads(pickle.dumps(identity))
    assert "_representation" not in vars(restored.family_schema)
    assert restored.cache_token == original_token
