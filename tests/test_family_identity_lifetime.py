"""Immutable source/family cache keys reuse the existing invocation lifetime."""

from dataclasses import dataclass, fields, replace
import gc
import hashlib
import pickle
import weakref

import pytest

from nominal_refactor_advisor.ast_tools import (
    BuilderCallShapeFamily,
    CollectedFamilyCacheContext,
    CollectedFamilySchemaIdentity,
)
from nominal_refactor_advisor.scan_cache import ScanCache


def context(tmp_path, name="module"):
    return CollectedFamilyCacheContext(tmp_path / f"{name}.py", name, "source", None)


def test_unscoped_identity_remains_fresh(tmp_path):
    source = context(tmp_path)
    first = source.identity(BuilderCallShapeFamily)
    second = source.identity(BuilderCallShapeFamily)
    assert first == second
    assert first is not second
    assert first.family_schema is not second.family_schema


def test_shared_schema_and_source_identity_live_only_within_outer_scan(tmp_path):
    source = context(tmp_path)
    with ScanCache.scope():
        first = source.identity(BuilderCallShapeFamily)
        with ScanCache.scope():
            assert source.identity(BuilderCallShapeFamily) is first
        another = context(tmp_path, "another").identity(BuilderCallShapeFamily)
        assert another is not first
        assert another.family_schema is first.family_schema
        assert (
            CollectedFamilySchemaIdentity.from_family(BuilderCallShapeFamily)
            is first.family_schema
        )
    with ScanCache.scope():
        refreshed = source.identity(BuilderCallShapeFamily)
        assert refreshed == first
        assert refreshed is not first
        assert refreshed.family_schema is not first.family_schema


@pytest.mark.parametrize("demand", ("", "focused-demand"))
def test_token_format_is_unchanged_and_computed_once_per_identity(
    tmp_path, monkeypatch, demand
):
    identity = context(tmp_path).identity(BuilderCallShapeFamily, demand)
    expected = hashlib.blake2s(
        repr(identity).encode("utf-8"), digest_size=16
    ).hexdigest()
    cls = type(identity)
    original_repr = cls.__repr__
    calls = []

    def observed_repr(self):
        calls.append(self)
        return original_repr(self)

    monkeypatch.setattr(cls, "__repr__", observed_repr)
    assert identity.cache_token == identity.cache_token == expected
    assert len(calls) == 1
    assert "cache_token" not in {field.name for field in fields(identity)}


def test_changed_source_demand_module_and_schema_keep_distinct_keys(tmp_path):
    source = context(tmp_path)
    with ScanCache.scope():
        original = source.identity(BuilderCallShapeFamily)
        candidates = (
            replace(source, source_signature="changed").identity(
                BuilderCallShapeFamily
            ),
            replace(source, module_name="changed").identity(BuilderCallShapeFamily),
            context(tmp_path, "other_path").identity(BuilderCallShapeFamily),
            source.identity(BuilderCallShapeFamily, "focused"),
            replace(
                original,
                family_schema=replace(
                    original.family_schema, item_schema_signature="changed"
                ),
            ),
        )
        assert (
            len(
                {
                    original.cache_token,
                    *(candidate.cache_token for candidate in candidates),
                }
            )
            == 6
        )


def test_changed_family_declaration_is_observed_on_the_next_invocation(
    tmp_path, monkeypatch
):
    @dataclass(frozen=True)
    class Item:
        changed: int

    source = context(tmp_path)
    with ScanCache.scope():
        original = source.identity(BuilderCallShapeFamily)
    monkeypatch.setattr(BuilderCallShapeFamily, "item_type", Item)
    with ScanCache.scope():
        changed = source.identity(BuilderCallShapeFamily)
    assert (
        changed.family_schema.item_schema_signature
        != original.family_schema.item_schema_signature
    )
    assert changed.cache_token != original.cache_token


def test_replacement_does_not_inherit_an_old_derived_token(tmp_path):
    identity = context(tmp_path).identity(BuilderCallShapeFamily)
    original_token = identity.cache_token
    changed = replace(identity, source_signature="new-source")
    assert "cache_token" not in vars(changed)
    assert changed.cache_token != original_token
    restored = pickle.loads(pickle.dumps(identity))
    assert "cache_token" not in vars(restored)
    assert restored == identity
    assert restored.cache_token == original_token
    assert (
        replace(restored, source_signature="new-source").cache_token
        == changed.cache_token
    )


def test_exception_releases_scoped_identities_and_contexts(tmp_path):
    with pytest.raises(RuntimeError):
        with ScanCache.scope():
            source = context(tmp_path)
            source_ref = weakref.ref(source)
            identity_ref = weakref.ref(source.identity(BuilderCallShapeFamily))
            del source
            assert source_ref() is not None
            assert identity_ref() is not None
            raise RuntimeError("end invocation")
    gc.collect()
    assert source_ref() is None
    assert identity_ref() is None
