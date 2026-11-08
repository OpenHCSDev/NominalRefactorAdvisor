"""Derived state is keyed by content and checkout-relative paths, never by where a
checkout sits: a new worktree reads what another worktree already paid for."""

from __future__ import annotations

import io
from pathlib import Path, PurePosixPath

from nominal_refactor_advisor.analysis_cache import (
    AnalysisCacheStorage,
    SourceFileSignatureCache,
)
from nominal_refactor_advisor.ast_tools import (
    CollectedFamilyCacheContext,
    parse_python_modules,
)
from nominal_refactor_advisor.cache_checkout import CheckoutRelocation
from nominal_refactor_advisor.cache_paths import (
    default_analysis_cache_dir,
    default_parse_cache_dir,
)
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily


def _checkout(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    (root / ".git").mkdir(parents=True)
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "mod.py").write_text("class Shared:\n    pass\n")
    return root


def _context(root: Path, cache_dir: Path) -> CollectedFamilyCacheContext:
    path = root / "pkg" / "mod.py"
    return CollectedFamilyCacheContext.from_source(
        path=path,
        module_name="pkg.mod",
        source=path.read_text(),
        family_cache_dir=cache_dir,
    )


def _collected(root: Path) -> tuple[object, ...]:
    (module,) = (
        module
        for module in parse_python_modules(root, use_parse_cache=False)
        if module.module_name == "pkg.mod"
    )
    return tuple(CompactModuleClassProjectionFamily.collect(module))


def test_a_family_entry_one_checkout_stored_is_read_by_another(tmp_path: Path) -> None:
    """A fresh agent-comms worktree re-collected every module (103.9 s for one edit,
    against 8-9 s in the worktree that wrote the cache): entries were keyed by,
    and their items spelled, the absolute checkout path (11-08)."""

    cache_dir = tmp_path / "collected-family"
    first, second = _checkout(tmp_path, "first"), _checkout(tmp_path, "second")
    stored = _collected(first)
    assert str(first) in repr(stored)
    first_context, second_context = _context(first, cache_dir), _context(second, cache_dir)
    assert first_context.store_items(CompactModuleClassProjectionFamily, stored)

    read = second_context.load_items(CompactModuleClassProjectionFamily)

    assert read == _collected(second)
    assert str(first) not in repr(read)
    assert first_context.items_content_signature(stored) == (
        second_context.items_content_signature(read)
    )


def test_relocation_rebases_paths_and_binds_what_it_cannot_rebase(tmp_path: Path) -> None:
    first = CheckoutRelocation(tmp_path / "first")
    second = CheckoutRelocation(tmp_path / "second")
    value = (str(first.root), f"{first.root}/pkg/mod.py:3", PurePosixPath(first.root, "x"))

    data, portable = first.portable_dumps(value)

    assert portable
    assert second.load(io.BytesIO(data)) == (
        str(second.root),
        f"{second.root}/pkg/mod.py:3",
        PurePosixPath(second.root, "x"),
    )
    assert first.relative_id(f"{first.root}-sibling/x") is None
    _, embedded_is_portable = first.portable_dumps(f"read {first.root}/pkg")
    assert not embedded_is_portable


def test_source_signatures_keep_one_shard_per_checkout(tmp_path: Path) -> None:
    storage = AnalysisCacheStorage(tmp_path / "analysis")
    first, second = _checkout(tmp_path, "first"), _checkout(tmp_path, "second")
    paths = (first / "pkg" / "mod.py", second / "pkg" / "mod.py")

    signatures = SourceFileSignatureCache(storage).source_file_signatures(paths)

    assert {path.name for path in storage.storage_root.iterdir()} == {
        storage.source_signature_cache_path(CheckoutRelocation.containing(path)).name
        for path in paths
    }
    assert SourceFileSignatureCache(storage).source_file_signatures(paths) == signatures


def test_the_default_cache_is_one_home_shared_by_every_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NRA_CACHE_HOME", (tmp_path / "cache-home").as_posix())

    assert default_parse_cache_dir() == tmp_path / "cache-home" / "shared" / "ast"
    assert default_analysis_cache_dir().parent == default_parse_cache_dir().parent
