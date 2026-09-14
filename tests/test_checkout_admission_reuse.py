"""Completed path admission depends on all current roots, not source contents."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.cache_checkout import (
    CacheCheckoutPathError,
    CheckoutRoot,
    absolute_checkout_path,
    checkout_relative_path,
    lexical_absolute_path,
)
from nominal_refactor_advisor.scan_cache import ScanCache


def test_reuse_keeps_all_roots_in_the_dependency_key(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    source = first / "module.py"
    source.write_text("value = 1\n")
    calls = []
    relative_path = CheckoutRoot.relative_path

    def observed(root, candidate):
        calls.append(root.path)
        return relative_path(root, candidate)

    monkeypatch.setattr(CheckoutRoot, "relative_path", observed)
    with ScanCache.scope():
        assert checkout_relative_path(source, (first, second)) == "0:module.py"
        assert calls == [first, second]
        source.write_text("value = 200\n")
        assert checkout_relative_path(source, (first, second)) == "0:module.py"
        assert calls == [first, second]
        assert checkout_relative_path(source, (second, first)) == "1:module.py"
        assert calls == [first, second, second, first]
        with pytest.raises(CacheCheckoutPathError, match="multiple roots"):
            checkout_relative_path(source, (first, tmp_path))
    count = len(calls)
    assert checkout_relative_path(source, (first, second)) == "0:module.py"
    assert len(calls) == count + 2


def test_warm_admission_reobserves_root_file_kind(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    candidate = root / "module.py"
    with ScanCache.scope():
        assert checkout_relative_path(candidate, (root,)) == "0:module.py"
        root.rmdir()
        root.write_text("now a file")
        assert checkout_relative_path(root, (root,)) == "0:."
        with pytest.raises(CacheCheckoutPathError, match="outside every"):
            checkout_relative_path(candidate, (root,))
        root.unlink()
        root.mkdir()
        assert checkout_relative_path(candidate, (root,)) == "0:module.py"


def test_warm_relative_paths_follow_current_working_directory(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "pkg").mkdir(parents=True)
    (second / "pkg").mkdir(parents=True)
    candidate = first / "pkg" / "module.py"
    with ScanCache.scope():
        monkeypatch.chdir(first)
        assert lexical_absolute_path("pkg") == first / "pkg"
        assert checkout_relative_path(candidate, (Path("pkg"),)) == "0:module.py"
        monkeypatch.chdir(second)
        assert lexical_absolute_path("pkg") == second / "pkg"
        with pytest.raises(CacheCheckoutPathError, match="outside every"):
            checkout_relative_path(candidate, (Path("pkg"),))


def test_warm_file_root_supports_both_relative_request_forms(tmp_path, monkeypatch):
    path = tmp_path / "pkg" / "module.py"
    path.parent.mkdir()
    path.write_text("value = 1\n")
    monkeypatch.chdir(tmp_path)
    roots = (Path("pkg/module.py"),)
    with ScanCache.scope():
        for request in ("module.py", Path("pkg/module.py"), path):
            assert checkout_relative_path(request, roots) == "0:."
            assert absolute_checkout_path("0:.", roots) == str(path)


def test_source_symlink_target_changes_do_not_rewrite_lexical_admission(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("first")
    second.write_text("second")
    link = root / "module.py"
    link.symlink_to(first)
    with ScanCache.scope():
        assert checkout_relative_path(link, (root,)) == "0:module.py"
        link.unlink()
        link.symlink_to(second)
        assert checkout_relative_path(link, (root,)) == "0:module.py"
        assert absolute_checkout_path("0:module.py", (root,)) == str(link)


def test_root_symlink_file_kind_is_not_cached(tmp_path):
    directory = tmp_path / "directory"
    directory.mkdir()
    file = tmp_path / "module.py"
    file.write_text("value = 1\n")
    root = tmp_path / "root"
    root.symlink_to(directory, target_is_directory=True)
    candidate = root / "child.py"
    with ScanCache.scope():
        assert checkout_relative_path(candidate, (root,)) == "0:child.py"
        root.unlink()
        root.symlink_to(file)
        with pytest.raises(CacheCheckoutPathError, match="outside every"):
            checkout_relative_path(candidate, (root,))
        assert checkout_relative_path(root, (root,)) == "0:."


@pytest.mark.parametrize("path_request", ("../escape.py", "a/../../escape.py"))
def test_relative_traversal_never_enters_the_cached_admission(tmp_path, path_request):
    with ScanCache.scope():
        assert checkout_relative_path("valid.py", (tmp_path,)) == "0:valid.py"
        with pytest.raises(CacheCheckoutPathError, match="unsafe"):
            checkout_relative_path(path_request, (tmp_path,))


def test_repeated_and_absent_roots_stay_ambiguous(tmp_path):
    source = tmp_path / "module.py"
    with ScanCache.scope():
        assert checkout_relative_path(source, (tmp_path,)) == "0:module.py"
        with pytest.raises(CacheCheckoutPathError, match="multiple roots"):
            checkout_relative_path(source, (tmp_path, tmp_path))
        with pytest.raises(CacheCheckoutPathError, match="outside every"):
            checkout_relative_path(source, ())
        with pytest.raises(CacheCheckoutPathError, match="ambiguous"):
            checkout_relative_path("module.py", (tmp_path, tmp_path))


def test_absolute_spelling_reuse_is_scoped(tmp_path):
    source = tmp_path / "module.py"
    with ScanCache.scope():
        first = lexical_absolute_path(source)
        assert lexical_absolute_path(str(source)) is first
        assert lexical_absolute_path(source.parent / "." / source.name) is first
    assert lexical_absolute_path(source) == first
    assert lexical_absolute_path(source) is not first


def test_invalid_relative_requests_are_rejected_before_filesystem_observation(
    tmp_path, monkeypatch
):
    def unexpected_capture(cls, root):
        raise AssertionError("Invalid request reached filesystem observation")

    monkeypatch.setattr(CheckoutRoot, "capture", classmethod(unexpected_capture))
    with ScanCache.scope():
        with pytest.raises(CacheCheckoutPathError, match="unsafe"):
            checkout_relative_path("../escape.py", (tmp_path,))
        with pytest.raises(CacheCheckoutPathError, match="ambiguous"):
            checkout_relative_path("source.py", (tmp_path, tmp_path))
