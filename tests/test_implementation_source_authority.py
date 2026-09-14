"""Cache layers derive source provenance from the same loaded-module owner."""

import hashlib
import importlib.machinery
from contextlib import nullcontext
import os
import sys
from types import ModuleType

import pytest

from nominal_refactor_advisor.analysis_cache import (
    DetectorRegistrySignature,
    _module_source_signature,
)
from nominal_refactor_advisor.detectors import IssueDetector
from nominal_refactor_advisor.implementation_identity import ImplementationSource
from nominal_refactor_advisor.scan_cache import ScanCache


def loaded_module(monkeypatch, name, path):
    module = ModuleType(name)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_script_source_without_import_spec_has_content_identity(tmp_path, monkeypatch):
    path = tmp_path / "script.py"
    path.write_text("VALUE = 1\n")
    module = loaded_module(monkeypatch, "__main__", path)
    assert module.__spec__ is None
    signature = _module_source_signature(module.__name__)
    assert signature.path == str(path)
    assert (
        signature.source_hash
        == hashlib.blake2s(path.read_bytes(), digest_size=16).hexdigest()
    )


def test_signature_consumes_implementation_source_owner_once(monkeypatch):
    calls = []

    def observe(cls, module_name):
        calls.append(module_name)
        return cls(module_name, "content-digest", "/actual/declaration.py")

    monkeypatch.setattr(ImplementationSource, "from_module_name", classmethod(observe))
    signature = _module_source_signature("declaration")
    assert calls == ["declaration"]
    assert signature.path == "/actual/declaration.py"
    assert signature.source_hash == "content-digest"


def test_loaded_source_is_not_reselected_from_an_import_spec(tmp_path, monkeypatch):
    path = tmp_path / "actual.py"
    path.write_text("VALUE = 1\n")
    other = tmp_path / "import_candidate.py"
    other.write_text("VALUE = 2\n")
    module = loaded_module(monkeypatch, "loaded_fixture", path)
    module.__spec__ = importlib.machinery.ModuleSpec(
        module.__name__, None, origin=str(other)
    )
    observed = ImplementationSource.from_module_name(module.__name__)
    signature = _module_source_signature(module.__name__)
    assert signature.path == str(path)
    assert signature.source_hash == observed.source_signature


def test_source_edit_changes_both_cache_identities_in_one_scope(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    path.write_text("VALUE = 1\n")
    module = loaded_module(monkeypatch, "editable_fixture", path)
    with ScanCache.scope():
        before = _module_source_signature(module.__name__)
        path.write_text("VALUE = 123456\n")
        after = _module_source_signature(module.__name__)
        source = ImplementationSource.from_module_name(module.__name__)
    assert before != after
    assert after.source_hash == source.source_signature


def test_source_association_change_is_not_cached_by_module_name(tmp_path, monkeypatch):
    first = tmp_path / "one.py"
    second = tmp_path / "two.py"
    first.write_text("VALUE = 1\n")
    second.write_text("VALUE = 1\n")
    module = loaded_module(monkeypatch, "relocated_fixture", first)
    with ScanCache.scope():
        before = _module_source_signature(module.__name__)
        module.__file__ = str(second)
        after = _module_source_signature(module.__name__)
    assert before.source_hash == after.source_hash
    assert before.path == str(first)
    assert after.path == str(second)


@pytest.mark.parametrize("name", ("builtins", "missing_implementation_fixture"))
def test_no_source_file_preserves_named_identity(name):
    source = ImplementationSource.from_module_name(name)
    signature = _module_source_signature(name)
    assert signature.path == name
    assert signature.source_hash == source.source_signature


def test_missing_file_uses_same_explicit_path_identity(tmp_path, monkeypatch):
    path = tmp_path / "absent.py"
    module = loaded_module(monkeypatch, "missing_file_fixture", path)
    source = ImplementationSource.from_module_name(module.__name__)
    signature = _module_source_signature(module.__name__)
    assert signature.path == str(path)
    assert signature.source_hash == source.source_signature


@pytest.mark.parametrize("scoped", (False, True))
def test_detector_signature_refreshes_without_recreating_the_declaration(
    tmp_path, monkeypatch, scoped
):
    path = tmp_path / "detector.py"
    path.write_text("VALUE = 1\n")
    module = loaded_module(monkeypatch, "detector_source_fixture", path)

    class FixtureDetector(IssueDetector):
        pass

    FixtureDetector.__module__ = module.__name__
    scope = ScanCache.scope if scoped else nullcontext
    with scope():
        before = DetectorRegistrySignature.from_detector_types((FixtureDetector,))
        if scoped:
            assert (
                DetectorRegistrySignature.from_detector_types((FixtureDetector,))
                is before
            )
    stamp = path.stat()
    path.write_text("VALUE = 2\n")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with scope():
        after = DetectorRegistrySignature.from_detector_types((FixtureDetector,))
    assert before != after
    assert (
        before.detector_types[0].implementation_source_hash
        != after.detector_types[0].implementation_source_hash
    )


def test_source_location_does_not_duplicate_module_content_identity(
    tmp_path, monkeypatch
):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("VALUE = 1\n")
    second.write_text("VALUE = 1\n")
    module = loaded_module(monkeypatch, "portable_source_fixture", first)
    before = ImplementationSource.from_module_name(module.__name__)
    module.__file__ = str(second)
    after = ImplementationSource.from_module_name(module.__name__)
    assert before.path != after.path
    assert before == after
    assert repr(before) == repr(after)
