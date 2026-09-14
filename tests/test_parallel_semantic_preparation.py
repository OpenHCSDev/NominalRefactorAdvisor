"""Workers derive lexical identities; only their source cache publishes them."""

from dataclasses import replace
import os

import pytest

from nominal_refactor_advisor.analysis import DetectorAnalysisWorkerPlan
from nominal_refactor_advisor.analysis_cache import (
    AnalysisFindingCache,
    CachedSourceFileSignature,
    SourceFileSignatureCache,
)
from nominal_refactor_advisor.ast_tools import semantic_python_source_hash


@pytest.mark.parametrize("workers", (1, 2))
def test_workers_preserve_signatures_and_parent_owns_publication(tmp_path, workers):
    texts = ("value = 1\n", "# comment\r\nvalue = 'é'\r\n", "value = (\n  2\n)\n")
    paths = tuple(tmp_path / f"source_{index}.py" for index in range(len(texts)))
    for path, text in zip(paths, texts, strict=True):
        path.write_bytes(text.encode("utf-8"))
    cache = AnalysisFindingCache(tmp_path / "cache")
    owner = cache.source_signature_cache()
    assert owner is not None
    owner.source_file_signatures(paths)
    pending = owner.pending_semantic_hashes(paths)
    plan = DetectorAnalysisWorkerPlan(workers, len(pending))
    computed = plan.map(CachedSourceFileSignature.with_semantic_hash, pending)

    assert all(entry.semantic_hash is None for entry in owner.entries_by_path.values())
    separate = cache.source_signature_cache()
    assert separate is not None
    assert len(separate.pending_semantic_hashes(paths)) == len(paths)
    for signature, text in zip(computed, texts, strict=True):
        assert owner.record_semantic_hash(signature) == semantic_python_source_hash(text)
    owner.store_if_dirty()
    warm = cache.source_signature_cache()
    assert warm is not None
    assert warm.pending_semantic_hashes(paths) == ()
    assert tuple(warm.semantic_source_hash(path) for path in paths) == tuple(
        semantic_python_source_hash(text) for text in texts
    )


def test_preparation_rejects_changed_bytes_even_with_preserved_stat(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"value = 1\n")
    original_stat = path.stat()
    owner = SourceFileSignatureCache(None)
    (pending,) = owner.pending_semantic_hashes((path,))
    path.write_bytes(b"value = 2\n")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ValueError, match="Source changed during semantic preparation"):
        pending.with_semantic_hash()


def test_publication_rejects_source_changed_after_preparation(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"value = 1\n")
    owner = SourceFileSignatureCache(None)
    (pending,) = owner.pending_semantic_hashes((path,))
    result = pending.with_semantic_hash()
    path.write_bytes(b"value = 200\n")
    with pytest.raises(ValueError, match="Source changed before semantic publication"):
        owner.record_semantic_hash(result)
    assert owner.entries_by_path[result.path].semantic_hash is None


def test_publication_rejects_replaced_source_authority(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"value = 1\n")
    owner = SourceFileSignatureCache(None)
    (pending,) = owner.pending_semantic_hashes((path,))
    result = pending.with_semantic_hash()
    owner.entries_by_path[result.path] = replace(pending, source_hash="different")
    with pytest.raises(ValueError, match="Source changed before semantic publication"):
        owner.record_semantic_hash(result)


def test_raw_signature_cannot_be_published_as_semantic_evidence(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"value = 1\n")
    owner = SourceFileSignatureCache(None)
    (pending,) = owner.pending_semantic_hashes((path,))
    with pytest.raises(ValueError, match="no lexical identity"):
        owner.record_semantic_hash(pending)


def test_changed_file_is_the_only_pending_semantic_task(tmp_path):
    paths = (tmp_path / "first.py", tmp_path / "second.py")
    owner = SourceFileSignatureCache(None)
    for path in paths:
        path.write_bytes(b"value = 1\n")
        owner.semantic_source_hash(path)
    paths[1].write_bytes(b"value = 200\n")
    (pending,) = owner.pending_semantic_hashes(paths)
    assert pending.path == str(paths[1])


def test_single_preparation_task_does_not_start_idle_workers(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    path.write_bytes(b"value = 1\n")
    owner = SourceFileSignatureCache(None)
    pending = owner.pending_semantic_hashes((path,))

    def forbidden(*args, **kwargs):
        raise AssertionError("One task must execute in the current process")

    monkeypatch.setattr("nominal_refactor_advisor.analysis.ProcessPoolExecutor", forbidden)
    result = DetectorAnalysisWorkerPlan(16, len(pending)).map(
        CachedSourceFileSignature.with_semantic_hash, pending,
    )
    assert len(result) == 1
    assert result[0].semantic_hash == semantic_python_source_hash("value = 1\n")
