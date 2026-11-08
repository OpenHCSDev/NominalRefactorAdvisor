"""The type checker's answers are kept under what they depend on, so a scan
after an unrelated change, or in another worktree, reads them without a build."""

from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.cache_paths import ActiveScanCache
from nominal_refactor_advisor.probe_owner import InferredReceiverType, ReceiverSpan
from nominal_refactor_advisor.type_checker_session import SourceFiles, TypeCheckerSession

pytest.importorskip("mypy")

_SOURCE = "class Goal:\n    paused_at: int | None = None\n\n\nmade = Goal()\nmade.paused_at\n"


def _checkout(tmp_path: Path, name: str) -> tuple[Path, ReceiverSpan]:
    root = tmp_path / name
    (root / ".git").mkdir(parents=True)
    module = root / "owner.py"
    module.write_text(_SOURCE)
    return root, ReceiverSpan(str(module), 6, 0, 6, 4)


def test_a_second_checkout_reads_the_answers_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mypy build was 14-19 s of every cold scan, rerun after every NRA
    merge that re-keyed the caches although the scanned files had not changed
    (11-08)."""

    first, first_span = _checkout(tmp_path, "first")
    second, second_span = _checkout(tmp_path, "second")
    with ActiveScanCache.opened(tmp_path / "cache"):
        built = SourceFiles(frozenset((first_span.file_path,))).receiver_types(
            (first_span,)
        )
        builds: list[object] = []
        started = TypeCheckerSession.started
        monkeypatch.setattr(
            TypeCheckerSession,
            "started",
            staticmethod(lambda sources: builds.append(sources) or started(sources)),
        )
        read = SourceFiles(frozenset((second_span.file_path,))).receiver_types(
            (second_span,)
        )

    assert isinstance(built[first_span], InferredReceiverType)
    assert read == {second_span: built[first_span]}
    assert builds == []


def test_an_edited_file_is_answered_by_a_new_build(tmp_path: Path) -> None:
    root, span = _checkout(tmp_path, "first")
    sources = SourceFiles(frozenset((span.file_path,)))
    with ActiveScanCache.opened(tmp_path / "cache"):
        before = sources.answer_key((span,))
        first = sources.receiver_types((span,))[span]
        Path(span.file_path).write_text(_SOURCE.replace("made = Goal()", "made = 3"))
        after = sources.answer_key((span,))
        edited = sources.receiver_types((span,))[span]

    assert before != after
    assert isinstance(first, InferredReceiverType) and first.classes == ("owner.Goal",)
    assert isinstance(edited, InferredReceiverType) and edited.classes == ("builtins.int",)
