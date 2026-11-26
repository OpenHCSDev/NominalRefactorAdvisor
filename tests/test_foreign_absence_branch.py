"""Tests for pushing scanned-class foreign absence branches onto their owner class."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    codemod_plan_from_findings,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._relocations import (
    OwnStateBranchAnswerDetector,
)
from nominal_refactor_advisor.detectors._restated_facts import (
    ForeignAbsenceProbeDetector,
)
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.owner_picks import OwnerPick, OwnerPicks
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _scan(root: Path):
    modules = parse_python_modules(
        root,
        use_parse_cache=False,
        source_policy=PythonSourcePathPolicy(include_tests=False),
    )
    snap = CodemodSourceSnapshot.from_modules(modules)
    with ScanCache.scope():
        findings = ForeignAbsenceProbeDetector().detect(modules, DetectorConfig())
    return modules, snap, findings


def _apply_one(finding, snap: CodemodSourceSnapshot) -> None:
    evaluation = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(
        finding, snap
    )
    assert evaluation.rejection_reason == ""
    sim = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snap)
    assert sim.is_clean
    sim.apply()


def test_whole_function_early_return_and_guard_push_cross_module_with_parity(
    tmp_path: Path,
) -> None:
    """Whole-function early-return and compound `and` guard probes on scanned classes
    synthesize directly as executable recipes, push onto the owner class, clear the
    finding, introduce no `OWN_STATE_BRANCH_ANSWER`, and preserve runtime parity."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/models.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Budget:\n"
                "    token_limit: int | None\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Snapshot:\n"
                "    attempt: int | None\n"
            ),
            "pkg/service.py": (
                "from pkg.models import Budget, Snapshot\n\n\n"
                "def char_ceiling(budget: Budget, chars_per_token: int) -> int:\n"
                "    if budget.token_limit is None:\n"
                "        return 0\n"
                "    return budget.token_limit * chars_per_token\n\n\n"
                "def require_deferral(snapshot: Snapshot, authorized: bool) -> None:\n"
                "    if snapshot.attempt is not None and not authorized:\n"
                "        raise ValueError('unauthorized')\n"
            ),
        },
    )

    _mods, snap, findings = _scan(tmp_path)
    by_symbol = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_symbol) == {"char_ceiling", "require_deferral"}

    for sym in ("char_ceiling", "require_deferral"):
        _mods, snap, cur = _scan(tmp_path)
        finding = next(f for f in cur if f.primary_evidence.symbol == sym)
        _apply_one(finding, snap)

    after_models = (tmp_path / "pkg/models.py").read_text()
    after_service = (tmp_path / "pkg/service.py").read_text()
    assert "def char_ceiling(self, chars_per_token: int) -> int:" in after_models
    assert "def require_deferral(self, authorized: bool) -> None:" in after_models
    assert "return budget.char_ceiling(chars_per_token)" in after_service
    assert "snapshot.require_deferral(authorized)" in after_service

    # Re-scan: zero foreign absence probes and zero own-state branch answers.
    after_mods, _after_snap, remaining = _scan(tmp_path)
    assert remaining == []
    with ScanCache.scope():
        assert OwnStateBranchAnswerDetector().detect(after_mods, DetectorConfig()) == []

    # Runtime execution parity.
    sys.path.insert(0, str(tmp_path))
    try:
        for mod_name in [m for m in sys.modules if m == "pkg" or m.startswith("pkg.")]:
            del sys.modules[mod_name]
        import pkg.models as models  # type: ignore[import-not-found]
        import pkg.service as service  # type: ignore[import-not-found]

        assert service.char_ceiling(models.Budget(None), 4) == 0
        assert service.char_ceiling(models.Budget(10), 4) == 40
        service.require_deferral(models.Snapshot(None), False)
        service.require_deferral(models.Snapshot(1), True)
        try:
            service.require_deferral(models.Snapshot(1), False)
        except ValueError as err:
            assert "unauthorized" in str(err)
        else:
            raise AssertionError("expected ValueError")
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in [m for m in sys.modules if m == "pkg" or m.startswith("pkg.")]:
            del sys.modules[mod_name]


def test_embedded_void_ternary_and_loop_continue_use_owner_picks_with_parity(
    tmp_path: Path,
) -> None:
    """Embedded void `if`, non-boolean ternary `IfExp`, and loop `continue` probes stop at
    `NeededInput.OWNER_PICK` and apply cleanly once `--codemod-owner-picks` supplies names."""

    _write(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/work.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Event:\n"
                "    payload: str | None\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Row:\n"
                "    sequence: int | None\n\n\n"
                "def record_event(event: Event, sink: list[str]) -> int:\n"
                "    before = len(sink)\n"
                "    if event.payload is not None:\n"
                "        sink.append(event.payload)\n"
                "    return len(sink) - before\n\n\n"
                "def classify_row(row: Row) -> str:\n"
                "    label = 'missing' if row.sequence is None else f'seq:{row.sequence}'\n"
                "    return label\n\n\n"
                "def collect_present(events: list[Event]) -> list[str]:\n"
                "    out: list[str] = []\n"
                "    for event in events:\n"
                "        if event.payload is None:\n"
                "            continue\n"
                "        out.append(event.payload.upper())\n"
                "    return out\n"
            ),
        },
    )

    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"record_event", "classify_row", "collect_present"}

    # Without picks, all three stop cleanly at OWNER_PICK.
    for sym in ("record_event", "classify_row", "collect_present"):
        ev = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(
            by_sym[sym], snap
        )
        assert ev.needs == (NeededInput.OWNER_PICK,), (sym, ev.rejection_reason, ev.needs)

    # Apply each with an OwnerPicks entry.
    pick_specs = {
        "record_event": ("append_payload", "Append payload to sink when present."),
        "classify_row": ("sequence_label", "Format sequence label."),
        "collect_present": ("append_upper_payload", "Append uppercased payload when present."),
    }
    for sym, (name, judgement) in pick_specs.items():
        _mods, snap, cur = _scan(tmp_path)
        finding = next(f for f in cur if f.primary_evidence.symbol == sym)
        picks_path = tmp_path / f"{sym}.picks.json"
        picks_path.write_text(
            json.dumps(
                {
                    "picks": [
                        {
                            "finding": finding.stable_id,
                            "owner": sym,
                            "because": "test pick",
                            "name": name,
                            "judgement": judgement,
                        }
                    ]
                }
            )
        )
        picked_findings, report = OwnerPicks.read(picks_path).applied(cur)
        assert not report.unmatched
        picked = next(f for f in picked_findings if f.primary_evidence.symbol == sym)
        _apply_one(picked, snap)

    after_source = (tmp_path / "app/work.py").read_text()
    assert "def append_payload(self, sink: list[str]) -> None:" in after_source
    assert "def sequence_label(self):" in after_source
    assert "def append_upper_payload(self, out: list[str]) -> None:" in after_source
    assert "event.append_payload(sink)" in after_source
    assert "row.sequence_label()" in after_source
    assert "event.append_upper_payload(out)" in after_source

    _after_mods, _after_snap, remaining = _scan(tmp_path)
    assert remaining == []

    ns: dict[str, object] = {}
    exec(compile(after_source, str(tmp_path / "app/work.py"), "exec"), ns)
    Event = ns["Event"]
    Row = ns["Row"]
    sink: list[str] = []
    assert ns["record_event"](Event(None), sink) == 0  # type: ignore[operator]
    assert ns["record_event"](Event("hi"), sink) == 1  # type: ignore[operator]
    assert sink == ["hi"]
    assert ns["classify_row"](Row(None)) == "missing"  # type: ignore[operator]
    assert ns["classify_row"](Row(7)) == "seq:7"  # type: ignore[operator]
    assert ns["collect_present"]([Event(None), Event("a"), Event("b")]) == ["A", "B"]  # type: ignore[operator]


def test_soundness_refusals_for_bool_predicates_escaping_locals_protocols_and_collisions(
    tmp_path: Path,
) -> None:
    """Refuses bare boolean predicates, escaping outer local assignments, Protocol owners,
    and method-name collisions without producing dirty edits."""

    _write(
        tmp_path,
        {
            "bad/__init__.py": "",
            "bad/mod.py": (
                "from dataclasses import dataclass\n"
                "from typing import Protocol\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Box:\n"
                "    value: int | None\n\n"
                "    def existing(self) -> int:\n"
                "        return 1\n\n\n"
                "class BoxProto(Protocol):\n"
                "    value: int | None\n\n\n"
                "def is_filled(box: Box) -> bool:\n"
                "    return box.value is not None\n\n\n"
                "def bool_ternary(box: Box) -> bool:\n"
                "    return True if box.value is not None else False\n\n\n"
                "def mutates_outer_local(box: Box) -> tuple[int, bool]:\n"
                "    total = 0\n"
                "    seen = False\n"
                "    if box.value is not None:\n"
                "        total = box.value\n"
                "        seen = True\n"
                "    return total, seen\n\n\n"
                "def proto_consumer(box: BoxProto) -> int:\n"
                "    if box.value is None:\n"
                "        return 0\n"
                "    return box.value\n\n\n"
                "def existing(box: Box) -> int:\n"
                "    if box.value is None:\n"
                "        return 0\n"
                "    return box.value\n\n\n"
                "def nested_short_circuit_guard(ready: bool, flag: bool, box: Box | None) -> None:\n"
                "    if ready and (flag or box.value is not None):\n"
                "        raise ValueError('boom')\n\n\n"
                "def zero_one_tuple_ternary(box: Box) -> tuple[int, ...]:\n"
                "    return (box.value,) if box.value is not None else ()\n"
            ),
        },
    )

    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {
        "is_filled",
        "bool_ternary",
        "mutates_outer_local",
        "proto_consumer",
        "existing",
        "nested_short_circuit_guard",
        "zero_one_tuple_ternary",
    }

    det = ForeignAbsenceProbeDetector()
    for sym in (
        "is_filled",
        "bool_ternary",
        "mutates_outer_local",
        "proto_consumer",
        "nested_short_circuit_guard",
        "zero_one_tuple_ternary",
    ):
        ev = det.evaluate_recipe_for_finding(by_sym[sym], snap)
        assert ev.rejection_reason != "", f"{sym} must be refused"
        assert NeededInput.OWNER_PICK not in ev.needs, (sym, ev.rejection_reason)

    collision_ev = det.evaluate_recipe_for_finding(by_sym["existing"], snap)
    assert "already declares `existing`" in collision_ev.rejection_reason
    assert collision_ev.needs == (NeededInput.OWNER_PICK,)


def test_pushes_prefix_guarded_or_and_multi_return_and_multi_continue_absence_branches(
    tmp_path: Path,
) -> None:
    """Pushes prefix-guarded `or`/`and` absence checks (preserving exact short-circuiting,
    including `box is None or box.value is None`), multi-early-return void functions,
    and multi-`continue` loop tails cleanly with runtime parity."""

    _write(
        tmp_path,
        {
            "shapes/__init__.py": "",
            "shapes/work.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Box:\n"
                "    value: int | None\n"
                "    tag: str\n\n\n"
                "def nullable_or_value(box: Box | None) -> int:\n"
                "    if box is None or box.value is None:\n"
                "        return 0\n"
                "    return box.value + 1\n\n\n"
                "def reject_if_ready_or_present(ready: bool, box: Box) -> None:\n"
                "    if ready or box.value is not None:\n"
                "        raise ValueError('bad')\n\n\n"
                "def record_if_ready_and_present(ready: bool, box: Box, out: list[int]) -> None:\n"
                "    if ready and box.value is not None:\n"
                "        out.append(box.value)\n\n\n"
                "def multi_void_return(box: Box, out: list[int]) -> None:\n"
                "    if box.value is None:\n"
                "        return\n"
                "    if box.tag == 'skip':\n"
                "        return\n"
                "    out.append(box.value)\n\n\n"
                "def multi_continue_loop(boxes: list[Box]) -> list[int]:\n"
                "    out: list[int] = []\n"
                "    for box in boxes:\n"
                "        if box.value is None:\n"
                "            continue\n"
                "        if box.tag == 'skip':\n"
                "            continue\n"
                "        out.append(box.value)\n"
                "    return out\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {
        "nullable_or_value",
        "reject_if_ready_or_present",
        "record_if_ready_and_present",
        "multi_void_return",
        "multi_continue_loop",
    }

    det = ForeignAbsenceProbeDetector()
    # Whole-function void multi-return needs no owner pick; the 4 guarded/loop sites ask for OWNER_PICK.
    whole_ev = det.evaluate_recipe_for_finding(by_sym["multi_void_return"], snap)
    assert whole_ev.rejection_reason == ""
    for sym in (
        "nullable_or_value",
        "reject_if_ready_or_present",
        "record_if_ready_and_present",
        "multi_continue_loop",
    ):
        ev = det.evaluate_recipe_for_finding(by_sym[sym], snap)
        assert ev.needs == (NeededInput.OWNER_PICK,), (sym, ev.rejection_reason)

    pick_specs = (
        ("multi_void_return", None, None),
        (
            "nullable_or_value",
            "incremented_or_zero",
            "Return value + 1 or zero when absent.",
        ),
        (
            "reject_if_ready_or_present",
            "reject_if_present",
            "Raise when value is present.",
        ),
        (
            "record_if_ready_and_present",
            "append_value",
            "Append value when present.",
        ),
        (
            "multi_continue_loop",
            "append_unless_skipped",
            "Append value unless absent or skipped.",
        ),
    )
    for sym, method_name, judgement in pick_specs:
        _mods, snap, cur = _scan(tmp_path)
        finding = next(f for f in cur if f.primary_evidence.symbol == sym)
        if method_name is not None and judgement is not None:
            picks = OwnerPicks(
                "picks.json",
                "",
                (
                    OwnerPick(
                        finding=finding.stable_id,
                        owner=sym,
                        because="test pick",
                        name=method_name,
                        judgement=judgement,
                    ),
                ),
            )
            picked, report = picks.applied(cur)
            assert not report.unmatched
            finding = next(f for f in picked if f.primary_evidence.symbol == sym)
        _apply_one(finding, snap)

    after = (tmp_path / "shapes" / "work.py").read_text()
    _after_mods, _after_snap, remaining = _scan(tmp_path)
    assert remaining == []

    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "shapes/work.py"), "exec"), ns)
    Box = ns["Box"]
    assert ns["nullable_or_value"](None) == 0  # type: ignore[operator]
    assert ns["nullable_or_value"](Box(None, "a")) == 0  # type: ignore[operator]
    assert ns["nullable_or_value"](Box(5, "a")) == 6  # type: ignore[operator]

    try:
        ns["reject_if_ready_or_present"](True, None)  # type: ignore[operator]
        raise AssertionError("expected ValueError on short-circuit prefix")
    except ValueError:
        pass
    try:
        ns["reject_if_ready_or_present"](False, Box(1, "a"))  # type: ignore[operator]
        raise AssertionError("expected ValueError on present box")
    except ValueError:
        pass
    ns["reject_if_ready_or_present"](False, Box(None, "a"))  # type: ignore[operator]

    sink: list[int] = []
    ns["record_if_ready_and_present"](False, None, sink)  # type: ignore[operator]
    ns["record_if_ready_and_present"](True, Box(None, "a"), sink)  # type: ignore[operator]
    ns["record_if_ready_and_present"](True, Box(9, "a"), sink)  # type: ignore[operator]
    assert sink == [9]

    void_sink: list[int] = []
    ns["multi_void_return"](Box(None, "a"), void_sink)  # type: ignore[operator]
    ns["multi_void_return"](Box(3, "skip"), void_sink)  # type: ignore[operator]
    ns["multi_void_return"](Box(4, "ok"), void_sink)  # type: ignore[operator]
    assert void_sink == [4]

    assert ns["multi_continue_loop"]([Box(None, "ok"), Box(2, "skip"), Box(8, "ok")]) == [8]  # type: ignore[operator]


def test_two_subjects_on_same_or_line_replay_cleanly(tmp_path: Path) -> None:
    """When one `if` line probes two distinct subjects (`parent.pos is None or child.after is not None`),
    persisting `subject_source` in `PushForeignAbsenceBranchOperation` lets replay resolve the
    exact probed subject without losing the pick."""

    _write(
        tmp_path,
        {
            "dual/__init__.py": "",
            "dual/pair.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Parent:\n"
                "    pos: int | None\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Child:\n"
                "    after: int | None\n\n\n"
                "def check_pair(parent: Parent, child: Child) -> None:\n"
                "    if parent.pos is None or child.after is not None:\n"
                "        raise ValueError('bad pair')\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    assert len(findings) == 2
    child_finding = next(f for f in findings if f.metrics.subject_source == "child")
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=child_finding.stable_id,
                owner="check_pair",
                because="Child owns rejecting a set after",
                name="require_fresh",
                judgement="Raise when after is already set.",
            ),
        ),
    )
    picked, report = picks.applied(findings)
    assert not report.unmatched
    picked_child = next(f for f in picked if f.stable_id == child_finding.stable_id)
    _apply_one(picked_child, snap)

    after = (tmp_path / "dual" / "pair.py").read_text()
    assert "if parent.pos is None:" in after
    assert "child.require_fresh()" in after
    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "dual/pair.py"), "exec"), ns)
    Parent = ns["Parent"]
    Child = ns["Child"]
    ns["check_pair"](Parent(1), Child(None))  # type: ignore[operator]
    for args in ((Parent(None), Child(None)), (Parent(1), Child(2))):
        try:
            ns["check_pair"](*args)  # type: ignore[operator]
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_cross_module_free_global_constant_is_refused(tmp_path: Path) -> None:
    """Moving a branch across modules that reads an unproved module-level constant
    must be refused instead of dropping the name and raising NameError at runtime."""

    _write(
        tmp_path,
        {
            "cross/__init__.py": "",
            "cross/owner.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Item:\n"
                "    cached: str | None\n"
            ),
            "cross/consumer.py": (
                "from cross.owner import Item\n\n\n"
                "_SENTINEL = '__cached__'\n\n\n"
                "def read_cached(item: Item) -> str:\n"
                "    if item.cached is not None:\n"
                "        return item.cached\n"
                "    return _SENTINEL\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    (finding,) = findings
    ev = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(finding, snap)
    assert "_SENTINEL" in ev.rejection_reason
    assert ev.needs == (NeededInput.SOURCE_CHANGE,)


def test_reuses_existing_identical_owner_method_without_owner_pick_or_duplication(
    tmp_path: Path,
) -> None:
    """When the owner class already declares a method with the exact normalized body
    (including alpha-equivalent parameter names), NRA reuses it as a zero-pick
    executable candidate instead of asking for a pick or creating a duplicate method."""

    _write(
        tmp_path,
        {
            "reuse/__init__.py": "",
            "reuse/flow.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class FlowPrefix:\n"
                "    owner: str\n"
                "    after: int | None\n\n"
                "    def require_fresh_body_interval(self) -> None:\n"
                "        '''Raise when this prefix already carries an after cut.'''\n"
                "        if self.after is not None:\n"
                "            raise ValueError('fresh body required')\n\n"
                "    def append_if_present(self, bucket: list[int]) -> None:\n"
                "        if self.after is not None:\n"
                "            bucket.append(self.after)\n\n\n"
                "def check_entry(prefix: FlowPrefix, expected: str) -> None:\n"
                "    if prefix.owner != expected or prefix.after is not None:\n"
                "        raise ValueError('fresh body required')\n\n\n"
                "def record_after(prefix: FlowPrefix, out: list[int]) -> int:\n"
                "    before = len(out)\n"
                "    if prefix.after is not None:\n"
                "        out.append(prefix.after)\n"
                "    return len(out) - before\n"
            ),
        },
    )

    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"check_entry", "record_after"}

    # Both synthesize with ZERO owner picks and reuse the existing methods!
    for sym in ("check_entry", "record_after"):
        _mods, snap, cur = _scan(tmp_path)
        finding = next(f for f in cur if f.primary_evidence.symbol == sym)
        _apply_one(finding, snap)

    after = (tmp_path / "reuse" / "flow.py").read_text()
    assert after.count("def require_fresh_body_interval") == 1
    assert after.count("def append_if_present") == 1
    assert "prefix.require_fresh_body_interval()" in after
    assert "prefix.append_if_present(out)" in after
    _after_mods, _after_snap, remaining = _scan(tmp_path)
    assert remaining == []

    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "reuse/flow.py"), "exec"), ns)
    FlowPrefix = ns["FlowPrefix"]
    ns["check_entry"](FlowPrefix("a", None), "a")  # type: ignore[operator]
    for args in ((FlowPrefix("b", None), "a"), (FlowPrefix("a", 1), "a")):
        try:
            ns["check_entry"](*args)  # type: ignore[operator]
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    out: list[int] = []
    assert ns["record_after"](FlowPrefix("a", None), out) == 0  # type: ignore[operator]
    assert ns["record_after"](FlowPrefix("a", 7), out) == 1  # type: ignore[operator]
    assert out == [7]


def test_does_not_reuse_existing_method_overridden_by_descendant(
    tmp_path: Path,
) -> None:
    """If a subclass overrides the matching method, calling it could change runtime
    behavior on subclass instances, so NRA must not auto-reuse it."""

    _write(
        tmp_path,
        {
            "sub/__init__.py": "",
            "sub/flow.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class FlowPrefix:\n"
                "    after: int | None\n\n"
                "    def require_fresh(self) -> None:\n"
                "        if self.after is not None:\n"
                "            raise ValueError('fresh required')\n\n\n"
                "@dataclass(frozen=True)\n"
                "class CustomPrefix(FlowPrefix):\n"
                "    def require_fresh(self) -> None:\n"
                "        return None\n\n\n"
                "def check(prefix: FlowPrefix, flag: bool) -> None:\n"
                "    if flag:\n"
                "        return\n"
                "    if prefix.after is not None:\n"
                "        raise ValueError('fresh required')\n"
            ),
        },
    )

    _mods, snap, findings = _scan(tmp_path)
    (finding,) = findings
    ev = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(finding, snap)
    assert ev.recipe is None
    assert ev.needs == (NeededInput.OWNER_PICK,)


def test_two_foreign_absence_branches_pushing_distinct_methods_onto_same_owner_compose_in_one_batch(
    tmp_path: Path,
) -> None:
    """Two distinct foreign absence pushes targeting the same owner class (exact
    `CachedSourceFileSignature` shape) coalesce commutatively in one batch, while
    two pushes declaring the same method name with different bodies still conflict."""

    _write(
        tmp_path,
        {
            "batch/__init__.py": "",
            "batch/cache.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Sig:\n"
                "    semantic_hash: str | None\n\n"
                "    def compute(self) -> str:\n"
                "        return 'computed'\n\n\n"
                "def resolve_hash(sig: Sig) -> str:\n"
                "    if sig.semantic_hash is not None:\n"
                "        return sig.semantic_hash\n"
                "    return sig.compute()\n\n\n"
                "def collect_pending(sigs: list[Sig]) -> list[Sig]:\n"
                "    pending: list[Sig] = []\n"
                "    for sig in sigs:\n"
                "        if sig.semantic_hash is None:\n"
                "            pending.append(sig)\n"
                "    return pending\n"
            ),
        },
    )

    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"resolve_hash", "collect_pending"}

    # Soundness: two distinct bodies claiming the SAME method name on Sig must conflict.
    colliding_picks = OwnerPicks(
        "collide.json",
        "",
        (
            OwnerPick(
                finding=by_sym["resolve_hash"].stable_id,
                owner="resolve_hash",
                because="owner",
                name="same_name",
                judgement="Resolve hash.",
            ),
            OwnerPick(
                finding=by_sym["collect_pending"].stable_id,
                owner="collect_pending",
                because="owner",
                name="same_name",
                judgement="Collect pending.",
            ),
        ),
    )
    colliding_findings, _ = colliding_picks.applied(tuple(findings))
    colliding_plan = codemod_plan_from_findings(colliding_findings, selector_context=snap)
    assert colliding_plan.candidate_count == 0

    # Distinct method names on Sig compose commutatively in ONE batch.
    picks = OwnerPicks(
        "batch.json",
        "",
        (
            OwnerPick(
                finding=by_sym["resolve_hash"].stable_id,
                owner="resolve_hash",
                because="owner",
                name="resolved_hash",
                judgement="Return cached hash or compute it.",
            ),
            OwnerPick(
                finding=by_sym["collect_pending"].stable_id,
                owner="collect_pending",
                because="owner",
                name="append_if_pending",
                judgement="Append when hash is missing.",
            ),
        ),
    )
    picked_findings, report = picks.applied(tuple(findings))
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked_findings, selector_context=snap)
    assert plan.candidate_count == 2
    sim = plan.document.simulate(snap)
    assert sim.is_clean
    sim.apply()

    after = (tmp_path / "batch" / "cache.py").read_text()
    # `batch/cache.py` omits `from __future__ import annotations`, so self-referential
    # `pending: list[Sig]` inside `class Sig` is omitted to avoid a runtime NameError.
    assert "def append_if_pending(self, pending) -> None:" in after
    assert "def resolved_hash(self) -> str:" in after
    assert "return sig.resolved_hash()" in after
    assert "sig.append_if_pending(pending)" in after
    _after_mods, _after_snap, remaining = _scan(tmp_path)
    assert remaining == []

    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "batch/cache.py"), "exec"), ns)
    Sig = ns["Sig"]
    assert ns["resolve_hash"](Sig("abc")) == "abc"  # type: ignore[operator]
    assert ns["resolve_hash"](Sig(None)) == "computed"  # type: ignore[operator]
    assert ns["collect_pending"]([Sig("abc"), Sig(None)]) == [Sig(None)]  # type: ignore[operator]


def test_partial_return_if_without_else_and_nested_function_foreign_absence_push(
    tmp_path: Path,
) -> None:
    """Partial-return `if box.cached is not None:` without `else:` followed by a
    terminating tail, and a foreign absence probe inside a nested helper capturing
    enclosing locals, both push cleanly onto their scanned owner class."""

    _write(
        tmp_path,
        {
            "ext/__init__.py": "",
            "ext/boxes.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Box:\n"
                "    cached: int | None\n\n\n"
                "def read_positive(box: Box, default: int) -> int:\n"
                "    if box.cached is not None:\n"
                "        if box.cached > 0:\n"
                "            return box.cached\n"
                "    return default\n\n\n"
                "class Collector:\n"
                "    def __init__(self, scale: int) -> None:\n"
                "        self.scale = scale\n\n"
                "    def collect(self, offset: int, boxes: list[Box]) -> list[int]:\n"
                "        out: list[int] = []\n"
                "        def record_box(box: Box) -> None:\n"
                "            if box.cached is not None:\n"
                "                out.append(box.cached * self.scale + offset)\n"
                "        for box in boxes:\n"
                "            record_box(box)\n"
                "        return out\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"read_positive", "Collector.collect.record_box"}
    plan = codemod_plan_from_findings(findings, selector_context=snap)
    assert plan.candidate_count == 2
    sim = plan.document.simulate(snap)
    assert sim.is_clean
    sim.apply()

    after = (tmp_path / "ext" / "boxes.py").read_text()
    assert "box.cached" not in after
    assert "return box.read_positive(default)" in after
    assert "box.record_box(out, offset, self)" in after
    _after_mods, _after_snap, remaining = _scan(tmp_path)
    assert remaining == []

    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "ext/boxes.py"), "exec"), ns)
    Box = ns["Box"]
    assert ns["read_positive"](Box(5), -1) == 5  # type: ignore[operator]
    assert ns["read_positive"](Box(0), -1) == -1  # type: ignore[operator]
    assert ns["read_positive"](Box(None), -1) == -1  # type: ignore[operator]
    collector = ns["Collector"](10)  # type: ignore[operator]
    assert collector.collect(1, [Box(2), Box(None), Box(4)]) == [21, 41]


def test_preserved_owner_annotation_unlocks_follow_up_foreign_absence_push(
    tmp_path: Path,
) -> None:
    """Preserving `owner: EnclosingClass` on a same-module pushed method keeps
    `owner.cached is None` resolvable for a subsequent foreign-absence-probe push."""

    _write(
        tmp_path,
        {
            "chain/__init__.py": "",
            "chain/work.py": (
                "from __future__ import annotations\n"
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Box:\n"
                "    value: int | None\n\n\n"
                "class Holder:\n"
                "    def __init__(self, fallback: int | None) -> None:\n"
                "        self.fallback = fallback\n\n"
                "    def resolve(self, box: Box) -> int:\n"
                "        if box.value is not None:\n"
                "            return box.value\n"
                "        if self.fallback is not None:\n"
                "            return self.fallback\n"
                "        return 0\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    assert [f.primary_evidence.symbol for f in findings] == ["Holder.resolve"]
    _apply_one(findings[0], snap)

    after_first = (tmp_path / "chain" / "work.py").read_text()
    assert "def resolve(self, owner: Holder) -> int:" in after_first

    # Follow-up scan sees `owner.fallback is not None` inside `Box.resolve` with
    # `owner: Holder` already typed and pushes it straight back onto `Holder`.
    _mods2, snap2, findings2 = _scan(tmp_path)
    assert [f.primary_evidence.symbol for f in findings2] == ["Box.resolve"]
    ev2 = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(findings2[0], snap2)
    assert ev2.needs == (NeededInput.OWNER_PICK,)


def test_prefix_guarded_boolop_does_not_steal_earlier_line_probe(
    tmp_path: Path,
) -> None:
    """In `if self.load is None or self.load.span is None or store.span is None:`,
    targeting `store.span is None` keeps the earlier `self.load` prefix guard instead
    of breaking on `self.load.span` and failing."""

    _write(
        tmp_path,
        {
            "win/__init__.py": "",
            "win/store.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Site:\n"
                "    span: tuple[int, int] | None\n\n\n"
                "class Window:\n"
                "    def __init__(self, load: Site | None) -> None:\n"
                "        self.load = load\n\n"
                "    def require_store(self, store: Site) -> None:\n"
                "        if (\n"
                "            self.load is None\n"
                "            or self.load.span is None\n"
                "            or store.span is None\n"
                "        ):\n"
                "            raise ValueError('missing span')\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    (store_finding,) = [
        f for f in findings if f.metrics.subject_source == "store"
    ]
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=store_finding.stable_id,
                owner="Window.require_store",
                because="site owns its span check",
                name="require_span",
                judgement="Raise when this site lacks a source span.",
            ),
        ),
    )
    picked, report = picks.applied((store_finding,))
    assert not report.unmatched
    _apply_one(picked[0], snap)

    after = (tmp_path / "win" / "store.py").read_text()
    assert "if self.load is None or self.load.span is None:" in after
    assert "store.require_span()" in after
    assert "def require_span(self) -> None:" in after

    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "win/store.py"), "exec"), ns)
    Site = ns["Site"]
    Window = ns["Window"]
    Window(Site((0, 1))).require_store(Site((2, 3)))  # type: ignore[operator]
    for load, store in (
        (None, Site((2, 3))),
        (Site(None), Site((2, 3))),
        (Site((0, 1)), Site(None)),
    ):
        try:
            Window(load).require_store(store)  # type: ignore[operator]
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_terminal_elif_and_returning_elif_else_push_onto_owner(
    tmp_path: Path,
) -> None:
    """Terminal raising `elif` and returning `elif ... else:` rewrite to `else:`
    calling the pushed owner method with runtime parity."""

    _write(
        tmp_path,
        {
            "el/__init__.py": "",
            "el/intervals.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Interval:\n"
                "    after: int | None\n\n\n"
                "def check_interval(present: bool, interval: Interval) -> None:\n"
                "    if present:\n"
                "        return\n"
                "    elif interval.after is not None:\n"
                "        raise ValueError('must be initial')\n\n\n"
                "def resolve_cut(override: int | None, interval: Interval) -> int:\n"
                "    if override is not None:\n"
                "        return override\n"
                "    elif interval.after is not None:\n"
                "        return interval.after\n"
                "    else:\n"
                "        return 0\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"check_interval", "resolve_cut"}

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=by_sym["check_interval"].stable_id,
                owner="check_interval",
                because="interval validates initial state",
                name="require_initial",
                judgement="Raise if this interval has a predecessor.",
            ),
            OwnerPick(
                finding=by_sym["resolve_cut"].stable_id,
                owner="resolve_cut",
                because="interval resolves its cut",
                name="cut_or_zero",
                judgement="Return the cut position or zero.",
            ),
        ),
    )
    picked, report = picks.applied(tuple(findings))
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked, selector_context=snap)
    assert plan.candidate_count == 2
    sim = plan.document.simulate(snap)
    assert sim.is_clean
    sim.apply()

    after = (tmp_path / "el" / "intervals.py").read_text()
    assert "else:\n        interval.require_initial()" in after
    assert "else:\n        return interval.cut_or_zero()" in after
    after_mods, _after_snap, remaining = _scan(tmp_path)
    assert remaining == []
    with ScanCache.scope():
        assert OwnStateBranchAnswerDetector().detect(after_mods, DetectorConfig()) == []

    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "el/intervals.py"), "exec"), ns)
    Interval = ns["Interval"]
    ns["check_interval"](True, Interval(5))  # type: ignore[operator]
    ns["check_interval"](False, Interval(None))  # type: ignore[operator]
    try:
        ns["check_interval"](False, Interval(5))  # type: ignore[operator]
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert ns["resolve_cut"](9, Interval(5)) == 9  # type: ignore[operator]
    assert ns["resolve_cut"](None, Interval(5)) == 5  # type: ignore[operator]
    assert ns["resolve_cut"](None, Interval(None)) == 0  # type: ignore[operator]


def test_prefix_guarded_and_returning_pushes_trivial_tail_and_rejects_complex_tail(
    tmp_path: Path,
) -> None:
    """`if graph is not None and graph.index is None and ...: return None, None`
    followed by a trivial pure `return graph, token` pushes cleanly, while a
    call-bearing fallback tail is refused to avoid duplicating computation."""

    _write(
        tmp_path,
        {
            "pg/__init__.py": "",
            "pg/cache.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Graph:\n"
                "    index: int | None\n\n\n"
                "def compute_fallback(graph: Graph | None) -> int:\n"
                "    return 0 if graph is None else 1\n\n\n"
                "def latest_graph(graph: Graph | None, token: str, current: str) -> tuple[Graph | None, str | None]:\n"
                "    if graph is not None and graph.index is None and token != current:\n"
                "        return None, None\n"
                "    return graph, token\n\n\n"
                "def complex_tail(graph: Graph | None, token: str) -> int:\n"
                "    if graph is not None and graph.index is None and token == 'bad':\n"
                "        return -1\n"
                "    return compute_fallback(graph)\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"latest_graph", "complex_tail"}

    # Complex tail must NOT be duplicated.
    complex_ev = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(
        by_sym["complex_tail"], snap
    )
    assert complex_ev.recipe is None
    assert "not a pushable branch shape" in complex_ev.rejection_reason

    # Trivial pure tail pushes cleanly with OWNER_PICK.
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=by_sym["latest_graph"].stable_id,
                owner="latest_graph",
                because="graph validates predecessor compatibility",
                name="as_predecessor",
                judgement="Return this graph and token if usable as a predecessor.",
            ),
        ),
    )
    picked, report = picks.applied((by_sym["latest_graph"],))
    assert not report.unmatched
    _apply_one(picked[0], snap)

    after = (tmp_path / "pg" / "cache.py").read_text()
    assert (
        "if graph is not None:\n"
        "        return graph.as_predecessor(token, current)\n"
        "    return (graph, token)"
    ) in after
    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "pg/cache.py"), "exec"), ns)
    Graph = ns["Graph"]
    assert ns["latest_graph"](None, "a", "b") == (None, "a")  # type: ignore[operator]
    assert ns["latest_graph"](Graph(None), "a", "b") == (None, None)  # type: ignore[operator]
    assert ns["latest_graph"](Graph(None), "a", "a") == (Graph(None), "a")  # type: ignore[operator]
    assert ns["latest_graph"](Graph(7), "a", "b") == (Graph(7), "a")  # type: ignore[operator]


def test_nested_if_loop_continue_tail_preserves_continue_and_rejects_fallthrough(
    tmp_path: Path,
) -> None:
    """Inside a nested `if` within a `for` loop, an `if ns.sym is None: continue`
    followed by a tail ending in `continue` pushes as `ns.method(...)\\ncontinue`,
    preserving the loop skip; a nested `if` without trailing `continue` stays refused."""

    _write(
        tmp_path,
        {
            "lp/__init__.py": "",
            "lp/scan.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Summary:\n"
                "    sym: str | None\n\n\n"
                "def collect_dirty(items: list[tuple[bool, Summary]]) -> tuple[list[str], list[str]]:\n"
                "    dirty: list[str] = []\n"
                "    fallback: list[str] = []\n"
                "    for fast, ns in items:\n"
                "        if fast:\n"
                "            if ns.sym is None:\n"
                "                continue\n"
                "            if ns.sym.startswith('d'):\n"
                "                dirty.append(ns.sym)\n"
                "            continue\n"
                "        fallback.append(str(ns.sym))\n"
                "    return dirty, fallback\n\n\n"
                "def fallthrough_nested(items: list[tuple[bool, Summary]]) -> list[str]:\n"
                "    seen: list[str] = []\n"
                "    for fast, ns in items:\n"
                "        if fast:\n"
                "            if ns.sym is None:\n"
                "                continue\n"
                "            seen.append(ns.sym)\n"
                "        seen.append('after')\n"
                "    return seen\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"collect_dirty", "fallthrough_nested"}

    # Fallthrough nested `if` MUST stay refused (pushing it would drop `continue` or mis-continue).
    ft_ev = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(
        by_sym["fallthrough_nested"], snap
    )
    assert ft_ev.recipe is None
    assert "not a pushable branch shape" in ft_ev.rejection_reason

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=by_sym["collect_dirty"].stable_id,
                owner="collect_dirty",
                because="summary records dirty symbol when present",
                name="record_dirty",
                judgement="Record this summary symbol if dirty.",
            ),
        ),
    )
    picked, report = picks.applied((by_sym["collect_dirty"],))
    assert not report.unmatched
    _apply_one(picked[0], snap)

    after = (tmp_path / "lp" / "scan.py").read_text()
    assert "ns.record_dirty(dirty)\n            continue" in after
    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "lp/scan.py"), "exec"), ns)
    Summary = ns["Summary"]
    dirty, fallback = ns["collect_dirty"](  # type: ignore[operator]
        [(True, Summary(None)), (True, Summary("d1")), (True, Summary("ok")), (False, Summary("x"))]
    )
    assert dirty == ["d1"]
    assert fallback == ["x"]


def test_composite_multi_ternary_expression_pushes_and_auto_reuses_on_second_site(
    tmp_path: Path,
) -> None:
    """Two non-boolean `IfExp`s probing the same subject inside one tuple push as
    one method on the owner, and a second function with the same tuple auto-reuses
    that method with zero `OWNER_PICK` and zero `OwnStateBranchAnswer`."""

    _write(
        tmp_path,
        {
            "ct/__init__.py": "",
            "ct/scope.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Scope:\n"
                "    name: str\n"
                "    decl: str | None\n"
                "    final: str | None\n\n\n"
                "def _slice(val: str, prefix: str) -> str:\n"
                "    return f'{prefix}:{val}'\n\n\n"
                "def build_seed_key(scope: Scope, prefix: str) -> tuple[str, str | None, str | None]:\n"
                "    key = (\n"
                "        scope.name,\n"
                "        None if scope.decl is None else _slice(scope.decl, prefix),\n"
                "        None if scope.final is None else _slice(scope.final, prefix),\n"
                "    )\n"
                "    return key\n\n\n"
                "def build_live_key(scope: Scope, prefix: str) -> tuple[str, str | None, str | None]:\n"
                "    key = (\n"
                "        scope.name,\n"
                "        None if scope.decl is None else _slice(scope.decl, prefix),\n"
                "        None if scope.final is None else _slice(scope.final, prefix),\n"
                "    )\n"
                "    return key\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"build_seed_key", "build_live_key"}

    # Push first site with OWNER_PICK.
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=by_sym["build_seed_key"].stable_id,
                owner="build_seed_key",
                because="scope owns its digest key",
                name="digest_key",
                judgement="Build the digest key tuple for this scope.",
            ),
        ),
    )
    picked, report = picks.applied((by_sym["build_seed_key"],))
    assert not report.unmatched
    _apply_one(picked[0], snap)

    # Second site auto-reuses `Scope.digest_key` with ZERO OWNER_PICK!
    _mods2, snap2, remaining2 = _scan(tmp_path)
    assert [f.primary_evidence.symbol for f in remaining2] == ["build_live_key"]
    _apply_one(remaining2[0], snap2)

    after = (tmp_path / "ct" / "scope.py").read_text()
    assert after.count("def digest_key(") == 1
    assert after.count("scope.digest_key(prefix)") == 2
    after_mods, _after_snap, remaining3 = _scan(tmp_path)
    assert remaining3 == []
    with ScanCache.scope():
        assert OwnStateBranchAnswerDetector().detect(after_mods, DetectorConfig()) == []

    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "ct/scope.py"), "exec"), ns)
    Scope = ns["Scope"]
    s1 = Scope("m", "a", None)
    s2 = Scope("m", None, "b")
    assert ns["build_seed_key"](s1, "p") == ("m", "p:a", None)  # type: ignore[operator]
    assert ns["build_live_key"](s2, "p") == ("m", None, "p:b")  # type: ignore[operator]


def test_comprehension_and_outside_walrus_scopes_do_not_trigger_false_escape(
    tmp_path: Path,
) -> None:
    """Comprehension loop targets inside an absence branch and freshly walrus-bound
    variables in a later comprehension must not trigger false escaping-local refusals."""

    _write(
        tmp_path,
        {
            "sc/__init__.py": "",
            "sc/scopes.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Graph:\n"
                "    paths: tuple[str, ...] | None\n\n\n"
                "@dataclass(frozen=True)\n"
                "class History:\n"
                "    revision: str | None\n"
                "    label: str\n\n\n"
                "def collect_paths(graph: Graph, extra: list[str]) -> list[str]:\n"
                "    out: list[str] = []\n"
                "    for file_path in extra:\n"
                "        out.append(file_path)\n"
                "    if graph.paths is not None:\n"
                "        out.extend(file_path for file_path in graph.paths if file_path)\n"
                "    return out\n\n\n"
                "def retain_histories(items: list[tuple[str, History]]) -> dict[str, str]:\n"
                "    saved: dict[str, str] = {}\n"
                "    for key, history in items:\n"
                "        if history.revision is None:\n"
                "            continue\n"
                "        successor = f'{history.label}:{history.revision}'\n"
                "        saved[key] = successor\n"
                "    return {\n"
                "        key: successor\n"
                "        for key, _history in items\n"
                "        if (successor := saved.get(key)) is not None\n"
                "        and len(successor) > 1\n"
                "    }\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"collect_paths", "retain_histories"}

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=by_sym["collect_paths"].stable_id,
                owner="collect_paths",
                because="graph appends its own paths",
                name="extend_paths",
                judgement="Append non-empty paths owned by this graph.",
            ),
            OwnerPick(
                finding=by_sym["retain_histories"].stable_id,
                owner="retain_histories",
                because="history records itself when revision is present",
                name="record_if_present",
                judgement="Record this history under key when it has a revision.",
            ),
        ),
    )
    picked, report = picks.applied(tuple(findings))
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked, selector_context=snap)
    assert plan.candidate_count == 2
    sim = plan.document.simulate(snap)
    assert sim.is_clean
    sim.apply()

    after = (tmp_path / "sc" / "scopes.py").read_text()
    assert "graph.extend_paths(out)" in after
    assert "history.record_if_present(saved, key)" in after
    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "sc/scopes.py"), "exec"), ns)
    Graph = ns["Graph"]
    History = ns["History"]
    assert ns["collect_paths"](Graph(("a.py", "", "b.py")), ["e.py"]) == ["e.py", "a.py", "b.py"]  # type: ignore[operator]
    assert ns["collect_paths"](Graph(None), ["e.py"]) == ["e.py"]  # type: ignore[operator]
    assert ns["retain_histories"]([("k1", History("r1", "L")), ("k2", History(None, "X"))]) == {"k1": "L:r1"}  # type: ignore[operator]


def test_assigning_absence_branch_sites_push_with_runtime_parity(
    tmp_path: Path,
) -> None:
    """Single-target definite-assign `if/elif/else`, subject self-rebind, conditional
    outer-local rebind, prefix-guarded assign, and `AugAssign` push cleanly onto the
    owner while unbound conditional assigns and multi-local escapes stay refused."""

    _write(
        tmp_path,
        {
            "asgn/__init__.py": "",
            "asgn/branches.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Context:\n"
                "    index: int | None\n"
                "    fallback: int = 0\n\n"
                "    def materialized(self) -> 'Context':\n"
                "        return Context(index=self.fallback, fallback=self.fallback)\n\n\n"
                "def ensure_index(context: Context) -> int:\n"
                "    if context.index is None:\n"
                "        context = context.materialized()\n"
                "    return context.fallback + (context.index or 0)\n\n\n"
                "def select_label(context: Context, flag: bool) -> str:\n"
                "    if context.index is not None:\n"
                "        label = f'idx:{context.index}'\n"
                "    elif flag:\n"
                "        label = 'flagged'\n"
                "    else:\n"
                "        label = 'plain'\n"
                "    return label\n\n\n"
                "def update_selectors(context: Context) -> tuple[str, ...]:\n"
                "    selectors = ('base',)\n"
                "    if context.index is not None:\n"
                "        selectors = ('base', f'n:{context.index}')\n"
                "    return selectors\n\n\n"
                "def guarded_rebind(context: Context, enabled: bool) -> list[int]:\n"
                "    items = [1]\n"
                "    if enabled and context.index is None:\n"
                "        items = [*items, context.fallback]\n"
                "    return items\n\n\n"
                "def append_detail(context: Context, detail: str, log: list[str]) -> tuple[str, list[str]]:\n"
                "    if context.index is not None:\n"
                "        detail += f':{context.index}'\n"
                "    return detail, log\n\n\n"
                "def append_items_inplace(context: Context, items: list[int]) -> list[int]:\n"
                "    if context.index is not None:\n"
                "        items += [context.index]\n"
                "    return items\n\n\n"
                "def unbound_conditional(context: Context) -> int:\n"
                "    if context.index is not None:\n"
                "        bound = context.index\n"
                "    return bound\n\n\n"
                "def multi_escape(context: Context) -> tuple[int, int]:\n"
                "    a, b = 0, 0\n"
                "    if context.index is not None:\n"
                "        a = context.index\n"
                "        b = context.index + 1\n"
                "    return a, b\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {
        "ensure_index",
        "select_label",
        "update_selectors",
        "guarded_rebind",
        "append_detail",
        "append_items_inplace",
        "unbound_conditional",
        "multi_escape",
    }

    # Unbound conditional assign and multi-local escape MUST stay refused.
    for refused_sym in ("unbound_conditional", "multi_escape"):
        ev = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(
            by_sym[refused_sym], snap
        )
        assert ev.recipe is None
        assert "assigns outer local" in ev.rejection_reason

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=by_sym["ensure_index"].stable_id,
                owner="ensure_index",
                because="context materializes itself when index is absent",
                name="with_index",
                judgement="Return this context materialized if its index is absent.",
            ),
            OwnerPick(
                finding=by_sym["select_label"].stable_id,
                owner="select_label",
                because="context owns its label choice",
                name="label_for",
                judgement="Compute the display label for this context.",
            ),
            OwnerPick(
                finding=by_sym["update_selectors"].stable_id,
                owner="update_selectors",
                because="context extends selectors when indexed",
                name="selectors_with",
                judgement="Return updated selectors for this context.",
            ),
            OwnerPick(
                finding=by_sym["guarded_rebind"].stable_id,
                owner="guarded_rebind",
                because="context appends fallback when index is absent",
                name="items_with_fallback",
                judgement="Return items extended with fallback when index is absent.",
            ),
            OwnerPick(
                finding=by_sym["append_detail"].stable_id,
                owner="append_detail",
                because="context formats its index suffix onto detail",
                name="extend_detail",
                judgement="Append index suffix to detail when present.",
            ),
            OwnerPick(
                finding=by_sym["append_items_inplace"].stable_id,
                owner="append_items_inplace",
                because="context appends its index onto items",
                name="extend_items",
                judgement="Append index onto items in place when present.",
            ),
        ),
    )
    pushable = tuple(
        by_sym[sym]
        for sym in (
            "ensure_index",
            "select_label",
            "update_selectors",
            "guarded_rebind",
            "append_detail",
            "append_items_inplace",
        )
    )
    picked, report = picks.applied(pushable)
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked, selector_context=snap)
    assert plan.candidate_count == 6
    sim = plan.document.simulate(snap)
    assert sim.is_clean
    sim.apply()

    after = (tmp_path / "asgn" / "branches.py").read_text()
    assert "context = context.with_index()" in after
    assert "def with_index(self)" in after
    assert "label = context.label_for(flag)" in after
    assert "selectors = context.selectors_with(selectors)" in after
    assert "if enabled:\n        items = context.items_with_fallback(items)" in after
    assert "detail = context.extend_detail(detail)" in after
    assert "items = context.extend_items(items)" in after

    ns: dict[str, object] = {}
    exec(compile(after, str(tmp_path / "asgn/branches.py"), "exec"), ns)
    Context = ns["Context"]
    assert ns["ensure_index"](Context(None, 5)) == 10  # type: ignore[operator]
    assert ns["ensure_index"](Context(3, 5)) == 8  # type: ignore[operator]
    assert ns["select_label"](Context(4), False) == "idx:4"  # type: ignore[operator]
    assert ns["select_label"](Context(None), True) == "flagged"  # type: ignore[operator]
    assert ns["select_label"](Context(None), False) == "plain"  # type: ignore[operator]
    assert ns["update_selectors"](Context(2)) == ("base", "n:2")  # type: ignore[operator]
    assert ns["update_selectors"](Context(None)) == ("base",)  # type: ignore[operator]
    assert ns["guarded_rebind"](Context(None, 9), True) == [1, 9]  # type: ignore[operator]
    assert ns["guarded_rebind"](Context(None, 9), False) == [1]  # type: ignore[operator]
    assert ns["guarded_rebind"](Context(3, 9), True) == [1]  # type: ignore[operator]
    assert ns["append_detail"](Context(7), "d", []) == ("d:7", [])  # type: ignore[operator]
    assert ns["append_detail"](Context(None), "d", []) == ("d", [])  # type: ignore[operator]
    shared: list[int] = [10]
    res = ns["append_items_inplace"](Context(42), shared)  # type: ignore[operator]
    assert res is shared and shared == [10, 42]
    assert ns["append_items_inplace"](Context(None), shared) is shared and shared == [10, 42]






