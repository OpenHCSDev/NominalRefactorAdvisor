"""Pushing CONSTANT_SUBTYPE_ANSWER branches onto polymorphic methods of their family."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._relocations import ConstantSubtypeAnswerDetector
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.models import ConsumerMethodPick
from nominal_refactor_advisor.owner_picks import OwnerPick, OwnerPicks
from nominal_refactor_advisor.refactors.constant_subtype_answer import (
    PushConstantSubtypeAnswerOperation,
)
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")


def _findings(root: Path):
    with ScanCache.scope():
        return ConstantSubtypeAnswerDetector().detect(
            parse_python_modules(
                root,
                use_parse_cache=False,
                source_policy=PythonSourcePathPolicy(include_tests=False),
            ),
            DetectorConfig(),
        )


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(
        parse_python_modules(
            root,
            use_parse_cache=False,
            source_policy=PythonSourcePathPolicy(include_tests=False),
        )
    )


def _apply(root: Path, recipe) -> None:
    simulation = recipe.simulate(_snapshot(root))
    assert simulation.is_clean
    simulation.apply()


def test_pushes_in_place_base_method_and_deletes_unread_constant_answer(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/process.py": (
                "class ProcessState:\n"
                "    def allows_owner_start(self) -> bool:\n"
                "        return False\n\n"
                "    def require_owner_startable(self) -> None:\n"
                "        if not self.allows_owner_start():\n"
                "            raise RuntimeError('already running')\n\n\n"
                "class NoProcess(ProcessState):\n"
                "    def allows_owner_start(self) -> bool:\n"
                "        return True\n\n\n"
                "class ProcessIdentity(ProcessState):\n"
                "    def allows_owner_start(self) -> bool:\n"
                "        return False\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    (op,) = recipe.operations
    assert isinstance(op, PushConstantSubtypeAnswerOperation)
    assert len(op.consumers) == 1
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "process.py").read_text()
    assert "allows_owner_start" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    ns["NoProcess"]().require_owner_startable()  # type: ignore[operator]
    for cls_name in ("ProcessState", "ProcessIdentity"):
        try:
            ns[cls_name]().require_owner_startable()  # type: ignore[operator]
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"expected RuntimeError on {cls_name}")
    assert _findings(tmp_path) == []


def test_pushes_multi_consumer_classmethod_and_ternary_and_deletes_answer(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/verdicts.py": (
                "class HeadVerdict:\n"
                "    @classmethod\n"
                "    def allows_integration(cls) -> bool:\n"
                "        return False\n\n"
                "    @classmethod\n"
                "    def require_ok(cls) -> None:\n"
                "        if not cls.allows_integration():\n"
                "            raise ValueError('blocked')\n\n\n"
                "class ReviewedHeadVerdict(HeadVerdict):\n"
                "    @classmethod\n"
                "    def allows_integration(cls) -> bool:\n"
                "        return True\n\n\n"
                "class SubBlockedVerdict(ReviewedHeadVerdict):\n"
                "    @classmethod\n"
                "    def allows_integration(cls) -> bool:\n"
                "        return False\n\n\n"
                "def accepted_sources(verdict: type[HeadVerdict], req: str) -> tuple[str, ...]:\n"
                "    return (req,) if verdict.allows_integration() else ()\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    (op,) = recipe.operations
    assert isinstance(op, PushConstantSubtypeAnswerOperation)
    assert len(op.consumers) == 2
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "verdicts.py").read_text()
    assert "allows_integration" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    ns["ReviewedHeadVerdict"].require_ok()  # type: ignore[attr-defined]
    for cls_name in ("HeadVerdict", "SubBlockedVerdict"):
        try:
            ns[cls_name].require_ok()  # type: ignore[attr-defined]
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError on {cls_name}")
    assert ns["accepted_sources"](ns["ReviewedHeadVerdict"], "r1") == ("r1",)  # type: ignore[operator]
    assert ns["accepted_sources"](ns["HeadVerdict"], "r1") == ()  # type: ignore[operator]
    assert ns["accepted_sources"](ns["SubBlockedVerdict"], "r1") == ()  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_embedded_loop_continue_with_owner_pick_and_keeps_answer_if_other_reader_exists(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/members.py": (
                "from typing import ClassVar\n\n\n"
                "class DestinationMember:\n"
                "    callers_import_destination: ClassVar[bool] = False\n\n"
                "    def __init__(self, name: str) -> None:\n"
                "        self.name = name\n\n\n"
                "class DestinationClassMethod(DestinationMember):\n"
                "    callers_import_destination = True\n\n\n"
                "def collect_imports(members: list[DestinationMember], out: list[str]) -> None:\n"
                "    for member in members:\n"
                "        if not member.callers_import_destination:\n"
                "            continue\n"
                "        out.append(member.name)\n\n\n"
                "def describe(member: DestinationMember) -> bool:\n"
                "    return member.callers_import_destination\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    unpicked = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    assert unpicked.recipe is None
    assert NeededInput.OWNER_PICK in unpicked.needs

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=finding.stable_id,
                owner="collect_imports",
                because="destination members own whether callers import them",
                name="record_required_import",
                judgement="Record this member's name when callers must import its destination.",
            ),
        ),
    )
    picked, report = picks.applied((finding,))
    assert [o.outcome for o in report.outcomes] == ["picked"]
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        picked[0], _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "members.py").read_text()
    assert "member.record_required_import(out)" in after
    assert "@classmethod" not in after
    # Non-branch reader `describe` still reads `callers_import_destination`, so it stays declared.
    assert "callers_import_destination" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    out: list[str] = []
    plain = ns["DestinationMember"]("a")  # type: ignore[operator]
    cm = ns["DestinationClassMethod"]("b")  # type: ignore[operator]
    ns["collect_imports"]([plain, cm], out)  # type: ignore[operator]
    assert out == ["b"]
    assert ns["describe"](plain) is False  # type: ignore[operator]
    assert ns["describe"](cm) is True  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_refuses_in_place_push_when_branch_calls_super(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/base.py": (
                "class Root:\n"
                "    def check(self) -> int:\n"
                "        return 1\n\n\n"
                "class Mid(Root):\n"
                "    allow = False\n\n"
                "    def check(self) -> int:\n"
                "        if self.allow:\n"
                "            return super().check() + 1\n"
                "        return 0\n\n\n"
                "class Leaf(Mid):\n"
                "    allow = True\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    assert evaluation.recipe is None
    assert NeededInput.SOURCE_CHANGE in evaluation.needs


def test_pushes_intermediate_subclass_in_place_and_uninitialized_annotation_base(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/strategy.py": (
                "class Strategy:\n"
                "    allows_projection: bool\n\n"
                "    @classmethod\n"
                "    def require_valid(cls, requested: bool) -> None:\n"
                "        if requested and not cls.allows_projection:\n"
                "            raise ValueError('needs simulate')\n\n\n"
                "class RunStrategy(Strategy):\n"
                "    allows_projection = False\n\n\n"
                "class SimulateStrategy(Strategy):\n"
                "    allows_projection = True\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "strategy.py").read_text()
    assert "allows_projection" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    ns["SimulateStrategy"].require_valid(True)  # type: ignore[attr-defined]
    ns["RunStrategy"].require_valid(False)  # type: ignore[attr-defined]
    try:
        ns["RunStrategy"].require_valid(True)  # type: ignore[attr-defined]
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on RunStrategy(requested=True)")
    assert _findings(tmp_path) == []


def test_pushes_or_else_guard_and_compound_ternary_with_owner_pick(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/channel.py": (
                "class ChannelSelection:\n"
                "    def requires_catalog(self) -> bool:\n"
                "        return False\n\n\n"
                "class NamedChannel(ChannelSelection):\n"
                "    def requires_catalog(self) -> bool:\n"
                "        return True\n\n\n"
                "def load_catalog(channel: ChannelSelection, fallback: bool) -> str:\n"
                "    return 'loaded' if channel.requires_catalog() or fallback else 'none'\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "channel.py").read_text()
    assert "requires_catalog" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["load_catalog"](ns["NamedChannel"](), False) == "loaded"  # type: ignore[operator]
    assert ns["load_catalog"](ns["ChannelSelection"](), True) == "loaded"  # type: ignore[operator]
    assert ns["load_catalog"](ns["ChannelSelection"](), False) == "none"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_prefix_guarded_or_early_return_and_non_first_compound_ternary(tmp_path: Path):
    """`if PREFIX or not subject.flag(): return fallback; ...` and
    `a if (p1 and p2 and subject.flag) else b` push cleanly while preserving
    PREFIX short-circuiting and deleting the dead constant answer."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/roles.py": (
                "from typing import ClassVar\n\n\n"
                "class ThreadRole:\n"
                "    def hears_each_reply(self) -> bool:\n"
                "        return False\n\n\n"
                "class PersonRole(ThreadRole):\n"
                "    def hears_each_reply(self) -> bool:\n"
                "        return True\n\n\n"
                "def format_route(reply_route: str, role: ThreadRole, target: str) -> str:\n"
                "    if not reply_route or not role.hears_each_reply():\n"
                "        return ''\n"
                "    return reply_route + ':' + target\n\n\n"
                "class ViewportRequest:\n"
                "    animates_arrivals: ClassVar[bool] = False\n\n\n"
                "class LiveViewportRequest(ViewportRequest):\n"
                "    animates_arrivals: ClassVar[bool] = True\n\n\n"
                "def select_keys(arrivals: bool, drawn: bool, request: ViewportRequest, keys: list[str]) -> list[str] | None:\n"
                "    chosen = list(keys) if (not arrivals and drawn and request.animates_arrivals) else None\n"
                "    return chosen\n"
            ),
        },
    )

    # 1. `format_route` (`if not reply_route or not role.hears_each_reply(): return ''`)
    for member_name, consumer, method_name in (
        ("hears_each_reply", "format_route", "format_reply_route"),
        ("animates_arrivals", "select_keys", "animated_keys"),
    ):
        findings = _findings(tmp_path)
        target = next(f for f in findings if f.metrics.member_name == member_name)
        unpicked = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            target, _snapshot(tmp_path)
        )
        assert unpicked.recipe is None
        assert NeededInput.OWNER_PICK in unpicked.needs
        picks = OwnerPicks(
            "picks.json",
            "",
            (
                OwnerPick(
                    finding=target.stable_id,
                    owner=consumer,
                    because="family owns its own branch behavior",
                    name=method_name,
                    judgement="Push constant answer branch onto the family.",
                ),
            ),
        )
        picked, _ = picks.applied((target,))
        ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            picked[0], _snapshot(tmp_path)
        )
        _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "roles.py").read_text()
    assert "hears_each_reply" not in after
    assert "animates_arrivals" not in after
    assert "if not reply_route:" in after
    assert "return role.format_reply_route(" in after
    assert "request.animated_keys(" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["format_route"]("", ns["PersonRole"](), "t") == ""  # type: ignore[operator]
    assert ns["format_route"]("r", ns["ThreadRole"](), "t") == ""  # type: ignore[operator]
    assert ns["format_route"]("r", ns["PersonRole"](), "t") == "r:t"  # type: ignore[operator]
    assert ns["select_keys"](False, True, ns["LiveViewportRequest"](), ["a"]) == ["a"]  # type: ignore[operator]
    assert ns["select_keys"](True, True, ns["LiveViewportRequest"](), ["a"]) is None  # type: ignore[operator]
    assert ns["select_keys"](False, True, ns["ViewportRequest"](), ["a"]) is None  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_sum_and_comprehension_filter_answers_with_parity_and_deletes_flags(
    tmp_path: Path,
):
    """`sum(1 for ... if x.flag)` (including inside an f-string) and filtered
    `GeneratorExp`/`ListComp`/`frozenset(...)` comprehensions (including prefix-guarded
    `and`, captured loop keys, and negated `@classmethod` filters) push cleanly onto
    the family without introducing `ZERO_ONE_TUPLE` and delete the dead constant flags."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/filters.py": (
                "from dataclasses import dataclass\n"
                "from typing import ClassVar\n\n\n"
                "class ShapeModel:\n"
                "    bypasses: ClassVar[bool] = False\n\n\n"
                "class BypassedModel(ShapeModel):\n"
                "    bypasses: ClassVar[bool] = True\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Finding:\n"
                "    model: ShapeModel\n\n\n"
                "def summary_note(findings: list[Finding]) -> str:\n"
                "    return f'bypass={sum(1 for f in findings if f.model.bypasses)}'\n\n\n"
                "class State:\n"
                "    def attention(self) -> bool:\n"
                "        return False\n\n\n"
                "class AlertState(State):\n"
                "    def __init__(self, label: str) -> None:\n"
                "        self.label = label\n\n"
                "    def attention(self) -> bool:\n"
                "        return True\n\n\n"
                "def ping_labels(states: list[State]) -> tuple[str, ...]:\n"
                "    return tuple(f'ping:{state.label}' for state in states if state.attention())\n\n\n"
                "class Status:\n"
                "    def is_working(self) -> bool:\n"
                "        return False\n\n\n"
                "class WorkingStatus(Status):\n"
                "    def is_working(self) -> bool:\n"
                "        return True\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Row:\n"
                "    active: bool\n"
                "    status: Status\n\n\n"
                "def working_names(rows: dict[str, Row]) -> frozenset[str]:\n"
                "    return frozenset(\n"
                "        name for name, row in rows.items() if row.active and row.status.is_working()\n"
                "    )\n\n\n"
                "class Detector:\n"
                "    @classmethod\n"
                "    def prefers_in_process(cls) -> bool:\n"
                "        return False\n\n\n"
                "class LocalDetector(Detector):\n"
                "    @classmethod\n"
                "    def prefers_in_process(cls) -> bool:\n"
                "        return True\n\n\n"
                "def pooled_detectors(detectors: list[type[Detector]]) -> list[type[Detector]]:\n"
                "    return [dt for dt in detectors if not dt.prefers_in_process()]\n\n\n"
                "class Attempt:\n"
                "    def __init__(self, owner: str) -> None:\n"
                "        self.owner = owner\n\n"
                "    def counts_as_started(self) -> bool:\n"
                "        return False\n\n\n"
                "class StartedAttempt(Attempt):\n"
                "    def counts_as_started(self) -> bool:\n"
                "        return True\n\n\n"
                "def started_for(rows: dict[str, Attempt], owners: set[str]) -> frozenset[str]:\n"
                "    return frozenset(\n"
                "        key for key, row in rows.items() if row.counts_as_started() and row.owner in owners\n"
                "    )\n"
            ),
        },
    )

    for member_name, consumer, method_name in (
        ("bypasses", "summary_note", "bypass_count"),
        ("attention", "ping_labels", "attention_pings"),
        ("is_working", "working_names", "working_name_entries"),
        ("prefers_in_process", "pooled_detectors", "pooled_entries"),
        ("counts_as_started", "started_for", "started_key_entries"),
    ):
        findings = _findings(tmp_path)
        target = next(f for f in findings if f.metrics.member_name == member_name)
        unpicked = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            target, _snapshot(tmp_path)
        )
        assert unpicked.recipe is None
        assert NeededInput.OWNER_PICK in unpicked.needs, (member_name, unpicked.rejection_reason)
        picks = OwnerPicks(
            "picks.json",
            "",
            (
                OwnerPick(
                    finding=target.stable_id,
                    owner=consumer,
                    because="family owns its own comprehension projection",
                    name=method_name,
                    judgement="Project member contribution for comprehension.",
                ),
            ),
        )
        picked, _ = picks.applied((target,))
        ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            picked[0], _snapshot(tmp_path)
        )
        _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "filters.py").read_text()
    for deleted in ("bypasses", "attention(", "is_working", "prefers_in_process"):
        assert deleted not in after
    assert "sum((f.model.bypass_count() for f in findings))" in after or "sum(f.model.bypass_count() for f in findings)" in after

    ns: dict[str, object] = {}
    exec(after, ns)
    Finding = ns["Finding"]
    ShapeModel = ns["ShapeModel"]
    BypassedModel = ns["BypassedModel"]
    assert ns["summary_note"]([Finding(ShapeModel()), Finding(BypassedModel()), Finding(BypassedModel())]) == "bypass=2"  # type: ignore[operator]
    assert ns["ping_labels"]([ns["State"](), ns["AlertState"]("a"), ns["AlertState"]("b")]) == ("ping:a", "ping:b")  # type: ignore[operator]
    Row = ns["Row"]
    assert ns["working_names"](  # type: ignore[operator]
        {
            "a": Row(True, ns["WorkingStatus"]()),
            "b": Row(False, ns["WorkingStatus"]()),
            "c": Row(True, ns["Status"]()),
        }
    ) == frozenset({"a"})
    assert ns["pooled_detectors"]([ns["Detector"], ns["LocalDetector"]]) == [ns["Detector"]]  # type: ignore[operator]
    Attempt = ns["Attempt"]
    StartedAttempt = ns["StartedAttempt"]
    assert ns["started_for"](  # type: ignore[operator]
        {"k1": StartedAttempt("a"), "k2": StartedAttempt("b"), "k3": Attempt("a")},
        {"a"},
    ) == frozenset({"k1"})
    assert _findings(tmp_path) == []


def test_pushes_call_and_subscript_receivers_in_comprehensions_and_prefix_or(
    tmp_path: Path,
):
    """Non-pure `Call` receivers (when absent from bodies) and pure `Subscript` receivers
    (including when reused in comprehension `elt`, rewritten to `self`) push cleanly while
    preserving single evaluation and short-circuiting; an impure `Call` repeated inside the
    body stays rejected."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/receivers.py": (
                "from dataclasses import dataclass, replace\n"
                "from typing import ClassVar\n\n\n"
                "class Policy:\n"
                "    selected: ClassVar[bool] = False\n\n\n"
                "class SelectedPolicy(Policy):\n"
                "    selected: ClassVar[bool] = True\n\n\n"
                "class Catalog:\n"
                "    def __init__(self, mapping: dict[str, Policy]) -> None:\n"
                "        self.mapping = mapping\n"
                "        self.calls: list[str] = []\n\n"
                "    def policy_for(self, key: str) -> Policy:\n"
                "        self.calls.append(key)\n"
                "        return self.mapping[key]\n\n\n"
                "def selected_keys(catalog: Catalog, keys: list[str]) -> tuple[str, ...]:\n"
                "    return tuple(\n"
                "        key\n"
                "        for key in keys\n"
                "        if catalog.policy_for(\n"
                "            key\n"
                "        ).selected\n"
                "    )\n\n\n"
                "class Progress:\n"
                "    def for_verdict(self) -> bool:\n"
                "        return False\n\n\n"
                "class OpenProgress(Progress):\n"
                "    def for_verdict(self) -> bool:\n"
                "        return True\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Read:\n"
                "    key: str\n"
                "    progress: Progress | None = None\n\n\n"
                "def judgeable(reads: list[Read], progress: dict[str, Progress]) -> tuple[Read, ...]:\n"
                "    return tuple(\n"
                "        replace(read, progress=progress[read.key])\n"
                "        for read in reads\n"
                "        if progress[read.key].for_verdict()\n"
                "    )\n\n\n"
                "class Binding:\n"
                "    removable: ClassVar[bool] = False\n\n\n"
                "class RemovableBinding(Binding):\n"
                "    removable: ClassVar[bool] = True\n\n\n"
                "def first_removable(pairs: list[tuple[Binding, str]]) -> tuple[Binding, str]:\n"
                "    return next((pair for pair in pairs if pair[0].removable), pairs[0])\n\n\n"
                "class ThreadStatus:\n"
                "    def is_stopped(self) -> bool:\n"
                "        return False\n\n\n"
                "class StoppedStatus(ThreadStatus):\n"
                "    def is_stopped(self) -> bool:\n"
                "        return True\n\n\n"
                "class Registry:\n"
                "    def __init__(self, statuses: dict[str, ThreadStatus]) -> None:\n"
                "        self.statuses = statuses\n"
                "        self.lookups: list[str] = []\n\n"
                "    def status(self, name: str) -> ThreadStatus:\n"
                "        self.lookups.append(name)\n"
                "        return self.statuses[name]\n\n\n"
                "def drain(wakes_disabled: bool, reg: Registry, name: str) -> str:\n"
                "    if wakes_disabled or reg.status(name).is_stopped():\n"
                "        return 'held'\n"
                "    return 'run:' + name\n\n\n"
                "class ImpureFlag:\n"
                "    active: ClassVar[bool] = False\n\n\n"
                "class ActiveImpureFlag(ImpureFlag):\n"
                "    active: ClassVar[bool] = True\n\n\n"
                "def make_flag(items: list[ImpureFlag]) -> ImpureFlag:\n"
                "    return items.pop(0)\n\n\n"
                "def impure_double(items: list[ImpureFlag]) -> ImpureFlag | None:\n"
                "    if make_flag(items).active:\n"
                "        return make_flag(items)\n"
                "    return None\n"
            ),
        },
    )

    for member_name, consumer, method_name in (
        ("selected", "selected_keys", "selected_entries"),
        ("for_verdict", "judgeable", "verdict_entries"),
        ("removable", "first_removable", "removable_entries"),
        ("is_stopped", "drain", "drain_outcome"),
    ):
        findings = _findings(tmp_path)
        target = next(f for f in findings if f.metrics.member_name == member_name)
        unpicked = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            target, _snapshot(tmp_path)
        )
        assert unpicked.recipe is None
        assert NeededInput.OWNER_PICK in unpicked.needs, (member_name, unpicked.reason)
        picks = OwnerPicks(
            "picks.json",
            "",
            (
                OwnerPick(
                    finding=target.stable_id,
                    owner=consumer,
                    because="family owns its own receiver dispatch",
                    name=method_name,
                    judgement="Push constant answer branch onto the family.",
                ),
            ),
        )
        picked, _ = picks.applied((target,))
        ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            picked[0], _snapshot(tmp_path)
        )
        _apply(tmp_path, ev.executable_recipe)

    # Impure call repeated inside arm body must remain rejected (SOURCE_CHANGE).
    remaining = _findings(tmp_path)
    assert [f.metrics.member_name for f in remaining] == ["active"]
    impure_eval = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        remaining[0], _snapshot(tmp_path)
    )
    assert impure_eval.recipe is None
    assert NeededInput.SOURCE_CHANGE in impure_eval.needs

    after = (tmp_path / "pkg" / "receivers.py").read_text()
    for deleted in ("selected:", "for_verdict", "removable:", "is_stopped"):
        assert deleted not in after
    assert "replace(read, progress=self)" in after

    ns: dict[str, object] = {}
    exec(after, ns)
    cat = ns["Catalog"]({"a": ns["Policy"](), "b": ns["SelectedPolicy"]()})  # type: ignore[operator]
    assert ns["selected_keys"](cat, ["a", "b"]) == ("b",)  # type: ignore[operator]
    assert cat.calls == ["a", "b"]  # single evaluation per key

    p_closed = ns["Progress"]()  # type: ignore[operator]
    p_open = ns["OpenProgress"]()  # type: ignore[operator]
    Read = ns["Read"]
    reads = [Read("k1"), Read("k2")]  # type: ignore[operator]
    assert ns["judgeable"](reads, {"k1": p_closed, "k2": p_open}) == (Read("k2", p_open),)  # type: ignore[operator]

    b0, b1 = ns["Binding"](), ns["RemovableBinding"]()  # type: ignore[operator]
    assert ns["first_removable"]([(b0, "x"), (b1, "y")]) == (b1, "y")  # type: ignore[operator]
    assert ns["first_removable"]([(b0, "x")]) == (b0, "x")  # type: ignore[operator]

    reg = ns["Registry"]({"a": ns["ThreadStatus"](), "b": ns["StoppedStatus"]()})  # type: ignore[operator]
    assert ns["drain"](True, reg, "a") == "held"  # type: ignore[operator]
    assert reg.lookups == []  # prefix short-circuited before status() call
    assert ns["drain"](False, reg, "b") == "held"  # type: ignore[operator]
    assert ns["drain"](False, reg, "a") == "run:a"  # type: ignore[operator]
    assert reg.lookups == ["b", "a"]


def test_pushes_assigning_and_prefix_guarded_assigning_branches(tmp_path: Path):
    """Single-target assigning branches and prefix-guarded conditional rebinds promote
    via `_upgrade_dispatch_site_to_assigning` and delete the dead constant flags."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/assigning.py": (
                "from typing import ClassVar\n\n\n"
                "class TurnPhase:\n"
                "    def pauses_input_clock(self) -> bool:\n"
                "        return False\n\n\n"
                "class ModelWaitPhase(TurnPhase):\n"
                "    def pauses_input_clock(self) -> bool:\n"
                "        return True\n\n\n"
                "def wait_for(phase: TurnPhase, awaiting_start: bool, model_wait: str, input_wait: str) -> str:\n"
                "    active = model_wait\n"
                "    if awaiting_start and not phase.pauses_input_clock():\n"
                "        active = input_wait\n"
                "    return 'wait:' + active\n\n\n"
                "class Mode:\n"
                "    is_fast: ClassVar[bool] = False\n\n\n"
                "class FastMode(Mode):\n"
                "    is_fast: ClassVar[bool] = True\n\n\n"
                "def compute_budget(mode: Mode, base: int) -> int:\n"
                "    if mode.is_fast:\n"
                "        budget = base // 2\n"
                "    else:\n"
                "        budget = base * 2\n"
                "    return budget + 1\n"
            ),
        },
    )

    for member_name, consumer, method_name in (
        ("pauses_input_clock", "wait_for", "effective_wait"),
        ("is_fast", "compute_budget", "scaled_budget"),
    ):
        findings = _findings(tmp_path)
        target = next(f for f in findings if f.metrics.member_name == member_name)
        unpicked = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            target, _snapshot(tmp_path)
        )
        assert unpicked.recipe is None
        assert NeededInput.OWNER_PICK in unpicked.needs, (member_name, unpicked.reason)
        picks = OwnerPicks(
            "picks.json",
            "",
            (
                OwnerPick(
                    finding=target.stable_id,
                    owner=consumer,
                    because="family owns its assigned value",
                    name=method_name,
                    judgement="Compute the branch value polymorphically.",
                ),
            ),
        )
        picked, _ = picks.applied((target,))
        ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            picked[0], _snapshot(tmp_path)
        )
        _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "assigning.py").read_text()
    assert "pauses_input_clock" not in after
    assert "is_fast" not in after
    assert "active = phase.effective_wait(" in after
    assert "budget = mode.scaled_budget(" in after

    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["wait_for"](ns["TurnPhase"](), True, "m", "i") == "wait:i"  # type: ignore[operator]
    assert ns["wait_for"](ns["TurnPhase"](), False, "m", "i") == "wait:m"  # type: ignore[operator]
    assert ns["wait_for"](ns["ModelWaitPhase"](), True, "m", "i") == "wait:m"  # type: ignore[operator]
    assert ns["compute_budget"](ns["Mode"](), 10) == 21  # type: ignore[operator]
    assert ns["compute_budget"](ns["FastMode"](), 10) == 6  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_unrelated_elif_partial_return_and_prefix_guarded_returning(
    tmp_path: Path,
):
    """Unrelated `elif` chains (positive and negated), partially-returning `if` with
    terminating tail, and `if PREFIX and x.flag [and SUFFIX]: return ...; return <trivial>`
    push cleanly with full runtime parity and delete the dead constant flags."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/branches.py": (
                "from typing import ClassVar\n\n\n"
                "class SourceShape:\n"
                "    keeps_receiver: ClassVar[bool] = False\n\n\n"
                "class PlainMethodShape(SourceShape):\n"
                "    keeps_receiver: ClassVar[bool] = True\n\n\n"
                "def emit_arg(shape: SourceShape, idx: int, total: int, out: list[str]) -> None:\n"
                "    if shape.keeps_receiver:\n"
                "        out.append('keep')\n"
                "    elif idx + 1 < total:\n"
                "        out.append('mid')\n"
                "    else:\n"
                "        out.append('last')\n\n\n"
                "class Status:\n"
                "    def is_stopped(self) -> bool:\n"
                "        return False\n\n\n"
                "class Stopped(Status):\n"
                "    def is_stopped(self) -> bool:\n"
                "        return True\n\n\n"
                "def classify(status: Status, unread: int, buckets: dict[str, list[str]], name: str) -> None:\n"
                "    if not status.is_stopped():\n"
                "        buckets['live'].append(name)\n"
                "    elif unread > 0:\n"
                "        buckets['unread'].append(name)\n"
                "    else:\n"
                "        buckets['idle'].append(name)\n\n\n"
                "class ScopeKind:\n"
                "    is_module: ClassVar[bool] = False\n\n\n"
                "class ModuleScope(ScopeKind):\n"
                "    is_module: ClassVar[bool] = True\n\n\n"
                "def resolve_symbol(kind: ScopeKind, is_nonlocal: bool, is_mutated: bool) -> str | None:\n"
                "    if not kind.is_module:\n"
                "        if is_nonlocal:\n"
                "            return None\n"
                "        if is_mutated:\n"
                "            return None\n"
                "    return 'resolved'\n\n\n"
                "class GuardFlag:\n"
                "    enabled: ClassVar[bool] = False\n\n\n"
                "class EnabledFlag(GuardFlag):\n"
                "    enabled: ClassVar[bool] = True\n\n\n"
                "def maybe_emit(ready: bool, flag: GuardFlag, extra: bool) -> str | None:\n"
                "    if ready and flag.enabled and extra:\n"
                "        return 'emitted'\n"
                "    return None\n"
            ),
        },
    )

    for member_name, consumer, method_name in (
        ("keeps_receiver", "emit_arg", "append_arg"),
        ("is_stopped", "classify", "record_bucket"),
        ("is_module", "resolve_symbol", "resolve_symbol"),
        ("enabled", "maybe_emit", "emit_when_ready"),
    ):
        findings = _findings(tmp_path)
        target = next(f for f in findings if f.metrics.member_name == member_name)
        ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            target, _snapshot(tmp_path)
        )
        if ev.recipe is None:
            assert NeededInput.OWNER_PICK in ev.needs, (member_name, ev.reason)
            picks = OwnerPicks(
                "picks.json",
                "",
                (
                    OwnerPick(
                        finding=target.stable_id,
                        owner=consumer,
                        because="family owns its branch behavior",
                        name=method_name,
                        judgement="Push constant answer branch onto the family.",
                    ),
                ),
            )
            picked, _ = picks.applied((target,))
            ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
                picked[0], _snapshot(tmp_path)
            )
        _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "branches.py").read_text()
    for deleted in ("keeps_receiver", "is_stopped", "is_module", "enabled:"):
        assert deleted not in after
    assert "if ready:\n        return flag.emit_when_ready(" in after

    ns: dict[str, object] = {}
    exec(after, ns)
    out: list[str] = []
    ns["emit_arg"](ns["PlainMethodShape"](), 0, 2, out)  # type: ignore[operator]
    ns["emit_arg"](ns["SourceShape"](), 0, 2, out)  # type: ignore[operator]
    ns["emit_arg"](ns["SourceShape"](), 1, 2, out)  # type: ignore[operator]
    assert out == ["keep", "mid", "last"]

    buckets: dict[str, list[str]] = {"live": [], "unread": [], "idle": []}
    ns["classify"](ns["Status"](), 5, buckets, "a")  # type: ignore[operator]
    ns["classify"](ns["Stopped"](), 2, buckets, "b")  # type: ignore[operator]
    ns["classify"](ns["Stopped"](), 0, buckets, "c")  # type: ignore[operator]
    assert buckets == {"live": ["a"], "unread": ["b"], "idle": ["c"]}

    assert ns["resolve_symbol"](ns["ScopeKind"](), True, False) is None  # type: ignore[operator]
    assert ns["resolve_symbol"](ns["ScopeKind"](), False, True) is None  # type: ignore[operator]
    assert ns["resolve_symbol"](ns["ScopeKind"](), False, False) == "resolved"  # type: ignore[operator]
    assert ns["resolve_symbol"](ns["ModuleScope"](), True, True) == "resolved"  # type: ignore[operator]

    assert ns["maybe_emit"](False, ns["EnabledFlag"](), True) is None  # type: ignore[operator]
    assert ns["maybe_emit"](True, ns["GuardFlag"](), True) is None  # type: ignore[operator]
    assert ns["maybe_emit"](True, ns["EnabledFlag"](), False) is None  # type: ignore[operator]
    assert ns["maybe_emit"](True, ns["EnabledFlag"](), True) == "emitted"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushing_cross_module_external_import_and_prefix_guarded_else_assign(
    tmp_path: Path,
):
    """Pushing a branch across modules forwards top-level external/stdlib imports
    (e.g. `import math`), and `if PREFIX and subject.flag: target = ... else: target = ...`
    with a pure independent `PREFIX` upgrades to an assigning polymorphic call."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/policy.py": (
                "from typing import ClassVar\n\n\n"
                "class Policy:\n"
                "    is_active: ClassVar[bool] = False\n\n\n"
                "class ActivePolicy(Policy):\n"
                "    is_active: ClassVar[bool] = True\n\n\n"
                "class TwinRun:\n"
                "    needs_taps: ClassVar[bool] = False\n\n\n"
                "class EnabledTwinRun(TwinRun):\n"
                "    needs_taps: ClassVar[bool] = True\n"
            ),
            "pkg/runner.py": (
                "import math\n\n"
                "from pkg.policy import Policy, TwinRun\n\n\n"
                "def score_policy(policy: Policy, value: float) -> int:\n"
                "    if policy.is_active:\n"
                "        return math.floor(value * 10)\n"
                "    return 0\n\n\n"
                "def accumulate_loss(\n"
                "    twin_ids: tuple[int, ...] | None, run: TwinRun, base: int\n"
                ") -> int:\n"
                "    total = base\n"
                "    if twin_ids is not None and run.needs_taps:\n"
                "        total = total + len(twin_ids) * 10\n"
                "    else:\n"
                "        total = total + 1\n"
                "    return total\n"
            ),
        },
    )

    for member_name, consumer, method_name in (
        ("is_active", "score_policy", "score_value"),
        ("needs_taps", "accumulate_loss", "add_twin_loss"),
    ):
        findings = _findings(tmp_path)
        target = next(f for f in findings if f.metrics.member_name == member_name)
        ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            target, _snapshot(tmp_path)
        )
        if ev.recipe is None:
            assert NeededInput.OWNER_PICK in ev.needs, (member_name, ev.reason)
            picks = OwnerPicks(
                "picks.json",
                "",
                (
                    OwnerPick(
                        finding=target.stable_id,
                        owner=consumer,
                        because="family owns branch behavior",
                        name=method_name,
                        judgement="Push constant answer branch onto the family.",
                    ),
                ),
            )
            picked, _ = picks.applied((target,))
            ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
                picked[0], _snapshot(tmp_path)
            )
        _apply(tmp_path, ev.executable_recipe)

    policy_after = (tmp_path / "pkg" / "policy.py").read_text()
    runner_after = (tmp_path / "pkg" / "runner.py").read_text()
    assert "import math" in policy_after
    assert "is_active" not in policy_after
    assert "needs_taps" not in policy_after
    assert "total = run.add_twin_loss(" in runner_after

    policy_ns: dict[str, object] = {}
    exec(policy_after, policy_ns)
    runner_ns: dict[str, object] = {
        "Policy": policy_ns["Policy"],
        "TwinRun": policy_ns["TwinRun"],
    }
    runner_exec = "\n".join(
        line
        for line in runner_after.splitlines()
        if not line.startswith("from pkg.policy import")
    )
    exec(runner_exec, runner_ns)

    assert runner_ns["score_policy"](policy_ns["Policy"](), 3.7) == 0  # type: ignore[operator]
    assert runner_ns["score_policy"](policy_ns["ActivePolicy"](), 3.7) == 37  # type: ignore[operator]
    assert runner_ns["accumulate_loss"](None, policy_ns["TwinRun"](), 5) == 6  # type: ignore[operator]
    assert runner_ns["accumulate_loss"]((1, 2), policy_ns["TwinRun"](), 5) == 6  # type: ignore[operator]
    assert runner_ns["accumulate_loss"](None, policy_ns["EnabledTwinRun"](), 5) == 6  # type: ignore[operator]
    assert runner_ns["accumulate_loss"]((1, 2), policy_ns["EnabledTwinRun"](), 5) == 25  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_in_place_classmethod_ternary_return_does_not_double_return(tmp_path: Path) -> None:
    """In-place ternary push (`return rule if cls.flag else ''`) uses
    `TernaryDispatchSite.render_in_place_default` so the base method becomes `return rule`,
    never `return return rule` (bench regression from `toad.client.input_modality`)."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/modality.py": (
                "from abc import ABC\n"
                "from typing import ClassVar\n"
                "\n"
                "class InputModality(ABC):\n"
                "    hover_highlights: ClassVar[bool]\n"
                "\n"
                "    @classmethod\n"
                "    def hover_rule(cls, rule: str) -> str:\n"
                "        return rule if cls.hover_highlights else ''\n"
                "\n"
                "    @classmethod\n"
                "    def hover_or_pressed(cls, selector: str) -> str:\n"
                "        if cls.hover_highlights:\n"
                "            return f'{selector}:hover, {selector}:pressed'\n"
                "        return f'{selector}:pressed'\n"
                "\n"
                "class KeyboardInput(InputModality):\n"
                "    hover_highlights = True\n"
                "\n"
                "class TouchInput(InputModality):\n"
                "    hover_highlights = False\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    target = next(f for f in findings if f.metrics.member_name == "hover_highlights")
    ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        target, _snapshot(tmp_path)
    )
    assert ev.executable_recipe is not None, ev.reason
    _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "modality.py").read_text()
    assert "return return" not in after
    assert "hover_highlights" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    kb = ns["KeyboardInput"]
    touch = ns["TouchInput"]
    assert kb.hover_rule("QWidget:hover {}") == "QWidget:hover {}"  # type: ignore[attr-defined]
    assert touch.hover_rule("QWidget:hover {}") == ""  # type: ignore[attr-defined]
    assert kb.hover_or_pressed("QPushButton") == "QPushButton:hover, QPushButton:pressed"  # type: ignore[attr-defined]
    assert touch.hover_or_pressed("QPushButton") == "QPushButton:pressed"  # type: ignore[attr-defined]
    assert _findings(tmp_path) == []


def test_pushes_multiple_embedded_branches_with_consumer_method_picks(tmp_path: Path) -> None:
    """Reproduces `LogView.newest_first` (`comms-tui` `acp_log.py:205,207,211`):
    multiple embedded branches on one constant subtype answer report
    `NeededInput.OWNER_PICK` naming each site, refuse duplicate method names, and
    push every branch cleanly via `consumer_methods` in `--codemod-owner-picks`."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/views.py": (
                "from typing import ClassVar\n\n\n"
                "class LogView:\n"
                "    newest_first: ClassVar[bool] = False\n\n\n"
                "class ErrorsLogView(LogView):\n"
                "    newest_first: ClassVar[bool] = True\n"
            ),
            "pkg/preview.py": (
                "from pkg.views import LogView\n\n\n"
                "class FakeArea:\n"
                "    def __init__(self, scroll_y: int, max_scroll_y: int) -> None:\n"
                "        self.scroll_y = scroll_y\n"
                "        self.max_scroll_y = max_scroll_y\n"
                "        self.cursor = (-1, -1)\n"
                "        self.scrolled = ''\n\n"
                "    def move_cursor(self, pos: tuple[int, int]) -> None:\n"
                "        self.cursor = pos\n\n"
                "    def scroll_home(self, *, animate: bool = True) -> None:\n"
                "        self.scrolled = f'home:{animate}'\n\n"
                "    def scroll_end(self, *, animate: bool = True) -> None:\n"
                "        self.scrolled = f'end:{animate}'\n\n\n"
                "class Preview:\n"
                "    view: LogView\n\n"
                "    def __init__(self, view: LogView) -> None:\n"
                "        self.view = view\n\n"
                "    def draw(self, area: FakeArea) -> None:\n"
                "        area.move_cursor((0, 0) if self.view.newest_first else (9, 9))\n"
                "        (area.scroll_home if self.view.newest_first else area.scroll_end)(animate=False)\n\n"
                "    def at_newest(self, area: FakeArea) -> bool:\n"
                "        return area.scroll_y <= 0 if self.view.newest_first else area.scroll_y >= area.max_scroll_y\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    snap = _snapshot(tmp_path)

    # Without picks: reports OWNER_PICK (not SOURCE_CHANGE) naming every embedded site.
    # (`Preview.at_newest` is whole-function so it takes its own name automatically.)
    ev0 = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(finding, snap)
    assert ev0.recipe is None
    assert NeededInput.OWNER_PICK in ev0.needs
    assert "consumer_methods" in ev0.rejection_reason
    assert "Preview.draw:28" in ev0.rejection_reason
    assert "Preview.draw:29" in ev0.rejection_reason
    assert "Preview.at_newest" not in ev0.rejection_reason

    # Duplicate method names refuse cleanly.
    dup_picks = OwnerPicks(
        "dup.json",
        "",
        (
            OwnerPick(
                finding=finding.stable_id,
                owner=finding.metrics.consumer_symbol,
                because="LogView owns ordering",
                consumer_methods=(
                    ConsumerMethodPick(line=28, name="same_name", judgement="First."),
                    ConsumerMethodPick(line=29, name="same_name", judgement="Second."),
                ),
            ),
        ),
    )
    (dup_finding,), _ = dup_picks.applied((finding,))
    ev_dup = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(dup_finding, snap)
    assert ev_dup.recipe is None
    assert "must be distinct" in ev_dup.rejection_reason

    # Read from JSON file to verify OwnerPicks.read parsing end-to-end.
    picks_file = tmp_path / "picks.json"
    picks_file.write_text(
        '{"picks": [{"finding": "' + finding.stable_id + '", '
        '"owner": "' + finding.metrics.consumer_symbol + '", '
        '"because": "LogView owns newest-end positioning", '
        '"consumer_methods": ['
        '{"line": 28, "symbol": "Preview.draw", "name": "newest_cursor", "judgement": "Cursor position for newest log entry."},'
        '{"line": 29, "symbol": "Preview.draw", "name": "scroll_to_newest", "judgement": "Scroll area to newest log entry."}'
        ']}]}\n',
        encoding="utf-8",
    )
    loaded = OwnerPicks.read(picks_file)
    (picked_finding,), report = loaded.applied((finding,))
    assert report.unmatched == ()
    ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(picked_finding, snap)
    assert ev.executable_recipe is not None, ev.rejection_reason
    _apply(tmp_path, ev.executable_recipe)

    views_after = (tmp_path / "pkg" / "views.py").read_text()
    preview_after = (tmp_path / "pkg" / "preview.py").read_text()
    assert "newest_first" not in views_after
    assert "newest_first" not in preview_after
    assert "def newest_cursor" in views_after
    assert "def scroll_to_newest" in views_after
    assert "def at_newest" in views_after

    ns: dict[str, object] = {}
    exec(views_after, ns)
    exec(preview_after.replace("from pkg.views import LogView\n", ""), ns)
    a_log = ns["FakeArea"](10, 10)  # type: ignore[operator]
    p_log = ns["Preview"](ns["LogView"]())  # type: ignore[operator]
    p_log.draw(a_log)
    assert a_log.cursor == (9, 9)
    assert a_log.scrolled == "end:False"
    assert p_log.at_newest(a_log) is True

    a_err = ns["FakeArea"](0, 10)  # type: ignore[operator]
    p_err = ns["Preview"](ns["ErrorsLogView"]())  # type: ignore[operator]
    p_err.draw(a_err)
    assert a_err.cursor == (0, 0)
    assert a_err.scrolled == "home:False"
    assert p_err.at_newest(a_err) is True
    assert _findings(tmp_path) == []





