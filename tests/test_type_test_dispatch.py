"""Pushing OWN_TYPE_TEST dispatches onto polymorphic methods of the tested family."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    codemod_plan_from_findings,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._restated_facts import OwnTypeTestDetector
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.owner_picks import OwnerPick, OwnerPicks
from nominal_refactor_advisor.refactors.type_test_dispatch import (
    PushTypeTestDispatchOperation,
)
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")


def _findings(root: Path):
    with ScanCache.scope():
        return OwnTypeTestDetector().detect(
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
    simulation = CodemodPlanSequence.from_operations(recipe.operations).simulate(
        _snapshot(root)
    )
    assert simulation.is_clean
    simulation.apply()


def test_pushes_whole_function_if_elif_else_dispatch_and_clears_finding(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/failures.py": (
                "class Failure:\n"
                "    pass\n\n\n"
                "class NotANumber(Failure):\n"
                "    pass\n\n\n"
                "class OutOfRange(Failure):\n"
                "    def __init__(self, limit: int) -> None:\n"
                "        self.limit = limit\n\n\n"
                "def describe_failure(failure: Failure, prefix: str) -> str:\n"
                "    if isinstance(failure, NotANumber):\n"
                "        return f'{prefix}: not a number'\n"
                "    elif isinstance(failure, OutOfRange):\n"
                "        return f'{prefix}: exceeds {failure.limit}'\n"
                "    else:\n"
                "        return f'{prefix}: invalid'\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    (operation,) = recipe.operations
    assert isinstance(operation, PushTypeTestDispatchOperation)
    assert operation.method_name == "describe_failure"
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "failures.py").read_text()
    assert "isinstance" not in after
    assert "return failure.describe_failure(prefix)" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["describe_failure"](ns["Failure"](), "err") == "err: invalid"  # type: ignore[operator]
    assert ns["describe_failure"](ns["NotANumber"](), "err") == "err: not a number"  # type: ignore[operator]
    assert ns["describe_failure"](ns["OutOfRange"](5), "err") == "err: exceeds 5"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_ternary_tuple_dispatch_and_migrates_cross_module_imports(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/helpers.py": "def format_symbol(name: str) -> str:\n    return f'<{name}>'\n",
            "pkg/bindings.py": (
                "class Binding:\n"
                "    pass\n\n\n"
                "class NamedBinding(Binding):\n"
                "    def __init__(self, symbol: str) -> None:\n"
                "        self.symbol = symbol\n\n\n"
                "class OtherBinding(Binding):\n"
                "    def __init__(self, symbol: str) -> None:\n"
                "        self.symbol = symbol\n"
            ),
            "pkg/use.py": (
                "from pkg.bindings import Binding, NamedBinding, OtherBinding\n"
                "from pkg.helpers import format_symbol\n\n\n"
                "def binding_label(binding: Binding) -> str | None:\n"
                "    return format_symbol(binding.symbol) if isinstance(binding, (NamedBinding, OtherBinding)) else None\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    _apply(tmp_path, evaluation.executable_recipe)

    use_after = (tmp_path / "pkg" / "use.py").read_text()
    bindings_after = (tmp_path / "pkg" / "bindings.py").read_text()
    assert "return binding.binding_label()" in use_after
    assert "NamedBinding" not in use_after
    assert "OtherBinding" not in use_after
    assert "from pkg.helpers import format_symbol" in bindings_after
    assert "def binding_label(self) -> str | None:" in bindings_after
    assert _findings(tmp_path) == []


def test_pushes_class_identity_and_issubclass_as_classmethod_and_refuses_exact_with_descendant(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/bars.py": (
                "class Bar:\n"
                "    pass\n\n\n"
                "class SolidBar(Bar):\n"
                "    pass\n\n\n"
                "class NoBar(Bar):\n"
                "    pass\n\n\n"
                "def barred(bar: type[Bar], text: str) -> str:\n"
                "    if bar is NoBar:\n"
                "        return text\n"
                "    return f'|{text}'\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    _apply(tmp_path, evaluation.executable_recipe)
    after = (tmp_path / "pkg" / "bars.py").read_text()
    assert "@classmethod" in after
    assert "return bar.barred(text)" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["barred"](ns["NoBar"], "hi") == "hi"  # type: ignore[operator]
    assert ns["barred"](ns["SolidBar"], "hi") == "|hi"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_refuses_exact_identity_test_when_tested_class_has_scanned_descendant(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/bars.py": (
                "class Bar:\n"
                "    pass\n\n\n"
                "class NoBar(Bar):\n"
                "    pass\n\n\n"
                "class CustomNoBar(NoBar):\n"
                "    pass\n\n\n"
                "def barred(bar: type[Bar], text: str) -> str:\n"
                "    if bar is NoBar:\n"
                "        return text\n"
                "    return f'|{text}'\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    assert evaluation.recipe is None
    assert "CustomNoBar" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_pushes_match_class_pattern_and_void_guard_with_owner_pick(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/layouts.py": (
                "class Layout:\n"
                "    def __init__(self) -> None:\n"
                "        self.height = 0\n\n\n"
                "class GridLayout(Layout):\n"
                "    pass\n\n\n"
                "def configure(layout: Layout, height: int) -> int:\n"
                "    applied = height + 1\n"
                "    if isinstance(layout, GridLayout):\n"
                "        layout.height = applied\n"
                "    return applied\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    # Embedded inside a larger function: asks for OWNER_PICK first.
    first_eval = OwnTypeTestDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    assert first_eval.recipe is None
    assert NeededInput.OWNER_PICK in first_eval.needs

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=finding.stable_id,
                owner="configure",
                because="grid layouts apply stretch height",
                name="apply_height",
                judgement="Apply configured height to layouts that support it.",
            ),
        ),
    )
    picked, report = picks.applied((finding,))
    assert [outcome.outcome for outcome in report.outcomes] == ["picked"]
    evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
        picked[0], _snapshot(tmp_path)
    )
    _apply(tmp_path, evaluation.executable_recipe)
    after = (tmp_path / "pkg" / "layouts.py").read_text()
    assert "layout.apply_height(applied)" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    grid = ns["GridLayout"]()  # type: ignore[operator]
    plain = ns["Layout"]()  # type: ignore[operator]
    assert ns["configure"](grid, 9) == 10  # type: ignore[operator]
    assert grid.height == 10
    assert ns["configure"](plain, 9) == 10  # type: ignore[operator]
    assert plain.height == 0
    assert _findings(tmp_path) == []


def test_pushes_match_class_patterns_with_alias(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/items.py": (
                "class Item:\n"
                "    pass\n\n\n"
                "class NumberItem(Item):\n"
                "    def __init__(self, value: int) -> None:\n"
                "        self.value = value\n\n\n"
                "def score(item: Item) -> int:\n"
                "    match item:\n"
                "        case NumberItem() as num:\n"
                "            return num.value * 2\n"
                "        case _:\n"
                "            return 0\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    _apply(tmp_path, evaluation.executable_recipe)
    after = (tmp_path / "pkg" / "items.py").read_text()
    assert "return item.score()" in after
    assert "match " not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["score"](ns["NumberItem"](7)) == 14  # type: ignore[operator]
    assert ns["score"](ns["Item"]()) == 0  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_negated_raise_guard_and_boolean_return_predicate(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/events.py": (
                "class Event:\n"
                "    pass\n\n\n"
                "class Resize(Event):\n"
                "    pass\n\n\n"
                "def is_resize(event: Event) -> bool:\n"
                "    return isinstance(event, Resize)\n\n\n"
                "def require_resize(event: Event) -> None:\n"
                "    if not isinstance(event, Resize):\n"
                "        raise RuntimeError('need resize')\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    _apply(tmp_path, evaluation.executable_recipe)

    events_file = tmp_path / "pkg" / "events.py"
    events_path = str(events_file.resolve())
    guard_line = next(
        idx
        for idx, line in enumerate(events_file.read_text().splitlines(), start=1)
        if "if not isinstance(event, Resize):" in line
    )
    guard_op = PushTypeTestDispatchOperation(
        target=SourceRewriteTarget(file_path=events_path, qualname="require_resize"),
        test_line=guard_line,
        base=SourceRewriteTarget(file_path=events_path, qualname="Event"),
        method_name="require_resize",
    )
    simulation = CodemodPlanSequence.from_operations((guard_op,)).simulate(
        _snapshot(tmp_path)
    )
    assert simulation.is_clean
    simulation.apply()

    after = (tmp_path / "pkg" / "events.py").read_text()
    assert "isinstance" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["is_resize"](ns["Resize"]()) is True  # type: ignore[operator]
    assert ns["is_resize"](ns["Event"]()) is False  # type: ignore[operator]
    ns["require_resize"](ns["Resize"]())  # type: ignore[operator]
    try:
        ns["require_resize"](ns["Event"]())  # type: ignore[operator]
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for non-Resize Event")
    assert _findings(tmp_path) == []


def test_pushes_compound_and_return_with_outer_self_rename(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messages.py": (
                "class Message:\n"
                "    pass\n\n\n"
                "class CoreEventMessage(Message):\n"
                "    def __init__(self, topic: str, seq: int) -> None:\n"
                "        self.topic = topic\n"
                "        self.seq = seq\n\n\n"
                "class CarrierMatcher:\n"
                "    def __init__(self, topic: str, seq: int) -> None:\n"
                "        self.topic = topic\n"
                "        self.seq = seq\n\n"
                "    def matches_carrier(self, message: Message) -> bool:\n"
                "        return (\n"
                "            isinstance(message, CoreEventMessage)\n"
                "            and self.topic == message.topic\n"
                "            and self.seq <= message.seq\n"
                "        )\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "messages.py").read_text()
    assert "isinstance" not in after
    assert "message.matches_carrier(self)" in after
    assert "def matches_carrier(self, owner) -> bool:" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    matcher = ns["CarrierMatcher"]("bus", 1)  # type: ignore[operator]
    b = ns["CoreEventMessage"]("bus", 2)  # type: ignore[operator]
    earlier = ns["CoreEventMessage"]("bus", 0)  # type: ignore[operator]
    c = ns["CoreEventMessage"]("other", 2)  # type: ignore[operator]
    plain = ns["Message"]()  # type: ignore[operator]
    assert matcher.matches_carrier(b) is True
    assert matcher.matches_carrier(earlier) is False
    assert matcher.matches_carrier(c) is False
    assert matcher.matches_carrier(plain) is False
    assert _findings(tmp_path) == []


def test_pushes_prefix_guarded_void_if_and_match_keyword_bindings(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/offers.py": (
                "class Offer:\n"
                "    pass\n\n\n"
                "class TailnetOffer(Offer):\n"
                "    def __init__(self, endpoint: str) -> None:\n"
                "        self.endpoint = endpoint\n"
                "        self.settled: list[str] = []\n\n\n"
                "def resolve_endpoint(offer: Offer) -> str:\n"
                "    match offer:\n"
                "        case TailnetOffer(endpoint=served):\n"
                "            return f'https://{served}'\n"
                "    return 'none'\n\n\n"
                "def publish(offer: Offer, answers: tuple[str, ...]) -> None:\n"
                "    if answers and isinstance(offer, TailnetOffer):\n"
                "        offer.settled.extend(answers)\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    assert len(findings) == 2
    by_fn = {f.evidence[0].symbol: f for f in findings}

    match_eval = OwnTypeTestDetector().evaluate_recipe_for_finding(
        by_fn["resolve_endpoint"], _snapshot(tmp_path)
    )
    _apply(tmp_path, match_eval.executable_recipe)

    (publish_finding,) = _findings(tmp_path)
    unpicked = OwnTypeTestDetector().evaluate_recipe_for_finding(
        publish_finding, _snapshot(tmp_path)
    )
    assert unpicked.recipe is None
    assert NeededInput.OWNER_PICK in unpicked.needs

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=publish_finding.stable_id,
                owner="publish",
                because="tailnet offers settle answered deliveries",
                name="settle_answers",
                judgement="Record settled answer IDs when applicable.",
            ),
        ),
    )
    picked, _ = picks.applied((publish_finding,))
    publish_eval = OwnTypeTestDetector().evaluate_recipe_for_finding(
        picked[0], _snapshot(tmp_path)
    )
    _apply(tmp_path, publish_eval.executable_recipe)

    after = (tmp_path / "pkg" / "offers.py").read_text()
    assert "isinstance" not in after
    assert "match " not in after
    assert "if answers:\n        offer.settle_answers(answers)" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    tailnet = ns["TailnetOffer"]("bus.ts.net")  # type: ignore[operator]
    plain = ns["Offer"]()  # type: ignore[operator]
    assert ns["resolve_endpoint"](tailnet) == "https://bus.ts.net"  # type: ignore[operator]
    assert ns["resolve_endpoint"](plain) == "none"  # type: ignore[operator]
    ns["publish"](tailnet, ())  # type: ignore[operator]
    assert tailnet.settled == []
    ns["publish"](tailnet, ("a1", "a2"))  # type: ignore[operator]
    assert tailnet.settled == ["a1", "a2"]
    ns["publish"](plain, ("a1",))  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_compound_or_return_and_embedded_issubclass_conjunction(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/settings.py": (
                "class Declaration:\n"
                "    pass\n\n\n"
                "class SettingsDeclaration(Declaration):\n"
                "    @classmethod\n"
                "    def admits_field(cls, field_name: str) -> bool:\n"
                "        return field_name.startswith('ok_')\n\n\n"
                "def allows_missing_or_ready(decl: Declaration) -> bool:\n"
                "    return not isinstance(decl, SettingsDeclaration) or decl.admits_field('ok_x')\n\n\n"
                "def admits(owner: type[Declaration], field_name: str) -> bool:\n"
                "    key = field_name.strip()\n"
                "    return (\n"
                "        issubclass(owner, SettingsDeclaration)\n"
                "        and owner.admits_field(key)\n"
                "    )\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    assert len(findings) == 2
    by_fn = {f.evidence[0].symbol: f for f in findings}

    # Whole-function `not isinstance(...) or REST` synthesizes automatically.
    or_eval = OwnTypeTestDetector().evaluate_recipe_for_finding(
        by_fn["allows_missing_or_ready"], _snapshot(tmp_path)
    )
    _apply(tmp_path, or_eval.executable_recipe)

    # Embedded `issubclass(...) and REST` asks for OWNER_PICK, then executes cleanly.
    re_findings = _findings(tmp_path)
    assert len(re_findings) == 1
    admits_finding = re_findings[0]
    unpicked = OwnTypeTestDetector().evaluate_recipe_for_finding(
        admits_finding, _snapshot(tmp_path)
    )
    assert unpicked.recipe is None
    assert NeededInput.OWNER_PICK in unpicked.needs
    assert "supply `owner` `admits`" in unpicked.rejection_reason

    wrong_picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=admits_finding.stable_id,
                owner="Declaration",
                because="wrong owner",
                name="admits_setting_field",
                judgement="Whether this declaration admits the named field.",
            ),
        ),
    )
    _, wrong_report = wrong_picks.applied((admits_finding,))
    assert wrong_report.outcomes[0].outcome == "rejected"
    assert "is not the consumer `admits`" in wrong_report.outcomes[0].detail

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=admits_finding.stable_id,
                owner="admits",
                because="settings declarations own field admission",
                name="admits_setting_field",
                judgement="Whether this declaration admits the named field.",
            ),
        ),
    )
    picked, _ = picks.applied((admits_finding,))
    admits_eval = OwnTypeTestDetector().evaluate_recipe_for_finding(
        picked[0], _snapshot(tmp_path)
    )
    _apply(tmp_path, admits_eval.executable_recipe)

    after = (tmp_path / "pkg" / "settings.py").read_text()
    assert "isinstance" not in after
    assert "issubclass" not in after
    assert "@classmethod" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["allows_missing_or_ready"](ns["Declaration"]()) is True  # type: ignore[operator]
    assert ns["allows_missing_or_ready"](ns["SettingsDeclaration"]()) is True  # type: ignore[operator]
    assert ns["admits"](ns["SettingsDeclaration"], " ok_a ") is True  # type: ignore[operator]
    assert ns["admits"](ns["SettingsDeclaration"], " bad ") is False  # type: ignore[operator]
    assert ns["admits"](ns["Declaration"], " ok_a ") is False  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_compound_if_and_guard_and_or_early_return(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/nodes.py": (
                "class Node:\n"
                "    pass\n\n\n"
                "class ActiveNode(Node):\n"
                "    def __init__(self, label: str, active: bool) -> None:\n"
                "        self.label = label\n"
                "        self.active = active\n\n\n"
                "def active_label(node: Node) -> str:\n"
                "    if isinstance(node, ActiveNode) and node.active:\n"
                "        return node.label\n"
                "    return 'idle'\n\n\n"
                "def required_label(node: Node) -> str:\n"
                "    if not isinstance(node, ActiveNode) or not node.active:\n"
                "        return 'none'\n"
                "    return node.label\n"
            ),
        },
    )
    for fn_name in ("active_label", "required_label"):
        findings = _findings(tmp_path)
        target_finding = next(f for f in findings if f.evidence[0].symbol == fn_name)
        evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
            target_finding, _snapshot(tmp_path)
        )
        _apply(tmp_path, evaluation.executable_recipe)

    after = (tmp_path / "pkg" / "nodes.py").read_text()
    assert "isinstance" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    on = ns["ActiveNode"]("go", True)  # type: ignore[operator]
    off = ns["ActiveNode"]("go", False)  # type: ignore[operator]
    plain = ns["Node"]()  # type: ignore[operator]
    assert ns["active_label"](on) == "go"  # type: ignore[operator]
    assert ns["active_label"](off) == "idle"  # type: ignore[operator]
    assert ns["active_label"](plain) == "idle"  # type: ignore[operator]
    assert ns["required_label"](on) == "go"  # type: ignore[operator]
    assert ns["required_label"](off) == "none"  # type: ignore[operator]
    assert ns["required_label"](plain) == "none"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_void_early_return_and_rejects_loop_early_return(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/handles.py": (
                "class Handle:\n"
                "    def __init__(self) -> None:\n"
                "        self.events: list[str] = []\n\n"
                "    def release(self, tag: str) -> None:\n"
                "        self.events.append(f'release:{tag}')\n\n\n"
                "class OpenHandle(Handle):\n"
                "    def close(self) -> None:\n"
                "        self.events.append('close')\n\n\n"
                "def settle_handle(handle: Handle, tag: str) -> None:\n"
                "    if isinstance(handle, OpenHandle):\n"
                "        handle.close()\n"
                "        return\n"
                "    handle.release(tag)\n\n\n"
                "def stop_on_open(handles: list[Handle], log: list[str]) -> None:\n"
                "    for handle in handles:\n"
                "        if isinstance(handle, OpenHandle):\n"
                "            handle.close()\n"
                "            return\n"
                "        log.append('kept')\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    by_fn = {f.evidence[0].symbol: f for f in findings}

    # Whole-function void early-return pushes automatically.
    settle_eval = OwnTypeTestDetector().evaluate_recipe_for_finding(
        by_fn["settle_handle"], _snapshot(tmp_path)
    )
    _apply(tmp_path, settle_eval.executable_recipe)

    # Loop-nested early return exits the whole function, not just the loop iteration, so must not push.
    loop_eval = OwnTypeTestDetector().evaluate_recipe_for_finding(
        by_fn["stop_on_open"], _snapshot(tmp_path)
    )
    assert loop_eval.recipe is None
    assert "not a pushable dispatch shape" in loop_eval.rejection_reason

    after = (tmp_path / "pkg" / "handles.py").read_text()
    assert "handle.settle_handle(tag)" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    open_h = ns["OpenHandle"]()  # type: ignore[operator]
    plain_h = ns["Handle"]()  # type: ignore[operator]
    ns["settle_handle"](open_h, "x")  # type: ignore[operator]
    ns["settle_handle"](plain_h, "x")  # type: ignore[operator]
    assert open_h.events == ["close"]
    assert plain_h.events == ["release:x"]


def test_pushes_loop_continue_guard_and_non_first_boolop_operand(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/commands.py": (
                "class Command:\n"
                "    pass\n\n\n"
                "class SpecialCommand(Command):\n"
                "    def __init__(self, name: str, active: bool = True) -> None:\n"
                "        self.name = name\n"
                "        self.active = active\n\n\n"
                "def collect_special(commands: list[Command], out: list[str]) -> None:\n"
                "    for cmd in commands:\n"
                "        if not isinstance(cmd, SpecialCommand):\n"
                "            continue\n"
                "        out.append(cmd.name)\n\n\n"
                "def accepts(flag: bool, cmd_type: type[Command]) -> bool:\n"
                "    return flag or issubclass(\n"
                "        cmd_type,\n"
                "        SpecialCommand,\n"
                "    )\n\n\n"
                "def accepts_active(flag: bool, cmd: Command) -> bool:\n"
                "    return (\n"
                "        flag\n"
                "        and isinstance(cmd, SpecialCommand)\n"
                "        and cmd.active\n"
                "    )\n"
            ),
        },
    )
    for fn_name, method_name in (
        ("collect_special", "append_if_special"),
        ("accepts", "is_special_command"),
        ("accepts_active", "is_active_special"),
    ):
        findings = _findings(tmp_path)
        target_finding = next(f for f in findings if f.evidence[0].symbol == fn_name)
        unpicked = OwnTypeTestDetector().evaluate_recipe_for_finding(
            target_finding, _snapshot(tmp_path)
        )
        assert unpicked.recipe is None
        assert NeededInput.OWNER_PICK in unpicked.needs
        picks = OwnerPicks(
            "picks.json",
            "",
            (
                OwnerPick(
                    finding=target_finding.stable_id,
                    owner=fn_name,
                    because="command subclasses own their behavior",
                    name=method_name,
                    judgement="Command family dispatch method.",
                ),
            ),
        )
        picked, _ = picks.applied((target_finding,))
        evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
            picked[0], _snapshot(tmp_path)
        )
        _apply(tmp_path, evaluation.executable_recipe)

    after = (tmp_path / "pkg" / "commands.py").read_text()
    assert "isinstance" not in after
    assert "issubclass" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    out: list[str] = []
    special_on = ns["SpecialCommand"]("go", True)  # type: ignore[operator]
    special_off = ns["SpecialCommand"]("no", False)  # type: ignore[operator]
    plain = ns["Command"]()  # type: ignore[operator]
    ns["collect_special"]([plain, special_on, plain], out)  # type: ignore[operator]
    assert out == ["go"]
    assert ns["accepts"](True, ns["Command"]) is True  # type: ignore[operator]
    assert ns["accepts"](False, ns["Command"]) is False  # type: ignore[operator]
    assert ns["accepts"](False, ns["SpecialCommand"]) is True  # type: ignore[operator]
    assert ns["accepts_active"](True, special_on) is True  # type: ignore[operator]
    assert ns["accepts_active"](True, special_off) is False  # type: ignore[operator]
    assert ns["accepts_active"](False, special_on) is False  # type: ignore[operator]
    assert ns["accepts_active"](True, plain) is False  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_decodes_mypy_intersection_type_for_pushed_dispatch(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/detectors.py": (
                "class Detector:\n"
                "    pass\n\n\n"
                "class CompactMixin:\n"
                "    pass\n\n\n"
                "class MultiCompactMixin(CompactMixin):\n"
                "    pass\n\n\n"
                "class SingleDetector(CompactMixin, Detector):\n"
                "    pass\n\n\n"
                "class MultiDetector(MultiCompactMixin, Detector):\n"
                "    pass\n\n\n"
                "def classify(detectors: list[type[Detector]]) -> list[str]:\n"
                "    compact = [d for d in detectors if issubclass(d, CompactMixin)]\n"
                "    out: list[str] = []\n"
                "    for det in compact:\n"
                "        if issubclass(det, MultiCompactMixin):\n"
                "            out.append('multi')\n"
                "        else:\n"
                "            out.append('single')\n"
                "    return out\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    inner_finding = next(
        f for f in findings if "MultiCompactMixin" in f.summary or f.evidence[0].line > 23
    )
    unpicked = OwnTypeTestDetector().evaluate_recipe_for_finding(
        inner_finding, _snapshot(tmp_path)
    )
    assert unpicked.recipe is None
    assert NeededInput.OWNER_PICK in unpicked.needs
    assert "CompactMixin" in unpicked.rejection_reason

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=inner_finding.stable_id,
                owner="classify",
                because="compact mixins own single vs multi classification",
                name="record_compact_kind",
                judgement="Append this compact mixin's kind label to out.",
            ),
        ),
    )
    picked, _ = picks.applied((inner_finding,))
    evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
        picked[0], _snapshot(tmp_path)
    )
    _apply(tmp_path, evaluation.executable_recipe)

    after = (tmp_path / "pkg" / "detectors.py").read_text()
    assert "MultiCompactMixin" in after
    assert "det.record_compact_kind(out)" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["classify"]([ns["SingleDetector"], ns["MultiDetector"], ns["Detector"]]) == [  # type: ignore[operator]
        "single",
        "multi",
    ]


def test_pushes_multiple_inheritance_mixin_dispatch_and_rejects_bad_c3_or_collision(
    tmp_path: Path,
):
    """Pushes `isinstance`/`issubclass` against a composed mixin onto `(Base, Mixin)` when
    C3 MRO puts `Mixin` ahead of `Base`, and refuses reversed C3 order or mixin hierarchy collisions."""

    good = tmp_path / "good"
    _write(
        good,
        {
            "pkg/__init__.py": "",
            "pkg/cli.py": (
                "class Command:\n"
                "    pass\n\n\n"
                "class ApplyMixin:\n"
                "    pass\n\n\n"
                "class ApplyCommand(ApplyMixin, Command):\n"
                "    pass\n\n\n"
                "class Unrelated(ApplyMixin):\n"
                "    def colliding(self) -> bool:\n"
                "        return False\n\n\n"
                "def accepts(cmd_type: type[Command]) -> bool:\n"
                "    return issubclass(cmd_type, ApplyMixin)\n\n\n"
                "def colliding(cmd_type: type[Command]) -> bool:\n"
                "    return issubclass(cmd_type, ApplyMixin)\n"
            ),
        },
    )
    findings = _findings(good)
    by_sym = {f.evidence[0].symbol: f for f in findings}
    assert set(by_sym) == {"accepts", "colliding"}

    # Unrelated subclass of ApplyMixin already declares `colliding`: must be refused.
    colliding_eval = OwnTypeTestDetector().evaluate_recipe_for_finding(
        by_sym["colliding"], _snapshot(good)
    )
    assert colliding_eval.recipe is None
    assert "along mixin `ApplyMixin`'s hierarchy" in colliding_eval.rejection_reason

    # `accepts` pushes cleanly onto Command (default False) and ApplyMixin (override True).
    accepts_eval = OwnTypeTestDetector().evaluate_recipe_for_finding(
        by_sym["accepts"], _snapshot(good)
    )
    _apply(good, accepts_eval.executable_recipe)
    after = (good / "pkg" / "cli.py").read_text()
    assert "return cmd_type.accepts()" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["accepts"](ns["Command"]) is False  # type: ignore[operator]
    assert ns["accepts"](ns["ApplyCommand"]) is True  # type: ignore[operator]

    # Reversed C3 order `class BadOrder(Command, ApplyMixin):` puts Command ahead of ApplyMixin -> refused.
    bad = tmp_path / "bad"
    _write(
        bad,
        {
            "pkg/__init__.py": "",
            "pkg/cli.py": (
                "class Command:\n"
                "    pass\n\n\n"
                "class ApplyMixin:\n"
                "    pass\n\n\n"
                "class BadOrder(Command, ApplyMixin):\n"
                "    pass\n\n\n"
                "def accepts(cmd_type: type[Command]) -> bool:\n"
                "    return issubclass(cmd_type, ApplyMixin)\n"
            ),
        },
    )
    (bad_finding,) = _findings(bad)
    bad_eval = OwnTypeTestDetector().evaluate_recipe_for_finding(
        bad_finding, _snapshot(bad)
    )
    assert bad_eval.recipe is None
    assert "disagrees with the original type-test branch order" in bad_eval.rejection_reason


def test_pushes_prefix_guarded_or_not_isinstance_preserving_short_circuit(tmp_path: Path):
    """`if PREFIX or not isinstance(x, Sub): return fallback; ...` preserves PREFIX
    short-circuiting and pushes the `isinstance` dispatch onto `(Base, Sub)`."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/metrics.py": (
                "class Base:\n"
                "    pass\n\n\n"
                "class Sub(Base):\n"
                "    def __init__(self, label: str) -> None:\n"
                "        self.label = label\n\n\n"
                "def evaluate(ctx: object | None, item: Base) -> str:\n"
                "    if ctx is None or not isinstance(item, Sub):\n"
                "        return 'skip'\n"
                "    return f'ok:{item.label}'\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    unpicked = OwnTypeTestDetector().evaluate_recipe_for_finding(
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
                owner="evaluate",
                because="Base family owns item evaluation",
                name="eval_item",
                judgement="Evaluate item when context is present.",
            ),
        ),
    )
    picked, _ = picks.applied((finding,))
    evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(
        picked[0], _snapshot(tmp_path)
    )
    _apply(tmp_path, evaluation.executable_recipe)

    after = (tmp_path / "pkg" / "metrics.py").read_text()
    assert "isinstance" not in after
    assert "if ctx is None:" in after
    assert "return item.eval_item()" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["evaluate"](None, None) == "skip"  # type: ignore[operator]
    assert ns["evaluate"](object(), ns["Base"]()) == "skip"  # type: ignore[operator]
    assert ns["evaluate"](object(), ns["Sub"]("win")) == "ok:win"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_multi_continue_loop_and_refuses_escaping_loop_local(tmp_path: Path):
    """Loop-`continue` tails with further `continue` guards push cleanly by rewriting
    direct-loop `continue`s to `return`, while escaping outer-local stores are refused."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/items.py": (
                "class Item:\n"
                "    pass\n\n\n"
                "class Named(Item):\n"
                "    def __init__(self, name: str) -> None:\n"
                "        self.name = name\n\n\n"
                "def collect(items: list[Item]) -> list[str]:\n"
                "    out: list[str] = []\n"
                "    for item in items:\n"
                "        if not isinstance(item, Named):\n"
                "            continue\n"
                "        if not item.name:\n"
                "            continue\n"
                "        out.append(item.name)\n"
                "    return out\n\n\n"
                "def escaping_local(items: list[Item]) -> str | None:\n"
                "    last: str | None = None\n"
                "    for item in items:\n"
                "        if not isinstance(item, Named):\n"
                "            continue\n"
                "        last = item.name\n"
                "    return last\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"collect", "escaping_local"}

    det = OwnTypeTestDetector()
    esc_ev = det.evaluate_recipe_for_finding(by_sym["escaping_local"], _snapshot(tmp_path))
    assert esc_ev.recipe is None
    assert "assigns outer local(s) last" in esc_ev.rejection_reason
    assert NeededInput.OWNER_PICK not in esc_ev.needs

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=by_sym["collect"].stable_id,
                owner="collect",
                because="Item owns appending its non-empty name",
                name="append_non_empty",
                judgement="Append name to out when named and non-empty.",
            ),
        ),
    )
    picked, _ = picks.applied((by_sym["collect"],))
    ev = det.evaluate_recipe_for_finding(picked[0], _snapshot(tmp_path))
    _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "items.py").read_text()
    ns: dict[str, object] = {}
    exec(after, ns)
    Item = ns["Item"]
    Named = ns["Named"]
    assert ns["collect"]([Item(), Named(""), Named("ok")]) == ["ok"]  # type: ignore[operator]
    assert [f.primary_evidence.symbol for f in _findings(tmp_path)] == ["escaping_local"]


def test_except_handler_variable_is_not_captured_as_outer_param(tmp_path: Path):
    """An `except ValueError as error:` binding inside a pushed arm is bound locally
    in that arm and must never be captured as a parameter from the caller."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/plans.py": (
                "class Plan:\n"
                "    pass\n\n\n"
                "class CutPlan(Plan):\n"
                "    def __init__(self, fail: bool) -> None:\n"
                "        self.fail = fail\n\n"
                "    def run(self) -> str:\n"
                "        if self.fail:\n"
                "            raise ValueError('bad cut')\n"
                "        return 'cut'\n\n\n"
                "def execute(plan: Plan) -> str:\n"
                "    if isinstance(plan, CutPlan):\n"
                "        try:\n"
                "            return plan.run()\n"
                "        except ValueError as error:\n"
                "            return f'err:{error}'\n"
                "    return 'skip'\n\n\n"
                "def execute_trailing(plan: Plan) -> str:\n"
                "    if isinstance(plan, CutPlan):\n"
                "        try:\n"
                "            result = plan.run()\n"
                "        except ValueError as error:\n"
                "            return f'err:{error}'\n"
                "        return result\n"
                "    return 'skip'\n"
            ),
        },
    )
    for sym in ("execute", "execute_trailing"):
        findings = _findings(tmp_path)
        finding = next(f for f in findings if f.primary_evidence.symbol == sym)
        ev = OwnTypeTestDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
        _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "plans.py").read_text()
    assert "plan.execute()" in after
    assert "plan.execute_trailing()" in after
    assert "def execute(self) -> str:" in after
    assert "def execute_trailing(self) -> str:" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    for fn_name in ("execute", "execute_trailing"):
        assert ns[fn_name](ns["Plan"]()) == "skip"  # type: ignore[operator]
        assert ns[fn_name](ns["CutPlan"](False)) == "cut"  # type: ignore[operator]
        assert ns[fn_name](ns["CutPlan"](True)) == "err:bad cut"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_negated_void_and_prefix_guarded_negated_and_positive_or(tmp_path: Path):
    """Covers symmetric polarities: standalone void `if not isinstance`,
    `if PREFIX and not issubclass`, and `if PREFIX or cls is Sub`."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/neg.py": (
                "class Ref:\n"
                "    def __init__(self) -> None:\n"
                "        self.admitted = 0\n\n"
                "    def admit(self) -> None:\n"
                "        self.admitted += 1\n\n\n"
                "class OpenRef(Ref):\n"
                "    pass\n\n\n"
                "class Cmd:\n"
                "    pass\n\n\n"
                "class PlanCmd(Cmd):\n"
                "    pass\n\n\n"
                "def admit_all(refs: list[Ref]) -> None:\n"
                "    for ref in refs:\n"
                "        if not isinstance(ref, OpenRef):\n"
                "            ref.admit()\n\n\n"
                "def record_non_plan(flag: bool, cls: type[Cmd], log: list[str]) -> None:\n"
                "    if flag and not issubclass(cls, PlanCmd):\n"
                "        log.append(cls.__name__)\n\n\n"
                "def route_cmd(fast: bool, cls: type[Cmd]) -> str:\n"
                "    if fast or cls is PlanCmd:\n"
                "        return 'direct'\n"
                "    return 'fallback'\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"admit_all", "record_non_plan", "route_cmd"}

    det = OwnTypeTestDetector()
    for sym, method_name, judgement in (
        ("admit_all", "admit_if_closed", "Admit non-open references."),
        ("record_non_plan", "record_if_non_plan", "Record non-plan command names."),
        ("route_cmd", "route_label", "Return routing label."),
    ):
        current = next(f for f in _findings(tmp_path) if f.primary_evidence.symbol == sym)
        picks = OwnerPicks(
            "picks.json",
            "",
            (
                OwnerPick(
                    finding=current.stable_id,
                    owner=sym,
                    because="family owns dispatch",
                    name=method_name,
                    judgement=judgement,
                ),
            ),
        )
        picked, _ = picks.applied((current,))
        ev = det.evaluate_recipe_for_finding(picked[0], _snapshot(tmp_path))
        assert ev.recipe is not None, ev.rejection_reason
        _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "neg.py").read_text()
    assert "isinstance" not in after
    assert "issubclass" not in after
    assert " is PlanCmd" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    r1, r2 = ns["Ref"](), ns["OpenRef"]()  # type: ignore[operator]
    ns["admit_all"]([r1, r2])  # type: ignore[operator]
    assert (r1.admitted, r2.admitted) == (1, 0)  # type: ignore[attr-defined]
    log: list[str] = []
    ns["record_non_plan"](False, ns["Cmd"], log)  # type: ignore[operator]
    ns["record_non_plan"](True, ns["PlanCmd"], log)  # type: ignore[operator]
    ns["record_non_plan"](True, ns["Cmd"], log)  # type: ignore[operator]
    assert log == ["Cmd"]
    assert ns["route_cmd"](True, ns["Cmd"]) == "direct"  # type: ignore[operator]
    assert ns["route_cmd"](False, ns["PlanCmd"]) == "direct"  # type: ignore[operator]
    assert ns["route_cmd"](False, ns["Cmd"]) == "fallback"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_if_elif_else_with_mixed_early_returns_and_terminating_tail(tmp_path: Path):
    """An `if/elif/else` where some arms return early and others assign a local
    consumed by a terminating tail absorbs the tail into non-terminating arms."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/moves.py": (
                "class Move:\n"
                "    pass\n\n\n"
                "class Compose(Move):\n"
                "    pass\n\n\n"
                "class Extract(Move):\n"
                "    pass\n\n\n"
                "def build(move: type[Move], ok: bool) -> str:\n"
                "    if move is Compose:\n"
                "        if not ok:\n"
                "            return 'bad-compose'\n"
                "        label = 'composed'\n"
                "    elif move is Extract:\n"
                "        label = 'extracted'\n"
                "    else:\n"
                "        return 'unsupported'\n"
                "    return f'done:{label}'\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    ev = OwnTypeTestDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert ev.executable_recipe is not None, ev.rejection_reason
    _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "moves.py").read_text()
    assert " is Compose" not in after
    assert " is Extract" not in after
    assert "return move.build(ok)" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["build"](ns["Compose"], False) == "bad-compose"  # type: ignore[operator]
    assert ns["build"](ns["Compose"], True) == "done:composed"  # type: ignore[operator]
    assert ns["build"](ns["Extract"], False) == "done:extracted"  # type: ignore[operator]
    assert ns["build"](ns["Move"], True) == "unsupported"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_boolop_suffix_preserves_balanced_parens_around_trailing_boolop(tmp_path: Path):
    """`BoolOpSuffixDispatchSite` consumes the closing `)` of a parenthesized
    trailing `BoolOp` so `if not (a and isinstance(x, Sub) and (p or q)):` stays valid."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/parens.py": (
                "class Owner:\n"
                "    pass\n\n\n"
                "class DeclOwner(Owner):\n"
                "    pass\n\n\n"
                "def check(ready: bool, owner: Owner, pos: int | None) -> bool:\n"
                "    if not (\n"
                "        ready\n"
                "        and isinstance(owner, DeclOwner)\n"
                "        and (pos is None or pos > 0)\n"
                "    ):\n"
                "        return False\n"
                "    return True\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=finding.stable_id,
                owner="check",
                because="Owner owns position match",
                name="matches_pos",
                judgement="Return whether owner matches position.",
            ),
        ),
    )
    picked, _ = picks.applied((finding,))
    ev = OwnTypeTestDetector().evaluate_recipe_for_finding(picked[0], _snapshot(tmp_path))
    assert ev.recipe is not None, ev.rejection_reason
    _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "parens.py").read_text()
    assert "isinstance" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["check"](True, ns["DeclOwner"](), None) is True  # type: ignore[operator]
    assert ns["check"](True, ns["DeclOwner"](), 5) is True  # type: ignore[operator]
    assert ns["check"](True, ns["DeclOwner"](), 0) is False  # type: ignore[operator]
    assert ns["check"](True, ns["Owner"](), None) is False  # type: ignore[operator]
    assert ns["check"](False, ns["DeclOwner"](), None) is False  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_mro_dispatch_order_proof_fails_before_owner_pick_on_name_collision(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/order.py": (
                "class Base:\n"
                "    def handle(self) -> int:\n"
                "        return -1\n\n\n"
                "class Mid(Base):\n"
                "    pass\n\n\n"
                "class Leaf(Mid):\n"
                "    pass\n\n\n"
                "def handle(x: Base) -> int:\n"
                "    if isinstance(x, Mid):\n"
                "        return 1\n"
                "    elif isinstance(x, Leaf):\n"
                "        return 2\n"
                "    return 0\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    ev = OwnTypeTestDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert ev.recipe is None
    assert ev.needs == (NeededInput.PROOF,)
    assert "branch order" in (ev.rejection_reason or "")


def test_two_own_type_test_sites_in_one_function_compose_without_nominal_conflict(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/multi.py": (
                "class Left:\n"
                "    pass\n\n\n"
                "class LeftSpecial(Left):\n"
                "    pass\n\n\n"
                "class Right:\n"
                "    pass\n\n\n"
                "class RightSpecial(Right):\n"
                "    pass\n\n\n"
                "def score(left: Left, right: Right) -> int:\n"
                "    total = 1 if isinstance(left, LeftSpecial) else 0\n"
                "    total += 2 if isinstance(right, RightSpecial) else 0\n"
                "    return total\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    assert len(findings) == 2
    by_line = {f.primary_evidence.line: f for f in findings}
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=by_line[18].stable_id,
                owner="score",
                because="Left family owns its score contribution",
                name="left_score",
                judgement="Return this Left member's score contribution.",
            ),
            OwnerPick(
                finding=by_line[19].stable_id,
                owner="score",
                because="Right family owns its score contribution",
                name="right_score",
                judgement="Return this Right member's score contribution.",
            ),
        ),
    )
    picked, _ = picks.applied(findings)
    snapshot = _snapshot(tmp_path)
    plan = codemod_plan_from_findings(picked, selector_context=snapshot)
    assert plan.candidate_count == 2
    simulation = plan.document.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    after = (tmp_path / "pkg" / "multi.py").read_text()
    assert "isinstance" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["score"](ns["Left"](), ns["Right"]()) == 0  # type: ignore[operator]
    assert ns["score"](ns["LeftSpecial"](), ns["Right"]()) == 1  # type: ignore[operator]
    assert ns["score"](ns["Left"](), ns["RightSpecial"]()) == 2  # type: ignore[operator]
    assert ns["score"](ns["LeftSpecial"](), ns["RightSpecial"]()) == 3  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_two_own_type_tests_pushing_distinct_methods_onto_same_pass_class_family_compose_in_one_batch(
    tmp_path: Path,
) -> None:
    """Two dispatches targeting the exact same empty `pass`-body family compose in
    one batch: `pass` deletions coalesce and both methods insert commutatively."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/same_family.py": (
                "class Node:\n"
                "    pass\n\n\n"
                "class Special(Node):\n"
                "    pass\n\n\n"
                "def weight(node: Node) -> int:\n"
                "    return 10 if isinstance(node, Special) else 1\n\n\n"
                "def tag(node: Node) -> str:\n"
                "    return 'special' if isinstance(node, Special) else 'plain'\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    assert len(findings) == 2
    snapshot = _snapshot(tmp_path)
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    assert plan.candidate_count == 2
    sim = plan.document.simulate(snapshot)
    assert sim.is_clean
    sim.apply()

    after = (tmp_path / "pkg" / "same_family.py").read_text()
    assert "isinstance" not in after
    assert "    pass" not in after
    assert "def tag(self) -> str:" in after
    assert "def weight(self) -> int:" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["weight"](ns["Node"]()) == 1  # type: ignore[operator]
    assert ns["weight"](ns["Special"]()) == 10  # type: ignore[operator]
    assert ns["tag"](ns["Node"]()) == "plain"  # type: ignore[operator]
    assert ns["tag"](ns["Special"]()) == "special"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_partial_return_if_without_else_falls_through_to_terminating_tail(
    tmp_path: Path,
) -> None:
    """`if isinstance(x, Sub): if cond: return A` without `else:` followed by a
    terminating tail extends the arm with the tail and pushes onto the family."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/partial.py": (
                "class Evaluator:\n"
                "    def fallback(self, target: int) -> bool:\n"
                "        return target > 0\n\n\n"
                "class DelegatedEvaluator(Evaluator):\n"
                "    def shortcut(self, target: int) -> bool | None:\n"
                "        return True if target == 42 else None\n\n\n"
                "def check(evaluator: Evaluator, target: int) -> bool:\n"
                "    if isinstance(evaluator, DelegatedEvaluator):\n"
                "        possible = evaluator.shortcut(target)\n"
                "        if possible is not None:\n"
                "            return possible\n"
                "    return evaluator.fallback(target)\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    ev = OwnTypeTestDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert ev.executable_recipe is not None, ev.rejection_reason
    _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "partial.py").read_text()
    assert "isinstance" not in after
    assert "return evaluator.check(target)" in after
    assert "return super().check(target)" in after
    assert after.count("self.fallback(target)") == 1
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["check"](ns["Evaluator"](), 5) is True  # type: ignore[operator]
    assert ns["check"](ns["Evaluator"](), -1) is False  # type: ignore[operator]
    assert ns["check"](ns["DelegatedEvaluator"](), 42) is True  # type: ignore[operator]
    assert ns["check"](ns["DelegatedEvaluator"](), -1) is False  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_match_with_call_subject_subpatterns_and_guard_folds_and_evaluates_once(
    tmp_path: Path,
) -> None:
    """`match <call>:` with keyword subpatterns and a guard evaluates `<call>` once,
    folds single-arm base matches directly onto the base class, and pushes subclass
    matches onto the tested subclass."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/match_dispatch.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Payload:\n"
                "    value: int\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Lookup:\n"
                "    hit: bool\n"
                "    payload: Payload | None\n\n\n"
                "class Loader:\n"
                "    def __init__(self, lookup: Lookup) -> None:\n"
                "        self.lookup = lookup\n"
                "        self.calls = 0\n\n"
                "    def load(self) -> Lookup:\n"
                "        self.calls += 1\n"
                "        return self.lookup\n\n\n"
                "def resolve_payload(loader: Loader, fallback: int) -> Payload:\n"
                "    match loader.load():\n"
                "        case Lookup(hit=hit, payload=Payload() as payload) if hit:\n"
                "            return payload\n"
                "    return Payload(fallback)\n\n\n"
                "class CallExpr:\n"
                "    pass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class SetattrCall(CallExpr):\n"
                "    target: str\n\n\n"
                "def decode_call(name: str, counter: list[int]) -> CallExpr:\n"
                "    counter.append(1)\n"
                "    return SetattrCall(name) if name else CallExpr()\n\n\n"
                "def collect_type_setattr(name: str, counter: list[int], out: list[str]) -> None:\n"
                "    match decode_call(name, counter):\n"
                "        case SetattrCall(target=target) as call if target.endswith('type'):\n"
                "            out.append(f'{call.target}!')\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    # 3 findings: outer Lookup match, nested Payload subpattern on the same line, and SetattrCall match.
    # Only the outer Lookup and SetattrCall findings execute; nested Payload must neither piggyback nor collide.
    assert len(findings) == 3
    snap = _snapshot(tmp_path)
    plan = codemod_plan_from_findings(findings, selector_context=snap)
    assert plan.candidate_count == 2
    sim = plan.document.simulate(snap)
    assert sim.is_clean
    sim.apply()

    after = (tmp_path / "pkg" / "match_dispatch.py").read_text()
    assert "case Lookup(" not in after
    assert "case SetattrCall(" not in after
    assert "return loader.load().resolve_payload(fallback)" in after
    assert "decode_call(name, counter).collect_type_setattr(out)" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    Lookup = ns["Lookup"]
    Payload = ns["Payload"]
    Loader = ns["Loader"]
    loader_hit = Loader(Lookup(True, Payload(7)))  # type: ignore[operator]
    assert ns["resolve_payload"](loader_hit, 0) == Payload(7)  # type: ignore[operator]
    assert loader_hit.calls == 1
    loader_miss = Loader(Lookup(False, Payload(7)))  # type: ignore[operator]
    assert ns["resolve_payload"](loader_miss, 9) == Payload(9)  # type: ignore[operator]
    assert loader_miss.calls == 1
    counter: list[int] = []
    out: list[str] = []
    ns["collect_type_setattr"]("my_type", counter, out)  # type: ignore[operator]
    ns["collect_type_setattr"]("other", counter, out)  # type: ignore[operator]
    ns["collect_type_setattr"]("", counter, out)  # type: ignore[operator]
    assert counter == [1, 1, 1]
    assert out == ["my_type!"]
    remaining = _findings(tmp_path)
    assert len(remaining) == 1
    assert "Payload" in remaining[0].summary


def test_nested_function_dispatch_captures_enclosing_locals_and_outer_self(
    tmp_path: Path,
) -> None:
    """A type test inside a nested helper captures enclosing-function locals and
    aliases outer `self` cleanly instead of refusing `nested inside non-class`."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/nested.py": (
                "class Item:\n"
                "    pass\n\n\n"
                "class SpecialItem(Item):\n"
                "    pass\n\n\n"
                "class Formatter:\n"
                "    def __init__(self, tag: str) -> None:\n"
                "        self.tag = tag\n\n"
                "    def render_all(self, prefix: str, items: list[Item]) -> list[str]:\n"
                "        def format_one(item: Item) -> str:\n"
                "            if isinstance(item, SpecialItem):\n"
                "                return f'{prefix}:{self.tag}:special'\n"
                "            return f'{prefix}:{self.tag}:plain'\n"
                "        return [format_one(item) for item in items]\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    ev = OwnTypeTestDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert ev.executable_recipe is not None, ev.rejection_reason
    _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "nested.py").read_text()
    assert "isinstance" not in after
    assert "return item.format_one(prefix, self)" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    fmt = ns["Formatter"]("v1")  # type: ignore[operator]
    assert fmt.render_all("p", [ns["Item"](), ns["SpecialItem"]()]) == [
        "p:v1:plain",
        "p:v1:special",
    ]
    assert _findings(tmp_path) == []


def test_partial_return_fallthrough_skips_super_when_intermediate_arm_sits_between(
    tmp_path: Path,
) -> None:
    """When `Leaf -> Mid -> Base` has an arm on `Mid` and a partial-return arm on `Leaf`,
    `Leaf` must NOT call `super()` (which would dispatch to `Mid`), and instead inlines
    the base tail directly."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/chain.py": (
                "class Base:\n"
                "    pass\n\n\n"
                "class Mid(Base):\n"
                "    pass\n\n\n"
                "class Leaf(Mid):\n"
                "    pass\n\n\n"
                "def classify(x: Base, flag: bool) -> int:\n"
                "    if isinstance(x, Leaf):\n"
                "        if flag:\n"
                "            return 1\n"
                "    elif isinstance(x, Mid):\n"
                "        return 2\n"
                "    return 0\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    ev = OwnTypeTestDetector().evaluate_recipe_for_finding(finding, _snapshot(tmp_path))
    assert ev.executable_recipe is not None, ev.rejection_reason
    _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "chain.py").read_text()
    assert "isinstance" not in after
    assert "super()" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["classify"](ns["Leaf"](), True) == 1  # type: ignore[operator]
    assert ns["classify"](ns["Leaf"](), False) == 0  # type: ignore[operator]
    assert ns["classify"](ns["Mid"](), False) == 2  # type: ignore[operator]
    assert ns["classify"](ns["Base"](), False) == 0  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_defensive_not_isinstance_base_guard_is_not_folded_and_cross_module_imports_clean_up(
    tmp_path: Path,
) -> None:
    """(1) `if not isinstance(x, Base):` must not fold onto `Base` and drop its guard.
    (2) Moving a `match` body cross-module drops imports used only inside the moved
    subpattern/body (such as `Kind`) while keeping imports still used at the consumer."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/models.py": (
                "from dataclasses import dataclass\n"
                "from enum import Enum\n\n\n"
                "class Kind(Enum):\n"
                "    PLAIN = 1\n"
                "    OTHER = 2\n\n\n"
                "class Facet:\n"
                "    pass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Authority(Facet):\n"
                "    kind: Kind\n\n\n"
                "@dataclass(frozen=True)\n"
                "class CacheKey:\n"
                "    value: int\n"
            ),
            "pkg/consumer.py": (
                "from pkg.models import Authority, CacheKey, Facet, Kind\n\n\n"
                "def check_guard(key: CacheKey) -> str:\n"
                "    if not isinstance(key, CacheKey):\n"
                "        return 'corrupt'\n"
                "    return 'ok'\n\n\n"
                "def is_plain(facet: Facet) -> bool:\n"
                "    match facet:\n"
                "        case Authority(kind=Kind.PLAIN):\n"
                "            return True\n"
                "    return False\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"check_guard", "is_plain"}
    snap = _snapshot(tmp_path)

    guard_ev = OwnTypeTestDetector().evaluate_recipe_for_finding(by_sym["check_guard"], snap)
    assert guard_ev.recipe is None

    plain_ev = OwnTypeTestDetector().evaluate_recipe_for_finding(by_sym["is_plain"], snap)
    assert plain_ev.executable_recipe is not None, plain_ev.rejection_reason
    _apply(tmp_path, plain_ev.executable_recipe)

    consumer_after = (tmp_path / "pkg" / "consumer.py").read_text()
    assert "Authority" not in consumer_after
    assert "Kind" not in consumer_after
    assert "return facet.is_plain()" in consumer_after


def test_preserves_parameter_owner_and_return_annotations_and_skips_cyclic_imports(
    tmp_path: Path,
) -> None:
    """Pushed methods preserve parameter, annotated-local, `owner: EnclosingClass`
    (under PEP 563), and return annotations, auto-import cycle-free annotation types,
    and omit only annotations whose cross-module import would create a cycle."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/items.py": (
                "from __future__ import annotations\n\n\n"
                "class Item:\n"
                "    pass\n\n\n"
                "class SpecialItem(Item):\n"
                "    def __init__(self, tag: str) -> None:\n"
                "        self.tag = tag\n\n\n"
                "class SameModuleConsumer:\n"
                "    def __init__(self, prefix: str | None) -> None:\n"
                "        self.prefix = prefix\n\n"
                "    def format_item(self, item: Item, count: int) -> str:\n"
                "        if isinstance(item, SpecialItem):\n"
                "            return f'{self.prefix}:{item.tag}:{count}'\n"
                "        return f'{self.prefix}:plain:{count}'\n"
            ),
            "pkg/consumer.py": (
                "from __future__ import annotations\n"
                "from pathlib import Path\n"
                "from pkg.items import Item, SpecialItem\n\n\n"
                "class ExternalConsumer:\n"
                "    def __init__(self, suffix: str) -> None:\n"
                "        self.suffix = suffix\n\n"
                "    def item_path_label(self, item: Item, root: Path) -> str:\n"
                "        if isinstance(item, SpecialItem):\n"
                "            return f'{root}/{item.tag}{self.suffix}'\n"
                "        return f'{root}/plain{self.suffix}'\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {
        "SameModuleConsumer.format_item",
        "ExternalConsumer.item_path_label",
    }
    snap = _snapshot(tmp_path)
    plan = codemod_plan_from_findings(findings, selector_context=snap)
    assert plan.candidate_count == 2
    sim = plan.document.simulate(snap)
    assert sim.is_clean
    sim.apply()

    items_after = (tmp_path / "pkg" / "items.py").read_text()
    # Same-module push preserves `owner: SameModuleConsumer`, `count: int`, and `-> str`.
    assert (
        "def format_item(self, owner: SameModuleConsumer, count: int) -> str:"
        in items_after
    )
    # Cross-module push imports `Path` cleanly, preserves `root: Path` and `-> str`,
    # and omits `owner: ExternalConsumer` because importing `pkg.consumer` would cycle.
    assert "from pathlib import Path" in items_after
    assert "def item_path_label(self, root: Path, owner) -> str:" in items_after
    assert "from pkg.consumer" not in items_after
    assert _findings(tmp_path) == []




