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
    assert "def binding_label(self):" in bindings_after
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
    assert "def matches_carrier(self, owner):" in after
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
