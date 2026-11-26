"""Classes overriding one inherited method differently and taking the same step, found."""

from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._sibling_steps import (
    SiblingOverrideStepDetector,
)
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.sibling_override_steps import (
    HoistSiblingOverrideStepOperation,
)
from nominal_refactor_advisor.scan_cache import ScanCache

HOST = (
    "from abc import abstractmethod\n\n\n"
    "class App:\n    def history_displayed(self) -> None:\n        pass\n\n\n"
    "class Host:\n"
    "    @abstractmethod\n    def history_painted(self) -> None:\n        \"\"\"A publication painted the history.\"\"\"\n\n"
    "    @abstractmethod\n    def label(self) -> str: ...\n\n"
    "    def closed(self) -> None:\n        pass\n\n\n"
)
TEXT = (
    "class TextView(Host):\n"
    "    def __init__(self, app, submissions):\n        self.app, self.submissions = app, submissions\n\n"
    "    def history_painted(self) -> None:\n"
    "        self.retire_splash()\n        self.app.history_displayed()\n        self.submissions.settle()\n\n"
    "    def retire_splash(self) -> None:\n        pass\n\n"
    "    def label(self) -> str:\n        return 'text'\n\n\n"
)
QT = (
    "class QtView(Host):\n"
    "    def __init__(self, app, submissions):\n        self.app, self.submissions = app, submissions\n\n"
    "    def history_painted(self) -> None:\n        self.submissions.settle()\n\n"
    "    def opened(self) -> None:\n        self.app.show()\n\n"
    "    def label(self) -> str:\n        return 'qt'\n"
)


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _findings(root: Path):
    with ScanCache.scope():
        return SiblingOverrideStepDetector().detect(
            parse_python_modules(
                root, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False)
            ),
            DetectorConfig(),
        )


def test_sibling_override_step_continues_the_stable_sequence():
    assert PatternId.SIBLING_OVERRIDE_STEP == 59


def test_a_step_every_override_takes_and_one_a_class_leaves_out_are_one_finding(tmp_path: Path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + TEXT + QT})
    (finding,) = _findings(tmp_path)
    assert tuple(evidence.symbol for evidence in finding.evidence) == ("TextView.history_painted", "QtView.history_painted")
    assert "every one of them calls `self.submissions.settle()`" in finding.summary
    assert "`QtView` alone does not call `self.app.history_displayed()`, and it reads `self.app`" in finding.summary
    # A call on the receiver itself is the class's own helper, not a collaborator's step.
    assert "retire_splash" not in finding.summary
    # Overrides with nothing in common (`label`) are each class's own answer.
    assert "label" not in finding.summary


def test_a_member_the_other_class_never_reads_is_not_an_omitted_step(tmp_path: Path):
    qt = QT.replace("    def opened(self) -> None:\n        self.app.show()\n\n", "").replace(
        "        self.app, self.submissions = app, submissions\n\n    def history_painted(self) -> None:\n        self.submissions.settle()",
        "        self.submissions = submissions\n\n    def history_painted(self) -> None:\n        self.submissions.settle()",
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + TEXT + qt})
    (finding,) = _findings(tmp_path)
    assert "every one of them calls `self.submissions.settle()`" in finding.summary
    assert "alone does not call" not in finding.summary


def test_overrides_with_no_step_in_common_are_not_found(tmp_path: Path):
    qt = QT.replace("self.submissions.settle()", "self.repaint()")
    text = TEXT.replace("        self.app.history_displayed()\n        self.submissions.settle()\n", "")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + text + qt})
    assert not _findings(tmp_path)


def test_two_siblings_with_no_shared_step_do_not_report_omitted_steps(tmp_path: Path):
    """With only two independent overrides and zero shared steps, one backend calling a
    collaborator method is not consensus on an ancestor sequence; with three overrides,
    two agreeing still flags the third's omission."""

    qt = QT.replace("self.submissions.settle()", "self.repaint()")
    text = TEXT.replace("        self.submissions.settle()\n", "")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + text + qt})
    assert not _findings(tmp_path)

    third = (
        "\n\nclass WebView(Host):\n"
        "    def __init__(self, app, submissions):\n        self.app, self.submissions = app, submissions\n\n"
        "    def history_painted(self) -> None:\n"
        "        self.app.history_displayed()\n"
        "        self.flush()\n\n"
        "    def label(self) -> str:\n        return 'web'\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + text + qt + third})
    (finding,) = _findings(tmp_path)
    assert "`QtView` alone does not call `self.app.history_displayed()`" in finding.summary


def test_overrides_that_are_one_declaration_are_the_sibling_override_finding(tmp_path: Path):
    same = "    def history_painted(self) -> None:\n        self.submissions.settle()\n\n"
    text = TEXT.replace(
        "    def history_painted(self) -> None:\n"
        "        self.retire_splash()\n        self.app.history_displayed()\n        self.submissions.settle()\n\n",
        same,
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + text + QT})
    assert not _findings(tmp_path)


def test_a_class_overriding_through_another_is_compared_once(tmp_path: Path):
    wire = (
        "\n\nclass QtWireView(QtView):\n"
        "    def history_painted(self) -> None:\n        self.submissions.settle()\n        self.repaint()\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + TEXT + QT + wire})
    (finding,) = _findings(tmp_path)
    assert tuple(evidence.symbol for evidence in finding.evidence) == ("TextView.history_painted", "QtView.history_painted")


def test_overrides_that_each_call_something_else_on_one_member_are_each_class_own_spelling(tmp_path: Path):
    text = TEXT.replace("        self.app.history_displayed()\n", "        self.app.history_displayed()\n        self.bar.update('x')\n")
    qt = QT.replace("        self.submissions.settle()\n\n    def opened", "        self.submissions.settle()\n        self.bar.setText('x')\n        self.app.show()\n\n    def opened")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + text + qt})
    (finding,) = _findings(tmp_path)
    assert "every one of them calls `self.submissions.settle()`" in finding.summary
    assert "alone does not call" not in finding.summary


def test_a_call_into_a_library_is_not_a_step_another_class_omits(tmp_path: Path):
    """`history_displayed` is declared by a scanned class; `setVisible` by none."""
    qt = QT.replace("        self.submissions.settle()\n\n    def opened", "        self.submissions.settle()\n        self.panel.setVisible(True)\n\n    def opened")
    text = TEXT.replace("    def label(self) -> str:\n        return 'text'", "    def shown(self) -> None:\n        self.panel.refresh()\n\n    def label(self) -> str:\n        return 'text'")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + text + qt})
    (finding,) = _findings(tmp_path)
    assert "`QtView` alone does not call `self.app.history_displayed()`" in finding.summary
    assert "setVisible" not in finding.summary


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(
        parse_python_modules(
            root,
            use_parse_cache=False,
            source_policy=PythonSourcePathPolicy(include_tests=False),
        )
    )


def _apply_finding(root: Path):
    (finding,) = _findings(root)
    snapshot = _snapshot(root)
    evaluation = SiblingOverrideStepDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    return finding, snapshot, evaluation


def test_hoisting_trailing_step_creates_template_method_and_deletes_empty_sibling(
    tmp_path: Path,
):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + TEXT + QT})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    recipe = evaluation.executable_recipe
    (operation,) = recipe.operations
    assert isinstance(operation, HoistSiblingOverrideStepOperation)
    assert operation.member_name == "history_painted"
    assert operation.hook_name == "_history_painted_step"

    simulation = recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/views.py")]
    assert "def history_painted(self) -> None:" in rewritten
    assert "self._history_painted_step()" in rewritten
    assert "self.submissions.settle()" in rewritten
    assert "def _history_painted_step(self) -> None:\n        pass" in rewritten
    assert "def _history_painted_step(self) -> None:\n        self.retire_splash()" in rewritten
    # QtView's entire body was hoisted, so its override is deleted completely.
    qt_section = rewritten.split("class QtView(Host):")[1]
    assert "history_painted" not in qt_section

    # Execute the rewritten module and confirm runtime behavior matches identically.
    namespace: dict[str, object] = {}
    exec(compile(rewritten, "views.py", "exec"), namespace)
    calls: list[str] = []

    class FakeApp:
        def history_displayed(self) -> None:
            calls.append("displayed")

    class FakeSubmissions:
        def settle(self) -> None:
            calls.append("settle")

    text_view = namespace["TextView"](FakeApp(), FakeSubmissions())  # type: ignore[operator]
    qt_view = namespace["QtView"](FakeApp(), FakeSubmissions())  # type: ignore[operator]
    text_view.history_painted()
    assert calls == ["displayed", "settle"]
    calls.clear()
    qt_view.history_painted()
    assert calls == ["settle"]

    (tmp_path / "pkg/views.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_leading_early_return_guard_preserves_abstract_hook_and_return_value(
    tmp_path: Path,
):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Demand(ABC):\n"
        "    def accepts(self, end: int) -> bool:\n"
        "        return end > 0\n\n"
        "    @abstractmethod\n"
        "    def read_rows(self, end: int, limit: int = 10) -> int:\n"
        "        \"\"\"Rows to read.\"\"\"\n\n\n"
        "class MovingDemand(Demand):\n"
        "    def read_rows(self, end: int, limit: int = 10) -> int:\n"
        "        if not self.accepts(end):\n"
        "            return 0\n"
        "        return min(end, limit)\n\n\n"
        "class TargetDemand(Demand):\n"
        "    def read_rows(self, end: int, limit: int = 10) -> int:\n"
        "        if not self.accepts(end):\n"
        "            return 0\n"
        "        return end + limit\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/demands.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    recipe = evaluation.executable_recipe
    simulation = recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/demands.py")]
    assert "return self._read_rows_step(end, limit)" in rewritten
    assert "@abstractmethod\n    def _read_rows_step(self, end: int, limit: int = 10) -> int:" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "demands.py", "exec"), namespace)
    moving = namespace["MovingDemand"]()  # type: ignore[operator]
    target = namespace["TargetDemand"]()  # type: ignore[operator]
    assert moving.read_rows(0) == 0
    assert moving.read_rows(4, 10) == 4
    assert target.read_rows(0) == 0
    assert target.read_rows(4, 10) == 14


def test_hoisting_moves_required_imports_across_modules_and_cleans_obsolete_imports(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/constants.py": "def factor() -> int:\n    return 3\n",
            "pkg/base.py": (
                "from abc import ABC, abstractmethod\n\n\n"
                "class Base(ABC):\n"
                "    def scale(self, x: int) -> None:\n"
                "        self.value = x\n\n"
                "    @abstractmethod\n"
                "    def run(self) -> int:\n"
                "        ...\n"
            ),
            "pkg/one.py": (
                "from pkg.base import Base\n"
                "from pkg.constants import factor\n\n\n"
                "class One(Base):\n"
                "    def run(self) -> int:\n"
                "        self.scale(factor())\n"
                "        return 1\n"
            ),
            "pkg/two.py": (
                "from pkg.base import Base\n"
                "from pkg.constants import factor\n\n\n"
                "class Two(Base):\n"
                "    def run(self) -> int:\n"
                "        self.scale(factor())\n"
                "        return 2\n"
            ),
        },
    )
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten_base = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/base.py")]
    rewritten_one = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/one.py")]
    rewritten_two = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/two.py")]
    assert "from pkg.constants import factor" in rewritten_base
    assert "self.scale(factor())" in rewritten_base
    assert "factor" not in rewritten_one
    assert "factor" not in rewritten_two


def test_hoisting_supports_classmethod_and_property_descriptors(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Loader(ABC):\n"
        "    @classmethod\n"
        "    def validate(cls, raw: str) -> None:\n"
        "        if not raw:\n"
        "            raise ValueError('empty')\n\n"
        "    @classmethod\n"
        "    @abstractmethod\n"
        "    def load(cls, raw: str) -> str:\n"
        "        ...\n\n\n"
        "class JsonLoader(Loader):\n"
        "    @classmethod\n"
        "    def load(cls, raw: str) -> str:\n"
        "        cls.validate(raw)\n"
        "        return 'json:' + raw\n\n\n"
        "class TomlLoader(Loader):\n"
        "    @classmethod\n"
        "    def load(cls, raw: str) -> str:\n"
        "        cls.validate(raw)\n"
        "        return 'toml:' + raw\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/loaders.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/loaders.py")]
    assert "return cls._load_step(raw)" in rewritten
    assert "@classmethod\n    @abstractmethod\n    def _load_step(cls, raw: str) -> str:" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "loaders.py", "exec"), namespace)
    assert namespace["JsonLoader"].load("a") == "json:a"  # type: ignore[attr-defined]
    assert namespace["TomlLoader"].load("b") == "toml:b"  # type: ignore[attr-defined]

    prop_source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Panel(ABC):\n"
        "    def ready(self) -> bool:\n"
        "        return self.active\n\n"
        "    @property\n"
        "    @abstractmethod\n"
        "    def title(self) -> str:\n"
        "        ...\n\n\n"
        "class FirstPanel(Panel):\n"
        "    active = True\n\n"
        "    @property\n"
        "    def title(self) -> str:\n"
        "        if not self.ready():\n"
        "            return ''\n"
        "        return 'first'\n\n\n"
        "class SecondPanel(Panel):\n"
        "    active = False\n\n"
        "    @property\n"
        "    def title(self) -> str:\n"
        "        if not self.ready():\n"
        "            return ''\n"
        "        return 'second'\n"
    )
    prop_root = tmp_path / "prop"
    _write(prop_root, {"pkg/__init__.py": "", "pkg/panels.py": prop_source})
    _finding_p, snap_p, eval_p = _apply_finding(prop_root)
    sim_p = eval_p.executable_recipe.simulate(snap_p)
    assert sim_p.is_clean, sim_p.preflight_report
    rewritten_p = sim_p.simulation.rewritten_sources[str(prop_root / "pkg/panels.py")]
    assert "return self._title_step()" in rewritten_p
    ns_p: dict[str, object] = {}
    exec(compile(rewritten_p, "panels.py", "exec"), ns_p)
    assert ns_p["FirstPanel"]().title == "first"  # type: ignore[operator]
    assert ns_p["SecondPanel"]().title == ""  # type: ignore[operator]


def test_hoisting_forwards_prefix_bound_local_to_hook(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Provider(ABC):\n"
        "    def make_matcher(self, q: str) -> str:\n"
        "        return q.lower()\n\n"
        "    @abstractmethod\n"
        "    def search(self, query: str) -> str:\n"
        "        ...\n\n\n"
        "class First(Provider):\n"
        "    def search(self, query: str) -> str:\n"
        "        matcher = self.make_matcher(query)\n"
        "        return matcher + '!'\n\n\n"
        "class Second(Provider):\n"
        "    def search(self, query: str) -> str:\n"
        "        matcher = self.make_matcher(query)\n"
        "        return matcher + '?'\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/providers.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/providers.py")]
    assert "matcher = self.make_matcher(query)" in rewritten
    assert "return self._search_step(query, matcher)" in rewritten
    assert "def _search_step(self, query: str, matcher) -> str:" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "providers.py", "exec"), namespace)
    assert namespace["First"]().search("Ab") == "ab!"  # type: ignore[operator]
    assert namespace["Second"]().search("Ab") == "ab?"  # type: ignore[operator]

    (tmp_path / "pkg/providers.py").write_text(rewritten)
    assert _findings(tmp_path) == []

    # When the declaring method has defaults, forwarded locals become keyword-only.
    defaulted = source.replace("query: str", "query: str = 'hi'")
    default_root = tmp_path / "defaulted"
    _write(default_root, {"pkg/__init__.py": "", "pkg/providers.py": defaulted})
    _f2, snap2, eval2 = _apply_finding(default_root)
    sim2 = eval2.executable_recipe.simulate(snap2)
    assert sim2.is_clean, sim2.preflight_report
    rewritten2 = sim2.simulation.rewritten_sources[str(default_root / "pkg/providers.py")]
    assert "return self._search_step(query, matcher=matcher)" in rewritten2
    assert "def _search_step(self, query: str='hi', *, matcher) -> str:" in rewritten2
    ns2: dict[str, object] = {}
    exec(compile(rewritten2, "providers.py", "exec"), ns2)
    assert ns2["First"]().search() == "hi!"  # type: ignore[operator]
    assert ns2["Second"]().search("YO") == "yo?"  # type: ignore[operator]


def test_hoisting_stops_when_prefix_local_is_only_conditionally_bound(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Base(ABC):\n"
        "    def prepare(self) -> int:\n"
        "        return 1\n\n"
        "    @abstractmethod\n"
        "    def run(self, flag: bool) -> int:\n"
        "        ...\n\n\n"
        "class One(Base):\n"
        "    def run(self, flag: bool) -> int:\n"
        "        if flag:\n"
        "            val = self.prepare()\n"
        "        return val if flag else 0\n\n\n"
        "class Two(Base):\n"
        "    def run(self, flag: bool) -> int:\n"
        "        if flag:\n"
        "            val = self.prepare()\n"
        "        return val + 1 if flag else 2\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/cond.py": source})
    _finding, _snapshot, evaluation = _apply_finding(tmp_path)
    assert "not independent leading or trailing statements" in evaluation.rejection_reason
    assert NeededInput.SOURCE_CHANGE in evaluation.needs


def test_hoisting_leading_unconditional_step_expression_from_return(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Outcome(ABC):\n"
        "    def display_name(self) -> str:\n"
        "        return 'worker'\n\n"
        "    @abstractmethod\n"
        "    def start(self) -> tuple[str, ...]:\n"
        "        ...\n\n\n"
        "class Completed(Outcome):\n"
        "    def start(self) -> tuple[str, ...]:\n"
        "        return ('ok', self.display_name())\n\n\n"
        "class Failed(Outcome):\n"
        "    def start(self) -> tuple[str, ...]:\n"
        "        return ('fail', self.display_name(), 'err')\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/outcomes.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/outcomes.py")]
    assert "display_name = self.display_name()" in rewritten
    assert "return self._start_step(display_name)" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "outcomes.py", "exec"), namespace)
    assert namespace["Completed"]().start() == ("ok", "worker")  # type: ignore[operator]
    assert namespace["Failed"]().start() == ("fail", "worker", "err")  # type: ignore[operator]

    (tmp_path / "pkg/outcomes.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_stops_when_impure_call_precedes_shared_step_expression(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Worker(ABC):\n"
        "    def step(self) -> int:\n"
        "        return 1\n\n"
        "    def side_effect(self) -> int:\n"
        "        return 2\n\n"
        "    @abstractmethod\n"
        "    def run(self) -> int:\n"
        "        ...\n\n\n"
        "class First(Worker):\n"
        "    def run(self) -> int:\n"
        "        return self.side_effect() + self.step()\n\n\n"
        "class Second(Worker):\n"
        "    def run(self) -> int:\n"
        "        return self.step() + 10\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/order.py": source})
    _finding, _snapshot, evaluation = _apply_finding(tmp_path)
    assert "not independent leading or trailing statements" in evaluation.rejection_reason


def test_hoisting_generator_methods_delegates_with_yield_from(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n"
        "from collections.abc import Iterator\n\n\n"
        "class Source(ABC):\n"
        "    def observed_directory(self) -> str:\n"
        "        return '/tmp/dir'\n\n"
        "    @abstractmethod\n"
        "    def watches(self) -> Iterator[str]:\n"
        "        ...\n\n\n"
        "class FileSource(Source):\n"
        "    def watches(self) -> Iterator[str]:\n"
        "        yield f'file:{self.observed_directory()}'\n"
        "        yield 'extra'\n\n\n"
        "class DirSource(Source):\n"
        "    def watches(self) -> Iterator[str]:\n"
        "        directory = self.observed_directory()\n"
        "        yield f'dir:{directory}'\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/sources.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/sources.py")]
    assert "directory = self.observed_directory()" in rewritten
    assert "yield from self._watches_step(directory)" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "sources.py", "exec"), namespace)
    assert list(namespace["FileSource"]().watches()) == ["file:/tmp/dir", "extra"]  # type: ignore[operator]
    assert list(namespace["DirSource"]().watches()) == ["dir:/tmp/dir"]  # type: ignore[operator]

    (tmp_path / "pkg/sources.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_promotes_abstract_property_when_all_siblings_use_cached_property(
    tmp_path: Path,
):
    source = (
        "from abc import ABC, abstractmethod\n"
        "from functools import cached_property\n\n\n"
        "class Contents(ABC):\n"
        "    def __init__(self) -> None:\n"
        "        self.checks = 0\n\n"
        "    def require_closed(self) -> None:\n"
        "        self.checks += 1\n\n"
        "    @property\n"
        "    @abstractmethod\n"
        "    def names(self) -> tuple[str, ...]:\n"
        "        ...\n\n\n"
        "class InitialContents(Contents):\n"
        "    @cached_property\n"
        "    def names(self) -> tuple[str, ...]:\n"
        "        self.require_closed()\n"
        "        return ('a',)\n\n\n"
        "class FullContents(Contents):\n"
        "    @cached_property\n"
        "    def names(self) -> tuple[str, ...]:\n"
        "        self.require_closed()\n"
        "        return ('a', 'b')\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/contents.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/contents.py")]
    assert "@cached_property\n    def names(self) -> tuple[str, ...]:" in rewritten
    assert "return self._names_step()" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "contents.py", "exec"), namespace)
    initial = namespace["InitialContents"]()  # type: ignore[operator]
    assert initial.names == ("a",)
    assert initial.names == ("a",)
    assert initial.checks == 1
    full = namespace["FullContents"]()  # type: ignore[operator]
    assert full.names == ("a", "b")
    assert full.checks == 1

    (tmp_path / "pkg/contents.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_return_call_template_substitutes_single_varying_argument(
    tmp_path: Path,
):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class PickerPane(ABC):\n"
        "    def query_one(self, target: str) -> str:\n"
        "        return f'found:{target}'\n\n"
        "    @abstractmethod\n"
        "    def toggled(self) -> str:\n"
        "        ...\n\n\n"
        "class FuzzyPane(PickerPane):\n"
        "    def toggled(self) -> str:\n"
        "        return self.query_one('tree')\n\n\n"
        "class TreePane(PickerPane):\n"
        "    def toggled(self) -> str:\n"
        "        return self.query_one('fuzzy')\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/picker.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/picker.py")]
    assert "return self.query_one(self._toggled_step())" in rewritten
    assert "def _toggled_step(self):\n" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "picker.py", "exec"), namespace)
    assert namespace["FuzzyPane"]().toggled() == "found:tree"  # type: ignore[operator]
    assert namespace["TreePane"]().toggled() == "found:fuzzy"  # type: ignore[operator]

    (tmp_path / "pkg/picker.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_stops_when_middle_returns_early_before_trailing_step(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Base(ABC):\n"
        "    def settle(self) -> None:\n"
        "        pass\n\n"
        "    @abstractmethod\n"
        "    def finish(self, skip: bool) -> None:\n"
        "        ...\n\n\n"
        "class One(Base):\n"
        "    def finish(self, skip: bool) -> None:\n"
        "        if skip:\n"
        "            return\n"
        "        self.settle()\n\n\n"
        "class Two(Base):\n"
        "    def finish(self, skip: bool) -> None:\n"
        "        self.settle()\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/finish.py": source})
    _finding, _snapshot, evaluation = _apply_finding(tmp_path)
    assert "not independent leading or trailing statements" in evaluation.rejection_reason


def test_hoisting_stops_when_descendant_calls_super_or_inherits_without_overriding(
    tmp_path: Path,
):
    super_source = (
        HOST
        + TEXT
        + QT
        + "\n\nclass ExtraQt(QtView):\n"
        "    def history_painted(self) -> None:\n"
        "        super().history_painted()\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": super_source})
    _finding, _snapshot, evaluation = _apply_finding(tmp_path)
    assert "calls `super().history_painted()`" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs

    unoverridden = (
        "class Base:\n"
        "    def notify(self) -> None:\n"
        "        pass\n\n"
        "    def run(self) -> None:\n"
        "        pass\n\n\n"
        "class One(Base):\n"
        "    def run(self) -> None:\n"
        "        self.notify()\n"
        "        self.a = 1\n\n\n"
        "class Two(Base):\n"
        "    def run(self) -> None:\n"
        "        self.notify()\n"
        "        self.b = 2\n\n\n"
        "class Bystander(Base):\n"
        "    pass\n"
    )
    other_root = tmp_path / "other"
    _write(other_root, {"pkg/__init__.py": "", "pkg/run.py": unoverridden})
    _finding2, _snapshot2, evaluation2 = _apply_finding(other_root)
    assert "`Bystander` inherits `Base.run` without overriding it" in evaluation2.rejection_reason
    assert NeededInput.PROOF in evaluation2.needs


def test_hoisting_stops_on_dunder_method_and_hook_collision(tmp_path: Path):
    colliding = HOST.replace(
        "    def closed(self) -> None:\n        pass",
        "    def _history_painted_step(self) -> None:\n        pass",
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": colliding + TEXT + QT})
    _finding, _snapshot, evaluation = _apply_finding(tmp_path)
    assert "already binds hook name `_history_painted_step`" in evaluation.rejection_reason
    assert NeededInput.SOURCE_CHANGE in evaluation.needs


def test_hoisting_leading_step_inside_subscript_slice(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Registry(ABC):\n"
        "    def canonical_name(self, name: str) -> str:\n"
        "        return name.strip().lower()\n\n"
        "    @abstractmethod\n"
        "    def generation(self, name: str) -> int:\n"
        "        ...\n\n\n"
        "class LiveRegistry(Registry):\n"
        "    def __init__(self, gens: dict[str, int]) -> None:\n"
        "        self.live = gens\n\n"
        "    def generation(self, name: str) -> int:\n"
        "        return self.live[self.canonical_name(name)]\n\n\n"
        "class SnapshotRegistry(Registry):\n"
        "    def __init__(self, gens: dict[str, int]) -> None:\n"
        "        self.saved = gens\n\n"
        "    def generation(self, name: str) -> int:\n"
        "        return self.saved[self.canonical_name(name)]\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/registry.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/registry.py")]
    assert "canonical_name = self.canonical_name(name)" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "registry.py", "exec"), namespace)
    assert namespace["LiveRegistry"]({"a": 3}).generation(" A ") == 3  # type: ignore[operator]
    assert namespace["SnapshotRegistry"]({"a": 7}).generation(" A ") == 7  # type: ignore[operator]

    (tmp_path / "pkg/registry.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_void_trailing_call_template_with_prep_statements(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Bus:\n"
        "    def __init__(self) -> None:\n"
        "        self.items: list[str] = []\n\n"
        "    def publish(self, item: str) -> None:\n"
        "        self.items.append(item)\n\n\n"
        "class ContentEffect(ABC):\n"
        "    def __init__(self, bus: Bus) -> None:\n"
        "        self.bus = bus\n\n"
        "    @abstractmethod\n"
        "    def emit(self, text: str) -> None:\n"
        "        ...\n\n\n"
        "class UserEffect(ContentEffect):\n"
        "    def emit(self, text: str) -> None:\n"
        "        self.bus.publish(f'user:{text}')\n\n\n"
        "class AgentEffect(ContentEffect):\n"
        "    def emit(self, text: str) -> None:\n"
        "        formatted = text\n"
        "        if formatted.startswith('err:'):\n"
        "            formatted = formatted + '!'\n"
        "        self.bus.publish(f'agent:{formatted}')\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/effects.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/effects.py")]
    assert "_step_value = self._emit_step(text)" in rewritten
    assert "self.bus.publish(_step_value)" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "effects.py", "exec"), namespace)
    bus = namespace["Bus"]()  # type: ignore[operator]
    namespace["UserEffect"](bus).emit("hi")  # type: ignore[operator]
    namespace["AgentEffect"](bus).emit("err:boom")  # type: ignore[operator]
    assert bus.items == ["user:hi", "agent:err:boom!"]  # type: ignore[attr-defined]

    (tmp_path / "pkg/effects.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_trailing_boolop_template_preserves_short_circuit(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Rule(ABC):\n"
        "    def __init__(self) -> None:\n"
        "        self.scope_calls = 0\n\n"
        "    def scope_applies(self, owner: str) -> bool:\n"
        "        self.scope_calls += 1\n"
        "        return owner == 'ok'\n\n"
        "    @abstractmethod\n"
        "    def applies(self, owner: str, tag: str) -> bool:\n"
        "        ...\n\n\n"
        "class AuthorRule(Rule):\n"
        "    def applies(self, owner: str, tag: str) -> bool:\n"
        "        return tag == 'author' and self.scope_applies(owner)\n\n\n"
        "class RecipientRule(Rule):\n"
        "    def applies(self, owner: str, tag: str) -> bool:\n"
        "        return tag == 'recipient' and owner != '' and self.scope_applies(owner)\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/rules.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/rules.py")]
    assert "return self._applies_step(owner, tag) and self.scope_applies(owner)" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "rules.py", "exec"), namespace)
    author = namespace["AuthorRule"]()  # type: ignore[operator]
    assert author.applies("ok", "wrong") is False
    assert author.scope_calls == 0
    assert author.applies("ok", "author") is True
    assert author.scope_calls == 1
    recip = namespace["RecipientRule"]()  # type: ignore[operator]
    assert recip.applies("", "recipient") is False
    assert recip.scope_calls == 0
    assert recip.applies("ok", "recipient") is True
    assert recip.scope_calls == 1

    (tmp_path / "pkg/rules.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_trailing_guarded_step_template(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Resolution(ABC):\n"
        "    def __init__(self) -> None:\n"
        "        self.released = 0\n"
        "        self.admitted = 0\n\n"
        "    def require_release(self) -> None:\n"
        "        self.released += 1\n\n"
        "    def require_admitted(self) -> None:\n"
        "        self.admitted += 1\n\n"
        "    @abstractmethod\n"
        "    def check(self, keep: bool) -> None:\n"
        "        ...\n\n\n"
        "class NativeRes(Resolution):\n"
        "    def check(self, keep: bool) -> None:\n"
        "        if not keep:\n"
        "            self.require_release()\n\n\n"
        "class DictRes(Resolution):\n"
        "    def check(self, keep: bool) -> None:\n"
        "        self.require_admitted()\n"
        "        if keep:\n"
        "            self.require_release()\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/res.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/res.py")]
    assert "if self._check_step(keep):\n            self.require_release()" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "res.py", "exec"), namespace)
    native = namespace["NativeRes"]()  # type: ignore[operator]
    native.check(True)
    assert native.released == 0
    native.check(False)
    assert native.released == 1
    created = namespace["DictRes"]()  # type: ignore[operator]
    created.check(True)
    assert (created.admitted, created.released) == (1, 1)

    (tmp_path / "pkg/res.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_guarded_early_return_before_shared_tail(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Attachment(ABC):\n"
        "    def root_source(self, msg: str) -> str:\n"
        "        return f'root:{msg}'\n\n"
        "    @abstractmethod\n"
        "    def current_roots(self, msg: str, allow: bool) -> str | None:\n"
        "        ...\n\n\n"
        "class Supersession(Attachment):\n"
        "    def current_roots(self, msg: str, allow: bool) -> str | None:\n"
        "        if not allow:\n"
        "            return None\n"
        "        return self.root_source(msg)\n\n\n"
        "class Scoped(Attachment):\n"
        "    def current_roots(self, msg: str, allow: bool) -> str | None:\n"
        "        if not msg:\n"
        "            return None\n"
        "        return self.root_source(msg)\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/attach.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/attach.py")]
    assert "if self._current_roots_step(msg, allow):\n            return None" in rewritten
    assert "return self.root_source(msg)" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "attach.py", "exec"), namespace)
    assert namespace["Supersession"]().current_roots("m", False) is None  # type: ignore[operator]
    assert namespace["Supersession"]().current_roots("m", True) == "root:m"  # type: ignore[operator]
    assert namespace["Scoped"]().current_roots("", True) is None  # type: ignore[operator]
    assert namespace["Scoped"]().current_roots("x", False) == "root:x"  # type: ignore[operator]

    (tmp_path / "pkg/attach.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_variadic_return_call_template_preserves_prep_order_and_arity(
    tmp_path: Path,
):
    """Siblings ending with the same receiver call at differing positional arities
    (e.g. `DirectiveVerb.respond` calling `self.applied(...)` with 1 or 2 args) hoist
    into `return self.applied(*_step_args)` with prep-first evaluation order."""

    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Verb(ABC):\n"
        "    def __init__(self) -> None:\n"
        "        self.events: list[str] = []\n\n"
        "    def applied(self, *items: str) -> tuple[str, ...]:\n"
        "        self.events.append('applied')\n"
        "        return items\n\n"
        "    @abstractmethod\n"
        "    def respond(self, tag: str) -> tuple[str, ...]:\n"
        "        ...\n\n\n"
        "class SendVerb(Verb):\n"
        "    def respond(self, tag: str) -> tuple[str, ...]:\n"
        "        self.events.append(f'send:{tag}')\n"
        "        return self.applied(f'order:{tag}', f'msg:{tag}')\n\n\n"
        "class StrikeVerb(Verb):\n"
        "    def respond(self, tag: str) -> tuple[str, ...]:\n"
        "        self.events.append(f'strike:{tag}')\n"
        "        return self.applied(f'problem:{tag}')\n\n\n"
        "class EmptyVerb(Verb):\n"
        "    def respond(self, tag: str) -> tuple[str, ...]:\n"
        "        return self.applied()\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/verbs.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/verbs.py")]
    assert "_step_args = self._respond_step(tag)" in rewritten
    assert "return self.applied(*_step_args)" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "verbs.py", "exec"), namespace)
    send = namespace["SendVerb"]()  # type: ignore[operator]
    assert send.respond("x") == ("order:x", "msg:x")
    assert send.events == ["send:x", "applied"]
    strike = namespace["StrikeVerb"]()  # type: ignore[operator]
    assert strike.respond("y") == ("problem:y",)
    assert strike.events == ["strike:y", "applied"]
    empty = namespace["EmptyVerb"]()  # type: ignore[operator]
    assert empty.respond("z") == ()
    assert empty.events == ["applied"]

    (tmp_path / "pkg/verbs.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_multi_arg_return_call_with_fixed_keyword_and_rejecting_varying_keyword(
    tmp_path: Path,
):
    """Fixed-arity calls with 2+ varying positional args and identical fixed keyword
    arguments hoist via `*self._hook(...)`; varying keyword arguments remain refused."""

    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Formatter(ABC):\n"
        "    def join(self, a: str, b: str, *, sep: str = '-') -> str:\n"
        "        return f'{a}{sep}{b}'\n\n"
        "    @abstractmethod\n"
        "    def render(self, x: str) -> str:\n"
        "        ...\n\n\n"
        "class LeftFmt(Formatter):\n"
        "    def render(self, x: str) -> str:\n"
        "        return self.join('L', x, sep=':')\n\n\n"
        "class RightFmt(Formatter):\n"
        "    def render(self, x: str) -> str:\n"
        "        return self.join(x, 'R', sep=':')\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/fmt.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/fmt.py")]
    assert "return self.join(*self._render_step(x), sep=':')" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "fmt.py", "exec"), namespace)
    assert namespace["LeftFmt"]().render("v") == "L:v"  # type: ignore[operator]
    assert namespace["RightFmt"]().render("v") == "v:R"  # type: ignore[operator]

    (tmp_path / "pkg/fmt.py").write_text(rewritten)
    assert _findings(tmp_path) == []

    # Varying keyword values must stay refused.
    bad_source = source.replace("sep=':')", "sep='/')", 1)
    bad_dir = tmp_path / "bad"
    _write(bad_dir, {"pkg/__init__.py": "", "pkg/fmt.py": bad_source})
    (bad_finding,) = _findings(bad_dir)
    bad_snap = CodemodSourceSnapshot.from_modules(
        parse_python_modules(
            bad_dir,
            use_parse_cache=False,
            source_policy=PythonSourcePathPolicy(include_tests=False),
        )
    )
    bad_eval = SiblingOverrideStepDetector().evaluate_recipe_for_finding(
        bad_finding, bad_snap
    )
    assert bad_eval.recipe is None
    assert "not independent leading or trailing statements" in bad_eval.rejection_reason


def test_hoisting_concrete_base_with_shared_prefix_and_non_overriding_descendant(
    tmp_path: Path,
):
    """When a concrete base method shares the same leading step as its overriding
    siblings, hoisting preserves the base's own remaining middle as a concrete
    default hook so direct callers and non-overriding descendants keep identical
    behavior."""

    source = (
        "class Gate:\n"
        "    def __init__(self) -> None:\n"
        "        self.events: list[str] = []\n"
        "        self.last_kind: str = ''\n\n"
        "    def template_once(self, tag: str) -> str:\n"
        "        self.events.append(f'tpl:{tag}')\n"
        "        return f'[{tag}]'\n\n"
        "    def blocked(self, tag: str) -> str:\n"
        "        tpl = self.template_once(tag)\n"
        "        self.last_kind = 'base'\n"
        "        return f'base:{tpl}'\n\n\n"
        "class InheritingGate(Gate):\n"
        "    pass\n\n\n"
        "class FastGate(Gate):\n"
        "    def blocked(self, tag: str) -> str:\n"
        "        tpl = self.template_once(tag)\n"
        "        self.last_kind = 'fast'\n"
        "        return f'fast:{tpl}'\n\n\n"
        "class AuditedGate(Gate):\n"
        "    def blocked(self, tag: str) -> str:\n"
        "        tpl = self.template_once(tag)\n"
        "        self.last_kind = 'audited'\n"
        "        return f'audited:{tpl}'\n\n\n"
        "def run_direct(tag: str) -> str:\n"
        "    return Gate().blocked(tag)\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/gates.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/gates.py")]
    assert "return self._blocked_step(tag, tpl)" in rewritten
    assert "def _blocked_step(self, tag: str, tpl) -> str:" in rewritten

    ns: dict[str, object] = {}
    exec(compile(rewritten, "gates.py", "exec"), ns)
    assert ns["run_direct"]("x") == "base:[x]"  # type: ignore[operator]
    inh = ns["InheritingGate"]()  # type: ignore[operator]
    assert inh.blocked("a") == "base:[a]"
    assert inh.events == ["tpl:a"]
    assert inh.last_kind == "base"
    fast = ns["FastGate"]()  # type: ignore[operator]
    assert fast.blocked("b") == "fast:[b]"
    assert fast.events == ["tpl:b"]
    assert fast.last_kind == "fast"
    aud = ns["AuditedGate"]()  # type: ignore[operator]
    assert aud.blocked("c") == "audited:[c]"
    assert aud.events == ["tpl:c"]
    assert aud.last_kind == "audited"

    (tmp_path / "pkg/gates.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_concrete_base_default_hook_for_guarded_templates(tmp_path: Path):
    """Concrete base methods matching `TrailingGuardedStepTemplate` (either the
    guarded step -> `return True`, or no-op -> `return False`) or
    `GuardedEarlyReturnSuffixTemplate` (`early_return_body` -> `return True`)
    hoist cleanly even with direct base callers and non-overriding descendants."""

    source = (
        "class Resolution:\n"
        "    def __init__(self) -> None:\n"
        "        self.released = 0\n\n"
        "    def require_release(self) -> None:\n"
        "        self.released += 1\n\n"
        "    def require_release_in(self, retained: bool) -> None:\n"
        "        self.require_release()\n\n\n"
        "class PlainResolution(Resolution):\n"
        "    pass\n\n\n"
        "class NativeResolution(Resolution):\n"
        "    def require_release_in(self, retained: bool) -> None:\n"
        "        if not retained:\n"
        "            self.require_release()\n\n\n"
        "class DictResolution(Resolution):\n"
        "    def require_release_in(self, retained: bool) -> None:\n"
        "        if retained:\n"
        "            self.require_release()\n\n\n"
        "class Attachment:\n"
        "    def root_source(self, msg: str) -> str:\n"
        "        return f'root:{msg}'\n\n"
        "    def current_roots(self, msg: str, allow: bool) -> str | None:\n"
        "        return None\n\n\n"
        "class PlainAttachment(Attachment):\n"
        "    pass\n\n\n"
        "class Supersession(Attachment):\n"
        "    def current_roots(self, msg: str, allow: bool) -> str | None:\n"
        "        if not allow:\n"
        "            return None\n"
        "        return self.root_source(msg)\n\n\n"
        "class Scoped(Attachment):\n"
        "    def current_roots(self, msg: str, allow: bool) -> str | None:\n"
        "        if not msg:\n"
        "            return None\n"
        "        return self.root_source(msg)\n\n\n"
        "def direct_attachment(msg: str) -> str | None:\n"
        "    return Attachment().current_roots(msg, True)\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/concrete.py": source})
    for _ in range(2):
        findings = _findings(tmp_path)
        assert findings
        snap = CodemodSourceSnapshot.from_modules(
            parse_python_modules(
                tmp_path,
                use_parse_cache=False,
                source_policy=PythonSourcePathPolicy(include_tests=False),
            )
        )
        ev = SiblingOverrideStepDetector().evaluate_recipe_for_finding(findings[0], snap)
        sim = ev.executable_recipe.simulate(snap)
        assert sim.is_clean, sim.preflight_report
        sim.apply()

    rewritten = (tmp_path / "pkg/concrete.py").read_text()
    assert "return True" in rewritten
    assert "if self._require_release_in_step(retained):" in rewritten
    assert "if self._current_roots_step(msg, allow):" in rewritten

    ns: dict[str, object] = {}
    exec(compile(rewritten, "concrete.py", "exec"), ns)
    plain_res = ns["PlainResolution"]()  # type: ignore[operator]
    plain_res.require_release_in(True)
    assert plain_res.released == 1
    nat = ns["NativeResolution"]()  # type: ignore[operator]
    nat.require_release_in(True)
    assert nat.released == 0
    nat.require_release_in(False)
    assert nat.released == 1

    assert ns["direct_attachment"]("hi") is None  # type: ignore[operator]
    assert ns["PlainAttachment"]().current_roots("hi", True) is None  # type: ignore[operator]
    assert ns["Supersession"]().current_roots("hi", False) is None  # type: ignore[operator]
    assert ns["Supersession"]().current_roots("hi", True) == "root:hi"  # type: ignore[operator]
    assert ns["Scoped"]().current_roots("", True) is None  # type: ignore[operator]
    assert ns["Scoped"]().current_roots("x", False) == "root:x"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_hoisting_shared_if_guard_template_with_walrus_and_shared_for_loop_template(
    tmp_path: Path,
):
    """`if (found := cls.found(text)): <body_i>` and `for item in cls.items(src): <body_i>`
    (with outer-loop `continue` rewritten to `return`, preserving nested-loop `continue`)
    hoist cleanly onto their ancestor."""

    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Words(ABC):\n"
        "    @classmethod\n"
        "    def found(cls, text: str) -> list[str]:\n"
        "        return [w for w in text.split() if w.isupper()]\n\n"
        "    @classmethod\n"
        "    @abstractmethod\n"
        "    def flag(cls, text: str, out: list[str]) -> None:\n"
        "        ...\n\n\n"
        "class PlainWords(Words):\n"
        "    @classmethod\n"
        "    def flag(cls, text: str, out: list[str]) -> None:\n"
        "        if (hits := cls.found(text)):\n"
        "            out.append('plain:' + ','.join(hits))\n\n\n"
        "class OrderWords(Words):\n"
        "    @classmethod\n"
        "    def flag(cls, text: str, out: list[str]) -> None:\n"
        "        if (hits := cls.found(text)):\n"
        "            out.append('order:' + ','.join(hits))\n\n\n"
        "class Store(ABC):\n"
        "    @classmethod\n"
        "    def resources(cls, items: list[str]) -> list[str]:\n"
        "        return [item for item in items if item]\n\n"
        "    @classmethod\n"
        "    @abstractmethod\n"
        "    def process(cls, items: list[str], out: list[str]) -> None:\n"
        "        ...\n\n\n"
        "class SkippingStore(Store):\n"
        "    @classmethod\n"
        "    def process(cls, items: list[str], out: list[str]) -> None:\n"
        "        for res in cls.resources(items):\n"
        "            if res.startswith('skip:'):\n"
        "                continue\n"
        "            for ch in res:\n"
        "                if ch == '-':\n"
        "                    continue\n"
        "                out.append(ch)\n\n\n"
        "class DirectStore(Store):\n"
        "    @classmethod\n"
        "    def process(cls, items: list[str], out: list[str]) -> None:\n"
        "        for res in cls.resources(items):\n"
        "            out.append(res.upper())\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/guards.py": source})
    for _ in range(2):
        findings = _findings(tmp_path)
        assert findings
        snap = CodemodSourceSnapshot.from_modules(
            parse_python_modules(
                tmp_path,
                use_parse_cache=False,
                source_policy=PythonSourcePathPolicy(include_tests=False),
            )
        )
        ev = SiblingOverrideStepDetector().evaluate_recipe_for_finding(findings[0], snap)
        sim = ev.executable_recipe.simulate(snap)
        assert sim.is_clean, sim.preflight_report
        sim.apply()

    rewritten = (tmp_path / "pkg/guards.py").read_text()
    assert "if (hits := cls.found(text)):" in rewritten
    assert "cls._flag_step(text, out, hits)" in rewritten
    assert "for res in cls.resources(items):" in rewritten
    assert "cls._process_step(items, out, res)" in rewritten

    ns: dict[str, object] = {}
    exec(compile(rewritten, "guards.py", "exec"), ns)
    w_out: list[str] = []
    ns["PlainWords"].flag("hello world", w_out)  # type: ignore[attr-defined]
    ns["PlainWords"].flag("hello BIG WORLD", w_out)  # type: ignore[attr-defined]
    ns["OrderWords"].flag("hi NOW", w_out)  # type: ignore[attr-defined]
    assert w_out == ["plain:BIG,WORLD", "order:NOW"]

    s_out: list[str] = []
    ns["SkippingStore"].process(["", "skip:x", "a-b"], s_out)  # type: ignore[attr-defined]
    ns["DirectStore"].process(["", "ok"], s_out)  # type: ignore[attr-defined]
    assert s_out == ["a", "b", "OK"]
    assert _findings(tmp_path) == []


def test_hoisting_step_expr_past_pure_prep_and_pure_builtins_rejects_impure_prep(
    tmp_path: Path,
):
    """Unconditional step calls following pure local prep assignments or pure builtin
    calls (`str(...)`) hoist cleanly with single evaluation; impure prep calls before
    the step remain refused."""

    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Route(ABC):\n"
        "    def __init__(self) -> None:\n"
        "        self.calls = 0\n\n"
        "    def target(self, col: str) -> str:\n"
        "        self.calls += 1\n"
        "        return f't.{col}'\n\n"
        "    @abstractmethod\n"
        "    def sql(self, col: str, names: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:\n"
        "        ...\n\n\n"
        "class MultiRoute(Route):\n"
        "    def sql(self, col: str, names: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:\n"
        "        marks = ','.join(('?' for _ in names))\n"
        "        return (f'{self.target(col)} IN ({marks})', names)\n\n\n"
        "class SingleRoute(Route):\n"
        "    def sql(self, col: str, names: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:\n"
        "        return (f'{self.target(col)}=?', names)\n\n\n"
        "class Rule(ABC):\n"
        "    def __init__(self, num: int) -> None:\n"
        "        self.num = num\n"
        "        self.wire_calls = 0\n\n"
        "    def to_wire(self) -> str:\n"
        "        self.wire_calls += 1\n"
        "        return f'w:{self.num}'\n\n"
        "    @abstractmethod\n"
        "    def values(self) -> dict[str, str]:\n"
        "        ...\n\n\n"
        "class FirstRule(Rule):\n"
        "    def values(self) -> dict[str, str]:\n"
        "        return {'num': str(self.num), 'wire': self.to_wire()}\n\n\n"
        "class SecondRule(Rule):\n"
        "    def values(self) -> dict[str, str]:\n"
        "        return {'wire': self.to_wire(), 'num': str(self.num)}\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/hoist_pure.py": source})
    for _ in range(2):
        findings = _findings(tmp_path)
        assert findings
        snap = CodemodSourceSnapshot.from_modules(
            parse_python_modules(
                tmp_path,
                use_parse_cache=False,
                source_policy=PythonSourcePathPolicy(include_tests=False),
            )
        )
        ev = SiblingOverrideStepDetector().evaluate_recipe_for_finding(findings[0], snap)
        sim = ev.executable_recipe.simulate(snap)
        assert sim.is_clean, sim.preflight_report
        sim.apply()

    rewritten = (tmp_path / "pkg/hoist_pure.py").read_text()
    assert "target = self.target(col)" in rewritten
    assert "to_wire = self.to_wire()" in rewritten

    ns: dict[str, object] = {}
    exec(compile(rewritten, "hoist_pure.py", "exec"), ns)
    mr = ns["MultiRoute"]()  # type: ignore[operator]
    assert mr.sql("id", ("a", "b")) == ("t.id IN (?,?)", ("a", "b"))
    assert mr.calls == 1
    sr = ns["SingleRoute"]()  # type: ignore[operator]
    assert sr.sql("id", ("a",)) == ("t.id=?", ("a",))
    assert sr.calls == 1
    r1 = ns["FirstRule"](7)  # type: ignore[operator]
    assert r1.values() == {"num": "7", "wire": "w:7"}
    assert r1.wire_calls == 1
    r2 = ns["SecondRule"](8)  # type: ignore[operator]
    assert r2.values() == {"wire": "w:8", "num": "8"}
    assert r2.wire_calls == 1
    assert _findings(tmp_path) == []

    # Impure prep call before the shared step must stay refused.
    impure_dir = tmp_path / "impure"
    _write(
        impure_dir,
        {
            "pkg/__init__.py": "",
            "pkg/impure.py": (
                "from abc import ABC, abstractmethod\n\n\n"
                "def side_effect() -> str:\n"
                "    return 'x'\n\n\n"
                "class Base(ABC):\n"
                "    def step(self) -> str:\n"
                "        return 's'\n\n"
                "    @abstractmethod\n"
                "    def run(self) -> str:\n"
                "        ...\n\n\n"
                "class SubA(Base):\n"
                "    def run(self) -> str:\n"
                "        val = side_effect()\n"
                "        return val + self.step()\n\n\n"
                "class SubB(Base):\n"
                "    def run(self) -> str:\n"
                "        return self.step() + '!'\n"
            ),
        },
    )
    (impure_finding,) = _findings(impure_dir)
    impure_snap = CodemodSourceSnapshot.from_modules(
        parse_python_modules(
            impure_dir,
            use_parse_cache=False,
            source_policy=PythonSourcePathPolicy(include_tests=False),
        )
    )
    impure_eval = SiblingOverrideStepDetector().evaluate_recipe_for_finding(
        impure_finding, impure_snap
    )
    assert impure_eval.recipe is None
    assert "not independent leading or trailing statements" in impure_eval.rejection_reason


