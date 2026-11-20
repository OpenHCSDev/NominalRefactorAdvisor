"""A restated base dropped, and a restated member deleted, under a class whose installed
ancestors run creation code (a metaclass and a hook that walk the order and every
namespace, as Textual's do) and beside the repository's own readers of bases and namespaces.

Each case writes an installed toolkit and a scanned application, runs the application,
applies the recipe the finding offers, runs it again and re-reads the source.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import PythonSourcePathPolicy, parse_python_modules
from nominal_refactor_advisor.class_index import InstalledClassDeclarations
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._restated_bases import RestatedBaseDetector
from nominal_refactor_advisor.detectors._restated_overrides import RestatedOverrideDetector
from nominal_refactor_advisor.installed_source import InstalledModule
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.refactors.restated_bases import InstalledPackageBaseReads
from nominal_refactor_advisor.scan_cache import ScanCache

META = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Scope:\n    owner: type\n\n"
    "    @classmethod\n    def of(cls, owner):\n        return cls(owner=owner)\n\n\n"
    "class HooksMeta(type):\n"
    "    def __new__(mcs, name, bases, namespace, **kwargs):\n"
    "        for key, value in namespace.items():\n"
    "            if key.startswith('compute_') and callable(value):\n"
    "                namespace.setdefault('_computes', True)\n"
    "        built = super().__new__(mcs, name, bases, namespace, **kwargs)\n"
    "        built._compile()\n"
    "        return built\n\n"
    "    def __init__(cls, name, bases, namespace, **kwargs):\n"
    "        super().__init__(name, bases, namespace, **kwargs)\n"
    "        cls._abstract = frozenset(\n"
    "            member\n"
    "            for base in cls.__mro__\n"
    "            for member, value in vars(base).items()\n"
    "            if getattr(value, '__isabstractmethod__', False)\n"
    "            and getattr(getattr(cls, member, None), '__isabstractmethod__', False)\n"
    "        )\n\n"
    "    def _compile(cls):\n"
    "        cls._scopes = tuple(Scope.of(owner) for owner in cls.__mro__)\n"
)
NODE = (
    "from typing import TYPE_CHECKING\n\n"
    "from toolkit.meta import HooksMeta\n\n\n"
    "class Marker:\n    pass\n\n\n"
    "class Node(metaclass=HooksMeta):\n"
    "    if TYPE_CHECKING:\n\n        def typed(self) -> int: ...\n\n"
    "    def __init_subclass__(cls, inherit: bool = True, **kwargs):\n"
    "        super().__init_subclass__(**kwargs)\n"
    "        marked = cls._marked = {}\n"
    "        for base in reversed(cls.__mro__):\n"
    "            marked.update({name: value for name, value in base.__dict__.items() if isinstance(value, Marker)})\n"
    "        cls._inherit = inherit\n"
    "        cls._node_bases = cls._css_bases(cls)\n"
    "        if '_paint' not in cls.__dict__ and 'render' in cls.__dict__:\n"
    "            cls._paint = False\n"
    "        cls._computed = frozenset(name for name in dir(cls) if name.startswith('compute_'))\n\n"
    "    @classmethod\n"
    "    def _css_bases(cls, base):\n"
    "        classes: list[type] = []\n"
    "        for ancestor in base.__mro__:\n"
    "            if issubclass(ancestor, Node):\n"
    "                classes.append(ancestor)\n"
    "        return classes\n\n"
    "    def settled(self):\n        return self\n\n\n"
    "class Widget(Node):\n    pass\n\n\n"
    "class Box[Item](Node):\n    pass\n"
)
VIEWS = (
    "from toolkit.node import Widget\n\n\n"
    "class Hidden(Widget):\n    def render(self):\n        return ''\n\n\n"
    "class Slot(Hidden, Widget):\n    pass\n"
)
MAIN = (
    "from app.views import Slot\n\n"
    "print([c.__name__ for c in Slot.__mro__], [c.__name__ for c in Slot._node_bases],"
    " sorted(Slot._marked), Slot._paint, sorted(Slot._abstract), len(Slot._scopes))\n"
)
CACHES = (
    InstalledModule.located_source,
    InstalledModule.parsed,
    InstalledClassDeclarations.module,
    InstalledClassDeclarations.stub,
    InstalledPackageBaseReads.of,
)


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


@pytest.fixture
def roots(tmp_path: Path, monkeypatch):
    """`site` holds the installed toolkit, on the path of this process (NRA reads it)
    and of the application's run; `scan` holds the application NRA scans."""

    site, scan = tmp_path / "site", tmp_path / "scan"
    _write(site, {"toolkit/__init__.py": "", "toolkit/meta.py": META, "toolkit/node.py": NODE})
    _write(scan, {"app/__init__.py": "", "app/views.py": VIEWS, "main.py": MAIN})
    monkeypatch.syspath_prepend(str(site))
    for cached in CACHES:
        cached.cache_clear()
    yield site, scan
    for cached in CACHES:
        cached.cache_clear()
    for name in [name for name in sys.modules if name == "toolkit" or name.startswith("toolkit.")]:
        del sys.modules[name]


def _run(site: Path, scan: Path) -> str:
    return subprocess.check_output(
        [sys.executable, "main.py"], cwd=scan, text=True, env={**os.environ, "PYTHONPATH": str(site)}
    )


def _modules(scan: Path):
    return parse_python_modules(
        scan, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False)
    )


def _evaluation(detector, scan: Path, symbol: str):
    for cached in CACHES:
        cached.cache_clear()
    with ScanCache.scope():
        finding = next(f for f in detector.detect(_modules(scan), DetectorConfig()) if f.primary_evidence.symbol == symbol)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(scan, use_parse_cache=False))
    return detector.evaluate_recipe_for_finding(finding, snapshot), snapshot


def _apply(evaluation, snapshot) -> None:
    simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()


def test_a_base_restated_under_an_installed_class_with_hooks_is_dropped(roots):
    site, scan = roots
    expected = _run(site, scan)
    evaluation, snapshot = _evaluation(RestatedBaseDetector(), scan, "Slot")
    # The metaclass and the hook are the toolkit's: they walk the order, each class's
    # namespace and `dir`, keep the class in a local list and a dataclass, and call a
    # method the metaclass defines. None of it reads the header's bases.
    assert evaluation.rejection_reason == ""
    _apply(evaluation, snapshot)
    assert (scan / "app/views.py").read_text() == VIEWS.replace("class Slot(Hidden, Widget):", "class Slot(Hidden):")
    assert _run(site, scan) == expected


def test_installed_creation_code_that_reads_bases_keeps_the_drop_unproved(roots):
    site, scan = roots
    _write(site, {"toolkit/node.py": NODE.replace("for ancestor in base.__mro__:", "for ancestor in base.__bases__:")})
    evaluation, _snapshot = _evaluation(RestatedBaseDetector(), scan, "Slot")
    assert "toolkit/node.py" in evaluation.rejection_reason
    assert "`base.__bases__`" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_a_bases_read_anywhere_in_an_installed_package_of_the_ancestry_is_judged(roots):
    site, scan = roots
    _write(site, {"toolkit/registry.py": "def direct(cls):\n    return [base.__name__ for base in cls.__bases__]\n"})
    evaluation, _snapshot = _evaluation(RestatedBaseDetector(), scan, "Slot")
    # No creation code calls it; the toolkit keeps the class (`_scopes`), so its read
    # of a class's bases may be a read of this one's.
    assert "toolkit/registry.py:2" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs
    _write(site, {"toolkit/registry.py": "from toolkit.node import Marker\n\n\ndef direct(cls):\n    return Marker in cls.__bases__\n"})
    evaluation, _snapshot = _evaluation(RestatedBaseDetector(), scan, "Slot")
    assert evaluation.rejection_reason == ""


SUBSCRIPTED = (
    "from toolkit.node import Box, Widget\n\n\n"
    "class Hidden(Widget):\n    def render(self):\n        return ''\n\n\n"
    "class Slot(Hidden, Widget, Box[int]):\n    pass\n"
)
WALKS = (
    "        found = {}\n"
    "        for owner in cls.__mro__:\n"
    "            for name, value in vars(owner).items():\n"
    "                found.setdefault(name, value)\n"
    "        cls._handled = tuple(name for name, value in found.items() if getattr(value, '__handled__', ()))\n"
    "        cls._union = cls._declared('_own')\n"
    "        cls._paints = any(method in cls.__dict__ for method in ('render', '_render'))\n"
    "        cls._first = [base.__dict__[key] for base in cls.__mro__ for key in ('_own',) if key in base.__dict__]\n"
)
DECLARED = (
    "    @classmethod\n"
    "    def _declared(cls, name):\n"
    "        return tuple(base.__dict__[name] for base in cls.__mro__ if name in base.__dict__)\n\n"
)


def test_a_subscripted_header_s_bases_are_passed_by_in_every_namespace_walk(roots):
    site, scan = roots
    # `__orig_bases__` is in `Slot`'s own namespace and the drop shortens it. The toolkit
    # walks namespaces by a constant key, a parameter handed a constant, a display of
    # constants, the first-found walk, and by value behind `isinstance`, `getattr`,
    # `callable` and a name prefix: none of them does anything with that entry.
    _write(site, {"toolkit/node.py": NODE.replace(
        "        cls._inherit = inherit\n", "        cls._inherit = inherit\n" + WALKS
    ).replace("    @classmethod\n    def _css_bases", DECLARED + "    @classmethod\n    def _css_bases")})
    _write(scan, {"app/views.py": SUBSCRIPTED})
    expected = _run(site, scan)
    evaluation, snapshot = _evaluation(RestatedBaseDetector(), scan, "Slot")
    assert evaluation.rejection_reason == ""
    _apply(evaluation, snapshot)
    assert (scan / "app/views.py").read_text() == SUBSCRIPTED.replace("Slot(Hidden, Widget, Box[int])", "Slot(Hidden, Box[int])")
    assert _run(site, scan) == expected


@pytest.mark.parametrize(
    "walk",
    [
        "        cls._all = list(cls.__dict__.values())\n",
        "        for name, value in cls.__dict__.items():\n            marked[name] = value\n",
        "        cls._listed = cls.__dict__.get(inherit and '__orig_bases__')\n",
    ],
)
def test_a_walk_that_keeps_a_subscripted_header_s_bases_leaves_the_drop_unproved(roots, walk: str):
    site, scan = roots
    _write(site, {"toolkit/node.py": NODE.replace("        cls._inherit = inherit\n", "        cls._inherit = inherit\n" + walk)})
    _write(scan, {"app/views.py": SUBSCRIPTED})
    evaluation, _snapshot = _evaluation(RestatedBaseDetector(), scan, "Slot")
    assert "toolkit/node.py" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs
    # The same walk is the same body's namespace when no base is subscripted.
    _write(scan, {"app/views.py": VIEWS})
    evaluation, _snapshot = _evaluation(RestatedBaseDetector(), scan, "Slot")
    assert evaluation.rejection_reason == ""


def test_a_loop_over_a_family_s_subclasses_reads_only_that_family_s_bases(roots):
    site, scan = roots
    reader = (
        "class Block:\n"
        "    @classmethod\n"
        "    def blocks_for(cls, native):\n"
        "        found = {}\n"
        "        for declaration in cls.__subclasses__():\n"
        "            for base in declaration.__bases__:\n"
        "                if base in native:\n"
        "                    found[base] = declaration\n"
        "        return found\n"
    )
    _write(scan, {"app/blocks.py": reader})
    evaluation, _snapshot = _evaluation(RestatedBaseDetector(), scan, "Slot")
    # `declaration` is a subclass of `Block`, which `Slot` is not.
    assert evaluation.rejection_reason == ""
    _write(scan, {"app/blocks.py": reader.replace("class Block:", "from toolkit.node import Node\n\n\nclass Block(Node):").replace(
        "cls.__subclasses__()", "Node.__subclasses__()")})
    evaluation, _snapshot = _evaluation(RestatedBaseDetector(), scan, "Slot")
    assert "app/blocks.py" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


OVERRIDES = (
    "from functools import partialmethod\n\n\n"
    "class Actions:\n"
    "    def label(self):\n        return 'panel'\n\n\n"
    "class Panel(Actions):\n    def label(self):\n        return 'panel'\n\n\n"
    "def written(cls, declared):\n"
    "    name = f'action_{declared}'\n"
    "    return tuple(\n"
    "        base.__name__ for base in cls.__mro__\n"
    "        if name in base.__dict__ and not isinstance(base.__dict__[name], partialmethod)\n"
    "    )\n\n\n"
    "def widths(member):\n"
    "    return 'width' in {name for ancestor in member.__mro__ for name in vars(ancestor)}\n"
)
OVERRIDES_MAIN = "from app.actions import Panel, widths, written\n\nprint(Panel().label(), written(Panel, 'open'), widths(Panel))\n"


def test_a_namespace_read_by_a_prefixed_text_or_for_one_constant_leaves_a_deletion_proved(roots):
    site, scan = roots
    _write(scan, {"app/actions.py": OVERRIDES, "main.py": OVERRIDES_MAIN})
    expected = _run(site, scan)
    evaluation, snapshot = _evaluation(RestatedOverrideDetector(), scan, "Panel")
    # `name` is always `action_<declared>`, and the collected names are asked only for `width`.
    assert evaluation.rejection_reason == ""
    _apply(evaluation, snapshot)
    assert "class Panel(Actions):\n    pass\n" in (scan / "app/actions.py").read_text()
    assert _run(site, scan) == expected


@pytest.mark.parametrize(
    ("member", "reader"),
    [("action_open", "name in base.__dict__"), ("width", "for name in vars(ancestor)")],
)
def test_a_deleted_name_the_read_can_ask_for_keeps_the_deletion_unproved(roots, member: str, reader: str):
    site, scan = roots
    _write(scan, {"app/actions.py": OVERRIDES.replace("def label(self):\n        return 'panel'", f"def {member}(self):\n        return 'panel'")})
    evaluation, _snapshot = _evaluation(RestatedOverrideDetector(), scan, "Panel")
    assert reader in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_names_an_installed_class_declares_for_the_type_checker_only_are_not_bound(roots):
    site, scan = roots
    _write(site, {"toolkit/plain.py": (
        "from typing import TYPE_CHECKING\n\n\n"
        "class Plain:\n"
        "    if TYPE_CHECKING:\n\n        def typed(self) -> int: ...\n\n"
        "    def settled(self):\n        return self\n"
    )})
    _write(scan, {
        "app/plain.py": (
            "from toolkit.plain import Plain\n\n\n"
            "class Mid(Plain):\n    def settled(self):\n        return None\n\n\n"
            "class Leaf(Mid):\n    def settled(self):\n        return None\n"
        ),
        "main.py": "from app.plain import Leaf\n\nprint(Leaf().settled())\n",
    })
    expected = _run(site, scan)
    evaluation, snapshot = _evaluation(RestatedOverrideDetector(), scan, "Leaf")
    # The walk of `Leaf`'s order reads what `Plain`'s body binds: not `typed`.
    assert evaluation.rejection_reason == ""
    _apply(evaluation, snapshot)
    assert _run(site, scan) == expected
