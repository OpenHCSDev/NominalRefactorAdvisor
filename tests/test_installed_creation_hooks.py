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
from nominal_refactor_advisor.refactors.class_attribute_reads import InstalledPackageBaseReads
from nominal_refactor_advisor.refactors.class_overlap import ComposedAncestry, OverlapClass
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


COMPOSITION_NODE = (
    "class Marker:\n    pass\n\n\n"
    "class Node:\n"
    "    def __init_subclass__(cls, **kwargs):\n"
    "        super().__init_subclass__(**kwargs)\n"
    "        marked = cls._marked = {}\n"
    "        for base in reversed(cls.__mro__):\n"
    "            marked.update({name: value for name, value in base.__dict__.items() if isinstance(value, Marker)})\n"
    "        bases = cls._css_bases(cls)\n"
    "        cls._css_names = tuple(base.__name__ for base in bases)\n"
    "        cls._gen_bases = tuple(base for base in cls.__mro__ if issubclass(base, Node))\n"
    "        for base in bases:\n"
    "            cls._last_css = base.__name__\n"
    "        cls._paints = any(method in cls.__dict__ for method in ('render', '_render'))\n"
    "        cls._own_markers = cls._collect_own('_tag')\n"
    "        cls._computed = frozenset(name for name in dir(cls) if name.startswith('compute_'))\n\n"
    "    @classmethod\n"
    "    def _css_bases(cls, base):\n"
    "        classes: list[type] = []\n"
    "        for ancestor in base.__mro__:\n"
    "            if issubclass(ancestor, Node):\n"
    "                classes.append(ancestor)\n"
    "        return classes\n\n"
    "    @classmethod\n"
    "    def _collect_own(cls, key, bases=None):\n"
    "        found = []\n"
    "        for base in cls.__mro__ if bases is None else bases:\n"
    "            if key in base.__dict__:\n"
    "                found.append(base.__dict__[key])\n"
    "        return tuple(found)\n\n\n"
    "class Widget(Node):\n    pass\n"
)


def _compose_proof(site: Path, scan: Path, app_source: str, main_source: str, composer: str, owner: str, moved: tuple[str, ...]):
    _write(site, {"toolkit/comp.py": COMPOSITION_NODE})
    _write(scan, {"app/comp.py": app_source, "main.py": main_source})
    for cached in CACHES:
        cached.cache_clear()
    snap = CodemodSourceSnapshot.from_modules(parse_python_modules(scan, use_parse_cache=False))
    comp_cls = OverlapClass.for_symbol(snap, f"app.comp.{composer}")
    owner_cls = OverlapClass.for_symbol(snap, f"app.comp.{owner}")
    return ComposedAncestry(snap, comp_cls, (owner_cls,), frozenset(moved)).creation()


def test_class_creation_proves_plain_mixin_under_textual_style_init_subclass(roots):
    site, scan = roots
    app_source = (
        "from toolkit.comp import Marker, Widget\n\n\n"
        "class SharedStyle:\n"
        "    def format_label(self):\n        return 'ok'\n\n\n"
        "class Panel(Widget):\n"
        "    flag = Marker()\n"
        "    _tag = 'panel'\n\n"
        "    def render(self):\n        return 'r'\n\n"
        "    def format_label(self):\n        return 'ok'\n"
    )
    main_source = (
        "from app.comp import Panel\n"
        "print(Panel._css_names, [c.__name__ for c in Panel._gen_bases], sorted(Panel._marked), "
        "Panel._paints, Panel._own_markers, sorted(Panel._computed), Panel().format_label())\n"
    )
    proof = _compose_proof(site, scan, app_source, main_source, "Panel", "SharedStyle", ("format_label",))
    proof.require_unchanged()
    before = _run(site, scan)
    _write(
        scan,
        {
            "app/comp.py": app_source.replace(
                "class Panel(Widget):", "class Panel(SharedStyle, Widget):"
            ).replace(
                "    def render(self):\n        return 'r'\n\n    def format_label(self):\n        return 'ok'\n",
                "    def render(self):\n        return 'r'\n",
            )
        },
    )
    assert _run(site, scan) == before


def test_class_creation_refuses_node_subclass_or_marker_or_dir_or_moved_dict_key(roots):
    site, scan = roots
    base_app = (
        "from toolkit.comp import Marker, Widget\n\n\n"
        "class CenteredText(Widget):\n"
        "    def format_label(self):\n        return 'ok'\n\n\n"
        "class MarkerMixin:\n"
        "    extra = Marker()\n"
        "    def format_label(self):\n        return 'ok'\n\n\n"
        "class ExtraAnnMixin:\n"
        "    extra_ann: int = 1\n"
        "    def format_label(self):\n        return 'ok'\n\n\n"
        "class RenderMixin:\n"
        "    def render(self):\n        return 'r'\n\n\n"
        "class Panel(Widget):\n"
        "    extra_ann = 1\n\n"
        "    def render(self):\n        return 'r'\n\n"
        "    def format_label(self):\n        return 'ok'\n"
    )
    main_source = "from app.comp import Panel\nprint(Panel._css_names)\n"

    # Composing another Node subclass changes `issubclass(ancestor, Node)` walks.
    with pytest.raises(Exception, match="subclasses"):
        _compose_proof(site, scan, base_app, main_source, "Panel", "CenteredText", ("format_label",)).require_unchanged()

    # Composing a mixin with a Marker changes `isinstance(v, Marker)` reversed MRO walks.
    with pytest.raises(Exception, match="Marker"):
        _compose_proof(site, scan, base_app, main_source, "Panel", "MarkerMixin", ("format_label",)).require_unchanged()

    # Composing a mixin with an annotation changes unfiltered `dir(cls)` (`__annotations__`/`__annotate_func__`),
    # whereas a `startswith('compute_')`-filtered `dir(cls)` walk ignores `__annotations__` and refuses `compute_*`.
    _compose_proof(site, scan, base_app, main_source, "Panel", "ExtraAnnMixin", ("format_label",)).require_unchanged()
    unfiltered_node = COMPOSITION_NODE.replace(
        "cls._computed = frozenset(name for name in dir(cls) if name.startswith('compute_'))",
        "cls._all_dir = frozenset(dir(cls))",
    )
    _write(site, {"toolkit/comp.py": unfiltered_node})
    _write(scan, {"app/comp.py": base_app, "main.py": main_source})
    for cached in CACHES:
        cached.cache_clear()
    unfiltered_snap = CodemodSourceSnapshot.from_modules(parse_python_modules(scan, use_parse_cache=False))
    with pytest.raises(Exception, match="dir"):
        ComposedAncestry(
            unfiltered_snap,
            OverlapClass.for_symbol(unfiltered_snap, "app.comp.Panel"),
            (OverlapClass.for_symbol(unfiltered_snap, "app.comp.ExtraAnnMixin"),),
            frozenset(("format_label",)),
        ).creation().require_unchanged()

    # Moving `'render'` out of `Panel` changes `any(method in cls.__dict__ for method in ('render', '_render'))`.
    with pytest.raises(Exception, match="reads the class namespace"):
        _compose_proof(site, scan, base_app, main_source, "Panel", "RenderMixin", ("render",)).require_unchanged()


def test_class_creation_proves_unmatched_prefix_namespace_loop_and_refuses_matching(roots):
    """Textual `_MessagePumpMeta.__new__` walks `class_dict.items()` behind `if attr_name.startswith(prefix)`
    (`prefix = 'compute_'`). When the composer binds no `compute_*` names before or after, the loop body
    never runs; when it does bind a `compute_*` name (or moves one), NRA still refuses."""
    site, scan = roots
    prefix_meta_node = (
        "class Reactive:\n    pass\n\n\n"
        "class PumpMeta(type):\n"
        "    def __new__(mcs, name, bases, class_dict, **kwargs):\n"
        "        prefix = 'compute_'\n"
        "        prefix_len = len(prefix)\n"
        "        for attr_name, value in class_dict.items():\n"
        "            if attr_name.startswith(prefix) and callable(value):\n"
        "                reactive_name = attr_name[prefix_len:]\n"
        "                if reactive_name in class_dict and isinstance(class_dict[reactive_name], Reactive):\n"
        "                    raise RuntimeError(reactive_name)\n"
        "        built = super().__new__(mcs, name, bases, class_dict, **kwargs)\n"
        "        built._compile_dispatch()\n"
        "        return built\n\n"
        "    def _compile_dispatch(cls):\n"
        "        cls._compiled = True\n\n\n"
        "class PumpWidget(metaclass=PumpMeta):\n    pass\n"
    )
    _write(site, {"toolkit/pump.py": prefix_meta_node})
    app_source = (
        "from toolkit.pump import PumpWidget, Reactive\n\n\n"
        "class SharedBehavior:\n"
        "    def toggled(self):\n        return True\n\n\n"
        "class SharedCompute:\n"
        "    def compute_x(self):\n        return 1\n\n\n"
        "class PlainPane(PumpWidget):\n"
        "    def toggled(self):\n        return True\n\n\n"
        "class ComputePane(PumpWidget):\n"
        "    def compute_x(self):\n        return 1\n\n\n"
        "class ReactivePane(PumpWidget):\n"
        "    x = Reactive()\n\n"
        "    def compute_x(self):\n        return 1\n\n"
        "    def toggled(self):\n        return True\n"
    )
    _write(scan, {"app/panes.py": app_source})
    for cached in CACHES:
        cached.cache_clear()
    snap = CodemodSourceSnapshot.from_modules(parse_python_modules(scan, use_parse_cache=False))
    owner_cls = OverlapClass.for_symbol(snap, "app.panes.SharedBehavior")
    compute_owner = OverlapClass.for_symbol(snap, "app.panes.SharedCompute")

    plain = OverlapClass.for_symbol(snap, "app.panes.PlainPane")
    ComposedAncestry(snap, plain, (owner_cls,), frozenset({"toggled"})).creation().require_unchanged()

    reactive = OverlapClass.for_symbol(snap, "app.panes.ReactivePane")
    with pytest.raises(Exception, match="reads the class namespace"):
        ComposedAncestry(snap, reactive, (owner_cls,), frozenset({"toggled"})).creation().require_unchanged()
    compute_pane = OverlapClass.for_symbol(snap, "app.panes.ComputePane")
    with pytest.raises(Exception, match="reads the class namespace"):
        ComposedAncestry(snap, compute_pane, (compute_owner,), frozenset({"compute_x"})).creation().require_unchanged()


def test_member_deletion_proves_textual_pump_meta_dispatch_scope_and_actions_and_refuses_handler_or_action(roots):
    """Order-preserving member deletion under a Textual-style metaclass (`AbstractHooksMeta`,
    `_MessagePumpMeta._compile_dispatch` -> `DispatchScope.from_class(owner)` walking
    `owner.__dict__.values()` behind `isinstance(value, MessageHandler)`, `dir(cls)` behind
    `startswith(('_compute_', 'compute_'))`, and `DeclaredWidgetActions`-style `setattr` +
    `f'action_{...}'` MRO check) proves deleting plain methods/properties and refuses deleting
    a `MessageHandler`, a `compute_*` member, or an `action_*` member."""
    site, scan = roots
    toolkit_source = (
        "from dataclasses import dataclass\n"
        "from functools import partialmethod\n"
        "from metaclass_registry.abstract_hooks import AbstractHooksMeta\n\n\n"
        "class MessageHandler:\n    pass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class DispatchScope:\n"
        "    owner: type\n\n"
        "    @classmethod\n"
        "    def from_class(cls, owner: type) -> 'DispatchScope':\n"
        "        routes = []\n"
        "        for value in owner.__dict__.values():\n"
        "            if isinstance(value, MessageHandler):\n"
        "                routes.append(value)\n"
        "        if routes:\n"
        "            return ClassScope(owner=owner, routes=tuple(routes))\n"
        "        return cls(owner=owner)\n\n\n"
        "@dataclass(frozen=True)\n"
        "class ClassScope(DispatchScope):\n"
        "    routes: tuple[MessageHandler, ...]\n\n\n"
        "class PumpMeta(AbstractHooksMeta):\n"
        "    def __new__(mcs, name, bases, class_dict, **kwargs):\n"
        "        prefix = 'compute_'\n"
        "        for attr_name, value in class_dict.items():\n"
        "            if attr_name.startswith(prefix) and callable(value):\n"
        "                raise RuntimeError(attr_name)\n"
        "        built = super().__new__(mcs, name, bases, class_dict, **kwargs)\n"
        "        built._compile_dispatch()\n"
        "        return built\n\n"
        "    def _compile_dispatch(cls) -> None:\n"
        "        cls._dispatch_declarations = tuple(\n"
        "            DispatchScope.from_class(owner) for owner in cls.__mro__\n"
        "        )\n\n\n"
        "class PumpNode(metaclass=PumpMeta):\n"
        "    def __init_subclass__(cls, actions: tuple[str, ...] = (), **kwargs):\n"
        "        super().__init_subclass__(**kwargs)\n"
        "        for declared in actions:\n"
        "            name = f'action_{declared}'\n"
        "            _ = tuple(\n"
        "                base for base in cls.__mro__\n"
        "                if name in base.__dict__ and not isinstance(base.__dict__[name], partialmethod)\n"
        "            )\n"
        "            setattr(cls, name, partialmethod(cls._run, declared))\n"
        "        cls._computes = frozenset(\n"
        "            name for name in dir(cls) if name.startswith(('_compute_', 'compute_'))\n"
        "        )\n\n"
        "    def _run(self, declared: str) -> str:\n"
        "        return declared\n"
    )
    app_source = (
        "from toolkit.textual_like import MessageHandler, PumpNode\n\n\n"
        "class BasePane(PumpNode, actions=('open',)):\n"
        "    on_click = MessageHandler()\n\n"
        "    @property\n"
        "    def toggled(self) -> bool:\n"
        "        return True\n\n\n"
        "class LeafPane(BasePane):\n"
        "    @property\n"
        "    def toggled(self) -> bool:\n"
        "        return True\n"
    )
    main_source = (
        "from app.panes import LeafPane\n"
        "print(LeafPane().toggled, len(LeafPane._dispatch_declarations), sorted(LeafPane._computes))\n"
    )
    _write(site, {"toolkit/textual_like.py": toolkit_source})
    _write(scan, {"app/panes.py": app_source, "main.py": main_source})
    expected = _run(site, scan)
    evaluation, snapshot = _evaluation(RestatedOverrideDetector(), scan, "LeafPane")
    assert evaluation.rejection_reason == ""
    _apply(evaluation, snapshot)
    assert "class LeafPane(BasePane):\n    pass\n" in (scan / "app/panes.py").read_text()
    assert _run(site, scan) == expected

    # Deleting a MessageHandler binding must be refused by `_is_type_filtered_namespace_loop`.
    handler_app = (
        "from toolkit.textual_like import MessageHandler, PumpNode\n\n\n"
        "HANDLER = MessageHandler()\n\n\n"
        "class BasePane(PumpNode):\n"
        "    on_click = HANDLER\n\n\n"
        "class LeafPane(BasePane):\n"
        "    on_click = HANDLER\n"
    )
    _write(scan, {"app/panes.py": handler_app})
    handler_eval, _ = _evaluation(RestatedOverrideDetector(), scan, "LeafPane")
    assert handler_eval.rejection_reason != ""
    assert NeededInput.PROOF in handler_eval.needs

    # Deleting an `action_*` method must be refused by `f'action_{declared}'` key avoidance.
    action_app = (
        "from toolkit.textual_like import PumpNode\n\n\n"
        "class BasePane(PumpNode):\n"
        "    def action_open(self) -> int:\n"
        "        return 1\n\n\n"
        "class LeafPane(BasePane):\n"
        "    def action_open(self) -> int:\n"
        "        return 1\n"
    )
    _write(scan, {"app/panes.py": action_app})
    action_eval, _ = _evaluation(RestatedOverrideDetector(), scan, "LeafPane")
    assert action_eval.rejection_reason != ""
    assert NeededInput.PROOF in action_eval.needs

    # Deleting a `_compute_*` method must be refused by prefix-filtered `dir(cls)`.
    compute_app = (
        "from toolkit.textual_like import PumpNode\n\n\n"
        "class BasePane(PumpNode):\n"
        "    def _compute_x(self) -> int:\n"
        "        return 1\n\n\n"
        "class LeafPane(BasePane):\n"
        "    def _compute_x(self) -> int:\n"
        "        return 1\n"
    )
    _write(scan, {"app/panes.py": compute_app})
    compute_eval, _ = _evaluation(RestatedOverrideDetector(), scan, "LeafPane")
    assert "dir()" in compute_eval.rejection_reason
    assert NeededInput.PROOF in compute_eval.needs
