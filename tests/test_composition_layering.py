"""A composition root's declared layering, and the declarations placed above it.

The root builds each service from the services it hands it; a declaration only
a higher service loads, read by code a lower one loads, belongs in the lower
layer. Each applying case writes a package, runs it, applies the move the
finding offers, and runs it again.
"""

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.analysis import (
    analyze_compact_roots_with_cache,
    analyze_detector_types,
)
from nominal_refactor_advisor.ast_tools import PythonSourcePathPolicy, parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.composition_layering import CompositionRootLayering
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._composition_layering import LayeringViolationDetector
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache

PYPROJECT = '[tool.nominal-refactor-advisor]\ncomposition-roots = ["pkg.root:Root"]\n'
ROOT = (
    "from pkg.registry import Registry\nfrom pkg.relations import Relations\nfrom pkg.threads import Threads\n\n\n"
    "class Root:\n"
    "    def __init__(self, path):\n"
    "        self.root = path\n"
    "        self.registry = Registry(self.root)\n"
    "        self.threads = Threads(self.root, self.registry)\n"
    "        self.relations = Relations(self.threads)\n"
)
REGISTRY = "class Registry:\n    def __init__(self, root):\n        self.root = root\n"
MAIN = "from pkg.root import Root\n\nprint(Root('r').threads.states(), Root('r').relations.threads.root)\n"


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in {"pyproject.toml": PYPROJECT, "pkg/__init__.py": "", "main.py": MAIN, **files}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(root: Path) -> str:
    return subprocess.check_output([sys.executable, "main.py"], cwd=root, text=True)


def _modules(root: Path):
    return parse_python_modules(root, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False))


def _findings(root: Path):
    with ScanCache.scope():
        return LayeringViolationDetector().detect(_modules(root), DetectorConfig())


def _apply(root: Path, finding) -> None:
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))
    evaluation = LayeringViolationDetector().evaluate_recipe_for_finding(finding, snapshot)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()


def test_layering_violation_continues_the_stable_sequence():
    assert PatternId.LAYERING_VIOLATION == 56


def test_a_store_declared_with_the_higher_service_moves_into_the_layer_reading_it(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/root.py": ROOT,
            "pkg/registry.py": REGISTRY,
            "pkg/threads.py": (
                "from pkg.relations import RelationStore\n\n\n"
                "class Threads:\n"
                "    def __init__(self, root, registry):\n"
                "        self.root = root\n"
                "        self.registry = registry\n\n"
                "    def states(self):\n"
                "        return RelationStore.at(self.root)\n"
            ),
            "pkg/relations.py": (
                "class RelationStore:\n"
                "    @classmethod\n"
                "    def at(cls, root):\n"
                "        return f'store at {root}'\n\n\n"
                "class Relations:\n"
                "    def __init__(self, threads):\n"
                "        self.threads = threads\n"
            ),
        },
    )
    expected = _run(tmp_path)
    [finding] = _findings(tmp_path)
    assert "`pkg.relations.RelationStore`" in finding.summary
    assert "`threads` (below `relations`)" in finding.summary
    assert "home is the `threads` layer" in finding.summary
    _apply(tmp_path, finding)
    assert "RelationStore" not in (tmp_path / "pkg/relations.py").read_text()
    assert _findings(tmp_path) == []
    assert _run(tmp_path) == expected


def test_a_moved_store_gets_a_module_named_for_it_rather_than_one_merely_low_enough(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/root.py": ROOT,
            "pkg/registry.py": REGISTRY,
            "pkg/support/__init__.py": "",
            "pkg/support/clock.py": "TICK = 1\n",
            "pkg/threads.py": (
                "from pkg.relations import RelationStore\nfrom pkg.support.clock import TICK\n\n\n"
                "class Threads:\n"
                "    def __init__(self, root, registry):\n"
                "        self.root = root\n"
                "        self.registry = registry\n\n"
                "    def states(self):\n"
                "        return RelationStore.at(self.root), TICK\n"
            ),
            "pkg/relations.py": (
                "from pkg.support.clock import TICK\n\n\n"
                "def _key(root):\n"
                "    return f'store at {root}'\n\n\n"
                "class RelationStore:\n"
                "    @classmethod\n"
                "    def at(cls, root):\n"
                "        return _key(root)\n\n\n"
                "class Relations:\n"
                "    def __init__(self, threads):\n"
                "        self.threads = threads\n"
                "        self.tick = TICK\n"
            ),
        },
    )
    expected = _run(tmp_path)
    [finding] = _findings(tmp_path)
    _apply(tmp_path, finding)
    assert (tmp_path / "pkg/support/clock.py").read_text() == "TICK = 1\n"
    home = (tmp_path / "pkg/relation_store.py").read_text()
    assert "def _key" in home and "class RelationStore" in home
    assert _findings(tmp_path) == []
    assert _run(tmp_path) == expected


def test_a_hidden_import_of_a_higher_layer_names_its_home(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/root.py": ROOT,
            "pkg/registry.py": (
                "class Registry:\n"
                "    def __init__(self, root):\n"
                "        self.root = root\n\n"
                "    def thread_count(self):\n"
                "        from pkg.threads import LIMIT\n\n"
                "        return LIMIT\n"
            ),
            "pkg/threads.py": (
                "LIMIT = 3\n\n\n"
                "class Threads:\n"
                "    def __init__(self, root, registry):\n"
                "        self.root = root\n"
                "        self.registry = registry\n\n"
                "    def states(self):\n"
                "        return self.registry.thread_count()\n"
            ),
            "pkg/relations.py": "class Relations:\n    def __init__(self, threads):\n        self.threads = threads\n",
        },
    )
    layering = _modules(tmp_path).fact(CompositionRootLayering)
    [read] = [read for read in layering.reads if read.declaration[1] is not None]
    assert (read.reader, read.declaration, read.lower, read.higher) == (
        "pkg.registry",
        ("pkg.threads", "LIMIT"),
        "registry",
        "threads",
    )
    assert read.symbol == "Registry.thread_count"
    assert layering.home(read.declaration)[1] == "registry"
    [finding] = _findings(tmp_path)
    assert "`pkg.threads.LIMIT`" in finding.summary


def test_a_conforming_root_has_no_findings(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/root.py": ROOT,
            "pkg/registry.py": REGISTRY,
            "pkg/store.py": "class RelationStore:\n    @classmethod\n    def at(cls, root):\n        return root\n",
            "pkg/threads.py": (
                "from pkg.store import RelationStore\n\n\n"
                "class Threads:\n"
                "    def __init__(self, root, registry):\n"
                "        self.root = root\n\n"
                "    def states(self):\n"
                "        return RelationStore.at(self.root)\n"
            ),
            "pkg/relations.py": (
                "from pkg.store import RelationStore\n\n\n"
                "class Relations:\n"
                "    def __init__(self, threads):\n"
                "        self.threads = threads\n"
                "        self.store = RelationStore\n"
            ),
        },
    )
    [root] = _modules(tmp_path).fact(CompositionRootLayering).roots
    assert [service.name for service in root.layers] == ["registry", "threads", "relations"]
    assert _findings(tmp_path) == []


def test_dependencies_declared_through_a_base_init_and_a_cached_property(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/registry.py": REGISTRY,
            "pkg/bus.py": "class Bus:\n    def __init__(self, root, registry):\n        self.registry = registry\n",
            "pkg/goals.py": (
                "class Goals:\n"
                "    def __init__(self, root, registry, bus):\n"
                "        self.bus = bus\n"
            ),
            "pkg/client.py": (
                "from functools import cached_property\n\n"
                "from pkg.bus import Bus\nfrom pkg.registry import Registry\n\n\n"
                "class Client:\n"
                "    def __init__(self, root):\n"
                "        self.root = root\n"
                "        self.registry = Registry(self.root)\n\n"
                "    @cached_property\n"
                "    def bus(self):\n"
                "        return Bus(self.root, self.registry)\n"
            ),
            "pkg/root.py": (
                "from pkg.client import Client\nfrom pkg.goals import Goals\n\n\n"
                "class Root(Client):\n"
                "    def __init__(self, root):\n"
                "        super().__init__(root)\n"
                "        self.goals = Goals(self.root, self.registry, self.bus)\n"
            ),
        },
    )
    [root] = _modules(tmp_path).fact(CompositionRootLayering).roots
    assert {service.name: service.qualname for service in root.services} == {
        "registry": "Registry",
        "bus": "Bus",
        "goals": "Goals",
    }
    assert root.built_on == {
        "registry": frozenset(),
        "bus": frozenset({"registry"}),
        "goals": frozenset({"registry", "bus"}),
    }


def test_a_module_read_as_a_namespace_names_the_declarations_read_off_it(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/root.py": ROOT,
            "pkg/registry.py": (
                "class Registry:\n"
                "    def __init__(self, root):\n"
                "        self.root = root\n\n"
                "    def thread_count(self):\n"
                "        from pkg import threads\n\n"
                "        return threads.LIMIT\n"
            ),
            "pkg/threads.py": (
                "LIMIT = 3\n\n\n"
                "class Threads:\n"
                "    def __init__(self, root, registry):\n"
                "        self.root = root\n"
                "        self.registry = registry\n\n"
                "    def states(self):\n"
                "        return self.registry.thread_count()\n"
            ),
            "pkg/relations.py": "class Relations:\n    def __init__(self, threads):\n        self.threads = threads\n",
        },
    )
    [finding] = _findings(tmp_path)
    assert "`pkg.threads.LIMIT`" in finding.summary
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path, use_parse_cache=False))
    evaluation = LayeringViolationDetector().evaluate_recipe_for_finding(finding, snapshot)
    assert "inside 'thread_count' and reads 'LIMIT' off it there" in evaluation.rejection_reason


def test_a_module_object_used_whole_stays_a_module_read(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/root.py": ROOT,
            "pkg/registry.py": (
                "class Registry:\n"
                "    def __init__(self, root):\n"
                "        self.root = root\n\n"
                "    def thread_count(self):\n"
                "        from pkg import threads\n\n"
                "        return getattr(threads, 'LIMIT')\n"
            ),
            "pkg/threads.py": (
                "LIMIT = 3\n\n\n"
                "class Threads:\n"
                "    def __init__(self, root, registry):\n"
                "        self.root = root\n"
                "        self.registry = registry\n\n"
                "    def states(self):\n"
                "        return self.registry.thread_count()\n"
            ),
            "pkg/relations.py": "class Relations:\n    def __init__(self, threads):\n        self.threads = threads\n",
        },
    )
    [finding] = _findings(tmp_path)
    assert "`pkg.threads` is read" in finding.summary


def test_compact_root_analysis_matches_full_for_layering_violations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(
        root,
        {
            "pkg/root.py": ROOT,
            "pkg/registry.py": (
                "class Registry:\n"
                "    def __init__(self, root):\n"
                "        self.root = root\n\n"
                "    def thread_count(self):\n"
                "        from pkg import threads\n\n"
                "        return threads.LIMIT\n"
            ),
            "pkg/threads.py": (
                "LIMIT = 3\n\n\n"
                "class Threads:\n"
                "    def __init__(self, root, registry):\n"
                "        self.root = root\n"
                "        self.registry = registry\n\n"
                "    def states(self):\n"
                "        return self.registry.thread_count()\n"
            ),
            "pkg/relations.py": (
                "class Relations:\n"
                "    def __init__(self, threads):\n"
                "        self.threads = threads\n"
            ),
        },
    )
    detectors = (LayeringViolationDetector,)
    config = DetectorConfig()
    full_findings = analyze_detector_types(_modules(root), config, detector_types=detectors)
    cache_dir = tmp_path / "cache" / "ast"
    analysis_cache_dir = tmp_path / "cache" / "analysis"
    cold = analyze_compact_roots_with_cache(
        (root,),
        config,
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
        detector_types=detectors,
    )
    warm = analyze_compact_roots_with_cache(
        (root,),
        config,
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
        detector_types=detectors,
    )
    assert cold.findings == full_findings
    assert warm.findings == full_findings
    assert len(cold.findings) == 1
    assert "`pkg.threads.LIMIT`" in cold.findings[0].summary
