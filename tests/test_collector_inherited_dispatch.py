"""Collector migration must preserve inherited dispatch across base branches."""

import ast
from pathlib import Path
import subprocess
import sys
from textwrap import indent

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DeriveCandidateCollectorOperation,
    SourceRewriteTarget,
)


@pytest.mark.parametrize(
    ("other_source", "bases", "safe"),
    (
        (
            "class Root:\n    def _candidate_items(self, modules, config): return ('other',)\n"
            "class Other(Root): pass\n",
            "Other, CrossModuleCandidateDetector[int]",
            False,
        ),
        (
            "class Root:\n    def _candidate_items(self, modules, config): return ('other',)\n"
            "class Other(Root): pass\n",
            "CrossModuleCandidateDetector[int], Other",
            True,
        ),
        (
            "class Other(object):\n    marker = 'independent'\n",
            "Other, CrossModuleCandidateDetector[int]",
            True,
        ),
        (
            "class Other:\n    exec(\"def _candidate_items(self, modules, config): return ('other',)\")\n",
            "Other, CrossModuleCandidateDetector[int]",
            False,
        ),
        (
            "class Other:\n"
            "    def classmethod(function):\n"
            "        import sys\n"
            "        sys._getframe(1).f_locals['_candidate_items'] = lambda self, modules, config: ('other',)\n"
            "        return function\n"
            "    @classmethod\n"
            "    def helper(): pass\n"
            "    del classmethod\n",
            "Other, CrossModuleCandidateDetector[int]",
            False,
        ),
        (
            "class Other:\n"
            "    def _candidate_items(self, modules, config): return ('other',)\n"
            "    del _candidate_items\n",
            "Other, CrossModuleCandidateDetector[int]",
            True,
        ),
        (
            "class Other:\n    _candidate_items: int\n",
            "Other, CrossModuleCandidateDetector[int]",
            True,
        ),
        (
            "class Other:\n"
            "    def helper(self, value: dict[str, list[int]]) -> tuple[int, str]: pass\n",
            "Other, CrossModuleCandidateDetector[int]",
            True,
        ),
        (
            "class Other:\n"
            "    def helper(value=(exec('raise RuntimeError()') for item in ())): pass\n",
            "Other, CrossModuleCandidateDetector[int]",
            True,
        ),
        *(
            (
                future + "class Annotation:\n"
                "    def __class_getitem__(cls, item):\n"
                "        import sys\n"
                "        sys._getframe(1).f_locals['_candidate_items'] = lambda self, modules, config: ('other',)\n"
                "        return int\n"
                "class Other:\n"
                "    def helper(self, value: Annotation[int]): pass\n",
                "Other, CrossModuleCandidateDetector[int]",
                bool(future) or sys.version_info >= (3, 14),
            )
            for future in ("", "from __future__ import annotations\n")
        ),
        (
            "class Other:\n"
            "    def unused(self):\n"
            "        exec(\"raise RuntimeError('not called')\")\n",
            "Other, CrossModuleCandidateDetector[int]",
            True,
        ),
        (
            "class Predicate:\n"
            "    def __bool__(self):\n"
            "        import sys\n"
            "        sys._getframe(1).f_locals['_candidate_items'] = lambda self, modules, config: ('other',)\n"
            "        return True\n"
            "predicate = Predicate()\n"
            "class Other:\n    if predicate: pass\n",
            "Other, CrossModuleCandidateDetector[int]",
            False,
        ),
        (
            "class Iterable:\n"
            "    def __iter__(self):\n"
            "        import sys\n"
            "        sys._getframe(1).f_locals['_candidate_items'] = lambda self, modules, config: ('other',)\n"
            "        return iter(())\n"
            "items = Iterable()\n"
            "class Other:\n    for item in items: pass\n",
            "Other, CrossModuleCandidateDetector[int]",
            False,
        ),
        (
            "class Other(ConfiguredCrossModuleCollectorCandidateDetector[int]): pass\n",
            "Other, CrossModuleCandidateDetector[int]",
            False,
        ),
        (
            "class Other:\n    def _candidate_items(self, modules, config): return ('other',)\n",
            "Other, CrossModuleCandidateDetector[int]",
            False,
        ),
        (
            "class Other:\n    def _candidate_items(self, modules, config): return ('other',)\n",
            "CrossModuleCandidateDetector[int], Other",
            True,
        ),
        (
            "class Other:\n    @classmethod\n    def required_candidate_collector(cls): return lambda modules: ('other',)\n",
            "Other, CrossModuleCandidateDetector[int]",
            True,
        ),
        (
            "class Other:\n    @classmethod\n    def required_candidate_collector(cls): return lambda modules: ('other',)\n",
            "CrossModuleCandidateDetector[int], Other",
            True,
        ),
        (
            "class Other:\n    marker = 'independent'\n",
            "Other, CrossModuleCandidateDetector[int]",
            True,
        ),
        (
            "class Other:\n    marker = 'independent'\n",
            "CrossModuleCandidateDetector[int], Other",
            True,
        ),
        *(
            (
                "class Effect:\n"
                + "".join(
                    f"    def {signature}:\n"
                    "        import sys\n"
                    "        sys._getframe(1).f_locals['_candidate_items'] = lambda self, modules, config: ('other',)\n"
                    f"        return {result}\n"
                    for signature, result in (
                        ("__pos__(self)", "1"),
                        ("__bool__(self)", "True"),
                        ("__hash__(self)", "1"),
                        ("__radd__(self, other)", "1"),
                        ("__iter__(self)", "iter(())"),
                    )
                )
                + "effect = Effect()\nclass Other:\n"
                + indent(body, "    "),
                "Other, CrossModuleCandidateDetector[int]",
                False,
            )
            for body in (
                "def helper(value=+effect): pass\n",
                "def helper(value=(effect and 1)): pass\n",
                "def helper(value={effect: 1}): pass\n",
                "marker = 0\nmarker += effect\n",
                "def helper(value=(item for item in effect)): pass\n",
            )
        ),
    ),
)
def test_collector_dispatch_follows_native_mro(
    tmp_path: Path,
    native_collector_module: ParsedModule,
    other_source: str,
    bases: str,
    safe: bool,
) -> None:
    future = "from __future__ import annotations\n"
    source = (
        (future if other_source.startswith(future) else "")
        + "from nominal_refactor_advisor.detectors._base import CrossModuleCandidateDetector, ConfiguredCrossModuleCollectorCandidateDetector\n"
        + other_source.removeprefix(future)
        + "def collect(modules): return ('collected',)\n"
        + f"class Owner({bases}):\n"
        + "    def _candidate_items(self, modules, config): return collect(modules)\n"
        + "print(Owner()._candidate_items([], None))\n"
    )
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    assert before.strip() == "('collected',)"
    module = ParsedModule(path, "probe", False, ast.parse(source), source)
    snapshot = CodemodSourceSnapshot.from_modules((module, native_collector_module))
    plan = CodemodPlanSequence.from_operations(
        (
            DeriveCandidateCollectorOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(),
                    qualname="Owner._candidate_items",
                )
            ),
        )
    )
    if safe:
        simulation = plan.simulate(snapshot)
        assert simulation.is_clean
        simulation.apply()
        assert subprocess.check_output([sys.executable, str(path)], text=True) == before
    else:
        with pytest.raises(ValueError):
            plan.simulate(snapshot)
        assert path.read_bytes() == source.encode("utf-8")
