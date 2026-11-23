"""Guarded and function-local imports that close no cycle, found, planned
together and moved to their module's top.

Each applying case writes a package, runs it, applies the recipe the finding
offers, checks the rewritten source, runs it again and re-scans.
"""

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.analysis import (
    AnalysisPathScope,
    analyze_compact_roots_with_cache,
    analyze_detector_types,
)
from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_module_roots,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._hidden_import_cuts import HiddenImportCycleDetector
from nominal_refactor_advisor.detectors._hidden_imports import HiddenImportDetector
from nominal_refactor_advisor.hidden_imports import (
    ClosesCycle,
    GuardedImport,
    HiddenImportPlan,
    LocalImport,
    UnpromotableImport,
    UnscannedImportBridge,
)
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in {"pkg/__init__.py": "", **files}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(root: Path) -> str:
    return subprocess.check_output([sys.executable, "main.py"], cwd=root, text=True)


def _modules(root: Path):
    return parse_python_modules(root, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False))


def _plan(root: Path) -> HiddenImportPlan:
    return HiddenImportPlan.derive(_modules(root))


def _findings(root: Path):
    with ScanCache.scope():
        return HiddenImportDetector().detect(_modules(root), DetectorConfig())


def _apply_all(root: Path) -> None:
    for finding in _findings(root):
        snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))
        evaluation = HiddenImportDetector().evaluate_recipe_for_finding(finding, snapshot)
        assert evaluation.rejection_reason == ""
        simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
        assert simulation.is_clean
        simulation.apply()


def test_hidden_import_continues_the_stable_sequence():
    assert PatternId.HIDDEN_IMPORT == 54


def test_imports_that_close_no_cycle_move_to_the_top(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/shapes.py": "class Square:\n    side = 2\n",
            "pkg/area.py": (
                "from __future__ import annotations\n\n"
                "from typing import TYPE_CHECKING\n\n"
                "if TYPE_CHECKING:\n    from pkg.shapes import Square\n\n\n"
                "def area(square: 'Square') -> int:\n    return square.side ** 2\n\n\n"
                "def unit():\n    from pkg.shapes import Square\n\n    return area(Square())\n\n\n"
                "def again():\n    from pkg.shapes import Square\n"
            ),
            "main.py": "from pkg.area import unit\n\nprint(unit())\n",
        },
    )
    expected = _run(tmp_path)
    plan = _plan(tmp_path)
    assert [type(hidden) for hidden in plan.accepted] == [GuardedImport, LocalImport, LocalImport]
    assert plan.refused == ()
    [finding] = _findings(tmp_path)
    assert finding.summary == "`pkg.area` hides 3 import(s) that close no cycle."
    _apply_all(tmp_path)
    assert (tmp_path / "pkg/area.py").read_text() == (
        "from __future__ import annotations\n\n\nfrom pkg.shapes import Square\n\n"
        "def area(square: 'Square') -> int:\n    return square.side ** 2\n\n\n"
        "def unit():\n\n    return area(Square())\n\n\n"
        "def again():\n    pass\n"
    )
    assert _run(tmp_path) == expected
    assert _findings(tmp_path) == []


def test_an_import_that_closes_a_cycle_names_the_module_that_reaches_back(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/a.py": "def b_value():\n    from pkg.b import VALUE\n\n    return VALUE\n",
            "pkg/b.py": "import pkg.a\n\nVALUE = 1\n",
        },
    )
    [refusal] = _plan(tmp_path).refused
    assert isinstance(refusal, ClosesCycle)
    assert refusal.reaching == "pkg.b"
    assert "`pkg.b` already imports `pkg.a` at load" in str(refusal.need)
    assert refusal.need.needs == (NeededInput.SOURCE_CHANGE,)
    assert _findings(tmp_path) == []


def test_two_imports_free_alone_but_cyclic_together_promote_one(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/a.py": "A = 1\n\n\ndef b_value():\n    from pkg.b import B\n\n    return B\n",
            "pkg/b.py": "B = 2\n\n\ndef a_value():\n    from pkg.a import A\n\n    return A\n",
            "main.py": "from pkg.a import b_value\nfrom pkg.b import a_value\n\nprint(b_value(), a_value())\n",
        },
    )
    expected = _run(tmp_path)
    plan = _plan(tmp_path)
    assert [hidden.location for hidden in plan.accepted] == [f"{tmp_path / 'pkg/a.py'}:5"]
    [refusal] = plan.refused
    assert isinstance(refusal, ClosesCycle)
    assert (refusal.hidden.module.module_name, refusal.reaching) == ("pkg.b", "pkg.a")
    _apply_all(tmp_path)
    assert (tmp_path / "pkg/a.py").read_text().startswith("from pkg.b import B\n")
    assert _run(tmp_path) == expected
    assert _findings(tmp_path) == []


def test_an_import_whose_name_the_module_binds_otherwise_stays(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/b.py": "def VALUE():\n    return 1\n",
            "pkg/a.py": (
                "VALUE = 'module'\n\n\ndef read():\n    from pkg.b import VALUE\n\n    return VALUE()\n\n\n"
                "def outer():\n    open = 1\n\n    def inner():\n        from pkg.b import VALUE as open\n\n"
                "        return open\n\n    return inner\n\n\n"
                "def builtin():\n    from pkg.b import VALUE as len\n\n    return len\n"
            ),
        },
    )
    plan = _plan(tmp_path)
    assert plan.accepted == ()
    assert all(isinstance(refusal, UnpromotableImport) for refusal in plan.refused)
    assert [str(refusal.need).partition(": ")[2] for refusal in plan.refused] == [
        "`VALUE` is already bound at module scope to something else",
        "an enclosing function binds `open`, which would shadow the module's",
        "`len` is already bound at module scope to something else",
    ]
    assert all(refusal.need.needs == (NeededInput.SOURCE_CHANGE,) for refusal in plan.refused)
    assert _findings(tmp_path) == []


def test_an_import_of_a_name_its_module_does_not_have_stays_hidden(tmp_path: Path):
    """A guarded import never runs, so it can name a class that moved away; at the module's
    top it would run at load time and the module would stop importing."""
    _write(
        tmp_path,
        {
            "pkg/moved.py": "class Moved:\n    pass\n",
            "pkg/old_home.py": "OTHER = 1\n",
            "pkg/a.py": (
                "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from pkg.old_home import Moved\n"
                "    from pkg.moved import Moved as Here\n\n\ndef read(value: 'Moved') -> None:\n    return None\n"
            ),
            "main.py": "import pkg.a\nprint('ok')\n",
        },
    )
    plan = _plan(tmp_path)
    stale = [refusal for refusal in plan.refused if "does not bind" in str(refusal.need)]
    assert [str(refusal.need).partition(": ")[2] for refusal in stale] == [
        "`pkg.old_home` does not bind `Moved`: the import names something its module does not have"
    ]
    assert stale[0].need.needs == (NeededInput.SOURCE_CHANGE,)
    assert not any("old_home" in hidden.identity for hidden in plan.accepted)
    assert _run(tmp_path) == "ok\n"


def test_an_import_after_its_function_changes_sys_path_names_the_path_setup_it_needs(tmp_path: Path):
    """A hook that puts a checkout on `sys.path` and then imports from it: at the
    module's top the import would run before the path is set. Another function's
    import, with no path change before it, still moves."""

    _write(
        tmp_path,
        {
            "pkg/base.py": "LIMIT = 3\n",
            "pkg/hook.py": (
                "import sys\nfrom pathlib import Path\n\n\n"
                "def gate():\n"
                "    sys.path.insert(0, str(Path(__file__).parent))\n"
                "    from pkg.base import LIMIT\n\n"
                "    return LIMIT\n\n\n"
                "def plain():\n"
                "    from pkg.base import LIMIT\n\n"
                "    return LIMIT\n"
            ),
        },
    )
    plan = _plan(tmp_path)
    [refused] = plan.refused
    assert isinstance(refused, UnpromotableImport) and refused.hidden.symbol == "gate"
    assert "changes `sys.path` at line 6" in str(refused.need)
    assert NeededInput.SOURCE_CHANGE in refused.need.needs
    [accepted] = plan.accepted
    assert accepted.symbol == "plain"


def test_compact_root_analysis_matches_full_for_hidden_imports_and_cycles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(
        root,
        {
            "pkg/leaf.py": "VALUE = 1\n",
            "pkg/hub.py": (
                "from pkg.reader import helper\n\n\n"
                "class Shared:\n"
                "    pass\n\n\n"
                "def use() -> int:\n"
                "    return helper()\n"
            ),
            "pkg/reader.py": (
                "from __future__ import annotations\n\n"
                "from typing import TYPE_CHECKING\n\n"
                "if TYPE_CHECKING:\n"
                "    from pkg.hub import Shared\n"
                "    from pkg.leaf import VALUE\n\n\n"
                "def helper() -> int:\n"
                "    from pkg.leaf import VALUE\n\n"
                "    return VALUE\n"
            ),
        },
    )
    detectors = (HiddenImportDetector, HiddenImportCycleDetector)
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
    assert {finding.detector_id for finding in cold.findings} == {
        HiddenImportDetector.detector_id,
        HiddenImportCycleDetector.detector_id,
    }


def test_three_hop_cross_subpackage_cycle_is_refused_in_auto_and_partial_scans(
    tmp_path: Path,
) -> None:
    """Regression for agent-comms thread_presentation -> recipient_activity ->
    messaging.notification_progress -> thread_presentation: subpackage directory
    scans auto-widen to the enclosing package and catch ClosesCycle, and even a
    partial scan without messaging/ refuses via UnscannedImportBridge."""

    src = tmp_path / "src"
    _write(
        src,
        {
            "pkg/registry/__init__.py": "",
            "pkg/cli_verbs/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/registry/thread_presentation.py": (
                "class ThreadPresentationStatus:\n    pass\n\n\n"
                "def make_activity():\n"
                "    from pkg.registry.recipient_activity import UnavailableRecipientActivity\n"
                "    return UnavailableRecipientActivity()\n"
            ),
            "pkg/registry/recipient_activity.py": (
                "from pkg.messaging.notification_progress import AssignmentProgress\n\n\n"
                "class UnavailableRecipientActivity:\n"
                "    progress = AssignmentProgress\n"
            ),
            "pkg/messaging/notification_progress.py": (
                "from pkg.registry.thread_presentation import ThreadPresentationStatus\n\n\n"
                "class AssignmentProgress:\n"
                "    status = ThreadPresentationStatus\n"
            ),
            "pkg/cli_verbs/status.py": "STATUS = 'ok'\n",
        },
    )
    registry = src / "pkg" / "registry"
    cli_verbs = src / "pkg" / "cli_verbs"

    # 1. Default auto-context widens subpackage directories to src/pkg and sees the 3-hop cycle.
    scope = AnalysisPathScope.from_requested_roots((registry, cli_verbs), auto_context=True)
    assert scope.analysis_roots == (src / "pkg",)
    full_modules = parse_python_module_roots(scope.analysis_roots, use_parse_cache=False)
    full_plan = HiddenImportPlan.derive(full_modules)
    assert full_plan.accepted == ()
    assert [type(refusal) for refusal in full_plan.refused] == [ClosesCycle]
    assert (
        analyze_detector_types(
            full_modules,
            DetectorConfig(),
            detector_types=(HiddenImportDetector,),
        )
        == []
    )

    # 2. Even when auto-context is disabled (partial scan omitting pkg/messaging),
    # promoting through an unscanned same-package import refuses with NeededInput.PROOF.
    partial_modules = parse_python_module_roots((registry, cli_verbs), use_parse_cache=False)
    partial_plan = HiddenImportPlan.derive(partial_modules)
    assert partial_plan.accepted == ()
    [refusal] = partial_plan.refused
    assert isinstance(refusal, UnscannedImportBridge)
    assert refusal.loader == "pkg.registry.recipient_activity"
    assert refusal.unscanned == "pkg.messaging.notification_progress"
    assert refusal.need.needs == (NeededInput.PROOF,)
    assert (
        analyze_detector_types(
            partial_modules,
            DetectorConfig(),
            detector_types=(HiddenImportDetector,),
        )
        == []
    )
    compact_partial = analyze_compact_roots_with_cache(
        (registry, cli_verbs),
        DetectorConfig(),
        cache_dir=tmp_path / "cache" / "ast",
        analysis_cache_dir=tmp_path / "cache" / "analysis",
        detector_types=(HiddenImportDetector,),
    )
    assert compact_partial.findings == []
