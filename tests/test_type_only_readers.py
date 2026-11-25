"""A module that imports a class under `if TYPE_CHECKING:` only to annotate with
it, where promoting the import would close a cycle, is retyped to a base of the
class below it; the whole-tree type check refuses a retype a consumer needs more of.

Each case writes a package, runs it, applies the cycle finding's recipe, and runs
it again.
"""

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._hidden_import_cuts import HiddenImportCycleDetector
from nominal_refactor_advisor.hidden_imports import HiddenImportPlan
from nominal_refactor_advisor.refactors.hidden_import_cuts import HiddenImportCuts, HubRetype, HubStays
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.type_checker_session import (
    FineGrainedTypeChecker,
    SourceTexts,
    TypeCheckerSession,
    TypeErrors,
    _incremental_type_errors_slot,
)

BASE = (
    "from abc import ABC, abstractmethod\n\n\nclass Service(ABC):\n"
    "    @abstractmethod\n    def ping(self) -> int: ...\n"
)
ROWS = 'class StoreRows:\n    """A marker rows inherit."""\n\n    __slots__ = ()\n'
REGISTRY = (
    "class Service:\n    members: dict[str, type] = {}\n\n"
    "    def __init_subclass__(cls, **kwargs: object) -> None:\n"
    "        super().__init_subclass__()\n        Service.members[cls.__name__] = cls\n\n"
    "    def ping(self) -> int:\n        return 1\n"
)
READER = (
    "from __future__ import annotations\n\nfrom typing import TYPE_CHECKING\n\n"
    "if TYPE_CHECKING:\n    from pkg.hub import Hub\n\n\n"
    "def run(hub: Hub) -> int:\n    return relay(hub)\n\n\n"
    "def relay(hub: 'Hub') -> int:\n    return hub.ping()\n"
)


def _write(root: Path, hub: str, reader: str = READER, base: str = BASE) -> None:
    files = {
        "pkg/__init__.py": "",
        "pkg/service.py": base,
        "pkg/rows.py": ROWS,
        "pkg/hub.py": hub,
        "pkg/reader.py": reader,
        "main.py": "from pkg.hub import main\n\nprint(main())\n",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(root: Path) -> str:
    return subprocess.check_output([sys.executable, "main.py"], cwd=root, text=True)


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))


def _evaluation(root: Path):
    detector = HiddenImportCycleDetector()
    with ScanCache.scope():
        [finding] = detector.detect(parse_python_modules(root, use_parse_cache=False), DetectorConfig())
    snapshot = _snapshot(root)
    return detector.evaluate_recipe_for_finding(finding, snapshot), snapshot


HUB = (
    "from pkg.reader import run\nfrom pkg.service import Service\n\n\n"
    "class Hub(Service):\n    def ping(self) -> int:\n        return 1\n\n    def only(self) -> int:\n        return 2\n\n\n"
    "def main() -> int:\n    return run(Hub())\n"
)


def test_a_reader_that_only_annotates_with_the_class_names_its_base_and_drops_the_guard(tmp_path: Path):
    _write(tmp_path, HUB)
    expected = _run(tmp_path)
    [plan] = [plan for plan in HiddenImportCuts.of(_snapshot(tmp_path)).hubs() if isinstance(plan, HubRetype)]
    assert [(read.name, read.base_module, read.base_name) for read in plan.reads] == [("Hub", "pkg.service", "Service")]
    evaluation, snapshot = _evaluation(tmp_path)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    reader = (tmp_path / "pkg/reader.py").read_text()
    assert "TYPE_CHECKING:\n    from pkg.hub" not in reader
    assert "from pkg.service import Service\n" in reader
    assert "def run(hub: Service) -> int:" in reader and "def relay(hub: 'Service') -> int:" in reader
    assert _run(tmp_path) == expected
    assert HiddenImportPlan.derive(parse_python_modules(tmp_path, use_parse_cache=False)).refused == ()


def test_a_retype_whose_value_reaches_a_member_only_the_class_declares_is_refused_naming_it(tmp_path: Path):
    reader = READER + "\n\ndef keep(hub: Hub) -> Hub:\n    return hub\n"
    hub = HUB.replace("return run(Hub())", "return run(Hub()) + keep(Hub()).only()").replace(
        "from pkg.reader import run", "from pkg.reader import keep, run"
    )
    _write(tmp_path, hub, reader)
    evaluation, _snapshot_ = _evaluation(tmp_path)
    reason = evaluation.rejection_reason
    assert reason.startswith("retyping `Hub` to `Service` in `reader.py` breaks 1 consumer(s): ")
    assert "pkg/hub.py:" in reason and '"Service" has no attribute "only"' in reason


def _stays(root: Path) -> list[str]:
    return [
        str(need)
        for plan in HiddenImportCuts.of(_snapshot(root)).hubs()
        if isinstance(plan, HubStays)
        for _refusal, need in plan.needs
    ]


def test_a_guarded_import_the_module_reads_nowhere_is_a_re_export_for_type_checkers(tmp_path: Path):
    reader = (
        '"""Serves `Hub` to type checkers."""\n\nfrom __future__ import annotations\n\n'
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from pkg.hub import Hub\n\n"
        '__all__ = ["Hub", "run"]\n\n\ndef run(hub) -> int:\n    return hub.ping()\n'
    )
    _write(tmp_path, HUB.replace("class Hub(Service):", "class Hub:"), reader)
    [need] = [need for need in _stays(tmp_path) if "pkg.reader" in need]
    assert need == (
        "`pkg.reader` guards `Hub` but reads it nowhere: "
        "a re-export for type checkers, whose place is the module's owner's to decide"
    )


def test_a_guarded_type_alias_has_no_base_to_name(tmp_path: Path):
    hub = HUB.replace("return 2", "return run(self)").replace(
        "\n\ndef main()", "\n\nHandle = Hub | int\n\n\ndef main()"
    )
    reader = READER.replace("import Hub", "import Handle").replace("hub: Hub", "hub: Handle").replace(
        "hub: 'Hub'", "hub: 'Handle'"
    ).replace("return hub.ping()", "return 1")
    _write(tmp_path, hub, reader)
    [need] = [need for need in _stays(tmp_path) if "pkg.reader" in need]
    assert need == (
        "`pkg.reader` reads `Handle` only as a type, but `pkg.hub.Handle` is a type alias, not a class: "
        "it has no base to name; the alias sitting below it is the input"
    )


def _retyped_to(root: Path) -> list[tuple[str, str, str]]:
    return [
        (read.name, read.base_module, read.base_name)
        for plan in HiddenImportCuts.of(_snapshot(root)).hubs()
        if isinstance(plan, HubRetype)
        for read in plan.reads
    ]


def test_a_base_that_only_supplies_code_is_never_named_for_the_class(tmp_path: Path):
    hub = HUB.replace("from pkg.service import Service", "from pkg.rows import StoreRows").replace(
        "class Hub(Service):", "class Hub(StoreRows):"
    )
    _write(tmp_path, hub)
    assert _retyped_to(tmp_path) == []
    [need] = [need for need in _stays(tmp_path) if "pkg.reader" in need]
    assert need == (
        "`pkg.reader` reads `Hub` only as a type, but its bases below it, `pkg.rows.StoreRows`, "
        "supply code and declare no contract (no abstract member, no registry family): "
        "a contract base declared below it, which the class inherits, is the input"
    )


def test_a_mixin_nearer_than_the_contract_base_is_passed_over_for_it(tmp_path: Path):
    hub = HUB.replace(
        "from pkg.service import Service", "from pkg.rows import StoreRows\nfrom pkg.service import Service"
    ).replace("class Hub(Service):", "class Hub(StoreRows, Service):")
    _write(tmp_path, hub)
    assert _retyped_to(tmp_path) == [("Hub", "pkg.service", "Service")]


def test_a_registry_family_class_is_a_contract_base(tmp_path: Path):
    _write(tmp_path, HUB, base=REGISTRY)
    assert _retyped_to(tmp_path) == [("Hub", "pkg.service", "Service")]


def test_fine_grained_type_errors_match_cold_build_across_edits_and_reverts(tmp_path: Path):
    """Incremental `TypeErrors.of` inside `ScanCache.scope()` must match cold
    `TypeErrors._built` across clean edits, newly introduced errors, pre-existing
    errors in untouched modules, and reverts back to the baseline."""

    pkg = tmp_path / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "base.py").write_text("class Base:\n    x: int = 1\n\n\nclass Sub(Base):\n    y: str = 'ok'\n")
    reader_path = pkg / "reader.py"
    original_reader = "from pkg.base import Sub\n\n\ndef read(item: Sub) -> str:\n    return item.y\n"
    broken_reader = "from pkg.base import Base\n\n\ndef read(item: Base) -> str:\n    return item.y\n"
    reader_path.write_text(original_reader)
    (pkg / "preexisting.py").write_text("def broken() -> int:\n    return 'not-an-int'\n")

    with ScanCache.scope():
        mods_initial = parse_python_modules(tmp_path, use_parse_cache=False)
        st_initial = SourceTexts(tuple(mods_initial))
        inc_initial = TypeErrors.of(st_initial)
        cold_initial = TypeErrors._built(st_initial)
        assert inc_initial.by_file == cold_initial.by_file
        engine = _incremental_type_errors_slot()[0]
        assert engine is not None

        reader_path.write_text(broken_reader)
        mods_broken = parse_python_modules(tmp_path, use_parse_cache=False)
        st_broken = SourceTexts(tuple(mods_broken))
        inc_broken = TypeErrors.of(st_broken)
        cold_broken = TypeErrors._built(st_broken)
        assert inc_broken.by_file == cold_broken.by_file
        assert inc_broken.new_since(inc_initial) == cold_broken.new_since(cold_initial)
        assert len(inc_broken.new_since(inc_initial)) == 1
        assert _incremental_type_errors_slot()[0] is engine

        reader_path.write_text(original_reader)
        mods_reverted = parse_python_modules(tmp_path, use_parse_cache=False)
        st_reverted = SourceTexts(tuple(mods_reverted))
        inc_reverted = TypeErrors.of(st_reverted)
        assert inc_reverted.by_file == cold_initial.by_file
        assert inc_reverted.new_since(inc_initial) == ()
        assert _incremental_type_errors_slot()[0] is engine


def test_forked_fine_grained_type_errors_keep_parent_engine_at_baseline(
    tmp_path: Path, capsys
):
    """On a 16+ module corpus, speculative `TypeErrors.of` deltas run in a COW
    child on Linux so the parent engine stays at baseline across sibling trials,
    and stray mypy stdout/stderr is suppressed."""

    pkg = tmp_path / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "base.py").write_text(
        "class Base:\n    x: int = 1\n\n\nclass Sub(Base):\n    y: str = 'ok'\n"
    )
    for idx in range(15):
        (pkg / f"m{idx}.py").write_text(
            f"from pkg.base import Sub\n\n\ndef f{idx}(item: Sub) -> str:\n    return item.y\n"
        )
    target_path = pkg / "m0.py"
    original_target = target_path.read_text()
    broken_target = (
        "from pkg.base import Base\n\n\ndef f0(item: Base) -> str:\n    return item.y\n"
    )

    with ScanCache.scope():
        mods_initial = parse_python_modules(tmp_path, use_parse_cache=False)
        st_initial = SourceTexts(tuple(mods_initial))
        seeded_checker = TypeCheckerSession.seed_incremental(st_initial)
        inc_initial = TypeErrors.of(st_initial)
        engine = _incremental_type_errors_slot()[0]
        assert engine is not None
        if sys.platform.startswith("linux") and isinstance(
            seeded_checker, FineGrainedTypeChecker
        ):
            assert engine._manager is seeded_checker._manager
        baseline_text = engine._text_by_module["pkg.m0"]

        target_path.write_text(broken_target)
        mods_broken = parse_python_modules(tmp_path, use_parse_cache=False)
        st_broken = SourceTexts(tuple(mods_broken))
        inc_broken = TypeErrors.of(st_broken)
        assert len(inc_broken.new_since(inc_initial)) == 1
        assert _incremental_type_errors_slot()[0] is engine
        if sys.platform.startswith("linux"):
            assert engine._text_by_module["pkg.m0"] == baseline_text

        target_path.write_text(original_target)
        mods_reverted = parse_python_modules(tmp_path, use_parse_cache=False)
        inc_reverted = TypeErrors.of(SourceTexts(tuple(mods_reverted)))
        assert inc_reverted.new_since(inc_initial) == ()

        # Syncing FineGrainedTypeChecker in-place updates the shared error table
        # in lockstep instead of dropping the shared engine for a second build.
        target_path.write_text(broken_target)
        mods_synced_broken = parse_python_modules(tmp_path, use_parse_cache=False)
        st_synced_broken = SourceTexts(tuple(mods_synced_broken))
        synced = TypeCheckerSession.started(st_synced_broken)
        assert synced is seeded_checker
        assert _incremental_type_errors_slot()[0] is engine
        if sys.platform.startswith("linux"):
            assert len(TypeErrors.of(st_synced_broken).new_since(inc_initial)) == 1
            assert TypeErrors.of(st_initial).new_since(inc_initial) == ()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
