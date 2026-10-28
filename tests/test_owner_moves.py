"""The two owner-choice moves: create a mixin, extend a partial owner (#67)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis import analyze_modules
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_overlap import (
    ClassOverlapMetrics,
    ComposeExisting,
    ExtractMixin,
)
from nominal_refactor_advisor.class_overlap_codemod import (
    CreateMixinOperation,
    ExtendOwnerOperation,
)
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_architecture_guards import ArchitectureGuardSuite
from nominal_refactor_advisor.codemod_workflow import (
    CodemodRefactorGoalRunner,
    CodemodWorkflowScan,
)
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.owner_candidates import OwnerPick
from nominal_refactor_advisor.owner_picks import OwnerPicks
from nominal_refactor_advisor.refactor_concepts import ClassBodyCompositionConcept

RECORDS = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class First:\n"
    "    alpha: int\n"
    "    beta: str\n\n"
    "    def gamma(self) -> str:\n"
    "        return f'{self.beta}{self.alpha}'\n\n"
    "    own_first: int = 0\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Second:\n"
    "    alpha: int\n"
    "    beta: str\n\n"
    "    def gamma(self) -> str:\n"
    "        return f'{self.beta}{self.alpha}'\n\n"
    "    own_second: int = 0\n"
)

HOLDER = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Holder:\n"
    "    alpha: int\n"
    "    beta: str\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Uses(Holder):\n"
    "    extra: int = 0\n\n\n"
    "def build():\n"
    "    return Uses(alpha=1, beta='b')\n"
)

PROBE = (
    "from pkg.records import First, Second\n"
    "print(First(alpha=1, beta='x', own_first=2).gamma(), "
    "Second(alpha=3, beta='y').gamma())\n"
)


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _apply(root: Path, operation: RefactorRecipeOperation) -> dict[str, str]:
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    result = RefactorRecipe(recipe_id="owner-move").with_operation(replayed).simulate(snapshot)
    assert result.is_clean is True
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source, encoding="utf-8")
    return result.simulation.rewritten_sources


def _run(root: Path, program: str) -> str:
    return subprocess.run(
        [sys.executable, "-c", program], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def test_a_new_mixin_is_declared_and_each_class_then_composes_it(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS})
    before = _run(tmp_path, PROBE)
    target = SourceRewriteTarget(file_path=str(tmp_path / "pkg/records.py"), qualname="First")

    _apply(
        tmp_path,
        CreateMixinOperation(
            target=target, member_names=("alpha", "beta", "gamma"), mixin_name="Lettered"
        ),
    )

    source = (tmp_path / "pkg/records.py").read_text()
    assert "@dataclass(frozen=True)\nclass Lettered:\n" in source
    assert _run(tmp_path, PROBE) == before
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    composes = {
        candidate.restating.qualname
        for candidate in snapshot.class_overlap_verdicts.candidates
        if candidate.move is ComposeExisting
        and candidate.owner.qualname == "Lettered"
        and candidate.permitted
    }
    assert composes == {"First", "Second"}


def test_a_mixin_name_already_bound_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = CreateMixinOperation(
        target=SourceRewriteTarget(file_path=str(tmp_path / "pkg/records.py"), qualname="First"),
        member_names=("alpha", "beta", "gamma"),
        mixin_name="Second",
    )

    with pytest.raises(CodemodOperationPreflightError, match="already bound"):
        operation.source_edits(snapshot)


def test_extending_an_owner_its_descendants_construct_without_the_field_is_refused(
    tmp_path: Path,
) -> None:
    records = RECORDS.replace("    alpha: int\n    beta: str\n", "    alpha: int\n    beta: str\n    delta: float\n")
    _write(
        tmp_path,
        {"pkg/__init__.py": "", "pkg/records.py": records, "pkg/holder.py": HOLDER},
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = ExtendOwnerOperation(
        target=SourceRewriteTarget(file_path=str(tmp_path / "pkg/holder.py"), qualname="Holder"),
        anchor=SourceRewriteTarget(file_path=str(tmp_path / "pkg/records.py"), qualname="First"),
        member_names=("delta",),
    )

    with pytest.raises(CodemodOperationPreflightError, match="breaks 1 construction"):
        operation.source_edits(snapshot)


def test_an_owner_gains_the_declarations_it_lacks_and_is_then_composed(
    tmp_path: Path,
) -> None:
    records = RECORDS.replace(
        "    alpha: int\n    beta: str\n",
        "    alpha: int\n    beta: str\n    delta: float = 0.0\n",
    )
    _write(
        tmp_path,
        {"pkg/__init__.py": "", "pkg/records.py": records, "pkg/holder.py": HOLDER},
    )
    before = _run(tmp_path, PROBE + "from pkg.holder import build\nprint(build())\n")

    _apply(
        tmp_path,
        ExtendOwnerOperation(
            target=SourceRewriteTarget(
                file_path=str(tmp_path / "pkg/holder.py"), qualname="Holder"
            ),
            anchor=SourceRewriteTarget(
                file_path=str(tmp_path / "pkg/records.py"), qualname="First"
            ),
            member_names=("delta", "gamma"),
        ),
    )

    after = _run(tmp_path, PROBE + "from pkg.holder import build\nprint(build())\n")
    assert after.splitlines()[0] == before.splitlines()[0]
    assert after.splitlines()[1] == "Uses(alpha=1, beta='b', delta=0.0, extra=0)"
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert {
        candidate.restating.qualname
        for candidate in snapshot.class_overlap_verdicts.candidates
        if candidate.move is ComposeExisting and candidate.owner.qualname == "Holder"
    } >= {"First", "Second"}


def test_a_goal_run_executes_the_picked_owner_and_reports_the_pick(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS})
    before = _run(tmp_path, PROBE)
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    unpicked = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks("picks.json", "0", ()),
    ).run()
    assert [outcome.outcome for outcome in unpicked.owner_picks.outcomes] == ["missing"]
    assert unpicked.stage_count == 0

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=mixin.stable_id,
                    owner="new_mixin:pkg.records",
                    because="new_class",
                    name="Lettered",
                ),
            ),
        ),
    ).run()

    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked"]
    source = (tmp_path / "pkg/records.py").read_text()
    assert "class First(Lettered):\n    own_first: int = 0\n" in source
    assert "class Second(Lettered):\n    own_second: int = 0\n" in source
    assert _run(tmp_path, PROBE) == before


def test_a_pick_citing_a_receipt_its_owner_lacks_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS})
    findings = analyze_modules(
        parse_python_modules(tmp_path, use_parse_cache=False), DetectorConfig()
    )
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    picks = OwnerPicks(
        "picks.json",
        "0",
        (
            OwnerPick(
                finding=mixin.stable_id,
                owner="new_mixin:pkg.records",
                because="holds_declaration",
                name="Lettered",
            ),
            OwnerPick(finding="0000000000", owner="new_mixin:pkg.records", because="new_class"),
        ),
    )

    applied, report = picks.applied(findings)

    (outcome,) = report.outcomes
    assert outcome.outcome == "rejected"
    assert "carries no receipt of" in outcome.detail
    assert report.unmatched == ("0000000000",)
    assert all(
        finding.metrics.owner_pick is None
        for finding in applied
        if isinstance(finding.metrics, ClassOverlapMetrics)
    )


def test_a_pick_without_an_owner_declines_the_finding(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS})
    findings = analyze_modules(
        parse_python_modules(tmp_path, use_parse_cache=False), DetectorConfig()
    )
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    picks = OwnerPicks(
        "picks.json",
        "0",
        (OwnerPick(finding=mixin.stable_id, owner=None, because="not one concept"),),
    )

    _applied, report = picks.applied(findings)

    assert [(outcome.outcome, outcome.detail) for outcome in report.outcomes] == [
        ("declined", "not one concept")
    ]


REORDERED = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class First:\n"
    "    lead: int\n"
    "    alpha: int\n"
    "    beta: str\n\n"
    "    def gamma(self) -> str:\n"
    "        return f'{self.beta}{self.alpha}'\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Second:\n"
    "    alpha: int\n"
    "    beta: str\n\n"
    "    def gamma(self) -> str:\n"
    "        return f'{self.beta}{self.alpha}'\n\n"
    "    own_second: int = 0\n"
)

REORDERED_PROBE = (
    "from pkg.records import First, Second\n"
    "print(First(lead=0, alpha=1, beta='x').gamma(), "
    "Second(alpha=3, beta='y', own_second=4).gamma())\n"
)


def _goal_with_pick(root: Path, name: str) -> object:
    modules = parse_python_modules(root, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    return CodemodRefactorGoalRunner(
        roots=(root,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=mixin.stable_id,
                    owner="new_mixin:pkg.records",
                    because="new_class",
                    name=name,
                ),
            ),
        ),
    ).run()


def test_a_reordering_mixin_is_keyword_only_and_so_is_each_composer(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": REORDERED})
    before = _run(tmp_path, REORDERED_PROBE)

    report = _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 3
    assert "@dataclass(frozen=True, kw_only=True)\nclass Lettered:\n" in source
    assert "@dataclass(frozen=True, kw_only=True)\nclass First(Lettered):\n    lead: int\n" in source
    assert "@dataclass(frozen=True, kw_only=True)\nclass Second(Lettered):\n" in source
    assert _run(tmp_path, REORDERED_PROBE) == before


def test_a_reordered_composer_that_orders_by_field_position_is_not_composed(
    tmp_path: Path,
) -> None:
    ordered = REORDERED.replace(
        "@dataclass(frozen=True)\nclass First:", "@dataclass(frozen=True, order=True)\nclass First:"
    ).replace(
        "@dataclass(frozen=True)\nclass Second:", "@dataclass(frozen=True, order=True)\nclass Second:"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": ordered})

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 0
    assert (tmp_path / "pkg/records.py").read_text() == ordered


def test_a_reordered_composer_whose_field_order_reaches_a_digest_is_not_composed(
    tmp_path: Path,
) -> None:
    digested = "import hashlib\n" + REORDERED.replace(
        "    own_second: int = 0\n",
        "    own_second: int = 0\n\n"
        "    def identity(self) -> str:\n"
        "        return hashlib.blake2b(repr(self).encode()).hexdigest()\n",
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": digested})

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 0
    assert (tmp_path / "pkg/records.py").read_text() == digested
