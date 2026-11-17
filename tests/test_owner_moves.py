"""The two owner-choice moves: create a mixin, extend a partial owner (#67)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis import analyze_modules
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_overlap import (
    NO_FLOOR,
    ClassOverlapMetrics,
    ComposeExisting,
    ExtractMixin,
)
from nominal_refactor_advisor.overlap_receipts import OverlapReceiptKind
from nominal_refactor_advisor.refactors.carrier_fields import (
    CarrierFieldProjection,
    ReplaceFieldsWithCarrierOperation,
)
from nominal_refactor_advisor.refactors.class_overlap import (
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
from nominal_refactor_advisor.detectors._base import DetectorConfig
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
            target=target,
            member_names=("alpha", "beta", "gamma"),
            mixin_name="Lettered",
            docstring="A lettered record.",
        ),
    )

    source = (tmp_path / "pkg/records.py").read_text()
    assert '@dataclass(frozen=True)\nclass Lettered:\n    """A lettered record."""\n' in source
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
        docstring="A lettered record.",
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
                    judgement="a lettered record",
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
                judgement="a lettered record",
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
                    judgement="a lettered record",
                ),
            ),
        ),
    ).run()


def test_a_reordering_mixin_is_keyword_only_and_so_is_each_composer(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": REORDERED})
    before = _run(tmp_path, REORDERED_PROBE)

    report = _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 1
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


def test_a_picks_file_makes_the_picked_findings_the_only_targets(tmp_path: Path) -> None:
    """A compose-existing recipe nobody picked is in the source; the goal leaves it."""

    holder = (
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Point:\n"
        "    x: int\n"
        "    y: int\n"
        "    w: int\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Located:\n"
        "    x: int\n"
        "    y: int\n"
        "    w: int\n"
        "    z: int = 0\n"
    )
    _write(
        tmp_path,
        {"pkg/__init__.py": "", "pkg/records.py": RECORDS, "pkg/points.py": holder},
    )

    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    unpicked = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=True,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(
            modules=modules, findings=analyze_modules(modules, DetectorConfig())
        ),
        migration_type=ClassBodyCompositionConcept,
    ).run()
    assert unpicked.stage_count >= 1

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 1
    assert (tmp_path / "pkg/points.py").read_text() == holder


def test_a_mixin_one_composer_cannot_compose_is_not_created(tmp_path: Path) -> None:
    """Creating the mixin and composing it is one step: a refused composer refuses all."""

    positional = RECORDS + (
        "\n\ndef build():\n    return Second(1, 'b')\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": positional})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    records = str(tmp_path / "pkg/records.py")
    operation = CreateMixinOperation(
        target=SourceRewriteTarget(file_path=records, qualname="First"),
        member_names=("alpha", "beta", "gamma"),
        mixin_name="Lettered",
        docstring="A lettered record.",
        composers=(
            SourceRewriteTarget(file_path=records, qualname="First"),
            SourceRewriteTarget(file_path=records, qualname="Second"),
            SourceRewriteTarget(file_path=records, qualname="Missing"),
        ),
    )

    with pytest.raises(CodemodOperationPreflightError):
        operation.source_edits(snapshot)


def test_a_positional_construction_is_keyworded_first_then_the_pick_composes(
    tmp_path: Path,
) -> None:
    """A reordering pick refused only because a composer is built by position takes
    two stages: every construction bound by keyword, then the composition."""
    positional = REORDERED + "\n\ndef build():\n    return First(0, 1, 'x')\n"
    probe = "from pkg.records import build\nprint(build().gamma(), build().lead)\n"
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": positional})
    before = _run(tmp_path, probe)

    report = _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 2
    assert "return First(lead=0, alpha=1, beta='x')" in source
    assert "class First(Lettered):\n" in source
    assert _run(tmp_path, probe) == before



def test_a_positional_construction_in_a_script_is_keyworded_first_too(
    tmp_path: Path,
) -> None:
    """The only positional construction sits in Python a string holds and a child
    interpreter runs: the scan counts it, so the goal binds it by keyword first."""
    script = (
        "import subprocess, sys\n\n"
        "SCRIPT = (\n"
        '    "from pkg.records import First\\n"\n'
        '    "print(First(0, 1, \'x\').lead)\\n"\n'
        ")\n\n\n"
        "def run():\n"
        "    return subprocess.run([sys.executable, '-c', SCRIPT], check=True, capture_output=True, text=True).stdout\n"
    )
    probe = "from pkg.scripts import run\nprint(run())\n"
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": REORDERED, "pkg/scripts.py": script})
    before = _run(tmp_path, probe)

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 2
    assert "First(lead=0, alpha=1, beta=\'x\').lead" in (tmp_path / "pkg/scripts.py").read_text()
    assert "class First(Lettered):\n" in (tmp_path / "pkg/records.py").read_text()
    assert _run(tmp_path, probe) == before


DOCUMENTED = REORDERED.replace(
    "class Second:\n    alpha: int\n    beta: str\n\n    def gamma(self) -> str:\n        return",
    "class Second:\n    alpha: int\n    beta: str\n    \"\"\"The letter.\"\"\"\n\n"
    "    def gamma(self) -> str:\n        \"\"\"Letter then number.\"\"\"\n        return",
)


def test_the_one_copy_kept_carries_every_composers_documentation(tmp_path: Path) -> None:
    """The anchor's copies are undocumented; the other composer documents a
    method and a field. The mixin keeps both docstrings, and the field's
    docstring goes with its field instead of staying behind."""
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": DOCUMENTED})
    before = _run(tmp_path, REORDERED_PROBE)

    _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert '    beta: str\n    """The letter."""\n\n    def gamma(self) -> str:\n        """Letter then number."""\n' in source
    assert source.count('"""The letter."""') == 1
    assert "class Second(Lettered):\n    own_second: int = 0\n" in source
    assert _run(tmp_path, REORDERED_PROBE) == before


def test_two_docstrings_for_one_member_refuse(tmp_path: Path) -> None:
    documented = DOCUMENTED.replace(
        "class First:\n    lead: int\n    alpha: int\n    beta: str\n",
        "class First:\n    lead: int\n    alpha: int\n    beta: str\n    \"\"\"A letter.\"\"\"\n",
    )
    assert documented != DOCUMENTED
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": documented})

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 0
    assert "class Lettered" not in (tmp_path / "pkg/records.py").read_text()

COPIED = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class First:\n"
    "    alpha: int\n"
    "    beta: str\n"
    "    gamma: float\n"
    "    own_first: int = 0\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Second:\n"
    "    alpha: int\n"
    "    beta: str\n"
    "    gamma: float\n"
    "    own_second: int = 0\n\n\n"
    "def summarize(first: First) -> Second:\n"
    "    return Second(alpha=first.alpha, beta=first.beta, gamma=first.gamma)\n\n\n"
    "def describe(second: Second) -> str:\n"
    "    return f'{second.beta}{second.alpha}{second.gamma}'\n"
)


def test_a_hold_source_pick_holds_the_value_it_copied(tmp_path: Path) -> None:
    """A class built by copying one value's block holds that value instead."""
    probe = (
        "from pkg.records import First, describe, summarize\n"
        "print(describe(summarize(First(alpha=1, beta='x', gamma=2.5))))\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": COPIED})
    before = _run(tmp_path, probe)
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (copy,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    hold = next(
        owner for owner in copy.metrics.owner_candidates if owner.kind == "hold_source"
    )

    CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=copy.stable_id,
                    owner=hold.candidate_id,
                    because=hold.receipts[0].kind.value,
                    name="first",
                ),
            ),
        ),
    ).run()

    source = (tmp_path / "pkg/records.py").read_text()
    assert "class Second:\n    first: First\n" in source
    assert "return Second(first=first)" in source
    assert _run(tmp_path, probe) == before


def test_holding_the_source_rewrites_a_reader_in_another_file(
    tmp_path: Path,
) -> None:
    """A typed reader of the copier in another module reads through the held source."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/records.py": COPIED,
            "pkg/report.py": "from pkg.records import Second\n\n\ndef label(second: Second) -> str:\n    return second.beta\n",
        },
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = ReplaceFieldsWithCarrierOperation(
        target=SourceRewriteTarget(file_path=str(tmp_path / "pkg/records.py"), qualname="Second"),
        field_projections=tuple(
            CarrierFieldProjection(source_field=name, carrier_attribute=name)
            for name in ("alpha", "beta", "gamma")
        ),
        carrier_field_declaration="first: First",
    )

    simulation = RefactorRecipe("hold").with_operation(operation).simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    report = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/report.py")]
    assert "return second.first.beta\n" in report



FIELD_RECORDS = (
    "from collections import OrderedDict\n"
    "from dataclasses import dataclass, field\n\n\n"
    "@dataclass(frozen=True)\n"
    "class First:\n"
    "    alpha: int\n"
    "    beta: str\n"
    "    items: list = field(default_factory=list, compare=False)\n"
    "    own_first: int = 0\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Second:\n"
    "    alpha: int\n"
    "    beta: str\n"
    "    items: list = field(default_factory=list, compare=False)\n"
    "    own_second: int = 0\n"
)

FIELD_PROBE = (
    "from pkg.records import First, Second\n"
    "a, b = First(alpha=1, beta='x'), Second(alpha=2, beta='y')\n"
    "a.items.append(1)\n"
    "print(a, b, a.items is not First(alpha=1, beta='x').items)\n"
)


HOISTED = (
    'from dataclasses import dataclass\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class Action:\n'
    '    problem: str\n'
    '    order: int\n'
    "    request: str = ''\n"
    '\n'
    "    def bound_to(self, request: str) -> 'Action':\n"
    '        return self\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class Other(Action):\n'
    '    revision: int = 0\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class Create(Action):\n'
    '    repo: str\n'
    '    text: str\n'
    '\n'
    '    def reserve(self) -> str:\n'
    '        return self.repo + self.text\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class Phase:\n'
    '    review: float = 1.0\n'
    '\n'
    '    def due(self) -> float:\n'
    '        return self.review\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class Task(Phase):\n'
    '    problem: str\n'
    '    repo: str\n'
    '    text: str\n'
    '    order: int\n'
    '\n'
    '    def checked(self) -> bool:\n'
    '        return self.review > 0\n'
)


HOISTED_PROBE = (
    "from pkg.records import Create, Other, Task\n"
    "print(Create(problem='p', order=1, repo='r', text='t').reserve(), "
    "Other(problem='q', order=2, revision=3).bound_to('x'), "
    "Task(problem='p', repo='r', text='t', order=1).checked())\n"
)


def test_an_inherited_share_is_hoisted_out_of_its_base_then_the_rest_is_extracted(
    tmp_path: Path,
) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": HOISTED})
    before = _run(tmp_path, HOISTED_PROBE)
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
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
                    owner="hoisting_mixin:pkg.records",
                    because="hoists_inherited",
                    name="Filed",
                    judgement="the repository and text a problem is filed with",
                    hoisted_name="Numbered",
                    hoisted_judgement="the problem and the order it belongs to",
                ),
            ),
        ),
    ).run()

    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked"]
    source = (tmp_path / "pkg/records.py").read_text()
    assert "class Numbered:\n" in source and "class Filed:\n" in source
    assert "class Action(Numbered):\n    request: str = ''\n" in source
    assert "class Create(Filed, Action):\n    def reserve" in source
    assert "class Task(Filed, Numbered, Phase):\n    def checked" in source
    assert _run(tmp_path, HOISTED_PROBE) == before



ALONE = HOISTED.replace(
    "class Action:\n    problem: str\n    order: int\n    request: str = ''\n",
    "class Step:\n    request: str = ''\n\n    def bound_to(self, request: str) -> 'Step':\n        return self\n\n\n"
    "@dataclass(frozen=True, kw_only=True)\nclass Action(Step):\n    problem: str\n    order: int\n",
).replace("    def bound_to(self, request: str) -> 'Action':\n        return self\n\n\n", "\n\n")


def test_a_base_declaring_only_the_hoisted_names_is_hoisted_too(tmp_path: Path) -> None:
    """`Action` declares nothing but the names `Task` restates, so the pair
    reads as a duplicate class whose repair composes `Action`. The hoist extracts
    them instead, and its step is proved as that extraction: `Task` must not
    become a `Step`."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": ALONE})
    before = _run(tmp_path, HOISTED_PROBE)
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
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
                    owner="hoisting_mixin:pkg.records",
                    because="hoists_inherited",
                    name="Filed",
                    judgement="the repository and text a problem is filed with",
                    hoisted_name="Numbered",
                    hoisted_judgement="the problem and the order it belongs to",
                ),
            ),
        ),
    ).run()

    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked"]
    source = (tmp_path / "pkg/records.py").read_text()
    assert "class Numbered:\n" in source and "class Filed:\n" in source
    assert "class Action(Numbered, Step):\n    pass\n" in source
    assert "class Task(Filed, Numbered, Phase):\n    def checked" in source
    assert _run(tmp_path, HOISTED_PROBE) == before



@pytest.mark.parametrize(
    ("records", "gained"),
    (
        (ALONE, ["`Step.bound_to`", "`Step.request`"]),
        (ALONE.replace("class Action(Step):", "class Action:"), []),
    ),
)
def test_composing_an_owner_never_hands_a_class_its_ancestors_members(
    tmp_path: Path, records: str, gained: list[str]
) -> None:
    """`Task` restates all `Action` declares, but inheriting `Action` would also
    give it `Step`'s field and method, which nothing requires `Task` to have:
    unwanted supply, which the move must refuse. Without `Step` it is sound."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records})
    verdicts = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path, use_parse_cache=False)
    ).class_overlap_verdicts
    verdict = verdicts.verdict(
        verdicts.bodies.overlap("pkg.records.Action", "pkg.records.Task"), NO_FLOOR
    )
    assert verdict is not None and verdict.move is ComposeExisting
    refusals = [
        receipt.detail
        for receipt in verdict.refusals
        if receipt.kind is OverlapReceiptKind.UNWANTED_SUPPLY
    ]
    assert [name for detail in refusals for name in gained if name in detail] == gained
    assert bool(refusals) == bool(gained) and verdict.permitted is not bool(gained)


def _field_mixin(records: str) -> CreateMixinOperation:
    return CreateMixinOperation(
        target=SourceRewriteTarget(file_path=records, qualname="First"),
        member_names=("alpha", "beta", "items"),
        mixin_name="Lettered",
        docstring="A lettered record with its items.",
        composers=(
            SourceRewriteTarget(file_path=records, qualname="First"),
            SourceRewriteTarget(file_path=records, qualname="Second"),
        ),
    )


def test_a_field_default_moves_into_the_new_mixin(tmp_path: Path) -> None:
    """`dataclasses.field` builds no value that learns its class: the decorator
    consumes it, so the block moves and each instance still gets its own list."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": FIELD_RECORDS})
    before = _run(tmp_path, FIELD_PROBE)

    _apply(tmp_path, _field_mixin(str(tmp_path / "pkg/records.py")))

    source = (tmp_path / "pkg/records.py").read_text()
    assert source.count("field(default_factory=list, compare=False)") == 1
    assert "class First(Lettered):" in source and "class Second(Lettered):" in source
    assert _run(tmp_path, FIELD_PROBE) == before


def test_a_moved_value_from_an_unindexed_class_still_refuses(tmp_path: Path) -> None:
    """Only `dataclasses.field` is admitted: any other call may build a field or a
    value that learns its class, so the order stays unproved and the move refuses."""

    records = FIELD_RECORDS.replace(
        "field(default_factory=list, compare=False)", "OrderedDict()"
    ).replace("    items: list", "    items: OrderedDict")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(CodemodOperationPreflightError, match="field order .* is unknown"):
        _field_mixin(str(tmp_path / "pkg/records.py")).source_edits(snapshot)
