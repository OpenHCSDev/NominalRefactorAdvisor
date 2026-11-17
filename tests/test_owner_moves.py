"""The two owner-choice moves: create a mixin, extend a partial owner (#67)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from dataclasses import replace

import pytest

from nominal_refactor_advisor.analysis import analyze_modules
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_overlap import (
    NO_FLOOR,
    ClassOverlapMetrics,
    ComposeExisting,
    ExtractMixin,
)
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
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



def test_a_member_inherited_past_an_unordered_base_is_already_had(tmp_path: Path) -> None:
    """`Task` already inherits `Step`, behind a base the scan cannot order: it has
    `Step`'s members whichever ancestor wins their lookup, so composing `Action`
    hands it nothing new."""

    records = ALONE.replace(
        "from dataclasses import dataclass\n", "from dataclasses import dataclass\n\nfrom vendor import External\n", 1
    ).replace("class Task(Phase):", "class Task(External, Step, Phase):")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records})
    verdicts = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path, use_parse_cache=False)
    ).class_overlap_verdicts
    verdict = verdicts.verdict(
        verdicts.bodies.overlap("pkg.records.Action", "pkg.records.Task"), NO_FLOOR
    )
    assert verdict is not None
    assert not [r for r in verdict.receipts if r.kind is OverlapReceiptKind.UNWANTED_SUPPLY]


LAYOUTS = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True, slots=True)\n"
    "class Viewer:\n"
    "    viewer: str\n\n\n"
    "@dataclass(frozen=True, slots=True)\n"
    "class Basis:\n"
    "    root: str\n"
    "    project: str\n"
    "    peer: str\n"
    "    shown: int\n\n\n"
    "@dataclass(frozen=True, slots=True)\n"
    "class Identity(Viewer):\n"
    "    root: str\n"
    "    project: str\n"
    "    peer: str\n"
    "    seen: int\n\n\n"
    "@dataclass(frozen=True, eq=False)\n"
    "class Loose:\n"
    "    root: str\n"
    "    project: str\n"
    "    peer: str\n"
    "    note: str = ''\n"
)

LAYOUTS_PROBE = (
    "from pkg.records import Basis, Identity, Loose\n"
    "made = (Basis(root='r', project='p', peer='q', shown=1), Identity(viewer='v', root='r', project='p', peer='q', seen=2),"
    " Loose(root='r', project='p', peer='q'))\n"
    "for value in made:\n"
    "    kind = type(value)\n"
    "    print(repr(value), value == kind(**{k: getattr(value, k) for k in kind.__dataclass_fields__}),"
    " hasattr(value, '__dict__'), kind.__eq__ is object.__eq__, list(kind.__dataclass_fields__))\n"
)


def test_a_mixin_takes_the_declaration_every_composer_can_inherit(tmp_path: Path) -> None:
    """`Basis` and `Identity` are slotted and `Identity` already has a slotted
    base with a field, so a slotted mixin would be a second layout; `Loose` has
    a `__dict__` and no `__eq__`. The mixin lays out nothing, generates no
    `__eq__`, and each class keeps its layout, equality and fields."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": LAYOUTS})
    before = _run(tmp_path, LAYOUTS_PROBE)
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    mixin = next(
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.summary.startswith("`Identity` restate")
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
                    owner="new_mixin:pkg.records",
                    because="new_class",
                    name="Located",
                    judgement="the root, project and peer a record is about",
                ),
            ),
        ),
    ).run()

    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked", "covered", "covered"]
    source = (tmp_path / "pkg/records.py").read_text()
    assert (
        "@dataclass(eq=False, frozen=True)\nclass Located:\n"
        '    """The root, project and peer a record is about."""\n\n    __slots__ = ()\n\n    root: str\n'
    ) in source, source
    assert "class Identity(Located, Viewer):\n" in source, source
    assert _run(tmp_path, LAYOUTS_PROBE) == before


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


def test_a_mixin_step_names_every_input_it_still_needs(tmp_path: Path) -> None:
    """Independent checks all run: a bad class name and a quoted docstring are
    both named in one refusal, so one scan shows everything the pick lacks."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": FIELD_RECORDS})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = replace(_field_mixin(str(tmp_path / "pkg/records.py")), mixin_name="class", docstring='say """hi"""')

    with pytest.raises(CodemodOperationPreflightError) as refused:
        operation.source_edits(snapshot)
    assert "one-line docstring" in str(refused.value) and "`class` is no class name" in str(refused.value)


def test_every_raises_one_refusal_with_each_need() -> None:
    def proof() -> None:
        raise NeedsInput("a proof", NeededInput.PROOF)

    def pick() -> None:
        raise NeedsInput("a pick", NeededInput.OWNER_PICK)

    with pytest.raises(NeedsInput) as refused:
        NeedsInput.every(proof, lambda: None, pick)
    assert str(refused.value) == "a proof; a pick"
    assert refused.value.needs == (NeededInput.OWNER_PICK, NeededInput.PROOF)
    NeedsInput.every(lambda: None)


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


LETTERED = (
    "    alpha: int\n"
    "    beta: str\n\n"
    "    def gamma(self) -> str:\n"
    "        return f'{self.beta}{self.alpha}'\n\n"
)
LAYERED = {
    "pkg/__init__.py": "",
    "pkg/base.py": "UNIT = 1\n",
    "pkg/low.py": (
        "from dataclasses import dataclass\n\nfrom pkg.base import UNIT\n\n\n"
        "@dataclass(frozen=True)\nclass First:\n" + LETTERED + "    own_first: int = 0\n\n\n"
        "def unit():\n    return UNIT\n"
    ),
    "pkg/high.py": (
        "from dataclasses import dataclass\n\nfrom pkg.base import UNIT\nfrom pkg.low import First\n\n\n"
        "@dataclass(frozen=True)\nclass Second:\n" + LETTERED + "    own_second: int = 0\n\n\n"
        "def pair():\n    return First(alpha=1, beta='x'), Second(alpha=2, beta='y')\n"
    ),
}
LAYERED_PROBE = (
    "from pkg.high import pair\n"
    "print([(value.gamma(), value) for value in pair()])\n"
)


def _mixin_finding(root: Path):
    return next(
        finding
        for finding in analyze_modules(
            parse_python_modules(root, use_parse_cache=False), DetectorConfig()
        )
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )


def test_a_mixin_whose_anchor_module_would_close_a_cycle_lands_in_a_module_both_import(
    tmp_path: Path,
) -> None:
    """`pkg.high` imports `pkg.low`, so a mixin declared beside `Second` would close
    a cycle when `First` composes it. Copied from `Second`, it is declared in
    `pkg.base` instead, which both import and which imports neither; both compose
    it there and the program runs as before."""

    _write(tmp_path, LAYERED)
    before = _run(tmp_path, LAYERED_PROBE)
    finding = _mixin_finding(tmp_path)
    assert "new_mixin:pkg.base" in {
        owner.candidate_id for owner in finding.metrics.owner_candidates
    }
    modules = parse_python_modules(tmp_path, use_parse_cache=False)

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(
            modules=modules, findings=analyze_modules(modules, DetectorConfig())
        ),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=finding.stable_id,
                    owner="new_mixin:pkg.base",
                    because="placed_below",
                    name="Lettered",
                    judgement="a lettered record",
                ),
            ),
        ),
    ).run()

    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked"]
    base = (tmp_path / "pkg/base.py").read_text()
    assert "class Lettered:" in base and "import pkg" not in base and "from pkg" not in base
    assert "class First(Lettered):" in (tmp_path / "pkg/low.py").read_text()
    assert "class Second(Lettered):" in (tmp_path / "pkg/high.py").read_text()
    assert _run(tmp_path, LAYERED_PROBE) == before
    assert _run(
        tmp_path,
        "import pkg.low, pkg.high\n"
        "print(pkg.low.First.__mro__[1] is pkg.high.Second.__mro__[1], "
        "pkg.low.First.__mro__[1].__module__)\n",
    ) == "True pkg.base\n"


def _owner_files(package: str) -> dict[str, str]:
    prefix, dotted = (f"{package}/", f"{package}.") if package else ("", "")
    body = (
        "    name: str\n    reason: str\n\n"
        "    def value_or(self, default):\n        return default\n\n"
        "    def label(self):\n        return f'{self.name}: {self.reason}'\n"
    )
    files = {
        f"{prefix}base.py": "UNIT = 1\n",
        f"{prefix}wire.py": (
            f"from dataclasses import dataclass\n\nfrom {dotted}base import UNIT\n\n\n"
            "@dataclass(frozen=True)\nclass Omitted:\n" + body + "\n"
            "    def wire(self):\n        return {'omitted': self.name}\n\n\n"
            "def omitted(name):\n    return Omitted(name=name, reason='unit')\n"
        ),
        f"{prefix}tools.py": (
            f"from dataclasses import dataclass\n\nfrom {dotted}base import UNIT\n"
            f"from {dotted}wire import omitted\n\n\n"
            "@dataclass(frozen=True)\nclass Unsupplied:\n" + body + "\n\n"
            "def unsupplied(name):\n    return Unsupplied(name=name, reason='missing'), omitted(name)\n"
        ),
    }
    if package:
        files[f"{package}/__init__.py"] = ""
    return files


def _composition(root: Path):
    from nominal_refactor_advisor.refactors.class_overlap import (
        ClassOverlapFindingRecipeSynthesizer,
    )

    (finding,) = (
        finding
        for finding in analyze_modules(
            parse_python_modules(root, use_parse_cache=False), DetectorConfig()
        )
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ComposeExisting
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))
    return ClassOverlapFindingRecipeSynthesizer().evaluate_recipe_for_finding(finding, snapshot), snapshot


def test_an_owner_composing_would_close_a_cycle_moves_down_first(tmp_path: Path) -> None:
    """`Omitted` restates `Unsupplied`, but `pkg.tools`, which declares the owner,
    imports `pkg.wire`: composing it there closes a cycle. The owner moves to
    `pkg.base`, which both import and which imports neither, and `Omitted`
    composes it there."""

    from nominal_refactor_advisor.codemod import CodemodPlanSequence

    _write(tmp_path, _owner_files("pkg"))
    probe = (
        "from pkg.tools import unsupplied\n"
        "print([(value.label(), value.value_or(3), value) for value in unsupplied('n')])\n"
    )
    before = _run(tmp_path, probe)

    evaluation, snapshot = _composition(tmp_path)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()

    assert "class Unsupplied:" in (tmp_path / "pkg/base.py").read_text()
    assert "class Unsupplied" not in (tmp_path / "pkg/tools.py").read_text()
    assert "class Omitted(Unsupplied):" in (tmp_path / "pkg/wire.py").read_text()
    assert _run(tmp_path, probe) == before


def test_an_owner_no_module_can_hold_without_a_cycle_refuses_saying_why(tmp_path: Path) -> None:
    """Here `wire` does not import `base`, so no module both import avoids the
    cycle, and top-level modules share no package for a new one: the move
    refuses and says so."""

    files = _owner_files("")
    files["wire.py"] = files["wire.py"].replace("from base import UNIT\n", "")
    _write(tmp_path, files)

    evaluation, _ = _composition(tmp_path)

    assert evaluation.rejection_reason == (
        "`tools` already imports, directly or not, `wire`, so `Unsupplied` moves "
        "first; but no module that `tools`, `wire` all import avoids the cycle, and "
        "they share no package to hold a new one"
    )


def _delivery_files(package: str) -> dict[str, str]:
    """`After` restates `Delivered`, whose module imports `After`'s: composing it
    there closes a cycle. `Delivered` reads `Reference` from `After`'s module and
    its base `Delivery` from its own, so no module below both can hold it."""

    prefix, dotted = (f"{package}/", f"{package}.") if package else ("", "")
    files = {
        f"{prefix}low.py": "UNIT = 1\n",
        f"{prefix}ref.py": (
            f"from dataclasses import dataclass\n\nfrom {dotted}low import UNIT\n\n\n"
            "@dataclass(frozen=True)\nclass Reference:\n    seq: int\n\n\n"
            "@dataclass(frozen=True)\nclass After:\n    reference: Reference\n\n"
            "    def refs(self):\n        return (self.reference,)\n\n"
            "    def cursor(self):\n        return self.reference.seq + UNIT\n"
        ),
        f"{prefix}origin.py": (
            f"from dataclasses import dataclass\n\nfrom {dotted}low import UNIT\n"
            f"from {dotted}ref import Reference\n\n\n"
            "class Delivery:\n    def refs(self):\n        raise NotImplementedError\n\n\n"
            "@dataclass(frozen=True)\nclass Delivered(Delivery):\n    reference: Reference\n\n"
            "    def refs(self):\n        return (self.reference,)\n\n\n"
            "class Undelivered(Delivery):\n    def refs(self):\n        return ()\n\n\n"
            "def both():\n    return Delivered(reference=Reference(seq=3)).refs(), Undelivered().refs(), UNIT\n"
        ),
    }
    if package:
        files[f"{package}/__init__.py"] = ""
    return files


def test_a_moved_owner_takes_what_it_reads_to_a_home_that_can_import_the_rest(tmp_path: Path) -> None:
    """`pkg.low` is imported by both and imports neither, but `Delivered` reads
    `Reference` from `pkg.ref`, which `pkg.low` cannot import without a cycle.
    So `Delivered` moves into `pkg.ref` itself, taking its base `Delivery` from
    its own module with it, and `After` composes it there."""

    from nominal_refactor_advisor.codemod import CodemodPlanSequence

    _write(tmp_path, _delivery_files("pkg"))
    probe = (
        "from pkg.origin import both\nfrom pkg.ref import After, Reference\n"
        "after = After(reference=Reference(seq=1))\nprint(both(), after.refs(), after.cursor(), after)\n"
    )
    before = _run(tmp_path, probe)

    evaluation, snapshot = _composition(tmp_path)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()

    ref = (tmp_path / "pkg/ref.py").read_text()
    origin = (tmp_path / "pkg/origin.py").read_text()
    assert "class Delivery:" in ref and "class Delivered(Delivery):" in ref, ref
    assert "class After(Delivered):" in ref, ref
    assert "class Delivered" not in origin and "class Delivery:" not in origin, origin
    assert "class Undelivered(Delivery):" in origin
    assert "from pkg.low" not in ref.replace("from pkg.low import UNIT", "")
    assert _run(tmp_path, probe) == before


def test_a_moved_owner_whose_reads_pin_it_refuses_naming_them(tmp_path: Path) -> None:
    """Top-level modules share no package, so only a module both import could
    hold `Delivered`, and it reads `Reference` from `ref`, one of them."""

    _write(tmp_path, _delivery_files(""))

    evaluation, _ = _composition(tmp_path)

    assert evaluation.rejection_reason == (
        "`origin` already imports, directly or not, `ref`, so `Delivered` moves first; "
        "but `Delivery` and `Delivered` read names from `ref`, which no module below "
        "`origin`, `ref` can import without a cycle"
    )


def test_a_moved_owner_prefers_a_home_evaluating_annotations_as_its_module_does(
    tmp_path: Path,
) -> None:
    """`pkg.annotated` and `pkg.base` both qualify; `pkg.annotated` comes first by name,
    but it stringizes annotations and `pkg.tools` does not, so the owner goes to
    `pkg.base`, whose annotations evaluate as before."""

    from nominal_refactor_advisor.codemod import CodemodPlanSequence

    files = _owner_files("pkg")
    files["pkg/annotated.py"] = "from __future__ import annotations\n\nTWO = 2\n"
    for name in ("pkg/wire.py", "pkg/tools.py"):
        files[name] = files[name].replace(
            "from pkg.base import UNIT\n", "from pkg.base import UNIT\nfrom pkg.annotated import TWO\n"
        )
    _write(tmp_path, files)

    evaluation, snapshot = _composition(tmp_path)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()

    assert "class Unsupplied:" in (tmp_path / "pkg/base.py").read_text()
    assert "class Unsupplied" not in (tmp_path / "pkg/annotated.py").read_text()
