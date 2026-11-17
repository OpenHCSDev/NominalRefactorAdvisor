"""Ranked owner candidates for a block two classes share (#67)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    CompactModuleClassProjectionFamily,
    build_compact_class_family_index,
)
from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts, ExtractMixin
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.owner_candidates import OwnerReceiptKind

BLOCK = (
    "    alpha: int\n"
    "    beta: str\n"
    "    def gamma(self):\n"
    "        return self.alpha\n"
)


def _module(module_name: str, source: str) -> ParsedModule:
    path = Path(*module_name.split(".")).with_suffix(".py")
    return ParsedModule(
        path=path,
        module_name=module_name,
        module=ast.parse(source, filename=str(path)),
        source=source,
    )


def _mixin_verdicts(*modules: tuple[str, str]):
    projections = CompactModuleClassProjectionFamily.collect_modules(
        tuple(_module(name, source) for name, source in modules)
    )
    verdicts = ClassOverlapVerdicts(
        tuple(projections), build_compact_class_family_index(projections)
    )
    return [
        candidate for candidate in verdicts.candidates if candidate.move is ExtractMixin
    ]


def test_candidates_are_ranked_by_a_softmax_over_their_receipts() -> None:
    (verdict,) = _mixin_verdicts(
        (
            "pkg.records",
            "class First:\n" + BLOCK + "    own_first: int\n\n\n"
            "class Second:\n" + BLOCK + "    own_second: int\n",
        ),
    )

    (owner,) = verdict.owner_candidates
    assert owner.candidate_id == "new_mixin:pkg.records"
    assert owner.confidence == 1.0
    assert owner.cost.imports_added == 0
    payload = json_report_object(owner)
    assert payload["kind"] == "new_mixin"
    assert payload["log_odds"] == sum(receipt["weight"] for receipt in payload["receipts"])


LOW = "class First:\n" + BLOCK + "    own_first: int\n"
HIGH = "class Second:\n" + BLOCK + "    own_second: int\n"


def test_a_new_mixin_that_would_close_a_cycle_moves_to_a_module_both_import() -> None:
    (verdict,) = _mixin_verdicts(
        ("pkg.base", "UNIT = 1\n"),
        ("pkg.low", "from pkg.base import UNIT\n\n\n" + LOW),
        ("pkg.high", "from pkg.base import UNIT\nfrom pkg.low import First\n\n\n" + HIGH),
    )
    owners = {owner.candidate_id: owner for owner in verdict.owner_candidates}

    assert set(owners) == {"new_mixin:pkg.low", "new_mixin:pkg.base"}
    placed = owners["new_mixin:pkg.base"]
    assert placed.permitted and placed.anchor == "pkg.high.Second"
    assert placed.cost.cycles_created == 0
    (moved,) = (r for r in placed.receipts if r.kind is OwnerReceiptKind.PLACED_BELOW)
    assert moved.detail == (
        "not in `pkg.high`: `pkg.high` already imports, directly or not, `pkg.low`: "
        "the new import closes a cycle; `pkg.base` is imported, directly or not, by "
        "`pkg.high`, `pkg.low` and imports none of them"
    )


def test_with_no_module_both_import_the_mixin_gets_a_new_module_of_their_package() -> None:
    (verdict,) = _mixin_verdicts(
        ("pkg.low", LOW), ("pkg.high", "from pkg.low import First\n\n\n" + HIGH)
    )
    owners = {owner.candidate_id: owner for owner in verdict.owner_candidates}

    low, new = owners["new_mixin:pkg.low"], owners["new_module_mixin:pkg"]
    assert low.permitted and new.permitted
    assert low.confidence > new.confidence
    assert "new_mixin:pkg.high" not in owners


def test_a_new_mixin_no_module_can_hold_without_a_cycle_is_refused_saying_why() -> None:
    (verdict,) = _mixin_verdicts(("low", LOW), ("high", "from low import First\n\n\n" + HIGH))
    owners = {owner.candidate_id: owner for owner in verdict.owner_candidates}

    high = owners["new_mixin:high"]
    assert not high.permitted and high.confidence == 0.0
    assert [receipt.detail for receipt in high.refusals] == [
        "`high` already imports, directly or not, `low`: the new import closes a cycle",
        (
            "no module that `high`, `low` all import avoids the cycle, and they share "
            "no package to hold a new one"
        ),
    ]
    assert high.cost.cycles_created == 1


def test_an_existing_composed_class_holding_most_of_the_block_outranks_a_new_mixin() -> None:
    (verdict,) = _mixin_verdicts(
        (
            "pkg.records",
            "class Holder:\n    alpha: int\n    beta: str\n\n\n"
            "class Uses(Holder):\n    pass\n\n\n"
            "class First:\n" + BLOCK + "    own_first: int\n\n\n"
            "class Second:\n" + BLOCK + "    own_second: int\n",
        ),
    )

    ranked = verdict.owner_candidates
    assert ranked[0].candidate_id == "existing_class:pkg.records.Holder"
    assert {receipt.kind for receipt in ranked[0].receipts} >= {
        OwnerReceiptKind.HOLDS_DECLARATION
    }
    assert math.isclose(sum(owner.confidence for owner in ranked), 1.0, abs_tol=1e-3)


REORDERED = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\nclass First:\n"
    "    lead: int\n    alpha: int\n    beta: str\n    gamma: float\n\n\n"
    "@dataclass(frozen=True)\nclass Second:\n"
    "    alpha: int\n    beta: str\n    gamma: float\n    own: int\n"
)


def test_a_mixin_that_reorders_dataclass_fields_makes_them_keyword_only() -> None:
    (verdict,) = _mixin_verdicts(("pkg.records", REORDERED))

    (owner,) = verdict.owner_candidates
    assert owner.permitted
    assert OwnerReceiptKind.KEYWORD_ONLY_COMPOSITION in {
        receipt.kind for receipt in owner.receipts
    }


def test_a_reordered_composer_built_by_position_refuses_the_mixin() -> None:
    (verdict,) = _mixin_verdicts(
        ("pkg.records", REORDERED),
        ("pkg.use", "from pkg.records import First\n\n\ndef build():\n    return First(0, 1, 'b', 2.0)\n"),
    )

    (owner,) = verdict.owner_candidates
    assert {receipt.kind for receipt in owner.refusals} == {
        OwnerReceiptKind.POSITIONAL_CONSTRUCTION
    }


def test_a_finding_id_names_the_obligation_not_its_score() -> None:
    from dataclasses import replace

    from nominal_refactor_advisor.class_overlap import ClassOverlapMetrics
    from nominal_refactor_advisor.detectors._class_overlap import MissingMixinDetector

    (verdict,) = _mixin_verdicts(
        (
            "pkg.records",
            "class First:\n" + BLOCK + "    own_first: int\n\n\n"
            "class Second:\n" + BLOCK + "    own_second: int\n",
        ),
    )
    metrics = ClassOverlapMetrics.of(verdict)
    finding = MissingMixinDetector()._finding_for_candidate(verdict)
    rescored = replace(
        finding,
        summary="0.12 summary",
        metrics=replace(metrics, confidence_score=0.12, owner_candidates=()),
    )
    assert rescored.stable_id == finding.stable_id
    other_block = replace(
        finding, metrics=replace(metrics, shared_declaration_names=("alpha",))
    )
    assert other_block.stable_id != finding.stable_id


def test_a_finding_a_stage_does_not_touch_keeps_its_id_when_its_lines_move() -> None:
    """A stage that inserts a class above a pair shifts every evidence line; the
    pair's finding is the same obligation and keeps its id."""

    from nominal_refactor_advisor.detectors._class_overlap import MissingMixinDetector

    pair = (
        "class First:\n" + BLOCK + "    own_first: int\n\n\n"
        "class Second:\n" + BLOCK + "    own_second: int\n"
    )
    (before,) = _mixin_verdicts(("pkg.records", pair))
    (after,) = _mixin_verdicts(
        ("pkg.records", "class Inserted:\n    delta: int\n\n\n" + pair)
    )
    assert before.owner.line != after.owner.line
    detector = MissingMixinDetector()
    assert (
        detector._finding_for_candidate(after).stable_id
        == detector._finding_for_candidate(before).stable_id
    )


def test_a_star_imported_module_s_field_order_is_settled_the_same_way_on_every_path() -> None:
    """A dataclass in a module with `from x import *` reads `str` and `int`; the
    star origin's `__all__` excludes them. The detector context and the
    collected projections settle the field order through the same resolution."""

    from nominal_refactor_advisor.ast_tools import collect_family_items
    from nominal_refactor_advisor.detectors._base import CompactClassRepositoryContext
    from nominal_refactor_advisor.detectors._base import DetectorConfig

    modules = (
        _module("pkg.base", '__all__ = ["Thing"]\n\n\nclass Thing:\n    pass\n'),
        _module(
            "pkg.records",
            "from dataclasses import dataclass\n"
            "from pkg.base import *\n\n\n"
            "@dataclass(frozen=True)\nclass Record:\n    name: str\n    size: int\n",
        ),
    )
    collected = CompactModuleClassProjectionFamily.collect_modules(modules)
    per_module = tuple(
        item
        for module in modules
        for item in collect_family_items(module, CompactModuleClassProjectionFamily)
    )
    context = CompactClassRepositoryContext.from_projections(per_module, DetectorConfig())
    symbol = "pkg.records.Record"

    assert build_compact_class_family_index(collected).class_for(
        symbol
    ).own_dataclass_field_names == ("name", "size")
    assert context.class_index.class_for(symbol).own_dataclass_field_names == ("name", "size")


def test_a_block_copied_off_one_value_refuses_a_mixin_and_ranks_holding_the_source() -> None:
    (verdict,) = _mixin_verdicts(
        (
            "pkg.records",
            "class First:\n" + BLOCK + "    own_first: int\n\n\n"
            "class Second:\n" + BLOCK + "    own_second: int\n\n\n"
            "def summarize(first: First) -> Second:\n"
            "    return Second(alpha=first.alpha, beta=first.beta)\n",
        ),
    )

    kinds = {owner.kind: owner for owner in verdict.owner_candidates}
    assert OwnerReceiptKind.COPIED_FROM in {
        receipt.kind for receipt in kinds["new_mixin"].refusals
    }
    hold = kinds["hold_source"]
    assert hold.permitted
    assert hold.composers == ("pkg.records.Second",)
    assert hold.symbol == "pkg.records.First"
    assert verdict.owner_candidates[0] is hold


def test_a_classmethod_building_its_own_class_off_one_value_is_a_copy() -> None:
    (verdict,) = _mixin_verdicts(
        (
            "pkg.records",
            "class First:\n" + BLOCK + "    own_first: int\n\n\n"
            "class Second:\n" + BLOCK + "    own_second: int\n\n"
            "    @classmethod\n"
            "    def of(cls, first: First) -> 'Second':\n"
            "        return cls(alpha=first.alpha, beta=first.beta, own_second=0)\n",
        ),
    )

    kinds = {owner.kind: owner for owner in verdict.owner_candidates}
    assert OwnerReceiptKind.COPIED_FROM in {
        receipt.kind for receipt in kinds["new_mixin"].refusals
    }
    assert kinds["hold_source"].composers == ("pkg.records.Second",)


SPLIT_OWNERS = (
    "pkg.owners",
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\nclass SourceLineReference:\n"
    "    file_path: str\n    line: int\n\n\n"
    "@dataclass(frozen=True)\nclass ClassNameMixin:\n    class_name: str\n\n\n"
    "@dataclass(frozen=True)\nclass ClassLine(ClassNameMixin, SourceLineReference):\n"
    "    pass\n",
)
SPLIT_RESTATERS = (
    "pkg.facts",
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\nclass AxisFact:\n"
    "    file_path: str\n    line: int\n    class_name: str\n    signals: tuple[str, ...]\n\n\n"
    "@dataclass(frozen=True)\nclass AuthorityShape:\n"
    "    file_path: str\n    line: int\n    class_name: str\n    method_names: tuple[str, ...]\n",
)


def test_existing_classes_that_cover_the_block_refuse_a_new_mixin() -> None:
    (verdict,) = _mixin_verdicts(SPLIT_OWNERS, SPLIT_RESTATERS)
    owners = {owner.candidate_id: owner for owner in verdict.owner_candidates}

    covering = owners[
        "compose_existing:pkg.owners.ClassNameMixin+pkg.owners.SourceLineReference"
    ]
    assert covering.kind == "compose_existing"
    assert covering.permitted
    assert covering.symbols == (
        "pkg.owners.ClassNameMixin",
        "pkg.owners.SourceLineReference",
    )
    new_mixins = [owner for owner in owners.values() if owner.kind == "new_mixin"]
    assert new_mixins
    assert all(
        OwnerReceiptKind.OWNED_BY_EXISTING_CLASSES
        in {receipt.kind for receipt in owner.refusals}
        for owner in new_mixins
    )
    assert verdict.owner_candidates[0] is covering


def test_a_covering_owner_that_brings_behaviour_is_offered_but_refused() -> None:
    """The covering owners cannot be composed, so their identical declarations
    are evidence against a new mixin, not a refusal: whether they are one
    concept is the pick's nominal judgement (paper1 Thm 4.8)."""
    owners_module = (
        SPLIT_OWNERS[0],
        SPLIT_OWNERS[1].replace(
            "    file_path: str\n    line: int\n\n\n",
            "    file_path: str\n    line: int\n\n"
            "    def __post_init__(self):\n        pass\n\n\n",
            1,
        ),
    )
    (verdict,) = _mixin_verdicts(owners_module, SPLIT_RESTATERS)
    kinds = {owner.kind: owner for owner in verdict.owner_candidates}

    assert OwnerReceiptKind.OWNER_NOT_CONTAINED in {
        receipt.kind for receipt in kinds["compose_existing"].refusals
    }
    assert kinds["new_mixin"].permitted
    assert OwnerReceiptKind.ALSO_DECLARED_BY in {
        receipt.kind for receipt in kinds["new_mixin"].receipts
    }


FIELDS = "    alpha: int\n    beta: str\n    gamma: float\n"


def _dataclass(name: str, options: str, own: str) -> str:
    return f"@dataclass({options})\nclass {name}:\n" + FIELDS + f"    {own}: int = 0\n\n\n"


def _new_mixin(verdict):
    return next(owner for owner in verdict.owner_candidates if owner.kind == "new_mixin")


def test_a_frozen_and_a_mutable_dataclass_never_share_one_mixin() -> None:
    """Python refuses a frozen dataclass with a non-frozen dataclass base and the
    reverse, so no one mixin serves both."""

    for options in ("", "frozen=FLAG"):
        (verdict,) = _mixin_verdicts(
            (
                "pkg.records",
                "from dataclasses import dataclass\n\nFLAG = True\n\n\n"
                + _dataclass("First", "frozen=True", "own_first")
                + _dataclass("Second", options, "own_second"),
            ),
        )
        owner = _new_mixin(verdict)
        assert not owner.permitted
        (refusal,) = (r for r in owner.refusals if r.kind is OwnerReceiptKind.FROZEN_DIFFERS)
        assert "`First`" in refusal.detail and "`Second`" in refusal.detail


def test_a_co_restater_of_the_other_frozenness_keeps_its_own_declaration() -> None:
    (verdict,) = (
        verdict
        for verdict in _mixin_verdicts(
            (
                "pkg.records",
                "from dataclasses import dataclass\n\n\n"
                + _dataclass("First", "frozen=True", "own_first")
                + _dataclass("Second", "frozen=True", "own_second")
                + _dataclass("Third", "", "own_third"),
            ),
        )
        if {verdict.owner.symbol, *verdict.restating_symbols}
        == {"pkg.records.First", "pkg.records.Second"}
    )
    owner = _new_mixin(verdict)
    assert owner.permitted
    assert "pkg.records.Third" not in owner.composers
    (kept,) = (r for r in owner.receipts if r.kind is OwnerReceiptKind.KEEPS_OWN_DECLARATION)
    assert "`Third`" in kept.detail


HOIST = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(kw_only=True)\n"
    "class Action:\n"
    "    problem: str\n"
    "    order: int\n"
    "    request: str = ''\n\n\n"
    "@dataclass(kw_only=True)\n"
    "class Create(Action):\n"
    "    repo: str\n"
    "    text: str\n\n"
    "    def reserve(self) -> str:\n"
    "        return self.repo\n\n\n"
    "@dataclass(kw_only=True)\n"
    "class Task:\n"
    "    problem: str\n"
    "    repo: str\n"
    "    text: str\n"
    "    order: int\n\n"
    "    def checked(self) -> bool:\n"
    "        return True\n\n\n"
    "@dataclass(frozen=True, kw_only=True)\n"
    "class Frozen:\n"
    "    problem: str\n"
    "    order: int\n"
    "    label: str = ''\n"
)


def test_a_hoist_leaves_out_a_composer_of_the_other_frozenness() -> None:
    (verdict,) = (
        verdict
        for verdict in _mixin_verdicts(("pkg.records", HOIST))
        if {verdict.owner.symbol, *verdict.restating_symbols}
        == {"pkg.records.Create", "pkg.records.Task"}
    )
    (hoist,) = (owner for owner in verdict.owner_candidates if owner.kind == "hoisting_mixin")
    assert hoist.hoist_composers == ("pkg.records.Action", "pkg.records.Task")
    (kept,) = (r for r in hoist.receipts if r.kind is OwnerReceiptKind.KEEPS_OWN_DECLARATION)
    assert "`Frozen`" in kept.detail


def test_a_hoist_out_of_a_base_of_the_other_frozenness_refuses() -> None:
    source = HOIST.replace(
        "@dataclass(kw_only=True)\nclass Action:", "@dataclass(frozen=True, kw_only=True)\nclass Action:"
    ).replace(
        "@dataclass(kw_only=True)\nclass Create(Action):",
        "@dataclass(frozen=True, kw_only=True)\nclass Create(Action):",
    )
    (verdict,) = (
        verdict
        for verdict in _mixin_verdicts(("pkg.records", source))
        if {verdict.owner.symbol, *verdict.restating_symbols}
        == {"pkg.records.Create", "pkg.records.Task"}
    )
    (hoist,) = (owner for owner in verdict.owner_candidates if owner.kind == "hoisting_mixin")
    assert not hoist.permitted
    assert any(
        "`Action`" in refusal.detail and "`Task`" in refusal.detail
        for refusal in hoist.refusals
        if refusal.kind is OwnerReceiptKind.FROZEN_DIFFERS
    )
