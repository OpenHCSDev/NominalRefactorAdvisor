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
        is_package_init=False,
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


def test_a_new_mixin_that_closes_an_import_cycle_is_refused() -> None:
    verdicts = _mixin_verdicts(
        ("pkg.low", "class First:\n" + BLOCK + "    own_first: int\n"),
        (
            "pkg.high",
            "from pkg.low import First\n\n\n"
            "class Second:\n" + BLOCK + "    own_second: int\n",
        ),
    )
    (verdict,) = verdicts
    owners = {owner.candidate_id: owner for owner in verdict.owner_candidates}

    low, high = owners["new_mixin:pkg.low"], owners["new_mixin:pkg.high"]
    assert low.permitted and low.confidence == 1.0
    assert not high.permitted and high.confidence == 0.0
    assert {receipt.kind for receipt in high.refusals} == {OwnerReceiptKind.CYCLE_CREATED}
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
    from nominal_refactor_advisor.detectors import DetectorConfig

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
