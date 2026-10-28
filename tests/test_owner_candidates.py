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
    assert owner.probability == 1.0
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
    assert low.permitted and low.probability == 1.0
    assert not high.permitted and high.probability == 0.0
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
    assert math.isclose(sum(owner.probability for owner in ranked), 1.0, abs_tol=1e-3)
