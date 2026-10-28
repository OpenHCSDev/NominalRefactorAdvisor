"""A picker's choice of owner for each verdict that needs one (#67).

NRA ranks where an extracted block could live; it does not choose. A picker
(a person or an agent) writes a picks file::

    {"picks": [
      {"finding": "<finding stable id>",
       "owner": "<candidate_id from metrics.owner_candidates>",
       "because": "<the receipt kind the pick rests on>",
       "name": "<class name, for a new mixin>"}
    ]}

Each pick must name a candidate the finding offers and no guard refuses, and
must cite a receipt kind that candidate carries, so a wrong pick shows up as
a disagreement with a cited fact. A new mixin needs a name: naming a new
identity is the picker's decision. A pick whose `owner` is null declines the
finding, with `because` saying why (the classes are not one concept, or
another pick already decides them). A pick also covers every other verdict
among the classes that would compose the same owner. An extract-mixin
verdict with no valid pick and no covering pick is reported as missing,
never guessed; a pick naming no current finding is reported as unmatched.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from .class_overlap import ClassOverlapMetrics, ExtractMixin
from .json_reports import SemanticRecord
from .models import RefactorFinding
from .owner_candidates import OwnerPick, OwnerPickOutcome, PickedOwner


@dataclass(frozen=True)
class OwnerPickReport(SemanticRecord):
    """Every extract-mixin finding's outcome, and the picks that matched nothing."""

    picks_file: str
    picks_sha256: str
    outcomes: tuple[OwnerPickOutcome, ...]
    unmatched: tuple[str, ...]


@dataclass(frozen=True)
class OwnerPicks:
    """A picks file, read once."""

    path: str
    digest: str
    picks: tuple[OwnerPick, ...]

    @classmethod
    def read(cls, path: Path) -> OwnerPicks:
        content = path.read_bytes()
        payload = json.loads(content)
        return cls(
            str(path),
            hashlib.sha256(content).hexdigest(),
            tuple(
                OwnerPick(
                    finding=str(item["finding"]),
                    owner=None if item.get("owner") is None else str(item["owner"]),
                    because=str(item.get("because", "")),
                    name=None if item.get("name") is None else str(item["name"]),
                    judgement=(
                        None if item.get("judgement") is None else str(item["judgement"])
                    ),
                )
                for item in payload["picks"]
            ),
        )

    def applied(
        self, findings: Iterable[RefactorFinding]
    ) -> tuple[list[RefactorFinding], OwnerPickReport]:
        """The findings with each valid pick on its metrics, and the report."""

        findings = list(findings)
        by_finding = {pick.finding: pick for pick in self.picks}
        mixins = [
            finding
            for finding in findings
            if isinstance(finding.metrics, ClassOverlapMetrics)
            and finding.metrics.move is ExtractMixin
        ]
        outcomes: dict[str, OwnerPickOutcome] = {}
        chosen: dict[str, PickedOwner] = {}
        for finding in mixins:
            pick = by_finding.get(finding.stable_id)
            if pick is None:
                continue
            resolution = pick.resolved(finding.metrics.owner_candidates)
            outcomes[finding.stable_id] = resolution.outcome(finding.stable_id)
            for picked in resolution.picked():
                chosen[finding.stable_id] = picked
        picks_by_finding: dict[str, PickedOwner] = dict(chosen)
        for finding in mixins:
            if finding.stable_id in outcomes:
                continue
            classes = frozenset(
                (finding.metrics.owner_symbol, *finding.metrics.restating_symbols)
            )
            covering = next(
                (
                    (finding_id, picked)
                    for finding_id, picked in chosen.items()
                    if classes <= frozenset(picked.candidate.composers)
                    and any(
                        owner.candidate_id == picked.candidate.candidate_id
                        for owner in finding.metrics.owner_candidates
                    )
                ),
                None,
            )
            if covering is not None:
                picks_by_finding[finding.stable_id] = covering[1]
                outcomes[finding.stable_id] = OwnerPickOutcome(
                    finding.stable_id, "covered", f"by the pick on {covering[0]}"
                )
                continue
            outcomes[finding.stable_id] = OwnerPickOutcome(
                finding.stable_id,
                "missing",
                "no pick; candidates: "
                + ", ".join(
                    f"{owner.candidate_id} ({owner.confidence:.2f})"
                    for owner in finding.metrics.owner_candidates
                    if owner.permitted
                ),
            )
        present = frozenset(finding.stable_id for finding in mixins)
        return (
            [
                finding
                if finding.stable_id not in picks_by_finding
                else replace(
                    finding,
                    metrics=replace(
                        finding.metrics, owner_pick=picks_by_finding[finding.stable_id]
                    ),
                )
                for finding in findings
            ],
            OwnerPickReport(
                picks_file=self.path,
                picks_sha256=self.digest,
                outcomes=tuple(outcomes[finding.stable_id] for finding in mixins),
                unmatched=tuple(
                    pick.finding for pick in self.picks if pick.finding not in present
                ),
            ),
        )

    @property
    def picked_finding_ids(self) -> frozenset[str]:
        """The findings a pick chooses an owner for: with a picks file, the goal's
        only targets."""

        return frozenset(pick.finding for pick in self.picks if pick.owner is not None)


__all__ = ("OwnerPick", "OwnerPickOutcome", "OwnerPickReport", "OwnerPicks")
