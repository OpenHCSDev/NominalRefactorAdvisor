"""Restated facts: a second copy of a fact that its owner should hold.

Each finding names the restating sites, the fact they restate, the owner that
should hold it, and the factoring that moves it there. A restated fact is slop
because the copy can disagree with its owner; the finding is not a guess, only
the factoring's equivalence with the old behaviour needs proof, and that proof
is the recipe's job. Seven detectors share this base and differ in what they
match and which fact they name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar, cast

from ..codemod import (
    DispatchToPolymorphismOperation,
    LiteralDispatchFindingRecipeSynthesizer,
    RefactorRecipeOperation,
)
from ..models import DispatchCountMetrics, RefactorFinding, SourceLocation
from ..observation_families import FieldObservationFamily
from ..observation_shapes import FieldObservation, FieldOriginKind
from ..patterns import PatternId
from ..restated_fact_observations import (
    ExactKeySetSiteFamily,
    NamedAttributeSiteFamily,
    RestatingSite,
    RestatingSiteFamily,
    RestatingSiteModule,
    StringDispatchSiteFamily,
)
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    CompactModuleProjectionDetectorMixin,
    CompactMultiModuleProjectionDetectorMixin,
    CompactProjectionGroups,
    DetectorConfig,
    IssueDetector,
    high_confidence_certified_spec,
)

ContextT = TypeVar("ContextT")

# A finding lists this many consumers or keys in its text; evidence keeps them all.
LISTED_SITES = 8


@dataclass(frozen=True)
class RestatedFact:
    """The claim every restated-fact finding makes.

    `fact` is what the sites restate, `owner` is what should hold it, `factoring`
    is the move from one to the other, and `operation` is the existing codemod
    operation that performs it, or None when only the prescription exists.
    """

    fact: str
    owner: str
    factoring: str
    operation: type[RefactorRecipeOperation] | None = None

    @property
    def why(self) -> str:
        return f"{self.fact} {self.owner}"

    @property
    def recipe_status(self) -> str:
        if self.operation is None:
            return "No codemod operation performs this factoring yet; this finding is the prescription."
        return f"Recipe: `{self.operation.__name__}`, proved against current source before it is offered."

    def spec(
        self,
        pattern_id: PatternId,
        title: str,
        capability_gap: str,
        relation_context: str,
        capability_tags: tuple[CapabilityTag, ...],
        observation_tags: tuple[ObservationTag, ...],
    ):
        return high_confidence_certified_spec(
            pattern_id,
            title,
            self.why,
            capability_gap,
            relation_context,
            (CapabilityTag.PROVENANCE, *capability_tags),
            observation_tags,
        )


def _listed(items: Iterable[str]) -> str:
    items = tuple(items)
    shown = ", ".join(items[:LISTED_SITES])
    hidden = len(items) - LISTED_SITES
    return f"{shown}, and {hidden} more" if hidden > 0 else shown


class RestatedFactDetector(IssueDetector, Generic[ContextT], ABC):
    """One finding per restater: the sites, the fact, the owner, the factoring."""

    restated: ClassVar[RestatedFact]

    @abstractmethod
    def describe(self, sites: Sequence[RestatingSite], context: ContextT) -> str:
        """What the restater's sites say, in the terms of their own detector."""

        raise NotImplementedError

    def admits(self, sites: Sequence[RestatingSite], context: ContextT) -> bool:
        """Whether this restater is a finding; every restater is unless a join says otherwise."""

        return True

    def owner(self, sites: Sequence[RestatingSite], context: ContextT) -> str:
        """The owner that should hold the fact: a resolved declaration or the shape to create."""

        return self.restated.owner

    def evidence(
        self, sites: Sequence[RestatingSite], context: ContextT
    ) -> tuple[SourceLocation, ...]:
        return tuple(site.location for site in sites)

    def finding_context(
        self, sites: Sequence[RestatingSite], context: ContextT
    ) -> dict[str, object]:
        """Typed extras a recipe reads from the finding (empty unless a recipe needs them)."""

        return {}

    def findings_from_sites(
        self, sites: Iterable[RestatingSite], context: ContextT
    ) -> list[RefactorFinding]:
        groups: dict[tuple[str, str, str], list[RestatingSite]] = defaultdict(list)
        for site in sites:
            groups[(site.file_path, site.symbol, site.group)].append(site)
        findings = []
        for _key, group in sorted(groups.items()):
            ordered = sorted(group, key=lambda site: (site.lineno, site.detail))
            if not self.admits(ordered, context):
                continue
            summary = (
                f"{self.describe(ordered, context)} {self.restated.fact} "
                f"Owner: {self.owner(ordered, context)} "
                f"Factoring: {self.restated.factoring} {self.restated.recipe_status}"
            )
            findings.append(
                self.build_finding(
                    summary,
                    self.evidence(ordered, context),
                    **self.finding_context(ordered, context),
                )
            )
        return findings


class SingleFamilyRestatedFactDetector(
    CompactModuleProjectionDetectorMixin[RestatingSiteModule],
    RestatedFactDetector[None],
    ABC,
):
    """A restated-fact detector whose sites come from one family alone."""

    module_projection_family: ClassVar[type[RestatingSiteFamily]]

    def _findings_from_compact_projections(
        self,
        projections: Sequence[RestatingSiteModule],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        return self.findings_from_sites(
            (site for module in projections for site in module.sites), None
        )


class JoinedRestatedFactDetector(
    CompactMultiModuleProjectionDetectorMixin, RestatedFactDetector[ContextT], ABC
):
    """A restated-fact detector that joins its sites with a second family.

    The first of `module_projection_families` carries the sites;
    `compact_shared_group_context_builder` builds what the other families add,
    once per scan.
    """

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: ContextT,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        (site_family, *_joined) = type(self).module_projection_families
        return self.findings_from_sites(
            (
                site
                for module in projections_by_family[site_family]
                for site in module.sites
            ),
            context,
        )


class StringKindDispatchDetector(
    SingleFamilyRestatedFactDetector, LiteralDispatchFindingRecipeSynthesizer
):
    """One subject compared with three or more string literals in one function."""

    module_projection_family = StringDispatchSiteFamily
    restated = RestatedFact(
        fact="Each comparison restates which strings belong to the kind, a second copy of the family's membership that can disagree with the family.",
        owner="The owner is the class family of the kind: an abstract base with one subclass per string.",
        factoring="Declare an abstract base per kind and one subclass per string with the behaviour on the subclass, fill the registry at class creation (AutoRegisterMeta), and keep the strings only in the boundary decoder.",
        operation=DispatchToPolymorphismOperation,
    )
    finding_spec = restated.spec(
        PatternId.STRING_KIND_DISPATCH,
        "String dispatch restates a kind's family",
        "a class family per kind whose subclasses own the behaviour the string comparisons select",
        "one subject compared with three or more distinct string literals inside one function",
        (CapabilityTag.CLOSED_FAMILY_DISPATCH,),
        (ObservationTag.STRING_DISPATCH,),
    )

    def describe(self, sites: Sequence[RestatingSite], context: None) -> str:
        arms = sorted({literal for site in sites for literal in site.detail})
        return (
            f"`{sites[0].group}` is compared with {len(arms)} strings "
            f"({_listed(repr(arm) for arm in arms)}) at {len(sites)} comparison(s) in "
            f"{sites[0].symbol}."
        )

    def finding_context(
        self, sites: Sequence[RestatingSite], context: None
    ) -> dict[str, object]:
        arms = tuple(sorted({literal for site in sites for literal in site.detail}))
        return {"metrics": DispatchCountMetrics.from_literal_family(sites[0].group, arms)}


class NamedAttributeDetector(SingleFamilyRestatedFactDetector):
    """getattr, hasattr, setattr or delattr by a literal name, and getattr with a default."""

    module_projection_family = NamedAttributeSiteFamily
    restated = RestatedFact(
        fact="Each call names an attribute the receiver's type should declare, a second copy of that type's attribute list.",
        owner="The owner is the type that declares the attribute, or an adapter family or capability base when the receiver is another module's object.",
        factoring="Declare the attribute on the type (an abstract property on the base or a dataclass field) and read it directly; for a foreign object's optional attribute, decode once into an adapter family (ForeignAdapter) or a capability base.",
    )
    finding_spec = restated.spec(
        PatternId.ATTRIBUTE_ACCESS_BY_NAME,
        "Attribute access by name restates a type's attributes",
        "attributes declared on the receiver's type and read directly",
        "getattr, hasattr, setattr or delattr with a string-literal name, or getattr with a default, on one receiver in one function",
        (),
        (ObservationTag.ATTRIBUTE_PROBE,),
    )

    def describe(self, sites: Sequence[RestatingSite], context: None) -> str:
        spelled = [f"{site.detail[0]}(.., {site.detail[1]!r})" for site in sites]
        return (
            f"`{sites[0].group}` is reached by attribute name {len(sites)} time(s) in "
            f"{sites[0].symbol}: {_listed(spelled)}."
        )


@dataclass(frozen=True)
class DeclaredRecords:
    """Record classes of the scan by their exact field sets."""

    owners_by_fields: dict[frozenset[str], tuple[SourceLocation, ...]]

    @classmethod
    def from_observations(cls, fields: Iterable[FieldObservation]) -> "DeclaredRecords":
        names: dict[tuple[str, str], set[str]] = defaultdict(set)
        first: dict[tuple[str, str], int] = {}
        for observed in fields:
            if observed.origin_kind not in (
                FieldOriginKind.DATACLASS_FIELD,
                FieldOriginKind.CLASS_ANNOTATION,
            ):
                continue
            key = (observed.file_path, observed.class_name)
            names[key].add(observed.field_name)
            first[key] = min(first.get(key, observed.lineno), observed.lineno)
        owners: dict[frozenset[str], list[SourceLocation]] = defaultdict(list)
        for (file_path, class_name), found in sorted(names.items()):
            owners[frozenset(found)].append(
                SourceLocation(file_path, first[(file_path, class_name)], class_name)
            )
        return cls({fields: tuple(found) for fields, found in owners.items()})


def _declared_records(projections_by_family: CompactProjectionGroups) -> DeclaredRecords:
    return DeclaredRecords.from_observations(
        cast(Sequence[FieldObservation], projections_by_family[FieldObservationFamily])
    )


class ExactKeySetDetector(JoinedRestatedFactDetector[DeclaredRecords]):
    """A literal key collection compared with, or subtracted from, a mapping's keys."""

    module_projection_families = (ExactKeySetSiteFamily, FieldObservationFamily)
    restated = RestatedFact(
        fact="The literal restates the field list of a record, a second copy of that record's fields.",
        owner="The owner is a record class whose fields are these keys.",
        factoring="Make a dataclass (or use the record class that already has these fields) whose fields are the key set, and decode the mapping into it once at the boundary.",
    )
    finding_spec = restated.spec(
        PatternId.EXACT_KEY_SET,
        "Exact key set restates a record's fields",
        "a record class whose fields are the key set, decoded once at the boundary",
        "a literal collection of two or more string keys compared with, subtracted from or checked against a mapping's keys",
        (),
        (ObservationTag.SEMANTIC_STRING_LITERAL,),
    )

    compact_shared_group_context_builder = staticmethod(_declared_records)

    def describe(self, sites: Sequence[RestatingSite], context: DeclaredRecords) -> str:
        (site,) = sites
        mapping, *keys = site.detail
        return (
            f"The keys of `{mapping}` are checked against the literal "
            f"{{{_listed(repr(key) for key in keys)}}} in {site.symbol}."
        )

    def declared(
        self, sites: Sequence[RestatingSite], context: DeclaredRecords
    ) -> tuple[SourceLocation, ...]:
        (site,) = sites
        return context.owners_by_fields.get(frozenset(site.detail[1:]), ())

    def owner(self, sites: Sequence[RestatingSite], context: DeclaredRecords) -> str:
        declared = self.declared(sites, context)
        if not declared:
            return self.restated.owner
        names = _listed(f"`{location.symbol}` ({location.file_path}:{location.line})" for location in declared)
        return f"{names} already declares exactly these fields."

    def evidence(
        self, sites: Sequence[RestatingSite], context: DeclaredRecords
    ) -> tuple[SourceLocation, ...]:
        return (*super().evidence(sites, context), *self.declared(sites, context))

    def finding_context(
        self, sites: Sequence[RestatingSite], context: DeclaredRecords
    ) -> dict[str, object]:
        declared = self.declared(sites, context)
        return {} if not declared else {"authority_evidence": declared[0]}


__all__ = (
    "ExactKeySetDetector",
    "NamedAttributeDetector",
    "StringKindDispatchDetector",
)
