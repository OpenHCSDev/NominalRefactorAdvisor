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
from typing import ClassVar, Generic, TypeVar

from ..codemod import (
    DispatchToPolymorphismOperation,
    EraseDeadCompatibilityOperation,
    LiteralDispatchFindingRecipeSynthesizer,
    RefactorRecipeOperation,
)
from ..class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from ..probe_owner import ProbeOwnerContext, ScannedClasses
from ..type_checker_session import TypeCheckerSession
from ..codemod_legacy_path_recipe import LegacyPathErasureFindingRecipeSynthesizer
from ..codemod_optional_ast_field import (
    OptionalAstFieldDecoderFindingRecipeSynthesizer,
    RouteOptionalAstFieldProbesOperation,
)
from ..models import (
    DispatchCountMetrics,
    ForwardingFacadeMetrics,
    RefactorFinding,
    SourceLocation,
)
from ..observation_families import FieldObservationFamily
from ..observation_shapes import FieldObservation
from ..patterns import PatternId
from ..restated_fact_observations import (
    DeadModuleSiteFamily,
    ExactKeySetSiteFamily,
    ForwardingFacadeSiteFamily,
    ForeignAbsenceSite,
    ForeignAbsenceSiteFamily,
    ForeignAbsenceSiteModule,
    LegacyMarkerSiteFamily,
    ModuleReferenceFacts,
    ModuleReferenceFamily,
    NamedAttributeSiteFamily,
    ReferenceIndex,
    RestatingSite,
    RestatingSiteFamily,
    RestatingSiteModule,
    StringDispatchSiteFamily,
    UnreachedModules,
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
SiteT = TypeVar("SiteT", bound=RestatingSite)

# A finding lists this many consumers or keys in its text; evidence keeps them all.
LISTED_SITES = 8
# Consumers of a legacy path are kept as evidence up to this many.
EVIDENCED_CONSUMERS = 25


@dataclass(frozen=True)
class RestatedFact:
    """The claim every restated-fact finding makes.

    `fact` is what the sites restate, `owner` is what should hold it, `factoring`
    is the move from one to the other, and `operations` are the codemod
    operations that perform it, empty when only the prescription exists.
    """

    fact: str
    owner: str
    factoring: str
    operations: tuple[type[RefactorRecipeOperation], ...] = ()

    @property
    def why(self) -> str:
        return f"{self.fact} {self.owner}"

    @property
    def recipe_status(self) -> str:
        if not self.operations:
            return "No codemod operation performs this factoring yet; this finding is the prescription."
        named = " or ".join(f"`{operation.__name__}`" for operation in self.operations)
        return f"Recipe: {named}, proved against current source before it is offered; a site no recipe rewrites keeps this prescription."

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


class RestatedFactDetector(IssueDetector, Generic[ContextT, SiteT], ABC):
    """One finding per restater: the sites, the fact, the owner, the factoring."""

    restated: ClassVar[RestatedFact]

    @abstractmethod
    def describe(self, sites: Sequence[SiteT], context: ContextT) -> str:
        """What the restater's sites say, in the terms of their own detector."""

        raise NotImplementedError

    def admits(self, sites: Sequence[SiteT], context: ContextT) -> bool:
        """Whether this restater is a finding; every restater is unless a join says otherwise."""

        return True

    def owner(self, sites: Sequence[SiteT], context: ContextT) -> str:
        """The owner that should hold the fact: a resolved declaration or the shape to create."""

        return self.restated.owner

    def evidence(
        self, sites: Sequence[SiteT], context: ContextT
    ) -> tuple[SourceLocation, ...]:
        return tuple(site.location for site in sites)

    def finding_context(
        self, sites: Sequence[SiteT], context: ContextT
    ) -> dict[str, object]:
        """Typed extras a recipe reads from the finding (empty unless a recipe needs them)."""

        return {}

    def findings_from_sites(
        self, sites: Iterable[SiteT], context: ContextT
    ) -> list[RefactorFinding]:
        groups: dict[tuple[str, str, str], list[SiteT]] = defaultdict(list)
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
    RestatedFactDetector[None, RestatingSite],
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
    CompactMultiModuleProjectionDetectorMixin,
    RestatedFactDetector[ContextT, SiteT],
    ABC,
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
        site_family, *_joined = type(self).module_projection_families
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
        operations=(DispatchToPolymorphismOperation,),
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
        return {
            "metrics": DispatchCountMetrics.from_literal_family(sites[0].group, arms)
        }


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
    def from_observations(cls, fields: Iterable[FieldObservation]) -> DeclaredRecords:
        names: dict[tuple[str, str], set[str]] = defaultdict(set)
        first: dict[tuple[str, str], int] = {}
        for observed in fields:
            if not observed.origin_kind.declares_record_field:
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


def _declared_records(
    projections_by_family: CompactProjectionGroups,
) -> DeclaredRecords:
    return DeclaredRecords.from_observations(
        FieldObservationFamily.modules_in(projections_by_family)
    )


def _reference_facts(
    projections_by_family: CompactProjectionGroups,
) -> Sequence[ModuleReferenceFacts]:
    return ModuleReferenceFamily.modules_in(projections_by_family)


def _reference_index(projections_by_family: CompactProjectionGroups) -> ReferenceIndex:
    return ReferenceIndex.from_facts(_reference_facts(projections_by_family))


def _unreached_modules(
    projections_by_family: CompactProjectionGroups,
) -> UnreachedModules:
    return UnreachedModules.from_facts(_reference_facts(projections_by_family))


class ExactKeySetDetector(JoinedRestatedFactDetector[DeclaredRecords, RestatingSite]):
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
        names = _listed(
            f"`{location.symbol}` ({location.file_path}:{location.line})"
            for location in declared
        )
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


def _probe_owner_context(
    projections_by_family: CompactProjectionGroups,
) -> ProbeOwnerContext:
    """The scan's classes, and the type checker's owner for each unannotated receiver.

    The type checker runs only when some receiver declares no annotation.
    """

    classes = ScannedClasses.of(
        {
            indexed_class.symbol: indexed_class
            for projection in CompactModuleClassProjectionFamily.modules_in(projections_by_family)
            for indexed_class in projection.classes
        }
    )
    spans = tuple(
        span
        for module in ForeignAbsenceSiteFamily.modules_in(projections_by_family)
        for site in module.sites
        for span in site.owner.type_checker_spans()
    )
    if not spans:
        return ProbeOwnerContext(classes, {})
    with TypeCheckerSession.session(span.file_path for span in spans) as session:
        answers = session.receiver_types(spans)
    return ProbeOwnerContext(
        classes, {span: answer.owner(classes) for span, answer in answers.items()}
    )


class ForeignAbsenceProbeDetector(
    JoinedRestatedFactDetector[ProbeOwnerContext, ForeignAbsenceSite],
    OptionalAstFieldDecoderFindingRecipeSynthesizer,
):
    """`other.attr is None` or `is not None` on an object that is not the method's own."""

    module_projection_families = (
        ForeignAbsenceSiteFamily,
        CompactModuleClassProjectionFamily,
    )

    compact_shared_group_context_builder = staticmethod(_probe_owner_context)

    restated = RestatedFact(
        fact="Each probe restates a state of the other object's type, a second copy of that type's state that the type does not declare.",
        owner="The owner is the other object's type: its states as subclasses, or a null member.",
        factoring="Give the type its states as subclasses (for example PausedGoal and ActiveGoal) or a null member, and let the consumer call a method instead of probing the attribute.",
        operations=(RouteOptionalAstFieldProbesOperation,),
    )
    finding_spec = restated.spec(
        PatternId.FOREIGN_ABSENCE_PROBE,
        "Absence probes on another object restate its type's states",
        "the other type's states declared as subclasses or a null member, and consumers calling a method",
        "an attribute of an object that is neither the method's own nor an imported module compared with None",
        (),
        (ObservationTag.ATTRIBUTE_PROBE,),
    )

    def describe(
        self, sites: Sequence[ForeignAbsenceSite], context: ProbeOwnerContext
    ) -> str:
        probes = sorted({f"{site.detail[0]} {site.detail[1]}" for site in sites})
        located = "; ".join(
            f"{site.file_path}:{site.lineno} `{site.group}.{site.detail[0]} "
            f"{site.detail[1]}`, {site.owner.describe(site.detail[0], context)}"
            for site in sites
        )
        return (
            f"`{sites[0].group}` is probed for absence {len(sites)} time(s) in "
            f"{sites[0].symbol}: {_listed(probes)}. Probes: {located}."
        )


class LegacyPathMarkerDetector(
    JoinedRestatedFactDetector[ReferenceIndex, RestatingSite],
    LegacyPathErasureFindingRecipeSynthesizer,
):
    """A declaration, parameter, import alias or warning that says it is the old path."""

    module_projection_families = (LegacyMarkerSiteFamily, ModuleReferenceFamily)
    restated = RestatedFact(
        fact="The path declares itself legacy, compatibility or deprecated, so it restates behaviour its replacement already owns.",
        owner="The owner is the replacement; the old path has nothing left to own.",
        factoring="Move the consumers listed onto the replacement, then delete the old path.",
        operations=(EraseDeadCompatibilityOperation,),
    )
    finding_spec = restated.spec(
        PatternId.LEGACY_PATH_MARKER,
        "A legacy path is kept beside its replacement",
        "the old path deleted once its consumers use the replacement",
        "a function, class, parameter, assignment, import alias or warning that names itself legacy, compatibility, deprecated, backward or shim",
        (),
        (ObservationTag.PARTIAL_VIEW,),
    )

    compact_shared_group_context_builder = staticmethod(_reference_index)

    def consumers(
        self, sites: Sequence[RestatingSite], context: ReferenceIndex
    ) -> tuple[tuple[str, int], ...]:
        site = sites[0]
        return context.consumers_of(site.group, (site.file_path, site.lineno))

    def describe(self, sites: Sequence[RestatingSite], context: ReferenceIndex) -> str:
        site = sites[0]
        kind, why = site.detail
        consumers = self.consumers(sites, context)
        where = _listed(f"{path}:{line}" for path, line in consumers)
        still = (
            f"{len(consumers)} module(s) still use the identifier `{site.group}`: {where}."
            if consumers
            else f"no scanned module uses the identifier `{site.group}`."
        )
        return f"`{site.symbol}` ({kind}, {why}) is the old path; {still}"

    def evidence(
        self, sites: Sequence[RestatingSite], context: ReferenceIndex
    ) -> tuple[SourceLocation, ...]:
        return (
            *super().evidence(sites, context),
            *(
                SourceLocation(path, line, f"uses {sites[0].group}")
                for path, line in self.consumers(sites, context)[:EVIDENCED_CONSUMERS]
            ),
        )


class ForwardingFacadeDetector(SingleFamilyRestatedFactDetector):
    """A class holding an object in an attribute, with members that only forward to it.

    It stands beside `RepeatedPropertyAliasHookDetector` and does not subsume it:
    that detector matches `return self.x` hooks repeated across sibling
    subclasses of one base (one fact, owned by the base), while this one matches
    `return self.held.attr` and `return self.held.method(same args)` inside one
    class (a different fact, owned by the held class).
    """

    module_projection_family = ForwardingFacadeSiteFamily
    restated = RestatedFact(
        fact="The forwarding members restate the held class's attribute and method set, so they silently go stale when the held class changes.",
        owner="The owner is the held class: consumers read it directly, or the facade is-a held class and inherits it.",
        factoring="Either consumers read the held object directly and the forwards are deleted, or, when the facade is-a the held thing, the held class becomes a parent and the forwards are deleted; never keep forwards for API compatibility.",
    )
    finding_spec = restated.spec(
        PatternId.FORWARDING_FACADE,
        "A facade forwards members to the object it holds",
        "consumers reading the held class directly, or the held class as a parent, with the forwards deleted",
        "a class holding an object in an attribute with two or more members that only forward to it: properties returning the held attribute's attribute, and methods returning the held attribute's method called with the member's own parameters",
        (CapabilityTag.NOMINAL_IDENTITY,),
        (ObservationTag.PROJECTION_HELPER,),
    )

    @staticmethod
    def forwards(sites: Sequence[RestatingSite]) -> list[RestatingSite]:
        return [
            site
            for site in sites
            if site.detail[0] != ForwardingFacadeSiteFamily.CLASS_SITE
        ]

    def describe(self, sites: Sequence[RestatingSite], context: None) -> str:
        facade = sites[0]
        held_class = facade.detail[1]
        resolved = (
            f"of class `{held_class}`"
            if held_class
            else "of a class neither annotated nor constructed in the facade"
        )
        forwarded = [
            f"{site.detail[0]} -> {site.detail[2]}" for site in self.forwards(sites)
        ]
        return (
            f"`{facade.symbol}` holds `self.{facade.group}` ({resolved}) and forwards "
            f"{len(forwarded)} member(s) to it: {_listed(forwarded)}."
        )

    def evidence(
        self, sites: Sequence[RestatingSite], context: None
    ) -> tuple[SourceLocation, ...]:
        facade, *members = sites
        return (
            facade.location,
            *(
                SourceLocation(
                    site.file_path, site.lineno, f"{site.symbol}.{site.detail[0]}"
                )
                for site in members
            ),
        )

    def finding_context(
        self, sites: Sequence[RestatingSite], context: None
    ) -> dict[str, object]:
        return {"metrics": ForwardingFacadeMetrics(len(self.forwards(sites)))}


class DeadModuleDetector(JoinedRestatedFactDetector[UnreachedModules, RestatingSite]):
    """A scanned module that nothing imports and that is not an entry point."""

    module_projection_families = (DeadModuleSiteFamily, ModuleReferenceFamily)
    restated = RestatedFact(
        fact="No scanned module imports or names this module, so whatever it holds is a second copy of what its readers use instead, or nothing.",
        owner="The owner of its behaviour is the module the scan already imports; the dead module owns nothing.",
        factoring="Delete the module.",
    )
    finding_spec = restated.spec(
        PatternId.DEAD_MODULE,
        "A module nothing imports",
        "the module deleted, or imported by the entry point that needs it",
        "a scanned module no other scanned module imports or names, that is not __init__ or __main__, is declared by no pyproject entry point, is listed by no package-enumerating loader, and has no main guard",
        (),
        (ObservationTag.PARTIAL_VIEW,),
    )

    compact_shared_group_context_builder = staticmethod(_unreached_modules)

    def admits(self, sites: Sequence[RestatingSite], context: UnreachedModules) -> bool:
        return sites[0].group in context.module_names

    def describe(
        self, sites: Sequence[RestatingSite], context: UnreachedModules
    ) -> str:
        return (
            f"`{sites[0].group}` ({sites[0].file_path}) is imported by no scanned "
            "module, named by no string, listed by no module loader, declared by no "
            "entry point, and has no main guard."
        )


__all__ = (
    "DeadModuleDetector",
    "ExactKeySetDetector",
    "ForeignAbsenceProbeDetector",
    "ForwardingFacadeDetector",
    "LegacyPathMarkerDetector",
    "NamedAttributeDetector",
    "StringKindDispatchDetector",
)
