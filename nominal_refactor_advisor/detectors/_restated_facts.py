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

from nominal_refactor_advisor.refactors.target_deletion import (
    EraseDeadCompatibilityOperation,
)
from nominal_refactor_advisor.codemod import RefactorRecipeOperation
from nominal_refactor_advisor.class_index import (
    CompactClassFamilyIndex,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from nominal_refactor_advisor.ast_tools import ParsedModuleSet
from nominal_refactor_advisor.declared_receivers import DeclaredReceiverTyping
from nominal_refactor_advisor.probe_owner import (
    ProbeOwnerContext,
    ReceiverSpan,
    ScannedClasses,
)
from nominal_refactor_advisor.refactors.dispatch_polymorphism import (
    DispatchToPolymorphismOperation,
    LiteralDispatchFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.type_checker_session import (
    ReceiverTypingDetector,
    SourceFiles,
    SourceTexts,
    TypeCheckerSession,
    TypeCheckerSources,
)
from nominal_refactor_advisor.refactors.function_operations import InlineForwarderOperation
from nominal_refactor_advisor.refactors.module_move import DissolveModuleOperation
from nominal_refactor_advisor.refactors.restated_fact_recipes import (
    DeadModuleDissolutionFindingRecipeSynthesizer,
    ForwarderInlineFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.codemod_legacy_path_recipe import (
    LegacyPathErasureFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.refactors.optional_ast_field import (
    OptionalAstFieldDecoderFindingRecipeSynthesizer,
    RouteOptionalAstFieldProbesOperation,
)
from nominal_refactor_advisor.cache_checkout import (
    checkout_display_path,
    checkout_identity_text,
)
from nominal_refactor_advisor.models import (
    DispatchCountMetrics,
    FindingMetrics,
    ForwardingFacadeMetrics,
    RefactorFinding,
    RestaterMetrics,
    SourceLineReference,
    SourceLocation,
)
from nominal_refactor_advisor.observation_families import (
    FieldObservationFamily,
    TypeTestModuleFamily,
)
from nominal_refactor_advisor.observation_shapes import (
    DeclaredClassScope,
    FieldObservation,
    TypeTestShape,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.restated_fact_observations import (
    DeadModuleSiteFamily,
    ExactKeySetSite,
    ExactKeySetSiteFamily,
    FacadeForwardSite,
    FacadeSite,
    ForwardingFacadeSiteFamily,
    ForeignAbsenceSite,
    ForeignAbsenceSiteFamily,
    LegacyMarkerSite,
    LegacyMarkerSiteFamily,
    ModuleReferenceFacts,
    ModuleReferenceFamily,
    NamedAttributeSite,
    NamedAttributeSiteFamily,
    ReferenceIndex,
    RestatingSiteFamily,
    RestatingSiteModule,
    StringDispatchSite,
    StringDispatchSiteFamily,
    UnreachedModules,
)
from nominal_refactor_advisor.observation_shapes import RestatingSite
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag
from nominal_refactor_advisor.detectors._base import (
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

    def restates(self, site: SiteT, context: ContextT) -> bool:
        """Whether one site restates the fact; every site does unless a join says otherwise."""

        return True

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

    def group_of(self, site: SiteT, context: ContextT) -> tuple[str, str, str]:
        """What one finding covers: a file or a family, a symbol and a group."""

        return (site.file_path, site.symbol, site.group)

    def site_order(self, site: SiteT) -> tuple[object, ...]:
        """How the sites of one finding are listed."""

        return (site.lineno, site.detail)

    def restater(self, sites: Sequence[SiteT], context: ContextT) -> tuple[str, ...]:
        """What makes these sites the same finding in the next scan: what they
        cover, a file spelled as it sits in its checkout, and what they say.
        No line and no directory is in it."""

        where, symbol, group = self.group_of(sites[0], context)
        return (
            checkout_identity_text(where),
            symbol,
            group,
            *sorted({" ".join(site.detail) for site in sites}),
        )

    def metrics(self, sites: Sequence[SiteT], context: ContextT) -> FindingMetrics:
        """The finding's typed facts. By default only which restater it is,
        which is what makes it the same finding in the next scan."""

        return RestaterMetrics(restater=self.restater(sites, context))

    def finding_context(
        self, sites: Sequence[SiteT], context: ContextT
    ) -> dict[str, object]:
        """Evidence roles a recipe reads from the finding (empty unless a recipe needs them)."""

        return {}

    def findings_from_sites(
        self, sites: Iterable[SiteT], context: ContextT
    ) -> list[RefactorFinding]:
        groups: dict[tuple[str, str, str], list[SiteT]] = defaultdict(list)
        for site in sites:
            if self.restates(site, context):
                groups[self.group_of(site, context)].append(site)
        findings = []
        for _key, group in sorted(groups.items()):
            ordered = sorted(group, key=self.site_order)
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
                    metrics=self.metrics(ordered, context),
                    **self.finding_context(ordered, context),
                )
            )
        return findings


class SingleFamilyRestatedFactDetector(
    CompactModuleProjectionDetectorMixin[RestatingSiteModule[SiteT]],
    RestatedFactDetector[None, SiteT],
    Generic[SiteT],
    ABC,
):
    """A restated-fact detector whose sites come from one family alone."""

    module_projection_family: ClassVar[type[RestatingSiteFamily]]

    def _findings_from_compact_projections(
        self,
        projections: Sequence[RestatingSiteModule[SiteT]],
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
    `compact_projection_group_context` builds what the other families add,
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


class OwnTypeTestDetector(
    JoinedRestatedFactDetector[DeclaredClassScope, TypeTestShape],
):
    """A type test (`isinstance`, `issubclass`, `type(x) is`, `x is C`, a class
    pattern) on a class the scan declares."""

    module_projection_families = (
        TypeTestModuleFamily,
        CompactModuleClassProjectionFamily,
    )

    @staticmethod
    def compact_projection_group_context(
        projections_by_family: CompactProjectionGroups,
    ) -> DeclaredClassScope:
        """The scan's module names and the symbol of every class in one of its own
        families: a class one of whose resolved bases the scan declares, and each
        such base."""

        index = CompactClassFamilyIndex.from_projection_groups(projections_by_family)
        edges = tuple(
            (symbol, base)
            for symbol, indexed_class in index.classes_by_symbol.items()
            for base in indexed_class.resolved_base_symbols
            if base in index.known_symbols
        )
        return DeclaredClassScope(
            module_names=tuple(
                module.module_name
                for module in TypeTestModuleFamily.modules_in(projections_by_family)
            ),
            family_class_symbols=frozenset(symbol for edge in edges for symbol in edge),
            decoder_class_symbols=frozenset(
                symbol
                for symbol, indexed_class in index.classes_by_symbol.items()
                if indexed_class.decodes_instances
            ),
               )

    restated = RestatedFact(
        fact="The test re-derives at the consumer which class the value is, a fact the class already is; every own class added later must find this site.",
        owner="The owner is the tested class family: a method on the base that answers the default, overridden in each tested class.",
        factoring="Move each arm's body onto its tested class as the override of one base method, keep the else as the base default, and replace the test with the call; foreign values are tested once, at the decoder that turns them into the family.",
    )
    finding_spec = restated.spec(
        PatternId.OWN_TYPE_TEST,
        "Type test on a class the scan declares",
        "a method on the tested class family whose overrides own each arm",
        "a type test whose tested classes resolve, at least one of them a class declared in the scanned modules",
        (CapabilityTag.CLOSED_FAMILY_DISPATCH,),
        (ObservationTag.BRANCH_DISPATCH,),
    )

    def restates(self, site: TypeTestShape, context: DeclaredClassScope) -> bool:
        return bool(site.own_type_names(context))

    def describe(self, sites: Sequence[TypeTestShape], context: DeclaredClassScope) -> str:
        classes = sorted({name for site in sites for name in site.own_type_names(context)})
        return (
            f"`{sites[0].symbol}` tests one value against the scan's own "
            f"{_listed(f'`{name}`' for name in classes)} at "
            f"{_listed(f'line {site.lineno}' for site in sites)}."
        )


class StringKindDispatchDetector(
    SingleFamilyRestatedFactDetector[StringDispatchSite], LiteralDispatchFindingRecipeSynthesizer
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

    def describe(self, sites: Sequence[StringDispatchSite], context: None) -> str:
        arms = sorted({literal for site in sites for literal in site.literals})
        return (
            f"`{sites[0].group}` is compared with {len(arms)} strings "
            f"({_listed(repr(arm) for arm in arms)}) at {len(sites)} comparison(s) in "
            f"{sites[0].symbol}."
        )

    def metrics(self, sites: Sequence[StringDispatchSite], context: None) -> FindingMetrics:
        arms = tuple(sorted({literal for site in sites for literal in site.literals}))
        return DispatchCountMetrics.from_literal_family(sites[0].group, arms)


class NamedAttributeDetector(SingleFamilyRestatedFactDetector[NamedAttributeSite]):
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

    def describe(self, sites: Sequence[NamedAttributeSite], context: None) -> str:
        spelled = [f"{site.access}(.., {site.attribute!r})" for site in sites]
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
                SourceLocation(file_path=file_path, line=first[(file_path, class_name)], symbol=class_name)
            )
        return cls({fields: tuple(found) for fields, found in owners.items()})


def _reference_facts(
    projections_by_family: CompactProjectionGroups,
) -> Sequence[ModuleReferenceFacts]:
    return ModuleReferenceFamily.modules_in(projections_by_family)


def _reference_index(projections_by_family: CompactProjectionGroups) -> ReferenceIndex:
    return ReferenceIndex.from_facts(_reference_facts(projections_by_family))


class ExactKeySetDetector(JoinedRestatedFactDetector[DeclaredRecords, ExactKeySetSite]):
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

    @staticmethod
    def compact_projection_group_context(
        projections_by_family: CompactProjectionGroups,
    ) -> DeclaredRecords:
        return DeclaredRecords.from_observations(
            FieldObservationFamily.modules_in(projections_by_family)
        )

    def describe(self, sites: Sequence[ExactKeySetSite], context: DeclaredRecords) -> str:
        # One finding covers one mapping checked against one key set; a symbol
        # that writes the same check twice has two sites that say the same.
        site = sites[0]
        return (
            f"The keys of `{site.mapping}` are checked against the literal "
            f"{{{_listed(repr(key) for key in site.keys)}}} in {site.symbol}."
        )

    def declared(
        self, sites: Sequence[ExactKeySetSite], context: DeclaredRecords
    ) -> tuple[SourceLocation, ...]:
        return context.owners_by_fields.get(frozenset(sites[0].keys), ())

    def owner(self, sites: Sequence[ExactKeySetSite], context: DeclaredRecords) -> str:
        declared = self.declared(sites, context)
        if not declared:
            return self.restated.owner
        names = _listed(f"`{location.symbol}` ({location.where})" for location in declared)
        return f"{names} already declares exactly these fields."

    def evidence(
        self, sites: Sequence[ExactKeySetSite], context: DeclaredRecords
    ) -> tuple[SourceLocation, ...]:
        return (*super().evidence(sites, context), *self.declared(sites, context))

    def finding_context(
        self, sites: Sequence[ExactKeySetSite], context: DeclaredRecords
    ) -> dict[str, object]:
        declared = self.declared(sites, context)
        return {} if not declared else {"authority_evidence": declared[0]}


@dataclass(frozen=True)
class ProbeOwnerQuery:
    """The scan's classes and declared receiver typing, and the receivers it leaves undeclared.

    An owner resolves only to a scanned class, so its declaring file is a
    source of the type checker's build too.
    """

    classes: ScannedClasses
    typing: DeclaredReceiverTyping
    spans: tuple[ReceiverSpan, ...]
    modules: tuple[CompactModuleClassProjection, ...]

    @classmethod
    def of(cls, projections_by_family: CompactProjectionGroups) -> ProbeOwnerQuery:
        projections = CompactModuleClassProjectionFamily.modules_in(projections_by_family)
        index = CompactClassFamilyIndex.from_projection_groups(projections_by_family)
        typing = DeclaredReceiverTyping.from_projections(projections, index)
        return cls(
            ScannedClasses(index.classes_by_symbol),
            typing,
            tuple(
                span
                for module in ForeignAbsenceSiteFamily.modules_in(projections_by_family)
                for site in module.sites
                for span in site.owner.resolved(typing).type_checker_spans()
            ),
            tuple(projections),
        )

    @property
    def file_paths(self) -> frozenset[str]:
        """The files holding a receiver to type, and every scanned module they
        import, transitively: everything a type of those receivers can come from.
        A scanned class no such file reaches can be no receiver's type."""

        by_name = {module.module_name: module for module in self.modules}

        def imported(module: CompactModuleClassProjection) -> set[str]:
            names: set[str] = set()
            for _local_name, target in module.import_aliases:
                parts = target.split(".")
                for end in range(len(parts), 0, -1):
                    candidate = ".".join(parts[:end])
                    if candidate in by_name:
                        names.add(candidate)
                        break
            names.update(
                name
                for origin in module.star_import_origins
                for name in origin.imported_modules()
                if name in by_name
            )
            return names

        by_path = {module.file_path: module for module in self.modules}
        pending = [by_path[span.file_path] for span in self.spans if span.file_path in by_path]
        reached: dict[str, CompactModuleClassProjection] = {}
        while pending:
            module = pending.pop()
            if module.module_name in reached:
                continue
            reached[module.module_name] = module
            pending.extend(by_name[name] for name in imported(module) - reached.keys())
        return frozenset(span.file_path for span in self.spans) | frozenset(
            module.file_path for module in reached.values()
        )

    def answered(self, sources: TypeCheckerSources) -> ProbeOwnerContext:
        """The type checker's owner for each undeclared receiver; it runs only when there is one."""

        if not self.spans:
            return ProbeOwnerContext(classes=self.classes, typed={}, typing=self.typing)
        answers = TypeCheckerSession.started(sources).receiver_types(self.spans)
        return ProbeOwnerContext(
            classes=self.classes,
            typed={span: answer.owner(self.classes) for span, answer in answers.items()},
            typing=self.typing,
               )


class ForeignAbsenceProbeDetector(
    JoinedRestatedFactDetector[ProbeOwnerContext, ForeignAbsenceSite],
    OptionalAstFieldDecoderFindingRecipeSynthesizer,
    ReceiverTypingDetector,
):
    """`other.attr is None` or `is not None` on an object that is not the method's own."""

    module_projection_families = (
        ForeignAbsenceSiteFamily,
        CompactModuleClassProjectionFamily,
    )

    @staticmethod
    def compact_projection_group_context(
        projections_by_family: CompactProjectionGroups,
    ) -> ProbeOwnerContext:
        """The compact roots path holds projections, not parsed text: it types the files it read."""

        query = ProbeOwnerQuery.of(projections_by_family)
        return query.answered(SourceFiles(query.file_paths))

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

    def _collect_findings(
        self, modules: ParsedModuleSet, config: DetectorConfig
    ) -> list[RefactorFinding]:
        """Parsed modules type as their own text, so a state held only in memory
        types as itself, not as the files on disk."""

        projections_by_family = type(self).compact_module_projection_groups(modules)
        query = ProbeOwnerQuery.of(projections_by_family)
        return self._findings_from_compact_projection_groups_context(
            projections_by_family,
            query.answered(
                SourceTexts(
                    tuple(module for module in modules if module.file_path in query.file_paths)
                )
            ),
            config,
        )

    def describe(
        self, sites: Sequence[ForeignAbsenceSite], context: ProbeOwnerContext
    ) -> str:
        probes = sorted({f"{site.attribute} {site.test}" for site in sites})
        located = "; ".join(
            f"{site.location.where} `{site.group}.{site.attribute} "
            f"{site.test}`, {site.owner.describe(site.attribute, context)}"
            for site in sites
        )
        return (
            f"`{sites[0].group}` is probed for absence {len(sites)} time(s) in "
            f"{sites[0].symbol}: {_listed(probes)}. Probes: {located}."
        )


class LegacyPathMarkerDetector(
    JoinedRestatedFactDetector[ReferenceIndex, LegacyMarkerSite],
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

    compact_projection_group_context = staticmethod(_reference_index)

    def consumers(
        self, sites: Sequence[LegacyMarkerSite], context: ReferenceIndex
    ) -> tuple[tuple[str, int], ...]:
        site = sites[0]
        return context.consumers_of(site.group, (site.file_path, site.lineno))

    def describe(self, sites: Sequence[LegacyMarkerSite], context: ReferenceIndex) -> str:
        site = sites[0]
        kind, why = site.kind, site.why
        consumers = self.consumers(sites, context)
        where = _listed(SourceLineReference(file_path=path, line=line).where for path, line in consumers)
        still = (
            f"{len(consumers)} module(s) still use the identifier `{site.group}`: {where}."
            if consumers
            else f"no scanned module uses the identifier `{site.group}`."
        )
        return f"`{site.symbol}` ({kind}, {why}) is the old path; {still}"

    def evidence(
        self, sites: Sequence[LegacyMarkerSite], context: ReferenceIndex
    ) -> tuple[SourceLocation, ...]:
        return (
            *super().evidence(sites, context),
            *(
                SourceLocation(file_path=path, line=line, symbol=f"uses {sites[0].group}")
                for path, line in self.consumers(sites, context)[:EVIDENCED_CONSUMERS]
            ),
        )


def _class_family_index(
    projections_by_family: CompactProjectionGroups,
) -> CompactClassFamilyIndex:
    return CompactClassFamilyIndex.from_projection_groups(projections_by_family)


class ForwardingFacadeDetector(
    JoinedRestatedFactDetector[CompactClassFamilyIndex, FacadeSite],
    ForwarderInlineFindingRecipeSynthesizer,
):
    """A class holding an object in an attribute, with members that only forward to it.

    It stands beside `RepeatedPropertyAliasHookDetector` and does not subsume it:
    that detector matches `return self.x` hooks repeated across sibling
    subclasses of one base (one fact, owned by the base), while this one matches
    `return self.held.attr` and `return self.held.method(same args)` inside one
    class (a different fact, owned by the held class).
    """

    module_projection_families = (
        ForwardingFacadeSiteFamily,
        CompactModuleClassProjectionFamily,
    )
    compact_projection_group_context = staticmethod(_class_family_index)
    restated = RestatedFact(
        fact="The forwarding members restate the held class's attribute and method set, so they silently go stale when the held class changes.",
        owner="The owner is the held class: consumers read it directly, or the facade is-a held class and inherits it.",
        factoring="Either consumers read the held object directly and the forwards are deleted, or, when the facade is-a the held thing, the held class becomes a parent and the forwards are deleted; never keep forwards for API compatibility.",
        operations=(InlineForwarderOperation,),
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
    def forwards(sites: Sequence[FacadeSite]) -> list[FacadeForwardSite]:
        return [forward for site in sites for forward in site.forwarded()]

    def restates(self, site: FacadeSite, context: CompactClassFamilyIndex) -> bool:
        """A member that overrides what an ancestor declares is the family's dispatch,
        not a forward: a receiver typed as the ancestor reaches it through the MRO."""

        return not site.overrides_inherited(
            lambda member: self._overrides_inherited(
                context, site.file_path, site.symbol, member
            )
        )

    def admits(self, sites: Sequence[FacadeSite], context: CompactClassFamilyIndex) -> bool:
        return len(self.forwards(sites)) >= ForwardingFacadeSiteFamily.MIN_FORWARDS

    def describe(self, sites: Sequence[FacadeSite], context: CompactClassFamilyIndex) -> str:
        facade = sites[0]
        held_class = facade.held_class_text()
        resolved = (
            f"of class `{held_class}`"
            if held_class
            else "of a class neither annotated nor constructed in the facade"
        )
        forwarded = [
            f"{site.member} -> {site.target}" for site in self.forwards(sites)
        ]
        return (
            f"`{facade.symbol}` holds `self.{facade.group}` ({resolved}) and forwards "
            f"{len(forwarded)} member(s) to it: {_listed(forwarded)}."
        )

    def evidence(
        self, sites: Sequence[FacadeSite], context: CompactClassFamilyIndex
    ) -> tuple[SourceLocation, ...]:
        facade = sites[0]
        return (
            facade.location,
            *(
                SourceLocation(file_path=site.file_path, line=site.lineno, symbol=f"{site.symbol}.{site.member}")
                for site in self.forwards(sites)
            ),
        )

    def metrics(
        self, sites: Sequence[FacadeSite], context: CompactClassFamilyIndex
    ) -> FindingMetrics:
        return ForwardingFacadeMetrics(
            len(self.forwards(sites)), restater=self.restater(sites, context)
        )

    @staticmethod
    def _overrides_inherited(
        index: CompactClassFamilyIndex, file_path: str, qualname: str, member: str
    ) -> bool:
        """Whether the member overrides a declaration the class inherits; a class the
        index does not know is not proven free of one."""

        symbol = index.symbol_for(file_path=file_path, qualname=qualname)
        return symbol is None or bool(index.overridden_declarations(symbol, member))


class DeadModuleDetector(
    JoinedRestatedFactDetector[UnreachedModules, RestatingSite],
    DeadModuleDissolutionFindingRecipeSynthesizer,
):
    """A scanned module that nothing imports and that is not an entry point."""

    module_projection_families = (DeadModuleSiteFamily, ModuleReferenceFamily)
    restated = RestatedFact(
        fact="No scanned module imports or names this module, so whatever it holds is a second copy of what its readers use instead, or nothing.",
        owner="The owner of its behaviour is the module the scan already imports; the dead module owns nothing.",
        factoring="Delete the module.",
        operations=(DissolveModuleOperation,),
    )
    finding_spec = restated.spec(
        PatternId.DEAD_MODULE,
        "A module nothing imports",
        "the module deleted, or imported by the entry point that needs it",
        "a scanned module no other scanned module imports or names, that is not __init__ or __main__, is declared by no pyproject entry point, is listed by no package-enumerating loader, and has no main guard",
        (),
        (ObservationTag.PARTIAL_VIEW,),
    )

    @staticmethod
    def compact_projection_group_context(
        projections_by_family: CompactProjectionGroups,
    ) -> UnreachedModules:
        return UnreachedModules.from_facts(_reference_facts(projections_by_family))

    def admits(self, sites: Sequence[RestatingSite], context: UnreachedModules) -> bool:
        return sites[0].group in context.module_names

    def describe(
        self, sites: Sequence[RestatingSite], context: UnreachedModules
    ) -> str:
        return (
            f"`{sites[0].group}` ({checkout_display_path(sites[0].file_path)}) is imported by no scanned "
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
