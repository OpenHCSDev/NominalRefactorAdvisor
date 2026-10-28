"""Relocated probes: the shapes an absence or kind branch takes after its probe is removed.

A count of `x is None` probes drops when the probe moves behind a family
member's constant answer, a frozen record's derived bool, an exception caught
around one accessor, a fold over call-site lambdas, a 0/1 tuple or an
`X | None` parameter; the branch stays. Each detector here names one of those
spellings with the fact it restates, the owner, and the factoring. No codemod
performs these factorings yet, so every finding carries the prescription.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from operator import attrgetter

from ..class_index import (
    CompactClassFamilyIndex,
    CompactModuleClassProjectionFamily,
)

from ..models import RefactorFinding, SourceLocation
from ..patterns import PatternId
from ..relocation_observations import (
    BUILTIN_EXCEPTIONS,
    BranchRead,
    ClassShape,
    ClassShapeFamily,
    ConstantAnswerSiteFamily,
    ConsumerShapeFamily,
    ExceptionAbsenceSiteFamily,
    LambdaCall,
    LambdaFoldSiteFamily,
    OwnStateBoolSiteFamily,
    ThreadedAbsenceSiteFamily,
    ZeroOneTupleSiteFamily,
)
from ..restated_fact_observations import RestatingSite
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import CompactProjectionGroups
from ._restated_facts import (
    EVIDENCED_CONSUMERS,
    JoinedRestatedFactDetector,
    RestatedFact,
    SingleFamilyRestatedFactDetector,
    _listed,
)


@dataclass
class MemberFamilies:
    """Which declarations of one member answer the same question.

    The class graph is the compact class index's: each class's bases as
    `resolved_base_symbols`, never a spelled name. A declaration's family is the
    set of its topmost declaring ancestors. A class whose own or an ancestor's
    base references are incomplete is open: it stays alone, never joined by a
    guess.
    """

    class_index: CompactClassFamilyIndex
    shapes: dict[str, ClassShape]

    def is_open(self, symbol: str) -> bool:
        classes = self.class_index.classes_by_symbol
        return symbol not in classes or not all(
            classes[lineage].base_references_are_complete
            for lineage in (symbol, *self.class_index.ancestor_symbols(symbol))
        )

    def declares(self, symbol: str, member: str) -> bool:
        return symbol in self.shapes and member in self.shapes[symbol].declared

    def family(self, symbol: str, member: str) -> str:
        """The topmost declarations of `member` above `symbol`, as one key."""

        if self.is_open(symbol):
            return f"open {symbol}"
        declaring = [
            lineage
            for lineage in (symbol, *self.class_index.ancestor_symbols(symbol))
            if self.declares(lineage, member)
        ]
        return ", ".join(
            sorted(
                lineage
                for lineage in declaring
                if not any(
                    self.declares(ancestor, member)
                    for ancestor in self.class_index.ancestor_symbols(lineage)
                )
            )
        )

    @cached_property
    def computed_by_family(self) -> dict[tuple[str, str], list[str]]:
        computed: dict[tuple[str, str], list[str]] = defaultdict(list)
        for symbol, shape in self.shapes.items():
            for member in shape.computed:
                computed[(self.family(symbol, member), member)].append(symbol)
        return dict(computed)

    @cached_property
    def exception_symbols(self) -> frozenset[str]:
        """Scanned classes whose own or inherited bases bind a builtin exception."""

        builtin = frozenset(f"builtins.{name}" for name in BUILTIN_EXCEPTIONS)
        classes = self.class_index.classes_by_symbol
        return frozenset(
            symbol
            for symbol in classes
            if any(
                reference.qualified_name in builtin
                for lineage in (symbol, *self.class_index.ancestor_symbols(symbol))
                for reference in classes[lineage].base_references
            )
        )


@dataclass
class RelocationIndex:
    """The scan's member families and the consumers that branch on members."""

    families: MemberFamilies
    branch_reads: dict[str, list[BranchRead]]
    lambda_calls: dict[str, list[LambdaCall]]

    def reads_of(self, member: str, called: bool) -> list[BranchRead]:
        return [read for read in self.branch_reads.get(member, ()) if read.called is called]


def _relocation_index(projections_by_family: CompactProjectionGroups) -> RelocationIndex:
    reads: dict[str, list[BranchRead]] = defaultdict(list)
    calls: dict[str, list[LambdaCall]] = defaultdict(list)
    for module in ConsumerShapeFamily.modules_in(projections_by_family):
        for read in module.branch_reads:
            reads[read.member].append(read)
        for call in module.lambda_calls:
            calls[call.member].append(call)
    return RelocationIndex(
        MemberFamilies(
            CompactClassFamilyIndex.from_projection_groups(projections_by_family),
            {
                shape.symbol: shape
                for module in ClassShapeFamily.modules_in(projections_by_family)
                for shape in module.classes
            },
        ),
        dict(reads),
        dict(calls),
    )


def _consumer_locations(
    consumers: Sequence[BranchRead | LambdaCall],
) -> tuple[SourceLocation, ...]:
    return tuple(
        SourceLocation(consumer.file_path, consumer.lineno, consumer.symbol)
        for consumer in consumers[:EVIDENCED_CONSUMERS]
    )


def _consumer_text(consumers: Sequence[BranchRead | LambdaCall], what: str) -> str:
    if not consumers:
        return f"no scanned {what}."
    located = _listed(f"{consumer.file_path}:{consumer.lineno}" for consumer in consumers)
    return f"{len(consumers)} {what}: {located}."


class RelocationDetector(
    JoinedRestatedFactDetector[RelocationIndex, RestatingSite], ABC
):
    """A relocation joined with the scan's class families and branch consumers."""

    compact_shared_group_context_builder = staticmethod(_relocation_index)

    @abstractmethod
    def group_of(self, site: RestatingSite, context: RelocationIndex) -> tuple[str, str, str]:
        """The (file, symbol, group) one finding covers."""
        raise NotImplementedError

    def findings_from_sites(
        self, sites: Iterable[RestatingSite], context: RelocationIndex
    ) -> list[RefactorFinding]:
        groups: dict[tuple[str, str, str], list[RestatingSite]] = defaultdict(list)
        for site in sites:
            groups[self.group_of(site, context)].append(site)
        findings = []
        for _key, group in sorted(groups.items()):
            ordered = sorted(group, key=attrgetter("file_path", "lineno"))
            if not self.admits(ordered, context):
                continue
            summary = (
                f"{self.describe(ordered, context)} {self.restated.fact} "
                f"Owner: {self.owner(ordered, context)} "
                f"Factoring: {self.restated.factoring} {self.restated.recipe_status}"
            )
            findings.append(
                self.build_finding(summary, self.evidence(ordered, context))
            )
        return findings


class FamilyMemberRelocationDetector(RelocationDetector, ABC):
    """One finding per (member family, member): the sites are its declarations.

    `detail[0]` of each site is the declaring class's symbol.
    """

    def group_of(self, site: RestatingSite, context: RelocationIndex) -> tuple[str, str, str]:
        return (context.families.family(site.detail[0], site.group), site.group, "")

    def owner(self, sites: Sequence[RestatingSite], context: RelocationIndex) -> str:
        family = context.families.family(sites[0].detail[0], sites[0].group)
        return f"{self.restated.owner} Declared first by: {family}."


class ConstantSubtypeAnswerDetector(FamilyMemberRelocationDetector):
    """Every member of a family answers one zero-argument question with a constant."""

    module_projection_families = (
        ConstantAnswerSiteFamily,
        CompactModuleClassProjectionFamily,
        ClassShapeFamily,
        ConsumerShapeFamily,
    )
    restated = RestatedFact(
        fact="Each constant answer restates which member of the family the object is, a second copy of the subtype identity the class already is; a consumer that branches on it re-derives the subtype.",
        owner="The owner is the family: the work each consumer selects with the answer belongs on the members as an override.",
        factoring="Replace each consumer's `if x.answer:` with a call of a method named for that consumer's work, declared abstract on the family base and overridden per member with the branch body; then delete the answer.",
    )
    finding_spec = restated.spec(
        PatternId.CONSTANT_SUBTYPE_ANSWER,
        "A family's members answer which member they are",
        "a method per consumer's work, overridden on each member, instead of a constant answer the consumer branches on",
        "a zero-argument method or property returning one bool or str constant, a bool class attribute, or a bool or str ClassVar, declared by two or more members of one class family and computed by none",
        (CapabilityTag.CLOSED_FAMILY_DISPATCH,),
        (ObservationTag.CLASS_FAMILY, ObservationTag.BRANCH_DISPATCH),
    )

    def computed_elsewhere(
        self, sites: Sequence[RestatingSite], context: RelocationIndex
    ) -> list[str]:
        member = sites[0].group
        family = context.families.family(sites[0].detail[0], member)
        return context.families.computed_by_family.get((family, member), [])

    def admits(self, sites: Sequence[RestatingSite], context: RelocationIndex) -> bool:
        return len(sites) >= 2 and not self.computed_elsewhere(sites, context)

    def consumers(
        self, sites: Sequence[RestatingSite], context: RelocationIndex
    ) -> list[BranchRead]:
        return context.branch_reads.get(sites[0].group, [])

    def describe(self, sites: Sequence[RestatingSite], context: RelocationIndex) -> str:
        answers = _listed(
            f"{site.detail[0].rsplit('.', 1)[-1]}={site.detail[1]}" for site in sites
        )
        return (
            f"`{sites[0].group}` is answered by a constant in {len(sites)} members "
            f"of one family ({answers}); "
            f"{_consumer_text(self.consumers(sites, context), 'branch test(s) read the name')}"
        )

    def evidence(
        self, sites: Sequence[RestatingSite], context: RelocationIndex
    ) -> tuple[SourceLocation, ...]:
        return (
            *super().evidence(sites, context),
            *_consumer_locations(self.consumers(sites, context)),
        )


class OwnStateBranchAnswerDetector(RelocationDetector):
    """A frozen record's `-> bool` derived from its own state, and branches on it."""

    module_projection_families = (
        OwnStateBoolSiteFamily,
        CompactModuleClassProjectionFamily,
        ClassShapeFamily,
        ConsumerShapeFamily,
    )
    restated = RestatedFact(
        fact="The record is frozen, so the bool is fixed when the record is built; each branch on it restates a state of the record (an optional field present or absent, a kind) that its type does not declare.",
        owner="The owner is the record's type: its states as subtypes decided once at construction (present/absent subtypes for an optional field, one subtype per kind).",
        factoring="Decide the state where the record is built and build a subtype; move each caller's branch body onto the subtypes as a method named for the caller's work; delete the bool.",
    )
    finding_spec = restated.spec(
        PatternId.OWN_STATE_BRANCH_ANSWER,
        "A frozen record's derived bool selects its callers' branches",
        "the record's states declared as subtypes, with each caller's branch body as a method on them",
        "a `-> bool` property or zero-argument method of a frozen dataclass or NamedTuple whose single return reads the record's own state, read by name in a branch test",
        (CapabilityTag.CLOSED_FAMILY_DISPATCH,),
        (ObservationTag.BRANCH_DISPATCH, ObservationTag.ATTRIBUTE_PROBE),
    )

    def group_of(self, site: RestatingSite, context: RelocationIndex) -> tuple[str, str, str]:
        return (site.file_path, site.symbol, site.group)

    def consumers(
        self, sites: Sequence[RestatingSite], context: RelocationIndex
    ) -> list[BranchRead]:
        return context.reads_of(sites[0].group, sites[0].detail[1] == "method")

    def admits(self, sites: Sequence[RestatingSite], context: RelocationIndex) -> bool:
        return bool(self.consumers(sites, context))

    def describe(self, sites: Sequence[RestatingSite], context: RelocationIndex) -> str:
        site = sites[0]
        return (
            f"`{site.symbol}` ({site.detail[1]}) {site.detail[2]}; "
            f"{_consumer_text(self.consumers(sites, context), 'branch test(s) select on it')}"
        )

    def evidence(
        self, sites: Sequence[RestatingSite], context: RelocationIndex
    ) -> tuple[SourceLocation, ...]:
        return (
            *super().evidence(sites, context),
            *_consumer_locations(self.consumers(sites, context)),
        )


class ExceptionAsAbsenceDetector(RelocationDetector):
    """`try:` around one statement catching an exception the scan declares, not re-raised."""

    module_projection_families = (
        ExceptionAbsenceSiteFamily,
        CompactModuleClassProjectionFamily,
        ClassShapeFamily,
        ConsumerShapeFamily,
    )
    restated = RestatedFact(
        fact="The handler restates the guarded value's absent state as control flow: the exception is the scan's own, raised by its accessor to say the value is not there, and the except arm is the absent branch.",
        owner="The owner is the family of the value the accessor reads: a present member and an absent member.",
        factoring="Give the value present and absent members; the try arm becomes the present member's method and the except arm the absent member's, the caller calls that method, and the exception and the try are deleted.",
    )
    finding_spec = restated.spec(
        PatternId.EXCEPTION_AS_ABSENCE,
        "An exception the scan declares stands for absence",
        "present and absent members of the value's family, with the try and except arms as their methods",
        "a try statement guarding one statement that calls, with a handler for a class the scan declares as an exception whose body is not only a raise",
        (),
        (ObservationTag.BRANCH_DISPATCH,),
    )

    def group_of(self, site: RestatingSite, context: RelocationIndex) -> tuple[str, str, str]:
        return (site.file_path, site.symbol, site.group)

    def admits(self, sites: Sequence[RestatingSite], context: RelocationIndex) -> bool:
        return sites[0].group in context.families.exception_symbols

    def describe(self, sites: Sequence[RestatingSite], context: RelocationIndex) -> str:
        guarded = _listed(f"line {site.lineno}: `{site.detail[0]}`" for site in sites)
        return (
            f"{sites[0].symbol} catches `{sites[0].group}`, an exception the scan "
            f"declares, {len(sites)} time(s) around one statement: {guarded}."
        )


class LambdaFoldDetector(FamilyMemberRelocationDetector):
    """Each override calls exactly one callable parameter and callers pass lambdas."""

    module_projection_families = (
        LambdaFoldSiteFamily,
        CompactModuleClassProjectionFamily,
        ClassShapeFamily,
        ConsumerShapeFamily,
    )
    restated = RestatedFact(
        fact="The method is an optional encoded as a fold: each member calls exactly one of the callables it is given, so every caller restates the family's members as lambdas at its call site.",
        owner="The owner is the family: one named method per caller's question, overridden on each member with that member's lambda body.",
        factoring="Replace each `fold(lambda ..., lambda ...)` call with a method named for the caller's question whose overrides hold the corresponding lambda bodies; delete the fold.",
    )
    finding_spec = restated.spec(
        PatternId.LAMBDA_FOLD,
        "A fold over call-site lambdas selects a family member",
        "a named method per caller's question, overridden on each member",
        "a method declared by two or more members of one family whose overrides each return a call of a different one of its parameters, called with two or more lambdas",
        (CapabilityTag.CLOSED_FAMILY_DISPATCH,),
        (ObservationTag.CLASS_FAMILY, ObservationTag.BRANCH_DISPATCH),
    )

    def calls(
        self, sites: Sequence[RestatingSite], context: RelocationIndex
    ) -> list[LambdaCall]:
        return context.lambda_calls.get(sites[0].group, [])

    def admits(self, sites: Sequence[RestatingSite], context: RelocationIndex) -> bool:
        return len({site.detail[1] for site in sites}) >= 2 and bool(
            self.calls(sites, context)
        )

    def describe(self, sites: Sequence[RestatingSite], context: RelocationIndex) -> str:
        overrides = _listed(
            f"{site.detail[0].rsplit('.', 1)[-1]} calls `{site.detail[1]}`" for site in sites
        )
        return (
            f"`{sites[0].group}` folds over its callables ({overrides}); "
            f"{_consumer_text(self.calls(sites, context), 'call(s) pass lambdas')}"
        )

    def evidence(
        self, sites: Sequence[RestatingSite], context: RelocationIndex
    ) -> tuple[SourceLocation, ...]:
        return (
            *super().evidence(sites, context),
            *_consumer_locations(self.calls(sites, context)),
        )


class ZeroOneTupleDetector(RelocationDetector):
    """Returns that are all `()` or `(x,)`, in one function or across a family's overrides."""

    module_projection_families = (
        ZeroOneTupleSiteFamily,
        CompactModuleClassProjectionFamily,
        ClassShapeFamily,
        ConsumerShapeFamily,
    )
    restated = RestatedFact(
        fact="Returning `()` or a one-element tuple spells present and absent as a collection, so each caller that iterates it branches on absence without naming it.",
        owner="The owner is a family with a present member and an absent member (or a null member) that the function returns instead.",
        factoring="Return a member of a present/absent family instead of the tuple, and move each `for x in ...:` consumer body onto the present member as a method the absent member answers with nothing.",
    )
    finding_spec = restated.spec(
        PatternId.ZERO_ONE_TUPLE,
        "A 0/1 tuple stands for presence and absence",
        "a present/absent family returned instead of the tuple",
        "a function whose every return is `()` or a one-element tuple with both arities, or a method whose overrides in one family return only `()` and only one-element tuples",
        (),
        (ObservationTag.BRANCH_DISPATCH,),
    )

    def group_of(self, site: RestatingSite, context: RelocationIndex) -> tuple[str, str, str]:
        owner = site.detail[0]
        if not owner:
            return (site.file_path, site.symbol, site.group)
        return (context.families.family(owner, site.group), site.group, "")

    def admits(self, sites: Sequence[RestatingSite], context: RelocationIndex) -> bool:
        return {arity for site in sites for arity in site.detail[1:]} == {"0", "1"}

    def describe(self, sites: Sequence[RestatingSite], context: RelocationIndex) -> str:
        declared = _listed(
            f"{site.symbol} returns {' or '.join('()' if arity == '0' else '(x,)' for arity in site.detail[1:])}"
            for site in sites
        )
        return f"`{sites[0].group}` encodes absence as a 0/1 tuple: {declared}."


class ThreadedAbsenceDetector(SingleFamilyRestatedFactDetector):
    """An undefaulted `X | None` parameter compared with None in its own body."""

    module_projection_family = ThreadedAbsenceSiteFamily
    restated = RestatedFact(
        fact="The caller already knows whether the value is there, yet passes it as `X | None` and the callee asks again, so absence is restated at every function the value is threaded through.",
        owner="The owner is the value's type: a holding family (a held member and an absent member) that the caller builds once.",
        factoring="Pass a member of a held/absent family instead of `X | None`; the callee calls one method on it whose overrides hold the two branch bodies, and the parameter's `is None` probes are deleted.",
    )
    finding_spec = restated.spec(
        PatternId.THREADED_ABSENCE,
        "Absence is threaded through a parameter and asked again",
        "a held/absent family passed instead of `X | None`, with the callee's branches as its methods",
        "a parameter annotated `X | None` or `Optional[X]` without a default, compared with None in the function's own body",
        (),
        (ObservationTag.ATTRIBUTE_PROBE,),
    )

    def describe(self, sites: Sequence[RestatingSite], context: None) -> str:
        probes = _listed(f"line {site.lineno} `{site.group} {site.detail[0]}`" for site in sites)
        return (
            f"{sites[0].symbol} takes `{sites[0].group}` as `X | None` and probes it "
            f"{len(sites)} time(s): {probes}."
        )


__all__ = (
    "ConstantSubtypeAnswerDetector",
    "ExceptionAsAbsenceDetector",
    "LambdaFoldDetector",
    "OwnStateBranchAnswerDetector",
    "ThreadedAbsenceDetector",
    "ZeroOneTupleDetector",
)
