"""Lift a member several sibling subclasses override identically onto their shared ancestor.

When two or more sibling classes override an inherited member with the same
declaration and a shared ancestor already exists to hold it — either an
intermediate class only these siblings inherit beneath the declaring ancestor,
or the declaring ancestor itself when its declaration is an abstract hook or
unvalued annotation and no other class overrides it — lifting the declaration
onto that ancestor deletes every sibling copy in one proof-checked step.
"""

from __future__ import annotations

import ast
import keyword as keyword_module
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

from nominal_refactor_advisor.class_body import ClassBodyDeclaration
from nominal_refactor_advisor.class_index import (
    PER_CLASS_NAMES,
    ClassBodyDeclarationKind,
    CompactIndexedClass,
)
from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.class_references import ClassReferenceRole
from nominal_refactor_advisor.codemod_declaration_source import ClassBodySourceAuthority
from nominal_refactor_advisor.codemod_payload import (
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_source_edits import (
    Include,
    NominalSourceEdit,
    SourceNodeSpan,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.codemod_statement_source import StatementDeletionSource
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.refactor_concepts import SiblingOverrideConcept
from nominal_refactor_advisor.refactors.class_creation import (
    MroWalk,
    declared_names,
)
from nominal_refactor_advisor.refactors.class_member_authority import (
    ClassMemberMoveProofContext,
)
from nominal_refactor_advisor.refactors.class_overlap import (
    ClassReferenceSites,
    KeptDocumentation,
    OverlapClass,
    RestatedDeclarations,
    movable_members,
    obsolete_import_edits,
    require_static_c3,
    required_name_imports,
    source_rewrite_target,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
)
from nominal_refactor_advisor.refactors.restated_overrides import (
    MemberDeletionCreationProof,
    NamespaceReaders,
    RestatedOverrideDeletion,
    next_binder,
)


@dataclass(frozen=True)
class SiblingOverrideCandidate:
    """One inherited member and the classes that override it with one declaration."""

    declaring: CompactIndexedClass
    """The nearest class every one of them inherits from that declares the member."""
    name: str
    answering: tuple[tuple[CompactIndexedClass, int], ...]
    """Each class giving the shared answer, with the line of its copy."""
    answering_otherwise: int
    """How many other classes override the same declaration with another answer."""
    unordered: tuple[str, ...]
    """The answering classes whose lookup order is not proved (a base has no source the
    scan reads): a base outside the scan may declare the member ahead of `declaring`."""
    home: CompactIndexedClass | None
    """The class that already exists to hold the answer: the nearest class beneath
    `declaring` that every answering class inherits and no class answering otherwise
    does, or `declaring` itself when it has no answer and nothing answers otherwise."""

    @property
    def unanswered(self) -> bool:
        """Whether the overridden declaration is an abstract hook: it has no answer of
        its own, so the shared one is the only answer these classes could inherit."""

        return self.name in self.declaring.abstract_method_names

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(indexed.file_path, line, f"{indexed.qualname}.{self.name}")
            for indexed, line in self.answering
        )


@dataclass(frozen=True)
class Override:
    """One class's own declaration of a member some scanned ancestor declares."""

    symbol: str
    declaration: ClassBodyDeclaration
    ordered: bool
    """Whether the class's C3 order is proved."""


def _restates_next_binder(
    verdicts: ClassOverlapVerdicts,
    order: tuple[str, ...],
    name: str,
    mine: ClassBodyDeclaration,
) -> bool:
    """Whether lookup without the member reaches the same declaration: that is
    `RestatedOverrideDetector`'s finding, one class's and not siblings'."""

    inherited = next_binder(verdicts, order[1:], name)
    return inherited is not None and mine.same_declaration_as(
        verdicts.bodies.declarations_of(inherited)[name]
    )


def _shared_answers(index, overrides: list[Override]) -> list[list[Override]]:
    """The overrides grouped by declaration, keeping each one two or more classes give
    that do not inherit it from one another."""

    shared: list[list[Override]] = []
    for override in overrides:
        for answer in shared:
            if override.declaration.same_declaration_as(answer[0].declaration):
                answer.append(override)
                break
        else:
            shared.append([override])
    independent = [
        [
            override
            for override in answer
            if not any(
                other.symbol in index.ancestor_symbols(override.symbol)
                for other in answer
            )
        ]
        for answer in shared
    ]
    return [answer for answer in independent if len(answer) > 1]


def _home_for(
    index,
    declaring: str,
    name: str,
    answering: frozenset[str],
    otherwise: frozenset[str],
    unanswered: bool,
    bodies,
) -> str | None:
    """The existing class the shared answer belongs on, if one exists."""

    shared = frozenset.intersection(
        *(frozenset(index.ancestor_symbols(symbol)) for symbol in answering)
    )
    beneath = [
        symbol
        for symbol in shared
        if symbol in bodies.declarations_by_symbol
        and declaring in index.ancestor_symbols(symbol)
        and name not in bodies.declarations_of(symbol)
        and not any(symbol in index.ancestor_symbols(other) for other in otherwise)
    ]
    if beneath:
        return max(
            beneath, key=lambda symbol: (len(index.ancestor_symbols(symbol)), symbol)
        )
    return declaring if unanswered and not otherwise else None


def sibling_override_candidates(
    verdicts: ClassOverlapVerdicts,
) -> tuple[SiblingOverrideCandidate, ...]:
    """Every inherited member overridden identically by two or more independent siblings."""

    index, bodies = verdicts.class_index, verdicts.bodies
    overriding: defaultdict[tuple[str, str], list[Override]] = defaultdict(list)
    for symbol in bodies.declarations_by_symbol:
        ancestors = tuple(
            ancestor
            for ancestor in index.ancestor_symbols(symbol)
            if ancestor in bodies.declarations_by_symbol
        )
        if not ancestors:
            continue
        try:
            order: tuple[str, ...] | None = verdicts.linearized(symbol)
        except NeedsInput:
            order = None
        for name, mine in bodies.declarations_of(symbol).items():
            if (
                name in PER_CLASS_NAMES
                or mine.declaration.reads_its_class
                or mine.kind.is_field
            ):
                continue
            if order is not None and _restates_next_binder(verdicts, order, name, mine):
                continue
            for ancestor in ancestors:
                theirs = bodies.declarations_of(ancestor).get(name)
                if theirs is not None and not theirs.declaration.reads_its_class:
                    overriding[(ancestor, name)].append(
                        Override(symbol, mine, order is not None)
                    )
    nearest: dict[tuple[str, frozenset[str]], SiblingOverrideCandidate] = {}
    for (declaring, name), overrides in overriding.items():
        declaring_class = index.classes_by_symbol[declaring]
        theirs = bodies.declarations_of(declaring)[name]
        unanswered = (
            name in declaring_class.abstract_method_names
            or theirs.kind is ClassBodyDeclarationKind.FIELD
        )
        for answer in _shared_answers(index, overrides):
            answering = frozenset(override.symbol for override in answer)
            otherwise = frozenset(override.symbol for override in overrides) - answering
            home = _home_for(
                index, declaring, name, answering, otherwise, unanswered, bodies
            )
            if (
                home is None
                and answer[0].declaration.kind is not ClassBodyDeclarationKind.METHOD
            ):
                continue
            key = (name, answering)
            held = nearest.get(key)
            # Several ancestors can declare the member: the answer belongs beneath the
            # nearest one, which every other one is an ancestor of.
            if held is None or held.declaring.symbol in index.ancestor_symbols(
                declaring
            ):
                nearest[key] = SiblingOverrideCandidate(
                    home=None if home is None else index.classes_by_symbol[home],
                    declaring=declaring_class,
                    name=name,
                    answering=tuple(
                        sorted(
                            (
                                (
                                    index.classes_by_symbol[override.symbol],
                                    override.declaration.declaration.line,
                                )
                                for override in answer
                            ),
                            key=lambda found: (found[0].file_path, found[1]),
                        )
                    ),
                    answering_otherwise=len(overrides) - len(answer),
                    unordered=tuple(
                        sorted(
                            index.classes_by_symbol[override.symbol].qualname
                            for override in answer
                            if not override.ordered
                        )
                    ),
                )
    return tuple(
        sorted(
            nearest.values(),
            key=lambda candidate: (
                candidate.evidence[0].file_path,
                candidate.evidence[0].line,
                candidate.name,
            ),
        )
    )


def _is_property_statement(statement: ast.stmt) -> bool:
    if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in statement.decorator_list:
        match decorator:
            case ast.Name(id="property" | "cached_property"):
                return True
            case ast.Attribute(attr="property" | "cached_property"):
                return True
    return False


def _is_classvar_annotation(annotation: ast.expr) -> bool:
    match annotation:
        case ast.Name(id="ClassVar") | ast.Attribute(attr="ClassVar"):
            return True
        case ast.Subscript(
            value=ast.Name(id="ClassVar") | ast.Attribute(attr="ClassVar")
        ):
            return True
    return False


class HomePlacement(ABC):
    """How the lifted declaration lands in `home`: inserted anew or replacing a stub."""

    @abstractmethod
    def replacement(
        self,
        snapshot: CodemodSourceSnapshot,
        home: OverlapClass,
        first: RestatedDeclarations,
        documentation: KeptDocumentation,
    ) -> SourceTextSpanReplacement:
        raise NotImplementedError

    @abstractmethod
    def obsolete_home_edits(
        self,
        snapshot: CodemodSourceSnapshot,
        home: OverlapClass,
        lifted_statements: tuple[ast.stmt, ...],
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class InsertIntoHome(HomePlacement):
    """Insert the lifted member into an intermediate ancestor that does not bind it yet."""

    def replacement(
        self,
        snapshot: CodemodSourceSnapshot,
        home: OverlapClass,
        first: RestatedDeclarations,
        documentation: KeptDocumentation,
    ) -> SourceTextSpanReplacement:
        home_source = snapshot.sources_by_file_path[home.file_path]
        body = ClassBodySourceAuthority(home.node, home_source)
        return body.declaration_block_replacement(
            first.member_sources(snapshot, body.indentation, documentation)
        )

    def obsolete_home_edits(
        self,
        snapshot: CodemodSourceSnapshot,
        home: OverlapClass,
        lifted_statements: tuple[ast.stmt, ...],
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
        del snapshot, home, lifted_statements, rationale
        return ()


@dataclass(frozen=True)
class ReplaceHomeStub(HomePlacement):
    """Replace an abstract hook or unvalued contract annotation on the declaring ancestor."""

    stub: ast.stmt

    def replacement(
        self,
        snapshot: CodemodSourceSnapshot,
        home: OverlapClass,
        first: RestatedDeclarations,
        documentation: KeptDocumentation,
    ) -> SourceTextSpanReplacement:
        home_source = snapshot.sources_by_file_path[home.file_path]
        body = ClassBodySourceAuthority(home.node, home_source)
        start, end = body.geometry.node_span_offsets(SourceNodeSpan(self.stub, Include))
        member_sources = first.member_sources(snapshot, body.indentation, documentation)
        rendered = self._rendered_replacement(
            home_source, body.indentation, first, member_sources
        )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start,
            end_offset=end,
            replacement_source=rendered,
        )

    def _rendered_replacement(
        self,
        home_source: str,
        indentation: str,
        first: RestatedDeclarations,
        member_sources: tuple[tuple[str, bool], ...],
    ) -> str:
        # Preserve an unvalued contract annotation (e.g. `kind: ClassVar[type]`) when
        # replacing it with a plain assignment (`kind = Export`), so dataclass/type
        # checkers keep the ClassVar or declared type contract.
        if (
            isinstance(self.stub, ast.AnnAssign)
            and self.stub.value is None
            and isinstance(self.stub.target, ast.Name)
            and len(first.statements) == 1
            and isinstance(first.statements[0], ast.Assign)
            and len(first.statements[0].targets) == 1
            and isinstance(first.statements[0].targets[0], ast.Name)
            and first.statements[0].targets[0].id == self.stub.target.id
        ):
            stub_line = home_source.splitlines(keepends=True)[self.stub.lineno - 1]
            newline = "\r\n" if stub_line.endswith("\r\n") else "\n"
            annotation = ast.unparse(self.stub.annotation)
            value = ast.unparse(first.statements[0].value)
            extras = "".join(text for text, _is_fn in member_sources[1:])
            return f"{indentation}{self.stub.target.id}: {annotation} = {value}{newline}{extras}"
        return "".join(text for text, _is_fn in member_sources)

    def obsolete_home_edits(
        self,
        snapshot: CodemodSourceSnapshot,
        home: OverlapClass,
        lifted_statements: tuple[ast.stmt, ...],
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
        if isinstance(self.stub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = tuple(self.stub.decorator_list)
            if decorators:
                lifted_names = {
                    node.id
                    for statement in lifted_statements
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Name)
                }
                dropped = tuple(
                    decorator
                    for decorator in decorators
                    if not (
                        {
                            node.id
                            for node in ast.walk(decorator)
                            if isinstance(node, ast.Name)
                        }
                        & lifted_names
                    )
                )
                if dropped:
                    return obsolete_import_edits(
                        snapshot, home.file_path, dropped, rationale
                    )
        return ()


@dataclass(frozen=True)
class SiblingOverrideLift:
    """A sibling override proved safe to lift onto its shared ancestor `home`."""

    candidate: SiblingOverrideCandidate
    home: OverlapClass
    placement: HomePlacement
    children: tuple[OverlapClass, ...]
    restated: tuple[RestatedDeclarations, ...]

    @classmethod
    def proved(
        cls,
        snapshot: CodemodSourceSnapshot,
        home_target,
        member_name: str,
    ) -> SiblingOverrideLift:
        home = OverlapClass.require(snapshot, home_target)
        if member_name.startswith("__") and member_name.endswith("__"):
            raise NeedsInput(
                f"`{home.qualname}.{member_name}` is a dunder member, which class creation "
                "and decorators treat by name: NRA needs a proof that lifting it leaves "
                "each class unchanged",
                NeededInput.PROOF,
            )
        verdicts = snapshot.class_overlap_verdicts
        matches = [
            candidate
            for candidate in sibling_override_candidates(verdicts)
            if candidate.home is not None
            and candidate.home.symbol == home.symbol
            and candidate.name == member_name
        ]
        if len(matches) != 1:
            raise NeedsInput(
                f"expected one sibling-override group for `{home.qualname}.{member_name}`, "
                f"found {len(matches)}",
                NeededInput.SOURCE_CHANGE,
            )
        (candidate,) = matches
        if candidate.unordered:
            raise NeedsInput(
                f"lookup order of {', '.join(f'`{name}`' for name in candidate.unordered)} "
                "is not proved (a base has no source the scan reads)",
                NeededInput.PROOF,
            )
        home_ancestry = require_static_c3(snapshot, home.facts)
        children = tuple(
            OverlapClass.for_symbol(snapshot, indexed.symbol)
            for indexed, _line in candidate.answering
        )
        restated = tuple(
            RestatedDeclarations.require(snapshot, children[0], child, (member_name,))
            for child in children
        )
        first = restated[0]
        first.require_no_owner_bound_values(snapshot)
        movable_members(
            ClassMemberMoveProofContext.between(
                snapshot, children[0].target, home.target
            ),
            children[0],
            first.statements,
        )
        placement = cls._require_home_placement(snapshot, candidate, home, member_name)
        cls._require_mro_and_descendants_safe(
            snapshot,
            verdicts,
            candidate,
            home,
            children,
            member_name,
            lifted_is_property=any(
                _is_property_statement(statement) for statement in first.statements
            ),
        )
        names = frozenset((member_name,))
        for child, restatement in zip(children, restated):
            ancestry = require_static_c3(snapshot, child.facts)
            order = verdicts.linearized(child.symbol)
            proof = MemberDeletionCreationProof(
                snapshot,
                child.facts,
                ancestry,
                f"`{child.qualname}.{member_name}`",
                names,
                frozenset(),
                MroWalk(order, order, child.symbol, names),
            )
            for statement in restatement.statements:
                RestatedOverrideDeletion.require_once_per_class(proof, child, statement)
            RestatedOverrideDeletion.require_plain_decorators(proof, child)
            proof.require_unchanged()
            NamespaceReaders(snapshot).require_unread(child.facts, names)
        home_order = verdicts.linearized(home.symbol)
        home_proof = MemberDeletionCreationProof(
            snapshot,
            home.facts,
            home_ancestry,
            f"`{home.qualname}.{member_name}`",
            names,
            frozenset(),
            MroWalk(home_order, home_order, home.symbol, names),
        )
        RestatedOverrideDeletion.require_plain_decorators(home_proof, home)
        home_proof.require_unchanged()
        NamespaceReaders(snapshot).require_unread(home.facts, names)
        return cls(candidate, home, placement, children, restated)

    @classmethod
    def _require_home_placement(
        cls,
        snapshot: CodemodSourceSnapshot,
        candidate: SiblingOverrideCandidate,
        home: OverlapClass,
        member_name: str,
    ) -> HomePlacement:
        if home.symbol != candidate.declaring.symbol:
            if member_name in home.target.bound_names:
                raise NeedsInput(
                    f"`{home.qualname}` already binds `{member_name}`",
                    NeededInput.SOURCE_CHANGE,
                )
            if not candidate.unanswered:
                cls._require_home_not_directly_used(snapshot, candidate, home, member_name)
            return InsertIntoHome()
        statements = [
            statement
            for statement in home.node.body
            if member_name in declared_names(statement)
        ]
        if len(statements) != 1:
            raise NeedsInput(
                f"`{home.qualname}` binds `{member_name}` in {len(statements)} statements, not one",
                NeededInput.PROOF,
            )
        (stub,) = statements
        if member_name in home.facts.abstract_method_names and isinstance(
            stub, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            return ReplaceHomeStub(stub)
        if isinstance(stub, ast.AnnAssign) and stub.value is None:
            if (
                home.facts.dataclass_declaration.generates_construction
                and not _is_classvar_annotation(stub.annotation)
            ):
                raise NeedsInput(
                    f"`{home.qualname}.{member_name}` is a dataclass field, not an abstract contract",
                    NeededInput.PROOF,
                )
            return ReplaceHomeStub(stub)
        raise NeedsInput(
            f"`{home.qualname}.{member_name}` is a concrete declaration; NRA needs a proof "
            "that replacing it leaves every consumer unchanged",
            NeededInput.PROOF,
        )

    @staticmethod
    def _require_home_not_directly_used(
        snapshot: CodemodSourceSnapshot,
        candidate: SiblingOverrideCandidate,
        home: OverlapClass,
        member_name: str,
    ) -> None:
        """When `declaring.name` is concrete, lifting a different answer onto `home`
        is only safe if `home` itself is never instantiated or read directly for the
        old default."""

        if home.facts.abstract_method_names:
            return
        allowed_roles = frozenset((ClassReferenceRole.TYPE_TEST,))
        for site in ClassReferenceSites(snapshot, frozenset((home.symbol,))).sites:
            if site.is_base or site.is_annotation or site.role in allowed_roles:
                continue
            raise NeedsInput(
                f"{site.source.file_path}:{site.node.lineno} uses `{home.qualname}` directly, "
                f"which currently inherits `{candidate.declaring.qualname}.{member_name}`",
                NeededInput.PROOF,
            )

    @classmethod
    def _require_mro_and_descendants_safe(
        cls,
        snapshot: CodemodSourceSnapshot,
        verdicts: ClassOverlapVerdicts,
        candidate: SiblingOverrideCandidate,
        home: OverlapClass,
        children: tuple[OverlapClass, ...],
        member_name: str,
        *,
        lifted_is_property: bool,
    ) -> None:
        answering_symbols = frozenset(child.symbol for child in children)
        for child in children:
            order = verdicts.linearized(child.symbol)
            if home.symbol not in order:
                raise NeedsInput(
                    f"`{home.qualname}` is not in `{child.qualname}`'s MRO",
                    NeededInput.PROOF,
                )
            between = order[1 : order.index(home.symbol)]
            intervening = next_binder(verdicts, between, member_name)
            if intervening is not None:
                raise NeedsInput(
                    f"`{intervening}` binds `{member_name}` between `{child.qualname}` "
                    f"and `{home.qualname}`",
                    NeededInput.PROOF,
                )
        class_index = verdicts.class_index
        bodies = verdicts.bodies
        for desc_symbol in class_index.descendant_symbols(home.symbol):
            if desc_symbol in answering_symbols:
                continue
            desc = OverlapClass.for_symbol(snapshot, desc_symbol)
            require_static_c3(snapshot, desc.facts)
            order = verdicts.linearized(desc_symbol)
            if home.symbol not in order:
                continue
            before_home = order[: order.index(home.symbol)]
            if any(symbol in answering_symbols for symbol in before_home):
                continue
            binder = next_binder(verdicts, before_home, member_name)
            if binder is None:
                if cls._remains_abstract_after_lift(verdicts, order, member_name):
                    continue
                raise NeedsInput(
                    f"`{desc.qualname}` inherits from `{home.qualname}` without overriding "
                    f"`{member_name}`",
                    NeededInput.PROOF,
                )
            binder_decl = bodies.declarations_of(binder).get(member_name)
            if lifted_is_property and binder_decl is not None and binder_decl.kind.is_field:
                raise NeedsInput(
                    f"`{desc.qualname}` binds `{member_name}` as a dataclass field, which "
                    f"a property on `{home.qualname}` would shadow",
                    NeededInput.PROOF,
                )

    @staticmethod
    def _remains_abstract_after_lift(
        verdicts: ClassOverlapVerdicts,
        order: tuple[str, ...],
        lifted_member: str,
    ) -> bool:
        """Whether a descendant still has another unimplemented abstract method in its
        MRO, so it cannot be instantiated directly (all of its concrete subclasses are
        checked independently by the same descendant walk)."""

        class_index = verdicts.class_index
        if not any(
            symbol in ("abc.ABC", "abc.ABCMeta")
            or (
                (indexed := class_index.class_for(symbol)) is not None
                and any(
                    meta in ("abc.ABCMeta", "ABCMeta")
                    for meta in indexed.metaclass_names
                )
            )
            for symbol in order
        ):
            return False
        abstract_names: set[str] = set()
        for symbol in order:
            indexed = class_index.class_for(symbol)
            if indexed is not None:
                abstract_names.update(indexed.abstract_method_names)
        abstract_names.discard(lifted_member)
        for abstract_name in abstract_names:
            first = next_binder(verdicts, order, abstract_name)
            if first is None:
                continue
            owner = class_index.class_for(first)
            if owner is not None and abstract_name in owner.abstract_method_names:
                return True
        return False

    def edits(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        home = self.home
        first = self.restated[0]
        documentation = KeptDocumentation(self.children, first.deleted_names)
        rationale = documentation.rationale(rationale)
        home_source = snapshot.sources_by_file_path[home.file_path]
        body = ClassBodySourceAuthority(home.node, home_source)
        home_replacement = self.placement.replacement(
            snapshot, home, first, documentation
        )
        home_edits: list[NominalSourceEdit] = [
            *body.geometry.physical_edits(
                file_path=home.file_path,
                replacements=(home_replacement,),
                rationale=rationale,
            ),
            *required_name_imports(
                snapshot, first.statements, first.restating, home, rationale
            ),
            *self.placement.obsolete_home_edits(
                snapshot, home, first.statements, rationale
            ),
        ]
        for restatement in self.restated:
            child = restatement.restating
            deletion = StatementDeletionSource(
                source=snapshot.sources_by_file_path[child.file_path],
                node=child.node,
                file_path=child.file_path,
            )
            home_edits.extend(
                deletion.physical_edits(
                    file_path=child.file_path,
                    replacements=deletion.replacements_for_statements(
                        restatement.written_statements
                    ),
                    rationale=rationale,
                )
            )
            if child.file_path != home.file_path:
                home_edits.extend(
                    obsolete_import_edits(
                        snapshot, child.file_path, restatement.statements, rationale
                    )
                )
        return tuple(home_edits)

    def rationale(self) -> str:
        siblings = ", ".join(f"`{child.qualname}`" for child in self.children)
        return (
            f"Lifts `{self.candidate.name}` onto `{self.home.qualname}` and deletes "
            f"the identical sibling copies from {siblings}."
        )


@dataclass(frozen=True, kw_only=True)
class LiftSiblingOverrideOperation(RepositorySourceReprovedOperation):
    """Lift one identically overridden member onto the shared ancestor that owns it
    and delete the sibling copies."""

    member_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        if not self.member_name.isidentifier() or keyword_module.iskeyword(
            self.member_name
        ):
            raise ValueError("member_name must be a Python identifier")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        lift = SiblingOverrideLift.proved(snapshot, self.target, self.member_name)
        return lift.edits(snapshot, self.rationale_text(lift.rationale()))


class SiblingOverrideFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    SiblingOverrideConcept,
):
    """Lift a sibling override onto its existing shared ancestor when one is proved."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None or not finding.evidence:
            return self.rejected_evaluation(
                "lifting a sibling override needs source context and its sibling evidence"
            )
        snapshot = context.execution_snapshot()
        evidence_keys = frozenset(
            (location.file_path, location.symbol) for location in finding.evidence
        )
        candidate = next(
            (
                cand
                for cand in sibling_override_candidates(snapshot.class_overlap_verdicts)
                if frozenset(
                    (indexed.file_path, f"{indexed.qualname}.{cand.name}")
                    for indexed, _line in cand.answering
                )
                == evidence_keys
            ),
            None,
        )
        if candidate is None:
            return self.rejected_evaluation("the sibling override left the source")
        if candidate.home is None:
            return self.stopped_evaluation(
                NeedsInput(
                    f"no existing class beneath `{candidate.declaring.qualname}` holds "
                    f"`{candidate.name}` for only these siblings",
                    NeededInput.SOURCE_CHANGE,
                )
            )
        operation = LiftSiblingOverrideOperation(
            target=source_rewrite_target(candidate.home),
            member_name=candidate.name,
        )
        try:
            operation.source_edits(context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=(
                    f"Every sibling overrides `{candidate.declaring.qualname}.{candidate.name}` "
                    f"identically; `{candidate.home.qualname}` already exists to hold it once."
                ),
            ).with_operation(operation)
        )


__all__ = (
    "HomePlacement",
    "InsertIntoHome",
    "LiftSiblingOverrideOperation",
    "Override",
    "ReplaceHomeStub",
    "SiblingOverrideCandidate",
    "SiblingOverrideFindingRecipeSynthesizer",
    "SiblingOverrideLift",
    "sibling_override_candidates",
)
