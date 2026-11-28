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
import copy
import keyword as keyword_module
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace

from nominal_refactor_advisor.ast_field_decoders import alias_names, optional_values
from nominal_refactor_advisor.ast_tools import ParsedModuleSet, ParsedModuleSetFact
from nominal_refactor_advisor.class_body import ClassBodyDeclaration
from nominal_refactor_advisor.class_index import (
    PER_CLASS_NAMES,
    ClassBodyDeclarationKind,
    CompactClassFamilyIndex,
    CompactIndexedClass,
    CompactModuleClassProjection,
)
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.class_references import ClassReferenceRole
from nominal_refactor_advisor.class_symbols import ClassSymbolResolutionAuthority
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassBodySourceAuthority,
    ClassHeaderSpanSourceAuthority,
)
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_payload import (
    PayloadRecordArrayValueCodec,
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
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    Include,
    NominalSourceEdit,
    SourceInsertion,
    SourceNodeSpan,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.codemod_statement_source import (
    StatementDeletionSource,
    StatementSource,
)
from nominal_refactor_advisor.models import (
    RefactorFinding,
    SourceLocation,
)
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
    PrewarmTypeCheckerMixin,
)
from nominal_refactor_advisor.refactors.restated_overrides import (
    MemberDeletionCreationProof,
    NamespaceReaders,
    RestatedOverrideDeletion,
    method_header_annotations,
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

    def require_no_existing_home(self, member_name, snapshot, declaring, verdicts, cls, base_name, docstring):
        """Refuse declaring a new base when an existing class can already hold the member."""
        if self.home is not None:
            raise NeedsInput(f'`{self.home.qualname}` already exists to hold `{member_name}`; lift onto it directly instead of declaring a new base', NeededInput.SOURCE_CHANGE)
        if self.unordered:
            raise NeedsInput(f"lookup order of {', '.join((f'`{name}`' for name in self.unordered))} is not proved (a base has no source the scan reads)", NeededInput.PROOF)
        require_static_c3(snapshot, declaring.facts)
        children = tuple((OverlapClass.for_symbol(snapshot, indexed.symbol) for indexed, _ in self.answering))
        for child in children:
            if '.' in child.qualname:
                raise NeedsInput(f'`{child.qualname}` is nested inside another class; declaring an intermediate base needs top-level sibling classes', NeededInput.SOURCE_CHANGE)
        restated = tuple((RestatedDeclarations.require(snapshot, children[0], child, (member_name,)) for child in children))
        rebase = cls._select_rebase(snapshot, verdicts, declaring, children)
        if '.' in rebase.parent.qualname:
            raise NeedsInput(f'`{rebase.parent.qualname}` is nested inside another class', NeededInput.SOURCE_CHANGE)
        require_static_c3(snapshot, rebase.parent.facts)
        rebase.require_parent_safe(snapshot)
        placement, parent_spelling = cls._select_placement(snapshot, rebase, children, base_name)
        cls._require_base_name_unbound(snapshot, base_name, placement, children)
        creation_keywords = _creation_keywords_for_intermediate(snapshot, rebase.parent)
        lifted_index, annotation_plan = SiblingOverrideMove.select_lifted_choice(
            snapshot, children, restated, placement.anchor, member_name
        )
        first = restated[lifted_index]
        first.require_no_owner_bound_values(snapshot)
        cls._require_post_rebase_mro_safe(snapshot, verdicts, rebase, declaring, children, member_name, base_name, lifted_is_property=any((_is_property_statement(statement) for statement in first.statements)))
        names = frozenset((member_name,))
        for child, restatement in zip(children, restated):
            ancestry = require_static_c3(snapshot, child.facts)
            order = verdicts.linearized(child.symbol)
            proof = MemberDeletionCreationProof(
                snapshot,
                child.facts,
                ancestry,
                f'`{child.qualname}.{member_name}`',
                names,
                frozenset(),
                MroWalk(order, order, child.symbol, names),
                bases_modified=True,
                changed_direct_bases=rebase.changed_direct_bases(),
            )
            for statement in restatement.statements:
                RestatedOverrideDeletion.require_once_per_class(proof, child, statement)
            RestatedOverrideDeletion.require_plain_decorators(proof, child)
            proof.require_unchanged()
            NamespaceReaders(snapshot).require_unread(child.facts, names)
        return cls(candidate=self, rebase=rebase, placement=placement, parent_spelling=parent_spelling, base_name=base_name, docstring=docstring.strip(), creation_keywords=creation_keywords, children=children, restated=restated, lifted_index=lifted_index, annotation_plan=annotation_plan)

    def existing_home_note(self):
        """Format the summary clause naming an existing home class when one exists."""
        return '' if self.home is None else f'; `{self.home.qualname}` already exists to hold it: every one of them inherits it' + (' and no class answering otherwise does' if self.answering_otherwise else '')

    def evaluate_override_recipe(self, finding, context, owner):
        """Evaluate the sibling-override recipe for this candidate."""
        if self.home is None:
            return owner._evaluate_intermediate_base_recipe(finding, context, self)
        operation = LiftSiblingOverrideOperation(target=source_rewrite_target(self.home), member_name=self.name)
        try:
            recipe = RefactorRecipe(recipe_id=f'{finding.stable_id}-{operation.operation_key()}', reason=f'Every sibling overrides `{self.declaring.qualname}.{self.name}` identically; `{self.home.qualname}` already exists to hold it once.').with_proved_operation(operation, context)
        except ValueError as error:
            return owner.stopped_evaluation(error)
        return owner.executable_evaluation(recipe)


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


def _every_non_answering_descendant_is_abstract(
    verdicts: ClassOverlapVerdicts,
    declaring: str,
    answering: frozenset[str],
    name: str,
) -> bool:
    """Whether no concrete descendant outside `answering` inherits `declaring.name`."""

    index = verdicts.class_index
    for desc_symbol in index.descendant_symbols(declaring):
        if desc_symbol in answering:
            continue
        if any(ancestor in answering for ancestor in index.ancestor_symbols(desc_symbol)):
            continue
        try:
            order = verdicts.linearized(desc_symbol)
        except NeedsInput:
            return False
        if not _remains_abstract_after_lift(verdicts, order, name):
            return False
    return True


def _home_for(
    verdicts: ClassOverlapVerdicts,
    declaring: str,
    name: str,
    answering: frozenset[str],
    otherwise: frozenset[str],
    unanswered: bool,
) -> str | None:
    """The existing class the shared answer belongs on, if one exists."""

    index, bodies = verdicts.class_index, verdicts.bodies
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
    if not otherwise and (
        unanswered
        or _every_non_answering_descendant_is_abstract(
            verdicts, declaring, answering, name
        )
    ):
        return declaring
    return None


@ScanCache.cached
def _base_sibling_override_slots() -> dict[
    bool,
    tuple[
        tuple[CompactModuleClassProjection, ...],
        CompactClassFamilyIndex,
        dict[str, tuple[tuple[str, str, Override], ...]],
        dict[str, tuple[tuple[tuple[str, frozenset[str]], SiblingOverrideCandidate], ...]],
    ],
]:
    return {}


def _symbol_overrides(
    verdicts: ClassOverlapVerdicts,
    symbol: str,
) -> tuple[tuple[str, str, Override], ...]:
    index, bodies = verdicts.class_index, verdicts.bodies
    own_decls = bodies.declarations_of(symbol)
    if not own_decls:
        return ()
    ancestors = tuple(
        ancestor
        for ancestor in index.ancestor_symbols(symbol)
        if ancestor in bodies.declarations_by_symbol
    )
    if not ancestors:
        return ()
    ancestor_decls = tuple(
        (ancestor, bodies.declarations_of(ancestor)) for ancestor in ancestors
    )
    candidates: list[tuple[str, ClassBodyDeclaration, list[str]]] = []
    for name, mine in own_decls.items():
        if (
            name in PER_CLASS_NAMES
            or mine.declaration.reads_its_class
            or mine.kind.is_field
        ):
            continue
        matched_ancestors = [
            ancestor
            for ancestor, decls in ancestor_decls
            if (theirs := decls.get(name)) is not None
            and not theirs.declaration.reads_its_class
        ]
        if matched_ancestors:
            candidates.append((name, mine, matched_ancestors))
    if not candidates:
        return ()
    try:
        order: tuple[str, ...] | None = verdicts.linearized(symbol)
    except NeedsInput:
        order = None
    emitted: list[tuple[str, str, Override]] = []
    ordered = order is not None
    for name, mine, matched_ancestors in candidates:
        if order is not None and _restates_next_binder(verdicts, order, name, mine):
            continue
        override = Override(symbol, mine, ordered)
        for ancestor in matched_ancestors:
            emitted.append((ancestor, name, override))
    return tuple(emitted)


def _candidates_for_declaring(
    verdicts: ClassOverlapVerdicts,
    declaring: str,
    by_name: dict[str, list[Override]],
) -> tuple[tuple[tuple[str, frozenset[str]], SiblingOverrideCandidate], ...]:
    index, bodies = verdicts.class_index, verdicts.bodies
    declaring_class = index.classes_by_symbol[declaring]
    declaring_decls = bodies.declarations_of(declaring)
    emitted: list[tuple[tuple[str, frozenset[str]], SiblingOverrideCandidate]] = []
    for name, overrides in by_name.items():
        theirs = declaring_decls[name]
        unanswered = (
            name in declaring_class.abstract_method_names
            or theirs.kind is ClassBodyDeclarationKind.FIELD
        )
        for answer in _shared_answers(index, overrides):
            answering = frozenset(override.symbol for override in answer)
            otherwise = frozenset(override.symbol for override in overrides) - answering
            home = _home_for(
                verdicts, declaring, name, answering, otherwise, unanswered
            )
            if (
                home is None
                and not answer[0].declaration.kind.is_method
            ):
                continue
            emitted.append(
                (
                    (name, answering),
                    SiblingOverrideCandidate(
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
                    ),
                )
            )
    return tuple(emitted)


def _merge_nearest_sibling_candidates(
    index: CompactClassFamilyIndex,
    candidates_by_declaring: dict[
        str, tuple[tuple[tuple[str, frozenset[str]], SiblingOverrideCandidate], ...]
    ],
) -> tuple[SiblingOverrideCandidate, ...]:
    nearest: dict[tuple[str, frozenset[str]], SiblingOverrideCandidate] = {}
    for declaring, entries in candidates_by_declaring.items():
        declaring_ancestors: tuple[str, ...] | None = None
        for key, candidate in entries:
            held = nearest.get(key)
            if held is None:
                nearest[key] = candidate
            else:
                if declaring_ancestors is None:
                    declaring_ancestors = index.ancestor_symbols(declaring)
                if held.declaring.symbol in declaring_ancestors:
                    nearest[key] = candidate
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


def sibling_override_candidates(
    verdicts: ClassOverlapVerdicts,
) -> tuple[SiblingOverrideCandidate, ...]:
    """Every inherited member overridden identically by two or more independent siblings."""

    index, bodies = verdicts.class_index, verdicts.bodies
    settled_key = not any(
        p.has_unsettled_field_declarations for p in verdicts.projections
    )
    if ScanCache.active() is not None:
        slots = _base_sibling_override_slots()
        base_entry = slots.get(settled_key)
        if base_entry is not None:
            (
                base_projs,
                base_index,
                base_overrides_by_symbol,
                base_candidates_by_declaring,
            ) = base_entry
            base_by_path = {p.file_path: p for p in base_projs}
            cur_projs = tuple(verdicts.projections)
            cur_by_path = {p.file_path: p for p in cur_projs}
            changed = tuple(
                p for p in cur_projs if base_by_path.get(p.file_path) is not p
            )
            removed = tuple(
                p for path, p in base_by_path.items() if path not in cur_by_path
            )
            if len(changed) + len(removed) <= 64:
                old_by_sym = {
                    c.symbol: c
                    for p in (
                        *(
                            base_by_path[p.file_path]
                            for p in changed
                            if p.file_path in base_by_path
                        ),
                        *removed,
                    )
                    for c in p.classes
                }
                new_by_sym = {c.symbol: c for p in changed for c in p.classes}

                def _so_sig(c: CompactIndexedClass) -> tuple[object, ...]:
                    return (
                        c.base_references,
                        c.base_references_are_complete,
                        c.resolved_base_symbols,
                        c.declared_base_names,
                        c.mro_bases_are_static,
                        c.star_resolution,
                        c.abstract_method_names,
                        c.metaclass_names,
                        tuple(
                            (
                                d.kind,
                                d.name,
                                d.digest,
                                d.bindings,
                                d.deferred,
                                d.unbound,
                                d.field_declaration,
                                d.reads_its_class,
                            )
                            for d in c.body_declarations
                        ),
                    )

                semantically_changed: set[str] = set()
                line_shifted_syms: dict[str, CompactIndexedClass] = {}
                for sym in old_by_sym.keys() | new_by_sym.keys():
                    old_c = old_by_sym.get(sym)
                    new_c = new_by_sym.get(sym)
                    if old_c is None or new_c is None or _so_sig(old_c) != _so_sig(new_c):
                        semantically_changed.add(sym)
                    elif old_c.line != new_c.line or any(
                        od.line != nd.line
                        for od, nd in zip(old_c.body_declarations, new_c.body_declarations)
                    ):
                        line_shifted_syms[sym] = new_c
                affected_symbols: set[str] = set(semantically_changed)
                for sym in semantically_changed:
                    affected_symbols.update(base_index.descendant_symbols(sym))
                    affected_symbols.update(index.descendant_symbols(sym))
                affected_declarers: set[str] = set(affected_symbols)
                for sym in affected_symbols:
                    affected_declarers.update(base_index.ancestor_symbols(sym))
                    affected_declarers.update(index.ancestor_symbols(sym))
                for sym in line_shifted_syms:
                    if sym in base_overrides_by_symbol:
                        affected_symbols.add(sym)
                        for anc, _name, _ov in base_overrides_by_symbol[sym]:
                            affected_declarers.add(anc)
                candidates_by_declaring: dict[
                    str,
                    tuple[tuple[tuple[str, frozenset[str]], SiblingOverrideCandidate], ...],
                ] = {}
                for decl, entries in base_candidates_by_declaring.items():
                    if decl not in bodies.declarations_by_symbol or decl in affected_declarers:
                        continue
                    if decl in line_shifted_syms:
                        new_decl = line_shifted_syms[decl]
                        entries = tuple(
                            (key, replace(cand, declaring=new_decl))
                            for key, cand in entries
                        )
                    elif any(
                        cand.home is not None and cand.home.symbol in line_shifted_syms
                        for _key, cand in entries
                    ):
                        entries = tuple(
                            (
                                key,
                                (
                                    replace(cand, home=line_shifted_syms[cand.home.symbol])
                                    if cand.home is not None
                                    and cand.home.symbol in line_shifted_syms
                                    else cand
                                ),
                            )
                            for key, cand in entries
                        )
                    candidates_by_declaring[decl] = entries
                # Gather overrides for affected_declarers only, reusing cached
                # per-symbol overrides for unchanged descendants.
                relevant_descendants: set[str] = set()
                for decl in affected_declarers:
                    if decl in bodies.declarations_by_symbol:
                        relevant_descendants.update(index.descendant_symbols(decl))
                overriding_by_decl: defaultdict[str, defaultdict[str, list[Override]]] = (
                    defaultdict(lambda: defaultdict(list))
                )
                for sym in relevant_descendants:
                    if sym not in bodies.declarations_by_symbol:
                        continue
                    if sym in affected_symbols:
                        sym_overrides = _symbol_overrides(verdicts, sym)
                    else:
                        sym_overrides = base_overrides_by_symbol.get(sym, ())
                    for ancestor, name, ov in sym_overrides:
                        if ancestor in affected_declarers:
                            overriding_by_decl[ancestor][name].append(ov)
                for decl, by_name in overriding_by_decl.items():
                    entries = _candidates_for_declaring(verdicts, decl, by_name)
                    if entries:
                        candidates_by_declaring[decl] = entries
                return _merge_nearest_sibling_candidates(index, candidates_by_declaring)

    overrides_by_symbol: dict[str, tuple[tuple[str, str, Override], ...]] = {}
    overriding_by_decl: defaultdict[str, defaultdict[str, list[Override]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for symbol in bodies.declarations_by_symbol:
        sym_overrides = _symbol_overrides(verdicts, symbol)
        if sym_overrides:
            overrides_by_symbol[symbol] = sym_overrides
            for ancestor, name, ov in sym_overrides:
                overriding_by_decl[ancestor][name].append(ov)
    candidates_by_declaring: dict[
        str, tuple[tuple[tuple[str, frozenset[str]], SiblingOverrideCandidate], ...]
    ] = {}
    for declaring, by_name in overriding_by_decl.items():
        entries = _candidates_for_declaring(verdicts, declaring, by_name)
        if entries:
            candidates_by_declaring[declaring] = entries
    if ScanCache.active() is not None:
        slots = _base_sibling_override_slots()
        if settled_key not in slots:
            slots[settled_key] = (
                tuple(verdicts.projections),
                index,
                overrides_by_symbol,
                candidates_by_declaring,
            )
    return _merge_nearest_sibling_candidates(index, candidates_by_declaring)


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


def _self_return_form(spelling: str, class_name: str) -> str | None:
    """Map a sibling's self-typed return annotation to `Self` or `Self | None`."""

    if spelling in (class_name, "Self", "typing.Self"):
        return "Self"
    if spelling in (
        f"{class_name} | None",
        f"None | {class_name}",
        "Self | None",
        "None | Self",
        "typing.Self | None",
        "None | typing.Self",
    ):
        return "Self | None"
    return None


def _slot_annotation_node(statement: ast.stmt, slot: str) -> ast.expr | None:
    """Locate the annotation AST node on `statement` for a `method_header_annotations` slot."""

    if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    if slot == "return":
        return statement.returns
    args = statement.args
    if slot.startswith("posonly:"):
        index = int(slot.partition(":")[2])
        return args.posonlyargs[index].annotation if index < len(args.posonlyargs) else None
    if slot.startswith("arg:"):
        index = int(slot.partition(":")[2])
        return args.args[index].annotation if index < len(args.args) else None
    if slot == "vararg":
        return None if args.vararg is None else args.vararg.annotation
    if slot.startswith("kwonly:"):
        name = slot.partition(":")[2]
        for arg in args.kwonlyargs:
            if arg.arg == name:
                return arg.annotation
        return None
    if slot == "kwarg":
        return None if args.kwarg is None else args.kwarg.annotation
    return None


def _set_slot_annotation(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
    slot: str,
    replacement: ast.expr | None,
) -> None:
    """Update one annotation slot on a copied function AST node."""

    if slot == "return":
        statement.returns = replacement
        return
    args = statement.args
    if slot.startswith("posonly:"):
        index = int(slot.partition(":")[2])
        args.posonlyargs = list(args.posonlyargs)
        arg = copy.copy(args.posonlyargs[index])
        arg.annotation = replacement
        args.posonlyargs[index] = arg
    elif slot.startswith("arg:"):
        index = int(slot.partition(":")[2])
        args.args = list(args.args)
        arg = copy.copy(args.args[index])
        arg.annotation = replacement
        args.args[index] = arg
    elif slot == "vararg" and args.vararg is not None:
        arg = copy.copy(args.vararg)
        arg.annotation = replacement
        args.vararg = arg
    elif slot.startswith("kwonly:"):
        name = slot.partition(":")[2]
        args.kwonlyargs = list(args.kwonlyargs)
        for index, arg in enumerate(args.kwonlyargs):
            if arg.arg == name:
                copied = copy.copy(arg)
                copied.annotation = replacement
                args.kwonlyargs[index] = copied
                break
    elif slot == "kwarg" and args.kwarg is not None:
        arg = copy.copy(args.kwarg)
        arg.annotation = replacement
        args.kwarg = arg


def _aligned_contravariant_param_annotation(
    snapshot: CodemodSourceSnapshot,
    children: tuple[OverlapClass, ...],
    annotations_by_index: list[dict[str, str]],
    slot: str,
) -> str | None:
    """Return a sibling's parameter annotation that is a supertype of every other sibling's
    annotation on `slot` (either `object`, or a scanned class ancestor of all other siblings'
    parameter types), so widening to it is contravariantly sound for every caller and already
    typechecked on the shared body."""

    annotated = [
        (children[index], slot_map[slot])
        for index, slot_map in enumerate(annotations_by_index)
        if slot in slot_map
    ]
    if any(spelling == "object" for _child, spelling in annotated):
        return "object"
    bindings = snapshot.module_binding_proof
    class_index = snapshot.class_overlap_verdicts.class_index
    resolved: list[tuple[str, str]] = []
    for child, spelling in annotated:
        try:
            expr = ast.parse(spelling, mode="eval").body
        except SyntaxError:
            return None
        if not isinstance(expr, (ast.Name, ast.Attribute)):
            return None
        parsed = snapshot.parsed_module_for_source_path(child.file_path)
        witness = bindings.reference_or_builtin_witness_at(
            parsed, expr, line=child.node.lineno
        )
        if witness is None or witness.qualified_name not in class_index.classes_by_symbol:
            return None
        resolved.append((witness.qualified_name, spelling))
    for candidate_sym, candidate_spelling in resolved:
        if all(
            other_sym == candidate_sym
            or candidate_sym in class_index.ancestor_symbols(other_sym)
            for other_sym, _other_spelling in resolved
        ):
            return candidate_spelling
    return None


@dataclass(frozen=True)
class HeaderAnnotationPlan:
    """Header annotation rewrites needed when siblings annotate a lifted method differently."""

    overrides: tuple[tuple[str, str], ...] = ()
    imports_typing_self: bool = False

    @classmethod
    def for_siblings(
        cls,
        snapshot: CodemodSourceSnapshot,
        children: tuple[OverlapClass, ...],
        annotations_by_index: list[dict[str, str]],
        destination: OverlapClass,
        member_name: str,
    ) -> HeaderAnnotationPlan:
        slots_order: list[str] = []
        by_slot: dict[str, list[str]] = {}
        for slot_map in annotations_by_index:
            for slot, spelling in slot_map.items():
                if slot not in by_slot:
                    slots_order.append(slot)
                    by_slot[slot] = []
                if spelling not in by_slot[slot]:
                    by_slot[slot].append(spelling)

        overrides: list[tuple[str, str]] = []
        imports_typing_self = False
        for slot in slots_order:
            spellings = by_slot[slot]
            if len(spellings) <= 1:
                continue
            if slot == "return":
                forms = {
                    _self_return_form(slot_map["return"], child.node.name)
                    for child, slot_map in zip(children, annotations_by_index)
                    if "return" in slot_map
                }
                if len(forms) == 1 and (aligned := next(iter(forms))) is not None:
                    dest_module = snapshot.parsed_module_for_source_path(destination.file_path)
                    actual = snapshot.module_binding_proof.reference_or_builtin_witness_at(
                        dest_module,
                        ast.Name(id="Self", ctx=ast.Load()),
                        line=destination.node.lineno,
                    )
                    if actual is None:
                        imports_typing_self = True
                    elif actual.qualified_name not in ("typing.Self", "typing_extensions.Self"):
                        raise NeedsInput(
                            f"`Self` is bound to `{actual.qualified_name}` in {destination.file_path}, "
                            "not `typing.Self`",
                            NeededInput.SOURCE_CHANGE,
                        )
                    overrides.append(("return", aligned))
                    continue
            elif slot.startswith(("arg:", "posonly:", "kwonly:")):
                aligned_param = _aligned_contravariant_param_annotation(
                    snapshot, children, annotations_by_index, slot
                )
                if aligned_param is not None:
                    overrides.append((slot, aligned_param))
                    continue
            raise NeedsInput(
                f"siblings annotate `{member_name}` differently on {slot} "
                f"(`{spellings[0]}` vs `{spellings[1]}`): align the signature before lifting",
                NeededInput.SOURCE_CHANGE,
            )
        return cls(tuple(overrides), imports_typing_self)

    def rewritten_statements(
        self,
        statements: tuple[ast.stmt, ...],
        *,
        strip_typing_self: bool = False,
    ) -> tuple[ast.stmt, ...]:
        if not self.overrides or len(statements) != 1:
            return statements
        statement = statements[0]
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return statements
        copied = copy.copy(statement)
        copied.args = copy.copy(statement.args)
        for slot, replacement in self.overrides:
            if strip_typing_self and slot == "return" and replacement in ("Self", "Self | None"):
                _set_slot_annotation(copied, slot, None)
            else:
                _set_slot_annotation(copied, slot, ast.parse(replacement, mode="eval").body)
        return (copied,)

    def require_renderable(self, statements: tuple[ast.stmt, ...]) -> None:
        if not self.overrides:
            return
        if len(statements) != 1 or not isinstance(
            statements[0], (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            raise NeedsInput(
                "header annotation rewrite requires a single lifted method statement",
                NeededInput.SOURCE_CHANGE,
            )
        stmt = statements[0]
        for slot, _replacement in self.overrides:
            ann_node = _slot_annotation_node(stmt, slot)
            if ann_node is None:
                raise NeedsInput(
                    f"lifted sibling omits annotation on `{slot}` needed for signature alignment",
                    NeededInput.SOURCE_CHANGE,
                )
            if ann_node.lineno != (ann_node.end_lineno or ann_node.lineno):
                raise NeedsInput(
                    f"multiline annotation on `{slot}` must be aligned before lifting",
                    NeededInput.SOURCE_CHANGE,
                )

    def member_sources(
        self,
        snapshot: CodemodSourceSnapshot,
        first: RestatedDeclarations,
        indentation: str,
        documentation: KeptDocumentation,
    ) -> tuple[tuple[str, bool], ...]:
        if not self.overrides:
            return first.member_sources(snapshot, indentation, documentation)
        source = snapshot.sources_by_file_path[first.restating.file_path]
        stmt = first.statements[0]
        geometry = SourceTextGeometry(source)
        spans: list[tuple[int, int, str]] = []
        for slot, replacement in self.overrides:
            ann_node = _slot_annotation_node(stmt, slot)
            assert ann_node is not None
            start, end = geometry.required_node_offsets(ann_node)
            spans.append((start, end, replacement))
        patched = source
        for start, end, replacement in sorted(spans, key=lambda item: item[0], reverse=True):
            patched = patched[:start] + replacement + patched[end:]
        keeps = documentation.kept(min(declared_names(stmt))) is not None
        written = StatementSource(source=patched, node=stmt).member_source(indentation)
        assert isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
        return ((written if keeps else first.undocumented(patched, stmt, written), True),)

    def extra_imports(
        self, destination_file_path: str, rationale: str
    ) -> tuple[ModuleImportMutation, ...]:
        if not self.imports_typing_self:
            return ()
        return (
            ModuleImportMutation.from_source(
                file_path=destination_file_path,
                import_source="from typing import Self",
                rationale=rationale,
            ),
        )


class HomePlacement(ABC):
    """How the lifted declaration lands in `home`: inserted anew or replacing a stub."""

    @abstractmethod
    def replacement(
        self,
        snapshot: CodemodSourceSnapshot,
        home: OverlapClass,
        first: RestatedDeclarations,
        documentation: KeptDocumentation,
        annotation_plan: HeaderAnnotationPlan = HeaderAnnotationPlan(),
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
        annotation_plan: HeaderAnnotationPlan = HeaderAnnotationPlan(),
    ) -> SourceTextSpanReplacement:
        home_source = snapshot.sources_by_file_path[home.file_path]
        body = ClassBodySourceAuthority(home.node, home_source)
        return body.declaration_block_replacement(
            annotation_plan.member_sources(snapshot, first, body.indentation, documentation)
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
        annotation_plan: HeaderAnnotationPlan = HeaderAnnotationPlan(),
    ) -> SourceTextSpanReplacement:
        home_source = snapshot.sources_by_file_path[home.file_path]
        body = ClassBodySourceAuthority(home.node, home_source)
        start, end = body.geometry.node_span_offsets(SourceNodeSpan(self.stub, Include))
        member_sources = annotation_plan.member_sources(
            snapshot, first, body.indentation, documentation
        )
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
        lifted_names = {
            node.id
            for statement in lifted_statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
        }
        dropped_loads = tuple(
            node
            for node in ast.walk(self.stub)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in lifted_names
        )
        if dropped_loads:
            return obsolete_import_edits(
                snapshot, home.file_path, dropped_loads, rationale
            )
        return ()


@dataclass(frozen=True)
class SiblingOverrideMove(ABC):
    """Proved state shared by lifting onto an existing home or a new intermediate base."""

    candidate: SiblingOverrideCandidate
    children: tuple[OverlapClass, ...]
    restated: tuple[RestatedDeclarations, ...]
    lifted_index: int
    annotation_plan: HeaderAnnotationPlan

    @property
    def lifted_restatement(self) -> RestatedDeclarations:
        return self.restated[self.lifted_index]

    @staticmethod
    def select_lifted_choice(
        snapshot: CodemodSourceSnapshot,
        children: tuple[OverlapClass, ...],
        restated: tuple[RestatedDeclarations, ...],
        destination: OverlapClass,
        member_name: str,
    ) -> tuple[int, HeaderAnnotationPlan]:
        """Choose which sibling's source statement moves onto the shared ancestor and
        any sound header annotation alignment (`Self` return or contravariant parameter
        supertype). Refuse when siblings declare incompatible annotations on the same slot.
        """

        annotations_by_index = [
            method_header_annotations(restatement.statements[0])
            if len(restatement.statements) == 1
            else {}
            for restatement in restated
        ]
        annotation_plan = HeaderAnnotationPlan.for_siblings(
            snapshot, children, annotations_by_index, destination, member_name
        )
        override_map = dict(annotation_plan.overrides)
        candidate_indices = sorted(
            range(len(children)),
            key=lambda index: (
                -int(all(slot in annotations_by_index[index] for slot in override_map)),
                -sum(
                    1
                    for slot, replacement in override_map.items()
                    if annotations_by_index[index].get(slot) == replacement
                ),
                -len(annotations_by_index[index]),
                index,
            ),
        )
        first_error: NeedsInput | None = None
        for index in candidate_indices:
            try:
                annotation_plan.require_renderable(restated[index].statements)
                checked_statements = annotation_plan.rewritten_statements(
                    restated[index].statements, strip_typing_self=True
                )
                movable_members(
                    ClassMemberMoveProofContext.between(
                        snapshot, children[index].target, destination.target
                    ),
                    children[index],
                    checked_statements,
                )
            except NeedsInput as error:
                if first_error is None:
                    first_error = error
                continue
            return index, annotation_plan
        assert first_error is not None
        raise first_error


@dataclass(frozen=True)
class SiblingOverrideLift(SiblingOverrideMove):
    """A sibling override proved safe to lift onto its shared ancestor `home`."""

    home: OverlapClass
    placement: HomePlacement

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
        lifted_index, annotation_plan = cls.select_lifted_choice(
            snapshot, children, restated, home, member_name
        )
        first = restated[lifted_index]
        first.require_no_owner_bound_values(snapshot)
        placement = cls._require_home_placement(
            snapshot, verdicts, candidate, home, member_name
        )
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
        return cls(
            candidate=candidate,
            children=children,
            restated=restated,
            lifted_index=lifted_index,
            annotation_plan=annotation_plan,
            home=home,
            placement=placement,
        )

    @classmethod
    def _require_home_placement(
        cls,
        snapshot: CodemodSourceSnapshot,
        verdicts: ClassOverlapVerdicts,
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
        if isinstance(stub, ast.AnnAssign) and not optional_values(stub):
            if (
                home.facts.dataclass_declaration.generates_construction
                and not _is_classvar_annotation(stub.annotation)
            ):
                raise NeedsInput(
                    f"`{home.qualname}.{member_name}` is a dataclass field, not an abstract contract",
                    NeededInput.PROOF,
                )
            return ReplaceHomeStub(stub)
        if isinstance(stub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign)):
            if (
                isinstance(stub, ast.AnnAssign)
                and home.facts.dataclass_declaration.generates_construction
                and not _is_classvar_annotation(stub.annotation)
            ):
                raise NeedsInput(
                    f"`{home.qualname}.{member_name}` is a dataclass field, not an abstract contract",
                    NeededInput.PROOF,
                )
            cls._require_home_not_directly_used(
                snapshot, candidate, home, member_name, allow_abstract_home=False
            )
            cls._require_no_super_calls_on_member(snapshot, verdicts, home, member_name)
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
        *,
        allow_abstract_home: bool = True,
    ) -> None:
        """When `declaring.name` is concrete, lifting a different answer onto `home`
        is only safe if `home` itself is never instantiated or read directly for the
        old default."""

        if allow_abstract_home and home.facts.abstract_method_names:
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

    @staticmethod
    def _require_no_super_calls_on_member(
        snapshot: CodemodSourceSnapshot,
        verdicts: ClassOverlapVerdicts,
        home: OverlapClass,
        member_name: str,
    ) -> None:
        """Replacing a concrete declaration on `home` is unsafe if any descendant
        reaches the old default through `super().<member_name>`."""

        for desc_symbol in verdicts.class_index.descendant_symbols(home.symbol):
            desc = OverlapClass.for_symbol(snapshot, desc_symbol)
            for node in ast.walk(desc.node):
                match node:
                    case ast.Attribute(
                        attr=attr,
                        value=ast.Call(func=ast.Name(id="super")),
                    ) if attr == member_name:
                        raise NeedsInput(
                            f"{desc.file_path}:{node.lineno} calls `super().{member_name}` "
                            f"in `{desc.qualname}`, which reads `{home.qualname}.{member_name}`",
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
                if _remains_abstract_after_lift(verdicts, order, member_name):
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

    def edits(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        home = self.home
        first = self.lifted_restatement
        documentation = KeptDocumentation(self.children, first.deleted_names)
        rationale = documentation.rationale(rationale)
        home_source = snapshot.sources_by_file_path[home.file_path]
        body = ClassBodySourceAuthority(home.node, home_source)
        home_replacement = self.placement.replacement(
            snapshot, home, first, documentation, self.annotation_plan
        )
        home_edits: list[NominalSourceEdit] = [
            *body.geometry.physical_edits(
                file_path=home.file_path,
                replacements=(home_replacement,),
                rationale=rationale,
            ),
            *required_name_imports(
                snapshot,
                self.annotation_plan.rewritten_statements(
                    first.statements, strip_typing_self=True
                ),
                first.restating,
                home,
                rationale,
            ),
            *self.annotation_plan.extra_imports(home.file_path, rationale),
            *self.placement.obsolete_home_edits(
                snapshot,
                home,
                self.annotation_plan.rewritten_statements(first.statements),
                rationale,
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


def _c3_merge(sequences: list[list[str]]) -> tuple[str, ...] | None:
    """Python C3 linearization merge; returns None if no consistent MRO exists."""

    pending = [list(seq) for seq in sequences if seq]
    result: list[str] = []
    while pending:
        head = next(
            (
                seq[0]
                for seq in pending
                if all(seq[0] not in other[1:] for other in pending)
            ),
            None,
        )
        if head is None:
            return None
        result.append(head)
        pending = [
            seq[1:] if seq[0] == head else seq
            for seq in pending
            if len(seq) > 1 or seq[0] != head
        ]
    return tuple(result)


def _module_top_level_bound_names(module: ast.Module) -> frozenset[str]:
    names: set[str] = set()
    for statement in module.body:
        match statement:
            case ast.ClassDef(name=name) | ast.FunctionDef(name=name) | ast.AsyncFunctionDef(
                name=name
            ):
                names.add(name)
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                for alias in aliases:
                    bound = next(iter(alias_names(alias)), alias.name)
                    names.add(bound.split(".")[0])
            case ast.Assign(targets=targets):
                for target in targets:
                    for node in ast.walk(target):
                        if isinstance(node, ast.Name):
                            names.add(node.id)
            case ast.AnnAssign(target=ast.Name(id=name)):
                names.add(name)
    return frozenset(names)


def _creation_keywords_for_intermediate(
    snapshot: CodemodSourceSnapshot, parent: OverlapClass
) -> tuple[str, ...]:
    """Copy `declared_name=None`-style opt-out keywords from a registering parent."""

    keywords: list[str] = []
    for keyword in parent.node.keywords:
        match keyword:
            case ast.keyword(arg=str() as arg, value=ast.Constant(value=None)):
                keywords.append(arg)
    if keywords:
        return tuple(keywords)
    index = snapshot.class_overlap_verdicts.class_index
    verdicts = snapshot.class_overlap_verdicts
    order = verdicts.linearized(parent.symbol)
    for symbol in order:
        if symbol.rsplit(".", 1)[-1] == "DeclaredFamily":
            return ("declared_name",)
        indexed = index.class_for(symbol)
        if indexed is None:
            continue
        if any(
            base.rsplit(".", 1)[-1] == "DeclaredFamily"
            for base in indexed.declared_base_names
        ):
            return ("declared_name",)
    for symbol in order:
        indexed = index.class_for(symbol)
        if indexed is None:
            continue
        if any(
            meta.rsplit(".", 1)[-1] == "AutoRegisterMeta"
            for meta in indexed.metaclass_names
        ):
            raise NeedsInput(
                f"`{parent.qualname}` uses `AutoRegisterMeta` and declares no `None` "
                "creation keyword that keeps an unnamed intermediate out of the registry",
                NeededInput.PROOF,
            )
    return ()


@dataclass(frozen=True)
class DirectSubclassReadIndex(ParsedModuleSetFact):
    """First `X.__subclasses__` attribute read per simple class name across the module set."""

    first_read_by_name: dict[str, tuple[str, int]]

    @classmethod
    def derive(cls, modules: ParsedModuleSet) -> DirectSubclassReadIndex:
        first_read: dict[str, tuple[str, int]] = {}
        for parsed_module in modules:
            if "__subclasses__" not in parsed_module.source:
                continue
            for node in ast.walk(parsed_module.module):
                match node:
                    case ast.Attribute(attr="__subclasses__", value=value):
                        name = ast.unparse(value).rsplit(".", 1)[-1]
                        first_read.setdefault(
                            name, (parsed_module.file_path, node.lineno)
                        )
        return cls(first_read)


def _require_no_direct_subclass_reads(
    snapshot: CodemodSourceSnapshot, parent_simple_name: str
) -> None:
    found = snapshot.module_set.fact(DirectSubclassReadIndex).first_read_by_name.get(
        parent_simple_name
    )
    if found is not None:
        path, lineno = found
        raise NeedsInput(
            f"`{parent_simple_name}.__subclasses__()` is read at "
            f"{path}:{lineno}, and rebasing the siblings moves them out of it",
            NeededInput.SOURCE_CHANGE,
        )


class SiblingBaseRebase(ABC):
    """How each answering sibling inherits the newly extracted intermediate base."""

    parent: OverlapClass

    @abstractmethod
    def new_base_items_for(
        self,
        child_index: int,
        header: ClassHeaderSpanSourceAuthority,
        base_name: str,
    ) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def replaced_base_nodes(self, child_index: int) -> tuple[ast.AST, ...]:
        raise NotImplementedError

    @abstractmethod
    def new_base_symbols_for(
        self,
        child: OverlapClass,
        new_symbol: str,
    ) -> tuple[str, ...]:
        raise NotImplementedError

    def require_parent_safe(self, snapshot: CodemodSourceSnapshot) -> None:
        del snapshot

    def direct_base_spelling_for(self, child_index: int) -> str | None:
        del child_index
        return None

    def changed_direct_bases(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True)
class DirectBaseRebase(SiblingBaseRebase):
    """Every answering sibling directly inherits `parent`: replace `parent` with `base_name`."""

    parent: OverlapClass
    base_exprs: tuple[ast.expr, ...]

    def new_base_items_for(
        self,
        child_index: int,
        header: ClassHeaderSpanSourceAuthority,
        base_name: str,
    ) -> tuple[str, ...]:
        target_expr = self.base_exprs[child_index]
        return tuple(
            base_name if base is target_expr else header.item_source(base)
            for base in header.node.bases
        )

    def replaced_base_nodes(self, child_index: int) -> tuple[ast.AST, ...]:
        return (self.base_exprs[child_index],)

    def new_base_symbols_for(
        self,
        child: OverlapClass,
        new_symbol: str,
    ) -> tuple[str, ...]:
        return tuple(
            new_symbol if symbol == self.parent.symbol else symbol
            for symbol in child.facts.resolved_base_symbols
        )

    def require_parent_safe(self, snapshot: CodemodSourceSnapshot) -> None:
        _require_no_direct_subclass_reads(snapshot, self.parent.facts.simple_name)

    def direct_base_spelling_for(self, child_index: int) -> str | None:
        return ast.unparse(self.base_exprs[child_index])

    def changed_direct_bases(self) -> frozenset[str]:
        return frozenset((self.parent.symbol,))


@dataclass(frozen=True)
class PrependBaseRebase(SiblingBaseRebase):
    """Answering siblings inherit `parent` through distinct branches: prepend `base_name`."""

    parent: OverlapClass

    def new_base_items_for(
        self,
        child_index: int,
        header: ClassHeaderSpanSourceAuthority,
        base_name: str,
    ) -> tuple[str, ...]:
        del child_index
        return (base_name, *(header.item_source(base) for base in header.node.bases))

    def replaced_base_nodes(self, child_index: int) -> tuple[ast.AST, ...]:
        del child_index
        return ()

    def new_base_symbols_for(
        self,
        child: OverlapClass,
        new_symbol: str,
    ) -> tuple[str, ...]:
        return (new_symbol, *child.facts.resolved_base_symbols)


class NewBasePlacement(ABC):
    """Where the newly extracted intermediate base is declared."""

    @property
    @abstractmethod
    def file_path(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def anchor(self) -> OverlapClass:
        raise NotImplementedError

    @abstractmethod
    def insertion_edit(
        self,
        snapshot: CodemodSourceSnapshot,
        base_name: str,
        parent_spelling: str,
        creation_keywords: tuple[str, ...],
        docstring: str,
        first: RestatedDeclarations,
        documentation: KeptDocumentation,
        rationale: str,
        annotation_plan: HeaderAnnotationPlan = HeaderAnnotationPlan(),
    ) -> SourceInsertion:
        raise NotImplementedError

    @abstractmethod
    def extra_imports(
        self,
        snapshot: CodemodSourceSnapshot,
        base_name: str,
        parent: OverlapClass,
        children: tuple[OverlapClass, ...],
        rationale: str,
    ) -> tuple[ModuleImportMutation, ...]:
        raise NotImplementedError


def _rendered_class_lines(
    snapshot: CodemodSourceSnapshot,
    indentation: str,
    base_name: str,
    parent_spelling: str,
    creation_keywords: tuple[str, ...],
    docstring: str,
    first: RestatedDeclarations,
    documentation: KeptDocumentation,
    annotation_plan: HeaderAnnotationPlan = HeaderAnnotationPlan(),
) -> tuple[str, ...]:
    member_indentation = indentation + "    "
    keywords = "".join(f", {keyword}=None" for keyword in creation_keywords)
    members: list[str] = []
    for member_text, is_fn in annotation_plan.member_sources(
        snapshot, first, member_indentation, documentation
    ):
        if is_fn and members:
            members.append("\n")
        members.append(member_text if member_text.endswith("\n") else member_text + "\n")
    return (
        f"{indentation}class {base_name}({parent_spelling}{keywords}):\n",
        f'{member_indentation}"""{docstring}"""\n\n',
        *members,
    )


@dataclass(frozen=True)
class SameModuleBeforeFirstSibling(NewBasePlacement):
    """Declare the new base right above the earliest sibling in their shared module."""

    first_sibling: OverlapClass

    @property
    def file_path(self) -> str:
        return self.first_sibling.file_path

    @property
    def anchor(self) -> OverlapClass:
        return self.first_sibling

    def insertion_edit(
        self,
        snapshot: CodemodSourceSnapshot,
        base_name: str,
        parent_spelling: str,
        creation_keywords: tuple[str, ...],
        docstring: str,
        first: RestatedDeclarations,
        documentation: KeptDocumentation,
        rationale: str,
        annotation_plan: HeaderAnnotationPlan = HeaderAnnotationPlan(),
    ) -> SourceInsertion:
        source = snapshot.sources_by_file_path[self.file_path]
        lines = source.splitlines(keepends=True)
        first_line = min(
            (dec.lineno for dec in self.first_sibling.node.decorator_list),
            default=self.first_sibling.node.lineno,
        )
        header = lines[first_line - 1]
        indentation = header[: len(header) - len(header.lstrip())]
        class_lines = _rendered_class_lines(
            snapshot,
            indentation,
            base_name,
            parent_spelling,
            creation_keywords,
            docstring,
            first,
            documentation,
            annotation_plan,
        )
        return SourceInsertion(
            file_path=self.file_path,
            insertion_line=first_line,
            inserted_lines=(*class_lines, "\n\n"),
            rationale=rationale,
        )

    def extra_imports(
        self,
        snapshot: CodemodSourceSnapshot,
        base_name: str,
        parent: OverlapClass,
        children: tuple[OverlapClass, ...],
        rationale: str,
    ) -> tuple[ModuleImportMutation, ...]:
        del snapshot, base_name, parent, children, rationale
        return ()


@dataclass(frozen=True)
class SameModuleImportingParent(SameModuleBeforeFirstSibling):
    """Declare the new base above the earliest sibling and import `parent` there."""

    def extra_imports(
        self,
        snapshot: CodemodSourceSnapshot,
        base_name: str,
        parent: OverlapClass,
        children: tuple[OverlapClass, ...],
        rationale: str,
    ) -> tuple[ModuleImportMutation, ...]:
        del base_name, children
        return (
            ModuleImportMutation.from_source(
                file_path=self.file_path,
                import_source=snapshot.module_import_graph.required_import_source(
                    importing_file_path=self.file_path,
                    imported_file_path=parent.file_path,
                    imported_name=parent.facts.simple_name,
                ),
                rationale=rationale,
            ),
        )


@dataclass(frozen=True)
class ParentModuleAfterParent(NewBasePlacement):
    """Declare the new base right after `parent` in `parent`'s module and import it into
    each sibling module."""

    parent: OverlapClass

    @property
    def file_path(self) -> str:
        return self.parent.file_path

    @property
    def anchor(self) -> OverlapClass:
        return self.parent

    def insertion_edit(
        self,
        snapshot: CodemodSourceSnapshot,
        base_name: str,
        parent_spelling: str,
        creation_keywords: tuple[str, ...],
        docstring: str,
        first: RestatedDeclarations,
        documentation: KeptDocumentation,
        rationale: str,
        annotation_plan: HeaderAnnotationPlan = HeaderAnnotationPlan(),
    ) -> SourceInsertion:
        source = snapshot.sources_by_file_path[self.file_path]
        lines = source.splitlines(keepends=True)
        end_line = self.parent.node.end_lineno or self.parent.node.lineno
        header = lines[self.parent.node.lineno - 1]
        indentation = header[: len(header) - len(header.lstrip())]
        class_lines = _rendered_class_lines(
            snapshot,
            indentation,
            base_name,
            parent_spelling,
            creation_keywords,
            docstring,
            first,
            documentation,
            annotation_plan,
        )
        return SourceInsertion(
            file_path=self.file_path,
            insertion_line=end_line + 1,
            inserted_lines=("\n\n", *class_lines),
            rationale=rationale,
        )

    def extra_imports(
        self,
        snapshot: CodemodSourceSnapshot,
        base_name: str,
        parent: OverlapClass,
        children: tuple[OverlapClass, ...],
        rationale: str,
    ) -> tuple[ModuleImportMutation, ...]:
        sibling_paths = sorted(
            {child.file_path for child in children if child.file_path != self.file_path}
        )
        return tuple(
            ModuleImportMutation.from_source(
                file_path=sibling_path,
                import_source=snapshot.module_import_graph.required_import_source(
                    importing_file_path=sibling_path,
                    imported_file_path=parent.file_path,
                    imported_name=base_name,
                ),
                rationale=rationale,
            )
            for sibling_path in sibling_paths
        )


@dataclass(frozen=True)
class SiblingOverrideBaseExtraction(SiblingOverrideMove):
    """A partial-subset sibling override proved safe to lift into a new intermediate base."""

    rebase: SiblingBaseRebase
    placement: NewBasePlacement
    parent_spelling: str
    base_name: str
    docstring: str
    creation_keywords: tuple[str, ...]

    @classmethod
    def proved(
        cls,
        snapshot: CodemodSourceSnapshot,
        declaring_target: SourceRewriteTarget,
        member_name: str,
        sibling_targets: tuple[SourceRewriteTarget, ...],
        base_name: str,
        docstring: str,
    ) -> SiblingOverrideBaseExtraction:
        cls._require_valid_pick(base_name, docstring)
        declaring = OverlapClass.require(snapshot, declaring_target)
        if member_name.startswith("__") and member_name.endswith("__"):
            raise NeedsInput(
                f"`{declaring.qualname}.{member_name}` is a dunder member, which class "
                "creation and decorators treat by name: NRA needs a proof that lifting "
                "it into a new base leaves each class unchanged",
                NeededInput.PROOF,
            )
        target_symbols = frozenset(
            OverlapClass.require(snapshot, target).symbol for target in sibling_targets
        )
        verdicts = snapshot.class_overlap_verdicts
        matches = [
            cand
            for cand in sibling_override_candidates(verdicts)
            if cand.declaring.symbol == declaring.symbol
            and cand.name == member_name
            and frozenset(indexed.symbol for indexed, _ in cand.answering)
            == target_symbols
        ]
        if len(matches) != 1:
            raise NeedsInput(
                f"expected one sibling-override group for `{declaring.qualname}.{member_name}`, "
                f"found {len(matches)}",
                NeededInput.SOURCE_CHANGE,
            )
        (candidate,) = matches
        return candidate.require_no_existing_home(member_name, snapshot, declaring, verdicts, cls, base_name, docstring)

    @staticmethod
    def _require_valid_pick(base_name: str, docstring: str) -> None:
        if not base_name.isidentifier() or keyword_module.iskeyword(base_name):
            raise NeedsInput(
                f"`{base_name}` is not a valid class name",
                NeededInput.OWNER_PICK,
            )
        cleaned = docstring.strip()
        forbidden = ('"""', "\\", "\n")
        if not cleaned or any(token in cleaned for token in forbidden):
            raise NeedsInput(
                f"`{base_name}` needs a one-line docstring without triple quotes or backslashes",
                NeededInput.OWNER_PICK,
            )

    @staticmethod
    def _select_rebase(
        snapshot: CodemodSourceSnapshot,
        verdicts: ClassOverlapVerdicts,
        declaring: OverlapClass,
        children: tuple[OverlapClass, ...],
    ) -> SiblingBaseRebase:
        index = verdicts.class_index
        shared_direct = [
            symbol
            for symbol in children[0].facts.resolved_base_symbols
            if symbol in index.classes_by_symbol
            and (
                symbol == declaring.symbol
                or declaring.symbol in index.ancestor_symbols(symbol)
            )
            and all(symbol in child.facts.resolved_base_symbols for child in children)
        ]
        if shared_direct:
            parent_symbol = max(
                shared_direct,
                key=lambda sym: (len(index.ancestor_symbols(sym)), sym),
            )
            parent = OverlapClass.for_symbol(snapshot, parent_symbol)
            simple = parent.facts.simple_name
            base_exprs: list[ast.expr] = []
            for child in children:
                matches = [
                    base
                    for base in child.node.bases
                    if (declared := ClassSymbolResolutionAuthority.declared_base_name(base))
                    is not None
                    and declared.rsplit(".", 1)[-1] == simple
                ]
                if len(matches) != 1:
                    return PrependBaseRebase(declaring)
                base_exprs.append(matches[0])
            return DirectBaseRebase(parent, tuple(base_exprs))
        return PrependBaseRebase(declaring)

    @staticmethod
    def _select_placement(
        snapshot: CodemodSourceSnapshot,
        rebase: SiblingBaseRebase,
        children: tuple[OverlapClass, ...],
        base_name: str,
    ) -> tuple[NewBasePlacement, str]:
        del base_name
        parent = rebase.parent
        sibling_files = {child.file_path for child in children}
        if len(sibling_files) == 1:
            first_index, first_sibling = min(
                enumerate(children), key=lambda pair: pair[1].node.lineno
            )
            direct_spelling = rebase.direct_base_spelling_for(first_index)
            if direct_spelling is not None:
                return SameModuleBeforeFirstSibling(first_sibling), direct_spelling
            parsed = snapshot.parsed_module_for_source_path(first_sibling.file_path)
            witness = snapshot.module_binding_proof.reference_or_builtin_witness_at(
                parsed,
                ast.Name(id=parent.facts.simple_name, ctx=ast.Load()),
                line=first_sibling.node.lineno,
            )
            if witness is not None:
                if witness.qualified_name == parent.symbol:
                    return (
                        SameModuleBeforeFirstSibling(first_sibling),
                        parent.facts.simple_name,
                    )
                raise NeedsInput(
                    f"`{parent.facts.simple_name}` in {first_sibling.file_path} refers "
                    f"to `{witness.qualified_name}`, not `{parent.symbol}`",
                    NeededInput.SOURCE_CHANGE,
                )
            return (
                SameModuleImportingParent(first_sibling),
                parent.facts.simple_name,
            )
        return ParentModuleAfterParent(parent), parent.facts.simple_name

    @staticmethod
    def _require_base_name_unbound(
        snapshot: CodemodSourceSnapshot,
        base_name: str,
        placement: NewBasePlacement,
        children: tuple[OverlapClass, ...],
    ) -> None:
        checked_paths = sorted({placement.file_path, *(c.file_path for c in children)})
        for file_path in checked_paths:
            module = snapshot.module_nodes_by_file_path[file_path]
            if base_name in _module_top_level_bound_names(module):
                raise NeedsInput(
                    f"`{base_name}` is already bound in {file_path}",
                    NeededInput.OWNER_PICK,
                )

    @staticmethod
    def _require_post_rebase_mro_safe(
        snapshot: CodemodSourceSnapshot,
        verdicts: ClassOverlapVerdicts,
        rebase: SiblingBaseRebase,
        declaring: OverlapClass,
        children: tuple[OverlapClass, ...],
        member_name: str,
        base_name: str,
        *,
        lifted_is_property: bool,
    ) -> None:
        index = verdicts.class_index
        bodies = verdicts.bodies
        parent_mro = verdicts.linearized(rebase.parent.symbol)
        new_symbol = f"<extracted>.{base_name}"
        new_mros: dict[str, tuple[str, ...]] = {
            new_symbol: (new_symbol, *parent_mro),
        }
        answering_by_symbol = {child.symbol: child for child in children}
        for child in children:
            old_order = verdicts.linearized(child.symbol)
            if declaring.symbol not in old_order:
                raise NeedsInput(
                    f"`{declaring.qualname}` is not in `{child.qualname}`'s MRO",
                    NeededInput.PROOF,
                )
            between = old_order[1 : old_order.index(declaring.symbol)]
            intervening = next_binder(verdicts, between, member_name)
            if intervening is not None:
                raise NeedsInput(
                    f"`{intervening}` binds `{member_name}` between `{child.qualname}` "
                    f"and `{declaring.qualname}`",
                    NeededInput.PROOF,
                )
            rebased_bases = rebase.new_base_symbols_for(child, new_symbol)
            base_mros = [
                list(new_mros[b] if b in new_mros else verdicts.linearized(b))
                for b in rebased_bases
            ]
            merged = _c3_merge([*base_mros, list(rebased_bases)])
            if merged is None:
                raise NeedsInput(
                    f"inserting `{base_name}` produces an inconsistent C3 MRO for `{child.qualname}`",
                    NeededInput.PROOF,
                )
            new_order = (child.symbol, *merged)
            if tuple(s for s in new_order if s != new_symbol) != old_order:
                raise NeedsInput(
                    f"inserting `{base_name}` changes `{child.qualname}`'s MRO order beyond `{member_name}`",
                    NeededInput.PROOF,
                )
            new_mros[child.symbol] = new_order

        # Check all descendants of answering siblings in topological base-before-child order.
        descendants: set[str] = set()
        for child in children:
            descendants.update(index.descendant_symbols(child.symbol))
        ordered_descendants = sorted(
            descendants - frozenset(answering_by_symbol),
            key=lambda sym: (len(index.ancestor_symbols(sym)), sym),
        )
        for desc_symbol in ordered_descendants:
            desc = OverlapClass.for_symbol(snapshot, desc_symbol)
            require_static_c3(snapshot, desc.facts)
            old_order = verdicts.linearized(desc_symbol)
            base_mros = [
                list(new_mros[b] if b in new_mros else verdicts.linearized(b))
                for b in desc.facts.resolved_base_symbols
            ]
            merged = _c3_merge([*base_mros, list(desc.facts.resolved_base_symbols)])
            if merged is None:
                raise NeedsInput(
                    f"inserting `{base_name}` produces an inconsistent C3 MRO for descendant `{desc.qualname}`",
                    NeededInput.PROOF,
                )
            new_order = (desc_symbol, *merged)
            if tuple(s for s in new_order if s != new_symbol) != old_order:
                raise NeedsInput(
                    f"inserting `{base_name}` changes descendant `{desc.qualname}`'s MRO order",
                    NeededInput.PROOF,
                )
            new_mros[desc_symbol] = new_order
            if lifted_is_property:
                before_new = new_order[: new_order.index(new_symbol)]
                binder = next_binder(verdicts, before_new, member_name)
                if binder is not None:
                    binder_decl = bodies.declarations_of(binder).get(member_name)
                    if binder_decl is not None and binder_decl.kind.is_field:
                        raise NeedsInput(
                            f"`{desc.qualname}` binds `{member_name}` as a dataclass field, "
                            f"which a property on `{base_name}` would shadow",
                            NeededInput.PROOF,
                        )

    def edits(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        first = self.lifted_restatement
        documentation = KeptDocumentation(self.children, first.deleted_names)
        rationale = documentation.rationale(rationale)
        edits: list[NominalSourceEdit] = [
            self.placement.insertion_edit(
                snapshot,
                self.base_name,
                self.parent_spelling,
                self.creation_keywords,
                self.docstring,
                first,
                documentation,
                rationale,
                self.annotation_plan,
            ),
            *required_name_imports(
                snapshot,
                self.annotation_plan.rewritten_statements(
                    first.statements, strip_typing_self=True
                ),
                first.restating,
                self.placement.anchor,
                rationale,
            ),
            *self.annotation_plan.extra_imports(self.placement.file_path, rationale),
            *self.placement.extra_imports(
                snapshot, self.base_name, self.rebase.parent, self.children, rationale
            ),
        ]
        deleted_by_file: defaultdict[str, list[ast.AST]] = defaultdict(list)
        for child_index, restatement in enumerate(self.restated):
            child = restatement.restating
            source = snapshot.sources_by_file_path[child.file_path]
            header = ClassHeaderSpanSourceAuthority(child.node, source)
            edits.extend(
                header.source_edits(
                    header.with_base_items(
                        self.rebase.new_base_items_for(
                            child_index, header, self.base_name
                        )
                    ),
                    file_path=child.file_path,
                    rationale=rationale,
                )
            )
            deletion = StatementDeletionSource(
                source=source,
                node=child.node,
                file_path=child.file_path,
            )
            edits.extend(
                deletion.physical_edits(
                    file_path=child.file_path,
                    replacements=deletion.replacements_for_statements(
                        restatement.written_statements
                    ),
                    rationale=rationale,
                )
            )
            if child.file_path != self.placement.file_path:
                deleted_by_file[child.file_path].extend(restatement.statements)
                deleted_by_file[child.file_path].extend(
                    self.rebase.replaced_base_nodes(child_index)
                )
        for file_path, dropped in deleted_by_file.items():
            edits.extend(
                obsolete_import_edits(snapshot, file_path, tuple(dropped), rationale)
            )
        return tuple(edits)

    def rationale(self) -> str:
        siblings = ", ".join(f"`{child.qualname}`" for child in self.children)
        return (
            f"Declares `{self.base_name}` beneath `{self.rebase.parent.qualname}` "
            f"holding `{self.candidate.name}` once and deletes the identical sibling "
            f"copies from {siblings}."
        )


@dataclass(frozen=True, kw_only=True)
class ExtractSiblingOverrideBaseOperation(RepositorySourceReprovedOperation):
    """Declare a new intermediate base beneath `target` holding one identically
    overridden member and rebase the answering sibling classes onto it."""

    member_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    sibling_targets: tuple[SourceRewriteTarget, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(SourceRewriteTarget)
    )
    base_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    docstring: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        if not self.member_name.isidentifier() or keyword_module.iskeyword(
            self.member_name
        ):
            raise ValueError("member_name must be a Python identifier")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        extraction = SiblingOverrideBaseExtraction.proved(
            snapshot,
            self.target,
            self.member_name,
            self.sibling_targets,
            self.base_name,
            self.docstring,
        )
        return extraction.edits(snapshot, self.rationale_text(extraction.rationale()))


class SiblingOverrideFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    PrewarmTypeCheckerMixin,
    SiblingOverrideConcept,
):
    """Lift a sibling override onto its existing shared ancestor, or into a picked
    intermediate base when no shared ancestor holds only these siblings."""

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
        return candidate.evaluate_override_recipe(finding, context, self)

    def _evaluate_intermediate_base_recipe(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
        candidate: SiblingOverrideCandidate,
    ) -> FindingRecipeEvaluation:
        base_pick = finding.metrics.picked_base
        if base_pick is None:
            if candidate.unordered:
                return self.stopped_evaluation(
                    NeedsInput(
                        f"lookup order of {', '.join(f'`{name}`' for name in candidate.unordered)} "
                        "is not proved (a base has no source the scan reads)",
                        NeededInput.PROOF,
                    )
                )
            if candidate.name.startswith("__") and candidate.name.endswith("__"):
                return self.stopped_evaluation(
                    NeedsInput(
                        f"`{candidate.declaring.qualname}.{candidate.name}` is a dunder member",
                        NeededInput.PROOF,
                    )
                )
            return self.stopped_evaluation(
                NeedsInput(
                    f"no existing class beneath `{candidate.declaring.qualname}` holds "
                    f"`{candidate.name}` for only these siblings; supply a new intermediate "
                    "base name and one-line judgement in `--codemod-owner-picks`",
                    NeededInput.OWNER_PICK,
                )
            )
        operation = ExtractSiblingOverrideBaseOperation(
            target=source_rewrite_target(candidate.declaring),
            member_name=candidate.name,
            sibling_targets=tuple(
                source_rewrite_target(indexed) for indexed, _ in candidate.answering
            ),
            base_name=base_pick.name,
            docstring=base_pick.judgement,
        )
        try:
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=(
                    f"Declares `{base_pick.name}` beneath `{candidate.declaring.qualname}` "
                    f"to hold `{candidate.name}` once for these siblings."
                ),
            ).with_proved_operation(operation, context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(recipe)


__all__ = (
    "DirectBaseRebase",
    "ExtractSiblingOverrideBaseOperation",
    "HomePlacement",
    "InsertIntoHome",
    "LiftSiblingOverrideOperation",
    "NewBasePlacement",
    "Override",
    "ParentModuleAfterParent",
    "PrependBaseRebase",
    "ReplaceHomeStub",
    "SameModuleBeforeFirstSibling",
    "SameModuleImportingParent",
    "SiblingBaseRebase",
    "SiblingOverrideBaseExtraction",
    "SiblingOverrideCandidate",
    "SiblingOverrideFindingRecipeSynthesizer",
    "SiblingOverrideLift",
    "SiblingOverrideMove",
    "sibling_override_candidates",
)
