from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import cast

from nominal_refactor_advisor.assignment_projection import (
    SingleAssignmentAndValueNameProjection,
)
from nominal_refactor_advisor.ast_tools import BuiltinCallName
from nominal_refactor_advisor.class_families.parsed_index import (
    ClassFamilyIndex,
    IndexedClass,
    ModuleClassReferenceResolver,
)
from nominal_refactor_advisor.refactors.semantic_mirror import (
    RegistrationSemanticMirrorRecipeBuilder,
    SemanticMirrorOperationTargets,
)
from nominal_refactor_advisor.refactors.class_base import ClassAuthorityReferenceProof
from nominal_refactor_advisor.refactors.authority_extraction import (
    SourceDerivedAuthorityProjectionOperation,
)
from nominal_refactor_advisor.codemod_authority_claims import AstTargetAuthorityClaim
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceLineSpan,
    SourceSpanReplacement,
    SourceTargetEditor,
)
from nominal_refactor_advisor.refactor_concepts import ClassFamilyAuthorityConcept
from nominal_refactor_advisor.semantic_descent import SemanticAuthorityKind
from nominal_refactor_advisor.source_index import AstTargetNodeKind
from nominal_refactor_advisor.codemod_target_selectors import SourceIndexTargetSelector
from nominal_refactor_advisor.semantic_match import single_item
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput


class ClassFamilyCollectionFactory(StrEnum):
    """Collection syntax and ordering semantics for one derived family view."""

    def __new__(
        cls,
        value: str,
        literal_node_type: type[ast.Tuple | ast.List | ast.Set] | None,
        preserves_order: bool,
    ) -> ClassFamilyCollectionFactory:
        member = str.__new__(cls, value)
        member._value_ = value
        member._literal_node_type = literal_node_type
        member._preserves_order = preserves_order
        return member

    TUPLE = (BuiltinCallName.TUPLE.value, ast.Tuple, True)
    LIST = (BuiltinCallName.LIST.value, ast.List, True)
    SET = (BuiltinCallName.SET.value, ast.Set, False)
    FROZENSET = (BuiltinCallName.FROZENSET.value, None, False)

    def is_shadowed(self, unavailable_builtin_names: frozenset[str]) -> bool:
        """Whether the module rebinds this factory's builtin name."""

        return self.value in unavailable_builtin_names

    def elements(
        self,
        value: ast.AST,
        unavailable_builtin_names: frozenset[str],
    ) -> tuple[ast.expr, ...] | None:
        if self._literal_node_type is not None and isinstance(
            value, self._literal_node_type
        ):
            return tuple(value.elts)
        if (
            not isinstance(value, ast.Call)
            or not isinstance(value.func, ast.Name)
            or value.func.id != self.value
            or self.is_shadowed(unavailable_builtin_names)
            or len(value.args) != 1
            or value.keywords
            or not isinstance(value.args[0], ast.Tuple | ast.List | ast.Set)
        ):
            return None
        return tuple(value.args[0].elts)

    def preserves_member_sequence(
        self,
        observed: tuple[str, ...],
        expected: tuple[str, ...],
    ) -> bool:
        if len(observed) != len(expected):
            return False
        if self._preserves_order:
            return observed == expected
        return frozenset(observed) == frozenset(expected)

    def runtime_member_sequence(
        self,
        member_symbols: tuple[str, ...],
        class_index: ClassFamilyIndex,
    ) -> tuple[str, ...] | None:
        if not self._preserves_order:
            return member_symbols
        members = tuple(class_index.class_for(symbol) for symbol in member_symbols)
        if any(member is None for member in members):
            return None
        indexed_members = cast(tuple[IndexedClass, ...], members)
        if len({member.file_path for member in indexed_members}) != 1:
            return None
        return tuple(
            member.symbol
            for member in sorted(indexed_members, key=lambda member: member.line)
        )


def _class_object_family_symbols(
    elements: tuple[ast.expr, ...],
    resolver: ModuleClassReferenceResolver,
    family_symbols: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Each element as the class it names. A call constructs an instance, so a
    collection of `Member(...)` values is a collection of instances, each its own
    fact; its class-object view would change the value (zero-argument instance
    views are `AutoregisterInstanceView`'s)."""

    del family_symbols
    if not all(isinstance(element, (ast.Name, ast.Attribute)) for element in elements):
        return None
    symbols = tuple(resolver.symbol_for_reference(element) for element in elements)
    if any(symbol is None for symbol in symbols):
        return None
    return cast(tuple[str, ...], symbols)


def _class_name_family_symbols(
    elements: tuple[ast.expr, ...],
    resolver: ModuleClassReferenceResolver,
    family_symbols: tuple[str, ...],
) -> tuple[str, ...] | None:
    del resolver
    symbols_by_name: dict[str, str] = {}
    for symbol in family_symbols:
        name = symbol.rsplit(".", 1)[-1]
        if name in symbols_by_name:
            return None
        symbols_by_name[name] = symbol
    names = tuple(
        element.value
        for element in elements
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    )
    if len(names) != len(elements):
        return None
    symbols = tuple(symbols_by_name.get(name) for name in names)
    if any(symbol is None for symbol in symbols):
        return None
    return cast(tuple[str, ...], symbols)


class ClassFamilyCollectionElementProjection(StrEnum):
    """How one collection projection references a class-family member."""

    def __new__(
        cls,
        value: str,
        symbol_projector: Callable[
            [
                tuple[ast.expr, ...],
                ModuleClassReferenceResolver,
                tuple[str, ...],
            ],
            tuple[str, ...] | None,
        ],
        value_source_builder: Callable[[str, str], str],
    ) -> ClassFamilyCollectionElementProjection:
        member = str.__new__(cls, value)
        member._value_ = value
        member._symbol_projector = symbol_projector
        member._value_source_builder = value_source_builder
        return member

    CLASS_OBJECT = (
        "class_object",
        _class_object_family_symbols,
        lambda factory_name, member_source: f"{factory_name}({member_source})",
    )
    CLASS_NAME = (
        "class_name",
        _class_name_family_symbols,
        lambda factory_name, member_source: (
            f"{factory_name}(member_type.__name__ for member_type in {member_source})"
        ),
    )

    def projected_symbols(
        self,
        elements: tuple[ast.expr, ...],
        resolver: ModuleClassReferenceResolver,
        family_symbols: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        return self._symbol_projector(elements, resolver, family_symbols)

    def value_source(
        self,
        factory: ClassFamilyCollectionFactory,
        member_source: str,
    ) -> str:
        return self._value_source_builder(factory.value, member_source)


class ClassFamilyCollectionMembershipProjection(StrEnum):
    """Runtime member query selected from the nominal authority declaration."""

    def __new__(
        cls,
        value: str,
        authority_matcher: Callable[[bool, bool, bool], bool],
        member_symbol_projector: Callable[[ClassFamilyIndex, str], tuple[str, ...]],
        value_source_builder: Callable[[str], str],
    ) -> ClassFamilyCollectionMembershipProjection:
        member = str.__new__(cls, value)
        member._value_ = value
        member._authority_matcher = authority_matcher
        member._member_symbol_projector = member_symbol_projector
        member._value_source_builder = value_source_builder
        return member

    AUTOREGISTER_REGISTRY = (
        "autoregister_registry",
        lambda declares_autoregister, covers_family, _all_direct: (
            declares_autoregister and covers_family
        ),
        lambda class_index, authority_symbol: class_index.descendant_symbols(
            authority_symbol
        ),
        lambda authority_name: f"{authority_name}.__registry__.values()",
    )
    DIRECT_SUBCLASSES = (
        "direct_subclasses",
        lambda declares_autoregister, covers_family, all_direct: (
            not declares_autoregister and covers_family and all_direct
        ),
        lambda class_index, authority_symbol: class_index.children_by_symbol.get(
            authority_symbol, ()
        ),
        lambda authority_name: f"{authority_name}.__subclasses__()",
    )

    @classmethod
    def for_authority_declaration(
        cls,
        declares_autoregister_meta: bool,
        covers_complete_family: bool,
        all_members_are_direct: bool,
    ) -> ClassFamilyCollectionMembershipProjection | None:
        return single_item(
            tuple(
                projection
                for projection in cls
                if projection._authority_matcher(
                    declares_autoregister_meta,
                    covers_complete_family,
                    all_members_are_direct,
                )
            )
        )

    def value_source(self, authority_name: str) -> str:
        return self._value_source_builder(authority_name)

    def member_symbols(
        self,
        class_index: ClassFamilyIndex,
        authority_symbol: str,
    ) -> tuple[str, ...]:
        return self._member_symbol_projector(class_index, authority_symbol)


@dataclass(frozen=True)
class ClassFamilyCollectionProjection:
    """Source-level collection shape proven to mirror class-family members."""

    factory: ClassFamilyCollectionFactory
    element_projection: ClassFamilyCollectionElementProjection
    projected_symbols: tuple[str, ...]

    @classmethod
    def from_value(
        cls,
        value: ast.AST,
        unavailable_builtin_names: frozenset[str],
        resolver: ModuleClassReferenceResolver,
        family_symbols: tuple[str, ...],
    ) -> tuple[ClassFamilyCollectionProjection, ...]:
        return tuple(
            cls(
                factory=factory,
                element_projection=element_projection,
                projected_symbols=projected_symbols,
            )
            for factory in ClassFamilyCollectionFactory
            if (elements := factory.elements(value, unavailable_builtin_names))
            is not None
            for element_projection in ClassFamilyCollectionElementProjection
            if (
                projected_symbols := element_projection.projected_symbols(
                    elements,
                    resolver,
                    family_symbols,
                )
            )
            is not None
        )

    def value_source(
        self,
        authority_name: str,
        membership_projection: ClassFamilyCollectionMembershipProjection,
    ) -> str:
        return self.element_projection.value_source(
            self.factory,
            membership_projection.value_source(authority_name),
        )


@dataclass(frozen=True)
class ClassFamilyCollectionCandidate:
    """One source projection proven equal to a complete nominal class family."""

    statement: ast.Assign | ast.AnnAssign
    collection: ClassFamilyCollectionProjection
    membership: ClassFamilyCollectionMembershipProjection

    @property
    def assignment_name(self) -> str:
        return SingleAssignmentAndValueNameProjection(self.statement).required_name


@dataclass(frozen=True)
class ClassFamilyCollectionMiss:
    """Why one literal of family classes is not the family: the fact it lacks, and
    the input that would supply it when one can."""

    assignment_name: str
    reason: str
    needs: tuple[NeededInput, ...] = ()

    def refusal(self) -> ValueError:
        message = f"`{self.assignment_name}` {self.reason}"
        return NeedsInput(message, *self.needs) if self.needs else ValueError(message)

    @staticmethod
    def nearest(misses: tuple[ClassFamilyCollectionMiss, ...]) -> ClassFamilyCollectionMiss | None:
        """A miss an input can close comes first: it is an offer, the rest are refusals."""

        return next((miss for miss in misses if miss.needs), next(iter(misses), None))


@dataclass(frozen=True)
class ClassFamilyCollectionAuthorityProof:
    """Authority and source context for proving one family projection."""

    reference: ClassAuthorityReferenceProof
    class_index: ClassFamilyIndex
    authority_symbol: str
    authority_declaration: IndexedClass
    descendant_symbols: tuple[str, ...]
    holder_symbol: str | None = None
    """The class whose body holds the literal, when it is one. A member cannot list
    itself in its own body, and at that point the family does not hold it yet, so
    a literal naming every other member is the whole family as the body sees it."""

    def without_holder(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(symbol for symbol in symbols if symbol != self.holder_symbol)

    def outcomes_for_statement(
        self,
        statement: ast.stmt,
    ) -> tuple[ClassFamilyCollectionCandidate | ClassFamilyCollectionMiss, ...]:
        pair = SingleAssignmentAndValueNameProjection(statement).pair
        if pair is None or pair[0] == "__all__":
            return ()
        return tuple(
            self.candidate_for_projection(
                cast(ast.Assign | ast.AnnAssign, statement),
                collection,
            )
            for collection in ClassFamilyCollectionProjection.from_value(
                pair[1],
                self.reference.unavailable_builtin_names,
                self.reference.resolver,
                self.descendant_symbols,
            )
        )

    def candidate_for_statement(
        self,
        statement: ast.stmt,
    ) -> ClassFamilyCollectionCandidate | None:
        return single_item(
            tuple(
                outcome
                for outcome in self.outcomes_for_statement(statement)
                if isinstance(outcome, ClassFamilyCollectionCandidate)
            )
        )

    def misses_for_statement(self, statement: ast.stmt) -> tuple[ClassFamilyCollectionMiss, ...]:
        return tuple(
            outcome
            for outcome in self.outcomes_for_statement(statement)
            if isinstance(outcome, ClassFamilyCollectionMiss)
        )

    @property
    def authority_name(self) -> str:
        return self.authority_symbol.rsplit(".", 1)[-1]

    def membership_miss(
        self,
        assignment_name: str,
        projected_symbols: tuple[str, ...],
    ) -> ClassFamilyCollectionMiss:
        members = frozenset(
            self.without_holder(
                self.descendant_symbols
                if self.authority_declaration.declares_autoregister_meta
                else self.class_index.children_by_symbol.get(self.authority_symbol, ())
            )
        )
        named = frozenset(projected_symbols)
        below = frozenset(
            symbol for symbol in members if not self.declared_before_holder((symbol,))
        )
        if below and named == members - below:
            return ClassFamilyCollectionMiss(
                assignment_name,
                f"names every `{self.authority_name}` member its body can see, but "
                + ", ".join(f"`{symbol.rsplit('.', 1)[-1]}`" for symbol in sorted(below))
                + " is declared below it: move that member above to derive the family, "
                "or pick a base for the ones it names",
                (NeededInput.SOURCE_CHANGE, NeededInput.OWNER_PICK),
            )
        if named < members:
            return ClassFamilyCollectionMiss(
                assignment_name,
                f"names {len(named)} of the {len(members)} `{self.authority_name}` members: "
                f"deriving it needs a declared base between `{self.authority_name}` and those "
                f"{len(named)}, whose class name and one-line docstring a pick supplies",
                (NeededInput.OWNER_PICK,),
            )
        return ClassFamilyCollectionMiss(
            assignment_name,
            f"names `{self.authority_name}` descendants below its direct members, which "
            f"`{self.authority_name}.__subclasses__()` does not hold",
        )

    def candidate_for_projection(
        self,
        statement: ast.Assign | ast.AnnAssign,
        collection: ClassFamilyCollectionProjection,
    ) -> ClassFamilyCollectionCandidate | ClassFamilyCollectionMiss:
        membership = (
            ClassFamilyCollectionMembershipProjection.for_authority_declaration(
                self.authority_declaration.declares_autoregister_meta,
                self.same_members(
                    collection.projected_symbols,
                    self.without_holder(self.descendant_symbols),
                ),
                self.same_members(
                    collection.projected_symbols,
                    self.without_holder(
                        self.class_index.children_by_symbol.get(self.authority_symbol, ())
                    ),
                ),
            )
        )
        assignment_name = SingleAssignmentAndValueNameProjection(statement).required_name
        if membership is None:
            return self.membership_miss(assignment_name, collection.projected_symbols)
        if not self.declared_before_holder(collection.projected_symbols):
            return ClassFamilyCollectionMiss(
                assignment_name,
                "names a member declared below the class that holds it: that body runs "
                "before the member exists, so the family it sees does not hold it",
            )
        runtime_symbols = collection.factory.runtime_member_sequence(
            self.without_holder(
                membership.member_symbols(self.class_index, self.authority_symbol)
            ),
            self.class_index,
        )
        if runtime_symbols is None:
            return ClassFamilyCollectionMiss(
                assignment_name,
                f"is an ordered {collection.factory.value} of members declared in more than one "
                "module, so declaration gives no one order to derive",
            )
        if not collection.factory.preserves_member_sequence(
            collection.projected_symbols,
            runtime_symbols,
        ):
            return ClassFamilyCollectionMiss(
                assignment_name,
                f"lists the members out of declaration order, and a {collection.factory.value} "
                "keeps that order: the order is its own fact",
            )
        return ClassFamilyCollectionCandidate(
            statement=statement,
            collection=collection,
            membership=membership,
        )

    @staticmethod
    def same_members(
        left: tuple[str, ...],
        right: tuple[str, ...],
    ) -> bool:
        return len(left) == len(right) and frozenset(left) == frozenset(right)

    def declared_before_holder(self, symbols: tuple[str, ...]) -> bool:
        """Whether the family view the holder's body evaluates already holds every
        member: each is declared above the holder, or in another module the literal
        imports, so loaded first."""

        holder = self.class_index.class_for(self.holder_symbol) if self.holder_symbol else None
        if holder is None:
            return True
        members = tuple(self.class_index.class_for(symbol) for symbol in symbols)
        return all(
            member is not None
            and (member.file_path != holder.file_path or member.line < holder.line)
            for member in members
        )


@dataclass(frozen=True)
class ClassFamilyCollectionDerivation(SemanticMirrorOperationTargets):
    """Exact source proof for deriving one collection from its class authority."""

    candidate: ClassFamilyCollectionCandidate
    import_source: str | None

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> ClassFamilyCollectionDerivation:
        _authority_id, authority_digest, authority_node = (
            context.target_node_for_rewrite_target(authority_reference)
        )
        if not authority_digest.is_class or not isinstance(
            authority_node, ast.ClassDef
        ):
            raise ValueError("Class-family collection authority must be a class")
        if "." in authority_digest.qualname:
            raise ValueError("Class-family collection authority must be top level")
        projection_id = projection_reference.required_target_id(context.source_index)
        projection_module = context.source_index.target_by_id[projection_id]
        holder_node = context.ast_target_nodes_by_id.get(projection_id)
        if projection_module.is_module:
            holder_node = None
        elif not (
            projection_module.is_class
            and "." not in projection_module.qualname
            and isinstance(holder_node, ast.ClassDef)
        ):
            raise ValueError(
                "Class-family collection projection must target a module or a top-level class"
            )
        authority = ResolvedClassTarget(authority_digest, authority_node)
        class_index = context.required_class_family_index
        reference_proof = ClassAuthorityReferenceProof.from_context(
            context,
            authority,
            projection_module.file_path,
        )
        authority_symbol = reference_proof.authority_symbol
        authority_declaration = class_index.class_for(authority_symbol)
        if authority_declaration is None:
            raise ValueError("Class-family authority declaration is unavailable")
        descendant_symbols = class_index.descendant_symbols(authority_symbol)
        if not descendant_symbols:
            raise ValueError("Class-family authority has no indexed descendants")
        proof = ClassFamilyCollectionAuthorityProof(
            reference=reference_proof,
            class_index=class_index,
            authority_symbol=authority_symbol,
            authority_declaration=authority_declaration,
            descendant_symbols=descendant_symbols,
            holder_symbol=(
                None
                if holder_node is None
                else reference_proof.resolver.symbol_for_reference(
                    ast.Name(id=holder_node.name, ctx=ast.Load())
                )
            ),
        )
        statements = (
            reference_proof.projection_module.module.body
            if holder_node is None
            else holder_node.body
        )
        candidates = tuple(
            candidate
            for statement in statements
            if (candidate := proof.candidate_for_statement(statement)) is not None
        )
        if not candidates:
            miss = ClassFamilyCollectionMiss.nearest(
                tuple(miss for statement in statements for miss in proof.misses_for_statement(statement))
            )
            if miss is not None:
                raise miss.refusal()
            raise ValueError(
                f"`{projection_module.qualname or projection_module.file_path}` holds no "
                f"builtin tuple, list, set or frozenset literal of `{proof.authority_name}` "
                "classes: a constructed value or text it mirrors is each element's own fact"
            )
        if len(candidates) != 1:
            raise ValueError(
                "Class-family authority and projection module must expose exactly "
                f"one complete literal collection; found {len(candidates)}"
            )
        PositionalCollectionReads.require_none(context, candidates[0])
        return cls(
            authority=authority,
            projection_module=projection_module,
            candidate=candidates[0],
            import_source=reference_proof.required_import_source(context),
        )

    def replacement_source(self) -> str:
        candidate = self.candidate
        value_source = candidate.collection.value_source(
            self.authority.name,
            candidate.membership,
        )
        indentation = " " * candidate.statement.col_offset
        if isinstance(candidate.statement, ast.AnnAssign):
            return (
                f"{indentation}{candidate.assignment_name}: "
                f"{ast.unparse(candidate.statement.annotation)} = {value_source}"
            )
        return f"{indentation}{candidate.assignment_name} = {value_source}"


class PositionalCollectionReads:
    """A collection read by position gives each position a role: `row_tones[alternate]`
    picks the plain or the alternate tone. Equal to the family today, it still
    declares which member plays which role, a fact the family does not hold, so a
    new member would silently shift or overflow it (paper1 Thm 4.8)."""

    @staticmethod
    def require_none(
        context: CodemodSelectorContext, candidate: ClassFamilyCollectionCandidate
    ) -> None:
        name = candidate.assignment_name
        for path, module in context.module_nodes_by_file_path.items():
            for node in ast.walk(module):
                match node:
                    case ast.Subscript(
                        value=ast.Attribute(attr=read) | ast.Name(id=read), slice=index
                    ) if read == name and not isinstance(index, ast.Slice):
                        raise ValueError(
                            f"`{name}` is read by position at {path}:{node.lineno}: "
                            "each position is a role the family does not declare"
                        )


@dataclass(frozen=True, kw_only=True)
class DeriveClassFamilyCollectionOperation(SourceDerivedAuthorityProjectionOperation):
    """Derive one complete collection projection from its class authority."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        derivation = self.required_derivation(snapshot)
        edits: list[NominalSourceEdit] = []
        if derivation.import_source is not None:
            edits.extend(
                self.required_import_mutations(
                    derivation.projection_path,
                    import_source=derivation.import_source,
                    default_rationale="Import the class-family authority.",
                )
            )
        statement = derivation.candidate.statement
        edits.append(
            SourceSpanReplacement(
                file_path=derivation.projection_path,
                span=SourceLineSpan(statement.lineno, statement.end_lineno or statement.lineno),
                replacement_lines=SourceTargetEditor.source_lines(
                    derivation.replacement_source()
                ),
                rationale=self.rationale_text(
                    f"Derive {derivation.candidate.assignment_name!r} from "
                    f"{derivation.authority.name!r}."
                ),
            )
        )
        return tuple(edits)

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> ClassFamilyCollectionDerivation:
        return ClassFamilyCollectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class ClassFamilyCollectionSemanticMirrorRecipeBuilder(
    RegistrationSemanticMirrorRecipeBuilder,
    ClassFamilyAuthorityConcept,
):
    """Build a source-derived class-family projection recipe."""

    @cached_property
    def targets(self) -> SemanticMirrorOperationTargets | None:
        return SemanticMirrorOperationTargets.from_finding(self, self.finding)

    @cached_property
    def candidate_operation(self) -> DeriveClassFamilyCollectionOperation | None:
        if self.targets is None:
            return None
        return DeriveClassFamilyCollectionOperation(
            target=SourceRewriteTarget(
                target_id=self.targets.authority.target.target_id
            ),
            projection_target=SourceRewriteTarget(target_id=self.projection_target_id),
        )

    @cached_property
    def projection_target_id(self) -> str:
        """The top-level class whose body holds the finding's literal, else its module."""

        assert self.targets is not None
        module_target = self.targets.projection_module
        projection = self.finding.projection_evidence
        module = self.module_nodes_by_file_path.get(module_target.file_path)
        holder = next(
            (
                statement
                for statement in (module.body if module is not None else ())
                if isinstance(statement, ast.ClassDef)
                and projection is not None
                and statement.lineno < projection.line <= (statement.end_lineno or statement.lineno)
            ),
            None,
        )
        if holder is None:
            return module_target.target_id
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(module_target.file_path,),
            qualnames=(holder.name,),
        ).target_ids(self)
        return target_ids[0] if len(target_ids) == 1 else module_target.target_id

    @cached_property
    def proven_operation(self) -> DeriveClassFamilyCollectionOperation | None:
        operation = self.candidate_operation
        if operation is None:
            return None
        try:
            operation.required_derivation(self)
        except ValueError:
            return None
        return operation

    def recipe(self) -> RefactorRecipe | None:
        operation = self.proven_operation
        if operation is None or self.targets is None:
            return None
        return (
            RefactorRecipe(
                recipe_id=(f"{self.finding.stable_id}-derive-class-family-collection"),
                reason="Derive subclass collection from the class-family authority.",
            )
            .with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    self.targets.authority.target,
                    authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                )
            )
            .with_operation(operation)
        )

    @cached_property
    def refusal(self) -> ValueError | None:
        operation = self.candidate_operation
        if operation is None:
            return ValueError(
                "semantic mirror finding does not resolve one class authority and "
                "one projection module"
            )
        try:
            operation.required_derivation(self)
        except ValueError as error:
            return error
        return None

    def rejection_reason(self) -> str:
        return "class-family collection derivation is available" if self.refusal is None else str(self.refusal)

    def rejection_needs(self) -> tuple[NeededInput, ...]:
        return self.refusal.needs if isinstance(self.refusal, NeedsInput) else ()
