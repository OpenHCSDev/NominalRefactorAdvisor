from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
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
from nominal_refactor_advisor.fielded_enum import FieldedStrEnum
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
from nominal_refactor_advisor.semantic_match import single_item


class ClassFamilyCollectionFactory(FieldedStrEnum):
    """Collection syntax and ordering semantics for one derived family view."""

    _literal_node_type: type[ast.Tuple | ast.List | ast.Set] | None
    _preserves_order: bool

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
    del family_symbols
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


class ClassFamilyCollectionElementProjection(FieldedStrEnum):
    """How one collection projection references a class-family member."""

    _symbol_projector: Callable[[tuple[ast.expr, ...], ModuleClassReferenceResolver, tuple[str, ...]], tuple[str, ...] | None]
    _value_source_builder: Callable[[str, str], str]

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


class ClassFamilyCollectionMembershipProjection(FieldedStrEnum):
    """Runtime member query selected from the nominal authority declaration."""

    _authority_matcher: Callable[[bool, bool, bool], bool]
    _member_symbol_projector: Callable[[ClassFamilyIndex, str], tuple[str, ...]]
    _value_source_builder: Callable[[str], str]

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
class ClassFamilyCollectionAuthorityProof:
    """Authority and source context for proving one family projection."""

    reference: ClassAuthorityReferenceProof
    class_index: ClassFamilyIndex
    authority_symbol: str
    authority_declaration: IndexedClass
    descendant_symbols: tuple[str, ...]

    def candidate_for_statement(
        self,
        statement: ast.stmt,
    ) -> ClassFamilyCollectionCandidate | None:
        pair = SingleAssignmentAndValueNameProjection(statement).pair
        if pair is None or pair[0] == "__all__":
            return None
        assignment_name, value = pair
        return single_item(
            tuple(
                candidate
                for collection in ClassFamilyCollectionProjection.from_value(
                    value,
                    self.reference.unavailable_builtin_names,
                    self.reference.resolver,
                    self.descendant_symbols,
                )
                if (
                    candidate := self.candidate_for_projection(
                        cast(ast.Assign | ast.AnnAssign, statement),
                        collection,
                    )
                )
                is not None
            )
        )

    def candidate_for_projection(
        self,
        statement: ast.Assign | ast.AnnAssign,
        collection: ClassFamilyCollectionProjection,
    ) -> ClassFamilyCollectionCandidate | None:
        membership = (
            ClassFamilyCollectionMembershipProjection.for_authority_declaration(
                self.authority_declaration.declares_autoregister_meta,
                self.same_members(
                    collection.projected_symbols,
                    self.descendant_symbols,
                ),
                self.same_members(
                    collection.projected_symbols,
                    self.class_index.children_by_symbol.get(self.authority_symbol, ()),
                ),
            )
        )
        if membership is None:
            return None
        runtime_symbols = collection.factory.runtime_member_sequence(
            membership.member_symbols(self.class_index, self.authority_symbol),
            self.class_index,
        )
        if runtime_symbols is None or not collection.factory.preserves_member_sequence(
            collection.projected_symbols,
            runtime_symbols,
        ):
            return None
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
        if not projection_module.is_module:
            raise ValueError("Class-family collection projection must target a module")
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
        )
        candidates = tuple(
            candidate
            for statement in reference_proof.projection_module.module.body
            if (candidate := proof.candidate_for_statement(statement)) is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "Class-family authority and projection module must expose exactly "
                f"one complete literal collection; found {len(candidates)}"
            )
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
        if isinstance(candidate.statement, ast.AnnAssign):
            return (
                f"{candidate.assignment_name}: "
                f"{ast.unparse(candidate.statement.annotation)} = {value_source}"
            )
        return f"{candidate.assignment_name} = {value_source}"


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
            projection_target=SourceRewriteTarget(
                target_id=self.targets.projection_module.target_id
            ),
        )

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

    def rejection_reason(self) -> str:
        operation = self.candidate_operation
        if operation is None:
            return (
                "semantic mirror finding does not resolve one class authority and "
                "one projection module"
            )
        try:
            operation.required_derivation(self)
        except ValueError as error:
            return str(error)
        return "class-family collection derivation is available"
