from __future__ import annotations

import ast
import keyword as keyword_module
import re
from dataclasses import dataclass
from functools import cached_property
from typing import cast

from ..assignment_projection import SingleAssignmentAndValueNameProjection
from ..codemod import (
    ClassAuthorityReferenceProof,
    ClassDeclarationPromotionClass,
    ClassFamilyCollectionFactory,
    MappingSemanticMirrorRecipeBuilder,
    SemanticMirrorOperationTargets,
    SourceDerivedAuthorityProjectionOperation,
)
from ..codemod_authority_claims import AstTargetAuthorityClaim
from ..codemod_declaration_source import ClassBodySourceAuthority
from ..codemod_runtime import (
    CodemodSourceSnapshot,
    RefactorRecipe,
)
from ..codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from ..codemod_selector_models import SourceRewriteTarget
from ..codemod_source_edits import (
    NominalSourceEdit,
    SourceInsertion,
    SourceLineSpan,
    SourceSpanReplacement,
    SourceTargetEditor,
)
from ..lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from ..models import RefactorFinding
from ..refactor_concepts import DerivedProjectionConcept
from ..semantic_descent import SemanticAuthorityKind
from ..source_index import AstTargetDigest


@dataclass(frozen=True)
class EnumStringMemberDeclaration:
    """One direct enum member with a source-declared string value."""

    name: str
    value: str

    @classmethod
    def from_statement(
        cls, statement: ast.stmt
    ) -> EnumStringMemberDeclaration | None:
        pair = SingleAssignmentAndValueNameProjection(statement).pair
        if pair is None:
            return None
        name, value = pair
        if (
            name.startswith("_")
            or not isinstance(value, ast.Constant)
            or not isinstance(value.value, str)
        ):
            return None
        return cls(name=name, value=value.value)


@dataclass(frozen=True)
class EnumStringAuthority:
    """Exact enum class and its unambiguous string-valued members."""

    target: ResolvedClassTarget
    members: tuple[EnumStringMemberDeclaration, ...]

    @classmethod
    def from_target(cls, target: ResolvedClassTarget) -> EnumStringAuthority:
        if not ClassDeclarationPromotionClass(target.node).is_enum_class:
            raise ValueError("Enum subset authority must be an enum class")
        members = tuple(
            member
            for statement in target.node.body
            if (member := EnumStringMemberDeclaration.from_statement(statement))
            is not None
        )
        if not members:
            raise ValueError("Enum subset authority has no string-valued members")
        member_values = tuple(member.value for member in members)
        if len(frozenset(member_values)) != len(member_values):
            raise ValueError("Enum subset authority has aliased string values")
        return cls(target=target, members=members)

    def members_for_values(
        self,
        values: frozenset[str],
    ) -> tuple[EnumStringMemberDeclaration, ...] | None:
        selected = tuple(member for member in self.members if member.value in values)
        if not selected or frozenset(member.value for member in selected) != values:
            return None
        return selected


@dataclass(frozen=True)
class EnumSubsetProjection:
    """One literal enum-value subset to derive from its enum authority."""

    statement: ast.Assign | ast.AnnAssign
    members: tuple[EnumStringMemberDeclaration, ...]

    @property
    def assignment_name(self) -> str:
        return SingleAssignmentAndValueNameProjection(self.statement).required_name

    @property
    def accessor_name(self) -> str:
        return self.accessor_name_for_assignment(self.assignment_name)

    @classmethod
    def from_statement(
        cls,
        statement: ast.stmt,
        authority: EnumStringAuthority,
        reference: ClassAuthorityReferenceProof,
    ) -> EnumSubsetProjection | None:
        pair = SingleAssignmentAndValueNameProjection(statement).pair
        if pair is None or pair[0] == "__all__":
            return None
        _assignment_name, value = pair
        values = cls.frozenset_values(value, reference.unavailable_builtin_names)
        if values is None:
            return None
        members = authority.members_for_values(values)
        if members is None:
            return None
        return cls(
            statement=cast(ast.Assign | ast.AnnAssign, statement),
            members=members,
        )

    @staticmethod
    def frozenset_values(
        value: ast.AST,
        unavailable_builtin_names: frozenset[str],
    ) -> frozenset[str] | None:
        elements = ClassFamilyCollectionFactory.FROZENSET.elements(
            value, unavailable_builtin_names
        )
        if elements is None:
            return None
        values = frozenset(
            element.value
            for element in elements
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        )
        if not values or len(values) != len(elements):
            return None
        return values

    @staticmethod
    def accessor_name_for_assignment(assignment_name: str) -> str:
        identifier = re.sub(
            r"[^0-9A-Za-z_]+",
            "_",
            assignment_name.strip("_").lower(),
        )
        identifier = re.sub(r"_+", "_", identifier).strip("_")
        if not identifier:
            return "derived_values"
        if identifier[0].isdigit() or keyword_module.iskeyword(identifier):
            return f"derived_{identifier}"
        return identifier


@dataclass(frozen=True)
class EnumSubsetDerivation:
    """Current-source proof for one enum-owned subset projection."""

    authority: EnumStringAuthority
    projection_module: AstTargetDigest
    projection: EnumSubsetProjection
    import_source: str | None

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> EnumSubsetDerivation:
        _authority_id, authority_digest, authority_node = (
            context.target_node_for_rewrite_target(authority_reference)
        )
        if not authority_digest.is_class or not isinstance(
            authority_node, ast.ClassDef
        ):
            raise ValueError("Enum subset authority must target a class")
        if "." in authority_digest.qualname:
            raise ValueError("Enum subset authority must be top level")
        projection_id = projection_reference.required_target_id(context.source_index)
        projection_module = context.source_index.target_by_id[projection_id]
        if not projection_module.is_module:
            raise ValueError("Enum subset projection must target a module")
        resolved_authority = ResolvedClassTarget(authority_digest, authority_node)
        authority = EnumStringAuthority.from_target(resolved_authority)
        authority_reference_proof = ClassAuthorityReferenceProof.from_context(
            context,
            resolved_authority,
            resolved_authority.file_path,
        )
        authority_reference_proof.required_import_source(context)
        if ClassFamilyCollectionFactory.FROZENSET.is_shadowed(
            authority_reference_proof.unavailable_builtin_names
        ):
            raise ValueError("Enum authority shadows the frozenset constructor")
        projection_reference_proof = ClassAuthorityReferenceProof.from_context(
            context,
            resolved_authority,
            projection_module.file_path,
        )
        projections = tuple(
            projection
            for statement in projection_reference_proof.projection_module.module.body
            if (
                projection := EnumSubsetProjection.from_statement(
                    statement,
                    authority,
                    projection_reference_proof,
                )
            )
            is not None
        )
        if len(projections) != 1:
            raise ValueError(
                "Enum authority and projection module must expose exactly one "
                f"literal frozenset subset; found {len(projections)}"
            )
        projection = projections[0]
        if projection.accessor_name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            authority_node.body
        ):
            raise ValueError(
                f"Enum authority already binds {projection.accessor_name!r}"
            )
        return cls(
            authority=authority,
            projection_module=projection_module,
            projection=projection,
            import_source=projection_reference_proof.required_import_source(context),
        )

    @property
    def projection_path(self) -> str:
        return self.projection_module.file_path

    def method_source(self, indentation: str) -> str:
        member_lines = "".join(
            f"{indentation}        cls.{member.name}.value,\n"
            for member in self.projection.members
        )
        return (
            "\n"
            f"{indentation}@classmethod\n"
            f"{indentation}def {self.projection.accessor_name}("
            "cls) -> frozenset[str]:\n"
            f"{indentation}    return frozenset((\n"
            f"{member_lines}"
            f"{indentation}    ))\n"
        )

    def assignment_source(self) -> str:
        projection = self.projection
        value_source = f"{self.authority.target.name}.{projection.accessor_name}()"
        if isinstance(projection.statement, ast.AnnAssign):
            return (
                f"{projection.assignment_name}: "
                f"{ast.unparse(projection.statement.annotation)} = {value_source}"
            )
        return f"{projection.assignment_name} = {value_source}"


@dataclass(frozen=True, kw_only=True)
class DeriveEnumSubsetOperation(SourceDerivedAuthorityProjectionOperation):
    """Move one literal enum-value subset behind its enum authority."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        derivation = self.required_derivation(snapshot)
        authority_target = derivation.authority.target
        body_authority = ClassBodySourceAuthority(
            authority_target.node,
            snapshot.sources_by_file_path[authority_target.file_path],
        )
        edits: list[NominalSourceEdit] = [
            SourceInsertion(
                file_path=authority_target.file_path,
                insertion_line=(authority_target.node.end_lineno or 0) + 1,
                inserted_lines=SourceTargetEditor.source_lines(
                    derivation.method_source(body_authority.indentation)
                ),
                rationale=self.rationale_text(
                    f"Declare {derivation.projection.accessor_name!r} on "
                    f"{authority_target.name!r}."
                ),
            )
        ]
        if derivation.import_source is not None:
            edits.extend(
                self.required_import_mutations(
                    derivation.projection_path,
                    import_source=derivation.import_source,
                    default_rationale="Import the enum subset authority.",
                )
            )
        statement = derivation.projection.statement
        edits.append(
            SourceSpanReplacement(
                file_path=derivation.projection_path,
                span=SourceLineSpan(statement.lineno, statement.end_lineno or statement.lineno),
                replacement_lines=SourceTargetEditor.source_lines(
                    derivation.assignment_source()
                ),
                rationale=self.rationale_text(
                    f"Derive {derivation.projection.assignment_name!r} from "
                    f"{authority_target.name!r}."
                ),
            )
        )
        return tuple(edits)

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> EnumSubsetDerivation:
        return EnumSubsetDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class EnumSubsetSemanticMirrorRecipeBuilder(
    MappingSemanticMirrorRecipeBuilder,
    DerivedProjectionConcept,
):
    """Build a source-derived enum subset recipe."""

    finding: RefactorFinding

    @cached_property
    def targets(self) -> SemanticMirrorOperationTargets | None:
        targets = SemanticMirrorOperationTargets.from_finding(self, self.finding)
        if (
            targets is None
            or not ClassDeclarationPromotionClass(targets.authority.node).is_enum_class
        ):
            return None
        return targets

    @cached_property
    def candidate_operation(self) -> DeriveEnumSubsetOperation | None:
        if self.targets is None:
            return None
        return DeriveEnumSubsetOperation(
            target=SourceRewriteTarget(
                target_id=self.targets.authority.target.target_id
            ),
            projection_target=SourceRewriteTarget(
                target_id=self.targets.projection_module.target_id
            ),
        )

    def is_applicable(self) -> bool:
        return self.candidate_operation is not None

    @cached_property
    def proven_operation(self) -> DeriveEnumSubsetOperation | None:
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
                recipe_id=f"{self.finding.stable_id}-derive-enum-subset-mapping",
                reason="Move enum subset projection behind the enum authority.",
            )
            .with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    self.targets.authority.target,
                    authority_kind=SemanticAuthorityKind.ENUM,
                )
            )
            .with_operation(operation)
        )

    def rejection_reason(self) -> str:
        operation = self.candidate_operation
        if operation is None:
            return (
                "semantic mirror finding does not resolve one enum authority and "
                "one projection module"
            )
        try:
            operation.required_derivation(self)
        except ValueError as error:
            return str(error)
        return "enum subset projection has an executable authority recipe"
