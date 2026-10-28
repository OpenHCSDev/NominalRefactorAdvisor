"""Executable recipes that delete a distinction the code then treats identically.

Each operation targets the scope that owns a branch (a function, a method or
the module) and the branch's line, and re-derives its edits from current
source: the branch must still be there, and its deleted tests must still be
free of effects.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeVar, cast

from .codemod import (
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
    RefactorRecipeOperation,
)
from .codemod_imports import ModuleImportMutation
from .codemod_foreign_family_source import RewrittenLineFit
from .codemod_payload import (
    RequiredIntegerPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_reproof import RepositorySourceReprovedOperation, SourceReprovedOperation
from .codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewriteTarget
from .ast_tools import ParsedModule
from .class_index import ModuleNominalBindingAuthority
from .codemod_source_edits import (
    NominalSourceEdit,
    SourceSpanDeletion,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .enum_semantics import PythonEnumBaseKind
from .codemod_target_selectors import SourceIndexTargetSelector
from .declaration_authority_rename import (
    DeclarationAuthorityModuleRedirectProof,
    DeclarationAuthorityModuleRenameProof,
    DeclarationAuthorityRenameProof,
    DeclarationRedirectProof,
)
from .declared_root import DeclaredRoot
from .false_distinction_observations import (
    EmptySubclass,
    NameUse,
    SiblingTestModules,
    EnumClassUse,
    EnumMemberIndex,
    EnumMemberUse,
    EnumMembers,
    EnumUseKind,
    MemberGroup,
    NameUseKind,
    enum_uses,
    IdenticalArms,
    Scope,
    ScopeNode,
    UndistinguishedTypeArms,
)
from .models import RefactorFinding
from .refactor_concepts import FalseDistinctionErasureConcept
from .source_index import AstTargetNodeKind

FoundT = TypeVar("FoundT")


@dataclass(frozen=True, kw_only=True)
class ScopeBranchOperation(SourceReprovedOperation, ABC):
    """Rewrite the branch at `line` of the scope the target names."""

    line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())

    @abstractmethod
    def replacements(
        self, scope: Scope, geometry: SourceTextGeometry, fit: RewrittenLineFit
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """The edits for the one branch at `line`; a ValueError when it is gone or unsafe."""

        raise NotImplementedError

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        geometry = SourceTextGeometry(snapshot.sources_by_file_path[target.file_path])
        fit = RewrittenLineFit(
            geometry, DeclaredRoot.line_length_for(Path(target.file_path))
        )
        return geometry.physical_edits(
            file_path=target.file_path,
            replacements=self.replacements(
                Scope(cast(ScopeNode, node).body), geometry, fit
            ),
            rationale=self.rationale
            or f"Delete the distinction at line {self.line} that the code treats identically.",
        )

    def only(self, found: list[FoundT], what: str) -> FoundT:
        """The one `what` at `line`; current source must still hold exactly one."""

        if len(found) != 1:
            raise ValueError(f"{len(found)} {what} at line {self.line} of the target scope")
        return found[0]


@dataclass(frozen=True, kw_only=True)
class CollapseIdenticalArmsOperation(ScopeBranchOperation):
    """Replace the branch at `line`, whose arms are all the same code, with that code."""

    def replacements(
        self, scope: Scope, geometry: SourceTextGeometry, fit: RewrittenLineFit
    ) -> tuple[SourceTextSpanReplacement, ...]:
        del fit
        found = [arms for arms in IdenticalArms.in_scope(scope) if arms.node.lineno == self.line]
        return (self.only(found, "branch(es) with identical arms").collapse(geometry),)


@dataclass(frozen=True, kw_only=True)
class MergeTypeTestArmsOperation(ScopeBranchOperation):
    """Merge `isinstance` rungs at `line` that run the same code into one test,
    or delete trailing rungs that run the `else` body."""

    def replacements(
        self, scope: Scope, geometry: SourceTextGeometry, fit: RewrittenLineFit
    ) -> tuple[SourceTextSpanReplacement, ...]:
        found = [
            arms
            for arms in UndistinguishedTypeArms.in_scope(scope)
            if arms.line == self.line
        ]
        return self.only(found, "undistinguished type test group(s)").collapse(
            geometry, fit
        )


class ScopeBranchFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    FalseDistinctionErasureConcept,
    ABC,
):
    """One operation on the branch the finding's primary witness names, in its scope."""

    operation_type: ClassVar[type[ScopeBranchOperation]]
    recipe_reason: ClassVar[str]

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation(
                "deleting a false distinction requires source context and a primary witness"
            )
        selector = (
            SourceIndexTargetSelector(
                node_kinds=(AstTargetNodeKind.MODULE,), file_paths=(evidence.file_path,)
            )
            if evidence.symbol == "<module>"
            else SourceIndexTargetSelector.for_function_or_method(
                evidence.file_path, evidence.symbol
            )
        )
        target_ids = selector.target_ids(context)
        if len(target_ids) != 1:
            return self.rejected_evaluation(
                f"`{evidence.symbol}` is not one function, method or module"
            )
        operation = self.operation_type(
            target=SourceRewriteTarget(target_id=target_ids[0]), line=evidence.line
        )
        try:
            operation.source_edits(context)
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=self.recipe_reason,
            ).with_operation(operation)
        )


class IdenticalArmsFindingRecipeSynthesizer(ScopeBranchFindingRecipeSynthesizer, ABC):
    operation_type = CollapseIdenticalArmsOperation
    recipe_reason = "Every arm runs the same code; the branch becomes that code."


class UndistinguishedTypeTestFindingRecipeSynthesizer(
    ScopeBranchFindingRecipeSynthesizer, ABC
):
    operation_type = MergeTypeTestArmsOperation
    recipe_reason = (
        "Type tests whose arms run the same code become one test; "
        "rungs that run the else body are deleted."
    )


@dataclass(frozen=True, kw_only=True)
class RedirectEmptySubclassOperation(RepositorySourceReprovedOperation):
    """Delete an empty subclass and point every consumer in the repository at its base.

    The repository proof is the declaration-rename proof with the base's name
    as the new name (`DeclarationRedirectProof`): imports, direct and
    qualified references are the rename's surfaces, and a string or comment
    spelling the class refuses. Every reference must construct or raise the
    class; any other use tells it apart from its base, so the operation refuses
    and names the site.
    """

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        found = list(EmptySubclass.decoded(node))
        if len(found) != 1:
            raise ValueError(
                f"`{target.qualname}` is not one undecorated class with a single named base and no body"
            )
        (subclass,) = found
        proof = DeclarationRedirectProof.require_redirect(
            snapshot.parsed_modules, target, node, bound_name=subclass.base.id
        )
        tests = SiblingTestModules.beside(snapshot.parsed_modules)
        refusals = (*self.refusals(proof, subclass), *self.test_refusals(tests, subclass))
        if refusals:
            raise ValueError("; ".join(dict.fromkeys(refusals)))
        rationale = self.rationale or (
            f"Delete `{subclass.name}`: it differs from `{subclass.base.id}` only by name."
        )
        return (
            SourceSpanDeletion.for_statement(snapshot, target, rationale=rationale),
            *(
                edit
                for module in proof.modules
                for edit in self.module_edits(module, subclass, rationale)
            ),
        )

    @staticmethod
    def refusals(
        proof: DeclarationAuthorityRenameProof, subclass: EmptySubclass
    ) -> Iterator[str]:
        for module in proof.modules:
            parents = {
                id(child): parent
                for parent in ast.walk(module.module.module)
                for child in ast.iter_child_nodes(parent)
            }
            for reference in (*module.direct_references, *module.qualified_references):
                kind = NameUseKind.of(reference, parents)
                if kind not in NameUseKind.only_builds():
                    yield (
                        f"{module.module.file_path}:{reference.lineno} uses "
                        f"`{subclass.name}` as more than a constructor"
                    )
            for annotation in module.annotation_references:
                yield (
                    f"{module.module.file_path}:{annotation.literal.lineno} spells "
                    f"`{subclass.name}` in a string annotation"
                )
            for export in (*module.public_exports, *module.stable_public_exports):
                yield (
                    f"{module.module.file_path}:{export.literal.lineno} spells "
                    f"`{subclass.name}` in a public export string"
                )

    @staticmethod
    def test_refusals(tests: SiblingTestModules, subclass: EmptySubclass) -> Iterator[str]:
        """A test spelling the class: the redirect rewrites scanned modules only."""

        for test in tests.modules:
            for name, kind, line in NameUse.spellings(test):
                if name == subclass.name:
                    yield (
                        f"{test.file_path}:{line} spells `{name}` in a test ({kind}); "
                        "the redirect rewrites scanned modules only"
                    )

    @staticmethod
    def module_edits(
        module: DeclarationAuthorityModuleRenameProof,
        subclass: EmptySubclass,
        rationale: str,
    ) -> Iterator[NominalSourceEdit]:
        geometry = SourceTextGeometry(module.module.source)
        file_path = module.module.file_path
        base = subclass.base.id
        replacements = [
            *(
                module._name_replacement(geometry.required_node_offsets(reference), base)
                for reference in module.direct_references
            ),
            *(
                module._name_replacement(
                    module._qualified_name_offsets(geometry, reference), base
                )
                for reference in module.qualified_references
            ),
        ]
        match module:
            case DeclarationAuthorityModuleRedirectProof(redirected_imports_are_removed=True):
                for reference in module.imports:
                    yield ModuleImportMutation.remove_names(
                        file_path=file_path,
                        module_name="." * reference.statement.level
                        + (reference.statement.module or ""),
                        names=(reference.alias.name,),
                        rationale=rationale,
                    )
            case _:
                replacements.extend(
                    module._import_replacement(geometry, reference, base)
                    for reference in module.imports
                )
        if replacements:
            yield from geometry.physical_edits(
                file_path=file_path, replacements=replacements, rationale=rationale
            )


class ClassTargetFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    FalseDistinctionErasureConcept,
    ABC,
):
    """One operation on the top-level class the finding's primary witness names."""

    recipe_reason: ClassVar[str]

    @abstractmethod
    def operation_for(
        self, target: SourceRewriteTarget, context: CodemodSelectorContext
    ) -> RefactorRecipeOperation:
        """The operation on `target`; a ValueError when current source admits none."""

        raise NotImplementedError

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation(
                "deleting a false distinction requires source context and a primary witness"
            )
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(evidence.file_path,),
            qualnames=(evidence.symbol,),
        ).target_ids(context)
        if len(target_ids) != 1:
            return self.rejected_evaluation(f"`{evidence.symbol}` is not one class")
        try:
            operation = self.operation_for(SourceRewriteTarget(target_id=target_ids[0]), context)
            operation.source_edits(context)
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=self.recipe_reason,
            ).with_operation(operation)
        )


class EmptySubclassFindingRecipeSynthesizer(ClassTargetFindingRecipeSynthesizer, ABC):
    recipe_reason = "The subclass differs from its base only by name; its consumers use the base."

    def operation_for(
        self, target: SourceRewriteTarget, context: CodemodSelectorContext
    ) -> RefactorRecipeOperation:
        del context
        return RedirectEmptySubclassOperation(target=target)


@dataclass(frozen=True)
class EnumFoldProof:
    """Every spelling of one plain Enum in the repository, read for a fold."""

    name: str
    members: EnumMembers
    uses: tuple[tuple[ParsedModule, list[EnumMemberUse], list[EnumClassUse], list[MemberGroup]], ...]
    tests: tuple[tuple[ParsedModule, list[EnumMemberUse], list[EnumClassUse], list[MemberGroup]], ...]

    @classmethod
    def of(cls, snapshot: CodemodSourceSnapshot, node: ast.AST, file_path: str) -> EnumFoldProof:
        found = list(EnumMembers.decoded(node))
        if len(found) != 1:
            raise ValueError("The target is not a class whose body only binds names to constants")
        (members,) = found
        module = snapshot.parsed_module_for_source_path(file_path)
        match members.node.bases:
            case [spelled] if cls.binds_plain_enum(module, spelled):
                pass
            case _:
                raise ValueError(f"`{members.node.name}` is not a plain Enum")
        name = members.node.name
        return cls(
            name,
            members,
            tuple(
                (parsed, *enum_uses(parsed.module, name.__eq__))
                for parsed in snapshot.parsed_modules
            ),
            tuple(
                (test, *enum_uses(test.module, name.__eq__))
                for test in SiblingTestModules.beside(snapshot.parsed_modules).modules
            ),
        )

    @staticmethod
    def binds_plain_enum(module: ParsedModule, spelled: ast.expr) -> bool:
        snapshot = ModuleNominalBindingAuthority(module).snapshot_before()
        match spelled:
            case ast.Name(id=bound):
                binding = snapshot.binding_for(bound)
                return (
                    binding is not None
                    and binding.qualified_name == PythonEnumBaseKind.ENUM.qualified_name
                )
            case ast.Attribute(value=ast.Name(id=bound), attr=attribute):
                binding = snapshot.binding_for(bound)
                return (
                    binding is not None
                    and f"{binding.qualified_name}.{attribute}"
                    == PythonEnumBaseKind.ENUM.qualified_name
                )
        return False

    def folds(self) -> tuple[tuple[str, ...], ...]:
        return EnumMemberIndex.undistinguished(
            [
                use
                for _parsed, member_uses, _classes, _groups in (*self.uses, *self.tests)
                for use in member_uses
            ]
        )

    def refusals(self, folded: tuple[str, ...]) -> Iterator[str]:
        for parsed, member_uses, class_uses, _groups in self.uses:
            for class_use in class_uses:
                if class_use.kind is EnumUseKind.OTHER:
                    yield (
                        f"{parsed.file_path}:{class_use.line} uses `{self.name}` other than "
                        "by a member or as a type (calling, indexing or iterating it reads members by value, name or order)"
                    )
            for use in member_uses:
                yield from use.refusals(parsed.file_path, folded)
        for test, member_uses, class_uses, _groups in self.tests:
            for class_use in class_uses:
                if class_use.kind is EnumUseKind.OTHER:
                    yield (
                        f"{test.file_path}:{class_use.line} uses `{self.name}` other than "
                        "by a member or as a type in a test"
                    )
            for use in member_uses:
                if use.member in folded:
                    yield (
                        f"{test.file_path}:{use.line} spells `{self.name}.{use.member}` in a "
                        "test; the fold rewrites scanned modules only"
                    )


@dataclass(frozen=True, kw_only=True)
class FoldEnumMembersOperation(RepositorySourceReprovedOperation):
    """Fold plain-Enum members that every spelling names together into the first of them.

    The folded members' declarations are deleted and each grouped test drops
    their spellings: `x in (E.A, E.B)` keeps `E.A`, `case E.A | E.B` becomes
    `case E.A`, `x is E.A or x is E.B` becomes `x is E.A`. The enum class must
    be a plain `Enum` with literal values, spelled only by member accesses
    and type uses, and every spelling of a folded member must be grouped with
    all of them.
    """

    members: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        proof = EnumFoldProof.of(snapshot, node, target.file_path)
        statements = proof.members.statements_by_member
        if len(self.members) < 2 or not set(self.members) <= set(statements):
            raise ValueError(f"`{proof.name}` does not declare the members {self.members!r}")
        kept, *folded = sorted(self.members, key=list(statements).index)
        refusals = tuple(proof.refusals(tuple(self.members)))
        if refusals:
            raise ValueError("; ".join(dict.fromkeys(refusals)))
        rationale = self.rationale or (
            f"Fold {', '.join(f'`{proof.name}.{name}`' for name in folded)} into "
            f"`{proof.name}.{kept}`: nothing tells them apart."
        )
        source = snapshot.sources_by_file_path[target.file_path]
        return (
            *(
                SourceSpanDeletion.for_statement_node(
                    file_path=target.file_path,
                    source=source,
                    statement=statements[name],
                    rationale=rationale,
                )
                for name in folded
            ),
            *(
                edit
                for parsed, _members, _classes, groups in proof.uses
                for edit in self.group_edits(parsed, groups, proof.name, frozenset(folded), rationale)
            ),
        )

    @staticmethod
    def group_edits(
        parsed: ParsedModule,
        groups: list[MemberGroup],
        owner: str,
        folded: frozenset[str],
        rationale: str,
    ) -> Iterator[NominalSourceEdit]:
        geometry = SourceTextGeometry(parsed.source)
        unique = {tuple(map(id, group.members)): group for group in groups}
        replacements = [
            replacement
            for group in unique.values()
            for replacement in group.folded(geometry, owner, folded, parsed.file_path)
        ]
        if replacements:
            yield from geometry.physical_edits(
                file_path=parsed.file_path, replacements=replacements, rationale=rationale
            )


class EnumFoldFindingRecipeSynthesizer(ClassTargetFindingRecipeSynthesizer, ABC):
    recipe_reason = "The members are only ever named together; one member is all the code distinguishes."

    def operation_for(
        self, target: SourceRewriteTarget, context: CodemodSelectorContext
    ) -> RefactorRecipeOperation:
        snapshot = context.execution_snapshot()
        _, digest, node = snapshot.target_node_for_rewrite_target(target)
        folds = EnumFoldProof.of(snapshot, node, digest.file_path).folds()
        if not folds:
            raise ValueError(f"`{digest.qualname}` has no members that are only named together")
        return FoldEnumMembersOperation(target=target, members=folds[0])
