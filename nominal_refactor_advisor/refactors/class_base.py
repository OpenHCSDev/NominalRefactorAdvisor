from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import ClassVar

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_authority_collapse import (
    ClassAuthorityCollapseProof,
    IntermediateClassAuthorityCollapseProof,
    RedundantClassAuthorityCollapseProof,
)
from nominal_refactor_advisor.class_families.parsed_index import (
    ModuleClassReferenceResolver,
)
from nominal_refactor_advisor.class_index import declared_nominal_base_count
from nominal_refactor_advisor.refactors.target_deletion import DeleteTargetOperation
from nominal_refactor_advisor.codemod_authority_claims import AstTargetAuthorityClaim
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassHeaderSpanSourceAuthority,
)
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_module_declarations import ModuleSymbolTable
from nominal_refactor_advisor.codemod_payload import (
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import (
    RepositorySourceReprovedOperation,
    SourceReprovedOperation,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    PhysicalSourceEdit,
)
from nominal_refactor_advisor.codemod_target_selectors import ClassFamilyTargetSelector
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.semantic_descent import (
    AuthorityClaim,
    SemanticAuthorityKind,
)


@dataclass(frozen=True, kw_only=True)
class ClassBaseMutationOperationABC(SourceReprovedOperation, ABC):
    """Source-proved mutation of one class declaration's direct bases."""

    base_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_identifier, target, node = self.target_node_from_context(snapshot)
        if not isinstance(node, ast.ClassDef):
            raise ValueError(f"Target {target.qualname!r} is not a class definition")
        header_authority = ClassHeaderSpanSourceAuthority(
            node=node,
            source=snapshot.sources_by_file_path[target.file_path],
        )
        return header_authority.source_edits(
            self.replacement_header_lines(header_authority),
            file_path=target.file_path,
            rationale=self.rationale or f"Update direct bases of {target.qualname!r}.",
        )

    @abstractmethod
    def replacement_header_lines(
        self,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> tuple[str, ...]:
        """Return the leaf operation's complete replacement class header."""

        raise NotImplementedError


@dataclass(frozen=True)
class ClassAuthorityReferenceProof:
    """Prove one generated class-authority reference at a module boundary."""

    authority: ResolvedClassTarget
    authority_symbol: str
    projection_module: ParsedModule
    resolver: ModuleClassReferenceResolver
    symbol_table: ModuleSymbolTable

    @classmethod
    def from_native_declaration(
        cls,
        context: CodemodSelectorContext,
        declaration: type,
        projection_path: str,
    ) -> ClassAuthorityReferenceProof:
        native = NativeDeclaration(declaration)
        targets = context.source_index.targets_matching_repository_symbol(
            native.qualified_name
        )
        if len(targets) != 1 or not targets[0].is_class:
            raise ValueError(
                f"Native class authority {native.qualified_name!r} requires one source declaration"
            )
        target = targets[0]
        node = context.ast_target_nodes_by_id[target.target_id]
        if not isinstance(node, ast.ClassDef):
            raise ValueError("Native class authority is not a class declaration")
        native.require_source_matches(node)
        return cls.from_context(
            context, ResolvedClassTarget(target, node), projection_path
        )

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority: ResolvedClassTarget,
        projection_path: str,
    ) -> ClassAuthorityReferenceProof:
        projection_module = context.parsed_module_for_source_path(projection_path)
        authority_symbol = authority.required_symbol(context)
        return cls(
            authority=authority,
            authority_symbol=authority_symbol,
            projection_module=projection_module,
            resolver=context.class_reference_resolver_for_source_path(projection_path),
            symbol_table=ModuleSymbolTable(
                file_path=projection_module.file_path,
                source=projection_module.source,
                module=projection_module.module,
            ),
        )

    @property
    def unavailable_builtin_names(self) -> frozenset[str]:
        return frozenset(
            (
                *self.symbol_table.top_level_names,
                *self.symbol_table.import_sources_by_name,
            )
        )

    def required_import_source(
        self,
        context: CodemodSelectorContext,
    ) -> str | None:
        authority_name = self.authority.name
        declaration_bindings = self.symbol_table.binding_statements(authority_name)
        import_binding = self.symbol_table.import_sources_by_name.get(authority_name)
        if self.projection_module.file_path == self.authority.file_path:
            authority_binding_is_exact = (
                len(declaration_bindings) == 1
                and isinstance(declaration_bindings[0], ast.ClassDef)
                and declaration_bindings[0].lineno == self.authority.target.line
                and declaration_bindings[0].name == authority_name
            )
            if not authority_binding_is_exact or import_binding is not None:
                raise ValueError(f"Class authority name {authority_name!r} is rebound")
            return None
        if declaration_bindings:
            raise ValueError(f"Class authority name {authority_name!r} is rebound")
        reference = ast.Name(id=authority_name, ctx=ast.Load())
        if self.resolver.symbol_for_reference(reference) == self.authority_symbol:
            return None
        if import_binding is not None:
            raise ValueError(
                f"Class authority name {authority_name!r} is imported from another "
                "declaration"
            )
        return context.module_import_graph.required_import_source(
            importing_file_path=self.projection_module.file_path,
            imported_file_path=self.authority.file_path,
            imported_name=authority_name,
        )


@dataclass(frozen=True, kw_only=True)
class AddClassBaseOperation(ClassBaseMutationOperationABC):
    """Add one base class to a class declaration."""

    def replacement_header_lines(
        self,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> tuple[str, ...]:
        return header_authority.with_added_base(self.base_name)


@dataclass(frozen=True, kw_only=True)
class RemoveClassBaseOperation(ClassBaseMutationOperationABC):
    """Remove one base class from a class declaration."""

    def replacement_header_lines(
        self,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> tuple[str, ...]:
        return header_authority.without_base(self.base_name)


@dataclass(frozen=True, kw_only=True)
class ReplaceClassBaseOperation(ClassBaseMutationOperationABC):
    """Replace one authored direct base while preserving direct-base precedence."""

    replacement_base_name: str = codemod_payload_field(
        RequiredStringPayloadValueCodec()
    )

    def replacement_header_lines(
        self,
        header_authority: ClassHeaderSpanSourceAuthority,
    ) -> tuple[str, ...]:
        return header_authority.with_replaced_base(
            self.base_name, self.replacement_base_name
        )


@dataclass(frozen=True, kw_only=True)
class DirectClassBaseReplacementOperationABC(
    RepositorySourceReprovedOperation,
    ABC,
):
    """Shared source proof for replacing one complete direct-child cohort."""

    replacement_base: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )

    @abstractmethod
    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError

    def direct_class_base_source_edits(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        replaced = ResolvedClassTarget.from_rewrite_target(snapshot, self.target)
        replacement = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            self.replacement_base,
        )
        replaced_symbol = replaced.required_symbol(snapshot)
        replacement_symbol = replacement.required_symbol(snapshot)
        if replaced_symbol == replacement_symbol:
            raise ValueError("Direct class-base replacement requires distinct classes")
        if "." in replacement.qualname:
            raise ValueError("Replacement class base must be a top-level declaration")
        if (
            replacement_symbol
            in snapshot.required_class_family_index.descendant_symbols(replaced_symbol)
        ):
            raise ValueError(
                "Direct class-base replacement cannot introduce an inheritance cycle"
            )
        child_symbols = snapshot.required_class_family_index.children_by_symbol.get(
            replaced_symbol,
            (),
        )
        if not child_symbols:
            raise ValueError("Replaced class base has no direct children")
        child_target_ids = ClassFamilyTargetSelector.target_ids_for_symbols(
            snapshot.source_index,
            snapshot.required_class_family_index,
            child_symbols,
        )
        if len(child_target_ids) != len(child_symbols):
            raise ValueError("Direct-child class targets are incomplete")
        return tuple(
            edit
            for child_target_id in child_target_ids
            for edit in self.child_source_edits(
                snapshot,
                replaced_symbol,
                replacement_symbol,
                replacement,
                child_target_id,
            )
        )

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        snapshot = context.execution_snapshot()
        self.source_edits_from_snapshot(snapshot)
        replacement = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            self.replacement_base,
        )
        return (
            AstTargetAuthorityClaim.from_target(
                replacement.target,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            ),
        )

    def child_source_edits(
        self,
        snapshot: CodemodSourceSnapshot,
        replaced_symbol: str,
        replacement_symbol: str,
        replacement: ResolvedClassTarget,
        child_target_id: str,
    ) -> tuple[NominalSourceEdit, ...]:
        child_target = snapshot.source_index.target_by_id[child_target_id]
        child_node = snapshot.ast_target_nodes_by_id[child_target_id]
        if not isinstance(child_node, ast.ClassDef):
            raise ValueError("Direct-child source target is not a class")
        indexed_child = snapshot.required_class_family_index.class_for(
            snapshot.source_index.symbol_for_target(child_target)
        )
        if indexed_child is None:
            raise ValueError("Direct-child class is absent from the family index")
        if len(indexed_child.resolved_base_symbols) != declared_nominal_base_count(
            indexed_child
        ):
            raise ValueError(
                f"Direct child {child_target.qualname!r} has unresolved nominal bases"
            )
        replacement_relatives = frozenset(
            (
                replacement_symbol,
                *snapshot.required_class_family_index.ancestor_symbols(
                    replacement_symbol
                ),
                *snapshot.required_class_family_index.descendant_symbols(
                    replacement_symbol
                ),
            )
        )
        if (
            frozenset(indexed_child.resolved_base_symbols) - {replaced_symbol}
        ) & replacement_relatives:
            raise ValueError(
                f"Direct child {child_target.qualname!r} has a replacement-related "
                "sibling base"
            )
        resolver = snapshot.class_reference_resolver_for_source_path(
            child_target.file_path
        )
        replaced_bases = tuple(
            base
            for base in child_node.bases
            if resolver.symbol_for_reference(base) == replaced_symbol
        )
        if len(replaced_bases) != 1:
            raise ValueError(
                f"Direct child {child_target.qualname!r} has {len(replaced_bases)} "
                "source-resolved replaced bases"
            )
        header = ClassHeaderSpanSourceAuthority(
            child_node,
            snapshot.sources_by_file_path[child_target.file_path],
        )
        if not header.can_rewrite:
            raise ValueError(
                f"Class header for {child_target.qualname!r} is not reconstructible"
            )
        import_source = ClassAuthorityReferenceProof.from_context(
            snapshot,
            replacement,
            child_target.file_path,
        ).required_import_source(snapshot)
        import_edits = (
            ()
            if import_source is None
            else self.required_import_mutations(
                child_target.file_path,
                import_source=import_source,
                default_rationale="Import the replacement class authority.",
            )
        )
        return (
            *import_edits,
            *header.source_edits(
                header.with_replaced_base(
                    ast.unparse(replaced_bases[0]),
                    replacement.target.name,
                ),
                file_path=child_target.file_path,
                rationale=self.rationale_text(
                    f"Replace direct base {ast.unparse(replaced_bases[0])!r} with "
                    f"{replacement.target.name!r}."
                ),
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceDirectClassBaseOperation(DirectClassBaseReplacementOperationABC):
    """Replace one class authority across its complete direct-child cohort."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.direct_class_base_source_edits(snapshot)


@dataclass(frozen=True, kw_only=True)
class ClassAuthorityCollapseOperationABC(DirectClassBaseReplacementOperationABC, ABC):
    """Replace a displaced class authority at its children and delete it, under a proof.

    A member names the proof its kind of collapse requires and the two sentences that say
    why an import and the class go.
    """

    obsolete_import_reason: ClassVar[str]
    deletion_reason: ClassVar[str]

    @classmethod
    @abstractmethod
    def collapse_proof(cls) -> type[ClassAuthorityCollapseProof]:
        raise NotImplementedError

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        displaced = ResolvedClassTarget.from_rewrite_target(snapshot, self.target)
        replacement = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            self.replacement_base,
        )
        proof = type(self).collapse_proof().require(
            snapshot.parsed_modules,
            snapshot.required_class_family_index,
            displaced_symbol=displaced.required_symbol(snapshot),
            replacement_symbol=replacement.required_symbol(snapshot),
        )
        return (
            *self.direct_class_base_source_edits(snapshot),
            *(
                ModuleImportMutation.remove_names(
                    file_path=displaced.file_path,
                    module_name=obsolete_import.module_name,
                    names=(obsolete_import.imported_name,),
                    rationale=self.rationale_text(type(self).obsolete_import_reason),
                )
                for obsolete_import in proof.obsolete_imports
            ),
            *DeleteTargetOperation(
                target=self.target,
                rationale=self.rationale_text(type(self).deletion_reason),
            ).source_edits(snapshot),
        )


@dataclass(frozen=True, kw_only=True)
class CollapseRedundantClassAuthorityOperation(ClassAuthorityCollapseOperationABC):
    """Replace and delete one behaviorally redundant local class authority."""

    obsolete_import_reason = "Remove an import used only by the displaced class authority."
    deletion_reason = "Delete the displaced redundant class authority."

    @classmethod
    def collapse_proof(cls) -> type[ClassAuthorityCollapseProof]:
        return RedundantClassAuthorityCollapseProof


@dataclass(frozen=True, kw_only=True)
class CollapseIntermediateClassAuthorityOperation(ClassAuthorityCollapseOperationABC):
    """Delete one behavior-free class between its children and direct ancestor."""

    obsolete_import_reason = "Remove an import used only by the displaced intermediary."
    deletion_reason = "Delete the behavior-free intermediate class authority."

    @classmethod
    def collapse_proof(cls) -> type[ClassAuthorityCollapseProof]:
        return IntermediateClassAuthorityCollapseProof
