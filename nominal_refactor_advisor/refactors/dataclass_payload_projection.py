from __future__ import annotations

import ast
from dataclasses import dataclass

from nominal_refactor_advisor.refactors.function_projection_targets import (
    FunctionReturnProjectionTarget,
    ResolvedFunctionProjectionTarget,
    ReturnDictFieldValueExtractor,
    ReturnDictProjectionTarget,
    ReturnDictProjectionTargetAuthority,
)
from nominal_refactor_advisor.refactors.dataclass_projection import (
    DataclassAuthorityMappingRecipeBuilder,
    DataclassInstanceFieldRunProjection,
    DataclassPayloadAuthorityTarget,
    DataclassProjectionBoundary,
    DataclassesModuleReference,
    SourceDerivedDataclassProjection,
    SourceDerivedDataclassProjectionOperation,
    SourceDerivedDataclassProjectionRecipeParts,
)
from nominal_refactor_advisor.refactors.semantic_mirror import (
    SemanticMirrorImportBoundary,
    SemanticMirrorRecipeSeedLocations,
)
from nominal_refactor_advisor.ast_tools import walk_function_body_nodes
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextGeometry,
    SourceTextReplacement,
)
from nominal_refactor_advisor.models import (
    RefactorFinding,
)
from nominal_refactor_advisor.refactor_concepts import DataclassPayloadProjectionConcept


@dataclass(frozen=True)
class DataclassPayloadProjectionCandidate:
    """One exhaustive return-dict projection proved against a dataclass."""

    projection: ReturnDictProjectionTarget
    field_run: DataclassInstanceFieldRunProjection
    dataclasses_reference: DataclassesModuleReference
    source_replacement: SourceTextReplacement


@dataclass(frozen=True)
class DataclassPayloadProjectionDerivation(
    SourceDerivedDataclassProjection[ReturnDictProjectionTarget]
):
    """Current-source proof for one exhaustive dataclass return-dict projection."""

    field_run: DataclassInstanceFieldRunProjection
    dataclasses_reference: DataclassesModuleReference

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassPayloadProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node.body)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
            if (
                candidate := cls.candidate_from_return(
                    context,
                    boundary.authority,
                    boundary.function,
                    node,
                    boundary.authority_import_source,
                )
            )
            is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one exhaustive return-dict projection; found {len(candidates)}"
            )
        candidate = candidates[0]
        return cls(
            authority=boundary.authority,
            projection=candidate.projection,
            source_replacement=candidate.source_replacement,
            import_sources=tuple(
                import_source
                for import_source in (
                    boundary.authority_import_source,
                    candidate.dataclasses_reference.import_source,
                )
                if import_source is not None
            ),
            field_run=candidate.field_run,
            dataclasses_reference=candidate.dataclasses_reference,
        )

    @classmethod
    def candidate_from_return(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        function: ResolvedFunctionProjectionTarget,
        return_node: ast.Return,
        authority_import_source: str | None,
    ) -> DataclassPayloadProjectionCandidate | None:
        projection = ReturnDictProjectionTargetAuthority.from_return_node(
            function,
            return_node,
            authority.field_names,
        )
        if projection is None:
            return None
        field_run = DataclassInstanceFieldRunProjection.from_targets(
            authority,
            projection,
        )
        dataclasses_reference = DataclassesModuleReference.from_projection(
            context,
            projection,
        )
        if (
            field_run is None
            or dataclasses_reference is None
            or not field_run.owner_has_nominal_authority_type(
                context,
                authority,
                projection,
                authority_import_source,
            )
        ):
            return None
        source_replacement = cls.projection_replacement(
            context,
            authority,
            projection,
            field_run,
            dataclasses_reference,
        )
        if source_replacement is None:
            return None
        return DataclassPayloadProjectionCandidate(
            projection=projection,
            field_run=field_run,
            dataclasses_reference=dataclasses_reference,
            source_replacement=source_replacement,
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnDictProjectionTarget,
        field_run: DataclassInstanceFieldRunProjection,
        dataclasses_reference: DataclassesModuleReference,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        replacement_span = field_run.replacement_span(geometry)
        owner_source = geometry.segment_for_node(field_run.owner_node)
        if replacement_span is None or owner_source is None:
            return None
        if replacement_span.contains_comment(source):
            return None
        indentation = " " * field_run.first_key_node.col_offset
        continuation_indentation = f"{indentation}    "
        nested_indentation = f"{continuation_indentation}    "
        replacement_source = (
            f"{field_run.comprehension_prefix}{{\n"
            f"{continuation_indentation}field.name: getattr(\n"
            f"{nested_indentation}{owner_source},\n"
            f"{nested_indentation}field.name,\n"
            f"{continuation_indentation})\n"
            f"{continuation_indentation}for field in "
            f"{dataclasses_reference.expression}.fields(\n"
            f"{nested_indentation}{authority.name}\n"
            f"{continuation_indentation})\n"
            f"{indentation}}}"
        )
        return replacement_span.replacement(source, replacement_source)


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassPayloadProjectionOperation(
    SourceDerivedDataclassProjectionOperation[ReturnDictProjectionTarget]
):
    """Derive one exhaustive return-dict projection from a dataclass authority."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[ReturnDictProjectionTarget]:
        return DataclassPayloadProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class DataclassPayloadProjectionMappingRecipeBuilder(
    ReturnDictFieldValueExtractor,
    DataclassAuthorityMappingRecipeBuilder[
        ReturnDictProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[ReturnDictProjectionTarget],
    ],
    DataclassPayloadProjectionConcept,
):
    """Derive an exhaustive direct-instance mapping from dataclass fields."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        seed = SemanticMirrorRecipeSeedLocations.from_finding(self.finding)
        if seed is None:
            return (
                "dataclass payload projection requires projection and authority "
                "locations"
            )
        import_boundary = SemanticMirrorImportBoundary.from_seed(seed, self)
        if import_boundary is None:
            return "dataclass payload projection requires source-index-resolved files"
        if import_boundary.import_would_create_cycle(self):
            return "dataclass payload projection import would create a module cycle"
        if self.parts is not None:
            return (
                "dataclass payload projection has an executable instance-field recipe"
            )
        return (
            "dataclass payload projection requires one contiguous, exhaustive, "
            "declaration-ordered run of direct field reads from a nominally typed "
            "authority instance"
        )

    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
        )
        return function_return is not None and isinstance(
            function_return.return_node.value,
            ast.Dict,
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> ReturnDictProjectionTarget | None:
        return ReturnDictProjectionTargetAuthority.from_function_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
            field_names=self.finding.metrics.plan_field_names,
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnDictProjectionTarget,
    ) -> SourceDerivedDataclassProjectionRecipeParts[ReturnDictProjectionTarget] | None:
        operation = DeriveDataclassPayloadProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target=SourceRewriteTarget(
                target_id=projection.target.target_id
            ),
        )
        return SourceDerivedDataclassProjectionRecipeParts.from_proven_operation(
            self,
            authority=authority,
            operation=operation,
        )
