from __future__ import annotations

import ast
from dataclasses import dataclass

from .function_projection_targets import ResolvedFunctionProjectionTarget
from .dataclass_projection import (
    DataclassAuthorityMappingRecipeBuilder,
    DataclassFieldNameCollectionProjectionTarget,
    DataclassPayloadAuthorityTarget,
    DataclassProjectionBoundary,
    DataclassesModuleReference,
    SourceDerivedDataclassProjection,
    SourceDerivedDataclassProjectionOperation,
    SourceDerivedDataclassProjectionRecipeParts,
)
from .semantic_mirror import SemanticMirrorRecipeSeedLocations
from ..codemod_selection_context import CodemodSelectorContext
from ..codemod_selector_models import SourceRewriteTarget
from ..codemod_source_edits import (
    SourceTextGeometry,
    SourceTextReplacement,
    SourceTextSpan,
)
from ..models import (
    MappingMetrics,
    RefactorFinding,
)
from ..refactor_concepts import DataclassPayloadProjectionConcept


@dataclass(frozen=True)
class DataclassFieldNameCollectionProjectionDerivation(
    SourceDerivedDataclassProjection[DataclassFieldNameCollectionProjectionTarget]
):
    """Current-source proof for one exhaustive dataclass field-name collection."""

    dataclasses_reference: DataclassesModuleReference

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassFieldNameCollectionProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        if boundary.authority_import_source is not None:
            raise ValueError(
                "Dataclass field-name projection requires an existing runtime "
                "authority reference"
            )
        candidates = (
            DataclassFieldNameCollectionProjectionTarget.candidates_from_function(
                boundary.function,
                boundary.authority,
            )
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one exhaustive field-name collection; found {len(candidates)}"
            )
        projection = candidates[0]
        dataclasses_reference = DataclassesModuleReference.from_projection(
            context,
            projection,
        )
        if dataclasses_reference is None:
            raise ValueError(
                "Dataclass field-name projection has no collision-free dataclasses "
                "reference"
            )
        source_replacement = cls.projection_replacement(
            context,
            boundary.authority,
            projection,
            dataclasses_reference,
        )
        if source_replacement is None:
            raise ValueError(
                "Dataclass field-name projection cannot preserve its source span"
            )
        return cls(
            authority=boundary.authority,
            projection=projection,
            source_replacement=source_replacement,
            import_sources=(
                (dataclasses_reference.import_source,)
                if dataclasses_reference.import_source is not None
                else ()
            ),
            dataclasses_reference=dataclasses_reference,
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: DataclassFieldNameCollectionProjectionTarget,
        dataclasses_reference: DataclassesModuleReference,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        offsets = geometry.node_offsets(projection.collection_node)
        if offsets is None:
            return None
        replacement_span = SourceTextSpan.from_offsets(offsets)
        if replacement_span.contains_comment(source):
            return None
        return replacement_span.replacement(
            source,
            projection.derived_source(dataclasses_reference, authority),
        )


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassFieldNameCollectionProjectionOperation(
    SourceDerivedDataclassProjectionOperation[
        DataclassFieldNameCollectionProjectionTarget
    ]
):
    """Derive one exhaustive field-name collection from a dataclass authority."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[DataclassFieldNameCollectionProjectionTarget]:
        return DataclassFieldNameCollectionProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class DataclassFieldNameCollectionProjectionMappingRecipeBuilder(
    DataclassAuthorityMappingRecipeBuilder[
        DataclassFieldNameCollectionProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassFieldNameCollectionProjectionTarget
        ],
    ],
    DataclassPayloadProjectionConcept,
):
    """Derive an exhaustive local field-name collection from a dataclass."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return "dataclass field-name projection requires mapping metrics"
        if self.parts is not None:
            return "dataclass field-name projection has an executable recipe"
        return (
            "dataclass field-name projection requires one local tuple or list that "
            "exhaustively names direct dataclass fields in declaration order, with "
            "the authority already available at runtime"
        )

    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        function = ResolvedFunctionProjectionTarget.from_source_line(
            self,
            source_path=source_path,
            line=seed.projection.line,
        )
        return function is not None and any(
            DataclassFieldNameCollectionProjectionTarget.bound_collection(
                statement,
                seed.projection.subject_symbol,
                seed.projection.line,
            )
            for statement in ast.walk(function.node)
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> DataclassFieldNameCollectionProjectionTarget | None:
        return DataclassFieldNameCollectionProjectionTarget.from_binding_location(
            self,
            source_path=source_path,
            binding_name=seed.projection.subject_symbol,
            line=seed.projection.line,
            field_names=frozenset(self.finding.metrics.plan_field_names),
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: DataclassFieldNameCollectionProjectionTarget,
    ) -> (
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassFieldNameCollectionProjectionTarget
        ]
        | None
    ):
        operation = DeriveDataclassFieldNameCollectionProjectionOperation(
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
