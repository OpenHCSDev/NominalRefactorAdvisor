from __future__ import annotations

import ast
from dataclasses import dataclass

from nominal_refactor_advisor.refactors.function_projection_targets import (
    FunctionReturnProjectionTarget,
    ReturnKeyValueSequenceProjectionTarget,
    ReturnKeyValueSequenceProjectionTargetAuthority,
)
from nominal_refactor_advisor.refactors.dataclass_projection import (
    DataclassAuthorityMappingRecipeBuilder,
    DataclassKeyValueElementRunProjection,
    DataclassPayloadAuthorityTarget,
    DataclassProjectionBoundary,
    DataclassesModuleReference,
    SourceDerivedDataclassProjection,
    SourceDerivedDataclassProjectionOperation,
    SourceDerivedDataclassProjectionRecipeParts,
)
from nominal_refactor_advisor.refactors.semantic_mirror import (
    SemanticMirrorRecipeSeedLocations,
)
from nominal_refactor_advisor.ast_tools import walk_function_body_nodes
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextGeometry,
    SourceTextReplacement,
    SourceTextSpan,
)
from nominal_refactor_advisor.models import (
    RefactorFinding,
)
from nominal_refactor_advisor.refactor_concepts import DataclassPayloadProjectionConcept


@dataclass(frozen=True)
class DataclassKeyValueSequenceProjectionCandidate:
    """One exhaustive return-pair projection proved against a dataclass."""

    projection: ReturnKeyValueSequenceProjectionTarget
    element_run: DataclassKeyValueElementRunProjection
    dataclasses_reference: DataclassesModuleReference
    source_replacement: SourceTextReplacement


@dataclass(frozen=True)
class DataclassKeyValueSequenceProjectionDerivation(
    SourceDerivedDataclassProjection[ReturnKeyValueSequenceProjectionTarget]
):
    """Current-source proof for one exhaustive dataclass return-pair sequence."""

    element_run: DataclassKeyValueElementRunProjection
    dataclasses_reference: DataclassesModuleReference

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassKeyValueSequenceProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node.body)
            if isinstance(node, ast.Return)
            if (
                candidate := cls.candidate_from_return(
                    context,
                    boundary,
                    node,
                )
            )
            is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one exhaustive return-pair sequence; found {len(candidates)}"
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
            element_run=candidate.element_run,
            dataclasses_reference=candidate.dataclasses_reference,
        )

    @classmethod
    def candidate_from_return(
        cls,
        context: CodemodSelectorContext,
        boundary: DataclassProjectionBoundary,
        return_node: ast.Return,
    ) -> DataclassKeyValueSequenceProjectionCandidate | None:
        projection = ReturnKeyValueSequenceProjectionTargetAuthority.from_return_node(
            boundary.function,
            return_node,
            boundary.authority.field_names,
        )
        if projection is None:
            return None
        element_run = DataclassKeyValueElementRunProjection.from_targets(
            boundary.authority,
            projection,
        )
        dataclasses_reference = DataclassesModuleReference.from_projection(
            context,
            projection,
        )
        if (
            element_run is None
            or dataclasses_reference is None
            or not element_run.owner_has_nominal_authority_type(
                context,
                boundary.authority,
                projection,
                boundary.authority_import_source,
            )
        ):
            return None
        source_replacement = cls.projection_replacement(
            context,
            boundary.authority,
            projection,
            element_run,
            dataclasses_reference,
        )
        if source_replacement is None:
            return None
        return DataclassKeyValueSequenceProjectionCandidate(
            projection=projection,
            element_run=element_run,
            dataclasses_reference=dataclasses_reference,
            source_replacement=source_replacement,
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnKeyValueSequenceProjectionTarget,
        element_run: DataclassKeyValueElementRunProjection,
        dataclasses_reference: DataclassesModuleReference,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        first_offsets = geometry.node_offsets(element_run.first_element_node)
        last_offsets = geometry.node_offsets(element_run.last_element_node)
        sequence_offsets = geometry.node_offsets(projection.sequence_node)
        owner_source = geometry.segment_for_node(element_run.owner_node)
        if (
            first_offsets is None
            or last_offsets is None
            or sequence_offsets is None
            or owner_source is None
        ):
            return None
        replacement_span = SourceTextSpan(
            start_offset=first_offsets[0],
            end_offset=last_offsets[1],
        )
        if replacement_span.contains_comment(source):
            return None
        has_trailing_comma = (
            source[last_offsets[1] : sequence_offsets[1]].lstrip().startswith(",")
        )
        indentation = " " * element_run.first_element_node.col_offset
        continuation_indentation = f"{indentation}    "
        nested_indentation = f"{continuation_indentation}    "
        value_indentation = f"{nested_indentation}    "
        replacement_source = (
            "*(\n"
            f"{continuation_indentation}(\n"
            f"{nested_indentation}field.name,\n"
            f"{nested_indentation}getattr(\n"
            f"{value_indentation}{owner_source},\n"
            f"{value_indentation}field.name,\n"
            f"{nested_indentation})\n"
            f"{continuation_indentation})\n"
            f"{continuation_indentation}for field in "
            f"{dataclasses_reference.expression}.fields(\n"
            f"{nested_indentation}{authority.name}\n"
            f"{continuation_indentation})\n"
            f"{indentation})"
            f"{'' if has_trailing_comma else ','}"
        )
        return replacement_span.replacement(source, replacement_source)


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassKeyValueSequenceProjectionOperation(
    SourceDerivedDataclassProjectionOperation[ReturnKeyValueSequenceProjectionTarget]
):
    """Derive one exhaustive return-pair sequence from a dataclass authority."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[ReturnKeyValueSequenceProjectionTarget]:
        return DataclassKeyValueSequenceProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class DataclassKeyValueSequenceProjectionMappingRecipeBuilder(
    DataclassAuthorityMappingRecipeBuilder[
        ReturnKeyValueSequenceProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[
            ReturnKeyValueSequenceProjectionTarget
        ],
    ],
    DataclassPayloadProjectionConcept,
):
    """Derive returned ``("field", value)`` items from a dataclass authority."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if self.parts is not None:
            return (
                "dataclass key/value sequence projection has an executable "
                "instance-field recipe"
            )
        return (
            "dataclass key/value sequence projection requires one contiguous, "
            "exhaustive, declaration-ordered run of direct pair values read from "
            "a nominally typed authority instance"
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
        if function_return is None or not isinstance(
            function_return.return_node.value,
            ast.Tuple | ast.List,
        ):
            return False
        field_names = frozenset(self.finding.metrics.plan_field_names)
        return any(
            field_value is not None and field_value.field_name in field_names
            for element in function_return.return_node.value.elts
            for field_value in (
                ReturnKeyValueSequenceProjectionTargetAuthority.field_value(element),
            )
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> ReturnKeyValueSequenceProjectionTarget | None:
        return ReturnKeyValueSequenceProjectionTargetAuthority.from_function_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
            field_names=self.finding.metrics.plan_field_names,
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnKeyValueSequenceProjectionTarget,
    ) -> (
        SourceDerivedDataclassProjectionRecipeParts[
            ReturnKeyValueSequenceProjectionTarget
        ]
        | None
    ):
        operation = DeriveDataclassKeyValueSequenceProjectionOperation(
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
