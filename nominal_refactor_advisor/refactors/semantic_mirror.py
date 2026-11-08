from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import (
    ClassVar,
    Generic,
    Self,
    TypeVar,
    cast,
)

from metaclass_registry import AutoRegisterMeta

from ..codemod_paths import SourcePathResolutionAuthority
from ..codemod_runtime import (
    CodemodSourceSnapshot,
    ExecutableRecipeEvaluation,
    FindingRecipeEvaluation,
    FindingRecipeProofObstacle,
    RefactorRecipe,
    RejectedRecipeEvaluation,
)
from ..codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from ..codemod_selector_models import SourceRewriteTarget
from ..codemod_target_selectors import SourceIndexTargetSelector
from ..finding_recipe_actions import FindingRecipeActionKey
from ..models import (
    FindingMetrics,
    MappingMetrics,
    RefactorFinding,
    SourceLocation,
)
from ..registry_identity import mro_registry_value
from ..semantic_match import loaded_concrete_nominal_descendants
from ..source_index import (
    AstTargetDigest,
    AstTargetNodeKind,
)


@dataclass(frozen=True, kw_only=True)
class SemanticDescentRecipeEvaluation(ExecutableRecipeEvaluation):
    """Executable outcome declared by one semantic-mirror strategy leaf."""

    strategy_type: type[SemanticMirrorFindingRecipeStrategy]

    def gated_by_authority_claim(
        self,
        context: CodemodSelectorContext | None,
        finding: RefactorFinding,
    ) -> FindingRecipeEvaluation:
        del finding
        # Preflight first: an operation whose source no longer proves its
        # claims is reported by the preflight, not raised from claim lookup.
        return self.gated_by_recipe_preflight(context).gated_by_effective_claims(
            context
        )

    def gated_by_effective_claims(
        self, context: CodemodSelectorContext | None
    ) -> FindingRecipeEvaluation:
        if not self.executable_recipe.effective_authority_claims(context):
            return RejectedRecipeEvaluation(
                reason=(
                    "semantic-descent recipe requires a source-resolved AuthorityClaim"
                ),
                evaluation_declaration_type=self.evaluation_declaration_type,
            )
        return self


class SemanticMirrorFindingRecipeStrategy(ABC, metaclass=AutoRegisterMeta):
    """Metric-specific recipe strategy for semantic mirror findings."""

    metric_type: ClassVar[type[FindingMetrics]]
    __registry__: ClassVar[
        dict[type[FindingMetrics], type[SemanticMirrorFindingRecipeStrategy]]
    ] = {}
    __registry_key__ = "metric_type"
    __skip_if_no_key__ = True

    @classmethod
    def strategy_for(
        cls,
        metrics: FindingMetrics,
    ) -> SemanticMirrorFindingRecipeStrategy | None:
        strategy_type = mro_registry_value(cls.__registry__, type(metrics))
        return strategy_type() if strategy_type is not None else None

    @abstractmethod
    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        raise NotImplementedError

    def evaluation_from_recipe(
        self,
        finding: RefactorFinding,
        recipe: RefactorRecipe,
        declaration_type: type[object],
    ) -> FindingRecipeEvaluation:
        del finding
        return SemanticDescentRecipeEvaluation(
            executable_recipe=recipe,
            evaluation_declaration_type=declaration_type,
            strategy_type=type(self),
        )

    @abstractmethod
    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        raise NotImplementedError


@dataclass(frozen=True)
class SemanticMirrorOperationTargets:
    """Exact authority class and projection module for a mirror finding."""

    authority: ResolvedClassTarget
    projection_module: AstTargetDigest

    @staticmethod
    def from_finding(
        context: CodemodSelectorContext,
        finding: RefactorFinding,
    ) -> SemanticMirrorOperationTargets | None:
        seed = SemanticMirrorRecipeSeedLocations.from_finding(finding)
        if seed is None:
            return None
        projection_location = seed.projection
        authority_location = seed.authority
        try:
            projection_paths = context.resolve_source_paths(
                (projection_location.file_path,)
            )
            authority_paths = context.resolve_source_paths(
                (authority_location.file_path,)
            )
        except ValueError:
            return None
        if len(projection_paths) != 1 or len(authority_paths) != 1:
            return None
        authority_target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=tuple(authority_paths),
            qualnames=(authority_location.symbol,),
        ).target_ids(context)
        if len(authority_target_ids) != 1:
            return None
        authority_target = context.source_index.target_by_id[authority_target_ids[0]]
        authority_node = context.ast_target_nodes_by_id.get(authority_target.target_id)
        if not isinstance(authority_node, ast.ClassDef):
            return None
        projection_target_id = SourceRewriteTarget(
            file_path=next(iter(projection_paths))
        ).optional_target_id(context.source_index)
        if projection_target_id is None:
            return None
        projection_module = context.source_index.target_by_id[projection_target_id]
        if not projection_module.is_module:
            return None
        return SemanticMirrorOperationTargets(
            authority=ResolvedClassTarget(authority_target, authority_node),
            projection_module=projection_module,
        )

    @property
    def projection_path(self) -> str:
        return self.projection_module.file_path


@dataclass(frozen=True)
class SemanticMirrorRecipeSeedLocations:
    """Declared projection and authority witnesses for a semantic mirror."""

    projection: SourceLocation
    authority: SourceLocation

    @classmethod
    def from_finding(
        cls,
        finding: RefactorFinding,
    ) -> SemanticMirrorRecipeSeedLocations | None:
        projection = finding.projection_evidence
        authority = finding.authority_evidence
        if projection is None or authority is None:
            return None
        return cls(projection=projection, authority=authority)


@dataclass(frozen=True)
class SemanticMirrorImportBoundary:
    """Resolved source paths for one projection-to-authority descent."""

    projection_path: str
    authority_path: str

    @classmethod
    def from_seed(
        cls,
        seed: SemanticMirrorRecipeSeedLocations,
        context: CodemodSelectorContext,
    ) -> SemanticMirrorImportBoundary | None:
        projection_path = SourcePathResolutionAuthority.from_source_index(
            seed.projection.file_path,
            context.source_index,
        ).optional_path()
        authority_path = SourcePathResolutionAuthority.from_source_index(
            seed.authority.file_path,
            context.source_index,
        ).optional_path()
        if projection_path is None or authority_path is None:
            return None
        return cls(
            projection_path=projection_path,
            authority_path=authority_path,
        )

    def import_would_create_cycle(self, context: CodemodSelectorContext) -> bool:
        return context.module_import_graph.import_would_create_cycle(
            importing_file_path=self.projection_path,
            imported_file_path=self.authority_path,
        )


@dataclass(frozen=True, kw_only=True)
class SemanticMirrorRecipeBuilder(CodemodSourceSnapshot, ABC):
    """Shared source-backed lifecycle for one semantic-mirror recipe domain."""

    finding: RefactorFinding

    @classmethod
    def builder_types(cls) -> tuple[type[Self], ...]:
        """Derive concrete builders from this nominal domain branch."""

        return tuple(
            cast(type[Self], builder_type)
            for builder_type in loaded_concrete_nominal_descendants(cls)
        )

    @classmethod
    def builders_from_context(
        cls,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None,
    ) -> tuple[Self, ...]:
        if context is None:
            return ()
        return tuple(
            builder
            for builder_type in cls.builder_types()
            if (
                builder := builder_type(
                    source_index=context.source_index,
                    sources_by_file_path=context.sources_by_file_path,
                    class_family_index=context.class_family_index,
                    module_node_cache=context.module_nodes_by_file_path,
                    ast_target_node_cache=context.ast_target_nodes_by_id,
                    module_import_graph_cache=context.module_import_graph,
                    finding=finding,
                )
            ).is_applicable()
        )

    @staticmethod
    def proof_obstacles(
        builders: tuple[Self, ...],
    ) -> tuple[FindingRecipeProofObstacle, ...]:
        return tuple(builder.proof_obstacle() for builder in builders)

    def is_applicable(self) -> bool:
        """Return whether this declaration owns the finding's semantic domain."""

        return True

    @abstractmethod
    def recipe(self) -> RefactorRecipe | None:
        raise NotImplementedError

    @abstractmethod
    def rejection_reason(self) -> str:
        raise NotImplementedError

    def proof_obstacle(self) -> FindingRecipeProofObstacle:
        return FindingRecipeProofObstacle(
            executable_declaration_type=type(self),
            reason=self.rejection_reason(),
        )


class MappingSemanticMirrorRecipeBuilder(
    SemanticMirrorRecipeBuilder,
    ABC,
):
    """Nominal domain for mapping-mirror recipe declarations."""


class RegistrationSemanticMirrorRecipeBuilder(
    SemanticMirrorRecipeBuilder,
    ABC,
):
    """Nominal domain for registration-mirror recipe declarations."""


class FindingRecipeParts(ABC):
    """Executable recipe facts owned by a recipe builder."""

    @abstractmethod
    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        raise NotImplementedError


RecipePartsT = TypeVar("RecipePartsT", bound=FindingRecipeParts)


class PartsBackedMappingRecipeBuilder(
    MappingSemanticMirrorRecipeBuilder,
    Generic[RecipePartsT],
    ABC,
):
    """Mapping recipe builder whose actionability is owned by a parts record."""

    @property
    @abstractmethod
    def parts(self) -> RecipePartsT | None:
        raise NotImplementedError

    def recipe(self) -> RefactorRecipe | None:
        if self.parts is None:
            return None
        return self.parts.recipe_for(self.finding)


@dataclass(frozen=True)
class SemanticMirrorRecipeSelection:
    """One unambiguous source-proved builder and its recipe."""

    builder: MappingSemanticMirrorRecipeBuilder
    recipe: RefactorRecipe

    @classmethod
    def from_builders(
        cls,
        builders: tuple[MappingSemanticMirrorRecipeBuilder, ...],
    ) -> SemanticMirrorRecipeSelection | None:
        candidates = tuple(
            cls(builder=builder, recipe=recipe)
            for builder in builders
            for recipe in (builder.recipe(),)
            if recipe is not None
        )
        if len(candidates) > 1:
            raise ValueError(
                "Mapping mirror finding produced multiple inferred recipes: "
                f"{tuple(type(candidate.builder).__name__ for candidate in candidates)!r}"
            )
        return candidates[0] if candidates else None


class MappingSemanticMirrorRecipeStrategy(SemanticMirrorFindingRecipeStrategy):
    """Represent mapping/schema semantic mirrors as first-class DSL targets."""

    metric_type = MappingMetrics

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        builders = MappingSemanticMirrorRecipeBuilder.builders_from_context(
            finding,
            context,
        )
        selection = SemanticMirrorRecipeSelection.from_builders(builders)
        if selection is not None:
            return self.evaluation_from_recipe(
                finding,
                selection.recipe,
                type(selection.builder),
            )
        return self.rejected_evaluation(
            finding,
            context,
            builders=builders,
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = finding.primary_evidence
        mapping_name = finding.metrics.plan_mapping_name
        source_name = finding.metrics.plan_source_name
        if evidence is None or mapping_name is None or source_name is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((evidence.file_path, f"{mapping_name}->{source_name}"),),
        )

    def rejected_evaluation(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
        *,
        builders: tuple[MappingSemanticMirrorRecipeBuilder, ...] | None = None,
    ) -> RejectedRecipeEvaluation:
        if context is None:
            return RejectedRecipeEvaluation(
                reason=(
                    "semantic mapping mirror recipes require a source selector context"
                ),
                evaluation_declaration_type=type(self),
            )
        seed = SemanticMirrorRecipeSeedLocations.from_finding(finding)
        import_boundary = (
            SemanticMirrorImportBoundary.from_seed(seed, context)
            if seed is not None
            else None
        )
        if import_boundary is not None and import_boundary.import_would_create_cycle(
            context
        ):
            reason = "semantic authority import would create a module cycle"
            return RejectedRecipeEvaluation(
                reason=reason,
                evaluation_declaration_type=type(self),
                obstacles=(
                    FindingRecipeProofObstacle(
                        executable_declaration_type=SemanticMirrorImportBoundary,
                        reason=reason,
                    ),
                ),
            )
        resolved_builders = (
            builders
            if builders is not None
            else MappingSemanticMirrorRecipeBuilder.builders_from_context(
                finding,
                context,
            )
        )
        proof_obstacles = MappingSemanticMirrorRecipeBuilder.proof_obstacles(
            resolved_builders,
        )
        if proof_obstacles:
            return RejectedRecipeEvaluation(
                reason=(
                    "no inferred mapping recipe builder proved an executable "
                    "exact derivation"
                ),
                evaluation_declaration_type=type(self),
                obstacles=proof_obstacles,
            )
        return RejectedRecipeEvaluation(
            reason=(
                "semantic mapping mirror has a stable DSL action key, but no safe "
                f"mapping recipe exists yet to derive "
                f"`{finding.metrics.plan_mapping_name}` from "
                f"`{finding.metrics.plan_source_name}`"
            ),
            evaluation_declaration_type=type(self),
        )

    @staticmethod
    def import_source_for_path(
        context: CodemodSelectorContext,
        *,
        projection_path: str,
        authority_path: str,
        authority_name: str,
    ) -> str | None:
        return context.module_import_graph.import_source(
            importing_file_path=projection_path,
            imported_file_path=authority_path,
            imported_name=authority_name,
        )

    @staticmethod
    def authority_class_target(
        context: CodemodSelectorContext,
        authority_location: SourceLocation,
        authority_name: str,
    ) -> ResolvedClassTarget | None:
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(authority_location.file_path,),
            qualnames=(authority_name,),
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        target = context.source_index.target_by_id[target_ids[0]]
        node = context.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, ast.ClassDef):
            return None
        return ResolvedClassTarget(target=target, node=node)
