"""Core declarations for composable codemod operations."""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import (
    ClassVar,
    Self,
    cast,
)

from metaclass_registry import AutoRegisterMeta

from .codemod_architecture_guards import ArchitectureGuardRule
from .codemod_imports import ModuleImportMutation
from .codemod_payload import (
    DiscriminatedPayloadRecord,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_preflight import CodemodOperationPreflightReport
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewritePlanItem
from .codemod_semantics import CodemodSourceDependencyScope
from .codemod_source_edits import (
    NominalSourceEdit,
    SourceEditOrigin,
    SourceFileCreation,
)
from .registry_identity import suffix_trimmed_class_name_registry_key
from .semantic_descent import AuthorityClaim
from .source_index import (
    AstTargetDigest,
    AstTargetNode,
)


@dataclass(frozen=True, kw_only=True)
class RefactorRecipeOperation(
    SourceRewritePlanItem,
    DiscriminatedPayloadRecord,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Agent-authored codemod operation compiled through source-index geometry."""

    __registry_key__ = "operation_key_value"
    __key_extractor__ = staticmethod(suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True
    registry_key_suffix: ClassVar[str] = "Operation"
    operation_key_value: ClassVar[str]
    discriminator_field_name: ClassVar[str] = "operation"
    omit_none_payload_values: ClassVar[bool] = True
    source_dependency_scope: ClassVar[CodemodSourceDependencyScope] = (
        CodemodSourceDependencyScope.EXPLICIT_TARGETS
    )

    @classmethod
    def operation_key(cls) -> str:
        return cls.operation_key_value

    @classmethod
    def record_type_for_discriminator(cls, discriminator: str) -> type[Self]:
        operation_type = cls.__registry__.get(discriminator)
        if operation_type is None or not issubclass(operation_type, cls):
            raise ValueError(f"Unsupported recipe operation: {discriminator}")
        return cast(type[Self], operation_type)

    @classmethod
    def discriminator_key(cls) -> str:
        return cls.operation_key()

    @abstractmethod
    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError

    def originated_edits(
        self,
        context: CodemodSelectorContext,
        *,
        recipe_id: str,
        plan_item_index: int,
    ) -> tuple[NominalSourceEdit, ...]:
        origin = SourceEditOrigin(
            recipe_id=recipe_id,
            plan_item_declaration=type(self).__name__,
            plan_item_index=plan_item_index,
        )
        return tuple(edit.with_origin(origin) for edit in self.source_edits(context))

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        del context
        return ()

    def source_file_creations(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileCreation, ...]:
        del context
        return ()

    def created_source_paths(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, ...]:
        return tuple(
            creation.file_path for creation in self.source_file_creations(context)
        )

    def declared_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        """Derive authority claims established by this operation."""

        del context
        return ()

    def declared_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        """Derive post-refactor invariants established by this operation."""

        del context
        return ()

    def required_source_path(
        self,
        context: CodemodSelectorContext,
        operation_name: str,
    ) -> str:
        if self.target.file_path is None:
            raise ValueError(f"{operation_name} requires file_path")
        return self.target.required_file_path(context.source_index)

    def required_import_mutations(
        self,
        source_path: str,
        *,
        import_source: str,
        default_rationale: str,
    ) -> tuple["ModuleImportMutation", ...]:
        return (
            ModuleImportMutation.from_source(
                file_path=source_path,
                import_source=import_source,
                rationale=self.rationale_text(default_rationale),
            ),
        )

    def target_digest(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, AstTargetDigest]:
        target_identifier = self.target.required_target_id(context.source_index)
        return target_identifier, context.source_index.target_by_id[target_identifier]

    def target_node_from_context(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, AstTargetDigest, AstTargetNode]:
        return context.target_node_for_rewrite_target(self.target)


@dataclass(frozen=True, kw_only=True)
class SourcePayloadOperation(RefactorRecipeOperation, ABC):
    """Recipe operation whose declaration owns required Python source text."""

    source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
