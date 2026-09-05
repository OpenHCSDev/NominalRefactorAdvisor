"""Source-proven codemods for consolidating class-member authority."""

from __future__ import annotations

import ast
import keyword as keyword_module
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Mapping
from dataclasses import (
    dataclass,
    replace,
)
from functools import cached_property

from .ast_tools import (
    EagerNameLoadCollector,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    statements_without_docstring,
)
from .class_index import (
    ClassMethodPromotionSafetyProfile,
    ClassMethodReceiverRequirements,
    ClassSymbolResolutionAuthority,
    IndexedClass,
)
from .codemod_authority_claims import AstTargetAuthorityClaim
from .codemod_declaration_source import (
    ClassBodySourceAuthority,
    ClassHeaderSpanSourceAuthority,
)
from .codemod_paths import SourcePathResolutionAuthority
from .codemod_payload import (
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_reproof import RepositorySourceReprovedOperation
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from .codemod_selector_models import SourceRewriteTarget
from .codemod_statement_source import StatementDeletionSource, StatementSource
from .codemod_source_edits import (
    PhysicalSourceEdit,
    SourceInsertion,
    SourceNodeDecoratorPolicy,
    SourceNodeSpan,
    SourceSpanDeletion,
    SourceSpanReplacement,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceLineSpan,
)
from .exact_field_authority import ExactDataclassFieldAuthorityComponent
from .exact_method_authority import (
    ExactLeafMethodAncestorPromotionComponent,
    ExactMethodRoleComponent,
    ParallelMirroredLeafFamilyComponent,
)
from .semantic_descent import (
    AuthorityClaim,
    SemanticAuthorityKind,
)
from .semantic_match import loaded_concrete_nominal_descendants
from .source_index import (
    AstTargetDigest,
    AstTargetNode,
    AstTargetNodeKind,
    SourceIndex,
)


@dataclass(frozen=True, kw_only=True)
class ClassMemberPromotionTargets(CodemodSourceSnapshot):
    """Resolved class nodes participating in a class-member promotion."""

    targets: tuple[ResolvedClassTarget, ...]

    @classmethod
    def resolve(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str | None,
        class_names: tuple[str, ...],
    ) -> "ClassMemberPromotionTargets":
        nodes_by_target_id = context.ast_target_nodes_by_id
        return cls(
            source_index=context.source_index,
            sources_by_file_path=context.sources_by_file_path,
            class_family_index=context.class_family_index,
            module_node_cache=context.module_nodes_by_file_path,
            ast_target_node_cache=nodes_by_target_id,
            targets=tuple(
                cls.class_target(
                    context.source_index,
                    nodes_by_target_id,
                    source_path=source_path,
                    class_name=class_name,
                )
                for class_name in class_names
            ),
        )

    @classmethod
    def require_new_authority(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        class_names: tuple[str, ...],
        authority_name: str,
    ) -> "ClassMemberPromotionTargets":
        """Resolve a cohort and prove one new local base can own its members."""

        targets = cls.resolve(
            context,
            source_path=source_path,
            class_names=class_names,
        )
        if not targets.supports_base_rewrites():
            raise ValueError("Class-member factoring requires lossless class headers")
        insertion_module = targets.module_nodes_by_file_path[
            targets.insertion_target.file_path
        ]
        if authority_name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            insertion_module.body
        ):
            raise ValueError(
                f"Class-member authority name {authority_name!r} is already bound"
            )
        return targets

    def new_authority_claim(self, authority_name: str) -> AuthorityClaim:
        """Derive the class-family claim established at this cohort's anchor."""

        return AuthorityClaim(
            claimed_symbol=authority_name,
            authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            file_path=self.insertion_target.file_path,
            qualname=authority_name,
        )

    @classmethod
    def resolve_or_none(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str | None,
        class_names: tuple[str, ...],
    ) -> "ClassMemberPromotionTargets | None":
        nodes_by_target_id = context.ast_target_nodes_by_id
        targets: list[ResolvedClassTarget] = []
        for class_name in class_names:
            target = cls.optional_class_target(
                context.source_index,
                nodes_by_target_id,
                source_path=source_path,
                class_name=class_name,
            )
            if target is None:
                return None
            targets.append(target)
        return cls(
            source_index=context.source_index,
            sources_by_file_path=context.sources_by_file_path,
            class_family_index=context.class_family_index,
            module_node_cache=context.module_nodes_by_file_path,
            ast_target_node_cache=nodes_by_target_id,
            targets=tuple(targets),
        )

    @classmethod
    def unresolved_class_target_reason(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str | None,
        class_names: tuple[str, ...],
    ) -> str:
        nodes_by_target_id = context.ast_target_nodes_by_id
        for class_name in class_names:
            reason = cls.optional_class_target_rejection_reason(
                context.source_index,
                nodes_by_target_id,
                source_path=source_path,
                class_name=class_name,
            )
            if reason is not None:
                return reason
        return "class targets are unresolved"

    @staticmethod
    def class_target(
        source_index: SourceIndex,
        nodes_by_target_id: Mapping[str, AstTargetNode],
        *,
        source_path: str | None,
        class_name: str,
    ) -> ResolvedClassTarget:
        matches = ClassMemberPromotionTargets.matching_class_targets(
            source_index,
            source_path=source_path,
            class_name=class_name,
        )
        if len(matches) != 1:
            raise ValueError(f"Expected one class target for {class_name!r}")
        target = matches[0]
        node = nodes_by_target_id[target.target_id]
        if not isinstance(node, ast.ClassDef):
            raise ValueError(f"Target {target.qualname!r} is not a class definition")
        return ResolvedClassTarget(target=target, node=node)

    @staticmethod
    def optional_class_target(
        source_index: SourceIndex,
        nodes_by_target_id: Mapping[str, AstTargetNode],
        *,
        source_path: str | None,
        class_name: str,
    ) -> ResolvedClassTarget | None:
        matches = ClassMemberPromotionTargets.matching_class_targets(
            source_index,
            source_path=source_path,
            class_name=class_name,
        )
        if len(matches) != 1:
            return None
        target = matches[0]
        node = nodes_by_target_id[target.target_id]
        if not isinstance(node, ast.ClassDef):
            return None
        return ResolvedClassTarget(target=target, node=node)

    @staticmethod
    def optional_class_target_rejection_reason(
        source_index: SourceIndex,
        nodes_by_target_id: Mapping[str, AstTargetNode],
        *,
        source_path: str | None,
        class_name: str,
    ) -> str | None:
        matches = ClassMemberPromotionTargets.matching_class_targets(
            source_index,
            source_path=source_path,
            class_name=class_name,
        )
        if len(matches) != 1:
            return f"Expected one class target for {class_name!r}"
        target = matches[0]
        node = nodes_by_target_id[target.target_id]
        if not isinstance(node, ast.ClassDef):
            return f"Target {target.qualname!r} is not a class definition"
        return None

    @staticmethod
    def matching_class_targets(
        source_index: SourceIndex,
        *,
        source_path: str | None,
        class_name: str,
    ) -> tuple[AstTargetDigest, ...]:
        resolved_source_path = (
            None
            if source_path is None
            else SourcePathResolutionAuthority.from_source_index(
                source_path,
                source_index,
            ).optional_path()
        )
        if source_path is not None and resolved_source_path is None:
            return ()
        return tuple(
            target
            for target in source_index.targets_matching_symbol(class_name)
            if target.is_class
            and (source_path is None or target.file_path == resolved_source_path)
        )

    @property
    def insertion_target(self) -> ResolvedClassTarget:
        return min(self.targets, key=lambda item: (item.file_path, item.line))

    @property
    def insertion_line(self) -> int:
        class_target = self.insertion_target
        decorator_lines = tuple(
            decorator.lineno for decorator in class_target.node.decorator_list
        )
        return min((*decorator_lines, class_target.line))

    @property
    def first_source(self) -> str:
        return self.source_for(self.insertion_target.file_path)

    def supports_base_rewrites(self) -> bool:
        return all(
            ClassHeaderSpanSourceAuthority(
                node=class_target.node,
                source=self.source_for(class_target.file_path),
            ).can_rewrite
            for class_target in self.targets
        )

    @cached_property
    def required_class_symbols(self) -> tuple[str, ...]:
        return tuple(target.required_symbol(self) for target in self.targets)

    @cached_property
    def indexed_classes(self) -> tuple[IndexedClass, ...]:
        indexed_classes = tuple(
            self.required_class_family_index.class_for(symbol)
            for symbol in self.required_class_symbols
        )
        if any(indexed_class is None for indexed_class in indexed_classes):
            raise ValueError("Method promotion requires indexed class-family targets")
        return tuple(
            indexed_class
            for indexed_class in indexed_classes
            if indexed_class is not None
        )

    @cached_property
    def shared_resolved_ancestor_symbols(self) -> frozenset[str]:
        ancestor_sets = tuple(
            set(self.required_class_family_index.ancestor_symbols(symbol))
            for symbol in self.required_class_symbols
        )
        if not ancestor_sets:
            return frozenset()
        return frozenset(set.intersection(*ancestor_sets))

    @cached_property
    def shared_declared_nominal_base_names(self) -> frozenset[str]:
        base_name_sets = tuple(
            {
                base_name
                for base_name in indexed_class.declared_base_names
                if ClassSymbolResolutionAuthority.establishes_nominal_family(base_name)
            }
            for indexed_class in self.indexed_classes
        )
        if not base_name_sets:
            return frozenset()
        return frozenset(set.intersection(*base_name_sets))

    def exact_method_declaration_failure(
        self,
        method_names: tuple[str, ...],
    ) -> str | None:
        """Return the first source-level obstacle shared by method promotions."""

        if any("." in target.qualname for target in self.targets):
            return "Method promotion requires top-level class targets"
        for class_target in self.targets:
            module = self.module_nodes_by_file_path[class_target.file_path]
            source_lines = tuple(
                self.source_for(class_target.file_path).splitlines(keepends=True)
            )
            module_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                module.body
            )
            class_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                class_target.node.body
            )
            for statement in class_target.node.body:
                if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if statement.name not in method_names:
                    continue
                profile = ClassMethodPromotionSafetyProfile.from_method(
                    statement,
                    module_bound_names,
                    class_bound_names,
                    source_lines=source_lines,
                )
                if profile.hazards:
                    return (
                        f"Method {class_target.qualname}.{statement.name} has "
                        "promotion hazards "
                        f"{tuple(hazard.value for hazard in profile.hazards)!r}"
                    )
        if not self.methods_match_exactly(method_names):
            return (
                "Method promotion requires one exact declaration source per method role"
            )
        return None

    def methods_match_exactly(self, method_names: tuple[str, ...]) -> bool:
        """Prove one complete declaration source for every promoted method role."""

        for method_name in method_names:
            shapes = []
            for class_target in self.targets:
                matching_methods = tuple(
                    statement
                    for statement in class_target.node.body
                    if ClassMethodPromotionStatement(statement).name == method_name
                )
                if len(matching_methods) != 1:
                    return False
                shapes.append(
                    ClassMethodPromotionStatement(
                        matching_methods[0],
                    ).source_from(self.source_for(class_target.file_path))
                )
            if len(frozenset(shapes)) != 1:
                return False
        return True

    def receiver_member_names(
        self,
        method_names: tuple[str, ...],
    ) -> frozenset[str]:
        return frozenset(
            member_name
            for class_target in self.targets
            for statement in class_target.node.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement.name in method_names
            for member_name in ClassMethodReceiverRequirements.from_method(
                statement
            ).member_names
        )

    def source_for(self, file_path: str) -> str:
        return self.sources_by_file_path[file_path]


@dataclass(frozen=True)
class ExactLeafMethodAncestorPromotionTargets:
    """Physical targets for one currently proven exact-method component."""

    component: ExactLeafMethodAncestorPromotionComponent
    authority: ResolvedClassTarget
    participants: ClassMemberPromotionTargets

    @classmethod
    def resolve(
        cls,
        context: CodemodSelectorContext,
        component: ExactLeafMethodAncestorPromotionComponent,
    ) -> "ExactLeafMethodAncestorPromotionTargets":
        resolved = ClassMemberPromotionTargets.resolve(
            context,
            source_path=component.file_path,
            class_names=(
                component.authority_name,
                *component.participant_class_names,
            ),
        )
        return cls(
            component=component,
            authority=resolved.targets[0],
            participants=replace(resolved, targets=resolved.targets[1:]),
        )

    def validation_failure(self) -> str | None:
        if not self.participants.targets:
            return "Existing-ancestor method promotion requires participating leaves"
        if "." in self.authority.qualname:
            return "Existing-ancestor method promotion requires a top-level authority"
        if any(
            target.file_path != self.authority.file_path
            for target in self.participants.targets
        ):
            return "Existing-ancestor method promotion requires one source file"
        declaration_failure = self.participants.exact_method_declaration_failure(
            self.component.method_names
        )
        if declaration_failure is not None:
            return declaration_failure
        return None


@dataclass(frozen=True)
class ParallelMirroredLeafFamilyTargets:
    """Physical class targets for one currently proven role product."""

    component: ParallelMirroredLeafFamilyComponent
    all_classes: ClassMemberPromotionTargets
    role_classes: tuple[ClassMemberPromotionTargets, ...]

    @classmethod
    def required_for_root_target(
        cls,
        snapshot: CodemodSourceSnapshot,
        root_target: AstTargetDigest,
    ) -> "ParallelMirroredLeafFamilyTargets":
        if not root_target.is_class:
            raise ValueError("parallel leaf-family authority target must be a class")
        root_symbol = snapshot.source_index.symbol_for_target(root_target)
        component = snapshot.parallel_mirrored_leaf_family_component_builder.required_proven_component(
            root_symbol
        )
        targets = cls.resolve(snapshot, component)
        failure = targets.validation_failure()
        if failure is not None:
            raise ValueError(failure)
        return targets

    @classmethod
    def resolve(
        cls,
        context: CodemodSelectorContext,
        component: ParallelMirroredLeafFamilyComponent,
    ) -> "ParallelMirroredLeafFamilyTargets":
        class_names = (
            *(root.qualname for root in component.roots),
            *(
                indexed_class.qualname
                for role in component.roles
                for indexed_class in role.classes
            ),
        )
        all_classes = ClassMemberPromotionTargets.resolve(
            context,
            source_path=component.roots[0].file_path,
            class_names=class_names,
        )
        targets_by_qualname = {
            target.qualname: target for target in all_classes.targets
        }
        return cls(
            component=component,
            all_classes=all_classes,
            role_classes=tuple(
                replace(
                    all_classes,
                    targets=tuple(
                        targets_by_qualname[indexed_class.qualname]
                        for indexed_class in role.classes
                    ),
                )
                for role in component.roles
            ),
        )

    def validation_failure(self) -> str | None:
        if not self.role_classes:
            return "Parallel leaf-family factoring requires proven role classes"
        authority_names = tuple(role.authority_name for role in self.component.roles)
        if len(frozenset(authority_names)) != len(authority_names):
            return "Parallel leaf-family role authority names are ambiguous"
        module = self.all_classes.module_nodes_by_file_path[
            self.component.roots[0].file_path
        ]
        bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.body)
        colliding_names = tuple(name for name in authority_names if name in bound_names)
        if colliding_names:
            return f"Role authority names are already bound: {colliding_names!r}"
        for role_targets in self.role_classes:
            if not role_targets.supports_base_rewrites():
                return "Parallel leaf-family factoring requires lossless class headers"
            declaration_failure = role_targets.exact_method_declaration_failure(
                self.component.contract_method_names
            )
            if declaration_failure is not None:
                return declaration_failure
        return None


@dataclass(frozen=True)
class ClassMemberSetSpec:
    """One typed set of class-body members."""

    member_names: tuple[str, ...]
    statement_type: type["ClassMemberPromotionStatement"]


@dataclass(frozen=True)
class ClassMemberPromotionSpec(ClassMemberSetSpec):
    """Shared member-promotion identity used by plans and generated bases."""

    base_name: str


@dataclass(frozen=True)
class ClassMemberDeletionReplacementPlan(ClassMemberSetSpec):
    """Delete promoted members from their former concrete owners."""

    rationale: str

    def source_edits(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[PhysicalSourceEdit, ...]:
        replacements = []
        for class_target in targets.targets:
            authority = StatementDeletionSource(
                source=targets.source_for(class_target.file_path),
                node=class_target.node,
                file_path=class_target.file_path,
            )
            replacements.extend(
                authority.physical_edits(
                    file_path=authority.file_path,
                    replacements=authority.replacements_for_statements(
                        self.promoted_statements(class_target.node)
                    ),
                    rationale=self.rationale
                    or f"Delete promoted members from {class_target.qualname!r}.",
                )
            )
        return tuple(replacements)

    def promoted_statements(self, node: ast.ClassDef) -> tuple[ast.stmt, ...]:
        return tuple(
            statement
            for statement in node.body
            if self.statement_type(statement).name in self.member_names
        )

@dataclass(frozen=True)
class ClassBaseAdditionReplacementPlan:
    """Add one nominal base to a resolved class cohort."""

    base_name: str
    rationale: str

    def source_edits(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[PhysicalSourceEdit, ...]:
        replacements = []
        for class_target in targets.targets:
            if self.base_name in _class_base_source_names(class_target.node):
                continue
            header_authority = ClassHeaderSpanSourceAuthority(
                node=class_target.node,
                source=targets.source_for(class_target.file_path),
            )
            replacements.extend(
                header_authority.source_edits(
                    header_authority.with_prepended_base(self.base_name),
                    file_path=class_target.file_path,
                    rationale=self.rationale
                    or f"Add base {self.base_name!r} to {class_target.qualname!r}.",
                )
            )
        return tuple(replacements)


@dataclass(frozen=True)
class ClassMemberPromotionReplacementPlanABC(ClassMemberPromotionSpec, ABC):
    """Shared rewrites for promoting class members into one nominal base."""

    rationale: str

    @abstractmethod
    def promoted_base_source(self, targets: ClassMemberPromotionTargets) -> str:
        raise NotImplementedError

    def promoted_member_source(self, targets: ClassMemberPromotionTargets) -> str:
        """Derive the complete selected member source from the insertion owner."""

        return "".join(
            ClassMemberSourceSelection(
                member_names=self.member_names,
                statement_type=self.statement_type,
                source_text=targets.first_source,
                source_class=targets.insertion_target.node,
            ).member_sources
        )

    def source_edits(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return (
            self.base_insertion_replacement(targets),
            *ClassBaseAdditionReplacementPlan(
                base_name=self.base_name,
                rationale=self.rationale,
            ).source_edits(targets),
            *ClassMemberDeletionReplacementPlan(
                member_names=self.member_names,
                statement_type=self.statement_type,
                rationale=self.rationale,
            ).source_edits(targets),
        )

    def base_insertion_replacement(
        self,
        targets: ClassMemberPromotionTargets,
    ) -> SourceInsertion:
        class_target = targets.insertion_target
        base_source = self.promoted_base_source(targets)
        return SourceInsertion(
            file_path=class_target.file_path,
            insertion_line=targets.insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(f"{base_source}\n\n"),
            rationale=self.rationale
            or f"Insert promoted-member base {self.base_name!r}.",
        )


@dataclass(frozen=True)
class ClassMemberSourceSelection(ClassMemberSetSpec):
    """Exact source for a proved set of class-body members."""

    source_text: str
    source_class: ast.ClassDef

    @cached_property
    def member_sources(self) -> tuple[str, ...]:
        members = tuple(
            self.statement_type(statement).source_from(self.source_text)
            for statement in self.source_class.body
            if self.statement_type(statement).name in self.member_names
        )
        if len(members) != len(self.member_names):
            raise ValueError(
                f"Could not find promoted members {self.member_names!r} "
                f"on {self.source_class.name!r}"
            )
        return members


@dataclass(frozen=True)
class LayoutNeutralClassMemberPromotionReplacementPlan(
    ClassMemberPromotionReplacementPlanABC
):
    """Promote behavior into a layout-neutral mixin authority."""

    def promoted_base_source(self, targets: ClassMemberPromotionTargets) -> str:
        return (
            f"class {self.base_name}:\n"
            f"    __slots__ = ()\n\n"
            f"{self.promoted_member_source(targets)}"
        )


@dataclass(frozen=True)
class DataclassFieldPromotionReplacementPlan(ClassMemberPromotionReplacementPlanABC):
    """Promote exact fields into a standard dataclass authority."""

    decorator_source: str

    def promoted_base_source(self, targets: ClassMemberPromotionTargets) -> str:
        return (
            f"@{self.decorator_source}\n"
            f"class {self.base_name}:\n"
            f"{self.promoted_member_source(targets)}"
        )


@dataclass(frozen=True)
class _ExactLeafMethodAncestorPromotionSourceRewrite:
    """Source edits derived from one currently proven method component."""

    targets: ExactLeafMethodAncestorPromotionTargets
    rationale: str

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        return (
            self.authority_replacement(),
            *ClassMemberDeletionReplacementPlan(
                member_names=self.targets.component.method_names,
                statement_type=ClassMethodPromotionStatement,
                rationale=self.rationale,
            ).source_edits(self.targets.participants),
        )

    def authority_replacement(self) -> SourceSpanReplacement:
        authority = self.targets.authority
        source = self.targets.participants.source_for(authority.file_path)
        source_class = self.targets.participants.targets[0].node
        member_sources = ClassMemberSourceSelection(
            member_names=self.targets.component.method_names,
            statement_type=ClassMethodPromotionStatement,
            source_text=source,
            source_class=source_class,
        ).member_sources
        insertion_point = ClassBodySourceAuthority(
            node=authority.node,
            source=source,
        )
        replacement_source = SourceTextGeometry(source).target_source_with_replacements(
            authority.target,
            (insertion_point.member_insertion_replacement(member_sources),),
        )
        return SourceSpanReplacement(
            file_path=authority.file_path,
            start_line=authority.target.line,
            end_line=authority.target.end_line,
            replacement_lines=SourceTargetEditor.source_lines(replacement_source),
            rationale=self.rationale
            or f"Move exact shared methods to {authority.qualname!r}.",
        )


@dataclass(frozen=True)
class NamedClassMemberAuthoritySourceRewriteABC(ABC):
    """Shared claim surface for one source-proved class-member authority."""

    targets: ClassMemberPromotionTargets
    base_name: str
    rationale: str

    @property
    def authority_claim(self) -> AuthorityClaim:
        return self.targets.new_authority_claim(self.base_name)

    @abstractmethod
    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class ExactDataclassFieldEvidence:
    """One source anchor that re-proves an exact repeated-field component."""

    field_name: str

    def __post_init__(self) -> None:
        if not self.field_name.isidentifier() or keyword_module.iskeyword(
            self.field_name
        ):
            raise ValueError("Evidence field name must be an identifier")

    def required_component(
        self,
        snapshot: CodemodSourceSnapshot,
        target: AstTargetDigest,
    ) -> ExactDataclassFieldAuthorityComponent:
        target.require_kind(
            AstTargetNodeKind.CLASS,
            "Exact dataclass field factoring requires a class target",
        )
        return snapshot.exact_dataclass_field_authority_component_builder.required_component_for_field(
            file_path=target.file_path,
            class_qualname=target.qualname,
            field_name=self.field_name,
        )


@dataclass(frozen=True)
class _ExactDataclassFieldAuthoritySourceRewrite(
    NamedClassMemberAuthoritySourceRewriteABC
):
    """Physical rewrite derived from one current repeated-field proof."""

    component: ExactDataclassFieldAuthorityComponent

    @classmethod
    def required(
        cls,
        snapshot: CodemodSourceSnapshot,
        component: ExactDataclassFieldAuthorityComponent,
        *,
        base_name: str,
        rationale: str,
    ) -> "_ExactDataclassFieldAuthoritySourceRewrite":
        targets = ClassMemberPromotionTargets.require_new_authority(
            snapshot,
            source_path=component.file_path,
            class_names=component.participant_class_names,
            authority_name=base_name,
        )
        return cls(targets, base_name, rationale, component)

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        return DataclassFieldPromotionReplacementPlan(
            base_name=self.base_name,
            member_names=self.component.field_names,
            statement_type=ClassDeclarationPromotionStatement,
            rationale=self.rationale,
            decorator_source=self.component.decorator_source,
        ).source_edits(self.targets)


@dataclass(frozen=True, kw_only=True)
class FactorNamedClassMemberAuthorityOperationABC(
    RepositorySourceReprovedOperation,
    ABC,
):
    """Shared execution shell for a newly named, source-reproved member owner."""

    base_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        if not self.base_name.isidentifier() or keyword_module.iskeyword(
            self.base_name
        ):
            raise ValueError("Class-member authority name must be an identifier")

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self._source_rewrite(snapshot).source_edits()

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        return (self._source_rewrite(context.execution_snapshot()).authority_claim,)

    @abstractmethod
    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> NamedClassMemberAuthoritySourceRewriteABC:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class FactorExactDataclassFieldAuthorityOperation(
    FactorNamedClassMemberAuthorityOperationABC
):
    """Re-prove repeated leading fields and give them one dataclass authority."""

    evidence_field_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        super().__post_init__()
        ExactDataclassFieldEvidence(self.evidence_field_name)

    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _ExactDataclassFieldAuthoritySourceRewrite:
        component = self.required_component(snapshot)
        return _ExactDataclassFieldAuthoritySourceRewrite.required(
            snapshot,
            component,
            base_name=self.base_name,
            rationale=self.rationale,
        )

    def required_component(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> ExactDataclassFieldAuthorityComponent:
        _target_id, target, _node = self.target_node_from_context(snapshot)
        return ExactDataclassFieldEvidence(self.evidence_field_name).required_component(
            snapshot,
            target,
        )


@dataclass(frozen=True)
class ExistingDataclassFieldAuthorityTargets:
    """A behavior-free field owner and every class that should descend from it."""

    component: ExactDataclassFieldAuthorityComponent
    authority: ResolvedClassTarget
    participants: ClassMemberPromotionTargets

    @classmethod
    def required(
        cls,
        snapshot: CodemodSourceSnapshot,
        component: ExactDataclassFieldAuthorityComponent,
        authority: ResolvedClassTarget,
    ) -> "ExistingDataclassFieldAuthorityTargets":
        authority_participants = tuple(
            participant
            for participant in component.participants
            if participant.indexed_class.qualname == authority.qualname
        )
        if len(authority_participants) != 1:
            raise ValueError(
                "Existing field authority must belong to the proved component"
            )
        if authority_participants[0].fields != component.fields:
            raise ValueError(
                "Existing field authority must own exactly the repeated fields"
            )
        executable_body = tuple(statements_without_docstring(authority.node.body))
        if (
            tuple(
                ClassDeclarationPromotionStatement(statement).name
                for statement in executable_body
            )
            != component.field_names
        ):
            raise ValueError(
                "Existing field authority must be behavior-free outside its fields"
            )

        resolved = ClassMemberPromotionTargets.resolve(
            snapshot,
            source_path=component.file_path,
            class_names=component.participant_class_names,
        )
        resolved_authorities = tuple(
            target
            for target in resolved.targets
            if target.target.target_id == authority.target.target_id
        )
        if len(resolved_authorities) != 1:
            raise ValueError("Existing field authority source target is ambiguous")
        participants = replace(
            resolved,
            targets=tuple(
                target
                for target in resolved.targets
                if target.target.target_id != authority.target.target_id
            ),
        )
        if not participants.targets:
            raise ValueError("Existing field authority has no participating classes")
        if not participants.supports_base_rewrites():
            raise ValueError("Existing field authority requires lossless class headers")
        targets = cls(component, resolved_authorities[0], participants)
        targets.require_safe_relocation(snapshot)
        return targets

    @property
    def authority_name(self) -> str:
        return self.authority.node.name

    @cached_property
    def authority_span(self) -> SourceLineSpan:
        return SourceTextGeometry(
            self.participants.source_for(self.authority.file_path)
        ).node_line_span(
            SourceNodeSpan(
                self.authority.node,
                SourceNodeDecoratorPolicy.INCLUDE,
            )
        )

    @property
    def requires_relocation(self) -> bool:
        return self.authority_span.start_line > self.participants.insertion_line

    def require_safe_relocation(self, snapshot: CodemodSourceSnapshot) -> None:
        if not self.requires_relocation:
            return
        source = self.participants.source_for(self.authority.file_path)
        source_lines = source.splitlines()
        preceding_separator = source_lines[
            self.authority_span.start_line - 3 : self.authority_span.start_line - 1
        ]
        if len(preceding_separator) != 2 or any(
            line.strip() for line in preceding_separator
        ):
            raise ValueError(
                "Existing field authority relocation requires a complete "
                "top-level separator"
            )
        module = snapshot.module_nodes_by_file_path[self.authority.file_path]
        intervening_statements = tuple(
            statement
            for statement in module.body
            if self.participants.insertion_line
            <= statement.lineno
            < self.authority_span.start_line
        )
        if EagerNameLoadCollector.collect(
            module,
            self.authority_name,
            intervening_statements,
        ):
            raise ValueError(
                "Existing field authority is referenced before its current declaration"
            )
        preceding_statements = tuple(
            statement
            for statement in module.body
            if statement.lineno < self.participants.insertion_line
        )
        if self.authority_name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            preceding_statements
        ):
            raise ValueError(
                "Existing field authority name is already bound before relocation"
            )

    def relocation_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        if not self.requires_relocation:
            return ()
        source = self.participants.source_for(self.authority.file_path)
        authority_source = self.authority_span.source_from(source)
        rationale = f"Move field authority {self.authority_name!r} before its users."
        return (
            SourceInsertion(
                file_path=self.authority.file_path,
                insertion_line=self.participants.insertion_line,
                inserted_lines=(
                    *SourceTargetEditor.source_lines(authority_source),
                    "\n",
                    "\n",
                ),
                rationale=rationale,
            ),
            SourceSpanDeletion(
                file_path=self.authority.file_path,
                start_line=self.authority_span.start_line - 2,
                end_line=self.authority_span.end_line,
                rationale=rationale,
            ),
        )


@dataclass(frozen=True)
class _ExistingDataclassFieldAuthoritySourceRewrite:
    """Physical rewrite descending a field cohort from its existing owner."""

    targets: ExistingDataclassFieldAuthorityTargets
    rationale: str

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        return (
            *self.targets.relocation_edits(),
            *ClassBaseAdditionReplacementPlan(
                base_name=self.targets.authority_name,
                rationale=self.rationale,
            ).source_edits(self.targets.participants),
            *ClassMemberDeletionReplacementPlan(
                member_names=self.targets.component.field_names,
                statement_type=ClassDeclarationPromotionStatement,
                rationale=self.rationale,
            ).source_edits(self.targets.participants),
        )


@dataclass(frozen=True, kw_only=True)
class PromoteExactDataclassFieldsToExistingAuthorityOperation(
    RepositorySourceReprovedOperation
):
    """Re-prove repeated fields and descend their cohort from an existing owner."""

    evidence_field_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        ExactDataclassFieldEvidence(self.evidence_field_name)

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self._source_rewrite(snapshot).source_edits()

    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _ExistingDataclassFieldAuthoritySourceRewrite:
        _target_id, target, node = self.target_node_from_context(snapshot)
        if not isinstance(node, ast.ClassDef):
            raise ValueError("Existing field authority target must be a class")
        component = ExactDataclassFieldEvidence(
            self.evidence_field_name
        ).required_component(snapshot, target)
        return _ExistingDataclassFieldAuthoritySourceRewrite(
            targets=ExistingDataclassFieldAuthorityTargets.required(
                snapshot,
                component,
                ResolvedClassTarget(target, node),
            ),
            rationale=self.rationale,
        )


@dataclass(frozen=True)
class _ExactMethodRoleSourceRewrite(NamedClassMemberAuthoritySourceRewriteABC):
    """Physical rewrite derived from one currently proven method role."""

    component: ExactMethodRoleComponent

    @classmethod
    def required(
        cls,
        snapshot: CodemodSourceSnapshot,
        component: ExactMethodRoleComponent,
        *,
        base_name: str,
        rationale: str,
    ) -> "_ExactMethodRoleSourceRewrite":
        targets = ClassMemberPromotionTargets.require_new_authority(
            snapshot,
            source_path=component.file_path,
            class_names=component.participant_class_names,
            authority_name=base_name,
        )
        return cls(targets, base_name, rationale, component)

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        return LayoutNeutralClassMemberPromotionReplacementPlan(
            base_name=self.base_name,
            member_names=self.component.method_names,
            statement_type=ClassMethodPromotionStatement,
            rationale=self.rationale,
        ).source_edits(self.targets)


@dataclass(frozen=True, kw_only=True)
class FactorExactMethodRoleOperation(FactorNamedClassMemberAuthorityOperationABC):
    """Re-prove one exact-method cohort and give it a named MI authority."""

    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _ExactMethodRoleSourceRewrite:
        component = self.required_component(snapshot)
        return _ExactMethodRoleSourceRewrite.required(
            snapshot,
            component,
            base_name=self.base_name,
            rationale=self.rationale,
        )

    def required_component(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> ExactMethodRoleComponent:
        _target_id, target, _node = self.target_node_from_context(snapshot)
        target.require_kind(
            AstTargetNodeKind.METHOD,
            "Exact-method role factoring requires a method target",
        )
        return (
            snapshot.exact_method_role_component_builder.required_component_for_method(
                file_path=target.file_path,
                method_qualname=target.qualname,
            )
        )


@dataclass(frozen=True, kw_only=True)
class PromoteExactLeafMethodsToAncestorOperation(RepositorySourceReprovedOperation):
    """Re-prove and promote one authority-wide exact leaf-method component."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self._source_rewrite(snapshot).source_edits()

    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _ExactLeafMethodAncestorPromotionSourceRewrite:
        _target_identifier, authority_target, _authority_node = (
            self.target_node_from_context(snapshot)
        )
        if not authority_target.is_class:
            raise ValueError("exact method authority target must be a class")
        authority_symbol = snapshot.source_index.symbol_for_target(authority_target)
        component = (
            snapshot.exact_leaf_method_component_builder.required_proven_component(
                authority_symbol
            )
        )
        targets = ExactLeafMethodAncestorPromotionTargets.resolve(
            snapshot,
            component,
        )
        failure = targets.validation_failure()
        if failure is not None:
            raise ValueError(failure)
        return _ExactLeafMethodAncestorPromotionSourceRewrite(
            targets=targets,
            rationale=self.rationale,
        )


@dataclass(frozen=True)
class _ParallelMirroredLeafFamilySourceRewrite:
    """Generic promotion plans composed for every proved role axis."""

    targets: ParallelMirroredLeafFamilyTargets
    rationale: str

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        return tuple(
            edit
            for role, role_targets in zip(
                self.targets.component.roles,
                self.targets.role_classes,
                strict=True,
            )
            for edit in LayoutNeutralClassMemberPromotionReplacementPlan(
                base_name=role.authority_name,
                member_names=self.targets.component.contract_method_names,
                statement_type=ClassMethodPromotionStatement,
                rationale=self.rationale,
            ).source_edits(role_targets)
        )


@dataclass(frozen=True, kw_only=True)
class FactorParallelMirroredLeafFamilyOperation(RepositorySourceReprovedOperation):
    """Re-prove and factor parallel leaf behavior into MI role authorities."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self._source_rewrite(snapshot).source_edits()

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        component = self.required_targets(context.execution_snapshot()).component
        return tuple(
            AuthorityClaim(
                claimed_symbol=role.authority_name,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                file_path=component.file_path,
                qualname=role.authority_name,
            )
            for role in component.roles
        )

    def required_targets(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> ParallelMirroredLeafFamilyTargets:
        _target_identifier, root_target, _root_node = self.target_node_from_context(
            snapshot
        )
        return ParallelMirroredLeafFamilyTargets.required_for_root_target(
            snapshot,
            root_target,
        )

    def _source_rewrite(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _ParallelMirroredLeafFamilySourceRewrite:
        return _ParallelMirroredLeafFamilySourceRewrite(
            targets=self.required_targets(snapshot),
            rationale=self.rationale,
        )


@dataclass(frozen=True)
class ClassMemberPromotionStatement(ABC):
    """Class-body statement projected as a promotable member."""

    statement: ast.stmt

    @classmethod
    def for_statement(
        cls,
        statement: ast.stmt,
    ) -> "ClassMemberPromotionStatement | None":
        """Resolve one promotable member through its nominal statement family."""

        matches = tuple(
            projection
            for projection_type in loaded_concrete_nominal_descendants(cls)
            if (projection := projection_type(statement)).name is not None
        )
        if len(matches) > 1:
            raise ValueError(
                "Class member statement has competing promotable declaration "
                f"projection; found {len(matches)}"
            )
        return matches[0] if matches else None

    @property
    @abstractmethod
    def name(self) -> str | None:
        raise NotImplementedError

    @property
    def source_span(self) -> SourceNodeSpan:
        return SourceNodeSpan(self.statement, SourceNodeDecoratorPolicy.INCLUDE)

    @property
    def end_line(self) -> int:
        return self.statement.end_lineno or self.statement.lineno

    def source_from(self, source: str, *, indentation: str = "    ") -> str:
        """Return the complete member source selected by this declaration."""

        return StatementSource(source=source, node=self.statement).member_source(
            indentation
        )

    def require_safe_move(
        self,
        context: "ClassMemberMoveProofContext",
    ) -> None:
        """Reject ownership-sensitive syntax before moving this member."""

        context.require_no_attached_leading_comment(self)


@dataclass(frozen=True)
class ClassDeclarationPromotionStatement(ClassMemberPromotionStatement):
    """Class-body declaration eligible for declaration promotion."""

    @property
    def name(self) -> str | None:
        if isinstance(self.statement, ast.Assign):
            if len(self.statement.targets) != 1:
                return None
            target = self.statement.targets[0]
            if isinstance(target, ast.Name):
                return target.id
        if isinstance(self.statement, ast.AnnAssign) and isinstance(
            self.statement.target,
            ast.Name,
        ):
            return self.statement.target.id
        return None

    def require_safe_move(
        self,
        context: "ClassMemberMoveProofContext",
    ) -> None:
        super().require_safe_move(context)
        member_name = self.name
        if member_name is None:
            raise ValueError("Class declaration does not bind one direct member")
        if member_name.startswith("__") and not member_name.endswith("__"):
            raise ValueError(
                f"Class member {member_name!r} has owner-dependent name mangling"
            )
        class_local_references = frozenset(
            node.id
            for node in ast.walk(self.statement)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in context.source_class_bound_names
        )
        if class_local_references:
            raise ValueError(
                f"Class member {member_name!r} has class-local references "
                f"{tuple(sorted(class_local_references))!r}"
            )


@dataclass(frozen=True)
class ClassMethodPromotionStatement(ClassMemberPromotionStatement):
    """Class-body method eligible for method promotion."""

    @property
    def name(self) -> str | None:
        if isinstance(self.statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self.statement.name
        return None

    def require_safe_move(
        self,
        context: "ClassMemberMoveProofContext",
    ) -> None:
        super().require_safe_move(context)
        if not isinstance(self.statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError("Class method projection does not contain a method")
        profile = ClassMethodPromotionSafetyProfile.from_method(
            self.statement,
            context.module_bound_names,
            context.source_class_bound_names,
            source_lines=context.source_lines,
        )
        if profile.hazards:
            raise ValueError(
                f"Class method {self.statement.name!r} has promotion hazards "
                f"{tuple(hazard.value for hazard in profile.hazards)!r}"
            )


@dataclass(frozen=True)
class ClassMemberMoveProofContext(SourceTextGeometry):
    """Current-source facts shared by every selected class member."""

    source_class: ResolvedClassTarget
    destination_class: ResolvedClassTarget
    module_bound_names: frozenset[str]
    source_class_bound_names: frozenset[str]

    @property
    def source_lines(self) -> tuple[str, ...]:
        return self.lines

    def require_no_attached_leading_comment(
        self,
        member: ClassMemberPromotionStatement,
    ) -> None:
        start_line = self.node_start_line(member.source_span)
        if start_line <= self.source_class.node.lineno + 1:
            return
        member_line = self.lines[start_line - 1]
        preceding_line = self.lines[start_line - 2]
        indentation = member_line[: len(member_line) - len(member_line.lstrip())]
        if preceding_line.startswith(indentation) and preceding_line.removeprefix(
            indentation
        ).startswith("#"):
            raise ValueError(
                f"Class member {member.name!r} has an attached leading comment"
            )


@dataclass(frozen=True)
class ClassMemberMoveSelection:
    """One source-proved member set moving upward to an existing authority."""

    context: ClassMemberMoveProofContext
    members: tuple[ClassMemberPromotionStatement, ...]

    @classmethod
    def require(
        cls,
        snapshot: CodemodSourceSnapshot,
        source_class: ResolvedClassTarget,
        destination_class: ResolvedClassTarget,
        member_names: tuple[str, ...],
    ) -> "ClassMemberMoveSelection":
        if source_class.file_path != destination_class.file_path:
            raise ValueError(
                "Class-member promotion currently requires one source module"
            )
        if source_class.target.target_id == destination_class.target.target_id:
            raise ValueError("Class-member promotion requires distinct class owners")
        source_symbol = source_class.required_symbol(snapshot)
        destination_symbol = destination_class.required_symbol(snapshot)
        if (
            destination_symbol
            not in snapshot.required_class_family_index.ancestor_symbols(source_symbol)
        ):
            raise ValueError(
                f"Destination class {destination_class.qualname!r} is not an ancestor "
                f"of {source_class.qualname!r}"
            )
        source = snapshot.sources_by_file_path[source_class.file_path]
        module = snapshot.module_nodes_by_file_path[source_class.file_path]
        context = ClassMemberMoveProofContext(
            source_class=source_class,
            destination_class=destination_class,
            source=source,
            module_bound_names=LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.body),
            source_class_bound_names=LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                source_class.node.body
            ),
        )
        requested_names = frozenset(member_names)
        available_members: list[ClassMemberPromotionStatement] = []
        for statement in source_class.node.body:
            projection = ClassMemberPromotionStatement.for_statement(statement)
            if projection is None or projection.name not in requested_names:
                continue
            available_members.append(projection)
        resolved_names = tuple(member.name for member in available_members)
        unresolved_names = requested_names - set(resolved_names)
        if unresolved_names:
            raise ValueError(
                "Class-member promotion cannot resolve direct declarations "
                f"{tuple(sorted(unresolved_names))!r}"
            )
        if len(resolved_names) != len(frozenset(resolved_names)):
            raise ValueError("Class-member promotion found rebound member declarations")
        members = tuple(available_members)
        destination_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            destination_class.node.body
        )
        collisions = destination_names.intersection(member_names)
        if collisions:
            raise ValueError(
                f"Destination class {destination_class.qualname!r} already binds "
                f"{tuple(sorted(collisions))!r}"
            )
        for member in members:
            member.require_safe_move(context)
        return cls(context=context, members=members)

    def source_edits(self, rationale: str) -> tuple[PhysicalSourceEdit, ...]:
        source = self.context.source
        destination = self.context.destination_class
        body_authority = ClassBodySourceAuthority(destination.node, source)
        deletion = StatementDeletionSource(
            source=source,
            node=self.context.source_class.node,
            file_path=self.context.source_class.file_path,
        )
        return (
            *body_authority.geometry.physical_edits(
                file_path=destination.file_path,
                replacements=(
                    body_authority.member_insertion_replacement(
                        tuple(
                            member.source_from(
                                source, indentation=body_authority.indentation
                            )
                            for member in self.members
                        )
                    ),
                ),
                rationale=rationale
                or f"Promote class members into {destination.qualname!r}.",
            ),
            *deletion.physical_edits(
                file_path=deletion.file_path,
                replacements=deletion.replacements_for_statements(
                    tuple(member.statement for member in self.members)
                ),
                rationale=rationale
                or f"Remove promoted members from {self.context.source_class.qualname!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class PromoteClassMembersToAncestorOperation(RepositorySourceReprovedOperation):
    """Move selected direct members into one existing nominal ancestor."""

    destination: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    member_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def __post_init__(self) -> None:
        if not self.member_names:
            raise ValueError("Class-member promotion requires member_names")
        if len(frozenset(self.member_names)) != len(self.member_names):
            raise ValueError("Class-member promotion requires unique member_names")
        if any(
            not name.isidentifier() or keyword_module.iskeyword(name)
            for name in self.member_names
        ):
            raise ValueError("Class-member promotion requires Python identifiers")

    def selection(self, snapshot: CodemodSourceSnapshot) -> ClassMemberMoveSelection:
        return ClassMemberMoveSelection.require(
            snapshot,
            ResolvedClassTarget.from_rewrite_target(snapshot, self.target),
            ResolvedClassTarget.from_rewrite_target(snapshot, self.destination),
            self.member_names,
        )

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self.selection(snapshot).source_edits(self.rationale)

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        selection = self.selection(context.execution_snapshot())
        return (
            AstTargetAuthorityClaim.from_target(
                selection.context.destination_class.target,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            ),
        )


def _class_base_source_names(node: ast.ClassDef) -> frozenset[str]:
    return frozenset(ast.unparse(base) for base in node.bases)
