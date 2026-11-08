from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import (
    dataclass,
    field,
)
from typing import ClassVar

from ..assignment_projection import (
    AssignmentStatementNameProjection,
    SingleAssignmentAndValueNameProjection,
)
from ..ast_projection import ExpressionReference
from ..class_member_authority_codemod import (
    ClassDeclarationPromotionStatement,
    ClassMemberPromotionTargets,
)
from ..codemod import (
    DeleteClassAssignmentsOperation,
    DeleteInheritedAutoRegisterConfigurationOperation,
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    InsertAfterTargetOperation,
    PatchTargetOperation,
    RegistrationSemanticMirrorRecipeBuilder,
    ResolvedFunctionProjectionTarget,
    SemanticMirrorFindingRecipeStrategy,
    SemanticMirrorRecipeSeedLocations,
    SingleSourcePathFindingMixin,
)
from ..codemod_authority_claims import AstTargetAuthorityClaim
from ..codemod_declaration_source import (
    ClassBodyTailSourceAuthority,
    ClassHeaderSpanSourceAuthority,
)
from ..codemod_native_requirements import (
    DeclaredNativeUseInvariants,
    NativeDefinitionUseRequirement,
    NativeUseRequirement,
)
from ..codemod_payload import (
    PayloadRecordValueCodec,
    codemod_payload_field,
)
from ..codemod_preflight import CodemodOperationPreflightReport
from ..codemod_reproof import (
    RepositorySourceReprovedOperation,
    SourceReprovedOperation,
)
from ..codemod_runtime import (
    CodemodPlanDocumentSimulation,
    CodemodSourceSnapshot,
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
from ..codemod_source_edits import (
    NominalSourceEdit,
    PhysicalSourceEdit,
    SourceInsertion,
    SourceLineSpan,
    SourceSpanDeletion,
    SourceSpanReplacement,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceTextMutation,
    SourceTextReplacement,
)
from ..codemod_target_selectors import SourceIndexTargetSelector
from ..collection_algebra import sorted_tuple
from ..enum_semantics import PYTHON_ENUM_BASE_AUTHORITY
from ..finding_recipe_actions import FindingRecipeActionKey
from ..foreign_family import FunctionDefinition
from ..lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from ..manual_registry import (
    AutoRegisterInstanceViewComponent,
    DirectManualRegistryComponent,
    RegistryAssignment,
    SourceClassKeyEntry,
)
from ..models import (
    MappingMetrics,
    RefactorFinding,
    RegistrationMetrics,
)
from ..refactor_concepts import (
    AutoRegisterClassRegistryConcept,
    AutoRegisterConcept,
    AutoRegisterMroOrderingConcept,
)
from ..registry_identity import (
    AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
    AUTOREGISTER_META_NAME,
    AutoRegisterClassAuthority,
    DEFAULT_REGISTRY_KEY_ATTRIBUTE,
    REGISTRY_ATTRIBUTE_NAME,
    REGISTRY_KEY_ATTRIBUTE_NAME,
)
from ..semantic_descent import (
    AuthorityClaim,
    SemanticAuthorityKind,
)
from ..source_index import (
    AstTargetDigest,
    AstTargetNode,
    AstTargetNodeKind,
)
from ..value_expression import CompactValueExpression


class RegistryKeyDeclarationRewriteMixin:
    """Reuse exact class-key declaration rewrites across registry operations."""

    def registry_key_declaration_replacements(
        self,
        targets: ClassMemberPromotionTargets,
        entries: tuple[SourceClassKeyEntry, ...],
        registry_key_attribute: str,
    ) -> tuple[SourceTextMutation, ...]:
        entries_by_class = {entry.class_name: entry for entry in entries}
        replacements = []
        for class_target in targets.targets:
            entry = entries_by_class[class_target.node.name]
            existing = tuple(
                statement
                for statement in class_target.node.body
                if ClassDeclarationPromotionStatement(statement).name
                == registry_key_attribute
            )
            if existing:
                if len(existing) != 1 or not self.declaration_matches_value(
                    existing[0], entry.key_node
                ):
                    raise ValueError(
                        f"Registry key on {class_target.qualname!r} conflicts with "
                        "the source registry"
                    )
                continue
            replacements.append(
                self.registry_key_declaration_replacement(
                    targets,
                    class_target,
                    entry,
                    registry_key_attribute,
                )
            )
        return tuple(replacements)

    def registry_key_declaration_replacement(
        self,
        targets: ClassMemberPromotionTargets,
        target: ResolvedClassTarget,
        entry: SourceClassKeyEntry,
        registry_key_attribute: str,
    ) -> SourceTextMutation:
        body = ClassBodyTailSourceAuthority(
            target.node, targets.source_for(target.file_path)
        )
        member = f"{body.indentation}{registry_key_attribute} = {entry.key_source}\n"
        return body.geometry.nominal_edit(
            file_path=target.file_path,
            replacements=(body.member_insertion_replacement((member,)),),
            rationale=self.rationale_text(
                f"Declare registry key on {target.qualname!r}."
            ),
        )

    @staticmethod
    def declaration_matches_value(statement: ast.stmt, expected: ast.expr) -> bool:
        value = (
            statement.value
            if isinstance(statement, ast.Assign | ast.AnnAssign)
            else None
        )
        return value is not None and ast.dump(
            value, include_attributes=False
        ) == ast.dump(expected, include_attributes=False)


@dataclass(frozen=True, kw_only=True)
class DeriveAutoregisterInstanceViewOperation(
    RegistryKeyDeclarationRewriteMixin,
    SourceReprovedOperation,
):
    """Derive an instance-valued module view from an AutoRegisterMeta family."""

    instance_view_method_name: ClassVar[str] = "instances_by_registry_key"

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        _target_id, authority_digest, authority_node = self.target_node_from_context(
            snapshot
        )
        if not authority_digest.is_class or not isinstance(
            authority_node, ast.ClassDef
        ):
            raise ValueError("Instance-view derivation target must be a class")
        if "." in authority_digest.qualname:
            raise ValueError("Instance-view derivation requires a top-level authority")
        source_path = authority_digest.file_path
        component = AutoRegisterInstanceViewComponent.from_module_authority(
            snapshot.module_nodes_by_file_path[source_path],
            authority_node.name,
        )
        component.require_destination_key_compatibility(component.authority)
        authority_target = ResolvedClassTarget(
            target=authority_digest,
            node=component.authority_node,
        )
        authority_replacements = self.authority_replacements(
            authority_target,
            component,
            snapshot.sources_by_file_path,
        )
        component.require_original_entry_values(
            snapshot.module_binding_proof.native_reference_environment(
                snapshot.parsed_module_for_source_path(source_path)
            )
        )
        concrete_targets = ClassMemberPromotionTargets.resolve(
            snapshot,
            source_path=source_path,
            class_names=component.class_names,
        )
        return (
            *self.registry_key_declaration_replacements(
                concrete_targets,
                component.entries,
                component.registry_key_attribute,
            ),
            *authority_replacements,
            self.assignment_replacement(source_path, component),
        )

    def authority_replacements(
        self,
        authority_target: ResolvedClassTarget,
        component: AutoRegisterInstanceViewComponent,
        source_by_path: Mapping[str, str],
    ) -> tuple[SourceTextMutation, ...]:
        if (
            self.instance_view_method_name
            in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                component.authority_node.body
            )
        ):
            raise ValueError(
                f"AutoRegister authority {authority_target.qualname!r} already binds "
                f"{self.instance_view_method_name!r}"
            )
        body = ClassBodyTailSourceAuthority(
            component.authority_node, source_by_path[authority_target.file_path]
        )
        members = [self.instance_method_source(body.indentation)]
        if not component.authority.declares_registry:
            members.insert(0, f"{body.indentation}{REGISTRY_ATTRIBUTE_NAME} = {{}}\n")
        return (
            body.geometry.nominal_edit(
                file_path=authority_target.file_path,
                replacements=(body.member_insertion_replacement(tuple(members)),),
                rationale=self.rationale_text(
                    f"Derive {self.instance_view_method_name!r} from {authority_target.qualname!r}."
                ),
            ),
        )

    def instance_method_source(
        self,
        indent: str,
    ) -> str:
        return (
            "\n"
            f"{indent}@classmethod\n"
            f"{indent}def {self.instance_view_method_name}(cls):\n"
            f"{indent}    key_attribute = cls.{REGISTRY_KEY_ATTRIBUTE_NAME}\n"
            f"{indent}    return {{\n"
            f"{indent}        registered_type.__dict__[key_attribute]: registered_type()\n"
            f"{indent}        for registered_type in "
            f"cls.{REGISTRY_ATTRIBUTE_NAME}.values()\n"
            f"{indent}        if key_attribute in registered_type.__dict__\n"
            f"{indent}    }}\n"
        )

    def assignment_replacement(
        self,
        source_path: str,
        component: AutoRegisterInstanceViewComponent,
    ) -> PhysicalSourceEdit:
        statement = component.assignment
        value_source = f"{component.authority_name}.{self.instance_view_method_name}()"
        if isinstance(statement, ast.AnnAssign):
            assignment_source = (
                f"{component.assignment_name}: {ast.unparse(statement.annotation)} = "
                f"{value_source}"
            )
        else:
            assignment_source = f"{component.assignment_name} = {value_source}"
        return SourceSpanReplacement(
            file_path=source_path,
            span=SourceLineSpan(statement.lineno, statement.end_lineno or statement.lineno),
            replacement_lines=SourceTargetEditor.source_lines(assignment_source),
            rationale=self.rationale_text(
                f"Derive {component.assignment_name!r} from "
                f"{component.authority_name!r}."
            ),
        )


@dataclass(frozen=True)
class ManualRegistryConversionTargets:
    """Current component and physical targets for one registry conversion."""

    component: DirectManualRegistryComponent
    registered_classes: ClassMemberPromotionTargets
    authority: ResolvedClassTarget | None

    @classmethod
    def required_for_anchor(
        cls,
        snapshot: CodemodSourceSnapshot,
        anchor_target: AstTargetDigest,
        anchor_node: AstTargetNode,
    ) -> ManualRegistryConversionTargets:
        if not anchor_target.is_class or not isinstance(anchor_node, ast.ClassDef):
            raise ValueError("Manual registry conversion target must be a class")
        if "." in anchor_target.qualname:
            raise ValueError("Manual registry conversion requires a top-level class")
        source_path = anchor_target.file_path
        module = snapshot.module_nodes_by_file_path[source_path]
        component = DirectManualRegistryComponent.from_module_anchor(
            module,
            anchor_node.name,
        )
        component.require_destination_key_compatibility(component.destination_authority)
        component.require_original_entry_values(
            snapshot.module_binding_proof.native_reference_environment(
                snapshot.parsed_module_for_source_path(source_path)
            )
        )
        registered_classes = ClassMemberPromotionTargets.resolve(
            snapshot,
            source_path=source_path,
            class_names=component.class_names,
        )
        if not registered_classes.supports_base_rewrites():
            raise ValueError("Registry classes require lossless header rewrites")
        authority_node = component.existing_authority_node
        authority = (
            None
            if authority_node is None
            else ClassMemberPromotionTargets.class_target(
                snapshot.source_index,
                snapshot.ast_target_nodes_by_id,
                source_path=source_path,
                class_name=authority_node.name,
            )
        )
        return cls(
            component=component,
            registered_classes=registered_classes,
            authority=authority,
        )

    @property
    def file_path(self) -> str:
        return self.registered_classes.targets[0].file_path


@dataclass(frozen=True, kw_only=True)
class ConvertManualRegistryToAutoregisterOperation(
    RegistryKeyDeclarationRewriteMixin,
    SourceReprovedOperation,
):
    """Derive and convert one direct registry component from an anchor class."""

    supported_execution: DeclaredNativeUseInvariants = codemod_payload_field(
        PayloadRecordValueCodec(DeclaredNativeUseInvariants),
        default_factory=DeclaredNativeUseInvariants,
    )

    def simulation_reports(
        self, simulation: CodemodPlanDocumentSimulation
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        """Keep rendered candidates with unproved creator uses out of application."""
        return self.required_reproof(
            lambda: tuple(
                resolution.preflight_report()
                for resolution in self.supported_execution.resolve(
                    self.candidate_native_use_requirements(simulation)
                )
            )
        )

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        targets = self.required_targets(context.execution_snapshot())
        if targets.authority is not None:
            return (
                AstTargetAuthorityClaim.from_target(
                    targets.authority.target,
                    authority_kind=SemanticAuthorityKind.AUTOREGISTER_FAMILY,
                ),
            )
        return (
            AuthorityClaim(
                claimed_symbol=targets.component.authority_name,
                authority_kind=SemanticAuthorityKind.AUTOREGISTER_FAMILY,
                file_path=targets.file_path,
                qualname=targets.component.authority_name,
            ),
        )

    def required_targets(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> ManualRegistryConversionTargets:
        _target_id, anchor_target, anchor_node = self.target_node_from_context(snapshot)
        return ManualRegistryConversionTargets.required_for_anchor(
            snapshot,
            anchor_target,
            anchor_node,
        )

    def candidate_native_use_requirements(
        self, simulation: CodemodPlanDocumentSimulation
    ) -> tuple[NativeUseRequirement, ...]:
        """Expose candidate creator operands, not assert registration equivalence.

        Requirements retain the actual after revision and native-use provenance.
        Import execution, construction effects and selected observation relations
        remain separate obligations; inspecting these does not admit those effects.
        """
        targets = self.required_targets(
            simulation.after_snapshot_projection.base_snapshot
        )
        snapshot = simulation.required_after_snapshot
        candidate = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            SourceRewriteTarget(
                file_path=targets.file_path,
                qualname=targets.component.authority_name,
            ),
        )
        authority = AutoRegisterClassAuthority(candidate.node)
        environment = snapshot.product_flow_repository.native_reference_environment(
            snapshot.parsed_module_for_source_path(targets.file_path)
        )
        return (
            NativeDefinitionUseRequirement(
                type(self),
                authority.metaclass_operand,
                (authority.native_metaclass,),
                environment,
                source_state=snapshot.product_flow_repository,
            ),
        )

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        targets = self.required_targets(snapshot)
        return (
            *self.required_import_mutations(
                targets.file_path,
                import_source=(
                    f"from metaclass_registry import {AUTOREGISTER_META_NAME}\n"
                ),
                default_rationale="Import AutoRegisterMeta for class-time registration.",
            ),
            *self.authority_replacements(
                targets.file_path,
                snapshot.sources_by_file_path[targets.file_path],
                targets.component,
                targets.authority,
                targets.registered_classes,
            ),
            *self.registry_key_declaration_replacements(
                targets.registered_classes,
                targets.component.entries,
                DEFAULT_REGISTRY_KEY_ATTRIBUTE,
            ),
            *self.registration_replacements(
                targets.file_path,
                targets.component,
            ),
        )

    def authority_replacements(
        self,
        source_path: str,
        source: str,
        component: DirectManualRegistryComponent,
        authority_target: ResolvedClassTarget | None,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[NominalSourceEdit, ...]:
        if authority_target is None:
            return (
                self.generated_authority_insertion(component, targets),
                *self.generated_authority_base_replacements(component, targets),
            )
        return (
            *self.existing_authority_header_replacements(authority_target, source),
            *self.existing_authority_declaration_replacements(
                authority_target,
                source,
                component,
            ),
        )

    def generated_authority_insertion(
        self,
        component: DirectManualRegistryComponent,
        targets: ClassMemberPromotionTargets,
    ) -> PhysicalSourceEdit:
        class_target = targets.insertion_target
        authority_source = ast.unparse(component.destination_authority.node) + "\n\n"
        return SourceInsertion(
            file_path=class_target.file_path,
            insertion_line=targets.insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(authority_source),
            rationale=self.rationale_text(
                f"Insert AutoRegisterMeta base {component.authority_name!r}."
            ),
        )

    def generated_authority_base_replacements(
        self,
        component: DirectManualRegistryComponent,
        targets: ClassMemberPromotionTargets,
    ) -> tuple[SourceTextMutation, ...]:
        replacements = []
        for target in targets.targets:
            header = ClassHeaderSpanSourceAuthority(
                target.node, targets.source_for(target.file_path)
            )
            replacements.append(
                header.geometry.nominal_edit(
                    file_path=target.file_path,
                    replacements=(
                        header.header_replacement(
                            header.with_added_base(component.authority_name)
                        ),
                    ),
                    rationale=self.rationale_text(
                        f"Add registry authority to {target.qualname!r}."
                    ),
                )
            )
        return tuple(replacements)

    def existing_authority_header_replacements(
        self,
        authority_target: ResolvedClassTarget,
        source: str,
    ) -> tuple[SourceTextMutation, ...]:
        header = ClassHeaderSpanSourceAuthority(authority_target.node, source)
        metaclass_keywords = tuple(
            keyword
            for keyword in authority_target.node.keywords
            if keyword.arg == "metaclass"
        )
        if metaclass_keywords:
            # The operand's identity is proved on the result
            # (NativeDefinitionUseRequirement), not read from its name.
            return ()
        return (
            header.geometry.nominal_edit(
                replacements=(
                    header.header_replacement(
                        header.with_items(
                            header.base_items,
                            (
                                *header.keyword_items,
                                f"metaclass={AUTOREGISTER_META_NAME}",
                            ),
                        )
                    ),
                ),
                file_path=authority_target.file_path,
                rationale=self.rationale_text(
                    f"Make {authority_target.qualname!r} own class registration."
                ),
            ),
        )

    def existing_authority_declaration_replacements(
        self,
        authority_target: ResolvedClassTarget,
        source: str,
        component: DirectManualRegistryComponent,
    ) -> tuple[SourceTextMutation, ...]:
        required_values = component.destination_authority.assignment_pairs
        missing = []
        for name, expected_value in required_values:
            declarations = tuple(
                statement
                for statement in authority_target.node.body
                if ClassDeclarationPromotionStatement(statement).name == name
            )
            if not declarations:
                missing.append((name, expected_value))
                continue
            if len(declarations) != 1 or not self.declaration_matches_value(
                declarations[0], expected_value
            ):
                raise ValueError(
                    f"Registry authority declaration {name!r} conflicts with "
                    "the derived registry component"
                )
        if not missing:
            return ()
        body_authority = ClassBodyTailSourceAuthority(
            authority_target.node,
            source,
        )
        inserted_lines = tuple(
            f"{body_authority.indentation}{name} = {ast.unparse(value)}\n"
            for name, value in missing
        )
        return (
            body_authority.geometry.nominal_edit(
                file_path=authority_target.file_path,
                replacements=(
                    body_authority.member_insertion_replacement(inserted_lines),
                ),
                rationale=self.rationale_text(
                    f"Declare registry semantics on {authority_target.qualname!r}."
                ),
            ),
        )

    def registration_replacements(
        self,
        source_path: str,
        component: DirectManualRegistryComponent,
    ) -> tuple[PhysicalSourceEdit, ...]:
        replacements = []
        if not component.reuses_registry_binding:
            statement = component.registry_assignment
            replacements.append(
                SourceSpanReplacement(
                    file_path=source_path,
                    span=SourceLineSpan(statement.lineno, statement.end_lineno or statement.lineno),
                    replacement_lines=SourceTargetEditor.source_lines(
                        self.derived_registry_assignment_source(statement, component)
                    ),
                    rationale=self.rationale_text(
                        f"Derive {component.registry_name!r} from its class authority."
                    ),
                )
            )
        return (
            *replacements,
            *(
                SourceSpanDeletion(
                    file_path=source_path,
                    span=SourceLineSpan(statement.lineno, statement.end_lineno or statement.lineno),
                    rationale=self.rationale_text("Delete manual registry write."),
                )
                for statement in component.registration_statements
                if statement is not component.registry_assignment
            ),
        )

    @staticmethod
    def derived_registry_assignment_source(
        statement: RegistryAssignment,
        component: DirectManualRegistryComponent,
    ) -> str:
        value_source = f"{component.authority_name}.{REGISTRY_ATTRIBUTE_NAME}"
        if isinstance(statement, ast.AnnAssign):
            return (
                f"{component.registry_name}: {ast.unparse(statement.annotation)} = "
                f"{value_source}"
            )
        return f"{component.registry_name} = {value_source}"


class InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    AutoRegisterConcept,
):
    """Delete AutoRegister protocol fields repeated from inherited bases."""

    recipe_id_suffix = "delete-inherited-autoregister-config"
    recipe_reason = (
        "Delete AutoRegister registry protocol assignments already inherited "
        "from a nominal base."
    )

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "inherited AutoRegister cleanup requires source context"
            )
        evidence = finding.primary_evidence
        if evidence is None:
            return self.rejected_evaluation(
                "inherited AutoRegister cleanup lacks class evidence"
            )
        try:
            snapshot = context.execution_snapshot()
            target_ids = SourceIndexTargetSelector(
                node_kinds=(AstTargetNodeKind.CLASS,),
                file_paths=(evidence.file_path,),
                qualnames=(evidence.symbol,),
            ).target_ids(snapshot)
            if len(target_ids) != 1:
                raise ValueError(
                    "Inherited AutoRegister evidence must resolve one exact class"
                )
            target = snapshot.source_index.target_by_id[target_ids[0]]
            operation = DeleteInheritedAutoRegisterConfigurationOperation(
                target=SourceRewriteTarget(target_id=target.target_id),
                rationale="",
            )
            operation.source_edits_from_snapshot(snapshot)
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        recipe = (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{self.recipe_id_suffix}",
                reason=self.recipe_reason,
            )
            .with_authority_claim(AstTargetAuthorityClaim.from_target(target))
            .with_operation(operation)
        )
        return self.executable_evaluation(recipe)


@dataclass(frozen=True)
class AutoRegisterMroOrderingDerivation:
    """Current-source proof that one registered family can own ordering in its MRO."""

    context: CodemodSelectorContext = field(repr=False, compare=False)
    root: ResolvedClassTarget
    registered_leaves: tuple[tuple[int, ResolvedClassTarget], ...]
    registry_key_name: str
    ordering_field_name: str
    ordering_method: ResolvedFunctionProjectionTarget
    sorted_call: ast.Call = field(repr=False, compare=False)

    @classmethod
    def discover(
        cls,
        context: CodemodSelectorContext,
        root_reference: SourceRewriteTarget,
    ) -> AutoRegisterMroOrderingDerivation:
        root = ResolvedClassTarget.from_rewrite_target(context, root_reference)
        if "." in root.qualname:
            raise ValueError("MRO ordering derivation requires a top-level authority")
        root_registry_authority = AutoRegisterClassAuthority(root.node)
        registry_key_name = root_registry_authority.registry_key_attribute
        if (
            registry_key_name is None
            or not root_registry_authority.skips_missing_keys
            or root_registry_authority.declares_key_extractor
            or not cls.has_plain_root_bases(root.node)
        ):
            raise ValueError(
                "MRO ordering derivation requires a plain enum-keyed root without "
                "a custom key extractor"
            )
        ordering_projection = cls.ordering_projection(root.node)
        if ordering_projection is None:
            raise ValueError(
                "MRO ordering derivation requires one registry ordering projection"
            )
        ordering_node, sorted_call, ordering_field_name = ordering_projection
        if not cls.direct_assignment_declared(root.node, ordering_field_name):
            raise ValueError(
                "MRO ordering derivation requires the root to declare its ordering axis"
            )
        ordering_method = ResolvedFunctionProjectionTarget.from_function_identity(
            context,
            source_path=root.file_path,
            function_qualname=f"{root.qualname}.{ordering_node.name}",
        )
        if ordering_method is None:
            raise ValueError("MRO ordering derivation cannot resolve its consumer")
        class_targets = cls.top_level_class_targets(context, root.file_path)
        class_nodes_by_name = {target.node.name: target for target in class_targets}
        descendant_names = cls.descendant_names(
            class_nodes_by_name,
            root.node.name,
        )
        registered_leaves = cls.registered_leaf_targets(
            class_nodes_by_name,
            descendant_names,
            root.node.name,
            registry_key_name,
            ordering_field_name,
        )
        if registered_leaves is None or len(registered_leaves) < 2:
            raise ValueError(
                "MRO ordering derivation requires incomparable single-inheritance "
                "leaves with unique integer ordering values"
            )
        if not cls.registered_leaves_exhaust_enum_key(
            root.node,
            class_nodes_by_name,
            registered_leaves,
            registry_key_name,
        ):
            raise ValueError(
                "MRO ordering derivation requires registered leaves to exhaust one "
                "local enum key"
            )
        resolution_class_name = cls.resolution_class_name_for(root.node.name)
        module = context.module_nodes_by_file_path[root.file_path]
        if resolution_class_name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            module.body
        ):
            raise ValueError(
                "MRO ordering derivation will not overwrite its resolution authority"
            )
        return cls(
            context=context,
            root=root,
            registered_leaves=registered_leaves,
            registry_key_name=registry_key_name,
            ordering_field_name=ordering_field_name,
            ordering_method=ordering_method,
            sorted_call=sorted_call,
        )

    @property
    def ordering_axis_targets(self) -> tuple[ResolvedClassTarget, ...]:
        return (self.root, *(leaf for _priority, leaf in self.registered_leaves))

    @property
    def resolution_class_name(self) -> str:
        return self.resolution_class_name_for(self.root.node.name)

    @staticmethod
    def resolution_class_name_for(root_name: str) -> str:
        return f"_{root_name}ResolutionMro"

    @property
    def insertion_target(self) -> ResolvedClassTarget:
        return max(
            (leaf for _priority, leaf in self.registered_leaves),
            key=lambda leaf: leaf.target.end_line,
        )

    @property
    def registered_types_call_source(self) -> str:
        return f"{self.resolution_class_name}.registered_types()"

    @property
    def resolution_class_source(self) -> str:
        bases = "".join(
            f"    {leaf.node.name},\n" for _priority, leaf in self.registered_leaves
        )
        return (
            f"\n\nclass {self.resolution_class_name}(\n"
            f"{bases}"
            "):\n"
            f"    {self.registry_key_name} = None\n\n"
            "    @classmethod\n"
            f"    def registered_types(cls) -> tuple[type[{self.root.node.name}], ...]:\n"
            "        return tuple(\n"
            "            candidate\n"
            "            for candidate in cls.__mro__[1:]\n"
            f"            if candidate in {self.root.node.name}.{REGISTRY_ATTRIBUTE_NAME}.values()\n"
            "        )\n"
        )

    def source_edits(self) -> tuple[PhysicalSourceEdit, ...]:
        source_by_path = self.context.sources_by_file_path
        sorted_call_source = SourceTextGeometry(
            source_by_path[self.root.file_path]
        ).segment_for_node(self.sorted_call)
        if sorted_call_source is None:
            raise ValueError(
                "MRO ordering derivation cannot recover its current ordering source"
            )
        deletion_edits = tuple(
            edit
            for target in self.ordering_axis_targets
            for edit in DeleteClassAssignmentsOperation(
                target=SourceRewriteTarget(target_id=target.target.target_id),
                assignment_names=(self.ordering_field_name,),
                rationale=(
                    "Delete the explicit ordering axis superseded by the family MRO."
                ),
            ).source_edits(self.context)
        )
        ordering_edits = PatchTargetOperation(
            target=SourceRewriteTarget(target_id=self.ordering_method.target.target_id),
            replacements=(
                SourceTextReplacement(
                    old_source=sorted_call_source,
                    new_source=self.registered_types_call_source,
                ),
            ),
            rationale="Read family precedence from the declared MRO projection.",
        ).source_edits(self.context)
        insertion_edits = InsertAfterTargetOperation(
            target=SourceRewriteTarget(
                target_id=self.insertion_target.target.target_id
            ),
            source=self.resolution_class_source,
            rationale="Declare the family MRO projection beside its leaves.",
        ).source_edits(self.context)
        return (*deletion_edits, *ordering_edits, *insertion_edits)

    @staticmethod
    def top_level_class_targets(
        context: CodemodSelectorContext,
        source_path: str,
    ) -> tuple[ResolvedClassTarget, ...]:
        rows = []
        for target in context.source_index.ast_targets:
            if (
                target.file_path != source_path
                or not target.is_class
                or "." in target.qualname
            ):
                continue
            node = context.ast_target_nodes_by_id.get(target.target_id)
            if isinstance(node, ast.ClassDef):
                rows.append(ResolvedClassTarget(target=target, node=node))
        return sorted_tuple(rows, key=lambda row: row.line)

    @staticmethod
    def descendant_names(
        class_nodes_by_name: Mapping[str, ResolvedClassTarget],
        root_name: str,
    ) -> frozenset[str]:
        descendants: set[str] = set()
        changed = True
        while changed:
            changed = False
            family_names = descendants | {root_name}
            for class_name, target in class_nodes_by_name.items():
                if class_name in family_names:
                    continue
                base_names = {
                    base_name
                    for base in target.node.bases
                    if (base_name := ExpressionReference.decoded(base).terminal_name)
                    is not None
                }
                if family_names.isdisjoint(base_names):
                    continue
                descendants.add(class_name)
                changed = True
        return frozenset(descendants)

    @classmethod
    def registered_leaf_targets(
        cls,
        class_nodes_by_name: Mapping[str, ResolvedClassTarget],
        descendant_names: frozenset[str],
        root_name: str,
        registry_key_name: str,
        ordering_field_name: str,
    ) -> tuple[tuple[int, ResolvedClassTarget], ...] | None:
        family_names = descendant_names | {root_name}
        child_names_by_parent: dict[str, set[str]] = defaultdict(set)
        for class_name in descendant_names:
            target = class_nodes_by_name[class_name]
            direct_assignment_names = frozenset(
                name
                for statement in target.node.body
                for name in AssignmentStatementNameProjection(statement).names
            )
            if (
                len(target.node.bases) != 1
                or direct_assignment_names & AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES
                or any(
                    isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
                    and statement.name == "__init_subclass__"
                    for statement in target.node.body
                )
            ):
                return None
            base_name = ExpressionReference.decoded(target.node.bases[0]).terminal_name
            if base_name not in family_names:
                return None
            child_names_by_parent[base_name].add(class_name)

        leaves = []
        for class_name in descendant_names:
            target = class_nodes_by_name[class_name]
            registry_key = cls.direct_assignment_value(
                target.node,
                registry_key_name,
            )
            if registry_key is None or CompactValueExpression.project(registry_key).value_is_none_literal:
                continue
            if child_names_by_parent[class_name]:
                return None
            priority = cls.direct_assignment_value(
                target.node,
                ordering_field_name,
            )
            if not (
                isinstance(priority, ast.Constant)
                and isinstance(priority.value, int)
                and not isinstance(priority.value, bool)
            ):
                return None
            leaves.append((priority.value, target))
        if len({priority for priority, _target in leaves}) != len(leaves):
            return None
        return sorted_tuple(leaves, key=lambda row: row[0])

    @classmethod
    def registered_leaves_exhaust_enum_key(
        cls,
        root_node: ast.ClassDef,
        class_nodes_by_name: Mapping[str, ResolvedClassTarget],
        registered_leaves: tuple[tuple[int, ResolvedClassTarget], ...],
        registry_key_name: str,
    ) -> bool:
        enum_declaration = cls.registry_key_enum_declaration(
            root_node,
            class_nodes_by_name,
            registry_key_name,
        )
        if enum_declaration is None:
            return False
        enum_name, enum_node = enum_declaration
        enum_members = frozenset(
            name
            for statement in enum_node.body
            for name in AssignmentStatementNameProjection(statement).names
            if not name.startswith("_")
        )
        registered_members = tuple(
            cls.enum_member_name(
                cls.direct_assignment_value(target.node, registry_key_name),
                enum_name,
            )
            for _priority, target in registered_leaves
        )
        return bool(
            enum_members
            and None not in registered_members
            and len(registered_members) == len(set(registered_members))
            and frozenset(registered_members) == enum_members
        )

    @staticmethod
    def registry_key_enum_declaration(
        root_node: ast.ClassDef,
        class_nodes_by_name: Mapping[str, ResolvedClassTarget],
        registry_key_name: str,
    ) -> tuple[str, ast.ClassDef] | None:
        annotations = tuple(
            statement.annotation
            for statement in root_node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == registry_key_name
        )
        if len(annotations) != 1:
            return None
        annotation_names = frozenset(
            node.id for node in ast.walk(annotations[0]) if isinstance(node, ast.Name)
        )
        enum_declarations = tuple(
            (class_name, target.node)
            for class_name, target in class_nodes_by_name.items()
            if class_name in annotation_names
            and PYTHON_ENUM_BASE_AUTHORITY.matches_any(
                ExpressionReference.decoded(base).terminal_name
                for base in target.node.bases
            )
        )
        return enum_declarations[0] if len(enum_declarations) == 1 else None

    @staticmethod
    def enum_member_name(node: ast.expr | None, enum_name: str) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == enum_name
        ):
            return node.attr
        return None

    @staticmethod
    def direct_assignment_value(
        node: ast.ClassDef,
        assignment_name: str,
    ) -> ast.expr | None:
        values = tuple(
            pair[1]
            for statement in node.body
            if (pair := SingleAssignmentAndValueNameProjection(statement).pair)
            is not None
            and pair[0] == assignment_name
        )
        return (
            values[0] if len(values) == 1 and isinstance(values[0], ast.expr) else None
        )

    @staticmethod
    def direct_assignment_declared(
        node: ast.ClassDef,
        assignment_name: str,
    ) -> bool:
        return any(
            assignment_name in AssignmentStatementNameProjection(statement).names
            for statement in node.body
        )

    @staticmethod
    def has_plain_root_bases(root_node: ast.ClassDef) -> bool:
        return all(
            ExpressionReference.decoded(base).terminal_name
            in {"ABC", "Generic", "object"}
            for base in root_node.bases
        ) and not any(
            isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement.name == "__init_subclass__"
            for statement in root_node.body
        )

    @classmethod
    def ordering_projection(
        cls,
        root_node: ast.ClassDef,
    ) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call, str] | None:
        matches = tuple(
            (statement, node, ordering_field_name)
            for statement in FunctionDefinition.within(root_node.body)
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and (ordering_field_name := cls.registry_ordering_field_name(node))
            is not None
        )
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def registry_ordering_field_name(node: ast.Call) -> str | None:
        if not isinstance(node.func, ast.Name) or node.func.id != "sorted":
            return None
        if len(node.args) != 1 or len(node.keywords) != 1:
            return None
        registry_values = node.args[0]
        if not (
            isinstance(registry_values, ast.Call)
            and not registry_values.args
            and not registry_values.keywords
            and isinstance(registry_values.func, ast.Attribute)
            and registry_values.func.attr == "values"
            and isinstance(registry_values.func.value, ast.Attribute)
            and registry_values.func.value.attr == REGISTRY_ATTRIBUTE_NAME
            and isinstance(registry_values.func.value.value, ast.Name)
            and registry_values.func.value.value.id == "cls"
        ):
            return None
        keyword = node.keywords[0]
        key_function = keyword.value
        if not (
            keyword.arg == "key"
            and isinstance(key_function, ast.Lambda)
            and isinstance(key_function.body, ast.Attribute)
            and isinstance(key_function.body.value, ast.Name)
            and len(key_function.args.args) == 1
            and key_function.body.value.id == key_function.args.args[0].arg
        ):
            return None
        return key_function.body.attr


@dataclass(frozen=True, kw_only=True)
class DeriveAutoRegisterMroOrderingOperation(RepositorySourceReprovedOperation):
    """Re-prove one registered family and derive its ordering from current source."""

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        _target_identifier, root_target, _root_node = self.target_node_from_context(
            context
        )
        if not root_target.is_class:
            raise ValueError("MRO ordering authority target must be a class")
        authority_name = AutoRegisterMroOrderingDerivation.resolution_class_name_for(
            root_target.name
        )
        return (
            AuthorityClaim(
                claimed_symbol=authority_name,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                file_path=root_target.file_path,
                qualname=authority_name,
            ),
        )

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> AutoRegisterMroOrderingDerivation:
        return AutoRegisterMroOrderingDerivation.discover(
            context,
            self.target,
        )

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return self.required_derivation(snapshot).source_edits()


class AutoRegisterExplicitPriorityOrderingFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    AutoRegisterMroOrderingConcept,
    SingleSourcePathFindingMixin,
):
    """Batch an explicit registered priority axis into one nominal MRO view."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "MRO ordering extraction requires a source selector context"
            )
        recipe, rejection_reason = self.recipe_for_finding(finding, context)
        if recipe is None:
            return self.rejected_evaluation(rejection_reason)
        return self.executable_evaluation(recipe)

    def recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> tuple[RefactorRecipe | None, str]:
        source_path = self.source_path(finding)
        evidence = finding.primary_evidence
        if source_path is None or evidence is None:
            return None, "MRO ordering extraction requires one source file and root"
        if not isinstance(finding.metrics, MappingMetrics):
            return None, "MRO ordering extraction requires mapping metrics"
        if len(finding.metrics.plan_field_names) != 1:
            return None, "MRO ordering extraction requires one priority field"
        root = ClassMemberPromotionTargets.optional_class_target(
            context.source_index,
            context.ast_target_nodes_by_id,
            source_path=source_path,
            class_name=evidence.symbol,
        )
        if root is None:
            return None, "MRO ordering extraction cannot resolve the family root"
        root_target = root.target
        try:
            derivation = AutoRegisterMroOrderingDerivation.discover(
                context,
                SourceRewriteTarget(target_id=root_target.target_id),
            )
        except ValueError as error:
            return None, str(error)
        if derivation.ordering_field_name != finding.metrics.plan_field_names[0]:
            return None, "MRO ordering extraction axis differs from finding evidence"
        if len(derivation.ordering_axis_targets) != finding.metrics.mapping_site_count:
            return (
                None,
                "MRO ordering extraction priority sites do not match finding evidence",
            )
        operation = DeriveAutoRegisterMroOrderingOperation(
            target=SourceRewriteTarget(target_id=root_target.target_id),
            rationale="Derive registered-family precedence from its nominal MRO.",
        )
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-derive-mro-ordering",
            reason=(
                "Derive registered-family precedence from one nominal MRO composition."
            ),
        ).with_operation(operation)
        return (
            recipe,
            "",
        )


@dataclass(frozen=True)
class ManualRegistryRecipeParts:
    """Source-proved manual registry component and its exact operation anchor."""

    anchor_target: AstTargetDigest


class ManualClassRegistrationFindingRecipeSynthesizer(
    FindingRecipeSynthesizer,
    AutoRegisterClassRegistryConcept,
):
    """Build AutoRegisterMeta conversion recipes for manual class registries."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "manual-registry conversion requires a source selector context"
            )
        try:
            parts = self.recipe_parts_for_finding(finding, context)
        except ValueError as refusal:
            return self.rejected_evaluation(f"manual-registry conversion: {refusal}")
        return self.executable_evaluation(self.recipe_from_parts(finding, parts))

    def recipe_parts_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> ManualRegistryRecipeParts:
        """The proved component and anchor, or a ValueError naming the premise that failed."""

        registry_name = finding.metrics.plan_registry_name
        expected_class_names = frozenset(finding.metrics.plan_class_names)
        if registry_name is None:
            raise ValueError("the finding exposes no registry name")
        if not expected_class_names:
            raise ValueError("the finding exposes no registered classes")
        evidence = finding.primary_evidence
        if evidence is None:
            raise ValueError("the finding has no primary evidence")
        source_paths = context.resolve_source_paths((evidence.file_path,))
        if len(source_paths) != 1:
            raise ValueError(
                f"`{evidence.file_path}` resolves to {len(source_paths)} source files, not one"
            )
        source_path = next(iter(source_paths))
        targets = {
            class_name: ClassMemberPromotionTargets.optional_class_target(
                context.source_index,
                context.ast_target_nodes_by_id,
                source_path=source_path,
                class_name=class_name,
            )
            for class_name in sorted(expected_class_names)
        }
        missing = sorted(name for name, target in targets.items() if target is None)
        if missing:
            raise ValueError(
                f"registered class(es) {', '.join(f'`{name}`' for name in missing)} "
                f"are not declared in `{source_path}`"
            )
        resolved_targets = tuple(target for target in targets.values() if target is not None)
        nested = sorted(target.qualname for target in resolved_targets if "." in target.qualname)
        if nested:
            raise ValueError(
                f"registered class(es) {', '.join(f'`{name}`' for name in nested)} "
                "are nested, not module-level"
            )
        anchor_target = min(resolved_targets, key=lambda target: target.line)
        component = DirectManualRegistryComponent.from_module_anchor(
            context.module_nodes_by_file_path[source_path],
            anchor_target.node.name,
        )
        if component.registry_name != registry_name:
            raise ValueError(
                f"the registry at `{anchor_target.node.name}` is "
                f"`{component.registry_name}`, not the finding's `{registry_name}`"
            )
        if frozenset(component.class_names) != expected_class_names:
            raise ValueError(
                "the registry's classes "
                f"({', '.join(sorted(component.class_names))}) differ from the finding's "
                f"({', '.join(sorted(expected_class_names))})"
            )
        return ManualRegistryRecipeParts(anchor_target=anchor_target.target)

    def recipe_from_parts(
        self,
        finding: RefactorFinding,
        parts: ManualRegistryRecipeParts,
    ) -> RefactorRecipe:
        return RefactorRecipe(
            recipe_id=f"{finding.stable_id}-convert-manual-registry",
            reason="Replace manual registry writes with AutoRegisterMeta.",
        ).with_operation(
            ConvertManualRegistryToAutoregisterOperation(
                target=SourceRewriteTarget(target_id=parts.anchor_target.target_id),
                rationale="",
            )
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = finding.primary_evidence
        if evidence is None:
            return ()
        registry_name = finding.metrics.plan_registry_name
        if registry_name is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            (
                (evidence.file_path, class_name)
                for class_name in finding.metrics.plan_class_names
            ),
        )


class RegistrationSemanticMirrorRecipeStrategy(
    ManualClassRegistrationFindingRecipeSynthesizer,
    SemanticMirrorFindingRecipeStrategy,
):
    """Route class-family semantic mirrors through AutoRegisterMeta recipes."""

    metric_type = RegistrationMetrics

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        contextual_builders = (
            RegistrationSemanticMirrorRecipeBuilder.builders_from_context(
                finding,
                context,
            )
        )
        contextual_evaluations = tuple(
            self.evaluation_from_recipe(finding, recipe, type(builder))
            for builder in contextual_builders
            if (recipe := builder.recipe()) is not None
        )
        manual_evaluation = super().evaluate_recipe_for_finding(
            finding,
            context,
        )
        evaluations = (
            *contextual_evaluations,
            *(
                (
                    self.evaluation_from_recipe(
                        finding,
                        manual_evaluation.required_recipe,
                        manual_evaluation.required_executable_declaration_type,
                    ),
                )
                if manual_evaluation.candidate_recipes
                else ()
            ),
        )
        if len(evaluations) > 1:
            raise ValueError(
                "Registration mirror finding matched multiple recipe declarations: "
                f"{tuple(evaluation.recipe_id for evaluation in evaluations)!r}"
            )
        if evaluations:
            return evaluations[0]
        obstacles = (
            *RegistrationSemanticMirrorRecipeBuilder.proof_obstacles(
                contextual_builders,
            ),
            FindingRecipeProofObstacle(
                executable_declaration_type=(
                    manual_evaluation.required_evaluation_declaration_type
                ),
                reason=manual_evaluation.rejection_reason,
            ),
        )
        return RejectedRecipeEvaluation(
            reason=(
                "no class-family recipe declaration proved an executable exact "
                "derivation"
            ),
            evaluation_declaration_type=type(self),
            obstacles=obstacles,
        )


@dataclass(frozen=True, kw_only=True)
class AutoregisterInstanceViewRecipeBuilder(
    RegistrationSemanticMirrorRecipeBuilder,
    AutoRegisterClassRegistryConcept,
):
    """Build recipes for constructor-valued views over AutoRegisterMeta families."""

    def recipe(self) -> RefactorRecipe | None:
        authority_target = self.authority_target()
        if authority_target is None:
            return None
        return (
            RefactorRecipe(
                recipe_id=f"{self.finding.stable_id}-derive-autoregister-instance-view",
                reason="Derive instance view from existing AutoRegisterMeta registry.",
            )
            .with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    authority_target,
                    authority_kind=SemanticAuthorityKind.AUTOREGISTER_FAMILY,
                )
            )
            .with_operation(
                DeriveAutoregisterInstanceViewOperation(
                    target=SourceRewriteTarget(target_id=authority_target.target_id),
                    rationale="",
                )
            )
        )

    def authority_target(self) -> AstTargetDigest | None:
        seed = SemanticMirrorRecipeSeedLocations.from_finding(self.finding)
        assignment_name = self.finding.metrics.plan_registry_name
        expected_class_names = frozenset(self.finding.metrics.plan_class_names)
        if seed is None or assignment_name is None or not expected_class_names:
            return None
        projection_paths = self.resolve_source_paths((seed.projection.file_path,))
        if len(projection_paths) != 1:
            return None
        projection_path = next(iter(projection_paths))
        authority_targets = ClassMemberPromotionTargets.resolve_or_none(
            self,
            source_path=projection_path,
            class_names=(seed.authority.symbol,),
        )
        if authority_targets is None:
            return None
        authority_target = authority_targets.targets[0].target
        try:
            component = AutoRegisterInstanceViewComponent.from_module_authority(
                self.module_nodes_by_file_path[projection_path],
                authority_target.name,
            )
        except ValueError:
            return None
        if (
            component.assignment_name != assignment_name
            or frozenset(component.class_names) != expected_class_names
        ):
            return None
        return authority_target

    def rejection_reason(self) -> str:
        if SemanticMirrorRecipeSeedLocations.from_finding(self.finding) is None:
            return "semantic mirror finding does not expose projection and authority locations"
        if self.finding.metrics.plan_registry_name is None:
            return "semantic mirror finding exposes no instance-view assignment"
        if self.authority_target() is not None:
            return "instance-view derivation is available"
        return (
            "source does not prove one complete zero-argument constructor view "
            "owned by the AutoRegisterMeta family"
        )
