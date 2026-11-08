from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from functools import cached_property
from types import FunctionType
from typing import (
    ClassVar,
    Generic,
    TypeVar,
    cast,
)

from ..ast_tools import ParsedModule
from ..binding_proofs import RepositoryModuleBindingProof
from ..class_index import ModuleNominalBindingAuthority
from .class_member_authority import ClassDeclarationPromotionStatement
from .finding_recipes import (
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)
from .class_base import ClassAuthorityReferenceProof
from .source_insertion import EnsureImportOperation
from ..codemod import PatchTargetOperation
from ..codemod_declaration_source import (
    ClassBodySourceAuthority,
    ClassHeaderSpanSourceAuthority,
    NamedDeclarationSourceAuthority,
)
from ..codemod_paths import SourcePathResolutionAuthority
from ..codemod_preflight import CodemodOperationPreflightError
from ..codemod_reproof import RepositorySourceReprovedOperation
from ..codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
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
    SourceSpanDeletion,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceTextReplacement,
    SourceTextSpan,
)
from ..codemod_target_selectors import SourceIndexTargetSelector
from ..declaration_binding_transfer import ClassBodyReferenceCapture
from ..declaration_dependencies import ModuleLexicalDependencyProjection
from ..detectors._base import (
    CallableCandidateFindingRenderer,
    CandidateCollectorBaseReference,
    CandidateCollectorBoilerplateCandidate,
    DeclarativeDetectorClassCandidate,
    DerivedCandidateCollectorMixin,
    DetectorDeclaration,
    DetectorDeclarationOptions,
    DirectBuildFindingRendererCandidate,
    IssueDetector,
    ModuleCollectedLineWitnessCandidate,
    declare_module_detector,
)
from ..models import RefactorFinding
from ..native_class_mro import NativeClassMroDeclaration
from ..refactor_concepts import ClassFamilyAuthorityConcept
from ..source_index import (
    AstTargetDigest,
    AstTargetNodeKind,
)
from ..source_native_mro import (
    NativeClassBaseSubstitution,
    SourceNativeClassMro,
)


@dataclass(frozen=True)
class CandidateCollectorMigration:
    """One source-proved detector collector migration."""

    candidate: CandidateCollectorBoilerplateCandidate
    target: AstTargetDigest
    node: ast.ClassDef
    source: str
    import_source: str | None
    rationale: str

    replaced_base: ast.expr

    @property
    def contextual_base_source(self) -> str:
        return (
            f"{self.candidate.recommended_base_name}"
            f"[{self.candidate.candidate_type_source}]"
        )

    def source_edits(
        self, context: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        RepositoryModuleBindingProof(
            context.parsed_modules
        ).require_native_type_in_class(
            context.parsed_module_for_source_path(self.target.file_path),
            self.node,
            self.candidate.collector_descriptor_type,
        )
        ClassBodyReferenceCapture(
            context.parsed_module_for_source_path(self.target.file_path),
            self.node,
            self.candidate_method,
            self.candidate.forwarding.callee,
        ).require_preserved()
        if any(
            ClassDeclarationPromotionStatement(statement).name
            == self.candidate.collector_declaration_name
            for statement in self.node.body
        ):
            raise ValueError(
                f"{self.node.name!r} already declares "
                f"{self.candidate.collector_declaration_name}"
            )
        import_edits = (
            ()
            if self.import_source is None
            else EnsureImportOperation(
                target=SourceRewriteTarget(file_path=self.target.file_path),
                import_source=self.import_source,
                rationale=self.rationale,
            ).source_edits(context)
        )
        return (
            *import_edits,
            *self.class_header_replacements(),
            self.candidate_declaration_insertion(),
            self.candidate_method_deletion(),
        )

    def class_header_replacements(self) -> tuple[PhysicalSourceEdit, ...]:
        header = ClassHeaderSpanSourceAuthority(node=self.node, source=self.source)
        return header.source_edits(
            header.with_replaced_base(
                ast.unparse(self.replaced_base), self.contextual_base_source
            ),
            file_path=self.target.file_path,
            rationale=self.rationale
            or f"Derive {self.node.name!r} candidate traversal from its collector.",
        )

    def candidate_declaration_insertion(self) -> SourceInsertion:
        body = ClassBodySourceAuthority(node=self.node, source=self.source)
        anchor = next(
            (
                statement
                for statement in self.node.body
                if isinstance(statement, ast.FunctionDef)
                and statement.name == IssueDetector._collect_findings.__name__
            ),
            None,
        )
        insertion_line = (
            NamedDeclarationSourceAuthority(
                anchor, self.source
            ).declaration_line_span.start_line
            if anchor is not None
            else body.declaration_insert_line + 1
        )
        indent = body.indentation
        return SourceInsertion(
            file_path=self.target.file_path,
            insertion_line=insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(
                f"{indent}{self.candidate.collector_declaration_source}\n\n"
            ),
            rationale=self.rationale
            or "Declare the detector candidate collector strategy.",
        )

    @cached_property
    def candidate_method(self) -> ast.FunctionDef:
        method = next(
            (
                statement
                for statement in self.node.body
                if isinstance(statement, ast.FunctionDef)
                and statement.name == self.candidate.method_name
            ),
            None,
        )
        if method is None:
            raise ValueError(
                f"{self.candidate.symbol!r} is no longer declared by the target class"
            )
        return method

    def candidate_method_deletion(self) -> SourceSpanDeletion:
        return NamedDeclarationSourceAuthority(
            self.candidate_method,
            self.source,
        ).declaration_line_span.line_deletion(
            file_path=self.target.file_path,
            rationale=self.rationale
            or "Delete candidate traversal now owned by the collector base.",
        )


SourceRecipeCandidateT = TypeVar(
    "SourceRecipeCandidateT",
    bound=ModuleCollectedLineWitnessCandidate,
)


SourceRecipeNodeT = TypeVar("SourceRecipeNodeT", bound=ast.AST)


@dataclass(frozen=True)
class CurrentLineWitness(Generic[SourceRecipeCandidateT, SourceRecipeNodeT]):
    """Current source evidence, derived during execution and never serialized."""

    target: AstTargetDigest
    node: SourceRecipeNodeT
    candidate: SourceRecipeCandidateT
    module: ParsedModule


@dataclass(frozen=True, kw_only=True)
class LineWitnessSourceReprovedOperation(
    RepositorySourceReprovedOperation,
    Generic[SourceRecipeCandidateT, SourceRecipeNodeT],
    ABC,
):
    """Persist a target; rederive its candidate and edits on every execution."""

    candidate_type: ClassVar[type[ModuleCollectedLineWitnessCandidate]]
    target_node_kind: ClassVar[AstTargetNodeKind]

    def required_witness(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> CurrentLineWitness[SourceRecipeCandidateT, SourceRecipeNodeT]:
        _identifier, target, node = self.target_node_from_context(snapshot)
        target.require_kind(
            type(self).target_node_kind, "Unexpected line witness target kind"
        )
        if not type(self).target_node_kind.accepts(node):
            raise ValueError(
                "Line witness source does not match its indexed target kind"
            )
        module = snapshot.parsed_module_for_source_path(target.file_path)
        candidates = tuple(
            candidate
            for candidate in type(self).candidate_type.from_module(module)
            if candidate.witness_name == target.qualname
            and candidate.line == target.line
        )
        if len(candidates) != 1:
            raise ValueError(
                f"{target.qualname!r} belongs to {len(candidates)} current "
                f"{type(self).candidate_type.__name__} witnesses"
            )
        return CurrentLineWitness(
            target,
            cast(SourceRecipeNodeT, node),
            cast(SourceRecipeCandidateT, candidates[0]),
            module,
        )

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.source_edits_for_witness(snapshot, self.required_witness(snapshot))

    @abstractmethod
    def source_edits_for_witness(
        self,
        snapshot: CodemodSourceSnapshot,
        witness: CurrentLineWitness[SourceRecipeCandidateT, SourceRecipeNodeT],
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class DeriveCandidateCollectorOperation(
    LineWitnessSourceReprovedOperation[
        CandidateCollectorBoilerplateCandidate, ast.FunctionDef
    ],
):
    """Replace one proved forwarding method with its collector declaration."""

    candidate_type = CandidateCollectorBoilerplateCandidate
    target_node_kind = AstTargetNodeKind.METHOD

    def source_edits_for_witness(
        self,
        snapshot: CodemodSourceSnapshot,
        witness: CurrentLineWitness[
            CandidateCollectorBoilerplateCandidate, ast.FunctionDef
        ],
    ) -> tuple[NominalSourceEdit, ...]:
        return self.required_migration(snapshot, witness).source_edits(snapshot)

    def required_migration(
        self,
        snapshot: CodemodSourceSnapshot,
        witness: CurrentLineWitness[
            CandidateCollectorBoilerplateCandidate, ast.FunctionDef
        ],
    ) -> CandidateCollectorMigration:
        candidate = witness.candidate
        owner = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            SourceRewriteTarget(
                file_path=candidate.file_path, qualname=candidate.class_name
            ),
        )
        original = ClassAuthorityReferenceProof.from_native_declaration(
            snapshot,
            candidate.collector_scope.forwarding_detector_type,
            owner.file_path,
        )
        replacement = ClassAuthorityReferenceProof.from_native_declaration(
            snapshot,
            candidate.recommended_base_type,
            owner.file_path,
        )
        bindings = ModuleNominalBindingAuthority(
            snapshot.parsed_module_for_source_path(owner.file_path)
        )
        replaced_bases = tuple(
            base
            for base in owner.node.bases
            if bindings.qualified_name_at(base, line=owner.node.lineno)
            == original.authority_symbol
        )
        if len(replaced_bases) != 1:
            raise ValueError(
                "Collector migration requires one exact native forwarding base"
            )
        SourceNativeClassMro(
            snapshot,
            DerivedCandidateCollectorMixin.registered_collector_base_types(),
        ).require_inherited_method(
            NativeClassBaseSubstitution(
                snapshot.required_class_family_index.classes_by_symbol[
                    owner.required_symbol(snapshot)
                ],
                replaced_bases[0],
                NativeClassMroDeclaration(candidate.recommended_base_type),
            ),
            candidate.method_name,
        )
        return CandidateCollectorMigration(
            candidate=candidate,
            target=owner.target,
            node=owner.node,
            source=snapshot.sources_by_file_path[owner.file_path],
            import_source=replacement.required_import_source(snapshot),
            rationale=self.rationale,
            replaced_base=replaced_bases[0],
        )


class SourceReprovedLineWitnessFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ABC,
):
    """Compile evidence into an operation that owns current-source reproof."""

    operation_type: ClassVar[type[LineWitnessSourceReprovedOperation]]

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "line-witness recipe synthesis requires source context"
            )
        evidence = finding.primary_evidence
        if evidence is None:
            return self.rejected_evaluation(
                "line-witness recipe synthesis requires primary evidence"
            )
        try:
            source_path = SourcePathResolutionAuthority.from_source_index(
                evidence.file_path,
                context.source_index,
            ).required_path()
            target_ids = SourceIndexTargetSelector(
                node_kinds=(type(self).operation_type.target_node_kind,),
                file_paths=(source_path,),
                qualnames=(evidence.subject_symbol,),
            ).target_ids(context)
            if len(target_ids) != 1:
                raise ValueError(f"Line witness target count is {len(target_ids)}")
            operation = type(self).operation_type(
                target=SourceRewriteTarget(target_id=target_ids[0]),
            )
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=finding.summary,
            ).with_operation(operation)
        except (
            CodemodOperationPreflightError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(recipe)


@dataclass(frozen=True, kw_only=True)
class CollectorDeclarationOperation(
    LineWitnessSourceReprovedOperation[SourceRecipeCandidateT, SourceRecipeNodeT],
    ABC,
):
    """Derive a declaration rewrite and its import from current collector source."""

    declaration_factory: ClassVar[type | FunctionType]

    def source_edits_for_witness(
        self,
        snapshot: CodemodSourceSnapshot,
        witness: CurrentLineWitness[SourceRecipeCandidateT, SourceRecipeNodeT],
    ) -> tuple[NominalSourceEdit, ...]:
        CandidateCollectorBaseReference.require_for_target(witness.module, witness.node)
        source = snapshot.sources_by_file_path[witness.target.file_path]
        span = SourceTextSpan.from_offsets(
            SourceTextGeometry(source).required_node_offsets(witness.node)
        )
        if span.contains_comment(source):
            raise ValueError("declaration source contains comments")
        original = witness.module.source_segments.segment_for_node(witness.node)
        if original is None:
            raise ValueError("declaration source is unavailable")
        replacement = type(self).replacement_source(witness)
        factory = type(self).declaration_factory
        operations = (
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=witness.target.file_path),
                import_source=f"from {factory.__module__} import {factory.__name__}\n",
            ),
            PatchTargetOperation(
                target=SourceRewriteTarget(target_id=witness.target.target_id),
                replacements=(
                    SourceTextReplacement(old_source=original, new_source=replacement),
                ),
            ),
        )
        return tuple(
            edit
            for operation in operations
            for edit in operation.source_edits(snapshot)
        )

    @classmethod
    @abstractmethod
    def replacement_source(
        cls,
        witness: CurrentLineWitness[SourceRecipeCandidateT, SourceRecipeNodeT],
    ) -> str:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class DeclareCandidateFindingRendererOperation(
    CollectorDeclarationOperation[
        DirectBuildFindingRendererCandidate,
        ast.FunctionDef,
    ],
):
    """Replace one direct finding-builder method with a typed renderer value."""

    candidate_type = DirectBuildFindingRendererCandidate
    target_node_kind = AstTargetNodeKind.METHOD
    declaration_factory = CallableCandidateFindingRenderer

    @staticmethod
    def renderer_lambda(parameter_name: str, value: ast.expr) -> ast.Lambda:
        return ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self"), ast.arg(arg=parameter_name)],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=value,
        )

    @classmethod
    def replacement_source(
        cls,
        witness: CurrentLineWitness[
            DirectBuildFindingRendererCandidate, ast.FunctionDef
        ],
    ) -> str:
        call = witness.candidate.build_call(witness.node)
        if call is None:
            raise ValueError("finding renderer call is no longer derivable")
        assignment = ast.Assign(
            targets=[
                ast.Name(
                    id=DetectorDeclaration.finding_renderer_field_name,
                    ctx=ast.Store(),
                )
            ],
            value=ast.Call(
                func=ast.Name(
                    id=cls.declaration_factory.__name__,
                    ctx=ast.Load(),
                ),
                args=[cls.renderer_lambda(witness.candidate.parameter_name, call)],
                keywords=[],
            ),
        )
        return ast.unparse(ast.fix_missing_locations(assignment))


@dataclass(frozen=True, kw_only=True)
class DeclareDetectorClassOperation(
    CollectorDeclarationOperation[
        DeclarativeDetectorClassCandidate,
        ast.ClassDef,
    ],
):
    """Replace one metadata-only detector shell with its declaration call."""

    candidate_type = DeclarativeDetectorClassCandidate
    target_node_kind = AstTargetNodeKind.CLASS
    declaration_factory = staticmethod(declare_module_detector)

    @classmethod
    def replacement_source(
        cls,
        witness: CurrentLineWitness[DeclarativeDetectorClassCandidate, ast.ClassDef],
    ) -> str:
        candidate, node = witness.candidate, witness.node
        ModuleLexicalDependencyProjection.require_class_body_independence(node)
        assignment_values = candidate.assignment_values(node)
        if assignment_values is None:
            raise ValueError(
                f"{candidate.class_name!r} is no longer a declarative detector shell"
            )
        arguments = [
            ast.Name(id=candidate.candidate_type_name, ctx=ast.Load()),
        ]
        derived_names = DetectorDeclaration.derived_class_shell_field_names()
        keywords = [
            ast.keyword(arg=name, value=assignment_values[name])
            for name in assignment_values
            if name not in derived_names
        ]
        keywords.append(
            ast.keyword(
                arg=DetectorDeclarationOptions.detector_base_field_name,
                value=cast(ast.Subscript, node.bases[0]).value,
            )
        )
        if candidate.class_name != DetectorDeclaration.class_name_from_candidate_name(
            candidate.candidate_type_name
        ):
            keywords.append(
                ast.keyword(
                    arg=DetectorDeclarationOptions.detector_name_field_name,
                    value=ast.Constant(value=candidate.class_name),
                )
            )
        return ast.unparse(
            ast.Call(
                func=ast.Name(id=cls.declaration_factory.__name__, ctx=ast.Load()),
                args=arguments,
                keywords=keywords,
            )
        )


class DirectBuildFindingRendererFindingRecipeSynthesizer(
    SourceReprovedLineWitnessFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    operation_type = DeclareCandidateFindingRendererOperation


class DeclarativeDetectorClassFindingRecipeSynthesizer(
    SourceReprovedLineWitnessFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    operation_type = DeclareDetectorClassOperation


class CandidateCollectorBoilerplateFindingRecipeSynthesizer(
    SourceReprovedLineWitnessFindingRecipeSynthesizer,
    ClassFamilyAuthorityConcept,
):
    """Compile a forwarding-method finding through its current source witness."""

    operation_type = DeriveCandidateCollectorOperation
