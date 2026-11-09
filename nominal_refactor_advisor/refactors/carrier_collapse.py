from __future__ import annotations

import ast
import copy
from abc import (
    ABC,
    abstractmethod,
)
from collections import defaultdict
from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    ClassVar,
    cast,
)

from nominal_refactor_advisor.ast_field_decoders import variadic_parameters
from nominal_refactor_advisor.ast_tools import walk_function_body_nodes
from nominal_refactor_advisor.carrier_collapse import (
    CarrierCollapseCallEdge,
    CarrierCollapseParticipant,
    ClosedCarrierCollapseComponent,
)
from nominal_refactor_advisor.carrier_expansion import DeclaredCarrierExpansionBuilder
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
)
from nominal_refactor_advisor.refactors.class_base import ClassAuthorityReferenceProof
from nominal_refactor_advisor.codemod_authority_claims import AstTargetAuthorityClaim
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.name_algebra import CLASS_NAME_ALGEBRA
from nominal_refactor_advisor.parameter_conveyor import (
    ClosedParameterConveyorComponentBuilder,
)
from nominal_refactor_advisor.quoted_reference import (
    DeclaredInterpreter,
    ModuleReferenceScope,
)
from nominal_refactor_advisor.refactor_concepts import SemanticCarrierConcept
from nominal_refactor_advisor.semantic_descent import SemanticAuthorityKind
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from nominal_refactor_advisor.source_index import AstTargetDigest


class _CarrierCollapseNameLoadTransformer(ast.NodeTransformer):
    """Rewrite one participant's proven flat field parameters to its carrier."""

    def __init__(
        self,
        *,
        carrier_parameter_name: str,
        fields_by_parameter_name: Mapping[str, str],
    ) -> None:
        self.carrier_parameter_name = carrier_parameter_name
        self.fields_by_parameter_name = fields_by_parameter_name

    def visit_Name(self, node: ast.Name) -> ast.expr:
        field_name = self.fields_by_parameter_name.get(node.id)
        if field_name is None or not isinstance(node.ctx, ast.Load):
            return node
        return ast.copy_location(
            ast.Attribute(
                value=ast.Name(
                    id=self.carrier_parameter_name,
                    ctx=ast.Load(),
                ),
                attr=field_name,
                ctx=ast.Load(),
            ),
            node,
        )


@dataclass(frozen=True)
class _CarrierCollapseParticipantRewrite:
    participant: CarrierCollapseParticipant
    target: AstTargetDigest
    node: ast.FunctionDef | ast.AsyncFunctionDef
    field_mapping: tuple[tuple[str, str], ...]
    carrier_parameter_name: str
    carrier_annotation: ast.expr

    @property
    def fields_by_parameter_name(self) -> dict[str, str]:
        return {
            parameter_name: field_name
            for field_name, parameter_name in self.field_mapping
        }

    @property
    def mapped_parameter_names(self) -> frozenset[str]:
        return frozenset(self.fields_by_parameter_name)

    @property
    def transformer(self) -> _CarrierCollapseNameLoadTransformer:
        return _CarrierCollapseNameLoadTransformer(
            carrier_parameter_name=self.carrier_parameter_name,
            fields_by_parameter_name=self.fields_by_parameter_name,
        )

    @property
    def rewritten_arguments_source(self) -> str:
        arguments = copy.deepcopy(self.node.args)
        mapped_names = self.mapped_parameter_names
        positional_parameters = (*arguments.posonlyargs, *arguments.args)
        positional_defaults = (
            *(
                None
                for _ in range(len(positional_parameters) - len(arguments.defaults))
            ),
            *arguments.defaults,
        )
        retained_positional = tuple(
            (parameter, default)
            for parameter, default in zip(
                positional_parameters,
                positional_defaults,
                strict=True,
            )
            if parameter.arg not in mapped_names
        )
        retained_positional_only_count = sum(
            parameter.arg not in mapped_names for parameter in arguments.posonlyargs
        )
        arguments.posonlyargs = [
            parameter
            for parameter, _default in retained_positional[
                :retained_positional_only_count
            ]
        ]
        arguments.args = [
            parameter
            for parameter, _default in retained_positional[
                retained_positional_only_count:
            ]
        ]
        arguments.defaults = [
            default
            for _parameter, default in retained_positional
            if default is not None
        ]
        retained_keyword_only = tuple(
            (parameter, default)
            for parameter, default in zip(
                arguments.kwonlyargs,
                arguments.kw_defaults,
                strict=True,
            )
            if parameter.arg not in mapped_names
        )
        arguments.kwonlyargs = [
            parameter for parameter, _default in retained_keyword_only
        ]
        arguments.kw_defaults = [
            default for _parameter, default in retained_keyword_only
        ]
        arguments.kwonlyargs.append(
            ast.arg(
                arg=self.carrier_parameter_name,
                annotation=self.carrier_annotation,
            )
        )
        arguments.kw_defaults.append(None)
        return ast.unparse(arguments)


@dataclass(frozen=True)
class _ClosedCarrierCollapseSourceRewrite:
    """Derive one atomic physical rewrite from a current proven component."""

    context: CodemodSourceSnapshot
    component: ClosedCarrierCollapseComponent
    rationale: str

    _nested_scope_types: ClassVar[tuple[type[ast.AST], ...]] = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ClassDef,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    def __post_init__(self) -> None:
        self.component.require_rewrite_authority()

    @cached_property
    def geometries_by_file_path(self) -> dict[str, SourceTextGeometry]:
        return {
            file_path: SourceTextGeometry(source)
            for file_path, source in self.context.sources_by_file_path.items()
        }

    @cached_property
    def authority_target(self) -> ResolvedClassTarget:
        authority_symbol = self.component.authority.class_symbol
        matches = tuple(
            target
            for target in self.context.source_index.targets_matching_repository_symbol(
                authority_symbol
            )
            if target.is_class
        )
        if len(matches) != 1:
            raise ValueError(
                f"Carrier authority {authority_symbol!r} has {len(matches)} "
                "source targets"
            )
        target = matches[0]
        node = self.context.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"Carrier authority {authority_symbol!r} has no class node"
            )
        return ResolvedClassTarget(target, node)

    @cached_property
    def participant_rewrites(
        self,
    ) -> tuple[_CarrierCollapseParticipantRewrite, ...]:
        rewrites = []
        for participant in self.component.participants:
            target, node = self._participant_target(participant)
            field_mapping = self.component.field_mapping_by_participant[
                participant.symbol
            ]
            self._require_reconstructible_participant(
                node,
                self.geometries_by_file_path[target.file_path],
            )
            rewrites.append(
                _CarrierCollapseParticipantRewrite(
                    participant=participant,
                    target=target,
                    node=node,
                    field_mapping=field_mapping,
                    carrier_parameter_name=self._carrier_parameter_name(
                        participant,
                        node,
                        frozenset(
                            parameter_name
                            for _field_name, parameter_name in field_mapping
                        ),
                    ),
                    carrier_annotation=self._carrier_annotation(target, node),
                )
            )
        return tuple(rewrites)

    def _carrier_annotation(
        self,
        target: AstTargetDigest,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ast.expr:
        """The carrier class as a reference where the participant's module resolves it, else quoted."""

        name = self.authority_target.name
        if ModuleReferenceScope(
            self.context.module_nodes_by_file_path[target.file_path]
        ).resolves(
            name,
            node.lineno,
            annotation=True,
            interpreter=DeclaredInterpreter.of(Path(target.file_path)),
        ):
            return ast.Name(id=name, ctx=ast.Load())
        return ast.Constant(value=name)

    @cached_property
    def participant_rewrites_by_symbol(
        self,
    ) -> dict[str, _CarrierCollapseParticipantRewrite]:
        return {
            rewrite.participant.symbol: rewrite for rewrite in self.participant_rewrites
        }

    @cached_property
    def carrier_parameter_names(self) -> dict[str, str]:
        return {
            participant_symbol: rewrite.carrier_parameter_name
            for participant_symbol, rewrite in self.participant_rewrites_by_symbol.items()
        }

    def source_edits(self) -> tuple[NominalSourceEdit, ...]:
        call_replacements = tuple(
            (edge.resolved_call.context.file_path, self._call_replacement(edge))
            for edge in self.component.edges
        )
        call_spans_by_file_path: dict[str, list[SourceTextSpan]] = defaultdict(list)
        replacements_by_file_path: dict[
            str,
            list[SourceTextSpanReplacement],
        ] = defaultdict(list)
        for file_path, replacement in call_replacements:
            call_spans_by_file_path[file_path].append(
                SourceTextSpan(replacement.start_offset, replacement.end_offset)
            )
            replacements_by_file_path[file_path].append(replacement)
        for rewrite in self.participant_rewrites:
            geometry = self.geometries_by_file_path[rewrite.target.file_path]
            parameter_span = geometry.function_parameter_span(rewrite.node)
            replacements_by_file_path[rewrite.target.file_path].append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=parameter_span.start_offset,
                    end_offset=parameter_span.end_offset,
                    replacement_source=rewrite.rewritten_arguments_source,
                )
            )
            replacements_by_file_path[rewrite.target.file_path].extend(
                self._participant_name_replacements(
                    rewrite,
                    call_spans_by_file_path.get(rewrite.target.file_path, ()),
                )
            )
        physical_edits = tuple(
            edit
            for file_path, replacements in sorted(replacements_by_file_path.items())
            for edit in self.geometries_by_file_path[file_path].physical_edits(
                file_path=file_path,
                replacements=replacements,
                rationale=self.rationale
                or (
                    "Replace a closed flat parameter component with its existing "
                    "nominal carrier."
                ),
            )
        )
        return (*self.import_mutations, *physical_edits)

    @cached_property
    def import_mutations(self) -> tuple[ModuleImportMutation, ...]:
        imports_by_path_and_source = {
            (rewrite.target.file_path, import_source): ModuleImportMutation.from_source(
                file_path=rewrite.target.file_path,
                import_source=import_source,
                rationale=(
                    self.rationale
                    or "Import the nominal carrier used by a collapsed signature."
                ),
            )
            for rewrite in self.participant_rewrites
            if (
                import_source := ClassAuthorityReferenceProof.from_context(
                    self.context,
                    self.authority_target,
                    rewrite.target.file_path,
                ).required_import_source(self.context)
            )
            is not None
        }
        return tuple(imports_by_path_and_source.values())

    def _participant_target(
        self,
        participant: CarrierCollapseParticipant,
    ) -> tuple[AstTargetDigest, ast.FunctionDef | ast.AsyncFunctionDef]:
        declaration = participant.declaration
        matches = tuple(
            target
            for target in self.context.source_index.ast_targets
            if target.is_function_like
            and target.file_path == participant.context.file_path
            and target.qualname == declaration.identity.qualname
            and target.line == declaration.line
        )
        if len(matches) != 1:
            raise ValueError(
                f"Participant {participant.symbol!r} has {len(matches)} source targets"
            )
        target = matches[0]
        node = self.context.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Participant {participant.symbol!r} has no function node")
        return target, node

    def _require_reconstructible_participant(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        geometry: SourceTextGeometry,
    ) -> None:
        parameter_span = geometry.function_parameter_span(node)
        if geometry.span_contains_comment(parameter_span):
            raise ValueError(
                f"Participant {node.name!r} has comments inside its signature"
            )
        nested_scopes = tuple(
            nested
            for nested in walk_function_body_nodes(node.body)
            if isinstance(nested, self._nested_scope_types)
        )
        if nested_scopes:
            raise ValueError(
                f"Participant {node.name!r} contains nested lexical scopes"
            )
        if node.type_comment is not None:
            raise ValueError(f"Participant {node.name!r} has a function type comment")

    def _carrier_parameter_name(
        self,
        participant: CarrierCollapseParticipant,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        mapped_parameter_names: frozenset[str],
    ) -> str:
        class_name = self.component.authority.class_symbol.rsplit(".", 1)[-1]
        stem = "_".join(CLASS_NAME_ALGEBRA.ordered_tokens(class_name)) or "carrier"
        occupied_names = {
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
                *variadic_parameters(node.args),
            )
            if argument.arg not in mapped_parameter_names
        }
        occupied_names.update(
            name.id
            for name in walk_function_body_nodes(node.body)
            if isinstance(name, ast.Name) and name.id not in mapped_parameter_names
        )
        occupied_names.update(
            participant.context.flow.mutations_by_root_name.keys()
            - mapped_parameter_names
        )
        candidate = stem
        suffix = 2
        while candidate in occupied_names:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        return candidate

    def _call_replacement(
        self,
        edge: CarrierCollapseCallEdge,
    ) -> SourceTextSpanReplacement:
        resolved_call = edge.resolved_call
        geometry = self.geometries_by_file_path[resolved_call.context.file_path]
        source_span = resolved_call.call.source_span
        start_offset, end_offset = geometry.byte_span_offsets(source_span)
        span = SourceTextSpan(start_offset, end_offset)
        if geometry.span_contains_comment(span):
            raise ValueError(
                f"Component call at {resolved_call.context.file_path}:"
                f"{resolved_call.call.line} contains comments"
            )
        node = self._call_node(resolved_call.context.file_path, source_span)
        rewritten = copy.deepcopy(node)
        mapped_names = frozenset(
            parameter_name for _field_name, parameter_name in edge.field_mapping
        )
        positional_parameter_names = tuple(
            parameter.name
            for parameter in resolved_call.call_signature.parameters
            if parameter.kind.accepts_positional and not parameter.kind.variadic
        )
        rewritten.args = [
            argument
            for index, argument in enumerate(rewritten.args)
            if index >= len(positional_parameter_names)
            or positional_parameter_names[index] not in mapped_names
        ]
        rewritten.keywords = [
            keyword for keyword in rewritten.keywords if keyword.arg not in mapped_names
        ]
        for source_participant_symbol in edge.carrier_source_participant_symbols:
            transformer = self.participant_rewrites_by_symbol[
                source_participant_symbol
            ].transformer
            rewritten.args = [
                cast(ast.expr, transformer.visit(argument))
                for argument in rewritten.args
            ]
            rewritten.keywords = [
                ast.keyword(
                    arg=keyword.arg,
                    value=cast(ast.expr, transformer.visit(keyword.value)),
                )
                for keyword in rewritten.keywords
            ]
        rewritten.keywords.append(
            ast.keyword(
                arg=self.carrier_parameter_names[edge.callee_symbol],
                value=edge.carrier_value_reference(
                    self.carrier_parameter_names
                ).as_expression(),
            )
        )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=ast.unparse(rewritten),
        )

    def _call_node(self, file_path: str, source_span: SourceByteSpan) -> ast.Call:
        matches = tuple(
            node
            for node in ast.walk(self.context.module_nodes_by_file_path[file_path])
            if isinstance(node, ast.Call)
            and SourceByteSpan.from_node(node) == source_span
        )
        if len(matches) != 1:
            raise ValueError(
                f"Component call span in {file_path!r} resolved to {len(matches)} nodes"
            )
        return matches[0]

    def _participant_name_replacements(
        self,
        rewrite: _CarrierCollapseParticipantRewrite,
        call_spans: Iterable[SourceTextSpan],
    ) -> tuple[SourceTextSpanReplacement, ...]:
        geometry = self.geometries_by_file_path[rewrite.target.file_path]
        excluded_spans = tuple(call_spans)
        replacements = []
        for node in walk_function_body_nodes(rewrite.node.body):
            if not (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in rewrite.fields_by_parameter_name
            ):
                continue
            start_offset, end_offset = geometry.byte_span_offsets(
                SourceByteSpan.require_node(node)
            )
            if any(
                span.start_offset <= start_offset and end_offset <= span.end_offset
                for span in excluded_spans
            ):
                continue
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    replacement_source=(
                        f"{rewrite.carrier_parameter_name}."
                        f"{rewrite.fields_by_parameter_name[node.id]}"
                    ),
                )
            )
        return tuple(replacements)


@dataclass(frozen=True, kw_only=True)
class CarrierCollapseOperationABC(RepositorySourceReprovedOperation, ABC):
    """Re-prove every carrier component before one authority-wide collapse."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return tuple(
            edit
            for component in self._current_components(snapshot)
            for edit in _ClosedCarrierCollapseSourceRewrite(
                context=snapshot,
                component=component,
                rationale=self.rationale,
            ).source_edits()
        )

    def _current_components(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        (
            _target_identifier,
            authority_target,
            _authority_node,
        ) = self.target_node_from_context(snapshot)
        if not authority_target.is_class:
            raise ValueError("carrier-collapse authority target must be a class")
        components = self.current_components_for_authority(
            snapshot,
            authority_target,
        )
        if not components:
            raise ValueError(
                f"Authority {authority_target.qualname!r} has no current "
                "carrier-collapse components"
            )
        for component in components:
            component.require_rewrite_authority()
        return components

    @abstractmethod
    def current_components_for_authority(
        self,
        snapshot: CodemodSourceSnapshot,
        authority_target: AstTargetDigest,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class CollapseClosedParameterConveyorOperation(CarrierCollapseOperationABC):
    """Collapse every closed constructor-derived conveyor for one authority."""

    def current_components_for_authority(
        self,
        snapshot: CodemodSourceSnapshot,
        authority_target: AstTargetDigest,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        authority_symbol = snapshot.source_index.symbol_for_target(authority_target)
        return tuple(
            component
            for component in ClosedParameterConveyorComponentBuilder.from_modules(
                snapshot.parsed_modules
            ).assessed_components()
            if component.authority.class_symbol == authority_symbol
        )


@dataclass(frozen=True, kw_only=True)
class CollapseDeclaredCarrierExpansionOperation(CarrierCollapseOperationABC):
    """Collapse every declaration-typed carrier expansion for one authority."""

    def current_components_for_authority(
        self,
        snapshot: CodemodSourceSnapshot,
        authority_target: AstTargetDigest,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        authority_symbol = snapshot.source_index.symbol_for_target(authority_target)
        builder = DeclaredCarrierExpansionBuilder.from_modules(snapshot.parsed_modules)
        return tuple(
            assessment
            for assessment in builder.assessed_components()
            if assessment.component.carrier_class_symbol == authority_symbol
        )


class CarrierCollapseFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    SemanticCarrierConcept,
    ABC,
):
    """Collapse a currently re-proven flat component into its carrier."""

    @classmethod
    @abstractmethod
    def carrier_collapse_operation(
        cls,
        target: SourceRewriteTarget,
    ) -> CarrierCollapseOperationABC:
        raise NotImplementedError

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation("carrier collapse requires source context")
        authority_location = finding.authority_evidence
        if authority_location is None:
            return self.rejected_evaluation(
                "carrier-collapse finding lacks authority evidence"
            )
        try:
            authority_target = context.required_class_target_for_authority_evidence(
                authority_location
            )
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        operation = type(self).carrier_collapse_operation(
            SourceRewriteTarget(target_id=authority_target.target_id)
        )
        recipe = (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=(
                    "Replace the complete flat parameter component with its "
                    "existing nominal carrier."
                ),
            )
            .with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    authority_target,
                    authority_kind=SemanticAuthorityKind.DATACLASS_SCHEMA,
                )
            )
            .with_operation(operation)
        )
        return self.executable_evaluation(recipe)
