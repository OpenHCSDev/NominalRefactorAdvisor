from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import cast

from .function_projection_targets import (
    FunctionReturnProjectionTarget,
    ProductFieldValue,
    ResolvedFunctionProjectionTarget,
)
from .dataclass_projection import (
    DataclassAuthorityMappingRecipeBuilder,
    DataclassPayloadAuthorityTarget,
    DataclassProjectionBoundary,
    SourceDerivedDataclassProjection,
    SourceDerivedDataclassProjectionOperation,
    SourceDerivedDataclassProjectionRecipeParts,
)
from .semantic_mirror import SemanticMirrorRecipeSeedLocations
from ..ast_field_decoders import variadic_parameters
from ..ast_projection import (
    CallKeywords,
    StatementValue,
)
from ..ast_tools import (
    statements_without_docstring,
    walk_function_body_nodes,
)
from ..codemod import NominalConstructorCall
from ..codemod_declaration_source import PythonExpressionSourceFormatter
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
from ..refactor_concepts import ConstructorKwargCarrierProjectionConcept


@dataclass(frozen=True)
class DataclassConstructorFieldArgument(ProductFieldValue):
    """One authority field and its value at an external constructor call."""


@dataclass(frozen=True)
class DataclassConstructorProjectionTarget(ResolvedFunctionProjectionTarget):
    """External nominal constructor call carrying all dataclass authority fields."""

    constructor: NominalConstructorCall
    field_arguments: tuple[DataclassConstructorFieldArgument, ...]
    remaining_keywords: tuple[ast.keyword, ...]

    @property
    def call_node(self) -> ast.Call:
        return self.constructor.call_node


@dataclass(frozen=True)
class DataclassConstructorProjectionMethod:
    """Direct authority method that forwards fields to one nominal constructor."""

    node: ast.FunctionDef
    constructor: NominalConstructorCall
    receiver_name: str
    parameter_names: tuple[str, ...]

    @property
    def method_name(self) -> str:
        return self.node.name

    @classmethod
    def candidates_from_authority(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        constructor_symbol: str,
        remaining_keyword_names: tuple[str, ...],
    ) -> tuple[DataclassConstructorProjectionMethod, ...]:
        return tuple(
            candidate
            for statement in authority.node.body
            if isinstance(statement, ast.FunctionDef)
            if (
                candidate := cls.from_method(
                    context,
                    authority,
                    statement,
                    constructor_symbol,
                    remaining_keyword_names,
                )
            )
            is not None
        )

    @classmethod
    def from_method(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        method_node: ast.FunctionDef,
        constructor_symbol: str,
        remaining_keyword_names: tuple[str, ...],
    ) -> DataclassConstructorProjectionMethod | None:
        body = statements_without_docstring(method_node.body)
        if (
            method_node.decorator_list
            or len(body) != 1
            or not isinstance(body[0], ast.Return)
            or not isinstance(body[0].value, ast.Call)
            or variadic_parameters(method_node.args)
        ):
            return None
        positional_parameters = (
            *method_node.args.posonlyargs,
            *method_node.args.args,
        )
        if not positional_parameters or len(method_node.args.posonlyargs) > 1:
            return None
        receiver = positional_parameters[0]
        if method_node.args.posonlyargs:
            keyword_parameters = (
                *method_node.args.args,
                *method_node.args.kwonlyargs,
            )
        else:
            keyword_parameters = (
                *method_node.args.args[1:],
                *method_node.args.kwonlyargs,
            )
        parameter_names = tuple(parameter.arg for parameter in keyword_parameters)
        if len(frozenset(parameter_names)) != len(parameter_names) or frozenset(
            parameter_names
        ) != frozenset(remaining_keyword_names):
            return None
        constructor = NominalConstructorCall.from_context(
            context,
            authority.file_path,
            method_node,
            body[0].value,
        )
        if (
            constructor is None
            or constructor.constructor_symbol != constructor_symbol
            or frozenset(constructor.keyword_names)
            != frozenset((*authority.field_names, *parameter_names))
        ):
            return None
        if any(
            not cls.keyword_forwards_receiver_field(
                constructor,
                field_name,
                receiver.arg,
            )
            for field_name in authority.field_names
        ):
            return None
        if any(
            not cls.keyword_forwards_parameter(constructor, parameter_name)
            for parameter_name in parameter_names
        ):
            return None
        return cls(
            node=method_node,
            constructor=constructor,
            receiver_name=receiver.arg,
            parameter_names=parameter_names,
        )

    @staticmethod
    def keyword_forwards_receiver_field(
        constructor: NominalConstructorCall,
        field_name: str,
        receiver_name: str,
    ) -> bool:
        keyword = constructor.keyword_argument(field_name)
        return bool(
            keyword is not None
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == receiver_name
            and keyword.value.attr == field_name
        )

    @staticmethod
    def keyword_forwards_parameter(
        constructor: NominalConstructorCall,
        parameter_name: str,
    ) -> bool:
        keyword = constructor.keyword_argument(parameter_name)
        return bool(
            keyword is not None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == parameter_name
        )


@dataclass(frozen=True)
class DataclassConstructorProjectionCandidate:
    """One constructor projection and its exact authority-method relation."""

    projection: DataclassConstructorProjectionTarget
    authority_method: DataclassConstructorProjectionMethod
    source_replacement: SourceTextReplacement


@dataclass(frozen=True)
class DataclassConstructorProjectionDerivation(
    SourceDerivedDataclassProjection[DataclassConstructorProjectionTarget]
):
    """Current-source proof for one equivalent constructor projection."""

    authority_method: DataclassConstructorProjectionMethod

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassConstructorProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        boundary.authority.require_transparent_direct_construction()
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node.body)
            if isinstance(node, ast.Return)
            for call_node in StatementValue.of(node).walk()
            if isinstance(call_node, ast.Call)
            if (
                candidate := cls.candidate_from_call(
                    context,
                    boundary,
                    call_node,
                )
            )
            is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one equivalent constructor projection; found {len(candidates)}"
            )
        candidate = candidates[0]
        return cls(
            authority=boundary.authority,
            projection=candidate.projection,
            source_replacement=candidate.source_replacement,
            import_sources=(
                (boundary.authority_import_source,)
                if boundary.authority_import_source is not None
                else ()
            ),
            authority_method=candidate.authority_method,
        )

    @classmethod
    def candidate_from_call(
        cls,
        context: CodemodSelectorContext,
        boundary: DataclassProjectionBoundary,
        call_node: ast.Call,
    ) -> DataclassConstructorProjectionCandidate | None:
        constructor = NominalConstructorCall.from_context(
            context,
            boundary.function.source_path,
            boundary.function.node,
            call_node,
        )
        if constructor is None:
            return None
        field_name_set = frozenset(boundary.authority.field_names)
        projected_field_names = tuple(
            name for name in constructor.keyword_names if name in field_name_set
        )
        if projected_field_names != boundary.authority.field_names:
            return None
        field_arguments = tuple(
            DataclassConstructorFieldArgument(
                field_name=field_name,
                value_node=constructor.required_keyword_argument(field_name).value,
            )
            for field_name in boundary.authority.field_names
        )
        remaining_keywords = tuple(
            keyword
            for keyword in constructor.keyword_arguments
            if keyword.arg not in field_name_set
        )
        if not cls.remaining_values_are_post_construction_safe(
            boundary.function,
            remaining_keywords,
        ):
            return None
        authority_methods = (
            DataclassConstructorProjectionMethod.candidates_from_authority(
                context,
                boundary.authority,
                constructor.constructor_symbol,
                tuple(cast(str, keyword.arg) for keyword in remaining_keywords),
            )
        )
        if len(authority_methods) != 1:
            return None
        projection = DataclassConstructorProjectionTarget(
            source_path=boundary.function.source_path,
            function_qualname=boundary.function.function_qualname,
            target=boundary.function.target,
            node=boundary.function.node,
            constructor=constructor,
            field_arguments=field_arguments,
            remaining_keywords=remaining_keywords,
        )
        source_replacement = cls.projection_replacement(
            context,
            boundary.authority,
            projection,
            authority_methods[0],
        )
        if source_replacement is None:
            return None
        return DataclassConstructorProjectionCandidate(
            projection=projection,
            authority_method=authority_methods[0],
            source_replacement=source_replacement,
        )

    @staticmethod
    def remaining_values_are_post_construction_safe(
        function: ResolvedFunctionProjectionTarget,
        keywords: tuple[ast.keyword, ...],
    ) -> bool:
        parameter_names = frozenset(function.target.parameters)
        return all(
            isinstance(keyword.value, ast.Constant)
            or (
                isinstance(keyword.value, ast.Name)
                and keyword.value.id in parameter_names
            )
            for keyword in keywords
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: DataclassConstructorProjectionTarget,
        authority_method: DataclassConstructorProjectionMethod,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        offsets = geometry.node_offsets(projection.call_node)
        if offsets is None:
            return None
        replacement_span = SourceTextSpan.from_offsets(offsets)
        if replacement_span.contains_comment(source):
            return None
        authority_instance = ast.Call(
            func=ast.Name(id=authority.name, ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(
                    arg=field.field_name,
                    value=copy.deepcopy(field.value_node),
                )
                for field in projection.field_arguments
            ],
        )
        replacement_call = ast.Call(
            func=ast.Attribute(
                value=authority_instance,
                attr=authority_method.method_name,
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                copy.deepcopy(keyword) for keyword in projection.remaining_keywords
            ],
        )
        replacement_source = PythonExpressionSourceFormatter().replacement_source(
            ast.fix_missing_locations(replacement_call),
            line_prefix=geometry.line_indent(replacement_span.start_offset),
        )
        return replacement_span.replacement(source, replacement_source)


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassConstructorProjectionOperation(
    SourceDerivedDataclassProjectionOperation[DataclassConstructorProjectionTarget]
):
    """Derive one constructor call through an equivalent dataclass method."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[DataclassConstructorProjectionTarget]:
        return DataclassConstructorProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class DataclassConstructorProjectionMappingRecipeBuilder(
    DataclassAuthorityMappingRecipeBuilder[
        ResolvedFunctionProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassConstructorProjectionTarget
        ],
    ],
    ConstructorKwargCarrierProjectionConcept,
):
    """Derive constructor keyword mirrors through an existing dataclass method."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return "dataclass constructor projection requires mapping metrics"
        if self.parts is not None:
            return "dataclass constructor projection has an executable authority recipe"
        return (
            "dataclass constructor projection requires one nominal constructor call "
            "that is equivalent to a direct authority method"
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
        if function_return is None:
            return False
        field_names = frozenset(self.finding.metrics.plan_field_names)
        return any(
            field_names
            <= frozenset(
                CallKeywords.of(call.keywords).names
            )
            for call in ast.walk(function_return.return_node.value)
            if isinstance(call, ast.Call)
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> ResolvedFunctionProjectionTarget | None:
        return FunctionReturnProjectionTarget.from_return_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: ResolvedFunctionProjectionTarget,
    ) -> (
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassConstructorProjectionTarget
        ]
        | None
    ):
        operation = DeriveDataclassConstructorProjectionOperation(
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
