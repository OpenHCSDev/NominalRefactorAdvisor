from __future__ import annotations

import ast
import copy
from abc import (
    ABC,
    abstractmethod,
)
from collections import defaultdict
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
)
from dataclasses import (
    dataclass,
    replace,
)
from enum import StrEnum
from typing import (
    Generic,
    Self,
    TypeVar,
    cast,
)

from .authority_extraction import SourceDerivedAuthorityProjectionOperation
from .dataclass_projection import (
    DataclassAuthorityReferenceProof,
    DataclassPayloadAuthorityTarget,
    ResolvedFunctionProjectionTarget,
)
from .finding_recipes import FindingRecipeSynthesizer
from ..annotation_semantics import NOMINAL_ANNOTATION_SOURCE_AUTHORITY
from ..ast_field_decoders import declared_annotations
from ..ast_projection import (
    CallKeywords,
    ExpressionReference,
)
from ..ast_tools import (
    ROOT_NAME_PROJECTION,
    root_agnostic_expression_fingerprint,
    statements_without_docstring,
    walk_function_body_nodes,
)
from ..class_families.parsed_index import IndexedClass
from ..class_index import ClassMethodReceiverRequirements
from ..codemod import NominalConstructorCall
from ..codemod_authority_claims import AstTargetAuthorityClaim
from ..codemod_declaration_source import (
    ClassBodySourceAuthority,
    PythonExpressionSourceFormatter,
)
from ..codemod_paths import SourcePathResolutionAuthority
from ..codemod_runtime import (
    CodemodSourceSnapshot,
    ExecutableRecipeEvaluation,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from ..codemod_selection_context import CodemodSelectorContext
from ..codemod_selector_models import SourceRewriteTarget
from ..codemod_source_edits import (
    NominalSourceEdit,
    SourceLineSpan,
    SourceSpanReplacement,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from ..codemod_target_selectors import SourceIndexTargetSelector
from ..finding_recipe_actions import FindingRecipeActionKey
from ..foreign_family import FunctionDefinition
from ..json_reports import json_report_field
from ..models import (
    MappingMetrics,
    RefactorFinding,
    SourceLocation,
)
from ..name_algebra import CLASS_NAME_ALGEBRA
from ..owner_candidates import (
    OwnerCandidate,
    OwnerCost,
    OwnerReceiptKind,
    ranked,
)
from ..receipts import Receipt
from ..refactor_concepts import (
    ConstructorKwargCarrierProjectionConcept,
    ConstructorKwargCollapseConcept,
    RefactorConcept,
)
from ..semantic_descent import (
    AuthorityClaimCarrier,
    SemanticAuthorityKind,
)
from ..semantic_match import (
    AstNameTemplateMatch,
    Maybe,
    single_item,
)
from ..source_index import (
    AstTargetDigest,
    AstTargetNodeKind,
)


@dataclass(frozen=True)
class RepeatedCallAuthorityParameter:
    """Shared generated parameter identity for repeated-call authorities."""

    name: str
    annotation: str


def _repeated_builder_value_source(
    geometry: SourceTextGeometry,
    value: ast.expr,
) -> str | None:
    return geometry.segment_for_node(value)


def _repeated_builder_root_name_source(
    geometry: SourceTextGeometry,
    value: ast.expr,
) -> str | None:
    del geometry
    roots = ROOT_NAME_PROJECTION.root_names(value)
    return next(iter(roots)) if len(roots) == 1 else None


class RepeatedBuilderParameterProjection(StrEnum):
    """How a generated builder parameter is recovered from a matched call."""

    VALUE = ("value", _repeated_builder_value_source)
    ROOT_NAME = ("root_name", _repeated_builder_root_name_source)

    def __new__(
        cls,
        value: str,
        source_projection: Callable[[SourceTextGeometry, ast.expr], str | None],
    ) -> RepeatedBuilderParameterProjection:
        member = str.__new__(cls, value)
        member._value_ = value
        member._source_projection = source_projection
        return member

    def source_from(
        self,
        geometry: SourceTextGeometry,
        value: ast.expr,
    ) -> str | None:
        """Project one matched argument through this declaration's semantics."""

        return self._source_projection(geometry, value)


RepeatedAuthorityParameterT = TypeVar(
    "RepeatedAuthorityParameterT",
    bound=RepeatedCallAuthorityParameter,
)


@dataclass(frozen=True)
class RepeatedBuilderAuthorityParameter(RepeatedCallAuthorityParameter):
    """One generated builder-authority parameter projected from call sites."""

    source_field_name: str
    value_projection: RepeatedBuilderParameterProjection = (
        RepeatedBuilderParameterProjection.VALUE
    )
    unwrap_single_tuple: bool = False


@dataclass(frozen=True)
class RepeatedBuilderConstructorArgument:
    """One constructor argument emitted by the generated builder authority."""

    field_name: str
    value_source: str


@dataclass(frozen=True)
class RepeatedAuthorityMethodName:
    """Shared method identity for generated repeated-call authorities."""

    method_name: str


@dataclass(frozen=True)
class RepeatedAuthorityMethodSpec(
    RepeatedAuthorityMethodName,
    Generic[RepeatedAuthorityParameterT],
):
    """Shared method signature for generated repeated-call authorities."""

    parameters: tuple[RepeatedAuthorityParameterT, ...]


@dataclass(frozen=True)
class RepeatedBuilderAuthorityMethod(
    RepeatedAuthorityMethodSpec[RepeatedBuilderAuthorityParameter],
    ConstructorKwargCollapseConcept,
):
    """Generated builder-authority method signature and constructor mapping."""

    constructor_arguments: tuple[RepeatedBuilderConstructorArgument, ...]

    @property
    def minimum_call_site_count(self) -> int:
        """Minimum repeated construction sites that prove this authority."""

        return 3


@dataclass(frozen=True)
class RepeatedBuilderSourceProjectionAuthorityMethod(
    RepeatedBuilderAuthorityMethod,
    ConstructorKwargCarrierProjectionConcept,
):
    """Builder method that derives constructor fields from one source object."""

    @property
    def minimum_call_site_count(self) -> int:
        """Two peer projections are sufficient to prove a shared mapping."""

        return 2


@dataclass(frozen=True)
class RepeatedBuilderCallSite:
    """One matching constructor call together with its lexical owner."""

    call: ast.Call
    participant: ResolvedFunctionProjectionTarget

    @property
    def source_identity(self) -> tuple[str, int, int]:
        """Physical identity used only to relate evidence to current source."""

        return (
            self.participant.target.target_id,
            self.call.lineno,
            self.call.col_offset,
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return the exact keyword schema observed at this constructor call."""

        keywords = CallKeywords.of(self.call.keywords)
        if self.call.args or keywords.splats:
            return ()
        return keywords.names

    @property
    def mapping_key(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return root-agnostic identity for this observed constructor mapping."""

        return (
            self.field_names,
            tuple(
                root_agnostic_expression_fingerprint(keyword.value)
                for keyword in self.call.keywords
            ),
        )

    def root_parameter(self, root_name: str) -> ast.arg | None:
        for parameter in (
            *self.participant.node.args.posonlyargs,
            *self.participant.node.args.args,
            *self.participant.node.args.kwonlyargs,
        ):
            if parameter.arg == root_name and declared_annotations(parameter):
                return parameter
        return None

    def owner_class_symbol(self, context: CodemodSelectorContext) -> str | None:
        """Return the nominal class that owns this participant method."""

        if not self.participant.target.is_method:
            return None
        if self.participant.owner_qualname is None:
            return None
        return context.required_class_family_index.symbol_for(
            file_path=self.participant.source_path,
            qualname=self.participant.owner_qualname,
        )


@dataclass(frozen=True, kw_only=True)
class ExistingMethodOwner(OwnerCandidate):
    """An existing shared-family method that constructs the observed record schema:
    an owner candidate for repeated construction (#67)."""

    declaration: IndexedClass = json_report_field(included=False)
    method: ast.FunctionDef = json_report_field(included=False)
    constructor: NominalConstructorCall = json_report_field(included=False)

    @property
    def subject(self) -> tuple[str, ...]:
        return (self.symbol,)

    def describe(self) -> str:
        return f"existing method `{self.symbol}`"

    @property
    def symbol(self) -> str:
        return f"{self.declaration.symbol}.{self.method.name}"

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> tuple[Self, ...]:
        """Find methods inherited by every participant without choosing among them."""

        class_index = context.required_class_family_index
        owner_symbols = tuple(
            call_site.owner_class_symbol(context) for call_site in call_sites
        )
        if not owner_symbols or any(symbol is None for symbol in owner_symbols):
            return ()
        nominal_families = tuple(
            frozenset((symbol, *class_index.ancestor_symbols(symbol)))
            for symbol in owner_symbols
            if symbol is not None
        )
        family_symbols = frozenset().union(*nominal_families)
        participant_nodes = frozenset(
            call_site.participant.node for call_site in call_sites
        )
        authority_symbol = authority.symbol(context)
        if authority_symbol is None or not call_sites:
            return ()
        field_names = call_sites[0].field_names
        return tuple(
            candidate
            for symbol in sorted(family_symbols)
            for declaration in (class_index.class_for(symbol),)
            if declaration is not None
            for method in declaration.node.body
            if isinstance(method, ast.FunctionDef)
            and (
                method not in participant_nodes
                or any(
                    owner_symbol != symbol and symbol in nominal_family
                    for owner_symbol, nominal_family in zip(
                        owner_symbols,
                        nominal_families,
                        strict=True,
                    )
                )
            )
            for candidate in (
                cls.from_method(
                    context,
                    declaration,
                    method,
                    authority_symbol,
                    field_names,
                ),
            )
            if candidate is not None
        )

    @classmethod
    def from_method(
        cls,
        context: CodemodSelectorContext,
        declaration: IndexedClass,
        method: ast.FunctionDef,
        authority_symbol: str,
        field_names: tuple[str, ...],
    ) -> Self | None:
        body = statements_without_docstring(method.body)
        if (
            len(body) != 1
            or not isinstance(body[0], ast.Return)
            or not isinstance(body[0].value, ast.Call)
        ):
            return None
        constructor = NominalConstructorCall.from_context(
            context,
            declaration.file_path,
            method,
            body[0].value,
        )
        if (
            constructor is None
            or constructor.constructor_symbol != authority_symbol
            or not RepeatedBuilderAuthorityDerivation.constructor_call_matches(
                constructor.call_node,
                field_names,
            )
        ):
            return None
        return cls(
            module_name=declaration.module_name,
            receipts=(),
            cost=OwnerCost(statements_rewritten=0, imports_added=0, cycles_created=0),
            declaration=declaration,
            method=method,
            constructor=constructor,
        )

    def invocation_signature(
        self,
    ) -> ConsumerFamilyBuilderInvocationSignature | None:
        arguments = self.method.args
        if (
            self.method.decorator_list
            or arguments.posonlyargs
            or arguments.vararg is not None
            or arguments.kwonlyargs
            or arguments.kwarg is not None
            or arguments.defaults
            or arguments.kw_defaults
            or not arguments.args
        ):
            return None
        receiver_name = arguments.args[0].arg
        parameter_names = tuple(argument.arg for argument in arguments.args[1:])
        parameter_occurrences = tuple(
            node.id
            for keyword in self.constructor.keyword_arguments
            for node in ast.walk(keyword.value)
            if isinstance(node, ast.Name) and node.id in parameter_names
        )
        if parameter_occurrences != parameter_names:
            return None
        if any(
            not self._field_expression_is_relocatable(
                keyword.value,
                receiver_name,
                frozenset(parameter_names),
            )
            for keyword in self.constructor.keyword_arguments
        ):
            return None
        return ConsumerFamilyBuilderInvocationSignature(
            receiver_name=receiver_name,
            parameter_names=parameter_names,
        )

    @staticmethod
    def _field_expression_is_relocatable(
        expression: ast.expr,
        receiver_name: str,
        parameter_names: frozenset[str],
    ) -> bool:
        referenced_parameters = frozenset(
            node.id
            for node in ast.walk(expression)
            if isinstance(node, ast.Name) and node.id in parameter_names
        )
        if referenced_parameters:
            return len(referenced_parameters) == 1
        return bool(
            isinstance(expression, ast.Constant)
            or (isinstance(expression, ast.Name) and expression.id == receiver_name)
            or (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Name)
                and expression.func.id == "type"
                and len(expression.args) == 1
                and isinstance(expression.args[0], ast.Name)
                and expression.args[0].id == receiver_name
                and not expression.keywords
            )
        )

    def required_target(self, context: CodemodSelectorContext) -> AstTargetDigest:
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=self.declaration.file_path,
            qualname=f"{self.declaration.qualname}.{self.method.name}",
        ).target_ids(context)
        if len(target_ids) != 1:
            raise ValueError(
                f"Consumer-family authority {self.symbol!r} is not one exact method"
            )
        return context.source_index.target_by_id[target_ids[0]]

    def scored(
        self,
        context: CodemodSelectorContext,
        call_sites: tuple[RepeatedBuilderCallSite, ...],
        descent: InheritedConsumerBuilderAuthorityDescent | None,
    ) -> Self:
        """The candidate with its receipts: the call sites it already serves, and
        whether every one of them substitutes into it exactly."""

        served = tuple(
            call_site for call_site in call_sites if self.is_inherited_by(context, call_site)
        )
        location = SourceLocation(self.declaration.file_path, self.method.lineno, self.symbol)
        receipts = [
            Receipt(
                OwnerReceiptKind.SERVES_CALL_SITES,
                f"`{self.declaration.qualname}` is inherited by the owners of "
                f"{len(served)} of the {len(call_sites)} constructor call(s)",
                (location,),
                count=len(served),
            )
        ]
        if descent is None:
            receipts.append(
                Receipt(
                    OwnerReceiptKind.NO_EXACT_SUBSTITUTION,
                    f"`{self.symbol}` has no complete exact parameter substitution "
                    "for its inheriting call sites",
                    (location,),
                )
            )
        return replace(
            self,
            receipts=tuple(receipts),
            cost=OwnerCost(
                statements_rewritten=0 if descent is None else len(descent.projections),
                imports_added=0,
                cycles_created=0,
            ),
        )

    def is_inherited_by(
        self,
        context: CodemodSelectorContext,
        call_site: RepeatedBuilderCallSite,
    ) -> bool:
        owner_symbol = call_site.owner_class_symbol(context)
        return bool(
            owner_symbol is not None
            and self.declaration.symbol
            in (
                owner_symbol,
                *context.required_class_family_index.ancestor_symbols(owner_symbol),
            )
        )

    def is_unique_method_authority_for(
        self,
        context: CodemodSelectorContext,
        call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> bool:
        """Prove no participant MRO has a competing repository declaration."""

        class_index = context.required_class_family_index
        owner_symbols = tuple(
            call_site.owner_class_symbol(context) for call_site in call_sites
        )
        if any(symbol is None for symbol in owner_symbols):
            return False
        return all(
            frozenset(
                symbol
                for symbol in (
                    owner_symbol,
                    *class_index.ancestor_symbols(owner_symbol),
                )
                for declaration in (class_index.class_for(symbol),)
                if declaration is not None
                if any(
                    isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
                    and statement.name == self.method.name
                    for statement in declaration.node.body
                )
            )
            == frozenset((self.declaration.symbol,))
            for owner_symbol in owner_symbols
            if owner_symbol is not None
        )


@dataclass(frozen=True)
class ConsumerFamilyBuilderInvocationSignature:
    """Exact instance-method signature available to inherited call sites."""

    receiver_name: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class ConsumerFamilyBuilderCallProjection:
    """One direct constructor call proven equal to an inherited builder call."""

    call_site: RepeatedBuilderCallSite
    receiver_name: str
    parameter_names: tuple[str, ...]
    match: AstNameTemplateMatch

    @classmethod
    def from_candidate(
        cls,
        candidate: ExistingMethodOwner,
        signature: ConsumerFamilyBuilderInvocationSignature,
        call_site: RepeatedBuilderCallSite,
    ) -> Self | None:
        receiver_name = ClassMethodReceiverRequirements.receiver_name(
            call_site.participant.node
        )
        call_keyword_names = tuple(keyword.arg for keyword in call_site.call.keywords)
        if (
            receiver_name is None
            or candidate.constructor.keyword_names != call_keyword_names
        ):
            return None
        match = AstNameTemplateMatch.from_expression_pairs(
            tuple(
                candidate.constructor.required_keyword_argument(field_name).value
                for field_name in candidate.constructor.keyword_names
            ),
            tuple(keyword.value for keyword in call_site.call.keywords),
            (signature.receiver_name, *signature.parameter_names),
        )
        if match is None or any(
            match.value_for(parameter_name) is None
            for parameter_name in signature.parameter_names
        ):
            return None
        matched_receiver = match.value_for(signature.receiver_name)
        if matched_receiver is not None and not (
            isinstance(matched_receiver, ast.Name)
            and matched_receiver.id == receiver_name
        ):
            return None
        return cls(
            call_site=call_site,
            receiver_name=receiver_name,
            parameter_names=signature.parameter_names,
            match=match,
        )

    def required_replacement(
        self,
        candidate: ExistingMethodOwner,
        geometry: SourceTextGeometry,
    ) -> SourceTextSpanReplacement:
        offsets = geometry.required_node_offsets(self.call_site.call)
        span = SourceTextSpan.from_offsets(offsets)
        if span.contains_comment(geometry.source):
            raise ValueError(
                "Inherited builder descent will not discard constructor comments"
            )
        parameter_values = tuple(
            (parameter_name, self.match.required_value_for(parameter_name))
            for parameter_name in self.parameter_names
        )
        replacement_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.receiver_name, ctx=ast.Load()),
                attr=candidate.method.name,
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                ast.keyword(arg=name, value=copy.deepcopy(value))
                for name, value in parameter_values
            ],
        )
        replacement_source = PythonExpressionSourceFormatter().replacement_source(
            ast.fix_missing_locations(replacement_call),
            line_prefix=geometry.line_indent(span.start_offset),
        )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=span.start_offset,
            end_offset=span.end_offset,
            replacement_source=replacement_source,
        )


class RepeatedBuilderSourceDerivation(ABC):
    """Source-reproved execution route for one repeated constructor family."""

    authority: DataclassPayloadAuthorityTarget
    call_sites: tuple[RepeatedBuilderCallSite, ...]

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> RepeatedBuilderSourceDerivation:
        authority = DataclassPayloadAuthorityTarget.from_rewrite_target(
            context,
            authority_reference,
        )
        call_sites = cls.anchored_call_sites(
            context,
            authority,
            projection_reference,
        )
        candidates = ExistingMethodOwner.from_context(
            context,
            authority,
            call_sites,
        )
        descents = {
            candidate.candidate_id: InheritedConsumerBuilderAuthorityDescent.from_candidate(
                context,
                authority,
                candidate,
                call_sites,
            )
            for candidate in candidates
        }
        owners = ranked(
            candidate.scored(context, call_sites, descents[candidate.candidate_id])
            for candidate in candidates
        )
        permitted = tuple(owner for owner in owners if owner.permitted)
        if len(permitted) == 1:
            return descents[permitted[0].candidate_id]
        if permitted:
            raise ValueError(
                "Repeated-builder descent has several executable consumer-family "
                "constructor authorities; pick one (confidence uncalibrated): "
                + OwnerCandidate.ranked_text(owners)
            )
        if owners:
            raise ValueError(
                "Existing consumer-family constructor authorities lack one "
                "complete exact parameter substitution: "
                + OwnerCandidate.ranked_text(owners)
            )
        return RepeatedBuilderAuthorityDerivation.from_authority(
            context,
            authority,
        )

    @staticmethod
    def anchored_call_sites(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection_reference: SourceRewriteTarget,
    ) -> tuple[RepeatedBuilderCallSite, ...]:
        participant = ResolvedFunctionProjectionTarget.from_rewrite_target(
            context,
            projection_reference,
        )
        call_sites = RepeatedBuilderAuthorityDerivation.constructor_call_sites(
            context,
            authority,
        )
        anchor_sites = tuple(
            call_site
            for call_site in call_sites
            if call_site.participant.target.target_id == participant.target.target_id
        )
        if len(anchor_sites) != 1:
            raise ValueError(
                "Repeated-builder participant must contain one nominal constructor "
                f"call; found {len(anchor_sites)}"
            )
        anchor_key = anchor_sites[0].mapping_key
        return tuple(
            call_site for call_site in call_sites if call_site.mapping_key == anchor_key
        )

    @property
    @abstractmethod
    def executable_declaration_type(self) -> type[RefactorConcept]:
        raise NotImplementedError

    @property
    @abstractmethod
    def authority_kind(self) -> SemanticAuthorityKind:
        raise NotImplementedError

    @abstractmethod
    def authority_target(self, context: CodemodSelectorContext) -> AstTargetDigest:
        raise NotImplementedError

    @property
    @abstractmethod
    def rewrite_call_sites(self) -> tuple[RepeatedBuilderCallSite, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def call_rewrite_rationale(self) -> str:
        raise NotImplementedError

    def authority_source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        return ()

    @abstractmethod
    def required_call_replacement(
        self,
        geometry: SourceTextGeometry,
        call_site: RepeatedBuilderCallSite,
    ) -> SourceTextSpanReplacement:
        raise NotImplementedError

    def required_source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        participants = tuple(
            dict.fromkeys(
                call_site.participant for call_site in self.rewrite_call_sites
            )
        )
        edits = list(self.authority_source_edits(context))
        for participant in participants:
            geometry = SourceTextGeometry(
                context.sources_by_file_path[participant.source_path]
            )
            replacements = tuple(
                self.required_call_replacement(geometry, call_site)
                for call_site in self.rewrite_call_sites
                if call_site.participant.target.target_id
                == participant.target.target_id
            )
            edits.append(
                SourceSpanReplacement(
                    file_path=participant.source_path,
                    span=SourceLineSpan(participant.target.line, participant.target.end_line),
                    replacement_lines=SourceTargetEditor.source_lines(
                        geometry.target_source_with_replacements(
                            participant.target,
                            replacements,
                        )
                    ),
                    rationale=self.call_rewrite_rationale,
                )
            )
        return tuple(edits)


@dataclass(frozen=True)
class InheritedConsumerBuilderAuthorityDescent(
    RepeatedBuilderSourceDerivation,
    ConstructorKwargCollapseConcept,
):
    """Route duplicated construction through one existing inherited method."""

    authority: DataclassPayloadAuthorityTarget
    candidate: ExistingMethodOwner
    call_sites: tuple[RepeatedBuilderCallSite, ...]
    projections: tuple[ConsumerFamilyBuilderCallProjection, ...]

    @classmethod
    def from_candidate(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        candidate: ExistingMethodOwner,
        call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> Self | None:
        signature = candidate.invocation_signature()
        if signature is None:
            return None
        family_call_sites = tuple(
            call_site
            for call_site in call_sites
            if candidate.is_inherited_by(context, call_site)
        )
        if not candidate.is_unique_method_authority_for(
            context,
            family_call_sites,
        ):
            return None
        consumer_call_sites = tuple(
            call_site
            for call_site in family_call_sites
            if call_site.participant.node is not candidate.method
        )
        projections = tuple(
            projection
            for call_site in consumer_call_sites
            if (
                projection := ConsumerFamilyBuilderCallProjection.from_candidate(
                    candidate,
                    signature,
                    call_site,
                )
            )
            is not None
        )
        if len(consumer_call_sites) < 2 or len(projections) != len(consumer_call_sites):
            return None
        return cls(
            authority=authority,
            candidate=candidate,
            call_sites=call_sites,
            projections=projections,
        )

    @property
    def executable_declaration_type(self) -> type[RefactorConcept]:
        return type(self)

    @property
    def authority_kind(self) -> SemanticAuthorityKind:
        return SemanticAuthorityKind.CLASS_FAMILY

    def authority_target(self, context: CodemodSelectorContext) -> AstTargetDigest:
        return self.candidate.required_target(context)

    @property
    def rewrite_call_sites(self) -> tuple[RepeatedBuilderCallSite, ...]:
        return tuple(projection.call_site for projection in self.projections)

    @property
    def call_rewrite_rationale(self) -> str:
        return (
            "Route repeated construction through its inherited consumer-family "
            "authority."
        )

    def required_call_replacement(
        self,
        geometry: SourceTextGeometry,
        call_site: RepeatedBuilderCallSite,
    ) -> SourceTextSpanReplacement:
        projection = single_item(
            tuple(
                projection
                for projection in self.projections
                if projection.call_site.source_identity == call_site.source_identity
            )
        )
        if projection is None:
            raise ValueError("Inherited builder descent lost one call projection")
        return projection.required_replacement(self.candidate, geometry)


@dataclass(frozen=True)
class RepeatedBuilderSourceProjectionTemplate:
    """One constructor call normalized by replacing its source root with `source`."""

    root_name: str
    source_annotation: str
    source_symbol: str
    normalized_value_fingerprints: tuple[str, ...]
    value_sources_by_field: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RepeatedBuilderInvariantFieldPlan:
    """One field slot in an invariant-selector builder authority."""

    constructor_argument: RepeatedBuilderConstructorArgument
    parameter: RepeatedBuilderAuthorityParameter | None = None
    constant_value: ast.AST | None = None


@dataclass(frozen=True)
class RepeatedBuilderAuthorityRecipeParts(AuthorityClaimCarrier):
    """Exact targets and source-derived operation for a builder extraction."""

    operation: DeriveRepeatedBuilderAuthorityOperation
    derivation: RepeatedBuilderSourceDerivation

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        return (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-extract-builder-authority",
                reason=(
                    "Move repeated constructor field mapping behind an owned "
                    "builder authority."
                ),
            )
            .with_authority_claim(self.authority_claim)
            .with_operation(self.operation)
        )


class RepeatedBuilderCallFindingRecipeSynthesizer(
    FindingRecipeSynthesizer,
    ConstructorKwargCollapseConcept,
):
    """Build class-owned constructor authority recipes for repeated builder calls."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "repeated-builder authority extraction requires a source selector context"
            )
        if context.class_family_index is None:
            context = context.execution_snapshot()
        parts, rejection_reason = self.recipe_parts_for_finding(finding, context)
        if rejection_reason:
            return self.rejected_evaluation(rejection_reason)
        if parts is None:
            return self.rejected_evaluation(
                "repeated-builder authority extraction found no recipe parts"
            )
        return ExecutableRecipeEvaluation(
            executable_recipe=parts.recipe_for(finding),
            evaluation_declaration_type=parts.derivation.executable_declaration_type,
        )

    def recipe_parts_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> tuple[RepeatedBuilderAuthorityRecipeParts | None, str]:
        try:
            evidence_targets = tuple(
                self.evidence_target(context, evidence) for evidence in finding.evidence
            )
            constructor_symbols = frozenset(
                context.class_reference_resolver_for_source_path(
                    target.file_path
                ).symbol_for_reference(call.func)
                for target, call in evidence_targets
            )
            if None in constructor_symbols or len(constructor_symbols) != 1:
                raise ValueError(
                    "Repeated-builder evidence must resolve one nominal constructor"
                )
            constructor_symbol = cast(str, next(iter(constructor_symbols)))
            indexed_class = context.required_class_family_index.class_for(
                constructor_symbol
            )
            if indexed_class is None:
                raise ValueError(
                    "Repeated-builder constructor is absent from the class index"
                )
            authority_target_ids = SourceIndexTargetSelector(
                node_kinds=(AstTargetNodeKind.CLASS,),
                file_paths=(indexed_class.file_path,),
                qualnames=(indexed_class.qualname,),
            ).target_ids(context)
            if len(authority_target_ids) != 1:
                raise ValueError(
                    "Repeated-builder constructor must resolve to one exact class"
                )
            constructor_target = context.source_index.target_by_id[
                authority_target_ids[0]
            ]
            projection_target = self.unique_constructor_participant(
                context,
                evidence_targets,
                constructor_symbol,
            )
            operation = DeriveRepeatedBuilderAuthorityOperation(
                target=SourceRewriteTarget(target_id=constructor_target.target_id),
                projection_target=SourceRewriteTarget.from_semantic_target(
                    projection_target
                ),
            )
            derivation = operation.required_derivation(context)
            evidence_source_identities = frozenset(
                (target.target_id, call.lineno, call.col_offset)
                for target, call in evidence_targets
            )
            if not evidence_source_identities.issubset(
                frozenset(
                    call_site.source_identity for call_site in derivation.call_sites
                )
            ):
                raise ValueError(
                    "Repeated-builder evidence does not belong to the unique "
                    "current proven family"
                )
            derivation.required_source_edits(context)
        except ValueError as error:
            return None, str(error)
        return (
            RepeatedBuilderAuthorityRecipeParts(
                authority_claim=AstTargetAuthorityClaim.from_target(
                    derivation.authority_target(context),
                    authority_kind=derivation.authority_kind,
                ),
                operation=operation,
                derivation=derivation,
            ),
            "",
        )

    @staticmethod
    def evidence_target(
        context: CodemodSelectorContext,
        evidence: SourceLocation,
    ) -> tuple[AstTargetDigest, ast.Call]:
        source_path = SourcePathResolutionAuthority.from_source_index(
            evidence.file_path,
            context.source_index,
        ).required_path()
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=source_path,
            qualname=evidence.subject_symbol,
        ).target_ids(context)
        if len(target_ids) != 1:
            raise ValueError(
                "Repeated-builder evidence must resolve one exact participant"
            )
        target = context.source_index.target_by_id[target_ids[0]]
        node = context.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            raise ValueError("Repeated-builder participant must be a function")
        resolver = context.class_reference_resolver_for_source_path(source_path)
        nominal_calls = tuple(
            child
            for child in walk_function_body_nodes(node.body)
            if isinstance(child, ast.Call)
            and child.lineno == evidence.line
            and resolver.symbol_for_reference(child.func) is not None
        )
        if len(nominal_calls) != 1:
            raise ValueError(
                "Repeated-builder evidence line must identify one nominal "
                "constructor call"
            )
        return target, nominal_calls[0]

    @staticmethod
    def unique_constructor_participant(
        context: CodemodSelectorContext,
        evidence_targets: tuple[tuple[AstTargetDigest, ast.Call], ...],
        constructor_symbol: str,
    ) -> AstTargetDigest:
        participants = tuple(
            dict.fromkeys(target for target, _call in evidence_targets)
        )
        candidates = tuple(
            target
            for target in participants
            for node in FunctionDefinition.within(
                (context.ast_target_nodes_by_id.get(target.target_id),)
            )
            if sum(
                1
                for call in walk_function_body_nodes(node.body)
                if isinstance(call, ast.Call)
                and context.class_reference_resolver_for_source_path(
                    target.file_path
                ).symbol_for_reference(call.func)
                == constructor_symbol
            )
            == 1
        )
        if not candidates:
            raise ValueError(
                "Repeated-builder evidence has no participant with one nominal "
                "constructor call"
            )
        return candidates[0]

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        if not isinstance(finding.metrics, MappingMetrics):
            return ()
        constructor_name = finding.metrics.plan_mapping_name
        if constructor_name is None:
            return ()
        subjects = {
            (evidence.file_path, evidence.subject_symbol)
            for evidence in finding.evidence
        }
        subjects.add((finding.evidence[0].file_path, constructor_name))
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            sorted(subjects),
        )


class RepeatedBuilderAuthorityMethodDeriver(ABC):
    """Derive one owned builder method from repeated constructor calls."""

    @classmethod
    def authority_method_or_none(
        cls,
        context: CodemodSelectorContext,
        field_names: tuple[str, ...],
        field_annotations: tuple[tuple[str, str], ...],
        matching_call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        return cls.source_projection_authority_method_or_none(
            context,
            field_annotations,
            matching_call_sites,
        ) or cls.invariant_selector_authority_method_or_none(
            context,
            field_names,
            field_annotations,
            matching_call_sites,
        )

    @classmethod
    def source_projection_authority_method_or_none(
        cls,
        context: CodemodSelectorContext,
        field_annotations: tuple[tuple[str, str], ...],
        matching_call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        field_names = tuple(field_name for field_name, _annotation in field_annotations)
        matching_calls = tuple(site.call for site in matching_call_sites)
        return (
            Maybe.of(matching_call_sites)
            .filter(bool)
            .project(
                lambda sites: cls.source_projection_templates(
                    context,
                    sites,
                    field_names,
                )
            )
            .filter(cls.source_projection_templates_share_shape)
            .combine(
                lambda templates: cls.source_projection_anchor_field_name(
                    matching_calls,
                    field_names,
                ),
                lambda templates, source_field_name: (
                    cls.source_projection_authority_method(
                        templates,
                        source_field_name,
                    )
                ),
            )
            .unwrap_or_none()
        )

    @classmethod
    def source_projection_authority_method(
        cls,
        templates: tuple[RepeatedBuilderSourceProjectionTemplate, ...],
        source_field_name: str,
    ) -> RepeatedBuilderAuthorityMethod:
        parameter_name = "source"
        return RepeatedBuilderSourceProjectionAuthorityMethod(
            method_name=f"from_{parameter_name}",
            parameters=(
                RepeatedBuilderAuthorityParameter(
                    name=parameter_name,
                    annotation=templates[0].source_annotation,
                    source_field_name=source_field_name,
                    value_projection=RepeatedBuilderParameterProjection.ROOT_NAME,
                ),
            ),
            constructor_arguments=tuple(
                RepeatedBuilderConstructorArgument(
                    field_name=field_name,
                    value_source=value_source,
                )
                for field_name, value_source in templates[0].value_sources_by_field
            ),
        )

    @classmethod
    def source_projection_templates(
        cls,
        context: CodemodSelectorContext,
        call_sites: tuple[RepeatedBuilderCallSite, ...],
        field_names: tuple[str, ...],
    ) -> tuple[RepeatedBuilderSourceProjectionTemplate, ...] | None:
        templates = tuple(
            cls.source_projection_template_for_call(context, site, field_names)
            for site in call_sites
        )
        if any(template is None for template in templates):
            return None
        return tuple(template for template in templates if template is not None)

    @staticmethod
    def source_projection_templates_share_shape(
        templates: tuple[RepeatedBuilderSourceProjectionTemplate, ...],
    ) -> bool:
        template_fingerprints = tuple(
            template.normalized_value_fingerprints for template in templates
        )
        source_symbols = tuple(template.source_symbol for template in templates)
        return len(set(template_fingerprints)) == 1 and len(set(source_symbols)) == 1

    @classmethod
    def source_projection_template_for_call(
        cls,
        context: CodemodSelectorContext,
        call_site: RepeatedBuilderCallSite,
        field_names: tuple[str, ...],
    ) -> RepeatedBuilderSourceProjectionTemplate | None:
        root_name = cls.call_source_root_name(call_site.call)
        if root_name is None:
            return None
        parameter = call_site.root_parameter(root_name)
        values_by_field = cls.call_keyword_values_by_field(
            call_site.call,
            field_names,
        )
        if parameter is None or values_by_field is None:
            return None
        annotation_reference = DataclassAuthorityReferenceProof.annotation_reference(
            parameter.annotation
        )
        if annotation_reference is None:
            return None
        source_symbol = context.class_reference_resolver_for_source_path(
            call_site.participant.source_path
        ).symbol_for_reference(annotation_reference)
        source_annotation = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.deferred_source_or_none(
            parameter.annotation
        )
        if source_symbol is None or source_annotation is None:
            return None
        return cls.source_projection_template(
            root_name,
            source_annotation,
            source_symbol,
            field_names,
            values_by_field,
        )

    @classmethod
    def source_projection_template(
        cls,
        root_name: str,
        source_annotation: str,
        source_symbol: str,
        field_names: tuple[str, ...],
        values_by_field: Mapping[str, ast.expr],
    ) -> RepeatedBuilderSourceProjectionTemplate:
        normalized_values = tuple(
            cls.source_value_with_root_name(value, root_name, "source")
            for value in values_by_field.values()
        )
        return RepeatedBuilderSourceProjectionTemplate(
            root_name=root_name,
            source_annotation=source_annotation,
            source_symbol=source_symbol,
            normalized_value_fingerprints=tuple(
                ast.dump(value, include_attributes=False) for value in normalized_values
            ),
            value_sources_by_field=tuple(
                (
                    field_name,
                    ast.unparse(
                        cls.source_value_with_root_name(
                            values_by_field[field_name],
                            root_name,
                            "source",
                        )
                    ),
                )
                for field_name in field_names
            ),
        )

    @staticmethod
    def call_source_root_name(call: ast.Call) -> str | None:
        roots: set[str] = set()
        for keyword in CallKeywords.of(call.keywords).named:
            roots.update(ROOT_NAME_PROJECTION.root_names(keyword.value))
        if len(roots) != 1:
            return None
        return next(iter(roots))

    @staticmethod
    def call_keyword_values_by_field(
        call: ast.Call,
        field_names: tuple[str, ...],
    ) -> dict[str, ast.expr] | None:
        values_by_field = dict(CallKeywords.of(call.keywords).by_name)
        if frozenset(values_by_field) != frozenset(field_names):
            return None
        return {field_name: values_by_field[field_name] for field_name in field_names}

    @classmethod
    def source_projection_anchor_field_name(
        cls,
        matching_calls: tuple[ast.Call, ...],
        field_names: tuple[str, ...],
    ) -> str | None:
        values_by_call = tuple(
            cls.call_keyword_values_by_field(call, field_names)
            for call in matching_calls
        )
        if any(values_by_field is None for values_by_field in values_by_call):
            return None
        for field_name in field_names:
            values = tuple(
                values_by_field[field_name]
                for values_by_field in values_by_call
                if values_by_field is not None
            )
            if all(
                len(ROOT_NAME_PROJECTION.root_names(value)) == 1 for value in values
            ):
                return field_name
        return None

    @staticmethod
    def source_value_with_root_name(
        value: ast.expr,
        old_root_name: str,
        new_root_name: str,
    ) -> ast.expr:
        class RootNameRewriter(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == old_root_name:
                    return ast.copy_location(
                        ast.Name(id=new_root_name, ctx=copy.deepcopy(node.ctx)),
                        node,
                    )
                return node

        rewritten = RootNameRewriter().visit(copy.deepcopy(value))
        if not isinstance(rewritten, ast.expr):
            raise TypeError(f"Expected expression rewrite, got {type(rewritten)!r}")
        return ast.fix_missing_locations(rewritten)

    @classmethod
    def invariant_selector_authority_method_or_none(
        cls,
        context: CodemodSelectorContext,
        field_names: tuple[str, ...],
        field_annotations: tuple[tuple[str, str], ...],
        matching_call_sites: tuple[RepeatedBuilderCallSite, ...],
    ) -> RepeatedBuilderAuthorityMethod | None:
        matching_calls = tuple(site.call for site in matching_call_sites)
        if not matching_call_sites:
            return None
        source_path = matching_call_sites[0].participant.source_path
        annotation_by_field = dict(field_annotations)
        return (
            Maybe.of(matching_calls)
            .filter(bool)
            .project(
                lambda calls: cls.invariant_selector_field_plans(
                    field_names,
                    annotation_by_field,
                    calls,
                    context=context,
                    source_path=source_path,
                )
            )
            .filter(cls.invariant_selector_plan_has_constant_and_parameter)
            .filter(cls.invariant_selector_plan_has_unique_parameters)
            .combine(
                cls.invariant_selector_method_name_for_plans,
                cls.invariant_selector_authority_method_from_plans,
            )
            .unwrap_or_none()
        )

    @classmethod
    def invariant_selector_field_plans(
        cls,
        field_names: tuple[str, ...],
        annotation_by_field: Mapping[str, str],
        matching_calls: tuple[ast.Call, ...],
        *,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> tuple[RepeatedBuilderInvariantFieldPlan, ...] | None:
        values_by_field = {
            field_name: tuple(
                keyword.value
                for call in matching_calls
                for keyword in call.keywords
                if keyword.arg == field_name
            )
            for field_name in field_names
        }
        plans = tuple(
            cls.invariant_selector_field_plan(
                field_name,
                annotation_by_field,
                values_by_field[field_name],
                call_count=len(matching_calls),
                context=context,
                source_path=source_path,
            )
            for field_name in field_names
        )
        if any(plan is None for plan in plans):
            return None
        return tuple(plan for plan in plans if plan is not None)

    @classmethod
    def invariant_selector_field_plan(
        cls,
        field_name: str,
        annotation_by_field: Mapping[str, str],
        values: tuple[ast.AST, ...],
        *,
        call_count: int,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> RepeatedBuilderInvariantFieldPlan | None:
        return (
            Maybe.of(values)
            .filter(lambda field_values: len(field_values) == call_count)
            .project(
                lambda field_values: cls.constant_invariant_field_plan(
                    field_name,
                    field_values,
                    context=context,
                    source_path=source_path,
                )
            )
            .unwrap_or_none()
        ) or (
            Maybe.of(values)
            .filter(lambda field_values: len(field_values) == call_count)
            .project(
                lambda field_values: cls.parameter_invariant_field_plan(
                    field_name,
                    annotation_by_field,
                    field_values,
                )
            )
            .unwrap_or_none()
        )

    @classmethod
    def constant_invariant_field_plan(
        cls,
        field_name: str,
        values: tuple[ast.AST, ...],
        *,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> RepeatedBuilderInvariantFieldPlan | None:
        value_sources = tuple(ast.unparse(value) for value in values)
        if len(set(value_sources)) != 1 or not cls.authority_constant_value(
            context,
            source_path,
            values[0],
        ):
            return None
        return RepeatedBuilderInvariantFieldPlan(
            constructor_argument=RepeatedBuilderConstructorArgument(
                field_name=field_name,
                value_source=value_sources[0],
            ),
            constant_value=values[0],
        )

    @classmethod
    def parameter_invariant_field_plan(
        cls,
        field_name: str,
        annotation_by_field: Mapping[str, str],
        values: tuple[ast.AST, ...],
    ) -> RepeatedBuilderInvariantFieldPlan | None:
        tuple_items = tuple(cls.single_tuple_item(value) for value in values)
        if any(item is None for item in tuple_items):
            return None
        parameter_annotation = cls.scalar_annotation(annotation_by_field[field_name])
        if parameter_annotation is None:
            return None
        parameter_name = cls.singular_field_name(field_name)
        return RepeatedBuilderInvariantFieldPlan(
            constructor_argument=RepeatedBuilderConstructorArgument(
                field_name=field_name,
                value_source=f"({parameter_name},)",
            ),
            parameter=RepeatedBuilderAuthorityParameter(
                name=parameter_name,
                annotation=parameter_annotation,
                source_field_name=field_name,
                unwrap_single_tuple=True,
            ),
        )

    @staticmethod
    def invariant_selector_plan_has_constant_and_parameter(
        plans: tuple[RepeatedBuilderInvariantFieldPlan, ...],
    ) -> bool:
        return any(plan.constant_value is not None for plan in plans) and any(
            plan.parameter is not None for plan in plans
        )

    @staticmethod
    def invariant_selector_plan_has_unique_parameters(
        plans: tuple[RepeatedBuilderInvariantFieldPlan, ...],
    ) -> bool:
        parameter_names = tuple(
            plan.parameter.name for plan in plans if plan.parameter is not None
        )
        return len(set(parameter_names)) == len(parameter_names)

    @classmethod
    def invariant_selector_method_name_for_plans(
        cls,
        plans: tuple[RepeatedBuilderInvariantFieldPlan, ...],
    ) -> str | None:
        return cls.invariant_selector_method_name(
            plan.constant_value for plan in plans if plan.constant_value is not None
        )

    @staticmethod
    def invariant_selector_authority_method_from_plans(
        plans: tuple[RepeatedBuilderInvariantFieldPlan, ...],
        method_name: str,
    ) -> RepeatedBuilderAuthorityMethod:
        parameters: list[RepeatedBuilderAuthorityParameter] = []
        for plan in plans:
            if plan.parameter is not None:
                parameters.append(plan.parameter)
        return RepeatedBuilderAuthorityMethod(
            method_name=method_name,
            parameters=tuple(parameters),
            constructor_arguments=tuple(plan.constructor_argument for plan in plans),
        )

    @classmethod
    def authority_constant_value(
        cls,
        context: CodemodSelectorContext,
        source_path: str,
        value: ast.AST,
    ) -> bool:
        if isinstance(value, ast.Constant):
            return True
        if isinstance(value, ast.Attribute):
            return (
                context.class_reference_resolver_for_source_path(
                    source_path
                ).symbol_for_reference(value.value)
                is not None
            )
        if isinstance(value, ast.Tuple | ast.List | ast.Set):
            return all(
                cls.authority_constant_value(context, source_path, item)
                for item in value.elts
            )
        return False

    @staticmethod
    def single_tuple_item(value: ast.AST) -> ast.AST | None:
        if not isinstance(value, ast.Tuple):
            return None
        if len(value.elts) != 1:
            return None
        return value.elts[0]

    @staticmethod
    def singular_field_name(field_name: str) -> str:
        if field_name.endswith("ies"):
            return f"{field_name[:-3]}y"
        if field_name.endswith("s"):
            return field_name[:-1]
        return field_name

    @staticmethod
    def scalar_annotation(annotation: str) -> str | None:
        try:
            annotation_node = ast.parse(annotation, mode="eval").body
        except SyntaxError:
            return None
        annotation_node = DataclassAuthorityReferenceProof.annotation_reference(
            annotation_node
        )
        if not isinstance(
            annotation_node, ast.Subscript
        ) or ExpressionReference.decoded(annotation_node.value).terminal_name not in {
            "tuple",
            "Tuple",
        }:
            return None
        slice_node = annotation_node.slice
        if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) != 2:
            return None
        element_type, repetition = slice_node.elts
        if not isinstance(repetition, ast.Constant) or repetition.value is not Ellipsis:
            return None
        return ast.unparse(element_type)

    @classmethod
    def invariant_selector_method_name(
        cls,
        constant_values: Iterable[ast.AST],
    ) -> str | None:
        tokens = tuple(
            token
            for value in constant_values
            for token in cls.invariant_value_tokens(value)
        )
        if not tokens:
            return None
        return f"for_{'_or_'.join(dict.fromkeys(tokens))}"

    @classmethod
    def invariant_value_tokens(cls, value: ast.AST) -> tuple[str, ...]:
        if isinstance(value, ast.Tuple | ast.List | ast.Set):
            return tuple(
                token
                for item in value.elts
                for token in cls.invariant_value_tokens(item)
            )
        if isinstance(value, ast.Attribute):
            return tuple(CLASS_NAME_ALGEBRA.ordered_tokens(value.attr))
        if isinstance(value, ast.Name):
            return tuple(CLASS_NAME_ALGEBRA.ordered_tokens(value.id))
        return ()

    def constructor_replacement_source(
        self,
        source: str,
        target: AstTargetDigest,
        node: ast.ClassDef,
        *,
        constructor_name: str,
        method: RepeatedBuilderAuthorityMethod,
    ) -> str:
        method_source = self.method_source(
            constructor_name=constructor_name,
            method=method,
        )
        insertion_point = ClassBodySourceAuthority(node=node, source=source)
        return SourceTextGeometry(source).target_source_with_replacements(
            target,
            (insertion_point.member_insertion_replacement((method_source,)),),
        )

    @staticmethod
    def method_source(
        *,
        constructor_name: str,
        method: RepeatedBuilderAuthorityMethod,
    ) -> str:
        parameter_lines = tuple(
            f"        {parameter.name}: {parameter.annotation},\n"
            for parameter in method.parameters
        )
        constructor_lines = tuple(
            f"            {argument.field_name}={argument.value_source},\n"
            for argument in method.constructor_arguments
        )
        return (
            "    @classmethod\n"
            f"    def {method.method_name}(\n"
            "        cls,\n"
            f"{''.join(parameter_lines)}"
            f'    ) -> "{constructor_name}":\n'
            "        return cls(\n"
            f"{''.join(constructor_lines)}"
            "        )\n\n"
        )

    @classmethod
    def call_replacement(
        cls,
        geometry: SourceTextGeometry,
        node: ast.AST,
        *,
        constructor_name: str,
        method: RepeatedBuilderAuthorityMethod,
    ) -> SourceTextSpanReplacement | None:
        if not isinstance(node, ast.Call):
            return None
        if not RepeatedBuilderAuthorityDerivation.constructor_call_matches(
            node,
            tuple(argument.field_name for argument in method.constructor_arguments),
        ):
            return None
        argument_sources = {
            parameter.name: cls.parameter_source(geometry, node, parameter)
            for parameter in method.parameters
        }
        if any(argument_sources[name] is None for name in argument_sources):
            return None
        start_offset, end_offset = geometry.required_node_offsets(node)
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=(
                f"{constructor_name}.{method.method_name}("
                f"{', '.join(f'{parameter.name}={argument_sources[parameter.name]}' for parameter in method.parameters)}"
                ")"
            ),
        )

    @classmethod
    def parameter_source(
        cls,
        geometry: SourceTextGeometry,
        node: ast.Call,
        parameter: RepeatedBuilderAuthorityParameter,
    ) -> str | None:
        values = tuple(
            keyword.value
            for keyword in node.keywords
            if keyword.arg == parameter.source_field_name
        )
        if len(values) != 1:
            return None
        value = values[0]
        if parameter.unwrap_single_tuple:
            value = cls.single_tuple_item(value)
            if value is None:
                return None
        return parameter.value_projection.source_from(geometry, value)


@dataclass(frozen=True)
class RepeatedBuilderAuthorityDerivation(
    RepeatedBuilderSourceDerivation,
    RepeatedBuilderAuthorityMethodDeriver,
):
    """Current-source proof for one batched constructor-authority extraction."""

    authority: DataclassPayloadAuthorityTarget
    participants: tuple[ResolvedFunctionProjectionTarget, ...]
    call_sites: tuple[RepeatedBuilderCallSite, ...]
    method: RepeatedBuilderAuthorityMethod

    @classmethod
    def from_authority(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
    ) -> RepeatedBuilderAuthorityDerivation:
        authority.require_complete_owned_schema(context)
        derivations = cls.proven_derivations(context, authority)
        if not derivations:
            raise ValueError(
                "Repeated-builder authority extraction requires a source projection "
                "or invariant selector axis"
            )
        if len(derivations) > 1:
            raise ValueError(
                f"Authority {authority.target.qualname!r} has {len(derivations)} "
                "current proven repeated-builder families"
            )
        derivation = derivations[0]
        method = derivation.method
        if authority.family_defines_method(context, method.method_name):
            raise ValueError(
                "Repeated-builder authority extraction will not overwrite or shadow "
                f"{method.method_name}"
            )
        return derivation

    @property
    def executable_declaration_type(self) -> type[RefactorConcept]:
        return type(self.method)

    @property
    def authority_kind(self) -> SemanticAuthorityKind:
        return SemanticAuthorityKind.DATACLASS_SCHEMA

    def authority_target(self, context: CodemodSelectorContext) -> AstTargetDigest:
        del context
        return self.authority.target

    @property
    def rewrite_call_sites(self) -> tuple[RepeatedBuilderCallSite, ...]:
        return self.call_sites

    @property
    def call_rewrite_rationale(self) -> str:
        return "Rewrite repeated construction through its owned authority."

    @classmethod
    def proven_derivations(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
    ) -> tuple[RepeatedBuilderAuthorityDerivation, ...]:
        grouped_call_sites: dict[tuple[str, ...], list[RepeatedBuilderCallSite]] = (
            defaultdict(list)
        )
        for call_site in cls.peer_call_sites(context, authority):
            fingerprint = cls.mapping_fingerprint(
                call_site.call,
                authority.field_names,
            )
            if fingerprint is not None:
                grouped_call_sites[fingerprint].append(call_site)
        derivations: list[RepeatedBuilderAuthorityDerivation] = []
        for grouped_sites in grouped_call_sites.values():
            call_sites = tuple(
                sorted(
                    grouped_sites,
                    key=lambda site: (
                        site.participant.source_path,
                        site.call.lineno,
                        site.call.col_offset,
                    ),
                )
            )
            participants = tuple(dict.fromkeys(site.participant for site in call_sites))
            if len(participants) < 2:
                continue
            method = cls.authority_method_or_none(
                context,
                authority.field_names,
                authority.field_annotations,
                call_sites,
            )
            if method is None or len(call_sites) < method.minimum_call_site_count:
                continue
            derivations.append(
                cls(
                    authority=authority,
                    participants=participants,
                    call_sites=call_sites,
                    method=method,
                )
            )
        return tuple(derivations)

    @classmethod
    def peer_call_sites(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
    ) -> tuple[RepeatedBuilderCallSite, ...]:
        return tuple(
            call_site
            for call_site in cls.constructor_call_sites(context, authority)
            if cls.constructor_call_matches(call_site.call, authority.field_names)
        )

    @classmethod
    def constructor_call_sites(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
    ) -> tuple[RepeatedBuilderCallSite, ...]:
        authority_symbol = authority.symbol(context)
        if authority_symbol is None:
            raise ValueError(
                "Repeated-builder authority extraction requires nominal class identity"
            )
        resolver = context.class_reference_resolver_for_source_path(authority.file_path)
        call_sites: list[RepeatedBuilderCallSite] = []
        for target in context.source_index.targets_by_file[authority.file_path]:
            if not target.is_function_like or target.qualname.startswith(
                f"{authority.target.qualname}."
            ):
                continue
            participant = ResolvedFunctionProjectionTarget.from_target(
                context,
                source_path=authority.file_path,
                target=target,
            )
            if participant is None:
                continue
            call_sites.extend(
                RepeatedBuilderCallSite(call=node, participant=participant)
                for node in walk_function_body_nodes(participant.node.body)
                if isinstance(node, ast.Call)
                and resolver.symbol_for_reference(node.func) == authority_symbol
                and not node.args
                and bool(node.keywords)
                and not CallKeywords.of(node.keywords).splats
            )
        return tuple(call_sites)

    @classmethod
    def mapping_fingerprint(
        cls,
        call: ast.Call,
        field_names: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        values_by_field = cls.call_keyword_values_by_field(call, field_names)
        if values_by_field is None:
            return None
        return tuple(
            root_agnostic_expression_fingerprint(values_by_field[field_name])
            for field_name in field_names
        )

    @staticmethod
    def constructor_call_matches(
        call: ast.Call,
        field_names: tuple[str, ...],
    ) -> bool:
        return bool(
            not call.args
            and not CallKeywords.of(call.keywords).splats
            and len(call.keywords) == len(field_names)
            and frozenset(CallKeywords.of(call.keywords).names)
            == frozenset(field_names)
        )

    def authority_source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        source = context.sources_by_file_path[self.authority.file_path]
        constructor_source = self.constructor_replacement_source(
            source,
            self.authority.target,
            self.authority.node,
            constructor_name=self.authority.name,
            method=self.method,
        )
        return (
            SourceSpanReplacement(
                file_path=self.authority.file_path,
                span=SourceLineSpan(self.authority.target.line, self.authority.target.end_line),
                replacement_lines=SourceTargetEditor.source_lines(constructor_source),
                rationale=(
                    "Insert the source-derived builder on its constructor authority."
                ),
            ),
        )

    def required_call_replacement(
        self,
        geometry: SourceTextGeometry,
        call_site: RepeatedBuilderCallSite,
    ) -> SourceTextSpanReplacement:
        call = call_site.call
        offsets = geometry.required_node_offsets(call)
        span = SourceTextSpan.from_offsets(offsets)
        if span.contains_comment(geometry.source):
            raise ValueError(
                "Repeated-builder authority extraction will not discard call comments"
            )
        replacement = self.call_replacement(
            geometry,
            call,
            constructor_name=self.authority.name,
            method=self.method,
        )
        if replacement is None:
            raise ValueError(
                "Repeated-builder call no longer satisfies its derived authority"
            )
        return replacement


@dataclass(frozen=True, kw_only=True)
class DeriveRepeatedBuilderAuthorityOperation(
    SourceDerivedAuthorityProjectionOperation
):
    """Re-prove the unique maximal builder family from its constructor owner."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> RepeatedBuilderSourceDerivation:
        if context.class_family_index is None:
            context = context.execution_snapshot()
        return RepeatedBuilderSourceDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.required_derivation(snapshot).required_source_edits(snapshot)
