"""Compact product-flow facts for closed-component refactor proofs.

The declarations in this module preserve enough source semantics to prove a
whole parameter-conveyor trajectory without retaining repository ASTs.  They do
not rank or emit refactors: a call edge is evidence, not an executable change.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import Self, TypeAlias

from .annotation_semantics import NOMINAL_ANNOTATION_SOURCE_AUTHORITY
from .ast_tools import (
    CollectedFamily,
    CompactModuleIdentity,
    ImportBoundNameProjection,
    ParsedModule,
    PythonModulePathIdentity,
)
from .call_binding import (
    CompactBoundCallArgument as CompactBoundCallArgument,
    CompactCallArgument as CompactCallArgument,
    CompactCallBinding as CompactCallBinding,
    CompactCallBindingViolation as CompactCallBindingViolation,
    CompactFunctionParameter as CompactFunctionParameter,
    CompactFunctionSignature as CompactFunctionSignature,
    CompactKeywordArgument as CompactKeywordArgument,
    CompactParameterKind as CompactParameterKind,
    ExactCompactCallBinding as ExactCompactCallBinding,
    ViolatedCompactCallBinding as ViolatedCompactCallBinding,
)
from .source_geometry import SourceByteSpan
from .value_expression import (
    CompactValueExpression as CompactValueExpression,
    LexicalValueReference as LexicalValueReference,
    OpaqueValueExpression as OpaqueValueExpression,
)

SourcePositionedNode: TypeAlias = ast.expr | ast.stmt | ast.ExceptHandler | ast.pattern


class CompactTransparentSignatureDecorator(StrEnum):
    """Known decorators which preserve callable binding/signature semantics."""

    ABSTRACT_METHOD = (
        "abstractmethod",
        (
            ("abstractmethod",),
            ("abc", "abstractmethod"),
        ),
    )
    CLASS_METHOD = "classmethod", (("classmethod",),)
    FINAL = (
        "final",
        (
            ("final",),
            ("typing", "final"),
            ("typing_extensions", "final"),
        ),
    )
    OVERRIDE = (
        "override",
        (
            ("override",),
            ("typing", "override"),
            ("typing_extensions", "override"),
        ),
    )
    STATIC_METHOD = "staticmethod", (("staticmethod",),)

    def __new__(
        cls,
        value: str,
        accepted_reference_parts: tuple[tuple[str, ...], ...],
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._accepted_reference_parts = accepted_reference_parts
        return member

    def matches(self, decorator: CompactValueExpression) -> bool:
        return bool(
            isinstance(decorator, LexicalValueReference)
            and decorator.parts in self._accepted_reference_parts
        )

    def matches_any(self, decorators: tuple[CompactValueExpression, ...]) -> bool:
        return any(self.matches(decorator) for decorator in decorators)

    @classmethod
    def recognizes(cls, decorator: CompactValueExpression) -> bool:
        return any(member.matches(decorator) for member in cls)


class CompactFunctionBindingKind(StrEnum):
    """Nominal callable binding form with member-owned matching semantics."""

    FUNCTION = "function", 0, False, None
    INSTANCE_METHOD = "instance_method", 1, True, None
    CLASS_METHOD = (
        "class_method",
        1,
        True,
        CompactTransparentSignatureDecorator.CLASS_METHOD,
    )
    STATIC_METHOD = (
        "static_method",
        0,
        True,
        CompactTransparentSignatureDecorator.STATIC_METHOD,
    )

    def __new__(
        cls,
        value: str,
        implicit_parameter_count: int,
        class_owned: bool,
        binding_decorator: CompactTransparentSignatureDecorator | None,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._implicit_parameter_count = implicit_parameter_count
        member._class_owned = class_owned
        member._binding_decorator = binding_decorator
        return member

    @property
    def implicit_parameter_count(self) -> int:
        return self._implicit_parameter_count

    def matches_declaration(
        self,
        owner_class_qualname: str | None,
        decorators: tuple[CompactValueExpression, ...],
    ) -> bool:
        if (owner_class_qualname is not None) != self._class_owned:
            return False
        if self._binding_decorator is not None:
            return self._binding_decorator.matches_any(decorators)
        if not self._class_owned:
            return True
        return not any(
            decorator.matches_any(decorators)
            for decorator in (
                CompactTransparentSignatureDecorator.CLASS_METHOD,
                CompactTransparentSignatureDecorator.STATIC_METHOD,
            )
        )

    @classmethod
    def from_declaration(
        cls,
        owner_class_qualname: str | None,
        decorators: tuple[CompactValueExpression, ...],
    ) -> Self:
        return next(
            member
            for member in cls
            if member.matches_declaration(owner_class_qualname, decorators)
        )


class CompactFlowOwnerKind(StrEnum):
    """Executable source scopes represented by compact flow facts."""

    MODULE = "module", True, False, False
    CLASS_BODY = "class_body", False, False, True
    FUNCTION = "function", False, True, False

    is_module_scope: bool
    is_function_scope: bool
    is_class_body_scope: bool

    def __new__(
        cls,
        value: str,
        is_module_scope: bool,
        is_function_scope: bool,
        is_class_body_scope: bool,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.is_module_scope = is_module_scope
        member.is_function_scope = is_function_scope
        member.is_class_body_scope = is_class_body_scope
        return member

    def deferred_binding_resolution(
        self, mutations: tuple[CompactLexicalMutation, ...]
    ) -> CompactBindingMutationResolution:
        """Module and class bodies finish before deferred namespace lookup."""

        if len(mutations) == 1 or not self.is_function_scope:
            return ExactCompactBindingMutation(mutations[-1])
        return OpenCompactBindingMutation(
            CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION
        )


class CompactControlBranchKind(StrEnum):
    """Typed child suites needed for conservative dominance reasoning."""

    IF_BODY = "if_body"
    IF_ELSE = "if_else"
    LOOP_BODY = "loop_body"
    LOOP_ELSE = "loop_else"
    TRY_BODY = "try_body"
    TRY_HANDLER = "try_handler"
    TRY_ELSE = "try_else"
    TRY_FINALLY = "try_finally"
    WITH_BODY = "with_body"
    MATCH_CASE = "match_case"


class CompactMutationKind(StrEnum):
    """Source operations which may change lexical value identity."""

    ASSIGNMENT = "assignment", False, False, False
    AUGMENTED_ASSIGNMENT = "augmented_assignment", False, False, False
    DELETION = "deletion", False, False, False
    DEFINITION = "definition", True, False, True
    IMPORT = "import", True, True, False
    ITERATION_BINDING = "iteration_binding", False, False, False
    CONTEXT_BINDING = "context_binding", False, False, False
    EXCEPTION_BINDING = "exception_binding", False, False, False
    PATTERN_BINDING = "pattern_binding", False, False, False

    preserves_nominal_identity: bool
    is_import_binding: bool
    is_definition_binding: bool

    def __new__(
        cls,
        value: str,
        preserves_nominal_identity: bool,
        is_import_binding: bool,
        is_definition_binding: bool,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.preserves_nominal_identity = preserves_nominal_identity
        member.is_import_binding = is_import_binding
        member.is_definition_binding = is_definition_binding
        return member

    def validate_import_origin(self, origin: str | None) -> None:
        if origin is not None and not self.is_import_binding:
            raise ValueError("Only import mutations carry an imported origin")


class CompactCallResultUse(StrEnum):
    """Immediate call-result use with member-owned binding requirements."""

    BOUND = "bound", True
    RETURNED = "returned", False
    DISCARDED = "discarded", False
    EMBEDDED = "embedded", False

    def __new__(cls, value: str, requires_binding: bool) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._requires_binding = requires_binding
        return member

    def validate_binding(self, binding: "LexicalValueReference | None") -> None:
        if (binding is not None) != self._requires_binding:
            raise ValueError(f"{self.value} call-result binding does not match its use")


@dataclass(frozen=True)
class CompactFunctionIdentity:
    """Import-stable identity of one declared Python function."""

    module_name: str
    qualname: str

    @property
    def symbol(self) -> str:
        return f"{self.module_name}.{self.qualname}"


@dataclass(frozen=True)
class CompactCallResult:
    """Validated call-result use and its declaration-owned binding payload."""

    use: CompactCallResultUse
    binding: LexicalValueReference | None = None

    def __post_init__(self) -> None:
        self.use.validate_binding(self.binding)


@dataclass(frozen=True)
class CompactCallArguments:
    """One argument list, shared by collected calls and authored call edits."""

    positional: tuple[CompactCallArgument, ...]
    keywords: tuple[CompactKeywordArgument, ...]

    @classmethod
    def from_call(cls, node: ast.Call) -> Self:
        return cls(
            positional=tuple(
                CompactCallArgument(
                    CompactValueExpression.project(
                        argument.value if isinstance(argument, ast.Starred) else argument
                    ),
                    is_unpacked=isinstance(argument, ast.Starred),
                )
                for argument in node.args
            ),
            keywords=tuple(
                CompactKeywordArgument(keyword.arg, CompactValueExpression.project(keyword.value))
                for keyword in node.keywords
            ),
        )

    @property
    def values(self) -> tuple[CompactValueExpression, ...]:
        return tuple(argument.value for argument in (*self.positional, *self.keywords))

    def bind_to(self, declaration: "CompactFunctionDeclaration") -> "CompactCallBinding":
        return declaration.bind_call(self.positional, self.keywords)


class CompactCallTargetReference(ABC):
    """Nominal call-target syntax with leaf-owned resolution behavior."""

    @property
    @abstractmethod
    def terminal_name(self) -> str | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def lexical_reference(self) -> LexicalValueReference | None:
        """Return exact lookup syntax when no receiver-type proof is required."""

    @abstractmethod
    def local_candidate_symbols(
        self,
        module_name: str,
        lexical_scope_qualnames: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return candidates provable without import or type resolution."""


class CurrentClassCallTargetReference(CompactCallTargetReference, ABC):
    """Call target whose terminal method is selected from the current class."""

    owner_class_qualname: str
    method_name: str

    @property
    def terminal_name(self) -> str:
        return self.method_name

    @property
    def lexical_reference(self) -> None:
        return None


@dataclass(frozen=True)
class BareCallTargetReference(CompactCallTargetReference):
    function_name: str

    @property
    def terminal_name(self) -> str:
        return self.function_name

    @property
    def lexical_reference(self) -> LexicalValueReference:
        return LexicalValueReference(self.function_name)

    def local_candidate_symbols(
        self,
        module_name: str,
        lexical_scope_qualnames: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            (
                f"{module_name}.{scope_qualname}.{self.function_name}"
                if scope_qualname
                else f"{module_name}.{self.function_name}"
            )
            for scope_qualname in lexical_scope_qualnames
        )


@dataclass(frozen=True)
class CurrentClassMethodReference(CurrentClassCallTargetReference):
    owner_class_qualname: str
    method_name: str

    def local_candidate_symbols(
        self,
        module_name: str,
        lexical_scope_qualnames: tuple[str, ...],
    ) -> tuple[str, ...]:
        del lexical_scope_qualnames
        return (f"{module_name}.{self.owner_class_qualname}.{self.method_name}",)


@dataclass(frozen=True)
class CurrentClassMemberMethodReference(CurrentClassCallTargetReference):
    """Method reached through an annotated member of the current class."""

    owner_class_qualname: str
    member_name: str
    method_name: str
    uses_runtime_class_lookup: bool

    @classmethod
    def from_expression(
        cls,
        expression: ast.expr,
        *,
        owner_class_qualname: str | None,
        receiver_name: str | None,
    ) -> "CurrentClassMemberMethodReference | None":
        if (
            owner_class_qualname is None
            or receiver_name is None
            or not isinstance(expression, ast.Attribute)
            or not isinstance(expression.value, ast.Attribute)
        ):
            return None
        member_access = expression.value
        if isinstance(member_access.value, ast.Name):
            if member_access.value.id != receiver_name:
                return None
            uses_runtime_class_lookup = False
        elif (
            isinstance(member_access.value, ast.Call)
            and isinstance(member_access.value.func, ast.Name)
            and member_access.value.func.id == "type"
            and len(member_access.value.args) == 1
            and isinstance(member_access.value.args[0], ast.Name)
            and member_access.value.args[0].id == receiver_name
            and not member_access.value.keywords
        ):
            uses_runtime_class_lookup = True
        else:
            return None
        return cls(
            owner_class_qualname=owner_class_qualname,
            member_name=member_access.attr,
            method_name=expression.attr,
            uses_runtime_class_lookup=uses_runtime_class_lookup,
        )

    def local_candidate_symbols(
        self,
        module_name: str,
        lexical_scope_qualnames: tuple[str, ...],
    ) -> tuple[str, ...]:
        del lexical_scope_qualnames
        return (
            f"{module_name}.{self.owner_class_qualname}."
            f"{self.member_name}.{self.method_name}",
        )


@dataclass(frozen=True)
class QualifiedCallTargetReference(CompactCallTargetReference):
    reference: LexicalValueReference

    @property
    def terminal_name(self) -> str:
        return self.reference.terminal_name

    @property
    def lexical_reference(self) -> LexicalValueReference:
        return self.reference

    def local_candidate_symbols(
        self,
        module_name: str,
        lexical_scope_qualnames: tuple[str, ...],
    ) -> tuple[str, ...]:
        del module_name, lexical_scope_qualnames
        return ()


@dataclass(frozen=True)
class DynamicCallTargetReference(CompactCallTargetReference):
    @property
    def terminal_name(self) -> None:
        return None

    @property
    def lexical_reference(self) -> None:
        return None

    def local_candidate_symbols(
        self,
        module_name: str,
        lexical_scope_qualnames: tuple[str, ...],
    ) -> tuple[str, ...]:
        del module_name, lexical_scope_qualnames
        return ()


@dataclass(frozen=True)
class CompactControlBranch:
    parent_statement_index: int
    kind: CompactControlBranchKind
    alternative_index: int = 0


@dataclass(frozen=True)
class CompactFlowPosition:
    """One event position in a typed statement-suite tree."""

    branch_path: tuple[CompactControlBranch, ...]
    statement_index: int
    event_index: int

    def dominates(self, other: "CompactFlowPosition") -> bool:
        if self.branch_path == other.branch_path:
            return (self.statement_index, self.event_index) < (
                other.statement_index,
                other.event_index,
            )
        if len(self.branch_path) >= len(other.branch_path):
            return False
        if other.branch_path[: len(self.branch_path)] != self.branch_path:
            return False
        child_branch = other.branch_path[len(self.branch_path)]
        return self.statement_index < child_branch.parent_statement_index


@dataclass(frozen=True)
class CompactLexicalMutation:
    reference: LexicalValueReference
    kind: CompactMutationKind
    position: CompactFlowPosition
    line: int
    imported_origin: str | None = None

    def __post_init__(self) -> None:
        self.kind.validate_import_origin(self.imported_origin)


class CompactFunctionTargetResolutionViolation(StrEnum):
    """Typed reasons a call target lacks one closed nominal declaration."""

    DYNAMIC_BINDING = "dynamic_binding"
    MISSING_DECLARATION = "missing_declaration"
    AMBIGUOUS_DECLARATION = "ambiguous_declaration"
    INCOMPLETE_RECEIVER_FAMILY = "incomplete_receiver_family"
    UNSUPPORTED_RECEIVER = "unsupported_receiver"
    CYCLIC_BINDING = "cyclic_binding"


class CompactBindingMutationResolution(ABC):
    """The source write selected for a name, or an unresolved lookup obligation."""

    @property
    @abstractmethod
    def mutation(self) -> CompactLexicalMutation | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def violation(self) -> CompactFunctionTargetResolutionViolation | None:
        raise NotImplementedError


@dataclass(frozen=True)
class ExactCompactBindingMutation(CompactBindingMutationResolution):
    selected_mutation: CompactLexicalMutation

    @property
    def mutation(self) -> CompactLexicalMutation:
        return self.selected_mutation

    @property
    def violation(self) -> None:
        return None


@dataclass(frozen=True)
class OpenCompactBindingMutation(CompactBindingMutationResolution):
    failure: CompactFunctionTargetResolutionViolation

    @property
    def mutation(self) -> None:
        return None

    @property
    def violation(self) -> CompactFunctionTargetResolutionViolation:
        return self.failure


class CompactValueOriginViolation(StrEnum):
    """Reason one lexical value lacks a single unchanged local origin."""

    CONTROL_FLOW_JOIN = "control_flow_join"
    INTERVENING_REBINDING = "intervening_rebinding"
    AMBIGUOUS_BINDING = "ambiguous_binding"
    CYCLIC_ALIAS = "cyclic_alias"


class CompactValueOriginResolution(ABC):
    """Nominal result of tracing one value through exact local aliases."""

    @property
    @abstractmethod
    def exact_origin(self) -> LexicalValueReference | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def possible_origins(self) -> tuple[LexicalValueReference, ...]:
        raise NotImplementedError

    @abstractmethod
    def through_alias(
        self,
        suffix: tuple[str, ...],
        binding_mutation: CompactLexicalMutation,
    ) -> "CompactValueOriginResolution":
        raise NotImplementedError


@dataclass(frozen=True)
class ExactCompactValueOrigin(CompactValueOriginResolution):
    origin: LexicalValueReference
    alias_chain: tuple[CompactLexicalMutation, ...] = ()

    @property
    def exact_origin(self) -> LexicalValueReference:
        return self.origin

    @property
    def possible_origins(self) -> tuple[LexicalValueReference, ...]:
        return (self.origin,)

    def through_alias(
        self,
        suffix: tuple[str, ...],
        binding_mutation: CompactLexicalMutation,
    ) -> "ExactCompactValueOrigin":
        return type(self)(
            LexicalValueReference(
                self.origin.root_name,
                (*self.origin.attribute_path, *suffix),
            ),
            (*self.alias_chain, binding_mutation),
        )


@dataclass(frozen=True)
class OpenCompactValueOrigin(CompactValueOriginResolution):
    candidates: tuple[LexicalValueReference, ...]
    violation: CompactValueOriginViolation

    @property
    def exact_origin(self) -> None:
        return None

    @property
    def possible_origins(self) -> tuple[LexicalValueReference, ...]:
        return self.candidates

    def through_alias(
        self,
        suffix: tuple[str, ...],
        binding_mutation: CompactLexicalMutation,
    ) -> "OpenCompactValueOrigin":
        del binding_mutation
        return type(self)(
            tuple(
                dict.fromkeys(
                    LexicalValueReference(
                        candidate.root_name,
                        (*candidate.attribute_path, *suffix),
                    )
                    for candidate in self.candidates
                )
            ),
            self.violation,
        )


@dataclass(frozen=True)
class CompactExactValueAlias:
    """One exact lexical binding to an unchanged value in a supported scope."""

    source: LexicalValueReference
    source_position: CompactFlowPosition
    binding_mutation: CompactLexicalMutation

    @property
    def target(self) -> LexicalValueReference:
        return self.binding_mutation.reference

    def source_for(self, reference: LexicalValueReference) -> LexicalValueReference:
        """Project a use's attribute suffix onto the captured source reference."""

        return LexicalValueReference(
            self.source.root_name,
            (*self.source.attribute_path, *reference.attribute_path),
        )


@dataclass(frozen=True)
class CompactCallableReferenceUse:
    target: CompactCallTargetReference
    position: CompactFlowPosition
    line: int


@dataclass(frozen=True)
class CompactFunctionCall:
    target: CompactCallTargetReference
    arguments: CompactCallArguments
    result: CompactCallResult
    position: CompactFlowPosition
    source_span: SourceByteSpan

    @property
    def line(self) -> int:
        return self.source_span.start_line_index + 1

    @property
    def result_use(self) -> CompactCallResultUse:
        return self.result.use

    @property
    def result_binding(self) -> LexicalValueReference | None:
        return self.result.binding

    def bind_to(self, declaration: "CompactFunctionDeclaration") -> CompactCallBinding:
        return self.arguments.bind_to(declaration)

    def product_construction(self) -> "CompactProductConstruction | None":
        if (
            self.result.use is not CompactCallResultUse.BOUND
            or self.result.binding is None
            or self.arguments.positional
            or any(argument.is_unpacked for argument in self.arguments.keywords)
            or len({argument.name for argument in self.arguments.keywords})
            != len(self.arguments.keywords)
        ):
            return None
        return CompactProductConstruction(
            target=self.target,
            result_binding=self.result.binding,
            field_arguments=self.arguments.keywords,
            position=self.position,
            line=self.line,
        )


class CompactLocalSignatureObserver(StrEnum):
    """Runtime operations which can observe a function's local signature."""

    LOCAL_MAPPING = (
        "local_mapping",
        (
            ("locals",),
            ("builtins", "locals"),
        ),
        True,
    )
    OBJECT_NAMESPACE = (
        "object_namespace",
        (
            ("vars",),
            ("builtins", "vars"),
        ),
        True,
    )
    LOCAL_NAMES = (
        "local_names",
        (
            ("dir",),
            ("builtins", "dir"),
        ),
        True,
    )
    DYNAMIC_EVALUATION = (
        "dynamic_evaluation",
        (
            ("eval",),
            ("exec",),
            ("builtins", "eval"),
            ("builtins", "exec"),
        ),
        False,
    )
    FRAME_ACCESS = (
        "frame_access",
        (
            ("_getframe",),
            ("currentframe",),
            ("inspect", "currentframe"),
            ("inspect", "stack"),
            ("sys", "_getframe"),
        ),
        False,
    )

    def __new__(
        cls,
        value: str,
        accepted_reference_parts: tuple[tuple[str, ...], ...],
        requires_no_arguments: bool,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._accepted_reference_parts = accepted_reference_parts
        member._requires_no_arguments = requires_no_arguments
        return member

    def observes(self, call: CompactFunctionCall) -> bool:
        reference = call.target.lexical_reference
        return bool(
            reference is not None
            and reference.parts in self._accepted_reference_parts
            and (
                not self._requires_no_arguments
                or not call.arguments.positional
                and not call.arguments.keywords
            )
        )

    @classmethod
    def observes_any(cls, calls: tuple[CompactFunctionCall, ...]) -> bool:
        return any(observer.observes(call) for observer in cls for call in calls)


@dataclass(frozen=True)
class CompactProductConstruction:
    """Derived explicit-keyword construction bound to one lexical value."""

    target: CompactCallTargetReference
    result_binding: LexicalValueReference
    field_arguments: tuple[CompactKeywordArgument, ...]
    position: CompactFlowPosition
    line: int

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(
            argument.name
            for argument in self.field_arguments
            if argument.name is not None
        )


@dataclass(frozen=True)
class CompactFunctionDeclaration:
    identity: CompactFunctionIdentity
    line: int
    end_line: int
    owner_class_qualname: str | None
    signature: CompactFunctionSignature
    decorators: tuple[CompactValueExpression, ...] = ()
    return_annotation_expression: str | None = None

    @property
    def return_annotation_reference_parts(self) -> tuple[str, ...] | None:
        return (
            None
            if self.return_annotation_expression is None
            else NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_parts_from_source(
                self.return_annotation_expression
            )
        )

    @cached_property
    def binding_kind(self) -> CompactFunctionBindingKind:
        return CompactFunctionBindingKind.from_declaration(
            self.owner_class_qualname,
            self.decorators,
        )

    def preserves_alias_call_binding(
        self, alias: CompactExactValueAlias, owner: CompactFlowOwner, module_name: str
    ) -> bool:
        """Prove free-function capture or same-class descriptor identity.

        Moving a descriptor through attribute access or into another class needs
        receiver evidence beyond a lexical alias; keep those bindings open.
        """

        return (
            self.owner_class_qualname is None and not owner.kind.is_class_body_scope
        ) or (
            owner.kind.is_class_body_scope
            and not alias.source.attribute_path
            and self.identity.module_name == module_name
            and self.owner_class_qualname == owner.qualname
        )

    @cached_property
    def signature_decorator_hazard(self) -> bool:
        binding_decorator_count = sum(
            decorator.matches_any(self.decorators)
            for decorator in (
                CompactTransparentSignatureDecorator.CLASS_METHOD,
                CompactTransparentSignatureDecorator.STATIC_METHOD,
            )
        )
        return binding_decorator_count > 1 or any(
            not CompactTransparentSignatureDecorator.recognizes(decorator)
            for decorator in self.decorators
        )

    @property
    def nominal_receiver_name(self) -> str | None:
        if (
            self.binding_kind.implicit_parameter_count != 1
            or not self.signature.parameters
        ):
            return None
        return self.signature.parameters[0].name

    @cached_property
    def call_signature(self) -> CompactFunctionSignature:
        return self.signature.without_leading_parameters(
            self.binding_kind.implicit_parameter_count
        )

    def bind_call(
        self,
        positional_arguments: tuple[CompactCallArgument, ...],
        keyword_arguments: tuple[CompactKeywordArgument, ...],
    ) -> CompactCallBinding:
        if self.signature_decorator_hazard:
            return ViolatedCompactCallBinding(
                CompactCallBindingViolation.SIGNATURE_DECORATOR_HAZARD
            )
        if self.binding_kind.implicit_parameter_count > len(self.signature.parameters):
            return ViolatedCompactCallBinding(
                CompactCallBindingViolation.INVALID_IMPLICIT_PARAMETER
            )
        return self.call_signature.bind(positional_arguments, keyword_arguments)


@dataclass(frozen=True)
class CompactFlowOwner:
    kind: CompactFlowOwnerKind
    qualname: str


@dataclass(frozen=True)
class CompactFunctionFlow:
    owner: CompactFlowOwner
    lexical_scope_qualnames: tuple[str, ...]
    loaded_value_root_names: tuple[str, ...]
    calls: tuple[CompactFunctionCall, ...]
    callable_reference_uses: tuple[CompactCallableReferenceUse, ...]
    mutations: tuple[CompactLexicalMutation, ...]
    exact_value_aliases: tuple[CompactExactValueAlias, ...]
    global_binding_names: tuple[str, ...]
    nonlocal_binding_names: tuple[str, ...]

    @property
    def local_signature_is_observed(self) -> bool:
        return CompactLocalSignatureObserver.observes_any(self.calls)

    @cached_property
    def mutations_by_root_name(self) -> dict[str, tuple[CompactLexicalMutation, ...]]:
        grouped: dict[str, list[CompactLexicalMutation]] = {}
        for mutation in self.mutations:
            if mutation.reference.attribute_path:
                continue
            grouped.setdefault(mutation.reference.root_name, []).append(mutation)
        return {name: tuple(mutations) for name, mutations in grouped.items()}

    def binding_resolution_for(
        self, root_name: str, use_position: CompactFlowPosition | None = None
    ) -> CompactBindingMutationResolution | None:
        """Select one binding from ordered flow facts; absence permits outer lookup."""

        mutations = self.mutations_by_root_name.get(root_name, ())
        if not mutations:
            return None
        if any(mutation.position.branch_path for mutation in mutations):
            return OpenCompactBindingMutation(
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
            )
        if use_position is None:
            return self.owner.kind.deferred_binding_resolution(mutations)
        dominating = tuple(
            mutation
            for mutation in mutations
            if mutation.position.dominates(use_position)
        )
        if not dominating:
            return OpenCompactBindingMutation(
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
            )
        return ExactCompactBindingMutation(dominating[-1])

    @cached_property
    def exact_aliases_by_binding_mutation(
        self,
    ) -> dict[CompactLexicalMutation, CompactExactValueAlias]:
        return {alias.binding_mutation: alias for alias in self.exact_value_aliases}

    def bound_call_result_for(
        self,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition,
    ) -> CompactFunctionCall | None:
        """Return the unique call whose unchanged result reaches one use."""

        matching_calls = []
        mutations = tuple(
            mutation
            for mutation in self.mutations
            if mutation.reference == reference
        )
        for call in self.calls:
            if (
                call.result.binding != reference
                or not call.position.dominates(use_position)
            ):
                continue
            binding_mutations = tuple(
                mutation
                for mutation in mutations
                if mutation.kind is CompactMutationKind.ASSIGNMENT
                and mutation.position.branch_path == call.position.branch_path
                and mutation.position.statement_index == call.position.statement_index
                and call.position.dominates(mutation.position)
            )
            if len(binding_mutations) != 1:
                continue
            binding_position = binding_mutations[0].position
            if any(
                binding_position.dominates(mutation.position)
                and mutation.position.dominates(use_position)
                for mutation in mutations
            ):
                continue
            matching_calls.append(call)
        return matching_calls[0] if len(matching_calls) == 1 else None

    def value_origin_for(
        self,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition,
    ) -> CompactValueOriginResolution:
        return self._value_origin_for(reference, use_position, frozenset())

    def _value_origin_for(
        self,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition,
        visited_root_names: frozenset[str],
    ) -> CompactValueOriginResolution:
        root_name = reference.root_name
        if root_name in visited_root_names:
            return OpenCompactValueOrigin(
                (reference,),
                CompactValueOriginViolation.CYCLIC_ALIAS,
            )
        mutations = self.mutations_by_root_name.get(root_name, ())
        if not mutations:
            return ExactCompactValueOrigin(reference)
        possible_origins = self._possible_alias_origins(reference, mutations)
        if len(mutations) != 1:
            return OpenCompactValueOrigin(
                possible_origins,
                CompactValueOriginViolation.AMBIGUOUS_BINDING,
            )
        mutation = mutations[0]
        if not mutation.position.dominates(use_position):
            return OpenCompactValueOrigin(
                possible_origins,
                CompactValueOriginViolation.CONTROL_FLOW_JOIN,
            )
        alias = self.exact_aliases_by_binding_mutation.get(mutation)
        if alias is None:
            return OpenCompactValueOrigin(
                possible_origins,
                CompactValueOriginViolation.INTERVENING_REBINDING,
            )
        source_resolution = self._value_origin_for(
            alias.source,
            alias.source_position,
            visited_root_names | {root_name},
        )
        return source_resolution.through_alias(
            reference.attribute_path,
            mutation,
        )

    def _possible_alias_origins(
        self,
        reference: LexicalValueReference,
        mutations: tuple[CompactLexicalMutation, ...],
    ) -> tuple[LexicalValueReference, ...]:
        return tuple(
            dict.fromkeys(
                (
                    reference,
                    *(
                        alias.source_for(reference)
                        for mutation in mutations
                        if (
                            alias := self.exact_aliases_by_binding_mutation.get(
                                mutation
                            )
                        )
                        is not None
                    ),
                )
            )
        )

    def local_candidate_symbols(
        self,
        target: CompactCallTargetReference,
        module_name: str,
    ) -> tuple[str, ...]:
        return target.local_candidate_symbols(
            module_name,
            self.lexical_scope_qualnames,
        )


@dataclass(frozen=True)
class CompactProductFlowModuleProjection(CompactModuleIdentity):
    """AST-free function declarations and source-ordered product-flow facts."""

    function_declarations: tuple[CompactFunctionDeclaration, ...]
    flows: tuple[CompactFunctionFlow, ...]


@dataclass(frozen=True)
class _FunctionContext:
    node: ast.FunctionDef | ast.AsyncFunctionDef
    declaration: CompactFunctionDeclaration
    lexical_scope_qualnames: tuple[str, ...]
    current_class_qualname: str | None


@dataclass(frozen=True)
class _ClassContext:
    node: ast.ClassDef
    qualname: str
    lexical_scope_qualnames: tuple[str, ...]
    current_class_qualname: str


class _DeclarationCollector(ast.NodeVisitor):
    """Collect declaration identities while preserving Python scope nesting."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.scope_names: list[str] = []
        self.scope_kinds: list[CompactFlowOwnerKind] = []
        self.function_qualnames: list[str] = []
        self.class_qualnames: list[str] = []
        self.imported_binding_names: set[str] = set()
        self.function_contexts: list[_FunctionContext] = []
        self.class_contexts: list[_ClassContext] = []

    def generic_visit(self, node: ast.AST) -> None:
        """Traverse declaration-bearing statement suites, not expressions."""

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self.visit(child)
            elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                self.generic_visit(child)

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        self.imported_binding_names.update(ImportBoundNameProjection(node).names())

    visit_ImportFrom = visit_Import

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = ".".join((*self.scope_names, node.name))
        lexical_scopes = _unique_strings(
            (qualname, *reversed(self.function_qualnames), "")
        )
        self.class_contexts.append(
            _ClassContext(node, qualname, lexical_scopes, qualname)
        )
        self.scope_names.append(node.name)
        self.scope_kinds.append(CompactFlowOwnerKind.CLASS_BODY)
        self.class_qualnames.append(qualname)
        self.generic_visit(node)
        self.class_qualnames.pop()
        self.scope_kinds.pop()
        self.scope_names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        qualname = ".".join((*self.scope_names, node.name))
        direct_class_owner = bool(
            self.scope_kinds and self.scope_kinds[-1].is_class_body_scope
        )
        decorators = tuple(
            (
                reference
                if (reference := LexicalValueReference.from_expression(decorator))
                is not None
                else OpaqueValueExpression()
            )
            for decorator in node.decorator_list
        )
        declaration = CompactFunctionDeclaration(
            identity=CompactFunctionIdentity(self.module_name, qualname),
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            owner_class_qualname=(
                self.class_qualnames[-1] if direct_class_owner else None
            ),
            signature=CompactFunctionSignature.from_arguments(node.args),
            decorators=decorators,
            return_annotation_expression=(
                None if node.returns is None else ast.unparse(node.returns)
            ),
        )
        lexical_scopes = _unique_strings(
            (qualname, *reversed(self.function_qualnames), "")
        )
        self.function_contexts.append(
            _FunctionContext(
                node=node,
                declaration=declaration,
                lexical_scope_qualnames=lexical_scopes,
                current_class_qualname=(
                    self.class_qualnames[-1] if self.class_qualnames else None
                ),
            )
        )
        self.scope_names.append(node.name)
        self.scope_kinds.append(CompactFlowOwnerKind.FUNCTION)
        self.function_qualnames.append(qualname)
        self.generic_visit(node)
        self.function_qualnames.pop()
        self.scope_kinds.pop()
        self.scope_names.pop()


class _CompactFlowCollector(ast.NodeVisitor):
    """Collect one source scope without descending into nested scope bodies."""

    def __init__(
        self,
        *,
        owner: CompactFlowOwner,
        module_identity: PythonModulePathIdentity,
        lexical_scope_qualnames: tuple[str, ...],
        current_class_qualname: str | None,
        current_class_receiver_name: str | None,
        declared_function_names: frozenset[str],
        declared_method_names: frozenset[str],
        imported_binding_names: frozenset[str],
        method_names_by_class: dict[str, frozenset[str]],
    ) -> None:
        self.owner = owner
        self.module_identity = module_identity
        self.lexical_scope_qualnames = lexical_scope_qualnames
        self.current_class_qualname = current_class_qualname
        self.current_class_receiver_name = current_class_receiver_name
        self.declared_function_names = declared_function_names
        self.declared_method_names = declared_method_names
        self.imported_binding_names = imported_binding_names
        self.method_names_by_class = method_names_by_class
        self.calls: list[CompactFunctionCall] = []
        self.callable_reference_uses: list[CompactCallableReferenceUse] = []
        self.mutations: list[CompactLexicalMutation] = []
        self.exact_value_aliases: list[CompactExactValueAlias] = []
        self.loaded_value_root_names: set[str] = set()
        self.global_binding_names: set[str] = set()
        self.nonlocal_binding_names: set[str] = set()
        self.branch_path: tuple[CompactControlBranch, ...] = ()
        self.statement_index = 0
        self.event_index = 0
        self.call_results: dict[int, CompactCallResult] = {}
        self.mutation_kind = CompactMutationKind.ASSIGNMENT

    def collect(self, statements: list[ast.stmt]) -> CompactFunctionFlow:
        self._collect_statements(statements)
        return CompactFunctionFlow(
            owner=self.owner,
            lexical_scope_qualnames=self.lexical_scope_qualnames,
            loaded_value_root_names=tuple(sorted(self.loaded_value_root_names)),
            calls=tuple(self.calls),
            callable_reference_uses=tuple(self.callable_reference_uses),
            mutations=tuple(self.mutations),
            exact_value_aliases=tuple(self.exact_value_aliases),
            global_binding_names=tuple(sorted(self.global_binding_names)),
            nonlocal_binding_names=tuple(sorted(self.nonlocal_binding_names)),
        )

    def _collect_statements(self, statements: list[ast.stmt]) -> None:
        saved_statement_index = self.statement_index
        saved_event_index = self.event_index
        for statement_index, statement in enumerate(statements):
            self.statement_index = statement_index
            self.event_index = 0
            self.visit(statement)
        self.statement_index = saved_statement_index
        self.event_index = saved_event_index

    def _collect_branch(
        self,
        statements: list[ast.stmt],
        kind: CompactControlBranchKind,
        alternative_index: int = 0,
    ) -> None:
        saved_path = self.branch_path
        self.branch_path = (
            *saved_path,
            CompactControlBranch(self.statement_index, kind, alternative_index),
        )
        self._collect_statements(statements)
        self.branch_path = saved_path

    def _position(self) -> CompactFlowPosition:
        position = CompactFlowPosition(
            self.branch_path,
            self.statement_index,
            self.event_index,
        )
        self.event_index += 1
        return position

    def _record_mutation(
        self,
        reference: LexicalValueReference,
        node: SourcePositionedNode,
        kind: CompactMutationKind | None = None,
        *,
        imported_origin: str | None = None,
    ) -> CompactLexicalMutation:
        mutation = CompactLexicalMutation(
            reference=reference,
            kind=self.mutation_kind if kind is None else kind,
            position=self._position(),
            line=node.lineno,
            imported_origin=imported_origin,
        )
        self.mutations.append(mutation)
        return mutation

    def _call_target(self, expression: ast.expr) -> CompactCallTargetReference:
        if isinstance(expression, ast.Name):
            return BareCallTargetReference(expression.id)
        member_method = CurrentClassMemberMethodReference.from_expression(
            expression,
            owner_class_qualname=self.current_class_qualname,
            receiver_name=self.current_class_receiver_name,
        )
        if member_method is not None:
            return member_method
        reference = LexicalValueReference.from_expression(expression)
        if reference is None:
            return DynamicCallTargetReference()
        if (
            self.current_class_qualname is not None
            and self.current_class_receiver_name is not None
            and reference.root_name == self.current_class_receiver_name
            and len(reference.attribute_path) == 1
        ):
            return CurrentClassMethodReference(
                self.current_class_qualname,
                reference.terminal_name,
            )
        return QualifiedCallTargetReference(reference)

    def _is_potential_callable_reference(
        self,
        reference: LexicalValueReference,
    ) -> bool:
        if reference.root_name in self.imported_binding_names:
            return True
        if not reference.attribute_path:
            return reference.root_name in self.declared_function_names
        if reference.terminal_name in self.declared_method_names:
            return True
        return bool(
            self.current_class_qualname is not None
            and self.current_class_receiver_name is not None
            and reference.root_name == self.current_class_receiver_name
            and len(reference.attribute_path) == 1
            and reference.terminal_name
            in self.method_names_by_class.get(self.current_class_qualname, ())
        )

    def _record_callable_reference(
        self,
        reference: LexicalValueReference,
        node: ast.expr,
    ) -> None:
        if not self._is_potential_callable_reference(reference):
            return
        self.callable_reference_uses.append(
            CompactCallableReferenceUse(
                target=self._call_target(node),
                position=self._position(),
                line=node.lineno,
            )
        )

    def visit_Call(self, node: ast.Call) -> None:
        target_reference = LexicalValueReference.from_expression(node.func)
        if target_reference is not None:
            self.loaded_value_root_names.add(target_reference.root_name)
        self._visit_call_target_evaluation(node.func)
        for argument in node.args:
            self.visit(
                argument.value if isinstance(argument, ast.Starred) else argument
            )
        for keyword in node.keywords:
            self.visit(keyword.value)
        result = self.call_results.get(
            id(node),
            CompactCallResult(CompactCallResultUse.EMBEDDED),
        )
        self.calls.append(
            CompactFunctionCall(
                target=self._call_target(node.func),
                arguments=CompactCallArguments.from_call(node),
                result=result,
                position=self._position(),
                source_span=SourceByteSpan.require_node(node),
            )
        )

    def _visit_call_target_evaluation(self, expression: ast.expr) -> None:
        if LexicalValueReference.from_expression(expression) is not None:
            return
        if isinstance(expression, ast.Attribute):
            self.visit(expression.value)
            return
        self.visit(expression)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        reference = LexicalValueReference.from_expression(node)
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            if reference is not None:
                self._record_mutation(reference, node)
            return
        if reference is not None:
            self.loaded_value_root_names.add(reference.root_name)
        if reference is not None and self._is_potential_callable_reference(reference):
            self._record_callable_reference(reference, node)
            return
        self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        reference = LexicalValueReference(node.id)
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._record_mutation(reference, node)
        elif isinstance(node.ctx, ast.Load):
            self.loaded_value_root_names.add(node.id)
            self._record_callable_reference(reference, node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.value, ast.Call):
            binding = LexicalValueReference.from_expression(node.targets[0])
            if binding is not None:
                self.call_results[id(node.value)] = CompactCallResult(
                    CompactCallResultUse.BOUND,
                    binding,
                )
        self.visit(node.value)
        mutations = self._visit_mutation_targets(
            node.targets,
            CompactMutationKind.ASSIGNMENT,
        )
        source = LexicalValueReference.from_expression(node.value)
        self._record_exact_value_aliases(node.targets, source, mutations)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None and not self.owner.kind.is_function_scope:
            return
        if isinstance(node.value, ast.Call):
            binding = LexicalValueReference.from_expression(node.target)
            if binding is not None:
                self.call_results[id(node.value)] = CompactCallResult(
                    CompactCallResultUse.BOUND,
                    binding,
                )
        if node.value is not None:
            self.visit(node.value)
        mutations = self._visit_mutation_targets(
            (node.target,),
            CompactMutationKind.ASSIGNMENT,
        )
        source = (
            None
            if node.value is None
            else LexicalValueReference.from_expression(node.value)
        )
        self._record_exact_value_aliases((node.target,), source, mutations)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._visit_mutation_targets(
            (node.target,),
            CompactMutationKind.AUGMENTED_ASSIGNMENT,
        )

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if isinstance(node.value, ast.Call):
            binding = LexicalValueReference.from_expression(node.target)
            if binding is not None:
                self.call_results[id(node.value)] = CompactCallResult(
                    CompactCallResultUse.BOUND,
                    binding,
                )
        self.visit(node.value)
        self._visit_mutation_targets((node.target,), CompactMutationKind.ASSIGNMENT)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._visit_mutation_targets(node.targets, CompactMutationKind.DELETION)

    def visit_Return(self, node: ast.Return) -> None:
        if isinstance(node.value, ast.Call):
            self.call_results[id(node.value)] = CompactCallResult(
                CompactCallResultUse.RETURNED
            )
        if node.value is not None:
            self.visit(node.value)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call):
            self.call_results[id(node.value)] = CompactCallResult(
                CompactCallResultUse.DISCARDED
            )
        self.visit(node.value)

    def _visit_mutation_targets(
        self,
        targets: tuple[ast.expr, ...] | list[ast.expr],
        kind: CompactMutationKind,
    ) -> tuple[CompactLexicalMutation, ...]:
        mutation_start = len(self.mutations)
        saved_kind = self.mutation_kind
        self.mutation_kind = kind
        for target in targets:
            self.visit(target)
        self.mutation_kind = saved_kind
        return tuple(self.mutations[mutation_start:])

    def _is_exact_value_alias_assignment(
        self,
        targets: tuple[ast.expr, ...] | list[ast.expr],
        source: LexicalValueReference | None,
    ) -> bool:
        return bool(
            source is not None
            and targets
            and all(
                isinstance(target, ast.Name)
                and (
                    not self.owner.kind.is_function_scope
                    or target.id not in self.global_binding_names
                    and target.id not in self.nonlocal_binding_names
                )
                for target in targets
            )
        )

    def _record_exact_value_aliases(
        self,
        targets: tuple[ast.expr, ...] | list[ast.expr],
        source: LexicalValueReference | None,
        mutations: tuple[CompactLexicalMutation, ...],
    ) -> None:
        if not self._is_exact_value_alias_assignment(targets, source):
            return
        assert source is not None
        self.exact_value_aliases.extend(
            CompactExactValueAlias(
                source=source,
                source_position=mutations[0].position,
                binding_mutation=mutation,
            )
            for mutation in mutations
        )

    def visit_Global(self, node: ast.Global) -> None:
        self.global_binding_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_binding_names.update(node.names)

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        for origin in ImportBoundNameProjection(node).origins(self.module_identity):
            self._record_mutation(
                LexicalValueReference(origin.bound_name),
                node,
                CompactMutationKind.IMPORT,
                imported_origin=origin.qualified_name,
            )

    visit_ImportFrom = visit_Import

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition_expressions(node)
        self._record_mutation(
            LexicalValueReference(node.name),
            node,
            CompactMutationKind.DEFINITION,
        )

    visit_AsyncFunctionDef = visit_FunctionDef

    def _visit_definition_expressions(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for annotation in (
            *(argument.annotation for argument in node.args.posonlyargs),
            *(argument.annotation for argument in node.args.args),
            *(argument.annotation for argument in node.args.kwonlyargs),
            None if node.args.vararg is None else node.args.vararg.annotation,
            None if node.args.kwarg is None else node.args.kwarg.annotation,
            node.returns,
        ):
            if annotation is not None:
                self.visit(annotation)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._record_mutation(
            LexicalValueReference(node.name),
            node,
            CompactMutationKind.DEFINITION,
        )

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self._collect_branch(node.body, CompactControlBranchKind.IF_BODY)
        self._collect_branch(node.orelse, CompactControlBranchKind.IF_ELSE)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._visit_mutation_targets(
            (node.target,), CompactMutationKind.ITERATION_BINDING
        )
        self._collect_branch(node.body, CompactControlBranchKind.LOOP_BODY)
        self._collect_branch(node.orelse, CompactControlBranchKind.LOOP_ELSE)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._collect_branch(node.body, CompactControlBranchKind.LOOP_BODY)
        self._collect_branch(node.orelse, CompactControlBranchKind.LOOP_ELSE)

    def visit_Try(self, node: ast.Try) -> None:
        self._collect_branch(node.body, CompactControlBranchKind.TRY_BODY)
        for index, handler in enumerate(node.handlers):
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name is not None:
                self._record_mutation(
                    LexicalValueReference(handler.name),
                    handler,
                    CompactMutationKind.EXCEPTION_BINDING,
                )
            self._collect_branch(
                handler.body,
                CompactControlBranchKind.TRY_HANDLER,
                index,
            )
        self._collect_branch(node.orelse, CompactControlBranchKind.TRY_ELSE)
        self._collect_branch(node.finalbody, CompactControlBranchKind.TRY_FINALLY)

    visit_TryStar = visit_Try

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._visit_mutation_targets(
                    (item.optional_vars,), CompactMutationKind.CONTEXT_BINDING
                )
        self._collect_branch(node.body, CompactControlBranchKind.WITH_BODY)

    visit_AsyncWith = visit_With

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for index, case in enumerate(node.cases):
            for name in _match_bound_names(case.pattern):
                self._record_mutation(
                    LexicalValueReference(name),
                    case.pattern,
                    CompactMutationKind.PATTERN_BINDING,
                )
            if case.guard is not None:
                self.visit(case.guard)
            self._collect_branch(
                case.body,
                CompactControlBranchKind.MATCH_CASE,
                index,
            )


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _match_bound_names(pattern: ast.pattern) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                node.name
                for node in ast.walk(pattern)
                if isinstance(node, (ast.MatchAs, ast.MatchStar))
                and node.name is not None
            }
        )
    )


def compact_product_flow_projection(
    parsed_module: ParsedModule,
) -> CompactProductFlowModuleProjection:
    """Project one parsed module into AST-free closed-flow evidence."""

    declarations = _DeclarationCollector(parsed_module.module_name)
    declarations.visit(parsed_module.module)
    declared_function_names = frozenset(
        context.declaration.identity.qualname.rsplit(".", 1)[-1]
        for context in declarations.function_contexts
    )
    class_qualnames = frozenset(
        context.qualname for context in declarations.class_contexts
    )
    methods_by_class: dict[str, set[str]] = {}
    for context in declarations.function_contexts:
        qualname = context.declaration.identity.qualname
        if "." not in qualname:
            continue
        owner_qualname, method_name = qualname.rsplit(".", 1)
        if owner_qualname in class_qualnames:
            methods_by_class.setdefault(owner_qualname, set()).add(method_name)
    method_names_by_class = {
        owner: frozenset(names) for owner, names in methods_by_class.items()
    }
    common = dict(
        module_identity=parsed_module.module_path_identity,
        declared_function_names=declared_function_names,
        declared_method_names=frozenset(
            method_name
            for method_names in method_names_by_class.values()
            for method_name in method_names
        ),
        imported_binding_names=frozenset(declarations.imported_binding_names),
        method_names_by_class=method_names_by_class,
    )
    flows = [
        _CompactFlowCollector(
            owner=CompactFlowOwner(CompactFlowOwnerKind.MODULE, ""),
            lexical_scope_qualnames=("",),
            current_class_qualname=None,
            current_class_receiver_name=None,
            **common,
        ).collect(parsed_module.module.body)
    ]
    flows.extend(
        _CompactFlowCollector(
            owner=CompactFlowOwner(
                CompactFlowOwnerKind.CLASS_BODY,
                context.qualname,
            ),
            lexical_scope_qualnames=context.lexical_scope_qualnames,
            current_class_qualname=context.current_class_qualname,
            current_class_receiver_name=None,
            **common,
        ).collect(context.node.body)
        for context in declarations.class_contexts
    )
    flows.extend(
        _CompactFlowCollector(
            owner=CompactFlowOwner(
                CompactFlowOwnerKind.FUNCTION,
                context.declaration.identity.qualname,
            ),
            lexical_scope_qualnames=context.lexical_scope_qualnames,
            current_class_qualname=context.current_class_qualname,
            current_class_receiver_name=context.declaration.nominal_receiver_name,
            **common,
        ).collect(context.node.body)
        for context in declarations.function_contexts
    )
    return CompactProductFlowModuleProjection(
        module_name=parsed_module.module_name,
        file_path=parsed_module.file_path,
        function_declarations=tuple(
            context.declaration for context in declarations.function_contexts
        ),
        flows=tuple(flows),
    )


class CompactProductFlowModuleProjectionFamily(
    CollectedFamily[CompactProductFlowModuleProjection]
):
    """Persist product-flow proof facts without retaining repository ASTs."""

    item_type = CompactProductFlowModuleProjection
    cache_payload_max_bytes = 5_000_000

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactProductFlowModuleProjection]:
        del cls
        return [compact_product_flow_projection(parsed_module)]
