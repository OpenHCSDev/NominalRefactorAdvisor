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
from itertools import chain
from typing import (
    Callable,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

from .annotation_semantics import NOMINAL_ANNOTATION_SOURCE_AUTHORITY
from .ast_tools import (
    CollectedFamily,
    CompactModuleIdentity,
    ParsedModule,
)
from .call_binding import (
    CallValueT,
    CompactBoundCallArgument as CompactBoundCallArgument,
    CompactCallArgument as CompactCallArgument,
    CompactCallBinding as CompactCallBinding,
    CompactCallBindingViolation as CompactCallBindingViolation,
    CompactFunctionParameter as CompactFunctionParameter,
    CompactFunctionSignature as CompactFunctionSignature,
    CompactKeywordArgument as CompactKeywordArgument,
    ExactCompactCallBinding as ExactCompactCallBinding,
    ViolatedCompactCallBinding as ViolatedCompactCallBinding,
)
from .descriptor_algebra import AliasProperty
from .lexical_bindings import (
    CompactParameterKind as CompactParameterKind,
    FunctionDefaultVisitor,
    ImportBoundNameProjection,
    ImportedNameOrigin,
)
from .native_compilation import NativeFunctionExecution
from .python_module_identity import PythonModulePathIdentity
from .source_geometry import SourceByteSpan
from .value_expression import (
    CompactValueExpression as CompactValueExpression,
    LexicalValueReference as LexicalValueReference,
    OpaqueValueExpression as OpaqueValueExpression,
)
from .value_graph import DataclassGraphValue

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


class CompactDescriptorAccess(StrEnum):
    """Descriptor lookup form with member-owned implicit argument projection."""

    DIRECT = "direct", lambda kind: kind.direct_implicit_parameter_count
    CLASS = "class", lambda kind: kind.class_implicit_parameter_count
    INSTANCE = "instance", lambda kind: kind.implicit_parameter_count
    UNKNOWN = "unknown", lambda kind: None

    def __new__(
        cls,
        value: str,
        parameter_count: Callable[[CompactFunctionBindingKind], int | None],
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._parameter_count = parameter_count
        return member

    def implicit_parameter_count(self, kind: CompactFunctionBindingKind) -> int | None:
        return self._parameter_count(kind)


class CompactFunctionBindingKind(StrEnum):
    """Nominal callable binding form with member-owned matching semantics."""

    FUNCTION = "function", 0, False, None, CompactDescriptorAccess.DIRECT, 0, 0
    INSTANCE_METHOD = (
        "instance_method",
        1,
        True,
        None,
        CompactDescriptorAccess.INSTANCE,
        0,
        0,
    )
    CLASS_METHOD = (
        "class_method",
        1,
        True,
        CompactTransparentSignatureDecorator.CLASS_METHOD,
        CompactDescriptorAccess.CLASS,
        1,
        None,
    )
    STATIC_METHOD = (
        "static_method",
        0,
        True,
        CompactTransparentSignatureDecorator.STATIC_METHOD,
        CompactDescriptorAccess.DIRECT,
        0,
        0,
    )

    receiver_access: CompactDescriptorAccess
    class_implicit_parameter_count: int
    direct_implicit_parameter_count: int | None

    def __new__(
        cls,
        value: str,
        implicit_parameter_count: int,
        class_owned: bool,
        binding_decorator: CompactTransparentSignatureDecorator | None,
        receiver_access: CompactDescriptorAccess,
        class_implicit_parameter_count: int,
        direct_implicit_parameter_count: int | None,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._implicit_parameter_count = implicit_parameter_count
        member._class_owned = class_owned
        member._binding_decorator = binding_decorator
        member.receiver_access = receiver_access
        member.class_implicit_parameter_count = class_implicit_parameter_count
        member.direct_implicit_parameter_count = direct_implicit_parameter_count
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
        self, bindings: tuple[CompactBindingSource, ...]
    ) -> CompactBindingSource:
        """Completed namespaces select their final binding; closures retain alternatives."""
        if len(bindings) == 1 or not self.is_function_scope:
            return bindings[-1]
        return OpenCompactBindingMutation(
            CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION
        )


class CompactControlBranchKind(StrEnum):
    """Child-suite repetition and try completion order for flow reasoning."""

    IF_BODY = "if_body", False, None
    IF_ELSE = "if_else", False, None
    LOOP_BODY = "loop_body", True, None
    LOOP_ELSE = "loop_else", False, None
    TRY_BODY = "try_body", False, 0
    TRY_HANDLER = "try_handler", False, 1
    TRY_ELSE = "try_else", False, 2
    TRY_FINALLY = "try_finally", False, 3
    WITH_BODY = "with_body", False, None
    MATCH_CASE = "match_case", False, None

    def __new__(
        cls, value: str, can_repeat: bool, completion_phase: int | None
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.can_repeat = can_repeat
        member._completion_phase = completion_phase
        return member

    def may_precede(self, other: CompactControlBranchKind) -> bool:
        """Known try stages order execution; other sibling suites remain open."""
        return (
            self._completion_phase is None
            or other._completion_phase is None
            or self._completion_phase <= other._completion_phase
        )


class CompactBindingOperationABC(ABC):
    """Source-kind behaviour owned by the mutation declaration."""

    is_import_binding = False
    is_definition_binding = False

    def pending_after(
        self,
        context: CompactFlowContext,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit],
    ) -> frozenset[CompactBindingVisit]:
        return pending | {(context.owner_symbol, binding)}

    def validate_import_origin(self, origin: ImportedNameOrigin | None) -> None:
        if origin is not None:
            raise ValueError("Only import mutations carry an imported origin")

    @abstractmethod
    def resolve_source(
        self,
        resolver: CompactBindingResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        raise NotImplementedError

    def resolve_definition(
        self,
        resolver: CompactDefinitionResolverABC[TargetResolutionT],
        symbol: str,
        binding: CompactMutation,
    ) -> TargetResolutionT:
        raise ValueError("Only definition mutations resolve a declaration")


class CompactValueBindingOperation(CompactBindingOperationABC):
    def resolve_source(
        self,
        resolver: CompactBindingResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        return resolver._possible_binding_resolution(
            context,
            reference,
            CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
            pending,
        )


class CompactImportBindingOperation(CompactBindingOperationABC):
    is_import_binding = True

    def validate_import_origin(self, origin: ImportedNameOrigin | None) -> None:
        if origin is None:
            raise ValueError("Import mutations require their source declaration")

    def resolve_source(
        self,
        resolver: CompactBindingResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        return resolver._imported_name_resolution(context, reference, binding, pending)


class CompactDefinitionBindingOperation(CompactBindingOperationABC):
    is_definition_binding = True

    def __init__(
        self,
        selector: Callable[
            [CompactDefinitionResolverABC[TargetResolutionT], str, CompactMutation],
            TargetResolutionT,
        ],
    ) -> None:
        self._selector = selector

    def pending_after(
        self,
        context: CompactFlowContext,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit],
    ) -> frozenset[CompactBindingVisit]:
        return pending

    def resolve_source(
        self,
        resolver: CompactBindingResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        return resolver._definition_binding_resolution(
            context,
            reference,
            cast(CompactMutation[CompactDefinitionTarget], binding),
            pending,
        )

    def resolve_definition(
        self,
        resolver: CompactDefinitionResolverABC[TargetResolutionT],
        symbol: str,
        binding: CompactMutation,
    ) -> TargetResolutionT:
        return self._selector(resolver, symbol, binding)


class CompactMutationKind(StrEnum):
    """Source operations with one declaration-owned binding interpretation."""

    ASSIGNMENT = "assignment"
    AUGMENTED_ASSIGNMENT = "augmented_assignment"
    DELETION = "deletion"
    FUNCTION_DEFINITION = "function_definition", CompactDefinitionBindingOperation(
        lambda resolver, symbol, binding: resolver._selected_function_resolution(
            symbol, binding
        )
    )
    CLASS_DEFINITION = "class_definition", CompactDefinitionBindingOperation(
        lambda resolver, symbol, binding: resolver._selected_class_resolution(
            symbol, binding
        )
    )
    IMPORT = "import", CompactImportBindingOperation()
    ITERATION_BINDING = "iteration_binding"
    CONTEXT_BINDING = "context_binding"
    EXCEPTION_BINDING = "exception_binding"
    PATTERN_BINDING = "pattern_binding"

    def __new__(
        cls,
        value: str,
        binding_operation: CompactBindingOperationABC = CompactValueBindingOperation(),
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.binding_operation = binding_operation
        return member

    is_import_binding = AliasProperty[bool]("binding_operation.is_import_binding")
    is_definition_binding = AliasProperty[bool](
        "binding_operation.is_definition_binding"
    )

    @property
    def introduces_nominal_binding(self) -> bool:
        """A syntactic binding category, not a runtime identity guarantee."""
        return self.is_import_binding or self.is_definition_binding

    def resolve_definition(
        self,
        resolver: CompactDefinitionResolverABC[TargetResolutionT],
        symbol: str,
        binding: CompactMutation,
    ) -> TargetResolutionT:
        return self.binding_operation.resolve_definition(resolver, symbol, binding)

    def validate_import_origin(self, origin: ImportedNameOrigin | None) -> None:
        self.binding_operation.validate_import_origin(origin)


class CompactValueDestinationKind(StrEnum):
    """Immediate value destination with member-owned binding requirements."""

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
            raise ValueError(f"{self.value} value binding does not match its use")


@dataclass(frozen=True)
class CompactFunctionIdentity:
    """Import-stable identity of one declared Python function."""

    module_name: str
    qualname: str

    @property
    def symbol(self) -> str:
        return f"{self.module_name}.{self.qualname}"


@dataclass(frozen=True)
class CompactValueDestination:
    """Validated value destination and its declaration-owned binding payload."""

    use: CompactValueDestinationKind
    binding: LexicalValueReference | None = None

    @classmethod
    def for_assignment(cls, targets: tuple[ast.expr, ...] | list[ast.expr]) -> Self:
        binding = (
            LexicalValueReference.from_expression(targets[0])
            if len(targets) == 1
            else None
        )
        return (
            cls(CompactValueDestinationKind.EMBEDDED)
            if binding is None
            else cls(CompactValueDestinationKind.BOUND, binding)
        )

    def __post_init__(self) -> None:
        self.use.validate_binding(self.binding)


@dataclass(frozen=True, eq=False)
class CompactCallArguments(Generic[CallValueT], DataclassGraphValue):
    """One argument list, shared by collected calls and authored call edits."""

    positional: tuple[CompactCallArgument[CallValueT], ...]
    keywords: tuple[CompactKeywordArgument[CallValueT], ...]

    @classmethod
    def from_call(
        cls, node: ast.Call, project_value: Callable[[ast.expr], CallValueT]
    ) -> Self:
        return cls(
            positional=tuple(
                CompactCallArgument(
                    project_value(
                        argument.value
                        if isinstance(argument, ast.Starred)
                        else argument
                    ),
                    is_unpacked=isinstance(argument, ast.Starred),
                )
                for argument in node.args
            ),
            keywords=tuple(
                CompactKeywordArgument(keyword.arg, project_value(keyword.value))
                for keyword in node.keywords
            ),
        )

    @property
    def values(self) -> tuple[CallValueT, ...]:
        return tuple(argument.value for argument in (*self.positional, *self.keywords))

    def bind_to(
        self, declaration: "CompactFunctionDeclaration"
    ) -> "CompactCallBinding[CallValueT]":
        return declaration.bind_call(self.positional, self.keywords)


ResolutionContextT = TypeVar("ResolutionContextT")
TargetResolutionT = TypeVar("TargetResolutionT")


class CompactDefinitionResolverABC(ABC, Generic[TargetResolutionT]):
    """Select source definitions independently of callable-use projection."""

    @abstractmethod
    def _selected_class_resolution(
        self, symbol: str, binding: CompactMutation
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _selected_function_resolution(
        self, symbol: str, binding: CompactMutation
    ) -> TargetResolutionT:
        raise NotImplementedError


class CompactCallTargetResolverABC(
    CompactDefinitionResolverABC[TargetResolutionT],
    Generic[ResolutionContextT, TargetResolutionT],
):
    """Repository obligations selected by nominal call-target syntax."""

    @abstractmethod
    def _through_attribute_suffix(
        self,
        resolution: TargetResolutionT,
        attribute_path: tuple[str, ...],
    ) -> TargetResolutionT:
        """Project attributes of a captured non-lexical target conservatively."""
        raise NotImplementedError

    @abstractmethod
    def _through_receiver_binding(
        self,
        context: ResolutionContextT,
        position: CompactFlowPosition,
        resolution: TargetResolutionT,
    ) -> TargetResolutionT:
        """Require the current-class receiver to retain its entry origin."""
        raise NotImplementedError

    @abstractmethod
    def _local_function_target_resolution(
        self,
        context: ResolutionContextT,
        target: CompactCallTargetReference,
    ) -> TargetResolutionT:
        """Resolve candidates supplied by a target's local lookup contract."""
        raise NotImplementedError

    @abstractmethod
    def _lexical_function_target_resolution(
        self,
        context: ResolutionContextT,
        reference: LexicalValueReference,
        position: CompactFlowPosition,
        pending_bindings: frozenset[CompactBindingVisit] = frozenset(),
    ) -> TargetResolutionT:
        """Resolve a lexical access path through its reaching bindings."""
        raise NotImplementedError

    @abstractmethod
    def _class_member_method_resolution(
        self,
        context: ResolutionContextT,
        target: CurrentClassMemberMethodReference,
        position: CompactFlowPosition,
    ) -> TargetResolutionT:
        """Resolve a method through a declared current-class member."""
        raise NotImplementedError


class CompactCallTargetReference(ABC):
    """Nominal call-target syntax with leaf-owned resolution behavior."""

    def resolve_target(
        self,
        resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        position: CompactFlowPosition,
    ) -> TargetResolutionT:
        return resolver._local_function_target_resolution(context, self)

    def resolve(
        self,
        resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        position: CompactFlowPosition,
        *,
        pending_bindings: frozenset[CompactBindingVisit] = frozenset(),
        attribute_path: tuple[str, ...] = (),
    ) -> TargetResolutionT:
        """Select the nominal lookup, then project any captured attribute access."""
        return resolver._through_attribute_suffix(
            self.resolve_target(resolver, context, position),
            attribute_path,
        )

    @property
    @abstractmethod
    def terminal_name(self) -> str | None:
        raise NotImplementedError

    def receiver_access(
        self, caller: CompactFunctionDeclaration | None
    ) -> CompactDescriptorAccess:
        """Receiver evidence for syntax requiring current-class resolution."""
        return CompactDescriptorAccess.UNKNOWN

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


class LexicalCallTargetReference(CompactCallTargetReference, ABC):
    """Call syntax whose target is supplied by an exact lexical path."""

    @property
    @abstractmethod
    def lexical_reference(self) -> LexicalValueReference:
        raise NotImplementedError

    def resolve(
        self,
        resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        position: CompactFlowPosition,
        *,
        pending_bindings: frozenset[CompactBindingVisit] = frozenset(),
        attribute_path: tuple[str, ...] = (),
    ) -> TargetResolutionT:
        reference = self.lexical_reference
        return resolver._lexical_function_target_resolution(
            context,
            LexicalValueReference(
                reference.root_name, (*reference.attribute_path, *attribute_path)
            ),
            position,
            pending_bindings,
        )


class CurrentClassCallTargetReference(CompactCallTargetReference, ABC):
    """Call target whose terminal method is selected from the current class."""

    receiver_name: str
    owner_class_qualname: str
    method_name: str

    @property
    def lexical_attribute_path(self) -> tuple[str, ...] | None:
        return (self.method_name,)

    def resolve_current_class_target(
        self,
        resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        position: CompactFlowPosition,
    ) -> TargetResolutionT:
        return resolver._local_function_target_resolution(context, self)

    def resolve_target(
        self,
        resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        position: CompactFlowPosition,
    ) -> TargetResolutionT:
        return resolver._through_receiver_binding(
            context,
            position,
            self.resolve_current_class_target(resolver, context, position),
        )

    @property
    def terminal_name(self) -> str:
        return self.method_name

    @property
    def lexical_reference(self) -> LexicalValueReference | None:
        path = self.lexical_attribute_path
        return None if path is None else LexicalValueReference(self.receiver_name, path)


@dataclass(frozen=True)
class BareCallTargetReference(LexicalCallTargetReference):
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
    receiver_name: str
    owner_class_qualname: str
    method_name: str

    def receiver_access(
        self, caller: CompactFunctionDeclaration | None
    ) -> CompactDescriptorAccess:
        return (
            CompactDescriptorAccess.UNKNOWN
            if caller is None
            else caller.binding_kind.receiver_access
        )

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

    receiver_name: str
    owner_class_qualname: str
    member_name: str
    method_name: str
    uses_runtime_class_lookup: bool

    @property
    def lexical_attribute_path(self) -> tuple[str, ...] | None:
        return (
            None
            if self.uses_runtime_class_lookup
            else (self.member_name, self.method_name)
        )

    def resolve_current_class_target(
        self,
        resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        position: CompactFlowPosition,
    ) -> TargetResolutionT:
        return resolver._class_member_method_resolution(context, self, position)

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
            receiver_name=receiver_name,
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
class QualifiedCallTargetReference(LexicalCallTargetReference):
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

    def may_precede(self, other: CompactFlowPosition) -> bool:
        """Exclude proved future events, retaining loop and header ambiguity."""
        for source, destination in zip(self.branch_path, other.branch_path):
            if source != destination:
                if source.parent_statement_index != destination.parent_statement_index:
                    return (
                        source.parent_statement_index
                        < destination.parent_statement_index
                    )
                return source.kind.may_precede(destination.kind)
            if source.kind.can_repeat:
                return True
        common_depth = min(len(self.branch_path), len(other.branch_path))
        if len(self.branch_path) > common_depth:
            return (
                self.branch_path[common_depth].parent_statement_index
                <= other.statement_index
            )
        if len(other.branch_path) > common_depth:
            return (
                self.statement_index
                <= other.branch_path[common_depth].parent_statement_index
            )
        return (self.statement_index, self.event_index) <= (
            other.statement_index,
            other.event_index,
        )

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


class CompactMutationResolverABC(ABC, Generic[ResolutionContextT, TargetResolutionT]):
    """Distinct namespace-binding and captured-object mutation obligations."""

    @abstractmethod
    def _binding_mutation_resolution(
        self,
        context: ResolutionContextT,
        mutation: CompactMutation,
        name: str,
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _receiver_mutation_resolution(
        self,
        context: ResolutionContextT,
        mutation: CompactMutation,
        receiver_use: CompactValueUse,
    ) -> TargetResolutionT:
        raise NotImplementedError


class CompactAssignmentTargetABC(ABC):
    """An evaluated write destination, distinct from the later write event."""

    imported_origin: ImportedNameOrigin | None = None

    @abstractmethod
    def resolve_mutation(
        self,
        resolver: CompactMutationResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        mutation: CompactMutation,
    ) -> TargetResolutionT:
        raise NotImplementedError

    @property
    @abstractmethod
    def lexical_reference(self) -> LexicalValueReference | None:
        raise NotImplementedError

    @property
    def bound_name(self) -> str | None:
        return None

    def may_replace(self, reference: LexicalValueReference) -> bool:
        """Without a binding proof, a write may affect any object slot."""
        return bool(reference.attribute_path)

    @abstractmethod
    def affected_roots_within(
        self,
        flow: CompactFunctionFlow,
        roots: frozenset[str],
    ) -> frozenset[str]:
        raise NotImplementedError


class CompactLexicalBindingTargetABC(CompactAssignmentTargetABC):
    """Lexical write behaviour independent of how the binding name is supplied."""

    @property
    @abstractmethod
    def bound_name(self) -> str:
        raise NotImplementedError

    def may_replace(self, reference: LexicalValueReference) -> bool:
        return self.lexical_reference.is_prefix_of(reference)

    def resolve_mutation(
        self,
        resolver: CompactMutationResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        mutation: CompactMutation,
    ) -> TargetResolutionT:
        return resolver._binding_mutation_resolution(context, mutation, self.bound_name)

    @cached_property
    def lexical_reference(self) -> LexicalValueReference:
        return LexicalValueReference(self.bound_name)

    def affected_roots_within(
        self,
        flow: CompactFunctionFlow,
        roots: frozenset[str],
    ) -> frozenset[str]:
        return roots.intersection((self.bound_name,))


@dataclass(frozen=True)
class CompactImportTarget(CompactLexicalBindingTargetABC):
    origin: ImportedNameOrigin

    bound_name = AliasProperty[str]("origin.bound_name")
    imported_origin = AliasProperty[ImportedNameOrigin]("origin")


@dataclass(frozen=True)
class CompactBindingTarget(CompactLexicalBindingTargetABC):
    name: str

    bound_name = AliasProperty[str]("name")


@dataclass(frozen=True)
class CompactDefinitionTarget(CompactBindingTarget):
    """A definition binding retains decorator values captured during its creation."""

    decorator_uses: tuple[CompactValueUse, ...]


@dataclass(frozen=True)
class CompactReceiverTarget(CompactAssignmentTargetABC, ABC):
    receiver_use: CompactValueUse

    def resolve_mutation(
        self,
        resolver: CompactMutationResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        mutation: CompactMutation,
    ) -> TargetResolutionT:
        return resolver._receiver_mutation_resolution(
            context, mutation, self.receiver_use
        )

    @property
    def lexical_reference(self) -> LexicalValueReference | None:
        return None

    def affected_roots_within(
        self,
        flow: CompactFunctionFlow,
        roots: frozenset[str],
    ) -> frozenset[str]:
        return self.receiver_use.origin_in(flow).candidate_roots_within(roots)


@dataclass(frozen=True)
class CompactAttributeTarget(CompactReceiverTarget):
    attribute_name: str

    @cached_property
    def lexical_reference(self) -> LexicalValueReference | None:
        receiver = self.receiver_use.lexical_reference
        return (
            None
            if receiver is None
            else LexicalValueReference(
                receiver.root_name,
                (*receiver.attribute_path, self.attribute_name),
            )
        )


@dataclass(frozen=True)
class CompactItemTarget(CompactReceiverTarget):
    index_use: CompactValueUse


AssignmentTargetT = TypeVar("AssignmentTargetT", bound=CompactAssignmentTargetABC)


@dataclass(frozen=True)
class CompactMutation(Generic[AssignmentTargetT]):
    target: AssignmentTargetT
    kind: CompactMutationKind
    position: CompactFlowPosition
    line: int
    imported_origin = AliasProperty[ImportedNameOrigin | None]("target.imported_origin")

    reference = AliasProperty[LexicalValueReference | None]("target.lexical_reference")

    def resolve(
        self,
        resolver: CompactMutationResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return self.target.resolve_mutation(resolver, context, self)

    def __post_init__(self) -> None:
        self.kind.validate_import_origin(self.imported_origin)
        if self.kind.is_definition_binding != isinstance(
            self.target, CompactDefinitionTarget
        ):
            raise TypeError(
                "Definition mutations require a definition target exclusively"
            )


CompactBindingVisit: TypeAlias = tuple[str, CompactMutation]


class CompactFunctionTargetResolutionViolation(StrEnum):
    """Typed reasons a call target lacks one closed nominal declaration."""

    DYNAMIC_BINDING = "dynamic_binding"
    MISSING_DECLARATION = "missing_declaration"
    AMBIGUOUS_DECLARATION = "ambiguous_declaration"
    INCOMPLETE_RECEIVER_FAMILY = "incomplete_receiver_family"
    UNSUPPORTED_RECEIVER = "unsupported_receiver"
    CYCLIC_BINDING = "cyclic_binding"


class CompactBindingResolverABC(ABC, Generic[TargetResolutionT]):
    """Shared interpretation of a source selected in its actual flow."""

    def _selected_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        visit = (context.owner_symbol, binding)
        if visit in pending_bindings:
            return self._cyclic_binding_resolution(pending_bindings)
        operation = binding.kind.binding_operation
        pending = operation.pending_after(context, binding, pending_bindings)
        alias = context.flow.exact_aliases_by_binding_mutation.get(binding)
        if alias is not None:
            resolution = self._captured_alias_resolution(
                alias, context, reference, use_position, pending
            )
            return self._installed_alias_resolution(resolution, alias, context)
        return operation.resolve_source(self, context, reference, binding, pending)

    @abstractmethod
    def _cyclic_binding_resolution(
        self, pending: frozenset[CompactBindingVisit]
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _captured_alias_resolution(
        self,
        alias: CompactExactValueAlias,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _installed_alias_resolution(
        self,
        resolution: TargetResolutionT,
        alias: CompactExactValueAlias,
        context: CompactFlowContext,
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _imported_name_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _definition_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation[CompactDefinitionTarget],
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _possible_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        violation: CompactFunctionTargetResolutionViolation,
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        raise NotImplementedError


class CompactBindingSource(ABC):
    """Selected source evidence, with distinct value and callable projections."""

    @abstractmethod
    def resolve_binding(
        self,
        resolver: CompactBindingResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        """Project this source without nullable-field dispatch in the consumer."""
        raise NotImplementedError

    @property
    def mutation(self) -> CompactMutation | None:
        return None

    @abstractmethod
    def value_origin(
        self,
        flow: CompactFunctionFlow,
        reference: LexicalValueReference,
        visited_mutations: frozenset[CompactMutation],
    ) -> CompactValueOriginResolution:
        raise NotImplementedError


@dataclass(frozen=True)
class ExactCompactBindingMutation(CompactBindingSource):
    selected_mutation: CompactMutation

    mutation = AliasProperty[CompactMutation]("selected_mutation")

    def resolve_binding(
        self,
        resolver: CompactBindingResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        return resolver._selected_binding_resolution(
            context,
            reference,
            self.selected_mutation,
            use_position,
            pending_bindings,
        )

    def value_origin(
        self,
        flow: CompactFunctionFlow,
        reference: LexicalValueReference,
        visited_mutations: frozenset[CompactMutation],
    ) -> CompactValueOriginResolution:
        mutation = self.selected_mutation
        possible_origins = flow._possible_alias_origins(
            reference, flow.mutations_by_root_name[reference.root_name]
        )
        if mutation in visited_mutations:
            return OpenCompactValueOrigin(
                possible_origins, CompactValueOriginViolation.CYCLIC_ALIAS
            )
        alias = flow.exact_aliases_by_binding_mutation.get(mutation)
        if alias is None:
            return OpenCompactValueOrigin(
                possible_origins, CompactValueOriginViolation.INTERVENING_REBINDING
            )
        source_resolution = flow._value_origin_for(
            alias.source, alias.source_position, visited_mutations | {mutation}
        )
        return source_resolution.through_alias(reference.attribute_path, mutation)


class UnresolvedCompactBindingSource(CompactBindingSource, ABC):
    """Sources whose callable identity remains an explicit obligation."""

    @property
    @abstractmethod
    def target_lookup_violation(self) -> CompactFunctionTargetResolutionViolation:
        raise NotImplementedError

    def resolve_binding(
        self,
        resolver: CompactBindingResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> TargetResolutionT:
        return resolver._possible_binding_resolution(
            context,
            reference,
            self.target_lookup_violation,
            pending_bindings,
        )


@dataclass(frozen=True)
class OpenCompactBindingMutation(UnresolvedCompactBindingSource):
    failure: CompactFunctionTargetResolutionViolation

    target_lookup_violation = AliasProperty[CompactFunctionTargetResolutionViolation](
        "failure"
    )

    def value_origin(
        self,
        flow: CompactFunctionFlow,
        reference: LexicalValueReference,
        visited_mutations: frozenset[CompactMutation],
    ) -> CompactValueOriginResolution:
        return OpenCompactValueOrigin(
            flow._possible_alias_origins(
                reference, flow.mutations_by_root_name[reference.root_name]
            ),
            CompactValueOriginViolation.AMBIGUOUS_BINDING,
        )


@dataclass(frozen=True)
class InitialCompactParameterBinding(UnresolvedCompactBindingSource):
    """The entry value of the exact parameter declared by the flow owner."""

    parameter: CompactFunctionParameter

    target_lookup_violation = CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING

    def value_origin(
        self,
        flow: CompactFunctionFlow,
        reference: LexicalValueReference,
        visited_mutations: frozenset[CompactMutation],
    ) -> CompactValueOriginResolution:
        return ExactCompactValueOrigin(reference)


class CompactValueOriginViolation(StrEnum):
    """Reason one lexical value lacks a single unchanged local origin."""

    INTERVENING_REBINDING = "intervening_rebinding"
    AMBIGUOUS_BINDING = "ambiguous_binding"
    CYCLIC_ALIAS = "cyclic_alias"

    OPAQUE_EXPRESSION = "opaque_expression"


class CompactValueOriginResolution(ABC):
    """Nominal result of tracing one value through exact local aliases."""

    def candidate_roots_within(self, roots: frozenset[str]) -> frozenset[str]:
        """Project potentially retained origins into a supplied lexical family."""
        return roots.intersection(origin.root_name for origin in self.possible_origins)

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
        binding_mutation: CompactMutation,
    ) -> "CompactValueOriginResolution":
        raise NotImplementedError


@dataclass(frozen=True)
class ExactCompactValueOrigin(CompactValueOriginResolution):
    origin: LexicalValueReference
    alias_chain: tuple[CompactMutation, ...] = ()

    @property
    def exact_origin(self) -> LexicalValueReference:
        return self.origin

    @property
    def possible_origins(self) -> tuple[LexicalValueReference, ...]:
        return (self.origin,)

    def through_alias(
        self,
        suffix: tuple[str, ...],
        binding_mutation: CompactMutation,
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

    def candidate_roots_within(self, roots: frozenset[str]) -> frozenset[str]:
        """Diagnostic origins do not bound an unresolved object's identity."""
        return roots

    @property
    def exact_origin(self) -> None:
        return None

    @property
    def possible_origins(self) -> tuple[LexicalValueReference, ...]:
        return self.candidates

    def through_alias(
        self,
        suffix: tuple[str, ...],
        binding_mutation: CompactMutation,
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
    """An exact binding retaining the already-collected source read."""

    source_use: CompactCallableReferenceUse
    binding_mutation: CompactMutation

    source = AliasProperty[LexicalValueReference]("source_use.target.lexical_reference")
    source_position = AliasProperty[CompactFlowPosition]("source_use.position")

    target = AliasProperty[LexicalValueReference]("binding_mutation.reference")

    def __post_init__(self) -> None:
        if self.source is None:
            raise ValueError("Exact value aliases require a lexical source read")
        if self.binding_mutation.target.bound_name is None:
            raise ValueError("Exact value aliases require a lexical name binding")

    def source_for(self, reference: LexicalValueReference) -> LexicalValueReference:
        """Project lexical origin syntax without replacing nominal lookup evidence."""
        return LexicalValueReference(
            self.source.root_name,
            (*self.source.attribute_path, *reference.attribute_path),
        )


@dataclass(frozen=True, eq=False)
class CompactValueUse(DataclassGraphValue):
    """One evaluated expression value, retaining its source event."""

    value: CompactValueExpression
    position: CompactFlowPosition

    lexical_reference = AliasProperty[LexicalValueReference | None](
        "value.lexical_reference"
    )

    def origin_in(self, flow: CompactFunctionFlow) -> CompactValueOriginResolution:
        reference = self.lexical_reference
        if reference is None:
            return OpenCompactValueOrigin(
                (), CompactValueOriginViolation.OPAQUE_EXPRESSION
            )
        return flow.value_origin_for(reference, self.position)

    def reference_equivalents_in(
        self, flow: CompactFunctionFlow
    ) -> tuple[LexicalValueReference, ...]:
        return tuple(
            dict.fromkeys(
                reference
                for reference in (
                    self.lexical_reference,
                    self.origin_in(flow).exact_origin,
                )
                if reference is not None
            )
        )


@dataclass(frozen=True, eq=False)
class CompactEvaluatedResult(DataclassGraphValue):
    """An evaluated value and its immediate disposition, not a completed path proof.

    A missing value use denotes the implicit None of a bare return.
    The position records disposition after expression evaluation; a
    finally suite or enclosing activation can still affect completion.
    """

    destination: CompactValueDestination
    value_use: CompactValueUse | None
    position: CompactFlowPosition
    source_span: SourceByteSpan

    line = AliasProperty[int]("source_span.start_line")

    @property
    def lexical_reference(self) -> LexicalValueReference | None:
        return None if self.value_use is None else self.value_use.lexical_reference


@dataclass(frozen=True)
class CompactCallableReferenceUse:
    target: CompactCallTargetReference
    position: CompactFlowPosition
    source_span: SourceByteSpan

    line = AliasProperty[int]("source_span.start_line")

    def resolve(
        self,
        resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        *,
        pending_bindings: frozenset[CompactBindingVisit] = frozenset(),
        attribute_path: tuple[str, ...] = (),
    ) -> TargetResolutionT:
        """Resolve the captured target, retaining lexical cycle and suffix evidence."""
        return self.target.resolve(
            resolver,
            context,
            self.position,
            pending_bindings=pending_bindings,
            attribute_path=attribute_path,
        )


@dataclass(frozen=True, eq=False)
class CompactFunctionCall(DataclassGraphValue):
    target_use: CompactCallableReferenceUse
    arguments: CompactCallArguments[CompactValueUse]
    result: CompactValueDestination
    position: CompactFlowPosition
    source_span: SourceByteSpan

    target = AliasProperty[CompactCallTargetReference]("target_use.target")

    line = AliasProperty[int]("source_span.start_line")

    @property
    def result_use(self) -> CompactValueDestinationKind:
        return self.result.use

    @property
    def result_binding(self) -> LexicalValueReference | None:
        return self.result.binding

    def bind_to(
        self, declaration: "CompactFunctionDeclaration"
    ) -> CompactCallBinding[CompactValueUse]:
        return self.arguments.bind_to(declaration)

    def product_construction(self) -> "CompactProductConstruction | None":
        if (
            self.result.use is not CompactValueDestinationKind.BOUND
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


@dataclass(frozen=True, eq=False)
class CallResultValue(DataclassGraphValue, OpaqueValueExpression):
    """A retained source call's value, without claiming completed execution."""

    invocation: CompactFunctionCall


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
    field_arguments: tuple[CompactKeywordArgument[CompactValueUse], ...]
    position: CompactFlowPosition
    line: int

    @cached_property
    def field_values(self) -> dict[str, CompactValueUse]:
        return {
            argument.name: argument.value
            for argument in self.field_arguments
            if argument.name is not None
        }

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.field_values)


class CompactFlowOwner(ABC):
    """Nominal scope owner, retaining its declaration when it is a function."""

    kind: CompactFlowOwnerKind
    qualname: str

    def initial_binding_for(self, root_name: str) -> CompactBindingSource | None:
        return None

    @property
    @abstractmethod
    def declaration(self) -> CompactFunctionDeclaration | None:
        raise NotImplementedError


@dataclass(frozen=True)
class CompactNamespaceFlowOwner(CompactFlowOwner):
    """Module or class-body scope without a function signature."""

    kind: CompactFlowOwnerKind
    qualname: str

    def __post_init__(self) -> None:
        if self.kind.is_function_scope:
            raise ValueError("Function flows must be owned by their declaration")

    @property
    def declaration(self) -> None:
        return None


@dataclass(frozen=True)
class CompactFunctionDeclaration(CompactFlowOwner):
    identity: CompactFunctionIdentity
    execution: NativeFunctionExecution
    owner_class_qualname: str | None
    signature: CompactFunctionSignature
    decorators: tuple[CompactValueExpression, ...] = ()
    return_annotation_expression: str | None = None

    kind = CompactFlowOwnerKind.FUNCTION

    qualname = AliasProperty[str]("identity.qualname")
    line = AliasProperty[int]("execution.source_span.start_line")
    end_line = AliasProperty[int]("execution.source_span.end_line")

    def initial_binding_for(self, root_name: str) -> CompactBindingSource | None:
        return next(
            (
                InitialCompactParameterBinding(parameter)
                for parameter in self.signature.parameters
                if parameter.name == root_name
            ),
            None,
        )

    @property
    def declaration(self) -> CompactFunctionDeclaration:
        return self

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

    def signature_for_access(
        self, access: CompactDescriptorAccess
    ) -> CompactFunctionSignature | None:
        count = access.implicit_parameter_count(self.binding_kind)
        if count is None:
            return None
        return self.call_signature if count else self.signature

    def bind_call(
        self,
        positional_arguments: tuple[CompactCallArgument[CallValueT], ...],
        keyword_arguments: tuple[CompactKeywordArgument[CallValueT], ...],
        *,
        access: CompactDescriptorAccess = CompactDescriptorAccess.INSTANCE,
    ) -> CompactCallBinding[CallValueT]:
        if self.signature_decorator_hazard:
            return ViolatedCompactCallBinding(
                CompactCallBindingViolation.SIGNATURE_DECORATOR_HAZARD
            )
        count = access.implicit_parameter_count(self.binding_kind)
        if count is None:
            return ViolatedCompactCallBinding(
                CompactCallBindingViolation.INVALID_DESCRIPTOR_ACCESS
            )
        if count > len(self.signature.parameters):
            return ViolatedCompactCallBinding(
                CompactCallBindingViolation.INVALID_IMPLICIT_PARAMETER
            )
        signature = self.signature_for_access(access)
        assert signature is not None
        return signature.bind(positional_arguments, keyword_arguments)


@dataclass(frozen=True, eq=False)
class CompactFunctionFlow(DataclassGraphValue):
    owner: CompactFlowOwner
    lexical_scope_qualnames: tuple[str, ...]
    calls: tuple[CompactFunctionCall, ...]
    evaluated_results: tuple[CompactEvaluatedResult, ...]
    callable_reference_uses: tuple[CompactCallableReferenceUse, ...]
    mutations: tuple[CompactMutation, ...]
    exact_value_aliases: tuple[CompactExactValueAlias, ...]
    global_binding_names: tuple[str, ...]
    nonlocal_binding_names: tuple[str, ...]

    def mutated_roots_within(self, roots: frozenset[str]) -> frozenset[str]:
        """Derive possible affected roots from captured mutation targets."""
        return frozenset(
            chain.from_iterable(
                mutation.target.affected_roots_within(self, roots)
                for mutation in self.mutations
            )
        )

    @cached_property
    def loaded_value_root_names(self) -> tuple[str, ...]:
        """Derive observed names from retained calls and value reads."""
        return tuple(
            sorted(
                {
                    reference.root_name
                    for use in chain(
                        self.callable_reference_uses,
                        (call.target_use for call in self.calls),
                    )
                    if (reference := use.target.lexical_reference) is not None
                }
            )
        )

    def _binding_resolution_for_mutations(
        self,
        mutations: tuple[CompactMutation, ...],
        use_position: CompactFlowPosition | None,
        root_name: str,
    ) -> CompactBindingSource | None:
        """Select a positioned write before materialising declaration entry evidence."""
        if use_position is not None:
            selected = next(
                (
                    mutation
                    for mutation in reversed(mutations)
                    if mutation.position.may_precede(use_position)
                ),
                None,
            )
            if (
                selected is not None
                and selected.position.dominates(use_position)
                and all(
                    other is selected
                    or not selected.position.may_precede(other.position)
                    or not other.position.may_precede(use_position)
                    for other in mutations
                )
            ):
                return ExactCompactBindingMutation(selected)
        if any(
            mutation.position.branch_path
            and (use_position is None or mutation.position.may_precede(use_position))
            for mutation in mutations
        ):
            return OpenCompactBindingMutation(
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
            )
        initial_binding = self.owner.initial_binding_for(root_name)
        if not mutations:
            return initial_binding
        if use_position is None:
            return self.owner.kind.deferred_binding_resolution(
                tuple(
                    binding
                    for binding in chain(
                        (initial_binding,),
                        (
                            ExactCompactBindingMutation(mutation)
                            for mutation in mutations
                        ),
                    )
                    if binding is not None
                )
            )
        return (
            initial_binding
            if initial_binding is not None
            else OpenCompactBindingMutation(
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
            )
        )

    @property
    def local_signature_is_observed(self) -> bool:
        return CompactLocalSignatureObserver.observes_any(self.calls)

    @cached_property
    def mutations_by_root_name(self) -> dict[str, tuple[CompactMutation, ...]]:
        grouped: dict[str, list[CompactMutation]] = {}
        for mutation in self.mutations:
            name = mutation.target.bound_name
            if name is not None:
                grouped.setdefault(name, []).append(mutation)
        return {name: tuple(mutations) for name, mutations in grouped.items()}

    def binding_resolution_for(
        self, root_name: str, use_position: CompactFlowPosition | None = None
    ) -> CompactBindingSource | None:
        """Select a declared entry binding or positioned write; absence permits outer lookup."""
        return self._binding_resolution_for_mutations(
            self.mutations_by_root_name.get(root_name, ()), use_position, root_name
        )

    @cached_property
    def exact_aliases_by_binding_mutation(
        self,
    ) -> dict[CompactMutation, CompactExactValueAlias]:
        return {alias.binding_mutation: alias for alias in self.exact_value_aliases}

    def bound_call_result_for(
        self,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition,
    ) -> CompactFunctionCall | None:
        """Return the unique call whose unchanged result reaches one use."""
        selection = self._binding_resolution_for_mutations(
            tuple(
                mutation
                for mutation in self.mutations
                if mutation.target.may_replace(reference)
            ),
            use_position,
            reference.root_name,
        )
        binding = None if selection is None else selection.mutation
        if binding is None or binding.kind is not CompactMutationKind.ASSIGNMENT:
            return None
        matching_calls = tuple(
            call
            for call in self.calls
            if call.result.binding == reference
            and binding.reference == reference
            and binding.position.branch_path == call.position.branch_path
            and binding.position.statement_index == call.position.statement_index
            and call.position.dominates(binding.position)
        )
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
        visited_mutations: frozenset[CompactMutation],
    ) -> CompactValueOriginResolution:
        selection = self.binding_resolution_for(reference.root_name, use_position)
        if selection is None:
            return ExactCompactValueOrigin(reference)
        return selection.value_origin(self, reference, visited_mutations)

    def _possible_alias_origins(
        self,
        reference: LexicalValueReference,
        mutations: tuple[CompactMutation, ...],
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
class CompactFlowContext:
    """One execution flow joined to its module and optional declaration."""

    module_name: str
    file_path: str
    flow: CompactFunctionFlow

    declaration = AliasProperty[CompactFunctionDeclaration | None](
        "flow.owner.declaration"
    )

    @property
    def owner_symbol(self) -> str:
        if self.flow.owner.kind.is_module_scope:
            return self.module_name
        return f"{self.module_name}.{self.flow.owner.qualname}"


@dataclass(frozen=True)
class CompactProductFlowModuleProjection(CompactModuleIdentity):
    """AST-free function declarations and source-ordered product-flow facts."""

    flows: tuple[CompactFunctionFlow, ...]

    @cached_property
    def function_declarations(self) -> tuple[CompactFunctionDeclaration, ...]:
        return tuple(
            declaration
            for flow in self.flows
            if (declaration := flow.owner.declaration) is not None
        )


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

    def __init__(self, module: ParsedModule) -> None:
        self.module = module
        self.scope_names: list[str] = []
        self.scope_kinds: list[CompactFlowOwnerKind] = []
        self.function_qualnames: list[str] = []
        self.class_qualnames: list[str] = []
        self.function_contexts: list[_FunctionContext] = []
        self.class_contexts: list[_ClassContext] = []

    def generic_visit(self, node: ast.AST) -> None:
        """Traverse declaration-bearing statement suites, not expressions."""

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self.visit(child)
            elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                self.generic_visit(child)

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
            identity=CompactFunctionIdentity(self.module.module_name, qualname),
            execution=self.module.native_compilation.execution_for(
                SourceByteSpan.require_node(node)
            ),
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


@dataclass
class _CompactMutationTargetCollector(ast.NodeVisitor):
    """Capture legal Python target leaves through native AST dispatch."""

    flow: _CompactFlowCollector

    def generic_visit(self, node: ast.AST) -> CompactAssignmentTargetABC:
        raise ValueError("Expected a binding, attribute or item assignment target")

    def visit_Name(self, node: ast.Name) -> CompactBindingTarget:
        return CompactBindingTarget(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> CompactAttributeTarget:
        return CompactAttributeTarget(
            self.flow._capture_value(node.value),
            node.attr,
        )

    def visit_Subscript(self, node: ast.Subscript) -> CompactItemTarget:
        return CompactItemTarget(
            self.flow._capture_value(node.value),
            self.flow._capture_value(node.slice),
        )


class _CompactFlowCollector(FunctionDefaultVisitor):
    """Collect one source scope without descending into nested scope bodies."""

    def _capture_result(
        self,
        expression: ast.expr | None,
        destination: CompactValueDestination,
        statement: SourcePositionedNode,
    ) -> CompactEvaluatedResult:
        if isinstance(expression, ast.Call):
            self.call_results[id(expression)] = destination
        value_use = None if expression is None else self._capture_value(expression)
        result = CompactEvaluatedResult(
            destination,
            value_use,
            self._position(),
            SourceByteSpan.require_node(statement),
        )
        self.evaluated_results.append(result)
        return result

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._record_target_mutation(node)
        else:
            self.generic_visit(node)

    def _record_target_mutation(
        self,
        node: ast.expr,
        kind: CompactMutationKind | None = None,
    ) -> None:
        self._record_mutation(self.mutation_targets.visit(node), node, kind)

    def _capture_value(self, expression: ast.expr) -> CompactValueUse:
        invocation: CompactFunctionCall | None = self.visit(expression)
        value = (
            CompactValueExpression.project(expression)
            if invocation is None
            else CallResultValue(invocation)
        )
        return CompactValueUse(value, self._position())

    def __init__(
        self,
        *,
        owner: CompactFlowOwner,
        module_identity: PythonModulePathIdentity,
        lexical_scope_qualnames: tuple[str, ...],
        current_class_qualname: str | None,
        current_class_receiver_name: str | None,
    ) -> None:
        self.owner = owner
        self.module_identity = module_identity
        self.lexical_scope_qualnames = lexical_scope_qualnames
        self.current_class_qualname = current_class_qualname
        self.current_class_receiver_name = current_class_receiver_name
        self.calls: list[CompactFunctionCall] = []
        self.evaluated_results: list[CompactEvaluatedResult] = []
        self.callable_reference_uses: list[CompactCallableReferenceUse] = []
        self.mutations: list[CompactMutation] = []
        self.exact_value_aliases: list[CompactExactValueAlias] = []
        self.global_binding_names: set[str] = set()
        self.nonlocal_binding_names: set[str] = set()
        self.branch_path: tuple[CompactControlBranch, ...] = ()
        self.statement_index = 0
        self.event_index = 0
        self.call_results: dict[int, CompactValueDestination] = {}
        self.mutation_kind = CompactMutationKind.ASSIGNMENT
        self.mutation_targets = _CompactMutationTargetCollector(self)

    def collect(self, statements: list[ast.stmt]) -> CompactFunctionFlow:
        self._collect_statements(statements)
        return CompactFunctionFlow(
            owner=self.owner,
            lexical_scope_qualnames=self.lexical_scope_qualnames,
            calls=tuple(self.calls),
            evaluated_results=tuple(self.evaluated_results),
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
        target: CompactAssignmentTargetABC,
        node: SourcePositionedNode,
        kind: CompactMutationKind | None = None,
    ) -> CompactMutation:
        mutation = CompactMutation(
            target=target,
            kind=self.mutation_kind if kind is None else kind,
            position=self._position(),
            line=node.lineno,
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
                self.current_class_receiver_name,
                self.current_class_qualname,
                reference.terminal_name,
            )
        return QualifiedCallTargetReference(reference)

    def _callable_reference_use(self, node: ast.expr) -> CompactCallableReferenceUse:
        return CompactCallableReferenceUse(
            target=self._call_target(node),
            position=self._position(),
            source_span=SourceByteSpan.require_node(node),
        )

    def visit_Call(self, node: ast.Call) -> CompactFunctionCall:
        self._visit_reference_evaluation(node.func)
        target_use = self._callable_reference_use(node.func)
        arguments = CompactCallArguments[CompactValueUse].from_call(
            node, self._capture_value
        )
        result = self.call_results.get(
            id(node), CompactValueDestination(CompactValueDestinationKind.EMBEDDED)
        )
        invocation = CompactFunctionCall(
            target_use=target_use,
            arguments=arguments,
            result=result,
            position=self._position(),
            source_span=SourceByteSpan.require_node(node),
        )
        self.calls.append(invocation)
        return invocation

    def _visit_reference_evaluation(self, expression: ast.expr) -> None:
        """Evaluate a reference's receiver and indices before its terminal access."""
        if isinstance(expression, ast.Attribute):
            self.visit(expression.value)
        elif not isinstance(expression, ast.Name):
            self.visit(expression)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._record_target_mutation(node)
            return
        self._visit_reference_evaluation(node)
        self.callable_reference_uses.append(self._callable_reference_use(node))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._record_target_mutation(node)
        elif isinstance(node.ctx, ast.Load):
            self.callable_reference_uses.append(self._callable_reference_use(node))

    def visit_Assign(self, node: ast.Assign) -> None:
        result = self._capture_result(
            node.value, CompactValueDestination.for_assignment(node.targets), node
        )
        mutations = self._visit_mutation_targets(
            node.targets, CompactMutationKind.ASSIGNMENT
        )
        self._record_exact_value_aliases(
            node.targets, result.lexical_reference, mutations
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        source = None
        if node.value is None:
            self.mutation_targets.visit(node.target)
            if not (
                isinstance(node.target, ast.Name) and self.owner.kind.is_function_scope
            ):
                return
        else:
            result = self._capture_result(
                node.value, CompactValueDestination.for_assignment((node.target,)), node
            )
            source = result.lexical_reference
        mutations = self._visit_mutation_targets(
            (node.target,), CompactMutationKind.ASSIGNMENT
        )
        self._record_exact_value_aliases((node.target,), source, mutations)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        target = self.mutation_targets.visit(node.target)
        self.callable_reference_uses.append(self._callable_reference_use(node.target))
        self.visit(node.value)
        self._record_mutation(target, node, CompactMutationKind.AUGMENTED_ASSIGNMENT)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._capture_result(
            node.value, CompactValueDestination.for_assignment((node.target,)), node
        )
        self._visit_mutation_targets((node.target,), CompactMutationKind.ASSIGNMENT)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._visit_mutation_targets(node.targets, CompactMutationKind.DELETION)

    def visit_Return(self, node: ast.Return) -> None:
        self._capture_result(
            node.value,
            CompactValueDestination(CompactValueDestinationKind.RETURNED),
            node,
        )

    def visit_Expr(self, node: ast.Expr) -> None:
        self._capture_result(
            node.value,
            CompactValueDestination(CompactValueDestinationKind.DISCARDED),
            node,
        )

    def _visit_mutation_targets(
        self,
        targets: tuple[ast.expr, ...] | list[ast.expr],
        kind: CompactMutationKind,
    ) -> tuple[CompactMutation, ...]:
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
        mutations: tuple[CompactMutation, ...],
    ) -> None:
        if not self._is_exact_value_alias_assignment(targets, source):
            return
        source_use = self.callable_reference_uses[-1]
        assert source_use.target.lexical_reference == source
        self.exact_value_aliases.extend(
            CompactExactValueAlias(
                source_use=source_use,
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
                CompactImportTarget(origin), node, CompactMutationKind.IMPORT
            )

    visit_ImportFrom = visit_Import

    def _capture_decorators(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> tuple[CompactValueUse, ...]:
        return tuple(
            self._capture_value(decorator) for decorator in node.decorator_list
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        decorator_uses = self._visit_definition_expressions(node)
        self._record_mutation(
            CompactDefinitionTarget(node.name, decorator_uses),
            node,
            CompactMutationKind.FUNCTION_DEFINITION,
        )

    visit_AsyncFunctionDef = visit_FunctionDef

    def _visit_definition_expressions(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[CompactValueUse, ...]:
        decorator_uses = self._capture_decorators(node)
        self.visit_argument_defaults(node.args)
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
        return decorator_uses

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        decorator_uses = self._capture_decorators(node)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._record_mutation(
            CompactDefinitionTarget(node.name, decorator_uses),
            node,
            CompactMutationKind.CLASS_DEFINITION,
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
                    CompactBindingTarget(handler.name),
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
                    CompactBindingTarget(name),
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

    declarations = _DeclarationCollector(parsed_module)
    declarations.visit(parsed_module.module)
    flows = [
        _CompactFlowCollector(
            owner=CompactNamespaceFlowOwner(CompactFlowOwnerKind.MODULE, ""),
            module_identity=parsed_module.module_path_identity,
            lexical_scope_qualnames=("",),
            current_class_qualname=None,
            current_class_receiver_name=None,
        ).collect(parsed_module.module.body)
    ]
    flows.extend(
        _CompactFlowCollector(
            owner=CompactNamespaceFlowOwner(
                CompactFlowOwnerKind.CLASS_BODY,
                context.qualname,
            ),
            module_identity=parsed_module.module_path_identity,
            lexical_scope_qualnames=context.lexical_scope_qualnames,
            current_class_qualname=context.current_class_qualname,
            current_class_receiver_name=None,
        ).collect(context.node.body)
        for context in declarations.class_contexts
    )
    flows.extend(
        _CompactFlowCollector(
            owner=context.declaration,
            module_identity=parsed_module.module_path_identity,
            lexical_scope_qualnames=context.lexical_scope_qualnames,
            current_class_qualname=context.current_class_qualname,
            current_class_receiver_name=context.declaration.nominal_receiver_name,
        ).collect(context.node.body)
        for context in declarations.function_contexts
    )
    return CompactProductFlowModuleProjection(
        module_name=parsed_module.module_name,
        file_path=parsed_module.file_path,
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
