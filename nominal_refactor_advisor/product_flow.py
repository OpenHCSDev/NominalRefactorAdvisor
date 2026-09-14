"""Compact product-flow facts for closed-component refactor proofs.

The declarations in this module preserve enough source semantics to prove a
whole parameter-conveyor trajectory without retaining repository ASTs.  They do
not rank or emit refactors: a call edge is evidence, not an executable change.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import (
    cached_property,
    partial,
)
from itertools import chain
from typing import (
    Callable,
    ClassVar,
    Generic,
    Self,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    cast,
)

from .annotation_semantics import NOMINAL_ANNOTATION_SOURCE_AUTHORITY
from .ast_tools import (
    CollectedFamily,
    CompactModuleIdentity,
    EagerFunctionAnnotationVisitor,
    ModuleAnnotationEvaluationMode,
    ParsedModule,
    VariableAnnotationVisitorABC,
    is_docstring_statement,
    module_syntax_index,
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
from .collection_algebra import UniqueIdentityIndexAuthority
from .descriptor_algebra import AliasProperty
from .lexical_bindings import (
    CompactParameterKind as CompactParameterKind,
    DictionaryEvaluationVisitor,
    ImportBoundNameProjection,
    ImportedNameOrigin,
)
from .native_compilation import (
    CPythonClassConstructionField,
    ExactNativeClassCapture,
    NativeCaptureSite,
    NativeClassCapture,
    NativeClassCaptureResolverABC,
    NativeFunctionExecution,
    NativeItemStoreOperand,
    NativeItemStoreValue,
    NativeProducedValue,
    NativeReturn,
    OpenNativeClassCapture,
)
from .python_module_identity import PythonModulePathIdentity
from .source_geometry import SourceByteSpan
from .value_expression import (
    CompactValueExpression as CompactValueExpression,
    LexicalValueReference as LexicalValueReference,
    NonLexicalValueShape,
    OpaqueValueExpression as OpaqueValueExpression,
    ResolutionContextT as ResolutionContextT,
    TargetResolutionT as TargetResolutionT,
    ValueExpressionNode,
    ValueExpressionResolverABC as ValueExpressionResolverABC,
    ValueExpressionShapeABC,
)
from .value_graph import (
    DataclassGraphNode,
    DataclassGraphValue,
    StoredDataclassState,
)

if TYPE_CHECKING:
    from .native_reference import NativeReferenceEnvironment


SourcePositionedNode: TypeAlias = (
    ValueExpressionNode | ast.stmt | ast.ExceptHandler | ast.pattern
)


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

    def documentation_statement(self, statements: list[ast.stmt]) -> ast.Expr | None:
        """Only the original module/class body entry installs documentation storage."""
        if (
            not self.is_function_scope
            and statements
            and is_docstring_statement(statements[0])
        ):
            return cast(ast.Expr, statements[0])
        return None

    def visit_assignment_annotation(
        self,
        visitor: VariableAnnotationVisitorABC,
        node: ast.AnnAssign,
        mode: ModuleAnnotationEvaluationMode,
    ) -> None:
        """Function-local annotations are declarations, never body evaluation."""
        if not self.is_function_scope:
            mode.visit_variable_annotation(visitor, node)

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


class CompactBranchPredicateResolverABC(ABC):
    """Resolve an exact predicate read for one supplied activation."""

    @abstractmethod
    def proves_boolean(
        self,
        predicate_use: CompactCallableReferenceUse,
        expected: bool,
    ) -> bool:
        """Return whether this exact read has the expected built-in Boolean value."""
        raise NotImplementedError


class CompactControlBranchKind(StrEnum):
    """Child-suite repetition and try completion order for flow reasoning."""

    @staticmethod
    def _possibly_selected(
        resolver: CompactBranchPredicateResolverABC,
        predicate_use: CompactCallableReferenceUse | None,
    ) -> bool:
        """A branch without a supported predicate remains possibly selected."""
        return False

    @staticmethod
    def _truthy_branch_excluded(
        resolver: CompactBranchPredicateResolverABC,
        predicate_use: CompactCallableReferenceUse | None,
    ) -> bool:
        return bool(
            predicate_use is not None and resolver.proves_boolean(predicate_use, False)
        )

    @staticmethod
    def _falsey_branch_excluded(
        resolver: CompactBranchPredicateResolverABC,
        predicate_use: CompactCallableReferenceUse | None,
    ) -> bool:
        return bool(
            predicate_use is not None and resolver.proves_boolean(predicate_use, True)
        )

    IF_BODY = "if_body", False, None, _truthy_branch_excluded
    IF_ELSE = "if_else", False, None, _falsey_branch_excluded
    LOOP_BODY = "loop_body", True, None
    LOOP_ELSE = "loop_else", False, None
    TRY_BODY = "try_body", False, 0
    TRY_HANDLER = "try_handler", False, 1
    TRY_ELSE = "try_else", False, 2
    TRY_FINALLY = "try_finally", False, 3
    WITH_BODY = "with_body", False, None
    MATCH_CASE = "match_case", False, None

    def __new__(
        cls,
        value: str,
        can_repeat: bool,
        completion_phase: int | None,
        excluded_by: Callable[
            [
                CompactBranchPredicateResolverABC,
                CompactCallableReferenceUse | None,
            ],
            bool,
        ] = _possibly_selected,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.can_repeat = can_repeat
        member._completion_phase = completion_phase
        member._excluded_by = excluded_by
        return member

    def may_precede(self, other: CompactControlBranchKind) -> bool:
        """Known try stages order execution; other sibling suites remain open."""
        return (
            self._completion_phase is None
            or other._completion_phase is None
            or self._completion_phase <= other._completion_phase
        )

    def is_proved_excluded(
        self,
        resolver: CompactBranchPredicateResolverABC,
        predicate_use: CompactCallableReferenceUse | None,
    ) -> bool:
        """Delegate selection semantics to this declared branch member."""
        return self._excluded_by(resolver, predicate_use)


NamespaceMemberT = TypeVar("NamespaceMemberT")


class CompactBindingOperationABC(ABC):
    """Source-kind behaviour owned by the mutation declaration."""

    def update_namespace_members(
        self, names: set[NamespaceMemberT], name: NamespaceMemberT
    ) -> None:
        names.add(name)

    def require_previous_binding(self, present: bool) -> None:
        """Installation permits either an existing or an absent destination slot."""

    def require_plain_store(self) -> None:
        """Require direct RHS installation without an additional value operation."""
        raise ValueError("Only direct assignment has plain storage semantics")

    def bound_call_result(
        self,
        flow: CompactFunctionFlow,
        binding: CompactMutation,
        reference: LexicalValueReference,
    ) -> CompactFunctionCall | None:
        """Other operations do not establish an unchanged call result store."""
        return None

    is_import_binding = False
    is_definition_binding = False

    def pending_after(
        self,
        context: CompactFlowContext,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> frozenset[CompactBindingVisit[CompactFlowContext]]:
        return pending | {CompactBindingVisit(context, binding)}

    def validate_import_origin(self, origin: ImportedNameOrigin | None) -> None:
        if origin is not None:
            raise ValueError("Only import mutations carry an imported origin")

    @abstractmethod
    def resolve_source(
        self,
        resolver: CompactBindingValueResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
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
        resolver: CompactBindingValueResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        return binding.resolve_binding_value(resolver, context, reference, pending)


class CompactAssignmentBindingOperation(CompactValueBindingOperation):
    """Direct assignment installs its already-evaluated RHS unchanged."""

    def require_plain_store(self) -> None:
        return None

    def bound_call_result(
        self,
        flow: CompactFunctionFlow,
        binding: CompactMutation,
        reference: LexicalValueReference,
    ) -> CompactFunctionCall | None:
        matching_calls = tuple(
            call
            for call in flow.calls
            if call.result.binding == reference
            and binding.reference == reference
            and binding.position.branch_path == call.position.branch_path
            and binding.position.statement_index == call.position.statement_index
            and call.position.dominates(binding.position)
        )
        return matching_calls[0] if len(matching_calls) == 1 else None


class CompactDeletionBindingOperation(CompactBindingOperationABC):
    """Remove an existing binding; absence is not an unknown stored value."""

    def require_previous_binding(self, present: bool) -> None:
        if not present:
            raise ValueError("Native deletion requires an existing destination binding")

    def update_namespace_members(
        self, names: set[NamespaceMemberT], name: NamespaceMemberT
    ) -> None:
        self.require_previous_binding(name in names)
        names.remove(name)

    def resolve_source(
        self,
        resolver: CompactBindingValueResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        return resolver._deleted_binding_resolution(
            context, reference, binding, pending
        )


class CompactImportBindingOperation(CompactBindingOperationABC):
    is_import_binding = True

    def validate_import_origin(self, origin: ImportedNameOrigin | None) -> None:
        if origin is None:
            raise ValueError("Import mutations require their source declaration")

    def resolve_source(
        self,
        resolver: CompactBindingValueResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        return resolver._imported_name_resolution(context, reference, binding, pending)


class CompactDefinitionBindingOperation(CompactBindingOperationABC):
    is_definition_binding = True

    def pending_after(
        self,
        context: CompactFlowContext,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> frozenset[CompactBindingVisit[CompactFlowContext]]:
        return pending

    def resolve_source(
        self,
        resolver: CompactBindingValueResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
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
        definition = cast(CompactMutation[CompactDefinitionTarget], binding)
        return definition.target.owner.resolve_definition(resolver, symbol, definition)


class CompactMutationKind(StrEnum):
    """Source operations with one declaration-owned binding interpretation."""

    ASSIGNMENT = "assignment", CompactAssignmentBindingOperation()
    AUGMENTED_ASSIGNMENT = "augmented_assignment"
    DELETION = "deletion", CompactDeletionBindingOperation()
    DEFINITION = "definition", CompactDefinitionBindingOperation()
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


class CompactResultCompletionResolverABC(ABC, Generic[TargetResolutionT]):
    """Interpret an evaluated result through its declaration-owned destination."""

    @abstractmethod
    def _returned_result_completion(
        self, result: CompactEvaluatedResult
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _discarded_result_completion(
        self, result: CompactEvaluatedResult
    ) -> TargetResolutionT:
        raise NotImplementedError


class CompactValueDestinationKind(StrEnum):
    """Immediate value destinations own storage, scheduling and effect obligations."""

    def _open_expression_operations(
        self,
        result: CompactEvaluatedResult,
        operations: tuple[SourceFlowOperation, ...],
    ) -> tuple[SourceFlowOperation, ...]:
        raise ValueError("Value destination has no expression-statement disposition")

    def _discard_operations(
        self,
        result: CompactEvaluatedResult,
        operations: tuple[SourceFlowOperation, ...],
    ) -> tuple[SourceFlowOperation, ...]:
        return tuple(operation for operation in operations if operation.event is result)

    def _binding_operations(
        self,
        result: CompactEvaluatedResult,
        operations: tuple[SourceFlowOperation, ...],
    ) -> tuple[SourceFlowOperation, ...]:
        return tuple(
            operation
            for operation in operations
            if isinstance(operation.event, CompactEvaluatedAssignment)
            and operation.event.result is result
        )

    def _open_expression(
        self, environment: NativeReferenceEnvironment, node: ast.Expr
    ) -> None:
        raise ValueError("Value destination has no expression-statement obligation")

    def _discard_expression(
        self, environment: NativeReferenceEnvironment, node: ast.Expr
    ) -> None:
        environment.require_discard(node)

    def _bound_expression(
        self, environment: NativeReferenceEnvironment, node: ast.Expr
    ) -> None:
        environment.capture_value(node.value).require_closed()
        environment.require_binding_write(node)

    def _open_completion(
        self,
        resolver: CompactResultCompletionResolverABC[TargetResolutionT],
        result: CompactEvaluatedResult,
    ) -> TargetResolutionT:
        raise ValueError("Value destination has no completed-body interpretation")

    def _returned_completion(
        self,
        resolver: CompactResultCompletionResolverABC[TargetResolutionT],
        result: CompactEvaluatedResult,
    ) -> TargetResolutionT:
        return resolver._returned_result_completion(result)

    def _discarded_completion(
        self,
        resolver: CompactResultCompletionResolverABC[TargetResolutionT],
        result: CompactEvaluatedResult,
    ) -> TargetResolutionT:
        return resolver._discarded_result_completion(result)

    BOUND = "bound", True, _binding_operations, _bound_expression
    RETURNED = (
        "returned",
        False,
        _open_expression_operations,
        _open_expression,
        _returned_completion,
    )
    DISCARDED = (
        "discarded",
        False,
        _discard_operations,
        _discard_expression,
        _discarded_completion,
    )
    EMBEDDED = "embedded", False

    def __new__(
        cls,
        value: str,
        requires_binding: bool,
        expression_operations: Callable[
            [
                CompactValueDestinationKind,
                CompactEvaluatedResult,
                tuple[SourceFlowOperation, ...],
            ],
            tuple[SourceFlowOperation, ...],
        ] = _open_expression_operations,
        expression_requirement: Callable[
            [CompactValueDestinationKind, NativeReferenceEnvironment, ast.Expr], None
        ] = _open_expression,
        completion_resolution: Callable[
            [
                CompactValueDestinationKind,
                CompactResultCompletionResolverABC[TargetResolutionT],
                CompactEvaluatedResult,
            ],
            TargetResolutionT,
        ] = _open_completion,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._requires_binding = requires_binding
        member._expression_operations = expression_operations
        member._expression_requirement = expression_requirement
        member._completion_resolution = completion_resolution
        return member

    def expression_operations(
        self,
        result: CompactEvaluatedResult,
        operations: tuple[SourceFlowOperation, ...],
    ) -> tuple[SourceFlowOperation, ...]:
        selected = self._expression_operations(self, result, operations)
        if len(selected) != 1:
            raise ValueError(
                "Expression disposition requires one original application operation"
            )
        return selected

    def require_expression(
        self, environment: NativeReferenceEnvironment, node: ast.Expr
    ) -> None:
        self._expression_requirement(self, environment, node)

    def resolve_completion(
        self,
        resolver: CompactResultCompletionResolverABC[TargetResolutionT],
        result: CompactEvaluatedResult,
    ) -> TargetResolutionT:
        return self._completion_resolution(self, resolver, result)

    def require_discarded_value(
        self, value_use: CompactValueUse | None
    ) -> CompactValueUse:
        """Return the actual discarded value, not an unrelated destination's receipt."""
        if self is not type(self).DISCARDED or value_use is None:
            raise ValueError("Discard requires its actual value disposition")
        return value_use

    def validate_binding(self, binding: LexicalValueReference | None) -> None:
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

    @property
    def direct_binding_name(self) -> str | None:
        """A singular lexical destination, excluding computed object locations."""
        binding = self.binding
        return (
            binding.root_name
            if binding is not None and not binding.attribute_path
            else None
        )

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


class CompactDefinitionResolverABC(ABC, Generic[TargetResolutionT]):
    """Select source definitions independently of callable-use projection."""

    def _non_definition_resolution(self) -> TargetResolutionT:
        """A proved non-definition is distinct from an unproved receiver."""
        raise ValueError("This operation requires a source definition")

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
        pending_bindings: frozenset[
            CompactBindingVisit[CompactFlowContext]
        ] = frozenset(),
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
        pending_bindings: frozenset[
            CompactBindingVisit[CompactFlowContext]
        ] = frozenset(),
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
        pending_bindings: frozenset[
            CompactBindingVisit[CompactFlowContext]
        ] = frozenset(),
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
    predicate_use: CompactCallableReferenceUse | None = None

    def is_proved_excluded(self, resolver: CompactBranchPredicateResolverABC) -> bool:
        return self.kind.is_proved_excluded(resolver, self.predicate_use)


@dataclass(frozen=True)
class CompactEvaluationBranch:
    """One unordered member within a reserved parent event slot.

    Member indices identify sibling expressions; they never order execution.
    Each member owns a local event sequence and can contain further regions.
    """

    parent_event_index: int
    member_index: int


@dataclass(frozen=True)
class CompactFlowPosition:
    """One event position in a typed statement-suite tree."""

    branch_path: tuple[CompactControlBranch, ...]
    statement_index: int
    event_index: int

    evaluation_path: tuple[CompactEvaluationBranch, ...] = ()

    def is_proved_excluded(self, resolver: CompactBranchPredicateResolverABC) -> bool:
        """Return whether an exact enclosing branch cannot run in this activation."""
        return any(branch.is_proved_excluded(resolver) for branch in self.branch_path)

    def may_precede_cut(self, other: CompactFlowPosition) -> bool:
        """Strict entry cut, retaining an earlier iteration of the same source event."""
        return self.may_precede(other) and (
            self != other or any(branch.kind.can_repeat for branch in self.branch_path)
        )

    def _comparison_indices(self, other: CompactFlowPosition) -> tuple[int, int] | None:
        """Project same-suite order without ordering unordered sibling members."""
        if self.statement_index != other.statement_index:
            return self.statement_index, other.statement_index
        if self.evaluation_path == other.evaluation_path:
            return self.event_index, other.event_index
        for source, destination in zip(self.evaluation_path, other.evaluation_path):
            if source != destination:
                if source.parent_event_index != destination.parent_event_index:
                    return source.parent_event_index, destination.parent_event_index
                return None
        common_depth = min(len(self.evaluation_path), len(other.evaluation_path))
        return (
            (
                self.evaluation_path[common_depth].parent_event_index
                if len(self.evaluation_path) > common_depth
                else self.event_index
            ),
            (
                other.evaluation_path[common_depth].parent_event_index
                if len(other.evaluation_path) > common_depth
                else other.event_index
            ),
        )

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
        indices = self._comparison_indices(other)
        return indices is None or indices[0] <= indices[1]

    def dominates(self, other: "CompactFlowPosition") -> bool:
        if self.branch_path == other.branch_path:
            indices = self._comparison_indices(other)
            return indices is not None and indices[0] < indices[1]
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

    def _attribute_mutation_resolution(
        self,
        context: ResolutionContextT,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> TargetResolutionT:
        return self._receiver_mutation_resolution(
            context, mutation, mutation.target.receiver_use
        )

    def _item_mutation_resolution(
        self,
        context: ResolutionContextT,
        mutation: CompactMutation[CompactItemTarget],
    ) -> TargetResolutionT:
        return self._receiver_mutation_resolution(
            context, mutation, mutation.target.receiver_use
        )

    @abstractmethod
    def _receiver_mutation_resolution(
        self,
        context: ResolutionContextT,
        mutation: CompactMutation,
        receiver_use: CompactValueUse,
    ) -> TargetResolutionT:
        raise NotImplementedError


class CompactAssignmentTargetABC(DataclassGraphNode):
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
class CompactDefinitionTarget(CompactLexicalBindingTargetABC):
    """Captured header evidence and its exact source body, not final object identity."""

    owner: CompactDefinitionFlowOwner
    decorator_uses: tuple[CompactValueUse, ...]
    input_uses: tuple[CompactValueUse, ...]
    header_position: CompactFlowPosition

    bound_name = AliasProperty[str]("owner.bound_name")

    @property
    def header_uses(self) -> tuple[CompactValueUse, ...]:
        """All eagerly consumed header receipts in their native evaluation order."""
        return (*self.decorator_uses, *self.input_uses)

    def __post_init__(self) -> None:
        if not isinstance(self.owner, CompactDefinitionFlowOwner):
            raise TypeError("Definition targets require their source declaration")


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

    def resolve_mutation(
        self,
        resolver: CompactMutationResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        mutation: CompactMutation,
    ) -> TargetResolutionT:
        return resolver._attribute_mutation_resolution(
            context, cast(CompactMutation[CompactAttributeTarget], mutation)
        )

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

    def resolve_mutation(
        self,
        resolver: CompactMutationResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        mutation: CompactMutation,
    ) -> TargetResolutionT:
        return resolver._item_mutation_resolution(
            context, cast(CompactMutation[CompactItemTarget], mutation)
        )


AssignmentTargetT = TypeVar("AssignmentTargetT", bound=CompactAssignmentTargetABC)


@dataclass(frozen=True)
class CompactMutation(DataclassGraphNode, Generic[AssignmentTargetT]):
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

    def resolve_binding_value(
        self,
        resolver: CompactBindingValueResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        return resolver._value_binding_resolution(context, reference, self, pending)

    def __post_init__(self) -> None:
        self.kind.validate_import_origin(self.imported_origin)
        if self.kind.is_definition_binding != isinstance(
            self.target, CompactDefinitionTarget
        ):
            raise TypeError(
                "Definition mutations require a definition target exclusively"
            )


@dataclass(frozen=True)
class CompactEvaluatedAssignment(CompactMutation[AssignmentTargetT]):
    """One plain write owns the actual RHS, independently of destination kind."""

    result: CompactEvaluatedResult
    value_use = AliasProperty["CompactValueUse"]("result.value_use")

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind.binding_operation.require_plain_store()
        if self.result.value_use is None:
            raise ValueError("Evaluated assignment requires an actual RHS value use")
        if not (
            self.value_use.position.dominates(self.result.position)
            and self.result.position.dominates(self.position)
        ):
            raise ValueError("Evaluated assignment must follow its actual RHS result")
        name = self.result.destination.direct_binding_name
        if name is not None and name != self.target.bound_name:
            raise ValueError(
                "Evaluated destination differs from the actual name target"
            )

    def resolve_binding_value(
        self,
        resolver: CompactBindingValueResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        return resolver._evaluated_binding_resolution(context, reference, self, pending)


@dataclass(frozen=True, eq=False)
class CompactBindingVisit(Generic[ResolutionContextT]):
    """One original binding in its query context, retained for cycle detection.

    Equal-looking snapshots or source events are not the same visit. Holding
    the owners keeps identity keys valid without hashing their value graphs.
    This is query-local traversal evidence, not a runtime object identity.
    """

    context: ResolutionContextT
    mutation: CompactMutation

    def __hash__(self) -> int:
        return hash((id(self.context), id(self.mutation)))

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        other = cast(CompactBindingVisit[object], other)
        return self.context is other.context and self.mutation is other.mutation

    def required_at_read(
        self,
        context: ResolutionContextT,
        position: CompactFlowPosition,
    ) -> bool:
        """Keep every visit that can participate in this historical lookup.

        Positioned binding selection excludes a proved-future mutation in the
        same activation. Its cycle guard is therefore irrelevant at that cut.
        Other contexts cannot be ordered by this position; loops and unordered
        evaluations remain possible under the existing position relation.
        """
        return self.context is not context or self.mutation.position.may_precede(
            position
        )


class CompactFunctionTargetResolutionViolation(StrEnum):
    """Typed reasons a call target lacks one closed nominal declaration."""

    DYNAMIC_BINDING = "dynamic_binding"
    MISSING_DECLARATION = "missing_declaration"
    AMBIGUOUS_DECLARATION = "ambiguous_declaration"
    INCOMPLETE_RECEIVER_FAMILY = "incomplete_receiver_family"
    UNSUPPORTED_RECEIVER = "unsupported_receiver"
    CYCLIC_BINDING = "cyclic_binding"


class CompactBindingValueResolverABC(ABC, Generic[TargetResolutionT]):
    """Interpret an already-selected operation, independently of alias traversal."""

    @abstractmethod
    def _possible_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        violation: CompactFunctionTargetResolutionViolation,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _definition_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation[CompactDefinitionTarget],
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _imported_name_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        raise NotImplementedError

    def _value_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        """A selected write alone does not prove the value which it installed."""
        return self._possible_binding_resolution(
            context,
            reference,
            CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
            pending,
        )

    def _evaluated_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactEvaluatedAssignment,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        """Retained evaluation does not by itself prove a consumer's value semantics."""
        return self._value_binding_resolution(context, reference, binding, pending)

    def _deleted_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        return self._possible_binding_resolution(
            context,
            reference,
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
            pending_bindings,
        )

    def _initial_parameter_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: InitialCompactParameterBinding,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        """Leave entry values open unless an activation supplies this parameter."""
        return self._possible_binding_resolution(
            context,
            reference,
            binding.target_lookup_violation,
            pending_bindings,
        )


class CompactBindingResolverABC(CompactBindingValueResolverABC[TargetResolutionT]):
    """Shared interpretation of a source selected in its actual flow."""

    def _selected_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        visit = CompactBindingVisit(context, binding)
        if visit in pending_bindings:
            return self._cyclic_binding_resolution(pending_bindings)
        operation = binding.kind.binding_operation
        pending = operation.pending_after(context, binding, pending_bindings)
        alias = context.flow.exact_alias_for(binding)
        if alias is not None:
            resolution = self._captured_alias_resolution(
                alias, context, reference, use_position, pending
            )
            return self._installed_alias_resolution(resolution, alias, context)
        return operation.resolve_source(self, context, reference, binding, pending)

    @abstractmethod
    def _cyclic_binding_resolution(
        self, pending: frozenset[CompactBindingVisit[CompactFlowContext]]
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _captured_alias_resolution(
        self,
        alias: CompactExactValueAlias,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
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


class CompactBindingSource(ABC):
    """Selected source evidence, with distinct value and callable projections."""

    @abstractmethod
    def resolve_binding(
        self,
        resolver: CompactBindingResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
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
        visited_bindings: frozenset[CompactBindingVisit[CompactFunctionFlow]],
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
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
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
        visited_bindings: frozenset[CompactBindingVisit[CompactFunctionFlow]],
    ) -> CompactValueOriginResolution:
        mutation = self.selected_mutation
        possible_origins = flow._possible_alias_origins(
            reference, flow.mutations_by_root_name[reference.root_name]
        )
        visit = CompactBindingVisit(flow, mutation)
        if visit in visited_bindings:
            return OpenCompactValueOrigin(
                possible_origins, CompactValueOriginViolation.CYCLIC_ALIAS
            )
        alias = flow.exact_alias_for(mutation)
        if alias is None:
            return OpenCompactValueOrigin(
                possible_origins, CompactValueOriginViolation.INTERVENING_REBINDING
            )
        source_resolution = flow._value_origin_for(
            alias.source, alias.source_position, visited_bindings | {visit}
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
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
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
        visited_bindings: frozenset[CompactBindingVisit[CompactFunctionFlow]],
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

    def resolve_binding(
        self,
        resolver: CompactBindingResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> TargetResolutionT:
        return resolver._initial_parameter_binding_resolution(
            context, reference, self, use_position, pending_bindings
        )

    def value_origin(
        self,
        flow: CompactFunctionFlow,
        reference: LexicalValueReference,
        visited_bindings: frozenset[CompactBindingVisit[CompactFunctionFlow]],
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

    source = AliasProperty[LexicalValueReference]("source_use.lexical_reference")
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


class CompactValueResolverABC(
    ValueExpressionResolverABC[ResolutionContextT, TargetResolutionT],
):
    """Interpret retained flow results independently of their source syntax."""

    def _compiler_operand_value_resolution(
        self, value: CompilerOperandValue, context: ResolutionContextT
    ) -> TargetResolutionT:
        """Compiler operand metadata alone supplies no runtime value proof."""
        return self._unproved_value_resolution(context)

    def _subscription_result_value_resolution(
        self,
        value: SubscriptionResultValue,
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        """Retained operands do not establish the subscription's native result."""
        return self._unproved_value_resolution(context)

    def _tuple_value_resolution(
        self,
        value: CompactTupleValue,
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        """Retained input shape alone provides no construction or lifetime proof."""
        return self._unproved_value_resolution(context)

    @abstractmethod
    def _compiler_stored_value_resolution(
        self, value: CompilerStoredValue, context: ResolutionContextT
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _forwarded_result_value_resolution(
        self,
        value: ForwardedResultValue,
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _call_result_value_resolution(
        self,
        value: CallResultValue,
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        raise NotImplementedError


class CompactResolvableValue(ValueExpressionShapeABC):
    """A retained flow value interpreted by the complete flow resolver."""

    @abstractmethod
    def resolve_value(
        self,
        resolver: CompactValueResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class CompactTupleValue(
    DataclassGraphValue, NonLexicalValueShape, CompactResolvableValue
):
    """Ordered original inputs of one non-unpacking tuple production."""

    inputs: tuple[CompactValueUse, ...]

    def resolve_value(
        self,
        resolver: CompactValueResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return resolver._tuple_value_resolution(self, context)


@dataclass(frozen=True, eq=False)
class CompilerStoredValue(
    DataclassGraphValue, NonLexicalValueShape, CompactResolvableValue
):
    """An original expression whose stored value is supplied by the compiler."""

    def resolve_value(
        self,
        resolver: CompactValueResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return resolver._compiler_stored_value_resolution(self, context)


@dataclass(frozen=True, eq=False)
class CompilerOperandValue(
    DataclassGraphValue, NonLexicalValueShape, CompactResolvableValue
):
    """One implicit input of an original compiler-generated item transfer."""

    role: NativeItemStoreOperand

    def __post_init__(self) -> None:
        if not isinstance(self.role, NativeItemStoreOperand):
            raise TypeError("Compiler operand requires a declared native input role")

    def resolve_value(
        self,
        resolver: CompactValueResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return resolver._compiler_operand_value_resolution(self, context)


class CompactPositionedReference(DataclassGraphNode, CompactResolvableValue):
    """A captured expression/reference with its actual source flow position."""

    position: CompactFlowPosition

    def require_native_origin(
        self,
        source: SourceProductFlowProjection,
        read: CompactFlowValue,
        value: NativeProducedValue,
    ) -> None:
        operation = read.source_operation(source)
        if read.use is not self or value.source_span != SourceByteSpan.require_node(
            operation.node
        ):
            raise ValueError(
                "Native operand does not belong to the original source expression"
            )

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

    def origin_in(self, flow: CompactFunctionFlow) -> CompactValueOriginResolution:
        reference = self.lexical_reference
        if reference is None:
            return OpenCompactValueOrigin(
                (), CompactValueOriginViolation.OPAQUE_EXPRESSION
            )
        return flow.value_origin_for(reference, self.position)


@dataclass(frozen=True, eq=False)
class CompactValueUse(DataclassGraphValue, CompactPositionedReference):
    """One evaluated expression value, retaining its source event."""

    value: CompactValueExpression | CompactResolvableValue
    position: CompactFlowPosition
    indexes_source_expression: ClassVar[bool] = True

    lexical_reference = AliasProperty[LexicalValueReference | None](
        "value.lexical_reference"
    )

    def resolve_value(
        self,
        resolver: CompactValueResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return self.value.resolve_value(resolver, context)

    def require_source_origin(
        self,
        source: SourceProductFlowProjection,
        read: CompactFlowValue,
        operation: SourceFlowOperation,
    ) -> None:
        canonical = source.value_reads_by_node.get(operation.node)
        if (
            not isinstance(operation.node, ValueExpressionNode)
            or canonical is None
            or canonical.context is not read.context
            or canonical.use is not self
        ):
            raise ValueError("Value read has no unique original expression operation")


@dataclass(frozen=True, eq=False)
class CompactCompilerOperandUse(CompactValueUse):
    """An observed implicit input attached to its actual enclosing source node."""

    value: CompilerOperandValue
    indexes_source_expression: ClassVar[bool] = False

    def native_receipt(
        self, source: SourceProductFlowProjection, read: CompactFlowValue
    ) -> NativeReturn:
        operation = source.value_operation(read)
        if read.use is not self:
            raise ValueError("Compiler operand requires its original source read")
        return source.module.native_compilation.return_after_effect(
            SourceByteSpan.require_node(operation.node), NativeItemStoreValue
        )

    def native_operand(
        self, source: SourceProductFlowProjection, read: CompactFlowValue
    ) -> NativeProducedValue:
        receipt = self.native_receipt(source, read)
        effect = receipt.effect_for(
            SourceByteSpan.require_node(source.value_operation(read).node),
            NativeItemStoreValue,
        )
        return self.value.role.select(effect)

    def require_native_origin(
        self,
        source: SourceProductFlowProjection,
        read: CompactFlowValue,
        value: NativeProducedValue,
    ) -> None:
        if self.native_operand(source, read) is not value:
            raise ValueError("Compiler input requires its original native operand")

    def require_source_origin(
        self,
        source: SourceProductFlowProjection,
        read: CompactFlowValue,
        operation: SourceFlowOperation,
    ) -> None:
        if not isinstance(self.value, CompilerOperandValue):
            raise ValueError("Compiler operand read requires its declared input role")


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
class CompactCallableReferenceUse(CompactPositionedReference):
    target: CompactCallTargetReference
    position: CompactFlowPosition
    source_span: SourceByteSpan

    line = AliasProperty[int]("source_span.start_line")
    lexical_reference = AliasProperty[LexicalValueReference | None](
        "target.lexical_reference"
    )

    def resolve_value(
        self,
        resolver: CompactValueResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        # Computed callable targets have no retained evaluated object here. A source
        # declaration candidate would not repair that missing captured identity.
        reference = self.lexical_reference
        return (
            resolver._unproved_value_resolution(context)
            if reference is None
            else reference.resolve_value(resolver, context)
        )

    def resolve(
        self,
        resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
        *,
        pending_bindings: frozenset[
            CompactBindingVisit[CompactFlowContext]
        ] = frozenset(),
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
            self.result.binding is None
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
class CompactSubscription(DataclassGraphValue):
    """One load subscription, after capturing its original receiver and argument."""

    receiver_use: CompactValueUse
    argument_use: CompactValueUse
    position: CompactFlowPosition
    source_span: SourceByteSpan

    line = AliasProperty[int]("source_span.start_line")


@dataclass(frozen=True, eq=False)
class ForwardedResultValue(
    DataclassGraphValue, NonLexicalValueShape, CompactResolvableValue
):
    """An expression forwards its retained value after its storage effects."""

    result: CompactEvaluatedResult

    def resolve_value(
        self,
        resolver: CompactValueResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return resolver._forwarded_result_value_resolution(self, context)


@dataclass(frozen=True, eq=False)
class CallResultValue(
    DataclassGraphValue, NonLexicalValueShape, CompactResolvableValue
):
    """A retained source call's value, without claiming completed execution."""

    invocation: CompactFunctionCall

    def resolve_value(
        self,
        resolver: CompactValueResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return resolver._call_result_value_resolution(self, context)


@dataclass(frozen=True, eq=False)
class SubscriptionResultValue(
    DataclassGraphValue, NonLexicalValueShape, CompactResolvableValue
):
    """An original subscription result, independent of its execution admission."""

    invocation: CompactSubscription

    def resolve_value(
        self,
        resolver: CompactValueResolverABC[ResolutionContextT, TargetResolutionT],
        context: ResolutionContextT,
    ) -> TargetResolutionT:
        return resolver._subscription_result_value_resolution(self, context)


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


class FlowFrameResolverABC(ABC, Generic[TargetResolutionT]):
    """Admission requirements selected by the existing nominal scope declaration."""

    @abstractmethod
    def _namespace_flow_frame(
        self, context: CompactFlowContext, position: CompactFlowPosition | None
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _class_flow_frame(
        self, context: CompactFlowContext, position: CompactFlowPosition | None
    ) -> TargetResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _function_flow_frame(
        self, context: CompactFlowContext, position: CompactFlowPosition | None
    ) -> TargetResolutionT:
        raise NotImplementedError


class CompactFlowOwner(ABC):
    """Nominal scope owner, retaining its declaration when it is a function."""

    kind: CompactFlowOwnerKind
    qualname: str

    @abstractmethod
    def resolve_frame(
        self,
        resolver: FlowFrameResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        position: CompactFlowPosition | None,
    ) -> TargetResolutionT:
        raise NotImplementedError

    def initial_binding_for(self, root_name: str) -> CompactBindingSource | None:
        return None

    @property
    @abstractmethod
    def declaration(self) -> CompactFunctionDeclaration | None:
        raise NotImplementedError


class CompactDefinitionFlowOwner(CompactFlowOwner, ABC):
    """One exact source body shared by its binding event and separate flow."""

    source_span: SourceByteSpan

    @property
    def bound_name(self) -> str:
        return self.qualname.rsplit(".", 1)[-1]

    @abstractmethod
    def resolve_definition(
        self,
        resolver: CompactDefinitionResolverABC[TargetResolutionT],
        symbol: str,
        binding: CompactMutation[CompactDefinitionTarget],
    ) -> TargetResolutionT:
        raise NotImplementedError


@dataclass(frozen=True)
class CompactClassDeclaration(CompactDefinitionFlowOwner):
    """A positioned class body, independent of builder or metaclass results."""

    qualname: str
    capture: NativeClassCapture

    source_span = AliasProperty[SourceByteSpan]("capture.source_span")

    kind = CompactFlowOwnerKind.CLASS_BODY

    def resolve_frame(
        self,
        resolver: FlowFrameResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        position: CompactFlowPosition | None,
    ) -> TargetResolutionT:
        return resolver._class_flow_frame(context, position)

    @property
    def declaration(self) -> None:
        return None

    def resolve_definition(
        self,
        resolver: CompactDefinitionResolverABC[TargetResolutionT],
        symbol: str,
        binding: CompactMutation[CompactDefinitionTarget],
    ) -> TargetResolutionT:
        return resolver._selected_class_resolution(symbol, binding)


@dataclass(frozen=True)
class CompactNamespaceFlowOwner(CompactFlowOwner):
    """A module scope; definition flows retain their actual source declaration."""

    kind: CompactFlowOwnerKind
    qualname: str

    def resolve_frame(
        self,
        resolver: FlowFrameResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        position: CompactFlowPosition | None,
    ) -> TargetResolutionT:
        return resolver._namespace_flow_frame(context, position)

    def __post_init__(self) -> None:
        if not self.kind.is_module_scope:
            raise ValueError("Definition flows must be owned by their declaration")

    @property
    def declaration(self) -> None:
        return None


@dataclass(frozen=True)
class CompactFunctionDeclaration(CompactDefinitionFlowOwner):
    identity: CompactFunctionIdentity
    execution: NativeFunctionExecution
    owner_class_qualname: str | None
    signature: CompactFunctionSignature
    decorators: tuple[CompactValueExpression, ...] = ()
    return_annotation_expression: str | None = None

    kind = CompactFlowOwnerKind.FUNCTION

    qualname = AliasProperty[str]("identity.qualname")
    source_span = AliasProperty[SourceByteSpan]("execution.source_span")
    line = AliasProperty[int]("source_span.start_line")
    end_line = AliasProperty[int]("source_span.end_line")

    def resolve_frame(
        self,
        resolver: FlowFrameResolverABC[TargetResolutionT],
        context: CompactFlowContext,
        position: CompactFlowPosition | None,
    ) -> TargetResolutionT:
        return resolver._function_flow_frame(context, position)

    def resolve_definition(
        self,
        resolver: CompactDefinitionResolverABC[TargetResolutionT],
        symbol: str,
        binding: CompactMutation[CompactDefinitionTarget],
    ) -> TargetResolutionT:
        return resolver._selected_function_resolution(symbol, binding)

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


@dataclass(frozen=True)
class CompactNativeCapture(DataclassGraphNode):
    """An actual native capture positioned within source header evaluation.

    The site's actual native frame origin is independent of the enclosing
    source flow. A generated frame stays unjoined; no activation is inferred.
    """

    site: NativeCaptureSite
    position: CompactFlowPosition


@dataclass(frozen=True, eq=False)
class CompactFunctionFlow(StoredDataclassState, DataclassGraphValue):
    owner: CompactFlowOwner
    lexical_scope_qualnames: tuple[str, ...]
    calls: tuple[CompactFunctionCall, ...]
    evaluated_results: tuple[CompactEvaluatedResult, ...]
    callable_reference_uses: tuple[CompactCallableReferenceUse, ...]
    mutations: tuple[CompactMutation, ...]
    exact_value_aliases: tuple[CompactExactValueAlias, ...]
    global_binding_names: tuple[str, ...]
    nonlocal_binding_names: tuple[str, ...]

    native_captures: tuple[CompactNativeCapture, ...]

    subscriptions: tuple[CompactSubscription, ...]

    def local_binding_hides_outer_lookup(self, root_name: str) -> bool:
        """A declared function local cannot fall through to globals when unbound."""
        return (
            self.owner.kind.is_function_scope
            and root_name not in self.global_binding_names
            and root_name not in self.nonlocal_binding_names
            and (
                root_name in self.mutations_by_root_name
                or self.owner.initial_binding_for(root_name) is not None
            )
        )

    @cached_property
    def graph_nodes_by_identity(self) -> dict[int, DataclassGraphNode]:
        """Retained source nodes, derived once from this immutable flow graph."""
        return {id(node): node for node in self.graph_nodes()}

    def stored_binding_resolution_for(
        self,
        root_name: str,
        cut: CompactFlowPosition | None,
    ) -> CompactBindingSource | None:
        """Select actual storage before a cut, excluding the operation at that cut.

        Incomparable possibly earlier writes remain candidates. Lexical forward
        binding and outer lookup are separate scope obligations, not storage facts.
        """
        candidates = tuple(
            mutation
            for mutation in self.mutations_by_root_name.get(root_name, ())
            if cut is None or mutation.position.may_precede_cut(cut)
        )
        return self._binding_resolution_for_mutations(candidates, cut, root_name)

    def stored_binding_resolution_for_activation(
        self,
        root_name: str,
        cut: CompactFlowPosition,
        resolver: CompactBranchPredicateResolverABC,
    ) -> CompactBindingSource | None:
        """Select storage after removing branches disproved by this activation."""
        if cut.is_proved_excluded(resolver):
            raise ValueError("Excluded source event has no activation binding")
        candidates = tuple(
            mutation
            for mutation in self.mutations_by_root_name.get(root_name, ())
            if mutation.position.may_precede_cut(cut)
            and not mutation.position.is_proved_excluded(resolver)
        )
        return self._binding_resolution_for_mutations(candidates, cut, root_name)

    @property
    def reference_uses(self) -> Iterator[CompactCallableReferenceUse]:
        """Actual source reads, including callees captured before their arguments."""
        return chain(
            self.callable_reference_uses,
            (call.target_use for call in self.calls),
        )

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
                    for use in self.reference_uses
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
        if not mutations:
            return self.owner.initial_binding_for(root_name)
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
            (
                mutation.position.branch_path
                if use_position is None
                else mutation.position.may_precede(use_position)
            )
            for mutation in mutations
        ):
            return OpenCompactBindingMutation(
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
            )
        initial_binding = self.owner.initial_binding_for(root_name)
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
    def _exact_aliases_by_binding_identity(
        self,
    ) -> dict[int, CompactExactValueAlias]:
        return {id(alias.binding_mutation): alias for alias in self.exact_value_aliases}

    def exact_alias_for(
        self, mutation: CompactMutation
    ) -> CompactExactValueAlias | None:
        """Look up the alias of this actual source event, not an equal snapshot.

        Aliases retain their binding events, so the derived identity index is
        valid for the flow's lifetime without hashing the events' value graphs.
        """
        return self._exact_aliases_by_binding_identity.get(id(mutation))

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
        if binding is None:
            return None
        return binding.kind.binding_operation.bound_call_result(
            self, binding, reference
        )

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
        visited_bindings: frozenset[CompactBindingVisit[CompactFunctionFlow]],
    ) -> CompactValueOriginResolution:
        selection = self.binding_resolution_for(reference.root_name, use_position)
        if selection is None:
            return ExactCompactValueOrigin(reference)
        return selection.value_origin(self, reference, visited_bindings)

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
                        if (alias := self.exact_alias_for(mutation)) is not None
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


CompactDefinitionSource: TypeAlias = tuple[
    CompactFlowContext, CompactMutation[CompactDefinitionTarget]
]


@dataclass(frozen=True)
class CompactFlowValue:
    """An actual evaluated value with its canonical source-flow context."""

    context: CompactFlowContext
    use: CompactPositionedReference

    def source_operation(
        self, source: SourceProductFlowProjection
    ) -> SourceFlowOperation:
        return source.value_operation(self)


@dataclass(frozen=True)
class CompactFlowRead(CompactFlowValue):
    """A retained source read in its actual module and flow context."""

    use: CompactCallableReferenceUse

    source_span = AliasProperty[SourceByteSpan]("use.source_span")

    def source_operation(
        self, source: SourceProductFlowProjection
    ) -> SourceFlowOperation:
        operation = source.source_operation(self.context, self.use)
        canonical = source.reference_reads_by_node.get(operation.node)
        if (
            canonical is None
            or canonical.context is not self.context
            or canonical.use is not self.use
        ):
            raise ValueError("Callable read has no original source operation")
        return operation


@dataclass(frozen=True)
class CompactProductFlowModuleProjection(StoredDataclassState, CompactModuleIdentity):
    """AST-free function declarations and source-ordered product-flow facts."""

    flows: tuple[CompactFunctionFlow, ...]

    @cached_property
    def value_captures_by_identity(self) -> dict[int, CompactFlowValue]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                CompactFlowValue(context, value)
                for context in self.flow_contexts
                for value in context.flow.graph_nodes_by_identity.values()
                if isinstance(value, CompactValueUse)
            ),
            lambda capture: id(capture.use),
        )

    @cached_property
    def definition_sources_by_owner(
        self,
    ) -> dict[CompactDefinitionFlowOwner, CompactDefinitionSource]:
        """Unique actual parent/event/body joins, without an activation claim."""
        sources = UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                (context, cast(CompactMutation[CompactDefinitionTarget], mutation))
                for context in self.flow_contexts
                for mutation in context.flow.mutations
                if mutation.kind.is_definition_binding
            ),
            lambda source: source[1].target.owner,
        )
        contexts = self.flow_contexts_by_owner
        return {
            owner: source
            for owner, source in sources.items()
            if owner in contexts and contexts.get(source[0].flow.owner) is source[0]
        }

    @cached_property
    def reference_reads_by_span(self) -> dict[SourceByteSpan, CompactFlowRead]:
        """Only uniquely retained source reads; computed values are not inferred."""
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                CompactFlowRead(context, use)
                for context in self.flow_contexts
                for use in context.flow.reference_uses
            ),
            lambda read: read.source_span,
        )

    @cached_property
    def flow_contexts_by_owner(self) -> dict[CompactFlowOwner, CompactFlowContext]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.flow_contexts, lambda context: context.flow.owner
        )

    @cached_property
    def flow_contexts(self) -> tuple[CompactFlowContext, ...]:
        return tuple(
            CompactFlowContext(self.module_name, self.file_path, flow)
            for flow in self.flows
        )

    @cached_property
    def flow_contexts_by_identity(self) -> dict[int, CompactFlowContext]:
        """Actual context membership, distinct from unique declaration ownership."""
        return {id(context): context for context in self.flow_contexts}

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
    declaration: CompactClassDeclaration
    lexical_scope_qualnames: tuple[str, ...]

    current_class_qualname = AliasProperty[str]("declaration.qualname")


class _DeclarationCollector(ast.NodeVisitor):
    """Collect declaration identities while preserving Python scope nesting."""

    @property
    def owners_by_node(
        self,
    ) -> dict[
        ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        CompactDefinitionFlowOwner,
    ]:
        return {
            context.node: context.declaration
            for context in chain(self.function_contexts, self.class_contexts)
        }

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
            _ClassContext(
                node,
                CompactClassDeclaration(
                    qualname,
                    self.module.native_compilation.class_capture_for(
                        SourceByteSpan.require_node(node)
                    ),
                ),
                lexical_scopes,
            )
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

    def visit_Name(self, node: ast.Name) -> CompactLexicalBindingTargetABC:
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


SourceFlowEvent: TypeAlias = (
    CompactSubscription
    | CompactFunctionCall
    | CompactMutation
    | CompactCallableReferenceUse
    | CompactEvaluatedResult
    | CompactValueUse
    | CompactNativeCapture
)


@dataclass(frozen=True)
class SourceFlowEvaluation:
    """Observed visitor interval, not an exact implicit-operation timestamp.

    The interval is retained even when it contains no compact event. Missing
    modeled calls never certify that an expression or statement is effect-free.
    These AST references belong to a task, not a cached compact flow payload.
    """

    node: ast.AST
    owner: CompactFlowOwner
    entry: CompactFlowPosition
    exit: CompactFlowPosition


@dataclass(frozen=True, eq=False)
class SourceFlowOperation:
    """Actual compact operation joined to its original source node and owner."""

    node: SourcePositionedNode
    owner: CompactFlowOwner
    event: SourceFlowEvent

    @property
    def position(self) -> CompactFlowPosition:
        return self.event.position


@dataclass(frozen=True)
class SourceProductFlowProjection:
    """One task's ordered source observations and the exact flows they created."""

    module: ParsedModule
    compact: CompactProductFlowModuleProjection
    evaluations: tuple[SourceFlowEvaluation, ...]
    operations: tuple[SourceFlowOperation, ...]

    def node_operation(
        self, node: ast.AST, event_type: type[SourceFlowEvent]
    ) -> SourceFlowOperation:
        """Join one declared event family at its original source node."""
        operations = tuple(
            operation
            for operation in self.operations_by_node.get(node, ())
            if isinstance(operation.event, event_type)
        )
        if len(operations) != 1:
            raise ValueError("Source node has no unique actual operation")
        operation = operations[0]
        canonical = self.event_operation(operation.event)
        if canonical is not operation or operation.node is not node:
            raise ValueError("Source operation has no original node association")
        return operation

    def definition_operation(self, node: ast.AST) -> SourceFlowOperation:
        """Join an actual source definition to its canonical creation binding."""
        operation = self.mutation_operation(node)
        if not isinstance(
            cast(CompactMutation, operation.event).target, CompactDefinitionTarget
        ):
            raise ValueError("Source definition requires an actual definition binding")
        return operation

    def event_operation(self, event: object) -> SourceFlowOperation:
        """Join an actual event to its original node and canonical owning flow."""
        operations = self.operations_by_event_identity.get(id(event), ())
        if len(operations) != 1:
            raise ValueError("Source event has no unique original operation")
        operation = operations[0]
        context = self.context_for_owner(operation.owner)
        if (
            operation.event is not event
            or context.flow.graph_nodes_by_identity.get(id(event)) is not event
        ):
            raise ValueError("Source event belongs to a different canonical context")
        return operation

    def call_operation(self, span: SourceByteSpan) -> SourceFlowOperation:
        """Resolve a complete call span to its canonical operation and context."""
        operation = self.call_operations_by_span.get(span)
        if operation is None:
            raise ValueError("Call span has no unique original operation")
        if (
            self.event_operation(operation.event) is not operation
            or not isinstance(operation.node, ast.Call)
            or SourceByteSpan.require_node(operation.node) != span
        ):
            raise ValueError("Call span has no original node association")
        return operation

    @cached_property
    def call_operations_by_span(self) -> dict[SourceByteSpan, SourceFlowOperation]:
        """Index actual call events; ambiguous spans cannot select an occurrence."""
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                site
                for site in self.operations
                if isinstance(site.event, CompactFunctionCall)
            ),
            lambda site: site.event.source_span,
        )

    def evaluation_operand_entry(
        self,
        evaluation: SourceFlowEvaluation,
        operand: ast.expr,
    ) -> CompactFlowPosition:
        """Join an actual contained operand interval without inventing an event."""
        if not any(
            original is evaluation
            for original in self.evaluation_bounds_by_node.get(evaluation.node, ())
        ):
            raise ValueError("Effect bound requires its original source evaluation")
        candidates = tuple(
            original
            for original in self.evaluation_bounds_by_node.get(operand, ())
            if original.owner is evaluation.owner
        )
        if len(candidates) != 1:
            raise ValueError("Effect operand has no unique original evaluation")
        original = candidates[0]
        if original.node is not operand or not (
            (
                evaluation.entry == original.entry
                or evaluation.entry.dominates(original.entry)
            )
            and (
                original.exit == evaluation.exit
                or original.exit.dominates(evaluation.exit)
            )
        ):
            raise ValueError("Effect operand is outside its original evaluation")
        return original.entry

    def __post_init__(self) -> None:
        nodes = module_syntax_index(self.module.module).node_membership
        if any(
            site.node not in nodes for site in chain(self.operations, self.evaluations)
        ):
            raise ValueError("Source flow sites must belong to their actual module AST")

    @cached_property
    def evaluation_bounds_by_node(
        self,
    ) -> dict[ast.AST, tuple[SourceFlowEvaluation, ...]]:
        grouped: dict[ast.AST, list[SourceFlowEvaluation]] = {}
        for evaluation in self.evaluations:
            grouped.setdefault(evaluation.node, []).append(evaluation)
        return {node: tuple(evaluations) for node, evaluations in grouped.items()}

    @cached_property
    def source_nodes_by_owner(self) -> dict[int, ast.AST]:
        return {
            id(self.module_context.flow.owner): self.module.module,
            **{
                id(site.event.target.owner): site.node
                for site in self.operations
                if isinstance(site.event, CompactMutation)
                and isinstance(site.event.target, CompactDefinitionTarget)
            },
        }

    @cached_property
    def module_context(self) -> CompactFlowContext:
        contexts = tuple(
            context
            for context in self.compact.flow_contexts
            if context.flow.owner.kind.is_module_scope
        )
        if len(contexts) != 1:
            raise ValueError("Source module entry requires one actual module flow")
        return contexts[0]

    def source_operation(
        self, context: CompactFlowContext, event: object
    ) -> SourceFlowOperation:
        operation = self.event_operation(event)
        if self.compact.flow_contexts_by_owner[operation.owner] is not context:
            raise ValueError("Source event belongs to a different canonical context")
        return operation

    def value_operation(self, read: CompactFlowValue) -> SourceFlowOperation:
        """Join an original evaluated value, not an earlier callable-reference read."""
        operation = self.source_operation(read.context, read.use)
        if not isinstance(read.use, CompactValueUse):
            raise ValueError("Value read has no unique original expression operation")
        read.use.require_source_origin(self, read, operation)
        return operation

    def mutation_operation(self, node: ast.AST) -> SourceFlowOperation:
        """Join one actual mutation without selecting its storage semantics."""
        return self.node_operation(node, CompactMutation)

    def context_for_owner(self, owner: CompactFlowOwner) -> CompactFlowContext:
        context = self.compact.flow_contexts_by_owner.get(owner)
        if (
            context is None
            or context.flow.owner is not owner
            or self.compact.flow_contexts_by_identity.get(id(context)) is not context
        ):
            raise ValueError("Source owner has no unique actual flow context")
        return context

    @cached_property
    def operations_by_event_identity(
        self,
    ) -> Mapping[int, tuple[SourceFlowOperation, ...]]:
        """Retain multiplicity when joining original source to actual compact events."""
        grouped: dict[int, list[SourceFlowOperation]] = {}
        for operation in self.operations:
            grouped.setdefault(id(operation.event), []).append(operation)
        return {identity: tuple(operations) for identity, operations in grouped.items()}

    @cached_property
    def value_reads_by_node(self) -> dict[ast.AST, CompactFlowValue]:
        """Exact evaluated-value captures, distinct from earlier lexical reads."""
        sites = UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                site
                for site in self.operations
                if isinstance(site.event, CompactValueUse)
                and site.event.indexes_source_expression
            ),
            lambda site: site.node,
        )
        captures = self.compact.value_captures_by_identity
        return {
            node: capture
            for node, site in sites.items()
            if (capture := captures.get(id(site.event))) is not None
            and capture.use is site.event
            and capture.context.flow.owner is site.owner
        }

    @cached_property
    def operations_by_node(self) -> dict[ast.AST, tuple[SourceFlowOperation, ...]]:
        """Keep all actual trigger operations; one node may own several phases."""
        grouped: dict[ast.AST, list[SourceFlowOperation]] = {}
        for operation in self.operations:
            grouped.setdefault(operation.node, []).append(operation)
        return {node: tuple(operations) for node, operations in grouped.items()}

    @cached_property
    def reference_reads_by_node(self) -> dict[ast.AST, CompactFlowRead]:
        """Join original AST nodes to canonical reads, never just equal spans.

        Callable capture may bypass a visitor evaluation interval. Operation
        receipts retain that actual capture; both event and owner must be the
        canonical objects in this projection. Duplicate node sites remain open.
        """
        sites = UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                site
                for site in self.operations
                if isinstance(site.event, CompactCallableReferenceUse)
            ),
            lambda site: site.node,
        )
        reads = self.compact.reference_reads_by_span
        return {
            node: read
            for node, site in sites.items()
            if (read := reads.get(SourceByteSpan.require_node(node))) is not None
            and read.use is site.event
            and read.context.flow.owner is site.owner
        }


@dataclass(frozen=True)
class _ClassHeaderCaptureProjection(NativeClassCaptureResolverABC[None]):
    """Project an admitted native class header into its actual source traversal."""

    collector: _CompactFlowCollector
    node: ast.ClassDef

    def _exact_class_capture_resolution(self, capture: ExactNativeClassCapture) -> None:
        # Original admitted class lowering captures the builder and body before
        # evaluating bases. Generic wrappers preserve that source phase order,
        # but their actual frame remains the native site's unresolved origin.
        for site in (capture.builder, capture.creation):
            event = CompactNativeCapture(site, self.collector._position())
            self.collector.native_captures.append(event)
            self.collector._record_operation(self.node, event)

    def _open_class_capture_resolution(self, capture: OpenNativeClassCapture) -> None:
        # The declaration retains the reason; absence is no capture/order proof.
        pass


class _CompactFlowCollector(
    EagerFunctionAnnotationVisitor,
    DictionaryEvaluationVisitor,
    VariableAnnotationVisitorABC,
):
    """Collect one source scope without descending into nested scope bodies."""

    def visit_Tuple(self, node: ast.Tuple) -> CompactTupleValue | None:
        if not isinstance(node.ctx, ast.Load) or any(
            isinstance(element, ast.Starred) for element in node.elts
        ):
            self.generic_visit(node)
            return None
        return CompactTupleValue(
            tuple(self._capture_value(element) for element in node.elts)
        )

    def visit_Constant(self, node: ast.Constant) -> CompilerStoredValue | None:
        if (
            self.documentation_statement is not None
            and node is self.documentation_statement.value
        ):
            return CompilerStoredValue()
        return None

    def _visit_assignment_targets(
        self,
        targets: tuple[ast.expr, ...] | list[ast.expr],
        result: CompactEvaluatedResult | None,
    ) -> tuple[CompactMutation, ...]:
        previous = self.assignment_result
        self.assignment_result = (
            (targets[0], result)
            if result is not None
            and len(targets) == 1
            and isinstance(targets[0], (ast.Name, ast.Attribute, ast.Subscript))
            else None
        )
        try:
            return self._visit_mutation_targets(targets, CompactMutationKind.ASSIGNMENT)
        finally:
            self.assignment_result = previous

    def _cursor_position(self) -> CompactFlowPosition:
        """Snapshot the shared event cursor without inventing a source event."""
        return CompactFlowPosition(
            self.branch_path,
            self.statement_index,
            self.event_index,
            self.evaluation_path,
        )

    def _record_operation(
        self, node: SourcePositionedNode, event: SourceFlowEvent
    ) -> None:
        """Task collectors can retain this exact generated operation without replay."""

    def visit_unordered_annotations(self, roots: tuple[ast.expr, ...]) -> None:
        """Preserve each root's order without ordering sibling root evaluations."""
        parent_path = self.evaluation_path
        parent_slot = self.event_index
        try:
            for member, expression in enumerate(roots):
                self.evaluation_path = (
                    *parent_path,
                    CompactEvaluationBranch(parent_slot, member),
                )
                self.event_index = 0
                self.visit_annotation(expression)
        finally:
            self.evaluation_path = parent_path
            self.event_index = parent_slot + 1

    def _capture_result(
        self,
        expression: ast.expr | None,
        destination: CompactValueDestination,
        statement: SourcePositionedNode,
    ) -> CompactEvaluatedResult:
        if isinstance(expression, ast.Call):
            self.call_results[id(expression)] = destination
        value_use = None if expression is None else self._capture_value(expression)
        return self._record_evaluated_result(value_use, destination, statement)

    def _record_evaluated_result(
        self,
        value_use: CompactValueUse | None,
        destination: CompactValueDestination,
        statement: SourcePositionedNode,
    ) -> CompactEvaluatedResult:
        result = CompactEvaluatedResult(
            destination,
            value_use,
            self._position(),
            SourceByteSpan.require_node(statement),
        )
        self.evaluated_results.append(result)
        self._record_operation(statement, result)
        return result

    def visit_Subscript(self, node: ast.Subscript) -> SubscriptionResultValue | None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._record_target_mutation(node)
            return None
        receiver = self._capture_value(node.value)
        argument = self._capture_value(node.slice)
        invocation = CompactSubscription(
            receiver_use=receiver,
            argument_use=argument,
            position=self._position(),
            source_span=SourceByteSpan.require_node(node),
        )
        self.subscriptions.append(invocation)
        self._record_operation(node, invocation)
        return SubscriptionResultValue(invocation)

    def _record_target_mutation(
        self,
        node: ast.expr,
        kind: CompactMutationKind | None = None,
    ) -> None:
        target = self.mutation_targets.visit(node)
        result = (
            self.assignment_result[1]
            if self.assignment_result is not None and self.assignment_result[0] is node
            else None
        )
        self._record_mutation(target, node, kind, result=result)

    def _capture_definition_input(self, expression: ast.expr) -> None:
        self.definition_input_uses.append(self._capture_value(expression))

    visit_annotation = _capture_definition_input

    def _capture_value(self, expression: ValueExpressionNode) -> CompactValueUse:
        value = self.visit(expression)
        if value is None:
            value = CompactValueExpression.project(expression)
        use = CompactValueUse(value, self._position())
        self._record_operation(expression, use)
        return use

    def __init__(
        self,
        *,
        owner: CompactFlowOwner,
        definition_owners: dict[
            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
            CompactDefinitionFlowOwner,
        ],
        module_identity: PythonModulePathIdentity,
        annotation_mode: ModuleAnnotationEvaluationMode,
        lexical_scope_qualnames: tuple[str, ...],
        current_class_qualname: str | None,
        current_class_receiver_name: str | None,
    ) -> None:
        self.owner = owner
        self.definition_owners = definition_owners
        self.module_identity = module_identity
        self.annotation_mode = annotation_mode
        self.lexical_scope_qualnames = lexical_scope_qualnames
        self.current_class_qualname = current_class_qualname
        self.current_class_receiver_name = current_class_receiver_name
        self.calls: list[CompactFunctionCall] = []
        self.subscriptions: list[CompactSubscription] = []
        self.native_captures: list[CompactNativeCapture] = []
        self.evaluated_results: list[CompactEvaluatedResult] = []
        self.definition_input_uses: list[CompactValueUse] = []
        self.callable_reference_uses: list[CompactCallableReferenceUse] = []
        self.mutations: list[CompactMutation] = []
        self.exact_value_aliases: list[CompactExactValueAlias] = []
        self.global_binding_names: set[str] = set()
        self.nonlocal_binding_names: set[str] = set()
        self.branch_path: tuple[CompactControlBranch, ...] = ()
        self.statement_index = 0
        self.event_index = 0
        self.evaluation_path: tuple[CompactEvaluationBranch, ...] = ()
        self.call_results: dict[int, CompactValueDestination] = {}
        self.mutation_kind = CompactMutationKind.ASSIGNMENT
        self.mutation_targets = _CompactMutationTargetCollector(self)
        self.assignment_result: tuple[ast.expr, CompactEvaluatedResult] | None = None

    def collect(self, statements: list[ast.stmt]) -> CompactFunctionFlow:
        self.documentation_statement = self.owner.kind.documentation_statement(
            statements
        )
        self._collect_statements(statements)
        return CompactFunctionFlow(
            owner=self.owner,
            lexical_scope_qualnames=self.lexical_scope_qualnames,
            calls=tuple(self.calls),
            subscriptions=tuple(self.subscriptions),
            native_captures=tuple(self.native_captures),
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
        predicate_use: CompactCallableReferenceUse | None = None,
    ) -> None:
        saved_path = self.branch_path
        self.branch_path = (
            *saved_path,
            CompactControlBranch(
                self.statement_index,
                kind,
                alternative_index,
                predicate_use,
            ),
        )
        self._collect_statements(statements)
        self.branch_path = saved_path

    def _position(self) -> CompactFlowPosition:
        position = self._cursor_position()
        self.event_index += 1
        return position

    def _record_mutation(
        self,
        target: CompactAssignmentTargetABC,
        node: SourcePositionedNode,
        kind: CompactMutationKind | None = None,
        *,
        result: CompactEvaluatedResult | None = None,
    ) -> CompactMutation:
        factory = (
            CompactMutation
            if result is None
            else partial(CompactEvaluatedAssignment, result=result)
        )
        mutation = factory(
            target=target,
            kind=self.mutation_kind if kind is None else kind,
            position=self._position(),
            line=node.lineno,
        )
        self.mutations.append(mutation)
        self._record_operation(node, mutation)
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
        use = CompactCallableReferenceUse(
            target=self._call_target(node),
            position=self._position(),
            source_span=SourceByteSpan.require_node(node),
        )
        self._record_operation(node, use)
        return use

    def visit_Call(self, node: ast.Call) -> CallResultValue:
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
        self._record_operation(node, invocation)
        return CallResultValue(invocation)

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
        mutations = self._visit_assignment_targets(node.targets, result)
        self._record_exact_value_aliases(
            node.targets, result.lexical_reference, mutations
        )

    def _visit_annotated_assignment(self, node: ast.AnnAssign) -> None:
        result = None
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
        mutations = self._visit_assignment_targets((node.target,), result)
        self._record_exact_value_aliases(
            (node.target,),
            None if result is None else result.lexical_reference,
            mutations,
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_annotated_assignment(node)
        self.owner.kind.visit_assignment_annotation(self, node, self.annotation_mode)

    def _capture_compiler_operand(
        self, node: ast.AnnAssign, role: NativeItemStoreOperand
    ) -> CompactCompilerOperandUse:
        use = CompactCompilerOperandUse(CompilerOperandValue(role), self._position())
        self._record_operation(node, use)
        return use

    def _store_variable_annotation(
        self, node: ast.AnnAssign, result: CompactEvaluatedResult
    ) -> None:
        receiver = self._capture_compiler_operand(node, NativeItemStoreOperand.RECEIVER)
        key = self._capture_compiler_operand(node, NativeItemStoreOperand.KEY)
        self._record_mutation(CompactItemTarget(receiver, key), node, result=result)

    def visit_eager_variable_annotation(self, node: ast.AnnAssign) -> None:
        if node.simple:
            result = self._capture_result(
                node.annotation,
                CompactValueDestination(CompactValueDestinationKind.EMBEDDED),
                node.annotation,
            )
            self._store_variable_annotation(node, result)
        else:
            self.visit(node.annotation)

    def visit_stringized_variable_annotation(self, node: ast.AnnAssign) -> None:
        if node.simple:
            use = self._capture_compiler_operand(node, NativeItemStoreOperand.VALUE)
            result = self._record_evaluated_result(
                use,
                CompactValueDestination(CompactValueDestinationKind.EMBEDDED),
                node.annotation,
            )
            self._store_variable_annotation(node, result)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        target = self.mutation_targets.visit(node.target)
        self.callable_reference_uses.append(self._callable_reference_use(node.target))
        self.visit(node.value)
        self._record_mutation(target, node, CompactMutationKind.AUGMENTED_ASSIGNMENT)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ForwardedResultValue:
        result = self._capture_result(
            node.value, CompactValueDestination.for_assignment((node.target,)), node
        )
        self._visit_assignment_targets((node.target,), result)
        return ForwardedResultValue(result)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._visit_mutation_targets(node.targets, CompactMutationKind.DELETION)

    def visit_Return(self, node: ast.Return) -> None:
        self._capture_result(
            node.value,
            CompactValueDestination(CompactValueDestinationKind.RETURNED),
            node,
        )

    def visit_Expr(self, node: ast.Expr) -> None:
        destination = (
            CompactValueDestination(
                CompactValueDestinationKind.BOUND,
                LexicalValueReference(
                    CPythonClassConstructionField.DOCUMENTATION.value
                ),
            )
            if node is self.documentation_statement
            else CompactValueDestination(CompactValueDestinationKind.DISCARDED)
        )
        result = self._capture_result(node.value, destination, node)
        if destination.direct_binding_name is not None:
            self._record_mutation(
                CompactBindingTarget(destination.direct_binding_name),
                node,
                result=result,
            )

    def _visit_mutation_targets(
        self,
        targets: tuple[ast.expr, ...] | list[ast.expr],
        kind: CompactMutationKind,
    ) -> tuple[CompactMutation, ...]:
        mutation_start = len(self.mutations)
        saved_kind = self.mutation_kind
        self.mutation_kind = kind
        try:
            for target in targets:
                self.visit(target)
        finally:
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

    def _bind_definition(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        decorator_uses: tuple[CompactValueUse, ...],
        input_start: int,
    ) -> None:
        self._record_mutation(
            CompactDefinitionTarget(
                self.definition_owners[node],
                decorator_uses,
                tuple(self.definition_input_uses[input_start:]),
                self._position(),
            ),
            node,
            CompactMutationKind.DEFINITION,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        input_start = len(self.definition_input_uses)
        self._bind_definition(
            node, self._visit_definition_expressions(node), input_start
        )

    visit_AsyncFunctionDef = visit_FunctionDef

    def _visit_definition_expressions(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[CompactValueUse, ...]:
        decorator_uses = self._capture_decorators(node)
        for default in self.default_roots(node.args):
            self._capture_definition_input(default)
        if self.annotation_mode.annotations_execute_at_declaration:
            self.visit_function_annotations(node)
        return decorator_uses

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        input_start = len(self.definition_input_uses)
        decorator_uses = self._capture_decorators(node)
        declaration = cast(CompactClassDeclaration, self.definition_owners[node])
        declaration.capture.resolve(_ClassHeaderCaptureProjection(self, node))
        for base in node.bases:
            self._capture_definition_input(base)
        for keyword in node.keywords:
            self._capture_definition_input(keyword.value)
        self._bind_definition(node, decorator_uses, input_start)

    def _direct_predicate_use(
        self, expression: ast.expr
    ) -> CompactCallableReferenceUse | None:
        """Retain the exact direct-name read used by one control predicate."""
        read_start = len(self.callable_reference_uses)
        self.visit(expression)
        reads = tuple(self.callable_reference_uses[read_start:])
        if (
            not isinstance(expression, ast.Name)
            or len(reads) != 1
            or reads[0].source_span != SourceByteSpan.require_node(expression)
        ):
            return None
        return reads[0]

    def visit_If(self, node: ast.If) -> None:
        predicate_use = self._direct_predicate_use(node.test)
        self._collect_branch(
            node.body,
            CompactControlBranchKind.IF_BODY,
            predicate_use=predicate_use,
        )
        self._collect_branch(
            node.orelse,
            CompactControlBranchKind.IF_ELSE,
            predicate_use=predicate_use,
        )

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


class _SourceFlowCollector(_CompactFlowCollector):
    """Observe the inherited evaluation traversal; never re-interpret its AST."""

    @cached_property
    def source_evaluations(self) -> list[SourceFlowEvaluation]:
        return []

    @cached_property
    def source_operations(self) -> list[SourceFlowOperation]:
        return []

    def visit(self, node: ast.AST) -> CompactResolvableValue | None:
        entry = self._cursor_position()
        result = super().visit(node)
        self.source_evaluations.append(
            SourceFlowEvaluation(node, self.owner, entry, self._cursor_position())
        )
        return result

    def _record_operation(
        self, node: SourcePositionedNode, event: SourceFlowEvent
    ) -> None:
        self.source_operations.append(SourceFlowOperation(node, self.owner, event))


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


FlowCollectorT = TypeVar("FlowCollectorT", bound=_CompactFlowCollector)


@dataclass(frozen=True)
class _ProductFlowCollection(Generic[FlowCollectorT]):
    """Instantiate the existing scope collectors once for either projection."""

    parsed_module: ParsedModule
    collector_type: type[FlowCollectorT]

    @cached_property
    def collectors(self) -> tuple[tuple[FlowCollectorT, list[ast.stmt]], ...]:
        parsed_module = self.parsed_module
        declarations = _DeclarationCollector(parsed_module)
        declarations.visit(parsed_module.module)
        definition_owners = declarations.owners_by_node
        annotation_mode = ModuleAnnotationEvaluationMode.from_module(
            parsed_module.module
        )
        collectors = [
            (
                self.collector_type(
                    owner=CompactNamespaceFlowOwner(CompactFlowOwnerKind.MODULE, ""),
                    definition_owners=definition_owners,
                    module_identity=parsed_module.module_path_identity,
                    annotation_mode=annotation_mode,
                    lexical_scope_qualnames=("",),
                    current_class_qualname=None,
                    current_class_receiver_name=None,
                ),
                parsed_module.module.body,
            )
        ]
        collectors.extend(
            (
                self.collector_type(
                    owner=context.declaration,
                    definition_owners=definition_owners,
                    module_identity=parsed_module.module_path_identity,
                    lexical_scope_qualnames=context.lexical_scope_qualnames,
                    current_class_qualname=context.current_class_qualname,
                    current_class_receiver_name=None,
                    annotation_mode=annotation_mode,
                ),
                context.node.body,
            )
            for context in declarations.class_contexts
        )
        collectors.extend(
            (
                self.collector_type(
                    owner=context.declaration,
                    definition_owners=definition_owners,
                    module_identity=parsed_module.module_path_identity,
                    lexical_scope_qualnames=context.lexical_scope_qualnames,
                    current_class_qualname=context.current_class_qualname,
                    current_class_receiver_name=context.declaration.nominal_receiver_name,
                    annotation_mode=annotation_mode,
                ),
                context.node.body,
            )
            for context in declarations.function_contexts
        )
        return tuple(collectors)

    @cached_property
    def projection(self) -> CompactProductFlowModuleProjection:
        return CompactProductFlowModuleProjection(
            module_name=self.parsed_module.module_name,
            file_path=self.parsed_module.file_path,
            flows=tuple(collector.collect(body) for collector, body in self.collectors),
        )


def compact_product_flow_projection(
    parsed_module: ParsedModule,
) -> CompactProductFlowModuleProjection:
    """Project one parsed module into AST-free closed-flow evidence."""
    return _ProductFlowCollection(parsed_module, _CompactFlowCollector).projection


def source_product_flow_projection(
    parsed_module: ParsedModule,
) -> SourceProductFlowProjection:
    """Collect task-local effect obligations alongside their actual compact flows.

    This is observation only. The source node-local effect authorities determine
    which intervals and operations are admitted; collection grants no safety.
    """
    collection = _ProductFlowCollection(parsed_module, _SourceFlowCollector)
    compact = collection.projection
    return SourceProductFlowProjection(
        collection.parsed_module,
        compact,
        tuple(
            site
            for collector, _ in collection.collectors
            for site in collector.source_evaluations
        ),
        tuple(
            site
            for collector, _ in collection.collectors
            for site in collector.source_operations
        ),
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
