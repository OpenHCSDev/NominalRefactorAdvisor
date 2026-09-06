"""Native subscription families own their argument-effect obligations."""

from __future__ import annotations

import ast
import builtins
from abc import ABC, abstractmethod
from types import BuiltinFunctionType, ClassMethodDescriptorType
from typing import ClassVar

from .native_declarations import NativeDeclaration
from .native_reference import (
    NativeArgumentEvidence as NativeArgumentEvidence,
    NativeReferenceEnvironment,
    ScopedNativeReference,
)
from .semantic_match import loaded_concrete_nominal_descendants

NATIVE_BUILTIN_DECLARATIONS = tuple(
    NativeDeclaration(declaration)
    for declaration in vars(builtins).values()
    if isinstance(declaration, (type, BuiltinFunctionType))
)


class NativeSubscriptionAuthority(ABC):
    native_declarations: ClassVar[tuple[NativeDeclaration, ...]]

    @classmethod
    def for_reference(
        cls, reference: ScopedNativeReference, environment: NativeReferenceEnvironment
    ) -> type[NativeSubscriptionAuthority]:
        witness = reference.require_binding(environment)
        matches = tuple(
            authority
            for authority in loaded_concrete_nominal_descendants(cls)
            if any(
                declaration.qualified_name == witness.qualified_name
                for declaration in authority.native_declarations
            )
        )
        if len(matches) != 1:
            raise ValueError(
                f"Class namespace execution at line {reference.node.lineno} has no unique subscription proof"
            )
        return matches[0]

    @classmethod
    @abstractmethod
    def require_argument(
        cls,
        argument: NativeArgumentEvidence,
        environment: NativeReferenceEnvironment,
    ) -> None:
        raise NotImplementedError


class BuiltinGenericAliasSubscription(NativeSubscriptionAuthority):
    native_declarations = tuple(
        native
        for native in NATIVE_BUILTIN_DECLARATIONS
        if isinstance(native.declaration, type)
        and "__class_getitem__" in vars(native.declaration)
        and isinstance(
            vars(native.declaration)["__class_getitem__"], ClassMethodDescriptorType
        )
    )

    @classmethod
    def require_argument(
        cls,
        argument: NativeArgumentEvidence,
        environment: NativeReferenceEnvironment,
    ) -> None:
        # Native GenericAlias construction stores, rather than hashes, arguments.
        pass


class ClassVariableSubscription(NativeSubscriptionAuthority):
    native_declarations = (NativeDeclaration(ClassVar),)

    @classmethod
    def require_argument(
        cls,
        argument: NativeArgumentEvidence,
        environment: NativeReferenceEnvironment,
    ) -> None:
        NativeArgumentInspection(argument, environment).visit(argument.node)


class NativeArgumentInspection(ast.NodeVisitor):
    """Close hashing and metadata reads without evaluating repository objects."""

    def __init__(
        self,
        argument: NativeArgumentEvidence,
        environment: NativeReferenceEnvironment,
    ) -> None:
        self.argument = argument
        self.environment = environment

    def generic_visit(self, node: ast.AST) -> None:
        raise ValueError("Native argument has unproved hashing or metadata effects")

    def visit_Name(self, node: ast.Name | ast.Attribute) -> None:
        self.argument.required_reference(node).require_native(
            self.environment, NATIVE_BUILTIN_DECLARATIONS
        )

    visit_Attribute = visit_Name

    def visit_Constant(self, node: ast.Constant) -> None:
        ast.literal_eval(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # A newly created Python function has native hash and metadata access.
        # Its defaults are evaluated separately by the lexical traversal.
        pass

    def visit_Tuple(self, node: ast.Tuple | ast.List) -> None:
        for element in node.elts:
            self.visit(element)

    visit_List = visit_Tuple

    def visit_Subscript(self, node: ast.Subscript) -> None:
        NativeSubscriptionAuthority.for_reference(
            self.argument.required_reference(node.value), self.environment
        )
        self.visit(node.slice)
