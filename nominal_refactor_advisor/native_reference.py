"""Use-point lexical evidence for references to known native declarations."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from .ast_projection import AstExpressionProjection
from .lexical_scopes import LexicalNameResolution, LexicalScopeContext
from .native_declarations import NativeDeclaration

if TYPE_CHECKING:
    from .ast_tools import ParsedModule
    from .class_index import ModuleNominalBindingView, ModuleNominalBindingWitness


@dataclass(frozen=True)
class NativeReferenceEnvironment:
    bindings: ModuleNominalBindingView
    module: ParsedModule
    definition_line: int


@dataclass(frozen=True)
class ScopedNativeReference:
    node: ast.expr
    resolution: LexicalNameResolution

    @classmethod
    def from_scope(
        cls, node: ast.expr, scope: LexicalScopeContext
    ) -> ScopedNativeReference:
        chain = AstExpressionProjection.attribute_chain(node)
        return cls(
            node,
            (
                LexicalNameResolution.UNPROVED
                if chain is None
                else scope._resolve_name(chain[0])
            ),
        )

    def require_binding(
        self, environment: NativeReferenceEnvironment
    ) -> ModuleNominalBindingWitness:
        if self.resolution is not LexicalNameResolution.EXTERNAL:
            raise ValueError(
                f"Class namespace execution at line {self.node.lineno} has no external binding proof"
            )
        witness = environment.bindings.reference_or_builtin_witness_at(
            environment.module, self.node, line=environment.definition_line
        )
        if witness is None:
            raise ValueError(
                f"Class namespace execution at line {self.node.lineno} remains unproved"
            )
        return witness

    def require_native(
        self,
        environment: NativeReferenceEnvironment,
        declarations: tuple[NativeDeclaration, ...],
    ) -> NativeDeclaration:
        witness = self.require_binding(environment)
        for declaration in declarations:
            if declaration.qualified_name == witness.qualified_name:
                return declaration
        raise ValueError(
            f"Class namespace execution at line {self.node.lineno} remains unproved"
        )


@dataclass(frozen=True)
class NativeArgumentEvidence:
    node: ast.expr
    references: tuple[ScopedNativeReference, ...]

    @classmethod
    def from_scope(
        cls, node: ast.expr, scope: LexicalScopeContext
    ) -> NativeArgumentEvidence:
        return cls(
            node,
            tuple(
                ScopedNativeReference.from_scope(reference, scope)
                for reference in ast.walk(node)
                if isinstance(reference, (ast.Name, ast.Attribute))
            ),
        )

    @cached_property
    def references_by_node(self) -> dict[ast.expr, ScopedNativeReference]:
        return {reference.node: reference for reference in self.references}

    def required_reference(self, node: ast.expr) -> ScopedNativeReference:
        reference = self.references_by_node.get(node)
        if reference is None:
            raise ValueError("Native argument has no closed declaration reference")
        return reference
