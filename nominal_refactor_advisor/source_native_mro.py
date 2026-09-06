"""Project source hierarchies across authenticated native class boundaries."""

from __future__ import annotations

import ast
from dataclasses import (
    dataclass,
    field,
    replace,
)
from functools import cached_property
from graphlib import CycleError, TopologicalSorter

from .class_index import (
    CLASS_METHOD_OWNERSHIP_HOOK_NAMES,
    ClassSymbolResolutionAuthority,
    IndexedClass,
    RepositoryModuleBindingProof,
)
from .class_member_lookup import ClassMemberLookupProof, ClassNamespaceDelta
from .class_mro import DeclarationMroType, NativeMroBase
from .collection_algebra import UniqueIdentityIndexAuthority
from .codemod_selection_context import CodemodSelectorContext
from .native_class_mro import NativeClassMroDeclaration
from .native_declarations import (
    NativeDeclaration,
    QualifiedDeclaration,
)
from .source_geometry import SourceByteSpan


@dataclass(frozen=True)
class NativeClassBaseSubstitution:
    """One source-owned base occurrence replaced by a native declaration."""

    owner: IndexedClass
    base: ast.expr
    replacement: NativeClassMroDeclaration

    def replaces(self, owner: IndexedClass, base: ast.expr) -> bool:
        return owner.symbol == self.owner.symbol and SourceByteSpan.require_node(
            base
        ) == SourceByteSpan.require_node(self.base)


@dataclass(frozen=True)
class SourceNativeClassMro:
    """Close reachable source bases, then delegate all precedence to native C3."""

    context: CodemodSelectorContext
    native_roots: tuple[type, ...] = tuple(base.python_type for base in NativeMroBase)

    substitution: NativeClassBaseSubstitution | None = None

    _mro_types: dict[str, type] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @cached_property
    def native_declarations(self) -> dict[str, NativeClassMroDeclaration]:
        return UniqueIdentityIndexAuthority.declarations_by_handle(
            (
                NativeClassMroDeclaration(base)
                for root in self.native_roots
                for base in root.__mro__
            ),
            lambda declaration: declaration.qualified_name,
        )

    @cached_property
    def bindings(self) -> RepositoryModuleBindingProof:
        return RepositoryModuleBindingProof(self.context.parsed_modules)

    def required_native_type(self, native: NativeClassMroDeclaration) -> type:
        # Every source declaration supplied for this native island must agree.
        for owner in native.declaration.__mro__:
            declaration = NativeDeclaration(owner)
            targets = self.context.source_index.targets_matching_repository_symbol(
                declaration.qualified_name
            )
            if targets:
                if len(targets) != 1:
                    raise ValueError("Native MRO has ambiguous source authority")
                declaration.require_source_matches(
                    self.context.ast_target_nodes_by_id[targets[0].target_id]
                )
        return native.mro_type

    def required_base_symbol(self, owner: IndexedClass, base: ast.expr) -> str:
        reference = ClassSymbolResolutionAuthority.reference_node(base)
        module = self.context.parsed_module_for_source_path(owner.file_path)
        witness = self.bindings.reference_or_builtin_witness_at(
            module,
            reference,
            line=owner.line,
        )
        if witness is None:
            raise ValueError("MRO base has no closed declaration binding")
        if isinstance(base, ast.Subscript):
            native = self.native_declarations.get(witness.qualified_name)
            if native is None:
                raise ValueError("Source generic base application remains unproved")
            native.require_generic_origin()
        elif not isinstance(base, (ast.Name, ast.Attribute)):
            raise ValueError("Dynamic MRO base application remains unproved")
        return witness.qualified_name

    def for_source_class(
        self, source_class: IndexedClass
    ) -> DeclarationMroType[QualifiedDeclaration]:
        classes = self.context.required_class_family_index.classes_by_symbol
        dependencies: dict[str, tuple[str, ...]] = {}
        projected = self._mro_types
        pending = [source_class.symbol]
        while pending:
            symbol = pending.pop()
            if symbol in dependencies or symbol in projected:
                continue
            native = self.native_declarations.get(symbol)
            if native is not None:
                projected[symbol] = self.required_native_type(native)
                continue
            owner = classes.get(symbol)
            if owner is None:
                raise ValueError(f"MRO source authority {symbol!r} is unavailable")
            namespace = owner.member_binding_names
            owner.namespace_execution.require_closed(
                self.bindings,
                self.context.parsed_module_for_source_path(owner.file_path),
                owner.node,
            )
            if (
                owner.node.keywords
                or not owner.class_decorators_are_promotion_safe
                or (not CLASS_METHOD_OWNERSHIP_HOOK_NAMES.isdisjoint(namespace))
            ):
                raise ValueError("Source class creation remains unproved")
            bases = []
            for base in owner.node.bases:
                base_symbol = self.required_base_symbol(owner, base)
                if self.substitution is not None and self.substitution.replaces(
                    owner, base
                ):
                    if isinstance(base, ast.Subscript):
                        self.substitution.replacement.require_generic_origin()
                    base_symbol = self.substitution.replacement.qualified_name
                bases.append(base_symbol)
            dependencies[symbol] = tuple(bases)
            pending.extend(bases)
        try:
            # Topological order only schedules construction. Python owns MRO order.
            for symbol in TopologicalSorter(dependencies).static_order():
                if symbol not in projected:
                    projected[symbol] = DeclarationMroType.from_declaration(
                        classes[symbol],
                        tuple(projected[base] for base in dependencies[symbol]),
                    )
        except (CycleError, TypeError) as error:
            raise ValueError("Replacement has no consistent native C3 order") from error
        result = projected[source_class.symbol]
        if not isinstance(result, DeclarationMroType):
            raise ValueError("Replacement target is not a source class")
        return result

    def require_inherited_method(
        self,
        substitution: NativeClassBaseSubstitution,
        method_name: str,
    ) -> None:
        projected = replace(self, substitution=substitution).for_source_class(
            substitution.owner
        )
        expected = substitution.replacement.declaration
        expected_owner = NativeClassMroDeclaration(
            next(owner for owner in expected.__mro__ if method_name in vars(owner))
        )
        ClassMemberLookupProof(
            projected,
            (
                ClassNamespaceDelta(
                    substitution.owner, removed_names=frozenset((method_name,))
                ),
            ),
        ).require_owner(method_name, expected_owner)
