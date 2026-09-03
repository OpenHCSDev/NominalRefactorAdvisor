"""Proof objects for behavior projected through an external type-keyed family."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property

from .annotation_semantics import CLASSVAR_ANNOTATION_AUTHORITY
from .class_index import (
    ATTRIBUTE_CHAIN_AUTHORITY,
    CompactClassFamilyIndex,
    CompactClassReferenceResolver,
    CompactIndexedClass,
    CompactModuleClassProjection,
)
from .collection_algebra import sorted_tuple
from .models import SourceLocation


@dataclass(frozen=True)
class TypeKeyedBehaviorBinding:
    """One projection leaf bound injectively to its nominal target type."""

    projection_class: CompactIndexedClass
    target_class: CompactIndexedClass

    @cached_property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(
                self.projection_class.file_path,
                self.projection_class.line,
                self.projection_class.symbol,
            ),
            SourceLocation(
                self.target_class.file_path,
                self.target_class.line,
                self.target_class.symbol,
            ),
        )


@dataclass(frozen=True)
class TypeKeyedBehaviorProjectionComponent:
    """External behavior family whose type keys duplicate a nominal hierarchy."""

    projection_root: CompactIndexedClass
    target_root: CompactIndexedClass
    key_attribute_name: str
    behavior_method_names: tuple[str, ...]
    bindings: tuple[TypeKeyedBehaviorBinding, ...]

    def __post_init__(self) -> None:
        if len(self.bindings) < 2:
            raise ValueError(
                "type-keyed behavior projection requires multiple bindings"
            )
        if not self.behavior_method_names:
            raise ValueError("type-keyed behavior projection requires shared behavior")
        if len({binding.projection_class.symbol for binding in self.bindings}) != len(
            self.bindings
        ):
            raise ValueError("projection leaves must be unique")
        if len({binding.target_class.symbol for binding in self.bindings}) != len(
            self.bindings
        ):
            raise ValueError("target types must be unique")

    @cached_property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(
                self.projection_root.file_path,
                self.projection_root.line,
                self.projection_root.symbol,
            ),
            *(
                location
                for binding in self.bindings
                for location in binding.evidence_locations
            ),
        )

    @property
    def authority_evidence(self) -> SourceLocation:
        return SourceLocation(
            self.target_root.file_path,
            self.target_root.line,
            self.target_root.symbol,
        )


@dataclass(frozen=True)
class TypeKeyedBehaviorProjectionComponentBuilder:
    """Prove that an AutoRegister family duplicates type-owned behavior."""

    class_index: CompactClassFamilyIndex
    class_reference_resolver: CompactClassReferenceResolver

    @classmethod
    def from_projections(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        class_index: CompactClassFamilyIndex,
    ) -> "TypeKeyedBehaviorProjectionComponentBuilder":
        return cls(
            class_index=class_index,
            class_reference_resolver=CompactClassReferenceResolver.from_index(
                projections,
                class_index,
            ),
        )

    def proven_components(self) -> tuple[TypeKeyedBehaviorProjectionComponent, ...]:
        return sorted_tuple(
            (
                component
                for projection_root in self.class_index.classes_by_symbol.values()
                if projection_root.declares_autoregister_meta
                if (component := self._component_for_root(projection_root)) is not None
            ),
            key=lambda component: component.projection_root.symbol,
        )

    def component_for_projection_root(
        self,
        projection_root_symbol: str,
    ) -> TypeKeyedBehaviorProjectionComponent | None:
        projection_root = self.class_index.class_for(projection_root_symbol)
        if projection_root is None or not projection_root.declares_autoregister_meta:
            return None
        return self._component_for_root(projection_root)

    def _component_for_root(
        self,
        projection_root: CompactIndexedClass,
    ) -> TypeKeyedBehaviorProjectionComponent | None:
        key_attribute_name = projection_root.autoregister_registry_key_attr_name
        if key_attribute_name is None:
            return None
        projection_leaves = tuple(
            descendant
            for symbol in self.class_index.descendant_symbols(projection_root.symbol)
            if (descendant := self.class_index.class_for(symbol)) is not None
            if descendant.assignments_by_name.get(key_attribute_name) is not None
        )
        if len(projection_leaves) < 2:
            return None
        bindings = self._resolved_bindings(
            projection_leaves,
            key_attribute_name=key_attribute_name,
        )
        if bindings is None:
            return None
        target_root = self._declared_target_root(
            projection_root,
            key_attribute_name=key_attribute_name,
        )
        if target_root is None or target_root.symbol == projection_root.symbol:
            return None
        target_symbols = frozenset(binding.target_class.symbol for binding in bindings)
        if target_root.symbol not in target_symbols:
            return None
        if any(
            binding.target_class.symbol != target_root.symbol
            and target_root.symbol
            not in self.class_index.ancestor_symbols(binding.target_class.symbol)
            and binding.target_class.base_resolution_is_complete
            for binding in bindings
        ):
            return None
        behavior_method_names = self._behavior_method_names(
            projection_root,
            bindings,
        )
        if not behavior_method_names:
            return None
        return TypeKeyedBehaviorProjectionComponent(
            projection_root=projection_root,
            target_root=target_root,
            key_attribute_name=key_attribute_name,
            behavior_method_names=behavior_method_names,
            bindings=bindings,
        )

    def _resolved_bindings(
        self,
        projection_leaves: tuple[CompactIndexedClass, ...],
        *,
        key_attribute_name: str,
    ) -> tuple[TypeKeyedBehaviorBinding, ...] | None:
        bindings: list[TypeKeyedBehaviorBinding] = []
        for projection_leaf in projection_leaves:
            expression = projection_leaf.assignments_by_name[key_attribute_name]
            if expression is None:
                return None
            reference_parts = self._reference_parts(expression)
            if reference_parts is None:
                return None
            target_symbol = self.class_reference_resolver.symbol_for(
                module_name=projection_leaf.module_name,
                reference_parts=reference_parts,
            )
            if target_symbol is None:
                return None
            target_class = self.class_index.class_for(target_symbol)
            if target_class is None:
                return None
            bindings.append(TypeKeyedBehaviorBinding(projection_leaf, target_class))
        if len({binding.target_class.symbol for binding in bindings}) != len(bindings):
            return None
        return sorted_tuple(
            bindings,
            key=lambda binding: binding.projection_class.symbol,
        )

    @staticmethod
    def _reference_parts(expression: str) -> tuple[str, ...] | None:
        try:
            node = ast.parse(expression, mode="eval").body
        except SyntaxError:
            return None
        return ATTRIBUTE_CHAIN_AUTHORITY.project(node)

    def _declared_target_root(
        self,
        projection_root: CompactIndexedClass,
        *,
        key_attribute_name: str,
    ) -> CompactIndexedClass | None:
        declaration = projection_root.direct_members_by_name.get(key_attribute_name)
        if declaration is None or declaration.annotation_expression is None:
            return None
        reference_parts = self._type_class_reference_parts(
            declaration.annotation_expression
        )
        if reference_parts is None:
            return None
        target_symbol = self.class_reference_resolver.symbol_for(
            module_name=projection_root.module_name,
            reference_parts=reference_parts,
        )
        return (
            None if target_symbol is None else self.class_index.class_for(target_symbol)
        )

    @staticmethod
    def _type_class_reference_parts(
        annotation_expression: str,
    ) -> tuple[str, ...] | None:
        try:
            annotation = ast.parse(annotation_expression, mode="eval").body
        except SyntaxError:
            return None
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                annotation = ast.parse(annotation.value, mode="eval").body
            except SyntaxError:
                return None
        if isinstance(
            annotation, ast.Subscript
        ) and CLASSVAR_ANNOTATION_AUTHORITY.matches(annotation):
            annotation = annotation.slice
        if not (
            isinstance(annotation, ast.Subscript)
            and (parts := ATTRIBUTE_CHAIN_AUTHORITY.project(annotation.value))
            and parts[-1] == "type"
        ):
            return None
        return ATTRIBUTE_CHAIN_AUTHORITY.project(annotation.slice)

    @staticmethod
    def _behavior_method_names(
        projection_root: CompactIndexedClass,
        bindings: tuple[TypeKeyedBehaviorBinding, ...],
    ) -> tuple[str, ...]:
        leaf_method_sets = tuple(
            frozenset(binding.projection_class.method_names) for binding in bindings
        )
        common_leaf_methods = set.intersection(
            *(set(names) for names in leaf_method_sets)
        )
        registry_projection_names = frozenset(
            projection_root.autoregister_registry_projection_names
        )
        return sorted_tuple(
            method_name
            for method_name in common_leaf_methods.intersection(
                projection_root.method_names
            )
            if not method_name.startswith("__")
            and method_name not in registry_projection_names
            and all(
                method_name not in binding.target_class.method_names
                for binding in bindings
            )
        )
