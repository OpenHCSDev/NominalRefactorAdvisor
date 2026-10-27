"""Which scanned classes declare the same class-body declarations.

This is the one place NRA compares class bodies. A declaration's identity is
its `ClassBodyNormalForm` (kind, name, syntax and the declaration each free
name is bound to), projected once per class into the compact class index as a
digest. Names a star import may rebind are settled here, against the
repository's export contracts. The index inverts the identities into holders
and reports, for two classes, the declarations they share, the names both
declare with different answers, and what each keeps alone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import NamedTuple

from .class_index import (
    ClassBodyDeclarationDigest,
    ClassBodyDeclarationKind,
    CompactIndexedClass,
    CompactModuleClassHeader,
    CompactRepositoryPublicExposureIndex,
    StarDeferredName,
)


class ClassBodyDeclarationIdentity(NamedTuple):
    """What makes two class-body declarations the same declaration: the module-local
    digest (which covers kind and name) and how the repository bound its star-hidden names."""

    digest: str
    star_bindings: tuple[tuple[str, str], ...]


class ClassBodyDeclaration(NamedTuple):
    """One declaration of one class, with its identity when every name is bound."""

    declaration: ClassBodyDeclarationDigest
    identity: ClassBodyDeclarationIdentity | None

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def kind(self) -> ClassBodyDeclarationKind:
        return self.declaration.kind

    @property
    def line(self) -> int:
        return self.declaration.line


class SharedClassBodyDeclaration(NamedTuple):
    """One declaration two classes both make, with each one's line."""

    kind: ClassBodyDeclarationKind
    name: str
    first_line: int
    second_line: int


class DivergentClassBodyDeclaration(NamedTuple):
    """One name two classes both declare, answering differently or unprovably."""

    name: str
    first: ClassBodyDeclaration
    second: ClassBodyDeclaration


@dataclass(frozen=True)
class ClassBodyOverlap:
    """Two classes' bodies side by side, partitioned by declaration name."""

    first: CompactIndexedClass
    second: CompactIndexedClass
    shared: tuple[SharedClassBodyDeclaration, ...]
    divergent: tuple[DivergentClassBodyDeclaration, ...]
    first_only: tuple[ClassBodyDeclaration, ...]
    second_only: tuple[ClassBodyDeclaration, ...]

    @classmethod
    def of(
        cls,
        first: CompactIndexedClass,
        first_declarations: Mapping[str, ClassBodyDeclaration],
        second: CompactIndexedClass,
        second_declarations: Mapping[str, ClassBodyDeclaration],
    ) -> ClassBodyOverlap:
        shared: list[SharedClassBodyDeclaration] = []
        divergent: list[DivergentClassBodyDeclaration] = []
        for name, first_declaration in first_declarations.items():
            second_declaration = second_declarations.get(name)
            if second_declaration is None:
                continue
            if (
                first_declaration.identity is not None
                and first_declaration.identity == second_declaration.identity
            ):
                shared.append(
                    SharedClassBodyDeclaration(
                        first_declaration.kind,
                        name,
                        first_declaration.line,
                        second_declaration.line,
                    )
                )
            else:
                divergent.append(
                    DivergentClassBodyDeclaration(
                        name, first_declaration, second_declaration
                    )
                )
        return cls(
            first=first,
            second=second,
            shared=tuple(shared),
            divergent=tuple(divergent),
            first_only=tuple(
                declaration
                for name, declaration in first_declarations.items()
                if name not in second_declarations
            ),
            second_only=tuple(
                declaration
                for name, declaration in second_declarations.items()
                if name not in first_declarations
            ),
        )

    @property
    def first_is_contained(self) -> bool:
        """Every declaration of the first class is one the second makes identically."""

        return bool(self.shared) and not self.divergent and not self.first_only

    @property
    def second_is_contained(self) -> bool:
        return bool(self.shared) and not self.divergent and not self.second_only


@dataclass(frozen=True)
class ClassBodyIndex:
    """Holders of each class-body declaration across one scan's classes.

    `holder_limit` bounds the join: a declaration more than that many classes
    make (a shared `__slots__ = ()`, say) still counts as shared between two
    classes, but does not by itself pair them. The bound is a declared
    resource limit, reported as `unpaired_declarations`.
    """

    projections: tuple[CompactModuleClassHeader, ...]
    holder_limit: int = 32

    @classmethod
    def from_projections(
        cls,
        projections: Sequence[CompactModuleClassHeader],
        *,
        holder_limit: int = 32,
    ) -> ClassBodyIndex:
        return cls(tuple(projections), holder_limit)

    @cached_property
    def exposure(self) -> CompactRepositoryPublicExposureIndex:
        return CompactRepositoryPublicExposureIndex(self.projections)

    @cached_property
    def classes_by_symbol(self) -> Mapping[str, CompactIndexedClass]:
        return {
            indexed_class.symbol: indexed_class
            for projection in sorted(
                self.projections, key=lambda projection: projection.module_name
            )
            for indexed_class in projection.classes
        }

    @cached_property
    def declarations_by_symbol(
        self,
    ) -> Mapping[str, Mapping[str, ClassBodyDeclaration]]:
        """Each class's effective declarations: a later statement replaces an earlier one."""

        return {
            symbol: {
                declaration.name: ClassBodyDeclaration(
                    declaration,
                    self._identity(indexed_class.module_name, declaration),
                )
                for declaration in indexed_class.body_declarations
            }
            for symbol, indexed_class in self.classes_by_symbol.items()
        }

    def declarations_of(self, symbol: str) -> Mapping[str, ClassBodyDeclaration]:
        return self.declarations_by_symbol[symbol]

    @cached_property
    def holders(self) -> Mapping[ClassBodyDeclarationIdentity, tuple[str, ...]]:
        holders: dict[ClassBodyDeclarationIdentity, list[str]] = defaultdict(list)
        for symbol, declarations in self.declarations_by_symbol.items():
            for declaration in declarations.values():
                if declaration.identity is not None:
                    holders[declaration.identity].append(symbol)
        return {identity: tuple(symbols) for identity, symbols in holders.items()}

    @cached_property
    def unpaired_declarations(self) -> tuple[ClassBodyDeclarationIdentity, ...]:
        return tuple(
            sorted(
                identity
                for identity, symbols in self.holders.items()
                if len(symbols) > self.holder_limit
            )
        )

    def overlap(self, first: str, second: str) -> ClassBodyOverlap:
        return ClassBodyOverlap.of(
            self.classes_by_symbol[first],
            self.declarations_by_symbol[first],
            self.classes_by_symbol[second],
            self.declarations_by_symbol[second],
        )

    @cached_property
    def overlaps(self) -> tuple[ClassBodyOverlap, ...]:
        """Every pair of classes sharing at least one pairing declaration."""

        pairs: set[tuple[str, str]] = set()
        for symbols in self.holders.values():
            if len(symbols) > self.holder_limit:
                continue
            pairs.update(combinations(sorted(symbols), 2))
        return tuple(self.overlap(first, second) for first, second in sorted(pairs))

    def overlaps_among(self, symbols: Iterable[str]) -> tuple[ClassBodyOverlap, ...]:
        wanted = frozenset(symbols)
        return tuple(
            overlap
            for overlap in self.overlaps
            if overlap.first.symbol in wanted and overlap.second.symbol in wanted
        )

    def _identity(
        self, module_name: str, declaration: ClassBodyDeclarationDigest
    ) -> ClassBodyDeclarationIdentity | None:
        if declaration.unbound:
            return None
        star_bindings: list[tuple[str, str]] = []
        for deferred in declaration.deferred:
            qualified_name = self._star_binding(module_name, deferred)
            if qualified_name is None:
                return None
            star_bindings.append((deferred.name, qualified_name))
        return ClassBodyDeclarationIdentity(declaration.digest, tuple(star_bindings))

    def _star_binding(self, module_name: str, deferred: StarDeferredName) -> str | None:
        """What a star-hidden name is bound to, when the repository's exports decide it.

        No star import exposing the name leaves the module's own binding; the
        last star import that exposes it rebinds it to that module's binding;
        an origin whose exports are unknown decides nothing.
        """

        if self.exposure.star_imports_exclude(module_name, deferred.name):
            return deferred.unshadowed
        if not self.exposure.contains_module(module_name):
            return None
        for origin in reversed(self.exposure.star_import_origins_for(module_name)):
            if origin.module_name is None:
                return None
            exposure = self.exposure.exposure_for(origin.module_name, deferred.name)
            if exposure.proves_public_exposure:
                origin_projection = self.exposure.projections_by_module_name[
                    origin.module_name
                ]
                return dict(origin_projection.import_aliases).get(
                    deferred.name, f"{origin.module_name}.{deferred.name}"
                )
            if exposure.introduces_uncertainty:
                return None
        return deferred.unshadowed


__all__ = (
    "ClassBodyDeclaration",
    "ClassBodyDeclarationIdentity",
    "ClassBodyIndex",
    "ClassBodyOverlap",
    "DivergentClassBodyDeclaration",
    "SharedClassBodyDeclaration",
)
