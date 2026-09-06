"""Member ownership projected from native C3 and declared namespace changes."""

from dataclasses import dataclass
from functools import cached_property

from .class_mro import DeclarationMroType
from .collection_algebra import UniqueIdentityIndexAuthority
from .native_declarations import ClassNamespaceDeclaration, QualifiedDeclaration
from .native_class_mro import NativeClassMroDeclaration


@dataclass(frozen=True)
class ClassNamespaceDelta:
    """An authored change projected over one declaration's existing bindings."""

    declaration: ClassNamespaceDeclaration
    added_names: frozenset[str] = frozenset()
    removed_names: frozenset[str] = frozenset()

    @property
    def member_binding_names(self) -> frozenset[str]:
        return (
            self.declaration.member_binding_names - self.removed_names
        ) | self.added_names


@dataclass(frozen=True)
class ClassMemberLookupProof:
    """Select a member owner without reconstructing or approximating C3 order."""

    mro_type: DeclarationMroType[QualifiedDeclaration]
    namespace_changes: tuple[ClassNamespaceDelta, ...] = ()

    @cached_property
    def namespaces(self) -> tuple[ClassNamespaceDelta, ...]:
        changes = UniqueIdentityIndexAuthority.declarations_by_handle(
            self.namespace_changes, lambda change: change.declaration.qualified_name
        )
        namespaces = []
        for owner in self.mro_type.__mro__:
            declaration = (
                owner.declaration
                if isinstance(owner, DeclarationMroType)
                else NativeClassMroDeclaration(owner)
            )
            if not isinstance(declaration, ClassNamespaceDeclaration):
                raise ValueError("MRO class namespace remains unproved")
            projection = changes.get(declaration.qualified_name)
            if projection is None:
                projection = ClassNamespaceDelta(declaration)
            if projection.declaration != declaration:
                raise ValueError(
                    "Namespace change has a different declaration identity"
                )
            namespaces.append(projection)
        return tuple(namespaces)

    def owner_of(self, member_name: str) -> ClassNamespaceDeclaration | None:
        for namespace in self.namespaces:
            if member_name in namespace.member_binding_names:
                return namespace.declaration
        return None

    def require_owner(
        self, member_name: str, expected: ClassNamespaceDeclaration
    ) -> None:
        declaration = self.owner_of(member_name)
        if declaration is None:
            raise ValueError(
                f"Replacement member ownership of {member_name!r} remains unproved"
            )
        if declaration != expected:
            raise ValueError(
                f"Replacement selects {declaration.qualified_name!r}, "
                f"a competing member authority for {member_name!r}"
            )
