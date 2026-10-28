# Extracted from 8eef04ab:nominal_refactor_advisor/class_index.py


class CompactNominalReference(ABC):
    """One source reference; its subtype says whether a declaration-time binding roots it."""

    source_parts: tuple[str, ...]

    @staticmethod
    def bound_as(
        parts: tuple[str, ...], binding: CompactNominalBinding | None
    ) -> CompactNominalReference:
        """The binding lookup's answer, read once, picks the reference kind."""

        if binding is None:
            return UnboundNominalReference(parts)
        return BoundNominalReference(parts, binding)

    @abstractmethod
    def fold_binding(
        self,
        bound: Callable[[CompactNominalBinding], _ReferenceFoldT],
        unbound: Callable[[], _ReferenceFoldT],
    ) -> _ReferenceFoldT:
        raise NotImplementedError

    @property
    def resolved_parts(self) -> tuple[str, ...]:
        return self.fold_binding(
            lambda binding: (
                *binding.qualified_name.split("."),
                *self.source_parts[1:],
            ),
            lambda: self.source_parts,
        )

    @property
    def qualified_name(self) -> str:
        return ".".join(self.resolved_parts)

    @property
    def simple_name(self) -> str:
        return self.resolved_parts[-1]

    @property
    def binds_exactly(self) -> bool:
        """Whether a declaration-time binding fixes what this reference names."""

        return self.fold_binding(lambda _binding: True, lambda: False)

    def qualified_name_or(self, fallback: Callable[[], str | None]) -> str | None:
        return self.fold_binding(lambda _binding: self.qualified_name, fallback)

    def shares_binding_with(self, other: CompactNominalReference) -> bool:
        return self.fold_binding(
            lambda _binding: other.fold_binding(
                lambda _other_binding: other.qualified_name == self.qualified_name,
                lambda: False,
            ),
            lambda: False,
        )


@dataclass(frozen=True)
class BoundNominalReference(CompactNominalReference):
    source_parts: tuple[str, ...]
    root_binding: CompactNominalBinding

    def fold_binding(
        self,
        bound: Callable[[CompactNominalBinding], _ReferenceFoldT],
        unbound: Callable[[], _ReferenceFoldT],
    ) -> _ReferenceFoldT:
        return bound(self.root_binding)


@dataclass(frozen=True)
class UnboundNominalReference(CompactNominalReference):
    """A reference no binding in scope roots: only its source spelling is known."""

    source_parts: tuple[str, ...]

    def fold_binding(
        self,
        bound: Callable[[CompactNominalBinding], _ReferenceFoldT],
        unbound: Callable[[], _ReferenceFoldT],
    ) -> _ReferenceFoldT:
        return unbound()
