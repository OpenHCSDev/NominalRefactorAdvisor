# Extracted from 580b14d4:nominal_refactor_advisor/class_mro.py


class UnboundReference(ValueError):
    """A nominal reference no declaration-time binding roots."""


@dataclass(frozen=True)
class ClassMroAuthority:
    """Lazily derive each required hierarchy from the authoritative class graph."""

    classes_by_symbol: Mapping[str, CompactIndexedClass]
    _resolutions: dict[str, ClassMroResolution] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def resolve(self, symbol: str) -> ClassMroResolution:
        return self._resolve(symbol, frozenset())

    def _resolve(self, symbol: str, pending: frozenset[str]) -> ClassMroResolution:
        if symbol in self._resolutions:
            return self._resolutions[symbol]
        if symbol in pending:
            return CyclicHierarchy(symbol)
        resolution = self._derive(symbol, pending | {symbol})
        self._resolutions[symbol] = resolution
        return resolution

    def lookup_prefix(self, symbol: str) -> tuple[CompactIndexedClass, ...]:
        """Known declaration prefix before an opaque or branching base.

        A single direct base guarantees the next declaration even when a later
        ancestor prevents closing the full C3 order. Never guess past that point.
        This does not strengthen an OpenClassMro into a runtime proof.
        """
        return self.resolve(symbol).known_prefix(self, symbol)

    def single_base_prefix(self, symbol: str) -> tuple[CompactIndexedClass, ...]:
        """The declarations reached from ``symbol`` through single static bases."""
        prefix = []
        pending = set()
        while symbol not in pending:
            pending.add(symbol)
            declaration = self.classes_by_symbol.get(symbol)
            if declaration is None:
                break
            prefix.append(declaration)
            if not self._creation_is_static(declaration):
                break
            if not declaration.base_references_are_complete:
                break
            if (
                declaration.direct_base_count != 1
                or len(declaration.resolved_base_symbols) != 1
            ):
                break
            symbol = declaration.resolved_base_symbols[0]
        return tuple(prefix)

    @staticmethod
    def _creation_is_static(declaration: CompactIndexedClass) -> bool:
        return (
            declaration.mro_bases_are_static
            and not declaration.class_keyword_names
            and declaration.class_decorators_are_promotion_safe
            and not declaration.has_class_creation_hook
        )

    def _derive(self, symbol: str, pending: frozenset[str]) -> ClassMroResolution:
        declaration = self.classes_by_symbol.get(symbol)
        if declaration is None:
            return MissingDeclaration(symbol)
        if not self._creation_is_static(declaration):
            return DynamicClassDeclaration(symbol)
        native_bases = tuple(
            (
                None
                if reference.qualified_name in self.classes_by_symbol
                else NativeMroBase.for_qualified_name(reference.qualified_name)
            )
            for reference in declaration.base_references
        )
        if (
            not declaration.base_references_are_complete
            or len(declaration.resolved_base_symbols)
            != sum(base is None for base in native_bases)
        ):
            return UnresolvedBases(symbol)
        try:
            for reference in declaration.base_references:
                reference.require_bound_name()
        except UnboundReference:
            return UnresolvedBases(symbol)
        source_bases = iter(declaration.resolved_base_symbols)
        bases = []
        for native_base in native_bases:
            if native_base is not None:
                bases.append(native_base.python_type)
                continue
            base_symbol = next(source_bases)
            resolution = self._resolve(base_symbol, pending)
            if resolution.mro_type is None:
                return resolution
            bases.append(resolution.mro_type)
        try:
            projected = DeclarationMroType.from_declaration(declaration, tuple(bases))
        except TypeError:
            return InconsistentHierarchy(symbol)
        return ResolvedClassMro(symbol, projected)
