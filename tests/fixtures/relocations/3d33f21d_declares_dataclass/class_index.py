# Extracted from 3d33f21d:nominal_refactor_advisor/class_index.py


class CompactDataclassFacet(ABC):
    """A class's dataclass facet: declared, or absent with empty answers."""

    runtime_declaration: DataclassRuntimeDeclaration | None
    fields: tuple[CompactDataclassFieldDeclaration, ...]
    failures: tuple[CompactProductDeclarationFailure, ...]

    @property
    def is_standard_dataclass(self) -> bool:
        return (
            self.runtime_declaration is not None
            and self.runtime_declaration.is_dataclass_decorator
        )

    @property
    @abstractmethod
    def declares_dataclass(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def require_declared(self, reason: str) -> CompactDataclassDeclaration:
        raise NotImplementedError


@dataclass(frozen=True)
class CompactDataclassDeclaration(CompactDataclassFacet):
    """One dataclass-like declaration, including fail-closed unknown semantics."""

    runtime_declaration: DataclassRuntimeDeclaration | None
    fields: tuple[CompactDataclassFieldDeclaration, ...]
    failures: tuple[CompactProductDeclarationFailure, ...] = ()

    @property
    def declares_dataclass(self) -> bool:
        return True

    def require_declared(self, reason: str) -> CompactDataclassDeclaration:
        return self


@dataclass(frozen=True)
class NoDataclassDeclaration(CompactDataclassFacet):
    """A class no dataclass-like decorator declares: no runtime, no fields."""

    runtime_declaration = None
    fields = ()
    failures = ()

    @property
    def declares_dataclass(self) -> bool:
        return False

    def require_declared(self, reason: str) -> CompactDataclassDeclaration:
        raise ValueError(reason)


@dataclass(frozen=True)
class CompactClassFamilyIndex(ClassDeclarationIndex[CompactIndexedClass]):
    """Repository inheritance graph reconstructed from compact declarations."""

    @cached_property
    def mro_authority(self) -> ClassMroAuthority:
        return ClassMroAuthority(self.classes_by_symbol)

    @classmethod
    def from_modules(
        cls,
        modules: tuple[ParsedModule, ...],
    ) -> Self:
        """Join complete class declarations without collecting unrelated analyses."""

        return CompactClassFamilyIndexBuilder(
            CompactModuleClassHeader.from_modules(modules)
        ).build()

    @classmethod
    def from_projection_groups(
        cls,
        projections_by_family: dict[
            type[CollectedFamily],
            Sequence[object],
        ],
    ) -> Self:
        """Build the class anchor declared by a compact multi-family join."""

        return CompactClassFamilyIndexBuilder(
            cast(
                Sequence[CompactModuleClassProjection],
                projections_by_family[CompactModuleClassProjectionFamily],
            )
        ).build()

    @classmethod
    def require(cls, context: object | None) -> Self:
        if not isinstance(context, cls):
            raise TypeError("shared compact class index is unavailable")
        return context

    def assignments_repeated_from_ancestors(
        self,
        class_symbol: str,
        assignment_names: Iterable[str],
    ) -> tuple[str, ...]:
        """Project direct assignments whose exact values already come from a base."""

        indexed_class = self.class_for(class_symbol)
        if indexed_class is None:
            return ()
        ancestors = tuple(
            ancestor
            for ancestor_symbol in self.ancestor_symbols(class_symbol)
            if (ancestor := self.class_for(ancestor_symbol)) is not None
        )
        return tuple(
            assignment_name
            for assignment_name in assignment_names
            if (expression := indexed_class.assignments_by_name.get(assignment_name))
            is not None
            if any(
                ancestor.assignments_by_name.get(assignment_name) == expression
                for ancestor in ancestors
            )
        )

    def direct_or_absent_value_declaration(
        self,
        class_symbol: str,
        member_name: str,
    ) -> CompactClassMemberDeclaration | None:
        """Require a direct value or proved absence throughout the native MRO."""

        owner = self.classes_by_symbol[class_symbol]
        direct = owner.direct_value_writes_by_name.get(member_name)
        if direct is not None:
            return direct
        resolution = self.mro_authority.resolve(class_symbol)
        if resolution.mro_type is None:
            raise ValueError("Class value lookup requires a closed MRO")
        inherited = any(
            member_name in ancestor.direct_value_writes_by_name
            for ancestor in resolution.mro_type.declarations[1:]
        )
        if inherited:
            raise ValueError("Class value is inherited rather than directly declared")
        return None

    @cached_property
    def product_authority_resolutions_by_symbol(
        self,
    ) -> dict[str, CompactProductAuthorityResolution]:
        """Compose every dataclass product through its exact direct-base chain."""

        lineages: dict[str, _CompactProductLineage] = {}

        def compose_lineage(
            class_symbol: str,
            pending_symbols: frozenset[str],
        ) -> _CompactProductLineage:
            if (lineage := lineages.get(class_symbol)) is not None:
                return lineage
            indexed_class = self.classes_by_symbol[class_symbol]
            declaration = indexed_class.dataclass_declaration.require_declared(
                "product lineage requires a dataclass declaration"
            )
            if class_symbol in pending_symbols:
                return _CompactProductLineage(
                    (),
                    (
                        self._product_authority_failure(
                            indexed_class,
                            CompactProductAuthorityViolation.CYCLIC_PRODUCT_LINEAGE,
                        ),
                    ),
                )
            failures = tuple(
                self._product_authority_failure(
                    indexed_class,
                    failure.violation,
                    line=failure.line,
                )
                for failure in declaration.failures
            )
            effective_fields: tuple[CompactProductField, ...] = ()
            direct_base_count = indexed_class.direct_base_count
            if (
                not indexed_class.base_references_are_complete
                or not indexed_class.product_base_bindings_are_exact
                or len(indexed_class.resolved_base_symbols) != direct_base_count
            ):
                failures = (
                    *failures,
                    self._product_authority_failure(
                        indexed_class,
                        CompactProductAuthorityViolation.INCOMPLETE_BASE_RESOLUTION,
                    ),
                )
            elif direct_base_count > 1:
                failures = (
                    *failures,
                    self._product_authority_failure(
                        indexed_class,
                        CompactProductAuthorityViolation.MULTIPLE_PRODUCT_BASES,
                    ),
                )
            elif direct_base_count == 1:
                base_symbol = indexed_class.resolved_base_symbols[0]
                base_class = self.classes_by_symbol[base_symbol]
                if not base_class.dataclass_declaration.declares_dataclass:
                    failures = (
                        *failures,
                        self._product_authority_failure(
                            indexed_class,
                            CompactProductAuthorityViolation.NON_DATACLASS_BASE,
                        ),
                    )
                else:
                    base_lineage = compose_lineage(
                        base_symbol,
                        pending_symbols | frozenset((class_symbol,)),
                    )
                    effective_fields = base_lineage.effective_fields
                    failures = (*failures, *base_lineage.failures)

            fields_by_name = {field.name: field for field in effective_fields}
            for field_declaration in declaration.fields:
                fields_by_name[field_declaration.name] = CompactProductField(
                    name=field_declaration.name,
                    role=field_declaration.role,
                    declaring_class_symbol=class_symbol,
                    file_path=indexed_class.file_path,
                    line=field_declaration.line,
                )
            lineage = _CompactProductLineage(
                effective_fields=tuple(fields_by_name.values()),
                failures=tuple(dict.fromkeys(failures)),
            )
            lineages[class_symbol] = lineage
            return lineage

        resolutions: dict[str, CompactProductAuthorityResolution] = {}
        for class_symbol, indexed_class in self.classes_by_symbol.items():
            if not indexed_class.dataclass_declaration.declares_dataclass:
                resolutions[class_symbol] = AbsentCompactProductAuthority(class_symbol)
                continue
            lineage = compose_lineage(class_symbol, frozenset())
            role_failures = tuple(
                CompactProductAuthorityFailure(
                    class_symbol=class_symbol,
                    file_path=field.file_path,
                    line=field.line,
                    violation=violation,
                )
                for field in lineage.effective_fields
                if (violation := field.role.authority_violation) is not None
            )
            failures = tuple(dict.fromkeys((*lineage.failures, *role_failures)))
            resolutions[class_symbol] = (
                OpenCompactProductAuthority(class_symbol, failures)
                if failures
                else ResolvedCompactProductAuthority(
                    class_symbol,
                    CompactProductAuthority(
                        class_symbol=class_symbol,
                        effective_fields=lineage.effective_fields,
                        file_path=indexed_class.file_path,
                        line=indexed_class.line,
                    ),
                )
            )
        return resolutions

    @staticmethod
    def _product_authority_failure(
        indexed_class: CompactIndexedClass,
        violation: CompactProductAuthorityViolation,
        *,
        line: int | None = None,
    ) -> CompactProductAuthorityFailure:
        return CompactProductAuthorityFailure(
            class_symbol=indexed_class.symbol,
            file_path=indexed_class.file_path,
            line=indexed_class.line if line is None else line,
            violation=violation,
        )
