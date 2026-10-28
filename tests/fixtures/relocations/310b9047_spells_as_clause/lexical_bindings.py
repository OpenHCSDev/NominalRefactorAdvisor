# Extracted from 310b9047:nominal_refactor_advisor/lexical_bindings.py


@dataclass(frozen=True)
class ImportAlias:
    """One import alias as spelled; its statement kind owns the name it binds."""

    name: str
    asname: str | None

    @property
    def source(self) -> str:
        return self.name if self.asname is None else f"{self.name} as {self.asname}"

    @classmethod
    def from_alias(cls, alias: ast.alias) -> ImportAlias:
        return cls(name=alias.name, asname=alias.asname)

    @property
    def spells_as_clause(self) -> bool:
        return self.asname is not None

    @property
    def declares_reexport(self) -> bool:
        """`X as X` is the explicit public re-export spelling."""

        return self.asname == self.name

    def renamed(self, new_name: str) -> ImportAlias:
        """The alias for a renamed declaration: keep a foreign `as` binding, else follow."""

        if self.asname is None or self.asname == self.name:
            return ImportAlias(new_name, self.asname and new_name)
        return ImportAlias(new_name, self.asname)

    @property
    def canonical_key(self) -> tuple[str, str]:
        """Return the source-spelling key for commutative import merging."""

        return self.name, self.asname or ""


class ModuleImportDeclaration(ImportDeclarationABC):
    source_prefix = "import"
    is_future_import = False

    @property
    def module_references(self) -> tuple[ImportFromModuleName, ...]:
        return tuple(ImportFromModuleName(alias.name) for alias in self.aliases)

    def resolve_origin(
        self,
        resolver: ImportOriginResolverABC[ImportContextT, ImportResolutionT],
        origin: ImportedNameOrigin,
        context: ImportContextT,
    ) -> ImportResolutionT:
        return resolver._module_import_resolution(origin, context)

    def _unaliased_bound_name(self, alias: ImportAlias) -> str:
        return alias.name.partition(".")[0]

    def requested_module_name(
        self,
        alias: ImportAlias,
        module_identity: PythonModulePathIdentity,
    ) -> str:
        return alias.name

    def qualified_name(
        self,
        alias: ImportAlias,
        module_identity: PythonModulePathIdentity,
    ) -> str:
        return (
            alias.name if alias.spells_as_clause else self._unaliased_bound_name(alias)
        )
