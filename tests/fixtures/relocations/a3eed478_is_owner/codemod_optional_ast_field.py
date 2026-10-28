# Extracted from a3eed478:nominal_refactor_advisor/codemod_optional_ast_field.py


@dataclass(frozen=True)
class NamedOwnerDeclaration:
    """A top-level declaration carrying a decoder owner's name. It is the owner
    only when its source is the owner's own (`NativeDeclaration`), not by name."""

    module_name: str
    file_path: str
    node: ast.ClassDef | ast.FunctionDef
    owner: NativeDeclaration

    @property
    def name(self) -> str:
        return self.node.name

    @cached_property
    def is_owner(self) -> bool:
        return self.owner.source_matches(self.node)

    @property
    def conflict(self) -> str:
        return (
            f"`{self.name}` at {self.file_path}:{self.node.lineno} is not the "
            f"decoder owner `{self.owner.qualified_name}` "
            f"({inspect.getsourcefile(self.owner.declaration)}:"
            f"{inspect.getsourcelines(self.owner.declaration)[1]}); "
            "the same name with a different declaration leaves the routing ambiguous"
        )


@dataclass(frozen=True)
class ProjectDecoderOwners:
    """Where one root package declares each decoder owner, and what it lacks.

    Owners declared outside the generated module are imported from where they
    are. The rest live in the generated module, whose source is copied from the
    owners this package declares.
    """

    snapshot: CodemodSourceSnapshot
    root: str

    @classmethod
    def of(cls, snapshot: CodemodSourceSnapshot, module_name: str) -> Self:
        root = module_name.partition(".")[0]
        return _PROJECT_OWNERS.for_snapshot(snapshot).setdefault(
            root, cls(snapshot, root)
        )

    @property
    def generated_module(self) -> str:
        return f"{self.root}.{GENERATED_MODULE}"

    @cached_property
    def named_declarations(self) -> tuple[NamedOwnerDeclaration, ...]:
        """Every top-level class or function in this root package that carries an
        owner's name, whether or not it is that owner."""

        owners = {
            owner.__name__: NativeDeclaration(owner)
            for decoder in OptionalAstFieldDecoder.__registry__.values()
            for owner in decoder.owners
        }
        return tuple(
            NamedOwnerDeclaration(parsed.module_name, str(parsed.path), statement, owner)
            for parsed in self.snapshot.parsed_modules
            if parsed.module_name.partition(".")[0] == self.root
            for statement in parsed.module.body
            if isinstance(statement, ast.ClassDef | ast.FunctionDef)
            and (owner := owners.get(statement.name)) is not None
        )

    @cached_property
    def top_level_names(self) -> Mapping[str, frozenset[str]]:
        """Module -> the owners it declares at top level, by declaration identity."""

        names: dict[str, set[str]] = {}
        for declaration in self.named_declarations:
            if declaration.is_owner:
                names.setdefault(declaration.module_name, set()).add(
                    declaration.name
                )
        return {module: frozenset(found) for module, found in names.items()}

    @cached_property
    def declared(self) -> Mapping[str, str]:
        """Owner name -> the one hand-declared module of this package defining it."""

        found: dict[str, list[str]] = {}
        for module, names in self.top_level_names.items():
            if module == self.generated_module:
                continue
            for name in names:
                found.setdefault(name, []).append(module)
        return {
            name: modules[0] for name, modules in found.items() if len(modules) == 1
        }

    def require_unshadowed(self, names: Iterable[str]) -> None:
        """Refuse an owner name the package also gives a different declaration:
        importing or generating it would leave two meanings under one name."""

        names = frozenset(names)
        shadows = [
            declaration
            for declaration in self.named_declarations
            if declaration.name in names and not declaration.is_owner
        ]
        if shadows:
            raise ValueError(
                "; ".join(declaration.conflict for declaration in shadows)
            )

    @cached_property
    def pending(self) -> tuple[type[OptionalAstFieldDecoder], ...]:
        """Decoders the package probes whose owners only generation can supply."""

        probed = {
            node.left.attr
            for parsed in self.snapshot.parsed_modules
            if parsed.module_name.partition(".")[0] == self.root
            for node in ast.walk(parsed.module)
            if _none_probe(node)
        }
        return tuple(
            decoder
            for decoder in OptionalAstFieldDecoder.__registry__.values()
            if not all(name in self.declared for name in decoder.owner_names())
            and probed & decoder.field_names()
        )

    def generated_path(self) -> str:
        package = self.snapshot.module_import_graph.unique_source_file_for_module_name(
            self.root
        )
        if package is None or Path(package.file_path).name != "__init__.py":
            raise ValueError(
                f"`{self.root}` is not a package to generate {GENERATED_MODULE} in"
            )
        return str(Path(package.file_path).parent / f"{GENERATED_MODULE}.py")

    @property
    def needs_declaration(self) -> bool:
        """Whether a pending decoder has no generated module to live in yet."""

        return bool(self.pending) and (
            self.generated_path() not in self.snapshot.sources_by_file_path
        )

    def generated_source(self) -> str:
        """The pending decoders, copied from the owners this package declares."""

        self.require_unshadowed(
            name for decoder in self.pending for name in decoder.owner_names()
        )
        body = "\n\n".join(
            inspect.getsource(owner)
            for decoder in self.pending
            for owner in decoder.owners
        )
        imports = [
            line
            for line, used in (
                ("import ast", "ast."),
                ("from abc import ABC, abstractmethod", "(ABC)"),
                ("from collections.abc import Iterable, Mapping", "Iterable["),
                ("from dataclasses import dataclass", "@dataclass"),
            )
            if used in body
        ]
        return (
            '"""The one boundary read of each optional ast field (generated by NRA)."""'
            "\n\nfrom __future__ import annotations\n\n"
            + "\n".join(imports)
            + "\n\n\n"
            + body.rstrip()
            + "\n"
        )

    def import_sources_for(
        self, parsed_module: ParsedModule, names: Iterable[str]
    ) -> tuple[str, ...]:
        """One import per owner name the rewrite reads, none inside its own module.

        A name the package does not declare must already be in the generated
        module: routing to an owner nothing declares would import nothing.
        """

        names = tuple(names)
        self.require_unshadowed(names)
        sources = []
        for name in names:
            module = self.declared.get(name, self.generated_module)
            if name not in self.top_level_names.get(module, frozenset()):
                raise ValueError(
                    f"`{name}` is not declared under `{self.root}`; the recipe "
                    f"declaring {self.generated_module} routes this function"
                )
            if module != parsed_module.module_name:
                sources.append(
                    ImportedClassReference(f"{module}.{name}").import_source_for(
                        self.snapshot, parsed_module
                    )
                )
        return tuple(sources)
