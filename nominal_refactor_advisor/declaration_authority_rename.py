"""Repository proof for renaming one top-level declaration authority."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import cached_property

from .annotation_semantics import StringizedAnnotationSurface
from .ast_tools import ModuleAnnotationEvaluationMode, ParsedModule
from .class_index import (
    ModuleNominalBindingAuthority,
    ModulePublicExportSourceAuthority,
    PublicExportNameReference,
    RepositoryModuleBindingProof,
    module_public_export_contract,
    module_star_import_origins,
    nominal_reference_root,
)
from .codemod_module_declarations import (
    SourceTopLevelDeclaration,
    SourceTopLevelDeclarationIndex,
)
from .codemod_source_edits import SourceTextGeometry, SourceTextSpanReplacement
from .declaration_dependencies import (
    ModuleLexicalDependencyProjection,
    ModuleNameReferenceSurface,
    MovableDeclaration,
)
from .source_index import AstTargetDigest


@dataclass(frozen=True)
class TopLevelBindingRenameTarget:
    """One exact top-level binding selected for repository-wide renaming."""

    source_module: ParsedModule
    declaration: SourceTopLevelDeclaration

    @classmethod
    def require(
        cls,
        parsed_modules: tuple[ParsedModule, ...],
        target: AstTargetDigest,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        new_name: str,
    ) -> "TopLevelBindingRenameTarget":
        if target.qualname != target.name:
            raise ValueError("Declaration rename requires a top-level target")
        rename_target = cls.require_binding(
            parsed_modules,
            source_path=target.file_path,
            binding_name=target.name,
            new_name=new_name,
        )
        if rename_target.declaration.node is not node:
            raise ValueError("Declaration rename target is not one exact binding")
        return rename_target

    @classmethod
    def require_binding(
        cls,
        parsed_modules: tuple[ParsedModule, ...],
        *,
        source_path: str,
        binding_name: str,
        new_name: str,
    ) -> "TopLevelBindingRenameTarget":
        modules_by_path = {module.file_path: module for module in parsed_modules}
        if len(modules_by_path) != len(parsed_modules):
            raise ValueError("Declaration rename requires unique source modules")
        if len({module.module_name for module in parsed_modules}) != len(
            parsed_modules
        ):
            raise ValueError("Declaration rename requires unique module identities")
        source_module = modules_by_path.get(source_path)
        if source_module is None:
            raise ValueError(f"Source module {source_path!r} is unavailable")
        declaration_index = SourceTopLevelDeclarationIndex(
            source_path=source_path,
            module=source_module.module,
        )
        declaration = declaration_index.required_declaration(binding_name)
        if binding_name == new_name:
            raise ValueError("Declaration rename requires a distinct name")
        if new_name in declaration_index.binding_statements_by_name:
            raise ValueError(f"Replacement name {new_name!r} is already bound")
        if (
            module_public_export_contract(source_module)
            .exposure_for(binding_name)
            .introduces_uncertainty
        ):
            raise ValueError("Declaration export policy is unresolved")
        return cls(source_module, declaration)

    @property
    def node(self) -> MovableDeclaration:
        return self.declaration.node

    @property
    def file_path(self) -> str:
        return self.source_module.file_path

    @property
    def module_name(self) -> str:
        return self.source_module.module_name

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def self_binding_owner(self) -> ast.ClassDef | None:
        return self.node if isinstance(self.node, ast.ClassDef) else None


@dataclass(frozen=True)
class DeclarationAuthorityImportReference:
    """One direct import of the renamed authority."""

    importing_module: ParsedModule
    imported_module_name: str
    statement: ast.ImportFrom
    alias: ast.alias

    @classmethod
    def for_name(
        cls,
        module: ParsedModule,
        imported_name: str,
    ) -> tuple["DeclarationAuthorityImportReference", ...]:
        return tuple(
            cls(module, imported_module_name, statement, alias)
            for statement in module.module.body
            if isinstance(statement, ast.ImportFrom)
            if (
                imported_module_name := module.module_path_identity.resolve_import_from_module(
                    imported_module=statement.module,
                    level=statement.level,
                )
            )
            is not None
            for alias in statement.names
            if alias.name == imported_name
        )

    @property
    def local_name(self) -> str:
        return self.alias.asname or self.alias.name

    @property
    def changes_local_binding(self) -> bool:
        return self.local_name == self.alias.name

    def renamed_source(self, new_name: str) -> str:
        alias_name = new_name if self.changes_local_binding else self.local_name
        return new_name if self.alias.asname is None else f"{new_name} as {alias_name}"


@dataclass(frozen=True)
class DeclarationAuthorityRenameBindingClosure:
    """Import-propagated repository bindings changed by one declaration rename."""

    target: TopLevelBindingRenameTarget
    import_references: tuple[DeclarationAuthorityImportReference, ...]
    renamed_module_names: frozenset[str]

    @classmethod
    def from_modules(
        cls,
        parsed_modules: tuple[ParsedModule, ...],
        target: TopLevelBindingRenameTarget,
    ) -> "DeclarationAuthorityRenameBindingClosure":
        import_references = tuple(
            reference
            for module in parsed_modules
            for reference in DeclarationAuthorityImportReference.for_name(
                module,
                target.name,
            )
        )
        consumers_by_origin: dict[str, list[DeclarationAuthorityImportReference]] = (
            defaultdict(list)
        )
        for reference in import_references:
            consumers_by_origin[reference.imported_module_name].append(reference)

        renamed_module_names = {target.module_name}
        pending_module_names = deque((target.module_name,))
        while pending_module_names:
            origin_module_name = pending_module_names.popleft()
            for reference in consumers_by_origin[origin_module_name]:
                importing_module_name = reference.importing_module.module_name
                if (
                    not reference.changes_local_binding
                    or importing_module_name in renamed_module_names
                ):
                    continue
                renamed_module_names.add(importing_module_name)
                pending_module_names.append(importing_module_name)
        return cls(
            target=target,
            import_references=import_references,
            renamed_module_names=frozenset(renamed_module_names),
        )

    @cached_property
    def renamed_symbols(self) -> frozenset[str]:
        return frozenset(
            f"{module_name}.{self.target.name}"
            for module_name in self.renamed_module_names
        )

    def imports_for(
        self,
        module: ParsedModule,
    ) -> tuple[DeclarationAuthorityImportReference, ...]:
        return tuple(
            reference
            for reference in self.import_references
            if reference.importing_module is module
            and reference.imported_module_name in self.renamed_module_names
        )


@dataclass(frozen=True)
class DeclarationAuthorityModuleReferenceProof:
    """Binding-derived reference proof for one repository module."""

    target: TopLevelBindingRenameTarget
    binding_closure: DeclarationAuthorityRenameBindingClosure
    binding_authority: ModuleNominalBindingAuthority
    lexical_dependencies: ModuleLexicalDependencyProjection
    annotation_mode: ModuleAnnotationEvaluationMode

    @cached_property
    def final_binding_is_renamed(self) -> bool:
        binding = self.binding_authority.snapshot_before().binding_for(self.target.name)
        return (
            binding is not None
            and binding.qualified_name in self.binding_closure.renamed_symbols
        )

    def proves_direct_surface_binding(
        self,
        surface: ModuleNameReferenceSurface,
    ) -> bool:
        return (
            self.binding_authority.qualified_name_at(
                surface.reference,
                line=surface.binding_snapshot_line,
            )
            in self.binding_closure.renamed_symbols
        )

    def proves_direct_reference(self, surface: ModuleNameReferenceSurface) -> bool:
        proved = self.proves_direct_surface_binding(surface) or (
            surface.is_direct_annotation
            and not self.annotation_mode.annotations_execute_at_declaration
            and self.final_binding_is_renamed
            and surface.resolves_module_name(
                self.target.name,
                self.target.self_binding_owner,
            )
        )
        if proved:
            surface.resolution.require_known(surface.reference.id)
        return proved

    def proves_qualified_reference(
        self,
        reference: ast.Attribute,
        root_surface: ModuleNameReferenceSurface,
    ) -> bool:
        proved = (
            self.binding_authority.qualified_name_at(
                reference,
                line=root_surface.use.binding_phase(
                    root_surface.binding_phase,
                    eager_annotations=self.annotation_mode.annotations_execute_at_declaration,
                ).snapshot_line_for(root_surface.reference),
            )
            in self.binding_closure.renamed_symbols
        )
        if proved:
            root_surface.resolution.require_known(root_surface.reference.id)
        return proved

    def proves_stringized_annotation(
        self,
        surface: StringizedAnnotationSurface,
    ) -> bool:
        return (
            self.final_binding_is_renamed
            and surface.reference_count(self.target.name) > 0
            and surface.resolves_module_name(
                self.target.name,
                self.target.self_binding_owner,
            )
        )

    def require_complete_eager_annotations(self) -> None:
        if not self.annotation_mode.annotations_execute_at_declaration:
            return
        for surface in self.lexical_dependencies.direct_annotation_name_surfaces:
            reference = surface.reference
            if (
                reference.id != self.target.name
                or not self.final_binding_is_renamed
                or self.proves_direct_surface_binding(surface)
                or not surface.resolves_module_name(
                    self.target.name,
                    self.target.self_binding_owner,
                )
                or self.binding_authority.snapshot_before(
                    surface.binding_snapshot_line
                ).resolves_unshadowed_builtin(reference.id)
            ):
                continue
            raise ValueError(
                f"Declaration authority {self.target.name!r} has an unresolved "
                "eager annotation reference"
            )


@dataclass(frozen=True)
class DeclarationAuthorityModuleRenameProof:
    """Exact rename surfaces proved inside one repository module."""

    module: ParsedModule
    declaration: SourceTopLevelDeclaration | None
    imports: tuple[DeclarationAuthorityImportReference, ...]
    public_exports: tuple[PublicExportNameReference, ...]
    stable_public_exports: tuple[PublicExportNameReference, ...]
    direct_references: tuple[ast.Name, ...]
    qualified_references: tuple[ast.Attribute, ...]
    annotation_references: tuple[StringizedAnnotationSurface, ...]

    @classmethod
    def require(
        cls,
        module: ParsedModule,
        new_name: str,
        binding_closure: DeclarationAuthorityRenameBindingClosure,
        repository_bindings: RepositoryModuleBindingProof,
    ) -> "DeclarationAuthorityModuleRenameProof":
        target = binding_closure.target
        imports = binding_closure.imports_for(module)
        cls._require_supported_import_surfaces(
            module,
            target,
            imports,
            binding_closure,
        )
        cls._require_import_binding_collisions_absent(
            module,
            target,
            new_name,
            imports,
        )
        lexical_dependencies = ModuleLexicalDependencyProjection.from_module(
            module.module
        )
        public_export_source = ModulePublicExportSourceAuthority.from_module(
            module.module
        )
        named_public_exports = (
            ()
            if public_export_source is None
            else public_export_source.name_references(target.name)
        )
        renames_local_binding = module.file_path == target.file_path or any(
            reference.changes_local_binding for reference in imports
        )
        public_exports = named_public_exports if renames_local_binding else ()
        stable_public_exports = (
            ()
            if renames_local_binding
            or target.name
            not in SourceTopLevelDeclarationIndex(
                source_path=module.file_path,
                module=module.module,
            ).binding_statements_by_name
            else named_public_exports
        )
        binding_authority = ModuleNominalBindingAuthority(
            module,
            declared_assignment_authority_names=(
                target.declaration.assigned_binding_names
                if module.file_path == target.file_path
                else frozenset()
            ),
        )
        reference_proof = DeclarationAuthorityModuleReferenceProof(
            target=target,
            binding_closure=binding_closure,
            binding_authority=binding_authority,
            lexical_dependencies=lexical_dependencies,
            annotation_mode=ModuleAnnotationEvaluationMode.from_module(module.module),
        )
        reference_proof.require_complete_eager_annotations()
        direct_references = tuple(
            surface.reference
            for surface in lexical_dependencies.external_surfaces_named(target.name)
            if reference_proof.proves_direct_reference(surface)
        )
        external_surfaces_by_reference_id = {
            id(surface.reference): surface
            for surface in lexical_dependencies.direct_name_surfaces
        }
        qualified_candidates = tuple(
            (node, root_surface)
            for node in ast.walk(module.module)
            if isinstance(node, ast.Attribute)
            if node.attr == target.name
            if (root_reference := nominal_reference_root(node)) is not None
            if (
                root_surface := external_surfaces_by_reference_id.get(
                    id(root_reference)
                )
            )
            is not None
        )
        annotation_references = tuple(
            surface
            for surface in lexical_dependencies.stringized_annotations
            if reference_proof.proves_stringized_annotation(surface)
        )
        cls._require_no_affected_star_imports(
            module,
            target,
            new_name,
            binding_closure,
            repository_bindings,
        )
        return cls(
            module=module,
            declaration=(
                target.declaration if module.file_path == target.file_path else None
            ),
            imports=imports,
            public_exports=public_exports,
            stable_public_exports=stable_public_exports,
            direct_references=direct_references,
            qualified_references=tuple(
                node
                for node, root_surface in qualified_candidates
                if reference_proof.proves_qualified_reference(node, root_surface)
            ),
            annotation_references=annotation_references,
        )

    @staticmethod
    def _require_supported_import_surfaces(
        module: ParsedModule,
        target: TopLevelBindingRenameTarget,
        imports: tuple[DeclarationAuthorityImportReference, ...],
        binding_closure: DeclarationAuthorityRenameBindingClosure,
    ) -> None:
        supported_alias_ids = frozenset(id(reference.alias) for reference in imports)
        unsupported_imports = tuple(
            alias
            for statement in ast.walk(module.module)
            if isinstance(statement, ast.ImportFrom)
            if module.module_path_identity.resolve_import_from_module(
                imported_module=statement.module,
                level=statement.level,
            )
            in binding_closure.renamed_module_names
            for alias in statement.names
            if alias.name == target.name
            if id(alias) not in supported_alias_ids
        )
        if unsupported_imports:
            raise ValueError(
                f"Declaration authority {target.name!r} has a nested import consumer"
            )

    @staticmethod
    def _require_no_affected_star_imports(
        module: ParsedModule,
        target: TopLevelBindingRenameTarget,
        new_name: str,
        binding_closure: DeclarationAuthorityRenameBindingClosure,
        repository_bindings: RepositoryModuleBindingProof,
    ) -> None:
        for origin in module_star_import_origins(module):
            if origin.module_name not in binding_closure.renamed_module_names:
                continue
            exposures = tuple(
                repository_bindings.exposure_for(origin.module_name, name)
                for name in (target.name, new_name)
            )
            if any(exposure.introduces_uncertainty for exposure in exposures):
                raise ValueError(
                    f"Declaration authority {target.name!r} has an unresolved "
                    "star-import boundary"
                )
            if any(exposure.proves_public_exposure for exposure in exposures):
                raise ValueError(
                    f"Declaration authority {target.name!r} has an affected "
                    "star-import boundary"
                )

    @staticmethod
    def _require_import_binding_collisions_absent(
        module: ParsedModule,
        target: TopLevelBindingRenameTarget,
        new_name: str,
        imports: tuple[DeclarationAuthorityImportReference, ...],
    ) -> None:
        changing_imports = tuple(
            reference for reference in imports if reference.changes_local_binding
        )
        if not changing_imports:
            return
        declaration_index = SourceTopLevelDeclarationIndex(
            source_path=module.file_path,
            module=module.module,
        )
        old_name_bindings = declaration_index.binding_statements_by_name.get(
            target.name,
            (),
        )
        import_statements = tuple(
            dict.fromkeys(reference.statement for reference in changing_imports)
        )
        if old_name_bindings != import_statements:
            raise ValueError(f"Imported declaration binding {target.name!r} is rebound")
        if new_name in declaration_index.binding_statements_by_name:
            raise ValueError(
                f"Replacement name {new_name!r} collides in {module.file_path!r}"
            )
        if (
            module_public_export_contract(module)
            .exposure_for(target.name)
            .introduces_uncertainty
        ):
            raise ValueError(
                f"Imported declaration binding {target.name!r} has unresolved "
                "export policy"
            )

    @property
    def has_replacements(self) -> bool:
        return self.declaration is not None or bool(
            self.imports
            or self.public_exports
            or self.direct_references
            or self.qualified_references
            or self.annotation_references
        )

    def source_replacements(
        self,
        *,
        old_name: str,
        new_name: str,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        geometry = SourceTextGeometry(self.module.source)
        declaration_replacements = (
            ()
            if self.declaration is None
            else (
                self._declaration_replacement(
                    self.declaration,
                    new_name,
                ),
            )
        )
        import_replacements = tuple(
            self._import_replacement(geometry, reference, new_name)
            for reference in self.imports
        )
        public_export_replacements = tuple(
            self._public_export_replacement(geometry, reference, new_name)
            for reference in self.public_exports
        )
        direct_replacements = tuple(
            self._name_replacement(
                geometry.required_node_offsets(reference),
                new_name,
            )
            for reference in self.direct_references
        )
        qualified_replacements = tuple(
            self._name_replacement(
                self._qualified_name_offsets(geometry, reference),
                new_name,
            )
            for reference in self.qualified_references
        )
        annotation_replacements = tuple(
            self._annotation_replacement(
                geometry,
                reference,
                old_name=old_name,
                new_name=new_name,
            )
            for reference in self.annotation_references
        )
        return (
            *declaration_replacements,
            *import_replacements,
            *public_export_replacements,
            *direct_replacements,
            *qualified_replacements,
            *annotation_replacements,
        )

    def _declaration_replacement(
        self,
        declaration: SourceTopLevelDeclaration,
        new_name: str,
    ) -> SourceTextSpanReplacement:
        span = declaration.name_span(self.module.source)
        return self._name_replacement(
            (span.start_offset, span.end_offset),
            new_name,
        )

    @staticmethod
    def _import_replacement(
        geometry: SourceTextGeometry,
        reference: DeclarationAuthorityImportReference,
        new_name: str,
    ) -> SourceTextSpanReplacement:
        start_offset, end_offset = geometry.required_node_offsets(reference.alias)
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=reference.renamed_source(new_name),
        )

    @staticmethod
    def _public_export_replacement(
        geometry: SourceTextGeometry,
        reference: PublicExportNameReference,
        new_name: str,
    ) -> SourceTextSpanReplacement:
        start_offset, end_offset = geometry.required_node_offsets(reference.literal)
        literal_source = geometry.source[start_offset:end_offset]
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=reference.renamed_source(literal_source, new_name),
        )

    @staticmethod
    def _name_replacement(
        offsets: tuple[int, int],
        new_name: str,
    ) -> SourceTextSpanReplacement:
        start_offset, end_offset = offsets
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=new_name,
        )

    @staticmethod
    def _annotation_replacement(
        geometry: SourceTextGeometry,
        reference: StringizedAnnotationSurface,
        *,
        old_name: str,
        new_name: str,
    ) -> SourceTextSpanReplacement:
        start_offset, end_offset = geometry.required_node_offsets(reference.literal)
        literal_source = geometry.source[start_offset:end_offset]
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=reference.renamed_source(
                literal_source,
                old_name=old_name,
                new_name=new_name,
            ),
        )

    @staticmethod
    def _qualified_name_offsets(
        geometry: SourceTextGeometry,
        reference: ast.Attribute,
    ) -> tuple[int, int]:
        _start_offset, end_offset = geometry.required_node_offsets(reference)
        start_offset = end_offset - len(reference.attr)
        if geometry.source[start_offset:end_offset] != reference.attr:
            raise ValueError(
                f"Cannot resolve qualified declaration name token {reference.attr!r}"
            )
        return start_offset, end_offset


@dataclass(frozen=True)
class DeclarationAuthorityRenameProof:
    """Closed repository proof for one top-level declaration-authority rename."""

    binding_closure: DeclarationAuthorityRenameBindingClosure
    modules: tuple[DeclarationAuthorityModuleRenameProof, ...]

    @property
    def target(self) -> TopLevelBindingRenameTarget:
        return self.binding_closure.target

    @classmethod
    def require(
        cls,
        parsed_modules: tuple[ParsedModule, ...],
        target: AstTargetDigest,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        new_name: str,
    ) -> "DeclarationAuthorityRenameProof":
        rename_target = TopLevelBindingRenameTarget.require(
            parsed_modules,
            target,
            node,
            new_name=new_name,
        )
        return cls._from_target(parsed_modules, rename_target, new_name=new_name)

    @classmethod
    def require_binding(
        cls,
        parsed_modules: tuple[ParsedModule, ...],
        *,
        source_path: str,
        binding_name: str,
        new_name: str,
    ) -> "DeclarationAuthorityRenameProof":
        rename_target = TopLevelBindingRenameTarget.require_binding(
            parsed_modules,
            source_path=source_path,
            binding_name=binding_name,
            new_name=new_name,
        )
        return cls._from_target(parsed_modules, rename_target, new_name=new_name)

    @classmethod
    def _from_target(
        cls,
        parsed_modules: tuple[ParsedModule, ...],
        rename_target: TopLevelBindingRenameTarget,
        *,
        new_name: str,
    ) -> "DeclarationAuthorityRenameProof":
        binding_closure = DeclarationAuthorityRenameBindingClosure.from_modules(
            parsed_modules,
            rename_target,
        )
        repository_bindings = RepositoryModuleBindingProof(parsed_modules)
        modules = tuple(
            DeclarationAuthorityModuleRenameProof.require(
                module,
                new_name,
                binding_closure,
                repository_bindings,
            )
            for module in parsed_modules
        )
        cls._require_no_dynamic_name_surfaces(
            parsed_modules,
            rename_target,
            modules,
        )
        return cls(binding_closure=binding_closure, modules=modules)

    @staticmethod
    def _require_no_dynamic_name_surfaces(
        parsed_modules: tuple[ParsedModule, ...],
        target: TopLevelBindingRenameTarget,
        modules: tuple[DeclarationAuthorityModuleRenameProof, ...],
    ) -> None:
        name_pattern = re.compile(rf"(?<![\w]){re.escape(target.name)}(?![\w])")
        annotation_literal_ids = frozenset(
            id(reference.literal)
            for module in modules
            for reference in module.annotation_references
        )
        public_export_literal_ids = frozenset(
            id(reference.literal)
            for module in modules
            for reference in (*module.public_exports, *module.stable_public_exports)
        )
        for parsed_module in parsed_modules:
            for node in ast.walk(parsed_module.module):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and name_pattern.search(node.value)
                    and id(node) not in annotation_literal_ids
                    and id(node) not in public_export_literal_ids
                ):
                    raise ValueError(
                        f"Declaration authority {target.name!r} has a string "
                        "name surface"
                    )
                if isinstance(node, ast.Global | ast.Nonlocal) and target.name in (
                    node.names
                ):
                    raise ValueError(
                        f"Declaration authority {target.name!r} has an explicit "
                        "lexical scope declaration"
                    )
            for token in tokenize.generate_tokens(
                io.StringIO(parsed_module.source).readline
            ):
                if token.type == tokenize.COMMENT and name_pattern.search(token.string):
                    raise ValueError(
                        f"Declaration authority {target.name!r} has a comment "
                        "name surface"
                    )
