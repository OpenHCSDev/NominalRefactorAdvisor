"""Repository proof for renaming one top-level class authority."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import cached_property

from .annotation_semantics import StringizedAnnotationSurface
from .ast_tools import ParsedModule
from .class_index import (
    ClassFamilyIndex,
    IndexedClass,
    ModuleNominalBindingAuthority,
    ModulePublicExportSourceAuthority,
    PublicExportNameReference,
    RepositoryModuleBindingProof,
    module_public_export_contract,
    module_star_import_origins,
    nominal_reference_root,
)
from .codemod_declaration_source import ClassHeaderSpanSourceAuthority
from .codemod_module_declarations import SourceTopLevelDeclarationIndex
from .codemod_source_edits import SourceTextGeometry, SourceTextSpanReplacement
from .declaration_dependencies import ModuleLexicalDependencyProjection


@dataclass(frozen=True)
class ClassAuthorityImportReference:
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
    ) -> tuple["ClassAuthorityImportReference", ...]:
        return tuple(
            cls(module, imported_module_name, statement, alias)
            for statement in module.module.body
            if isinstance(statement, ast.ImportFrom)
            if (
                imported_module_name
                := module.module_path_identity.resolve_import_from_module(
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
        return (
            new_name
            if self.alias.asname is None
            else f"{new_name} as {alias_name}"
        )


@dataclass(frozen=True)
class ClassAuthorityRenameBindingClosure:
    """Import-propagated repository bindings changed by one class rename."""

    target: IndexedClass
    import_references: tuple[ClassAuthorityImportReference, ...]
    renamed_module_names: frozenset[str]

    @classmethod
    def from_modules(
        cls,
        parsed_modules: tuple[ParsedModule, ...],
        target: IndexedClass,
    ) -> "ClassAuthorityRenameBindingClosure":
        import_references = tuple(
            reference
            for module in parsed_modules
            for reference in ClassAuthorityImportReference.for_name(
                module,
                target.simple_name,
            )
        )
        consumers_by_origin: dict[str, list[ClassAuthorityImportReference]] = (
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
            f"{module_name}.{self.target.simple_name}"
            for module_name in self.renamed_module_names
        )

    def imports_for(
        self,
        module: ParsedModule,
    ) -> tuple[ClassAuthorityImportReference, ...]:
        return tuple(
            reference
            for reference in self.import_references
            if reference.importing_module is module
            and reference.imported_module_name in self.renamed_module_names
        )


@dataclass(frozen=True)
class ClassAuthorityModuleRenameProof:
    """Exact rename surfaces proved inside one repository module."""

    module: ParsedModule
    declaration: ast.ClassDef | None
    imports: tuple[ClassAuthorityImportReference, ...]
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
        binding_closure: ClassAuthorityRenameBindingClosure,
        repository_bindings: RepositoryModuleBindingProof,
    ) -> "ClassAuthorityModuleRenameProof":
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
            else public_export_source.name_references(target.simple_name)
        )
        renames_local_binding = (
            module.file_path == target.file_path
            or any(reference.changes_local_binding for reference in imports)
        )
        public_exports = named_public_exports if renames_local_binding else ()
        stable_public_exports = (
            ()
            if renames_local_binding
            or target.simple_name
            not in SourceTopLevelDeclarationIndex(
                source_path=module.file_path,
                module=module.module,
            ).binding_statements_by_name
            else named_public_exports
        )
        binding_authority = ModuleNominalBindingAuthority(module)
        direct_references = tuple(
            reference
            for reference in lexical_dependencies.external_references_named(
                target.simple_name
            )
            if binding_authority.qualified_name_at(
                reference,
                line=reference.lineno,
            )
            in binding_closure.renamed_symbols
        )
        external_reference_ids = frozenset(
            id(reference)
            for reference in lexical_dependencies.external_name_references
        )
        qualified_candidates = tuple(
            node
            for node in ast.walk(module.module)
            if isinstance(node, ast.Attribute)
            if node.attr == target.simple_name
            if (root_reference := nominal_reference_root(node)) is not None
            and id(root_reference) in external_reference_ids
        )
        qualified_names = tuple(
            (
                node,
                binding_authority.qualified_name_at(node, line=node.lineno),
            )
            for node in qualified_candidates
        )
        annotation_references = tuple(
            surface
            for surface in lexical_dependencies.stringized_annotations
            if surface.reference_count(target.simple_name)
            and surface.resolves_module_name(target.simple_name, target.node)
            and binding_authority.qualified_name_at(
                ast.Name(id=target.simple_name, ctx=ast.Load()),
                line=surface.literal.lineno,
            )
            in binding_closure.renamed_symbols
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
            declaration=target.node if module.file_path == target.file_path else None,
            imports=imports,
            public_exports=public_exports,
            stable_public_exports=stable_public_exports,
            direct_references=direct_references,
            qualified_references=tuple(
                node
                for node, qualified_name in qualified_names
                if qualified_name in binding_closure.renamed_symbols
            ),
            annotation_references=annotation_references,
        )

    @staticmethod
    def _require_supported_import_surfaces(
        module: ParsedModule,
        target: IndexedClass,
        imports: tuple[ClassAuthorityImportReference, ...],
        binding_closure: ClassAuthorityRenameBindingClosure,
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
            if alias.name == target.simple_name
            if id(alias) not in supported_alias_ids
        )
        if unsupported_imports:
            raise ValueError(
                f"Class authority {target.qualname!r} has a nested import consumer"
            )

    @staticmethod
    def _require_no_affected_star_imports(
        module: ParsedModule,
        target: IndexedClass,
        new_name: str,
        binding_closure: ClassAuthorityRenameBindingClosure,
        repository_bindings: RepositoryModuleBindingProof,
    ) -> None:
        for origin in module_star_import_origins(module):
            if origin.module_name not in binding_closure.renamed_module_names:
                continue
            exposures = tuple(
                repository_bindings.exposure_for(origin.module_name, name)
                for name in (target.simple_name, new_name)
            )
            if any(exposure.introduces_uncertainty for exposure in exposures):
                raise ValueError(
                    f"Class authority {target.qualname!r} has an unresolved "
                    "star-import boundary"
                )
            if any(exposure.proves_public_exposure for exposure in exposures):
                raise ValueError(
                    f"Class authority {target.qualname!r} has an affected "
                    "star-import boundary"
                )

    @staticmethod
    def _require_import_binding_collisions_absent(
        module: ParsedModule,
        target: IndexedClass,
        new_name: str,
        imports: tuple[ClassAuthorityImportReference, ...],
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
            target.simple_name,
            (),
        )
        import_statements = tuple(
            dict.fromkeys(reference.statement for reference in changing_imports)
        )
        if old_name_bindings != import_statements:
            raise ValueError(
                f"Imported class binding {target.simple_name!r} is rebound"
            )
        if new_name in declaration_index.binding_statements_by_name:
            raise ValueError(
                f"Replacement class name {new_name!r} collides in {module.file_path!r}"
            )
        if module_public_export_contract(module).exposure_for(
            target.simple_name
        ).introduces_uncertainty:
            raise ValueError(
                f"Imported class binding {target.simple_name!r} has unresolved "
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
                    geometry,
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
        geometry: SourceTextGeometry,
        declaration: ast.ClassDef,
        new_name: str,
    ) -> SourceTextSpanReplacement:
        span = ClassHeaderSpanSourceAuthority(
            declaration,
            self.module.source,
        ).name_span
        return self._name_replacement(
            (span.start_offset, span.end_offset),
            new_name,
        )

    @staticmethod
    def _import_replacement(
        geometry: SourceTextGeometry,
        reference: ClassAuthorityImportReference,
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
                f"Cannot resolve qualified class name token {reference.attr!r}"
            )
        return start_offset, end_offset


@dataclass(frozen=True)
class ClassAuthorityRenameProof:
    """Closed repository proof for one top-level class-authority rename."""

    binding_closure: ClassAuthorityRenameBindingClosure
    modules: tuple[ClassAuthorityModuleRenameProof, ...]

    @property
    def target(self) -> IndexedClass:
        return self.binding_closure.target

    @classmethod
    def require(
        cls,
        parsed_modules: tuple[ParsedModule, ...],
        class_index: ClassFamilyIndex,
        *,
        target_symbol: str,
        new_name: str,
    ) -> "ClassAuthorityRenameProof":
        target = cls._required_target(
            parsed_modules,
            class_index,
            target_symbol=target_symbol,
            new_name=new_name,
        )
        binding_closure = ClassAuthorityRenameBindingClosure.from_modules(
            parsed_modules,
            target,
        )
        repository_bindings = RepositoryModuleBindingProof(parsed_modules)
        modules = tuple(
            ClassAuthorityModuleRenameProof.require(
                module,
                new_name,
                binding_closure,
                repository_bindings,
            )
            for module in parsed_modules
        )
        cls._require_no_dynamic_name_surfaces(parsed_modules, target, modules)
        return cls(binding_closure=binding_closure, modules=modules)

    @staticmethod
    def _required_target(
        parsed_modules: tuple[ParsedModule, ...],
        class_index: ClassFamilyIndex,
        *,
        target_symbol: str,
        new_name: str,
    ) -> IndexedClass:
        target = class_index.class_for(target_symbol)
        if target is None:
            raise ValueError(f"Class authority {target_symbol!r} is unavailable")
        if "." in target.qualname:
            raise ValueError("Class-authority rename requires a top-level class")
        if target.simple_name == new_name:
            raise ValueError("Class-authority rename requires a distinct name")
        modules_by_path = {module.file_path: module for module in parsed_modules}
        if len(modules_by_path) != len(parsed_modules):
            raise ValueError("Class-authority rename requires unique source modules")
        if len({module.module_name for module in parsed_modules}) != len(
            parsed_modules
        ):
            raise ValueError("Class-authority rename requires unique module identities")
        source_module = modules_by_path.get(target.file_path)
        if source_module is None:
            raise ValueError(f"Source module {target.file_path!r} is unavailable")
        declaration_index = SourceTopLevelDeclarationIndex(
            source_path=target.file_path,
            module=source_module.module,
        )
        declaration = declaration_index.required_declaration(target.simple_name)
        if declaration.node is not target.node or not isinstance(
            declaration.node,
            ast.ClassDef,
        ):
            raise ValueError("Class-authority rename target is not one exact binding")
        if new_name in declaration_index.binding_statements_by_name:
            raise ValueError(f"Replacement class name {new_name!r} is already bound")
        if module_public_export_contract(source_module).exposure_for(
            target.simple_name
        ).introduces_uncertainty:
            raise ValueError("Class-authority export policy is unresolved")
        return target

    @staticmethod
    def _require_no_dynamic_name_surfaces(
        parsed_modules: tuple[ParsedModule, ...],
        target: IndexedClass,
        modules: tuple[ClassAuthorityModuleRenameProof, ...],
    ) -> None:
        name_pattern = re.compile(
            rf"(?<![\w]){re.escape(target.simple_name)}(?![\w])"
        )
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
                        f"Class authority {target.qualname!r} has a string name surface"
                    )
                if isinstance(node, ast.Global | ast.Nonlocal) and target.simple_name in (
                    node.names
                ):
                    raise ValueError(
                        f"Class authority {target.qualname!r} has an explicit lexical "
                        "scope declaration"
                    )
            for token in tokenize.generate_tokens(
                io.StringIO(parsed_module.source).readline
            ):
                if token.type == tokenize.COMMENT and name_pattern.search(token.string):
                    raise ValueError(
                        f"Class authority {target.qualname!r} has a comment name surface"
                    )
