"""Repository proof for renaming one module-local class authority."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass

from .annotation_semantics import StringizedAnnotationSurface
from .ast_tools import ParsedModule
from .class_index import (
    ClassFamilyIndex,
    IndexedClass,
    ModuleNominalBindingAuthority,
    module_public_export_contract,
    module_star_import_origins,
    nominal_reference_root,
)
from .codemod_declaration_source import ClassHeaderSpanSourceAuthority
from .codemod_module_declarations import SourceTopLevelDeclarationIndex
from .codemod_source_edits import SourceTextGeometry, SourceTextSpanReplacement
from .declaration_dependencies import ModuleLexicalDependencyProjection


@dataclass(frozen=True)
class LocalClassAuthorityRenameProof:
    """Closed proof that one class name has no repository-external consumers."""

    target: IndexedClass
    source_module: ParsedModule
    direct_references: tuple[ast.Name, ...]
    qualified_references: tuple[ast.Attribute, ...]
    annotation_references: tuple[StringizedAnnotationSurface, ...]

    @classmethod
    def require(
        cls,
        parsed_modules: tuple[ParsedModule, ...],
        class_index: ClassFamilyIndex,
        *,
        target_symbol: str,
        new_name: str,
    ) -> "LocalClassAuthorityRenameProof":
        target = class_index.class_for(target_symbol)
        if target is None:
            raise ValueError(f"Class authority {target_symbol!r} is unavailable")
        if "." in target.qualname:
            raise ValueError("Local class-authority rename requires a top-level class")
        if target.simple_name == new_name:
            raise ValueError("Class-authority rename requires a distinct name")
        modules_by_path = {module.file_path: module for module in parsed_modules}
        if len(modules_by_path) != len(parsed_modules):
            raise ValueError("Class-authority rename requires unique source modules")
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
        lexical_dependencies = ModuleLexicalDependencyProjection.from_module(
            source_module.module
        )
        binding_authority = ModuleNominalBindingAuthority(source_module)
        direct_references = tuple(
            reference
            for reference in lexical_dependencies.external_references_named(
                target.simple_name
            )
            if binding_authority.qualified_name_at(
                reference,
                line=reference.lineno,
            )
            == target.symbol
        )
        external_reference_ids = frozenset(
            id(reference)
            for reference in lexical_dependencies.external_name_references
        )
        qualified_references = tuple(
            node
            for node in ast.walk(source_module.module)
            if isinstance(node, ast.Attribute)
            if (
                root_reference := nominal_reference_root(node)
            ) is not None
            and id(root_reference) in external_reference_ids
            and binding_authority.qualified_name_at(
                node,
                line=node.lineno,
            )
            == target.symbol
        )
        annotation_references = tuple(
            surface
            for surface in lexical_dependencies.stringized_annotations
            if surface.reference_count(target.simple_name)
            and surface.resolves_module_name(target.simple_name, target.node)
        )
        cls._require_no_dynamic_name_surfaces(
            parsed_modules,
            target,
            annotation_references,
        )
        cls._require_no_external_consumers(
            parsed_modules,
            target,
        )
        return cls(
            target=target,
            source_module=source_module,
            direct_references=direct_references,
            qualified_references=qualified_references,
            annotation_references=annotation_references,
        )

    def source_replacements(
        self,
        new_name: str,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        geometry = SourceTextGeometry(self.source_module.source)
        declaration_span = ClassHeaderSpanSourceAuthority(
            self.target.node,
            self.source_module.source,
        ).name_span
        direct_spans = tuple(
            geometry.required_node_offsets(reference)
            for reference in self.direct_references
        )
        qualified_spans = tuple(
            self._qualified_name_offsets(geometry, reference)
            for reference in self.qualified_references
        )
        name_replacements = tuple(
            SourceTextSpanReplacement.from_offsets(
                start_offset=start_offset,
                end_offset=end_offset,
                replacement_source=new_name,
            )
            for start_offset, end_offset in (
                (declaration_span.start_offset, declaration_span.end_offset),
                *direct_spans,
                *qualified_spans,
            )
        )
        annotation_replacements = tuple(
            self._annotation_replacement(geometry, reference, new_name)
            for reference in self.annotation_references
        )
        return (*name_replacements, *annotation_replacements)

    def _annotation_replacement(
        self,
        geometry: SourceTextGeometry,
        reference: StringizedAnnotationSurface,
        new_name: str,
    ) -> SourceTextSpanReplacement:
        start_offset, end_offset = geometry.required_node_offsets(reference.literal)
        literal_source = geometry.source[start_offset:end_offset]
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=reference.renamed_source(
                literal_source,
                old_name=self.target.simple_name,
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

    @staticmethod
    def _require_no_dynamic_name_surfaces(
        parsed_modules: tuple[ParsedModule, ...],
        target: IndexedClass,
        annotation_references: tuple[StringizedAnnotationSurface, ...],
    ) -> None:
        name_pattern = re.compile(
            rf"(?<![\w]){re.escape(target.simple_name)}(?![\w])"
        )
        annotation_literal_ids = frozenset(
            id(reference.literal) for reference in annotation_references
        )
        for parsed_module in parsed_modules:
            for node in ast.walk(parsed_module.module):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and name_pattern.search(node.value)
                    and id(node) not in annotation_literal_ids
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

    @staticmethod
    def _require_no_external_consumers(
        parsed_modules: tuple[ParsedModule, ...],
        target: IndexedClass,
    ) -> None:
        known_module_names = frozenset(
            parsed_module.module_name for parsed_module in parsed_modules
        )
        for parsed_module in parsed_modules:
            if parsed_module.file_path == target.file_path:
                continue
            if any(
                alias.name == target.simple_name
                and parsed_module.module_path_identity.resolve_import_from_module(
                    imported_module=node.module,
                    level=node.level,
                )
                in known_module_names
                for node in ast.walk(parsed_module.module)
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            ):
                raise ValueError(
                    f"Class authority {target.qualname!r} has an imported repository "
                    "consumer"
                )
            if module_star_import_origins(
                parsed_module
            ) and ModuleLexicalDependencyProjection.from_module(
                parsed_module.module
            ).external_references_named(
                target.simple_name
            ):
                raise ValueError(
                    f"Class authority {target.qualname!r} has a star-import consumer"
                )
            if any(
                isinstance(node, ast.Attribute)
                and node.attr == target.simple_name
                for node in ast.walk(parsed_module.module)
            ):
                raise ValueError(
                    f"Class authority {target.qualname!r} has a qualified repository "
                    "consumer"
                )
