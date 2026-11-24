"""The class-family index built from parsed modules: each class, its resolved bases and the families they form."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    ParsedModuleFact,
    ParsedModuleSet,
    ParsedModuleSetFact,
)
from nominal_refactor_advisor.class_index import (
    ModuleClassHeaderCore,
    ClassDeclaration,
    ClassDeclarationIndex,
    ClassLineBindingSnapshots,
    ClassMethodPromotionSafeDecorator,
    ClassRegistryFacts,
    CompactClassFamilyIndexBuilder,
    CompactModuleClassProjectionFamily,
    CompactRepositoryPublicExposureIndex,
    ModuleNominalBindingAuthority,
    ModuleNominalBindingSnapshot,
    RegisteringMetaclassAuthority,
    _class_scope_qualified_import_name,
    _dataclass_declaration,
    _single_assignment_target_name,
    module_import_aliases,
)
from nominal_refactor_advisor.class_namespace import ClassNamespaceExecutionEvidence
from nominal_refactor_advisor.class_symbols import (
    ClassSymbolResolutionAuthority,
    _unique_known_symbol_by_suffix,
    _UniqueKnownSymbolSuffixIndex,
    iter_class_definitions,
)
from nominal_refactor_advisor.collection_algebra import (
    UniqueIdentityIndexAuthority,
    sorted_tuple,
)
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.native_declarations import ClassNamespaceDeclaration


@dataclass(frozen=True)
class IndexedClass(ClassDeclaration, ClassNamespaceDeclaration):
    node: ast.ClassDef

    member_binding_names = AliasProperty[frozenset[str]](
        "namespace_execution.binding_names"
    )

    @cached_property
    def namespace_execution(self) -> ClassNamespaceExecutionEvidence:
        return ClassNamespaceExecutionEvidence.from_class(self.node)

    @property
    def is_final(self) -> bool:
        return any(
            (isinstance(decorator, ast.Name) and decorator.id == "final")
            or (isinstance(decorator, ast.Attribute) and decorator.attr == "final")
            for decorator in self.node.decorator_list
        )

    @classmethod
    def from_parsed_class(
        cls,
        parsed_module: ParsedModule,
        qualname: str,
        node: ast.ClassDef,
        module_binding_snapshot: ModuleNominalBindingSnapshot,
    ) -> IndexedClass:
        return cls(
            symbol=f"{parsed_module.module_name}.{qualname}",
            module_name=parsed_module.module_name,
            qualname=qualname,
            simple_name=qualname.rsplit(".", 1)[-1],
            file_path=parsed_module.file_path,
            line=node.lineno,
            node=node,
            declared_base_names=tuple(
                declared_base_name
                for base in node.bases
                if (
                    declared_base_name := ClassSymbolResolutionAuthority.declared_base_name(
                        base
                    )
                )
                is not None
            ),
            resolved_base_symbols=(),
            registry_facts=ClassRegistryFacts.of_class(node, module_binding_snapshot, parsed_module),
            dataclass_declaration=_dataclass_declaration(
                module_binding_snapshot,
                qualname,
                node,
            ),
            class_decorators_are_promotion_safe=all(
                ClassMethodPromotionSafeDecorator.for_qualified_name(
                    _class_scope_qualified_import_name(
                        module_binding_snapshot,
                        {},
                        (
                            decorator.func
                            if isinstance(decorator, ast.Call)
                            else decorator
                        ),
                        frozenset(),
                    )
                )
                is not None
                for decorator in node.decorator_list
            ),
        )


@dataclass(frozen=True)
class ClassFamilyIndex(ClassDeclarationIndex[IndexedClass], ParsedModuleSetFact):
    @classmethod
    def derive(cls, modules: ParsedModuleSet) -> ClassFamilyIndex:
        return ClassFamilyIndexBuilder(modules.modules).build()

    def projected(
        self,
        previous_modules: Iterable[ParsedModule],
        projected_modules: tuple[ParsedModule, ...],
    ) -> ClassFamilyIndex:
        """This index over a projected source state. Its inputs are each module's
        class records and header core; when every module that left or replaced
        a parse declares no class and keeps an equal header core, the inputs are
        unchanged and so is the index. Otherwise it is rebuilt from the facts
        its unchanged parses already hold, resolving again only the records a
        changed symbol could resolve differently."""

        previous = {module.file_path: module for module in previous_modules}
        projected = {module.file_path: module for module in projected_modules}
        replaced = tuple(
            (previous.get(path), projected.get(path))
            for path in previous.keys() | projected.keys()
            if previous.get(path) is not projected.get(path)
        )
        if all(
            old is not None
            and new is not None
            and not old.fact(ModuleIndexedClasses).records
            and not new.fact(ModuleIndexedClasses).records
            and old.fact(ModuleClassHeaderCore) == new.fact(ModuleClassHeaderCore)
            for old, new in replaced
        ):
            return self
        return ClassFamilyIndexBuilder(
            projected_modules,
            previous=ClassFamilyIndexPredecessor(self, previous),
        ).build()


@dataclass(frozen=True)
class ModuleClassReferenceResolver:
    """Resolve class references in expression syntax against a class index."""

    parsed_module: ParsedModule
    class_index: ClassFamilyIndex

    @cached_property
    def known_symbols(self) -> frozenset[str]:
        return self.class_index.known_symbols

    @cached_property
    def unique_symbols_by_name(self) -> dict[str, str]:
        return self.class_index.unique_symbols_by_name

    @cached_property
    def import_aliases(self) -> dict[str, str]:
        return module_import_aliases(self.parsed_module)

    @cached_property
    def constructor_assignment_symbols(self) -> dict[str, str]:
        assignments: dict[str, str] = {}
        for statement in self.parsed_module.module.body:
            if not isinstance(statement, ast.Assign | ast.AnnAssign):
                continue
            target_name = _single_assignment_target_name(statement)
            if target_name is None:
                continue
            value = statement.value
            if value is None:
                continue
            symbol = self._direct_constructor_symbol(value)
            if symbol is not None:
                assignments[target_name] = symbol
        return assignments

    @cached_property
    def reference_resolution(self) -> ClassSymbolResolutionAuthority:
        return ClassSymbolResolutionAuthority(
            parsed_module=self.parsed_module,
            import_aliases=self.import_aliases,
            known_symbols=self.known_symbols,
            unique_symbols_by_name=self.unique_symbols_by_name,
            allow_unique_unqualified=False,
        )

    def symbols_for_node(self, node: ast.AST) -> tuple[str, ...]:
        collector = ClassReferenceSymbolCollector(self)
        collector.visit(node)
        return sorted_tuple(collector.symbols)

    def symbol_for_reference(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Call):
            return self._direct_constructor_symbol(node)
        if isinstance(node, ast.Name):
            constructor_symbol = self.constructor_assignment_symbols.get(node.id)
            if constructor_symbol is not None:
                return constructor_symbol
        return self.reference_resolution.symbol_for_node(node)

    def _direct_constructor_symbol(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        return self.reference_resolution.symbol_for_node(node.func)


class ClassReferenceSymbolCollector(ast.NodeVisitor):
    """Collect expression nodes that reference classes without counting members."""

    def __init__(self, resolver: ModuleClassReferenceResolver) -> None:
        self.resolver = resolver
        self.symbols: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        self._add_symbol(self.resolver._direct_constructor_symbol(node))
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._add_symbol(self.resolver.reference_resolution.symbol_for_node(node))

    def visit_Name(self, node: ast.Name) -> None:
        self._add_symbol(self.resolver.symbol_for_reference(node))

    def _add_symbol(self, symbol: str | None) -> None:
        if symbol is not None:
            self.symbols.add(symbol)


@dataclass(frozen=True)
class ModuleIndexedClasses(ParsedModuleFact):
    """One module's class records before a family index resolves their bases. A
    fact of the parse, so an unchanged module is read once per plan, not once per
    index build."""

    records: tuple[IndexedClass, ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> ModuleIndexedClasses:
        indexed_class_nodes = iter_class_definitions(list(parsed_module.module.body))
        if not indexed_class_nodes:
            return cls(())
        if parsed_module.has_family_batch(CompactModuleClassProjectionFamily):
            (projection,) = parsed_module.family_batch(
                CompactModuleClassProjectionFamily
            ).items
            if len(projection.classes) == len(indexed_class_nodes):
                return cls(
                    tuple(
                        IndexedClass(
                            symbol=compact.symbol,
                            module_name=compact.module_name,
                            qualname=compact.qualname,
                            simple_name=compact.simple_name,
                            file_path=compact.file_path,
                            line=compact.line,
                            node=node,
                            declared_base_names=compact.declared_base_names,
                            resolved_base_symbols=compact.resolved_base_symbols,
                            dataclass_declaration=compact.dataclass_declaration,
                            class_decorators_are_promotion_safe=(
                                compact.class_decorators_are_promotion_safe
                            ),
                            registry_facts=compact.registry_facts,
                            declares_autoregister_meta=compact.declares_autoregister_meta,
                        )
                        for compact, (_qualname, node) in zip(
                            projection.classes, indexed_class_nodes, strict=True
                        )
                    )
                )
        binding_snapshots = parsed_module.fact(ClassLineBindingSnapshots).snapshots
        return cls(
            tuple(
                IndexedClass.from_parsed_class(
                    parsed_module, qualname, node, binding_snapshots[node.lineno]
                )
                for qualname, node in indexed_class_nodes
            )
        )


def build_class_family_index(modules: Sequence[ParsedModule]) -> ClassFamilyIndex:
    """Build one class-family index; the caller that owns the modules holds it."""

    return ClassFamilyIndexBuilder(tuple(modules)).build()


@dataclass(frozen=True)
class ClassFamilyIndexPredecessor:
    """The index a projection starts from and the parses it was built over,
    by path. A module that is the same parse in both states has the same class
    records, and its records' symbols were resolved from exactly those records:
    a symbol another module also declared would have been ambiguous in both."""

    index: ClassFamilyIndex
    modules_by_file_path: dict[str, ParsedModule]

    def resolution_of(
        self, record: IndexedClass, module: ParsedModule
    ) -> IndexedClass | None:
        """What the predecessor resolved this record to, when its module is the
        same parse there."""

        if self.modules_by_file_path.get(record.file_path) is not module:
            return None
        return self.index.classes_by_symbol.get(record.symbol)


@dataclass(frozen=True)
class ClassFamilyIndexBuilder:
    modules: tuple[ParsedModule, ...]
    previous: ClassFamilyIndexPredecessor | None = None

    def build(self) -> ClassFamilyIndex:
        class_records = tuple(
            UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
                self.module_class_records(),
                lambda record: record.symbol,
            ).values()
        )
        known_symbols = frozenset(record.symbol for record in class_records)
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex | None = None
        moved_terminal_names = self.moved_terminal_names(known_symbols)
        classes_by_symbol: dict[str, IndexedClass] = {}
        previous = self.previous
        module_by_file_path = self.module_by_file_path if previous is not None else {}
        prev_modules = previous.modules_by_file_path if previous is not None else {}
        prev_classes = previous.index.classes_by_symbol if previous is not None else {}
        for record in class_records:
            resolved: IndexedClass | None = None
            if previous is not None and prev_modules.get(record.file_path) is module_by_file_path[record.file_path]:
                prev_record = prev_classes.get(record.symbol)
                if prev_record is not None and (
                    not moved_terminal_names
                    or not any(
                        reference.resolved_parts[-1] in moved_terminal_names
                        for reference in record.registry_facts.base_references
                    )
                ):
                    resolved = prev_record
            if resolved is None:
                if unique_symbols_by_suffix is None:
                    unique_symbols_by_suffix = _unique_known_symbol_by_suffix(known_symbols)
                resolved = self.resolved_record(
                    record, known_symbols, unique_symbols_by_suffix
                )
            classes_by_symbol[record.symbol] = resolved
        metaclass_records = tuple(
            (symbol, record)
            for symbol, record in classes_by_symbol.items()
            if record.registry_facts.metaclass_references
        )
        need_metaclass_recheck = bool(metaclass_records)
        if need_metaclass_recheck and previous is not None:
            changed_paths = {
                module.file_path
                for module in self.modules
                if prev_modules.get(module.file_path) is not module
            }
            removed_paths = prev_modules.keys() - module_by_file_path.keys()
            changed_or_removed_classes = (
                *(
                    record
                    for module in self.modules
                    if module.file_path in changed_paths
                    for record in module.fact(ModuleIndexedClasses).records
                ),
                *(
                    record
                    for path in (*changed_paths, *removed_paths)
                    if path in prev_modules
                    for record in prev_modules[path].fact(ModuleIndexedClasses).records
                ),
            )
            if not any(
                record.registry_facts.stores_created_class
                or record.registry_facts.metaclass_references
                or any(
                    ref.resolved_parts[-1] == "type"
                    for ref in record.registry_facts.base_references
                )
                for record in changed_or_removed_classes
            ) and not any(
                record.file_path in changed_paths
                or (
                    moved_terminal_names
                    and any(
                        ref.resolved_parts[-1] in moved_terminal_names
                        for ref in record.registry_facts.metaclass_references
                    )
                )
                for _, record in metaclass_records
            ):
                need_metaclass_recheck = False
        if need_metaclass_recheck:
            authority = RegisteringMetaclassAuthority(
                classes_by_symbol,
                CompactRepositoryPublicExposureIndex(
                    tuple(
                        parsed_module.family_batch(CompactModuleClassProjectionFamily).items[0]
                        if parsed_module.has_family_batch(CompactModuleClassProjectionFamily)
                        else parsed_module.fact(ModuleClassHeaderCore).projection
                        for parsed_module in self.modules
                    )
                ),
                _suffix_index=unique_symbols_by_suffix,
            )
            for symbol, record in metaclass_records:
                classes_by_symbol[symbol] = record.with_registering_metaclass(authority)
        return ClassFamilyIndex(classes_by_symbol=classes_by_symbol)

    def moved_terminal_names(self, known_symbols: frozenset[str]) -> frozenset[str]:
        """The terminal names of symbols this build knows and the previous did
        not, or the reverse. A base reference resolves against the known
        symbols that end in its own terminal name and no others, so a record
        none of whose bases end in one of these resolves as it did."""

        if self.previous is None:
            return frozenset()
        return frozenset(
            symbol.rsplit(".", 1)[-1]
            for symbol in known_symbols ^ self.previous.index.known_symbols
        )

    def reused_record(
        self, record: IndexedClass, moved_terminal_names: frozenset[str]
    ) -> IndexedClass | None:
        """The previous build's resolution of this same record, when no changed
        symbol could resolve one of its bases differently."""

        if self.previous is None:
            return None
        previous = self.previous.resolution_of(
            record, self.module_by_file_path[record.file_path]
        )
        if previous is None:
            return None
        if any(
            reference.resolved_parts[-1] in moved_terminal_names
            for reference in record.registry_facts.base_references
        ):
            return None
        return previous

    @cached_property
    def module_by_file_path(self) -> dict[str, ParsedModule]:
        return {module.file_path: module for module in self.modules}

    def module_class_records(self) -> tuple[IndexedClass, ...]:
        return tuple(
            record
            for parsed_module in self.modules
            for record in parsed_module.fact(ModuleIndexedClasses).records
        )

    def resolved_record(
        self,
        record: IndexedClass,
        known_symbols: frozenset[str],
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex,
    ) -> IndexedClass:
        """One record's bases resolved against the build's one suffix index."""

        return record.with_resolved_base_symbols(
            tuple(
                resolved
                for reference in record.registry_facts.base_references
                if (
                    resolved := CompactClassFamilyIndexBuilder._resolved_bound_symbol(
                        reference,
                        record.module_name,
                        known_symbols,
                        unique_symbols_by_suffix,
                    )
                )
                is not None
            )
        )

