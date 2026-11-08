"""The class-family index built from parsed modules: each class, its resolved bases and the families they form."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable

from ..ast_tools import (
    ParsedModule,
    ParsedModuleFact,
    ParsedModuleSet,
    ParsedModuleSetFact,
)
from ..class_index import (
    ModuleClassHeaderCore,
    ClassDeclaration,
    ClassDeclarationIndex,
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
    _module_nominal_binding_snapshots,
    _single_assignment_target_name,
    module_import_aliases,
)
from ..class_namespace import ClassNamespaceExecutionEvidence
from ..class_symbols import (
    ClassSymbolResolutionAuthority,
    _unique_known_symbol_by_suffix,
    _UniqueKnownSymbolSuffixIndex,
    iter_class_definitions,
)
from ..collection_algebra import (
    UniqueIdentityIndexAuthority,
    sorted_tuple,
)
from ..descriptor_algebra import AliasProperty
from ..native_declarations import ClassNamespaceDeclaration


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
            registry_facts=ClassRegistryFacts.of_class(
                node, module_binding_snapshot, parsed_module.module
            ),
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
    resolved_by_record: dict[int, tuple[IndexedClass, IndexedClass]] = field(
        default_factory=dict, repr=False, compare=False
    )
    """Each module class record this build resolved, by identity, with what it
    resolved to; the record is held so its identity stays its own. Identities
    mean nothing in another process, so a pickled index carries none."""

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["resolved_by_record"] = {}
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)

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
        return ClassFamilyIndexBuilder(projected_modules, previous=self).build()


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
        binding_snapshots = _module_nominal_binding_snapshots(
            ModuleNominalBindingAuthority(parsed_module),
            tuple(node.lineno for _qualname, node in indexed_class_nodes),
            include_final=False,
        )
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
class ClassFamilyIndexBuilder:
    modules: tuple[ParsedModule, ...]
    previous: ClassFamilyIndex | None = None

    def build(self) -> ClassFamilyIndex:
        class_records = tuple(
            UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
                self.module_class_records(),
                lambda record: record.symbol,
            ).values()
        )
        known_symbols = frozenset(record.symbol for record in class_records)
        unique_symbols_by_suffix = _unique_known_symbol_by_suffix(known_symbols)
        moved_terminal_names = self.moved_terminal_names(known_symbols)
        resolved_by_record: dict[int, tuple[IndexedClass, IndexedClass]] = {}
        classes_by_symbol: dict[str, IndexedClass] = {}
        for record in class_records:
            resolved = self.reused_record(record, moved_terminal_names)
            if resolved is None:
                resolved = self.resolved_record(
                    record, known_symbols, unique_symbols_by_suffix
                )
            resolved_by_record[id(record)] = (record, resolved)
            classes_by_symbol[record.symbol] = resolved
        authority = RegisteringMetaclassAuthority(
            classes_by_symbol,
            CompactRepositoryPublicExposureIndex(
                CompactModuleClassProjectionFamily.header_core_projections(self.modules)
            ),
        )
        for symbol, record in classes_by_symbol.items():
            if record.registry_facts.metaclass_references:
                classes_by_symbol[symbol] = record.with_registering_metaclass(authority)
        return ClassFamilyIndex(
            classes_by_symbol=classes_by_symbol,
            resolved_by_record=resolved_by_record,
        )

    def moved_terminal_names(self, known_symbols: frozenset[str]) -> frozenset[str]:
        """The terminal names of symbols this build knows and the previous did
        not, or the reverse. A base reference resolves against the known
        symbols that end in its own terminal name and no others, so a record
        none of whose bases end in one of these resolves as it did."""

        if self.previous is None:
            return frozenset()
        return frozenset(
            symbol.rsplit(".", 1)[-1]
            for symbol in known_symbols ^ self.previous.known_symbols
        )

    def reused_record(
        self, record: IndexedClass, moved_terminal_names: frozenset[str]
    ) -> IndexedClass | None:
        """The previous build's resolution of this same record, when no changed
        symbol could resolve one of its bases differently."""

        if self.previous is None:
            return None
        previous = self.previous.resolved_by_record.get(id(record))
        if previous is None or previous[0] is not record:
            return None
        if any(
            reference.resolved_parts[-1] in moved_terminal_names
            for reference in record.registry_facts.base_references
        ):
            return None
        return previous[1]

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

