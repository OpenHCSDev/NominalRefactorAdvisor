"""The class-family index built from parsed modules: each class, its resolved bases and the families they form."""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

from nominal_refactor_advisor.ast_projection import ExpressionReference
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
from nominal_refactor_advisor.foreign_family import FunctionDefinition
from nominal_refactor_advisor.native_declarations import ClassNamespaceDeclaration


class DecoratedStatement:
    """A class or def statement, read for its name and the decorators it spells."""

    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef

    @property
    def name(self) -> str:
        return self.node.name

    @cached_property
    def decorator_names(self) -> frozenset[str]:
        """The last name of each decorator; a called one is read by what it calls.

        This reads names, not bindings: `abc.abstractmethod` and a local
        function named `abstractmethod` both read as `abstractmethod`.
        """

        names: set[str] = set()
        for decorator in self.node.decorator_list:
            while isinstance(decorator, ast.Call):
                decorator = decorator.func
            name = ExpressionReference.decoded(decorator).terminal_name
            if name is not None:
                names.add(name)
        return frozenset(names)

    def spells(self, decorator_name: str) -> bool:
        """Whether a decorator of this statement is spelled `decorator_name`."""

        return decorator_name in self.decorator_names


@dataclass(frozen=True)
class FunctionStatement(DecoratedStatement):
    """One def statement, each answer about its own source read once."""

    node: ast.FunctionDef | ast.AsyncFunctionDef


class ClassStatement(DecoratedStatement):
    """What one class statement's own source says, each answer read once.

    `IndexedClass` is a statement the family index names. Any other one is an
    `UnindexedClassStatement`, and its source answers the same questions.
    """

    node: ast.ClassDef

    @cached_property
    def methods(self) -> tuple[FunctionStatement, ...]:
        """The functions the body defines directly, in source order."""

        return tuple(
            FunctionStatement(node) for node in FunctionDefinition.within(self.node.body)
        )

    @cached_property
    def first_methods_by_name(self) -> dict[str, FunctionStatement]:
        found: dict[str, FunctionStatement] = {}
        for method in self.methods:
            found.setdefault(method.name, method)
        return found

    def method_named(self, name: str) -> FunctionStatement | None:
        """The body's first direct definition of `name`."""

        return self.first_methods_by_name.get(name)

    @cached_property
    def direct_assignments(self) -> dict[str, ast.AST | None]:
        """Each name the body assigns directly, with the value it gets last.

        A bare annotation assigns nothing and maps to None.
        """

        assignments: dict[str, ast.AST | None] = {}
        for statement in self.node.body:
            match statement:
                case ast.Assign(targets=[ast.Name(id=name)], value=value):
                    assignments[name] = value
                case ast.AnnAssign(target=ast.Name(id=name), value=value):
                    assignments[name] = value
        return assignments

    @cached_property
    def annotated_names(self) -> tuple[str, ...]:
        """Each name the body annotates directly, in source order."""

        return tuple(
            statement.target.id
            for statement in self.node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        )

    @cached_property
    def spelled_base_names(self) -> tuple[str, ...]:
        """The last name of each spelled base, in source order."""

        return tuple(
            name
            for base in self.node.bases
            if (name := ExpressionReference.decoded(base).terminal_name) is not None
        )

    @cached_property
    def base_terminal_names(self) -> tuple[str, ...]:
        """The spelled base names, sorted, each once."""

        return sorted_tuple(set(self.spelled_base_names))

    @cached_property
    def dataclass_decorator(self) -> ast.expr | None:
        """The first decorator spelled `dataclass`, bare or called, by any path.

        This reads names, not bindings. `ClassDeclaration.dataclass_declaration`
        is the binding-checked answer for a class the index names.
        """

        for decorator in self.node.decorator_list:
            spelled = decorator
            while isinstance(spelled, ast.Call):
                spelled = spelled.func
            match spelled:
                case ast.Name(id="dataclass") | ast.Attribute(attr="dataclass"):
                    return decorator
        return None

    @property
    def spells_dataclass(self) -> bool:
        return self.dataclass_decorator is not None

    def dataclass_flag(self, name: str) -> bool:
        """Whether the dataclass decorator is called with `name` set to a true constant."""

        match self.dataclass_decorator:
            case ast.Call(keywords=keywords):
                for keyword in keywords:
                    if keyword.arg == name:
                        return isinstance(keyword.value, ast.Constant) and bool(
                            keyword.value.value
                        )
        return False

    @cached_property
    def spells_abstract(self) -> bool:
        """Whether a base is spelled `ABC` or `ABCMeta`, or a direct method is
        decorated with something spelled `abstractmethod`.

        This reads names, not bindings. `ClassDeclaration.is_abstract` on the
        compact index is the binding-checked answer for the same question.
        """

        return bool({"ABC", "ABCMeta"} & set(self.base_terminal_names)) or any(
            method.spells("abstractmethod") for method in self.methods
        )


@dataclass(frozen=True)
class IndexedClass(ClassDeclaration, ClassNamespaceDeclaration, ClassStatement):
    node: ast.ClassDef

    member_binding_names = AliasProperty[frozenset[str]](
        "namespace_execution.binding_names"
    )

    @cached_property
    def namespace_execution(self) -> ClassNamespaceExecutionEvidence:
        return ClassNamespaceExecutionEvidence.from_class(self.node)

    @property
    def is_final(self) -> bool:
        return self.spells("final")

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


@dataclass(frozen=True)
class UnindexedClassStatement(ClassStatement):
    """A class statement the family index does not name.

    One written inside a function, or one a native syntax index parsed alone.
    """

    node: ast.ClassDef


@dataclass(frozen=True)
class ModuleClassStatements(ParsedModuleFact):
    """Every class statement of one module, one record each for the whole parse."""

    statements: tuple[ClassStatement, ...]
    """All of them, in `ast.walk` order."""
    top_level: tuple[ClassStatement, ...]
    """The ones the module body holds directly, in source order."""

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> ModuleClassStatements:
        indexed: dict[ast.ClassDef, ClassStatement] = {
            record.node: record
            for record in parsed_module.fact(ModuleIndexedClasses).records
        }
        statements = tuple(
            indexed.get(node) or UnindexedClassStatement(node)
            for node in parsed_module.syntax_index.instances_of(ast.ClassDef)
        )
        by_node = {statement.node: statement for statement in statements}
        return cls(
            statements,
            tuple(
                by_node[node]
                for node in parsed_module.module.body
                if isinstance(node, ast.ClassDef)
            ),
        )

    @cached_property
    def by_name(self) -> dict[str, ClassStatement]:
        """Each class name with the last statement declaring it in `ast.walk` order."""

        return {statement.node.name: statement for statement in self.statements}

    @cached_property
    def top_level_by_name(self) -> dict[str, ClassStatement]:
        """Each module-level class name with the last statement declaring it."""

        return {statement.node.name: statement for statement in self.top_level}

    @cached_property
    def child_names_by_base(self) -> dict[str, tuple[str, ...]]:
        """Each spelled base name with the classes that spell it, sorted."""

        children: dict[str, set[str]] = defaultdict(set)
        for name, statement in self.by_name.items():
            for base_name in statement.base_terminal_names:
                children[base_name].add(name)
        return {base: tuple(sorted(names)) for base, names in children.items()}

    def descendant_names(self, base_name: str) -> tuple[str, ...]:
        """The classes of this module reaching `base_name` through spelled bases,
        nearest first."""

        descendants: dict[str, None] = {}
        queue = deque(self.child_names_by_base.get(base_name, ()))
        while queue:
            current = queue.popleft()
            if current in descendants:
                continue
            descendants[current] = None
            queue.extend(
                child
                for child in self.child_names_by_base.get(current, ())
                if child not in descendants
            )
        return tuple(descendants)


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
        unique_symbols_by_suffix = _unique_known_symbol_by_suffix(known_symbols)
        moved_terminal_names = self.moved_terminal_names(known_symbols)
        classes_by_symbol: dict[str, IndexedClass] = {}
        for record in class_records:
            resolved = self.reused_record(record, moved_terminal_names)
            if resolved is None:
                resolved = self.resolved_record(
                    record, known_symbols, unique_symbols_by_suffix
                )
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

