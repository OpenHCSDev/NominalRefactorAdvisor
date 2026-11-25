"""Repository-wide class-family indexing helpers.

This module builds a lightweight cross-module view of declared classes and
their resolved inheritance edges. The index is intentionally conservative:
it resolves only import patterns and base expressions that can be recovered
reliably from the local AST.
"""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import inspect
import re
import sys
import typing
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import defaultdict, deque
from collections.abc import (
    Collection,
    Hashable,
    Mapping,
    Sequence,
)
from dataclasses import MISSING, Field, dataclass, field, fields, replace
from enum import StrEnum
from functools import (
    cache,
    cached_property,
    singledispatchmethod,
)
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    NamedTuple,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.annotation_semantics import (
    NOMINAL_ANNOTATION_SOURCE_AUTHORITY,
)
from nominal_refactor_advisor.ast_field_decoders import (
    TYPE_PARAMETERS,
    declared_annotations,
    variadic_parameters,
)
from nominal_refactor_advisor.ast_projection import ExpressionReference
from nominal_refactor_advisor.ast_projection import (
    CallKeywords,
    KeyT,
    NoStatementValue,
    StatementValue,
    StatementValueQuestion,
    StatementValues,
    SuppliedValue,
    single_name_targets,
)
from nominal_refactor_advisor.ast_tools import (
    CompactModuleIdentity,
    CollectedFamily,
    LexicalSyntaxScope,
    ModuleAnnotationEvaluationMode,
    ParsedModule,
    ParsedModuleFact,
    SourceModule,
    _walk_nodes,
    collect_family_items,
    statements_without_docstring,
)
from nominal_refactor_advisor.class_method_facets import (
    CompactClassMethod,
    DecoratorBindings,
    _compact_class_methods,
    _is_classmethod,
)
from nominal_refactor_advisor.field_order_summaries import FunctionOrderSummary
from nominal_refactor_advisor.class_references import (
    ClassReferenceRole,
    CompactClassReference,
    CompactScriptScope,
    reads_as_type,
    subclass_test_subject,
)
from nominal_refactor_advisor.class_symbols import (
    ClassSymbolResolutionAuthority,
    _UniqueKnownSymbolSuffixIndex,
    _referenced_bases,
    _terminal_reference_name,
    _unique_known_symbol_by_suffix,
    iter_class_definitions,
)
from nominal_refactor_advisor.class_syntax_facets import (
    CompactClassMemberDeclaration,
    CompactClassSyntaxFacets,
    CompactKeyedTableAxis,
    CompactNamedProjectionSurface,
    CompactSortedKeyCall,
    CompactSourceLocation,
    ModuleClassSyntaxFacets,
    ModuleClassSyntaxFacetsWithoutAutoregister,
    _class_nominal_reference_root_names,
    _class_reference_root_names,
    _compact_base_argument_sources,
    _compact_base_key_type_names,
    _compact_class_member_declarations,
    _compact_keyed_table_axes,
    _compact_named_projection_surfaces,
    _compact_sorted_key_calls,
    _trim_leading_docstring,
)
from nominal_refactor_advisor.collection_algebra import (
    IdentityHandleMultiplicityProjection,
    SequenceViews,
    UniqueIdentityIndexAuthority,
    sorted_tuple,
)
from nominal_refactor_advisor.class_mro import (
    ClassMroAuthority,
    NativeMroBase,
    DeclarationMroType,
    DeclaredBases,
    ProvedCreationBases,
)
from nominal_refactor_advisor.overlap_receipts import OverlapReceiptKind
from nominal_refactor_advisor.foreign_family import FunctionDefinition
from nominal_refactor_advisor.product_declarations import (
    CompactProductAuthorityFailure,
    CompactProductAuthorityViolation,
    CompactProductDeclarationFailure,
    ProductDeclarationSite,
    _dataclass_decorator_failures,
)
from nominal_refactor_advisor.public_exports import (
    CompactModulePublicExportContract,
    CompactPublicNameExposure,
    module_public_export_contract,
)
from nominal_refactor_advisor.registration_facets import (
    CompactManualSelectorAxis,
    CompactManualSubclassRosterRoot,
    FamilyWalkRoot,
    NoWalkRoot,
    _autoregister_registry_key_attr_name,
    _compact_manual_selector_axes,
    _compact_manual_subclass_roster_roots,
    _compact_predicate_selected_methods,
    _direct_class_assignments,
    _is_registration_authority,
    _string_constant,
    registry_config_walk_root,
)
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.source_entry import ImportedSourceModuleEntryPremise
from nominal_refactor_advisor.value_expression import (
    LexicalValueReference,
    MappingKeyResolverABC,
)
from nominal_refactor_advisor.declaration_dependencies import (
    ModuleLexicalDependencyProjection,
)
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.enum_semantics import (
    PYTHON_ENUM_BASE_AUTHORITY,
    NoEnumDeclaration,
    PythonEnumDeclarationFacet,
)
from nominal_refactor_advisor.export_tools import PYTHON_PUBLIC_EXPORT_ASSIGNMENT
from nominal_refactor_advisor.lexical_bindings import (
    ImportBoundNameProjection,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
)
from nominal_refactor_advisor.native_declarations import (
    DataclassRuntimeDeclaration as DataclassRuntimeDeclaration,
    NativeDeclaration,
    QualifiedDeclaration,
)
from nominal_refactor_advisor.native_syntax import NativePythonSyntaxIndex
from nominal_refactor_advisor.native_reference import NativeReferenceEnvironment
from nominal_refactor_advisor.read_sources import EmbeddedSource, EmbeddedSources
from nominal_refactor_advisor.product_flow import (
    SourceProductFlowProjection,
    source_product_flow_projection,
)
from nominal_refactor_advisor.class_creation_store import (
    AUTO_REGISTER_META,
    MODELLED_BINDINGS,
    MODELLED_METACLASSES,
    NO_BINDINGS,
    CreatedClassStore,
    CreationBindings,
    FunctionNode,
    _functions,
)
from nominal_refactor_advisor.installed_source import InstalledModule
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.semantic_algebra import DirectedGraph
from nominal_refactor_advisor.source_geometry import (
    ClassHeaderSourceSpan as ClassHeaderSourceSpan,
)


@dataclass(frozen=True)
class ModuleTopLevelFunctions(ParsedModuleFact):
    """Top-level functions of a module, indexed once per parse for class creation checks."""

    functions: Mapping[str, FunctionNode]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        return cls(_functions(parsed_module.module.body))


class ClassRegistryFacts(NamedTuple):
    """What one class statement says about registration, read from it and its module.

    `metaclass_references` and `base_references` are the `metaclass=` keyword
    and the bases, each with the binding its root name has where the class
    is declared. `stores_created_class` is whether, used as a metaclass, its
    creation path stores the class it creates into a mapping;
    `stores_subclass` is whether its `__init_subclass__` stores the subclass
    (`CreatedClassStore`). No class is recognized by its name, and no
    library's attribute names are consulted.
    """

    metaclass_references: tuple[CompactNominalReference, ...] = ()
    base_references: tuple[CompactNominalReference, ...] = ()
    stores_created_class: bool = False
    stores_subclass: bool = False
    keeps_bases: bool = True
    """Used as a metaclass, creation keeps the order C3 gives the declared bases
    (`CreatedClassStore.keeps_bases`)."""
    metaclass_bindings: CreationBindings = NO_BINDINGS
    """Used as a metaclass, what its creation binds on each class it creates."""
    subclass_hook_bindings: CreationBindings = NO_BINDINGS
    """Used as a base, what its `__init_subclass__` binds on each subclass."""
    module_stores_bases: bool = False
    """Its module stores some class's `__bases__` (`ModuleBasesStores`)."""
    walk_root: FamilyWalkRoot = NoWalkRoot()
    """Where the family the class declares looks for its members (`FamilyWalkRoot`)."""

    @classmethod
    def of_class(
        cls,
        node: ast.ClassDef,
        module_binding_snapshot: ModuleNominalBindingSnapshot,
        parsed_module: ParsedModule,
    ) -> ClassRegistryFacts:
        creation = CreatedClassStore.of_module_functions(
            node, parsed_module.fact(ModuleTopLevelFunctions).functions
        )
        return cls(
            metaclass_references=tuple(
                module_binding_snapshot.reference_for(parts)
                for keyword in node.keywords
                if keyword.arg == "metaclass"
                if (
                    parts := ExpressionReference.decoded(
                        ClassSymbolResolutionAuthority.reference_node(keyword.value)
                    ).attribute_chain
                )
                is not None
            ),
            base_references=_compact_base_references(node, module_binding_snapshot),
            stores_created_class=creation.stores_created_class(),
            stores_subclass=creation.stores_subclass(),
            keeps_bases=creation.keeps_bases(),
            metaclass_bindings=creation.metaclass_bindings(),
            subclass_hook_bindings=creation.subclass_hook_bindings(),
            module_stores_bases=bool(parsed_module.fact(ModuleBasesStores).lines),
            walk_root=registry_config_walk_root(parsed_module, node),
        )


class NotStandardDataclass(ValueError):
    """A class that is not a standard `dataclasses.dataclass` declaration."""


class DataclassFrozenness(StrEnum):
    """What a class's dataclass decorator declares about frozen instances.

    Python refuses a frozen dataclass whose dataclass bases are not frozen, and
    the reverse, so one dataclass mixin is composed only by dataclasses of one
    frozenness. A class no dataclass decorator declares takes no part in that
    check, and a non-literal `frozen=` leaves it unknown, which agrees with
    nothing.
    """

    FROZEN = "frozen"
    MUTABLE = "mutable"
    UNKNOWN = "unknown"
    UNDECLARED = "undeclared"

    @classmethod
    def of_keywords(cls, keywords: Iterable[ast.keyword]) -> DataclassFrozenness:
        """The frozenness a dataclass decorator call's `frozen=` keyword declares."""

        values = tuple(keyword.value for keyword in keywords if keyword.arg == "frozen")
        if not values:
            return cls.MUTABLE
        if len(values) > 1 or not (
            isinstance(values[0], ast.Constant) and isinstance(values[0].value, bool)
        ):
            return cls.UNKNOWN
        return cls.FROZEN if values[0].value else cls.MUTABLE

    @staticmethod
    def agree(frozenness: Iterable[DataclassFrozenness]) -> bool:
        """Whether classes of these frozennesses can compose one dataclass mixin."""

        declared = frozenset(frozenness) - {DataclassFrozenness.UNDECLARED}
        return len(declared) <= 1 and DataclassFrozenness.UNKNOWN not in declared


class CompactDataclassFacet(ABC):
    """A class's dataclass facet: a dataclass-like declaration, or none."""

    runtime_declaration: DataclassRuntimeDeclaration | None
    fields: tuple[CompactDataclassFieldDeclaration, ...]
    failures: tuple[CompactProductDeclarationFailure, ...]
    frozenness: DataclassFrozenness

    @property
    @abstractmethod
    def declared_by(self) -> str:
        """What declares the class, in a reason's words."""

    def with_settled_field_declarations(self, index: FieldDeclarationIndex) -> Self:
        settled = tuple(member.settled(index) for member in self.fields)
        if settled == self.fields:
            return self
        return replace(self, fields=settled)

    @abstractmethod
    def require_standard(self, reason: str) -> CompactDataclassDeclaration:
        """This standard dataclass declaration, or NotStandardDataclass."""
        raise NotImplementedError

    @property
    @abstractmethod
    def generates_construction(self) -> bool:
        """Whether a dataclass-like decorator builds the class's `__init__` from
        its fields, so an added field changes how the class is constructed."""
        raise NotImplementedError

    @abstractmethod
    def refuse_inherited_fields(self, heir_qualname: str) -> None:
        """An ancestor's dataclass declaration refuses a heir that adds the decorator."""
        raise NotImplementedError

    @abstractmethod
    def base_lineage_in(
        self,
        composer: _ProductLineageComposer,
        base_symbol: str,
        heir: CompactIndexedClass,
        pending_symbols: frozenset[str],
    ) -> _CompactProductLineage:
        """What this class contributes as the single direct base of a product."""
        raise NotImplementedError

    @abstractmethod
    def product_resolution_in(
        self,
        composer: _ProductLineageComposer,
        class_symbol: str,
        indexed_class: CompactIndexedClass,
    ) -> CompactProductAuthorityResolution:
        raise NotImplementedError


@dataclass(frozen=True)
class CompactDataclassDeclaration(CompactDataclassFacet):
    """One dataclass-like declaration, including fail-closed unknown semantics."""

    runtime_declaration: DataclassRuntimeDeclaration | None
    fields: tuple[CompactDataclassFieldDeclaration, ...]
    failures: tuple[CompactProductDeclarationFailure, ...] = ()
    frozenness: DataclassFrozenness = DataclassFrozenness.MUTABLE

    def require_standard(self, reason: str) -> CompactDataclassDeclaration:
        if (
            self.runtime_declaration is None
            or not self.runtime_declaration.is_dataclass_decorator
        ):
            raise NotStandardDataclass(reason)
        return self

    @property
    def generates_construction(self) -> bool:
        return True

    @property
    def declared_by(self) -> str:
        """What declares the class, in a reason's words."""

        if self.runtime_declaration is None:
            return "a dataclass-like decorator NRA does not know"
        return f"`{self.runtime_declaration.qualified_name}`"

    def refuse_inherited_fields(self, heir_qualname: str) -> None:
        raise ValueError(f"`{heir_qualname}` inherits dataclass fields")

    def base_lineage_in(
        self,
        composer: _ProductLineageComposer,
        base_symbol: str,
        heir: CompactIndexedClass,
        pending_symbols: frozenset[str],
    ) -> _CompactProductLineage:
        return composer.compose(
            base_symbol, self, pending_symbols | frozenset((heir.symbol,))
        )

    def product_resolution_in(
        self,
        composer: _ProductLineageComposer,
        class_symbol: str,
        indexed_class: CompactIndexedClass,
    ) -> CompactProductAuthorityResolution:
        return composer.resolve(class_symbol, indexed_class, self)


@dataclass(frozen=True)
class NoDataclassDeclaration(CompactDataclassFacet):
    """A class no dataclass-like decorator declares: no runtime, no fields."""

    runtime_declaration = None
    fields = ()
    failures = ()
    frozenness = DataclassFrozenness.UNDECLARED

    def require_standard(self, reason: str) -> CompactDataclassDeclaration:
        raise NotStandardDataclass(reason)

    @property
    def generates_construction(self) -> bool:
        return False

    @property
    def declared_by(self) -> str:
        """What declares the class, in a reason's words."""

        return "no dataclass decorator"

    def refuse_inherited_fields(self, heir_qualname: str) -> None:
        return None

    def base_lineage_in(
        self,
        composer: _ProductLineageComposer,
        base_symbol: str,
        heir: CompactIndexedClass,
        pending_symbols: frozenset[str],
    ) -> _CompactProductLineage:
        return _CompactProductLineage(
            (),
            (
                composer.failure(
                    heir, CompactProductAuthorityViolation.NON_DATACLASS_BASE
                ),
            ),
        )

    def product_resolution_in(
        self,
        composer: _ProductLineageComposer,
        class_symbol: str,
        indexed_class: CompactIndexedClass,
    ) -> CompactProductAuthorityResolution:
        return AbsentCompactProductAuthority(class_symbol)


@dataclass(frozen=True)
class ForeignBase:
    """A base outside the scan. One the standard library binds is asked directly
    whether it declares a name; any other may, so it counts as declaring it."""

    spelled: str

    STANDARD_MODULES: ClassVar[tuple[str, ...]] = (
        "builtins", "abc", "typing", "collections.abc", "enum",
    )

    def may_declare(self, name: str) -> bool:
        bare = self.spelled.split("[", 1)[0]
        module_name, _, attribute = bare.rpartition(".")
        for candidate in (module_name,) if module_name else self.STANDARD_MODULES:
            module = sys.modules.get(candidate)
            base = None if module is None else getattr(module, attribute, None)
            if isinstance(base, type) or hasattr(base, "__mro_entries__"):
                return hasattr(base, name) and getattr(base, name) is not getattr(object, name, None)
        return True


@dataclass(frozen=True)
class ClassDeclaration(QualifiedDeclaration):
    """Source-form-independent identity shared by repository class indexes."""

    symbol: str
    module_name: str
    qualname: str
    simple_name: str
    file_path: str
    line: int
    declared_base_names: tuple[str, ...]
    resolved_base_symbols: tuple[str, ...] = field(default=(), kw_only=True)
    dataclass_declaration: CompactDataclassFacet = field(
        default_factory=NoDataclassDeclaration,
        kw_only=True,
    )
    class_decorators_are_promotion_safe: bool = field(default=True, kw_only=True)
    registry_facts: ClassRegistryFacts = field(
        default=ClassRegistryFacts(), kw_only=True
    )
    declares_autoregister_meta: bool = field(default=False, kw_only=True)
    """Whether its metaclass registers the classes it creates; set by the index."""

    qualified_name = AliasProperty[str]("symbol")

    @property
    def member_binding_names(self) -> frozenset[str]:
        """Every name the class body binds; each index's declarations read it from their source."""

        raise NotImplementedError(type(self).__name__)

    @property
    def unresolved_base_names(self) -> tuple[str, ...]:
        """The spelled bases the index could not bind to a scanned class."""

        if len(self.resolved_base_symbols) == len(self.declared_base_names):
            return ()
        resolved = {symbol.rsplit(".", 1)[-1] for symbol in self.resolved_base_symbols}
        return tuple(
            spelled
            for spelled in self.declared_base_names
            if spelled.split("[", 1)[0].rsplit(".", 1)[-1] not in resolved
        )

    def with_resolved_base_symbols(
        self,
        resolved_base_symbols: tuple[str, ...],
    ) -> Self:
        if resolved_base_symbols == self.resolved_base_symbols:
            return self
        return replace(self, resolved_base_symbols=resolved_base_symbols)

    def with_registering_metaclass(
        self, authority: RegisteringMetaclassAuthority[Any]
    ) -> Self:
        registers = authority.registers_remembered(self)
        if registers == self.declares_autoregister_meta:
            return self
        return replace(self, declares_autoregister_meta=registers)


@dataclass(frozen=True)
class CompactClassValueConstruction:
    """One class-owned construction of a nominal value declaration."""

    assigned_name: str
    constructor_name: str
    keyword_names: tuple[str, ...]
    line: int


class CompactNominalReference(ABC):
    """One source reference; its subtype says whether a declaration-time binding roots it."""

    source_parts: tuple[str, ...]

    @staticmethod
    def bound_as(
        parts: tuple[str, ...], binding: CompactNominalBinding | None
    ) -> CompactNominalReference:
        """The binding lookup's answer, read once, picks the reference kind."""

        if binding is None:
            return UnboundNominalReference(parts)
        return BoundNominalReference(parts, binding)

    @property
    @abstractmethod
    def resolved_parts(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def qualified_name(self) -> str:
        return ".".join(self.resolved_parts)

    @property
    def simple_name(self) -> str:
        return self.resolved_parts[-1]

    @abstractmethod
    def sort_into(
        self,
        bound: list[BoundNominalReference],
        unbound: list[UnboundNominalReference],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def settled_qualified_name(
        self, authority: RegisteringMetaclassAuthority[Any], module_name: str
    ) -> str | None:
        """The bound name, or what the module's star imports settle the spelling to."""
        raise NotImplementedError

    @abstractmethod
    def shares_binding_with(self, other: CompactNominalReference) -> bool:
        raise NotImplementedError

    @abstractmethod
    def has_bound_name(self, qualified_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def names_builtin(self, expected_parts: tuple[str, ...]) -> bool:
        """Whether a binding proves this reference names exactly that builtin."""
        raise NotImplementedError

    @property
    @abstractmethod
    def builtin_qualified_name(self) -> str:
        """The qualified name, reading an unbound builtin spelling as `builtins.X`."""
        raise NotImplementedError

    @abstractmethod
    def known_symbol_beyond(
        self,
        candidate: str,
        module_name: str,
        known_symbols: Collection[str],
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex,
    ) -> str | None:
        """The known class symbol when the resolved candidate itself is not one."""
        raise NotImplementedError


@dataclass(frozen=True)
class BoundNominalReference(CompactNominalReference):
    source_parts: tuple[str, ...]
    root_binding: CompactNominalBinding

    @property
    def resolved_parts(self) -> tuple[str, ...]:
        return (
            *self.root_binding.qualified_name.split("."),
            *self.source_parts[1:],
        )

    def sort_into(
        self,
        bound: list[BoundNominalReference],
        unbound: list[UnboundNominalReference],
    ) -> None:
        bound.append(self)

    def settled_qualified_name(
        self, authority: RegisteringMetaclassAuthority[Any], module_name: str
    ) -> str | None:
        return self.qualified_name

    def shares_binding_with(self, other: CompactNominalReference) -> bool:
        return other.has_bound_name(self.qualified_name)

    def has_bound_name(self, qualified_name: str) -> bool:
        return self.qualified_name == qualified_name

    def names_builtin(self, expected_parts: tuple[str, ...]) -> bool:
        return self.resolved_parts[0] == "builtins" and self.resolved_parts == expected_parts

    @property
    def builtin_qualified_name(self) -> str:
        return self.qualified_name

    def known_symbol_beyond(
        self,
        candidate: str,
        module_name: str,
        known_symbols: Collection[str],
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex,
    ) -> str | None:
        if self.root_binding.kind.projects_as_import_alias:
            return unique_symbols_by_suffix.root_relative_match(candidate)
        return None


@dataclass(frozen=True)
class UnboundNominalReference(CompactNominalReference):
    """A reference no binding in scope roots: only its source spelling is known."""

    source_parts: tuple[str, ...]

    @property
    def resolved_parts(self) -> tuple[str, ...]:
        return self.source_parts

    def sort_into(
        self,
        bound: list[BoundNominalReference],
        unbound: list[UnboundNominalReference],
    ) -> None:
        unbound.append(self)

    def settled_qualified_name(
        self, authority: RegisteringMetaclassAuthority[Any], module_name: str
    ) -> str | None:
        return authority.star_settled_name(self, module_name)

    def shares_binding_with(self, other: CompactNominalReference) -> bool:
        return False

    def has_bound_name(self, qualified_name: str) -> bool:
        return False

    def names_builtin(self, expected_parts: tuple[str, ...]) -> bool:
        return False

    @property
    def builtin_qualified_name(self) -> str:
        parts = self.source_parts
        if len(parts) == 1 and hasattr(builtins, parts[0]):
            return f"{builtins.__name__}.{parts[0]}"
        return self.qualified_name

    def known_symbol_beyond(
        self,
        candidate: str,
        module_name: str,
        known_symbols: Collection[str],
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex,
    ) -> str | None:
        module_local = ".".join((module_name, *self.source_parts))
        return module_local if module_local in known_symbols else None


MemberAnswerT = TypeVar("MemberAnswerT")


@dataclass(frozen=True)
class CompactCallableMemberDeclaration(ABC):
    """One `def` a class body binds: a method or a property, with what it declares it returns.

    A property's return annotation is the type of reading the member; a
    method's is the type of calling it. A def with no return annotation
    declares neither.
    """

    name: str
    line: int

    @abstractmethod
    def answer(
        self,
        owner: CompactIndexedClass,
        answers: ClassMemberAnswersABC[MemberAnswerT],
    ) -> MemberAnswerT:
        raise NotImplementedError


@dataclass(frozen=True)
class CompactAnnotatedProperty(CompactCallableMemberDeclaration):
    """`@property` or `@cached_property` with a return annotation: reading it has that type."""

    return_annotation_expression: str

    def answer(self, owner: CompactIndexedClass, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.annotated_property(owner, self)


@dataclass(frozen=True)
class CompactAnnotatedMethod(CompactCallableMemberDeclaration):
    """A method with a return annotation: calling it has that type."""

    return_annotation_expression: str

    def answer(self, owner: CompactIndexedClass, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.annotated_method(owner, self)


@dataclass(frozen=True)
class CompactUnannotatedCallable(CompactCallableMemberDeclaration):
    """A method or property that declares no return type."""

    def answer(self, owner: CompactIndexedClass, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.unannotated_callable(owner, self)


class ClassMemberAnswersABC(ABC, Generic[MemberAnswerT]):
    """What a consumer makes of each outcome of looking a member up through a class's MRO."""

    @abstractmethod
    def assigned(
        self, owner: CompactIndexedClass, declaration: CompactClassMemberDeclaration
    ) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def annotated_property(
        self, owner: CompactIndexedClass, declaration: CompactAnnotatedProperty
    ) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def annotated_method(
        self, owner: CompactIndexedClass, declaration: CompactAnnotatedMethod
    ) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def unannotated_callable(
        self, owner: CompactIndexedClass, declaration: CompactUnannotatedCallable
    ) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def missing(self) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def ambiguous(self) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def open(self) -> MemberAnswerT:
        raise NotImplementedError


class ClassMemberLookup(ABC):
    """A member name looked up through a class's MRO: its one binding, or why there is none."""

    @abstractmethod
    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        raise NotImplementedError


@dataclass(frozen=True)
class AssignedClassMember(ClassMemberLookup):
    """The first class in the MRO binding the name assigns or annotates it in its body."""

    owner: CompactIndexedClass
    declaration: CompactClassMemberDeclaration

    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.assigned(self.owner, self.declaration)


@dataclass(frozen=True)
class CallableClassMember(ClassMemberLookup):
    """The first class in the MRO binding the name defines it with `def`."""

    owner: CompactIndexedClass
    declaration: CompactCallableMemberDeclaration

    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return self.declaration.answer(self.owner, answers)


class MissingClassMember(ClassMemberLookup):
    """No class of the MRO binds the name in its body."""

    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.missing()


class AmbiguousClassMember(ClassMemberLookup):
    """The first class binding the name binds it more than once."""

    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.ambiguous()


class OpenClassMember(ClassMemberLookup):
    """The class does not bind the name and its MRO is open, so no ancestor is known to."""

    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.open()


@dataclass(frozen=True)
class CompactAnnotatedFunctionReturn:
    """A module-level `def` and the return type it declares."""

    name: str
    line: int
    return_annotation_expression: str


class CompactDataclassFieldRole(StrEnum):
    """Dataclass field kinds with declaration-owned product admissibility."""

    STORED_INIT = "stored_init", True, True, None, (), True
    CLASS_VARIABLE = "class_variable", False, False, None, ("typing.ClassVar",), False
    KEYWORD_ONLY_SENTINEL = (
        "keyword_only_sentinel",
        False,
        False,
        None,
        ("dataclasses.KW_ONLY",),
        False,
    )
    STORED_NON_INIT = (
        "stored_non_init",
        False,
        True,
        CompactProductAuthorityViolation.NON_INIT_FIELD,
        (),
        False,
    )
    INIT_ONLY = (
        "init_only",
        False,
        True,
        CompactProductAuthorityViolation.INIT_ONLY_FIELD,
        ("dataclasses.InitVar",),
        True,
    )
    UNRESOLVED = (
        "unresolved",
        False,
        False,
        CompactProductAuthorityViolation.UNRESOLVED_FIELD_ROLE,
        (),
    )
    DECLARED_BY_CALL = (
        "declared_by_call",
        False,
        False,
        CompactProductAuthorityViolation.UNRESOLVED_FIELD_ROLE,
        (),
    )
    """A value that calls something, or names something, the module cannot
    judge alone; the repository's field declarations settle it to a role."""

    contributes_stored_init_field: bool
    contributes_semantic_field: bool
    authority_violation: CompactProductAuthorityViolation | None
    annotation_qualified_names: tuple[str, ...]
    init_parameter: bool | None
    """Whether the generated `__init__` takes the field as a parameter; None
    while the role is unresolved."""

    def __new__(
        cls,
        value: str,
        contributes_stored_init_field: bool,
        contributes_semantic_field: bool,
        authority_violation: CompactProductAuthorityViolation | None,
        annotation_qualified_names: tuple[str, ...],
        init_parameter: bool | None = None,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.contributes_stored_init_field = contributes_stored_init_field
        member.contributes_semantic_field = contributes_semantic_field
        member.authority_violation = authority_violation
        member.annotation_qualified_names = annotation_qualified_names
        member.init_parameter = init_parameter
        return member

    @classmethod
    def for_qualified_annotation(
        cls,
        qualified_name: str | None,
    ) -> Self | None:
        return next(
            (role for role in cls if qualified_name in role.annotation_qualified_names),
            None,
        )

    def declaration_call(
        self,
        statement: ast.AnnAssign,
        module_binding_snapshot: ModuleNominalBindingSnapshot,
        class_aliases: dict[str, str],
        preceding_class_bound_names: frozenset[str],
    ) -> CompactFieldDeclarationCall | None:
        """The call or reference a member's value makes, kept for the repository to judge."""

        value = statement.value
        if self is not CompactDataclassFieldRole.DECLARED_BY_CALL or value is None:
            return None
        callee = _class_scope_qualified_import_name(
            module_binding_snapshot,
            class_aliases,
            value.func if isinstance(value, ast.Call) else value,
            preceding_class_bound_names,
        )
        if callee is None:
            return None
        if isinstance(value, ast.Call):
            return CompactFieldDeclarationCall.of_call(callee, value)
        return CompactFieldDeclarationCall.of_reference(callee)

    def binding_of_later_fields(
        self, binding: DataclassInitBinding
    ) -> DataclassInitBinding:
        """How the fields after this member bind: a `KW_ONLY` sentinel makes
        them keyword-only; any other member leaves the binding as it was."""

        if self is CompactDataclassFieldRole.KEYWORD_ONLY_SENTINEL:
            return DataclassInitBinding.KEYWORD_ONLY
        return binding

    def takes_init_parameter(self, field: str) -> bool:
        """Whether the generated `__init__` takes `field` as a parameter. Raises
        while the role is unresolved, naming the field."""

        if self.init_parameter is None:
            raise ValueError(f"{field} has an unresolved role")
        return self.init_parameter

    @property
    def has_known_field_position(self) -> bool:
        """Whether this member's place in the dataclass field order is known.

        A KW_ONLY sentinel turns every later field keyword-only, and an
        unresolved annotation may or may not declare a field.
        """

        return self not in (
            CompactDataclassFieldRole.KEYWORD_ONLY_SENTINEL,
            CompactDataclassFieldRole.UNRESOLVED,
            CompactDataclassFieldRole.DECLARED_BY_CALL,
        )


class DataclassInitBinding(StrEnum):
    """How a dataclass field binds as an `__init__` parameter.

    `kw_only` comes from the field's own `field(kw_only=...)`, else from a
    preceding `KW_ONLY` sentinel, else from the decorator's `kw_only=`. A value
    that is not a literal bool leaves the binding unknown.
    """

    POSITIONAL = "positional"
    KEYWORD_ONLY = "keyword_only"
    UNKNOWN = "unknown"

    @classmethod
    def of_keywords(
        cls, keywords: Iterable[ast.keyword], otherwise: DataclassInitBinding
    ) -> DataclassInitBinding:
        """The binding a call's `kw_only=` keyword declares, or `otherwise`."""

        values = tuple(keyword.value for keyword in keywords if keyword.arg == "kw_only")
        if not values:
            return otherwise
        if len(values) > 1 or not (
            isinstance(values[0], ast.Constant) and isinstance(values[0].value, bool)
        ):
            return cls.UNKNOWN
        return cls.KEYWORD_ONLY if values[0].value else cls.POSITIONAL


class CompactDataclassFieldDeclaration(NamedTuple):
    """One direct annotated member, its exact dataclass role and its binding,
    with the call its value makes while the repository has yet to judge it."""

    name: str
    line: int
    role: CompactDataclassFieldRole
    binding: DataclassInitBinding
    declaration_call: CompactFieldDeclarationCall | None = None

    def binds_by_position(self, class_symbol: str) -> bool:
        """Whether a positional call to `class_symbol`'s generated `__init__`
        binds this field. Raises when its role or keyword-only-ness is unknown."""

        field = f"`{class_symbol}`'s field `{self.name}`"
        takes_parameter = self.role.takes_init_parameter(field)
        if self.binding is DataclassInitBinding.UNKNOWN:
            raise ValueError(f"{field} is keyword-only by a non-literal")
        return takes_parameter and self.binding is DataclassInitBinding.POSITIONAL

    def settled(self, index: FieldDeclarationIndex) -> CompactDataclassFieldDeclaration:
        """The member once the repository says whether its call declares a field."""

        if self.declaration_call is None:
            return self
        return self._replace(
            role=index.field_role(self.declaration_call), declaration_call=None
        )


@dataclass(frozen=True)
class CompactProductField:
    """One effective dataclass field retaining its nominal declaration source."""

    name: str
    role: CompactDataclassFieldRole
    binding: DataclassInitBinding
    declaring_class_symbol: str
    file_path: str
    line: int


@dataclass(frozen=True, kw_only=True)
class CompactProductAuthority(ProductDeclarationSite):
    """One linearly composed dataclass schema with source-visible fields."""

    effective_fields: tuple[CompactProductField, ...]

    @property
    def fields(self) -> tuple[CompactProductField, ...]:
        return tuple(
            field
            for field in self.effective_fields
            if field.role.contributes_stored_init_field
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)


@dataclass(frozen=True)
class CompactProductAuthorityResolution(ABC):
    """Nominal result of composing one class into a product authority."""

    class_symbol: str

    @property
    @abstractmethod
    def authority(self) -> CompactProductAuthority | None:
        raise NotImplementedError


@dataclass(frozen=True)
class AbsentCompactProductAuthority(CompactProductAuthorityResolution):
    """A class declaration which does not claim dataclass product semantics."""

    @property
    def authority(self) -> None:
        return None


@dataclass(frozen=True)
class OpenCompactProductAuthority(CompactProductAuthorityResolution):
    """A dataclass-like declaration with unresolved product obligations."""

    failures: tuple[CompactProductAuthorityFailure, ...]

    @property
    def authority(self) -> None:
        return None


@dataclass(frozen=True)
class ResolvedCompactProductAuthority(CompactProductAuthorityResolution):
    """A dataclass schema proven by a complete linear declaration chain."""

    resolved_authority: CompactProductAuthority

    @property
    def authority(self) -> CompactProductAuthority:
        return self.resolved_authority


@dataclass(frozen=True)
class _CompactProductLineage:
    """Internal source-carrying schema before target-specific admissibility."""

    effective_fields: tuple[CompactProductField, ...]
    failures: tuple[CompactProductAuthorityFailure, ...]


@dataclass(frozen=True)
class NominalReferences:
    """References split into those a declaration-time binding roots and the rest."""

    bound: tuple[BoundNominalReference, ...]
    unbound: tuple[UnboundNominalReference, ...]

    @classmethod
    def of(cls, references: Iterable[CompactNominalReference]) -> NominalReferences:
        bound: list[BoundNominalReference] = []
        unbound: list[UnboundNominalReference] = []
        for reference in references:
            reference.sort_into(bound, unbound)
        return cls(tuple(bound), tuple(unbound))

    @property
    def bound_names(self) -> tuple[str, ...]:
        return tuple(reference.qualified_name for reference in self.bound)


@dataclass(frozen=True)
class CompactClassBindingFacets:
    """Class facts whose authority depends on one module binding snapshot."""

    base_references: tuple[CompactNominalReference, ...]
    base_references_are_complete: bool
    class_decorators_are_promotion_safe: bool
    dataclass_declaration: CompactDataclassFacet
    enum_declaration: PythonEnumDeclarationFacet

    @property
    def product_base_bindings_are_exact(self) -> bool:
        return not NominalReferences.of(
            reference
            for reference in self.base_references
            if ClassSymbolResolutionAuthority.establishes_nominal_family(
                ".".join(reference.source_parts)
            )
        ).unbound


@dataclass(frozen=True)
class CompactStarResolvedFacets:
    """A class's binding-sensitive facts as they are when the module's star
    imports bind none of the names the class reads.

    A module with `from x import *` cannot tell, alone, whether a later name is
    the star import's or its own: its projection keeps the conservative facts.
    These are the facts with every name in `root_names` taken as the module's
    own. The class index uses them when the repository's export contracts
    prove that every star origin excludes every one of those names.
    """

    root_names: frozenset[str]
    facets: CompactClassBindingFacets


@dataclass(frozen=True)
class CompactClassHeader(ClassDeclaration):
    """Class-index surface sufficient for inheritance reconstruction."""

    base_references: tuple[CompactNominalReference, ...]
    base_key_type_names: tuple[str | None, ...] = ()
    """The key type each base reference's subscript names (`Family[Key]`), aligned
    with `base_references`; the family index reads a registry family's key from it."""
    direct_base_count: int = 0
    base_references_are_complete: bool = False
    product_base_bindings_are_exact: bool = False
    is_final: bool = False
    mro_bases_are_static: bool = False
    star_resolution: CompactStarResolvedFacets | None = field(default=None, kw_only=True)
    """The facts past the module's star imports, when it has any; the index
    settles them once the repository's exports are known."""

    @property
    def base_bindings(self) -> NominalReferences:
        return NominalReferences.of(self.base_references)

    @property
    def base_resolution_is_complete(self) -> bool:
        """Return whether every domain-bearing base resolves in the compact graph."""

        return self.base_references_are_complete and len(
            self.resolved_base_symbols
        ) == declared_nominal_base_count(self)


@dataclass(frozen=True)
class CompactIndexedClass(CompactClassHeader):
    """AST-free class declaration used to reconstruct inheritance globally."""

    @property
    def member_binding_names(self) -> frozenset[str]:
        """Every name the class body binds, by assignment or by `def`."""

        return frozenset(
            (
                *(declaration.name for declaration in self.direct_member_declarations),
                *(declaration.name for declaration in self.callable_member_declarations),
            )
        )

    def with_settled_field_declarations(self, index: FieldDeclarationIndex) -> Self:
        """The class once the repository judges the calls its fields' values make."""

        settled = self.dataclass_declaration.with_settled_field_declarations(index)
        if settled is self.dataclass_declaration:
            return self
        return replace(self, dataclass_declaration=settled)

    def resolved_past_star_imports(
        self, exposure: RepositoryPublicExposureAuthority
    ) -> Self:
        """The class with its star-resolved facts when every star origin of its
        module excludes every name they read; otherwise its conservative facts."""

        resolution = self.star_resolution
        if resolution is None:
            return self
        if not all(
            exposure.star_imports_exclude(self.module_name, name)
            for name in resolution.root_names
        ):
            return replace(self, star_resolution=None)
        facets = resolution.facets
        return replace(
            self,
            base_references=facets.base_references,
            base_references_are_complete=facets.base_references_are_complete,
            product_base_bindings_are_exact=facets.product_base_bindings_are_exact,
            class_decorators_are_promotion_safe=(
                facets.class_decorators_are_promotion_safe
            ),
            dataclass_declaration=facets.dataclass_declaration,
            enum_declaration=facets.enum_declaration,
            star_resolution=None,
        )

    def with_registering_metaclass(
        self, authority: RegisteringMetaclassAuthority[Any]
    ) -> Self:
        """Its own metaclass, whether it is a registry family, and its registry family's key type."""

        registers = authority.registers_remembered(self)
        is_authority = (
            self.is_registration_authority or authority.is_registry_family(self)
        )
        key_type_name = authority.registry_family_key_type_name(self)
        if (
            registers == self.declares_autoregister_meta
            and is_authority == self.is_registration_authority
            and key_type_name == self.keyed_family_key_type_name
        ):
            return self
        return replace(
            self,
            declares_autoregister_meta=registers,
            is_registration_authority=is_authority,
            keyed_family_key_type_name=key_type_name,
        )

    @property
    def declares_keyed_base(self) -> bool:
        """Whether a base names a key type (`Family[Key]`): a registry family's key, once the index resolves the base."""

        return any(name is not None for name in self.base_key_type_names)

    @property
    def own_dataclass_field_names(self) -> tuple[str, ...] | None:
        """The class's own dataclass fields in declaration order, ClassVars excluded;
        empty for a plain class, None when a field's position is unknown."""

        declaration = self.dataclass_declaration
        if not all(field.role.has_known_field_position for field in declaration.fields):
            return None
        return tuple(
            field.name
            for field in declaration.fields
            if field.role.contributes_semantic_field
        )

    direct_member_declarations: tuple[CompactClassMemberDeclaration, ...] = ()
    metaclass_names: tuple[str, ...] = ()
    class_keyword_names: tuple[str, ...] = ()
    class_header_is_reconstructible: bool = True
    keyed_family_key_type_name: str | None = None
    """The key type of the first subscripted base that is a registry family; set by the index."""
    end_line: int | None = None
    method_names: tuple[str, ...] = ()
    callable_member_declarations: tuple[CompactCallableMemberDeclaration, ...] = ()
    base_argument_sources: tuple[tuple[str, ...], ...] = ()
    """Each referenced base's subscript arguments (`Base[A, B]`), aligned with its base references."""
    type_parameter_names: tuple[str, ...] = ()
    """The class's own `class C[T]:` type parameters."""
    abstract_method_names: tuple[str, ...] = ()
    is_abstract: bool = False
    is_registration_authority: bool = False
    """Named or configured as a registration authority; the index adds a registering metaclass."""
    autoregister_registry_key_attr_name: str | None = None
    autoregister_registry_projection_names: tuple[str, ...] = ()
    keyed_registry_lookup_method_names: tuple[str, ...] = ()
    keyed_registry_reverse_lookup_method_names: tuple[str, ...] = ()
    predicate_selected_methods: tuple[tuple[int, str, str, str], ...] = ()
    enum_declaration: PythonEnumDeclarationFacet = NoEnumDeclaration()
    body_declarations: tuple[ClassBodyDeclarationDigest, ...] = ()

    @property
    def implicit_base_names(self) -> tuple[str, ...]:
        """What creation appends to the bases the statement names: `class C[T]:`
        ends its bases with `Generic`, unless it names `Protocol`, which is generic itself."""

        protocol = f"{typing.Protocol.__module__}.{typing.Protocol.__qualname__}"
        if not self.type_parameter_names or any(
            reference.qualified_name == protocol for reference in self.base_references
        ):
            return ()
        return (f"{typing.Generic.__module__}.{typing.Generic.__qualname__}",)

    @property
    def has_class_creation_hook(self) -> bool:
        """Class creation can invoke hooks declared as methods or assigned values."""

        return not CLASS_METHOD_OWNERSHIP_HOOK_NAMES.isdisjoint(
            (
                *self.method_names,
                *(
                    member.name
                    for member in self.direct_member_declarations
                    if member.expression is not None
                ),
            )
        )

    @property
    def direct_enum_member_names(self) -> tuple[str, ...]:
        """Return public members declared by a direct Python enum owner."""

        if not PYTHON_ENUM_BASE_AUTHORITY.matches_any(self.declared_base_names):
            return ()
        return PYTHON_ENUM_BASE_AUTHORITY.declared_member_names(
            (
                declaration.name,
                declaration.expression is not None,
            )
            for declaration in self.direct_member_declarations
        )

    @property
    def assignments_by_name(self) -> dict[str, str | None]:
        return {
            name: declaration.expression
            for name, declaration in self.direct_value_writes_by_name.items()
        }

    @cached_property
    def direct_value_writes_by_name(self) -> dict[str, CompactClassMemberDeclaration]:
        """Latest direct RHS writes, retaining their original declaration rows."""
        writes: dict[str, CompactClassMemberDeclaration] = {}
        for declaration in self.direct_member_declarations:
            declaration.value.record_write(writes, declaration.name, declaration)
        return writes

    @cached_property
    def direct_annotation_writes_by_name(
        self,
    ) -> dict[str, CompactClassMemberDeclaration]:
        """Annotation writes are independent of subsequent unannotated values."""
        return {
            declaration.name: declaration
            for declaration in self.direct_member_declarations
            if declaration.annotation_expression is not None
        }

    @cached_property
    def direct_declared_member_names(self) -> frozenset[str]:
        return frozenset(
            declaration.name for declaration in self.direct_member_declarations
        )

    def constant_string_assignment(self, name: str) -> str | None:
        declaration = self.direct_value_writes_by_name.get(name)
        return None if declaration is None else declaration.value.constant_string

    @property
    def direct_assignment_expressions(self) -> tuple[tuple[str, str | None], ...]:
        return tuple(self.assignments_by_name.items())

    @property
    def declared_member_lines_by_name(self) -> dict[str, int]:
        lines: dict[str, int] = {}
        for declaration in self.direct_member_declarations:
            lines.setdefault(declaration.name, declaration.line)
        return lines

    @property
    def direct_value_constructions(self) -> tuple[CompactClassValueConstruction, ...]:
        return tuple(
            CompactClassValueConstruction(
                assigned_name=declaration.name,
                constructor_name=declaration.constructor_name,
                keyword_names=declaration.constructor_keyword_names,
                line=declaration.line,
            )
            for declaration in self.direct_value_writes_by_name.values()
            if declaration.constructor_name is not None
        )

    @property
    def direct_constant_string_assignments(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (declaration.name, declaration.value.constant_string)
            for declaration in self.direct_value_writes_by_name.values()
            if declaration.value.constant_string is not None
        )

    @property
    def direct_non_none_assignment_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            declaration.name
            for declaration in self.direct_value_writes_by_name.values()
            if not declaration.value.value_is_none_literal
        )


class StarNameSettlement(ABC):
    """What the star imports seen so far (latest first) do to one module binding."""

    @staticmethod
    def unshadowed(binding: str | None) -> StarNameSettlement:
        """The ast-side lookup, read once: the module's own binding, if it has one."""

        if binding is None:
            return NameUnbound()
        return NameUntouched(binding)

    @abstractmethod
    def or_settled_by(
        self,
        origin: CompactModuleStarImportOrigin,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
    ) -> StarNameSettlement:
        """Fold in an earlier star import; only an untouched name lets it decide."""
        raise NotImplementedError

    @abstractmethod
    def sort_into(
        self,
        key: KeyT,
        settled: list[tuple[KeyT, str]],
        unsettled: list[KeyT],
    ) -> None:
        raise NotImplementedError


class UntouchedSettlement(StarNameSettlement):
    """No star import seen so far exposes the name: an earlier one may still decide."""

    def or_settled_by(
        self,
        origin: CompactModuleStarImportOrigin,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
    ) -> StarNameSettlement:
        return origin.settlement_in(authority, name, self)


@dataclass(frozen=True)
class NameUntouched(UntouchedSettlement):
    """No star import exposes the name: the module's own binding stands."""

    own_binding: str

    def sort_into(
        self,
        key: KeyT,
        settled: list[tuple[KeyT, str]],
        unsettled: list[KeyT],
    ) -> None:
        settled.append((key, self.own_binding))


class NameUnbound(UntouchedSettlement):
    """No star import exposes the name and the module binds it nowhere else."""

    def sort_into(
        self,
        key: KeyT,
        settled: list[tuple[KeyT, str]],
        unsettled: list[KeyT],
    ) -> None:
        unsettled.append(key)


class NameUncertain(StarNameSettlement):
    """A star import whose exports are unknown may rebind the name."""

    def or_settled_by(
        self,
        origin: CompactModuleStarImportOrigin,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
    ) -> StarNameSettlement:
        return self

    def sort_into(
        self,
        key: KeyT,
        settled: list[tuple[KeyT, str]],
        unsettled: list[KeyT],
    ) -> None:
        unsettled.append(key)


@dataclass(frozen=True)
class NameSettled(StarNameSettlement):
    """The last star import exposing the name rebinds it to this binding."""

    binding: str

    def or_settled_by(
        self,
        origin: CompactModuleStarImportOrigin,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
    ) -> StarNameSettlement:
        return self

    def sort_into(
        self,
        key: KeyT,
        settled: list[tuple[KeyT, str]],
        unsettled: list[KeyT],
    ) -> None:
        settled.append((key, self.binding))


@dataclass(frozen=True)
class StarSettlements(Generic[KeyT]):
    """Keyed names split into those the star imports settle to a binding and the rest."""

    settled: tuple[tuple[KeyT, str], ...]
    unsettled: tuple[KeyT, ...]

    @classmethod
    def keyed(
        cls, pairs: Iterable[tuple[KeyT, StarNameSettlement]]
    ) -> StarSettlements[KeyT]:
        settled: list[tuple[KeyT, str]] = []
        unsettled: list[KeyT] = []
        for key, settlement in pairs:
            settlement.sort_into(key, settled, unsettled)
        return cls(tuple(settled), tuple(unsettled))


class CompactModuleStarImportOrigin(ABC):
    """One module-scope star import; its subtype says whether its module resolved."""

    @staticmethod
    def resolved_as(module_name: str | None) -> CompactModuleStarImportOrigin:
        """The import resolver's answer, read once, picks the origin kind."""

        if module_name is None:
            return UnresolvedStarImportOrigin()
        return ResolvedStarImportOrigin(module_name)

    @abstractmethod
    def exposure_in(
        self, authority: RepositoryPublicExposureAuthority, binding_name: str
    ) -> CompactPublicNameExposure:
        raise NotImplementedError

    @abstractmethod
    def settlement_in(
        self,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
        untouched: UntouchedSettlement,
    ) -> StarNameSettlement:
        raise NotImplementedError

    @abstractmethod
    def within(self, module_names: Collection[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def leaves_open(self, consumer: CompactModuleClassHeader, name: str) -> bool:
        """An unknown origin in a consumer that does not close the name over."""
        raise NotImplementedError


@dataclass(frozen=True)
class ResolvedStarImportOrigin(CompactModuleStarImportOrigin):
    module_name: str

    def exposure_in(
        self, authority: RepositoryPublicExposureAuthority, binding_name: str
    ) -> CompactPublicNameExposure:
        return authority.exposure_for(self.module_name, binding_name)

    def settlement_in(
        self,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
        untouched: UntouchedSettlement,
    ) -> StarNameSettlement:
        exposure = self.exposure_in(authority, name)
        if exposure.proves_public_exposure:
            origin_projection = authority.projections_by_module_name[self.module_name]
            return NameSettled(
                dict(origin_projection.import_aliases).get(
                    name, f"{self.module_name}.{name}"
                )
            )
        if exposure.introduces_uncertainty:
            return NameUncertain()
        return untouched

    def within(self, module_names: Collection[str]) -> bool:
        return self.module_name in module_names

    def leaves_open(self, consumer: CompactModuleClassHeader, name: str) -> bool:
        return False


@dataclass(frozen=True)
class UnresolvedStarImportOrigin(CompactModuleStarImportOrigin):
    """A star import whose module the repository cannot name: it decides nothing."""

    def exposure_in(
        self, authority: RepositoryPublicExposureAuthority, binding_name: str
    ) -> CompactPublicNameExposure:
        return CompactPublicNameExposure.UNRESOLVED

    def settlement_in(
        self,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
        untouched: UntouchedSettlement,
    ) -> StarNameSettlement:
        return NameUncertain()

    def within(self, module_names: Collection[str]) -> bool:
        return False

    def leaves_open(self, consumer: CompactModuleClassHeader, name: str) -> bool:
        return consumer.public_export_contract.exposure_for(name).blocks_closed_boundary


class RepositoryPublicExposureAuthority(ABC):
    """Repository-owned proof that star imports preserve one local binding."""

    @abstractmethod
    def contains_module(self, module_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def exposure_for(
        self,
        module_name: str,
        binding_name: str,
    ) -> CompactPublicNameExposure:
        raise NotImplementedError

    @abstractmethod
    def star_import_origins_for(
        self,
        module_name: str,
    ) -> tuple[CompactModuleStarImportOrigin, ...]:
        raise NotImplementedError

    def star_imports_exclude(self, module_name: str, binding_name: str) -> bool:
        """Prove every star-import origin excludes the queried binding."""

        return self.contains_module(module_name) and all(
            not origin.exposure_in(self, binding_name).blocks_closed_boundary
            for origin in self.star_import_origins_for(module_name)
        )


@dataclass(frozen=True)
class CompactModuleClassHeader(CompactModuleIdentity):
    """Module namespace and class surface required by the compact family index."""

    import_aliases: tuple[tuple[str, str], ...]
    public_export_contract: CompactModulePublicExportContract
    star_import_origins: tuple[CompactModuleStarImportOrigin, ...]
    classes: tuple[CompactIndexedClass, ...]
    field_declaration_wrappers: tuple[CompactFieldDeclarationWrapper, ...] = ()

    @cached_property
    def has_unsettled_field_declarations(self) -> bool:
        return any(
            member.declaration_call is not None
            for indexed_class in self.classes
            for member in indexed_class.dataclass_declaration.fields
        )

    @staticmethod
    @ScanCache.cached
    def from_modules(
        modules: tuple[ParsedModule, ...],
    ) -> tuple[CompactModuleClassHeader, ...]:
        """Join original declarations with the repository's namespace proof."""
        return repository_settled_projections(
            tuple(CompactModuleClassHeader.from_module(module) for module in modules)
        )

    @staticmethod
    def from_module(parsed_module: ParsedModule) -> CompactModuleClassHeader:
        """Acquire complete class declarations without unrelated analysis surfaces:
        a fact of the parse, so every scan and every goal state reads an unchanged
        module's declarations once."""
        return parsed_module.fact(ModuleClassHeader).header

    @classmethod
    def declared_in(
        cls, parsed_module: ParsedModule, *, include_body_facets: bool
    ) -> Self:
        """The module's namespace, classes and field-declaration wrappers, from one
        binding pass: each class reads the bindings before it, each wrapper the
        module's final ones."""
        indexed_class_nodes = iter_class_definitions(list(parsed_module.module.body))
        binding_snapshots = parsed_module.fact(ClassLineBindingSnapshots).snapshots
        return cls(
            module_name=parsed_module.module_name,
            file_path=parsed_module.file_path,
            import_aliases=tuple(sorted(module_import_aliases(parsed_module).items())),
            public_export_contract=module_public_export_contract(parsed_module),
            star_import_origins=module_star_import_origins(parsed_module),
            classes=_compact_indexed_classes(
                parsed_module,
                indexed_class_nodes,
                binding_snapshots,
                include_body_facets=include_body_facets,
            ),
            field_declaration_wrappers=module_field_declaration_wrappers(
                parsed_module, binding_snapshots[None]
            ),
        )

ClassProjectionT = TypeVar("ClassProjectionT", bound=CompactModuleClassHeader)


@dataclass(frozen=True)
class ModuleClassHeader(ParsedModuleFact):
    """One module's complete class declarations, body facets included. A fact of
    the parse, like ModuleClassHeaderCore: a goal run's states hand unchanged
    parses from stage to stage, so each unchanged module is declared once per run
    instead of once per scan."""

    header: CompactModuleClassHeader

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        return cls(
            CompactModuleClassHeader.declared_in(parsed_module, include_body_facets=True)
        )


@dataclass(frozen=True)
class ModuleClassHeaderCore(ParsedModuleFact):
    """One module's namespace and class headers, the part of its projection a
    class-family index reads. A fact of the parse, so a plan's snapshots, which
    hand unchanged parses from stage to stage, project each unchanged module once."""

    projection: CompactModuleClassProjection

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        """The header-core view of the class projection family, through the
        parse's family batch: a later process over unchanged source reads it."""

        (projection,) = parsed_module.family_batch(
            CompactModuleClassProjectionFamily, HEADER_CORE_DEMAND
        ).items
        return cls(projection)


@dataclass(frozen=True)
class CompactRepositoryPublicExposureIndex(RepositoryPublicExposureAuthority):
    """Derive transitive public exposure from module-owned namespace contracts."""

    module_projections: tuple[CompactModuleClassHeader, ...]

    def star_settlement(
        self, module_name: str, name: str, unshadowed: str | None
    ) -> StarNameSettlement:
        """What a name a star import may rebind is bound to, as the repository's exports decide it.

        `unshadowed` is the module's own binding, star imports aside. No star
        import exposing the name leaves that binding; the last star import
        that exposes it rebinds it to that module's binding; an origin whose
        exports are unknown decides nothing.
        """

        settlement = StarNameSettlement.unshadowed(unshadowed)
        if self.star_imports_exclude(module_name, name):
            return settlement
        if not self.contains_module(module_name):
            return NameUncertain()
        for origin in reversed(self.star_import_origins_for(module_name)):
            settlement = settlement.or_settled_by(origin, self, name)
        return settlement

    @cached_property
    def module_projection_multiplicity(
        self,
    ) -> IdentityHandleMultiplicityProjection[str, CompactModuleClassHeader]:
        return UniqueIdentityIndexAuthority.declaration_multiplicity_by_handle(
            self.module_projections,
            lambda projection: projection.module_name,
        )

    def module_declaration_lookup(
        self, module_name: str
    ) -> tuple[dict[str, CompactIndexedClass], dict[str, str]] | None:
        lookups: dict[
            str, tuple[dict[str, CompactIndexedClass], dict[str, str]] | None
        ] = self.__dict__.setdefault("_lazy_module_declaration_lookups", {})
        if module_name in lookups:
            return lookups[module_name]
        projection = self.projections_by_module_name.get(module_name)
        lookup = (
            None
            if projection is None
            else (
                {declared.symbol: declared for declared in projection.classes},
                dict(projection.import_aliases),
            )
        )
        lookups[module_name] = lookup
        return lookup

    @cached_property
    def projections_by_module_name(self) -> dict[str, CompactModuleClassHeader]:
        return self.module_projection_multiplicity.unambiguous_declarations_by_handle

    def contains_module(self, module_name: str) -> bool:
        return module_name in self.projections_by_module_name

    @cached_property
    def named_reexports_by_target_symbol(
        self,
    ) -> dict[str, tuple[tuple[str, str], ...]]:
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for projection in self.module_projections:
            for local_name, target_symbol in projection.import_aliases:
                grouped[target_symbol].append((projection.module_name, local_name))
        return {
            symbol: tuple(dict.fromkeys(reexports))
            for symbol, reexports in grouped.items()
        }

    def exposure_for(
        self,
        module_name: str,
        binding_name: str,
    ) -> CompactPublicNameExposure:
        pending = deque(((module_name, binding_name),))
        visited: set[tuple[str, str]] = set()
        unresolved = False
        while pending:
            current_module, current_name = pending.popleft()
            binding = current_module, current_name
            if binding in visited:
                continue
            visited.add(binding)
            projection = self.projections_by_module_name.get(current_module)
            if projection is None:
                unresolved = True
                continue
            exposure = projection.public_export_contract.exposure_for(current_name)
            if exposure.proves_public_exposure:
                return exposure
            if exposure.introduces_uncertainty:
                unresolved = True
            symbol = f"{current_module}.{current_name}"
            pending.extend(self.named_reexports_by_target_symbol.get(symbol, ()))
            if any(
                origin.leaves_open(consumer, current_name)
                for consumer in self.module_projections
                for origin in consumer.star_import_origins
            ):
                unresolved = True
        return (
            CompactPublicNameExposure.UNRESOLVED
            if unresolved
            else CompactPublicNameExposure.PRIVATE
        )

    def star_import_origins_for(
        self,
        module_name: str,
    ) -> tuple[CompactModuleStarImportOrigin, ...]:
        projection = self.projections_by_module_name.get(module_name)
        return () if projection is None else projection.star_import_origins


@dataclass(frozen=True)
class CompactModuleClassProjection(
    CompactClassSyntaxFacets,
    CompactModuleClassHeader,
):
    """One module's class declarations and import aliases, without its AST."""

    sorted_key_calls: tuple[CompactSortedKeyCall, ...] = ()
    keyed_table_axes: tuple[CompactKeyedTableAxis, ...] = ()
    manual_selector_axes: tuple[CompactManualSelectorAxis, ...] = ()
    top_level_definitions: tuple[tuple[str, int], ...] = ()
    annotated_function_returns: tuple[CompactAnnotatedFunctionReturn, ...] = ()
    type_variable_names: tuple[str, ...] = ()
    """Module-level names bound to a `TypeVar`, `ParamSpec` or `TypeVarTuple` call."""
    repeated_keyed_family_roots: tuple[CompactRepeatedKeyedFamilyRoot, ...] = ()
    manual_subclass_roster_roots: tuple[CompactManualSubclassRosterRoot, ...] = ()
    latent_rosters: tuple[LatentRosterObservation, ...] = ()
    named_projection_surfaces: tuple[CompactNamedProjectionSurface, ...] = ()
    class_methods: tuple[CompactClassMethod, ...] = ()
    class_references: tuple[CompactClassReference, ...] = ()
    script_scopes: tuple[CompactScriptScope, ...] = ()
    field_order_summaries: tuple[FunctionOrderSummary, ...] = ()
    """What each function contributes to reading a dataclass's fields in order."""

    @cached_property
    def references_by_part(
        self,
    ) -> Mapping[str, tuple[tuple[str, CompactClassReference], ...]]:
        """This module's class references indexed by each part name they spell,
        built once on the immutable projection and reused across virtual states."""

        filed: dict[str, list[tuple[str, CompactClassReference]]] = defaultdict(list)
        for scope_name, references in (
            (self.module_name, self.class_references),
            *(
                (script.module_name, script.class_references)
                for script in self.script_scopes
            ),
        ):
            for reference in references:
                parts = reference.reference_parts
                if len(parts) == 1:
                    filed[parts[0]].append((scope_name, reference))
                else:
                    for name in dict.fromkeys(parts):
                        filed[name].append((scope_name, reference))
        return {name: tuple(entries) for name, entries in filed.items()}

    def header_core(self) -> CompactModuleClassProjection:
        """Project only the class declarations required by the family index."""

        return replace(
            self,
            classes=tuple(
                replace(
                    indexed_class,
                    **self._default_values_outside(
                        CompactIndexedClass,
                        CompactClassHeader,
                    ),
                )
                for indexed_class in self.classes
            ),
            **self._default_values_outside(
                CompactModuleClassProjection,
                CompactModuleClassHeader,
            ),
        )

    @staticmethod
    def _default_values_outside(
        declaration_type: type,
        preserved_type: type,
    ) -> dict[str, Any]:
        """Each field `declaration_type` adds past `preserved_type`, at its declared default.

        The values are each field's own default, read from `fields()`, so each
        has its field's type; `fields()` exposes them only as `Any`.
        """

        preserved_names = frozenset(
            dataclass_field.name for dataclass_field in fields(preserved_type)
        )
        values: dict[str, Any] = {}
        for dataclass_field in fields(declaration_type):
            if dataclass_field.name in preserved_names:
                continue
            if dataclass_field.default is not MISSING:
                values[dataclass_field.name] = dataclass_field.default
            elif dataclass_field.default_factory is not MISSING:
                values[dataclass_field.name] = dataclass_field.default_factory()
            else:
                raise TypeError(
                    f"{declaration_type.__name__}.{dataclass_field.name} has no default"
                )
        return values


class ClassMethodPromotionSafeDecorator(StrEnum):
    """Class decorators proven not to depend on direct method ownership."""

    DATACLASS = ("dataclass", frozenset(("dataclasses",)), True)
    FINAL = ("final", frozenset(("typing", "typing_extensions")), True)

    import_module_names: frozenset[str]
    preserves_product_schema: bool

    def __new__(
        cls,
        value: str,
        import_module_names: frozenset[str],
        preserves_product_schema: bool,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.import_module_names = import_module_names
        member.preserves_product_schema = preserves_product_schema
        return member

    @classmethod
    def for_qualified_name(cls, qualified_name: str | None) -> Self | None:
        return next(
            (
                decorator
                for decorator in cls
                if any(
                    qualified_name == f"{module_name}.{decorator.value}"
                    for module_name in decorator.import_module_names
                )
            ),
            None,
        )

    @classmethod
    def qualified_name_preserves_product_schema(
        cls,
        qualified_name: str | None,
    ) -> bool:
        declaration = cls.for_qualified_name(qualified_name)
        return declaration is not None and declaration.preserves_product_schema


@dataclass(frozen=True)
class LatentRosterMatch:
    """Typed agreement between one roster and one nominal family surface."""

    coverage_ratio: float
    missing_member_names: tuple[str, ...]
    projection_policy_hint: str | None


@dataclass(frozen=True)
class LatentRosterObservation:
    """One source roster, with parsing and family matching owned once."""

    policy_tokens: ClassVar[frozenset[str]] = frozenset(
        {
            "active",
            "all",
            "available",
            "default",
            "enabled",
            "public",
            "selected",
            "supported",
            "test",
            "visible",
        }
    )
    policy_noise_tokens: ClassVar[frozenset[str]] = frozenset(
        {"by", "classes", "formats", "map", "registry", "types"}
    )
    mutation_methods: ClassVar[frozenset[str]] = frozenset({"extend", "update"})

    file_path: str
    roster_name: str
    line: int
    roster_kind: str
    projection_role: str
    member_names: tuple[str, ...]
    line_count: int

    @property
    def is_public_export_surface(self) -> bool:
        """Return whether this roster declares Python's module export contract."""

        return self.roster_name == PYTHON_PUBLIC_EXPORT_ASSIGNMENT

    @classmethod
    def from_module(
        cls,
        parsed_module: ParsedModule,
    ) -> tuple[LatentRosterObservation, ...]:
        observations: list[LatentRosterObservation] = []
        for statement in _trim_leading_docstring(list(parsed_module.module.body)):
            observations.extend(cls.from_statement(parsed_module, statement))
            observations.extend(cls.from_mutation_statement(parsed_module, statement))
            if not isinstance(statement, ast.ClassDef):
                continue
            for class_statement in _trim_leading_docstring(list(statement.body)):
                observations.extend(
                    cls.from_statement(
                        parsed_module,
                        class_statement,
                        roster_prefix=statement.name,
                    )
                )
                observations.extend(
                    cls.from_mutation_statement(parsed_module, class_statement)
                )
        return tuple(observations)

    @classmethod
    def from_statement(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
        *,
        roster_prefix: str | None = None,
    ) -> tuple[LatentRosterObservation, ...]:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = _trim_leading_docstring(list(statement.body))
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                return ()
            return StatementValue.of(body[0]).answer(
                LatentRosterValue(parsed_module, statement, statement.name)
            )
        if not isinstance(statement, ast.Assign | ast.AnnAssign):
            return ()
        return tuple(
            observation
            for target in single_name_targets(statement)
            for observation in StatementValue.of(statement).answer(
                LatentRosterValue(
                    parsed_module,
                    statement,
                    (
                        f"{roster_prefix}.{target.id}"
                        if roster_prefix is not None
                        else target.id
                    ),
                )
            )
        )

    @classmethod
    def from_mutation_statement(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
    ) -> tuple[LatentRosterObservation, ...]:
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr in cls.mutation_methods
        ):
            return ()
        call = statement.value
        return tuple(
            observation.for_mutation(call.func.attr)
            for argument in call.args
            for observation in cls.from_value(
                parsed_module,
                statement,
                roster_name=ast.unparse(call.func.value),
                value=argument,
            )
        )

    @classmethod
    def from_value(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
        *,
        roster_name: str,
        value: ast.AST,
    ) -> tuple[LatentRosterObservation, ...]:
        if isinstance(value, ast.Dict):
            return tuple(
                cls.from_members(
                    parsed_module,
                    statement,
                    roster_name=roster_name,
                    value=value,
                    projection_role=projection_role,
                    member_names=member_names,
                )
                for projection_role, member_names in (
                    ("dict_keys", cls.dict_member_names(value.keys)),
                    ("dict_values", cls.dict_member_names(value.values)),
                )
                if len(member_names) >= 2
            )
        member_names = cls.names_from(value)
        if len(member_names) < 2:
            return ()
        return (
            cls.from_members(
                parsed_module,
                statement,
                roster_name=roster_name,
                value=value,
                projection_role="collection_members",
                member_names=member_names,
            ),
        )

    @classmethod
    def from_members(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
        *,
        roster_name: str,
        value: ast.AST,
        projection_role: str,
        member_names: tuple[str, ...],
    ) -> LatentRosterObservation:
        return cls(
            file_path=parsed_module.file_path,
            roster_name=roster_name,
            line=statement.lineno,
            roster_kind=type(value).__name__,
            projection_role=projection_role,
            member_names=member_names,
            line_count=(statement.end_lineno or statement.lineno)
            - statement.lineno
            + 1,
        )

    @staticmethod
    def names_from(node: ast.AST) -> tuple[str, ...]:
        if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return ()
        member_names: set[str] = set()
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                member_names.add(element.value)
                continue
            reference = element.func if isinstance(element, ast.Call) else element
            if (member_name := _terminal_reference_name(reference)) is not None:
                member_names.add(member_name)
        return sorted_tuple(member_names)

    @classmethod
    def dict_member_names(cls, nodes: list[ast.expr | None]) -> tuple[str, ...]:
        return sorted_tuple(
            {
                member_name
                for node in nodes
                if node is not None
                for member_name in cls.names_from(
                    ast.Tuple(elts=[node], ctx=ast.Load())
                )
            }
        )

    def for_mutation(self, mutation_name: str) -> LatentRosterObservation:
        return replace(
            self,
            roster_kind=f"inline_{self.roster_kind}.{mutation_name}",
            projection_role=f"{mutation_name}_{self.projection_role}",
        )

    @property
    def projection_policy_hint(self) -> str | None:
        policy_tokens = tuple(
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", self.roster_name)
            if token.lower() not in self.policy_noise_tokens
            and token.lower() in self.policy_tokens
        )
        return "_".join(policy_tokens) or None

    def match(
        self,
        authority_member_names: tuple[str, ...],
    ) -> LatentRosterMatch | None:
        roster_members = set(self.member_names)
        authority_members = set(authority_member_names)
        if not authority_members or not roster_members <= authority_members:
            return None
        missing_names = sorted_tuple(authority_members - roster_members)
        coverage_ratio = len(roster_members) / len(authority_members)
        if not missing_names:
            return LatentRosterMatch(coverage_ratio, (), None)
        if self.projection_policy_hint is None:
            return None
        return LatentRosterMatch(
            coverage_ratio,
            missing_names,
            self.projection_policy_hint,
        )


@dataclass(frozen=True)
class LatentRosterValue(StatementValueQuestion[tuple[LatentRosterObservation, ...]]):
    """The roster a statement's value spells; a bare statement spells none."""

    parsed_module: ParsedModule
    statement: ast.stmt
    roster_name: str

    def supplied(self, value: SuppliedValue) -> tuple[LatentRosterObservation, ...]:
        return LatentRosterObservation.from_value(
            self.parsed_module,
            self.statement,
            roster_name=self.roster_name,
            value=value.value,
        )

    def bare(self, value: NoStatementValue) -> tuple[LatentRosterObservation, ...]:
        return ()


class RegistryLookupStyle(StrEnum):
    TRY_EXCEPT = "try_except"
    MEMBERSHIP_GUARD = "membership_guard"


@dataclass(frozen=True)
class ClsRegistryMembership:
    operator_type: type[ast.cmpop]
    key_expr: str

    @classmethod
    def from_node(cls, node: ast.AST) -> ClsRegistryMembership | None:
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and len(node.comparators) == 1
            and RegistryLookupShape.is_cls_registry_attribute(node.comparators[0])
        ):
            return None
        return cls(type(node.ops[0]), ast.unparse(node.left))


@dataclass(frozen=True)
class RegistryLookupShape:
    """Recognized syntax for a class-owned ``_registry`` lookup."""

    key_expr: str
    error_type_name: str | None
    style: RegistryLookupStyle

    @staticmethod
    def is_cls_registry_attribute(node: ast.AST | None) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "_registry"
            and isinstance(node.value, ast.Name)
            and node.value.id == "cls"
        )

    @classmethod
    def key_expr_from_subscript(cls, node: ast.AST | None) -> str | None:
        if not (
            isinstance(node, ast.Subscript)
            and cls.is_cls_registry_attribute(node.value)
        ):
            return None
        return ast.unparse(node.slice)

    @classmethod
    def references_registry(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        return any(cls.is_cls_registry_attribute(node) for node in ast.walk(method))

    @staticmethod
    def _raise_type_name(node: ast.Raise | None) -> str | None:
        if node is None or node.exc is None:
            return None
        expression = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        return _terminal_reference_name(expression)

    @classmethod
    def from_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> RegistryLookupShape | None:
        body = _trim_method_docstring(method)
        if len(body) == 1 and isinstance(body[0], ast.Try):
            try_node = body[0]
            if (
                not try_node.orelse
                and not try_node.finalbody
                and len(try_node.handlers) == 1
                and _terminal_reference_name(try_node.handlers[0].type) == "KeyError"
                and len(try_node.body) == 1
                and isinstance(try_node.body[0], ast.Return)
                and (key_expr := cls.key_expr_from_subscript(try_node.body[0].value))
                is not None
            ):
                raised = next(
                    (
                        statement
                        for statement in try_node.handlers[0].body
                        if isinstance(statement, ast.Raise)
                    ),
                    None,
                )
                return cls(
                    key_expr,
                    cls._raise_type_name(raised),
                    RegistryLookupStyle.TRY_EXCEPT,
                )
        if len(body) < 2 or not isinstance(body[0], ast.If):
            return None
        guard = body[0]
        returned = body[-1]
        membership = ClsRegistryMembership.from_node(guard.test)
        if not (
            isinstance(returned, ast.Return)
            and membership is not None
            and membership.operator_type is ast.NotIn
            and cls.key_expr_from_subscript(returned.value) == membership.key_expr
        ):
            return None
        raised = next(
            (statement for statement in guard.body if isinstance(statement, ast.Raise)),
            None,
        )
        return cls(
            membership.key_expr,
            cls._raise_type_name(raised),
            RegistryLookupStyle.MEMBERSHIP_GUARD,
        )


class DeclaredFamilyBase(NamedTuple):
    """One base a class declares: the symbol its binding resolves to, shown by its terminal name."""

    symbol: str
    name: str

    @classmethod
    def of(cls, reference: CompactNominalReference) -> Self:
        return cls(reference.builtin_qualified_name, reference.simple_name)

    def is_standard_library(self, scanned_roots: frozenset[str]) -> bool:
        """Whether it resolves into the standard library: a top-level module no scanned module shadows."""

        root = self.symbol.partition(".")[0]
        return root in sys.stdlib_module_names and root not in scanned_roots


@dataclass(frozen=True)
class CompactRepeatedKeyedFamilyRoot(CompactSourceLocation):
    class_name: str
    family_bases: tuple[DeclaredFamilyBase, ...]
    registry_key_attr_name: str
    lookup_method_name: str
    lookup_style: RegistryLookupStyle
    error_type_name: str | None
    abstract_hook_names: tuple[str, ...]


ClassDeclarationT = TypeVar("ClassDeclarationT", bound=ClassDeclaration)


@dataclass(frozen=True)
class ClassDeclarationIndex(Generic[ClassDeclarationT]):
    """One declaration map with lazily derived identity and inheritance views."""

    classes_by_symbol: dict[str, ClassDeclarationT]

    @cached_property
    def known_symbols(self) -> frozenset[str]:
        return frozenset(self.classes_by_symbol)

    @cached_property
    def symbols_by_simple_name(self) -> dict[str, tuple[str, ...]]:
        symbols: dict[str, list[str]] = defaultdict(list)
        for record in self.classes_by_symbol.values():
            symbols[record.simple_name].append(record.symbol)
        return {name: sorted_tuple(values) for name, values in symbols.items()}

    @cached_property
    def unique_symbols_by_name(self) -> dict[str, str]:
        return {
            name: symbols[0]
            for name, symbols in self.symbols_by_simple_name.items()
            if len(symbols) == 1
        }

    @cached_property
    def unique_symbols_by_suffix(self) -> _UniqueKnownSymbolSuffixIndex:
        return _UniqueKnownSymbolSuffixIndex.from_terminal_buckets(
            self.symbols_by_simple_name
        )

    @cached_property
    def symbols_by_file_and_qualname(self) -> dict[tuple[str, str], str]:
        return {
            (record.file_path, record.qualname): record.symbol
            for record in self.classes_by_symbol.values()
        }

    @cached_property
    def inheritance_graph(self) -> DirectedGraph[str]:
        return DirectedGraph(
            {
                symbol: self.classes_by_symbol[symbol].resolved_base_symbols
                for symbol in sorted(self.classes_by_symbol)
            }
        )

    @cached_property
    def children_by_symbol(self) -> dict[str, tuple[str, ...]]:
        return {
            symbol: children
            for symbol, children in self.inheritance_graph.reversed.neighbors.items()
            if children
        }

    @cached_property
    def ancestors_by_symbol(self) -> dict[str, tuple[str, ...]]:
        return self.inheritance_graph.nonempty_reachability_from(
            self.inheritance_graph.neighbors
        )

    @cached_property
    def descendants_by_symbol(self) -> dict[str, tuple[str, ...]]:
        return self.inheritance_graph.reversed.nonempty_reachability_from(
            self.inheritance_graph.neighbors
        )

    def class_for(self, symbol: str) -> ClassDeclarationT | None:
        return self.classes_by_symbol.get(symbol)

    def symbol_for(self, *, file_path: str, qualname: str) -> str | None:
        return self.symbols_by_file_and_qualname.get((file_path, qualname))

    def ancestor_symbols(self, class_symbol: str) -> tuple[str, ...]:
        return self.inheritance_graph.reachable_from(class_symbol)

    def overridden_declarations(self, class_symbol: str, member: str) -> tuple[str, ...]:
        """What the class's `member` overrides: each scanned ancestor that declares
        it, each co-inherited ancestor in any descendant's MRO that declares it
        (cooperative multiple inheritance: a mixin placed before a sibling base in
        a subclass overrides/implements that base's member, and one placed after it
        participates in its `super()` chain), and each base outside the scan not
        known to lack it. A receiver typed as any of them reaches the class's
        member by dispatch."""

        descendants = self.descendant_symbols(class_symbol)
        family = frozenset((class_symbol, *descendants))
        ancestors = list(self.ancestor_symbols(class_symbol))
        seen_ancestors = set(ancestors)
        for descendant in descendants:
            for ancestor in self.ancestor_symbols(descendant):
                if ancestor not in family and ancestor not in seen_ancestors:
                    seen_ancestors.add(ancestor)
                    ancestors.append(ancestor)
        lineage = (class_symbol, *descendants, *ancestors)
        classes = self.classes_by_symbol
        return (
            *dict.fromkeys(
                classes[symbol].qualname
                for symbol in ancestors
                if symbol in classes and member in classes[symbol].member_binding_names
            ),
            *dict.fromkeys(
                spelled
                for symbol in lineage
                if symbol in classes
                for spelled in classes[symbol].unresolved_base_names
                if ForeignBase(spelled).may_declare(member)
            ),
        )

    def descendant_symbols(self, base_symbol: str) -> tuple[str, ...]:
        if base_symbol not in self.classes_by_symbol:
            return ()
        return self.inheritance_graph.reversed.reachable_from(base_symbol)


@dataclass
class _ProductLineageComposer:
    """Compose dataclass products through their exact direct-base chains, memoized."""

    classes_by_symbol: Mapping[str, CompactIndexedClass]
    lineages: dict[str, _CompactProductLineage] = field(default_factory=dict)

    @staticmethod
    def failure(
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

    def compose(
        self,
        class_symbol: str,
        declaration: CompactDataclassDeclaration,
        pending_symbols: frozenset[str],
    ) -> _CompactProductLineage:
        if (lineage := self.lineages.get(class_symbol)) is not None:
            return lineage
        indexed_class = self.classes_by_symbol[class_symbol]
        if class_symbol in pending_symbols:
            return _CompactProductLineage(
                (),
                (
                    self.failure(
                        indexed_class,
                        CompactProductAuthorityViolation.CYCLIC_PRODUCT_LINEAGE,
                    ),
                ),
            )
        failures = tuple(
            self.failure(indexed_class, failure.violation, line=failure.line)
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
                self.failure(
                    indexed_class,
                    CompactProductAuthorityViolation.INCOMPLETE_BASE_RESOLUTION,
                ),
            )
        elif direct_base_count > 1:
            failures = (
                *failures,
                self.failure(
                    indexed_class,
                    CompactProductAuthorityViolation.MULTIPLE_PRODUCT_BASES,
                ),
            )
        elif direct_base_count == 1:
            base_symbol = indexed_class.resolved_base_symbols[0]
            base_lineage = self.classes_by_symbol[
                base_symbol
            ].dataclass_declaration.base_lineage_in(
                self, base_symbol, indexed_class, pending_symbols
            )
            effective_fields = base_lineage.effective_fields
            failures = (*failures, *base_lineage.failures)

        fields_by_name = {field.name: field for field in effective_fields}
        for field_declaration in declaration.fields:
            fields_by_name[field_declaration.name] = CompactProductField(
                name=field_declaration.name,
                role=field_declaration.role,
                binding=field_declaration.binding,
                declaring_class_symbol=class_symbol,
                file_path=indexed_class.file_path,
                line=field_declaration.line,
            )
        lineage = _CompactProductLineage(
            effective_fields=tuple(fields_by_name.values()),
            failures=tuple(dict.fromkeys(failures)),
        )
        self.lineages[class_symbol] = lineage
        return lineage

    def resolve(
        self,
        class_symbol: str,
        indexed_class: CompactIndexedClass,
        declaration: CompactDataclassDeclaration,
    ) -> CompactProductAuthorityResolution:
        lineage = self.compose(class_symbol, declaration, frozenset())
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
        return (
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


@dataclass(frozen=True)
class CompactClassFamilyIndex(ClassDeclarationIndex[CompactIndexedClass]):
    """Repository inheritance graph reconstructed from compact declarations."""

    _has_duplicate_symbols: bool = field(default=False, repr=False, compare=False)

    @cached_property
    def creation(self) -> ClassCreationOrderProof:
        """What creating each class does to its order and its names."""

        return ClassCreationOrderProof(
            RegisteringMetaclassAuthority(
                self.classes_by_symbol, CompactRepositoryPublicExposureIndex(())
            ),
            # A module declaring no class is not read: its `__bases__` store goes unseen.
            any(declaration.registry_facts.module_stores_bases for declaration in self.classes_by_symbol.values()),
        )

    @cached_property
    def mro_authority(self) -> ClassMroAuthority:
        """The order Python builds, also for a class whose creation runs code the
        creation proof reads; every detector and refactor reads this one."""

        return ClassMroAuthority(self.classes_by_symbol, ProvedCreationBases(self.creation.keeps_order))

    @cached_property
    def declared_mro_authority(self) -> ClassMroAuthority:
        """The MRO the declarations state, for reading declared member types (`DeclaredBases`)."""

        scanned_modules = frozenset(declaration.module_name for declaration in self.classes_by_symbol.values())
        unscanned = CompactRepositoryPublicExposureIndex(())

        def installed(qualified_name: str) -> CompactIndexedClass | None:
            # A module the scan holds is read from the scan, never an installed copy.
            if qualified_name.rpartition(".")[0] in scanned_modules:
                return None
            return InstalledClassDeclarations.declaration(qualified_name, unscanned)

        return ClassMroAuthority(self.classes_by_symbol, DeclaredBases(installed))

    def member_in_mro(self, owner_symbol: str, member_name: str) -> ClassMemberLookup:
        """The binding of `member_name` that an instance of `owner_symbol` reads at runtime."""

        return self._member_along(owner_symbol, member_name, self.mro_authority)

    def declared_member(self, owner_symbol: str, member_name: str) -> ClassMemberLookup:
        """The binding of `member_name` the declarations give an instance of `owner_symbol`."""

        return self._member_along(owner_symbol, member_name, self.declared_mro_authority)

    def _member_along(
        self, owner_symbol: str, member_name: str, authority: ClassMroAuthority
    ) -> ClassMemberLookup:
        """The first binding of `member_name` along `authority`'s MRO of `owner_symbol`.

        A class binding the name in its own body owns it, whatever its MRO; any
        other class reads the first binding along its closed MRO, and a lookup
        that reaches a base whose members nothing declares learns nothing past
        it. Assignments, annotations and defs all bind; a class binding the
        name twice, or by both an assignment and a def, is ambiguous.
        """

        owner = self.class_for(owner_symbol)
        if owner is None:
            return MissingClassMember()
        owners: tuple[CompactIndexedClass, ...] = (owner,)
        if not self._binds_member(owner, member_name):
            mro = authority.resolve(owner_symbol).mro_type
            if mro is None:
                return OpenClassMember()
            owners = mro.declarations_before(lambda entry: authority.bases.stops_lookup(entry, member_name))
        for current in owners:
            assigned = tuple(
                declaration
                for declaration in current.direct_member_declarations
                if declaration.name == member_name
            )
            defined = tuple(
                declaration
                for declaration in current.callable_member_declarations
                if declaration.name == member_name
            )
            match assigned, defined:
                case (), ():
                    continue
                case (declaration,), ():
                    return AssignedClassMember(current, declaration)
                case (), (declaration,):
                    return CallableClassMember(current, declaration)
            return AmbiguousClassMember()
        if not self._binds_member(owner, member_name) and (
            self._reaches_opaque_base(owner_symbol, authority, member_name)
            or self._creation_may_bind(owner_symbol, authority, member_name)
        ):
            return OpenClassMember()
        return MissingClassMember()

    def _creation_may_bind(self, owner_symbol: str, authority: ClassMroAuthority, member_name: str) -> bool:
        """Whether creating the class may bind `member_name` though no body does."""

        mro = authority.resolve(owner_symbol).mro_type
        return mro is not None and self.creation.bindings(mro.declarations).may_bind(member_name)

    @staticmethod
    def _reaches_opaque_base(owner_symbol: str, authority: ClassMroAuthority, member_name: str) -> bool:
        mro = authority.resolve(owner_symbol).mro_type
        return mro is not None and any(authority.bases.stops_lookup(entry, member_name) for entry in mro.__mro__)

    @staticmethod
    def _binds_member(owner: CompactIndexedClass, member_name: str) -> bool:
        return member_name in owner.member_binding_names

    @classmethod
    def from_modules(
        cls,
        modules: tuple[ParsedModule, ...],
    ) -> Self:
        """Join complete class declarations without collecting unrelated analyses."""

        return build_compact_class_family_index(
            CompactModuleClassHeader.from_modules(modules)
        )

    def projected(
        self,
        previous_projections: Sequence[CompactModuleClassHeader],
        projected_projections: Sequence[CompactModuleClassHeader],
    ) -> CompactClassFamilyIndex:
        """Project this index across a source update, reusing every unchanged
        class whose bases, metaclass, star imports and field wrappers are
        unaffected."""

        prev_by_path = {projection.file_path: projection for projection in previous_projections}
        proj_tuple = tuple(projected_projections)
        proj_by_path = {projection.file_path: projection for projection in proj_tuple}
        changed_projs = tuple(
            projection
            for projection in proj_tuple
            if prev_by_path.get(projection.file_path) is not projection
        )
        removed_projs = tuple(
            projection
            for path, projection in prev_by_path.items()
            if path not in proj_by_path
        )
        if not changed_projs and not removed_projs:
            _remember_compact_class_family_index(proj_tuple, self)
            return self
        if any(projection.star_import_origins for projection in proj_tuple):
            built = CompactClassFamilyIndexBuilder(proj_tuple).build()
            _remember_compact_class_family_index(proj_tuple, built)
            return built
        old_changed = tuple(
            prev_by_path[projection.file_path]
            for projection in changed_projs
            if projection.file_path in prev_by_path
        )
        old_wrappers = tuple(
            wrapper
            for projection in (*old_changed, *removed_projs)
            for wrapper in projection.field_declaration_wrappers
        )
        new_wrappers = tuple(
            wrapper
            for projection in changed_projs
            for wrapper in projection.field_declaration_wrappers
        )
        if old_wrappers != new_wrappers:
            built = CompactClassFamilyIndexBuilder(proj_tuple).build()
            _remember_compact_class_family_index(proj_tuple, built)
            return built

        needs_field_settling = any(
            projection.has_unsettled_field_declarations for projection in changed_projs
        )
        field_index = (
            FieldDeclarationIndex.of_projections(proj_tuple)
            if needs_field_settling
            else None
        )
        if field_index is not None:
            settled_changed: tuple[CompactModuleClassHeader, ...] = tuple(
                projection
                if not projection.has_unsettled_field_declarations
                else replace(
                    projection,
                    classes=tuple(
                        indexed_class.with_settled_field_declarations(field_index)
                        for indexed_class in projection.classes
                    ),
                )
                for projection in changed_projs
            )
        else:
            settled_changed = changed_projs

        if any(
            indexed_class.registry_facts.stores_created_class
            or any(
                reference.resolved_parts[-1] == "type"
                for reference in indexed_class.base_references
            )
            for projection in (*settled_changed, *old_changed, *removed_projs)
            for indexed_class in projection.classes
        ):
            built = CompactClassFamilyIndexBuilder(proj_tuple).build()
            _remember_compact_class_family_index(proj_tuple, built)
            return built

        prev_classes = self.classes_by_symbol
        old_changed_symbols: set[str] = set()
        duplicate_collision = self._has_duplicate_symbols
        if not duplicate_collision:
            for projection in (*old_changed, *removed_projs):
                for indexed_class in projection.classes:
                    if indexed_class.symbol not in prev_classes:
                        duplicate_collision = True
                        break
                    old_changed_symbols.add(indexed_class.symbol)
                if duplicate_collision:
                    break
        settled_changed_by_symbol: dict[str, CompactIndexedClass] = {}
        if not duplicate_collision:
            for projection in settled_changed:
                for indexed_class in projection.classes:
                    symbol = indexed_class.symbol
                    if symbol in settled_changed_by_symbol or (
                        symbol in prev_classes and symbol not in old_changed_symbols
                    ):
                        duplicate_collision = True
                        break
                    settled_changed_by_symbol[symbol] = indexed_class
                if duplicate_collision:
                    break
        if duplicate_collision:
            built = CompactClassFamilyIndexBuilder(proj_tuple).build()
            _remember_compact_class_family_index(proj_tuple, built)
            return built

        added_symbols = settled_changed_by_symbol.keys() - old_changed_symbols
        removed_symbols = old_changed_symbols - settled_changed_by_symbol.keys()
        if not added_symbols and not removed_symbols:
            known_symbols = self.known_symbols
            unique_symbols_by_suffix = self.unique_symbols_by_suffix
            moved_terminal_names: frozenset[str] = frozenset()
        else:
            known_symbols = (self.known_symbols - removed_symbols) | added_symbols
            unique_symbols_by_suffix = _unique_known_symbol_by_suffix(known_symbols)
            moved_terminal_names = frozenset(
                symbol.rsplit(".", 1)[-1]
                for symbol in (added_symbols | removed_symbols)
            )

        changed_alias_symbols: set[str] = set()
        for projection in changed_projs:
            old_projection = prev_by_path.get(projection.file_path)
            old_aliases = (
                dict(old_projection.import_aliases) if old_projection is not None else {}
            )
            new_aliases = dict(projection.import_aliases)
            for name in old_aliases.keys() | new_aliases.keys():
                if old_aliases.get(name) != new_aliases.get(name):
                    changed_alias_symbols.add(f"{projection.module_name}.{name}")
        for projection in removed_projs:
            for name, _ in projection.import_aliases:
                changed_alias_symbols.add(f"{projection.module_name}.{name}")

        classes_by_symbol = dict(prev_classes)
        for symbol in removed_symbols:
            classes_by_symbol.pop(symbol, None)

        own_dirty: set[str] = set()
        propagating_dirty: set[str] = set()

        for symbol, record in settled_changed_by_symbol.items():
            resolved_record = record.with_resolved_base_symbols(
                tuple(
                    resolved
                    for reference in record.base_references
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
            classes_by_symbol[symbol] = resolved_record
            own_dirty.add(symbol)
            prev_record = prev_classes.get(symbol)
            if (
                prev_record is None
                or resolved_record.resolved_base_symbols != prev_record.resolved_base_symbols
                or resolved_record.registry_facts.metaclass_references
                != prev_record.registry_facts.metaclass_references
                or resolved_record.registry_facts.base_references
                != prev_record.registry_facts.base_references
                or resolved_record.registry_facts.stores_subclass
                != prev_record.registry_facts.stores_subclass
                or resolved_record.registry_facts.stores_created_class
                != prev_record.registry_facts.stores_created_class
                or (
                    changed_alias_symbols
                    and any(
                        ".".join(ref.resolved_parts) in changed_alias_symbols
                        for ref in (
                            *resolved_record.base_references,
                            *resolved_record.registry_facts.metaclass_references,
                        )
                    )
                )
            ):
                propagating_dirty.add(symbol)

        if moved_terminal_names or changed_alias_symbols:
            for symbol, prev_record in prev_classes.items():
                if symbol in old_changed_symbols:
                    continue
                if (
                    moved_terminal_names
                    and (
                        any(
                            ref.resolved_parts[-1] in moved_terminal_names
                            for ref in prev_record.base_references
                        )
                        or any(
                            ref.resolved_parts[-1] in moved_terminal_names
                            for ref in prev_record.registry_facts.metaclass_references
                        )
                    )
                ) or (
                    changed_alias_symbols
                    and (
                        any(
                            ".".join(ref.resolved_parts) in changed_alias_symbols
                            for ref in prev_record.base_references
                        )
                        or any(
                            ".".join(ref.resolved_parts) in changed_alias_symbols
                            for ref in prev_record.registry_facts.metaclass_references
                        )
                    )
                ):
                    raw_module = proj_by_path[prev_record.file_path]
                    raw_record = next(
                        c for c in raw_module.classes if c.symbol == symbol
                    )
                    base_record = (
                        raw_record.with_settled_field_declarations(field_index)
                        if field_index is not None and raw_module.has_unsettled_field_declarations
                        else raw_record
                    )
                    classes_by_symbol[symbol] = base_record.with_resolved_base_symbols(
                        tuple(
                            resolved
                            for reference in base_record.base_references
                            if (
                                resolved := CompactClassFamilyIndexBuilder._resolved_bound_symbol(
                                    reference,
                                    base_record.module_name,
                                    known_symbols,
                                    unique_symbols_by_suffix,
                                )
                            )
                            is not None
                        )
                    )
                    own_dirty.add(symbol)
                    propagating_dirty.add(symbol)

        if not own_dirty and not removed_symbols:
            _remember_compact_class_family_index(proj_tuple, self)
            return self
        seed_propagating = propagating_dirty | removed_symbols
        if seed_propagating:
            descendants_dirty: set[str] = set(propagating_dirty)
            for symbol in tuple(seed_propagating):
                for descendant in self.descendant_symbols(symbol):
                    if descendant in classes_by_symbol:
                        descendants_dirty.add(descendant)
            # Follow any newly added base edges from dirty classes as well.
            children_map: dict[str, list[str]] = defaultdict(list)
            for symbol, record in classes_by_symbol.items():
                for base in record.resolved_base_symbols:
                    children_map[base].append(symbol)
            stack = list(descendants_dirty)
            while stack:
                current = stack.pop()
                for child in children_map.get(current, ()):
                    if child not in descendants_dirty:
                        descendants_dirty.add(child)
                        stack.append(child)
            own_dirty.update(descendants_dirty)

        if own_dirty:
            for symbol in own_dirty:
                if symbol in settled_changed_by_symbol:
                    continue
                current_record = classes_by_symbol[symbol]
                raw_module = proj_by_path[current_record.file_path]
                raw = next(c for c in raw_module.classes if c.symbol == symbol)
                if (
                    current_record.declares_autoregister_meta != raw.declares_autoregister_meta
                    or current_record.is_registration_authority != raw.is_registration_authority
                    or current_record.keyed_family_key_type_name != raw.keyed_family_key_type_name
                ):
                    classes_by_symbol[symbol] = replace(
                        current_record,
                        declares_autoregister_meta=raw.declares_autoregister_meta,
                        is_registration_authority=raw.is_registration_authority,
                        keyed_family_key_type_name=raw.keyed_family_key_type_name,
                    )
            authority = RegisteringMetaclassAuthority(
                classes_by_symbol,
                CompactRepositoryPublicExposureIndex(proj_tuple),
                _suffix_index=unique_symbols_by_suffix,
            )
            for symbol in own_dirty:
                classes_by_symbol[symbol] = classes_by_symbol[
                    symbol
                ].with_registering_metaclass(authority)
        result = CompactClassFamilyIndex(
            classes_by_symbol=classes_by_symbol,
            _has_duplicate_symbols=False,
        )
        _remember_compact_class_family_index(proj_tuple, result)
        return result

    @classmethod
    def from_projection_groups(
        cls,
        projections_by_family: dict[
            type[CollectedFamily],
            Sequence[object],
        ],
    ) -> Self:
        """Build the class anchor declared by a compact multi-family join."""

        return build_compact_class_family_index(
            CompactModuleClassProjectionFamily.modules_in(projections_by_family)
        )

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

        composer = _ProductLineageComposer(self.classes_by_symbol)
        return {
            class_symbol: indexed_class.dataclass_declaration.product_resolution_in(
                composer, class_symbol, indexed_class
            )
            for class_symbol, indexed_class in self.classes_by_symbol.items()
        }

    def construction_parameters(self, symbol: str) -> tuple[str, ...]:
        """The positional parameters a call `symbol(...)` binds, in order: those
        of the `__init__` the first dataclass along its declared MRO generates.

        The caller proves first that creating the class keeps what its
        declarations bind (`ClassCreationProof.require_construction_kept`).
        Then only `__init__` and `__new__` decide: an `__init__` met first
        replaces the generated one, and a `__new__` anywhere along the MRO sees
        the arguments as passed. Every other hook (`__post_init__`,
        `__setattr__`, `__slots__`, ...) runs after the parameters are bound,
        and any number of parents is fine once Python orders them. Raises,
        naming the reason, when the parameters are unknown.
        """

        authority = self.declared_mro_authority
        mro = authority.resolve(symbol).mro_type
        if mro is None:
            raise NeedsInput(f"`{symbol}`'s bases admit no MRO", NeededInput.PROOF)
        outside = [entry.__name__ for entry in mro.__mro__ if authority.bases.is_opaque(entry)]
        if outside:
            raise NeedsInput(
                f"`{symbol}`'s MRO reaches {outside} outside the scan, which may define `__init__`",
                NeededInput.PROOF,
            )
        for declaration in mro.declarations:
            if "__new__" in declaration.member_binding_names:
                raise ValueError(
                    f"`{declaration.symbol}` defines `__new__`, which sees the arguments as passed"
                )
        for entry in mro.__mro__:
            if not isinstance(entry, DeclarationMroType):
                continue
            if "__init__" in entry.declaration.member_binding_names:
                raise ValueError(
                    f"`{entry.declaration.symbol}` defines the `__init__` `{symbol}` runs"
                )
            facet = entry.declaration.dataclass_declaration
            if facet.generates_construction and not any(
                failure.violation is CompactProductAuthorityViolation.GENERATED_INIT_DISABLED
                for failure in facet.failures
            ):
                return tuple(
                    field.name
                    for field in self.dataclass_fields(entry).values()
                    if field.binds_by_position(symbol)
                )
        return ()

    def dataclass_fields(
        self, mro: DeclarationMroType[CompactIndexedClass]
    ) -> dict[str, CompactDataclassFieldDeclaration]:
        """The `__dataclass_fields__` a class with this MRO reads, as `dataclasses`
        builds it: a dataclass merges every MRO entry's `__dataclass_fields__`
        from the root down, then its own, so a redeclared name keeps its first
        position and takes the latest value; any other class inherits the
        first dataclass's along its MRO. A `KW_ONLY` marker stays in, binding
        nothing. Raises when a dataclass's fields cannot be read from its
        source."""

        owner = next(
            (
                entry
                for entry in mro.__mro__
                if isinstance(entry, DeclarationMroType)
                and entry.declaration.dataclass_declaration.generates_construction
            ),
            None,
        )
        if owner is None:
            return {}
        declaration = owner.declaration.dataclass_declaration.require_standard(
            f"`{owner.declaration.symbol}` is no standard dataclass"
        )
        for failure in declaration.failures:
            if failure.violation.hides_fields:
                raise ValueError(
                    f"`{owner.declaration.symbol}`'s fields are not in its source: "
                    f"{failure.violation.explanation}"
                )
        fields: dict[str, CompactDataclassFieldDeclaration] = {}
        for base in reversed(owner.__mro__[1:]):
            if isinstance(base, DeclarationMroType):
                fields.update(self.dataclass_fields(base))
        fields.update((field.name, field) for field in declaration.fields)
        return fields


ClosedLeafMethodAuthorityPredicate: TypeAlias = Callable[
    ["ClosedLeafMethodAuthorityProof"],
    bool,
]
CLASS_METHOD_OWNERSHIP_HOOK_NAMES = frozenset(("__init_subclass__",))
# The metaclasses Python provides: a class reaching one creates classes.
NATIVE_METACLASS_BASES = frozenset(("builtins.type", "abc.ABCMeta"))


class ClosedLeafMethodAuthorityViolation(StrEnum):
    """One failed proof obligation for promoting leaf methods to an ancestor."""

    TOO_FEW_PARTICIPANTS = (
        "too_few_participants",
        "the authority relation requires at least two participating leaves",
        lambda proof: len(proof.participant_symbol_set) < 2,
    )
    AMBIGUOUS_DIRECT_AUTHORITY = (
        "ambiguous_direct_authority",
        "the participants do not have exactly one resolved direct authority",
        lambda proof: (
            frozenset(proof.common_direct_base_symbols)
            != frozenset((proof.authority_symbol,))
        ),
    )
    AMBIGUOUS_DECLARED_AUTHORITY = (
        "ambiguous_declared_authority",
        "the participants do not have exactly one declared nominal base",
        lambda proof: (
            proof.common_declared_nominal_base_simple_names
            != frozenset((proof.authority_simple_name,))
        ),
    )
    INCOMPLETE_DIRECT_FAMILY = (
        "incomplete_direct_family",
        "the participants are not the complete direct-child family",
        lambda proof: (
            frozenset(proof.authority_direct_child_symbols)
            != proof.participant_symbol_set
        ),
    )
    NON_LEAF_PARTICIPANT = (
        "non_leaf_participant",
        "at least one participant still owns a descendant branch",
        lambda proof: bool(proof.non_leaf_participant_symbols),
    )
    INCOMPLETE_BASE_RESOLUTION = (
        "incomplete_base_resolution",
        "a relevant nominal base cannot be resolved from the repository graph",
        lambda proof: bool(proof.incompletely_resolved_symbols),
    )
    METHOD_OWNERSHIP_SENSITIVE_DECLARATION = (
        "method_ownership_sensitive_declaration",
        "a class decorator or metaclass boundary can observe direct method ownership",
        lambda proof: bool(proof.method_ownership_sensitive_symbols),
    )
    EXISTING_AUTHORITY_MEMBER = (
        "existing_authority_member",
        "the authority lineage already binds a promoted member name",
        lambda proof: bool(
            proof.promoted_method_name_set & proof.authority_lineage_member_name_set
        ),
    )
    COMPETING_ANCESTOR_MEMBER = (
        "competing_ancestor_member",
        "another participant ancestor binds a promoted member name",
        lambda proof: bool(
            proof.promoted_method_name_set
            & frozenset(proof.competing_ancestor_member_names)
        ),
    )
    UNDECLARED_RECEIVER_MEMBER = (
        "undeclared_receiver_member",
        "a promoted method requires a receiver member outside the authority contract",
        lambda proof: bool(
            frozenset(proof.receiver_member_names)
            - (proof.authority_lineage_member_name_set | proof.promoted_method_name_set)
        ),
    )

    def __new__(
        cls,
        value: str,
        explanation: str,
        predicate: ClosedLeafMethodAuthorityPredicate,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._explanation = explanation
        member._predicate = predicate
        return member

    @property
    def explanation(self) -> str:
        return self._explanation

    def is_violated_by(self, proof: ClosedLeafMethodAuthorityProof) -> bool:
        return self._predicate(proof)


@dataclass(frozen=True)
class ClosedLeafMethodAuthorityProof:
    """Representation-independent proof of one exact method-promotion owner."""

    authority_symbol: str
    authority_simple_name: str
    participant_symbols: tuple[str, ...]
    common_direct_base_symbols: tuple[str, ...]
    common_declared_nominal_base_names: tuple[str, ...]
    authority_direct_child_symbols: tuple[str, ...]
    non_leaf_participant_symbols: tuple[str, ...]
    incompletely_resolved_symbols: tuple[str, ...]
    method_ownership_sensitive_symbols: tuple[str, ...]
    authority_lineage_member_names: tuple[str, ...]
    competing_ancestor_member_names: tuple[str, ...]
    promoted_method_names: tuple[str, ...]
    receiver_member_names: tuple[str, ...]

    @cached_property
    def participant_symbol_set(self) -> frozenset[str]:
        return frozenset(self.participant_symbols)

    @cached_property
    def promoted_method_name_set(self) -> frozenset[str]:
        return frozenset(self.promoted_method_names)

    @cached_property
    def authority_lineage_member_name_set(self) -> frozenset[str]:
        return frozenset(self.authority_lineage_member_names)

    @cached_property
    def common_declared_nominal_base_simple_names(self) -> frozenset[str]:
        return frozenset(
            name.rsplit(".", 1)[-1] for name in self.common_declared_nominal_base_names
        )

    @cached_property
    def violations(self) -> tuple[ClosedLeafMethodAuthorityViolation, ...]:
        return tuple(
            violation
            for violation in ClosedLeafMethodAuthorityViolation
            if violation.is_violated_by(self)
        )

    @property
    def is_proven(self) -> bool:
        return not self.violations

    @property
    def rejection_reason(self) -> str:
        return "; ".join(violation.explanation for violation in self.violations)


def declared_nominal_base_count(declaration: ClassDeclaration) -> int:
    """Count domain-bearing direct bases under the canonical neutral-base policy."""

    return sum(
        ClassSymbolResolutionAuthority.establishes_nominal_family(base_name)
        for base_name in declaration.declared_base_names
    )


class CompactNominalBindingKind(StrEnum):
    """Kinds of exact module bindings and their exported-alias behavior."""

    MODULE_IMPORT = "module_import", True, True
    NAME_IMPORT = "name_import", True, False
    LOCAL_DECLARATION = "local_declaration", False, False
    BUILTIN = "builtin", False, False

    projects_as_import_alias: bool
    binds_module: bool

    def __new__(
        cls, value: str, projects_as_import_alias: bool, binds_module: bool
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.projects_as_import_alias = projects_as_import_alias
        member.binds_module = binds_module
        return member


class ModuleNominalBindingSnapshotPolicy(StrEnum):
    """Typed ambiguity policy for exact proof versus named-import projection."""

    EXACT = "exact", True
    NAMED_IMPORT_PROJECTION = "named_import_projection", False

    invalidates_on_star_import: bool

    def __new__(cls, value: str, invalidates_on_star_import: bool) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.invalidates_on_star_import = invalidates_on_star_import
        return member


@dataclass(frozen=True)
class CompactNominalBinding:
    """One exact nominal origin selected by sequential module binding."""

    qualified_name: str
    kind: CompactNominalBindingKind


DecoratedDefinition: TypeAlias = (
    ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | FunctionDefinition
)


@dataclass(frozen=True)
class ModuleNominalBindingSnapshot(DecoratorBindings):
    """Exact known and unresolved bindings before one module source position."""

    bindings_by_name: dict[str, CompactNominalBinding]
    annotation_mode: type[ModuleAnnotationEvaluationMode]
    """How the module's annotations are evaluated: an annotation naming nothing
    bound reaches `dataclasses` as a string or forward reference unless they
    execute where they are declared."""
    unresolved_bound_names: frozenset[str] = frozenset()
    star_import_ambiguity: bool = False
    star_import_excluded_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        overlap = self.bindings_by_name.keys() & self.unresolved_bound_names
        if overlap:
            raise ValueError(
                "module binding names cannot be both resolved and unresolved: "
                f"{tuple(sorted(overlap))!r}"
            )

    def binding_for(self, name: str) -> CompactNominalBinding | None:
        return self.bindings_by_name.get(name)

    def reference_for(self, parts: tuple[str, ...]) -> CompactNominalReference:
        """Project a name path through its exact module or builtin binding."""

        binding = self.binding_for(parts[0])
        if binding is None and self.resolves_unshadowed_builtin(parts[0]):
            binding = CompactNominalBinding(
                qualified_name=f"builtins.{parts[0]}",
                kind=CompactNominalBindingKind.BUILTIN,
            )
        return CompactNominalReference.bound_as(parts, binding)

    def decorator_references(
        self, node: DecoratedDefinition
    ) -> tuple[CompactNominalReference, ...]:
        """What each decorator of the definition names through these bindings, in
        order; a decorator that is no name path (`@registry[key]`) is unbound."""

        return tuple(self._decorator_reference(decorator) for decorator in node.decorator_list)

    def decorators_bound_to(
        self, definition: ast.FunctionDef | ast.AsyncFunctionDef, identities: frozenset[str]
    ) -> tuple[bool, ...]:
        return tuple(
            any(reference.has_bound_name(identity) for identity in identities)
            for reference in self.decorator_references(definition)
        )

    def decorated_by(
        self,
        node: DecoratedDefinition,
        identities: Collection[str],
    ) -> bool:
        """Whether a decorator of the definition binds one of these objects."""

        return any(
            reference.has_bound_name(identity)
            for reference in self.decorator_references(node)
            for identity in identities
        )

    def _decorator_reference(self, decorator: ast.expr) -> CompactNominalReference:
        callee = decorator.func if isinstance(decorator, ast.Call) else decorator
        match ExpressionReference.decoded(callee).attribute_chain:
            case (str(), *_) as parts:
                return self.reference_for(parts)
        return UnboundNominalReference((ast.unparse(decorator),))

    def resolves_unshadowed_builtin(
        self,
        name: str,
        *,
        preceding_class_bound_names: frozenset[str] = frozenset(),
    ) -> bool:
        """Prove that one bare name still resolves through Python builtins."""

        return self.binds_nothing_to(
            name, preceding_class_bound_names=preceding_class_bound_names
        ) and name in vars(builtins)

    def binds_nothing_to(
        self,
        name: str,
        *,
        preceding_class_bound_names: frozenset[str] = frozenset(),
    ) -> bool:
        """Prove that the module's namespace holds no `name` here: no binding, no
        binding NRA cannot read, no star import that may bind it, and no earlier
        class-body binding. Builtins are not in that namespace."""

        return (
            (not self.star_import_ambiguity or name in self.star_import_excluded_names)
            and name not in self.bindings_by_name
            and name not in self.unresolved_bound_names
            and name not in preceding_class_bound_names
        )

    def reads_as_absent(
        self,
        annotation: ast.expr,
        root_name: str,
        *,
        preceding_class_bound_names: frozenset[str] = frozenset(),
    ) -> bool:
        """Whether `dataclasses` reads a string annotation rooted at `root_name` as
        naming nothing: the module binds nothing to the name at the class.

        For a string annotation, `dataclasses._is_type` matches its leading
        `[module.]name` and looks the name up in the module's namespace while
        the class is created (CPython `Lib/dataclasses.py`), never in builtins,
        so a name bound only later, or never, is no `ClassVar`, `InitVar` or
        `KW_ONLY`. A module that stores its annotations without executing them
        (`from __future__ import annotations`) stores strings. A deferred
        annotation (PEP 649) is evaluated with builtins, which another module
        may extend, so an unbound name there stays unjudged.
        """

        mode = self.annotation_mode
        return self.binds_nothing_to(
            root_name, preceding_class_bound_names=preceding_class_bound_names
        ) and (
            (mode.stores_variable_annotations_at_definition and not mode.annotations_execute_at_declaration)
            or (isinstance(annotation, ast.Constant) and isinstance(annotation.value, str))
        )


@dataclass(frozen=True)
class ModuleNominalBindingAuthority:
    """Resolve nominal module bindings at an exact source position, fail closed."""

    parsed_module: ParsedModule
    declared_assignment_authority_names: frozenset[str] = frozenset()

    def snapshot_before(
        self,
        line: int | None = None,
        *,
        policy: ModuleNominalBindingSnapshotPolicy = (
            ModuleNominalBindingSnapshotPolicy.EXACT
        ),
    ) -> ModuleNominalBindingSnapshot:
        return self.snapshots_before((line,), policy=policy)[line]

    def snapshots_before(
        self,
        lines: Iterable[int | None],
        *,
        policy: ModuleNominalBindingSnapshotPolicy = ModuleNominalBindingSnapshotPolicy.EXACT,
    ) -> dict[int | None, ModuleNominalBindingSnapshot]:
        """Resolve requested declaration positions and final bindings in one pass."""
        requested_lines = tuple(dict.fromkeys(lines))
        if not self.declared_assignment_authority_names and (
            policy is ModuleNominalBindingSnapshotPolicy.EXACT
            or not module_star_import_origins(self.parsed_module)
        ):
            timeline = self.parsed_module.fact(ModuleNominalBindingTimeline)
            return {
                line: (
                    timeline.final_snapshot
                    if line is None
                    else timeline.snapshot_before_line(line)
                )
                for line in requested_lines
            }
        return _module_nominal_binding_snapshots(
            self,
            tuple(line for line in requested_lines if line is not None),
            include_final=None in requested_lines,
            policy=policy,
        )

    def qualified_name_at(
        self,
        reference: ast.AST,
        *,
        line: int | None,
        policy: ModuleNominalBindingSnapshotPolicy = (
            ModuleNominalBindingSnapshotPolicy.EXACT
        ),
    ) -> str | None:
        parts = ExpressionReference.decoded(
            ClassSymbolResolutionAuthority.reference_node(reference)
        ).attribute_chain
        if parts is None:
            return None
        root_binding = self.snapshot_before(line, policy=policy).binding_for(parts[0])
        if root_binding is None:
            return None
        return ".".join((root_binding.qualified_name, *parts[1:]))

    def nominal_annotation_name_at(
        self,
        annotation: ast.AST,
        *,
        line: int,
        class_bound_names: frozenset[str] = frozenset(),
    ) -> str | None:
        """Resolve a single nominal annotation through its declaration-time binding."""

        reference = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_or_none(annotation)
        if (
            reference is None
            or nominal_reference_root_name(reference) in class_bound_names
        ):
            return None
        return self.qualified_name_at(reference, line=line)

    def class_annotation_name_at(
        self,
        annotation: ast.AST,
        *,
        line: int,
        class_bound_names: frozenset[str] = frozenset(),
    ) -> str | None:
        """Resolve type[C] separately from an instance annotation C."""
        envelope = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.subscript_or_none(annotation)
        if envelope is None:
            return None
        if not self._annotation_factory_is(
            envelope.value,
            ("builtins.type",),
            line=line,
            class_bound_names=class_bound_names,
        ):
            return None
        return self.nominal_annotation_name_at(
            envelope.slice, line=line, class_bound_names=class_bound_names
        )

    def _annotation_factory_is(
        self,
        reference: ast.AST,
        factory_names: tuple[str, ...],
        *,
        line: int,
        class_bound_names: frozenset[str],
    ) -> bool:
        """Whether a binding proves the annotation's factory is one of these names."""

        parts = ExpressionReference.decoded(reference).attribute_chain
        if parts is None or parts[0] in class_bound_names:
            return False
        resolved = self.snapshot_before(line).reference_for(parts)
        return any(resolved.has_bound_name(name) for name in factory_names)

    def iterated_class_annotation_name_at(
        self,
        annotation: ast.AST,
        *,
        line: int,
        class_bound_names: frozenset[str] = frozenset(),
    ) -> str | None:
        """One represented class in a declared tuple, not guessed iterable flow."""
        envelope = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.subscript_or_none(annotation)
        if envelope is None:
            return None
        if not self._annotation_factory_is(
            envelope.value,
            ("builtins.tuple", "typing.Tuple"),
            line=line,
            class_bound_names=class_bound_names,
        ):
            return None
        items = (
            envelope.slice.elts
            if isinstance(envelope.slice, ast.Tuple)
            else [envelope.slice]
        )
        if (
            len(items) == 2
            and isinstance(items[-1], ast.Constant)
            and items[-1].value is Ellipsis
        ):
            items = items[:1]
        types = {
            self.class_annotation_name_at(
                item, line=line, class_bound_names=class_bound_names
            )
            for item in items
        }
        return next(iter(types)) if len(types) == 1 else None

    def _direct_nominal_bindings(
        self,
        statement: ast.stmt,
        preceding_bindings: dict[str, CompactNominalBinding],
    ) -> dict[str, CompactNominalBinding]:
        if isinstance(statement, ast.Import | ast.ImportFrom):
            # `import x` always binds a module; `from x import y` may bind anything.
            kind = (
                CompactNominalBindingKind.MODULE_IMPORT
                if isinstance(statement, ast.Import)
                else CompactNominalBindingKind.NAME_IMPORT
            )
            return {
                origin.bound_name: CompactNominalBinding(
                    qualified_name=origin.qualified_name,
                    kind=kind,
                )
                for origin in ImportBoundNameProjection(statement).origins(
                    self.parsed_module.module_path_identity
                )
                if origin.qualified_name is not None
            }
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            return {
                statement.name: CompactNominalBinding(
                    qualified_name=f"{self.parsed_module.module_name}.{statement.name}",
                    kind=CompactNominalBindingKind.LOCAL_DECLARATION,
                )
            }
        if not isinstance(statement, ast.Assign | ast.AnnAssign):
            return {}
        local_name = _single_assignment_target_name(statement)
        value = statement.value
        if local_name is None or value is None:
            return {}
        if local_name in self.declared_assignment_authority_names:
            return {
                local_name: CompactNominalBinding(
                    qualified_name=f"{self.parsed_module.module_name}.{local_name}",
                    kind=CompactNominalBindingKind.LOCAL_DECLARATION,
                )
            }
        parts = ExpressionReference.decoded(
            ClassSymbolResolutionAuthority.reference_node(value)
        ).attribute_chain
        if parts is None or (root_binding := preceding_bindings.get(parts[0])) is None:
            return {}
        return {
            local_name: CompactNominalBinding(
                qualified_name=".".join((root_binding.qualified_name, *parts[1:])),
                kind=root_binding.kind,
            )
        }


def nominal_reference_root(reference: ast.AST) -> ast.Name | None:
    """Return the lexical root node whose binding owns a nominal reference."""

    while isinstance(reference, ast.Attribute):
        reference = reference.value
    return reference if isinstance(reference, ast.Name) else None


def nominal_reference_root_name(reference: ast.AST) -> str | None:
    """Return the lexical root name whose binding owns a nominal reference."""

    root = nominal_reference_root(reference)
    return None if root is None else root.id


class ModuleNominalBindingView(ABC):
    """Representation-independent nominal bindings at one module position."""

    source_entry = staticmethod(ImportedSourceModuleEntryPremise.from_source)

    @cached_property
    def _source_projections(self) -> dict[int, SourceProductFlowProjection]:
        return {}

    def source_projection(self, module: ParsedModule) -> SourceProductFlowProjection:
        """Retain one original observation graph per parsed owner in this proof."""
        source = self._source_projections.get(id(module))
        if source is None:
            source = source_product_flow_projection(module)
            self._source_projections[id(module)] = source
        return source

    def native_reference_environment(
        self, module: ParsedModule
    ) -> NativeReferenceEnvironment:
        """Admit execution over this proof's existing original observations."""
        environment = self._native_executions.get(id(module))
        if environment is None:
            source = self.source_projection(module)
            entry = self.source_entry(source)
            if entry.source is not source:
                raise ValueError(
                    "Source entry belongs to a different canonical projection"
                )
            environment = SourceModuleExecution(entry)
            self._native_executions[id(module)] = environment
        return environment

    @cached_property
    def _native_executions(self) -> dict[int, NativeReferenceEnvironment]:
        return {}

    def reference_or_builtin_witness_at(
        self,
        module: ParsedModule,
        reference: ast.expr,
        *,
        line: int,
        preceding_class_bound_names: frozenset[str] = frozenset(),
    ) -> ModuleNominalBindingWitness | None:
        root = nominal_reference_root_name(reference)
        if root in preceding_class_bound_names:
            return None
        witness = self.reference_witness_at(module, reference, line=line)
        if witness is None and isinstance(reference, ast.Name):
            return self.unshadowed_builtin_witness(
                module,
                reference.id,
                line=line,
                preceding_class_bound_names=preceding_class_bound_names,
            )
        return witness

    def require_native_type_in_class(
        self,
        module: ParsedModule,
        owner: ast.ClassDef,
        declaration: type,
    ) -> None:
        """Prove the native type emitted into a class resolves to its declaration."""

        name = declaration.__name__
        class_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(owner.body)
        if name in class_names:
            raise ValueError(f"Class namespace shadows native type {name!r}")
        witness = self.reference_or_builtin_witness_at(
            module,
            ast.Name(id=name, ctx=ast.Load()),
            line=owner.lineno,
            preceding_class_bound_names=class_names,
        )
        qualified_name = NativeDeclaration(declaration).qualified_name
        if witness is None or witness.qualified_name != qualified_name:
            raise ValueError(
                f"Class creation does not prove native type binding {qualified_name!r}"
            )

    @abstractmethod
    def unshadowed_builtin_witness(
        self,
        module: ParsedModule,
        name: str,
        *,
        line: int,
        preceding_class_bound_names: frozenset[str],
    ) -> ModuleNominalBindingWitness | None:
        raise NotImplementedError

    @abstractmethod
    def reference_witness_at(
        self,
        module: ParsedModule,
        reference: ast.expr,
        *,
        line: int,
    ) -> ModuleNominalBindingWitness | None:
        raise NotImplementedError


@dataclass(frozen=True)
class ModuleNominalBindingWitness:
    """One resolved declaration and the lexical root requiring star exclusion."""

    qualified_name: str
    root_name: str


class ClassBodyDeclarationKind(StrEnum):
    """What one direct class-body statement declares, the receipt two classes
    sharing it earn, and whether it is a field.

    A supplied value weighs as a value whether or not it is a field's default.
    """

    shared_receipt: OverlapReceiptKind
    is_field: bool
    is_method: bool

    def __new__(
        cls,
        value: str,
        shared_receipt: OverlapReceiptKind,
        is_field: bool,
        is_method: bool,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.shared_receipt = shared_receipt
        member.is_field = is_field
        member.is_method = is_method
        return member

    FIELD = "field", OverlapReceiptKind.SHARED_FIELD, True, False
    DEFAULTED_FIELD = "defaulted_field", OverlapReceiptKind.SHARED_VALUE, True, False
    VALUE = "value", OverlapReceiptKind.SHARED_VALUE, False, False
    NOOP_METHOD = "noop_method", OverlapReceiptKind.SHARED_NOOP_METHOD, False, True
    METHOD = "method", OverlapReceiptKind.SHARED_METHOD, False, True


class AnnotatedDeclarationKind(StatementValueQuestion[ClassBodyDeclarationKind]):
    """An annotation declares a field; with a value it also supplies the default."""

    def supplied(self, value: SuppliedValue) -> ClassBodyDeclarationKind:
        return ClassBodyDeclarationKind.DEFAULTED_FIELD

    def bare(self, value: NoStatementValue) -> ClassBodyDeclarationKind:
        return ClassBodyDeclarationKind.FIELD


class _ReceiverAliasSubstitution(ast.NodeTransformer):
    """Substitute one proven local receiver alias in a copied method tree."""

    def __init__(self, alias_name: str, receiver_name: str) -> None:
        self.alias_name = alias_name
        self.receiver_name = receiver_name

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id != self.alias_name:
            return node
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Receiver alias is rebound after its declaration")
        return ast.copy_location(ast.Name(id=self.receiver_name, ctx=node.ctx), node)


class StarDeferredName(NamedTuple):
    """A free name a star import may rebind, with the binding it has if none does.

    `unshadowed` is the qualified name the module binds without its star
    imports (`None` when nothing does); the repository decides which applies.
    """

    name: str
    unshadowed: str | None


IdentityT = TypeVar("IdentityT")


class CompactFieldFact(ABC):
    """The bare field, `name: annotation`, one class-body declaration makes:
    none, the declaration itself, or what a call that declares one names."""

    depends_on_repository: ClassVar[bool] = False

    def static_field_identity(self, identity: IdentityT) -> IdentityT | None:
        """The field identity when independent of repository field wrappers."""

        del identity
        return None

    @abstractmethod
    def field_identity(
        self,
        identity: IdentityT,
        bare_identity: Callable[[ClassBodyDeclarationDigest], IdentityT | None],
        field_declarations: FieldDeclarationIndex,
    ) -> IdentityT | None:
        raise NotImplementedError


@dataclass(frozen=True)
class NoFieldFact(CompactFieldFact):
    """A value or a method: it declares no field."""

    def field_identity(
        self,
        identity: IdentityT,
        bare_identity: Callable[[ClassBodyDeclarationDigest], IdentityT | None],
        field_declarations: FieldDeclarationIndex,
    ) -> IdentityT | None:
        return None


@dataclass(frozen=True)
class CompactBareFieldFact(CompactFieldFact):
    """`name: annotation` with no value: the declaration is its own field."""

    def static_field_identity(self, identity: IdentityT) -> IdentityT | None:
        return identity

    def field_identity(
        self,
        identity: IdentityT,
        bare_identity: Callable[[ClassBodyDeclarationDigest], IdentityT | None],
        field_declarations: FieldDeclarationIndex,
    ) -> IdentityT | None:
        return identity


@dataclass(frozen=True)
class CompactCalledFieldFact(CompactFieldFact):
    """A field whose value is a call: it declares `bare` when the repository
    proves the call declares a field without a default."""

    depends_on_repository: ClassVar[bool] = True

    call: CompactFieldDeclarationCall
    bare: ClassBodyDeclarationDigest

    def field_identity(
        self,
        identity: IdentityT,
        bare_identity: Callable[[ClassBodyDeclarationDigest], IdentityT | None],
        field_declarations: FieldDeclarationIndex,
    ) -> IdentityT | None:
        if field_declarations.declares_bare_field(self.call) is not True:
            return None
        return bare_identity(self.bare)


class AnnotatedFieldFact(StatementValueQuestion[CompactFieldFact]):
    """An annotation alone is its own field; with a plain value it is a value."""

    def supplied(self, value: SuppliedValue) -> CompactFieldFact:
        return NoFieldFact()

    def bare(self, value: NoStatementValue) -> CompactFieldFact:
        return CompactBareFieldFact()


class ClassBodyDeclarationDigest(NamedTuple):
    """One class-body declaration as the index compares it: no AST.

    `digest` covers the declaration's kind, name, syntax and class-scope
    names; `bindings` what each free name the module binds resolves to, as
    `(name, qualified name)`. `deferred` lists the free names only the
    repository's star-import exposure can resolve, which the index settles
    into the same bindings, and `unbound` the free names nothing binds, which
    make it equal to nothing.
    """

    kind: ClassBodyDeclarationKind
    name: str
    line: int
    digest: str
    bindings: tuple[tuple[str, str], ...]
    deferred: tuple[StarDeferredName, ...]
    unbound: tuple[str, ...]
    field_declaration: CompactFieldFact = NoFieldFact()
    """The bare field, `name: annotation`, the declaration makes, if any."""
    reads_its_class: bool = False
    """Whether its meaning depends on the class body it sits in
    (`ClassBodyNormalForm.reads_its_class`), so a copy in another class means
    something else even when the two are the same declaration."""


@dataclass(frozen=True)
class ClassBodyModuleScope:
    """What a free name in one class body is bound to.

    `declaration` is the module snapshot where the class statement runs and
    `final` the module after import. A name bound at the class statement keeps
    that binding; a name bound only later is read where evaluation is deferred
    (method bodies, postponed annotations), and Python raises at import for any
    other read of it, so code that imports reads the final binding. A name the
    module binds by assignment rather than import or definition is identified by
    the module that binds it.
    """

    module_name: str
    declaration: ModuleNominalBindingSnapshot
    final: ModuleNominalBindingSnapshot
    star_blind_declaration: ModuleNominalBindingSnapshot
    star_blind_final: ModuleNominalBindingSnapshot

    @classmethod
    def for_lines(
        cls, parsed_module: ParsedModule, lines: Iterable[int]
    ) -> dict[int, ClassBodyModuleScope]:
        """One scope per class line, from two passes over the module's bindings."""

        requested = tuple(dict.fromkeys(lines))
        authority = ModuleNominalBindingAuthority(parsed_module)
        exact = authority.snapshots_before((*requested, None))
        star_blind = authority.snapshots_before(
            (*requested, None),
            policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
        )
        return {
            line: cls(
                parsed_module.module_name,
                exact[line],
                exact[None],
                star_blind[line],
                star_blind[None],
            )
            for line in requested
        }

    def witness_for(self, name: str) -> ModuleNominalBindingWitness | None:
        return self._witness_in(name, self.declaration, self.final)

    def deferred_for(self, name: str) -> StarDeferredName | None:
        """A name a star import hides, with its binding if no star import rebinds it."""

        if not self.final.star_import_ambiguity:
            return None
        witness = self._witness_in(
            name, self.star_blind_declaration, self.star_blind_final
        )
        return StarDeferredName(
            name, None if witness is None else witness.qualified_name
        )

    def _witness_in(
        self,
        name: str,
        declaration: ModuleNominalBindingSnapshot,
        final: ModuleNominalBindingSnapshot,
    ) -> ModuleNominalBindingWitness | None:
        for snapshot in (declaration, final):
            if (binding := snapshot.binding_for(name)) is not None:
                return ModuleNominalBindingWitness(binding.qualified_name, name)
        if declaration.resolves_unshadowed_builtin(name):
            return ModuleNominalBindingWitness(f"{builtins.__name__}.{name}", name)
        if name in declaration.unresolved_bound_names or name in final.unresolved_bound_names:
            return ModuleNominalBindingWitness(f"{self.module_name}.{name}", name)
        return None


def _is_zero_arg_super_call(node: ast.Call) -> bool:
    return (
        not node.args
        and not node.keywords
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
    )


def reads_class_cell_or_private_name(statement: ast.stmt) -> bool:
    """Whether a class-body statement reads the class it sits in: a zero-argument
    `super()` or `__class__` reads the class cell, and a `__name` identifier is
    mangled with the class's name."""

    stack: list[ast.AST] = [statement]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Name):
            if node.id == "__class__" or _is_private_name(node.id):
                return True
        elif isinstance(node, ast.Attribute):
            if _is_private_name(node.attr):
                return True
        elif isinstance(node, ast.Call):
            if _is_zero_arg_super_call(node):
                return True
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if _is_private_name(node.name):
                return True
        elif isinstance(node, ast.arg):
            if _is_private_name(node.arg):
                return True
        for field_name in node._fields:
            child = getattr(node, field_name, None)
            if isinstance(child, ast.AST):
                stack.append(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, ast.AST):
                        stack.append(item)
    return False


def _is_private_name(name: str) -> bool:
    return name.startswith("__") and not name.endswith("__")


PER_CLASS_NAMES = frozenset(("__slots__",))
"""Names class creation reads from the class's own body, never through its
bases: `__slots__` restated in a subclass is not a copy, since without it the
subclass's instances get a `__dict__`, and one in a base is no declaration the
subclass shares."""


@dataclass(frozen=True)
class ClassBodyNormalForm:
    """One class-body declaration as Python evaluates it.

    Two declarations are the same declaration when their kind, name, syntax
    (positions and a leading docstring dropped, a method's `x: T = self`
    receiver alias substituted) and the declaration each free name is bound to
    are equal. A free name bound earlier in the class body is recorded by name,
    so it is equal only to the same class-scope name; a free name the module
    snapshot cannot bind leaves the declaration unresolved, and an unresolved
    declaration equals nothing in the class-body index.
    """

    kind: ClassBodyDeclarationKind
    name: str
    line: int
    syntax: str
    global_bindings: tuple[ModuleNominalBindingWitness, ...]
    class_scope_names: tuple[str, ...]
    deferred_names: tuple[StarDeferredName, ...]
    unresolved_names: tuple[str, ...]
    field_declaration: CompactFieldFact = NoFieldFact()
    reads_its_class: bool = False
    """It reads the class it sits in: a zero-argument `super()` or `__class__`
    (the class cell), a private name the class's name mangles, or a free name
    the class body binds. Equality ignores it; whether a copy in a subclass
    restates the definition it would inherit does not."""

    @property
    def is_resolved(self) -> bool:
        """Every free name is bound inside the module, star imports aside."""

        return not self.unresolved_names and not self.deferred_names

    @cached_property
    def digest(self) -> str:
        return hashlib.blake2s(
            repr(
                (
                    self.kind.value,
                    self.name,
                    self.syntax,
                    tuple(
                        (binding.qualified_name, binding.root_name)
                        for binding in self.global_bindings
                    ),
                    self.class_scope_names,
                )
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    @cached_property
    def syntax_digest(self) -> str:
        """The digest of everything but the free names' bindings."""

        return hashlib.blake2s(
            repr((self.kind.value, self.name, self.syntax, self.class_scope_names)).encode(
                "utf-8"
            ),
            digest_size=16,
        ).hexdigest()

    def compact(self) -> ClassBodyDeclarationDigest:
        return ClassBodyDeclarationDigest(
            self.kind,
            self.name,
            self.line,
            self.syntax_digest,
            tuple(
                (binding.root_name, binding.qualified_name)
                for binding in self.global_bindings
            ),
            self.deferred_names,
            self.unresolved_names,
            self.field_declaration,
            self.reads_its_class,
        )

    @classmethod
    def of_class(
        cls,
        node: ast.ClassDef,
        scope: ClassBodyModuleScope,
    ) -> tuple[ClassBodyNormalForm, ...]:
        """Every direct declaration of one class body a subclass can share, in
        source order: the per-class names are no such declaration."""

        class_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
        declarations: list[ClassBodyNormalForm] = []
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                declarations.append(cls.of_method(statement, scope, class_bound_names))
            elif isinstance(statement, (ast.AnnAssign, ast.Assign)):
                declarations.extend(
                    cls.of_assignment(statement, scope, class_bound_names)
                )
        return tuple(declaration for declaration in declarations if declaration.name not in PER_CLASS_NAMES)

    @classmethod
    def _is_noop_stub_method(
        cls,
        normalized: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        """A concrete method whose body after any docstring is only `pass`, `...`,
        `return`, or `return None` and that is not decorated as `@abstractmethod`
        supplies no implementation."""

        if any(
            _terminal_reference_name(decorator) == "abstractmethod"
            for decorator in normalized.decorator_list
        ):
            return False
        return all(
            isinstance(stmt, ast.Pass)
            or (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is Ellipsis
            )
            or (
                isinstance(stmt, ast.Return)
                and (
                    stmt.value is None
                    or (
                        isinstance(stmt.value, ast.Constant)
                        and stmt.value.value is None
                    )
                )
            )
            for stmt in normalized.body
        )

    @classmethod
    def of_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        scope: ClassBodyModuleScope,
        class_bound_names: frozenset[str] = frozenset(),
    ) -> ClassBodyNormalForm:
        """A method's header evaluates in the class scope and its body in the module's.

        A free name the class body binds is recorded as a class-scope name and,
        when the module binds it too, with that binding, so equality holds only
        when both scopes agree.
        """

        normalized = cls._normalized_method(method)
        kind = (
            ClassBodyDeclarationKind.NOOP_METHOD
            if cls._is_noop_stub_method(normalized)
            else ClassBodyDeclarationKind.METHOD
        )
        return cls._of_statement(
            kind,
            method.name,
            method,
            normalized,
            scope,
            class_bound_names,
        )

    @classmethod
    def of_assignment(
        cls,
        statement: ast.AnnAssign | ast.Assign,
        scope: ClassBodyModuleScope,
        class_bound_names: frozenset[str],
    ) -> tuple[ClassBodyNormalForm, ...]:
        if isinstance(statement, ast.AnnAssign):
            if not isinstance(statement.target, ast.Name):
                return ()
            kind = StatementValue.of(statement).answer(AnnotatedDeclarationKind())
            if isinstance(statement.value, ast.Call):
                return (
                    cls.of_called_field(statement, statement.value, scope, class_bound_names),
                )
            return (
                replace(
                    cls._of_statement(
                        kind,
                        statement.target.id,
                        statement,
                        statement,
                        scope,
                        class_bound_names - {statement.target.id},
                    ),
                    field_declaration=StatementValue.of(statement).answer(
                        AnnotatedFieldFact()
                    ),
                ),
            )
        names = tuple(
            target.id for target in statement.targets if isinstance(target, ast.Name)
        )
        return tuple(
            cls._of_statement(
                ClassBodyDeclarationKind.VALUE,
                name,
                statement,
                statement,
                scope,
                class_bound_names - {name},
            )
            for name in names
        )

    @classmethod
    def of_called_field(
        cls,
        statement: ast.AnnAssign,
        call: ast.Call,
        scope: ClassBodyModuleScope,
        class_bound_names: frozenset[str],
    ) -> ClassBodyNormalForm:
        """A field whose value is a call, with the call and the bare field it
        declares if the call declares a field without a default."""

        name = cast(ast.Name, statement.target).id
        bound_names = class_bound_names - {name}
        bare_statement = ast.copy_location(
            ast.AnnAssign(
                target=statement.target,
                annotation=statement.annotation,
                value=None,
                simple=statement.simple,
            ),
            statement,
        )
        parts = ExpressionReference.decoded(call.func).attribute_chain
        witness = (
            None
            if parts is None or parts[0] in bound_names
            else scope.witness_for(parts[0])
        )
        declaration = cls._of_statement(
            ClassBodyDeclarationKind.DEFAULTED_FIELD,
            name,
            statement,
            statement,
            scope,
            bound_names,
        )
        if witness is None:
            return declaration
        return replace(
            declaration,
            field_declaration=CompactCalledFieldFact(
                CompactFieldDeclarationCall.of_call(
                    ".".join((witness.qualified_name, *cast(tuple[str, ...], parts)[1:])),
                    call,
                ),
                cls._of_statement(
                    ClassBodyDeclarationKind.FIELD,
                    name,
                    bare_statement,
                    bare_statement,
                    scope,
                    bound_names,
                ).compact(),
            ),
        )

    @classmethod
    def _of_statement(
        cls,
        kind: ClassBodyDeclarationKind,
        name: str,
        statement: ast.stmt,
        normalized: ast.stmt,
        scope: ClassBodyModuleScope,
        class_bound_names: frozenset[str],
    ) -> ClassBodyNormalForm:
        free_names = sorted(
            {
                reference.id
                for reference in ModuleLexicalDependencyProjection.from_module(
                    ast.Module(body=[normalized], type_ignores=[])
                ).external_name_references
            }
        )
        global_bindings: list[ModuleNominalBindingWitness] = []
        class_scope_names: list[str] = []
        deferred_names: list[StarDeferredName] = []
        unresolved_names: list[str] = []
        for free_name in free_names:
            if free_name in class_bound_names:
                class_scope_names.append(free_name)
                if not kind.is_method:
                    continue
            if (witness := scope.witness_for(free_name)) is not None:
                global_bindings.append(witness)
            elif free_name in class_bound_names:
                continue
            elif (deferred := scope.deferred_for(free_name)) is not None:
                deferred_names.append(deferred)
            else:
                unresolved_names.append(free_name)
        return cls(
            kind=kind,
            name=name,
            line=statement.lineno,
            syntax=ast.dump(normalized, include_attributes=False),
            global_bindings=tuple(global_bindings),
            class_scope_names=tuple(class_scope_names),
            deferred_names=tuple(deferred_names),
            unresolved_names=tuple(unresolved_names),
            reads_its_class=bool(class_scope_names) or reads_class_cell_or_private_name(statement),
        )

    @classmethod
    def _normalized_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        normalized = copy.copy(method)
        normalized.body = _trim_leading_docstring(list(method.body)) or [ast.Pass()]
        positional = (*normalized.args.posonlyargs, *normalized.args.args)
        receiver_name = positional[0].arg if positional else None
        if receiver_name is None or len(normalized.body) != 2:
            return normalized
        alias_declaration, remaining_statement = normalized.body
        if not (
            isinstance(alias_declaration, ast.AnnAssign)
            and alias_declaration.simple == 1
            and isinstance(alias_declaration.target, ast.Name)
            and isinstance(alias_declaration.value, ast.Name)
            and alias_declaration.value.id == receiver_name
            and alias_declaration.target.id != receiver_name
        ):
            return normalized
        alias_name = alias_declaration.target.id
        alias_references = tuple(
            node
            for node in ast.walk(remaining_statement)
            if isinstance(node, ast.Name) and node.id == alias_name
        )
        if not alias_references or any(
            not isinstance(reference.ctx, ast.Load) for reference in alias_references
        ):
            return normalized
        normalized.body = [
            _ReceiverAliasSubstitution(alias_name, receiver_name).visit(
                copy.deepcopy(remaining_statement)
            )
        ]
        return ast.fix_missing_locations(normalized)


@dataclass(frozen=True)
class NamedImportModuleBindingProjection(ModuleNominalBindingView):
    """Module-local binding view whose star-import obligations remain explicit."""

    def unshadowed_builtin_witness(
        self,
        module: ParsedModule,
        name: str,
        *,
        line: int,
        preceding_class_bound_names: frozenset[str],
    ) -> ModuleNominalBindingWitness | None:
        is_unshadowed = (
            ModuleNominalBindingAuthority(module)
            .snapshot_before(
                line,
                policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
            )
            .resolves_unshadowed_builtin(
                name,
                preceding_class_bound_names=preceding_class_bound_names,
            )
        )
        return (
            ModuleNominalBindingWitness(f"builtins.{name}", name)
            if is_unshadowed
            else None
        )

    def reference_witness_at(
        self,
        module: ParsedModule,
        reference: ast.expr,
        *,
        line: int,
    ) -> ModuleNominalBindingWitness | None:
        root_name = nominal_reference_root_name(reference)
        if root_name is None:
            return None
        qualified_name = ModuleNominalBindingAuthority(module).qualified_name_at(
            reference,
            line=line,
            policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
        )
        if qualified_name is None:
            return None
        return ModuleNominalBindingWitness(qualified_name, root_name)


def _direct_binding_target_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Starred):
        return _direct_binding_target_names(target.value)
    if isinstance(target, ast.Tuple | ast.List):
        return tuple(
            name
            for element in target.elts
            for name in _direct_binding_target_names(element)
        )
    return ()


def _direct_statement_bound_names(statement: ast.stmt) -> frozenset[str]:
    """Project common direct bindings without allocating a scope visitor."""

    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return frozenset((statement.name,))
    if isinstance(statement, ast.Import | ast.ImportFrom):
        return frozenset(ImportBoundNameProjection(statement).names())
    if isinstance(statement, ast.Assign):
        return frozenset(
            name
            for target in statement.targets
            for name in _direct_binding_target_names(target)
        )
    if isinstance(statement, ast.AnnAssign | ast.AugAssign):
        return frozenset(_direct_binding_target_names(statement.target))
    if isinstance(statement, ast.Delete):
        return frozenset(
            name
            for target in statement.targets
            for name in _direct_binding_target_names(target)
        )
    return LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names((statement,))


def _module_nominal_binding_snapshots(
    authority: ModuleNominalBindingAuthority,
    lines: tuple[int, ...],
    *,
    include_final: bool,
    policy: ModuleNominalBindingSnapshotPolicy = (
        ModuleNominalBindingSnapshotPolicy.EXACT
    ),
    star_import_excluded_names: frozenset[str] = frozenset(),
) -> dict[int | None, ModuleNominalBindingSnapshot]:
    parsed_module = authority.parsed_module
    bindings: dict[str, CompactNominalBinding] = {}
    unresolved_bound_names: set[str] = set()
    star_import_ambiguity = False
    annotation_mode = ModuleAnnotationEvaluationMode.from_module(parsed_module.module)
    statements = iter(parsed_module.module.body)
    statement = next(statements, None)
    snapshots: dict[int | None, ModuleNominalBindingSnapshot] = {}

    def snapshot() -> ModuleNominalBindingSnapshot:
        return ModuleNominalBindingSnapshot(
            bindings_by_name=dict(bindings),
            annotation_mode=annotation_mode,
            unresolved_bound_names=frozenset(unresolved_bound_names),
            star_import_ambiguity=star_import_ambiguity,
            star_import_excluded_names=star_import_excluded_names,
        )

    def apply(current: ast.stmt) -> None:
        nonlocal star_import_ambiguity
        if (
            policy.invalidates_on_star_import
            and isinstance(current, ast.ImportFrom)
            and any(alias.name == "*" for alias in current.names)
        ):
            for binding_name in tuple(bindings):
                if binding_name not in star_import_excluded_names:
                    del bindings[binding_name]
            unresolved_bound_names.intersection_update(star_import_excluded_names)
            star_import_ambiguity = True
            return
        direct_bindings = authority._direct_nominal_bindings(current, bindings)
        bound_names = _direct_statement_bound_names(current)
        deleted_names = bound_names if isinstance(current, ast.Delete) else frozenset()
        for bound_name in bound_names:
            bindings.pop(bound_name, None)
            if bound_name in deleted_names:
                unresolved_bound_names.discard(bound_name)
            elif bound_name not in direct_bindings:
                unresolved_bound_names.add(bound_name)
        for bound_name in direct_bindings:
            unresolved_bound_names.discard(bound_name)
        bindings.update(direct_bindings)

    for requested_line in sorted(frozenset(lines)):
        while statement is not None and statement.lineno < requested_line:
            apply(statement)
            statement = next(statements, None)
        snapshots[requested_line] = snapshot()
    if include_final:
        while statement is not None:
            apply(statement)
            statement = next(statements, None)
        snapshots[None] = snapshot()
    return snapshots


@dataclass(frozen=True)
class ModuleBasesStores(ParsedModuleFact):
    """Lines of one module storing a class's `__bases__`, by assignment, `del`
    or `setattr`: any of them can reorder any class's MRO."""

    lines: tuple[int, ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        if "__bases__" not in parsed_module.source:
            return cls(())
        found = []
        for node in ast.walk(parsed_module.module):
            match node:
                case ast.Attribute(attr="__bases__", ctx=ast.Store() | ast.Del()):
                    found.append(node.lineno)
                case ast.Call(func=ast.Name(id="setattr" | "delattr"), args=[_, ast.Constant(value="__bases__"), *_]):
                    found.append(node.lineno)
        return cls(tuple(found))


@dataclass(frozen=True)
class ModuleNominalBindingTimeline(ParsedModuleFact):
    """Exact module binding snapshots indexed by statement position, built once per parse.

    `prefix_max_lines[i]` is the running maximum of `statement.lineno` over `body[:i+1]`,
    so `bisect_left(prefix_max_lines, line)` finds the exact count of leading statements
    whose lines are strictly less than `line` (matching the sequential `while` scan even
    on synthetic ASTs) in O(log N) instead of rescanning the module per query.
    """

    prefix_max_lines: tuple[int, ...]
    snapshots: tuple[ModuleNominalBindingSnapshot, ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        authority = ModuleNominalBindingAuthority(parsed_module)
        bindings: dict[str, CompactNominalBinding] = {}
        unresolved_bound_names: set[str] = set()
        star_import_ambiguity = False
        annotation_mode = ModuleAnnotationEvaluationMode.from_module(parsed_module.module)
        initial = ModuleNominalBindingSnapshot(
            bindings_by_name={},
            annotation_mode=annotation_mode,
            unresolved_bound_names=frozenset(),
            star_import_ambiguity=False,
            star_import_excluded_names=frozenset(),
        )
        prefix_max_lines: list[int] = []
        snapshots: list[ModuleNominalBindingSnapshot] = [initial]
        current_snapshot = initial
        running_max = -1
        for statement in parsed_module.module.body:
            line = statement.lineno
            if line > running_max:
                running_max = line
            prefix_max_lines.append(running_max)
            changed = False
            if isinstance(statement, ast.ImportFrom) and any(
                alias.name == "*" for alias in statement.names
            ):
                if bindings or unresolved_bound_names or not star_import_ambiguity:
                    bindings.clear()
                    unresolved_bound_names.clear()
                    star_import_ambiguity = True
                    changed = True
            else:
                direct_bindings = authority._direct_nominal_bindings(statement, bindings)
                bound_names = _direct_statement_bound_names(statement)
                if bound_names or direct_bindings:
                    deleted_names = (
                        bound_names if isinstance(statement, ast.Delete) else frozenset()
                    )
                    for bound_name in bound_names:
                        if bindings.pop(bound_name, None) is not None:
                            changed = True
                        if bound_name in deleted_names:
                            if bound_name in unresolved_bound_names:
                                unresolved_bound_names.discard(bound_name)
                                changed = True
                        elif bound_name not in direct_bindings:
                            if bound_name not in unresolved_bound_names:
                                unresolved_bound_names.add(bound_name)
                                changed = True
                    for bound_name, binding in direct_bindings.items():
                        if bound_name in unresolved_bound_names:
                            unresolved_bound_names.discard(bound_name)
                            changed = True
                        if bindings.get(bound_name) != binding:
                            bindings[bound_name] = binding
                            changed = True
            if changed:
                current_snapshot = ModuleNominalBindingSnapshot(
                    bindings_by_name=dict(bindings),
                    annotation_mode=annotation_mode,
                    unresolved_bound_names=frozenset(unresolved_bound_names),
                    star_import_ambiguity=star_import_ambiguity,
                    star_import_excluded_names=frozenset(),
                )
            snapshots.append(current_snapshot)
        return cls(tuple(prefix_max_lines), tuple(snapshots))

    @property
    def final_snapshot(self) -> ModuleNominalBindingSnapshot:
        return self.snapshots[-1]

    def snapshot_before_line(self, line: int) -> ModuleNominalBindingSnapshot:
        return self.snapshots[bisect_left(self.prefix_max_lines, line)]


@dataclass(frozen=True)
class ClassLineBindingSnapshots(ParsedModuleFact):
    """The module's bindings where each class statement runs, keyed by its line, and its
    final bindings under None: one pass, read by the class header and by every check of
    what a class body's decorators and annotations resolve to."""

    snapshots: dict[int | None, ModuleNominalBindingSnapshot]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        return cls(
            ModuleNominalBindingAuthority(parsed_module).snapshots_before(
                (
                    *(
                        node.lineno
                        for _qualname, node in iter_class_definitions(
                            list(parsed_module.module.body)
                        )
                    ),
                    None,
                )
            )
        )


@dataclass(frozen=True)
class ModuleImportBindings(ParsedModuleFact):
    """The import aliases one module binds, by local name."""

    bindings: dict[str, CompactNominalBinding]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        final = ModuleNominalBindingAuthority(parsed_module).snapshots_before(
            (None,),
            policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
        )[None]
        return cls(
            {
                local_name: binding
                for local_name, binding in final.bindings_by_name.items()
                if binding.kind.projects_as_import_alias
            }
        )


def module_import_aliases(parsed_module: ParsedModule) -> dict[str, str]:
    return {
        local_name: binding.qualified_name
        for local_name, binding in parsed_module.fact(ModuleImportBindings).bindings.items()
    }


def module_import_names(parsed_module: ParsedModule) -> frozenset[str]:
    """Names the module's final bindings resolve to an imported module."""

    return frozenset(
        local_name
        for local_name, binding in parsed_module.fact(ModuleImportBindings).bindings.items()
        if binding.kind.binds_module
    )


@dataclass(frozen=True)
class _ModuleStarImportOrigins(ParsedModuleFact):
    """Module-scope star-import origins cached once per parsed module."""

    origins: tuple[CompactModuleStarImportOrigin, ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        if "*" not in parsed_module.source:
            return cls(())
        origins: list[CompactModuleStarImportOrigin] = []

        class ModuleScopeStarImportCollector(ast.NodeVisitor):
            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if any(alias.name == "*" for alias in node.names):
                    origins.append(
                        CompactModuleStarImportOrigin.resolved_as(
                            parsed_module.module_path_identity.resolve_import_from_module(
                                imported_module=node.module,
                                level=node.level,
                            )
                        )
                    )

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

        ModuleScopeStarImportCollector().visit(parsed_module.module)
        return cls(tuple(origins))


def module_star_import_origins(
    parsed_module: ParsedModule,
) -> tuple[CompactModuleStarImportOrigin, ...]:
    return parsed_module.fact(_ModuleStarImportOrigins).origins


@dataclass(frozen=True)
class CompactClassProjectionDemand:
    """Target-correlated filters for expensive class-family facets."""

    class_method_names: frozenset[str]
    include_autoregister_references: bool = True
    header_core_only: bool = False


HEADER_CORE_DEMAND = CompactClassProjectionDemand(frozenset(), header_core_only=True)
"""The view `ModuleClassHeaderCore` holds: namespace and class headers only."""


def _native_definition_child(node: object, definition_type: str) -> object | None:
    node_type = getattr(node, "type", None)
    if node_type == definition_type:
        return node
    if node_type != "decorated_definition":
        return None
    return next(
        (
            child
            for child in getattr(node, "named_children", ())
            if child.type == definition_type
        ),
        None,
    )


def _native_sparse_class_header(
    syntax_index: NativePythonSyntaxIndex,
    node: object,
) -> ast.ClassDef:
    class_node = copy.deepcopy(syntax_index.class_header_for(node))
    body_node = node.child_by_field_name("body")
    body: list[ast.stmt] = []
    if body_node is not None:
        for child in body_node.named_children:
            nested = _native_definition_child(child, "class_definition")
            if nested is not None:
                body.append(_native_sparse_class_header(syntax_index, nested))
                continue
            function = _native_definition_child(child, "function_definition")
            if function is not None:
                continue
            if child.type != "expression_statement":
                continue
    if body:
        class_node.body = body
    return class_node


def _native_class_header_module(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> ParsedModule | None:
    if not syntax_index.is_complete:
        return None
    public_export_name = PYTHON_PUBLIC_EXPORT_ASSIGNMENT.encode("utf-8")
    has_public_export_syntax = any(
        syntax_index.source_for(node) == public_export_name
        for node in syntax_index.captures("(identifier) @identifier").get(
            "identifier",
            (),
        )
    )
    has_star_import = any(
        isinstance(statement := syntax_index.statement_for(node), ast.ImportFrom)
        and any(alias.name == "*" for alias in statement.names)
        for node in syntax_index.captures("(import_from_statement) @statement").get(
            "statement", ()
        )
    )
    if has_public_export_syntax or has_star_import:
        return source_module.parse()
    body: list[ast.stmt] = []
    for child in syntax_index.tree.root_node.named_children:
        class_node = _native_definition_child(child, "class_definition")
        if class_node is not None:
            body.append(_native_sparse_class_header(syntax_index, class_node))
        elif child.type in {
            "future_import_statement",
            "import_statement",
            "import_from_statement",
        }:
            body.append(copy.deepcopy(syntax_index.statement_for(child)))
    return source_module.parsed_module(
        ast.Module(body=body, type_ignores=[]),
    )


def _collect_demanded_class_projection_from_source(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    demand: object,
) -> list[object] | None:
    if not isinstance(demand, CompactClassProjectionDemand):
        raise TypeError("class projection demand has the wrong authority type")
    if not demand.header_core_only:
        return None
    parsed_module = _native_class_header_module(source_module, syntax_index)
    if parsed_module is None:
        return None
    return CompactModuleClassProjectionFamily._collect_header_core(parsed_module)


def _compact_base_references(
    node: ast.ClassDef,
    module_binding_snapshot: ModuleNominalBindingSnapshot,
) -> tuple[CompactNominalReference, ...]:
    return tuple(
        module_binding_snapshot.reference_for(parts)
        for _base, parts in _referenced_bases(node)
    )


def _compact_class_binding_facets(
    node: ast.ClassDef,
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    module_name: str,
    qualname: str,
    *,
    include_body_facets: bool,
) -> CompactClassBindingFacets:
    if include_body_facets:
        try:
            enum_declaration: PythonEnumDeclarationFacet = (
                PYTHON_ENUM_BASE_AUTHORITY.declaration_from_node(
                    node,
                    identity=f"{module_name}.{qualname}",
                    qualified_reference=lambda reference, shadowed_names: (
                        _class_scope_qualified_import_name(
                            module_binding_snapshot,
                            {},
                            reference,
                            shadowed_names,
                        )
                    ),
                )
            )
        except ValueError:
            enum_declaration = NoEnumDeclaration()
    else:
        enum_declaration = NoEnumDeclaration()
    base_references = _compact_base_references(node, module_binding_snapshot)
    return CompactClassBindingFacets(
        base_references=base_references,
        base_references_are_complete=len(base_references) == len(node.bases),
        class_decorators_are_promotion_safe=all(
            ClassMethodPromotionSafeDecorator.for_qualified_name(
                _class_scope_qualified_import_name(
                    module_binding_snapshot,
                    {},
                    decorator.func if isinstance(decorator, ast.Call) else decorator,
                    frozenset(),
                )
            )
            is not None
            for decorator in node.decorator_list
        ),
        dataclass_declaration=(
            _dataclass_declaration(module_binding_snapshot, qualname, node)
            if include_body_facets
            else NoDataclassDeclaration()
        ),
        enum_declaration=enum_declaration,
    )


_CLASS_BODY_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@ScanCache.cached
def _cached_class_body_digests() -> dict[
    tuple[str, str, tuple[object, ...]],
    tuple[int, tuple[ClassBodyDeclarationDigest, ...]],
]:
    """Reuse ``ClassBodyNormalForm.of_class`` digests for unchanged classes
    across virtual snapshots during a ``ScanCache.scope()``."""

    return {}


@ScanCache.cached
def _base_modules_for_class_digests() -> dict[str, ParsedModule]:
    """Baseline parsed modules whose already-collected ``body_declarations`` can
    seed ``_cached_class_body_digests`` lazily on the first edit to a file."""

    return {}


@ScanCache.cached
def _seeded_class_digest_paths() -> set[str]:
    return set()


def register_base_modules_for_class_digests(modules: Iterable[ParsedModule]) -> None:
    """Register baseline scan modules so edited copies reuse unchanged class digests."""

    if ScanCache.active() is None:
        return
    registry = _base_modules_for_class_digests()
    if registry:
        return
    for module in modules:
        registry[module.file_path] = module


def _seed_class_body_digests_from_baseline(
    file_path: str,
    cache: dict[
        tuple[str, str, tuple[object, ...]],
        tuple[int, tuple[ClassBodyDeclarationDigest, ...]],
    ],
) -> None:
    seeded = _seeded_class_digest_paths()
    if file_path in seeded:
        return
    seeded.add(file_path)
    base_module = _base_modules_for_class_digests().get(file_path)
    if base_module is None or not base_module.has_family_batch(
        CompactModuleClassProjectionFamily
    ):
        return
    items = base_module.family_batch(CompactModuleClassProjectionFamily).items
    if not items:
        return
    base_proj = items[0]
    if not isinstance(base_proj, CompactModuleClassProjection) or not base_proj.classes:
        return
    source_lines = base_module.source_segments.lines
    lines = tuple(c.line for c in base_proj.classes if c.end_line is not None)
    if not lines:
        return
    scopes = ClassBodyModuleScope.for_lines(base_module, lines)
    for cls_rec in base_proj.classes:
        if cls_rec.end_line is None:
            continue
        scope = scopes.get(cls_rec.line)
        if scope is None:
            continue
        class_src = "".join(source_lines[cls_rec.line - 1 : cls_rec.end_line])
        names = tuple(sorted(set(_CLASS_BODY_IDENT_RE.findall(class_src))))
        scope_key = (
            scope.module_name,
            _binding_snapshot_slice_key(scope.declaration, names),
            _binding_snapshot_slice_key(scope.final, names),
            None
            if scope.star_blind_declaration is None
            else _binding_snapshot_slice_key(scope.star_blind_declaration, names),
            None
            if scope.star_blind_final is None
            else _binding_snapshot_slice_key(scope.star_blind_final, names),
        )
        key = (scope.module_name, class_src, scope_key)
        cache.setdefault(key, (cls_rec.line, cls_rec.body_declarations))


def _binding_snapshot_slice_key(
    snapshot: ModuleNominalBindingSnapshot,
    names: tuple[str, ...],
) -> tuple[object, ...]:
    bindings = snapshot.bindings_by_name
    unresolved = snapshot.unresolved_bound_names
    return (
        tuple(
            (name, bindings[name].qualified_name)
            for name in names
            if name in bindings
        ),
        tuple(name for name in names if name in unresolved),
        snapshot.annotation_mode,
        snapshot.star_import_ambiguity,
        snapshot.star_import_excluded_names,
    )


def _class_body_digests_for_node(
    source_lines: tuple[str, ...],
    node: ast.ClassDef,
    scope: ClassBodyModuleScope,
    scope_keys: dict[int, tuple[object, ...]],
    cache: (
        dict[
            tuple[str, str, tuple[object, ...]],
            tuple[int, tuple[ClassBodyDeclarationDigest, ...]],
        ]
        | None
    ),
) -> tuple[ClassBodyDeclarationDigest, ...]:
    del scope_keys
    if cache is not None and node.end_lineno is not None:
        class_src = "".join(source_lines[node.lineno - 1 : node.end_lineno])
        names = tuple(sorted(set(_CLASS_BODY_IDENT_RE.findall(class_src))))
        scope_key = (
            scope.module_name,
            _binding_snapshot_slice_key(scope.declaration, names),
            _binding_snapshot_slice_key(scope.final, names),
            None
            if scope.star_blind_declaration is None
            else _binding_snapshot_slice_key(scope.star_blind_declaration, names),
            None
            if scope.star_blind_final is None
            else _binding_snapshot_slice_key(scope.star_blind_final, names),
        )
        key = (scope.module_name, class_src, scope_key)
        cached = cache.get(key)
        if cached is not None:
            cached_lineno, cached_digests = cached
            delta = node.lineno - cached_lineno
            if delta == 0:
                return cached_digests
            return tuple(
                digest._replace(line=digest.line + delta) for digest in cached_digests
            )
        digests = tuple(
            declaration.compact()
            for declaration in ClassBodyNormalForm.of_class(node, scope)
        )
        cache[key] = (node.lineno, digests)
        return digests
    return tuple(
        declaration.compact()
        for declaration in ClassBodyNormalForm.of_class(node, scope)
    )


def _compact_indexed_classes(
    parsed_module: ParsedModule,
    indexed_class_nodes: tuple[tuple[str, ast.ClassDef], ...],
    binding_snapshots: Mapping[int | None, ModuleNominalBindingSnapshot],
    *,
    include_body_facets: bool,
) -> tuple[CompactIndexedClass, ...]:
    file_path = parsed_module.file_path
    source_lines = parsed_module.source_segments.lines
    body_cache = (
        _cached_class_body_digests()
        if include_body_facets and ScanCache.active() is not None
        else None
    )
    if body_cache is not None:
        _seed_class_body_digests_from_baseline(file_path, body_cache)
    body_scope_keys: dict[int, tuple[object, ...]] = {}
    star_snapshots = (
        _module_nominal_binding_snapshots(
            ModuleNominalBindingAuthority(parsed_module),
            tuple(node.lineno for _qualname, node in indexed_class_nodes),
            include_final=False,
            star_import_excluded_names=_class_nominal_reference_root_names(parsed_module),
        )
        if module_star_import_origins(parsed_module)
        else None
    )
    body_scopes = (
        ClassBodyModuleScope.for_lines(
            parsed_module, (node.lineno for _qualname, node in indexed_class_nodes)
        )
        if include_body_facets
        else {}
    )
    return tuple(
        CompactIndexedClass(
            symbol=f"{parsed_module.module_name}.{qualname}",
            module_name=parsed_module.module_name,
            qualname=qualname,
            simple_name=qualname.rsplit(".", 1)[-1],
            file_path=file_path,
            line=node.lineno,
            declared_base_names=tuple(
                declared_name
                for base in node.bases
                if (
                    declared_name := ClassSymbolResolutionAuthority.declared_base_name(
                        base
                    )
                )
                is not None
            ),
            base_references=binding_facets.base_references,
            mro_bases_are_static=all(
                isinstance(base, ast.Name | ast.Attribute) for base in node.bases
            ),
            direct_base_count=sum(
                ClassSymbolResolutionAuthority.establishes_nominal_family(
                    declared_base_name
                )
                for base in node.bases
                if (
                    declared_base_name := ClassSymbolResolutionAuthority.declared_base_name(
                        base
                    )
                )
                is not None
            ),
            base_references_are_complete=binding_facets.base_references_are_complete,
            product_base_bindings_are_exact=(
                binding_facets.product_base_bindings_are_exact
            ),
            direct_member_declarations=_compact_class_member_declarations(node),
            metaclass_names=tuple(
                terminal_name
                for keyword in node.keywords
                if keyword.arg == "metaclass"
                if (terminal_name := _terminal_reference_name(keyword.value))
                is not None
            ),
            class_keyword_names=tuple(
                keyword.spelled_name for keyword in CallKeywords.of(node.keywords).entries
            ),
            class_decorators_are_promotion_safe=(
                binding_facets.class_decorators_are_promotion_safe
            ),
            class_header_is_reconstructible=ClassHeaderSourceSpan(
                node,
                parsed_module.source_segments.lines,
            ).is_reconstructible,
            base_key_type_names=_compact_base_key_type_names(node),
            base_argument_sources=_compact_base_argument_sources(node),
            type_parameter_names=TYPE_PARAMETERS.names(node),
            is_final=binding_snapshots[node.lineno].decorated_by(node, (FINAL_DECORATOR,)),
            end_line=node.end_lineno,
            method_names=tuple(
                statement.name
                for statement in FunctionDefinition.within(node.body)
            ),
            callable_member_declarations=_compact_callable_member_declarations(
                node, binding_snapshots[node.lineno]
            ),
            abstract_method_names=sorted_tuple(
                statement.name
                for statement in FunctionDefinition.within(node.body)
                if binding_snapshots[node.lineno].decorated_by(
                    statement, (ABSTRACT_METHOD_DECORATOR,)
                )
            ),
            is_abstract=_is_abstract_class(node, binding_snapshots[node.lineno]),
            dataclass_declaration=binding_facets.dataclass_declaration,
            enum_declaration=binding_facets.enum_declaration,
            registry_facts=ClassRegistryFacts.of_class(node, module_binding_snapshot, parsed_module),
            is_registration_authority=_is_registration_authority(node),
            autoregister_registry_key_attr_name=_autoregister_registry_key_attr_name(
                parsed_module,
                node,
            ),
            autoregister_registry_projection_names=registry_methods[0],
            keyed_registry_lookup_method_names=registry_methods[1],
            keyed_registry_reverse_lookup_method_names=registry_methods[2],
            predicate_selected_methods=(
                _compact_predicate_selected_methods(node) if include_body_facets else ()
            ),
            body_declarations=(
                _class_body_digests_for_node(
                    source_lines,
                    node,
                    body_scopes[node.lineno],
                    body_scope_keys,
                    body_cache,
                )
                if include_body_facets
                else ()
            ),
            star_resolution=(
                None
                if star_snapshots is None
                else _star_resolved_facets(
                    node,
                    star_snapshots[node.lineno],
                    parsed_module.module_name,
                    qualname,
                    include_body_facets=include_body_facets,
                )
            ),
        )
        for qualname, node in indexed_class_nodes
        for module_binding_snapshot in (binding_snapshots[node.lineno],)
        for binding_facets in (
            _compact_class_binding_facets(
                node,
                module_binding_snapshot,
                parsed_module.module_name,
                qualname,
                include_body_facets=include_body_facets,
            ),
        )
        for registry_methods in (
            _registry_method_names(node) if include_body_facets else ((), (), ()),
        )
    )


def _star_resolved_facets(
    node: ast.ClassDef,
    star_snapshot: ModuleNominalBindingSnapshot,
    module_name: str,
    qualname: str,
    *,
    include_body_facets: bool,
) -> CompactStarResolvedFacets:
    return CompactStarResolvedFacets(
        root_names=_class_reference_root_names(node),
        facets=_compact_class_binding_facets(
            node,
            star_snapshot,
            module_name,
            qualname,
            include_body_facets=include_body_facets,
        ),
    )


def star_resolved_projections(
    projections: tuple[ClassProjectionT, ...],
) -> tuple[ClassProjectionT, ...]:
    """Settle every class's star-import-dependent facts from the repository's
    export contracts: the one place those facts are resolved."""

    if not any(projection.star_import_origins for projection in projections):
        return projections
    exposure = CompactRepositoryPublicExposureIndex(projections)
    result: list[ClassProjectionT] = []
    changed_any = False
    for projection in projections:
        if not projection.star_import_origins:
            result.append(projection)
            continue
        resolved_classes = tuple(
            indexed_class.resolved_past_star_imports(exposure)
            for indexed_class in projection.classes
        )
        if all(
            resolved is original
            for resolved, original in zip(resolved_classes, projection.classes)
        ):
            result.append(projection)
        else:
            changed_any = True
            result.append(replace(projection, classes=resolved_classes))
    return tuple(result) if changed_any else projections


def repository_settled_projections(
    projections: tuple[ClassProjectionT, ...],
) -> tuple[ClassProjectionT, ...]:
    """Settle every module fact the repository decides: names a star import
    may rebind, then the role of each field whose value calls a function."""

    return field_declaration_settled_projections(star_resolved_projections(projections))


@ScanCache.cached
def _interned_settled_projections() -> dict[
    int,
    tuple[
        CompactModuleClassHeader,
        tuple[CompactDataclassFieldRole, ...],
        CompactModuleClassHeader,
    ],
]:
    """Preserve settled projection object identity across snapshots when a
    module's field-declaration call roles did not change."""

    return {}


@ScanCache.cached
def _base_field_decl_settlement_slot() -> list[
    tuple[
        dict[str, CompactModuleClassHeader],
        dict[int, CompactModuleClassHeader],
        FieldDeclarationIndex,
        frozenset[str],
        frozenset[str],
    ]
    | None
]:
    return [None]


def _field_role_class_signature(
    indexed_class: CompactIndexedClass,
) -> tuple[object, ...]:
    return (
        indexed_class.symbol,
        "__new__" in indexed_class.member_binding_names,
        "__call__" in indexed_class.member_binding_names,
        indexed_class.base_references_are_complete,
        indexed_class.registry_facts.base_references,
        indexed_class.registry_facts.metaclass_references,
    )


def _try_incremental_field_declaration_settlement(
    projections: tuple[ClassProjectionT, ...],
) -> tuple[ClassProjectionT, ...] | None:
    slot = _base_field_decl_settlement_slot()
    base_entry = slot[0]
    if base_entry is None:
        return None
    base_by_path, settled_by_raw_id, base_index, touched_modules, touched_symbols = (
        base_entry
    )
    if len(projections) != len(base_by_path):
        return None
    changed: list[tuple[ClassProjectionT, CompactModuleClassHeader]] = []
    for projection in projections:
        old_projection = base_by_path.get(projection.file_path)
        if old_projection is None or old_projection.module_name != projection.module_name:
            return None
        if old_projection is not projection:
            changed.append((projection, old_projection))
    if not changed:
        result_unchanged: list[ClassProjectionT] = []
        changed_any = False
        for projection in projections:
            settled = cast(ClassProjectionT, settled_by_raw_id[id(projection)])
            if settled is not projection:
                changed_any = True
            result_unchanged.append(settled)
        return tuple(result_unchanged) if changed_any else projections
    if len(changed) > 64:
        return None
    changed_module_names: set[str] = set()
    for projection, old_projection in changed:
        changed_module_names.add(projection.module_name)
        if projection.field_declaration_wrappers != old_projection.field_declaration_wrappers:
            return None
        mod_touched = projection.module_name in touched_modules
        if mod_touched and projection.import_aliases != old_projection.import_aliases:
            return None
        old_sigs = {c.symbol: _field_role_class_signature(c) for c in old_projection.classes}
        new_sigs = {c.symbol: _field_role_class_signature(c) for c in projection.classes}
        for sym in old_sigs.keys() | new_sigs.keys():
            if old_sigs.get(sym) != new_sigs.get(sym) and sym in touched_symbols:
                return None
    for projection, _old_projection in changed:
        if not projection.has_unsettled_field_declarations:
            continue
        for indexed_class in projection.classes:
            for member in indexed_class.dataclass_declaration.fields:
                call = member.declaration_call
                if call is not None:
                    callee_mod = call.callee.rpartition(".")[0]
                    if callee_mod in changed_module_names:
                        return None
    result: list[ClassProjectionT] = []
    changed_any = False
    for projection in projections:
        cached_settled = settled_by_raw_id.get(id(projection))
        if cached_settled is not None:
            settled_proj = cast(ClassProjectionT, cached_settled)
            if settled_proj is not projection:
                changed_any = True
            result.append(settled_proj)
            continue
        if not projection.has_unsettled_field_declarations:
            result.append(projection)
            continue
        settled_classes = tuple(
            indexed_class.with_settled_field_declarations(base_index)
            for indexed_class in projection.classes
        )
        if all(
            settled is original
            for settled, original in zip(settled_classes, projection.classes)
        ):
            result.append(projection)
        else:
            changed_any = True
            result.append(replace(projection, classes=settled_classes))
    return tuple(result) if changed_any else projections


def field_declaration_settled_projections(
    projections: tuple[ClassProjectionT, ...],
) -> tuple[ClassProjectionT, ...]:
    """Settle every field whose value calls a function, from the repository's
    field declarations: the one place those roles are decided."""

    if not any(projection.has_unsettled_field_declarations for projection in projections):
        return projections
    if ScanCache.active() is not None:
        incremental = _try_incremental_field_declaration_settlement(projections)
        if incremental is not None:
            return incremental
    index = FieldDeclarationIndex.of_projections(projections)
    cache = (
        _interned_settled_projections() if ScanCache.active() is not None else None
    )
    result: list[ClassProjectionT] = []
    changed_any = False
    for projection in projections:
        if not projection.has_unsettled_field_declarations:
            result.append(projection)
            continue
        if cache is not None:
            roles = tuple(
                index.field_role(member.declaration_call)
                for indexed_class in projection.classes
                for member in indexed_class.dataclass_declaration.fields
                if member.declaration_call is not None
            )
            cached = cache.get(id(projection))
            if cached is not None and cached[0] is projection and cached[1] == roles:
                settled_proj = cast(ClassProjectionT, cached[2])
                if settled_proj is not projection:
                    changed_any = True
                result.append(settled_proj)
                continue
        settled_classes = tuple(
            indexed_class.with_settled_field_declarations(index)
            for indexed_class in projection.classes
        )
        if all(
            settled is original
            for settled, original in zip(settled_classes, projection.classes)
        ):
            settled_proj = projection
        else:
            changed_any = True
            settled_proj = replace(projection, classes=settled_classes)
        if cache is not None:
            cache[id(projection)] = (projection, roles, settled_proj)
        result.append(settled_proj)
    settled_tuple = tuple(result) if changed_any else projections
    if ScanCache.active() is not None:
        slot = _base_field_decl_settlement_slot()
        if slot[0] is None:
            queried_callees = {
                member.declaration_call.callee
                for projection in projections
                if projection.has_unsettled_field_declarations
                for indexed_class in projection.classes
                for member in indexed_class.dataclass_declaration.fields
                if member.declaration_call is not None
            }
            touched_symbols: set[str] = (
                queried_callees
                | set(index.built)
                | set(index.kept_metaclasses)
            )
            touched_modules: set[str] = {
                sym.rpartition(".")[0] for sym in touched_symbols if "." in sym
            }
            if "classes" in index.__dict__:
                lookups = index.classes.exposure.__dict__.get(
                    "_lazy_module_declaration_lookups", ()
                )
                touched_modules.update(lookups)
                scanned = index.classes.scanned
                for sym in tuple(index.built) + tuple(index.kept_metaclasses):
                    decl = scanned.get(sym)
                    if decl is not None:
                        touched_modules.add(decl.module_name)
                        for ref in (
                            *decl.registry_facts.base_references,
                            *decl.registry_facts.metaclass_references,
                        ):
                            qname = ref.settled_qualified_name(
                                index.classes, decl.module_name
                            )
                            if qname and "." in qname:
                                touched_symbols.add(qname)
                                touched_modules.add(qname.rpartition(".")[0])
            slot[0] = (
                {p.file_path: p for p in projections},
                {id(raw): settled for raw, settled in zip(projections, settled_tuple)},
                index,
                frozenset(touched_modules),
                frozenset(touched_symbols),
            )
    return settled_tuple


class CompactModuleClassProjectionFamily(CollectedFamily[CompactModuleClassProjection]):
    """Persist class/import facts needed by the global inheritance graph."""

    item_type = CompactModuleClassProjection
    cache_payload_max_bytes = 3_000_000
    source_demand_collector = staticmethod(
        _collect_demanded_class_projection_from_source
    )

    @classmethod
    def _required_module_projection(
        cls, parsed_module: ParsedModule,
    ) -> CompactModuleClassProjection:
        projections = collect_family_items(parsed_module, cls)
        if len(projections) != 1:
            raise ValueError(
                "Compact class projection family must emit one item per module"
            )
        return projections[0]

    @classmethod
    def report_demand(
        cls,
        target_items: tuple[object, ...],
        config: object,
    ) -> CompactClassProjectionDemand:
        """Derive the exact class facets required by report-target projections."""

        del config
        projections = tuple(
            item for item in target_items if isinstance(item, cls.item_type)
        )
        return CompactClassProjectionDemand(
            class_method_names=frozenset(
                method.method_name
                for projection in projections
                for method in projection.class_methods
            ),
            include_autoregister_references=any(
                projection.named_projection_surfaces
                or any(
                    bool(indexed_class.registry_facts.metaclass_references)
                    or (
                        indexed_class.declares_keyed_base
                        and "registry_key_attr"
                        in indexed_class.direct_declared_member_names
                    )
                    for indexed_class in projection.classes
                )
                for projection in projections
            ),
        )

    @classmethod
    def project_cached_demand(
        cls,
        items: tuple[object, ...],
        demand: object,
    ) -> tuple[CompactModuleClassProjection, ...]:
        """Project cached class facts through their exact report demand."""

        if not isinstance(demand, CompactClassProjectionDemand):
            raise TypeError("class projection demand has the wrong authority type")
        projected = tuple(
            replace(
                item,
                class_methods=tuple(
                    method
                    for method in item.class_methods
                    if method.method_name in demand.class_method_names
                ),
                autoregister_reference_index=(
                    item.autoregister_reference_index
                    if demand.include_autoregister_references
                    else None
                ),
            )
            for item in items
            if isinstance(item, cls.item_type)
        )
        if not demand.header_core_only:
            return projected
        return tuple(item.header_core() for item in projected)

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactModuleClassProjection]:
        return cls._collect(parsed_module, None)

    @classmethod
    @ScanCache.scope()
    def collect_modules(
        cls,
        parsed_modules: Iterable[ParsedModule],
    ) -> tuple[CompactModuleClassProjection, ...]:
        """Collect the single module projection declared by this family."""

        return repository_settled_projections(
            tuple(cls._required_module_projection(module) for module in parsed_modules)
        )

    @classmethod
    @ScanCache.scope()
    def collect_overlap_modules(
        cls,
        parsed_modules: Iterable[ParsedModule],
    ) -> tuple[CompactModuleClassProjection, ...]:
        """Collect overlap-sufficient projections: reuse already-cached full module
        projections and derive only overlap facets for uncached virtual modules."""

        return repository_settled_projections(
            tuple(
                cls._required_module_projection(module)
                if module.has_family_batch(cls)
                else module.fact(ModuleClassOverlapProjection).projection
                for module in parsed_modules
            )
        )

    @classmethod
    def collect_demanded(
        cls,
        parsed_module: ParsedModule,
        demand: object,
    ) -> list[CompactModuleClassProjection] | None:
        if not isinstance(demand, CompactClassProjectionDemand):
            raise TypeError("class projection demand has the wrong authority type")
        if demand.header_core_only:
            return cls._collect_header_core(parsed_module)
        return cls._collect(parsed_module, demand)

    @classmethod
    def header_core_projections(
        cls, parsed_modules: Iterable[ParsedModule]
    ) -> tuple[CompactModuleClassProjection, ...]:
        """Namespace and class headers only: enough to settle imports and exports."""

        return tuple(
            parsed_module.fact(ModuleClassHeaderCore).projection
            for parsed_module in parsed_modules
        )

    @classmethod
    def _collect_header_core(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactModuleClassProjection]:
        del cls
        return [
            CompactModuleClassProjection.declared_in(
                parsed_module, include_body_facets=False
            ).header_core()
        ]

    @classmethod
    def collect_stub(
        cls,
        parsed_module: ParsedModule,
    ) -> CompactModuleClassProjection:
        """Project only the declarations stdlib stub typing reads."""
        del cls
        declarations = CompactModuleClassHeader.from_module(parsed_module)
        return CompactModuleClassProjection(
            module_name=declarations.module_name,
            file_path=declarations.file_path,
            import_aliases=declarations.import_aliases,
            public_export_contract=declarations.public_export_contract,
            star_import_origins=declarations.star_import_origins,
            classes=declarations.classes,
            field_declaration_wrappers=declarations.field_declaration_wrappers,
            top_level_definitions=tuple(
                (node.name, node.lineno)
                for node in parsed_module.module.body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            ),
            type_variable_names=_module_type_variable_names(parsed_module),
            annotated_function_returns=tuple(
                CompactAnnotatedFunctionReturn(
                    node.name, node.lineno, ast.unparse(returns)
                )
                for node in parsed_module.module.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                for returns in declared_annotations(node)
            ),
        )

    @classmethod
    def _collect(
        cls,
        parsed_module: ParsedModule,
        demand: CompactClassProjectionDemand | None,
    ) -> list[CompactModuleClassProjection]:
        del cls
        declarations = CompactModuleClassHeader.from_module(parsed_module)
        syntax_facets = parsed_module.fact(
            ModuleClassSyntaxFacets
            if demand is None or demand.include_autoregister_references
            else ModuleClassSyntaxFacetsWithoutAutoregister
        ).facets
        indexed_class_nodes = iter_class_definitions(list(parsed_module.module.body))
        class_methods = _compact_class_methods(
            parsed_module,
            indexed_class_nodes,
            parsed_module.fact(ClassLineBindingSnapshots).snapshots,
            method_names=(None if demand is None else demand.class_method_names),
        )
        return [
            CompactModuleClassProjection(
                **{
                    declaration.name: getattr(declarations, declaration.name)
                    for declaration in fields(CompactModuleClassHeader)
                },
                sorted_key_calls=_compact_sorted_key_calls(parsed_module),
                keyed_table_axes=_compact_keyed_table_axes(parsed_module),
                closed_axis_branch_functions=syntax_facets.closed_axis_branch_functions,
                manual_selector_axes=_compact_manual_selector_axes(parsed_module),
                top_level_definitions=tuple(
                    (node.name, node.lineno)
                    for node in parsed_module.module.body
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                ),
                type_variable_names=_module_type_variable_names(parsed_module),
                annotated_function_returns=tuple(
                    CompactAnnotatedFunctionReturn(
                        node.name, node.lineno, ast.unparse(returns)
                    )
                    for node in parsed_module.module.body
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    for returns in declared_annotations(node)
                ),
                exact_type_guards=syntax_facets.exact_type_guards,
                autoregister_reference_index=(
                    syntax_facets.autoregister_reference_index
                ),
                repeated_keyed_family_roots=_compact_repeated_keyed_family_roots(
                    parsed_module, declarations
                ),
                manual_subclass_roster_roots=_compact_manual_subclass_roster_roots(
                    parsed_module
                ),
                latent_rosters=LatentRosterObservation.from_module(parsed_module),
                named_projection_surfaces=_compact_named_projection_surfaces(
                    parsed_module
                ),
                class_methods=class_methods,
                class_references=_compact_class_references(parsed_module),
                field_order_summaries=FunctionOrderSummary.of_syntax_index(
                    parsed_module.module_name,
                    parsed_module.syntax_index,
                    module_import_aliases(parsed_module),
                ),
                script_scopes=tuple(
                    _compact_script_scope(source)
                    for source in parsed_module.fact(EmbeddedSources).sources
                ),
            )
        ]


@dataclass(frozen=True)
class ModuleClassOverlapProjection(ParsedModuleFact):
    """Lightweight class projection holding only the facets `ClassOverlapVerdicts`
    and `CompactClassFamilyIndex` read (header + class references + field order
    summaries + script scopes), avoiding unrelated syntax/method facets on
    temporary virtual snapshots."""

    projection: CompactModuleClassProjection

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        declarations = CompactModuleClassHeader.from_module(parsed_module)
        return cls(
            CompactModuleClassProjection(
                **{
                    declaration.name: getattr(declarations, declaration.name)
                    for declaration in fields(CompactModuleClassHeader)
                },
                class_references=_compact_class_references(parsed_module),
                field_order_summaries=FunctionOrderSummary.of_syntax_index(
                    parsed_module.module_name,
                    parsed_module.syntax_index,
                    module_import_aliases(parsed_module),
                ),
                script_scopes=tuple(
                    _compact_script_scope(source)
                    for source in parsed_module.fact(EmbeddedSources).sources
                ),
            )
        )


@dataclass(frozen=True)
class ModuleClassTypingProjection(ParsedModuleFact):
    """Lightweight class projection holding only the header + return-typing
    facets read by `DeclaredReceiverTyping` and `CompactClassFamilyIndex`."""

    projection: CompactModuleClassProjection

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        return cls(CompactModuleClassProjectionFamily.collect_stub(parsed_module))


_REGISTRY_PROJECTION_ATTRS = frozenset(("__registry__", "_registry", "registry"))


def _registry_method_names(
    node: ast.ClassDef,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Collect autoregister projection, keyed lookup, and reverse lookup method names
    in one iterative walk per method instead of three separate `ast.walk` passes."""

    projection_names: list[str] = []
    lookup_names: list[str] = []
    reverse_lookup_names: list[str] = []
    for raw_method in FunctionDefinition.within(node.body):
        method = cast(ast.FunctionDef | ast.AsyncFunctionDef, raw_method)
        is_classmethod = _is_classmethod(method)
        has_projection_attr = False
        has_cls_registry = False
        stack: list[ast.AST] = [method]
        while stack:
            subnode = stack.pop()
            if isinstance(subnode, ast.Attribute) and subnode.attr in _REGISTRY_PROJECTION_ATTRS:
                has_projection_attr = True
                if is_classmethod and RegistryLookupShape.is_cls_registry_attribute(subnode):
                    has_cls_registry = True
                    break
                if not is_classmethod:
                    break
            for field_name in subnode._fields:
                child = getattr(subnode, field_name, None)
                if isinstance(child, ast.AST):
                    stack.append(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, ast.AST):
                            stack.append(item)
        if has_projection_attr:
            projection_names.append(method.name)
        if has_cls_registry:
            lookup_names.append(method.name)
            if any(token in method.name for token in ("class", "type", "reverse")):
                reverse_lookup_names.append(method.name)
    return (
        tuple(projection_names),
        tuple(lookup_names),
        tuple(reverse_lookup_names),
    )


def _autoregister_registry_projection_names(
    node: ast.ClassDef,
) -> tuple[str, ...]:
    return _registry_method_names(node)[0]


def _method_references_cls_registry(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return RegistryLookupShape.references_registry(method)


def _keyed_registry_lookup_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return _registry_method_names(node)[1]


def _keyed_registry_reverse_lookup_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return _registry_method_names(node)[2]


def _trim_method_docstring(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    body = list(method.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _compact_repeated_keyed_family_roots(
    parsed_module: ParsedModule,
    declarations: CompactModuleClassHeader,
) -> tuple[CompactRepeatedKeyedFamilyRoot, ...]:
    roots: list[CompactRepeatedKeyedFamilyRoot] = []
    classes = {record.qualname: record for record in declarations.classes}
    for node in parsed_module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        assignments = _direct_class_assignments(node)
        registry_key_attr_name = _string_constant(assignments.get("registry_key_attr"))
        registry = assignments.get("_registry")
        registry_is_empty = (
            isinstance(registry, ast.Dict) and not registry.keys and not registry.values
        ) or (
            isinstance(registry, ast.Call)
            and isinstance(registry.func, ast.Name)
            and registry.func.id == "dict"
        )
        if registry_key_attr_name is None or not registry_is_empty:
            continue
        lookup_methods = tuple(
            (method, shape)
            for method in FunctionDefinition.within(node.body)
            if _is_classmethod(method)
            and method.name.startswith("for_")
            and (shape := RegistryLookupShape.from_method(method)) is not None
        )
        if len(lookup_methods) != 1:
            continue
        lookup_method, lookup_shape = lookup_methods[0]
        roots.append(
            CompactRepeatedKeyedFamilyRoot(
                file_path=parsed_module.file_path,
                line=node.lineno,
                class_name=node.name,
                family_bases=tuple(
                    DeclaredFamilyBase.of(reference)
                    for reference in classes[node.name].base_references
                ),
                registry_key_attr_name=registry_key_attr_name,
                lookup_method_name=lookup_method.name,
                lookup_style=lookup_shape.style,
                error_type_name=lookup_shape.error_type_name,
                abstract_hook_names=tuple(
                    method.name
                    for method in FunctionDefinition.within(node.body)
                    if any(
                        _terminal_reference_name(decorator) == "abstractmethod"
                        for decorator in method.decorator_list
                    )
                ),
            )
        )
    return tuple(roots)


def _copied_keyword_fields(call: ast.Call) -> tuple[str, ...]:
    """The keywords one call passes as `x.<same name>`, for the value `x` that
    supplies the most of them."""

    by_root: dict[str, list[str]] = defaultdict(list)
    for keyword in call.keywords:
        value = keyword.value
        if (
            keyword.arg is not None
            and isinstance(value, ast.Attribute)
            and value.attr == keyword.arg
            and isinstance(value.value, ast.Name)
        ):
            by_root[value.value.id].append(keyword.arg)
    return max(
        (tuple(sorted(names)) for names in by_root.values()),
        key=lambda names: (len(names), names),
        default=(),
    )


def _compact_class_references(
    parsed_module: ParsedModule,
) -> tuple[CompactClassReference, ...]:
    """Every read of a module-level name chain, by consumer and role.

    Only chains rooted at a name the module itself binds are kept, and a root a
    function rebinds locally is that function's own. The join resolves the
    chains against the class index; most name no class and are dropped there.
    """

    syntax_index = parsed_module.syntax_index
    module_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
        parsed_module.module.body
    )
    import_aliases = module_import_aliases(parsed_module)
    nodes = syntax_index.depth_first_nodes
    parent_indices = syntax_index.parent_indices
    scopes = syntax_index.scopes
    scope_ids = syntax_index.scope_ids

    function_indices = {
        index
        for scope in scopes
        for index in scope.function_node_indices
    }
    locals_by_function_index: dict[int, frozenset[str]] = {}
    classmethod_param_by_func_index: dict[int, str] = {}
    can_have_classmethod = "classmethod" not in module_bound_names
    for function_index in function_indices:
        function = nodes[function_index]
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            locals_by_function_index[function_index] = (
                LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(function.body)
                | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(function)
            )
            if (
                can_have_classmethod
                and isinstance(nodes[parent_indices[function_index]], ast.ClassDef)
                and any(
                    type(decorator) is ast.Name and decorator.id == "classmethod"
                    for decorator in function.decorator_list
                )
            ):
                positional = (*function.args.posonlyargs, *function.args.args)
                if positional:
                    classmethod_param_by_func_index[function_index] = positional[0].arg

    classmethod_param_names = frozenset(classmethod_param_by_func_index.values())
    empty_frozenset: frozenset[str] = frozenset()
    scope_locals = [
        empty_frozenset.union(
            *(locals_by_function_index.get(fi, empty_frozenset) for fi in sc.function_node_indices)
        )
        if sc.function_node_indices
        else empty_frozenset
        for sc in scopes
    ]

    references: dict[
        tuple[str, tuple[str, ...], ClassReferenceRole, tuple[str, ...]],
        tuple[int, int, bool, tuple[str, ...]],
    ] = {}

    def module_level_chain(
        expression: ast.expr | None, shadowed: frozenset[str]
    ) -> tuple[str, ...]:
        """The name chain `expression` spells, when its root is bound at module
        level and no enclosing function rebinds it."""

        parts = None if expression is None else ExpressionReference.decoded(expression).attribute_chain
        if parts is None or parts[0] not in module_bound_names or parts[0] in shadowed:
            return ()
        return parts

    def record_reference(
        node_index: int,
        node: ast.Name | ast.Attribute,
        parent_index: int,
        parent: ast.AST | None,
        scope: LexicalSyntaxScope,
        shadowed: frozenset[str],
        parts: tuple[str, ...],
    ) -> None:
        grandparent_index = parent_indices[parent_index] if parent is not None else -1
        grandparent = nodes[grandparent_index] if grandparent_index >= 0 else None
        role = ClassReferenceRole.of_reference(
            node, parent, grandparent, import_aliases, reads_as_type(node_index, nodes, parent_indices)
        )
        tested = module_level_chain(subclass_test_subject(node, parent, grandparent), shadowed)
        key = (".".join(scope.names), parts, role, tested)
        positional, unpacked = (
            (
                len(parent.args),
                any(isinstance(argument, ast.Starred) for argument in parent.args)
                or bool(CallKeywords.of(parent.keywords).splats),
            )
            if role is ClassReferenceRole.CONSTRUCTION and isinstance(parent, ast.Call)
            else (0, False)
        )
        copied = (
            _copied_keyword_fields(parent)
            if role is ClassReferenceRole.CONSTRUCTION and isinstance(parent, ast.Call)
            else ()
        )
        line, seen_positional, seen_unpacked, seen_copied = references.get(
            key, (node.lineno, 0, False, ())
        )
        references[key] = (
            min(line, node.lineno),
            max(seen_positional, positional),
            seen_unpacked or unpacked,
            max(seen_copied, copied, key=lambda names: (len(names), names)),
        )

    for node_index in syntax_index.node_indices_by_type.get(ast.Name, ()):
        node = nodes[node_index]
        if type(node.ctx) is not ast.Load:  # type: ignore[union-attr]
            continue
        name_id = node.id  # type: ignore[union-attr]
        in_mod = name_id in module_bound_names
        if not in_mod and name_id not in classmethod_param_names:
            continue
        parent_index = parent_indices[node_index]
        parent = nodes[parent_index] if parent_index >= 0 else None
        if type(parent) is ast.Attribute and parent.value is node:
            continue
        scope_id = scope_ids[node_index]
        scope = scopes[scope_id]
        shadowed = scope_locals[scope_id]
        if (
            classmethod_param_by_func_index
            and scope.function_node_indices
            and classmethod_param_by_func_index.get(scope.function_node_indices[-1]) == name_id
            and scope.class_names
            and scope.class_names[0] in module_bound_names
        ):
            parts: tuple[str, ...] = scope.class_names
        else:
            if not in_mod or name_id in shadowed:
                continue
            parts = (name_id,)
        record_reference(
            node_index, node, parent_index, parent, scope, shadowed, parts  # type: ignore[arg-type]
        )

    for node_index in syntax_index.node_indices_by_type.get(ast.Attribute, ()):
        node = nodes[node_index]
        if type(node.ctx) is not ast.Load:  # type: ignore[union-attr]
            continue
        root = node.value  # type: ignore[union-attr]
        root_type = type(root)
        if root_type is ast.Name:
            root_id = root.id
            if root_id not in module_bound_names:
                continue
            parent_index = parent_indices[node_index]
            parent = nodes[parent_index] if parent_index >= 0 else None
            if type(parent) is ast.Attribute and parent.value is node:
                continue
            scope_id = scope_ids[node_index]
            shadowed = scope_locals[scope_id]
            if root_id in shadowed:
                continue
            parts = (root_id, node.attr)  # type: ignore[union-attr]
        elif root_type is ast.Attribute or root_type is ast.Subscript:
            parent_index = parent_indices[node_index]
            parent = nodes[parent_index] if parent_index >= 0 else None
            if type(parent) is ast.Attribute and parent.value is node:
                continue
            rev_attrs: list[str] = [node.attr]  # type: ignore[union-attr]
            while True:
                if root_type is ast.Attribute:
                    rev_attrs.append(root.attr)
                    root = root.value
                    root_type = type(root)
                elif root_type is ast.Subscript:
                    root = root.value
                    root_type = type(root)
                else:
                    break
            if root_type is not ast.Name or root.id not in module_bound_names:
                continue
            scope_id = scope_ids[node_index]
            shadowed = scope_locals[scope_id]
            if root.id in shadowed:
                continue
            rev_attrs.append(root.id)
            rev_attrs.reverse()
            parts = tuple(rev_attrs)
        else:
            continue
        record_reference(
            node_index, node, parent_index, parent, scopes[scope_id], shadowed, parts  # type: ignore[arg-type]
        )

    return tuple(
        CompactClassReference(consumer, parts, role, *facts, tested_subclass=tested)
        for (consumer, parts, role, tested), facts in sorted(
            references.items(), key=lambda item: (item[1][0], item[0][0], item[0][1])
        )
    )


def _compact_script_scope(source: EmbeddedSource) -> CompactScriptScope:
    return CompactScriptScope(
        source.parsed_module.module_name,
        tuple(module_import_aliases(source.parsed_module).items()),
        tuple(
            reference._replace(line=source.holder_line(reference.line))
            for reference in _compact_class_references(source.parsed_module)
        ),
    )


def _is_abstract_class(
    node: ast.ClassDef, module_binding_snapshot: ModuleNominalBindingSnapshot
) -> bool:
    if {"ABC", "ABCMeta"} & {
        terminal_name
        for base in node.bases
        if (terminal_name := _terminal_reference_name(base)) is not None
    }:
        return True
    return any(
        module_binding_snapshot.decorated_by(statement, (ABSTRACT_METHOD_DECORATOR,))
        for statement in FunctionDefinition.within(node.body)
    )


def _class_scope_qualified_import_name(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    class_aliases: dict[str, str],
    reference: ast.AST,
    preceding_class_bound_names: frozenset[str],
) -> str | None:
    parts = ExpressionReference.decoded(
        ClassSymbolResolutionAuthority.reference_node(reference)
    ).attribute_chain
    if parts is None:
        return None
    if (class_alias := class_aliases.get(parts[0])) is not None:
        return ".".join((class_alias, *parts[1:]))
    if parts[0] in preceding_class_bound_names:
        return None
    root_binding = module_binding_snapshot.binding_for(parts[0])
    if root_binding is None:
        return None
    return ".".join((root_binding.qualified_name, *parts[1:]))


def _dataclass_field_role(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    class_aliases: dict[str, str],
    statement: ast.AnnAssign,
    preceding_class_bound_names: frozenset[str],
) -> CompactDataclassFieldRole:
    annotation_root = _dataclass_role_reference(statement.annotation)
    qualified_annotation = (
        None
        if annotation_root is None
        else _class_scope_qualified_import_name(
            module_binding_snapshot,
            class_aliases,
            annotation_root,
            preceding_class_bound_names,
        )
    )
    if (
        semantic_role := CompactDataclassFieldRole.for_qualified_annotation(
            qualified_annotation
        )
    ) is not None:
        return semantic_role
    annotation_is_resolved_plain_field = (
        (
            annotation_root is None
            and _dataclass_plain_annotation_is_resolved(
                module_binding_snapshot,
                class_aliases,
                statement.annotation,
                preceding_class_bound_names,
            )
        )
        or qualified_annotation is not None
        or (
            isinstance(annotation_root, ast.Name)
            and module_binding_snapshot.resolves_unshadowed_builtin(
                annotation_root.id,
                preceding_class_bound_names=preceding_class_bound_names,
            )
        )
        or (
            annotation_root is not None
            and (root_name := nominal_reference_root_name(annotation_root)) is not None
            and root_name not in class_aliases
            and module_binding_snapshot.reads_as_absent(
                statement.annotation,
                root_name,
                preceding_class_bound_names=preceding_class_bound_names,
            )
        )
    )
    if not annotation_is_resolved_plain_field:
        return CompactDataclassFieldRole.UNRESOLVED
    return StatementValue.of(statement).answer(
        DataclassFieldDefaultRole(
            module_binding_snapshot, class_aliases, preceding_class_bound_names
        )
    )


@dataclass(frozen=True)
class DataclassFieldDefaultRole(StatementValueQuestion[CompactDataclassFieldRole]):
    """The role a resolved plain field's default gives it; no default stores an init field."""

    module_binding_snapshot: ModuleNominalBindingSnapshot
    class_aliases: dict[str, str]
    preceding_class_bound_names: frozenset[str]

    def bare(self, value: NoStatementValue) -> CompactDataclassFieldRole:
        return CompactDataclassFieldRole.STORED_INIT

    def supplied(self, supplied: SuppliedValue) -> CompactDataclassFieldRole:
        value = supplied.value
        if _is_literal_default(value):
            return CompactDataclassFieldRole.STORED_INIT
        if isinstance(value, ast.Name | ast.Attribute):
            # A bare reference: the repository judges whether it names a class.
            return (
                CompactDataclassFieldRole.UNRESOLVED
                if _class_scope_qualified_import_name(
                    self.module_binding_snapshot,
                    self.class_aliases,
                    value,
                    self.preceding_class_bound_names,
                )
                is None
                else CompactDataclassFieldRole.DECLARED_BY_CALL
            )
        if not isinstance(value, ast.Call):
            return CompactDataclassFieldRole.UNRESOLVED
        if (
            isinstance(value.func, ast.Name)
            and self.module_binding_snapshot.resolves_unshadowed_builtin(
                value.func.id, preceding_class_bound_names=self.preceding_class_bound_names
            )
            and isinstance(vars(builtins)[value.func.id], type)
        ):
            # Calling a builtin type returns an instance of it or, for `type`,
            # a class: never a `dataclasses.Field`, so a plain default.
            return CompactDataclassFieldRole.STORED_INIT
        qualified_default_factory = _class_scope_qualified_import_name(
            self.module_binding_snapshot,
            self.class_aliases,
            value.func,
            self.preceding_class_bound_names,
        )
        default_factory_declaration = DataclassRuntimeDeclaration.for_qualified_name(
            qualified_default_factory
        )
        if qualified_default_factory is not None and default_factory_declaration is None:
            return CompactDataclassFieldRole.DECLARED_BY_CALL
        if (
            default_factory_declaration is None
            or not default_factory_declaration.is_field_factory
        ):
            return CompactDataclassFieldRole.UNRESOLVED
        if value.args or CallKeywords.of(value.keywords).splats:
            return CompactDataclassFieldRole.UNRESOLVED
        init_values = tuple(
            keyword.value for keyword in value.keywords if keyword.arg == "init"
        )
        if len(init_values) > 1:
            return CompactDataclassFieldRole.UNRESOLVED
        if not init_values:
            return CompactDataclassFieldRole.STORED_INIT
        init_value = init_values[0]
        if not isinstance(init_value, ast.Constant) or not isinstance(
            init_value.value,
            bool,
        ):
            return CompactDataclassFieldRole.UNRESOLVED
        return (
            CompactDataclassFieldRole.STORED_INIT
            if init_value.value
            else CompactDataclassFieldRole.STORED_NON_INIT
        )


def _native_qualified_name(value: object) -> str:
    """The name a module binding resolves one native module attribute to."""

    module = sys.modules[type(value).__module__]
    return next(
        f"{module.__name__}.{name}"
        for name, attribute in vars(module).items()
        if attribute is value
    )


_MISSING_QUALIFIED_NAME = _native_qualified_name(MISSING)
_CAST_QUALIFIED_NAME = f"{typing.cast.__module__}.{typing.cast.__qualname__}"


class CompactFieldDeclarationCall(NamedTuple):
    """A call that may declare a dataclass field: its callee and what it supplies.

    `positional_count` parameters are filled by position and `keywords` by
    name; a `*` or `**` argument sets `splats`, and what the call supplies is
    then unknown. A keyword forwarding a parameter the caller left absent
    supplies nothing and is not listed.
    """

    callee: str
    positional_count: int
    keywords: tuple[str, ...]
    splats: bool
    called: bool = True
    """False for a bare reference: the value is the callee itself."""

    @classmethod
    def of_reference(cls, callee: str) -> Self:
        return cls(callee=callee, positional_count=0, keywords=(), splats=False, called=False)

    @classmethod
    def of_call(
        cls, callee: str, call: ast.Call, absent: frozenset[str] = frozenset()
    ) -> Self:
        keywords = CallKeywords.of(call.keywords)
        return cls(
            callee=callee,
            positional_count=len(call.args),
            keywords=tuple(
                sorted(
                    name
                    for name, value in keywords.by_name.items()
                    if not (isinstance(value, ast.Name) and value.id in absent)
                )
            ),
            splats=bool(keywords.splats)
            or any(isinstance(argument, ast.Starred) for argument in call.args),
        )


class FieldDeclarationSignature(NamedTuple):
    """What a field declaration accepts: its positional parameters in order, the
    parameters that give the field a default when supplied, every
    parameter name its own calls supply on the way to `dataclasses.field`, and
    whether those calls already give the field a default."""

    positional: tuple[str, ...]
    missing: tuple[str, ...]
    supplied_on_the_way: tuple[str, ...] = ()
    defaulted: bool = False

    @classmethod
    def of_native(cls, declaration: Callable[..., object]) -> Self:
        parameters = inspect.signature(declaration).parameters.values()
        return cls(
            positional=tuple(
                parameter.name
                for parameter in parameters
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ),
            missing=tuple(
                parameter.name
                for parameter in parameters
                if parameter.default is MISSING
                and cls._gives_a_default(declaration(**{parameter.name: list}))
            ),
        )

    @staticmethod
    def _gives_a_default(declared: object) -> bool:
        """Whether a native field declaration, built with one parameter supplied,
        has a default: dataclasses reads a field as defaulted exactly when its
        `default` or `default_factory` is not MISSING (`kw_only` is not one)."""

        return isinstance(declared, Field) and (
            declared.default is not MISSING or declared.default_factory is not MISSING
        )

    def supplied(self, call: CompactFieldDeclarationCall) -> frozenset[str] | None:
        """The parameters the call names, or None when a splat hides them."""

        if call.splats or call.positional_count > len(self.positional):
            return None
        return frozenset((*self.positional[: call.positional_count], *call.keywords))

    def declares_bare_field(self, call: CompactFieldDeclarationCall) -> bool | None:
        """Whether the call declares a field with no default; None when it is open."""

        supplied = self.supplied(call)
        if supplied is None:
            return None
        return not self.defaulted and not supplied.intersection(self.missing)


class _FieldDeclarationFoldOutcome(ABC):
    """What a run of statements does once the absent parameters are folded."""

    @abstractmethod
    def then(
        self, fold: _FieldDeclarationFold, rest: Sequence[ast.stmt]
    ) -> _FieldDeclarationFoldOutcome:
        raise NotImplementedError

    def returned(self) -> ast.expr | None:
        return None


@dataclass(frozen=True)
class _FieldDeclarationReturned(_FieldDeclarationFoldOutcome):
    value: ast.expr

    def then(
        self, fold: _FieldDeclarationFold, rest: Sequence[ast.stmt]
    ) -> _FieldDeclarationFoldOutcome:
        return self

    def returned(self) -> ast.expr | None:
        return self.value


class _FieldDeclarationRaised(_FieldDeclarationFoldOutcome):
    def then(
        self, fold: _FieldDeclarationFold, rest: Sequence[ast.stmt]
    ) -> _FieldDeclarationFoldOutcome:
        return self


class _FieldDeclarationFellThrough(_FieldDeclarationFoldOutcome):
    def then(
        self, fold: _FieldDeclarationFold, rest: Sequence[ast.stmt]
    ) -> _FieldDeclarationFoldOutcome:
        return fold.statements(rest)


class _FieldDeclarationUnfolded(_FieldDeclarationFoldOutcome):
    """A statement or test the fold cannot decide: the function stays open."""

    def then(
        self, fold: _FieldDeclarationFold, rest: Sequence[ast.stmt]
    ) -> _FieldDeclarationFoldOutcome:
        return self


class _ReturnedFieldDeclaration(StatementValueQuestion[_FieldDeclarationFoldOutcome]):
    """A `return` hands back its value; a bare `return` declares no field."""

    def supplied(self, value: SuppliedValue) -> _FieldDeclarationFoldOutcome:
        return _FieldDeclarationReturned(value.value)

    def bare(self, value: NoStatementValue) -> _FieldDeclarationFoldOutcome:
        return _FieldDeclarationUnfolded()


@dataclass(frozen=True)
class _FieldDeclarationFold:
    """Run a function body with its MISSING-defaulted parameters absent: each
    `parameter is MISSING` test is decided, everything else stays open."""

    snapshot: ModuleNominalBindingSnapshot
    absent: frozenset[str]

    def statements(self, statements: Sequence[ast.stmt]) -> _FieldDeclarationFoldOutcome:
        if not statements:
            return _FieldDeclarationFellThrough()
        return self.statement(statements[0]).then(self, statements[1:])

    @singledispatchmethod
    def statement(self, statement: ast.stmt) -> _FieldDeclarationFoldOutcome:
        return _FieldDeclarationUnfolded()

    @statement.register
    def _return(self, statement: ast.Return) -> _FieldDeclarationFoldOutcome:
        return StatementValue.of(statement).answer(_ReturnedFieldDeclaration())

    @statement.register
    def _raise(self, statement: ast.Raise) -> _FieldDeclarationFoldOutcome:
        return _FieldDeclarationRaised()

    @statement.register
    def _pass(self, statement: ast.Pass) -> _FieldDeclarationFoldOutcome:
        return _FieldDeclarationFellThrough()

    @statement.register
    def _if(self, statement: ast.If) -> _FieldDeclarationFoldOutcome:
        taken = self.test(statement.test)
        if taken is None:
            return _FieldDeclarationUnfolded()
        return self.statements(statement.body if taken else statement.orelse)

    @singledispatchmethod
    def test(self, test: ast.expr) -> bool | None:
        return None

    @test.register
    def _compare(self, test: ast.Compare) -> bool | None:
        match test:
            case ast.Compare(
                left=ast.Name(id=name), ops=[op], comparators=[right]
            ) if name in self.absent and self.qualified(right) == _MISSING_QUALIFIED_NAME:
                return self.absence(op)
        return None

    @singledispatchmethod
    def absence(self, op: ast.cmpop) -> bool | None:
        """What `parameter <op> MISSING` says when the parameter is absent."""

        return None

    @absence.register
    def _is(self, op: ast.Is) -> bool | None:
        return True

    @absence.register
    def _is_not(self, op: ast.IsNot) -> bool | None:
        return False

    @test.register
    def _boolean(self, test: ast.BoolOp) -> bool | None:
        values = tuple(self.test(value) for value in test.values)
        if None in values:
            return None
        return self.combined(test.op, values)

    @singledispatchmethod
    def combined(self, op: ast.boolop, values: tuple[bool, ...]) -> bool:
        raise TypeError(f"no boolean operator {type(op).__name__}")

    @combined.register
    def _all(self, op: ast.And, values: tuple[bool, ...]) -> bool:
        return all(values)

    @combined.register
    def _any(self, op: ast.Or, values: tuple[bool, ...]) -> bool:
        return any(values)

    @test.register
    def _unary(self, test: ast.UnaryOp) -> bool | None:
        value = self.test(test.operand)
        return None if value is None else self.unary(test.op, value)

    @singledispatchmethod
    def unary(self, op: ast.unaryop, value: bool) -> bool | None:
        return None

    @unary.register
    def _not(self, op: ast.Not, value: bool) -> bool:
        return not value

    def qualified(self, reference: ast.expr) -> str | None:
        parts = ExpressionReference.decoded(reference).attribute_chain
        if parts is None:
            return None
        binding = self.snapshot.binding_for(parts[0])
        if binding is None:
            return None
        return ".".join((binding.qualified_name, *parts[1:]))

    def returned_call(self, value: ast.expr) -> CompactFieldDeclarationCall | None:
        """The field-declaration call a returned value makes, seeing through `cast`."""

        if not isinstance(value, ast.Call):
            return None
        callee = self.qualified(value.func)
        if callee is None:
            return None
        if callee == _CAST_QUALIFIED_NAME and len(value.args) == 2 and not value.keywords:
            return self.returned_call(value.args[1])
        return CompactFieldDeclarationCall.of_call(callee, value, self.absent)


class CompactFieldDeclarationWrapper(NamedTuple):
    """A module function that, with its MISSING-defaulted parameters absent,
    returns one call: a field declaration when that call is one."""

    qualified_name: str
    signature: FieldDeclarationSignature
    returned: CompactFieldDeclarationCall

    @classmethod
    def of_function(
        cls,
        module_name: str,
        function: ast.FunctionDef,
        snapshot: ModuleNominalBindingSnapshot,
    ) -> Self | None:
        """The wrapper the function is, or None when its body stays open."""

        arguments = function.args
        if variadic_parameters(arguments):
            return None
        positional = (*arguments.posonlyargs, *arguments.args)
        defaulted = (
            *zip(positional[len(positional) - len(arguments.defaults) :], arguments.defaults),
            *(
                (argument, default)
                for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
                if default is not None
            ),
        )
        fold = _FieldDeclarationFold(snapshot, frozenset())
        # Every MISSING parameter of a wrapper is read as giving a default when
        # supplied: wrong only toward "a different field", never a false match.
        missing = tuple(
            argument.arg
            for argument, default in defaulted
            if fold.qualified(default) == _MISSING_QUALIFIED_NAME
        )
        fold = _FieldDeclarationFold(snapshot, frozenset(missing))
        value = fold.statements(statements_without_docstring(function.body)).returned()
        returned = None if value is None else fold.returned_call(value)
        if returned is None:
            return None
        if not missing and DataclassRuntimeDeclaration.for_qualified_name(
            returned.callee
        ) is None:
            return None
        return cls(
            qualified_name=f"{module_name}.{function.name}",
            signature=FieldDeclarationSignature(
                positional=tuple(argument.arg for argument in positional),
                missing=missing,
            ),
            returned=returned,
        )


def module_field_declaration_wrappers(
    parsed_module: ParsedModule,
    snapshot: ModuleNominalBindingSnapshot,
) -> tuple[CompactFieldDeclarationWrapper, ...]:
    """Every top-level function of the module that may wrap `dataclasses.field`,
    its names read through the module's final bindings."""

    return tuple(
        wrapper
        for statement in parsed_module.module.body
        if isinstance(statement, ast.FunctionDef)
        if (
            wrapper := CompactFieldDeclarationWrapper.of_function(
                parsed_module.module_name, statement, snapshot
            )
        )
        is not None
    )


@dataclass(frozen=True)
class FieldDeclarationIndex:
    """Which callees declare a dataclass field, across one repository.

    A native field factory declares one directly. A wrapper declares one when
    the call it returns does; the field has a default when that call supplies
    one of the callee's MISSING parameters. A wrapper reaching itself again
    declares nothing.
    """

    wrappers: Mapping[str, CompactFieldDeclarationWrapper]
    _projections: tuple[CompactModuleClassHeader, ...]
    built: dict[str, bool] = field(default_factory=dict, compare=False, repr=False)
    """`builds_own_instance`, remembered per class symbol."""
    kept_metaclasses: dict[str, bool] = field(default_factory=dict, compare=False, repr=False)
    """`keeps_type_call`, remembered per metaclass symbol."""

    @cached_property
    def classes(self) -> RegisteringMetaclassAuthority[CompactIndexedClass]:
        """The repository's classes, and installed ones read from source, as bases
        and metaclasses resolve to them."""

        return RegisteringMetaclassAuthority(
            {
                indexed_class.symbol: indexed_class
                for projection in self._projections
                for indexed_class in projection.classes
            },
            CompactRepositoryPublicExposureIndex(self._projections),
        )

    @classmethod
    def of_projections(cls, projections: Iterable[CompactModuleClassHeader]) -> Self:
        projections = tuple(projections)
        return cls(
            {
                wrapper.qualified_name: wrapper
                for projection in projections
                for wrapper in projection.field_declaration_wrappers
            },
            projections,
        )

    def named_class(self, qualified_name: str) -> CompactIndexedClass | None:
        """The class a qualified name reaches, scanned or installed, re-exports followed."""

        found = InstalledClassDeclarations.declaration(qualified_name, self.classes.exposure)
        return None if found is None else self.classes.scanned.get(found.symbol, found)

    def builds_own_instance(
        self, declaration: CompactIndexedClass, pending: frozenset[str] = frozenset()
    ) -> bool:
        """Calling the class returns an instance of it, never a `dataclasses.Field`.

        `type.__call__` returns what the class's `__new__` returns, so no class
        along the ancestry defines `__new__`; every ancestor is read from source
        or is a native root; and no metaclass, nor any ancestor of one, defines
        `__call__`, which would replace `type.__call__`.
        """

        symbol = declaration.symbol
        if symbol in pending:
            return False
        if symbol not in self.built:
            module_name = declaration.module_name
            self.built[symbol] = (
                "__new__" not in declaration.member_binding_names
                and declaration.base_references_are_complete
                and all(
                    self.keeps_type_call(reference, module_name, pending | {symbol})
                    for reference in declaration.registry_facts.metaclass_references
                )
                and all(
                    self.native_root(reference, module_name)
                    or (
                        (base := self.classes.declaration_for(reference, module_name)) is not None
                        and self.builds_own_instance(base, pending | {symbol})
                    )
                    for reference in declaration.registry_facts.base_references
                )
            )
        return self.built[symbol]

    def native_root(self, reference: CompactNominalReference, module_name: str) -> bool:
        qualified_name = reference.settled_qualified_name(self.classes, module_name)
        return qualified_name is not None and NativeMroBase.for_qualified_name(qualified_name) is not None

    def keeps_type_call(
        self, reference: CompactNominalReference, module_name: str, pending: frozenset[str]
    ) -> bool:
        """A metaclass that calls its classes with `type.__call__`: neither it nor
        an ancestor defines `__call__`. `type` and `abc.ABCMeta` define none."""

        qualified_name = reference.settled_qualified_name(self.classes, module_name)
        if qualified_name in ("builtins.type", "abc.ABCMeta"):
            return True
        metaclass = self.classes.declaration_for(reference, module_name)
        if metaclass is None or metaclass.symbol in pending:
            return False
        cached = self.kept_metaclasses.get(metaclass.symbol)
        if cached is not None:
            return cached
        kept = (
            "__call__" not in metaclass.member_binding_names
            and metaclass.base_references_are_complete
            and all(
                self.keeps_type_call(base, metaclass.module_name, pending | {metaclass.symbol})
                for base in metaclass.registry_facts.base_references
            )
        )
        self.kept_metaclasses[metaclass.symbol] = kept
        return kept

    @cached_property
    def signatures(self) -> Mapping[str, FieldDeclarationSignature]:
        signatures: dict[str, FieldDeclarationSignature] = {
            member.qualified_name: FieldDeclarationSignature.of_native(member.declaration)
            for member in DataclassRuntimeDeclaration
            if member.is_field_factory
        }
        refused: set[str] = set()

        def settle(name: str, pending: frozenset[str]) -> FieldDeclarationSignature | None:
            if name in signatures:
                return signatures[name]
            wrapper = self.wrappers.get(name)
            if wrapper is None or name in refused or name in pending:
                return None
            callee = settle(wrapper.returned.callee, pending | {name})
            supplied = None if callee is None else callee.supplied(wrapper.returned)
            if callee is None or supplied is None:
                refused.add(name)
                return None
            signature = wrapper.signature._replace(
                supplied_on_the_way=tuple(sorted({*callee.supplied_on_the_way, *supplied})),
                defaulted=callee.defaulted or bool(supplied.intersection(callee.missing)),
            )
            signatures[name] = signature
            return signature

        for name in self.wrappers:
            settle(name, frozenset())
        return signatures

    def declares_bare_field(self, call: CompactFieldDeclarationCall) -> bool | None:
        """Whether the call declares a field without a default: False when its
        callee declares no field, None when a splat leaves it open."""

        signature = self.signatures.get(call.callee)
        if signature is None:
            return False
        return signature.declares_bare_field(call)

    def field_role(self, call: CompactFieldDeclarationCall) -> CompactDataclassFieldRole:
        """A stored init field when the call declares a field and nothing on its
        way to `dataclasses.field` names `init`; otherwise unresolved."""

        if call.called and (signature := self.signatures.get(call.callee)) is not None:
            if signature.supplied(call) is None or self.supplies(call, "init"):
                return CompactDataclassFieldRole.UNRESOLVED
            return CompactDataclassFieldRole.STORED_INIT
        named = self.named_class(call.callee)
        if named is not None and (not call.called or self.builds_own_instance(named)):
            # A class object is an instance of its metaclass, a `type`, which
            # `Field`'s instance layout cannot share: it is never a `Field`.
            return CompactDataclassFieldRole.STORED_INIT
        return CompactDataclassFieldRole.UNRESOLVED

    def supplies(self, call: CompactFieldDeclarationCall, parameter: str) -> bool:
        """Whether the call, or a declaration it reaches, names this parameter."""

        signature = self.signatures[call.callee]
        supplied = signature.supplied(call) or frozenset()
        return parameter in supplied or parameter in signature.supplied_on_the_way


def _dataclass_role_reference(annotation: ast.expr) -> ast.expr | None:
    """Return only a top-level reference that can alter dataclass field role."""

    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return annotation
    if isinstance(annotation, ast.Subscript):
        return annotation.value
    return annotation if isinstance(annotation, ast.Name | ast.Attribute) else None


_LITERAL_DEFAULT_NODES = (
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Dict,
    ast.UnaryOp,
    ast.BinOp,
    ast.unaryop,
    ast.operator,
    ast.expr_context,
)


def _is_literal_default(value: ast.expr) -> bool:
    """A default built only from literals: it names nothing, so it cannot be a
    `dataclasses.Field` and the member is a plain stored field."""

    return all(isinstance(node, _LITERAL_DEFAULT_NODES) for node in ast.walk(value))


def _dataclass_plain_annotation_is_resolved(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    class_aliases: dict[str, str],
    annotation: ast.expr,
    preceding_class_bound_names: frozenset[str],
) -> bool:
    """Prove every evaluated root in a non-role annotation expression."""

    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return False
    root_names = frozenset(
        node.id
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )
    return all(
        root_name in class_aliases
        or module_binding_snapshot.binding_for(root_name) is not None
        or module_binding_snapshot.resolves_unshadowed_builtin(
            root_name,
            preceding_class_bound_names=preceding_class_bound_names,
        )
        for root_name in root_names
    )


def _dataclass_fields(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    node: ast.ClassDef,
    class_binding: DataclassInitBinding,
) -> tuple[CompactDataclassFieldDeclaration, ...]:
    fields: list[CompactDataclassFieldDeclaration] = []
    preceding_bound_names: set[str] = set()
    class_aliases: dict[str, str] = {}
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target,
            ast.Name,
        ):
            role = _dataclass_field_role(
                module_binding_snapshot,
                class_aliases,
                statement,
                frozenset(preceding_bound_names),
            )
            fields.append(
                CompactDataclassFieldDeclaration(
                    name=statement.target.id,
                    line=statement.lineno,
                    role=role,
                    binding=DataclassInitBinding.of_keywords(
                        (
                            statement.value.keywords
                            if isinstance(statement.value, ast.Call)
                            else ()
                        ),
                        class_binding,
                    ),
                    declaration_call=role.declaration_call(
                        statement,
                        module_binding_snapshot,
                        class_aliases,
                        frozenset(preceding_bound_names),
                    ),
                )
            )
            class_binding = role.binding_of_later_fields(class_binding)
        direct_aliases: dict[str, str] = {}
        if isinstance(statement, ast.Assign | ast.AnnAssign):
            for alias_target, assigned in StatementValues.keyed(
                (target, statement) for target in single_name_targets(statement)
            ).supplied:
                qualified_value = _class_scope_qualified_import_name(
                    module_binding_snapshot,
                    class_aliases,
                    assigned.value,
                    frozenset(preceding_bound_names),
                )
                if qualified_value is not None:
                    direct_aliases[alias_target.id] = qualified_value
        statement_bound_names = _direct_statement_bound_names(statement)
        preceding_bound_names.update(statement_bound_names)
        for bound_name in statement_bound_names:
            class_aliases.pop(bound_name, None)
        class_aliases.update(direct_aliases)
    return tuple(fields)


def _field_member_collision_lines(
    node: ast.ClassDef,
    fields: tuple[CompactDataclassFieldDeclaration, ...],
) -> tuple[int, ...]:
    field_names = frozenset(field.name for field in fields)
    seen_field_bindings: set[str] = set()
    collision_lines: set[int] = set()
    for statement in node.body:
        bound_fields = field_names.intersection(
            _direct_statement_bound_names(statement)
        )
        if seen_field_bindings.intersection(bound_fields):
            collision_lines.add(statement.lineno)
        seen_field_bindings.update(bound_fields)
    return tuple(sorted(collision_lines))


def _dynamic_dataclass_schema_lines(node: ast.ClassDef) -> tuple[int, ...]:
    """Return class-execution sites that can change the direct field schema."""

    lines: set[int] = set()

    class DynamicSchemaVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, child: ast.ClassDef) -> None:
            return None

        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
            return None

        def visit_Lambda(self, child: ast.Lambda) -> None:
            return None

        def visit_AnnAssign(self, child: ast.AnnAssign) -> None:
            lines.add(child.lineno)

        def visit_Name(self, child: ast.Name) -> None:
            if child.id == "__annotations__" and isinstance(child.ctx, ast.Store):
                lines.add(child.lineno)

        def visit_Subscript(self, child: ast.Subscript) -> None:
            if (
                isinstance(child.ctx, ast.Store | ast.Del)
                and isinstance(child.value, ast.Name)
                and child.value.id == "__annotations__"
            ):
                lines.add(child.lineno)
            self.generic_visit(child)

        def visit_Call(self, child: ast.Call) -> None:
            called_parts = ExpressionReference.decoded(child.func).attribute_chain
            if called_parts == ("exec",) or called_parts in {
                ("__annotations__", "clear"),
                ("__annotations__", "pop"),
                ("__annotations__", "popitem"),
                ("__annotations__", "setdefault"),
                ("__annotations__", "update"),
            }:
                lines.add(child.lineno)
            elif (
                len(called_parts or ()) == 2
                and called_parts[-1] in {"setdefault", "update"}
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Call)
                and ExpressionReference.decoded(child.func.value.func).attribute_chain
                in {("locals",), ("vars",)}
            ):
                lines.add(child.lineno)
            self.generic_visit(child)

    visitor = DynamicSchemaVisitor()
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign):
            continue
        visitor.visit(statement)
    return tuple(sorted(lines))


def _dataclass_declaration(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    qualname: str,
    node: ast.ClassDef,
) -> CompactDataclassFacet:
    decorator_qualified_names = tuple(
        _class_scope_qualified_import_name(
            module_binding_snapshot,
            {},
            decorator.func if isinstance(decorator, ast.Call) else decorator,
            frozenset(),
        )
        for decorator in node.decorator_list
    )
    decorator_runtime_declarations = tuple(
        DataclassRuntimeDeclaration.dataclass_decorator_for_name(qualified_name)
        for qualified_name in decorator_qualified_names
    )
    dataclass_decorator_indexes = tuple(
        index
        for index, runtime_declaration in enumerate(decorator_runtime_declarations)
        if runtime_declaration is not None
    )
    dataclass_like_indexes = tuple(
        index
        for index, decorator in enumerate(node.decorator_list)
        if DataclassRuntimeDeclaration.dataclass_decorator_for_name(
            _terminal_reference_name(
                decorator.func if isinstance(decorator, ast.Call) else decorator
            )
        )
        is not None
    )
    dataclass_candidate_indexes = frozenset(
        (*dataclass_decorator_indexes, *dataclass_like_indexes)
    )
    if not dataclass_candidate_indexes:
        return NoDataclassDeclaration()

    decorator = node.decorator_list[min(dataclass_candidate_indexes)]
    fields = _dataclass_fields(
        module_binding_snapshot,
        node,
        DataclassInitBinding.of_keywords(
            decorator.keywords if isinstance(decorator, ast.Call) else (),
            DataclassInitBinding.POSITIONAL,
        ),
    )
    failures: list[CompactProductDeclarationFailure] = []
    if not dataclass_decorator_indexes:
        failures.extend(
            CompactProductDeclarationFailure(
                node.decorator_list[index].lineno,
                CompactProductAuthorityViolation.UNRESOLVED_DATACLASS_DECORATOR,
            )
            for index in dataclass_candidate_indexes
        )
    if len(dataclass_candidate_indexes) != 1:
        failures.extend(
            CompactProductDeclarationFailure(
                node.decorator_list[index].lineno,
                CompactProductAuthorityViolation.MULTIPLE_DATACLASS_DECORATORS,
            )
            for index in sorted(dataclass_candidate_indexes)
        )
    if dataclass_decorator_indexes:
        failures.extend(
            _dataclass_decorator_failures(
                node.decorator_list[dataclass_decorator_indexes[0]]
            )
        )
    failures.extend(
        CompactProductDeclarationFailure(
            node.decorator_list[index].lineno,
            CompactProductAuthorityViolation.CUSTOM_CLASS_DECORATOR,
        )
        for index, qualified_name in enumerate(decorator_qualified_names)
        if index not in dataclass_decorator_indexes
        if not ClassMethodPromotionSafeDecorator.qualified_name_preserves_product_schema(
            qualified_name
        )
    )
    failures.extend(
        CompactProductDeclarationFailure(
            keyword.value.lineno,
            CompactProductAuthorityViolation.CUSTOM_CLASS_CREATION,
        )
        for keyword in node.keywords
    )
    if "." in qualname:
        failures.append(
            CompactProductDeclarationFailure(
                node.lineno,
                CompactProductAuthorityViolation.NESTED_CLASS_SCOPE,
            )
        )
    lifecycle_violation = CompactProductAuthorityViolation.CUSTOM_PRODUCT_LIFECYCLE
    failures.extend(
        CompactProductDeclarationFailure(statement.lineno, lifecycle_violation)
        for statement in node.body
        if lifecycle_violation.is_violated_by_member_names(
            _direct_statement_bound_names(statement)
        )
    )
    failures.extend(
        CompactProductDeclarationFailure(
            line,
            CompactProductAuthorityViolation.FIELD_MEMBER_COLLISION,
        )
        for line in _field_member_collision_lines(node, fields)
    )
    failures.extend(
        CompactProductDeclarationFailure(
            line,
            CompactProductAuthorityViolation.DYNAMIC_FIELD_SCHEMA,
        )
        for line in _dynamic_dataclass_schema_lines(node)
    )
    return CompactDataclassDeclaration(
        runtime_declaration=next(
            (
                runtime_declaration
                for runtime_declaration in decorator_runtime_declarations
                if runtime_declaration is not None
            ),
            None,
        ),
        fields=fields,
        failures=tuple(dict.fromkeys(failures)),
        frozenness=DataclassFrozenness.of_keywords(
            decorator.keywords if isinstance(decorator, ast.Call) else ()
        ),
    )


TYPE_VARIABLE_CONSTRUCTORS = frozenset(
    f"{module}.{name}"
    for module in ("typing", "typing_extensions")
    for name in ("TypeVar", "ParamSpec", "TypeVarTuple")
)


def _module_type_variable_names(parsed_module: ParsedModule) -> tuple[str, ...]:
    """Names the module binds at top level to a type variable constructor."""

    aliases = module_import_aliases(parsed_module)
    names: list[str] = []
    for statement in parsed_module.module.body:
        match statement:
            case ast.Assign(targets=targets, value=ast.Call(func=constructor)):
                parts = ExpressionReference.decoded(constructor).attribute_chain
                if not parts or parts[0] not in aliases:
                    continue
                qualified = ".".join((aliases[parts[0]], *parts[1:]))
                if qualified in TYPE_VARIABLE_CONSTRUCTORS:
                    names.extend(
                        target.id for target in targets if isinstance(target, ast.Name)
                    )
    return tuple(names)


def _compact_callable_member_declarations(
    node: ast.ClassDef, module_binding_snapshot: ModuleNominalBindingSnapshot
) -> tuple[CompactCallableMemberDeclaration, ...]:
    """Each `def` of the class body, except overload stubs and a property's
    setter or deleter, which rebind no new member type."""

    declarations: list[CompactCallableMemberDeclaration] = []
    property_names: set[str] = set()
    for statement in node.body:
        if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if module_binding_snapshot.decorated_by(
            statement, (OVERLOAD_DECORATOR,)
        ) or _redefines_property(statement, property_names):
            continue
        reads_as_property = module_binding_snapshot.decorated_by(
            statement, PROPERTY_DECORATORS
        )
        if reads_as_property:
            property_names.add(statement.name)
        declared = tuple(
            (CompactAnnotatedProperty if reads_as_property else CompactAnnotatedMethod)(
                statement.name, statement.lineno, ast.unparse(returns)
            )
            for returns in declared_annotations(statement)
        )
        declarations.extend(
            declared or (CompactUnannotatedCallable(statement.name, statement.lineno),)
        )
    return tuple(declarations)


def _redefines_property(
    statement: ast.FunctionDef | ast.AsyncFunctionDef, property_names: Collection[str]
) -> bool:
    """`@name.setter` or `@name.deleter` on a property the class body declared above."""

    return any(
        isinstance(decorator, ast.Attribute)
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id in property_names
        and decorator.attr in PROPERTY_REDEFINITION_ATTRIBUTES
        for decorator in statement.decorator_list
    )


def native_identity(value: Callable[..., object]) -> str:
    """The qualified name a binding resolves a native object to."""

    return f"{value.__module__}.{value.__qualname__}"


PROPERTY_DECORATOR = native_identity(property)
PROPERTY_DECORATORS = frozenset((PROPERTY_DECORATOR, native_identity(cached_property)))
"""The descriptors whose read is the decorated function's return."""
PROPERTY_REDEFINITION_ATTRIBUTES = frozenset(
    (property.setter.__name__, property.deleter.__name__)
)
ABSTRACT_METHOD_DECORATOR = native_identity(abstractmethod)
FINAL_DECORATOR = native_identity(typing.final)
OVERLOAD_DECORATOR = native_identity(typing.overload)
STATIC_METHOD_DECORATOR = native_identity(staticmethod)
CLASS_METHOD_DECORATOR = native_identity(classmethod)


class InstalledClassDeclarations:
    """Classes of modules outside the scan, projected from installed source and never run."""

    MAX_REEXPORTS: ClassVar[int] = 8

    @staticmethod
    @cache
    def module(module_name: str) -> CompactModuleClassProjection | None:
        found = InstalledModule.family_items(
            module_name, CompactModuleClassProjectionFamily
        )
        if found is None:
            return None
        (projection,) = found
        return projection

    @staticmethod
    @cache
    def stub(module_name: str) -> CompactModuleClassProjection | None:
        """What the stub beside the module's `.py` states, for a name the `.py`
        binds only at run time."""

        found = InstalledModule.family_items(module_name, CompactModuleClassProjectionFamily, stub=True)
        if found is None:
            return None
        (projection,) = found
        return projection

    @classmethod
    def declaration(
        cls,
        qualified_name: str,
        scan: CompactRepositoryPublicExposureIndex,
    ) -> CompactIndexedClass | None:
        """The class a qualified name reaches, following re-exports.

        A module the scan holds is read from the scan, never from an installed
        copy: the installed bytes of a scanned package can be another checkout
        of it. A module the scan holds more than once is not known. Only a
        module outside the scan is read from its installed source.
        """

        modules = scan.module_projection_multiplicity
        installed_lookups: dict[
            str, tuple[dict[str, CompactIndexedClass], dict[str, str]] | None
        ] = {}
        for _hop in range(cls.MAX_REEXPORTS):
            module_name, _dot, name = qualified_name.rpartition(".")
            if module_name in modules.ambiguous_handles:
                return None
            lookup = scan.module_declaration_lookup(module_name)
            in_scan = lookup is not None
            if not in_scan:
                if module_name in installed_lookups:
                    lookup = installed_lookups[module_name]
                else:
                    projection = cls.module(module_name) if module_name else None
                    lookup = (
                        None
                        if projection is None
                        else (
                            {declared.symbol: declared for declared in projection.classes},
                            dict(projection.import_aliases),
                        )
                    )
                    installed_lookups[module_name] = lookup
            if lookup is None:
                return None
            classes_by_symbol, aliases = lookup
            declared = classes_by_symbol.get(qualified_name)
            if declared is not None:
                return declared
            target = aliases.get(name)
            if target is None and not in_scan:
                # The `.py` computes the name when it is read (`__getattr__`); its stub states it.
                stated = cls.stub(module_name)
                target = None if stated is None else dict(stated.import_aliases).get(name)
            if target is None or target == qualified_name:
                return None
            qualified_name = target
        return None


@dataclass(frozen=True)
class RegisteringMetaclassAuthority(Generic[ClassDeclarationT]):
    """Whether a class's metaclass registers the classes it creates, from what it does.

    The `metaclass=` binding is resolved like a base: to a scanned class, or
    else to an installed module's class read from source, following
    re-exports. The metaclass registers when it, or an ancestor reached the
    same way, stores the class it creates into a mapping
    (`ClassRegistryFacts.stores_created_class`). A native root stores none,
    and an ancestor that cannot be read is not known to register.

    A registry family is a class whose subclasses are stored in a mapping as
    they are created: a registering metaclass creates it
    (`registers`, `inherits_registering`), or it or an ancestor defines an
    `__init_subclass__` that stores the subclass
    (`ClassRegistryFacts.stores_subclass`, `stores_subclasses`).
    """

    scanned: Mapping[str, ClassDeclarationT]
    exposure: CompactRepositoryPublicExposureIndex
    _suffix_index: _UniqueKnownSymbolSuffixIndex | None = field(
        default=None, repr=False, compare=False
    )
    registering: dict[str, bool] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    """`registers`, remembered per class symbol while ancestors are walked."""
    _inherits_cache: dict[str, bool] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _stores_cache: dict[str, bool] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _declaration_cache: dict[
        tuple[CompactNominalReference, str], ClassDeclaration | None
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    MAX_ANCESTORS: ClassVar[int] = 64

    @cached_property
    def suffix_index(self) -> _UniqueKnownSymbolSuffixIndex:
        if self._suffix_index is not None:
            return self._suffix_index
        return _unique_known_symbol_by_suffix(frozenset(self.scanned))

    def registers(self, declaration: ClassDeclaration) -> bool:
        if not declaration.registry_facts.metaclass_references:
            return False
        pending = [
            (reference, declaration.module_name)
            for reference in declaration.registry_facts.metaclass_references
        ]
        seen: set[str] = set()
        for _step in range(self.MAX_ANCESTORS):
            if not pending:
                break
            reference, module_name = pending.pop()
            found = self.declaration_for(reference, module_name)
            if found is None or found.symbol in seen:
                continue
            seen.add(found.symbol)
            if found.registry_facts.stores_created_class:
                return True
            pending.extend(
                (base, found.module_name)
                for base in found.registry_facts.base_references
            )
        return False

    def is_registry_family(self, declaration: ClassDeclaration) -> bool:
        """Whether the subclasses created under it are stored in a mapping as they are created."""

        return (
            self.registers_remembered(declaration)
            or self.inherits_registering(declaration)
            or self.stores_subclasses(declaration)
        )

    def registry_family_key_type_name(
        self, declaration: CompactIndexedClass
    ) -> str | None:
        """The key type `Family[Key]` names, for the first subscripted base that is a registry family."""

        return next(
            (
                key_type_name
                for reference, key_type_name in zip(
                    declaration.base_references,
                    declaration.base_key_type_names,
                    strict=True,
                )
                if key_type_name is not None
                and (found := self.declaration_for(reference, declaration.module_name))
                is not None
                and self.is_registry_family(found)
            ),
            None,
        )

    def inherits_registering(self, declaration: ClassDeclaration) -> bool:
        """Whether a class it inherits from is created by a registering metaclass.

        Python creates a subclass with its bases' metaclass, so a class whose
        ancestor's metaclass registers is registered too, whatever it declares
        itself.
        """

        symbol = declaration.symbol
        cached = self._inherits_cache.get(symbol)
        if cached is not None:
            return cached
        answer = self._inherits_registering_recursive(declaration, {symbol}, 0)
        self._inherits_cache[symbol] = answer
        return answer

    def _inherits_registering_recursive(
        self, declaration: ClassDeclaration, visiting: set[str], depth: int
    ) -> bool:
        if depth >= self.MAX_ANCESTORS:
            return bool(declaration.registry_facts.base_references)
        for base in declaration.registry_facts.base_references:
            found = self.declaration_for(base, declaration.module_name)
            if found is None or found.symbol in visiting:
                continue
            if self.registers_remembered(found):
                return True
            cached = self._inherits_cache.get(found.symbol)
            if cached is not None:
                if cached:
                    return True
                continue
            visiting.add(found.symbol)
            found_inherits = self._inherits_registering_recursive(
                found, visiting, depth + 1
            )
            visiting.discard(found.symbol)
            if depth + 1 < self.MAX_ANCESTORS:
                self._inherits_cache[found.symbol] = found_inherits
            if found_inherits:
                return True
        return False

    def stores_subclasses(self, declaration: ClassDeclaration) -> bool:
        """Whether its own `__init_subclass__`, or one it inherits, stores each subclass."""

        symbol = declaration.symbol
        cached = self._stores_cache.get(symbol)
        if cached is not None:
            return cached
        answer = declaration.registry_facts.stores_subclass or self._stores_subclasses_recursive(
            declaration, {symbol}, 0
        )
        self._stores_cache[symbol] = answer
        return answer

    def _stores_subclasses_recursive(
        self, declaration: ClassDeclaration, visiting: set[str], depth: int
    ) -> bool:
        if depth >= self.MAX_ANCESTORS:
            return bool(declaration.registry_facts.base_references)
        for base in declaration.registry_facts.base_references:
            found = self.declaration_for(base, declaration.module_name)
            if found is None or found.symbol in visiting:
                continue
            cached = self._stores_cache.get(found.symbol)
            if cached is not None:
                if cached:
                    return True
                continue
            if found.registry_facts.stores_subclass:
                self._stores_cache[found.symbol] = True
                return True
            visiting.add(found.symbol)
            found_stores = self._stores_subclasses_recursive(found, visiting, depth + 1)
            visiting.discard(found.symbol)
            if depth + 1 < self.MAX_ANCESTORS:
                self._stores_cache[found.symbol] = found_stores
            if found_stores:
                return True
        return False

    def registers_remembered(self, declaration: ClassDeclaration) -> bool:
        if declaration.symbol not in self.registering:
            self.registering[declaration.symbol] = self.registers(declaration)
        return self.registering[declaration.symbol]

    def some_ancestor(
        self,
        declaration: ClassDeclaration,
        holds: Callable[[ClassDeclaration], bool],
    ) -> bool:
        """Whether `holds` is true of an ancestor.

        Bases resolve as `declaration_for` resolves them; an ancestor that
        cannot be read is not known to hold. An ancestry left unread past
        `MAX_ANCESTORS` is not known not to hold, so it counts as holding:
        the answer that refuses a move.
        """

        pending = [
            (base, declaration.module_name)
            for base in declaration.registry_facts.base_references
        ]
        seen: set[str] = set()
        for _step in range(self.MAX_ANCESTORS):
            if not pending:
                break
            reference, module_name = pending.pop()
            found = self.declaration_for(reference, module_name)
            if found is None or found.symbol in seen:
                continue
            seen.add(found.symbol)
            if holds(found):
                return True
            pending.extend(
                (base, found.module_name)
                for base in found.registry_facts.base_references
            )
        return bool(pending)

    def declaration_for(
        self, reference: CompactNominalReference, module_name: str
    ) -> ClassDeclaration | None:
        key = (reference, module_name)
        if key in self._declaration_cache:
            return self._declaration_cache[key]
        symbol = CompactClassFamilyIndexBuilder._resolved_bound_symbol(
            reference, module_name, self.scanned, self.suffix_index
        )
        if symbol is not None:
            found: ClassDeclaration | None = self.scanned[symbol]
            self._declaration_cache[key] = found
            return found
        qualified_name = reference.settled_qualified_name(self, module_name)
        if qualified_name is None:
            self._declaration_cache[key] = None
            return None
        installed = InstalledClassDeclarations.declaration(qualified_name, self.exposure)
        found = None if installed is None else self.scanned.get(installed.symbol, installed)
        self._declaration_cache[key] = found
        return found

    def star_settled_name(
        self, reference: CompactNominalReference, module_name: str
    ) -> str | None:
        """A reference whose root a star import may rebind, settled by the repository's exports."""

        root, *rest = reference.source_parts
        projection = self.exposure.projections_by_module_name.get(module_name)
        settlement = self.exposure.star_settlement(
            module_name,
            root,
            None if projection is None else dict(projection.import_aliases).get(root),
        )
        # settled_qualified_name's str | None protocol.
        return next(
            (
                ".".join((binding, *rest))
                for _, binding in StarSettlements.keyed(((root, settlement),)).settled
            ),
            None,
        )

    def resolved(self) -> dict[str, ClassDeclarationT]:
        return {
            symbol: declaration.with_registering_metaclass(self)
            for symbol, declaration in self.scanned.items()
        }


@dataclass
class ClassCreationOrderProof:
    """What running a class's creation code does, read from its declarations.

    Python builds the order from the bases a metaclass's `__new__` hands
    `type.__new__`, so creation keeps the order C3 gives the declared bases
    when every metaclass along the class's ancestry, and every ancestor of
    one, keeps its bases (`CreatedClassStore.keeps_bases`), and nothing in the
    scan stores a `__bases__`. A metaclass is resolved like a base: scanned,
    or read from installed source, never run; a modelled one
    (`MODELLED_METACLASSES`) ends its branch, and one NRA cannot read leaves
    the order unproved. Class keywords, `__init_subclass__` and `__set_name__`
    run after the order exists.

    The names creation binds are a separate answer (`bindings`): a member
    lookup that finds no body binding the name stays open when creation may
    bind it.
    """

    metaclasses: RegisteringMetaclassAuthority[CompactIndexedClass]
    stores_bases: bool
    MAX_METACLASS_ANCESTORS: ClassVar[int] = 64

    def keeps_order(
        self, declaration: CompactIndexedClass, ancestry: tuple[CompactIndexedClass, ...]
    ) -> bool:
        try:
            self.require_order(ancestry)
        except NeedsInput:
            return False
        return True

    def require_order(self, ancestry: tuple[CompactIndexedClass, ...]) -> None:
        """Refuses, naming why, unless creating a class with this ancestry keeps
        the order C3 gives its declared bases."""

        if self.stores_bases:
            raise NeedsInput(
                "the scan stores a `__bases__`, so no declared order is proved to be the created one",
                NeededInput.PROOF,
            )
        self._walk(ancestry)

    def bindings(self, ancestry: tuple[CompactIndexedClass, ...]) -> CreationBindings:
        """Every name creating a class with this ancestry may bind on it."""

        try:
            walked = self._walk(ancestry)
        except NeedsInput:
            return CreationBindings(computed=True)
        hooks = NO_BINDINGS
        for indexed in ancestry:
            hooks |= indexed.registry_facts.subclass_hook_bindings
            if any(
                member.expression is not None and member.name in CLASS_METHOD_OWNERSHIP_HOOK_NAMES
                for member in indexed.direct_member_declarations
            ):
                # A hook bound by assignment runs code NRA does not read.
                hooks |= CreationBindings(computed=True)
        return walked | hooks

    def _walk(self, ancestry: tuple[CompactIndexedClass, ...]) -> CreationBindings:
        """What the metaclass creation code along `ancestry` binds; refuses,
        naming the metaclass, when one may change the order or cannot be read."""

        pending = [
            (reference, indexed.module_name)
            for indexed in ancestry
            for reference in indexed.registry_facts.metaclass_references
        ]
        found_bindings = NO_BINDINGS
        modelled = False
        seen: set[str] = set()
        for _step in range(self.MAX_METACLASS_ANCESTORS):
            if not pending:
                break
            reference, module_name = pending.pop()
            qualified = reference.settled_qualified_name(self.metaclasses, module_name)
            if qualified in MODELLED_METACLASSES:
                modelled |= qualified == AUTO_REGISTER_META
                continue
            metaclass = self.metaclasses.declaration_for(reference, module_name)
            if metaclass is None:
                raise NeedsInput(
                    f"metaclass `{qualified or '.'.join(reference.source_parts)}` has no source "
                    "the scan or an installed module holds",
                    NeededInput.PROOF,
                )
            if metaclass.symbol in seen:
                continue
            seen.add(metaclass.symbol)
            if metaclass.symbol in MODELLED_METACLASSES:
                modelled |= metaclass.symbol == AUTO_REGISTER_META
                continue
            if not metaclass.registry_facts.keeps_bases:
                raise NeedsInput(
                    f"metaclass `{metaclass.symbol}` may change the bases it hands "
                    "`type.__new__`, or defines `mro`",
                    NeededInput.PROOF,
                )
            found_bindings |= metaclass.registry_facts.metaclass_bindings
            pending.extend(
                (base, metaclass.module_name) for base in metaclass.registry_facts.base_references
            )
        else:
            if pending:
                raise NeedsInput(
                    f"the metaclass ancestry is deeper than {self.MAX_METACLASS_ANCESTORS} classes",
                    NeededInput.PROOF,
                )
        if modelled:
            found_bindings |= MODELLED_BINDINGS
            keys = found_bindings.key_attributes | frozenset(
                key
                for indexed in ancestry
                if (key := indexed.autoregister_registry_key_attr_name) is not None
            )
            if not keys:
                found_bindings |= CreationBindings(computed=True)
            found_bindings |= CreationBindings(key_attributes=keys)
        return found_bindings


@dataclass(frozen=True)
class CompactClassFamilyIndexBuilder:
    projections: Sequence[CompactModuleClassHeader]

    def build(self) -> CompactClassFamilyIndex:
        settled_projections = repository_settled_projections(tuple(self.projections))
        raw_records = tuple(
            record
            for projection in settled_projections
            for record in projection.classes
        )
        records = tuple(
            UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
                raw_records,
                lambda record: record.symbol,
            ).values()
        )
        known_symbols = frozenset(record.symbol for record in records)
        unique_symbols_by_suffix = _unique_known_symbol_by_suffix(known_symbols)
        classes_by_symbol = {
            record.symbol: record.with_resolved_base_symbols(
                tuple(
                    resolved
                    for reference in record.base_references
                    if (
                        resolved := self._resolved_bound_symbol(
                            reference,
                            record.module_name,
                            known_symbols,
                            unique_symbols_by_suffix,
                        )
                    )
                    is not None
                )
            )
            for record in records
        }
        return CompactClassFamilyIndex(
            classes_by_symbol=RegisteringMetaclassAuthority(
                classes_by_symbol,
                CompactRepositoryPublicExposureIndex(tuple(self.projections)),
                _suffix_index=unique_symbols_by_suffix,
            ).resolved(),
            _has_duplicate_symbols=len(raw_records) != len(records),
        )

    @staticmethod
    def _resolved_bound_symbol(
        reference: CompactNominalReference,
        module_name: str,
        known_symbols: frozenset[str] | dict[str, CompactIndexedClass],
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex,
    ) -> str | None:
        candidate = ".".join(reference.resolved_parts)
        if candidate in known_symbols:
            return candidate
        return reference.known_symbol_beyond(
            candidate, module_name, known_symbols, unique_symbols_by_suffix
        )

    @staticmethod
    def _resolved_symbol(
        parts: tuple[str, ...],
        module_name: str,
        import_aliases_by_module_name: dict[str, dict[str, str]],
        known_symbols: frozenset[str] | dict[str, CompactIndexedClass],
        unique_symbols_by_name: dict[str, str],
        allow_unique_unqualified: bool = True,
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex | None = None,
    ) -> str | None:
        import_aliases = import_aliases_by_module_name.get(module_name, {})
        first, *rest = parts
        alias_target = import_aliases.get(first)
        if alias_target is not None:
            candidate = followed_reexports(
                ".".join((alias_target, *rest)) if rest else alias_target,
                import_aliases_by_module_name,
            )
            if candidate in known_symbols:
                return candidate
            unique_by_suffix = unique_symbols_by_suffix
            if unique_by_suffix is None:
                unique_by_suffix = _unique_known_symbol_by_suffix(
                    frozenset(known_symbols)
                )
            match = unique_by_suffix.root_relative_match(candidate)
            if match is not None:
                return match
        qualified = ".".join(parts)
        if qualified in known_symbols:
            return qualified
        module_local = ".".join((module_name, *parts))
        if module_local in known_symbols:
            return module_local
        if allow_unique_unqualified and len(parts) == 1:
            return unique_symbols_by_name.get(parts[0])
        return None


def followed_reexports(
    target: str, import_aliases_by_module_name: Mapping[str, Mapping[str, str]]
) -> str:
    """The name `target` finally binds, through every module that re-binds it.

    A package whose `__init__` says `from .a import X as Y` binds `Y` to
    `pkg.a.X`, so `pkg.Y` names `pkg.a.X`, as Python resolves it. The longest
    dotted prefix a module re-binds is replaced by what it binds, and the walk
    repeats until no prefix is re-bound or a name comes round again."""

    seen = {target}
    while True:
        segments = target.split(".")
        for end in range(len(segments), 1, -1):
            rebound = import_aliases_by_module_name.get(".".join(segments[: end - 1]), {}).get(
                segments[end - 1]
            )
            if rebound is not None:
                followed = ".".join((rebound, *segments[end:]))
                break
        else:
            return target
        if followed in seen:
            return target
        seen.add(followed)
        target = followed


@ScanCache.cached
def _compact_class_family_index_by_projection_ids() -> dict[
    tuple[int, ...],
    tuple[tuple[CompactModuleClassHeader, ...], frozenset[int], CompactClassFamilyIndex],
]:
    """Share built or projected ``CompactClassFamilyIndex`` instances across
    equivalent projection tuples during a ``ScanCache.scope()``."""

    return {}


def _remember_compact_class_family_index(
    projections: tuple[CompactModuleClassHeader, ...],
    index: CompactClassFamilyIndex,
) -> None:
    if ScanCache.active() is None:
        return
    key = tuple(id(projection) for projection in projections)
    _compact_class_family_index_by_projection_ids()[key] = (
        projections,
        frozenset(key),
        index,
    )


def build_compact_class_family_index(projections: Sequence[CompactModuleClassHeader]) -> CompactClassFamilyIndex:
    """The exact inheritance graph of AST-free per-module facts, built once per
    projection sequence that holds its views."""

    if ScanCache.active() is not None:
        cache = _compact_class_family_index_by_projection_ids()
        key = tuple(id(projection) for projection in projections)
        cached = cache.get(key)
        if cached is not None:
            return cached[2]
        if cache:
            best_diff = 65
            best_entry: tuple[tuple[CompactModuleClassHeader, ...], CompactClassFamilyIndex] | None = None
            key_len = len(key)
            for base_projs, base_ids, base_index in cache.values():
                if abs(key_len - len(base_projs)) > 64:
                    continue
                diff_count = sum(1 for item_id in key if item_id not in base_ids)
                if diff_count < best_diff:
                    best_diff = diff_count
                    best_entry = (base_projs, base_index)
                    if diff_count == 0:
                        break
            if best_entry is not None:
                base_projs, base_index = best_entry
                proj_tuple = (
                    projections if isinstance(projections, tuple) else tuple(projections)
                )
                return base_index.projected(base_projs, proj_tuple)
    built = SequenceViews.of(projections).view(_built_class_family_index)
    if ScanCache.active() is not None:
        proj_tuple = projections if isinstance(projections, tuple) else tuple(projections)
        _remember_compact_class_family_index(proj_tuple, built)
    return built


def _built_class_family_index(
    projections: Sequence[CompactModuleClassHeader],
) -> CompactClassFamilyIndex:
    return CompactClassFamilyIndexBuilder(projections).build()


@dataclass(frozen=True)
class CompactClassReferenceResolver:
    import_aliases_by_module_name: dict[str, dict[str, str]]
    known_symbols: dict[str, CompactIndexedClass]
    unique_symbols_by_name: dict[str, str]
    unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex

    @classmethod
    def from_index(cls, projections: Sequence[CompactModuleClassHeader], class_index: CompactClassFamilyIndex) -> CompactClassReferenceResolver:
        return cls(
            import_aliases_by_module_name={
                projection.module_name: dict(projection.import_aliases)
                for projection in projections
            },
            known_symbols=class_index.classes_by_symbol,
            unique_symbols_by_name=class_index.unique_symbols_by_name,
            unique_symbols_by_suffix=class_index.unique_symbols_by_suffix,
        )

    def symbol_for(
        self,
        *,
        module_name: str,
        reference_parts: tuple[str, ...],
        allow_unique_unqualified: bool = True,
    ) -> str | None:
        return CompactClassFamilyIndexBuilder._resolved_symbol(
            reference_parts,
            module_name,
            self.import_aliases_by_module_name,
            self.known_symbols,
            self.unique_symbols_by_name,
            allow_unique_unqualified,
            self.unique_symbols_by_suffix,
        )


@dataclass(frozen=True)
class CompactClassMappingKeyAuthority(MappingKeyResolverABC[CompactIndexedClass]):
    """Resolve class-member keys through the referenced declaration authority."""

    projections: Sequence[CompactModuleClassProjection]
    class_index: CompactClassFamilyIndex

    @cached_property
    def class_reference_resolver(self) -> CompactClassReferenceResolver:
        return CompactClassReferenceResolver.from_index(
            self.projections,
            self.class_index,
        )

    def _lexical_mapping_key(
        self,
        reference: LexicalValueReference,
        context: CompactIndexedClass,
    ) -> Hashable:
        if len(reference.attribute_path) != 1:
            raise ValueError("Mapping key reference requires one declaration member")
        symbol = self.class_reference_resolver.symbol_for(
            module_name=context.module_name,
            reference_parts=(reference.root_name,),
            allow_unique_unqualified=False,
        )
        declaration = None if symbol is None else self.class_index.class_for(symbol)
        if declaration is None:
            raise ValueError("Mapping key declaration semantics remain unproved")
        return declaration.enum_declaration.require_mapping_key(
            reference.attribute_path[0]
        )


def _single_assignment_target_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            return None
        target = node.targets[0]
    else:
        target = node.target
    if isinstance(target, ast.Name):
        return target.id
    return None
