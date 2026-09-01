"""Repository-wide class-family indexing helpers.

This module builds a lightweight cross-module view of declared classes and
their resolved inheritance edges. The index is intentionally conservative:
it resolves only import patterns and base expressions that can be recovered
reliably from the local AST.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import re
import tokenize
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import MISSING, dataclass, field, fields, replace
from enum import StrEnum
from functools import cached_property, lru_cache
from heapq import merge
from typing import Callable, ClassVar, Self, TypeAlias

from .annotation_semantics import CLASSVAR_ANNOTATION_AUTHORITY
from .ast_tools import (
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    CompactModuleIdentity,
    CollectedFamily,
    ParsedModule,
    PythonSourcePathPolicy,
    SourceModule,
    _walk_nodes,
    module_syntax_index,
    named_function_nodes,
)
from .collection_algebra import sorted_tuple
from .export_tools import PYTHON_PUBLIC_EXPORT_ASSIGNMENT
from .native_syntax import NativePythonSyntaxIndex
from .source_identity import resolved_source_path_text


@dataclass(frozen=True)
class ClassDeclaration:
    """Source-form-independent identity shared by repository class indexes."""

    symbol: str
    module_name: str
    qualname: str
    simple_name: str
    file_path: str
    line: int
    declared_base_names: tuple[str, ...]
    resolved_base_symbols: tuple[str, ...] = field(default=(), kw_only=True)

    def with_resolved_base_symbols(
        self,
        resolved_base_symbols: tuple[str, ...],
    ) -> Self:
        return replace(self, resolved_base_symbols=resolved_base_symbols)


@dataclass(frozen=True)
class IndexedClass(ClassDeclaration):
    node: ast.ClassDef

    @property
    def is_final(self) -> bool:
        return any(
            (isinstance(decorator, ast.Name) and decorator.id == "final")
            or (isinstance(decorator, ast.Attribute) and decorator.attr == "final")
            for decorator in self.node.decorator_list
        )

    @property
    def declares_autoregister_meta(self) -> bool:
        """Whether this declaration owns an AutoRegister-backed family."""

        return _declares_autoregister_meta(self.node)

    @classmethod
    def from_parsed_class(
        cls,
        parsed_module: ParsedModule,
        qualname: str,
        node: ast.ClassDef,
    ) -> "IndexedClass":
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
                    declared_base_name
                    := ClassSymbolResolutionAuthority.declared_base_name(base)
                )
                is not None
            ),
            resolved_base_symbols=(),
        )


@dataclass(frozen=True)
class CompactClassValueConstruction:
    """One class-owned construction of a nominal value declaration."""

    assigned_name: str
    constructor_name: str
    keyword_names: tuple[str, ...]
    line: int


@dataclass(frozen=True)
class CompactClassHeader(ClassDeclaration):
    """Class-index surface sufficient for inheritance reconstruction."""

    base_reference_parts: tuple[tuple[str, ...], ...]
    base_references_are_complete: bool = False
    is_final: bool = False

    @property
    def base_resolution_is_complete(self) -> bool:
        """Return whether every domain-bearing base resolves in the compact graph."""

        return self.base_references_are_complete and len(
            self.resolved_base_symbols
        ) == declared_nominal_base_count(self)


@dataclass(frozen=True)
class ClassHeaderSourceSpan:
    """Exact source span and reconstruction safety for one class header."""

    node: ast.ClassDef
    source_lines: tuple[str, ...]

    @classmethod
    def from_source(cls, node: ast.ClassDef, source: str) -> "ClassHeaderSourceSpan":
        return cls(node=node, source_lines=tuple(source.splitlines(keepends=True)))

    @property
    def start_line(self) -> int:
        return self.node.lineno

    @property
    def end_line(self) -> int:
        return (
            min(self.statement_start_line(statement) for statement in self.node.body)
            - 1
        )

    @staticmethod
    def statement_start_line(statement: ast.stmt) -> int:
        if not isinstance(
            statement,
            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        ):
            return statement.lineno
        decorator_lines = tuple(
            decorator.lineno
            for decorator in statement.decorator_list
            if decorator.lineno
        )
        return min((*decorator_lines, statement.lineno))

    @property
    def source(self) -> str:
        return "".join(self.source_lines[self.start_line - 1 : self.end_line])

    @cached_property
    def contains_comment(self) -> bool:
        if "#" not in self.source:
            return False
        try:
            tokens = tokenize.generate_tokens(io.StringIO(self.source).readline)
            return any(token.type == tokenize.COMMENT for token in tokens)
        except tokenize.TokenError:
            return True

    @property
    def is_reconstructible(self) -> bool:
        return not self.contains_comment


@dataclass(frozen=True)
class CompactIndexedClass(CompactClassHeader):
    """AST-free class declaration used to reconstruct inheritance globally."""

    direct_assignment_expressions: tuple[tuple[str, str | None], ...] = ()
    direct_assignment_lines: tuple[tuple[str, int], ...] = ()
    direct_value_constructions: tuple[CompactClassValueConstruction, ...] = ()
    direct_constant_string_assignments: tuple[tuple[str, str], ...] = ()
    direct_non_none_assignment_names: tuple[str, ...] = ()
    metaclass_names: tuple[str, ...] = ()
    class_keyword_names: tuple[str, ...] = ()
    class_decorators_are_promotion_safe: bool = True
    class_header_is_reconstructible: bool = True
    keyed_family_key_type_name: str | None = None
    end_line: int | None = None
    method_names: tuple[str, ...] = ()
    abstract_method_names: tuple[str, ...] = ()
    is_abstract: bool = False
    is_dataclass: bool = False
    declares_autoregister_meta: bool = False
    is_registration_authority: bool = False
    autoregister_registry_key_attr_name: str | None = None
    autoregister_key_extractor_name: str | None = None
    autoregister_registry_projection_names: tuple[str, ...] = ()
    keyed_registry_lookup_method_names: tuple[str, ...] = ()
    keyed_registry_reverse_lookup_method_names: tuple[str, ...] = ()
    predicate_selected_methods: tuple[tuple[int, str, str, str], ...] = ()

    @property
    def assignments_by_name(self) -> dict[str, str | None]:
        return dict(self.direct_assignment_expressions)

    @property
    def assignment_lines_by_name(self) -> dict[str, int]:
        lines: dict[str, int] = {}
        for name, line in self.direct_assignment_lines:
            lines.setdefault(name, line)
        return lines


def has_complete_concrete_mro_composite(
    direct_child_symbols: tuple[str, ...],
    concrete_descendants: tuple[IndexedClass, ...] | tuple[CompactIndexedClass, ...],
) -> bool:
    """Return whether one descendant composes every concrete root branch."""

    concrete_descendant_symbols = {
        descendant.symbol for descendant in concrete_descendants
    }
    concrete_branch_symbols = concrete_descendant_symbols.intersection(
        direct_child_symbols
    )
    if len(concrete_branch_symbols) < 2:
        return False
    return any(
        concrete_branch_symbols.issubset(descendant.resolved_base_symbols)
        for descendant in concrete_descendants
    )


class CompactPublicNameExposure(StrEnum):
    """Proof result for one name on a module's declared public surface."""

    PUBLIC = "public"
    PRIVATE = "private"
    UNRESOLVED = "unresolved"


class CompactModulePublicExportContract(ABC):
    """Representation-independent declaration of one module's export policy."""

    @abstractmethod
    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        raise NotImplementedError


@dataclass(frozen=True)
class CompactImplicitPublicExportContract(CompactModulePublicExportContract):
    """Python's implicit convention: leading-underscore bindings are private."""

    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        return (
            CompactPublicNameExposure.PRIVATE
            if name.startswith("_")
            else CompactPublicNameExposure.PUBLIC
        )


@dataclass(frozen=True)
class CompactExplicitPublicExportContract(CompactModulePublicExportContract):
    """One statically complete ``__all__`` membership declaration."""

    exported_names: tuple[str, ...]

    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        return (
            CompactPublicNameExposure.PUBLIC
            if name in self.exported_names
            else CompactPublicNameExposure.PRIVATE
        )


@dataclass(frozen=True)
class CompactUnresolvedPublicExportContract(CompactModulePublicExportContract):
    """A dynamic or otherwise incomplete ``__all__`` declaration."""

    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        del name
        return CompactPublicNameExposure.UNRESOLVED


@dataclass(frozen=True)
class CompactModuleStarImportOrigin:
    """One module-scope star import and its optional resolved module origin."""

    module_name: str | None

    @property
    def is_resolved(self) -> bool:
        return self.module_name is not None


@dataclass(frozen=True)
class CompactModuleClassHeader(CompactModuleIdentity):
    """Module namespace and class surface required by the compact family index."""

    import_aliases: tuple[tuple[str, str], ...]
    public_export_contract: CompactModulePublicExportContract
    star_import_origins: tuple[CompactModuleStarImportOrigin, ...]
    classes: tuple[CompactIndexedClass, ...]


@dataclass(frozen=True)
class CompactClassSyntaxFacets:
    """Class-family views derived together from the shared module traversal."""

    closed_axis_branch_functions: tuple["CompactClosedAxisBranchFunction", ...] = ()
    exact_type_guards: tuple["CompactExactTypeGuard", ...] = ()
    autoregister_function_references: tuple[
        "CompactAutoRegisterFunctionReference", ...
    ] = ()
    autoregister_reference_index: "CompactAutoRegisterReferenceIndex | None" = None


@dataclass(frozen=True)
class CompactModuleClassProjection(
    CompactClassSyntaxFacets,
    CompactModuleClassHeader,
):
    """One module's class declarations and import aliases, without its AST."""

    sorted_key_calls: tuple["CompactSortedKeyCall", ...] = ()
    keyed_table_axes: tuple["CompactKeyedTableAxis", ...] = ()
    manual_selector_axes: tuple["CompactManualSelectorAxis", ...] = ()
    top_level_definitions: tuple[tuple[str, int], ...] = ()
    repeated_keyed_family_roots: tuple["CompactRepeatedKeyedFamilyRoot", ...] = ()
    manual_subclass_roster_roots: tuple["CompactManualSubclassRosterRoot", ...] = ()
    latent_rosters: tuple["LatentRosterObservation", ...] = ()
    named_projection_surfaces: tuple["CompactNamedProjectionSurface", ...] = ()
    manual_family_rosters: tuple["CompactManualFamilyRosterObservation", ...] = ()
    nominal_wrapper_authorities: tuple["CompactNominalWrapperAuthority", ...] = ()
    pass_through_nominal_wrappers: tuple["CompactPassThroughNominalWrapper", ...] = ()
    class_methods: tuple["CompactClassMethod", ...] = ()

    def header_core(self) -> "CompactModuleClassProjection":
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
    ) -> dict[str, object]:
        preserved_names = frozenset(
            dataclass_field.name for dataclass_field in fields(preserved_type)
        )
        values: dict[str, object] = {}
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


@dataclass(frozen=True)
class CompactNominalWrapperAuthority:
    """Reusable authority member names in repository AST walk order."""

    file_path: str
    class_name: str
    line: int
    method_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactPassThroughNominalWrapper:
    """A locally proven forwarding shell awaiting its global authority join."""

    file_path: str
    class_name: str
    line: int
    delegate_field_name: str
    delegate_authority_name: str
    forwarded_member_names: tuple[str, ...]


CompactMethodSemanticCoordinate: TypeAlias = tuple[tuple[str, ...], str, str]


@dataclass(frozen=True)
class CompactClassMethodSemanticProfile:
    """Semantic body profile derived lazily from one compact class method."""

    skeleton: tuple[str, ...]
    coordinates: tuple[CompactMethodSemanticCoordinate, ...]


@dataclass(frozen=True)
class MethodPromotionInspection:
    """Current method syntax and lexical bindings relevant to promotion safety."""

    method: ast.FunctionDef | ast.AsyncFunctionDef
    module_bound_names: frozenset[str]
    class_bound_names: frozenset[str]
    source_lines: tuple[str, ...]


@dataclass(frozen=True)
class ClassMethodReceiverRequirements:
    """Receiver members required directly or through local aliases."""

    member_names: frozenset[str]

    @classmethod
    def from_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> "ClassMethodReceiverRequirements":
        receiver_name = cls.receiver_name(method)
        if receiver_name is None:
            return cls(frozenset())
        aliases = cls.receiver_aliases(method, receiver_name)
        return cls(
            frozenset(
                node.attr
                for node in ast.walk(method)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            )
        )

    @staticmethod
    def receiver_aliases(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        receiver_name: str,
    ) -> frozenset[str]:
        alias_targets_by_source: dict[str, set[str]] = defaultdict(set)
        for node in ast.walk(method):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                alias_targets_by_source[node.value.id].update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
            elif (
                isinstance(node, ast.AnnAssign | ast.NamedExpr)
                and isinstance(node.target, ast.Name)
                and isinstance(node.value, ast.Name)
            ):
                alias_targets_by_source[node.value.id].add(node.target.id)

        aliases = {receiver_name}
        pending = deque((receiver_name,))
        while pending:
            source_name = pending.popleft()
            for target_name in alias_targets_by_source[source_name]:
                if target_name in aliases:
                    continue
                aliases.add(target_name)
                pending.append(target_name)
        return frozenset(aliases)

    @staticmethod
    def receiver_name(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str | None:
        if any(
            isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
            for decorator in method.decorator_list
        ):
            return None
        positional_parameters = (*method.args.posonlyargs, *method.args.args)
        return positional_parameters[0].arg if positional_parameters else None


MethodPromotionHazardPredicate: TypeAlias = Callable[[MethodPromotionInspection], bool]


def _has_super_reference(inspection: MethodPromotionInspection) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == "super"
        for node in ast.walk(inspection.method)
    )


def _has_class_cell_reference(inspection: MethodPromotionInspection) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == "__class__"
        for node in ast.walk(inspection.method)
    )


def _has_private_name_mangling(inspection: MethodPromotionInspection) -> bool:
    names = (
        inspection.method.name,
        *(
            node.id
            for node in ast.walk(inspection.method)
            if isinstance(node, ast.Name)
        ),
        *(
            node.attr
            for node in ast.walk(inspection.method)
            if isinstance(node, ast.Attribute)
        ),
    )
    return any(name.startswith("__") and not name.endswith("__") for name in names)


_PROMOTABLE_METHOD_DECORATOR_NAMES = frozenset(
    ("classmethod", "property", "staticmethod")
)


class ClassMethodPromotionSafeDecorator(StrEnum):
    """Class decorators proven not to depend on direct method ownership."""

    DATACLASS = ("dataclass", frozenset(("dataclasses",)))
    FINAL = ("final", frozenset(("typing", "typing_extensions")))

    import_module_names: frozenset[str]

    def __new__(
        cls,
        value: str,
        import_module_names: frozenset[str],
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.import_module_names = import_module_names
        return member

    def is_proven_reference(self, module: ast.Module, decorator: ast.expr) -> bool:
        reference = decorator.func if isinstance(decorator, ast.Call) else decorator
        parts = ATTRIBUTE_CHAIN_AUTHORITY.project(reference)
        if parts is None or parts[-1] != self.value:
            return False
        return any(
            self._import_proves_parts(statement, parts)
            for statement in module.body
            if isinstance(statement, ast.Import | ast.ImportFrom)
        )

    def _import_proves_parts(
        self,
        statement: ast.Import | ast.ImportFrom,
        parts: tuple[str, ...],
    ) -> bool:
        if isinstance(statement, ast.ImportFrom):
            return (
                statement.level == 0
                and statement.module in self.import_module_names
                and len(parts) == 1
                and any(
                    alias.name == self.value
                    and (alias.asname or alias.name) == parts[0]
                    for alias in statement.names
                )
            )
        return len(parts) == 2 and any(
            alias.name in self.import_module_names
            and (alias.asname or alias.name) == parts[0]
            for alias in statement.names
        )


def _has_custom_method_decorator(inspection: MethodPromotionInspection) -> bool:
    shadowed_names = inspection.module_bound_names | inspection.class_bound_names
    return any(
        not isinstance(decorator, ast.Name)
        or decorator.id not in _PROMOTABLE_METHOD_DECORATOR_NAMES
        or decorator.id in shadowed_names
        for decorator in inspection.method.decorator_list
    )


def _is_direct_namespace_sensitive(inspection: MethodPromotionInspection) -> bool:
    name = inspection.method.name
    return name.startswith("__") and name.endswith("__")


def _has_evaluated_default(inspection: MethodPromotionInspection) -> bool:
    return bool(
        inspection.method.args.defaults
        or any(default is not None for default in inspection.method.args.kw_defaults)
    )


def _method_annotation_nodes(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.expr, ...]:
    arguments = (
        *method.args.posonlyargs,
        *method.args.args,
        *method.args.kwonlyargs,
        *((method.args.vararg,) if method.args.vararg is not None else ()),
        *((method.args.kwarg,) if method.args.kwarg is not None else ()),
    )
    return (
        *(
            argument.annotation
            for argument in arguments
            if argument.annotation is not None
        ),
        *((method.returns,) if method.returns is not None else ()),
    )


def _has_class_local_annotation_reference(
    inspection: MethodPromotionInspection,
) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id in inspection.class_bound_names
        for annotation in _method_annotation_nodes(inspection.method)
        for node in ast.walk(annotation)
    )


def _has_attached_leading_comment(inspection: MethodPromotionInspection) -> bool:
    decorator_lines = tuple(
        decorator.lineno
        for decorator in inspection.method.decorator_list
        if decorator.lineno
    )
    start_line = min((*decorator_lines, inspection.method.lineno))
    if start_line <= 1 or start_line > len(inspection.source_lines):
        return False
    method_line = inspection.source_lines[start_line - 1]
    preceding_line = inspection.source_lines[start_line - 2]
    method_indent = method_line[: len(method_line) - len(method_line.lstrip())]
    preceding_indent = preceding_line[
        : len(preceding_line) - len(preceding_line.lstrip())
    ]
    return preceding_indent == method_indent and preceding_line.lstrip().startswith("#")


class MethodPromotionHazard(StrEnum):
    """One promotion hazard carrying its own syntax recognition behavior."""

    SUPER_REFERENCE = ("super_reference", _has_super_reference)
    CLASS_CELL_REFERENCE = ("class_cell_reference", _has_class_cell_reference)
    PRIVATE_NAME_MANGLING = ("private_name_mangling", _has_private_name_mangling)
    CUSTOM_METHOD_DECORATOR = (
        "custom_method_decorator",
        _has_custom_method_decorator,
    )
    DIRECT_NAMESPACE_SENSITIVE_METHOD = (
        "direct_namespace_sensitive_method",
        _is_direct_namespace_sensitive,
    )
    EVALUATED_DEFAULT = ("evaluated_default", _has_evaluated_default)
    CLASS_LOCAL_ANNOTATION_REFERENCE = (
        "class_local_annotation_reference",
        _has_class_local_annotation_reference,
    )
    ATTACHED_LEADING_COMMENT = (
        "attached_leading_comment",
        _has_attached_leading_comment,
    )

    predicate: MethodPromotionHazardPredicate

    def __new__(
        cls,
        value: str,
        predicate: MethodPromotionHazardPredicate,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.predicate = predicate
        return member

    def is_present(self, inspection: MethodPromotionInspection) -> bool:
        return self.predicate(inspection)


@dataclass(frozen=True)
class ClassMethodPromotionSafetyProfile:
    """Complete promotion hazards derived from one current method declaration."""

    hazards: tuple[MethodPromotionHazard, ...]

    @classmethod
    def from_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        module_bound_names: frozenset[str],
        class_bound_names: frozenset[str] = frozenset(),
        *,
        source_lines: tuple[str, ...],
    ) -> "ClassMethodPromotionSafetyProfile":
        inspection = MethodPromotionInspection(
            method,
            module_bound_names,
            class_bound_names,
            source_lines,
        )
        return cls(
            tuple(
                hazard
                for hazard in MethodPromotionHazard
                if hazard.is_present(inspection)
            )
        )


@dataclass(frozen=True)
class CompactClassMethod:
    """AST-free method fact shared by exact promotion and anti-unification."""

    class_symbol: str
    method_name: str
    line: int
    line_count: int
    body_statement_count: int
    statement_sources: tuple[str, ...]
    exact_source_digest: str | None
    promotion_hazards: tuple[MethodPromotionHazard, ...]
    receiver_member_names: frozenset[str]

    @property
    def statement_count(self) -> int:
        return self.body_statement_count

    @cached_property
    def semantic_profile(self) -> CompactClassMethodSemanticProfile:
        statements = _compact_class_method_statements_from_sources(
            self.statement_sources
        )
        coordinates = tuple(
            coordinate
            for statement_index, statement in enumerate(statements)
            for coordinate in _compact_class_method_coordinates(
                statement,
                ("body", statement_index),
            )
        )
        return CompactClassMethodSemanticProfile(
            skeleton=tuple(
                _compact_class_method_statement_skeleton(statement)
                for statement in statements
            ),
            coordinates=coordinates,
        )


@dataclass(frozen=True)
class CompactManualSubclassRegistrationSite:
    registry_name: str
    guard_summary: str | None
    selector_attr_name: str | None
    requires_concrete_subclass: bool


@dataclass(frozen=True)
class CompactManualSubclassRosterRoot:
    class_symbol: str
    init_subclass_line: int
    registration_sites: tuple[CompactManualSubclassRegistrationSite, ...]
    consumer_locations: tuple[tuple[str, int, str, str], ...]


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
    ) -> tuple["LatentRosterObservation", ...]:
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
    ) -> tuple["LatentRosterObservation", ...]:
        target_value: tuple[ast.AST, ast.AST] | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target_value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target_value = statement.target, statement.value
        if target_value is None or not isinstance(target_value[0], ast.Name):
            return ()
        target, value = target_value
        roster_name = (
            f"{roster_prefix}.{target.id}" if roster_prefix is not None else target.id
        )
        return cls.from_value(
            parsed_module,
            statement,
            roster_name=roster_name,
            value=value,
        )

    @classmethod
    def from_mutation_statement(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
    ) -> tuple["LatentRosterObservation", ...]:
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
    ) -> tuple["LatentRosterObservation", ...]:
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
    ) -> "LatentRosterObservation":
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

    def for_mutation(self, mutation_name: str) -> "LatentRosterObservation":
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
class CompactNamedProjectionSurface:
    """Top-level tuple/list/dict references used by registry projections."""

    file_path: str
    surface_name: str
    line: int
    sequence_references: tuple[tuple[str, str | None], ...] = ()
    dict_key_references: tuple[tuple[str, str | None], ...] = ()
    dict_value_references: tuple[tuple[str, str | None], ...] = ()


@dataclass(frozen=True)
class CompactManualFamilyRosterObservation:
    """Top-level local class roster before its shared base is resolved."""

    file_path: str
    line: int
    owner_name: str
    member_names: tuple[str, ...]
    constructor_style: str


@dataclass(frozen=True)
class CompactAutoRegisterFunctionReference:
    """Sparse AST-free function facts used by AutoRegister rent analysis."""

    qualname: str
    referenced_symbols: tuple[str, ...]
    calls_autoregister_meta: bool
    receiver_attribute_refs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CompactAutoRegisterReferenceIndex:
    """Interned receiver-attribute edges for registry consumer resolution."""

    function_qualnames: tuple[str, ...]
    receiver_names: tuple[str, ...]
    attribute_names: tuple[str, ...]
    encoded_edges: str


class RegistryLookupStyle(StrEnum):
    TRY_EXCEPT = "try_except"
    MEMBERSHIP_GUARD = "membership_guard"


class SelectionGuardKind(StrEnum):
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"
    NOT_EXACTLY_ONE = "not_exactly_one"

    @classmethod
    def from_node(
        cls,
        node: ast.AST,
        match_name: str,
    ) -> "SelectionGuardKind | None":
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Name)
            and node.operand.id == match_name
        ):
            return cls.EMPTY
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "len"
            and len(node.left.args) == 1
            and isinstance(node.left.args[0], ast.Name)
            and node.left.args[0].id == match_name
            and isinstance(node.comparators[0], ast.Constant)
            and isinstance(node.comparators[0].value, int)
        ):
            return None
        operator = node.ops[0]
        comparator = node.comparators[0].value
        if isinstance(operator, ast.NotEq) and comparator == 1:
            return cls.NOT_EXACTLY_ONE
        if isinstance(operator, ast.Gt) and comparator == 1:
            return cls.AMBIGUOUS
        if isinstance(operator, ast.Eq) and comparator == 0:
            return cls.EMPTY
        return None


@dataclass(frozen=True)
class ClsRegistryMembership:
    operator_type: type[ast.cmpop]
    key_expr: str

    @classmethod
    def from_node(cls, node: ast.AST) -> "ClsRegistryMembership | None":
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
    ) -> "RegistryLookupShape | None":
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


@dataclass(frozen=True)
class CompactRepeatedKeyedFamilyRoot:
    file_path: str
    line: int
    class_name: str
    family_base_name: str
    registry_key_attr_name: str
    lookup_method_name: str
    lookup_style: RegistryLookupStyle
    error_type_name: str | None
    abstract_hook_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactSortedKeyCall:
    """One sorted call and the semantic attributes used by its key."""

    file_path: str
    line: int
    registry_owner_names: tuple[str, ...]
    key_attribute_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactKeyedTableAxis:
    """AST-free module-level dictionary keyed by one enum-like axis."""

    file_path: str
    line: int
    table_name: str
    key_type_name: str
    case_names: tuple[str, ...]
    value_shape_name: str | None


@dataclass(frozen=True)
class CompactClosedAxisBranchFact:
    key_type_name: str
    branch_site_count: int
    case_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactClosedAxisBranchFunction:
    file_path: str
    line: int
    qualname: str
    axes: tuple[CompactClosedAxisBranchFact, ...]


@dataclass(frozen=True)
class CompactManualSelectorAxis:
    file_path: str
    line: int
    family_name: str
    selector_method_name: str
    key_type_name: str
    case_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactExactTypeGuard:
    file_path: str
    line: int
    qualname: str
    subject_expression: str
    type_reference_expression: str
    type_reference_parts: tuple[str, ...]
    matches_exact_type_when_true: bool
    expression: str

    @property
    def structural_membership_expression(self) -> str:
        membership = (
            f"isinstance({self.subject_expression}, {self.type_reference_expression})"
        )
        return membership if self.matches_exact_type_when_true else f"not {membership}"


@dataclass(frozen=True)
class CompactClassFamilyIndex:
    """Repository inheritance graph reconstructed from compact declarations."""

    classes_by_symbol: dict[str, CompactIndexedClass]
    symbols_by_simple_name: dict[str, tuple[str, ...]]
    symbols_by_file_and_qualname: dict[tuple[str, str], str]
    children_by_symbol: dict[str, tuple[str, ...]]
    ancestors_by_symbol: dict[str, tuple[str, ...]]
    descendants_by_symbol: dict[str, tuple[str, ...]]

    def class_for(self, symbol: str) -> CompactIndexedClass | None:
        return self.classes_by_symbol.get(symbol)

    def symbol_for(self, *, file_path: str, qualname: str) -> str | None:
        return self.symbols_by_file_and_qualname.get((file_path, qualname))

    def descendant_symbols(self, base_symbol: str) -> tuple[str, ...]:
        return self.descendants_by_symbol.get(base_symbol, ())

    def ancestor_symbols(self, class_symbol: str) -> tuple[str, ...]:
        return self.ancestors_by_symbol.get(class_symbol, ())


ClosedLeafMethodAuthorityPredicate: TypeAlias = Callable[
    ["ClosedLeafMethodAuthorityProof"],
    bool,
]
CLASS_METHOD_OWNERSHIP_HOOK_NAMES = frozenset(("__init_subclass__",))


def _has_too_few_closed_leaf_participants(
    proof: "ClosedLeafMethodAuthorityProof",
) -> bool:
    return len(proof.participant_symbol_set) < 2


def _has_ambiguous_direct_method_authority(
    proof: "ClosedLeafMethodAuthorityProof",
) -> bool:
    return frozenset(proof.common_direct_base_symbols) != frozenset(
        (proof.authority_symbol,)
    )


def _has_ambiguous_declared_method_authority(
    proof: "ClosedLeafMethodAuthorityProof",
) -> bool:
    return proof.common_declared_nominal_base_simple_names != frozenset(
        (proof.authority_simple_name,)
    )


def _has_incomplete_direct_method_family(
    proof: "ClosedLeafMethodAuthorityProof",
) -> bool:
    return (
        frozenset(proof.authority_direct_child_symbols) != proof.participant_symbol_set
    )


def _has_non_leaf_method_participant(
    proof: "ClosedLeafMethodAuthorityProof",
) -> bool:
    return bool(proof.non_leaf_participant_symbols)


def _has_incomplete_method_base_resolution(
    proof: "ClosedLeafMethodAuthorityProof",
) -> bool:
    return bool(proof.incompletely_resolved_symbols)


def _crosses_method_ownership_sensitive_declaration(
    proof: "ClosedLeafMethodAuthorityProof",
) -> bool:
    return bool(proof.method_ownership_sensitive_symbols)


def _has_existing_authority_method_member(
    proof: "ClosedLeafMethodAuthorityProof",
) -> bool:
    return bool(
        proof.promoted_method_name_set & proof.authority_lineage_member_name_set
    )


def _has_competing_ancestor_method_member(
    proof: "ClosedLeafMethodAuthorityProof",
) -> bool:
    return bool(
        proof.promoted_method_name_set
        & frozenset(proof.competing_ancestor_member_names)
    )


def _has_undeclared_promoted_receiver_member(
    proof: "ClosedLeafMethodAuthorityProof",
) -> bool:
    return bool(
        frozenset(proof.receiver_member_names)
        - (proof.authority_lineage_member_name_set | proof.promoted_method_name_set)
    )


class ClosedLeafMethodAuthorityViolation(StrEnum):
    """One failed proof obligation for promoting leaf methods to an ancestor."""

    TOO_FEW_PARTICIPANTS = (
        "too_few_participants",
        "the authority relation requires at least two participating leaves",
        _has_too_few_closed_leaf_participants,
    )
    AMBIGUOUS_DIRECT_AUTHORITY = (
        "ambiguous_direct_authority",
        "the participants do not have exactly one resolved direct authority",
        _has_ambiguous_direct_method_authority,
    )
    AMBIGUOUS_DECLARED_AUTHORITY = (
        "ambiguous_declared_authority",
        "the participants do not have exactly one declared nominal base",
        _has_ambiguous_declared_method_authority,
    )
    INCOMPLETE_DIRECT_FAMILY = (
        "incomplete_direct_family",
        "the participants are not the complete direct-child family",
        _has_incomplete_direct_method_family,
    )
    NON_LEAF_PARTICIPANT = (
        "non_leaf_participant",
        "at least one participant still owns a descendant branch",
        _has_non_leaf_method_participant,
    )
    INCOMPLETE_BASE_RESOLUTION = (
        "incomplete_base_resolution",
        "a relevant nominal base cannot be resolved from the repository graph",
        _has_incomplete_method_base_resolution,
    )
    METHOD_OWNERSHIP_SENSITIVE_DECLARATION = (
        "method_ownership_sensitive_declaration",
        "a class decorator or metaclass boundary can observe direct method ownership",
        _crosses_method_ownership_sensitive_declaration,
    )
    EXISTING_AUTHORITY_MEMBER = (
        "existing_authority_member",
        "the authority lineage already binds a promoted member name",
        _has_existing_authority_method_member,
    )
    COMPETING_ANCESTOR_MEMBER = (
        "competing_ancestor_member",
        "another participant ancestor binds a promoted member name",
        _has_competing_ancestor_method_member,
    )
    UNDECLARED_RECEIVER_MEMBER = (
        "undeclared_receiver_member",
        "a promoted method requires a receiver member outside the authority contract",
        _has_undeclared_promoted_receiver_member,
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

    def is_violated_by(self, proof: "ClosedLeafMethodAuthorityProof") -> bool:
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


@dataclass(frozen=True)
class ClassFamilyIndex:
    classes_by_symbol: dict[str, IndexedClass]
    symbols_by_simple_name: dict[str, tuple[str, ...]]
    symbols_by_file_and_qualname: dict[tuple[str, str], str]
    children_by_symbol: dict[str, tuple[str, ...]]
    ancestors_by_symbol: dict[str, tuple[str, ...]]
    descendants_by_symbol: dict[str, tuple[str, ...]]

    @cached_property
    def known_symbols(self) -> frozenset[str]:
        """Repository class symbols shared by every module resolver."""

        return frozenset(self.classes_by_symbol)

    @cached_property
    def unique_symbols_by_name(self) -> dict[str, str]:
        """Unambiguous simple-name projection shared across module resolvers."""

        return {
            simple_name: symbols[0]
            for simple_name, symbols in self.symbols_by_simple_name.items()
            if len(symbols) == 1
        }

    def class_for(self, symbol: str) -> IndexedClass | None:
        return self.classes_by_symbol.get(symbol)

    def symbol_for(self, *, file_path: str, qualname: str) -> str | None:
        return self.symbols_by_file_and_qualname.get((file_path, qualname))

    def descendant_symbols(self, base_symbol: str) -> tuple[str, ...]:
        return self.descendants_by_symbol.get(base_symbol, ())

    def ancestor_symbols(self, class_symbol: str) -> tuple[str, ...]:
        return self.ancestors_by_symbol.get(class_symbol, ())

    def class_records_excluding_files(
        self,
        file_paths: frozenset[str],
    ) -> tuple[IndexedClass, ...]:
        return tuple(
            indexed_class
            for indexed_class in self.classes_by_symbol.values()
            if resolved_source_path_text(indexed_class.file_path) not in file_paths
        )


def _iter_class_defs(
    statements: list[ast.stmt],
    *,
    parent_qualname: str | None = None,
) -> tuple[tuple[str, ast.ClassDef], ...]:
    classes: list[tuple[str, ast.ClassDef]] = []
    for statement in statements:
        if not isinstance(statement, ast.ClassDef):
            continue
        qualname = (
            statement.name
            if parent_qualname is None
            else f"{parent_qualname}.{statement.name}"
        )
        classes.append((qualname, statement))
        classes.extend(_iter_class_defs(list(statement.body), parent_qualname=qualname))
    return tuple(classes)


@dataclass(frozen=True)
class AttributeChainAuthority:
    def project(self, node: ast.AST) -> tuple[str, ...] | None:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            parent = self.project(node.value)
            if parent is None:
                return None
            return (*parent, node.attr)
        return None


ATTRIBUTE_CHAIN_AUTHORITY = AttributeChainAuthority()


class _UniqueKnownSymbolSuffixIndex:
    """Resolve only requested suffixes from terminal-name candidate buckets."""

    __slots__ = ("_matches_by_suffix", "_symbols_by_terminal_name")

    def __init__(self, known_symbols: frozenset[str]) -> None:
        symbols_by_terminal_name: dict[str, list[str]] = defaultdict(list)
        for symbol in known_symbols:
            symbols_by_terminal_name[symbol.rsplit(".", 1)[-1]].append(symbol)
        self._symbols_by_terminal_name = {
            terminal_name: tuple(symbols)
            for terminal_name, symbols in symbols_by_terminal_name.items()
        }
        self._matches_by_suffix: dict[str, str | None] = {}

    @classmethod
    def from_terminal_buckets(
        cls,
        symbols_by_terminal_name: dict[str, tuple[str, ...]],
    ) -> "_UniqueKnownSymbolSuffixIndex":
        index = cls.__new__(cls)
        index._symbols_by_terminal_name = symbols_by_terminal_name
        index._matches_by_suffix = {}
        return index

    def get(self, suffix: str) -> str | None:
        if suffix in self._matches_by_suffix:
            return self._matches_by_suffix[suffix]
        qualified_suffix = f".{suffix}"
        match: str | None = None
        for symbol in self._symbols_by_terminal_name.get(suffix.rsplit(".", 1)[-1], ()):
            if symbol != suffix and not symbol.endswith(qualified_suffix):
                continue
            if match is not None and match != symbol:
                match = None
                break
            match = symbol
        self._matches_by_suffix[suffix] = match
        return match


@lru_cache(maxsize=8)
def _unique_known_symbol_by_suffix(
    known_symbols: frozenset[str],
) -> _UniqueKnownSymbolSuffixIndex:
    """Share a bounded lazy suffix resolver for one repository symbol set."""

    return _UniqueKnownSymbolSuffixIndex(known_symbols)


def _resolve_relative_module(
    parsed_module: ParsedModule,
    *,
    imported_module: str | None,
    level: int,
) -> str | None:
    if level == 0:
        return imported_module
    package_parts = parsed_module.module_name.split(".")
    if not parsed_module.is_package_init:
        package_parts = package_parts[:-1]
    if level > 1:
        if level - 1 > len(package_parts):
            return None
        package_parts = package_parts[: len(package_parts) - (level - 1)]
    if imported_module:
        return ".".join((*package_parts, *imported_module.split(".")))
    return ".".join(package_parts)


class _ModuleScopeNameReferenceCollector(ast.NodeVisitor):
    """Collect one module binding's syntax without entering child namespaces."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.references: list[ast.Name] = []

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name:
            self.references.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _module_scope_name_references(
    module: ast.Module,
    name: str,
) -> tuple[ast.Name, ...]:
    collector = _ModuleScopeNameReferenceCollector(name)
    collector.visit(module)
    return tuple(collector.references)


def _public_export_assignment(
    statement: ast.stmt,
) -> tuple[ast.Name, ast.expr] | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == PYTHON_PUBLIC_EXPORT_ASSIGNMENT
    ):
        return statement.targets[0], statement.value
    if (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == PYTHON_PUBLIC_EXPORT_ASSIGNMENT
        and statement.value is not None
    ):
        return statement.target, statement.value
    return None


def _literal_public_export_names(value: ast.expr) -> tuple[str, ...] | None:
    if not isinstance(value, ast.List | ast.Tuple | ast.Set):
        return None
    if any(
        not isinstance(element, ast.Constant) or not isinstance(element.value, str)
        for element in value.elts
    ):
        return None
    return sorted_tuple({element.value for element in value.elts})


def _compact_module_public_export_contract(
    parsed_module: ParsedModule,
) -> CompactModulePublicExportContract:
    module = parsed_module.module
    if PYTHON_PUBLIC_EXPORT_ASSIGNMENT not in (
        LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.body)
    ):
        return CompactImplicitPublicExportContract()
    assignments = tuple(
        assignment
        for statement in module.body
        if (assignment := _public_export_assignment(statement)) is not None
    )
    if len(assignments) != 1:
        return CompactUnresolvedPublicExportContract()
    target, value = assignments[0]
    references = _module_scope_name_references(
        module,
        PYTHON_PUBLIC_EXPORT_ASSIGNMENT,
    )
    if references != (target,):
        return CompactUnresolvedPublicExportContract()
    exported_names = _literal_public_export_names(value)
    if exported_names is None:
        return CompactUnresolvedPublicExportContract()
    return CompactExplicitPublicExportContract(exported_names)


@lru_cache(maxsize=None)
def _module_import_aliases(parsed_module: ParsedModule) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for statement in parsed_module.module.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                aliases[local_name] = (
                    alias.name if alias.asname else alias.name.split(".", 1)[0]
                )
        elif isinstance(statement, ast.ImportFrom):
            resolved_module = _resolve_relative_module(
                parsed_module, imported_module=statement.module, level=statement.level
            )
            if resolved_module is None:
                continue
            for alias in statement.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                aliases[local_name] = f"{resolved_module}.{alias.name}"
    return aliases


def _module_star_import_origins(
    parsed_module: ParsedModule,
) -> tuple[CompactModuleStarImportOrigin, ...]:
    origins: list[CompactModuleStarImportOrigin] = []

    class ModuleScopeStarImportCollector(ast.NodeVisitor):
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if any(alias.name == "*" for alias in node.names):
                origins.append(
                    CompactModuleStarImportOrigin(
                        _resolve_relative_module(
                            parsed_module,
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
    return tuple(origins)


@dataclass(frozen=True)
class CompactClassProjectionDemand:
    """Target-correlated filters for expensive class-family facets."""

    class_method_names: frozenset[str]
    include_autoregister_references: bool = True
    header_core_only: bool = False


def _class_report_demand(
    target_items: tuple[object, ...],
    config: object,
) -> CompactClassProjectionDemand:
    del config
    projections = tuple(
        item for item in target_items if isinstance(item, CompactModuleClassProjection)
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
                indexed_class.declares_autoregister_meta
                or (
                    indexed_class.keyed_family_key_type_name is not None
                    and "registry_key_attr" in indexed_class.assignments_by_name
                )
                for indexed_class in projection.classes
            )
            for projection in projections
        ),
    )


def _cached_class_demand_projection(
    items: tuple[object, ...],
    demand: object,
) -> tuple[object, ...]:
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
            autoregister_function_references=(
                item.autoregister_function_references
                if demand.include_autoregister_references
                else ()
            ),
            autoregister_reference_index=(
                item.autoregister_reference_index
                if demand.include_autoregister_references
                else None
            ),
        )
        for item in items
        if isinstance(item, CompactModuleClassProjection)
    )
    if not demand.header_core_only:
        return projected
    return tuple(item.header_core() for item in projected)


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


def _compact_base_reference_parts(node: ast.ClassDef) -> tuple[tuple[str, ...], ...]:
    return tuple(
        parts
        for base in node.bases
        if (
            parts := ATTRIBUTE_CHAIN_AUTHORITY.project(
                ClassSymbolResolutionAuthority.reference_node(base)
            )
        )
        is not None
    )


def _compact_indexed_classes(
    parsed_module: ParsedModule,
    indexed_class_nodes: tuple[tuple[str, ast.ClassDef], ...],
    *,
    include_body_facets: bool,
) -> tuple[CompactIndexedClass, ...]:
    file_path = parsed_module.file_path
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
            base_reference_parts=base_reference_parts,
            base_references_are_complete=len(base_reference_parts) == len(node.bases),
            direct_assignment_expressions=tuple(
                (target_name, ast.unparse(value) if value is not None else None)
                for target_name, value in direct_assignments.items()
            ),
            direct_assignment_lines=tuple(_direct_class_assignment_lines(node)),
            direct_value_constructions=_compact_class_value_constructions(
                direct_assignments
            ),
            direct_constant_string_assignments=tuple(
                sorted(
                    (name, value.value)
                    for name, value in direct_assignments.items()
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                )
            ),
            direct_non_none_assignment_names=sorted_tuple(
                name
                for name, value in direct_assignments.items()
                if not (isinstance(value, ast.Constant) and value.value is None)
            ),
            metaclass_names=tuple(
                terminal_name
                for keyword in node.keywords
                if keyword.arg == "metaclass"
                if (terminal_name := _terminal_reference_name(keyword.value))
                is not None
            ),
            class_keyword_names=tuple(keyword.arg or "**" for keyword in node.keywords),
            class_decorators_are_promotion_safe=all(
                any(
                    safe_decorator.is_proven_reference(
                        parsed_module.module,
                        decorator,
                    )
                    for safe_decorator in ClassMethodPromotionSafeDecorator
                )
                for decorator in node.decorator_list
            ),
            class_header_is_reconstructible=ClassHeaderSourceSpan.from_source(
                node,
                parsed_module.source,
            ).is_reconstructible,
            keyed_family_key_type_name=_keyed_family_key_type_name(node),
            is_final=any(
                (isinstance(decorator, ast.Name) and decorator.id == "final")
                or (isinstance(decorator, ast.Attribute) and decorator.attr == "final")
                for decorator in node.decorator_list
            ),
            end_line=node.end_lineno,
            method_names=tuple(
                statement.name
                for statement in node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            abstract_method_names=sorted_tuple(
                statement.name
                for statement in node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                if any(
                    _terminal_reference_name(decorator) == "abstractmethod"
                    for decorator in statement.decorator_list
                )
            ),
            is_abstract=_is_abstract_class(node),
            is_dataclass=_is_dataclass_class(node),
            declares_autoregister_meta=_declares_autoregister_meta(node),
            is_registration_authority=_is_registration_authority(node),
            autoregister_registry_key_attr_name=_autoregister_registry_key_attr_name(
                parsed_module,
                node,
            ),
            autoregister_key_extractor_name=_autoregister_key_extractor_name(node),
            autoregister_registry_projection_names=(
                _autoregister_registry_projection_names(node)
                if include_body_facets
                else ()
            ),
            keyed_registry_lookup_method_names=(
                _keyed_registry_lookup_method_names(node) if include_body_facets else ()
            ),
            keyed_registry_reverse_lookup_method_names=(
                _keyed_registry_reverse_lookup_method_names(node)
                if include_body_facets
                else ()
            ),
            predicate_selected_methods=(
                _compact_predicate_selected_methods(node) if include_body_facets else ()
            ),
        )
        for qualname, node in indexed_class_nodes
        for direct_assignments in (_direct_class_assignments(node),)
        for base_reference_parts in (_compact_base_reference_parts(node),)
    )


class CompactModuleClassProjectionFamily(CollectedFamily[CompactModuleClassProjection]):
    """Persist class/import facts needed by the global inheritance graph."""

    item_type = CompactModuleClassProjection
    cache_payload_max_bytes = 3_000_000
    report_demand_builder = staticmethod(_class_report_demand)
    cached_demand_projector = staticmethod(_cached_class_demand_projection)
    source_demand_collector = staticmethod(
        _collect_demanded_class_projection_from_source
    )

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactModuleClassProjection]:
        return cls._collect(parsed_module, None)

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
    def _collect_header_core(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactModuleClassProjection]:
        del cls
        indexed_class_nodes = _iter_class_defs(list(parsed_module.module.body))
        return [
            CompactModuleClassProjection(
                module_name=parsed_module.module_name,
                file_path=parsed_module.file_path,
                import_aliases=tuple(
                    sorted(_module_import_aliases(parsed_module).items())
                ),
                public_export_contract=_compact_module_public_export_contract(
                    parsed_module
                ),
                star_import_origins=_module_star_import_origins(parsed_module),
                classes=_compact_indexed_classes(
                    parsed_module,
                    indexed_class_nodes,
                    include_body_facets=False,
                ),
            ).header_core()
        ]

    @classmethod
    def _collect(
        cls,
        parsed_module: ParsedModule,
        demand: CompactClassProjectionDemand | None,
    ) -> list[CompactModuleClassProjection]:
        del cls
        file_path = parsed_module.file_path
        syntax_facets = _compact_class_syntax_facets(
            parsed_module,
            collect_autoregister=(
                demand is None or demand.include_autoregister_references
            ),
        )
        indexed_class_nodes = _iter_class_defs(list(parsed_module.module.body))
        all_class_nodes = tuple(
            node
            for node in _walk_nodes(parsed_module.module)
            if isinstance(node, ast.ClassDef)
        )
        class_methods = _compact_class_methods(
            parsed_module,
            indexed_class_nodes,
            method_names=(None if demand is None else demand.class_method_names),
        )
        (
            nominal_wrapper_authorities,
            pass_through_nominal_wrappers,
        ) = _compact_nominal_wrapper_scope_facts(
            parsed_module,
            all_class_nodes,
        )
        classes = _compact_indexed_classes(
            parsed_module,
            indexed_class_nodes,
            include_body_facets=True,
        )
        return [
            CompactModuleClassProjection(
                module_name=parsed_module.module_name,
                file_path=file_path,
                import_aliases=tuple(
                    sorted(_module_import_aliases(parsed_module).items())
                ),
                public_export_contract=_compact_module_public_export_contract(
                    parsed_module
                ),
                star_import_origins=_module_star_import_origins(parsed_module),
                classes=classes,
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
                exact_type_guards=syntax_facets.exact_type_guards,
                autoregister_function_references=(
                    syntax_facets.autoregister_function_references
                ),
                autoregister_reference_index=(
                    syntax_facets.autoregister_reference_index
                ),
                repeated_keyed_family_roots=_compact_repeated_keyed_family_roots(
                    parsed_module
                ),
                manual_subclass_roster_roots=_compact_manual_subclass_roster_roots(
                    parsed_module
                ),
                latent_rosters=LatentRosterObservation.from_module(parsed_module),
                named_projection_surfaces=_compact_named_projection_surfaces(
                    parsed_module
                ),
                manual_family_rosters=_compact_manual_family_rosters(parsed_module),
                nominal_wrapper_authorities=nominal_wrapper_authorities,
                pass_through_nominal_wrappers=pass_through_nominal_wrappers,
                class_methods=class_methods,
            )
        ]


def _compact_predicate_selected_methods(
    node: ast.ClassDef,
) -> tuple[tuple[int, str, str, str], ...]:
    """Project exact registered-types selector shapes without retaining method ASTs."""

    methods: list[tuple[int, str, str, str]] = []
    for statement in node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(
            _terminal_reference_name(decorator) == "classmethod"
            for decorator in statement.decorator_list
        ):
            continue
        shape = _compact_registered_type_match_shape(statement)
        if shape is None:
            continue
        match_name, predicate_name, context_name = shape
        guard_kinds = {
            SelectionGuardKind.from_node(candidate.test, match_name)
            for candidate in _trim_leading_docstring(list(statement.body))
            if isinstance(candidate, ast.If)
        }
        if not (
            SelectionGuardKind.NOT_EXACTLY_ONE in guard_kinds
            or ({SelectionGuardKind.EMPTY, SelectionGuardKind.AMBIGUOUS} <= guard_kinds)
        ):
            continue
        if not any(
            isinstance(candidate, ast.Subscript)
            and isinstance(candidate.value, ast.Name)
            and candidate.value.id == match_name
            and isinstance(candidate.slice, ast.Constant)
            and candidate.slice.value == 0
            for candidate in ast.walk(statement)
        ):
            continue
        methods.append((statement.lineno, statement.name, predicate_name, context_name))
    return tuple(methods)


def _trim_leading_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _compact_registered_type_match_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str] | None:
    parameter_names = {
        argument.arg
        for argument in (
            *method.args.posonlyargs,
            *method.args.args,
            *method.args.kwonlyargs,
        )
        if argument.arg not in {"self", "cls"}
    }
    for statement in _trim_leading_docstring(list(method.body)):
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.ListComp)
            and len(statement.value.generators) == 1
        ):
            continue
        match_name = statement.targets[0].id
        comprehension = statement.value
        generator = comprehension.generators[0]
        if not (
            not generator.is_async
            and isinstance(generator.target, ast.Name)
            and isinstance(comprehension.elt, ast.Name)
            and comprehension.elt.id == generator.target.id
            and isinstance(generator.iter, ast.Call)
            and not generator.iter.args
            and not generator.iter.keywords
            and isinstance(generator.iter.func, ast.Attribute)
            and generator.iter.func.attr == "registered_types"
            and isinstance(generator.iter.func.value, ast.Name)
            and generator.iter.func.value.id == "cls"
            and len(generator.ifs) == 1
            and isinstance(generator.ifs[0], ast.Call)
        ):
            continue
        predicate = generator.ifs[0]
        if not (
            not predicate.keywords
            and len(predicate.args) == 1
            and isinstance(predicate.args[0], ast.Name)
            and predicate.args[0].id in parameter_names
            and isinstance(predicate.func, ast.Attribute)
            and predicate.func.attr
            and isinstance(predicate.func.value, ast.Name)
            and predicate.func.value.id == generator.target.id
        ):
            continue
        return match_name, predicate.func.attr, predicate.args[0].id
    return None


def _compact_manual_subclass_roster_roots(
    parsed_module: ParsedModule,
) -> tuple[CompactManualSubclassRosterRoot, ...]:
    roots: list[CompactManualSubclassRosterRoot] = []
    file_path = parsed_module.file_path
    for qualname, node in _iter_class_defs(list(parsed_module.module.body)):
        registry_names = _compact_class_list_registry_names(node)
        if not registry_names:
            continue
        init_subclass = next(
            (
                statement
                for statement in node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == "__init_subclass__"
            ),
            None,
        )
        if init_subclass is None:
            continue
        sites = _compact_manual_subclass_registration_sites(
            init_subclass, registry_names, owner_name=node.name
        )
        if not sites:
            continue
        consumers: list[tuple[str, int, str, str]] = []
        for registry_name in registry_names:
            for method in node.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if method.name == "__init_subclass__":
                    continue
                if _compact_uses_named_registry(
                    method,
                    registry_name=registry_name,
                    owner_names=frozenset({"cls", "type", node.name}),
                ):
                    consumers.append(
                        (
                            registry_name,
                            method.lineno,
                            f"{node.name}.{method.name}",
                            file_path,
                        )
                    )
            for function_qualname, function in _named_functions(parsed_module.module):
                if "." in function_qualname:
                    continue
                if _compact_uses_named_registry(
                    function,
                    registry_name=registry_name,
                    owner_names=frozenset({node.name}),
                ):
                    consumers.append(
                        (
                            registry_name,
                            function.lineno,
                            function_qualname,
                            file_path,
                        )
                    )
        roots.append(
            CompactManualSubclassRosterRoot(
                class_symbol=f"{parsed_module.module_name}.{qualname}",
                init_subclass_line=init_subclass.lineno,
                registration_sites=sites,
                consumer_locations=sorted_tuple(
                    set(consumers), key=lambda item: (item[1], item[2], item[0])
                ),
            )
        )
    return tuple(roots)


def _compact_class_list_registry_names(node: ast.ClassDef) -> tuple[str, ...]:
    return sorted_tuple(
        {
            target_name
            for statement in node.body
            for target_name in (
                (
                    statement.targets[0].id
                    if isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and isinstance(statement.value, ast.List)
                    else (
                        statement.target.id
                        if isinstance(statement, ast.AnnAssign)
                        and isinstance(statement.target, ast.Name)
                        and isinstance(statement.value, ast.List)
                        else None
                    )
                ),
            )
            if target_name is not None
        }
    )


def _compact_manual_subclass_registration_sites(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    registry_names: tuple[str, ...],
    *,
    owner_name: str,
) -> tuple[CompactManualSubclassRegistrationSite, ...]:
    sites: dict[str, CompactManualSubclassRegistrationSite] = {}

    def walk_statements(
        statements: list[ast.stmt], guard_stack: tuple[ast.AST, ...]
    ) -> None:
        for statement in statements:
            if isinstance(statement, ast.If):
                walk_statements(statement.body, (*guard_stack, statement.test))
                walk_statements(statement.orelse, guard_stack)
                continue
            for subnode in ast.walk(statement):
                registry_name = _compact_registration_append_registry_name(
                    subnode, registry_names, owner_name
                )
                if registry_name is None:
                    continue
                sites[registry_name] = CompactManualSubclassRegistrationSite(
                    registry_name=registry_name,
                    guard_summary=(
                        " and ".join(ast.unparse(guard) for guard in guard_stack)
                        if guard_stack
                        else None
                    ),
                    selector_attr_name=next(
                        (
                            attr_name
                            for guard in guard_stack
                            if (attr_name := _compact_guarded_defined_attr_name(guard))
                            is not None
                        ),
                        None,
                    ),
                    requires_concrete_subclass=any(
                        _compact_guard_requires_concrete_subclass(guard)
                        for guard in guard_stack
                    ),
                )

    walk_statements(_trim_leading_docstring(list(method.body)), ())
    return tuple(sites[name] for name in sorted(sites))


def _compact_registration_append_registry_name(
    node: ast.AST, registry_names: tuple[str, ...], owner_name: str
) -> str | None:
    if not (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr in registry_names
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id in {"cls", "type", owner_name}
        and _compact_looks_like_cls_registration_value(node.args[0])
    ):
        return None
    return node.func.value.attr


def _compact_looks_like_cls_registration_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "cls"
    return bool(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cast"
        and node.args
        and _compact_looks_like_cls_registration_value(node.args[-1])
    )


def _compact_class_dict_get_attr_name(node: ast.AST) -> str | None:
    if not (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "__dict__"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "cls"
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return None
    return node.args[0].value


def _compact_guarded_defined_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _compact_class_dict_get_attr_name(node)
    if not (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.IsNot, ast.NotEq))
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    ):
        return None
    return _compact_class_dict_get_attr_name(node.left)


def _compact_guard_requires_concrete_subclass(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Call)
        and isinstance(node.operand.func, ast.Attribute)
        and isinstance(node.operand.func.value, ast.Name)
        and node.operand.func.value.id == "inspect"
        and node.operand.func.attr == "isabstract"
        and len(node.operand.args) == 1
        and isinstance(node.operand.args[0], ast.Name)
        and node.operand.args[0].id == "cls"
    )


def _compact_uses_named_registry(
    node: ast.AST,
    *,
    registry_name: str,
    owner_names: frozenset[str],
) -> bool:
    return any(
        isinstance(subnode, ast.Attribute)
        and subnode.attr == registry_name
        and isinstance(subnode.value, ast.Name)
        and subnode.value.id in owner_names
        for subnode in ast.walk(node)
    )


def _compact_assignment_target_value(
    statement: ast.stmt,
) -> tuple[ast.AST, ast.AST] | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        return statement.targets[0], statement.value
    if isinstance(statement, ast.AnnAssign) and statement.value is not None:
        return statement.target, statement.value
    return None


def _compact_named_projection_surfaces(
    parsed_module: ParsedModule,
) -> tuple[CompactNamedProjectionSurface, ...]:
    named_values: dict[str, tuple[int, ast.AST]] = {}
    named_sequences: dict[str, tuple[int, ast.Tuple | ast.List]] = {}
    for statement in _trim_leading_docstring(list(parsed_module.module.body)):
        target_value = _compact_assignment_target_value(statement)
        if target_value is None or not isinstance(target_value[0], ast.Name):
            continue
        target, value = target_value
        named_values[target.id] = (statement.lineno, value)
        if isinstance(value, (ast.Tuple, ast.List)):
            named_sequences[target.id] = (statement.lineno, value)

    surfaces: list[CompactNamedProjectionSurface] = []
    for surface_name, (line, value) in named_sequences.items():
        surfaces.append(
            CompactNamedProjectionSurface(
                file_path=parsed_module.file_path,
                surface_name=surface_name,
                line=line,
                sequence_references=tuple(
                    reference
                    for element in value.elts
                    if (reference := _compact_projection_reference(element)) is not None
                ),
            )
        )
    for surface_name, (line, value) in named_values.items():
        if isinstance(value, ast.Dict):
            surfaces.append(
                CompactNamedProjectionSurface(
                    file_path=parsed_module.file_path,
                    surface_name=surface_name,
                    line=line,
                    dict_key_references=tuple(
                        reference
                        for key in value.keys
                        if key is not None
                        if (reference := _compact_projection_reference(key)) is not None
                    ),
                    dict_value_references=tuple(
                        reference
                        for item in value.values
                        if (reference := _compact_projection_reference(item))
                        is not None
                    ),
                )
            )
    return tuple(surfaces)


def _compact_projection_reference(
    node: ast.AST,
) -> tuple[str, str | None] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, None
    if isinstance(node, ast.Name):
        return node.id, node.id
    if isinstance(node, ast.Attribute):
        parts = ATTRIBUTE_CHAIN_AUTHORITY.project(node)
        if parts is None:
            return ast.unparse(node), None
        return ".".join(parts), parts[0]
    return None


def _compact_self_assignment(
    statement: ast.stmt,
) -> tuple[str, ast.AST | None] | None:
    target: ast.AST | None = None
    value: ast.AST | None = None
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        target = statement.target
        value = statement.value
    if not (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return None
    return target.attr, value


def _compact_class_field_type_map(
    node: ast.ClassDef,
) -> tuple[tuple[str, str], ...]:
    nominal_fields: dict[str, str] = {}
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            if not CLASSVAR_ANNOTATION_AUTHORITY.matches(statement.annotation):
                nominal_fields.setdefault(
                    statement.target.id,
                    ast.unparse(statement.annotation),
                )
            continue
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if statement.name != "__init__":
            continue
        parameter_annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in (
                *statement.args.posonlyargs,
                *statement.args.args,
                *statement.args.kwonlyargs,
            )
            if argument.annotation is not None
        }
        for inner in statement.body:
            assignment = _compact_self_assignment(inner)
            if assignment is None:
                continue
            field_name, value = assignment
            if isinstance(value, ast.Name) and value.id in parameter_annotations:
                nominal_fields.setdefault(
                    field_name,
                    parameter_annotations[value.id],
                )
    return sorted_tuple(nominal_fields.items())


class _CompactClassMethodSemanticSkeletonNormalizer(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg) -> ast.arg:
        del node
        return ast.arg(arg="ARG", annotation=None, type_comment=None)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        return ast.copy_location(
            ast.Attribute(
                value=self.visit(node.value),
                attr="ATTR",
                ctx=node.ctx,
            ),
            node,
        )

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.copy_location(ast.Constant(value="CONST"), node)


def _compact_class_method_statement_skeleton(statement: ast.stmt) -> str:
    normalized = _CompactClassMethodSemanticSkeletonNormalizer().visit(statement)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _compact_class_method_coordinates(
    node: ast.AST,
    path: tuple[object, ...] = (),
) -> tuple[tuple[tuple[str, ...], str, str], ...]:
    coordinates: list[tuple[tuple[str, ...], str, str]] = []
    coordinate_path = tuple(str(item) for item in path)
    if isinstance(node, ast.Constant):
        coordinates.append((coordinate_path, "constant", repr(node.value)))
    elif isinstance(node, ast.Name):
        coordinates.append((coordinate_path, "name", node.id))
    elif isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            coordinates.append((coordinate_path, "self_attr", node.attr))
        else:
            coordinates.append((coordinate_path, "attribute", ast.unparse(node)))
    elif isinstance(node, ast.Call):
        coordinates.append(
            (
                tuple(str(item) for item in (*path, "func")),
                "call",
                ast.unparse(node.func),
            )
        )
    skipped_fields = {"func"} if isinstance(node, ast.Call) else set()
    for field_name, value in ast.iter_fields(node):
        if field_name in skipped_fields:
            continue
        if isinstance(value, ast.AST):
            coordinates.extend(
                _compact_class_method_coordinates(value, (*path, field_name))
            )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, ast.AST):
                    coordinates.extend(
                        _compact_class_method_coordinates(
                            item,
                            (*path, field_name, index),
                        )
                    )
    return tuple(coordinates)


def _compact_class_method_statements_from_sources(
    statement_sources: tuple[str, ...],
) -> tuple[ast.stmt, ...]:
    statements = tuple(ast.parse("\n".join(statement_sources)).body)
    if len(statements) != len(statement_sources):
        raise ValueError("Method-family statement source lost its AST boundary")
    return statements


def _compact_class_method(
    class_symbol: str,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: tuple[str, ...],
    module_bound_names: frozenset[str],
    class_bound_names: frozenset[str],
    *,
    include_family_semantics: bool,
) -> CompactClassMethod:
    body = _trim_leading_docstring(list(method.body))
    decorator_lines = tuple(
        decorator.lineno for decorator in method.decorator_list if decorator.lineno
    )
    start_line = min((*decorator_lines, method.lineno))
    is_tiny_role = len(body) <= 2
    safety_profile = (
        ClassMethodPromotionSafetyProfile.from_method(
            method,
            module_bound_names,
            class_bound_names,
            source_lines=source_lines,
        )
        if is_tiny_role
        else None
    )
    return CompactClassMethod(
        class_symbol=class_symbol,
        method_name=method.name,
        line=start_line,
        line_count=max(1, (method.end_lineno or method.lineno) - start_line + 1),
        body_statement_count=len(body),
        statement_sources=(
            tuple(ast.unparse(statement) for statement in body)
            if include_family_semantics and len(body) >= 3
            else ()
        ),
        exact_source_digest=(
            hashlib.blake2s(
                "".join(
                    source_lines[start_line - 1 : (method.end_lineno or method.lineno)]
                ).encode("utf-8"),
                digest_size=16,
            ).hexdigest()
            if is_tiny_role
            else None
        ),
        promotion_hazards=() if safety_profile is None else safety_profile.hazards,
        receiver_member_names=(
            ClassMethodReceiverRequirements.from_method(method).member_names
            if is_tiny_role
            else frozenset()
        ),
    )


def _compact_class_methods(
    parsed_module: ParsedModule,
    indexed_class_nodes: tuple[tuple[str, ast.ClassDef], ...],
    *,
    method_names: frozenset[str] | None = None,
) -> tuple[CompactClassMethod, ...]:
    methods: list[CompactClassMethod] = []
    source_lines = tuple(parsed_module.source.splitlines(keepends=True))
    module_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
        parsed_module.module.body
    )
    for qualname, node in indexed_class_nodes:
        class_symbol = f"{parsed_module.module_name}.{qualname}"
        class_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
        establishes_nominal_family = any(
            ClassSymbolResolutionAuthority.establishes_nominal_family(declared_name)
            for base in node.bases
            if (
                declared_name := ClassSymbolResolutionAuthority.declared_base_name(base)
            )
            is not None
        )
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if method_names is not None and statement.name not in method_names:
                continue
            method = _compact_class_method(
                class_symbol,
                statement,
                source_lines,
                module_bound_names,
                class_bound_names,
                include_family_semantics=establishes_nominal_family,
            )
            if establishes_nominal_family or method.statement_count <= 2:
                methods.append(method)
    return tuple(methods)


def _compact_manual_family_rosters(
    parsed_module: ParsedModule,
) -> tuple[CompactManualFamilyRosterObservation, ...]:
    known_class_names = {
        node.name
        for node in _walk_nodes(parsed_module.module)
        if isinstance(node, ast.ClassDef)
    }
    observations: list[CompactManualFamilyRosterObservation] = []
    for statement in _trim_leading_docstring(list(parsed_module.module.body)):
        owner_name: str | None = None
        source_node: ast.AST | None = None
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = _trim_leading_docstring(list(statement.body))
            if (
                len(body) == 1
                and isinstance(body[0], ast.Return)
                and body[0].value is not None
            ):
                owner_name = statement.name
                source_node = body[0].value
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            owner_name = statement.targets[0].id
            source_node = statement.value
        if owner_name is None or not isinstance(
            source_node, (ast.Tuple, ast.List, ast.Set)
        ):
            continue
        members = tuple(
            _compact_manual_family_roster_member(element, known_class_names)
            for element in source_node.elts
        )
        if len(members) < 2 or any(member is None for member in members):
            continue
        member_names, constructor_styles = zip(
            *(member for member in members if member is not None), strict=True
        )
        observations.append(
            CompactManualFamilyRosterObservation(
                file_path=parsed_module.file_path,
                line=statement.lineno,
                owner_name=owner_name,
                member_names=member_names,
                constructor_style="+".join(sorted(set(constructor_styles))),
            )
        )
    return tuple(observations)


def _compact_nominal_wrapper_scope_facts(
    parsed_module: ParsedModule,
    class_nodes: tuple[ast.ClassDef, ...],
) -> tuple[
    tuple[CompactNominalWrapperAuthority, ...],
    tuple[CompactPassThroughNominalWrapper, ...],
]:
    nominal_wrapper_authorities: list[CompactNominalWrapperAuthority] = []
    pass_through_nominal_wrappers: list[CompactPassThroughNominalWrapper] = []
    for node in class_nodes:
        field_type_map = _compact_class_field_type_map(node)
        if _compact_is_reusable_nominal_wrapper_authority(node):
            nominal_wrapper_authorities.append(
                CompactNominalWrapperAuthority(
                    file_path=parsed_module.file_path,
                    class_name=node.name,
                    line=node.lineno,
                    method_names=sorted_tuple(
                        statement.name
                        for statement in node.body
                        if isinstance(
                            statement, (ast.FunctionDef, ast.AsyncFunctionDef)
                        )
                    ),
                )
            )
        wrapper = _compact_pass_through_nominal_wrapper(
            parsed_module,
            node,
            field_type_map,
        )
        if wrapper is not None:
            pass_through_nominal_wrappers.append(wrapper)
    return (
        tuple(nominal_wrapper_authorities),
        tuple(pass_through_nominal_wrappers),
    )


def _compact_is_reusable_nominal_wrapper_authority(node: ast.ClassDef) -> bool:
    if node.name.endswith("Detector"):
        return False
    return _is_abstract_class(node) or node.name.endswith(("Base", "Mixin", "Carrier"))


def _compact_normalized_nominal_authority_name(annotation_text: str) -> str:
    text = annotation_text.strip("\"'")
    text = re.split("\\s*\\|\\s*", text, maxsplit=1)[0]
    text = re.split("[\\[,]", text, maxsplit=1)[0]
    return text.rsplit(".", 1)[-1].strip()


def _compact_forwarded_parameter_names(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    return tuple(
        argument.arg
        for argument in (
            *method.args.posonlyargs,
            *method.args.args[1:],
            *method.args.kwonlyargs,
        )
    )


def _compact_call_forwards_parameters(
    call: ast.Call,
    parameter_names: tuple[str, ...],
) -> bool:
    parameter_set = frozenset(parameter_names)

    def forwards_argument(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in parameter_set
        return bool(
            isinstance(node, ast.Starred)
            and isinstance(node.value, ast.Name)
            and node.value.id in parameter_set
        )

    return all(forwards_argument(argument) for argument in call.args) and all(
        keyword.arg is None
        or (
            keyword.arg in parameter_set
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == keyword.arg
        )
        for keyword in call.keywords
    )


def _compact_forwarded_nominal_member_name(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    delegate_field_name: str,
) -> str | None:
    body = _trim_leading_docstring(list(method.body))
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    returned = body[0].value
    is_property = any(
        _terminal_reference_name(decorator) == "property"
        for decorator in method.decorator_list
    )
    if is_property:
        if not (
            isinstance(returned, ast.Attribute)
            and returned.attr == method.name
            and isinstance(returned.value, ast.Attribute)
            and returned.value.attr == delegate_field_name
            and isinstance(returned.value.value, ast.Name)
            and returned.value.value.id == "self"
        ):
            return None
        return method.name
    if not (
        isinstance(returned, ast.Call)
        and isinstance(returned.func, ast.Attribute)
        and returned.func.attr == method.name
        and isinstance(returned.func.value, ast.Attribute)
        and returned.func.value.attr == delegate_field_name
        and isinstance(returned.func.value.value, ast.Name)
        and returned.func.value.value.id == "self"
        and _compact_call_forwards_parameters(
            returned,
            _compact_forwarded_parameter_names(method),
        )
    ):
        return None
    return method.name


def _compact_pass_through_nominal_wrapper(
    parsed_module: ParsedModule,
    node: ast.ClassDef,
    field_type_map: tuple[tuple[str, str], ...],
) -> CompactPassThroughNominalWrapper | None:
    if _is_abstract_class(node) or len(field_type_map) != 1:
        return None
    delegate_field_name, annotation_text = field_type_map[0]
    delegate_authority_name = _compact_normalized_nominal_authority_name(
        annotation_text
    )
    if not delegate_authority_name:
        return None
    declared_base_names = {
        terminal_name
        for base in node.bases
        if (terminal_name := _terminal_reference_name(base)) is not None
    }
    if delegate_authority_name in declared_base_names:
        return None
    forwarded_member_names: list[str] = []
    for statement in _trim_leading_docstring(list(node.body)):
        if isinstance(statement, (ast.AnnAssign, ast.Assign)):
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.name == "__init__":
                continue
            if statement.name.startswith("__") and statement.name.endswith("__"):
                return None
            forwarded_member_name = _compact_forwarded_nominal_member_name(
                statement,
                delegate_field_name,
            )
            if forwarded_member_name is None:
                return None
            forwarded_member_names.append(forwarded_member_name)
            continue
        return None
    if len(forwarded_member_names) < 2:
        return None
    return CompactPassThroughNominalWrapper(
        file_path=parsed_module.file_path,
        class_name=node.name,
        line=node.lineno,
        delegate_field_name=delegate_field_name,
        delegate_authority_name=delegate_authority_name,
        forwarded_member_names=sorted_tuple(set(forwarded_member_names)),
    )


def _compact_manual_family_roster_member(
    node: ast.AST,
    known_class_names: set[str],
) -> tuple[str, str] | None:
    if isinstance(node, ast.Name) and node.id in known_class_names:
        return node.id, "class_reference"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in known_class_names
        and not node.args
        and not node.keywords
    ):
        return node.func.id, "constructor_call"
    return None


def _annotation_type_names(node: ast.AST | None) -> tuple[str, ...]:
    if node is None:
        return ()
    if isinstance(node, ast.Constant) and node.value is None:
        return ()
    if isinstance(node, ast.Name):
        return () if node.id == "None" else (node.id,)
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    if isinstance(node, ast.Tuple):
        return sorted_tuple(
            {name for element in node.elts for name in _annotation_type_names(element)}
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return sorted_tuple(
            {*_annotation_type_names(node.left), *_annotation_type_names(node.right)}
        )
    if isinstance(node, ast.Subscript):
        base_name = _terminal_reference_name(node.value)
        if base_name in {"Optional", "Required", "NotRequired", "Type", "type"}:
            return _annotation_type_names(node.slice)
        if base_name == "Annotated":
            if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                return _annotation_type_names(node.slice.elts[0])
            return _annotation_type_names(node.slice)
    return ()


def _keyed_family_key_type_name(node: ast.ClassDef) -> str | None:
    for base in node.bases:
        if not isinstance(base, ast.Subscript):
            continue
        if _terminal_reference_name(base.value) != "KeyedNominalFamily":
            continue
        type_names = _annotation_type_names(base.slice)
        if type_names:
            return type_names[0]
    return None


def _compact_keyed_table_axes(
    parsed_module: ParsedModule,
) -> tuple[CompactKeyedTableAxis, ...]:
    axes: list[CompactKeyedTableAxis] = []
    for statement in parsed_module.module.body:
        table_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                table_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            table_name = statement.target.id
            value = statement.value
        if table_name is None or not isinstance(value, ast.Dict):
            continue
        if len(value.keys) < 2 or any(key is None for key in value.keys):
            continue
        case_names = tuple(ast.unparse(key) for key in value.keys if key is not None)
        key_type_names = {
            case_name.split(".", 1)[0] for case_name in case_names if "." in case_name
        }
        if len(key_type_names) != 1:
            continue
        value_shape_name: str | None = None
        value_constructor_names = {
            ast.unparse(item.func)
            for item in value.values
            if isinstance(item, ast.Call)
        }
        if (
            all(isinstance(item, ast.Call) for item in value.values)
            and len(value_constructor_names) == 1
        ):
            value_shape_name = next(iter(value_constructor_names))
        axes.append(
            CompactKeyedTableAxis(
                file_path=parsed_module.file_path,
                line=statement.lineno,
                table_name=table_name,
                key_type_name=next(iter(key_type_names)),
                case_names=sorted_tuple(case_names),
                value_shape_name=value_shape_name,
            )
        )
    return tuple(axes)


def _compact_manual_selector_axes(
    parsed_module: ParsedModule,
) -> tuple[CompactManualSelectorAxis, ...]:
    dict_literals: dict[str, ast.Dict] = {}
    for statement in parsed_module.module.body:
        name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            if isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            name = statement.target.id
            value = statement.value
        if name is not None and isinstance(value, ast.Dict):
            dict_literals[name] = value
    case_names_by_mapping = {
        name: tuple(ast.unparse(key) for key in mapping.keys if key is not None)
        for name, mapping in dict_literals.items()
    }
    known_mapping_names = frozenset(
        name
        for name, case_names in case_names_by_mapping.items()
        if len(case_names) >= 2
    )
    axes: list[CompactManualSelectorAxis] = []
    for node in _walk_nodes(parsed_module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not method.name.startswith("for_") or not any(
                _terminal_reference_name(decorator) == "classmethod"
                for decorator in method.decorator_list
            ):
                continue
            parameter_names = {
                item.arg
                for item in (
                    *method.args.posonlyargs,
                    *method.args.args,
                    *method.args.kwonlyargs,
                )
                if item.arg not in {"self", "cls"}
            }
            if not parameter_names:
                continue
            mapping_name: str | None = None
            for subnode in ast.walk(method):
                if not (
                    isinstance(subnode, ast.Subscript)
                    and isinstance(subnode.value, ast.Name)
                    and subnode.value.id in known_mapping_names
                    and ast.unparse(subnode.slice) in parameter_names
                ):
                    continue
                mapping_name = subnode.value.id
                break
            if mapping_name is None:
                continue
            case_names = case_names_by_mapping[mapping_name]
            key_type_names = {
                case_name.split(".", 1)[0]
                for case_name in case_names
                if "." in case_name
            }
            if len(key_type_names) != 1:
                continue
            axes.append(
                CompactManualSelectorAxis(
                    file_path=parsed_module.file_path,
                    line=method.lineno,
                    family_name=node.name,
                    selector_method_name=method.name,
                    key_type_name=next(iter(key_type_names)),
                    case_names=case_names,
                )
            )
    return tuple(axes)


def _exact_type_predicate(
    node: ast.AST,
) -> tuple[ast.AST, ast.AST, bool, str] | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        negated = True
        comparison = node.operand
    else:
        negated = False
        comparison = node
    if not (
        isinstance(comparison, ast.Compare)
        and len(comparison.ops) == 1
        and len(comparison.comparators) == 1
    ):
        return None
    operator = comparison.ops[0]
    if not isinstance(operator, (ast.Is, ast.Eq, ast.IsNot, ast.NotEq)):
        return None
    left = comparison.left
    right = comparison.comparators[0]
    subject = _type_call_subject(left)
    type_reference = right
    if subject is None:
        subject = _type_call_subject(right)
        type_reference = left
    if subject is None:
        return None
    matches_exact_type = isinstance(operator, (ast.Is, ast.Eq))
    return (
        subject,
        type_reference,
        not matches_exact_type if negated else matches_exact_type,
        ast.unparse(node),
    )


def _type_call_subject(node: ast.AST) -> ast.AST | None:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "type"
        and len(node.args) == 1
        and not node.keywords
    ):
        return None
    return node.args[0]


def _fail_loud_block(statements: list[ast.stmt]) -> bool:
    non_terminating_prefix_types = (
        ast.AnnAssign,
        ast.Assign,
        ast.AugAssign,
        ast.Delete,
        ast.Expr,
        ast.Import,
        ast.ImportFrom,
        ast.Pass,
    )
    return (
        bool(statements)
        and isinstance(statements[-1], ast.Raise)
        and all(
            isinstance(statement, non_terminating_prefix_types)
            for statement in statements[:-1]
        )
    )


def _named_functions(
    module: ast.Module,
) -> tuple[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef], ...]:
    return named_function_nodes(module)


def _enum_member_refs_by_key_type(node: ast.AST) -> dict[str, tuple[str, ...]]:
    refs: dict[str, set[str]] = defaultdict(set)
    for subnode in ast.walk(node):
        parts = ATTRIBUTE_CHAIN_AUTHORITY.project(subnode)
        if parts is None or len(parts) < 2:
            continue
        key_type_name = parts[-2]
        refs[key_type_name].add(f"{key_type_name}.{parts[-1]}")
    return {
        key_type_name: sorted_tuple(case_names)
        for key_type_name, case_names in refs.items()
    }


def _direct_class_assignments(node: ast.ClassDef) -> dict[str, ast.AST | None]:
    assignments: dict[str, ast.AST | None] = {}
    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                assignments[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target,
            ast.Name,
        ):
            assignments[statement.target.id] = statement.value
    return assignments


def _string_constant(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _module_string_constant_assignments(
    parsed_module: ParsedModule,
) -> dict[str, str]:
    constants: dict[str, str] = {}
    for statement in parsed_module.module.body:
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is not None and (string_value := _string_constant(value)):
            constants[target_name] = string_value
    return constants


def _class_direct_string_member_assignments(
    node: ast.ClassDef,
) -> dict[str, str]:
    return {
        name: string_value
        for name, value in _direct_class_assignments(node).items()
        if (string_value := _string_constant(value)) is not None
    }


def _module_string_enum_member_assignments(
    parsed_module: ParsedModule,
) -> dict[tuple[str, str], str]:
    enum_base_names = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
    members: dict[tuple[str, str], str] = {}
    for statement in parsed_module.module.body:
        if not isinstance(statement, ast.ClassDef):
            continue
        if not enum_base_names & {
            terminal_name
            for base in statement.bases
            if (terminal_name := _terminal_reference_name(base)) is not None
        }:
            continue
        for member_name, string_value in _class_direct_string_member_assignments(
            statement
        ).items():
            members[(statement.name, member_name)] = string_value
    return members


def _enum_member_value_reference(node: ast.AST | None) -> tuple[str, str] | None:
    if not (
        isinstance(node, ast.Attribute)
        and node.attr == "value"
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
    ):
        return None
    return node.value.value.id, node.value.attr


def _autoregister_constant_name(
    node: ast.AST | None,
    parsed_module: ParsedModule,
) -> str | None:
    if (string_value := _string_constant(node)) is not None:
        return string_value
    if (enum_ref := _enum_member_value_reference(node)) is not None:
        return _module_string_enum_member_assignments(parsed_module).get(enum_ref)
    if isinstance(node, ast.Name):
        return (
            _module_string_constant_assignments(parsed_module).get(node.id) or node.id
        )
    if isinstance(node, ast.Attribute) and node.attr == "__registry_key__":
        return node.attr
    return None


def _registry_family_key_attr_name(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if _terminal_reference_name(node.func) != "RegistryFamily" or not node.args:
        return None
    key_arg = node.args[0]
    if (key_literal := _string_constant(key_arg)) is not None:
        return key_literal
    if isinstance(key_arg, ast.Attribute):
        return key_arg.attr.lower()
    return None


def _registry_config_keyword(
    node: ast.AST | None,
    keyword_name: str,
) -> ast.AST | None:
    if not isinstance(node, ast.Call):
        return None
    if _terminal_reference_name(node.func) != "RegistryConfig":
        return None
    values = tuple(
        keyword.value for keyword in node.keywords if keyword.arg == keyword_name
    )
    return values[0] if len(values) == 1 else None


def _autoregister_registry_key_attr_name(
    parsed_module: ParsedModule,
    node: ast.ClassDef,
) -> str | None:
    assignments = _direct_class_assignments(node)
    explicit_key = _autoregister_constant_name(
        assignments.get("__registry_key__"), parsed_module
    )
    if explicit_key is not None:
        return explicit_key
    configured_key = _autoregister_constant_name(
        _registry_config_keyword(
            assignments.get("__registry_config__"),
            "key_attribute",
        ),
        parsed_module,
    )
    if configured_key is not None:
        return configured_key
    stable_key_axis = assignments.get("stable_key_axis")
    if (
        isinstance(stable_key_axis, ast.Name)
        and stable_key_axis.id == "__registry_key__"
    ):
        return stable_key_axis.id
    stable_key_name = _autoregister_constant_name(stable_key_axis, parsed_module)
    if stable_key_name is not None:
        return stable_key_name
    return _registry_family_key_attr_name(assignments.get("__registry_family__"))


def _autoregister_key_extractor_name(node: ast.ClassDef) -> str | None:
    assignments = _direct_class_assignments(node)
    extractor = assignments.get("__key_extractor__")
    if extractor is None:
        extractor = _registry_config_keyword(
            assignments.get("__registry_config__"),
            "key_extractor",
        )
    if isinstance(extractor, ast.Constant) and extractor.value is None:
        return None
    return ast.unparse(extractor) if extractor is not None else None


def _autoregister_registry_projection_names(
    node: ast.ClassDef,
) -> tuple[str, ...]:
    return tuple(
        method.name
        for method in node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        if any(
            isinstance(subnode, ast.Attribute)
            and subnode.attr in {"__registry__", "_registry", "registry"}
            for subnode in ast.walk(method)
        )
    )


def _is_classmethod(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _terminal_reference_name(decorator) == "classmethod"
        for decorator in method.decorator_list
    )


def _method_references_cls_registry(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return RegistryLookupShape.references_registry(method)


def _keyed_registry_lookup_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        method.name
        for method in node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        if _is_classmethod(method) and _method_references_cls_registry(method)
    )


def _keyed_registry_reverse_lookup_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        method.name
        for method in node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        if _is_classmethod(method)
        and _method_references_cls_registry(method)
        and any(token in method.name for token in ("class", "type", "reverse"))
    )


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
) -> tuple[CompactRepeatedKeyedFamilyRoot, ...]:
    roots: list[CompactRepeatedKeyedFamilyRoot] = []
    for node in parsed_module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if "AutoRegisterByClassVar" not in {
            terminal_name
            for base in node.bases
            if (terminal_name := _terminal_reference_name(base)) is not None
        }:
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
            for method in node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
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
                family_base_name="AutoRegisterByClassVar",
                registry_key_attr_name=registry_key_attr_name,
                lookup_method_name=lookup_method.name,
                lookup_style=lookup_shape.style,
                error_type_name=lookup_shape.error_type_name,
                abstract_hook_names=tuple(
                    method.name
                    for method in node.body
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    if any(
                        _terminal_reference_name(decorator) == "abstractmethod"
                        for decorator in method.decorator_list
                    )
                ),
            )
        )
    return tuple(roots)


@dataclass
class _CompactAutoRegisterFunctionReferenceBuilder:
    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    receiver_attribute_refs: set[tuple[str, str]]
    referenced_symbols: set[str]
    calls_autoregister_meta: bool = False


def _compact_autoregister_reference_projection(
    builders: tuple[_CompactAutoRegisterFunctionReferenceBuilder, ...],
) -> tuple[
    tuple[CompactAutoRegisterFunctionReference, ...],
    CompactAutoRegisterReferenceIndex | None,
]:
    references = tuple(
        CompactAutoRegisterFunctionReference(
            qualname=builder.qualname,
            referenced_symbols=sorted_tuple(builder.referenced_symbols),
            calls_autoregister_meta=True,
            receiver_attribute_refs=sorted_tuple(builder.receiver_attribute_refs),
        )
        for builder in builders
        if builder.calls_autoregister_meta
    )
    consumer_builders = tuple(
        builder for builder in builders if builder.receiver_attribute_refs
    )
    if not consumer_builders:
        return references, None
    receiver_names = sorted_tuple(
        {
            receiver_name
            for builder in consumer_builders
            for receiver_name, _attr_name in builder.receiver_attribute_refs
        }
    )
    attribute_names = sorted_tuple(
        {
            attr_name
            for builder in consumer_builders
            for _receiver_name, attr_name in builder.receiver_attribute_refs
        }
    )
    receiver_indexes = {name: index for index, name in enumerate(receiver_names)}
    attribute_indexes = {name: index for index, name in enumerate(attribute_names)}
    return references, CompactAutoRegisterReferenceIndex(
        function_qualnames=tuple(builder.qualname for builder in consumer_builders),
        receiver_names=receiver_names,
        attribute_names=attribute_names,
        encoded_edges=";".join(
            f"{function_index},{receiver_index},{attribute_index}"
            for function_index, receiver_index, attribute_index in sorted(
                {
                    (
                        function_index,
                        receiver_indexes[receiver_name],
                        attribute_indexes[attr_name],
                    )
                    for function_index, builder in enumerate(consumer_builders)
                    for receiver_name, attr_name in builder.receiver_attribute_refs
                }
            )
        ),
    )


@lru_cache(maxsize=None)
def _compact_class_syntax_facets(
    parsed_module: ParsedModule,
    *,
    collect_autoregister: bool = True,
) -> CompactClassSyntaxFacets:
    syntax_index = module_syntax_index(parsed_module.module)
    file_path = parsed_module.file_path
    collect_autoregister = (
        collect_autoregister
        and not PythonSourcePathPolicy.is_test_path(parsed_module.path)
    )
    builders_by_function_id = (
        {
            id(function): _CompactAutoRegisterFunctionReferenceBuilder(
                qualname=qualname,
                node=function,
                receiver_attribute_refs=set(),
                referenced_symbols=set(),
            )
            for qualname, function in syntax_index.named_functions
        }
        if collect_autoregister
        else {}
    )
    function_bindings_by_id = {
        id(function): LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(function.body)
        | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(function)
        for _qualname, function in syntax_index.named_functions
    }
    module_binds_type = "type" in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
        parsed_module.module.body
    )
    active_function_ids_by_scope = tuple(
        tuple(
            id(syntax_index.depth_first_nodes[function_index])
            for function_index in scope.function_node_indices
        )
        for scope in syntax_index.scopes
    )
    active_builders_by_scope = tuple(
        (
            tuple(
                builders_by_function_id[function_id]
                for function_id in active_function_ids
            )
            if builders_by_function_id
            else ()
        )
        for active_function_ids in active_function_ids_by_scope
    )
    scope_binds_type = tuple(
        module_binds_type
        or any(
            "type" in function_bindings_by_id[function_id]
            for function_id in active_function_ids
        )
        for active_function_ids in active_function_ids_by_scope
    )
    scope_qualnames = tuple(".".join(scope.names) for scope in syntax_index.scopes)
    branch_site_counts_by_function_id: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    case_names_by_function_and_key: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    exact_type_guards: list[CompactExactTypeGuard] = []
    indices_by_type = syntax_index.node_indices_by_type

    if builders_by_function_id:
        for node_index in indices_by_type.get(ast.Attribute, ()):
            node = syntax_index.depth_first_nodes[node_index]
            if not isinstance(node.value, ast.Name):
                continue
            receiver_reference = (node.value.id, node.attr)
            for builder in active_builders_by_scope[syntax_index.scope_ids[node_index]]:
                builder.receiver_attribute_refs.add(receiver_reference)
        for node_index in indices_by_type.get(ast.Call, ()):
            node = syntax_index.depth_first_nodes[node_index]
            if _terminal_reference_name(node.func) != "AutoRegisterMeta":
                continue
            for builder in active_builders_by_scope[syntax_index.scope_ids[node_index]]:
                builder.calls_autoregister_meta = True

    for node_index in indices_by_type.get(ast.If, ()):
        active_function_ids = active_function_ids_by_scope[
            syntax_index.scope_ids[node_index]
        ]
        if not active_function_ids:
            continue
        active_function_id = active_function_ids[-1]
        node = syntax_index.depth_first_nodes[node_index]
        for key_type_name, case_names in _enum_member_refs_by_key_type(
            node.test
        ).items():
            branch_site_counts_by_function_id[active_function_id][key_type_name] += 1
            case_names_by_function_and_key[active_function_id][key_type_name].update(
                case_names
            )

    for node_index in indices_by_type.get(ast.Match, ()):
        active_function_ids = active_function_ids_by_scope[
            syntax_index.scope_ids[node_index]
        ]
        if not active_function_ids:
            continue
        active_function_id = active_function_ids[-1]
        node = syntax_index.depth_first_nodes[node_index]
        refs_by_key: dict[str, set[str]] = defaultdict(set)
        for case in node.cases:
            for key_type_name, case_names in _enum_member_refs_by_key_type(
                case.pattern
            ).items():
                refs_by_key[key_type_name].update(case_names)
            if case.guard is not None:
                for key_type_name, case_names in _enum_member_refs_by_key_type(
                    case.guard
                ).items():
                    refs_by_key[key_type_name].update(case_names)
        for key_type_name, case_names in refs_by_key.items():
            branch_site_counts_by_function_id[active_function_id][key_type_name] += 1
            case_names_by_function_and_key[active_function_id][key_type_name].update(
                case_names
            )

    for node_index in merge(
        indices_by_type.get(ast.If, ()),
        indices_by_type.get(ast.Assert, ()),
    ):
        scope_id = syntax_index.scope_ids[node_index]
        if not active_function_ids_by_scope[scope_id]:
            continue
        node = syntax_index.depth_first_nodes[node_index]
        predicate = _exact_type_predicate(node.test)
        if predicate is None:
            continue
        if isinstance(node, ast.If):
            matches_exact_type_when_true = predicate[2]
            rejects_descendants = (
                not matches_exact_type_when_true and _fail_loud_block(node.body)
            ) or (matches_exact_type_when_true and _fail_loud_block(node.orelse))
            if not rejects_descendants:
                continue
        elif not predicate[2]:
            continue
        if scope_binds_type[scope_id]:
            continue
        subject, type_reference, matches_exact_type_when_true, expression = predicate
        reference_node = ClassSymbolResolutionAuthority.reference_node(type_reference)
        parts = ATTRIBUTE_CHAIN_AUTHORITY.project(reference_node)
        if parts is None:
            continue
        exact_type_guards.append(
            CompactExactTypeGuard(
                file_path=file_path,
                line=node.lineno,
                qualname=scope_qualnames[scope_id],
                subject_expression=ast.unparse(subject),
                type_reference_expression=ast.unparse(type_reference),
                type_reference_parts=parts,
                matches_exact_type_when_true=matches_exact_type_when_true,
                expression=expression,
            )
        )

    builders = (
        tuple(
            builders_by_function_id[id(function)]
            for _qualname, function in syntax_index.named_functions
        )
        if collect_autoregister
        else ()
    )
    for builder in builders:
        if not builder.calls_autoregister_meta:
            continue
        builder.referenced_symbols.update(
            symbol
            for subnode in ast.walk(builder.node)
            for symbol in (
                (
                    subnode.id
                    if isinstance(subnode, ast.Name)
                    else (
                        subnode.value
                        if isinstance(subnode, ast.Constant)
                        and isinstance(subnode.value, str)
                        else (
                            subnode.attr if isinstance(subnode, ast.Attribute) else None
                        )
                    )
                ),
            )
            if symbol is not None
        )
    autoregister_references, autoregister_index = (
        _compact_autoregister_reference_projection(builders)
    )
    closed_axis_functions: list[CompactClosedAxisBranchFunction] = []
    for qualname, function in syntax_index.named_functions:
        function_id = id(function)
        axes = tuple(
            CompactClosedAxisBranchFact(
                key_type_name=key_type_name,
                branch_site_count=branch_site_count,
                case_names=sorted_tuple(
                    case_names_by_function_and_key[function_id][key_type_name]
                ),
            )
            for key_type_name, branch_site_count in sorted(
                branch_site_counts_by_function_id[function_id].items()
            )
        )
        if axes:
            closed_axis_functions.append(
                CompactClosedAxisBranchFunction(
                    file_path=file_path,
                    line=function.lineno,
                    qualname=qualname,
                    axes=axes,
                )
            )
    return CompactClassSyntaxFacets(
        autoregister_function_references=autoregister_references,
        autoregister_reference_index=autoregister_index,
        closed_axis_branch_functions=tuple(closed_axis_functions),
        exact_type_guards=tuple(exact_type_guards),
    )


def _registration_authority_base_name(base_name: str) -> bool:
    tokens = frozenset(
        token.lower()
        for token in re.findall(
            r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+",
            base_name,
        )
        if token
    )
    return bool(
        tokens & {"autoregister", "registered", "registry"}
        or (
            "registration" in tokens
            and bool(tokens & {"authority", "base", "family", "meta", "root"})
        )
        or ("stable" in tokens and bool(tokens & {"axis", "key"}))
        or ("key" in tokens and "family" in tokens)
        or (
            "nominal" in tokens
            and "base" in tokens
            and bool(tokens & {"axis", "family", "formula", "policy"})
        )
    )


def _declares_autoregister_meta(node: ast.ClassDef) -> bool:
    return any(
        terminal_name == "AutoRegisterMeta"
        or terminal_name.endswith("AutoRegisterMeta")
        for keyword in node.keywords
        if keyword.arg == "metaclass"
        if (terminal_name := _terminal_reference_name(keyword.value)) is not None
    )


def _is_registration_authority(node: ast.ClassDef) -> bool:
    assignments = _direct_class_assignments(node)
    inherits_named_authority = any(
        _registration_authority_base_name(terminal_name)
        for base in node.bases
        if (terminal_name := _terminal_reference_name(base)) is not None
    )
    declares_named_authority = (
        "AutoRegister" in node.name
        or "Registered" in node.name
        or node.name.endswith("KeyFamily")
        or _registration_authority_base_name(node.name)
    )
    return bool(
        _declares_autoregister_meta(node)
        or inherits_named_authority
        or declares_named_authority
        or ("__registry__" in assignments and "__registry_key__" in assignments)
        or "stable_key_axis" in assignments
    )


def _is_abstract_class(node: ast.ClassDef) -> bool:
    if {"ABC", "ABCMeta"} & {
        terminal_name
        for base in node.bases
        if (terminal_name := _terminal_reference_name(base)) is not None
    }:
        return True
    return any(
        _terminal_reference_name(decorator) == "abstractmethod"
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in statement.decorator_list
    )


def _is_dataclass_class(node: ast.ClassDef) -> bool:
    return any(
        (isinstance(decorator, ast.Name) and decorator.id == "dataclass")
        or (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "dataclass"
        )
        for decorator in node.decorator_list
    )


def _direct_class_assignment_lines(node: ast.ClassDef) -> list[tuple[str, int]]:
    lines: list[tuple[str, int]] = []
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target,
            ast.Name,
        ):
            lines.append((statement.target.id, statement.lineno))
        elif isinstance(statement, ast.Assign):
            lines.extend(
                (target.id, statement.lineno)
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
    return lines


def _compact_class_value_constructions(
    direct_assignments: dict[str, ast.expr | None],
) -> tuple[CompactClassValueConstruction, ...]:
    return tuple(
        CompactClassValueConstruction(
            assigned_name=assigned_name,
            constructor_name=constructor_name,
            keyword_names=sorted_tuple(
                keyword.arg for keyword in value.keywords if keyword.arg is not None
            ),
            line=value.lineno,
        )
        for assigned_name, value in direct_assignments.items()
        if isinstance(value, ast.Call)
        if (constructor_name := _terminal_reference_name(value.func)) is not None
    )


def _compact_sorted_key_calls(
    parsed_module: ParsedModule,
) -> tuple[CompactSortedKeyCall, ...]:
    calls: list[CompactSortedKeyCall] = []
    for node in _walk_nodes(parsed_module.module):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sorted"
        ):
            continue
        registry_owner_names = sorted_tuple(
            {
                child.value.id
                for argument in node.args
                for child in ast.walk(argument)
                if isinstance(child, ast.Attribute)
                and child.attr == "__registry__"
                and isinstance(child.value, ast.Name)
            }
        )
        key_attribute_names: set[str] = set()
        for keyword in node.keywords:
            if keyword.arg != "key" or keyword.value is None:
                continue
            key_attribute_names.update(
                child.attr
                for child in ast.walk(keyword.value)
                if isinstance(child, ast.Attribute)
            )
            for child in ast.walk(keyword.value):
                if not (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "attrgetter"
                ):
                    continue
                key_attribute_names.update(
                    argument.value
                    for argument in child.args
                    if isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                )
        if key_attribute_names:
            calls.append(
                CompactSortedKeyCall(
                    file_path=parsed_module.file_path,
                    line=node.lineno,
                    registry_owner_names=registry_owner_names,
                    key_attribute_names=sorted_tuple(key_attribute_names),
                )
            )
    return tuple(calls)


def _terminal_reference_name(node: ast.AST) -> str | None:
    parts = ATTRIBUTE_CHAIN_AUTHORITY.project(
        ClassSymbolResolutionAuthority.reference_node(node)
    )
    return None if parts is None else parts[-1]


@dataclass(frozen=True)
class CompactClassFamilyIndexBuilder:
    projections: tuple[CompactModuleClassProjection, ...]

    def build(self) -> CompactClassFamilyIndex:
        records = tuple(
            record for projection in self.projections for record in projection.classes
        )
        known_symbols = frozenset(record.symbol for record in records)
        symbols_by_simple_name_lists: dict[str, list[str]] = defaultdict(list)
        for record in records:
            symbols_by_simple_name_lists[record.simple_name].append(record.symbol)
        unique_symbols_by_name = {
            name: symbols[0]
            for name, symbols in symbols_by_simple_name_lists.items()
            if len(symbols) == 1
        }
        import_aliases_by_module_name = {
            projection.module_name: dict(projection.import_aliases)
            for projection in self.projections
        }
        classes_by_symbol = {
            record.symbol: record.with_resolved_base_symbols(
                tuple(
                    resolved
                    for parts in record.base_reference_parts
                    if (
                        resolved := self._resolved_symbol(
                            parts,
                            record.module_name,
                            import_aliases_by_module_name,
                            known_symbols,
                            unique_symbols_by_name,
                        )
                    )
                    is not None
                )
            )
            for record in records
        }
        children_by_symbol = self._children_by_symbol(classes_by_symbol)
        return CompactClassFamilyIndex(
            classes_by_symbol=classes_by_symbol,
            symbols_by_simple_name={
                name: sorted_tuple(symbols)
                for name, symbols in symbols_by_simple_name_lists.items()
            },
            symbols_by_file_and_qualname={
                (record.file_path, record.qualname): record.symbol
                for record in classes_by_symbol.values()
            },
            children_by_symbol=children_by_symbol,
            ancestors_by_symbol=self._ancestors_by_symbol(classes_by_symbol),
            descendants_by_symbol=self._descendants_by_symbol(
                classes_by_symbol,
                children_by_symbol,
            ),
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
            candidate = ".".join((alias_target, *rest)) if rest else alias_target
            if candidate in known_symbols:
                return candidate
            candidate_parts = candidate.split(".")
            unique_by_suffix = unique_symbols_by_suffix
            if unique_by_suffix is None:
                unique_by_suffix = _unique_known_symbol_by_suffix(
                    frozenset(known_symbols)
                )
            for suffix_width in range(len(candidate_parts) - 1, 0, -1):
                suffix = ".".join(candidate_parts[-suffix_width:])
                match = unique_by_suffix.get(suffix)
                if match is not None:
                    return match
        module_local = ".".join((module_name, *parts))
        if module_local in known_symbols:
            return module_local
        if allow_unique_unqualified and len(parts) == 1:
            return unique_symbols_by_name.get(parts[0])
        return None

    @staticmethod
    def _children_by_symbol(
        classes_by_symbol: dict[str, CompactIndexedClass],
    ) -> dict[str, tuple[str, ...]]:
        children: dict[str, list[str]] = defaultdict(list)
        for record in classes_by_symbol.values():
            for base_symbol in record.resolved_base_symbols:
                children[base_symbol].append(record.symbol)
        return {
            symbol: sorted_tuple(child_symbols)
            for symbol, child_symbols in children.items()
        }

    @staticmethod
    def _ancestors_by_symbol(
        classes_by_symbol: dict[str, CompactIndexedClass],
    ) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        for symbol in sorted(classes_by_symbol):
            ancestors: list[str] = []
            queue = list(classes_by_symbol[symbol].resolved_base_symbols)
            seen: set[str] = set()
            while queue:
                current = queue.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                ancestors.append(current)
                if current in classes_by_symbol:
                    queue.extend(classes_by_symbol[current].resolved_base_symbols)
            if ancestors:
                result[symbol] = tuple(ancestors)
        return result

    @staticmethod
    def _descendants_by_symbol(
        classes_by_symbol: dict[str, CompactIndexedClass],
        children_by_symbol: dict[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        for symbol in sorted(classes_by_symbol):
            descendants: list[str] = []
            queue = list(children_by_symbol.get(symbol, ()))
            seen: set[str] = set()
            while queue:
                current = queue.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                descendants.append(current)
                queue.extend(children_by_symbol.get(current, ()))
            if descendants:
                result[symbol] = tuple(descendants)
        return result


def build_compact_class_family_index(
    projections: tuple[CompactModuleClassProjection, ...],
) -> CompactClassFamilyIndex:
    """Build an exact inheritance graph from AST-free per-module facts."""

    return CompactClassFamilyIndexBuilder(projections).build()


@dataclass(frozen=True)
class CompactClassReferenceResolver:
    import_aliases_by_module_name: dict[str, dict[str, str]]
    known_symbols: dict[str, CompactIndexedClass]
    unique_symbols_by_name: dict[str, str]
    unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex

    @classmethod
    def from_index(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        class_index: CompactClassFamilyIndex,
    ) -> "CompactClassReferenceResolver":
        return cls(
            import_aliases_by_module_name={
                projection.module_name: dict(projection.import_aliases)
                for projection in projections
            },
            known_symbols=class_index.classes_by_symbol,
            unique_symbols_by_name={
                name: symbols[0]
                for name, symbols in class_index.symbols_by_simple_name.items()
                if len(symbols) == 1
            },
            unique_symbols_by_suffix=(
                _UniqueKnownSymbolSuffixIndex.from_terminal_buckets(
                    class_index.symbols_by_simple_name
                )
            ),
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
class ClassSymbolResolutionAuthority:
    """Resolve AST name chains to indexed class symbols under an explicit policy."""

    NOMINAL_FAMILY_NEUTRAL_BASE_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"ABC", "Generic", "Protocol", "object"}
    )

    parsed_module: ParsedModule
    import_aliases: dict[str, str]
    known_symbols: frozenset[str]
    unique_symbols_by_name: dict[str, str]
    allow_unique_unqualified: bool

    def symbol_for_node(self, node: ast.AST) -> str | None:
        parts = ATTRIBUTE_CHAIN_AUTHORITY.project(self.reference_node(node))
        if parts is None:
            return None
        alias_symbol = self._import_alias_symbol(parts)
        if alias_symbol is not None:
            return alias_symbol
        module_local_symbol = self._module_local_symbol(parts)
        if module_local_symbol is not None:
            return module_local_symbol
        if self.allow_unique_unqualified:
            return self._unique_unqualified_symbol(parts)
        return None

    @staticmethod
    def reference_node(node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Subscript):
            return node.value
        return node

    def _import_alias_symbol(self, parts: tuple[str, ...]) -> str | None:
        first, *rest = parts
        alias_target = self.import_aliases.get(first)
        if alias_target is None:
            return None
        candidate = ".".join((alias_target, *rest)) if rest else alias_target
        if candidate in self.known_symbols:
            return candidate
        # A scan may start at a package subdirectory, making indexed symbols
        # source-root-relative while imports retain their full package prefix.
        # Resolve only a unique suffix match so unrelated same-named classes do
        # not create a speculative inheritance edge.
        candidate_parts = candidate.split(".")
        unique_symbol_by_suffix = _unique_known_symbol_by_suffix(self.known_symbols)
        for suffix_width in range(len(candidate_parts) - 1, 0, -1):
            suffix = ".".join(candidate_parts[-suffix_width:])
            match = unique_symbol_by_suffix.get(suffix)
            if match is not None:
                return match
        return None

    def _module_local_symbol(self, parts: tuple[str, ...]) -> str | None:
        candidate = ".".join((self.parsed_module.module_name, *parts))
        if candidate in self.known_symbols:
            return candidate
        return None

    def _unique_unqualified_symbol(self, parts: tuple[str, ...]) -> str | None:
        if len(parts) != 1:
            return None
        return self.unique_symbols_by_name.get(parts[0])

    @classmethod
    def declared_base_name(cls, node: ast.AST) -> str | None:
        reference_node = cls.reference_node(node)
        if ATTRIBUTE_CHAIN_AUTHORITY.project(reference_node) is None:
            return None
        return ast.unparse(reference_node)

    @classmethod
    def establishes_nominal_family(cls, declared_base_name: str) -> bool:
        """Return whether a base declaration can own a domain class family."""

        return (
            declared_base_name.rsplit(".", 1)[-1]
            not in cls.NOMINAL_FAMILY_NEUTRAL_BASE_NAMES
        )


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
        return _module_import_aliases(self.parsed_module)

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


def build_class_family_index(modules: list[ParsedModule]) -> ClassFamilyIndex:
    return _build_class_family_index_cached(tuple(modules))


def overlay_class_family_index(
    base_index: ClassFamilyIndex,
    changed_modules: tuple[ParsedModule, ...],
) -> ClassFamilyIndex:
    """Rebuild a class index by replacing changed-module class records."""

    changed_path_texts = _resolved_module_path_texts(changed_modules)
    unchanged_records = base_index.class_records_excluding_files(changed_path_texts)
    return ClassFamilyIndexBuilder(
        changed_modules,
        base_records=unchanged_records,
    ).build()


@lru_cache(maxsize=None)
def _build_class_family_index_cached(
    modules: tuple[ParsedModule, ...],
) -> ClassFamilyIndex:
    return ClassFamilyIndexBuilder(modules).build()


@dataclass(frozen=True)
class ClassFamilyIndexBuilder:
    modules: tuple[ParsedModule, ...]
    base_records: tuple[IndexedClass, ...] = ()

    def build(self) -> ClassFamilyIndex:
        class_records = (*self.base_records, *self.module_class_records())
        known_symbols = frozenset(record.symbol for record in class_records)
        symbols_by_simple_name_multimap = self.symbols_by_simple_name_multimap(
            class_records
        )
        unique_symbols_by_name = {
            name: symbols[0]
            for name, symbols in symbols_by_simple_name_multimap.items()
            if len(symbols) == 1
        }
        classes_by_symbol = {
            record.symbol: self.resolved_record(
                record,
                known_symbols,
                unique_symbols_by_name,
            )
            for record in class_records
        }
        symbols_by_file_and_qualname = {
            (record.file_path, record.qualname): record.symbol
            for record in classes_by_symbol.values()
        }
        children_by_symbol = self.children_by_symbol(classes_by_symbol)
        ancestors_by_symbol = self.ancestors_by_symbol(classes_by_symbol)
        descendants_by_symbol = self.descendants_by_symbol(
            classes_by_symbol,
            children_by_symbol,
        )
        return ClassFamilyIndex(
            classes_by_symbol=classes_by_symbol,
            symbols_by_simple_name={
                name: sorted_tuple(symbols)
                for name, symbols in symbols_by_simple_name_multimap.items()
            },
            symbols_by_file_and_qualname=symbols_by_file_and_qualname,
            children_by_symbol=children_by_symbol,
            ancestors_by_symbol=ancestors_by_symbol,
            descendants_by_symbol=descendants_by_symbol,
        )

    def module_class_records(self) -> tuple[IndexedClass, ...]:
        records: list[IndexedClass] = []
        for parsed_module in self.modules:
            for qualname, node in _iter_class_defs(list(parsed_module.module.body)):
                records.append(
                    IndexedClass.from_parsed_class(parsed_module, qualname, node)
                )
        return tuple(records)

    @staticmethod
    def symbols_by_simple_name_multimap(
        class_records: tuple[IndexedClass, ...],
    ) -> dict[str, list[str]]:
        symbols_by_simple_name_multimap: dict[str, list[str]] = defaultdict(list)
        for record in class_records:
            symbols_by_simple_name_multimap[record.simple_name].append(record.symbol)
        return symbols_by_simple_name_multimap

    def resolved_record(
        self,
        record: IndexedClass,
        known_symbols: frozenset[str],
        unique_symbols_by_name: dict[str, str],
    ) -> IndexedClass:
        parsed_module = self.parsed_module_by_name.get(record.module_name)
        if parsed_module is None:
            return self.base_record_with_current_bases(record, known_symbols)
        base_resolution = ClassSymbolResolutionAuthority(
            parsed_module=parsed_module,
            import_aliases=_module_import_aliases(parsed_module),
            known_symbols=known_symbols,
            unique_symbols_by_name=unique_symbols_by_name,
            allow_unique_unqualified=True,
        )
        return record.with_resolved_base_symbols(
            tuple(
                resolved
                for base in record.node.bases
                if (resolved := base_resolution.symbol_for_node(base)) is not None
            )
        )

    @cached_property
    def parsed_module_by_name(self) -> dict[str, ParsedModule]:
        return {module.module_name: module for module in self.modules}

    @staticmethod
    def base_record_with_current_bases(
        record: IndexedClass,
        known_symbols: frozenset[str],
    ) -> IndexedClass:
        return record.with_resolved_base_symbols(
            tuple(
                base_symbol
                for base_symbol in record.resolved_base_symbols
                if base_symbol in known_symbols
            )
        )

    @staticmethod
    def children_by_symbol(
        classes_by_symbol: dict[str, IndexedClass],
    ) -> dict[str, tuple[str, ...]]:
        children_by_symbol_lists: dict[str, list[str]] = defaultdict(list)
        for record in classes_by_symbol.values():
            for base_symbol in record.resolved_base_symbols:
                children_by_symbol_lists[base_symbol].append(record.symbol)
        return {
            symbol: sorted_tuple(children)
            for symbol, children in children_by_symbol_lists.items()
        }

    @staticmethod
    def ancestors_by_symbol(
        classes_by_symbol: dict[str, IndexedClass],
    ) -> dict[str, tuple[str, ...]]:
        ancestors_by_symbol: dict[str, tuple[str, ...]] = {}
        for symbol in sorted(classes_by_symbol):
            ancestors: list[str] = []
            queue = list(classes_by_symbol[symbol].resolved_base_symbols)
            seen: set[str] = set()
            while queue:
                current = queue.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                ancestors.append(current)
                indexed_class = classes_by_symbol.get(current)
                if indexed_class is not None:
                    queue.extend(indexed_class.resolved_base_symbols)
            if ancestors:
                ancestors_by_symbol[symbol] = tuple(ancestors)
        return ancestors_by_symbol

    @staticmethod
    def descendants_by_symbol(
        classes_by_symbol: dict[str, IndexedClass],
        children_by_symbol: dict[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        descendants_by_symbol: dict[str, tuple[str, ...]] = {}
        for symbol in sorted(classes_by_symbol):
            descendants: list[str] = []
            queue = (
                list(children_by_symbol[symbol]) if symbol in children_by_symbol else []
            )
            seen: set[str] = set()
            while queue:
                current = queue.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                descendants.append(current)
                if current in children_by_symbol:
                    queue.extend(children_by_symbol[current])
            if descendants:
                descendants_by_symbol[symbol] = tuple(descendants)
        return descendants_by_symbol


def _resolved_module_path_texts(modules: tuple[ParsedModule, ...]) -> frozenset[str]:
    return frozenset(module.resolved_file_path for module in modules)
