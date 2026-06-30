"""AST parsing, registration, and collection substrate.

This module provides the reusable machinery that turns Python source into parsed
modules, registered observation/spec families, and collected semantic shapes.
Most higher-level detectors depend on this substrate rather than walking raw ASTs
directly.
"""

from __future__ import annotations

import ast
import copy
import os
import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import StrEnum
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
from types import EllipsisType
from typing import Callable, ClassVar, Generic, TypeAlias, TypeVar, cast

from metaclass_registry import AutoRegisterMeta

from .cache_paths import ParseCacheDirectory, default_parse_cache_dir
from .collection_algebra import sorted_tuple
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from .semantic_match import (
    AstTypedEffectStep,
    GuardedEffectStep,
    Maybe,
    RegisteredEffectStep,
    SingleCompareEffectStep,
    as_ast,
    named_value_binding,
    registered_effect_steps,
    single_assign_target,
    single_call_arg,
    single_item,
    single_return_call,
)
from .observation_graph import (
    NominalWitnessGroup,
    ObservationCohort,
    ObservationFiber,
    ObservationGraph,
    ObservationKind,
    StructuralExecutionLevel,
    StructuralObservation,
    StructuralObservationCarrier,
    build_observation_graph,
    collect_structural_observations,
)
from .observation_shapes import (
    AccessorWrapperCandidate,
    AttributeProbeObservation,
    BuilderCallShape,
    ClassMarkerObservation,
    ConfigDispatchObservation,
    DualAxisResolutionObservation,
    DynamicMethodInjectionObservation,
    ExportDictShape,
    FieldObservation,
    FieldOriginKind,
    InterfaceGenerationObservation,
    LineageMappingObservation,
    LiteralDispatchObservation,
    LiteralKind,
    MethodShape,
    ProjectionHelperShape,
    RegistrationShape,
    RuntimeTypeGenerationObservation,
    ScopedShapeWrapperFunction,
    ScopedShapeWrapperSpec,
    SentinelTypeObservation,
)

_TYPE_BUILTIN = "type"
_SETATTR_BUILTIN = "setattr"
_RUNTIME_TYPE_GENERATORS = frozenset({_TYPE_BUILTIN, "make_dataclass", "new_class"})
_IGNORED_PYTHON_TREE_DIRS = frozenset(
    {
        ".eggs",
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".svn",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
        "site-packages",
        "venv",
    }
)
_DEFAULT_PARSE_WORKERS = 1
_MAX_AUTO_PARSE_WORKERS = 16


@dataclass(frozen=True)
class AstParseCacheSchema:
    """Nominal schema identity for persisted Python AST cache entries."""

    version: int = 1


class AstCachePayloadUnavailable:
    """Sentinel for unreadable or incompatible persisted AST cache payloads."""


@dataclass(frozen=True)
class AstParseCachePayload:
    """Persisted AST parse-cache entry for one source file signature."""

    version: int
    path: str
    mtime_ns: int
    size: int
    source_signature: str
    python_version: tuple[int, int]
    module: ast.Module

    def matches(
        self,
        path: Path,
        path_stat: os.stat_result,
        source_signature: str,
    ) -> bool:
        return (
            self.version == ast_parse_cache_schema.version
            and self.path == str(path.resolve())
            and self.source_signature == source_signature
            and self.mtime_ns == path_stat.st_mtime_ns
            and self.size == path_stat.st_size
            and self.python_version == (sys.version_info.major, sys.version_info.minor)
        )


ast_parse_cache_schema = AstParseCacheSchema()
ast_cache_payload_unavailable = AstCachePayloadUnavailable()


@dataclass(frozen=True)
class CollectedFamilyCacheSchema:
    """Schema identity for persisted collected-family item projections."""

    version: int = 2
    max_payload_bytes: int = 100_000


@dataclass(frozen=True)
class CollectedFamilyCacheIdentity:
    """Invalidation identity for one collected family in one parsed module."""

    path: str
    module_name: str
    source_signature: str
    family_module: str
    family_qualname: str
    item_type_module: str
    item_type_qualname: str
    python_version: tuple[int, int]
    schema: CollectedFamilyCacheSchema

    @property
    def cache_token(self) -> str:
        payload = repr(self).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=16).hexdigest()


collected_family_cache_schema = CollectedFamilyCacheSchema()


@dataclass(frozen=True, kw_only=True)
class PythonModuleParseContext(ParseCacheDirectory):
    """Parse-time context shared by sequential and concurrent module loading."""

    analysis_root: Path


@dataclass(frozen=True)
class PythonModulePathIdentity:
    """Module import identity derived from one source path and analysis root."""

    path: Path
    import_name: str
    is_package_init: bool

    @classmethod
    def from_path(
        cls,
        path: Path,
        analysis_root: Path,
    ) -> "PythonModulePathIdentity":
        relative = path.relative_to(analysis_root)
        module_parts = list(relative.with_suffix("").parts)
        is_package_init = bool(module_parts and module_parts[-1] == "__init__")
        if is_package_init:
            module_parts = module_parts[:-1]
        import_name = ".".join(module_parts) if module_parts else "__init__"
        return cls(
            path=path,
            import_name=import_name,
            is_package_init=is_package_init,
        )


def _source_signature(source: str) -> str:
    return hashlib.blake2s(source.encode("utf-8"), digest_size=16).hexdigest()


def _cache_entry_path(cache_dir: Path, path: Path) -> Path:
    token = hashlib.blake2s(
        str(path.resolve()).encode("utf-8"), digest_size=16
    ).hexdigest()
    return cache_dir / f"{token}.pickle"


def _load_cached_ast(
    path: Path,
    source_signature: str,
    *,
    cache_dir: Path | None = None,
) -> ast.Module | None:
    if cache_dir is None:
        return None
    try:
        path_stat = path.stat()
    except OSError:
        return None
    cache_path = _cache_entry_path(cache_dir, path)
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except (
        FileNotFoundError,
        OSError,
        pickle.PickleError,
        EOFError,
        TypeError,
        ValueError,
        AttributeError,
        ImportError,
    ):
        payload = ast_cache_payload_unavailable
    if not isinstance(payload, AstParseCachePayload):
        return None
    if not payload.matches(path, path_stat, source_signature):
        return None
    return payload.module


def _write_cached_ast(
    path: Path,
    module: ast.Module,
    source_signature: str,
    *,
    cache_dir: Path | None = None,
) -> None:
    if cache_dir is None:
        return
    try:
        path_stat = path.stat()
    except OSError:
        return
    cache_entry = _cache_entry_path(cache_dir, path)
    payload = AstParseCachePayload(
        version=ast_parse_cache_schema.version,
        path=str(path.resolve()),
        mtime_ns=path_stat.st_mtime_ns,
        size=path_stat.st_size,
        source_signature=source_signature,
        python_version=(sys.version_info.major, sys.version_info.minor),
        module=module,
    )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_entry.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        return


def _parse_source_module(
    path: Path,
    *,
    context: PythonModuleParseContext,
) -> ParsedModule:
    source = path.read_text(encoding="utf-8")
    source_signature = _source_signature(source)
    module = (
        _load_cached_ast(path, source_signature, cache_dir=context.parse_cache_dir)
        if context.use_parse_cache
        else None
    )
    if module is None:
        module = ast.parse(source, filename=str(path))
        if context.use_parse_cache:
            _write_cached_ast(
                path,
                module,
                source_signature,
                cache_dir=context.parse_cache_dir,
            )
    module_identity = PythonModulePathIdentity.from_path(
        path,
        analysis_root=context.analysis_root,
    )
    return ParsedModule(
        path=path,
        module_name=module_identity.import_name,
        is_package_init=module_identity.is_package_init,
        module=module,
        source=source,
        family_cache_dir=context.collected_family_cache_dir,
    )


def _effective_parse_workers(parse_workers: int) -> int:
    if parse_workers <= 0:
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 1
        return min(_MAX_AUTO_PARSE_WORKERS, cpu_count)
    return max(1, parse_workers)


@dataclass(frozen=True)
class ParsedModule:
    """Parsed Python module together with its source text and path."""

    path: Path
    module_name: str
    is_package_init: bool
    module: ast.Module
    source: str
    family_cache_dir: Path | None = None


@dataclass(frozen=True)
class AstNameFamily:
    names: frozenset[str]


class BuiltinCallName(StrEnum):
    """Built-in call names that detector and AST helpers treat semantically."""

    ABS = "abs"
    ALL = "all"
    ANY = "any"
    BOOL = "bool"
    BYTEARRAY = "bytearray"
    BYTES = "bytes"
    DICT = "dict"
    ENUMERATE = "enumerate"
    FLOAT = "float"
    FROZENSET = "frozenset"
    INT = "int"
    ISINSTANCE = "isinstance"
    ITER = "iter"
    LEN = "len"
    LIST = "list"
    MAP = "map"
    MAX = "max"
    MEMORYVIEW = "memoryview"
    MIN = "min"
    NEXT = "next"
    OBJECT = "object"
    OPEN = "open"
    PRINT = "print"
    RANGE = "range"
    SET = "set"
    SORTED = "sorted"
    STR = "str"
    SUM = "sum"
    TUPLE = "tuple"
    TYPE = "type"
    ZIP = "zip"

    @classmethod
    def sequence_wrapper_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset((cls.TUPLE, cls.LIST, cls.SET))

    @classmethod
    def collection_factory_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset((cls.TUPLE, cls.LIST, cls.SET, cls.FROZENSET))

    @classmethod
    def return_collection_kind_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset((cls.TUPLE, cls.LIST, cls.DICT))

    @classmethod
    def self_attribute_wrapper_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset(
            (
                cls.BOOL,
                cls.FROZENSET,
                cls.INT,
                cls.LEN,
                cls.LIST,
                cls.SET,
                cls.SORTED,
                cls.STR,
                cls.TUPLE,
            )
        )

    @classmethod
    def non_helper_call_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset(
            (
                cls.ALL,
                cls.ANY,
                cls.BOOL,
                cls.DICT,
                cls.FROZENSET,
                cls.INT,
                cls.LEN,
                cls.LIST,
                cls.MAX,
                cls.MIN,
                cls.SET,
                cls.SORTED,
                cls.STR,
                cls.SUM,
                cls.TUPLE,
            )
        )

    @classmethod
    def integer_result_call_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset((cls.LEN, cls.MAX, cls.MIN, cls.SUM))

    @classmethod
    def structural_alias_root_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset((cls.DICT, cls.FROZENSET, cls.LIST, cls.SET, cls.TUPLE))

    @classmethod
    def structural_alias_leaf_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset(
            (
                cls.BOOL,
                cls.BYTES,
                cls.DICT,
                cls.FLOAT,
                cls.FROZENSET,
                cls.INT,
                cls.LIST,
                cls.SET,
                cls.STR,
                cls.TUPLE,
            )
        )

    @classmethod
    def schema_accessor_copy_call_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset((cls.DICT, cls.LIST, cls.SET, cls.TUPLE))

    @classmethod
    def role_surface_iteration_call_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset(
            (
                cls.ALL,
                cls.ANY,
                cls.DICT,
                cls.ENUMERATE,
                cls.LEN,
                cls.LIST,
                cls.MAX,
                cls.MIN,
                cls.SET,
                cls.SUM,
                cls.TUPLE,
                cls.ZIP,
            )
        )

    @classmethod
    def non_lifecycle_stage_call_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset(
            (
                cls.ANY,
                cls.DICT,
                cls.FROZENSET,
                cls.ISINSTANCE,
                cls.ITER,
                cls.LEN,
                cls.LIST,
                cls.NEXT,
                cls.SET,
                cls.TUPLE,
                cls.TYPE,
            )
        )

    @classmethod
    def formula_builtin_callee_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset((cls.ABS, cls.ALL, cls.ANY, cls.MAX, cls.MIN, cls.SUM))

    @classmethod
    def smelly_type_alias_builtin_tokens(cls) -> frozenset["BuiltinCallName"]:
        return frozenset((cls.DICT, cls.LIST, cls.MAP, cls.SET, cls.TUPLE))

    @classmethod
    def isinstance_scatter_builtin_type_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset(
            (
                cls.BYTEARRAY,
                cls.BOOL,
                cls.BYTES,
                cls.DICT,
                cls.FLOAT,
                cls.FROZENSET,
                cls.INT,
                cls.LIST,
                cls.MEMORYVIEW,
                cls.OBJECT,
                cls.SET,
                cls.STR,
                cls.TUPLE,
                cls.TYPE,
            )
        )

    @classmethod
    def normalized_template_stable_builtin_names(cls) -> frozenset["BuiltinCallName"]:
        return frozenset(
            (
                cls.DICT,
                cls.ENUMERATE,
                cls.FLOAT,
                cls.INT,
                cls.LEN,
                cls.LIST,
                cls.MAX,
                cls.MIN,
                cls.OPEN,
                cls.PRINT,
                cls.RANGE,
                cls.SET,
                cls.SORTED,
                cls.STR,
                cls.SUM,
                cls.TUPLE,
            )
        )


@dataclass(frozen=True)
class AstCallObservation:
    call: ast.Call
    matched_name: str


@dataclass(frozen=True)
class _BuilderCallContext:
    call: ast.Call
    callee_name: str
    field_pairs: tuple[tuple[str, ast.AST], ...]


@dataclass(frozen=True)
class _ExportDictContext:
    dict_node: ast.Dict
    key_pairs: tuple[tuple[str, ast.AST], ...]


@dataclass(frozen=True)
class _ScopedShapeSpecCall:
    spec_name: str
    call: ast.Call


@dataclass(frozen=True)
class _ScopedShapeSpecKeywords:
    function_name: str
    node_types: tuple[str, ...]


AstScopedNode: TypeAlias = ast.AST
CollectedFamilyTypes: TypeAlias = tuple[type["CollectedFamily"], ...]


class ScopedAstObservationRole(StrEnum):
    """Semantic scope roles owned by ScopedAstObservation's field schema."""

    SCOPE_FILTERED = "scope_filtered"
    CLASS_SCOPE = "class_scope"
    FUNCTION_SCOPE = "function_scope"
    NODE_TYPE = "node_type"
    GENERIC_SCOPE = "generic_scope"
    MODULE_ONLY_GUARD = "module_only_guard"
    CLASS_ONLY_GUARD = "class_only_guard"
    MODULE_SCOPE_GUARD = "module_scope_guard"
    FUNCTION_SCOPE_GUARD = "function_scope_guard"
    NODE_TYPE_GUARD = "node_type_guard"
    GUARDED_DELEGATE = "guarded_delegate"


@dataclass(frozen=True)
class ScopedAstObservation:
    node: AstScopedNode
    class_name: str | None
    function_name: str | None

    @classmethod
    def class_scope_field_name(cls) -> str:
        return single_item(
            tuple(
                field.name for field in fields(cls) if field.name.startswith("class_")
            )
        )

    @classmethod
    def function_scope_field_name(cls) -> str:
        return single_item(
            tuple(
                field.name
                for field in fields(cls)
                if field.name.startswith("function_")
            )
        )

    @classmethod
    def scope_role_name_from_text(cls, text: str) -> str:
        class_field_name = cls.class_scope_field_name()
        function_field_name = cls.function_scope_field_name()
        mentions_class = class_field_name in text
        mentions_function = function_field_name in text
        if mentions_class and mentions_function:
            return ScopedAstObservationRole.SCOPE_FILTERED.value
        if mentions_class:
            return ScopedAstObservationRole.CLASS_SCOPE.value
        if mentions_function:
            return ScopedAstObservationRole.FUNCTION_SCOPE.value
        if "isinstance" in text:
            return ScopedAstObservationRole.NODE_TYPE.value
        return ScopedAstObservationRole.GENERIC_SCOPE.value

    @classmethod
    def guard_role_name_from_text(cls, text: str) -> str:
        class_ref = f"observation.{cls.class_scope_field_name()}"
        function_ref = f"observation.{cls.function_scope_field_name()}"
        if f"{class_ref} is not None" in text:
            return ScopedAstObservationRole.MODULE_ONLY_GUARD.value
        if f"{class_ref} is None" in text:
            return ScopedAstObservationRole.CLASS_ONLY_GUARD.value
        if f"{function_ref} is None" in text:
            return ScopedAstObservationRole.MODULE_SCOPE_GUARD.value
        if f"{function_ref} is not None" in text:
            return ScopedAstObservationRole.FUNCTION_SCOPE_GUARD.value
        if "isinstance" in text:
            return ScopedAstObservationRole.NODE_TYPE_GUARD.value
        return ScopedAstObservationRole.GUARDED_DELEGATE.value


@dataclass(frozen=True)
class ClassAstObservation:
    node: ast.ClassDef
    is_dataclass_family: bool


class ClassFunctionStackNodeVisitor(ast.NodeVisitor, ABC):
    """Nominal AST visitor base that owns class/function scope stack lifecycle."""

    def __init__(self) -> None:
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []

    @property
    def current_class_name(self) -> str | None:
        return self.class_stack[-1] if self.class_stack else None

    @property
    def current_function_name(self) -> str | None:
        return self.function_stack[-1] if self.function_stack else None

    @property
    def qualname(self) -> str:
        return ".".join((*self.class_stack, *self.function_stack)) or "<module>"

    def before_visit_class(self, node: ast.ClassDef) -> None:
        del node

    def before_visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        del node

    def traverse_statements(self, body: list[ast.stmt]) -> None:
        for statement in body:
            self.visit(statement)

    def traverse_trimmed_statements(self, body: list[ast.stmt]) -> None:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        self.traverse_statements(body)

    def traverse_node_body(
        self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        self.generic_visit(node)

    def traverse_trimmed_node_body(
        self, node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        self.traverse_trimmed_statements(node.body)

    traverse_class_body = traverse_node_body
    traverse_function_body = traverse_node_body

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.before_visit_class(node)
        self.class_stack.append(node.name)
        try:
            self.traverse_class_body(node)
        finally:
            self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.before_visit_function(node)
        self.function_stack.append(node.name)
        try:
            self.traverse_function_body(node)
        finally:
            self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef


_TRegistered = TypeVar("_TRegistered")
_TRegisteredType = TypeVar("_TRegisteredType")
ShapeItemT = TypeVar("ShapeItemT")
FlattenedItemT = TypeVar("FlattenedItemT")
ShapeEmission: TypeAlias = ShapeItemT | tuple[ShapeItemT, ...]
ContextShapeHelperArg: TypeAlias = ast.AST | str | None
LiteralDispatchScalar: TypeAlias = str | int
LiteralConstantValue: TypeAlias = str | int | float | complex | bool | bytes | None


@dataclass(frozen=True)
class CollectedFamilyCachePayload(Generic[ShapeItemT]):
    """Persisted items collected for one module/family pair."""

    identity: CollectedFamilyCacheIdentity
    items: tuple[ShapeItemT, ...]


def _registry_member_key(registered_type: type[_TRegistered]) -> tuple[str, int, str]:
    return (
        registered_type.__module__,
        cast(int, registered_type.__dict__.get("__firstlineno__", 0)),
        registered_type.__qualname__,
    )


def _registered_type_token(_name: str, cls: type[_TRegistered]) -> str | None:
    if cls.__dict__.get("_registry_skip", False):
        return None
    return f"{cls.__module__}:{cls.__qualname__}"


def _is_direct_registered_descendant(
    candidate: type[_TRegistered],
    root: type[_TRegisteredType],
    *,
    registry_base: type,
) -> bool:
    if not issubclass(candidate, root):
        return False
    if not root.__dict__.get("_registry_root", False):
        return True
    for ancestor in candidate.__mro__[1:]:
        if ancestor is root:
            return True
        if not issubclass(ancestor, registry_base):
            continue
        if ancestor.__dict__.get("_registry_root", False):
            return False
    return False


class RegisteredTypeLineage:
    def descendant_types(
        self, root: type[_TRegistered]
    ) -> tuple[type[_TRegistered], ...]:
        seen: set[type] = set()
        ordered: list[type[_TRegistered]] = []
        queue = list(root.__subclasses__())
        while queue:
            current = queue.pop(0)
            queue.extend(current.__subclasses__())
            if current in seen:
                continue
            seen.add(current)
            ordered.append(cast(type[_TRegistered], current))
        return tuple(ordered)

    def ordered_registered_types(
        self,
        root: type[_TRegisteredType],
    ) -> tuple[type[_TRegisteredType], ...]:
        registry = root.__registry__
        seen: set[type[_TRegisteredType]] = set()
        ordered: list[type[_TRegisteredType]] = []
        for registered_type in sorted(
            registry.values(),
            key=_registry_member_key,
        ):
            registered_class = cast(type[_TRegisteredType], registered_type)
            if registered_class in seen or not issubclass(registered_class, root):
                continue
            seen.add(registered_class)
            ordered.append(registered_class)
        return tuple(ordered)

    def direct_registered_types(
        self, root: type[_TRegisteredType], *, registry_base: type
    ) -> tuple[type[_TRegisteredType], ...]:
        return tuple(
            (
                registered_type
                for registered_type in self.ordered_registered_types(root)
                if _is_direct_registered_descendant(
                    registered_type, root, registry_base=registry_base
                )
            )
        )


REGISTERED_TYPE_LINEAGE = RegisteredTypeLineage()


class ModuleShapeSpec(Generic[ShapeItemT], ABC):
    """Abstract collector that emits semantic items from one parsed module."""

    @abstractmethod
    def collect(self, parsed_module: ParsedModule) -> list[ShapeEmission[ShapeItemT]]:
        raise NotImplementedError


class SharedRegistryRootBase:
    __registry_key__ = "__registry_token__"
    __key_extractor__ = _registered_type_token
    _registry_root: ClassVar[bool] = False


class AutoRegisteredModuleShapeSpec(
    SharedRegistryRootBase,
    ModuleShapeSpec[ShapeItemT],
    Generic[ShapeItemT],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Module shape spec family whose concrete subclasses self-register."""

    __registry__: ClassVar[dict[str, type["AutoRegisteredModuleShapeSpec"]]] = {}
    __skip_if_no_key__ = True

    @classmethod
    def registered_specs(cls) -> tuple["AutoRegisteredModuleShapeSpec", ...]:
        """Return concrete specs registered directly under this root."""
        return tuple(
            (
                spec_type()
                for spec_type in REGISTERED_TYPE_LINEAGE.direct_registered_types(
                    cls, registry_base=AutoRegisteredModuleShapeSpec
                )
            )
        )

    @classmethod
    def all_registered_specs(cls) -> tuple["AutoRegisteredModuleShapeSpec", ...]:
        """Return all concrete specs reachable from descendant registry roots."""
        return tuple(
            (
                spec_type()
                for spec_type in REGISTERED_TYPE_LINEAGE.ordered_registered_types(cls)
            )
        )


class CollectedFamily(
    SharedRegistryRootBase,
    Generic[ShapeItemT],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered family of collected items keyed by a runtime item type."""

    __registry__: ClassVar[dict[str, type["CollectedFamily"]]] = {}
    __skip_if_no_key__ = True
    item_type: ClassVar[type[ShapeItemT]]

    @classmethod
    def registered_families(cls) -> CollectedFamilyTypes:
        """Return concrete families registered directly under this root."""
        return REGISTERED_TYPE_LINEAGE.direct_registered_types(
            cls, registry_base=CollectedFamily
        )

    @classmethod
    def all_registered_families(cls) -> CollectedFamilyTypes:
        """Return all concrete families reachable from descendant registry roots."""
        return REGISTERED_TYPE_LINEAGE.ordered_registered_types(cls)

    @classmethod
    @abstractmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ShapeItemT]:
        raise NotImplementedError


def _collected_family_cache_identity(
    parsed_module: ParsedModule,
    family: type[CollectedFamily[ShapeItemT]],
) -> CollectedFamilyCacheIdentity:
    item_type = family.item_type
    return CollectedFamilyCacheIdentity(
        path=str(parsed_module.path.resolve()),
        module_name=parsed_module.module_name,
        source_signature=_source_signature(parsed_module.source),
        family_module=family.__module__,
        family_qualname=family.__qualname__,
        item_type_module=item_type.__module__,
        item_type_qualname=item_type.__qualname__,
        python_version=(sys.version_info.major, sys.version_info.minor),
        schema=collected_family_cache_schema,
    )


def _collected_family_cache_path(
    cache_dir: Path,
    identity: CollectedFamilyCacheIdentity,
) -> Path:
    return cache_dir / f"{identity.cache_token}.pickle"


def _load_cached_collected_family_items(
    parsed_module: ParsedModule,
    family: type[CollectedFamily[ShapeItemT]],
) -> tuple[ShapeItemT, ...] | None:
    cache_dir = parsed_module.family_cache_dir
    if cache_dir is None:
        return None
    identity = _collected_family_cache_identity(parsed_module, family)
    try:
        with _collected_family_cache_path(cache_dir, identity).open("rb") as handle:
            payload = pickle.load(handle)
    except (
        FileNotFoundError,
        OSError,
        pickle.PickleError,
        EOFError,
        TypeError,
        ValueError,
        AttributeError,
        ImportError,
    ):
        return None
    if not isinstance(payload, CollectedFamilyCachePayload):
        return None
    if payload.identity != identity:
        return None
    if not all(isinstance(item, family.item_type) for item in payload.items):
        return None
    return cast(tuple[ShapeItemT, ...], payload.items)


def _store_cached_collected_family_items(
    parsed_module: ParsedModule,
    family: type[CollectedFamily[ShapeItemT]],
    items: tuple[ShapeItemT, ...],
) -> None:
    cache_dir = parsed_module.family_cache_dir
    if cache_dir is None:
        return
    identity = _collected_family_cache_identity(parsed_module, family)
    payload = CollectedFamilyCachePayload(identity=identity, items=items)
    try:
        payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload_bytes) > identity.schema.max_payload_bytes:
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        with _collected_family_cache_path(cache_dir, identity).open("wb") as handle:
            handle.write(payload_bytes)
    except (OSError, pickle.PickleError, TypeError, AttributeError):
        return


@lru_cache(maxsize=None)
def _collect_family_items_cached(
    parsed_module: ParsedModule, family: type[CollectedFamily[ShapeItemT]]
) -> tuple[ShapeItemT, ...]:
    cached_items = _load_cached_collected_family_items(parsed_module, family)
    if cached_items is not None:
        return cached_items
    items = tuple(
        (
            item
            for item in COLLECTED_ITEM_PROJECTION.flatten(family.collect(parsed_module))
            if isinstance(item, family.item_type)
        )
    )
    _store_cached_collected_family_items(parsed_module, family, items)
    return items


def collect_family_items(
    parsed_module: ParsedModule,
    family: type[CollectedFamily[ShapeItemT]],
) -> list[ShapeItemT]:
    """Collect and flatten items from one registered family."""
    return list(_collect_family_items_cached(parsed_module, family))


class RegisteredSpecCollectedFamily(
    CollectedFamily[ShapeItemT], Generic[ShapeItemT], ABC
):
    """Collected family driven by an auto-registered spec root."""

    _registry_skip = True
    spec_root: ClassVar[type[AutoRegisteredModuleShapeSpec]]

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ShapeItemT]:
        return COLLECTED_ITEM_PROJECTION.from_spec_root(
            cls.spec_root, parsed_module, cls.item_type
        )


class SingleSpecCollectedFamily(CollectedFamily[ShapeItemT], Generic[ShapeItemT], ABC):
    """Collected family driven by one explicit spec instance."""

    _registry_skip = True
    spec: ClassVar[ModuleShapeSpec[ShapeItemT]]

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ShapeItemT]:
        return [
            item
            for item in COLLECTED_ITEM_PROJECTION.flatten(
                cls.spec.collect(parsed_module)
            )
            if isinstance(item, cls.item_type)
        ]


class ScopedShapeSpec(
    ModuleShapeSpec[ShapeItemT], Generic[ShapeItemT], ABC, metaclass=AutoRegisterMeta
):
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    @property
    @abstractmethod
    def node_types(self) -> tuple[type[ast.AST], ...]:
        raise NotImplementedError

    def collect(self, parsed_module: ParsedModule) -> list[ShapeEmission[ShapeItemT]]:
        shapes: list[ShapeEmission[ShapeItemT]] = []
        for observation in collect_scoped_observations(parsed_module, self.node_types):
            shape = self.build_shape(parsed_module, observation)
            if shape is not None:
                shapes.append(shape)
        return shapes

    @abstractmethod
    def build_shape(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> ShapeEmission[ShapeItemT] | None:
        raise NotImplementedError


class ObservationShapeSpec(ScopedShapeSpec[ShapeItemT], Generic[ShapeItemT], ABC):
    def build_shape(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> ShapeEmission[ShapeItemT] | None:
        if not isinstance(observation.node, self.node_types):
            return None
        return self.build_from_observation(parsed_module, observation)

    @abstractmethod
    def build_from_observation(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> ShapeEmission[ShapeItemT] | None:
        raise NotImplementedError


class ContextForwardingShapeSpec(
    ObservationShapeSpec[ShapeItemT], Generic[ShapeItemT], ABC
):
    node_type: ClassVar[type[ast.AST]]

    @property
    def node_types(self) -> tuple[type[ast.AST], ...]:
        return (type(self).node_type,)

    def build_from_observation(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> ShapeEmission[ShapeItemT] | None:
        node = observation.node
        assert isinstance(node, type(self).node_type)
        return self.build_from_context(parsed_module, node, observation)

    def shape_helper_args(
        self, node: ast.AST, observation: ScopedAstObservation
    ) -> tuple[ContextShapeHelperArg, ...]:
        raise NotImplementedError

    def build_from_context(
        self,
        parsed_module: ParsedModule,
        node: ast.AST,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[ShapeItemT] | None:
        raise NotImplementedError


class ContextHelperShapeSpec(
    ContextForwardingShapeSpec[ShapeItemT], Generic[ShapeItemT], ABC
):
    shape_helper: ClassVar[
        Callable[
            [ParsedModule, ast.AST, str | None, str | None],
            ShapeEmission[ShapeItemT] | None,
        ]
    ]

    def shape_helper_args(
        self, node: ast.AST, observation: ScopedAstObservation
    ) -> tuple[ast.AST, str | None, str | None]:
        return (node, observation.class_name, observation.function_name)

    def build_from_context(
        self,
        parsed_module: ParsedModule,
        node: ast.AST,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[ShapeItemT] | None:
        return type(self).shape_helper(
            parsed_module,
            *self.shape_helper_args(node, observation),
        )


class FunctionObservationSpec(
    ObservationShapeSpec[ShapeItemT], Generic[ShapeItemT], ABC
):
    @property
    def node_types(self) -> tuple[type[ast.AST], ...]:
        return (ast.FunctionDef, ast.AsyncFunctionDef)

    def build_from_observation(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> ShapeEmission[ShapeItemT] | None:
        node = observation.node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        return self.build_from_function(parsed_module, node, observation)

    @abstractmethod
    def build_from_function(
        self,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[ShapeItemT] | None:
        raise NotImplementedError


class AssignObservationSpec(ObservationShapeSpec[ShapeItemT], Generic[ShapeItemT], ABC):
    @property
    def node_types(self) -> tuple[type[ast.AST], ...]:
        return (ast.Assign,)

    def build_from_observation(
        self, parsed_module: ParsedModule, observation: ScopedAstObservation
    ) -> ShapeEmission[ShapeItemT] | None:
        node = observation.node
        if not isinstance(node, ast.Assign):
            return None
        return self.build_from_assign(parsed_module, node, observation)

    @abstractmethod
    def build_from_assign(
        self,
        parsed_module: ParsedModule,
        node: ast.Assign,
        observation: ScopedAstObservation,
    ) -> ShapeEmission[ShapeItemT] | None:
        raise NotImplementedError


class SentinelTypeObservationSpecRoot(AutoRegisteredModuleShapeSpec, ABC):
    _registry_root = True


def _parse_module_roots(
    root_parser: "PythonModuleRootParser", paths: tuple[Path, ...]
) -> list[ParsedModule]:
    return [_parse_source_module(path, context=root_parser) for path in paths]


def _parse_module_roots_concurrently(
    root_parser: "PythonModuleRootParser", paths: tuple[Path, ...]
) -> list[ParsedModule]:
    parse_workers = _effective_parse_workers(root_parser.parse_workers)

    def parse_path(path: Path) -> ParsedModule:
        return _parse_source_module(path, context=root_parser)

    with ThreadPoolExecutor(max_workers=parse_workers) as executor:
        modules = list(executor.map(parse_path, paths))
    return modules


@dataclass(frozen=True)
class PythonSourcePathPolicy:
    """Decide which Python source files belong to one scan."""

    include_tests: bool = True

    def allows_directory_name(self, directory_name: str) -> bool:
        if directory_name in _IGNORED_PYTHON_TREE_DIRS:
            return False
        if directory_name.endswith((".egg-info", ".dist-info")):
            return False
        if not self.include_tests and self.is_test_directory_name(directory_name):
            return False
        return True

    def allows_file_path(self, path: Path) -> bool:
        if path.suffix != ".py":
            return False
        if self.include_tests:
            return True
        return not self.is_test_path(path)

    @staticmethod
    def is_test_directory_name(directory_name: str) -> bool:
        return directory_name.lower() in {"test", "tests"}

    @classmethod
    def is_test_path(cls, path: Path) -> bool:
        if any(cls.is_test_directory_name(part) for part in path.parts):
            return True
        file_name = path.name.lower()
        return any(
            fnmatchcase(file_name, pattern) for pattern in ("test_*.py", "*_test.py")
        )


@dataclass(frozen=True)
class PythonSourcePathDiscovery:
    """Discover deterministic Python source paths for one advisor scan root."""

    root: Path
    source_policy: PythonSourcePathPolicy = field(
        default_factory=PythonSourcePathPolicy
    )

    def paths(self) -> tuple[Path, ...]:
        if self.root.is_file():
            if self.source_policy.allows_file_path(self.root):
                return (self.root,)
            return ()

        paths: list[Path] = []
        for directory, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(
                (
                    dirname
                    for dirname in dirnames
                    if self.source_policy.allows_directory_name(dirname)
                )
            )
            directory_path = Path(directory)
            for filename in sorted(filenames):
                path = directory_path / filename
                if self.source_policy.allows_file_path(path):
                    paths.append(path)
        return tuple(paths)


def python_source_paths_for_roots(
    roots: tuple[Path, ...],
    *,
    source_policy: PythonSourcePathPolicy | None = None,
) -> tuple[Path, ...]:
    """Return de-duplicated Python source paths for multiple scan roots."""

    paths: list[Path] = []
    seen_paths: set[Path] = set()
    active_source_policy = source_policy or PythonSourcePathPolicy()
    for root in roots:
        for path in PythonSourcePathDiscovery(root, active_source_policy).paths():
            normalized_path = path.resolve()
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            paths.append(path)
    return tuple(paths)


@dataclass(frozen=True)
class PythonModuleRootParser(PythonModuleParseContext):
    root: Path
    parse_workers: int = _DEFAULT_PARSE_WORKERS
    source_policy: PythonSourcePathPolicy = field(
        default_factory=PythonSourcePathPolicy
    )

    @classmethod
    def for_root(
        cls,
        root: Path,
        *,
        cache_dir: Path | None = None,
        use_parse_cache: bool = True,
        parse_workers: int = _DEFAULT_PARSE_WORKERS,
        source_policy: PythonSourcePathPolicy | None = None,
    ) -> PythonModuleRootParser:
        resolved_cache_dir = (
            cache_dir
            if cache_dir is not None or not use_parse_cache
            else default_parse_cache_dir(root)
        )
        active_source_policy = source_policy or PythonSourcePathPolicy()
        return cls(
            root=root,
            analysis_root=root.parent if root.is_file() else root,
            parse_cache_dir=resolved_cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=parse_workers,
            source_policy=active_source_policy,
        )

    @classmethod
    def parse(
        cls,
        root: Path,
        *,
        cache_dir: Path | None = None,
        use_parse_cache: bool = True,
        parse_workers: int = _DEFAULT_PARSE_WORKERS,
        source_policy: PythonSourcePathPolicy | None = None,
    ) -> list[ParsedModule]:
        parser = cls.for_root(
            root,
            cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=parse_workers,
            source_policy=source_policy,
        )
        return parser.parsed_modules()

    def parsed_modules(self) -> list[ParsedModule]:
        paths = PythonSourcePathDiscovery(self.root, self.source_policy).paths()
        return self.parsed_source_paths(paths)

    def source_path_identities(self) -> tuple[PythonModulePathIdentity, ...]:
        paths = PythonSourcePathDiscovery(self.root, self.source_policy).paths()
        return self.source_path_identities_for_paths(paths)

    def source_path_identities_for_paths(
        self,
        paths: tuple[Path, ...],
    ) -> tuple[PythonModulePathIdentity, ...]:
        return tuple(
            PythonModulePathIdentity.from_path(path, self.analysis_root)
            for path in paths
            if path.is_file() and self.source_policy.allows_file_path(path)
        )

    def parsed_source_paths(self, paths: tuple[Path, ...]) -> list[ParsedModule]:
        allowed_paths = tuple(
            path
            for path in paths
            if path.is_file() and self.source_policy.allows_file_path(path)
        )
        if _effective_parse_workers(self.parse_workers) <= 1 or len(allowed_paths) <= 1:
            return _parse_module_roots(self, allowed_paths)
        return _parse_module_roots_concurrently(self, allowed_paths)


def parse_python_modules(
    root: Path,
    *,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = _DEFAULT_PARSE_WORKERS,
    source_policy: PythonSourcePathPolicy | None = None,
) -> list[ParsedModule]:
    """Parse one path (file or directory) into canonical ParsedModule records."""
    return PythonModuleRootParser.parse(
        root,
        cache_dir=cache_dir,
        use_parse_cache=use_parse_cache,
        parse_workers=parse_workers,
        source_policy=source_policy,
    )


def parse_python_module_roots(
    roots: tuple[Path, ...],
    *,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = _DEFAULT_PARSE_WORKERS,
    source_policy: PythonSourcePathPolicy | None = None,
) -> list[ParsedModule]:
    """Parse multiple file or directory roots into one de-duplicated module set."""
    modules: list[ParsedModule] = []
    seen_paths: set[Path] = set()
    for root in roots:
        parser = PythonModuleRootParser.for_root(
            root,
            cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=parse_workers,
            source_policy=source_policy,
        )
        for module in parser.parsed_modules():
            normalized_path = module.path.resolve()
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            modules.append(module)
    return modules


def python_module_path_identities_for_roots(
    roots: tuple[Path, ...],
    *,
    source_policy: PythonSourcePathPolicy | None = None,
) -> tuple[PythonModulePathIdentity, ...]:
    """Return de-duplicated module path identities without parsing source."""

    identities: list[PythonModulePathIdentity] = []
    seen_paths: set[Path] = set()
    for root in roots:
        parser = PythonModuleRootParser.for_root(
            root,
            use_parse_cache=False,
            source_policy=source_policy,
        )
        for identity in parser.source_path_identities():
            normalized_path = identity.path.resolve()
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            identities.append(identity)
    return tuple(identities)


AstConstantValue: TypeAlias = (
    str | int | float | complex | bool | bytes | None | EllipsisType
)
AstFingerprintInput: TypeAlias = (
    ast.AST
    | list["AstFingerprintInput"]
    | tuple["AstFingerprintInput", ...]
    | AstConstantValue
)
AstFingerprintAtom: TypeAlias = str | int | bool | None
AstFingerprintKey: TypeAlias = AstFingerprintAtom | tuple["AstFingerprintKey", ...]


def _normalized_constant(value: AstConstantValue) -> AstFingerprintAtom:
    if isinstance(value, str):
        return "STR"
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, complex)):
        return 0
    if value is None:
        return None
    return "CONST"


def _normalized_ast_key(node: AstFingerprintInput) -> AstFingerprintKey:
    if isinstance(node, ast.FunctionDef):
        return (
            "FunctionDef",
            "FUNC",
            _normalized_ast_key(node.args),
            tuple((_normalized_ast_key(stmt) for stmt in node.body)),
            tuple((_normalized_ast_key(dec) for dec in node.decorator_list)),
        )
    if isinstance(node, ast.AsyncFunctionDef):
        return (
            "AsyncFunctionDef",
            "FUNC",
            _normalized_ast_key(node.args),
            tuple((_normalized_ast_key(stmt) for stmt in node.body)),
            tuple((_normalized_ast_key(dec) for dec in node.decorator_list)),
        )
    if isinstance(node, ast.arg):
        return ("arg", "ARG")
    if isinstance(node, ast.Name):
        return ("Name", "VAR", node.ctx.__class__.__name__)
    if isinstance(node, ast.Constant):
        return ("Constant", _normalized_constant(node.value))
    if isinstance(node, ast.Attribute):
        return (
            "Attribute",
            _normalized_ast_key(node.value),
            "ATTR",
            node.ctx.__class__.__name__,
        )
    if isinstance(node, ast.AST):
        return (
            node.__class__.__name__,
            tuple(
                (
                    (field_name, _normalized_ast_key(value))
                    for field_name, value in ast.iter_fields(node)
                )
            ),
        )
    if isinstance(node, list):
        return tuple((_normalized_ast_key(item) for item in node))
    if isinstance(node, tuple):
        return tuple((_normalized_ast_key(item) for item in node))
    return node


@lru_cache(maxsize=None)
def fingerprint_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return repr(_normalized_ast_key(node))


def _builder_value_key(node: AstFingerprintInput) -> str:
    if isinstance(node, ast.Name):
        return f"Name(id='ROOT', ctx={node.ctx.__class__.__name__}())"
    if isinstance(node, ast.Constant):
        return f"Constant(value={_normalized_constant(node.value)!r})"
    if isinstance(node, ast.AST):
        fields_text = ", ".join(
            (
                f"{field_name}={_builder_value_key(value)}"
                for field_name, value in ast.iter_fields(node)
            )
        )
        return f"{node.__class__.__name__}({fields_text})"
    if isinstance(node, list):
        return "[" + ", ".join(_builder_value_key(item) for item in node) + "]"
    if isinstance(node, tuple):
        return "(" + ", ".join(_builder_value_key(item) for item in node) + ")"
    return repr(node)


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _subscript_base_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    return _terminal_name(node.value)


_CLASSVAR_REFERENCE_FAMILY = AstNameFamily(frozenset({"ClassVar"}))
_DATACLASS_DECORATOR_FAMILY = AstNameFamily(frozenset({"dataclass"}))
_ATTRIBUTE_ERROR_FAMILY = AstNameFamily(frozenset({"AttributeError"}))
_HASATTR_CALL_FAMILY = AstNameFamily(frozenset({"hasattr"}))
_GETATTR_CALL_FAMILY = AstNameFamily(frozenset({"getattr"}))
_REGISTRATION_CALL_FAMILY = AstNameFamily(
    frozenset({"register", "add", "register_class", "register_type"})
)
_REGISTRATION_DECORATOR_FAMILY = AstNameFamily(
    _REGISTRATION_CALL_FAMILY.names | frozenset({"auto_register"})
)


def _node_matches_family(node: ast.AST, family: AstNameFamily) -> bool:
    if isinstance(node, ast.Call):
        return _node_matches_family(node.func, family)
    return (
        _terminal_name(node) in family.names
        or _subscript_base_name(node) in family.names
    )


def _terminal_name_in_family(node: ast.AST, family: AstNameFamily) -> str | None:
    terminal_name = _terminal_name(node)
    if terminal_name in family.names:
        return terminal_name
    return None


def _name_family(names: set[str] | frozenset[str]) -> AstNameFamily:
    return AstNameFamily(frozenset(names))


@lru_cache(maxsize=32768)
def _walk_nodes(node: ast.AST) -> tuple[ast.AST, ...]:
    return tuple(ast.walk(node))


def _iter_attribute_family_calls(
    parsed_module: ParsedModule, family: AstNameFamily
) -> tuple[AstCallObservation, ...]:
    observations: list[AstCallObservation] = []
    for node in _walk_nodes(parsed_module.module):
        if not isinstance(node, ast.Call):
            continue
        matched_name = _attribute_call_family_name(node, family)
        if matched_name is None:
            continue
        observations.append(AstCallObservation(call=node, matched_name=matched_name))
    return sorted_tuple(observations, key=lambda item: item.call.lineno)


def _attribute_call_family_name(node: ast.Call, family: AstNameFamily) -> str | None:
    if not isinstance(node.func, ast.Attribute):
        return None
    return _terminal_name_in_family(node.func, family)


def _iter_class_decorator_family_calls(
    parsed_module: ParsedModule, family: AstNameFamily
) -> tuple[tuple[ast.ClassDef, ast.Call, str], ...]:
    observations: list[tuple[ast.ClassDef, ast.Call, str]] = []
    for node in _walk_nodes(parsed_module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            matched_name = _terminal_name_in_family(decorator.func, family)
            if matched_name is None:
                continue
            observations.append((node, decorator, matched_name))
    return sorted_tuple(observations, key=lambda item: item[0].lineno)


def _node_display_name(node: ast.AST) -> str:
    return _terminal_name(node) or node.__class__.__name__


@lru_cache(maxsize=None)
def _collect_all_scoped_observations(
    parsed_module: ParsedModule,
) -> tuple[ScopedAstObservation, ...]:
    observations: list[ScopedAstObservation] = []

    class Visitor(ClassFunctionStackNodeVisitor):
        def _record(self, node: ast.AST) -> None:
            if "lineno" not in node._attributes:
                return
            observations.append(
                ScopedAstObservation(
                    node=node,
                    class_name=self.current_class_name,
                    function_name=self.current_function_name,
                )
            )

        def before_visit_class(self, node: ast.ClassDef) -> None:
            self._record(node)

        def before_visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            self._record(node)

        def generic_visit(self, node: ast.AST) -> None:
            if not isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                self._record(node)
            super().generic_visit(node)

    Visitor().visit(parsed_module.module)
    return tuple(observations)


@lru_cache(maxsize=None)
def collect_scoped_observations(
    parsed_module: ParsedModule, node_types: tuple[type[ast.AST], ...]
) -> tuple[ScopedAstObservation, ...]:
    return tuple(
        (
            observation
            for observation in _collect_all_scoped_observations(parsed_module)
            if isinstance(observation.node, node_types)
        )
    )


def collect_scoped_shapes(
    parsed_module: ParsedModule, spec: ScopedShapeSpec[ShapeItemT]
) -> list[ShapeEmission[ShapeItemT]]:
    return spec.collect(parsed_module)


class CollectedItemProjection:
    def flatten(
        self,
        items: list[FlattenedItemT | tuple[FlattenedItemT, ...]],
    ) -> tuple[FlattenedItemT, ...]:
        flattened: list[FlattenedItemT] = []
        for item in items:
            if isinstance(item, tuple):
                flattened.extend(item)
            else:
                flattened.append(item)
        return tuple(flattened)

    def from_spec_root(
        self,
        spec_root: type[AutoRegisteredModuleShapeSpec],
        parsed_module: ParsedModule,
        item_type: type[FlattenedItemT],
    ) -> list[FlattenedItemT]:
        items: list[FlattenedItemT] = []
        for spec in spec_root.registered_specs():
            items.extend(
                (
                    item
                    for item in self.flatten(spec.collect(parsed_module))
                    if isinstance(item, item_type)
                )
            )
        return items


COLLECTED_ITEM_PROJECTION = CollectedItemProjection()


def _execution_level_for_scope(function_name: str | None) -> StructuralExecutionLevel:
    if function_name is None:
        return StructuralExecutionLevel.MODULE_BODY
    return StructuralExecutionLevel.FUNCTION_BODY


class ClassObservationProjection:
    def project(self, parsed_module: ParsedModule) -> tuple[ClassAstObservation, ...]:
        observations: list[ClassAstObservation] = []
        for observation in collect_scoped_observations(parsed_module, (ast.ClassDef,)):
            node = observation.node
            assert isinstance(node, ast.ClassDef)
            observations.append(
                ClassAstObservation(
                    node=node,
                    is_dataclass_family=any(
                        (
                            _node_matches_family(decorator, _DATACLASS_DECORATOR_FAMILY)
                            for decorator in node.decorator_list
                        )
                    ),
                )
            )
        return tuple(observations)


CLASS_OBSERVATION_PROJECTION = ClassObservationProjection()


@lru_cache(maxsize=None)
def _class_nodes(root: ast.AST) -> tuple[ast.ClassDef, ...]:
    return tuple(node for node in ast.walk(root) if isinstance(node, ast.ClassDef))


def _known_class_family(parsed_module: ParsedModule) -> AstNameFamily:
    return _name_family({node.name for node in _class_nodes(parsed_module.module)})


def _class_body_field_observation(
    parsed_module: ParsedModule,
    class_name: str,
    is_dataclass_family: bool,
    stmt: ast.stmt,
) -> FieldObservation | None:
    if not is_dataclass_family:
        return None
    binding = named_value_binding(stmt)
    if binding is None:
        return None
    if isinstance(stmt, ast.AnnAssign):
        if _node_matches_family(stmt.annotation, _CLASSVAR_REFERENCE_FAMILY):
            return None
        return FieldObservation(
            file_path=str(parsed_module.path),
            class_name=class_name,
            field_name=binding.name,
            lineno=binding.line,
            execution_level=StructuralExecutionLevel.CLASS_BODY,
            origin_kind=(
                FieldOriginKind.DATACLASS_FIELD
                if is_dataclass_family
                else FieldOriginKind.CLASS_ANNOTATION
            ),
            is_dataclass_family=is_dataclass_family,
            value_fingerprint=(
                _fingerprint_builder_value(binding.value)
                if binding.value is not None
                else None
            ),
            annotation_text=ast.unparse(stmt.annotation),
            annotation_fingerprint=_annotation_fingerprint(stmt.annotation),
        )
    if isinstance(stmt, ast.Assign):
        return FieldObservation(
            file_path=str(parsed_module.path),
            class_name=class_name,
            field_name=binding.name,
            lineno=binding.line,
            execution_level=StructuralExecutionLevel.CLASS_BODY,
            origin_kind=FieldOriginKind.CLASS_ASSIGNMENT,
            is_dataclass_family=is_dataclass_family,
            value_fingerprint=_fingerprint_builder_value(binding.value),
        )
    return None


def _annotation_fingerprint(node: ast.AST) -> str:
    return ast.dump(copy.deepcopy(node), include_attributes=False)


def _parameter_annotation_map(
    function: ast.FunctionDef,
) -> dict[str, tuple[str, str]]:
    annotations: dict[str, tuple[str, str]] = {}
    for arg in function.args.args:
        if arg.annotation is None:
            continue
        annotations[arg.arg] = (
            ast.unparse(arg.annotation),
            _annotation_fingerprint(arg.annotation),
        )
    return annotations


def _init_field_observations(
    parsed_module: ParsedModule,
    class_name: str,
    is_dataclass_family: bool,
    function: ast.FunctionDef,
) -> list[FieldObservation]:
    observations: list[FieldObservation] = []
    parameter_annotations = _parameter_annotation_map(function)
    for stmt in function.body:
        value: ast.AST | None = None
        target: ast.AST | None = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            value = stmt.value
        else:
            continue
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and (target.value.id == "self")
        ):
            continue
        observations.append(
            FieldObservation(
                file_path=str(parsed_module.path),
                class_name=class_name,
                field_name=target.attr,
                lineno=stmt.lineno,
                execution_level=StructuralExecutionLevel.INIT_BODY,
                origin_kind=FieldOriginKind.INIT_ASSIGNMENT,
                is_dataclass_family=is_dataclass_family,
                value_fingerprint=(
                    _fingerprint_builder_value(value) if value is not None else None
                ),
                annotation_text=(
                    parameter_annotations[value.id][0]
                    if isinstance(value, ast.Name) and value.id in parameter_annotations
                    else None
                ),
                annotation_fingerprint=(
                    parameter_annotations[value.id][1]
                    if isinstance(value, ast.Name) and value.id in parameter_annotations
                    else None
                ),
            )
        )
    return observations


def _parent_map(module: ast.Module) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in _walk_nodes(module)
        for child in ast.iter_child_nodes(parent)
    }


class ScopeParentage:
    def enclosing_function_name(
        self, node: ast.AST, parent_map: dict[ast.AST, ast.AST]
    ) -> str | None:
        current: ast.AST | None = node
        while current is not None:
            current = parent_map.get(current)
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current.name
        return None


SCOPE_PARENTAGE = ScopeParentage()


class LiteralDispatchCaseMatcher:
    def match(
        self, test: ast.AST, literal_type: type[LiteralDispatchScalar]
    ) -> tuple[str, str, str] | None:
        return (
            Maybe.of(test)
            .bind(_LiteralDispatchCompareStep())
            .bind(_LiteralDispatchCaseStep(literal_type))
            .unwrap_or_none()
        )


LITERAL_DISPATCH_CASE_MATCHER = LiteralDispatchCaseMatcher()


@dataclass(frozen=True)
class _LiteralDispatchCompare:
    left: ast.AST
    right: ast.AST


class _LiteralDispatchCompareStep(SingleCompareEffectStep[_LiteralDispatchCompare]):
    step_id = "literal_dispatch_compare"
    operator_type = ast.Eq

    def project_compare(self, left: ast.AST, right: ast.AST) -> _LiteralDispatchCompare:
        return _LiteralDispatchCompare(left, right)


@dataclass(frozen=True)
class _LiteralDispatchCaseStep(
    GuardedEffectStep[_LiteralDispatchCompare, tuple[str, str, str]]
):
    literal_type: type[LiteralDispatchScalar]
    step_id = "literal_dispatch_case"

    def project(self, value: _LiteralDispatchCompare) -> tuple[str, str, str] | None:
        return _literal_dispatch_side(
            value.right, value.left, self.literal_type
        ) or _literal_dispatch_side(value.left, value.right, self.literal_type)


def _literal_dispatch_side(
    axis: ast.AST, literal: ast.AST, literal_type: type[LiteralDispatchScalar]
) -> tuple[str, str, str] | None:
    if not isinstance(literal, ast.Constant) or not isinstance(
        literal.value, literal_type
    ):
        return None
    return (
        ast.dump(axis, include_attributes=False),
        ast.unparse(axis),
        repr(literal.value),
    )


def _literal_dispatch_observation_from_if(
    parsed_module: ParsedModule,
    node: ast.If,
    literal_type: type[LiteralDispatchScalar],
    literal_kind: LiteralKind,
    parent_map: dict[ast.AST, ast.AST],
) -> LiteralDispatchObservation | None:
    literal_cases: list[str] = []
    branch_lines: list[int] = []
    axis_fingerprint: str | None = None
    dispatch_axis_expression: str | None = None
    current: ast.stmt | None = node
    while isinstance(current, ast.If):
        case = LITERAL_DISPATCH_CASE_MATCHER.match(current.test, literal_type)
        if case is None:
            return None
        current_fingerprint, current_expression, literal_case = case
        if axis_fingerprint is None:
            axis_fingerprint = current_fingerprint
            dispatch_axis_expression = current_expression
        elif axis_fingerprint != current_fingerprint:
            return None
        literal_cases.append(literal_case)
        branch_lines.append(current.lineno)
        current = current.orelse[0] if len(current.orelse) == 1 else None
    if (
        axis_fingerprint is None
        or dispatch_axis_expression is None
        or len(literal_cases) < 2
    ):
        return None
    function_name = SCOPE_PARENTAGE.enclosing_function_name(node, parent_map)
    return LiteralDispatchObservation(
        file_path=str(parsed_module.path),
        line=node.lineno,
        symbol=(function_name or "<module>") + ":literal-dispatch",
        axis_fingerprint=axis_fingerprint,
        dispatch_axis_expression=dispatch_axis_expression,
        literal_cases=tuple(literal_cases),
        literal_kind=literal_kind,
        execution_level=_execution_level_for_scope(function_name),
        branch_lines=tuple(branch_lines),
        scope_owner=function_name,
    )


def _literal_match_case(
    pattern: ast.pattern, literal_type: type[LiteralDispatchScalar]
) -> str | None:
    if not isinstance(pattern, ast.MatchValue):
        return None
    value = pattern.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, literal_type):
        return None
    return repr(value.value)


def _literal_dispatch_observation_from_match(
    parsed_module: ParsedModule,
    node: ast.Match,
    literal_type: type[LiteralDispatchScalar],
    literal_kind: LiteralKind,
    parent_map: dict[ast.AST, ast.AST],
) -> LiteralDispatchObservation | None:
    literal_cases = tuple(
        (
            literal_case
            for match_case in node.cases
            if (literal_case := _literal_match_case(match_case.pattern, literal_type))
            is not None
        )
    )
    if len(literal_cases) < 2:
        return None
    function_name = SCOPE_PARENTAGE.enclosing_function_name(node, parent_map)
    dispatch_axis_expression = ast.unparse(node.subject)
    return LiteralDispatchObservation(
        file_path=str(parsed_module.path),
        line=node.lineno,
        symbol=(function_name or "<module>") + ":literal-dispatch",
        axis_fingerprint=ast.dump(node.subject, include_attributes=False),
        dispatch_axis_expression=dispatch_axis_expression,
        literal_cases=literal_cases,
        literal_kind=literal_kind,
        execution_level=_execution_level_for_scope(function_name),
        branch_lines=tuple((match_case.pattern.lineno for match_case in node.cases)),
        scope_owner=function_name,
    )


def _iter_statement_blocks(
    module: ast.Module,
) -> tuple[tuple[str | None, list[ast.stmt]], ...]:
    blocks: list[tuple[str | None, list[ast.stmt]]] = [(None, module.body)]
    for node in _walk_nodes(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            blocks.append((node.name, node.body))
    return tuple(blocks)


def _inline_literal_dispatch_groups(
    parsed_module: ParsedModule,
    owner_name: str | None,
    block: list[ast.stmt],
    literal_type: type[LiteralDispatchScalar],
    literal_kind: LiteralKind,
) -> tuple[LiteralDispatchObservation, ...]:
    groups: dict[str, list[tuple[int, str, str]]] = {}
    for stmt in block:
        if not isinstance(stmt, ast.If):
            continue
        case = LITERAL_DISPATCH_CASE_MATCHER.match(stmt.test, literal_type)
        if case is None:
            continue
        axis_fingerprint, dispatch_axis_expression, literal_case = case
        groups.setdefault(axis_fingerprint, []).append(
            (stmt.lineno, dispatch_axis_expression, literal_case)
        )
    observations: list[LiteralDispatchObservation] = []
    for axis_fingerprint, items in groups.items():
        literal_cases = sorted_tuple(
            {literal_case for _, _, literal_case in items}, key=str
        )
        if len(literal_cases) < 2:
            continue
        observations.append(
            LiteralDispatchObservation(
                file_path=str(parsed_module.path),
                line=min((line for line, _, _ in items)),
                symbol=(owner_name or "<module>") + ":inline-literal-dispatch",
                axis_fingerprint=axis_fingerprint,
                dispatch_axis_expression=items[0][1],
                literal_cases=literal_cases,
                literal_kind=literal_kind,
                execution_level=_execution_level_for_scope(owner_name),
                branch_lines=sorted_tuple((line for line, _, _ in items)),
                scope_owner=owner_name,
            )
        )
    return sorted_tuple(observations, key=lambda item: item.line)


_LITERAL_DISPATCH_KINDS: tuple[tuple[type[LiteralDispatchScalar], LiteralKind], ...] = (
    (str, LiteralKind.STRING),
    (int, LiteralKind.NUMERIC),
)


@lru_cache(maxsize=None)
def _literal_dispatch_observations(
    parsed_module: ParsedModule,
) -> tuple[LiteralDispatchObservation, ...]:
    parent_map = _parent_map(parsed_module.module)
    observations: list[LiteralDispatchObservation] = []
    for node in _walk_nodes(parsed_module.module):
        for literal_type, literal_kind in _LITERAL_DISPATCH_KINDS:
            observation = None
            if isinstance(node, ast.If):
                parent = parent_map.get(node)
                if (
                    isinstance(parent, ast.If)
                    and len(parent.orelse) == 1
                    and (parent.orelse[0] is node)
                ):
                    continue
                observation = _literal_dispatch_observation_from_if(
                    parsed_module, node, literal_type, literal_kind, parent_map
                )
            elif isinstance(node, ast.Match):
                observation = _literal_dispatch_observation_from_match(
                    parsed_module, node, literal_type, literal_kind, parent_map
                )
            if observation is not None:
                observations.append(observation)
    return sorted_tuple(observations, key=lambda item: (item.line, item.literal_kind))


def _literal_dispatch_observations_for_kind(
    parsed_module: ParsedModule, literal_kind: LiteralKind
) -> tuple[LiteralDispatchObservation, ...]:
    return tuple(
        (
            item
            for item in _literal_dispatch_observations(parsed_module)
            if item.literal_kind is literal_kind
        )
    )


@lru_cache(maxsize=None)
def _inline_literal_dispatch_observations(
    parsed_module: ParsedModule,
) -> tuple[LiteralDispatchObservation, ...]:
    observations: list[LiteralDispatchObservation] = []
    for owner_name, block in _iter_statement_blocks(parsed_module.module):
        for literal_type, literal_kind in _LITERAL_DISPATCH_KINDS:
            observations.extend(
                _inline_literal_dispatch_groups(
                    parsed_module, owner_name, block, literal_type, literal_kind
                )
            )
    return sorted_tuple(observations, key=lambda item: (item.line, item.literal_kind))


def _inline_literal_dispatch_observations_for_kind(
    parsed_module: ParsedModule, literal_kind: LiteralKind
) -> tuple[LiteralDispatchObservation, ...]:
    return tuple(
        (
            item
            for item in _inline_literal_dispatch_observations(parsed_module)
            if item.literal_kind is literal_kind
        )
    )


def _is_docstring_expr(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _trim_docstring_body(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and _is_docstring_expr(body[0]):
        return body[1:]
    return body


class _ProjectionOuterCallStep(RegisteredEffectStep):
    pass


class _SingleReturnCallStep(
    _ProjectionOuterCallStep, GuardedEffectStep[list[ast.stmt], ast.Call]
):
    step_id = "single_return_call"
    registration_order = 10

    def project(self, value: list[ast.stmt]) -> ast.Call | None:
        return single_return_call(value)


class _SingleArgumentCallStep(
    _ProjectionOuterCallStep, GuardedEffectStep[ast.Call, ast.Call]
):
    step_id = "single_argument_call"
    registration_order = 20

    def project(self, value: ast.Call) -> ast.Call | None:
        return value if len(value.args) == 1 else None


class _TerminalCalleeFamilyStep(
    _ProjectionOuterCallStep, GuardedEffectStep[ast.Call, ast.Call]
):
    step_id = "terminal_callee_family"
    registration_order = 30
    terminal_names = BuiltinCallName.sequence_wrapper_names()

    def project(self, value: ast.Call) -> ast.Call | None:
        return value if _terminal_name(value.func) in self.terminal_names else None


def _projection_outer_inner_calls(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ast.Call] | None:
    outer_call = (
        Maybe.of(_trim_docstring_body(function.body))
        .bind_all(registered_effect_steps(_ProjectionOuterCallStep))
        .unwrap_or_none()
    )
    return (
        Maybe.of(outer_call)
        .combine(
            lambda call: as_ast(single_call_arg(call), ast.Call),
            lambda call, inner_call: (
                (call, inner_call) if len(inner_call.args) == 1 else None
            ),
        )
        .combine(
            lambda context: _terminal_name(context[0].func),
            lambda context, outer_call_name: (outer_call_name, context[1]),
        )
        .unwrap_or_none()
    )


@dataclass(frozen=True)
class _ProjectionGeneratorMatch:
    node: ast.GeneratorExp
    comprehension: ast.comprehension


class _ProjectionGeneratorAttributeStep(RegisteredEffectStep):
    pass


class _SingleProjectionGeneratorStep(
    _ProjectionGeneratorAttributeStep,
    AstTypedEffectStep[ast.GeneratorExp, _ProjectionGeneratorMatch],
):
    step_id = "single_projection_generator"
    registration_order = 10
    node_type = ast.GeneratorExp

    def project_ast(self, value: ast.GeneratorExp) -> _ProjectionGeneratorMatch | None:
        return (
            Maybe.of(single_item(value.generators))
            .map(lambda comprehension: _ProjectionGeneratorMatch(value, comprehension))
            .unwrap_or_none()
        )


class _ProjectionNameTargetStep(
    _ProjectionGeneratorAttributeStep,
    GuardedEffectStep[_ProjectionGeneratorMatch, _ProjectionGeneratorMatch],
):
    step_id = "projection_name_target"
    registration_order = 20

    def project(
        self, value: _ProjectionGeneratorMatch
    ) -> _ProjectionGeneratorMatch | None:
        if value.comprehension.is_async or value.comprehension.ifs:
            return None
        return value if isinstance(value.comprehension.target, ast.Name) else None


def _projected_attribute_name(value: _ProjectionGeneratorMatch) -> str | None:
    attribute = as_ast(value.node.elt, ast.Attribute)
    target = as_ast(value.comprehension.target, ast.Name)
    owner = as_ast(attribute.value if attribute else None, ast.Name)
    if attribute is None or target is None or owner is None or (owner.id != target.id):
        return None
    return attribute.attr


class _ProjectedAttributeStep(
    _ProjectionGeneratorAttributeStep,
    GuardedEffectStep[_ProjectionGeneratorMatch, str],
):
    step_id = "projected_attribute"
    registration_order = 30

    def project(self, value: _ProjectionGeneratorMatch) -> str | None:
        return _projected_attribute_name(value)


def _projection_generator_attribute(node: ast.AST) -> str | None:
    return cast(
        str | None,
        Maybe.of(node)
        .bind_all(registered_effect_steps(_ProjectionGeneratorAttributeStep))
        .unwrap_or_none(),
    )


def _projection_inner_shape(inner_call: ast.Call) -> tuple[str, str] | None:
    return (
        Maybe.of(_terminal_name(inner_call.func))
        .combine(
            lambda _aggregator_name: _projection_generator_attribute(
                inner_call.args[0]
            ),
            lambda aggregator_name, projected_attribute: (
                aggregator_name,
                projected_attribute,
            ),
        )
        .unwrap_or_none()
    )


def _projection_helper_shape_from_function(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ProjectionHelperShape | None:
    return (
        Maybe.of(_projection_outer_inner_calls(function))
        .combine(
            lambda call_pair: _projection_inner_shape(call_pair[1]),
            lambda call_pair, inner_shape: ProjectionHelperShape(
                file_path=str(parsed_module.path),
                function_name=function.name,
                lineno=function.lineno,
                outer_call_name=call_pair[0],
                aggregator_name=inner_shape[0],
                iterable_fingerprint=fingerprint_function(function),
                projected_attribute=inner_shape[1],
            ),
        )
        .unwrap_or_none()
    )


def _accessor_wrapper_candidate_from_function(
    parsed_module: ParsedModule,
    class_name: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> AccessorWrapperCandidate | None:
    if _is_dunder_method(function.name):
        return None
    if _has_property_like_decorator(function):
        return None
    body = _trim_docstring_body(function.body)
    if not body:
        return None
    getter_candidate = _getter_wrapper_candidate(function, body)
    if getter_candidate is not None:
        target_expression, observed_attribute, wrapper_shape = getter_candidate
        return AccessorWrapperCandidate(
            file_path=str(parsed_module.path),
            class_name=class_name,
            method_name=function.name,
            lineno=function.lineno,
            target_expression=target_expression,
            observed_attribute=observed_attribute,
            accessor_kind="getter",
            wrapper_shape=wrapper_shape,
        )
    setter_candidate = _setter_wrapper_candidate(function, body)
    if setter_candidate is not None:
        target_expression, observed_attribute = setter_candidate
        return AccessorWrapperCandidate(
            file_path=str(parsed_module.path),
            class_name=class_name,
            method_name=function.name,
            lineno=function.lineno,
            target_expression=target_expression,
            observed_attribute=observed_attribute,
            accessor_kind="setter",
            wrapper_shape="write_through",
        )
    return None


def _scoped_shape_wrapper_node_types(
    function: ast.FunctionDef,
    body: list[ast.stmt],
) -> tuple[str, ...] | None:
    if len(function.args.args) != 2 or len(body) < 3:
        return None
    first_stmt, second_stmt = body[:2]
    if not _assigns_observation_node(first_stmt, function.args.args[1].arg):
        return None
    if not isinstance(second_stmt, ast.If):
        return None
    node_types = TYPE_GUARD_PROJECTION.guarded_node_types(second_stmt.test, "node")
    if not node_types or not _if_returns_none(second_stmt):
        return None
    return node_types


def _assigns_observation_node(statement: ast.stmt, observation_arg_name: str) -> bool:
    return bool(
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and (statement.targets[0].id == "node")
        and isinstance(statement.value, ast.Attribute)
        and isinstance(statement.value.value, ast.Name)
        and (statement.value.value.id == observation_arg_name)
        and (statement.value.attr == "node")
    )


def _if_returns_none(statement: ast.If) -> bool:
    return bool(
        len(statement.body) == 1
        and isinstance(statement.body[0], ast.Return)
        and isinstance(statement.body[0].value, ast.Constant)
        and (statement.body[0].value.value is None)
    )


def _scoped_shape_wrapper_function_from_function(
    parsed_module: ParsedModule,
    function: ast.FunctionDef,
) -> ScopedShapeWrapperFunction | None:
    body = _trim_docstring_body(function.body)
    node_types = _scoped_shape_wrapper_node_types(function, body)
    if (
        node_types is None
        or not isinstance(body[-1], ast.Return)
        or body[-1].value is None
    ):
        return None
    return ScopedShapeWrapperFunction(
        file_path=str(parsed_module.path),
        function_name=function.name,
        lineno=function.lineno,
        node_types=node_types,
    )


def _scoped_shape_spec_call(node: ast.Assign) -> _ScopedShapeSpecCall | None:
    target = as_ast(single_assign_target(node), ast.Name)
    call = as_ast(node.value, ast.Call)
    if target is None or call is None:
        return None
    if _terminal_name(call.func) != "ScopedShapeSpec":
        return None
    return _ScopedShapeSpecCall(target.id, call)


def _scoped_shape_spec_keywords(call: ast.Call) -> _ScopedShapeSpecKeywords | None:
    node_types: tuple[str, ...] = ()
    function_name = None
    for keyword in call.keywords:
        if keyword.arg == "node_types":
            node_types = TYPE_GUARD_PROJECTION.type_name_tuple(keyword.value)
        if keyword.arg == "build_shape":
            function_name = _terminal_name(keyword.value)
    if not node_types or function_name is None:
        return None
    return _ScopedShapeSpecKeywords(function_name, node_types)


def _scoped_shape_wrapper_spec_from_assign(
    parsed_module: ParsedModule,
    node: ast.Assign,
) -> ScopedShapeWrapperSpec | None:
    return (
        Maybe.of(_scoped_shape_spec_call(node))
        .combine(
            lambda spec_call: _scoped_shape_spec_keywords(spec_call.call),
            lambda spec_call, keywords: ScopedShapeWrapperSpec(
                file_path=str(parsed_module.path),
                spec_name=spec_call.spec_name,
                lineno=node.lineno,
                function_name=keywords.function_name,
                node_types=keywords.node_types,
            ),
        )
        .unwrap_or_none()
    )


def _getter_wrapper_candidate(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    body: list[ast.stmt],
) -> tuple[str, str, str] | None:
    if len(function.args.args) != 1:
        return None
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    expr = body[0].value
    if _is_self_attribute_expression(expr):
        observed_attribute = _self_attribute_name(expr)
        if observed_attribute is None:
            return None
        return ast.unparse(expr), observed_attribute, "read_through"
    wrapped = _wrapped_self_attribute_expression(expr)
    if wrapped is not None:
        wrapper_name, observed_attribute = wrapped
        return ast.unparse(expr), observed_attribute, f"computed_{wrapper_name}"
    return None


def _setter_wrapper_candidate(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    body: list[ast.stmt],
) -> tuple[str, str] | None:
    return (
        Maybe.of(body[0] if len(function.args.args) == 2 and len(body) == 1 else None)
        .project(
            lambda statement: _self_attribute_assignment(
                statement,
                function.args.args[1].arg,
            )
        )
        .combine(
            _self_attribute_name,
            lambda target, observed_attribute: (
                ast.unparse(target),
                observed_attribute,
            ),
        )
        .unwrap_or_none()
    )


def _self_attribute_assignment(
    statement: ast.stmt, value_arg: str
) -> ast.Attribute | None:
    assignment = as_ast(statement, ast.Assign)
    if assignment is None:
        return None
    target = single_assign_target(assignment)
    if not _is_self_attribute_expression(target):
        return None
    if not (
        isinstance(assignment.value, ast.Name) and assignment.value.id == value_arg
    ):
        return None
    assert isinstance(target, ast.Attribute)
    return target


def _is_self_attribute_expression(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and (node.value.id == "self")
    )


def _wrapped_self_attribute_expression(node: ast.AST) -> tuple[str, str] | None:
    wrapper_names = BuiltinCallName.self_attribute_wrapper_names()
    return (
        Maybe.of(as_ast(node, ast.Call))
        .filter(
            lambda call: isinstance(call.func, ast.Name)
            and call.func.id in wrapper_names
        )
        .combine(
            lambda call: _self_attribute_name(single_call_arg(call)),
            lambda call, observed_attribute: (
                cast(ast.Name, call.func).id,
                observed_attribute,
            ),
        )
        .unwrap_or_none()
    )


def _self_attribute_name(node: ast.AST) -> str | None:
    if not _is_self_attribute_expression(node):
        return None
    assert isinstance(node, ast.Attribute)
    return node.attr.lstrip("_") or node.attr


def _is_dunder_method(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _has_property_like_decorator(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "setter":
            return True
    return False


class TypeGuardProjection:
    def guarded_node_types(self, test: ast.AST, expected_name: str) -> tuple[str, ...]:
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self.guarded_node_types(test.operand, expected_name)
        if not isinstance(test, ast.Call):
            return ()
        if not isinstance(test.func, ast.Name) or test.func.id != "isinstance":
            return ()
        if len(test.args) != 2:
            return ()
        if not isinstance(test.args[0], ast.Name) or test.args[0].id != expected_name:
            return ()
        return self.type_name_tuple(test.args[1])

    def type_name_tuple(self, node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            return (node.attr,)
        if isinstance(node, ast.Tuple):
            names: list[str] = []
            for item in node.elts:
                names.extend(self.type_name_tuple(item))
            return tuple(names)
        return ()


TYPE_GUARD_PROJECTION = TypeGuardProjection()


def _config_dispatch_observations(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ConfigDispatchObservation, ...]:
    seen: set[tuple[int, str]] = set()
    observations: list[ConfigDispatchObservation] = []
    for node in _walk_nodes(function):
        if isinstance(node, ast.If):
            for attr_name in _config_dispatch_attributes(node.test):
                key = (node.lineno, attr_name)
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    ConfigDispatchObservation(
                        file_path=str(parsed_module.path),
                        line=node.lineno,
                        symbol=function.name,
                        observed_attribute=attr_name,
                    )
                )
        if isinstance(node, ast.Match):
            for attr_name in _match_config_dispatch_attributes(node.subject):
                key = (node.lineno, attr_name)
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    ConfigDispatchObservation(
                        file_path=str(parsed_module.path),
                        line=node.lineno,
                        symbol=function.name,
                        observed_attribute=attr_name,
                    )
                )
    return sorted_tuple(
        observations, key=lambda item: (item.line, item.observed_attribute)
    )


def _config_dispatch_attributes(test: ast.AST) -> tuple[str, ...]:
    attrs: set[str] = set()
    for node in _walk_nodes(test):
        if isinstance(node, ast.Call) and _terminal_name_in_family(
            node.func, _HASATTR_CALL_FAMILY
        ):
            if _call_targets_name(node, "config") and len(node.args) >= 2:
                if isinstance(node.args[1], ast.Constant) and isinstance(
                    node.args[1].value, str
                ):
                    attrs.add(node.args[1].value)
        if isinstance(node, ast.Call) and _terminal_name_in_family(
            node.func, _GETATTR_CALL_FAMILY
        ):
            if _call_targets_name(node, "config") and len(node.args) >= 2:
                if isinstance(node.args[1], ast.Constant) and isinstance(
                    node.args[1].value, str
                ):
                    attrs.add(node.args[1].value)
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                continue
            if not isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                continue
            left_name = CONFIG_SUBJECT_PROJECTION.subject_name(node.left)
            right_name = CONFIG_SUBJECT_PROJECTION.subject_name(node.comparators[0])
            left_literal = _literal_dispatch_value(node.left)
            right_literal = _literal_dispatch_value(node.comparators[0])
            if left_name is not None and right_literal is not None:
                attrs.add(left_name)
            if right_name is not None and left_literal is not None:
                attrs.add(right_name)
    return sorted_tuple(attrs)


def _match_config_dispatch_attributes(subject: ast.AST) -> tuple[str, ...]:
    attr_name = CONFIG_SUBJECT_PROJECTION.subject_name(subject)
    if attr_name is not None:
        return (attr_name,)
    return ()


def _class_marker_observations(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ClassMarkerObservation, ...]:
    seen: set[tuple[int, str]] = set()
    observations: list[ClassMarkerObservation] = []
    for node in _walk_nodes(function):
        if isinstance(node, ast.Call) and _terminal_name_in_family(
            node.func, _HASATTR_CALL_FAMILY
        ):
            target = node.args[0] if node.args else None
            marker_name = None
            if _is_class_target(target):
                marker_name = (
                    _constant_string(node.args[1]) if len(node.args) >= 2 else None
                )
            if marker_name is not None:
                key = (node.lineno, marker_name)
                if key not in seen:
                    seen.add(key)
                    observations.append(
                        ClassMarkerObservation(
                            file_path=str(parsed_module.path),
                            line=node.lineno,
                            symbol=function.name,
                            marker_name=marker_name,
                        )
                    )
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_is_")
            and _is_class_target(node.value)
        ):
            key = (node.lineno, node.attr)
            if key not in seen:
                seen.add(key)
                observations.append(
                    ClassMarkerObservation(
                        file_path=str(parsed_module.path),
                        line=node.lineno,
                        symbol=function.name,
                        marker_name=node.attr,
                    )
                )
    return sorted_tuple(observations, key=lambda item: (item.line, item.marker_name))


def _interface_generation_observation(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> InterfaceGenerationObservation | None:
    for node in _walk_nodes(function):
        if not isinstance(node, ast.Call):
            continue
        if _terminal_name(node.func) != "type":
            continue
        if len(node.args) < 3:
            continue
        bases = node.args[1]
        namespace = node.args[2]
        if not isinstance(namespace, ast.Dict) or namespace.keys:
            continue
        if not isinstance(bases, ast.Tuple):
            continue
        if any((_terminal_name(base) == "ABC" for base in bases.elts)):
            return InterfaceGenerationObservation(
                file_path=str(parsed_module.path),
                line=node.lineno,
                symbol=function.name,
                generator_name=_TYPE_BUILTIN,
            )
    return None


def _sentinel_type_observation(
    parsed_module: ParsedModule,
    node: ast.Assign,
) -> SentinelTypeObservation | None:
    target = single_item(node.targets)
    if not isinstance(target, ast.Name) or not _is_type_call_constructor(node.value):
        return None
    return SentinelTypeObservation(
        file_path=str(parsed_module.path),
        line=node.lineno,
        symbol=target.id,
        sentinel_name=target.id,
    )


def _is_type_call_constructor(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Call)
        and (_terminal_name(node.func.func) == _TYPE_BUILTIN)
    )


def _sentinel_type_usage_observations(
    parsed_module: ParsedModule,
) -> tuple[SentinelTypeObservation, ...]:
    sentinel_names = {
        item.sentinel_name
        for node in _walk_nodes(parsed_module.module)
        if isinstance(node, ast.Assign)
        if (item := _sentinel_type_observation(parsed_module, node)) is not None
    }
    observations: list[SentinelTypeObservation] = []
    seen: set[tuple[int, str]] = set()
    for node in _walk_nodes(parsed_module.module):
        if isinstance(node, (ast.Compare, ast.Subscript)):
            names = {
                subnode.id
                for subnode in _walk_nodes(node)
                if isinstance(subnode, ast.Name)
            }
            for name in sorted(names & sentinel_names):
                key = (node.lineno, name)
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    SentinelTypeObservation(
                        file_path=str(parsed_module.path),
                        line=node.lineno,
                        symbol=f"sentinel:{name}",
                        sentinel_name=name,
                    )
                )
    return sorted_tuple(observations, key=lambda item: (item.line, item.sentinel_name))


def _dynamic_method_injection_observations(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[DynamicMethodInjectionObservation, ...]:
    observations: list[DynamicMethodInjectionObservation] = []
    for node in _walk_nodes(function):
        if not isinstance(node, ast.Call):
            continue
        if _terminal_name(node.func) != _SETATTR_BUILTIN:
            continue
        if len(node.args) < 3:
            continue
        target = node.args[0]
        if isinstance(target, ast.Name) and target.id.endswith("type"):
            observations.append(
                DynamicMethodInjectionObservation(
                    file_path=str(parsed_module.path),
                    line=node.lineno,
                    symbol=function.name,
                    mutator_name=_SETATTR_BUILTIN,
                )
            )
    return sorted_tuple(observations, key=lambda item: item.line)


def _runtime_type_generation_observation(
    parsed_module: ParsedModule,
    node: ast.Call,
    observation: ScopedAstObservation,
) -> RuntimeTypeGenerationObservation | None:
    generator_name = _terminal_name(node.func)
    if generator_name not in _RUNTIME_TYPE_GENERATORS:
        return None
    # `type(obj)` is ordinary type introspection, not runtime type generation.
    # Only the 3-argument `type(name, bases, namespace)` form constructs a type.
    if generator_name == _TYPE_BUILTIN and len(node.args) < 3:
        return None
    return RuntimeTypeGenerationObservation(
        file_path=str(parsed_module.path),
        line=node.lineno,
        symbol=observation.function_name or generator_name,
        generator_name=generator_name,
    )


def _lineage_mapping_observation(
    parsed_module: ParsedModule,
    node: ast.Assign,
) -> LineageMappingObservation | None:
    for target in node.targets:
        if not isinstance(target, ast.Subscript):
            continue
        name = _terminal_name(target.value)
        if name and any(
            (
                token in name.lower()
                for token in ("lazy", "base", "type", "mapping", "registry")
            )
        ):
            return LineageMappingObservation(
                file_path=str(parsed_module.path),
                line=node.lineno,
                symbol=name,
                mapping_name=name,
            )
    return None


def _dual_axis_resolution_observation(
    parsed_module: ParsedModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> DualAxisResolutionObservation | None:
    for node in _walk_nodes(function):
        if not isinstance(node, ast.For):
            continue
        inner_loops = [child for child in node.body if isinstance(child, ast.For)]
        if not inner_loops:
            continue
        outer_name = _loop_target_name(node.target)
        inner_name = _loop_target_name(inner_loops[0].target)
        text = ast.dump(inner_loops[0].iter, include_attributes=False)
        if "__mro__" in text or "mro" in text.lower() or "type" in text.lower():
            if outer_name and any(
                (token in outer_name.lower() for token in ("scope", "context", "level"))
            ):
                return DualAxisResolutionObservation(
                    file_path=str(parsed_module.path),
                    line=node.lineno,
                    symbol=function.name,
                    outer_axis_name=outer_name,
                    inner_axis_name=inner_name or "mro_type",
                )
    return None


def _loop_target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _call_targets_name(node: ast.Call, expected_name: str) -> bool:
    return bool(
        node.args
        and isinstance(node.args[0], ast.Name)
        and (node.args[0].id == expected_name)
    )


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _attribute_name_if_root(node: ast.AST, expected_root: str) -> str | None:
    if not isinstance(node, ast.Attribute):
        return None
    if isinstance(node.value, ast.Name) and node.value.id == expected_root:
        return node.attr
    return None


class ConfigSubjectProjection:
    def subject_name(self, node: ast.AST) -> str | None:
        attr_name = _attribute_name_if_root(node, "config")
        if attr_name is not None:
            return attr_name
        if isinstance(node, ast.Call) and _terminal_name_in_family(
            node.func, _GETATTR_CALL_FAMILY
        ):
            if _call_targets_name(node, "config") and len(node.args) >= 2:
                return _constant_string(node.args[1])
        return None


CONFIG_SUBJECT_PROJECTION = ConfigSubjectProjection()


def _literal_dispatch_value(node: ast.AST) -> LiteralConstantValue:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, bool)):
        return node.value
    return None


def _is_class_target(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Attribute) and node.attr == "__class__":
        return True
    if isinstance(node, ast.Call) and _terminal_name(node.func) == "type":
        return True
    return False


@lru_cache(maxsize=None)
def _module_class_names(parsed_module: ParsedModule) -> frozenset[str]:
    return frozenset(
        node.name
        for node in _walk_nodes(parsed_module.module)
        if isinstance(node, ast.ClassDef)
    )


def _builder_call_shape(
    parsed_module: ParsedModule,
    node: ast.AST,
    class_name: str | None,
    function_name: str | None,
    module_class_names: frozenset[str] | None = None,
) -> BuilderCallShape | None:
    module_class_names = module_class_names or _module_class_names(parsed_module)

    def owned_builder_authority_call(call: ast.Call) -> bool:
        if not isinstance(call.func, ast.Attribute):
            return False
        if not call.func.attr.startswith(("for_", "from_", "with_")):
            return False
        owner_name = _terminal_name(call.func.value)
        if owner_name is None:
            return False
        return owner_name in module_class_names

    def positional_builder_roles_allowed(callee_name: str) -> bool:
        return callee_name.startswith(("for_", "from_", "with_"))

    def positional_field_pairs(
        call: ast.Call,
        callee_name: str,
    ) -> tuple[tuple[str, ast.AST], ...]:
        if not positional_builder_roles_allowed(callee_name):
            return ()
        pairs: list[tuple[str, ast.AST]] = []
        for argument in call.args:
            field_name = _terminal_name(argument)
            if field_name is None:
                return ()
            pairs.append((field_name, argument))
        return tuple(pairs)

    call_node = as_ast(node, ast.Call)
    if call_node is not None and owned_builder_authority_call(call_node):
        return None

    context = (
        Maybe.of(as_ast(node, ast.Call))
        .filter(lambda _call: function_name is not None)
        .combine(
            lambda call: _terminal_name(call.func),
            lambda call, callee_name: _BuilderCallContext(
                call=call,
                callee_name=callee_name,
                field_pairs=(
                    positional_field_pairs(call, callee_name)
                    + tuple(
                        (kw.arg, kw.value) for kw in call.keywords if kw.arg is not None
                    )
                ),
            ),
        )
        .filter(lambda builder_context: bool(builder_context.field_pairs))
        .unwrap_or_none()
    )
    if context is None:
        return None
    field_names = tuple(name for name, _ in context.field_pairs)
    value_fingerprint = tuple(
        (_fingerprint_builder_value(value) for _, value in context.field_pairs)
    )
    source_roots = set()
    for _, value in context.field_pairs:
        source_roots.update(ROOT_NAME_PROJECTION.root_names(value))
    source_name = next(iter(source_roots)) if len(source_roots) == 1 else None
    identity_field_names = tuple(
        (name for name, value in context.field_pairs if _terminal_name(value) == name)
    )
    return BuilderCallShape(
        file_path=str(parsed_module.path),
        class_name=class_name,
        function_name=function_name,
        lineno=context.call.lineno,
        callee_name=context.callee_name,
        field_names=field_names,
        value_fingerprint=value_fingerprint,
        source_arity=len(source_roots),
        source_name=source_name,
        identity_field_names=identity_field_names,
    )


def _export_dict_shape(
    parsed_module: ParsedModule,
    node: ast.AST,
    class_name: str | None,
    function_name: str | None,
) -> ExportDictShape | None:
    context = (
        Maybe.of(as_ast(node, ast.Dict))
        .filter(lambda _dict: function_name is not None)
        .map(
            lambda dict_node: _ExportDictContext(
                dict_node=dict_node,
                key_pairs=tuple(
                    (key.value, value)
                    for key, value in zip(
                        dict_node.keys, dict_node.values, strict=False
                    )
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                ),
            )
        )
        .filter(
            lambda export_context: len(export_context.key_pairs) >= 3
            and len(export_context.key_pairs) == len(export_context.dict_node.keys)
        )
        .unwrap_or_none()
    )
    return (
        Maybe.of(context)
        .combine(
            lambda export_context: _source_roots_for_value_pairs(
                export_context.key_pairs
            ),
            lambda export_context, source_roots: ExportDictShape(
                file_path=str(parsed_module.path),
                class_name=class_name,
                function_name=function_name,
                lineno=export_context.dict_node.lineno,
                key_names=tuple(name for name, _ in export_context.key_pairs),
                value_fingerprint=tuple(
                    (
                        _fingerprint_builder_value(value)
                        for _, value in export_context.key_pairs
                    )
                ),
                source_arity=len(source_roots),
                source_name=(
                    next(iter(source_roots)) if len(source_roots) == 1 else None
                ),
                identity_field_names=tuple(
                    (
                        name
                        for name, value in export_context.key_pairs
                        if _terminal_name(value) == name
                    )
                ),
            ),
        )
        .unwrap_or_none()
    )


def _source_roots_for_value_pairs(
    value_pairs: tuple[tuple[str, ast.AST], ...],
) -> set[str] | None:
    source_roots: set[str] = set()
    for _, value in value_pairs:
        source_roots.update(ROOT_NAME_PROJECTION.root_names(value))
    return source_roots or None


def _fingerprint_builder_value(node: ast.AST) -> str:
    return _builder_value_key(node)


class RootNameProjection:
    def root_names(self, node: ast.AST) -> set[str]:
        roots: set[str] = set()

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                for argument in node.args:
                    self.visit(argument)
                for keyword in node.keywords:
                    self.visit(keyword.value)

            def visit_Attribute(self, node: ast.Attribute) -> None:
                current: ast.AST = node
                while isinstance(current, ast.Attribute):
                    current = current.value
                if isinstance(current, ast.Name):
                    roots.add(current.id)
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                roots.add(node.id)

        Visitor().visit(node)
        return roots


ROOT_NAME_PROJECTION = RootNameProjection()


def _registration_key_fingerprint(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    return _fingerprint_builder_value(node.slice)


def _class_name_from_expr(
    node: ast.AST, known_class_family: AstNameFamily
) -> str | None:
    return _terminal_name_in_family(node, known_class_family)


from .observation_families import (
    AccessorWrapperObservationFamily,
    AccessorWrapperObservationSpec,
    AssignmentRegistrationShapeSpec,
    AttributeErrorProbeObservationSpec,
    AttributeProbeObservationFamily,
    AttributeProbeObservationSpec,
    BuilderCallShapeFamily,
    BuilderCallShapeSpec,
    CallAttributeProbeObservationSpec,
    CallRegistrationShapeSpec,
    ClassMarkerObservationFamily,
    ClassMarkerObservationSpec,
    ClassObservationSpec,
    ConfigDispatchObservationFamily,
    ConfigDispatchObservationSpec,
    DataclassBodyFieldObservationSpec,
    DecoratorRegistrationShapeSpec,
    DualAxisResolutionObservationFamily,
    DualAxisResolutionObservationSpec,
    DynamicMethodInjectionObservationFamily,
    DynamicMethodInjectionObservationSpec,
    ExportDictShapeFamily,
    ExportDictShapeSpec,
    FieldObservationFamily,
    FieldObservationSpec,
    GetAttrProbeObservationSpec,
    HasAttrProbeObservationSpec,
    InitAssignmentFieldObservationSpec,
    InlineLiteralDispatchObservationSpec,
    InlineStringLiteralDispatchObservationFamily,
    InlineStringLiteralDispatchObservationSpec,
    InterfaceGenerationObservationFamily,
    InterfaceGenerationObservationSpec,
    KnownClassFamilyShapeSpec,
    LineageMappingObservationFamily,
    LineageMappingObservationSpec,
    LiteralDispatchObservationSpec,
    MethodShapeFamily,
    MethodShapeSpec,
    NumericLiteralDispatchObservationFamily,
    NumericLiteralDispatchObservationSpec,
    ObservationFamily,
    ProjectionHelperObservationFamily,
    ProjectionHelperObservationSpec,
    RegistrationShapeFamily,
    RegistrationShapeSpec,
    RuntimeTypeGenerationObservationFamily,
    RuntimeTypeGenerationObservationSpec,
    ScopedShapeWrapperFunctionFamily,
    ScopedShapeWrapperFunctionObservationSpec,
    ScopedShapeWrapperObservationSpec,
    ScopedShapeWrapperSpecFamily,
    ScopedShapeWrapperSpecObservationSpec,
    SentinelTypeAssignmentObservationSpec,
    SentinelTypeObservationFamily,
    SentinelTypeObservationSpec,
    SentinelTypeUsageObservationSpec,
    ShapeFamily,
    StandardAccessorWrapperObservationSpec,
    StandardClassMarkerObservationSpec,
    StandardConfigDispatchObservationSpec,
    StandardDualAxisResolutionObservationSpec,
    StandardDynamicMethodInjectionObservationSpec,
    StandardInterfaceGenerationObservationSpec,
    StandardLineageMappingObservationSpec,
    StandardProjectionHelperObservationSpec,
    StringLiteralDispatchObservationFamily,
    StringLiteralDispatchObservationSpec,
    TypeCallGenerationObservationSpec,
    TypedLiteralObservationFamily,
    TypedLiteralObservationSpec,
)
