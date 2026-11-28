"""Retype a module that imports a class only to annotate with it.

A guarded import (`if TYPE_CHECKING:`) that a cycle keeps hidden, read only in
annotations, needs the class as a type and nothing more. When a base of the
class sits below the module, the annotations name that base instead and the
base is imported at runtime: the guard is gone and so is the cycle. A whole-tree
type check proves the narrower type still serves every value's consumers; one
that reads a member only the class declares refuses the retype, named.
"""

from __future__ import annotations

import ast
import builtins
import copy
import keyword
import re
import typing
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from nominal_refactor_advisor.ast_field_decoders import declared_annotations
from nominal_refactor_advisor.audit.family import snake_case
from nominal_refactor_advisor.class_index import (
    CompactIndexedClass,
    CompactRepositoryPublicExposureIndex,
    InstalledClassDeclarations,
)
from nominal_refactor_advisor.codemod_import_scopes import ModuleImportScope
from nominal_refactor_advisor.codemod_imports import (
    ModuleImportInsertionPoint,
    ModuleImportMutation,
)
from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_payload import (
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceFileCreation,
)
from nominal_refactor_advisor.codemod_spacing import (
    DestinationInsertionSpacing,
    SourceInsertionBoundary,
)
from nominal_refactor_advisor.hidden_imports import ClosesCycle, GuardedImport
from nominal_refactor_advisor.lexical_bindings import ImportFromModuleName
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.owner_candidates import ModuleImportGraph, common_package
from nominal_refactor_advisor.refactors.class_base import AddClassBaseOperation
from nominal_refactor_advisor.refactors.class_overlap import (
    _line_difference_edits,
    annotation_modes,
    in_order_states,
)
from nominal_refactor_advisor.refactors.module_move import ImportedNameOwner, PromoteGuardedImportOperation
from nominal_refactor_advisor.refactors.source_insertion import EnsureImportOperation


class TypeOnlyReadRefused(NeedsInput):
    """A guarded import every one of whose names the module reads at most as a
    type, which no base below it can serve: each subclass names why."""


class NoLowerBase(TypeOnlyReadRefused):
    """The module reads the class only as a type, but no base of it sits below
    the module: a contract declared below, which the class inherits, is the input."""

    def __init__(
        self,
        reader: str,
        name: str,
        *,
        file_path: str = "",
        hub: str = "",
        owner: str = "",
    ) -> None:
        self.reader = reader
        self.name = name
        self.file_path = file_path
        self.hub = hub
        self.owner = owner or hub
        super().__init__(
            f"`{reader}` reads `{name}` only as a type, but no base of `{name}` sits below it: "
            "a contract base declared below it, which the class inherits, is the input",
            NeededInput.OPERATION,
        )


class BaseIsImplementationMixin(TypeOnlyReadRefused):
    """The bases below the module supply code but declare no contract: naming
    one would say the value is a mixin, erasing which type it is, and a type
    check cannot see that loss when the module reads no member."""

    def __init__(
        self,
        reader: str,
        name: str,
        bases: tuple[str, ...],
        *,
        file_path: str = "",
        hub: str = "",
        owner: str = "",
    ) -> None:
        self.reader = reader
        self.name = name
        self.bases = bases
        self.file_path = file_path
        self.hub = hub
        self.owner = owner or hub
        super().__init__(
            f"`{reader}` reads `{name}` only as a type, but its bases below it, "
            + ", ".join(f"`{base}`" for base in bases)
            + ", supply code and declare no contract (no abstract member, no registry family): "
            "a contract base declared below it, which the class inherits, is the input",
            NeededInput.OPERATION,
        )


class TypeCheckerReExport(TypeOnlyReadRefused):
    """Nothing in the module reads the name, not even an annotation: the import
    is there for type checkers reading the module's namespace, not for the module."""

    def __init__(self, reader: str, names: tuple[str, ...]) -> None:
        super().__init__(
            f"`{reader}` guards {', '.join(f'`{name}`' for name in names)} but reads it nowhere: "
            "a re-export for type checkers, whose place is the module's owner's to decide",
            NeededInput.SOURCE_CHANGE,
        )


class AliasNotClass(TypeOnlyReadRefused):
    """The name is a type alias, which has no bases: the alias itself, or what
    it spells, sitting below the module is the input."""

    def __init__(self, reader: str, owner: str, name: str) -> None:
        super().__init__(
            f"`{reader}` reads `{name}` only as a type, but `{owner}.{name}` is a type alias, not a class: "
            "it has no base to name; the alias sitting below it is the input",
            NeededInput.OPERATION,
        )


class RetypeReachesConsumer(NeedsInput):
    """The narrower type does not serve a consumer of the retyped values: each
    new type error names the consumer and what it needs of the class."""

    def __init__(self, retypes: str, errors: tuple[str, ...]) -> None:
        super().__init__(
            f"retyping {retypes} breaks {len(errors)} consumer(s): " + "; ".join(errors[:6]),
            NeededInput.SOURCE_CHANGE,
        )


@dataclass(frozen=True)
class TypeOnlyRead:
    """One guarded name a module reads only in its annotations, and the base it
    can name instead."""

    reader: str
    file_path: str
    hub: str
    name: str
    base_module: str
    base_name: str

    @classmethod
    def of(cls, snapshot: CodemodSourceSnapshot, imports: ModuleImportGraph, refusal: ClosesCycle) -> tuple[TypeOnlyRead, ...] | None:
        """Every name the refused guarded import binds, each with its base below
        the module; None when the import is not guarded or some name is read
        other than as a type. Raises `NoLowerBase` when a name has no base below."""

        hidden = refusal.hidden
        statement = hidden.statement
        if not isinstance(hidden, GuardedImport) or not isinstance(statement, ast.ImportFrom):
            return None
        if any(alias.asname for alias in statement.names):
            return None
        module = snapshot.module_nodes_by_file_path[hidden.module.file_path]
        hub = hidden.module.module_path_identity.resolve_import_from_module(
            imported_module=statement.module, level=statement.level
        )
        if hub is None:
            return None
        reader = hidden.module.module_name
        names = tuple(alias.name for alias in statement.names)
        if any(AnnotationReferences.of(module, name) is None for name in names):
            return None
        unread = tuple(name for name in names if AnnotationReferences.unread(module, name))
        if unread:
            raise TypeCheckerReExport(reader, unread)
        reads = []
        for name in names:
            owner = ImportedNameOwner.declaring_module(snapshot, hub, name) or hub
            if cls.alias(snapshot, owner, name):
                raise AliasNotClass(reader, owner, name)
            base = cls.lower_base(
                snapshot,
                imports,
                owner,
                name,
                reader,
                file_path=hidden.module.file_path,
                hub_module=hub,
            )
            reads.append(cls(reader, hidden.module.file_path, hub, name, *base))
        return tuple(reads)

    @staticmethod
    def alias(snapshot: CodemodSourceSnapshot, owner: str, name: str) -> bool:
        """Whether `owner` binds `name` at module level by assignment, not a class."""

        source = snapshot.module_import_graph.unique_source_file_for_module_name(owner)
        if source is None or snapshot.compact_class_index.class_for(f"{owner}.{name}") is not None:
            return False
        for node in snapshot.module_nodes_by_file_path[source.file_path].body:
            match node:
                case ast.Assign(targets=targets) if any(
                    isinstance(target, ast.Name) and target.id == name for target in targets
                ):
                    return True
                case ast.AnnAssign(target=ast.Name(id=bound)) | ast.TypeAlias(name=ast.Name(id=bound)) if bound == name:
                    return True
        return False

    @staticmethod
    def lower_base(
        snapshot: CodemodSourceSnapshot,
        imports: ModuleImportGraph,
        hub: str,
        name: str,
        reader: str,
        *,
        file_path: str = "",
        hub_module: str = "",
    ) -> tuple[str, str]:
        """The nearest contract base along the class's MRO declared in another
        scanned module that the reader can import at runtime without a cycle.

        An annotation says which type a value is; a base that only supplies code
        cannot say it, so only a contract base is named: one declaring an
        abstract member, or a registry family's class (its root or a member).
        Raises `NoLowerBase` when no base sits below the reader, and
        `BaseIsImplementationMixin` when only mixins do."""

        index = snapshot.compact_class_index
        authority = index.mro_authority
        symbol = f"{hub}.{name}"
        mro = authority.resolve(symbol).mro_type
        declarations = mro.declarations if mro is not None else authority.lookup_prefix(symbol)
        mixins: list[str] = []
        for declaration in declarations:
            if declaration.symbol == symbol or "." in declaration.qualname:
                continue
            module = declaration.module_name
            if module in (hub, reader) or module not in imports.imports:
                continue
            if imports.reaches(module, reader):
                continue
            if not ContractBase.declares(index.class_for(declaration.symbol) or declaration):
                mixins.append(declaration.symbol)
                continue
            return module, declaration.qualname
        if mixins:
            raise BaseIsImplementationMixin(
                reader,
                name,
                tuple(mixins),
                file_path=file_path,
                hub=hub_module or hub,
                owner=hub,
            )
        raise NoLowerBase(
            reader,
            name,
            file_path=file_path,
            hub=hub_module or hub,
            owner=hub,
        )


class ContractBase:
    """Whether a class declares what its subclasses are, not only code they reuse:
    an abstract member it declares, or membership in a registry family. A
    `Protocol` is structural, so it names no nominal identity."""

    PROTOCOL = f"{typing.Protocol.__module__}.{typing.Protocol.__qualname__}"

    @classmethod
    def declares(cls, declaration: CompactIndexedClass) -> bool:
        if any(reference.qualified_name == cls.PROTOCOL for reference in declaration.base_references):
            return False
        return bool(declaration.abstract_method_names) or declaration.is_registration_authority


@dataclass(frozen=True)
class AnnotationReferences:
    """Where a module names one of its names in annotations: every `Name` in an
    annotation and every string annotation spelling it, by source position."""

    names: tuple[ast.Name, ...]
    strings: tuple[ast.Constant, ...]

    @classmethod
    def of(cls, module: ast.Module, name: str) -> AnnotationReferences | None:
        """None when the module reads the name anywhere outside an annotation."""

        annotations = [root for root in cls.roots(module)]
        inside = {id(node) for root in annotations for node in ast.walk(root)}
        names, strings = [], []
        for node in ast.walk(module):
            if isinstance(node, ast.Name) and node.id == name:
                if id(node) not in inside:
                    return None
                names.append(node)
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) in inside
                and re.search(rf"\b{re.escape(name)}\b", node.value)
            ):
                strings.append(node)
        return cls(tuple(names), tuple(strings))

    @staticmethod
    def unread(module: ast.Module, name: str) -> bool:
        """Whether the module reads `name` nowhere: no `Name` spells it and no
        string does, save docstrings and `__all__`'s entries."""

        skipped = {
            id(node.value)
            for node in ast.walk(module)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        } | {
            id(entry)
            for node in module.body
            if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
            )
            and node.value is not None
            for entry in ast.walk(node.value)
        }
        for node in ast.walk(module):
            if isinstance(node, ast.Name) and node.id == name:
                return False
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in skipped
                and re.search(rf"\b{re.escape(name)}\b", node.value)
            ):
                return False
        return True

    @staticmethod
    def roots(module: ast.Module) -> Iterator[ast.expr]:
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                arguments = node.args
                for argument in (
                    *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                    *(a for a in (arguments.vararg, arguments.kwarg) if a is not None),
                ):
                    yield from declared_annotations(argument)
                for declared_annotation in declared_annotations(node):
                    yield declared_annotation
            elif isinstance(node, ast.AnnAssign):
                yield node.annotation

    def retyped(self, source: str, name: str, base: str) -> str:
        """`source` with each reference naming `base` instead of `name`."""

        lines = source.splitlines(keepends=True)
        spans: list[tuple[int, int, int, str]] = []
        for node in self.names:
            spans.append((node.lineno, node.col_offset, node.end_col_offset or node.col_offset, base))
        for node in self.strings:
            line = lines[node.lineno - 1].encode()
            if node.end_lineno != node.lineno:
                continue
            literal = line[node.col_offset : node.end_col_offset].decode()
            spans.append(
                (node.lineno, node.col_offset, node.end_col_offset or node.col_offset,
                 re.sub(rf"\b{re.escape(name)}\b", base, literal))
            )
        for lineno, start, end, text in sorted(spans, reverse=True):
            encoded = lines[lineno - 1].encode()
            lines[lineno - 1] = (encoded[:start] + text.encode() + encoded[end:]).decode()
        return "".join(lines)


@dataclass(frozen=True, kw_only=True)
class SwapGuardedImportOperation(RefactorRecipeOperation):
    """Inside the module's `if TYPE_CHECKING:` guard, import `base_name` from
    `base_module` instead of `name` from `module_name`."""

    module_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    base_module: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    base_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits(self, context: CodemodSelectorContext) -> tuple[ModuleImportMutation, ...]:
        source_path = self.required_source_path(context, self.operation_key())
        rationale = self.rationale or f"Annotate with `{self.base_name}`, the base of `{self.name}` below this module."
        return (
            ModuleImportMutation.remove_names(
                file_path=source_path,
                module_name=self.module_name,
                names=(self.name,),
                scope=ModuleImportScope.TYPE_CHECKING,
                rationale=rationale,
            ),
            ModuleImportMutation.from_source(
                file_path=source_path,
                import_source=f"from {self.base_module} import {self.base_name}\n",
                scope=ModuleImportScope.TYPE_CHECKING,
                rationale=rationale,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class RetypeTypeOnlyReadsOperation(RepositorySourceReprovedOperation):
    """Retype every listed module's annotations of a class to its base below
    them, swap each guarded import for the base, then promote it: one proof for
    them all, so a module and the callee it hands the value to change together.

    The promotion is `PromoteGuardedImportOperation`'s, which refuses an import
    that closes a cycle; the whole-tree type check refuses a retype whose value
    reaches a consumer of a member only the class declares."""

    readers: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())
    hubs: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())
    names: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())
    base_modules: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())
    base_names: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())

    @classmethod
    def of(cls, reads: tuple[TypeOnlyRead, ...]) -> RetypeTypeOnlyReadsOperation:
        return cls(
            target=SourceRewriteTarget(file_path=reads[0].file_path),
            readers=tuple(read.file_path for read in reads),
            hubs=tuple(read.hub for read in reads),
            names=tuple(read.name for read in reads),
            base_modules=tuple(read.base_module for read in reads),
            base_names=tuple(read.base_name for read in reads),
        )

    @property
    def described(self) -> str:
        return ", ".join(
            f"`{name}` to `{base}` in `{path.rsplit('/', 1)[-1]}`"
            for path, name, base in zip(self.readers, self.names, self.base_names, strict=True)
        )

    def source_edits_from_snapshot(self, snapshot: CodemodSourceSnapshot) -> tuple[NominalSourceEdit, ...]:
        overlay: dict[str, str] = {}
        for path, name, base in zip(self.readers, self.names, self.base_names, strict=True):
            source = overlay.get(path, snapshot.sources_by_file_path[path])
            references = AnnotationReferences.of(ast.parse(source), name)
            if references is None:
                raise NeedsInput(f"`{path}` reads `{name}` other than as a type", NeededInput.SOURCE_CHANGE)
            overlay[path] = references.retyped(source, name, base)
        steps: list[RefactorRecipeOperation] = []
        promoted: dict[str, set[tuple[str, str]]] = {}
        for path, hub, name, base_module, base in zip(
            self.readers, self.hubs, self.names, self.base_modules, self.base_names, strict=True
        ):
            spelled = self.spelled(snapshot, path, hub)
            steps.append(
                SwapGuardedImportOperation(
                    target=SourceRewriteTarget(file_path=path),
                    module_name=spelled, name=name, base_module=base_module, base_name=base,
                )
            )
            promoted.setdefault(path, set()).add((base_module, base))
        for path, bases in promoted.items():
            for base_module, base in sorted(bases):
                steps.append(
                    PromoteGuardedImportOperation(
                        target=SourceRewriteTarget(file_path=path), module_name=base_module, import_names=(base,)
                    )
                )
        states = in_order_states(snapshot.with_virtual_sources(overlay), tuple(steps), "retype-type-only")
        errors = states[-1].type_errors.new_since(snapshot.type_errors)
        if errors:
            raise RetypeReachesConsumer(self.described, errors)
        rationale = self.rationale_text(
            f"Annotate with each class's base below the module: {self.described}."
        )
        mutations_by_file: dict[str, list[ModuleImportMutation]] = {}
        for path, hub, name, base_module, base in zip(
            self.readers, self.hubs, self.names, self.base_modules, self.base_names, strict=True
        ):
            spelled = self.spelled(snapshot, path, hub)
            mutations_by_file.setdefault(path, []).extend(
                (
                    ModuleImportMutation.remove_names(
                        file_path=path,
                        module_name=spelled,
                        names=(name,),
                        scope=ModuleImportScope.TYPE_CHECKING,
                        rationale=rationale,
                    ),
                    ModuleImportMutation.from_source(
                        file_path=path,
                        import_source=f"from {base_module} import {base}\n",
                        scope=ModuleImportScope.RUNTIME,
                        rationale=rationale,
                    ),
                )
            )
        edits: list[NominalSourceEdit] = []
        for path, after in states[-1].sources_by_file_path.items():
            before = snapshot.sources_by_file_path.get(path)
            if before is None or before == after:
                continue
            composed = ModuleImportMutation.composed_sequence(
                mutations_by_file.get(path, ()),
                rationale=rationale,
            )
            if composed is not None and path in overlay:
                retyped = overlay[path]
                if retyped != before:
                    edits.extend(_line_difference_edits(path, before, retyped, rationale))
                if composed.additions or composed.removals:
                    edits.append(composed)
                continue
            edits.extend(_line_difference_edits(path, before, after, rationale))
        return tuple(edits)

    @staticmethod
    def spelled(snapshot: CodemodSourceSnapshot, path: str, hub: str) -> str:
        """The module name the guarded import spells for `hub`, relative or absolute."""

        identity = snapshot.module_import_graph.source_file_for_path(path).module_path_identity
        for statement in ModuleImportScope.TYPE_CHECKING.import_statements(snapshot.module_nodes_by_file_path[path]):
            if isinstance(statement, ast.ImportFrom) and identity.resolve_import_from_module(
                imported_module=statement.module, level=statement.level
            ) == hub:
                return ImportFromModuleName.from_node(statement).source
        return hub


INCOMPATIBLE_ABC_BASES = frozenset(
    (
        "typing.NamedTuple",
        "typing_extensions.NamedTuple",
        "typing.TypedDict",
        "typing_extensions.TypedDict",
        "typing.Protocol",
        "typing_extensions.Protocol",
        "enum.Enum",
        "enum.IntEnum",
        "enum.StrEnum",
        "enum.Flag",
        "enum.IntFlag",
        "enum.ReprEnum",
        "builtins.tuple",
    )
)

COMPATIBLE_ABC_METACLASSES = frozenset(
    (
        "builtins.type",
        "abc.ABCMeta",
        "pydantic._internal._model_construction.ModelMetaclass",
        "metaclass_registry.core.AutoRegisterMeta",
    )
)

PLAIN_METHOD_DECORATORS = frozenset(
    (
        "abc.abstractmethod",
        "typing.override",
        "typing_extensions.override",
        "typing.final",
        "typing_extensions.final",
        "functools.cache",
        "functools.lru_cache",
    )
)

PROPERTY_DECORATORS = frozenset(
    (
        "builtins.property",
        "functools.cached_property",
    )
)


@dataclass(frozen=True)
class ContractMember:
    """One member stub or typed attribute declared on an extracted contract ABC."""

    name: str
    source: str
    is_abstract: bool
    required_imports: tuple[str, ...]
    read_modules: frozenset[str]


@dataclass(frozen=True)
class ContractBaseCandidate:
    """One cycle-closing type-only class whose readers can share a new nominal ABC below them."""

    hub: str
    owner_module: str
    owner_file_path: str
    class_name: str
    readers: tuple[str, ...]
    reader_modules: tuple[str, ...]
    read_hubs: tuple[str, ...]
    accessed_members: tuple[str, ...]
    destination_module: str | None

    @classmethod
    def for_needs(
        cls,
        snapshot: CodemodSourceSnapshot,
        imports: ModuleImportGraph,
        hub: str,
        needs: tuple[tuple[ClosesCycle, NeedsInput], ...],
        *,
        contract_name: str | None = None,
    ) -> ContractBaseCandidate | None:
        """Select the highest-leverage extractable class among `NoLowerBase` /
        `BaseIsImplementationMixin` needs for `hub`, or `None` when none exist."""

        grouped: dict[tuple[str, str], list[NoLowerBase | BaseIsImplementationMixin]] = {}
        for _refusal, need in needs:
            if isinstance(need, (NoLowerBase, BaseIsImplementationMixin)) and need.file_path:
                grouped.setdefault((need.owner, need.name), []).append(need)
        if not grouped:
            return None
        ordered = sorted(
            grouped.items(),
            key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
        )
        first_error: NeedsInput | None = None
        for (owner_module, class_name), group in ordered:
            deduped: list[NoLowerBase | BaseIsImplementationMixin] = []
            seen_readers: set[str] = set()
            for item in group:
                if item.file_path not in seen_readers:
                    seen_readers.add(item.file_path)
                    deduped.append(item)
            try:
                return cls._build_candidate(
                    snapshot,
                    imports,
                    hub,
                    owner_module,
                    class_name,
                    tuple(deduped),
                    contract_name=contract_name,
                )
            except NeedsInput as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        return None

    @classmethod
    def _build_candidate(
        cls,
        snapshot: CodemodSourceSnapshot,
        imports: ModuleImportGraph,
        hub: str,
        owner_module: str,
        class_name: str,
        items: tuple[NoLowerBase | BaseIsImplementationMixin, ...],
        *,
        contract_name: str | None = None,
    ) -> ContractBaseCandidate:
        owner_source = snapshot.module_import_graph.unique_source_file_for_module_name(owner_module)
        if owner_source is None:
            raise NeedsInput(
                f"`{owner_module}.{class_name}` has no scanned declaring module",
                NeededInput.SOURCE_CHANGE,
            )
        _require_abc_compatible_class(snapshot, owner_module, class_name)
        readers = tuple(item.file_path for item in items)
        reader_modules = tuple(item.reader for item in items)
        read_hubs = tuple(item.hub for item in items)
        accessed = _accessed_members_across_readers(snapshot, readers, class_name)
        importers = (owner_module, *reader_modules)
        probe_dest = _probe_destination_module(
            snapshot,
            imports,
            owner_module,
            reader_modules,
            contract_name=contract_name,
        )
        members = _extract_contract_members(
            snapshot,
            imports,
            f"{owner_module}.{class_name}",
            accessed,
            dest_module=probe_dest or "",
            forbidden_modules=frozenset(importers),
        )
        member_reads = frozenset().union(*(member.read_modules for member in members))
        destination = _choose_destination_module(
            snapshot,
            imports,
            owner_module,
            reader_modules,
            member_reads,
            contract_name=contract_name,
        )
        return cls(
            hub=hub,
            owner_module=owner_module,
            owner_file_path=owner_source.file_path,
            class_name=class_name,
            readers=readers,
            reader_modules=reader_modules,
            read_hubs=read_hubs,
            accessed_members=accessed,
            destination_module=destination,
        )


def _require_valid_contract_pick(contract_name: str, docstring: str) -> None:
    if not contract_name.isidentifier() or keyword.iskeyword(contract_name):
        raise NeedsInput(
            f"`{contract_name}` is not a valid class name",
            NeededInput.OWNER_PICK,
        )
    cleaned = docstring.strip()
    if not cleaned or any(token in cleaned for token in ('"""', "\\", "\n")):
        raise NeedsInput(
            f"`{contract_name}` needs a one-line docstring without triple quotes or backslashes",
            NeededInput.OWNER_PICK,
        )


def _is_abc_compatible_metaclass(
    snapshot: CodemodSourceSnapshot,
    metaclass_name: str | None,
    *,
    decl: CompactIndexedClass | None = None,
) -> bool:
    if metaclass_name is None or metaclass_name in COMPATIBLE_ABC_METACLASSES:
        return True
    resolved_start = metaclass_name
    if "." not in resolved_start and decl is not None and decl.file_path in snapshot.module_nodes_by_file_path:
        parsed = snapshot.parsed_module_for_source_path(decl.file_path)
        class_node = _top_level_class_node(snapshot, decl.file_path, decl.qualname)
        line = class_node.lineno if class_node is not None else 1
        witness = snapshot.module_binding_proof.reference_or_builtin_witness_at(
            parsed, ast.Name(id=resolved_start, ctx=ast.Load()), line=line
        )
        if witness is not None:
            resolved_start = witness.qualified_name
    index = snapshot.compact_class_index
    exposure = CompactRepositoryPublicExposureIndex(())
    pending = [resolved_start]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in COMPATIBLE_ABC_METACLASSES:
            return True
        target = index.class_for(current) or InstalledClassDeclarations.declaration(
            current, exposure
        )
        if target is None:
            return False
        for ref in target.base_references:
            pending.append(ref.qualified_name)
    return False


def _require_abc_compatible_class(
    snapshot: CodemodSourceSnapshot, owner_module: str, class_name: str
) -> ast.ClassDef:
    source_file = snapshot.module_import_graph.unique_source_file_for_module_name(owner_module)
    if source_file is None:
        raise NeedsInput(
            f"`{owner_module}.{class_name}` has no scanned declaring module",
            NeededInput.SOURCE_CHANGE,
        )
    module_node = snapshot.module_nodes_by_file_path[source_file.file_path]
    class_node = next(
        (
            stmt
            for stmt in module_node.body
            if isinstance(stmt, ast.ClassDef) and stmt.name == class_name
        ),
        None,
    )
    if class_node is None:
        raise NeedsInput(
            f"`{owner_module}.{class_name}` is not a top-level class declaration",
            NeededInput.SOURCE_CHANGE,
        )
    symbol = f"{owner_module}.{class_name}"
    index = snapshot.compact_class_index
    exposure = CompactRepositoryPublicExposureIndex(())
    root = index.class_for(symbol)
    if root is None:
        raise NeedsInput(
            f"`{symbol}` is not indexed as a scanned class",
            NeededInput.PROOF,
        )
    pending = [root]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current.symbol in seen:
            continue
        seen.add(current.symbol)
        for metaclass_name in current.metaclass_names:
            if not _is_abc_compatible_metaclass(snapshot, metaclass_name, decl=current):
                raise NeedsInput(
                    f"`{symbol}` has metaclass `{metaclass_name}`, not proved compatible with `abc.ABC`",
                    NeededInput.PROOF,
                )
        for base_ref in current.base_references:
            base_name = base_ref.qualified_name
            if base_name in INCOMPATIBLE_ABC_BASES:
                raise NeedsInput(
                    f"`{symbol}` inherits `{base_name}`, which cannot add an `abc.ABC` contract base mechanically",
                    NeededInput.SOURCE_CHANGE,
                )
            if base_name in ("builtins.object", "abc.ABC", "typing.Generic", "typing_extensions.Generic"):
                continue
            ancestor = index.class_for(base_name) or InstalledClassDeclarations.declaration(
                base_name, exposure
            )
            if ancestor is not None:
                pending.append(ancestor)
    return class_node


def _annotation_mentions_name(annotation: ast.expr | None, name: str) -> bool:
    if annotation is None:
        return False
    return re.search(rf"\b{re.escape(name)}\b", ast.unparse(annotation)) is not None


def _accessed_members_in_module(module: ast.Module, class_name: str) -> set[str]:
    attrs: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local_vars: set[str] = set()
            for arg in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
                *(a for a in (node.args.vararg, node.args.kwarg) if a is not None),
            ):
                if _annotation_mentions_name(arg.annotation, class_name):
                    local_vars.add(arg.arg)
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.AnnAssign)
                    and isinstance(sub.target, ast.Name)
                    and _annotation_mentions_name(sub.annotation, class_name)
                ):
                    local_vars.add(sub.target.id)
            if local_vars:
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id in local_vars
                        and not (sub.attr.startswith("__") and sub.attr.endswith("__"))
                    ):
                        attrs.add(sub.attr)
        elif isinstance(node, ast.ClassDef):
            self_fields: set[str] = set()
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and _annotation_mentions_name(stmt.annotation, class_name)
                ):
                    self_fields.add(stmt.target.id)
            if self_fields:
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Attribute)
                        and isinstance(sub.value.value, ast.Name)
                        and sub.value.value.id == "self"
                        and sub.value.attr in self_fields
                        and not (sub.attr.startswith("__") and sub.attr.endswith("__"))
                    ):
                        attrs.add(sub.attr)
    return attrs


def _accessed_members_across_readers(
    snapshot: CodemodSourceSnapshot, readers: tuple[str, ...], class_name: str
) -> tuple[str, ...]:
    found: set[str] = set()
    for path in readers:
        found.update(_accessed_members_in_module(snapshot.module_nodes_by_file_path[path], class_name))
    return tuple(sorted(found))


def _top_level_class_node(snapshot: CodemodSourceSnapshot, file_path: str, qualname: str) -> ast.ClassDef | None:
    if "." in qualname:
        return None
    module = snapshot.module_nodes_by_file_path.get(file_path)
    if module is None:
        return None
    return next(
        (stmt for stmt in module.body if isinstance(stmt, ast.ClassDef) and stmt.name == qualname),
        None,
    )


def _iter_annotation_references(
    expr: ast.expr, *, inside_literal: bool = False
) -> Iterator[ast.expr]:
    match expr:
        case ast.Subscript(value=val, slice=slc):
            yield from _iter_annotation_references(val, inside_literal=inside_literal)
            is_lit = (
                (isinstance(val, ast.Name) and val.id == "Literal")
                or (isinstance(val, ast.Attribute) and val.attr == "Literal")
            )
            yield from _iter_annotation_references(slc, inside_literal=inside_literal or is_lit)
        case ast.Name():
            yield expr
        case ast.Attribute():
            yield expr
        case ast.Constant(value=str(text)) if not inside_literal:
            try:
                parsed = ast.parse(text, mode="eval").body
            except SyntaxError:
                return
            yield from _iter_annotation_references(parsed, inside_literal=False)
        case _:
            for child in ast.iter_child_nodes(expr):
                if isinstance(child, ast.expr):
                    yield from _iter_annotation_references(child, inside_literal=inside_literal)


def _explicit_import_in_module(
    module: ast.Module,
    identity: object,
    name: str,
) -> tuple[str, str, str | None] | None:
    """Return `(resolved_module, imported_name, asname)` for a top-level or
    `TYPE_CHECKING` import binding `name` in `module`, or `None`."""

    statements: list[ast.stmt] = list(module.body)
    statements.extend(ModuleImportScope.TYPE_CHECKING.import_statements(module))
    for stmt in statements:
        match stmt:
            case ast.ImportFrom(module=mod, names=aliases, level=level):
                resolved_mod = identity.resolve_import_from_module(  # type: ignore[attr-defined]
                    imported_module=mod, level=level
                )
                if resolved_mod is None:
                    continue
                for alias in aliases:
                    bound = alias.asname or alias.name
                    if bound == name and alias.name != "*":
                        return (resolved_mod, alias.name, alias.asname)
            case ast.Import(names=aliases):
                for alias in aliases:
                    bound = alias.asname or alias.name.partition(".")[0]
                    if bound == name:
                        return (alias.name, "", alias.asname)
    return None


def _resolve_annotation_imports(
    snapshot: CodemodSourceSnapshot,
    imports: ModuleImportGraph,
    decl_module: str,
    decl_file_path: str,
    member_name: str,
    exprs: tuple[ast.expr, ...],
    line: int,
    bound_type_params: frozenset[str],
    dest_module: str,
    forbidden_modules: frozenset[str],
) -> tuple[tuple[str, ...], frozenset[str]]:
    parsed_decl = snapshot.parsed_module_for_source_path(decl_file_path)
    identity = parsed_decl.module_path_identity
    bindings = snapshot.module_binding_proof
    builtin_names = frozenset(vars(builtins)) | frozenset(("None", "True", "False"))
    import_lines: list[str] = []
    read_modules: set[str] = set()
    seen_locals: set[str] = set()

    for expr in exprs:
        for ref in _iter_annotation_references(expr):
            root_name = ref.id if isinstance(ref, ast.Name) else None
            if isinstance(ref, ast.Attribute):
                cur: ast.expr = ref
                while isinstance(cur, ast.Attribute):
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    root_name = cur.id
            if root_name is None or root_name in builtin_names or root_name in bound_type_params:
                continue
            if root_name in seen_locals:
                continue
            seen_locals.add(root_name)

            explicit = _explicit_import_in_module(parsed_decl.module, identity, root_name)
            if explicit is not None:
                imp_mod, imp_name, asname = explicit
                if imp_mod in imports.imports:
                    owner = (
                        ImportedNameOwner.declaring_module(snapshot, imp_mod, imp_name)
                        if imp_name
                        else imp_mod
                    ) or imp_mod
                    if owner != dest_module:
                        if owner in forbidden_modules or any(
                            imports.reaches(owner, target) for target in forbidden_modules
                        ):
                            raise NeedsInput(
                                f"contract member `{member_name}` references `{root_name}` from `{owner}`, "
                                f"which reaches the cycle",
                                NeededInput.SOURCE_CHANGE,
                            )
                        read_modules.add(owner)
                        if imp_name:
                            line_str = (
                                f"from {owner} import {imp_name} as {asname}\n"
                                if asname
                                else f"from {owner} import {imp_name}\n"
                            )
                        else:
                            line_str = f"import {owner} as {asname}\n" if asname else f"import {owner}\n"
                        import_lines.append(line_str)
                else:
                    if imp_name:
                        line_str = (
                            f"from {imp_mod} import {imp_name} as {asname}\n"
                            if asname
                            else f"from {imp_mod} import {imp_name}\n"
                        )
                    else:
                        line_str = f"import {imp_mod} as {asname}\n" if asname else f"import {imp_mod}\n"
                    import_lines.append(line_str)
                continue

            witness = bindings.reference_or_builtin_witness_at(parsed_decl, ast.Name(id=root_name, ctx=ast.Load()), line=line)
            if witness is not None:
                qname = witness.qualified_name
                if qname.startswith("builtins."):
                    continue
                if qname.startswith(f"{decl_module}."):
                    if decl_module == dest_module:
                        continue
                    raise NeedsInput(
                        f"contract member `{member_name}` references `{root_name}` declared in `{decl_module}`, "
                        f"which sits above the readers",
                        NeededInput.SOURCE_CHANGE,
                    )
                mod_part, _, sym_part = qname.rpartition(".")
                if mod_part in imports.imports:
                    if mod_part == dest_module:
                        continue
                    if mod_part in forbidden_modules or any(
                        imports.reaches(mod_part, target) for target in forbidden_modules
                    ):
                        raise NeedsInput(
                            f"contract member `{member_name}` references `{qname}`, which reaches the cycle",
                            NeededInput.SOURCE_CHANGE,
                        )
                    read_modules.add(mod_part)
                    import_lines.append(f"from {mod_part} import {sym_part}\n")
                elif mod_part:
                    import_lines.append(f"from {mod_part} import {sym_part}\n")
                continue

            raise NeedsInput(
                f"contract member `{member_name}` references unresolved type `{root_name}` in `{decl_module}`",
                NeededInput.SOURCE_CHANGE,
            )

    return tuple(dict.fromkeys(import_lines)), frozenset(read_modules)


def _stub_arguments(args: ast.arguments) -> ast.arguments:
    cloned = copy.deepcopy(args)
    cloned.defaults = [
        default if isinstance(default, ast.Constant) else ast.Constant(value=...)
        for default in cloned.defaults
    ]
    cloned.kw_defaults = [
        None
        if default is None
        else (default if isinstance(default, ast.Constant) else ast.Constant(value=...))
        for default in cloned.kw_defaults
    ]
    return cloned


def _annotation_exprs_from_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.expr, ...]:
    exprs: list[ast.expr] = []
    for arg in (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        *(a for a in (node.args.vararg, node.args.kwarg) if a is not None),
    ):
        if arg.annotation is not None:
            exprs.append(arg.annotation)
    if node.returns is not None:
        exprs.append(node.returns)
    for param in getattr(node, "type_params", ()):
        bound = getattr(param, "bound", None)
        if isinstance(bound, ast.expr):
            exprs.append(bound)
    return tuple(exprs)


def _extract_contract_members(
    snapshot: CodemodSourceSnapshot,
    imports: ModuleImportGraph,
    owner_symbol: str,
    member_names: tuple[str, ...],
    *,
    dest_module: str,
    forbidden_modules: frozenset[str],
) -> tuple[ContractMember, ...]:
    if not member_names:
        return ()
    index = snapshot.compact_class_index
    authority = index.mro_authority
    mro = authority.resolve(owner_symbol).mro_type
    declarations = mro.declarations if mro is not None else authority.lookup_prefix(owner_symbol)
    scanned_decls = [
        decl
        for decl in declarations
        if decl.file_path in snapshot.module_nodes_by_file_path
    ]
    extracted: list[ContractMember] = []
    for member_name in member_names:
        member = _extract_one_contract_member(
            snapshot,
            imports,
            owner_symbol,
            scanned_decls,
            member_name,
            dest_module=dest_module,
            forbidden_modules=forbidden_modules,
        )
        extracted.append(member)
    return tuple(extracted)


def _extract_one_contract_member(
    snapshot: CodemodSourceSnapshot,
    imports: ModuleImportGraph,
    owner_symbol: str,
    scanned_decls: list[CompactIndexedClass],
    member_name: str,
    *,
    dest_module: str,
    forbidden_modules: frozenset[str],
) -> ContractMember:
    bindings = snapshot.module_binding_proof
    for decl in scanned_decls:
        class_node = _top_level_class_node(snapshot, decl.file_path, decl.qualname)
        if class_node is None:
            continue
        parsed_decl = snapshot.parsed_module_for_source_path(decl.file_path)
        class_type_params = frozenset(
            getattr(param, "name", "")
            for param in getattr(class_node, "type_params", ())
        )
        for stmt in class_node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == member_name:
                is_property = False
                is_classmethod = False
                is_staticmethod = False
                for dec in stmt.decorator_list:
                    dec_expr = dec.func if isinstance(dec, ast.Call) else dec
                    witness = bindings.reference_or_builtin_witness_at(parsed_decl, dec_expr, line=dec.lineno)
                    qname = witness.qualified_name if witness is not None else ""
                    if qname in PROPERTY_DECORATORS:
                        is_property = True
                    elif qname == "builtins.classmethod":
                        is_classmethod = True
                    elif qname == "builtins.staticmethod":
                        is_staticmethod = True
                    elif qname in PLAIN_METHOD_DECORATORS:
                        continue
                    else:
                        raise NeedsInput(
                            f"contract member `{owner_symbol}.{member_name}` uses decorator `@{ast.unparse(dec)}`",
                            NeededInput.PROOF,
                        )
                fn_type_params = class_type_params | frozenset(
                    getattr(param, "name", "") for param in getattr(stmt, "type_params", ())
                )
                exprs = _annotation_exprs_from_function(stmt)
                req_imports, read_mods = _resolve_annotation_imports(
                    snapshot,
                    imports,
                    decl.module_name,
                    decl.file_path,
                    member_name,
                    exprs,
                    stmt.lineno,
                    fn_type_params,
                    dest_module,
                    forbidden_modules,
                )
                stub_decorators: list[ast.expr] = []
                if is_property:
                    stub_decorators.append(ast.Name(id="property", ctx=ast.Load()))
                elif is_classmethod:
                    stub_decorators.append(ast.Name(id="classmethod", ctx=ast.Load()))
                elif is_staticmethod:
                    stub_decorators.append(ast.Name(id="staticmethod", ctx=ast.Load()))
                stub_decorators.append(ast.Name(id="abstractmethod", ctx=ast.Load()))
                stub_cls = ast.AsyncFunctionDef if isinstance(stmt, ast.AsyncFunctionDef) else ast.FunctionDef
                stub_node = stub_cls(
                    name=stmt.name,
                    args=_stub_arguments(stmt.args),
                    body=[ast.Expr(value=ast.Constant(value=...))],
                    decorator_list=stub_decorators,
                    returns=copy.deepcopy(stmt.returns),
                    type_params=copy.deepcopy(getattr(stmt, "type_params", [])),
                )
                ast.fix_missing_locations(stub_node)
                return ContractMember(
                    name=member_name,
                    source=ast.unparse(stub_node),
                    is_abstract=True,
                    required_imports=req_imports,
                    read_modules=read_mods,
                )
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == member_name
            ):
                req_imports, read_mods = _resolve_annotation_imports(
                    snapshot,
                    imports,
                    decl.module_name,
                    decl.file_path,
                    member_name,
                    (stmt.annotation,),
                    stmt.lineno,
                    class_type_params,
                    dest_module,
                    forbidden_modules,
                )
                return ContractMember(
                    name=member_name,
                    source=f"{member_name}: {ast.unparse(stmt.annotation)}",
                    is_abstract=False,
                    required_imports=req_imports,
                    read_modules=read_mods,
                )
        for stmt in class_node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                init_param_annotations: dict[str, ast.expr] = {
                    arg.arg: arg.annotation
                    for arg in (
                        *stmt.args.posonlyargs,
                        *stmt.args.args,
                        *stmt.args.kwonlyargs,
                    )
                    if arg.annotation is not None
                }
                for sub in ast.walk(stmt):
                    if (
                        isinstance(sub, ast.AnnAssign)
                        and isinstance(sub.target, ast.Attribute)
                        and isinstance(sub.target.value, ast.Name)
                        and sub.target.value.id == "self"
                        and sub.target.attr == member_name
                    ):
                        req_imports, read_mods = _resolve_annotation_imports(
                            snapshot,
                            imports,
                            decl.module_name,
                            decl.file_path,
                            member_name,
                            (sub.annotation,),
                            sub.lineno,
                            class_type_params,
                            dest_module,
                            forbidden_modules,
                        )
                        return ContractMember(
                            name=member_name,
                            source=f"{member_name}: {ast.unparse(sub.annotation)}",
                            is_abstract=False,
                            required_imports=req_imports,
                            read_modules=read_mods,
                        )
                    if isinstance(sub, ast.Assign) and any(
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == member_name
                        for target in sub.targets
                    ):
                        inferred_expr: ast.expr | None = None
                        if isinstance(sub.value, ast.Name) and sub.value.id in init_param_annotations:
                            inferred_expr = init_param_annotations[sub.value.id]
                        elif isinstance(sub.value, ast.Call) and isinstance(
                            sub.value.func, (ast.Name, ast.Attribute)
                        ):
                            callee_witness = bindings.reference_or_builtin_witness_at(
                                parsed_decl, sub.value.func, line=sub.lineno
                            )
                            if callee_witness is not None:
                                qn = callee_witness.qualified_name
                                is_class_callee = (
                                    snapshot.compact_class_index.class_for(qn) is not None
                                    or InstalledClassDeclarations.declaration(
                                        qn, CompactRepositoryPublicExposureIndex(())
                                    )
                                    is not None
                                )
                                if is_class_callee:
                                    inferred_expr = sub.value.func
                        if inferred_expr is not None:
                            req_imports, read_mods = _resolve_annotation_imports(
                                snapshot,
                                imports,
                                decl.module_name,
                                decl.file_path,
                                member_name,
                                (inferred_expr,),
                                sub.lineno,
                                class_type_params,
                                dest_module,
                                forbidden_modules,
                            )
                            return ContractMember(
                                name=member_name,
                                source=f"{member_name}: {ast.unparse(inferred_expr)}",
                                is_abstract=False,
                                required_imports=req_imports,
                                read_modules=read_mods,
                            )
    raise NeedsInput(
        f"cannot derive typed contract declaration for `{owner_symbol}.{member_name}`",
        NeededInput.SOURCE_CHANGE,
    )


def _name_bound_in_module(snapshot: CodemodSourceSnapshot, module_name: str, symbol: str) -> bool:
    source = snapshot.module_import_graph.unique_source_file_for_module_name(module_name)
    if source is None:
        return False
    module = snapshot.module_nodes_by_file_path[source.file_path]
    for stmt in module.body:
        match stmt:
            case ast.ClassDef(name=n) | ast.FunctionDef(name=n) | ast.AsyncFunctionDef(name=n) if n == symbol:
                return True
            case ast.Assign(targets=targets) if any(
                isinstance(t, ast.Name) and t.id == symbol for t in targets
            ):
                return True
            case ast.AnnAssign(target=ast.Name(id=n)) | ast.TypeAlias(name=ast.Name(id=n)) if n == symbol:
                return True
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases) if any(
                (a.asname or a.name) == symbol for a in aliases
            ):
                return True
    return False


def _package_chain_reaches_any(
    imports: ModuleImportGraph, module_name: str, targets: tuple[str, ...]
) -> bool:
    parts = module_name.split(".")
    for length in range(1, len(parts)):
        pkg = ".".join(parts[:length])
        if pkg in imports.imports and any(pkg == t or imports.reaches(pkg, t) for t in targets):
            return True
    return False


def _probe_destination_module(
    snapshot: CodemodSourceSnapshot,
    imports: ModuleImportGraph,
    owner_module: str,
    reader_modules: tuple[str, ...],
    *,
    contract_name: str | None = None,
) -> str | None:
    if (
        len(reader_modules) == 1
        and reader_modules[0] != owner_module
        and imports.reaches(owner_module, reader_modules[0])
        and not imports.reaches(reader_modules[0], owner_module)
        and (contract_name is None or not _name_bound_in_module(snapshot, reader_modules[0], contract_name))
    ):
        return reader_modules[0]
    importers = (owner_module, *reader_modules)
    lower = imports.shared_lower_module(
        importers,
        frozenset(),
        above=importers,
        modes=annotation_modes(snapshot),
        source=owner_module,
    )
    if lower is not None and lower.module != owner_module and (
        contract_name is None or not _name_bound_in_module(snapshot, lower.module, contract_name)
    ):
        return lower.module
    return None


def _choose_destination_module(
    snapshot: CodemodSourceSnapshot,
    imports: ModuleImportGraph,
    owner_module: str,
    reader_modules: tuple[str, ...],
    member_reads: frozenset[str],
    *,
    contract_name: str | None = None,
) -> str | None:
    importers = (owner_module, *reader_modules)
    if (
        len(reader_modules) == 1
        and reader_modules[0] != owner_module
        and imports.reaches(owner_module, reader_modules[0])
        and not imports.reaches(reader_modules[0], owner_module)
        and (contract_name is None or not _name_bound_in_module(snapshot, reader_modules[0], contract_name))
        and not any(
            read != reader_modules[0]
            and (read in importers or any(imports.reaches(read, imp) for imp in importers))
            for read in member_reads
        )
    ):
        return reader_modules[0]
    lower = imports.shared_lower_module(
        importers,
        member_reads,
        above=importers,
        modes=annotation_modes(snapshot),
        source=owner_module,
    )
    if lower is not None and lower.module != owner_module and (
        contract_name is None or not _name_bound_in_module(snapshot, lower.module, contract_name)
    ):
        return lower.module
    pkg = common_package(importers) or common_package((owner_module,))
    pkg_file = snapshot.module_import_graph.unique_source_file_for_module_name(pkg) if pkg else None
    if pkg_file is None or Path(pkg_file.file_path).name != "__init__.py":
        raise NeedsInput(
            f"no module or package below `{owner_module}` and its readers can host a contract base",
            NeededInput.SOURCE_CHANGE,
        )
    if _package_chain_reaches_any(imports, f"{pkg}.__probe__", importers):
        raise NeedsInput(
            f"package `{pkg}` reaches `{owner_module}` or its readers, so a new contract module inside it would cycle",
            NeededInput.SOURCE_CHANGE,
        )
    if any(
        read in importers or any(imports.reaches(read, imp) for imp in importers)
        for read in member_reads
    ):
        raise NeedsInput(
            f"contract member annotations read a module that reaches `{owner_module}` or its readers",
            NeededInput.SOURCE_CHANGE,
        )
    if contract_name is None:
        return None
    home = f"{pkg}.{snake_case(contract_name)}"
    if home in imports.imports:
        raise NeedsInput(
            f"module `{home}` already exists; pick another contract base name",
            NeededInput.OWNER_PICK,
        )
    return home


def _render_contract_class(
    contract_name: str,
    docstring: str,
    members: tuple[ContractMember, ...],
) -> str:
    lines = [
        f"class {contract_name}(ABC):",
        f'    """{docstring.strip()}"""',
        "",
        "    __slots__ = ()",
    ]
    attrs = [m for m in members if not m.is_abstract]
    methods = [m for m in members if m.is_abstract]
    for attr in attrs:
        lines.append(f"    {attr.source}")
    for method in methods:
        lines.append("")
        for method_line in method.source.splitlines():
            lines.append(f"    {method_line}")
    return "\n".join(lines) + "\n"


def _render_new_contract_module(
    contract_name: str,
    docstring: str,
    members: tuple[ContractMember, ...],
) -> str:
    needs_abstractmethod = any(m.is_abstract for m in members)
    abc_import = (
        "from abc import ABC, abstractmethod\n"
        if needs_abstractmethod
        else "from abc import ABC\n"
    )
    extra_imports = sorted({imp for m in members for imp in m.required_imports})
    header = [
        f'"""{docstring.strip()}"""',
        "",
        "from __future__ import annotations",
        "",
        abc_import.rstrip("\n"),
        *(imp.rstrip("\n") for imp in extra_imports),
        "",
        "",
    ]
    return "\n".join(header) + _render_contract_class(contract_name, docstring, members)


def _insert_class_after_imports(source: str, file_path: str, class_source: str) -> str:
    module = ast.parse(source)
    insertion_line = ModuleImportInsertionPoint(source, file_path, module).line_number
    spacing = DestinationInsertionSpacing.from_source(
        source,
        insertion_line,
        inserted_source_is_import_block=False,
        boundary=SourceInsertionBoundary.TWO_BLANK_LINES,
    )
    lines = source.splitlines(keepends=True)
    block = f"{spacing.leading_separator}{class_source.strip(chr(10))}{spacing.trailing_separator}"
    idx = max(0, min(len(lines), insertion_line - 1))
    lines.insert(idx, block)
    return "".join(lines)


@dataclass(frozen=True, kw_only=True)
class RemoveGuardedImportOperation(RefactorRecipeOperation):
    """Remove `name` from `module_name` inside the module's `if TYPE_CHECKING:` guard."""

    module_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits(self, context: CodemodSelectorContext) -> tuple[ModuleImportMutation, ...]:
        source_path = self.required_source_path(context, self.operation_key())
        return (
            ModuleImportMutation.remove_names(
                file_path=source_path,
                module_name=self.module_name,
                names=(self.name,),
                scope=ModuleImportScope.TYPE_CHECKING,
                rationale=self.rationale or f"Drop guarded import of `{self.name}`.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class DeclareContractBaseOperation(RepositorySourceReprovedOperation):
    """Declare a nominal `ABC` contract base below type-only cycle readers, make
    the hub class inherit it, retype each reader's annotations to the contract
    base, drop the `if TYPE_CHECKING:` import, and prove zero new type errors
    across the whole repository."""

    readers: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())
    hubs: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())
    class_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    contract_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    docstring: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    destination_module: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def _destination_file_path(self, snapshot: CodemodSourceSnapshot) -> tuple[str, bool]:
        existing = snapshot.module_import_graph.unique_source_file_for_module_name(
            self.destination_module
        )
        if existing is not None:
            return existing.file_path, False
        pkg, _, stem = self.destination_module.rpartition(".")
        pkg_file = (
            snapshot.module_import_graph.unique_source_file_for_module_name(pkg)
            if pkg
            else None
        )
        if pkg_file is None or Path(pkg_file.file_path).name != "__init__.py":
            raise NeedsInput(
                f"`{self.destination_module}` would be a new module, but `{pkg}` has no `__init__.py`",
                NeededInput.SOURCE_CHANGE,
            )
        return (Path(pkg_file.file_path).parent / f"{stem}.py").as_posix(), True

    def source_file_creations(
        self, context: CodemodSelectorContext
    ) -> tuple[SourceFileCreation, ...]:
        snapshot = context.execution_snapshot()
        dest_path, is_new = self._destination_file_path(snapshot)
        if not is_new:
            return ()
        return (
            SourceFileCreation.from_operation(
                self,
                requested_path=dest_path,
                source_index=context.source_index,
                source="",
            ),
        )

    @property
    def described(self) -> str:
        return ", ".join(
            f"`{self.class_name}` to `{self.contract_name}` in `{path.rsplit('/', 1)[-1]}`"
            for path in self.readers
        )

    def _build_state(
        self,
        snapshot: CodemodSourceSnapshot,
        owner_file_path: str,
        owner_module: str,
        dest_path: str,
        dest_is_new: bool,
        member_names: tuple[str, ...],
    ) -> CodemodSourceSnapshot:
        imports = ModuleImportGraph(snapshot.module_import_graph.import_edges_by_module)
        reader_modules = tuple(
            snapshot.parsed_module_for_source_path(path).module_name for path in self.readers
        )
        forbidden = frozenset((owner_module, *reader_modules))
        members = _extract_contract_members(
            snapshot,
            imports,
            f"{owner_module}.{self.class_name}",
            member_names,
            dest_module=self.destination_module,
            forbidden_modules=forbidden,
        )
        overlay: dict[str, str] = {}
        for path in self.readers:
            source = overlay.get(path, snapshot.sources_by_file_path[path])
            references = AnnotationReferences.of(ast.parse(source), self.class_name)
            if references is None:
                raise NeedsInput(
                    f"`{path}` reads `{self.class_name}` other than as a type",
                    NeededInput.SOURCE_CHANGE,
                )
            overlay[path] = references.retyped(source, self.class_name, self.contract_name)

        steps: list[RefactorRecipeOperation] = []
        if dest_is_new:
            overlay[dest_path] = _render_new_contract_module(
                self.contract_name, self.docstring, members
            )
        else:
            dest_src = overlay.get(dest_path, snapshot.sources_by_file_path[dest_path])
            class_src = _render_contract_class(self.contract_name, self.docstring, members)
            overlay[dest_path] = _insert_class_after_imports(dest_src, dest_path, class_src)
            needs_abstractmethod = any(m.is_abstract for m in members)
            abc_imp = (
                "from abc import ABC, abstractmethod\n"
                if needs_abstractmethod
                else "from abc import ABC\n"
            )
            steps.append(
                EnsureImportOperation(
                    target=SourceRewriteTarget(file_path=dest_path),
                    import_source=abc_imp,
                )
            )
            for req_imp in sorted({imp for m in members for imp in m.required_imports}):
                steps.append(
                    EnsureImportOperation(
                        target=SourceRewriteTarget(file_path=dest_path),
                        import_source=req_imp,
                    )
                )

        steps.append(
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=owner_file_path),
                import_source=f"from {self.destination_module} import {self.contract_name}\n",
            )
        )
        steps.append(
            AddClassBaseOperation(
                target=SourceRewriteTarget(
                    file_path=owner_file_path, qualname=self.class_name
                ),
                base_name=self.contract_name,
            )
        )
        for path, hub in zip(self.readers, self.hubs, strict=True):
            spelled = RetypeTypeOnlyReadsOperation.spelled(snapshot, path, hub)
            steps.append(
                RemoveGuardedImportOperation(
                    target=SourceRewriteTarget(file_path=path),
                    module_name=spelled,
                    name=self.class_name,
                )
            )
            if path != dest_path:
                steps.append(
                    EnsureImportOperation(
                        target=SourceRewriteTarget(file_path=path),
                        import_source=f"from {self.destination_module} import {self.contract_name}\n",
                    )
                )
        states = in_order_states(
            snapshot.with_virtual_sources(overlay),
            tuple(steps),
            f"declare-contract-{self.contract_name}",
        )
        return states[-1]

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        _require_valid_contract_pick(self.contract_name, self.docstring)
        owner_file_path = self.required_source_path(snapshot, self.operation_key())
        owner_module = snapshot.parsed_module_for_source_path(owner_file_path).module_name
        _require_abc_compatible_class(snapshot, owner_module, self.class_name)
        if _name_bound_in_module(snapshot, owner_module, self.contract_name):
            raise NeedsInput(
                f"`{self.contract_name}` is already bound in `{owner_module}`",
                NeededInput.OWNER_PICK,
            )
        dest_path, _ = self._destination_file_path(snapshot)
        dest_is_new = snapshot.sources_by_file_path.get(dest_path, "") == ""
        if not dest_is_new and _name_bound_in_module(
            snapshot, self.destination_module, self.contract_name
        ):
            raise NeedsInput(
                f"`{self.contract_name}` is already bound in `{self.destination_module}`",
                NeededInput.OWNER_PICK,
            )
        member_names = _accessed_members_across_readers(snapshot, self.readers, self.class_name)
        after = self._build_state(
            snapshot, owner_file_path, owner_module, dest_path, dest_is_new, member_names
        )
        errors = after.type_errors.new_since(snapshot.type_errors)
        if errors:
            missing_pattern = re.compile(
                rf'"{re.escape(self.contract_name)}" has no attribute "([A-Za-z_][A-Za-z0-9_]*)"'
            )
            missing_attrs = tuple(
                dict.fromkeys(
                    match.group(1)
                    for err in errors
                    if (match := missing_pattern.search(err)) is not None
                )
            )
            extra = tuple(attr for attr in missing_attrs if attr not in member_names)
            if extra and all(missing_pattern.search(err) is not None for err in errors):
                try:
                    after = self._build_state(
                        snapshot,
                        owner_file_path,
                        owner_module,
                        dest_path,
                        dest_is_new,
                        tuple(sorted({*member_names, *extra})),
                    )
                    errors = after.type_errors.new_since(snapshot.type_errors)
                except NeedsInput:
                    pass
        if errors:
            raise RetypeReachesConsumer(self.described, errors)

        after_graph = ModuleImportGraph(after.module_import_graph.import_edges_by_module)
        reader_modules = tuple(
            snapshot.parsed_module_for_source_path(path).module_name for path in self.readers
        )
        if after_graph.reaches(self.destination_module, owner_module) or any(
            after_graph.reaches(self.destination_module, reader_mod)
            for reader_mod in reader_modules
            if reader_mod != self.destination_module
        ):
            raise NeedsInput(
                f"declaring `{self.contract_name}` in `{self.destination_module}` closes an import cycle",
                NeededInput.SOURCE_CHANGE,
            )

        rationale = self.rationale_text(
            f"Declare nominal contract base `{self.contract_name}` in `{self.destination_module}` "
            f"for `{self.class_name}`: {self.described}."
        )
        edits: list[NominalSourceEdit] = []
        for path, after_source in after.sources_by_file_path.items():
            before_source = snapshot.sources_by_file_path.get(path)
            if before_source is None or before_source == after_source:
                continue
            edits.extend(_line_difference_edits(path, before_source, after_source, rationale))
        return tuple(edits)
