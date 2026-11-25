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
import re
import typing
from collections.abc import Iterator
from dataclasses import dataclass

from nominal_refactor_advisor.ast_field_decoders import declared_annotations
from nominal_refactor_advisor.codemod_import_scopes import ModuleImportScope
from nominal_refactor_advisor.class_index import CompactIndexedClass
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
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
from nominal_refactor_advisor.codemod_source_edits import NominalSourceEdit
from nominal_refactor_advisor.hidden_imports import ClosesCycle, GuardedImport
from nominal_refactor_advisor.lexical_bindings import ImportFromModuleName
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.owner_candidates import ModuleImportGraph
from nominal_refactor_advisor.refactors.class_overlap import (
    _line_difference_edits,
    in_order_states,
)
from nominal_refactor_advisor.refactors.module_move import ImportedNameOwner, PromoteGuardedImportOperation


class TypeOnlyReadRefused(NeedsInput):
    """A guarded import every one of whose names the module reads at most as a
    type, which no base below it can serve: each subclass names why."""


class NoLowerBase(TypeOnlyReadRefused):
    """The module reads the class only as a type, but no base of it sits below
    the module: a contract declared below, which the class inherits, is the input."""

    def __init__(self, reader: str, name: str) -> None:
        super().__init__(
            f"`{reader}` reads `{name}` only as a type, but no base of `{name}` sits below it: "
            "a contract base declared below it, which the class inherits, is the input",
            NeededInput.OPERATION,
        )


class BaseIsImplementationMixin(TypeOnlyReadRefused):
    """The bases below the module supply code but declare no contract: naming
    one would say the value is a mixin, erasing which type it is, and a type
    check cannot see that loss when the module reads no member."""

    def __init__(self, reader: str, name: str, bases: tuple[str, ...]) -> None:
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
            base = cls.lower_base(snapshot, imports, owner, name, reader)
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
        snapshot: CodemodSourceSnapshot, imports: ModuleImportGraph, hub: str, name: str, reader: str
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
            raise BaseIsImplementationMixin(reader, name, tuple(mixins))
        raise NoLowerBase(reader, name)


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
                    if argument.annotation is not None:
                        yield argument.annotation
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
