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
from collections.abc import Iterator
from dataclasses import dataclass

from nominal_refactor_advisor.codemod_import_scopes import ModuleImportScope
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
from nominal_refactor_advisor.refactors.class_overlap import changed_source_edits, in_order_states
from nominal_refactor_advisor.refactors.module_move import PromoteGuardedImportOperation


class NoLowerBase(NeedsInput):
    """The module reads the class only as a type, but no base of it sits below
    the module: a contract declared below, which the class inherits, is the input."""

    def __init__(self, reader: str, name: str) -> None:
        super().__init__(
            f"`{reader}` reads `{name}` only as a type, but no base of `{name}` sits below it: "
            "a contract base declared below it, which the class inherits, is the input",
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
        reads = []
        for alias in statement.names:
            if AnnotationReferences.of(module, alias.name) is None:
                return None
            base = cls.lower_base(snapshot, imports, hub, alias.name, hidden.module.module_name)
            if base is None:
                raise NoLowerBase(hidden.module.module_name, alias.name)
            reads.append(cls(hidden.module.module_name, hidden.module.file_path, hub, alias.name, *base))
        return tuple(reads)

    @staticmethod
    def lower_base(
        snapshot: CodemodSourceSnapshot, imports: ModuleImportGraph, hub: str, name: str, reader: str
    ) -> tuple[str, str] | None:
        """The nearest base along the class's MRO declared in another scanned
        module that the reader can import at runtime without a cycle."""

        authority = snapshot.compact_class_index.mro_authority
        symbol = f"{hub}.{name}"
        mro = authority.resolve(symbol).mro_type
        declarations = mro.declarations if mro is not None else authority.lookup_prefix(symbol)
        for declaration in declarations:
            if declaration.symbol == symbol or "." in declaration.qualname:
                continue
            module = declaration.module_name
            if module in (hub, reader) or module not in imports.imports:
                continue
            if imports.reaches(module, reader):
                continue
            return module, declaration.qualname
        return None


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
                if node.returns is not None:
                    yield node.returns
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
        return changed_source_edits(
            snapshot,
            states[-1],
            self.rationale_text(f"Annotate with each class's base below the module: {self.described}."),
        )

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
