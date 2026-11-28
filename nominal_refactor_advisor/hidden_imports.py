"""Imports a module makes where the runtime import graph cannot see them, and
which of them can move to the module's top.

A `TYPE_CHECKING`-guarded import and a function-local import are both edges the
load-time graph leaves out (`SourceModuleImportGraph` reads top-level statements
only). An edge that closes no cycle hides nothing but a fear of one, so it moves
to the top with no input from anyone. The plan accepts edges in one fixed order
and adds each accepted edge to the graph before judging the next, so a batch of
promotions never closes a cycle that no single proof saw. A refused edge keeps
what stopped it: the module that reaches back (`ClosesCycle`), which a cut of
that cycle starts from, or the binding that makes moving it unsound.
"""

from __future__ import annotations

import ast
import builtins
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    ParsedModuleFact,
    ParsedModuleSet,
    ParsedModuleSetFact,
)
from nominal_refactor_advisor.codemod_import_graph import SourceModuleImportGraph
from nominal_refactor_advisor.codemod_import_scopes import (
    ModuleImportScope,
    TypeCheckingGuardProjection,
)
from nominal_refactor_advisor.codemod_imports import (
    ModuleImportMutation,
    node_contains_comment,
)
from nominal_refactor_advisor.codemod_source_edits import NominalSourceEdit, SourceSpanEdit
from nominal_refactor_advisor.declared_root import DeclaredRoot, ForbiddenImportEdge
from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.observation_families import ShapeFamily
from nominal_refactor_advisor.scan_cache import ScanCache

ImportStatement = ast.Import | ast.ImportFrom
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
BUILTIN_NAMES = frozenset(vars(builtins))


def raw_import_targets(
    module: ParsedModule,
    statement: ImportStatement,
) -> tuple[str, ...]:
    """Candidate module names an import statement can trigger, before
    intersecting with the scanned module set."""

    if isinstance(statement, ast.Import):
        return tuple(alias.name for alias in statement.names)
    resolved = module.module_path_identity.resolve_import_from_module(
        imported_module=statement.module,
        level=statement.level,
    )
    if resolved is None:
        return ()
    return (
        resolved,
        *(f"{resolved}.{alias.name}" for alias in statement.names if alias.name != "*"),
    )


def top_level_imported_modules(module: ParsedModule) -> tuple[str, ...]:
    """Direct module names imported at module top level (excluding member names
    of `from mod import member`, where only `mod` is guaranteed to be a module)."""

    imported: list[str] = []
    for statement in module.module.body:
        if isinstance(statement, ast.Import):
            imported.extend(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            resolved = module.module_path_identity.resolve_import_from_module(
                imported_module=statement.module,
                level=statement.level,
            )
            if resolved is not None:
                imported.append(resolved)
    return tuple(dict.fromkeys(imported))


def unscanned_same_package_bridge(
    importing_module: str,
    targets: frozenset[str],
    known: frozenset[str],
    reachable_from: Callable[[str], frozenset[str]],
    unscanned_by_module: Mapping[str, tuple[str, ...]],
) -> tuple[str, str] | None:
    """Return `(loader, unscanned)` if promoting `targets` into `importing_module`
    loads a scanned module that imports an unscanned module in the same top-level
    package, making acyclicity unprovable without scanning that bridge."""

    if "." not in importing_module:
        return None
    importing_top = importing_module.partition(".")[0]
    for target in sorted(targets):
        triggered = known_import_targets((target,), known) - {importing_module}
        loaded: set[str] = set(triggered)
        for started in triggered:
            loaded.update(reachable_from(started))
        loaded.discard(importing_module)
        for loader in sorted(loaded):
            for unscanned in unscanned_by_module.get(loader, ()):
                if unscanned.partition(".")[0] == importing_top:
                    return loader, unscanned
    return None


def known_import_targets(
    candidate_targets: Iterable[str],
    known: frozenset[str],
) -> frozenset[str]:
    """Every scanned module on the package path to any target in `candidate_targets`."""

    targets: set[str] = set()
    for target in candidate_targets:
        parts = target.split(".")
        for depth in range(1, len(parts) + 1):
            prefix = ".".join(parts[:depth])
            if prefix in known:
                targets.add(prefix)
    return frozenset(targets)


@dataclass(frozen=True)
class ImportedBinding:
    """One name an import binds, as what it binds: the module and the member
    read from it (none for `import m`), so two spellings of one import agree."""

    bound: str
    module: str
    member: str | None

    @staticmethod
    def of(module: ParsedModule, statement: ImportStatement) -> tuple[ImportedBinding, ...]:
        if isinstance(statement, ast.Import):
            return tuple(
                ImportedBinding(alias.asname or alias.name.partition(".")[0], alias.name, None)
                for alias in statement.names
            )
        source = module.module_path_identity.resolve_import_from_module(
            imported_module=statement.module, level=statement.level
        )
        return tuple(
            ImportedBinding(alias.asname or alias.name, source or "", alias.name) for alias in statement.names
        )


@dataclass(frozen=True)
class RuntimeBindings:
    """What a module's scope binds once it has loaded: each name a plain
    top-level import binds, with what it binds, and every name bound any other
    way. A guard binds nothing at runtime; a builtin, and a name some function
    declares `global`, count as bound some other way."""

    imported: frozenset[ImportedBinding]
    other: frozenset[str]

    @classmethod
    def of(cls, module: ParsedModule) -> RuntimeBindings:
        body = module.module.body
        guards = TypeCheckingGuardProjection.from_module(module.module).guards
        runtime = (
            *(statement for statement in body if statement not in guards),
            *(statement for guard in guards for statement in guard.orelse),
        )
        others, sole = LEXICAL_SCOPE_BINDING_AUTHORITY.import_only_bindings(runtime)
        direct = {name: statement for name, statement in sole.items() if statement in body}
        declared_global = frozenset(
            name for node in module.syntax_index.instances_of(ast.Global) for name in node.names
        )
        return cls(
            frozenset(
                binding
                for statement in {id(statement): statement for statement in direct.values()}.values()
                for binding in ImportedBinding.of(module, statement)
                if binding.bound in direct
            ),
            others | (sole.keys() - direct.keys()) | declared_global | (BUILTIN_NAMES - direct.keys()),
        )

    def with_promoted(self, bindings: tuple[ImportedBinding, ...]) -> RuntimeBindings:
        return RuntimeBindings(self.imported | frozenset(bindings), self.other)

    def binds(self, name: str) -> bool:
        """Whether the loaded module has `name` for another module to import."""

        return name in self.other or any(held.bound == name for held in self.imported)

    def already_bound(self, binding: ImportedBinding) -> bool:
        """Whether the module already binds this name to exactly this; needs a
        source change when it binds the name to anything else."""

        if binding in self.imported:
            return True
        if binding.bound in self.other or any(held.bound == binding.bound for held in self.imported):
            raise NeedsInput(
                f"`{binding.bound}` is already bound at module scope to something else",
                NeededInput.SOURCE_CHANGE,
            )
        return False


@dataclass(frozen=True, eq=False)
class HiddenImport(ABC):
    """One import statement the load-time graph does not see, and the scanned
    modules it would load."""

    module: ParsedModule
    statement: ImportStatement
    targets: frozenset[str]

    @property
    def location(self) -> str:
        return f"{self.module.file_path}:{self.statement.lineno}"

    @property
    def bindings(self) -> tuple[ImportedBinding, ...]:
        return ImportedBinding.of(self.module, self.statement)

    @property
    def symbol(self) -> str:
        """The dotted class and function names enclosing the import; empty at
        module level."""

        return ".".join(
            node.name
            for node in reversed(self.module.syntax_index.parent_index.ancestors(self.statement))
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )

    @property
    def identity(self) -> str:
        """The import as a move leaves it: its file, the symbol it sits in and
        the names it binds. A move shifts its line and may change the module it
        imports from; none of these."""

        return f"{self.module.file_path}::{self.symbol}::{','.join(sorted(b.bound for b in self.bindings))}"

    @abstractmethod
    def require_static_promotable(self) -> None:
        """Raises, naming the input it needs, when local syntax or enclosing
        bindings make moving this import unsound regardless of module scope."""

        raise NotImplementedError

    def require_no_module_setup_before_import(self) -> None:
        """Refuse promotion when module top-level code (including `if __name__ == "__main__":`)
        mutates `sys.path` or `os.environ` before this import: promoting to the top import
        block would execute the import before that script setup."""

        setup = self.module.fact(_ModuleTopLevelSetupChanges).first_before(
            self.statement.lineno
        )
        if setup is not None:
            changed_line, kind = setup
            raise NeedsInput(
                f"module setup changes `{kind}` at line {changed_line} before this import; "
                "at the module's top it would run before that setup",
                NeededInput.SOURCE_CHANGE,
            )

    def require_promotable(self, scope: RuntimeBindings) -> None:
        """Raises, naming the input it needs, when moving this import to the
        module's top would change what a name means."""

        self.require_static_promotable()
        for binding in self.bindings:
            scope.already_bound(binding)

    @abstractmethod
    def source_edits(
        self, scope: RuntimeBindings, moved: frozenset[ast.stmt], rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        """The edits that move it, given the module's scope before it moves and
        every statement of the file the same plan moves."""

        raise NotImplementedError

    @staticmethod
    def collected_candidates(
        module: ParsedModule,
    ) -> tuple[tuple[HiddenImport, tuple[str, ...]], ...]:
        """Every guarded or function-local import in `module` paired with its
        unresolved candidate target module names: guarded first, then local,
        each in line order."""

        # The module's own traversal answers which nodes are imports: walking the tree
        # again here cost every module of every projected state its whole syntax tree
        # (Toad, 520 modules: 63 plans of one concept's search, 2026-11-20).
        syntax = module.syntax_index
        parents = syntax.parent_index
        guarded = (
            (
                GuardedImport(module, statement, frozenset(), guard),
                raw_import_targets(module, statement),
            )
            for guard in TypeCheckingGuardProjection.from_module(module.module).guards
            for statement in guard.body
            if isinstance(statement, ast.Import | ast.ImportFrom)
        )
        local = (
            (
                LocalImport(module, node, frozenset(), function),
                raw_import_targets(module, node),
            )
            for node in syntax.instances_of(ImportStatement)
            if (function := parents.enclosing_function(node)) is not None
        )
        return (
            *sorted(guarded, key=lambda pair: pair[0].statement.lineno),
            *sorted(local, key=lambda pair: pair[0].statement.lineno),
        )

    @staticmethod
    def collected(module: ParsedModule, graph: SourceModuleImportGraph) -> tuple[HiddenImport, ...]:
        """The module's hidden imports of scanned modules: guarded ones first,
        then local ones, each in line order."""

        if graph.source_file_for_path(module.file_path) is None:
            return ()
        known = graph.known_module_names
        return tuple(
            replace(hidden, targets=targets)
            for hidden, candidate_targets in HiddenImport.collected_candidates(module)
            if (targets := known_import_targets(candidate_targets, known) - {module.module_name})
        )


@dataclass(frozen=True, eq=False)
class GuardedImport(HiddenImport):
    """An import under `if TYPE_CHECKING:`; promoting it moves it out of the guard."""

    guard: ast.If

    def require_static_promotable(self) -> None:
        self.require_no_module_setup_before_import()
        if not isinstance(self.statement, ast.ImportFrom) or any(alias.asname for alias in self.statement.names):
            raise NeedsInput(
                "only an unaliased guarded `from ... import` moves out of its guard",
                NeededInput.OPERATION,
            )
        if self.guard.orelse:
            raise NeedsInput(
                "the guard's `else` binds names at runtime",
                NeededInput.SOURCE_CHANGE,
            )
        if node_contains_comment(self.module.source, self.guard):
            raise NeedsInput(
                "the `if TYPE_CHECKING:` guard carries a comment; remove or move the comment "
                "before promoting so the guard is not left behind as `pass`",
                NeededInput.SOURCE_CHANGE,
            )

    def source_edits(
        self, scope: RuntimeBindings, moved: frozenset[ast.stmt], rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        statement = self.statement
        assert isinstance(statement, ast.ImportFrom)
        return GuardedImport.promotion_edits(
            self.module.file_path,
            "." * statement.level + (statement.module or ""),
            tuple(alias.name for alias in statement.names),
            rationale,
        )

    @staticmethod
    def promotion_edits(
        file_path: str, module_name: str, names: tuple[str, ...], rationale: str
    ) -> tuple[ModuleImportMutation, ...]:
        """Remove the names from the guard and import them at runtime."""

        return (
            ModuleImportMutation.remove_names(
                file_path=file_path,
                module_name=module_name,
                names=names,
                scope=ModuleImportScope.TYPE_CHECKING,
                rationale=rationale,
            ),
            ModuleImportMutation.from_source(
                file_path=file_path,
                import_source=f"from {module_name} import {', '.join(names)}\n",
                rationale=rationale,
            ),
        )


@dataclass(frozen=True, eq=False)
class LocalImport(HiddenImport):
    """An import inside a function body; promoting it deletes it there and
    imports it once at the module's top."""

    function: FunctionNode

    @property
    def ancestors(self) -> tuple[ast.AST, ...]:
        return self.module.syntax_index.parent_index.ancestors(self.statement)

    @property
    def block(self) -> list[ast.stmt]:
        """The statement list that holds it."""

        parent = self.ancestors[0]
        return next(
            block
            for field in ("body", "orelse", "finalbody")
            if isinstance(block := getattr(parent, field, None), list) and self.statement in block
        )

    def require_static_promotable(self) -> None:
        self.require_no_module_setup_before_import()
        ancestors = self.ancestors
        within = ancestors[: ancestors.index(self.function)]
        if any(isinstance(ancestor, ast.ClassDef) for ancestor in within):
            raise NeedsInput(
                "an import in a class body binds a class attribute",
                NeededInput.OPERATION,
            )
        if any(isinstance(ancestor, ast.Try | ast.TryStar) for ancestor in within):
            raise NeedsInput(
                "an import inside `try` may be handling its own failure",
                NeededInput.PROOF,
            )
        first, last = self.statement.lineno, self.statement.end_lineno or self.statement.lineno
        if any(
            other is not self.statement and other.lineno <= last and (other.end_lineno or other.lineno) >= first
            for other in self.block
        ):
            raise NeedsInput("another statement shares its lines", NeededInput.OPERATION)
        changed = next(
            (
                (node.lineno, kind)
                for node in ast.walk(self.function)
                if isinstance(node, ast.stmt | ast.expr)
                and node.lineno < first
                and (kind := import_runtime_setup_change(node)) is not None
            ),
            None,
        )
        if changed is not None:
            changed_line, kind = changed
            raise NeedsInput(
                f"it runs after `{self.function.name}` changes `{kind}` at line {changed_line}; at the "
                "module's top it would run before, so the setup moves to load time first",
                NeededInput.SOURCE_CHANGE,
            )
        others, imported = LEXICAL_SCOPE_BINDING_AUTHORITY.import_only_bindings(self.function.body)
        here = others | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(self.function)
        enclosing = frozenset(
            name
            for outer in ancestors
            if isinstance(outer, FunctionNode) and outer is not self.function
            for name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(outer.body)
            | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(outer)
        )
        for binding in self.bindings:
            if binding.bound in here or imported.get(binding.bound) is not self.statement:
                raise NeedsInput(
                    f"`{binding.bound}` is bound in `{self.function.name}` by more than this import",
                    NeededInput.SOURCE_CHANGE,
                )
            if binding.bound in enclosing:
                raise NeedsInput(
                    f"an enclosing function binds `{binding.bound}`, which would shadow the module's",
                    NeededInput.SOURCE_CHANGE,
                )

    def source_edits(
        self, scope: RuntimeBindings, moved: frozenset[ast.stmt], rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        block = self.block
        emptied = block[0] is self.statement and all(statement in moved for statement in block)
        bound = all(scope.already_bound(binding) for binding in self.bindings)
        return (
            SourceSpanEdit.from_replacement_lines(
                file_path=self.module.file_path,
                start_line=self.statement.lineno,
                end_line=self.statement.end_lineno or self.statement.lineno,
                replacement_lines=(" " * self.statement.col_offset + "pass\n",) if emptied else (),
                rationale=rationale,
            ),
            *(
                ()
                if bound
                else (
                    ModuleImportMutation.from_source(
                        file_path=self.module.file_path,
                        import_source=ast.unparse(self.statement) + "\n",
                        rationale=rationale,
                    ),
                )
            ),
        )


SYS_PATH_CHANGES = frozenset({"insert", "append", "extend", "remove", "pop", "clear"})
OS_ENVIRON_CHANGES = frozenset({"setdefault", "update", "pop", "popitem", "clear"})
OS_ENV_CALLS = frozenset({"putenv", "unsetenv"})


def _is_sys_path(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Attribute)
        and expr.attr == "path"
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "sys"
    )


def _is_os_environ(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Attribute)
        and expr.attr == "environ"
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "os"
    )


def import_runtime_setup_change(node: ast.AST) -> str | None:
    """Return ``'sys.path'`` or ``'os.environ'`` when ``node`` mutates import-path
    or environment state that a later import can depend on."""

    match node:
        case ast.Call(func=ast.Attribute(value=target, attr=method)):
            if method in SYS_PATH_CHANGES and _is_sys_path(target):
                return "sys.path"
            if method in OS_ENVIRON_CHANGES and _is_os_environ(target):
                return "os.environ"
            if method in OS_ENV_CALLS and isinstance(target, ast.Name) and target.id == "os":
                return "os.environ"
        case ast.Assign(targets=targets):
            for target in targets:
                if _is_sys_path(target) or isinstance(target, ast.Subscript) and _is_sys_path(target.value):
                    return "sys.path"
                if isinstance(target, ast.Subscript) and _is_os_environ(target.value):
                    return "os.environ"
        case ast.AugAssign(target=target):
            if _is_sys_path(target):
                return "sys.path"
        case ast.Delete(targets=targets):
            for target in targets:
                if isinstance(target, ast.Subscript) and _is_sys_path(target.value):
                    return "sys.path"
                if isinstance(target, ast.Subscript) and _is_os_environ(target.value):
                    return "os.environ"
    return None


def changes_import_path(node: ast.AST) -> bool:
    """`sys.path.insert(...)` and its kin, or an assignment to `sys.path` or one
    of its items: what an import after it may need in order to resolve."""

    return import_runtime_setup_change(node) == "sys.path"


@dataclass(frozen=True)
class _ModuleTopLevelSetupChanges(ParsedModuleFact):
    """Module top-level `sys.path` / `os.environ` mutations (including inside
    `if __name__ == "__main__":` and other module-level control blocks), sorted
    by line number."""

    changes: tuple[tuple[int, str], ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> _ModuleTopLevelSetupChanges:
        found: list[tuple[int, str]] = []

        def walk(nodes: Iterable[ast.AST]) -> None:
            for node in nodes:
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    continue
                if isinstance(node, ast.stmt | ast.expr):
                    kind = import_runtime_setup_change(node)
                    if kind is not None:
                        found.append((node.lineno, kind))
                walk(ast.iter_child_nodes(node))

        walk(parsed_module.module.body)
        found.sort()
        return cls(tuple(found))

    def first_before(self, line: int) -> tuple[int, str] | None:
        for changed_line, kind in self.changes:
            if changed_line < line:
                return (changed_line, kind)
        return None


def _forbidden_edges_for_paths(
    file_paths: Iterable[str],
) -> tuple[ForbiddenImportEdge, ...]:
    """Distinct declared forbidden import edges for the repositories containing ``file_paths``."""

    seen_dirs: set[Path] = set()
    seen_pyprojects: set[Path] = set()
    edges: list[ForbiddenImportEdge] = []
    seen_edges: set[ForbiddenImportEdge] = set()
    for raw_path in file_paths:
        parent = Path(raw_path).parent
        if parent in seen_dirs:
            continue
        seen_dirs.add(parent)
        pyproject = DeclaredRoot.nearest_pyproject(parent)
        if pyproject is None or pyproject in seen_pyprojects:
            continue
        seen_pyprojects.add(pyproject)
        for edge in DeclaredRoot.forbidden_imports_for(pyproject):
            if edge not in seen_edges:
                seen_edges.add(edge)
                edges.append(edge)
    return tuple(edges)


def _forbidden_edge_violation(
    importing_module: str,
    targets: frozenset[str],
    bindings: tuple[ImportedBinding, ...],
    known: frozenset[str],
    reachable_from: Callable[[str], frozenset[str]],
    top_level_by_module: Mapping[str, tuple[str, ...]],
    edges: tuple[ForbiddenImportEdge, ...],
) -> tuple[str, str, ForbiddenImportEdge] | None:
    """Return ``(importer, loaded, edge)`` if promoting ``targets`` into
    ``importing_module`` would make ``importer`` (``importing_module`` or a
    module that already loads it) transitively load ``loaded`` across ``edge``."""

    if not edges:
        return None
    affected_importers = {importing_module}
    for candidate in known:
        if candidate != importing_module and importing_module in reachable_from(candidate):
            affected_importers.add(candidate)
    active_edges = tuple(
        (importer, edge)
        for importer in sorted(affected_importers)
        for edge in edges
        if edge.applies_to(importer)
    )
    if not active_edges:
        return None
    triggered = known_import_targets(targets, known) - {importing_module}
    loaded_scanned: set[str] = set(triggered)
    for started in triggered:
        loaded_scanned.update(reachable_from(started))
    loaded_all: set[str] = set(loaded_scanned)
    for binding in bindings:
        if binding.module:
            loaded_all.add(binding.module)
    for loaded_mod in loaded_scanned:
        loaded_all.update(top_level_by_module.get(loaded_mod, ()))
    ordered_loaded = sorted(loaded_all)
    for importer, edge in active_edges:
        for loaded_mod in ordered_loaded:
            if edge.forbids(loaded_mod):
                return importer, loaded_mod, edge
    return None


class HiddenImportRefusal(ABC):
    """Why one hidden import stays where it is, and the input moving it needs."""

    hidden: HiddenImport

    @property
    @abstractmethod
    def need(self) -> NeedsInput:
        raise NotImplementedError


@dataclass(frozen=True)
class ClosesCycle(HiddenImportRefusal):
    """Loading the import would run `reaching`, which already loads the importer
    at load time: the cycle a cut starts from."""

    hidden: HiddenImport
    reaching: str

    @property
    def need(self) -> NeedsInput:
        return NeedsInput(
            f"{self.hidden.location}: `{self.reaching}` already imports `{self.hidden.module.module_name}` "
            "at load; the import hides a cycle, so a declaration has to move first",
            NeededInput.SOURCE_CHANGE,
        )


@dataclass(frozen=True)
class UnscannedImportBridge(HiddenImportRefusal):
    """Promoting the import runs a scanned module that imports an unscanned
    module in the same package, which could import the target module back."""

    hidden: HiddenImport
    loader: str
    unscanned: str

    @property
    def need(self) -> NeedsInput:
        return NeedsInput(
            f"{self.hidden.location}: `{self.loader}` imports unscanned `{self.unscanned}` "
            f"at load; scan the enclosing package (`--context-root`) to prove it does not "
            f"reach `{self.hidden.module.module_name}`",
            NeededInput.PROOF,
        )


@dataclass(frozen=True)
class CrossesForbiddenImportEdge(HiddenImportRefusal):
    """Promoting the import crosses a repository-declared architectural layer
    boundary in `[tool.nominal-refactor-advisor] forbidden-imports`."""

    hidden: HiddenImport
    importer: str
    loaded: str
    edge: ForbiddenImportEdge

    @property
    def need(self) -> NeedsInput:
        return NeedsInput(
            f"{self.hidden.location}: promoting this import makes `{self.importer}` load "
            f"`{self.loaded}`, violating forbidden import edge "
            f"`{self.edge.source_prefix}` -> `{self.edge.target_prefix}`",
            NeededInput.SOURCE_CHANGE,
        )


@dataclass(frozen=True)
class UnpromotableImport(HiddenImportRefusal):
    """Moving the import would change what a name means, or no operation moves it."""

    hidden: HiddenImport
    reason: NeedsInput

    @property
    def need(self) -> NeedsInput:
        return NeedsInput(f"{self.hidden.location}: {self.reason}", *self.reason.needs)



@dataclass(eq=False)
class ScannedExports:
    """What each scanned module has for another to import: the names its loaded scope
    binds, and its submodules.

    A guarded import never runs, and a local one only when its function does, so either
    can name a member its module no longer has (a class that moved away) for as long as
    nothing runs it. At the module's top it runs at load time and the module stops
    importing (Toad, 2026-11-20: `from toad.forms.presentation import EntryPlacement`,
    which `toad.entry_placement` defines; the promotion had reported its goal achieved).
    """

    modules: Mapping[str, ParsedModule]
    scopes: dict[str, RuntimeBindings] = field(default_factory=dict)

    @classmethod
    def of(cls, modules: ParsedModuleSet) -> ScannedExports:
        return cls({module.module_name: module for module in modules})

    def has(self, module_name: str, member: str) -> bool:
        """Whether `module_name` has `member`; a module outside the scan is not judged."""

        module = self.modules.get(module_name)
        if module is None or f"{module_name}.{member}" in self.modules:
            return True
        if module_name not in self.scopes:
            self.scopes[module_name] = RuntimeBindings.of(module)
        return self.scopes[module_name].binds(member)

    def require_bound(self, bindings: tuple[ImportedBinding, ...]) -> None:
        missing = [binding for binding in bindings
                   if binding.member is not None and not self.has(binding.module, binding.member)]
        if missing:
            raise NeedsInput(
                "; ".join(f"`{binding.module}` does not bind `{binding.member}`" for binding in missing)
                + ": the import names something its module does not have",
                NeededInput.SOURCE_CHANGE,
            )


@dataclass(frozen=True)
class HiddenImportPlan(ParsedModuleSetFact):
    """Every hidden import of one module set, accepted or refused. Modules go in
    path order, and within one, guarded imports before local ones (a local
    import of a name its guard imports then needs no top-level import of its
    own), each in line order; each is judged against the graph holding every
    import accepted before it."""

    accepted: tuple[HiddenImport, ...]
    refused: tuple[HiddenImportRefusal, ...]
    scopes: Mapping[str, RuntimeBindings]
    """Each planned module's scope before any import moves."""

    @classmethod
    def derive(cls, modules: ParsedModuleSet) -> HiddenImportPlan:
        graph = SourceModuleImportGraph.of_modules(modules)
        known = graph.known_module_names
        top_level_by_module = {
            module.module_name: top_level_imported_modules(module)
            for module in modules
        }
        unscanned_by_module = {
            module_name: tuple(
                sorted(
                    imported
                    for imported in imported_modules
                    if imported not in known
                )
            )
            for module_name, imported_modules in top_level_by_module.items()
        }
        forbidden_edges = _forbidden_edges_for_paths(
            module.file_path for module in modules
        )
        accepted: list[HiddenImport] = []
        refused: list[HiddenImportRefusal] = []
        scopes: dict[str, RuntimeBindings] = {}
        exports = ScannedExports.of(modules)
        for module in sorted(modules, key=lambda parsed: parsed.file_path):
            found = HiddenImport.collected(module, graph)
            if not found:
                continue
            scope = scopes[module.file_path] = RuntimeBindings.of(module)
            for hidden in found:
                try:
                    exports.require_bound(hidden.bindings)
                    hidden.require_promotable(scope)
                except NeedsInput as need:
                    refused.append(UnpromotableImport(hidden, need))
                    continue
                reaching = next(
                    (
                        back
                        for target in sorted(hidden.targets)
                        if (back := graph.reaching_back(module.module_name, target)) is not None
                    ),
                    None,
                )
                if reaching is not None:
                    refused.append(ClosesCycle(hidden, reaching))
                    continue
                bridge = unscanned_same_package_bridge(
                    module.module_name,
                    hidden.targets,
                    known,
                    graph.reachable_modules,
                    unscanned_by_module,
                )
                if bridge is not None:
                    refused.append(UnscannedImportBridge(hidden, *bridge))
                    continue
                violation = _forbidden_edge_violation(
                    module.module_name,
                    hidden.targets,
                    hidden.bindings,
                    known,
                    graph.reachable_modules,
                    top_level_by_module,
                    forbidden_edges,
                )
                if violation is not None:
                    refused.append(CrossesForbiddenImportEdge(hidden, *violation))
                    continue
                accepted.append(hidden)
                graph = graph.with_edges(module.module_name, hidden.targets)
                scope = scope.with_promoted(hidden.bindings)
        return cls(tuple(accepted), tuple(refused), scopes)

    def accepted_in(self, file_path: str) -> tuple[HiddenImport, ...]:
        return tuple(hidden for hidden in self.accepted if hidden.module.file_path == file_path)

    def refused_in(self, file_path: str) -> tuple[HiddenImportRefusal, ...]:
        return tuple(refusal for refusal in self.refused if refusal.hidden.module.file_path == file_path)

    def freed_by(self, before: HiddenImportPlan) -> tuple[HiddenImport, ...]:
        """The imports this plan accepts that `before`, the plan of the source a
        change started from, refused: what the change freed."""

        refused = frozenset(refusal.hidden.identity for refusal in before.refused)
        return tuple(hidden for hidden in self.accepted if hidden.identity in refused)

    def edits_for(
        self, file_path: str, rationale: str, only: frozenset[str] | None = None
    ) -> tuple[NominalSourceEdit, ...]:
        """Every accepted import of one file, or those of them `only` names by
        identity, moved to its top, each seeing the scope with the imports
        accepted before it bound; when none is, what each refused one needs."""

        accepted = tuple(
            hidden for hidden in self.accepted_in(file_path) if only is None or hidden.identity in only
        )
        if not accepted:
            needs = tuple(refusal.need for refusal in self.refused_in(file_path))
            raise NeedsInput(
                "; ".join(map(str, needs)) or f"{file_path} hides no import of a scanned module",
                *(kind for need in needs for kind in need.needs),
            )
        moved = frozenset(hidden.statement for hidden in accepted)
        scope = self.scopes[file_path]
        edits: list[NominalSourceEdit] = []
        for hidden in accepted:
            edits.extend(hidden.source_edits(scope, moved, rationale))
            scope = scope.with_promoted(hidden.bindings)
        return tuple(edits)


@dataclass(frozen=True)
class CompactHiddenImport:
    """AST-free summary of one guarded or function-local import."""

    lineno: int
    is_import_from: bool
    bindings: tuple[ImportedBinding, ...]
    candidate_targets: tuple[str, ...]
    static_unpromotable: bool


@dataclass(frozen=True)
class HiddenImportModuleFacts:
    """AST-free per-module facts for hidden-import and cycle detection."""

    file_path: str
    module_name: str
    top_level_import_targets: tuple[str, ...]
    imported_bindings: tuple[ImportedBinding, ...]
    other_bindings: tuple[str, ...]
    hidden: tuple[CompactHiddenImport, ...]
    top_level_imported_modules: tuple[str, ...] = ()

    def runtime_bindings(self) -> RuntimeBindings:
        return RuntimeBindings(
            frozenset(self.imported_bindings),
            frozenset(self.other_bindings),
        )


class HiddenImportModuleFamily(ShapeFamily[HiddenImportModuleFacts]):
    """Compact per-module facts shared by hidden-import and cycle detectors."""

    item_type = HiddenImportModuleFacts

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[HiddenImportModuleFacts]:
        top_level_import_targets = tuple(
            target
            for statement in parsed_module.module.body
            if isinstance(statement, ast.Import | ast.ImportFrom)
            for target in raw_import_targets(parsed_module, statement)
        )
        scope = RuntimeBindings.of(parsed_module)
        compact_hidden: list[CompactHiddenImport] = []
        for hidden, candidate_targets in HiddenImport.collected_candidates(parsed_module):
            try:
                hidden.require_static_promotable()
            except NeedsInput:
                static_unpromotable = True
            else:
                static_unpromotable = False
            compact_hidden.append(
                CompactHiddenImport(
                    lineno=hidden.statement.lineno,
                    is_import_from=isinstance(hidden.statement, ast.ImportFrom),
                    bindings=hidden.bindings,
                    candidate_targets=candidate_targets,
                    static_unpromotable=static_unpromotable,
                )
            )
        return [
            HiddenImportModuleFacts(
                file_path=parsed_module.file_path,
                module_name=parsed_module.module_name,
                top_level_import_targets=top_level_import_targets,
                imported_bindings=tuple(
                    sorted(
                        scope.imported,
                        key=lambda binding: (binding.bound, binding.module, binding.member or ""),
                    )
                ),
                other_bindings=tuple(sorted(scope.other)),
                hidden=tuple(compact_hidden),
                top_level_imported_modules=top_level_imported_modules(parsed_module),
            )
        ]


@dataclass(frozen=True)
class CompactHiddenImportModuleSummary:
    """One module's judged hidden imports for `HiddenImportDetector`."""

    file_path: str
    module_name: str
    first_accepted_line: int
    accepted_count: int
    cycle_reaching: tuple[str, ...]
    unpromotable_count: int


@dataclass(frozen=True)
class _CompactHiddenImportBaseline:
    """Reusable full-scan state for incremental ``CompactHiddenImportPlan``."""

    by_path: dict[str, HiddenImportModuleFacts]
    known: frozenset[str]
    sorted_mods: tuple[str, ...]
    mod_to_idx: dict[str, int]
    paths_by_module: dict[str, str]
    initial_edges: dict[str, frozenset[str]]
    initial_adj_mask: tuple[int, ...]
    exports_by_module: dict[str, RuntimeBindings]
    unscanned_by_module: dict[str, tuple[str, ...]]
    initial_reach_mask: tuple[int | None, ...]
    referenced_export_members: dict[str, frozenset[str]]
    forbidden_edges: tuple[ForbiddenImportEdge, ...]
    plan: CompactHiddenImportPlan


@ScanCache.cached
def _base_hidden_import_plan_slot() -> list[_CompactHiddenImportBaseline | None]:
    return [None]


@dataclass(frozen=True)
class CompactHiddenImportPlan:
    """Shared compact judgment of hidden imports and cycle hubs across modules."""

    module_summaries: tuple[CompactHiddenImportModuleSummary, ...]
    hub_readers: Mapping[str, tuple[str, ...]]
    paths_by_module: Mapping[str, str]

    @classmethod
    def require(cls, context: object | None) -> CompactHiddenImportPlan:
        if not isinstance(context, cls):
            raise TypeError(
                f"Expected CompactHiddenImportPlan shared context, got {type(context).__name__}"
            )
        return context

    @classmethod
    def from_projections(
        cls,
        projections: Sequence[HiddenImportModuleFacts],
        config: object = None,
    ) -> CompactHiddenImportPlan:
        del config
        base: _CompactHiddenImportBaseline | None = None
        if ScanCache.active() is not None:
            base = _base_hidden_import_plan_slot()[0]
        known = frozenset(projection.module_name for projection in projections)
        forbidden_edges = _forbidden_edges_for_paths(
            projection.file_path for projection in projections
        )
        top_level_by_module = {
            projection.module_name: projection.top_level_imported_modules
            for projection in projections
        }
        if (
            base is not None
            and known == base.known
            and len(projections) == len(base.by_path)
            and forbidden_edges == base.forbidden_edges
        ):
            changed = [
                p
                for p in projections
                if (old_p := base.by_path.get(p.file_path)) is None
                or (old_p is not p and old_p != p)
            ]
            if not changed:
                return base.plan
            sorted_mods = base.sorted_mods
            mod_to_idx = base.mod_to_idx
            paths_by_module = base.paths_by_module
            adj_mask = list(base.initial_adj_mask)
            exports_by_module: dict[str, RuntimeBindings] = dict(base.exports_by_module)
            unscanned_by_module: dict[str, tuple[str, ...]] = dict(
                base.unscanned_by_module
            )
            dirty_edge_idxs: set[int] = set()
            dirty_bitmask = 0
            any_structural_change = False
            for p in changed:
                old_p = base.by_path[p.file_path]
                mod = p.module_name
                mod_idx = mod_to_idx[mod]
                new_edges = frozenset(
                    known_import_targets(p.top_level_import_targets, known)
                )
                if new_edges != base.initial_edges.get(mod):
                    mask = 0
                    for target in new_edges:
                        mask |= 1 << mod_to_idx[target]
                    adj_mask[mod_idx] = mask
                    dirty_edge_idxs.add(mod_idx)
                    dirty_bitmask |= 1 << mod_idx
                    any_structural_change = True
                new_unscanned = tuple(
                    sorted(
                        imported
                        for imported in p.top_level_imported_modules
                        if imported not in known
                    )
                )
                if new_unscanned != base.unscanned_by_module.get(mod, ()):
                    unscanned_by_module[mod] = new_unscanned
                    any_structural_change = True
                new_scope = p.runtime_bindings()
                old_scope = base.exports_by_module[mod]
                exports_by_module[mod] = new_scope
                if not any_structural_change:
                    if p.hidden != old_p.hidden:
                        any_structural_change = True
                    elif p.hidden and new_scope != old_scope:
                        any_structural_change = True
                    else:
                        ref_members = base.referenced_export_members.get(mod)
                        if ref_members and any(
                            new_scope.binds(member) != old_scope.binds(member)
                            for member in ref_members
                        ):
                            any_structural_change = True
            if not any_structural_change:
                return base.plan
            if not dirty_edge_idxs:
                reach_mask: list[int | None] = list(base.initial_reach_mask)
            else:
                reach_mask = [
                    m
                    if m is not None
                    and idx not in dirty_edge_idxs
                    and not (m & dirty_bitmask)
                    else None
                    for idx, m in enumerate(base.initial_reach_mask)
                ]
            populated_reach = [
                idx for idx, m in enumerate(reach_mask) if m is not None
            ]
            record_initial = False
            initial_edges: dict[str, frozenset[str]] = {}
            initial_adj_mask: tuple[int, ...] = ()
            initial_reach_mask: list[int | None] = []
        else:
            sorted_mods = tuple(sorted(known))
            mod_to_idx = {mod: idx for idx, mod in enumerate(sorted_mods)}
            n_mods = len(sorted_mods)
            paths_by_module = {
                projection.module_name: projection.file_path
                for projection in projections
            }
            initial_edges = {
                projection.module_name: frozenset(
                    known_import_targets(projection.top_level_import_targets, known)
                )
                for projection in projections
            }
            adj_mask = [0] * n_mods
            for mod, targets in initial_edges.items():
                mask = 0
                for target in targets:
                    mask |= 1 << mod_to_idx[target]
                adj_mask[mod_to_idx[mod]] = mask
            initial_adj_mask = tuple(adj_mask)
            exports_by_module = {
                projection.module_name: projection.runtime_bindings()
                for projection in projections
            }
            unscanned_by_module = {
                projection.module_name: tuple(
                    sorted(
                        imported
                        for imported in projection.top_level_imported_modules
                        if imported not in known
                    )
                )
                for projection in projections
            }
            reach_mask = [None] * n_mods
            populated_reach = []
            record_initial = (
                ScanCache.active() is not None
                and base is None
                and len(projections) >= 20
            )
            initial_reach_mask = [None] * n_mods if record_initial else []

        unscanned_mask_by_top: dict[str, int] = {}
        for mod, unscanned in unscanned_by_module.items():
            if not unscanned:
                continue
            mod_bit = 1 << mod_to_idx[mod]
            for imp in unscanned:
                top = imp.partition(".")[0]
                unscanned_mask_by_top[top] = unscanned_mask_by_top.get(top, 0) | mod_bit

        mutated_mask = 0

        def reachable_mask(start_idx: int) -> int:
            cached = reach_mask[start_idx]
            if cached is not None:
                return cached
            visited = 0
            reached = 0
            stack = [start_idx]
            while stack:
                cur = stack.pop()
                cur_bit = 1 << cur
                if visited & cur_bit:
                    continue
                visited |= cur_bit
                if cur != start_idx:
                    cur_cached = reach_mask[cur]
                    if cur_cached is not None:
                        reached |= cur_cached
                        continue
                nbrs = adj_mask[cur]
                reached |= nbrs
                rem = nbrs & ~visited
                while rem:
                    lsb = rem & -rem
                    stack.append(lsb.bit_length() - 1)
                    rem ^= lsb
            reach_mask[start_idx] = reached
            populated_reach.append(start_idx)
            if record_initial and (
                not mutated_mask
                or (
                    not (mutated_mask & (1 << start_idx))
                    and not (reached & mutated_mask)
                )
            ):
                initial_reach_mask[start_idx] = reached
            return reached

        def reachable_from(start_module: str) -> frozenset[str]:
            m_idx = mod_to_idx.get(start_module)
            if m_idx is None:
                return frozenset()
            mask = reachable_mask(m_idx)
            names: list[str] = []
            rem = mask
            while rem:
                lsb = rem & -rem
                names.append(sorted_mods[lsb.bit_length() - 1])
                rem ^= lsb
            return frozenset(names)

        def reaching_back(importing_module: str, importing_bit: int, imported_module: str) -> str | None:
            if importing_module == imported_module:
                return None
            triggered = known_import_targets((imported_module,), known) - {
                importing_module
            }
            for target in sorted(triggered):
                if reachable_mask(mod_to_idx[target]) & importing_bit:
                    return target
            return None

        def exports_has(module_name: str, member: str) -> bool:
            scope = exports_by_module.get(module_name)
            if scope is None or f"{module_name}.{member}" in known:
                return True
            return scope.binds(member)

        module_summaries: list[CompactHiddenImportModuleSummary] = []
        hub_readers: dict[str, set[str]] = defaultdict(set)
        referenced_exports: dict[str, set[str]] = defaultdict(set)

        for projection in sorted(projections, key=lambda item: item.file_path):
            if not projection.hidden:
                continue
            active = [
                (hidden, targets)
                for hidden in projection.hidden
                if (
                    targets := known_import_targets(hidden.candidate_targets, known)
                    - {projection.module_name}
                )
            ]
            if not active:
                continue
            mod = projection.module_name
            mod_idx = mod_to_idx[mod]
            mod_bit = 1 << mod_idx
            importing_top = mod.partition(".")[0]
            top_unscanned_mask = unscanned_mask_by_top.get(importing_top, 0)
            scope = exports_by_module[mod]
            accepted_lines: list[int] = []
            cycle_reaching: list[str] = []
            unpromotable_count = 0
            for hidden, targets in active:
                if record_initial:
                    for binding in hidden.bindings:
                        if binding.member is not None and binding.module in known:
                            referenced_exports[binding.module].add(binding.member)
                if any(
                    binding.member is not None
                    and not exports_has(binding.module, binding.member)
                    for binding in hidden.bindings
                ):
                    unpromotable_count += 1
                    continue
                if hidden.static_unpromotable:
                    unpromotable_count += 1
                    continue
                try:
                    for binding in hidden.bindings:
                        scope.already_bound(binding)
                except NeedsInput:
                    unpromotable_count += 1
                    continue
                reaching = next(
                    (
                        back
                        for target in sorted(targets)
                        if (back := reaching_back(mod, mod_bit, target))
                        is not None
                    ),
                    None,
                )
                if reaching is not None:
                    cycle_reaching.append(reaching)
                    hub = (
                        hidden.bindings[0].module
                        if hidden.is_import_from and hidden.bindings
                        else None
                    )
                    hub_readers[hub or reaching].add(mod)
                    continue
                if top_unscanned_mask:
                    loaded_mask = 0
                    for target in targets:
                        t_idx = mod_to_idx[target]
                        loaded_mask |= (1 << t_idx) | reachable_mask(t_idx)
                    loaded_mask &= ~mod_bit
                    if (
                        loaded_mask & top_unscanned_mask
                        and unscanned_same_package_bridge(
                            mod,
                            targets,
                            known,
                            reachable_from,
                            unscanned_by_module,
                        )
                        is not None
                    ):
                        unpromotable_count += 1
                        continue
                if forbidden_edges and _forbidden_edge_violation(
                    mod,
                    targets,
                    hidden.bindings,
                    known,
                    reachable_from,
                    top_level_by_module,
                    forbidden_edges,
                ) is not None:
                    unpromotable_count += 1
                    continue
                accepted_lines.append(hidden.lineno)
                target_mask = 0
                for target in targets:
                    target_mask |= 1 << mod_to_idx[target]
                new_bits = target_mask & ~adj_mask[mod_idx]
                if new_bits:
                    added_mask = new_bits
                    rem = new_bits
                    while rem:
                        lsb = rem & -rem
                        added_mask |= reachable_mask(lsb.bit_length() - 1)
                        rem ^= lsb
                    adj_mask[mod_idx] |= new_bits
                    mutated_mask |= mod_bit
                    for k in populated_reach:
                        rk = reach_mask[k]
                        if rk is not None and (k == mod_idx or (rk & mod_bit)):
                            reach_mask[k] = rk | added_mask
                scope = scope.with_promoted(hidden.bindings)
            if accepted_lines:
                module_summaries.append(
                    CompactHiddenImportModuleSummary(
                        file_path=projection.file_path,
                        module_name=mod,
                        first_accepted_line=accepted_lines[0],
                        accepted_count=len(accepted_lines),
                        cycle_reaching=tuple(cycle_reaching),
                        unpromotable_count=unpromotable_count,
                    )
                )

        built = cls(
            module_summaries=tuple(module_summaries),
            hub_readers={
                hub: tuple(sorted(readers)) for hub, readers in hub_readers.items()
            },
            paths_by_module=paths_by_module,
        )
        if record_initial:
            slot = _base_hidden_import_plan_slot()
            if slot[0] is None:
                slot[0] = _CompactHiddenImportBaseline(
                    by_path={p.file_path: p for p in projections},
                    known=known,
                    sorted_mods=sorted_mods,
                    mod_to_idx=mod_to_idx,
                    paths_by_module=paths_by_module,
                    initial_edges=initial_edges,
                    initial_adj_mask=initial_adj_mask,
                    exports_by_module=exports_by_module,
                    unscanned_by_module=unscanned_by_module,
                    initial_reach_mask=tuple(initial_reach_mask),
                    referenced_export_members={
                        mod: frozenset(members)
                        for mod, members in referenced_exports.items()
                    },
                    forbidden_edges=forbidden_edges,
                    plan=built,
                )
        return built


__all__ = (
    "BUILTIN_NAMES",
    "ClosesCycle",
    "CompactHiddenImport",
    "CompactHiddenImportModuleSummary",
    "CompactHiddenImportPlan",
    "CrossesForbiddenImportEdge",
    "GuardedImport",
    "HiddenImport",
    "HiddenImportModuleFacts",
    "HiddenImportModuleFamily",
    "HiddenImportPlan",
    "HiddenImportRefusal",
    "ImportedBinding",
    "LocalImport",
    "RuntimeBindings",
    "UnpromotableImport",
    "UnscannedImportBridge",
    "known_import_targets",
    "raw_import_targets",
    "top_level_imported_modules",
    "unscanned_same_package_bridge",
)
