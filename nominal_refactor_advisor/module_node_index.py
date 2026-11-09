"""One walk of a parsed module, shared by every pass that reads its nodes."""

from __future__ import annotations

import ast
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, field

from .ast_tools import ParsedModule, ParsedModuleFact

_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSION_SCOPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


@dataclass(frozen=True)
class ModuleNodeIndex(ParsedModuleFact):
    """The nodes of one module that consumer passes look up, from a single walk.

    The walk is `ast.walk`'s own breadth-first order, so every list keeps the
    order a pass that walked the module itself would have seen. It is a fact of the
    parse (`parsed.fact(ModuleNodeIndex)`), and a plan's snapshots hand unchanged
    parses from stage to stage, so each unchanged file is walked once per plan instead of once per
    pass, per operation, per stage.

    `bound_name_counts` counts each name's bindings, where every `import pkg.x`
    binds the same package object to `pkg` and so counts once however many.

    A function, lambda or comprehension that binds a name (without `global`)
    has its own variable of that name, shadowing the module's for its own uses
    and those of the scopes inside it. `module_scope_names`,
    `module_scope_bindings` and `module_scope_loads` read only the uses that
    resolve to the module's binding. A class body is not a scope here: its
    bindings count as the module's, which keeps every check that reads them
    conservative."""

    imports: tuple[ast.Import | ast.ImportFrom, ...]
    string_constants: tuple[ast.Constant, ...]
    calls: tuple[ast.Call, ...]
    names_by_id: dict[str, tuple[ast.Name, ...]] = field(repr=False)
    parents: dict[ast.AST, ast.AST] = field(repr=False)
    loads: Counter[str] = field(repr=False)
    attribute_bases: frozenset[str]
    bound_name_counts: Counter[str] = field(repr=False)
    binding_sites: dict[str, tuple[ast.AST, ...]] = field(repr=False)
    declared_global: dict[int, frozenset[str]] = field(repr=False)
    identifier_text: str = field(repr=False)
    _local_names: dict[int, frozenset[str]] = field(repr=False, default_factory=dict)

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> ModuleNodeIndex:
        imports: list[ast.Import | ast.ImportFrom] = []
        strings: list[ast.Constant] = []
        calls: list[ast.Call] = []
        names: dict[str, list[ast.Name]] = {}
        parents: dict[ast.AST, ast.AST] = {}
        loads: Counter[str] = Counter()
        attribute_bases: set[str] = set()
        bound: Counter[str] = Counter()
        package_heads: set[str] = set()
        sites: dict[str, list[ast.AST]] = {}
        declarations: list[ast.Global | ast.Nonlocal] = []
        pending: deque[ast.AST] = deque([parsed_module.module])
        while pending:
            node = pending.popleft()
            for child in ast.iter_child_nodes(node):
                parents[child] = node
                pending.append(child)
            match node:
                case ast.Import(names=aliases):
                    imports.append(node)
                    for alias in aliases:
                        if alias.asname is None:
                            package_heads.add(alias.name.split(".")[0])
                        else:
                            bound[alias.asname] += 1
                        sites.setdefault(alias.asname or alias.name.split(".")[0], []).append(node)
                case ast.ImportFrom(names=aliases):
                    imports.append(node)
                    for alias in aliases:
                        if alias.name != "*":
                            bound[alias.asname or alias.name] += 1
                            sites.setdefault(alias.asname or alias.name, []).append(node)
                case ast.arg(arg=name):
                    bound[name] += 1
                    sites.setdefault(name, []).append(node)
                case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(
                    name=name
                ):
                    bound[name] += 1
                    sites.setdefault(name, []).append(node)
                case ast.ExceptHandler(name=str() as name):
                    sites.setdefault(name, []).append(node)
                case ast.Global() | ast.Nonlocal():
                    declarations.append(node)
                case ast.Constant(value=str()):
                    strings.append(node)
                case ast.Call():
                    calls.append(node)
                case ast.Name(id=name, ctx=context):
                    names.setdefault(name, []).append(node)
                    if isinstance(context, ast.Load):
                        loads[name] += 1
                    else:
                        bound[name] += 1
                        sites.setdefault(name, []).append(node)
                case ast.Attribute(value=ast.Name(id=name)):
                    attribute_bases.add(name)
        bound.update(package_heads)
        declared_global: dict[int, set[str]] = {}
        for declaration in declarations:
            if isinstance(declaration, ast.Global):
                scope = cls._scope_in(parents, declaration)
                declared_global.setdefault(id(scope), set()).update(declaration.names)
        return cls(
            imports=tuple(imports),
            string_constants=tuple(strings),
            calls=tuple(calls),
            names_by_id={name: tuple(nodes) for name, nodes in names.items()},
            parents=parents,
            loads=loads,
            attribute_bases=frozenset(attribute_bases),
            bound_name_counts=bound,
            binding_sites={name: tuple(nodes) for name, nodes in sites.items()},
            declared_global={key: frozenset(names) for key, names in declared_global.items()},
            identifier_text=(
                parsed_module.source
                if parsed_module.source.isascii()
                else unicodedata.normalize("NFKC", parsed_module.source)
            ),
        )

    def import_froms(self) -> tuple[ast.ImportFrom, ...]:
        return tuple(node for node in self.imports if isinstance(node, ast.ImportFrom))

    def names(self, name: str) -> tuple[ast.Name, ...]:
        return self.names_by_id.get(name, ())

    def spells(self, identifier: str) -> bool:
        """Whether the module's text can name `identifier`: every identifier,
        import, comment or one-piece string naming it spells it in the text the
        compiler resolves identifiers against (NFKC, for a non-ASCII source). A
        string the parser assembles from pieces or escapes can hold it without
        spelling it; read `string_constants` for those."""
        return identifier in self.identifier_text

    @property
    def star_imports(self) -> bool:
        return any(alias.name == "*" for node in self.imports for alias in node.names)

    def module_scope_names(self, name: str) -> tuple[ast.Name, ...]:
        """The `name` nodes, loads and stores, that use the module's binding."""

        return self.uses_of(name, None)

    def module_scope_loads(self, name: str) -> int:
        return sum(isinstance(node.ctx, ast.Load) for node in self.uses_of(name, None))

    def module_scope_bindings(self, name: str) -> int:
        return self.bindings_in(name, None)

    def uses_of(self, name: str, scope: ast.AST | None) -> tuple[ast.Name, ...]:
        """The `name` nodes that use the variable `scope` (`None`: the module)
        holds: the innermost scope binding the name, from each use outward."""

        return tuple(node for node in self.names(name) if self.binder_of(node, name) is scope)

    def bindings_in(self, name: str, scope: ast.AST | None) -> int:
        """How many statements bind `name` in `scope`'s own variable; every
        `import pkg.x` there binds `pkg` once however many."""

        sites = [
            site
            for site in self.binding_sites.get(name, ())
            if self.binding_scope(site, name) is scope
        ]
        heads = [
            site
            for site in sites
            if isinstance(site, ast.Import)
            and any(alias.asname is None and alias.name.split(".")[0] == name for alias in site.names)
        ]
        return len(sites) - len(heads) + bool(heads)

    def binding_scope(self, site: ast.AST, name: str) -> ast.AST | None:
        """The scope whose variable a binding of `name` at `site` sets."""

        scope = self.scope_of(site)
        if scope is not None and name in self.declared_global.get(id(scope), frozenset()):
            return None
        return scope

    def binder_of(self, node: ast.AST, name: str) -> ast.AST | None:
        """The scope whose variable `name` at `node` reads or sets."""

        scope = self.scope_of(node)
        while scope is not None:
            if name in self.declared_global.get(id(scope), frozenset()):
                return None
            if name in self.local_names(scope):
                return scope
            scope = self.scope_of(scope)
        return None

    def local_names(self, scope: ast.AST) -> frozenset[str]:
        """The names `scope` binds itself, minus those it declares `global`."""

        key = id(scope)
        if key not in self._local_names:
            for scope_key, names in self._scope_bindings().items():
                self._local_names[scope_key] = names
        return self._local_names.get(key, frozenset())

    def _scope_bindings(self) -> dict[int, frozenset[str]]:
        found: dict[int, set[str]] = {}
        for name, sites in self.binding_sites.items():
            for site in sites:
                scope = self.scope_of(site)
                if scope is not None and name not in self.declared_global.get(id(scope), frozenset()):
                    found.setdefault(id(scope), set()).add(name)
        for scope_key in {id(scope) for scope in self.parents.values() if self._is_scope(scope)}:
            found.setdefault(scope_key, set())
        return {key: frozenset(names) for key, names in found.items()}

    @staticmethod
    def _is_scope(node: ast.AST) -> bool:
        return isinstance(node, (*_FUNCTION_SCOPES, *_COMPREHENSION_SCOPES))

    def scope_of(self, node: ast.AST) -> ast.AST | None:
        """The innermost function, lambda or comprehension whose variables
        `node` belongs to, or `None` for the module's. A function's decorators,
        defaults and annotations, and a comprehension's first iterable, are
        evaluated in the enclosing scope; its parameters belong to it."""

        return self._scope_in(self.parents, node)

    @staticmethod
    def _scope_in(parents: dict[ast.AST, ast.AST], node: ast.AST) -> ast.AST | None:
        below: ast.AST | None = None
        child, current = node, parents.get(node)
        while current is not None:
            if isinstance(current, _FUNCTION_SCOPES):
                body = current.body if isinstance(current.body, list) else [current.body]
                if child in body or (isinstance(node, ast.arg) and child is current.args):
                    return current
            elif isinstance(current, _COMPREHENSION_SCOPES):
                first = current.generators[0]
                if not (child is first and below is first.iter):
                    return current
            below, child, current = child, current, parents.get(current)
        return None
