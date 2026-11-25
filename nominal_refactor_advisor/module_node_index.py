"""One walk of a parsed module, shared by every pass that reads its nodes."""

from __future__ import annotations

import ast
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property

from nominal_refactor_advisor.ast_field_decoders import alias_names
from nominal_refactor_advisor.ast_tools import (
    AstParentIndex,
    ParsedModule,
    ParsedModuleFact,
)
from nominal_refactor_advisor.declaration_dependencies import (
    ModuleLexicalDependencyProjection,
    ModuleNameReferenceSurface,
)
from nominal_refactor_advisor.lexical_bindings import (
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    ImportBoundNameProjection,
)
from nominal_refactor_advisor.lexical_scopes import (
    FunctionBindingProjection,
    ScopeBindingProjection,
)

_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSION_SCOPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _collect_import_statements(
    statements: list[ast.stmt],
    imports: list[ast.Import | ast.ImportFrom],
    import_froms: list[ast.ImportFrom],
    inner_imports: list[ast.Import | ast.ImportFrom],
    *,
    in_inner_scope: bool = False,
) -> None:
    """Collect every import statement in source order by walking statement blocks only."""

    for statement in statements:
        if isinstance(statement, ast.ImportFrom):
            imports.append(statement)
            import_froms.append(statement)
            if in_inner_scope:
                inner_imports.append(statement)
        elif isinstance(statement, ast.Import):
            imports.append(statement)
            if in_inner_scope:
                inner_imports.append(statement)
        elif isinstance(
            statement,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            _collect_import_statements(
                statement.body, imports, import_froms, inner_imports, in_inner_scope=True
            )
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            _collect_import_statements(
                statement.body,
                imports,
                import_froms,
                inner_imports,
                in_inner_scope=in_inner_scope,
            )
        elif isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            _collect_import_statements(
                statement.body,
                imports,
                import_froms,
                inner_imports,
                in_inner_scope=in_inner_scope,
            )
            if statement.orelse:
                _collect_import_statements(
                    statement.orelse,
                    imports,
                    import_froms,
                    inner_imports,
                    in_inner_scope=in_inner_scope,
                )
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            _collect_import_statements(
                statement.body,
                imports,
                import_froms,
                inner_imports,
                in_inner_scope=in_inner_scope,
            )
            for handler in statement.handlers:
                _collect_import_statements(
                    handler.body,
                    imports,
                    import_froms,
                    inner_imports,
                    in_inner_scope=in_inner_scope,
                )
            if statement.orelse:
                _collect_import_statements(
                    statement.orelse,
                    imports,
                    import_froms,
                    inner_imports,
                    in_inner_scope=in_inner_scope,
                )
            if statement.finalbody:
                _collect_import_statements(
                    statement.finalbody,
                    imports,
                    import_froms,
                    inner_imports,
                    in_inner_scope=in_inner_scope,
                )
        elif isinstance(statement, ast.Match):
            for match_case in statement.cases:
                _collect_import_statements(
                    match_case.body,
                    imports,
                    import_froms,
                    inner_imports,
                    in_inner_scope=in_inner_scope,
                )


@dataclass(frozen=True)
class ModuleImportStatements(ParsedModuleFact):
    """Every `import` and `from ... import` in one module, without indexing expressions."""

    imports: tuple[ast.Import | ast.ImportFrom, ...]
    import_froms: tuple[ast.ImportFrom, ...]
    inner_imports: tuple[ast.Import | ast.ImportFrom, ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> ModuleImportStatements:
        imports: list[ast.Import | ast.ImportFrom] = []
        import_froms: list[ast.ImportFrom] = []
        inner_imports: list[ast.Import | ast.ImportFrom] = []
        _collect_import_statements(
            parsed_module.module.body, imports, import_froms, inner_imports
        )
        return cls(tuple(imports), tuple(import_froms), tuple(inner_imports))


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
    resolve to the module's binding. Class writes have their own frame; reads
    use the existing ordered lexical resolver, including class fallback and
    function/comprehension lookup that skips enclosing class locals."""

    module: ast.Module = field(repr=False)
    imports: tuple[ast.Import | ast.ImportFrom, ...]
    string_constants: tuple[ast.Constant, ...]
    calls: tuple[ast.Call, ...]
    names_by_id: dict[str, tuple[ast.Name, ...]] = field(repr=False)
    parents: AstParentIndex = field(repr=False)
    loads: Counter[str] = field(repr=False)
    attribute_bases: frozenset[str]
    attributes_by_attr: dict[str, tuple[ast.Attribute, ...]] = field(repr=False)
    bound_name_counts: Counter[str] = field(repr=False)
    binding_sites: dict[str, tuple[ast.AST, ...]] = field(repr=False)
    identifier_text: str = field(repr=False)

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> ModuleNodeIndex:
        imports: list[ast.Import | ast.ImportFrom] = []
        strings: list[ast.Constant] = []
        calls: list[ast.Call] = []
        names: dict[str, list[ast.Name]] = {}
        parents = parsed_module.syntax_index.parent_index
        loads: Counter[str] = Counter()
        attribute_bases: set[str] = set()
        attributes_by_attr: dict[str, list[ast.Attribute]] = {}
        bound: Counter[str] = Counter()
        package_heads: set[str] = set()
        sites: dict[str, list[ast.AST]] = {}
        for node in parsed_module.syntax_index.breadth_first_nodes:
            match node:
                case ast.Import(names=aliases):
                    imports.append(node)
                    for alias in aliases:
                        if alias.asname is None:
                            package_heads.add(alias.name.split(".")[0])
                        else:
                            bound[alias.asname] += 1
                        sites.setdefault(
                            ImportBoundNameProjection(node).alias_bound_name(alias), []
                        ).append(node)
                case ast.ImportFrom(names=aliases):
                    imports.append(node)
                    for alias in aliases:
                        if alias.name != "*":
                            bound_name = ImportBoundNameProjection(node).alias_bound_name(alias)
                            bound[bound_name] += 1
                            sites.setdefault(bound_name, []).append(node)
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
                case ast.Attribute(attr=attr_name, value=ast.Name(id=name)):
                    attribute_bases.add(name)
                    attributes_by_attr.setdefault(attr_name, []).append(node)
                case ast.Attribute(attr=attr_name) if not isinstance(node.value, ast.Name):
                    attributes_by_attr.setdefault(attr_name, []).append(node)
        bound.update(package_heads)
        return cls(
            module=parsed_module.module,
            imports=tuple(imports),
            string_constants=tuple(strings),
            calls=tuple(calls),
            names_by_id={name: tuple(nodes) for name, nodes in names.items()},
            parents=parents,
            loads=loads,
            attribute_bases=frozenset(attribute_bases),
            attributes_by_attr={
                name: tuple(nodes) for name, nodes in attributes_by_attr.items()
            },
            bound_name_counts=bound,
            binding_sites={name: tuple(nodes) for name, nodes in sites.items()},
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
            and any(not alias_names(alias) and alias.name.split(".")[0] == name for alias in site.names)
        ]
        return len(sites) - len(heads) + bool(heads)

    def binding_scope(self, site: ast.AST, name: str) -> ast.AST | None:
        """The scope whose variable a binding of `name` at `site` sets."""

        scope = self.scope_of(site)
        while scope is not None:
            declarations = self.scope_declarations[scope]
            if name in declarations.global_names:
                return None
            if name in declarations.nonlocal_names:
                scope = self.scope_of(scope)
                while scope is not None:
                    if not isinstance(scope, ast.ClassDef) and name in self.scope_declarations[scope].local_names:
                        return scope
                    scope = self.scope_of(scope)
                raise ValueError(f"No enclosing function binds nonlocal {name!r}")
            if name in declarations.local_names:
                return scope
            scope = self.scope_of(scope)
        return None

    def binder_of(self, node: ast.AST, name: str) -> ast.AST | None:
        """The scope whose variable `name` at `node` reads or sets."""

        reference = self.name_bindings.get(node)
        if reference is not None:
            reference.resolution.require_known(name)
            return reference.binding_scope
        return self.binding_scope(node, name)

    def local_names(self, scope: ast.AST) -> frozenset[str]:
        """The names `scope` binds itself, minus those it declares `global`."""

        return self.scope_declarations[scope].local_names

    @cached_property
    def name_bindings(self) -> dict[ast.AST, ModuleNameReferenceSurface]:
        """Project existing lexical decisions, never reconstruct class lookup here."""
        projection = ModuleLexicalDependencyProjection.from_module(self.module)
        return {
            surface.reference: surface
            for surface in projection.name_bindings
            if surface.use.is_direct_source
        }

    @cached_property
    def scope_declarations(self) -> dict[ast.AST, ScopeBindingProjection]:
        declarations: dict[ast.AST, ScopeBindingProjection] = {}
        for scope in self.parents.values():
            if scope in declarations:
                continue
            match scope:
                case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.Lambda():
                    declarations[scope] = FunctionBindingProjection.from_function(scope)
                case ast.ClassDef():
                    declarations[scope] = ScopeBindingProjection.from_nodes(scope.body, node=scope)
                case ast.ListComp() | ast.SetComp() | ast.DictComp() | ast.GeneratorExp():
                    declarations[scope] = FunctionBindingProjection.from_comprehension(scope)
        return declarations

    def scope_of(self, node: ast.AST) -> ast.AST | None:
        """The innermost class, function, lambda or comprehension whose variables
        `node` belongs to, or `None` for the module's. A function's decorators,
        defaults and annotations, and a comprehension's first iterable, are
        evaluated in the enclosing scope; its parameters belong to it."""

        return self._scope_in(self.parents, node)

    def bound_once(self, read: ast.Name) -> bool:
        """Whether every value `read` can hold there comes from one binding
        (`LexicalScopeBindingAuthority.bound_once`), so the type checker's type of
        it, which a later binding from an untyped value leaves in place, holds."""

        ancestors: list[ast.AST] = []
        node = self.parents.get(read)
        while node is not None:
            ancestors.append(node)
            node = self.parents.get(node)
        return LEXICAL_SCOPE_BINDING_AUTHORITY.bound_once(read, ancestors[::-1])

    @staticmethod
    def _scope_in(parents: AstParentIndex, node: ast.AST) -> ast.AST | None:
        below: ast.AST | None = None
        child, current = node, parents.get(node)
        while current is not None:
            if isinstance(current, _FUNCTION_SCOPES):
                body = current.body if isinstance(current.body, list) else [current.body]
                if child in body or (isinstance(node, ast.arg) and child is current.args):
                    return current
            elif isinstance(current, ast.ClassDef):
                if child in current.body:
                    return current
            elif isinstance(current, _COMPREHENSION_SCOPES):
                first = current.generators[0]
                if not (child is first and below is first.iter):
                    return current
            below, child, current = child, current, parents.get(current)
        return None


@dataclass(frozen=True)
class ScopedImportBindings(ParsedModuleFact):
    """What each name an inner scope binds by import reads, scope by scope.

    A class, function, lambda or comprehension that binds a name has its own
    variable of it (`ModuleNodeIndex.binder_of` says whose variable a read
    uses). That variable reads an imported name exactly when every binding of
    it in that scope is an import of one qualified name; bound any other way,
    or by imports of different names, it is the scope's own value and reads
    no import. The module's own imports are `ModuleImportBindings`'."""

    node_index: ModuleNodeIndex = field(repr=False)
    imported_names: dict[ast.AST, dict[str, str | None]] = field(repr=False)

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> ScopedImportBindings:
        node_index = parsed_module.fact(ModuleNodeIndex)
        targets: dict[ast.AST, dict[str, set[str | None]]] = {}
        for name, sites in node_index.binding_sites.items():
            for site in sites:
                scope = node_index.binding_scope(site, name)
                if scope is None:
                    continue
                targets.setdefault(scope, {}).setdefault(name, set()).update(
                    (
                        origin.qualified_name
                        for origin in ImportBoundNameProjection(site).origins(
                            parsed_module.module_path_identity
                        )
                        if origin.bound_name == name
                    )
                    if isinstance(site, ast.Import | ast.ImportFrom)
                    else (None,)
                )
        return cls(
            node_index=node_index,
            imported_names={
                scope: {
                    name: next(iter(names)) if len(names) == 1 else None
                    for name, names in names_by_name.items()
                }
                for scope, names_by_name in targets.items()
            },
        )

    @cached_property
    def inner_bound_names(self) -> frozenset[str]:
        """Every name some inner scope binds, however."""

        return frozenset(name for names in self.imported_names.values() for name in names)

    def aliases(self) -> Iterable[tuple[str, str]]:
        """Each inner-scope name and the qualified name its imports read."""

        for names in self.imported_names.values():
            for name, qualified_name in names.items():
                if qualified_name is not None:
                    yield name, qualified_name

    @classmethod
    def inner_import_aliases(cls, parsed_module: ParsedModule) -> Iterable[tuple[str, str]]:
        """Each inner-scope import alias without indexing modules that have no inner imports."""

        if not parsed_module.fact(ModuleImportStatements).inner_imports:
            return ()
        return parsed_module.fact(cls).aliases()
