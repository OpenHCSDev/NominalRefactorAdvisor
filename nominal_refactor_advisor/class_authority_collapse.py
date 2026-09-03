"""Repository proof for collapsing one redundant class authority."""

from __future__ import annotations

import ast
import builtins
import copy
from dataclasses import dataclass

from .ast_tools import LEXICAL_SCOPE_BINDING_AUTHORITY, ParsedModule
from .class_index import (
    ATTRIBUTE_CHAIN_AUTHORITY,
    ClassMethodPromotionSafetyProfile,
    ClassFamilyIndex,
    ClassSymbolResolutionAuthority,
    IndexedClass,
    ModuleClassReferenceResolver,
    ModuleNominalBindingAuthority,
    ModuleNominalBindingSnapshot,
    ModuleNominalBindingWitness,
    module_star_import_origins,
)

_NESTED_METHOD_SCOPES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.DictComp,
    ast.FunctionDef,
    ast.GeneratorExp,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
)


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


@dataclass(frozen=True)
class ClassBehaviorProofContext:
    """Class-local proof inputs derived once for all behavior comparisons."""

    owner: IndexedClass
    module_bound_names: frozenset[str]
    class_bound_names: frozenset[str]
    source_lines: tuple[str, ...]
    binding_snapshot: ModuleNominalBindingSnapshot

    @classmethod
    def from_declaration(
        cls,
        parsed_module: ParsedModule,
        owner: IndexedClass,
    ) -> "ClassBehaviorProofContext":
        return cls(
            owner=owner,
            module_bound_names=LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                parsed_module.module.body
            ),
            class_bound_names=LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                owner.node.body
            ),
            source_lines=tuple(parsed_module.source.splitlines(keepends=True)),
            binding_snapshot=ModuleNominalBindingAuthority(
                parsed_module
            ).snapshot_before(owner.line),
        )


@dataclass(frozen=True)
class ClassMethodBehaviorAuthority:
    """Normalized method syntax together with its exact global bindings."""

    syntax: str
    global_bindings: tuple[ModuleNominalBindingWitness, ...]

    @classmethod
    def from_declaration(
        cls,
        context: ClassBehaviorProofContext,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> "ClassMethodBehaviorAuthority":
        safety_profile = ClassMethodPromotionSafetyProfile.from_method(
            method,
            context.module_bound_names,
            context.class_bound_names,
            source_lines=context.source_lines,
        )
        if safety_profile.hazards:
            raise ValueError(
                f"Method {context.owner.qualname}.{method.name} has "
                "ownership-sensitive "
                f"behavior: {tuple(hazard.value for hazard in safety_profile.hazards)!r}"
            )
        normalized = cls._normalized_method(method)
        cls._require_flat_lexical_scope(normalized)
        local_names = cls._local_names(normalized)
        global_bindings: list[ModuleNominalBindingWitness] = []
        for name in sorted(
            {
                node.id
                for node in ast.walk(normalized)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in local_names
            }
        ):
            binding = context.binding_snapshot.binding_for(name)
            if binding is not None:
                global_bindings.append(
                    ModuleNominalBindingWitness(binding.qualified_name, name)
                )
                continue
            if context.binding_snapshot.resolves_unshadowed_builtin(name):
                global_bindings.append(
                    ModuleNominalBindingWitness(f"{builtins.__name__}.{name}", name)
                )
                continue
            raise ValueError(
                f"Method {context.owner.qualname}.{method.name} has unresolved global "
                f"binding {name!r}"
            )
        return cls(
            syntax=ast.dump(normalized, include_attributes=False),
            global_bindings=tuple(global_bindings),
        )

    @classmethod
    def _normalized_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        normalized = copy.deepcopy(method)
        receiver_name = cls._receiver_name(normalized)
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
                remaining_statement
            )
        ]
        return ast.fix_missing_locations(normalized)

    @staticmethod
    def _receiver_name(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str | None:
        positional = (*method.args.posonlyargs, *method.args.args)
        return positional[0].arg if positional else None

    @staticmethod
    def _require_flat_lexical_scope(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for node in ast.walk(method):
            if node is method:
                continue
            if isinstance(node, _NESTED_METHOD_SCOPES):
                raise ValueError(f"Method {method.name!r} has a nested lexical scope")
            if isinstance(node, (ast.Global, ast.Import, ast.ImportFrom, ast.Nonlocal)):
                raise ValueError(
                    f"Method {method.name!r} mutates its lexical binding surface"
                )

    @staticmethod
    def _local_names(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> frozenset[str]:
        arguments = (
            *method.args.posonlyargs,
            *method.args.args,
            *method.args.kwonlyargs,
            *((method.args.vararg,) if method.args.vararg is not None else ()),
            *((method.args.kwarg,) if method.args.kwarg is not None else ()),
        )
        return frozenset(
            (
                *(argument.arg for argument in arguments),
                *(
                    node.id
                    for node in ast.walk(method)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                ),
            )
        )


@dataclass(frozen=True)
class ObsoleteClassAuthorityImport:
    """One from-import binding used only by a displaced class declaration."""

    module_name: str
    imported_name: str


@dataclass(frozen=True)
class ClassBaseBehaviorAuthority:
    """One neutral base expression and its exact lexical declaration."""

    syntax: str
    qualified_name: str

    @classmethod
    def from_declaration(
        cls,
        context: ClassBehaviorProofContext,
    ) -> tuple["ClassBaseBehaviorAuthority", ...]:
        authorities: list[ClassBaseBehaviorAuthority] = []
        for base in context.owner.node.bases:
            parts = ATTRIBUTE_CHAIN_AUTHORITY.project(base)
            if parts is None:
                raise ValueError(
                    f"Class authority {context.owner.qualname!r} has parameterized or "
                    "computed base mechanics"
                )
            binding = context.binding_snapshot.binding_for(parts[0])
            if binding is not None:
                qualified_name = ".".join((binding.qualified_name, *parts[1:]))
            elif context.binding_snapshot.resolves_unshadowed_builtin(parts[0]):
                qualified_name = ".".join(
                    (f"{builtins.__name__}.{parts[0]}", *parts[1:])
                )
            else:
                raise ValueError(
                    f"Class authority {context.owner.qualname!r} has unresolved base "
                    f"binding {parts[0]!r}"
                )
            authorities.append(
                cls(
                    syntax=ast.dump(base, include_attributes=False),
                    qualified_name=qualified_name,
                )
            )
        return tuple(authorities)


@dataclass(frozen=True)
class RedundantClassAuthorityCollapseProof:
    """Closed repository proof for one class-authority substitution."""

    obsolete_imports: tuple[ObsoleteClassAuthorityImport, ...]

    @classmethod
    def require(
        cls,
        parsed_modules: tuple[ParsedModule, ...],
        class_index: ClassFamilyIndex,
        *,
        displaced_symbol: str,
        replacement_symbol: str,
    ) -> "RedundantClassAuthorityCollapseProof":
        displaced = cls._required_class(class_index, displaced_symbol)
        replacement = cls._required_class(class_index, replacement_symbol)
        if displaced_symbol == replacement_symbol:
            raise ValueError("Class-authority collapse requires distinct classes")
        if "." in displaced.qualname or "." in replacement.qualname:
            raise ValueError("Class-authority collapse requires top-level classes")
        related_symbols = frozenset(
            (
                *class_index.ancestor_symbols(displaced_symbol),
                *class_index.descendant_symbols(displaced_symbol),
            )
        )
        if replacement_symbol in related_symbols:
            raise ValueError(
                "Replacement and displaced class authorities are already related"
            )
        modules_by_path = {module.file_path: module for module in parsed_modules}
        if len(modules_by_path) != len(parsed_modules):
            raise ValueError("Class-authority proof requires unique source modules")
        displaced_module = cls._required_module(modules_by_path, displaced.file_path)
        replacement_module = cls._required_module(
            modules_by_path,
            replacement.file_path,
        )
        cls._require_standalone_authority(displaced)
        cls._require_standalone_authority(replacement)
        displaced_context = ClassBehaviorProofContext.from_declaration(
            displaced_module,
            displaced,
        )
        replacement_context = ClassBehaviorProofContext.from_declaration(
            replacement_module,
            replacement,
        )
        if ClassBaseBehaviorAuthority.from_declaration(
            displaced_context
        ) != ClassBaseBehaviorAuthority.from_declaration(replacement_context):
            raise ValueError("Class authorities do not have equivalent base mechanics")
        cls._require_equivalent_methods(
            displaced_context,
            replacement_context,
        )
        direct_children = tuple(
            cls._required_class(class_index, child_symbol)
            for child_symbol in class_index.children_by_symbol.get(
                displaced_symbol,
                (),
            )
        )
        if not direct_children:
            raise ValueError("Displaced class authority has no direct children")
        cls._require_closed_local_children(
            displaced,
            direct_children,
        )
        cls._require_reference_closure(
            modules_by_path,
            class_index,
            displaced,
            direct_children,
        )
        obsolete_imports = cls._obsolete_imports(displaced_module, displaced)
        return cls(obsolete_imports=obsolete_imports)

    @staticmethod
    def _required_class(
        class_index: ClassFamilyIndex,
        symbol: str,
    ) -> IndexedClass:
        indexed_class = class_index.class_for(symbol)
        if indexed_class is None:
            raise ValueError(f"Class authority {symbol!r} is unavailable")
        return indexed_class

    @staticmethod
    def _required_module(
        modules_by_path: dict[str, ParsedModule],
        file_path: str,
    ) -> ParsedModule:
        parsed_module = modules_by_path.get(file_path)
        if parsed_module is None:
            raise ValueError(f"Source module {file_path!r} is unavailable")
        return parsed_module

    @staticmethod
    def _require_standalone_authority(indexed_class: IndexedClass) -> None:
        node = indexed_class.node
        if len(indexed_class.declared_base_names) != len(node.bases) or any(
            ClassSymbolResolutionAuthority.establishes_nominal_family(base_name)
            for base_name in indexed_class.declared_base_names
        ):
            raise ValueError(
                f"Class authority {indexed_class.qualname!r} is not standalone"
            )
        if node.decorator_list or node.keywords:
            raise ValueError(
                f"Class authority {indexed_class.qualname!r} has class-creation "
                "semantics"
            )
        unsupported_statements = tuple(
            statement
            for statement in node.body
            if not isinstance(
                statement,
                (ast.AsyncFunctionDef, ast.Expr, ast.FunctionDef, ast.Pass),
            )
            or (
                isinstance(statement, ast.Expr)
                and not (
                    isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                )
            )
        )
        if unsupported_statements:
            raise ValueError(
                f"Class authority {indexed_class.qualname!r} has non-method state"
            )

    @classmethod
    def _require_equivalent_methods(
        cls,
        displaced_context: ClassBehaviorProofContext,
        replacement_context: ClassBehaviorProofContext,
    ) -> None:
        displaced = displaced_context.owner
        replacement = replacement_context.owner
        displaced_methods = cls._methods_by_name(displaced)
        replacement_methods = cls._methods_by_name(replacement)
        if displaced_methods.keys() != replacement_methods.keys():
            raise ValueError("Class authorities do not declare the same method set")
        for method_name in displaced_methods:
            displaced_behavior = ClassMethodBehaviorAuthority.from_declaration(
                displaced_context,
                displaced_methods[method_name],
            )
            replacement_behavior = ClassMethodBehaviorAuthority.from_declaration(
                replacement_context,
                replacement_methods[method_name],
            )
            if displaced_behavior != replacement_behavior:
                raise ValueError(
                    f"Class method {method_name!r} does not have equivalent behavior"
                )

    @staticmethod
    def _methods_by_name(
        indexed_class: IndexedClass,
    ) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        methods = tuple(
            statement
            for statement in indexed_class.node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        methods_by_name = {method.name: method for method in methods}
        if len(methods_by_name) != len(methods):
            raise ValueError(
                f"Class authority {indexed_class.qualname!r} rebinds a method"
            )
        return methods_by_name

    @staticmethod
    def _require_closed_local_children(
        displaced: IndexedClass,
        direct_children: tuple[IndexedClass, ...],
    ) -> None:
        for child in direct_children:
            if child.file_path != displaced.file_path or "." in child.qualname:
                raise ValueError(
                    "Class-authority collapse requires a local top-level child cohort"
                )
            if child.node.keywords:
                raise ValueError(
                    f"Direct child {child.qualname!r} has class-creation semantics"
                )

    @staticmethod
    def _require_reference_closure(
        modules_by_path: dict[str, ParsedModule],
        class_index: ClassFamilyIndex,
        displaced: IndexedClass,
        direct_children: tuple[IndexedClass, ...],
    ) -> None:
        resolvers_by_path = {
            file_path: ModuleClassReferenceResolver(module, class_index)
            for file_path, module in modules_by_path.items()
        }
        allowed_reference_node_ids: set[int] = set()
        for child in direct_children:
            resolver = resolvers_by_path[child.file_path]
            matching_bases = tuple(
                base
                for base in child.node.bases
                if resolver.symbol_for_reference(base) == displaced.symbol
            )
            if len(matching_bases) != 1:
                raise ValueError(
                    f"Direct child {child.qualname!r} does not expose one exact "
                    "displaced base reference"
                )
            allowed_reference_node_ids.update(
                id(node)
                for node in ast.walk(matching_bases[0])
                if resolver.symbol_for_reference(node) == displaced.symbol
            )

        for parsed_module in modules_by_path.values():
            resolver = resolvers_by_path[parsed_module.file_path]
            if any(
                resolver.symbol_for_reference(ast.Name(id=local_name, ctx=ast.Load()))
                == displaced.symbol
                for local_name in resolver.import_aliases
            ):
                raise ValueError(
                    f"Class authority {displaced.qualname!r} has an imported "
                    "repository reference"
                )
            if any(
                origin.module_name == displaced.module_name
                for origin in module_star_import_origins(parsed_module)
            ):
                raise ValueError(
                    f"Class authority {displaced.qualname!r} has an open star-import "
                    "boundary"
                )
            for node in ast.walk(parsed_module.module):
                if (
                    isinstance(node, (ast.Attribute, ast.Call, ast.Name, ast.Subscript))
                    and resolver.symbol_for_reference(node) == displaced.symbol
                    and id(node) not in allowed_reference_node_ids
                ):
                    raise ValueError(
                        f"Class authority {displaced.qualname!r} has a non-base "
                        "repository reference"
                    )
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value
                    in {displaced.simple_name, displaced.qualname, displaced.symbol}
                ):
                    raise ValueError(
                        f"Class authority {displaced.qualname!r} has a string "
                        "repository reference"
                    )

    @staticmethod
    def _obsolete_imports(
        parsed_module: ParsedModule,
        displaced: IndexedClass,
    ) -> tuple[ObsoleteClassAuthorityImport, ...]:
        displaced_nodes = tuple(ast.walk(displaced.node))
        displaced_node_ids = {id(node) for node in displaced_nodes}
        displaced_loaded_names = frozenset(
            node.id
            for node in displaced_nodes
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )
        remaining_loaded_names = frozenset(
            node.id
            for node in ast.walk(parsed_module.module)
            if id(node) not in displaced_node_ids
            and isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
        )
        deleted_only_names = displaced_loaded_names - remaining_loaded_names
        local_names_by_import: dict[tuple[str, str], list[str]] = {}
        for statement in parsed_module.module.body:
            if isinstance(statement, ast.Import):
                bound_names = {
                    alias.asname or alias.name.split(".", 1)[0]
                    for alias in statement.names
                }
                if bound_names & deleted_only_names:
                    raise ValueError(
                        "Class-authority collapse cannot clean an obsolete module "
                        "import"
                    )
                continue
            if not isinstance(statement, ast.ImportFrom):
                continue
            module_name = f"{'.' * statement.level}{statement.module or ''}"
            for alias in statement.names:
                if alias.name == "*":
                    continue
                local_names_by_import.setdefault((module_name, alias.name), []).append(
                    alias.asname or alias.name
                )
        obsolete: list[ObsoleteClassAuthorityImport] = []
        for (module_name, imported_name), local_names in local_names_by_import.items():
            removed_local_names = deleted_only_names.intersection(local_names)
            if not removed_local_names:
                continue
            if len(removed_local_names) != len(local_names):
                raise ValueError(
                    "Class-authority collapse cannot partially clean a shared "
                    "import binding"
                )
            obsolete.append(ObsoleteClassAuthorityImport(module_name, imported_name))
        return tuple(obsolete)
