"""Repository-wide class-family indexing helpers.

This module builds a lightweight cross-module view of declared classes and
their resolved inheritance edges. The index is intentionally conservative:
it resolves only import patterns and base expressions that can be recovered
reliably from the local AST.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from pathlib import Path

from .ast_tools import (
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    CollectedFamily,
    ParsedModule,
)
from .collection_algebra import sorted_tuple


@dataclass(frozen=True)
class IndexedClass:
    symbol: str
    module_name: str
    qualname: str
    simple_name: str
    file_path: str
    line: int
    node: ast.ClassDef
    declared_base_names: tuple[str, ...]
    resolved_base_symbols: tuple[str, ...]

    @property
    def is_final(self) -> bool:
        return any(
            (isinstance(decorator, ast.Name) and decorator.id == "final")
            or (isinstance(decorator, ast.Attribute) and decorator.attr == "final")
            for decorator in self.node.decorator_list
        )

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
            file_path=str(parsed_module.path),
            line=node.lineno,
            node=node,
            declared_base_names=tuple(
                declared_base_name
                for base in node.bases
                if (
                    declared_base_name := ClassSymbolResolutionAuthority.declared_base_name(
                        base
                    )
                )
                is not None
            ),
            resolved_base_symbols=(),
        )

    def with_resolved_base_symbols(
        self,
        resolved_base_symbols: tuple[str, ...],
    ) -> "IndexedClass":
        return replace(self, resolved_base_symbols=resolved_base_symbols)


@dataclass(frozen=True)
class CompactIndexedClass:
    """AST-free class declaration used to reconstruct inheritance globally."""

    symbol: str
    module_name: str
    qualname: str
    simple_name: str
    file_path: str
    line: int
    declared_base_names: tuple[str, ...]
    base_reference_parts: tuple[tuple[str, ...], ...]
    direct_assignment_expressions: tuple[tuple[str, str | None], ...] = ()
    direct_assignment_lines: tuple[tuple[str, int], ...] = ()
    metaclass_names: tuple[str, ...] = ()
    keyed_family_key_type_name: str | None = None
    is_final: bool = False
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
    resolved_base_symbols: tuple[str, ...] = ()

    @property
    def assignments_by_name(self) -> dict[str, str | None]:
        return dict(self.direct_assignment_expressions)

    @property
    def assignment_lines_by_name(self) -> dict[str, int]:
        lines: dict[str, int] = {}
        for name, line in self.direct_assignment_lines:
            lines.setdefault(name, line)
        return lines

    def with_resolved_base_symbols(
        self,
        resolved_base_symbols: tuple[str, ...],
    ) -> "CompactIndexedClass":
        return replace(self, resolved_base_symbols=resolved_base_symbols)


@dataclass(frozen=True)
class CompactModuleClassProjection:
    """One module's class declarations and import aliases, without its AST."""

    module_name: str
    file_path: str
    import_aliases: tuple[tuple[str, str], ...]
    classes: tuple[CompactIndexedClass, ...]
    registry_order_calls: tuple["CompactRegistryOrderCall", ...] = ()
    keyed_table_axes: tuple["CompactKeyedTableAxis", ...] = ()
    closed_axis_branch_functions: tuple["CompactClosedAxisBranchFunction", ...] = ()
    manual_selector_axes: tuple["CompactManualSelectorAxis", ...] = ()
    top_level_definitions: tuple[tuple[str, int], ...] = ()
    exact_type_guards: tuple["CompactExactTypeGuard", ...] = ()
    autoregister_function_references: tuple[
        "CompactAutoRegisterFunctionReference", ...
    ] = ()
    autoregister_reference_index: "CompactAutoRegisterReferenceIndex | None" = None


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


@dataclass(frozen=True)
class CompactRegistryOrderCall:
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


@dataclass(frozen=True)
class CompactClassFamilyIndex:
    """Repository inheritance graph reconstructed from compact declarations."""

    classes_by_symbol: dict[str, CompactIndexedClass]
    symbols_by_simple_name: dict[str, tuple[str, ...]]
    children_by_symbol: dict[str, tuple[str, ...]]
    ancestors_by_symbol: dict[str, tuple[str, ...]]
    descendants_by_symbol: dict[str, tuple[str, ...]]

    def class_for(self, symbol: str) -> CompactIndexedClass | None:
        return self.classes_by_symbol.get(symbol)

    def descendant_symbols(self, base_symbol: str) -> tuple[str, ...]:
        return self.descendants_by_symbol.get(base_symbol, ())

    def ancestor_symbols(self, class_symbol: str) -> tuple[str, ...]:
        return self.ancestors_by_symbol.get(class_symbol, ())


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
            if _resolved_path_text(indexed_class.file_path) not in file_paths
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


@lru_cache(maxsize=None)
def _unique_known_symbol_by_suffix(
    known_symbols: frozenset[str],
) -> dict[str, str]:
    """Index the suffix rule once instead of scanning every class per reference."""

    unique: dict[str, str] = {}
    ambiguous: set[str] = set()
    for symbol in known_symbols:
        parts = symbol.split(".")
        for start in range(len(parts)):
            suffix = ".".join(parts[start:])
            if suffix in ambiguous:
                continue
            previous = unique.get(suffix)
            if previous is None:
                unique[suffix] = symbol
            elif previous != symbol:
                del unique[suffix]
                ambiguous.add(suffix)
    return unique


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


class CompactModuleClassProjectionFamily(CollectedFamily[CompactModuleClassProjection]):
    """Persist class/import facts needed by the global inheritance graph."""

    item_type = CompactModuleClassProjection

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactModuleClassProjection]:
        del cls
        file_path = str(parsed_module.path)
        (
            autoregister_function_references,
            autoregister_reference_index,
        ) = _compact_autoregister_function_references(parsed_module)
        classes = tuple(
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
                base_reference_parts=tuple(
                    parts
                    for base in node.bases
                    if (
                        parts := ATTRIBUTE_CHAIN_AUTHORITY.project(
                            ClassSymbolResolutionAuthority.reference_node(base)
                        )
                    )
                    is not None
                ),
                direct_assignment_expressions=tuple(
                    (target_name, ast.unparse(value) if value is not None else None)
                    for target_name, value in _direct_class_assignments(node).items()
                ),
                direct_assignment_lines=tuple(_direct_class_assignment_lines(node)),
                metaclass_names=tuple(
                    terminal_name
                    for keyword in node.keywords
                    if keyword.arg == "metaclass"
                    if (terminal_name := _terminal_reference_name(keyword.value))
                    is not None
                ),
                keyed_family_key_type_name=_keyed_family_key_type_name(node),
                is_final=any(
                    (isinstance(decorator, ast.Name) and decorator.id == "final")
                    or (
                        isinstance(decorator, ast.Attribute)
                        and decorator.attr == "final"
                    )
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
                autoregister_registry_projection_names=_autoregister_registry_projection_names(
                    node
                ),
            )
            for qualname, node in _iter_class_defs(list(parsed_module.module.body))
        )
        return [
            CompactModuleClassProjection(
                module_name=parsed_module.module_name,
                file_path=file_path,
                import_aliases=tuple(
                    sorted(_module_import_aliases(parsed_module).items())
                ),
                classes=classes,
                registry_order_calls=_compact_registry_order_calls(
                    parsed_module.module
                ),
                keyed_table_axes=_compact_keyed_table_axes(parsed_module),
                closed_axis_branch_functions=_compact_closed_axis_branch_functions(
                    parsed_module
                ),
                manual_selector_axes=_compact_manual_selector_axes(parsed_module),
                top_level_definitions=tuple(
                    (node.name, node.lineno)
                    for node in parsed_module.module.body
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                ),
                exact_type_guards=_compact_exact_type_guards(parsed_module),
                autoregister_function_references=autoregister_function_references,
                autoregister_reference_index=autoregister_reference_index,
            )
        ]


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
                file_path=str(parsed_module.path),
                line=statement.lineno,
                table_name=table_name,
                key_type_name=next(iter(key_type_names)),
                case_names=sorted_tuple(case_names),
                value_shape_name=value_shape_name,
            )
        )
    return tuple(axes)


def _compact_closed_axis_branch_functions(
    parsed_module: ParsedModule,
) -> tuple[CompactClosedAxisBranchFunction, ...]:
    facts: list[CompactClosedAxisBranchFunction] = []
    for qualname, function in _named_functions(parsed_module.module):
        branch_site_counts: dict[str, int] = defaultdict(int)
        case_names_by_key: dict[str, set[str]] = defaultdict(set)
        for subnode in _non_nested_function_subnodes(function):
            if isinstance(subnode, ast.If):
                refs = _enum_member_refs_by_key_type(subnode.test)
                for key_type_name, case_names in refs.items():
                    branch_site_counts[key_type_name] += 1
                    case_names_by_key[key_type_name].update(case_names)
            elif isinstance(subnode, ast.Match):
                refs_by_key: dict[str, set[str]] = defaultdict(set)
                for case in subnode.cases:
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
                    branch_site_counts[key_type_name] += 1
                    case_names_by_key[key_type_name].update(case_names)
        axes = tuple(
            CompactClosedAxisBranchFact(
                key_type_name=key_type_name,
                branch_site_count=branch_site_count,
                case_names=sorted_tuple(case_names_by_key[key_type_name]),
            )
            for key_type_name, branch_site_count in sorted(branch_site_counts.items())
        )
        if axes:
            facts.append(
                CompactClosedAxisBranchFunction(
                    file_path=str(parsed_module.path),
                    line=function.lineno,
                    qualname=qualname,
                    axes=axes,
                )
            )
    return tuple(facts)


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
    for node in ast.walk(parsed_module.module):
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
                    file_path=str(parsed_module.path),
                    line=method.lineno,
                    family_name=node.name,
                    selector_method_name=method.name,
                    key_type_name=next(iter(key_type_names)),
                    case_names=case_names,
                )
            )
    return tuple(axes)


def _compact_exact_type_guards(
    parsed_module: ParsedModule,
) -> tuple[CompactExactTypeGuard, ...]:
    guards: list[CompactExactTypeGuard] = []
    module_bindings = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
        parsed_module.module.body
    )

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []
            self.scope_bindings: list[frozenset[str]] = [module_bindings]
            self.callable_depth = 0

        @property
        def qualname(self) -> str:
            return ".".join(self.scope) if self.scope else "<module>"

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_callable(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_callable(node)

        def _visit_callable(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            self.scope.append(node.name)
            self.scope_bindings.append(
                LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
                | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(node)
            )
            self.callable_depth += 1
            self.generic_visit(node)
            self.callable_depth -= 1
            self.scope_bindings.pop()
            self.scope.pop()

        def visit_If(self, node: ast.If) -> None:
            predicate = _exact_type_predicate(node.test)
            if predicate is not None and self.callable_depth:
                _, _, matches_exact_type_when_true, _ = predicate
                rejects_descendants = (
                    not matches_exact_type_when_true and _fail_loud_block(node.body)
                ) or (matches_exact_type_when_true and _fail_loud_block(node.orelse))
                if rejects_descendants:
                    self._append_guard(node, predicate)
            self.generic_visit(node)

        def visit_Assert(self, node: ast.Assert) -> None:
            predicate = _exact_type_predicate(node.test)
            if predicate is not None and predicate[2] and self.callable_depth:
                self._append_guard(node, predicate)
            self.generic_visit(node)

        def _append_guard(
            self,
            node: ast.If | ast.Assert,
            predicate: tuple[ast.AST, ast.AST, bool, str],
        ) -> None:
            if any("type" in bindings for bindings in self.scope_bindings):
                return
            subject, type_reference, matches_exact_type_when_true, expression = (
                predicate
            )
            reference_node = ClassSymbolResolutionAuthority.reference_node(
                type_reference
            )
            parts = ATTRIBUTE_CHAIN_AUTHORITY.project(reference_node)
            if parts is None:
                return
            guards.append(
                CompactExactTypeGuard(
                    file_path=str(parsed_module.path),
                    line=node.lineno,
                    qualname=self.qualname,
                    subject_expression=ast.unparse(subject),
                    type_reference_expression=ast.unparse(type_reference),
                    type_reference_parts=parts,
                    matches_exact_type_when_true=matches_exact_type_when_true,
                    expression=expression,
                )
            )

    Visitor().visit(parsed_module.module)
    return tuple(guards)


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
    functions: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            functions.append((".".join((*self.class_stack, node.name)), node))
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(module)
    return tuple(functions)


def _non_nested_function_subnodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    nodes: list[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def generic_visit(self, node: ast.AST) -> None:
            nodes.append(node)
            super().generic_visit(node)

    visitor = Visitor()
    for statement in function.body:
        visitor.visit(statement)
    return tuple(nodes)


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
    extractor = _direct_class_assignments(node).get("__key_extractor__")
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


@dataclass
class _CompactAutoRegisterFunctionReferenceBuilder:
    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    receiver_attribute_refs: set[tuple[str, str]]
    calls_autoregister_meta: bool = False


def _compact_autoregister_function_references(
    parsed_module: ParsedModule,
) -> tuple[
    tuple[CompactAutoRegisterFunctionReference, ...],
    CompactAutoRegisterReferenceIndex | None,
]:
    file_path = str(parsed_module.path)
    if file_path.startswith("tests/") or "/tests/" in file_path:
        return (), None
    builders: list[_CompactAutoRegisterFunctionReferenceBuilder] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.active_functions: list[
                _CompactAutoRegisterFunctionReferenceBuilder
            ] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            builder = _CompactAutoRegisterFunctionReferenceBuilder(
                qualname=".".join((*self.class_stack, node.name)),
                node=node,
                receiver_attribute_refs=set(),
            )
            builders.append(builder)
            self.active_functions.append(builder)
            self.generic_visit(node)
            self.active_functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name):
                reference = (node.value.id, node.attr)
                for builder in self.active_functions:
                    builder.receiver_attribute_refs.add(reference)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if _terminal_reference_name(node.func) == "AutoRegisterMeta":
                for builder in self.active_functions:
                    builder.calls_autoregister_meta = True
            self.generic_visit(node)

    Visitor().visit(parsed_module.module)
    references: list[CompactAutoRegisterFunctionReference] = []
    for builder in builders:
        if builder.calls_autoregister_meta:
            referenced_symbols = {
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
                                subnode.attr
                                if isinstance(subnode, ast.Attribute)
                                else None
                            )
                        )
                    ),
                )
                if symbol is not None
            }
            references.append(
                CompactAutoRegisterFunctionReference(
                    qualname=builder.qualname,
                    referenced_symbols=sorted_tuple(referenced_symbols),
                    calls_autoregister_meta=builder.calls_autoregister_meta,
                    receiver_attribute_refs=sorted_tuple(
                        builder.receiver_attribute_refs
                    ),
                )
            )
    consumer_builders = tuple(
        builder for builder in builders if builder.receiver_attribute_refs
    )
    if not consumer_builders:
        return tuple(references), None
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
    return tuple(references), CompactAutoRegisterReferenceIndex(
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
        or _registration_authority_base_name(terminal_name)
        or ("Registered" in terminal_name and terminal_name.endswith("Meta"))
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


def _compact_registry_order_calls(
    module: ast.Module,
) -> tuple[CompactRegistryOrderCall, ...]:
    calls: list[CompactRegistryOrderCall] = []
    for node in ast.walk(module):
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
        if not registry_owner_names:
            continue
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
                CompactRegistryOrderCall(
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
        projections_by_module_name = {
            projection.module_name: projection for projection in self.projections
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
                            projections_by_module_name,
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
        projections_by_module_name: dict[str, CompactModuleClassProjection],
        known_symbols: frozenset[str],
        unique_symbols_by_name: dict[str, str],
    ) -> str | None:
        projection = projections_by_module_name.get(module_name)
        import_aliases = {} if projection is None else dict(projection.import_aliases)
        first, *rest = parts
        alias_target = import_aliases.get(first)
        if alias_target is not None:
            candidate = ".".join((alias_target, *rest)) if rest else alias_target
            if candidate in known_symbols:
                return candidate
            candidate_parts = candidate.split(".")
            unique_by_suffix = _unique_known_symbol_by_suffix(known_symbols)
            for suffix_width in range(len(candidate_parts) - 1, 0, -1):
                suffix = ".".join(candidate_parts[-suffix_width:])
                if suffix in unique_by_suffix:
                    return unique_by_suffix[suffix]
        module_local = ".".join((module_name, *parts))
        if module_local in known_symbols:
            return module_local
        if len(parts) == 1:
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
    projections_by_module_name: dict[str, CompactModuleClassProjection]
    known_symbols: frozenset[str]
    unique_symbols_by_name: dict[str, str]

    @classmethod
    def from_index(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        class_index: CompactClassFamilyIndex,
    ) -> "CompactClassReferenceResolver":
        return cls(
            projections_by_module_name={
                projection.module_name: projection for projection in projections
            },
            known_symbols=frozenset(class_index.classes_by_symbol),
            unique_symbols_by_name={
                name: symbols[0]
                for name, symbols in class_index.symbols_by_simple_name.items()
                if len(symbols) == 1
            },
        )

    def symbol_for(
        self,
        *,
        module_name: str,
        reference_parts: tuple[str, ...],
    ) -> str | None:
        return CompactClassFamilyIndexBuilder._resolved_symbol(
            reference_parts,
            module_name,
            self.projections_by_module_name,
            self.known_symbols,
            self.unique_symbols_by_name,
        )


@dataclass(frozen=True)
class ClassSymbolResolutionAuthority:
    """Resolve AST name chains to indexed class symbols under an explicit policy."""

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
    return frozenset(_resolved_path_text(str(module.path)) for module in modules)


def _resolved_path_text(file_path: str) -> str:
    return str(Path(file_path).resolve())
