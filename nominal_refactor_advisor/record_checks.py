"""Compact declared-attribute check evidence joined to the existing class index."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .annotation_semantics import NOMINAL_ANNOTATION_SOURCE_AUTHORITY
from .ast_tools import ClassFunctionStackNodeVisitor, ParsedModule
from .class_index import (
    CompactClassFamilyIndex,
    CompactClassReferenceResolver,
    CompactIndexedClass,
    CompactClassMemberDeclaration,
    CompactNominalReference,
    ModuleNominalBindingAuthority,
    ModuleNominalBindingSnapshot,
)
from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from .models import SourceLocation


@dataclass(frozen=True)
class DeclaredAttributeCheck:
    location: SourceLocation
    subject_type: tuple[str, ...]
    attribute: str
    expected_type: CompactNominalReference
    expression: str


@dataclass(frozen=True)
class DeclaredTypeCheckModule:
    """Source bindings and check sites retained by the semantic module projection."""

    bindings: ModuleNominalBindingSnapshot
    checks: tuple[DeclaredAttributeCheck, ...]

    @classmethod
    def collect(
        cls, module: ParsedModule, *, include_checks: bool = True
    ) -> DeclaredTypeCheckModule:
        collector = DeclaredAttributeCheckCollector(module)
        if include_checks:
            collector.visit(module.module)
        return cls(collector.module_bindings, tuple(collector.checks))


class DeclaredAttributeCheckCollector(ClassFunctionStackNodeVisitor):
    """Collect contracts once at the lexical boundary, without a second class index."""

    def __init__(self, module: ParsedModule) -> None:
        super().__init__()
        self.module = module
        self.module_bindings = ModuleNominalBindingAuthority(module).snapshot_before(
            None
        )
        self.subjects: list[dict[str, tuple[str, ...]]] = []
        self.locals: list[frozenset[str]] = []
        self.checks: list[DeclaredAttributeCheck] = []

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        assigned = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
        parameters = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        subjects = {}
        for parameter in parameters:
            if parameter.arg not in assigned and parameter.annotation is not None:
                parts = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_parts_or_none(
                    parameter.annotation
                )
                if parts is not None:
                    subjects[parameter.arg] = parts
        if (
            self.class_stack
            and not self.function_stack
            and parameters
            and parameters[0].arg == "self"
            and "self" not in assigned
            and not any(
                isinstance(d, ast.Name) and d.id in {"staticmethod", "classmethod"}
                for d in node.decorator_list
            )
        ):
            subjects["self"] = (*self.class_stack,)
        self.subjects.append(subjects)
        self.locals.append(
            assigned | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(node)
        )
        try:
            super().visit_FunctionDef(node)
        finally:
            self.subjects.pop()
            self.locals.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def _builtin(self, node: ast.AST, name: str) -> bool:
        return (
            isinstance(node, ast.Name)
            and node.id == name
            and not any(name in scope for scope in self.locals)
            and self.module_bindings.resolves_unshadowed_builtin(name)
        )

    def visit_Call(self, node: ast.Call) -> None:
        if (
            self._builtin(node.func, "isinstance")
            and len(node.args) == 2
            and not node.keywords
        ):
            self._record(node, node.args[0], node.args[1])
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) == 1 and isinstance(
            node.ops[0], (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)
        ):
            for call, expected in (
                (node.left, node.comparators[0]),
                (node.comparators[0], node.left),
            ):
                if (
                    isinstance(call, ast.Call)
                    and self._builtin(call.func, "type")
                    and len(call.args) == 1
                    and not call.keywords
                ):
                    self._record(node, call.args[0], expected)
        self.generic_visit(node)

    def _record(self, node: ast.AST, value: ast.AST, expected: ast.AST) -> None:
        if (
            not self.subjects
            or not isinstance(value, ast.Attribute)
            or not isinstance(value.value, ast.Name)
        ):
            return
        subject_type = self.subjects[-1].get(value.value.id)
        expected_parts = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_parts_or_none(
            expected
        )
        if (
            subject_type is None
            or expected_parts is None
            or any(expected_parts[0] in scope for scope in self.locals)
        ):
            return
        self.checks.append(
            DeclaredAttributeCheck(
                SourceLocation(self.module.file_path, node.lineno, self.qualname),
                subject_type,
                value.attr,
                self.module_bindings.reference_for(expected_parts),
                ast.unparse(node),
            )
        )


@dataclass(frozen=True)
class DeclaredAttributeContractResolver:
    """Join compact check sites to original class-member and module bindings."""

    class_index: CompactClassFamilyIndex
    class_resolver: CompactClassReferenceResolver
    modules: dict[str, DeclaredTypeCheckModule]

    def field(
        self, symbol: str, name: str, seen: frozenset[str] = frozenset()
    ) -> tuple[CompactIndexedClass, CompactClassMemberDeclaration] | None:
        if symbol in seen:
            return None
        owner = self.class_index.classes_by_symbol[symbol]
        for member in owner.direct_member_declarations:
            if member.name == name:
                return (
                    (owner, member)
                    if member.annotation_reference_parts is not None
                    else None
                )
        if name in owner.method_names:
            return None
        candidates = [
            found
            for base in owner.resolved_base_symbols
            if (found := self.field(base, name, seen | {symbol})) is not None
        ]
        return candidates[0] if len(candidates) == 1 else None

    def resolve(
        self, module_name: str, check: DeclaredAttributeCheck
    ) -> tuple[SourceLocation, str] | None:
        symbol = self.class_resolver.symbol_for(
            module_name=module_name,
            reference_parts=check.subject_type,
            allow_unique_unqualified=False,
        )
        field = None if symbol is None else self.field(symbol, check.attribute)
        if field is None:
            return None
        owner, member = field
        parts = member.annotation_reference_parts
        if parts is None:
            return None
        declared = self.modules[owner.module_name].bindings.reference_for(parts)
        declared_symbol = self.class_resolver.symbol_for(
            module_name=owner.module_name,
            reference_parts=declared.resolved_parts,
            allow_unique_unqualified=False,
        )
        expected_symbol = self.class_resolver.symbol_for(
            module_name=module_name,
            reference_parts=check.expected_type.resolved_parts,
            allow_unique_unqualified=False,
        )
        matches = (
            declared_symbol == expected_symbol
            if declared_symbol is not None
            else declared.root_binding is not None
            and declared.resolved_parts[0] == "builtins"
            and declared.resolved_parts == check.expected_type.resolved_parts
        )
        if not matches:
            return None
        return (
            SourceLocation(
                owner.file_path, member.line, f"{owner.qualname}.{member.name}"
            ),
            member.annotation_expression,
        )
