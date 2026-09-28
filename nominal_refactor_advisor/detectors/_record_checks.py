"""Declared attribute-contract checks, using NRA's class and lexical authorities."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..annotation_semantics import NOMINAL_ANNOTATION_SOURCE_AUTHORITY
from ..ast_tools import ClassFunctionStackNodeVisitor, ParsedModule
from ..class_index import (
    ClassFamilyIndex, ModuleClassReferenceResolver, ModuleNominalBindingAuthority,
    build_class_family_index,
)
from ..lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from ..models import ProbeCountMetrics, RefactorFinding, SourceLocation
from ..patterns import PatternId
from ._base import DetectorConfig, IssueDetector, high_confidence_spec


@dataclass(frozen=True)
class DeclaredAttributeCheck:
    check: SourceLocation
    declaration: SourceLocation
    expression: str
    annotation: str


class DeclaredAttributeCheckCollector(ClassFunctionStackNodeVisitor):
    """Resolve annotated parameters and self through existing class declarations.

    Union/Any, dynamic attributes, rebound subjects, shadowed builtins, and
    unresolved classes provide no single declared type and produce no lead.
    A lead is about the source contract, not proof that Python enforces it.
    """

    def __init__(self, module: ParsedModule, index: ClassFamilyIndex,
                 modules_by_name: dict[str, ParsedModule]) -> None:
        super().__init__()
        self.module = module
        self.index = index
        self.modules_by_name = modules_by_name
        self.resolver = ModuleClassReferenceResolver(module, index)
        self.module_bindings = ModuleNominalBindingAuthority(module).snapshot_before(None)
        self.subjects: list[dict[str, str]] = []
        self.locals: list[frozenset[str]] = []
        self.checks: list[DeclaredAttributeCheck] = []

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        assigned = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
        parameters = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        subjects = {}
        for parameter in parameters:
            if parameter.arg in assigned or parameter.annotation is None:
                continue
            reference = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_or_none(parameter.annotation)
            symbol = None if reference is None else self.resolver.symbol_for_reference(reference)
            if symbol is not None:
                subjects[parameter.arg] = symbol
        # Only an actual instance method's first parameter supplies implicit self.
        if (self.class_stack and not self.function_stack and parameters
                and parameters[0].arg == "self" and "self" not in assigned
                and not any(isinstance(d, ast.Name) and d.id in {"staticmethod", "classmethod"}
                            for d in node.decorator_list)):
            symbol = self.index.symbol_for(file_path=self.module.file_path,
                                           qualname=".".join(self.class_stack))
            if symbol is not None:
                subjects["self"] = symbol
        self.subjects.append(subjects)
        self.locals.append(assigned | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(node))
        try:
            super().visit_FunctionDef(node)
        finally:
            self.subjects.pop()
            self.locals.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # Lambda parameters form a separate contract scope.
        pass

    def _builtin(self, node: ast.AST, name: str) -> bool:
        return (isinstance(node, ast.Name) and node.id == name
                and not any(name in scope for scope in self.locals)
                and self.module_bindings.resolves_unshadowed_builtin(name))

    def visit_Call(self, node: ast.Call) -> None:
        if self._builtin(node.func, "isinstance") and len(node.args) == 2 and not node.keywords:
            self._record(node, node.args[0], node.args[1])
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
            for call, expected in ((node.left, node.comparators[0]), (node.comparators[0], node.left)):
                if (isinstance(call, ast.Call) and self._builtin(call.func, "type")
                        and len(call.args) == 1 and not call.keywords):
                    self._record(node, call.args[0], expected)
        self.generic_visit(node)

    def _field(self, symbol: str, name: str, seen: frozenset[str] = frozenset()):
        if symbol in seen:
            return None
        owner = self.index.classes_by_symbol[symbol]
        for statement in owner.node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.target.id == name:
                return owner, statement.annotation, statement.lineno
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == name:
                return None  # a descriptor is not a stored field contract
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in statement.targets):
                return None
        inherited = [self._field(base, name, seen | {symbol}) for base in owner.resolved_base_symbols]
        candidates = [candidate for candidate in inherited if candidate is not None]
        # Conflicting inherited declarations need MRO/domain investigation, not a type-check verdict.
        return candidates[0] if len(candidates) == 1 else None

    def _record(self, node: ast.AST, value: ast.AST, expected: ast.AST) -> None:
        if (not self.subjects or not isinstance(value, ast.Attribute)
                or not isinstance(value.value, ast.Name)):
            return
        symbol = self.subjects[-1].get(value.value.id)
        field = None if symbol is None else self._field(symbol, value.attr)
        if field is None:
            return
        owner, annotation, line = field
        declared = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_or_none(annotation)
        target = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_or_none(expected)
        if declared is None or target is None:
            return
        if NOMINAL_ANNOTATION_SOURCE_AUTHORITY.source_or_none(declared) is None:
            return
        owner_module = self.modules_by_name[owner.module_name]
        declared_resolver = ModuleClassReferenceResolver(owner_module, self.index)
        declared_symbol = declared_resolver.symbol_for_reference(declared)
        target_symbol = self.resolver.symbol_for_reference(target)
        if declared_symbol is not None:
            matches = declared_symbol == target_symbol
        else:
            bindings = ModuleNominalBindingAuthority(owner_module).snapshot_before(None)
            matches = (isinstance(declared, ast.Name) and isinstance(target, ast.Name)
                       and declared.id == target.id and self._builtin(target, target.id)
                       and bindings.resolves_unshadowed_builtin(declared.id))
        if matches:
            self.checks.append(DeclaredAttributeCheck(
                SourceLocation(self.module.file_path, node.lineno, self.qualname),
                SourceLocation(owner.file_path, line, f"{owner.qualname}.{value.attr}"),
                ast.unparse(node), ast.unparse(annotation),
            ))


class RedundantTypeCheckDetector(IssueDetector):
    """Source-backed leads for rechecking an already declared attribute type."""

    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Type check repeats a declared attribute contract",
        "An annotated parameter or self already names the field's declared owner. "
        "Validate external values at decoding, then trust that contract. Python does "
        "not enforce annotations; exact-type checks can further exclude subclasses, "
        "so this lead does not certify deleting a runtime check.",
        "one type-validation boundary and trusted nominal field consumers",
        "attribute type is redeclared by a type or isinstance check",
    )

    def _collect_findings(self, modules: list[ParsedModule], config: DetectorConfig) -> list[RefactorFinding]:
        del config
        index = build_class_family_index(modules)
        modules_by_name = {module.module_name: module for module in modules}
        findings = []
        for module in modules:
            collector = DeclaredAttributeCheckCollector(module, index, modules_by_name)
            collector.visit(module.module)
            for check in collector.checks:
                findings.append(self.build_finding(
                    f"`{check.expression}` repeats declared attribute type `{check.annotation}`. "
                    "Boundary validation and exact-type/subclass intent remain to be checked before removal.",
                    (check.check, check.declaration), metrics=ProbeCountMetrics(1),
                ))
        return findings
