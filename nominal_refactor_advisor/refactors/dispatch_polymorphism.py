from __future__ import annotations

import ast
import hashlib
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    ClassVar,
    Self,
    TypeAlias,
)

from metaclass_registry import AutoRegisterMeta

from nominal_refactor_advisor.ast_field_decoders import match_guards
from nominal_refactor_advisor.ast_tools import (
    statements_without_docstring,
    walk_function_body_nodes,
)
from nominal_refactor_advisor.refactors.finding_recipes import FindingRecipeSynthesizer
from nominal_refactor_advisor.refactors.source_insertion import EnsureImportOperation
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardRule,
    ArchitectureGuardTargetScope,
    ForbiddenDispatchArchitectureGuardConstraint,
)
from nominal_refactor_advisor.codemod_declaration_source import (
    NamedDeclarationSourceAuthority,
)
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_reproof import SourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceInsertion,
    SourceLineSpan,
    SourceSpanReplacement,
    SourceTargetEditor,
)
from nominal_refactor_advisor.codemod_target_selectors import (
    FindingEvidenceTargetSelector,
    SourceIndexTargetSelector,
    TargetSetExpressionSelector,
)
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.finding_recipe_actions import FindingRecipeActionKey
from nominal_refactor_advisor.lexical_bindings import (
    ImportBoundNameProjection,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
)
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.name_algebra import CLASS_NAME_ALGEBRA
from nominal_refactor_advisor.quoted_reference import (
    DeclaredInterpreter,
    ModuleReferenceScope,
)
from nominal_refactor_advisor.refactor_concepts import AutoRegisterStrategyFamilyConcept
from nominal_refactor_advisor.semantic_descent import (
    AuthorityClaim,
    SemanticAuthorityKind,
)
from nominal_refactor_advisor.source_index import (
    AstTargetDigest,
    AstTargetNode,
    AstTargetNodeKind,
)


def _unparsed_lines(statements: Iterable[ast.stmt]) -> tuple[str, ...]:
    return tuple(
        line
        for statement in statements
        for line in ast.unparse(statement).splitlines()
    )


def _bound_target_names_in_order(target: ast.AST) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()

    def collect(node: ast.AST) -> None:
        match node:
            case ast.Name(id=name):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
            case ast.Tuple(elts=elts) | ast.List(elts=elts):
                for element in elts:
                    collect(element)
            case ast.Starred(value=value):
                collect(value)

    collect(target)
    return tuple(names)


def _parameter_names_of_args(arguments: ast.arguments) -> tuple[str, ...]:
    names = [
        arg.arg
        for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    ]
    vararg = arguments.vararg
    if vararg is not None:
        names.append(vararg.arg)
    kwarg = arguments.kwarg
    if kwarg is not None:
        names.append(kwarg.arg)
    return tuple(names)


def _free_loaded_names(statements: Iterable[ast.stmt]) -> frozenset[str]:
    """Free variable loads within ``statements``, respecting inner Python scopes."""

    loaded: set[str] = set()

    def walk(node: ast.AST, masked: frozenset[str]) -> None:
        match node:
            case ast.Name(id=name, ctx=ast.Load()):
                if name not in masked:
                    loaded.add(name)
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                for child in (
                    *node.decorator_list,
                    *node.args.defaults,
                    *(item for item in node.args.kw_defaults if item is not None),
                ):
                    walk(child, masked)
                for arg in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ):
                    annotation = arg.annotation
                    if annotation is not None:
                        walk(annotation, masked)
                returns = node.returns
                if returns is not None:
                    walk(returns, masked)
                inner_masked = (
                    masked
                    | frozenset(_parameter_names_of_args(node.args))
                    | LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
                )
                for statement in node.body:
                    walk(statement, inner_masked)
            case ast.Lambda():
                for child in (
                    *node.args.defaults,
                    *(item for item in node.args.kw_defaults if item is not None),
                ):
                    walk(child, masked)
                inner_masked = masked | frozenset(_parameter_names_of_args(node.args))
                walk(node.body, inner_masked)
            case ast.ClassDef():
                for child in (*node.bases, *node.keywords, *node.decorator_list):
                    walk(child, masked)
                inner_masked = masked | LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                    node.body
                )
                for statement in node.body:
                    walk(statement, inner_masked)
            case ast.ListComp() | ast.SetComp() | ast.GeneratorExp() | ast.DictComp():
                walk(node.generators[0].iter, masked)
                comp_masked = set(masked)
                for index, generator in enumerate(node.generators):
                    if index > 0:
                        walk(generator.iter, frozenset(comp_masked))
                    comp_masked.update(_bound_target_names_in_order(generator.target))
                    for condition in generator.ifs:
                        walk(condition, frozenset(comp_masked))
                frozen_comp = frozenset(comp_masked)
                if isinstance(node, ast.DictComp):
                    walk(node.key, frozen_comp)
                    walk(node.value, frozen_comp)
                else:
                    walk(node.elt, frozen_comp)
            case ast.ExceptHandler():
                exc_type = node.type
                if exc_type is not None:
                    walk(exc_type, masked)
                exc_name = node.name
                handler_masked = (
                    masked | frozenset((exc_name,)) if exc_name else masked
                )
                for statement in node.body:
                    walk(statement, handler_masked)
            case _:
                for child in ast.iter_child_nodes(node):
                    walk(child, masked)

    empty = frozenset[str]()
    for statement in statements:
        walk(statement, empty)
    return frozenset(loaded)


def _merge_ordered(
    first: Iterable[str],
    second: Iterable[str],
    *,
    exclude: frozenset[str],
) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set(exclude)
    for name in (*first, *second):
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def _definitely_bound_names_in_order(
    statements: Iterable[ast.stmt],
) -> tuple[str, ...]:
    """Locals unconditionally bound on every fall-through path through ``statements``."""

    ordered: list[str] = []
    bound: set[str] = set()

    def record(names: Iterable[str]) -> None:
        for name in names:
            if name not in bound:
                bound.add(name)
                ordered.append(name)

    for statement in statements:
        match statement:
            case ast.Assign(targets=targets):
                for target in targets:
                    record(_bound_target_names_in_order(target))
            case ast.AnnAssign(target=target, value=value) if value is not None:
                record(_bound_target_names_in_order(target))
            case ast.With(items=items, body=body) | ast.AsyncWith(
                items=items, body=body
            ):
                for item in items:
                    optional_vars = item.optional_vars
                    if optional_vars is not None:
                        record(_bound_target_names_in_order(optional_vars))
                record(_definitely_bound_names_in_order(body))
            case ast.If(body=body, orelse=orelse):
                body_terminal = _DispatchExtractor.terminal(body)
                orelse_terminal = _DispatchExtractor.terminal(orelse)
                if body_terminal and not orelse_terminal:
                    record(_definitely_bound_names_in_order(orelse))
                elif orelse_terminal and not body_terminal:
                    record(_definitely_bound_names_in_order(body))
                elif body and orelse:
                    body_bound = _definitely_bound_names_in_order(body)
                    orelse_bound = frozenset(_definitely_bound_names_in_order(orelse))
                    record(name for name in body_bound if name in orelse_bound)
            case ast.Delete(targets=targets):
                removed = frozenset(
                    name
                    for target in targets
                    for name in _bound_target_names_in_order(target)
                )
                if removed:
                    bound.difference_update(removed)
                    ordered = [name for name in ordered if name not in removed]
    return tuple(ordered)


def _has_control_escape(statements: Iterable[ast.stmt]) -> bool:
    return any(
        isinstance(node, (ast.Return, ast.Break, ast.Continue, ast.Yield, ast.YieldFrom))
        for node in walk_function_body_nodes(statements)
    )


@dataclass(frozen=True)
class DispatchAxis:
    """Bound-name-rooted chain of plain attribute loads that selects a dispatch case."""

    root_name: str
    attribute_path: tuple[str, ...]

    @classmethod
    def decoded(
        cls,
        expression: ast.expr,
        root_names: Iterable[str],
    ) -> DispatchAxis | None:
        attribute_path: list[str] = []
        current = expression
        while isinstance(current, ast.Attribute):
            attribute_path.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        allowed = (
            root_names
            if isinstance(root_names, frozenset)
            else frozenset(root_names)
        )
        if current.id not in allowed:
            return None
        return cls(current.id, tuple(reversed(attribute_path)))

    def selects(self, expression: ast.expr) -> bool:
        return self.decoded(expression, frozenset((self.root_name,))) == self

    @property
    def expression(self) -> ast.expr:
        expression: ast.expr = ast.Name(id=self.root_name, ctx=ast.Load())
        for attribute in self.attribute_path:
            expression = ast.Attribute(value=expression, attr=attribute, ctx=ast.Load())
        return expression

    @property
    def source(self) -> str:
        return ast.unparse(self.expression)


@dataclass(frozen=True)
class DispatchPolymorphismArm:
    """One branch body and every literal that selects it."""

    literals: tuple[ast.Constant, ...]
    statements: tuple[ast.stmt, ...]

    @property
    def cases(self) -> tuple[DispatchPolymorphismCase, ...]:
        return tuple(DispatchPolymorphismCase(literal, self) for literal in self.literals)


@dataclass(frozen=True)
class DispatchPolymorphismCase:
    """One literal dispatch case lifted into a concrete strategy class.

    The first literal of an arm owns the arm's body; every further literal
    of the same arm is a subclass of that leaf that only declares its key.
    """

    literal: ast.Constant
    arm: DispatchPolymorphismArm

    @property
    def registry_key(self) -> str | int | float:
        value = self.literal.value
        if not isinstance(value, str | int | float):
            raise ValueError(f"Unsupported dispatch registry key {value!r}")
        return value

    @property
    def literal_source(self) -> str:
        return ast.unparse(self.literal)

    def class_name_for(self, base_name: str) -> str:
        case_name = CLASS_NAME_ALGEBRA.pascal_identifier(str(self.registry_key))
        if not case_name or not case_name.isidentifier():
            digest = hashlib.blake2s(
                self.literal_source.encode("utf-8"),
                digest_size=3,
            ).hexdigest()
            case_name = f"Case{case_name or 'Value'}{digest}"
        return f"{case_name}{base_name}"


DispatchPolymorphismCases: TypeAlias = tuple[DispatchPolymorphismCase, ...]


@dataclass(frozen=True)
class DispatchPolymorphismExtraction(ABC):
    """AST-derived dispatch data for one mechanically convertible function."""

    arms: tuple[DispatchPolymorphismArm, ...]
    fallback_statements: tuple[ast.stmt, ...]
    replaced_statements: tuple[ast.stmt, ...]
    captured_locals: tuple[str, ...]

    @property
    def cases(self) -> DispatchPolymorphismCases:
        return tuple(case for arm in self.arms for case in arm.cases)

    def case_statement_lines(
        self,
        statements: tuple[ast.stmt, ...],
    ) -> tuple[str, ...]:
        return tuple(f"        {line}" for line in _unparsed_lines(statements))

    @abstractmethod
    def dispatch_call_lines(
        self,
        *,
        case_type_binding: str,
        base_name: str,
        axis_source: str,
        method_name: str,
        call_arguments: str,
    ) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class TerminalDispatchExtraction(DispatchPolymorphismExtraction):
    """Terminal dispatch where every arm and the fallback return or raise."""

    def dispatch_call_lines(
        self,
        *,
        case_type_binding: str,
        base_name: str,
        axis_source: str,
        method_name: str,
        call_arguments: str,
    ) -> tuple[str, ...]:
        fallback_lines = _unparsed_lines(self.fallback_statements)
        return (
            f"{case_type_binding} = {base_name}.__registry__.get({axis_source})",
            f"if {case_type_binding} is None:",
            *(f"    {line}" for line in fallback_lines),
            f"return {case_type_binding}().{method_name}({call_arguments})",
        )


@dataclass(frozen=True)
class VoidDispatchExtraction(DispatchPolymorphismExtraction):
    """Side-effect dispatch with no fallback and no local bindings escaping."""

    def dispatch_call_lines(
        self,
        *,
        case_type_binding: str,
        base_name: str,
        axis_source: str,
        method_name: str,
        call_arguments: str,
    ) -> tuple[str, ...]:
        return (
            f"{case_type_binding} = {base_name}.__registry__.get({axis_source})",
            f"if {case_type_binding} is not None:",
            f"    {case_type_binding}().{method_name}({call_arguments})",
        )


@dataclass(frozen=True)
class FallbackVoidDispatchExtraction(DispatchPolymorphismExtraction):
    """Side-effect dispatch with an inline fallback branch and no escaping locals."""

    def dispatch_call_lines(
        self,
        *,
        case_type_binding: str,
        base_name: str,
        axis_source: str,
        method_name: str,
        call_arguments: str,
    ) -> tuple[str, ...]:
        fallback_lines = _unparsed_lines(self.fallback_statements)
        return (
            f"{case_type_binding} = {base_name}.__registry__.get({axis_source})",
            f"if {case_type_binding} is None:",
            *(f"    {line}" for line in fallback_lines),
            "else:",
            f"    {case_type_binding}().{method_name}({call_arguments})",
        )


@dataclass(frozen=True)
class AssigningDispatchExtraction(DispatchPolymorphismExtraction, ABC):
    """Non-terminal dispatch whose arms return one or more reassigned local targets."""

    assigned_names: tuple[str, ...]

    @property
    def target_source(self) -> str:
        return ", ".join(self.assigned_names)

    @property
    def return_expression_source(self) -> str:
        if len(self.assigned_names) == 1:
            return self.assigned_names[0]
        return f"({self.target_source})"

    def case_statement_lines(
        self,
        statements: tuple[ast.stmt, ...],
    ) -> tuple[str, ...]:
        return (
            *super().case_statement_lines(statements),
            f"        return {self.return_expression_source}",
        )


@dataclass(frozen=True)
class GuardedAssigningDispatchExtraction(AssigningDispatchExtraction):
    """Assigning dispatch with no fallback, updating already-bound targets on hit."""

    def dispatch_call_lines(
        self,
        *,
        case_type_binding: str,
        base_name: str,
        axis_source: str,
        method_name: str,
        call_arguments: str,
    ) -> tuple[str, ...]:
        return (
            f"{case_type_binding} = {base_name}.__registry__.get({axis_source})",
            f"if {case_type_binding} is not None:",
            f"    {self.target_source} = {case_type_binding}().{method_name}({call_arguments})",
        )


@dataclass(frozen=True)
class FallbackAssigningDispatchExtraction(AssigningDispatchExtraction):
    """Assigning dispatch with an inline fallback binding the same targets."""

    def dispatch_call_lines(
        self,
        *,
        case_type_binding: str,
        base_name: str,
        axis_source: str,
        method_name: str,
        call_arguments: str,
    ) -> tuple[str, ...]:
        fallback_lines = _unparsed_lines(self.fallback_statements)
        return (
            f"{case_type_binding} = {base_name}.__registry__.get({axis_source})",
            f"if {case_type_binding} is None:",
            *(f"    {line}" for line in fallback_lines),
            "else:",
            f"    {self.target_source} = {case_type_binding}().{method_name}({call_arguments})",
        )


@dataclass(frozen=True)
class _DispatchScope:
    """One statement block (function top level or a top-level for loop) searched for a dispatch."""

    statements: tuple[ast.stmt, ...]
    enclosing_prefix_bound_order: tuple[str, ...]
    enclosing_all_prefix_bound: frozenset[str]
    enclosing_outside_statements: tuple[ast.stmt, ...]
    enclosing_outside_loads: frozenset[str]
    allows_terminal: bool


@dataclass(frozen=True, kw_only=True)
class _DispatchExtractor:
    """Strict recognizer for literal branch functions convertible to strategies.

    Supports terminal, void, and assigning literal dispatches rooted at a
    function parameter or a definitely-bound prefix local, either at function
    top level or inside a single top-level ``for`` loop.
    """

    node: ast.FunctionDef
    axis: DispatchAxis

    @classmethod
    def compared_expressions(cls, node: ast.FunctionDef) -> tuple[ast.expr, ...]:
        return tuple(
            expression
            for statement in node.body
            for expression in cls.compared_expressions_within(statement)
        )

    @staticmethod
    def compared_expressions_within(statement: ast.stmt) -> tuple[ast.expr, ...]:
        return tuple(
            expression
            for child in walk_function_body_nodes([statement])
            for expression in (
                (child.left, *child.comparators)
                if isinstance(child, ast.Compare)
                else (child.subject,)
                if isinstance(child, ast.Match)
                else ()
            )
        )

    def extract(self) -> DispatchPolymorphismExtraction | None:
        if self.unsupported_signature:
            return None
        for scope in self._candidate_scopes():
            extraction = self._extract_from_scope(scope)
            if extraction is not None:
                registry_keys = tuple(case.registry_key for case in extraction.cases)
                if len(registry_keys) >= 2 and len(frozenset(registry_keys)) == len(
                    registry_keys
                ):
                    return extraction
        return None

    def _candidate_scopes(self) -> tuple[_DispatchScope, ...]:
        body = self.executable_body
        scopes: list[_DispatchScope] = [
            _DispatchScope(
                statements=body,
                enclosing_prefix_bound_order=(),
                enclosing_all_prefix_bound=frozenset(),
                enclosing_outside_statements=(),
                enclosing_outside_loads=frozenset(),
                allows_terminal=True,
            )
        ]
        param_set = frozenset(self.parameter_names)
        for index, statement in enumerate(body):
            if not isinstance(statement, ast.For):
                continue
            before_loop = body[:index]
            after_loop = (*body[index + 1 :], *statement.orelse)
            loop_targets = _bound_target_names_in_order(statement.target)
            prefix_order = _merge_ordered(
                _definitely_bound_names_in_order(before_loop),
                loop_targets,
                exclude=param_set,
            )
            all_prefix_bound = (
                LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(before_loop)
                | frozenset(loop_targets)
            ) - param_set
            scopes.append(
                _DispatchScope(
                    statements=tuple(statement.body),
                    enclosing_prefix_bound_order=prefix_order,
                    enclosing_all_prefix_bound=all_prefix_bound,
                    enclosing_outside_statements=(*before_loop, *after_loop),
                    enclosing_outside_loads=_free_loaded_names(after_loop),
                    allows_terminal=False,
                )
            )
        return tuple(scopes)

    def _dispatch_index_in(self, statements: tuple[ast.stmt, ...]) -> int | None:
        for index, statement in enumerate(statements):
            if isinstance(statement, ast.If) and (
                self.test_literals(statement.test)
                or self.negated_test_literals(statement.test)
            ):
                return index
            if isinstance(statement, ast.Match) and self.axis.selects(
                statement.subject
            ):
                return index
        return None

    def _extract_from_scope(
        self,
        scope: _DispatchScope,
    ) -> DispatchPolymorphismExtraction | None:
        dispatch_index = self._dispatch_index_in(scope.statements)
        if dispatch_index is None:
            return None
        prefix_statements = scope.statements[:dispatch_index]
        if self.selects_axis_within(scope.enclosing_outside_statements):
            return None
        if self.selects_axis_within(prefix_statements):
            return None
        param_set = frozenset(self.parameter_names)
        prefix_bound_order = _merge_ordered(
            scope.enclosing_prefix_bound_order,
            _definitely_bound_names_in_order(prefix_statements),
            exclude=param_set,
        )
        prefix_definitely_bound = frozenset(prefix_bound_order)
        prefix_all_bound = scope.enclosing_all_prefix_bound | (
            LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(prefix_statements) - param_set
        )
        if (
            self.axis.root_name not in param_set
            and self.axis.root_name not in prefix_definitely_bound
        ):
            return None
        dispatch_body = scope.statements[dispatch_index:]
        if scope.allows_terminal:
            for extractor in (
                self._terminal_branch_extraction,
                self._terminal_match_extraction,
                self._terminal_sequential_guard_extraction,
            ):
                extraction = extractor(
                    dispatch_body,
                    prefix_bound_order,
                    prefix_definitely_bound,
                    prefix_all_bound,
                )
                if extraction is not None:
                    return extraction
        loop_prefix_loads = (
            frozenset[str]()
            if scope.allows_terminal
            else _free_loaded_names(prefix_statements)
        )
        outside_loads = scope.enclosing_outside_loads | loop_prefix_loads
        for non_terminal_extractor in (
            self._non_terminal_branch_extraction,
            self._non_terminal_match_extraction,
        ):
            extraction = non_terminal_extractor(
                dispatch_body,
                prefix_bound_order,
                prefix_definitely_bound,
                prefix_all_bound,
                outside_loads,
            )
            if extraction is not None:
                return extraction
        return None

    def selects_axis_within(self, statements: Iterable[ast.stmt]) -> bool:
        """Whether a residual statement still branches, matches, or literal-compares the axis."""

        for statement in statements:
            for child in walk_function_body_nodes([statement]):
                match child:
                    case ast.If(test=test):
                        for node in ast.walk(test):
                            if isinstance(node, ast.Compare) and any(
                                self.axis.selects(expression)
                                for expression in (node.left, *node.comparators)
                            ):
                                return True
                    case ast.Match(subject=subject):
                        subjects = (
                            subject.elts
                            if isinstance(subject, ast.Tuple)
                            else (subject,)
                        )
                        if any(self.axis.selects(item) for item in subjects):
                            return True
                    case ast.Compare():
                        if self.test_literals(child) or self.negated_test_literals(
                            child
                        ):
                            return True
                    case ast.Subscript(value=ast.Dict(), slice=slice_node):
                        if self.axis.selects(slice_node):
                            return True
                    case ast.Call(
                        func=ast.Attribute(value=ast.Dict(), attr="get"),
                        args=[first_arg, *_],
                    ):
                        if self.axis.selects(first_arg):
                            return True
        return False

    @property
    def unsupported_signature(self) -> bool:
        return bool(
            self.node.args.vararg
            or self.node.args.kwarg
            or any(
                isinstance(node, (ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal))
                for node in walk_function_body_nodes(self.node.body)
            )
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            parameter.arg
            for parameter in (
                *self.node.args.posonlyargs,
                *self.node.args.args,
                *self.node.args.kwonlyargs,
            )
        )

    @property
    def executable_body(self) -> tuple[ast.stmt, ...]:
        return tuple(statements_without_docstring(self.node.body))

    @cached_property
    def function_locals(self) -> frozenset[str]:
        return LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            self.executable_body
        ) - frozenset(self.parameter_names)

    def _captured_locals_for(
        self,
        arms: Iterable[DispatchPolymorphismArm],
        prefix_bound_order: tuple[str, ...],
    ) -> tuple[str, ...]:
        arm_loaded = frozenset[str]().union(
            *(_free_loaded_names(arm.statements) for arm in arms)
        )
        return tuple(name for name in prefix_bound_order if name in arm_loaded)

    def _terminal_extraction(
        self,
        arms: list[DispatchPolymorphismArm],
        fallback: tuple[ast.stmt, ...],
        body: tuple[ast.stmt, ...],
        prefix_bound_order: tuple[str, ...],
    ) -> TerminalDispatchExtraction:
        arm_tuple = tuple(arms)
        return TerminalDispatchExtraction(
            arms=arm_tuple,
            fallback_statements=fallback,
            replaced_statements=body,
            captured_locals=self._captured_locals_for(arm_tuple, prefix_bound_order),
        )

    def _terminal_branch_extraction(
        self,
        body: tuple[ast.stmt, ...],
        prefix_bound_order: tuple[str, ...],
        prefix_definitely_bound: frozenset[str],
        prefix_all_bound: frozenset[str],
    ) -> DispatchPolymorphismExtraction | None:
        if not body or not isinstance(body[0], ast.If):
            return None
        arms: list[DispatchPolymorphismArm] = []
        current = body[0]
        fallback: tuple[ast.stmt, ...] = body[1:]
        while True:
            literals = self.test_literals(current.test)
            statements = self.case_statements(
                current.body, prefix_definitely_bound, prefix_all_bound
            )
            if not literals or statements is None:
                return None
            arms.append(DispatchPolymorphismArm(literals, statements))
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                continue
            fallback = (*current.orelse, *fallback)
            break
        if not self.is_preservable_fallback(
            fallback, prefix_definitely_bound, prefix_all_bound
        ):
            return None
        return self._terminal_extraction(arms, fallback, body, prefix_bound_order)

    def _terminal_match_extraction(
        self,
        body: tuple[ast.stmt, ...],
        prefix_bound_order: tuple[str, ...],
        prefix_definitely_bound: frozenset[str],
        prefix_all_bound: frozenset[str],
    ) -> DispatchPolymorphismExtraction | None:
        if not body or not isinstance(body[0], ast.Match):
            return None
        match_node = body[0]
        if not self.axis.selects(match_node.subject):
            return None
        arms: list[DispatchPolymorphismArm] = []
        fallback: tuple[ast.stmt, ...] = body[1:]
        for index, match_case in enumerate(match_node.cases):
            if match_guards(match_case):
                return None
            if self.is_default_match_pattern(match_case.pattern):
                if index != len(match_node.cases) - 1 or fallback:
                    return None
                fallback = tuple(match_case.body)
                continue
            literals = self.pattern_literals(match_case.pattern)
            statements = self.case_statements(
                match_case.body, prefix_definitely_bound, prefix_all_bound
            )
            if not literals or statements is None:
                return None
            arms.append(DispatchPolymorphismArm(literals, statements))
        if not self.is_preservable_fallback(
            fallback, prefix_definitely_bound, prefix_all_bound
        ):
            return None
        return self._terminal_extraction(arms, fallback, body, prefix_bound_order)

    def _terminal_sequential_guard_extraction(
        self,
        body: tuple[ast.stmt, ...],
        prefix_bound_order: tuple[str, ...],
        prefix_definitely_bound: frozenset[str],
        prefix_all_bound: frozenset[str],
    ) -> DispatchPolymorphismExtraction | None:
        arms: list[DispatchPolymorphismArm] = []
        index = 0
        fallback: tuple[ast.stmt, ...] = ()
        consumed_negated_final = False
        while index < len(body):
            statement = body[index]
            if not isinstance(statement, ast.If) or statement.orelse:
                break
            literals = self.test_literals(statement.test)
            if literals:
                statements = self.case_statements(
                    statement.body, prefix_definitely_bound, prefix_all_bound
                )
                if statements is None:
                    return None
                arms.append(DispatchPolymorphismArm(literals, statements))
                index += 1
                continue
            negated_literals = self.negated_test_literals(statement.test)
            if negated_literals:
                tail = body[index + 1 :]
                negated_fallback = tuple(statement.body)
                tail_statements = self.case_statements(
                    list(tail), prefix_definitely_bound, prefix_all_bound
                )
                if (
                    tail_statements is None
                    or self.selects_axis_within(tail_statements)
                    or not self.is_preservable_fallback(
                        negated_fallback, prefix_definitely_bound, prefix_all_bound
                    )
                ):
                    return None
                arms.append(DispatchPolymorphismArm(negated_literals, tail_statements))
                fallback = negated_fallback
                consumed_negated_final = True
                break
            break
        if not consumed_negated_final:
            fallback = body[index:]
            if not arms or not self.is_preservable_fallback(
                fallback, prefix_definitely_bound, prefix_all_bound
            ):
                return None
        return self._terminal_extraction(arms, fallback, body, prefix_bound_order)

    def _non_terminal_branch_extraction(
        self,
        body: tuple[ast.stmt, ...],
        prefix_bound_order: tuple[str, ...],
        prefix_definitely_bound: frozenset[str],
        prefix_all_bound: frozenset[str],
        outside_loads: frozenset[str],
    ) -> DispatchPolymorphismExtraction | None:
        if not body or not isinstance(body[0], ast.If):
            return None
        dispatch_stmt = body[0]
        post_statements = body[1:]
        arms: list[DispatchPolymorphismArm] = []
        current = dispatch_stmt
        fallback: tuple[ast.stmt, ...] = ()
        while True:
            literals = self.test_literals(current.test)
            if not literals or not current.body:
                return None
            arms.append(DispatchPolymorphismArm(literals, tuple(current.body)))
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                continue
            fallback = tuple(current.orelse)
            break
        return self._finalize_non_terminal_extraction(
            tuple(arms),
            fallback,
            dispatch_stmt,
            post_statements,
            prefix_bound_order,
            prefix_definitely_bound,
            prefix_all_bound,
            outside_loads,
        )

    def _non_terminal_match_extraction(
        self,
        body: tuple[ast.stmt, ...],
        prefix_bound_order: tuple[str, ...],
        prefix_definitely_bound: frozenset[str],
        prefix_all_bound: frozenset[str],
        outside_loads: frozenset[str],
    ) -> DispatchPolymorphismExtraction | None:
        if not body or not isinstance(body[0], ast.Match):
            return None
        match_node = body[0]
        if not self.axis.selects(match_node.subject):
            return None
        post_statements = body[1:]
        arms: list[DispatchPolymorphismArm] = []
        fallback: tuple[ast.stmt, ...] = ()
        for index, match_case in enumerate(match_node.cases):
            if match_guards(match_case):
                return None
            if self.is_default_match_pattern(match_case.pattern):
                if index != len(match_node.cases) - 1:
                    return None
                fallback = tuple(match_case.body)
                continue
            literals = self.pattern_literals(match_case.pattern)
            if not literals or not match_case.body:
                return None
            arms.append(DispatchPolymorphismArm(literals, tuple(match_case.body)))
        if not arms:
            return None
        return self._finalize_non_terminal_extraction(
            tuple(arms),
            fallback,
            match_node,
            post_statements,
            prefix_bound_order,
            prefix_definitely_bound,
            prefix_all_bound,
            outside_loads,
        )

    def _finalize_non_terminal_extraction(
        self,
        arms: tuple[DispatchPolymorphismArm, ...],
        fallback: tuple[ast.stmt, ...],
        dispatch_stmt: ast.stmt,
        post_statements: tuple[ast.stmt, ...],
        prefix_bound_order: tuple[str, ...],
        prefix_definitely_bound: frozenset[str],
        prefix_all_bound: frozenset[str],
        outside_loads: frozenset[str],
    ) -> DispatchPolymorphismExtraction | None:
        if self.selects_axis_within(post_statements):
            return None
        if fallback and (
            _has_control_escape(fallback)
            or self.selects_axis_within(fallback)
            or not self.preserves_bindings(
                fallback, prefix_definitely_bound, prefix_all_bound
            )
        ):
            return None
        for arm in arms:
            if _has_control_escape(arm.statements) or not self.preserves_bindings(
                arm.statements, prefix_definitely_bound, prefix_all_bound
            ):
                return None
        post_loaded = _free_loaded_names(post_statements) | outside_loads
        arm_bound = frozenset[str]().union(
            *(
                LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(arm.statements)
                for arm in arms
            )
        )
        escaped = arm_bound & post_loaded
        captured_locals = self._captured_locals_for(arms, prefix_bound_order)
        if not escaped:
            if fallback:
                return FallbackVoidDispatchExtraction(
                    arms=arms,
                    fallback_statements=fallback,
                    replaced_statements=(dispatch_stmt,),
                    captured_locals=captured_locals,
                )
            return VoidDispatchExtraction(
                arms=arms,
                fallback_statements=(),
                replaced_statements=(dispatch_stmt,),
                captured_locals=captured_locals,
            )
        first_arm_order = _definitely_bound_names_in_order(arms[0].statements)
        if not escaped <= frozenset(first_arm_order):
            return None
        assigned_names = tuple(name for name in first_arm_order if name in escaped)
        if any(
            not escaped <= frozenset(_definitely_bound_names_in_order(arm.statements))
            for arm in arms[1:]
        ):
            return None
        if fallback:
            if not escaped <= frozenset(_definitely_bound_names_in_order(fallback)):
                return None
            return FallbackAssigningDispatchExtraction(
                arms=arms,
                fallback_statements=fallback,
                replaced_statements=(dispatch_stmt,),
                captured_locals=captured_locals,
                assigned_names=assigned_names,
            )
        already_bound = frozenset(self.parameter_names) | prefix_definitely_bound
        if not escaped <= already_bound:
            return None
        return GuardedAssigningDispatchExtraction(
            arms=arms,
            fallback_statements=(),
            replaced_statements=(dispatch_stmt,),
            captured_locals=captured_locals,
            assigned_names=assigned_names,
        )

    def test_literals(self, test: ast.expr) -> tuple[ast.Constant, ...]:
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            return ()
        operator = test.ops[0]
        comparator = test.comparators[0]
        sides = ((test.left, comparator, True), (comparator, test.left, False))
        for subject, candidate, allow_collection in sides:
            literals = self.dispatch_literals_for_side(
                subject,
                candidate,
                operator,
                allow_collection=allow_collection,
            )
            if literals:
                return literals
        return ()

    def negated_test_literals(self, test: ast.expr) -> tuple[ast.Constant, ...]:
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            return ()
        operator = test.ops[0]
        comparator = test.comparators[0]
        sides = ((test.left, comparator, True), (comparator, test.left, False))
        for subject, candidate, allow_collection in sides:
            if not self.axis.selects(subject):
                continue
            if (
                isinstance(operator, ast.NotEq)
                and isinstance(candidate, ast.Constant)
                and self.is_literal(candidate)
            ):
                return (candidate,)
            if allow_collection and isinstance(operator, ast.NotIn):
                return self.collection_literals(candidate)
        return ()

    def dispatch_literals_for_side(
        self,
        subject: ast.expr,
        candidate: ast.expr,
        operator: ast.cmpop,
        *,
        allow_collection: bool,
    ) -> tuple[ast.Constant, ...]:
        if not self.axis.selects(subject):
            return ()
        if (
            isinstance(operator, ast.Eq)
            and isinstance(candidate, ast.Constant)
            and self.is_literal(candidate)
        ):
            return (candidate,)
        if allow_collection and isinstance(operator, ast.In):
            return self.collection_literals(candidate)
        return ()

    def pattern_literals(self, pattern: ast.pattern) -> tuple[ast.Constant, ...]:
        if (
            isinstance(pattern, ast.MatchValue)
            and isinstance(pattern.value, ast.Constant)
            and self.is_literal(pattern.value)
        ):
            return (pattern.value,)
        if isinstance(pattern, ast.MatchOr):
            return tuple(
                literal
                for child_pattern in pattern.patterns
                for literal in self.pattern_literals(child_pattern)
            )
        return ()

    @staticmethod
    def collection_literals(node: ast.expr) -> tuple[ast.Constant, ...]:
        if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return ()
        literals = tuple(
            element for element in node.elts if isinstance(element, ast.Constant)
        )
        if not all(
            _DispatchExtractor.is_literal(element) for element in node.elts
        ) or len(literals) != len(node.elts):
            return ()
        return literals

    def case_statements(
        self,
        statements: list[ast.stmt],
        prefix_definitely_bound: frozenset[str],
        prefix_all_bound: frozenset[str],
    ) -> tuple[ast.stmt, ...] | None:
        """Statements that execute unchanged as one strategy leaf's body."""

        if not self.terminal(statements) or not self.preserves_bindings(
            statements, prefix_definitely_bound, prefix_all_bound
        ):
            return None
        return tuple(statements)

    def preserves_bindings(
        self,
        statements: Iterable[ast.stmt],
        prefix_definitely_bound: frozenset[str],
        prefix_all_bound: frozenset[str],
    ) -> bool:
        """Leave the axis root bound and read only parameters, own locals, or definitely-bound prefix locals."""

        statements = tuple(statements)
        bound_here = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(statements)
        loaded = _free_loaded_names(statements)
        conditionally_bound_prefix = prefix_all_bound - prefix_definitely_bound
        return (
            self.axis.root_name not in bound_here
            and not (loaded & conditionally_bound_prefix)
            and not (
                loaded
                & (self.function_locals - bound_here - prefix_definitely_bound)
            )
        )

    @staticmethod
    def terminal(statements: Iterable[ast.stmt]) -> bool:
        """Every path through the statements ends in a return or a raise."""

        statements = tuple(statements)
        if not statements:
            return False
        last = statements[-1]
        if isinstance(last, ast.If):
            return _DispatchExtractor.terminal(
                last.body
            ) and _DispatchExtractor.terminal(last.orelse)
        return isinstance(last, (ast.Return, ast.Raise))

    def is_preservable_fallback(
        self,
        statements: tuple[ast.stmt, ...],
        prefix_definitely_bound: frozenset[str],
        prefix_all_bound: frozenset[str],
    ) -> bool:
        return (
            self.terminal(statements)
            and self.preserves_bindings(
                statements, prefix_definitely_bound, prefix_all_bound
            )
            and not self.selects_axis_within(statements)
        )

    @staticmethod
    def is_default_match_pattern(pattern: ast.pattern) -> bool:
        return isinstance(pattern, ast.MatchAs) and pattern.name is None

    @staticmethod
    def is_literal(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (str, int, float))
            and not isinstance(node.value, bool)
        )


@dataclass(frozen=True, kw_only=True)
class DispatchPolymorphismFunction:
    """One proved mechanically convertible literal dispatch function."""

    node: ast.FunctionDef
    axis: DispatchAxis
    extraction: DispatchPolymorphismExtraction

    @classmethod
    def derived_from_function(
        cls,
        node: ast.FunctionDef,
    ) -> tuple[Self, ...]:
        """Recover supported literal dispatches from one function."""

        executable_body = tuple(statements_without_docstring(node.body))
        parameter_names = frozenset(_parameter_names_of_args(node.args))
        root_names = parameter_names | LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            executable_body
        )
        axes = dict.fromkeys(
            axis
            for expression in _DispatchExtractor.compared_expressions(node)
            if (axis := DispatchAxis.decoded(expression, root_names)) is not None
        )
        candidates: list[Self] = []
        for axis in axes:
            extracted = _DispatchExtractor(node=node, axis=axis).extract()
            if extracted is not None:
                candidates.append(cls(node=node, axis=axis, extraction=extracted))
        return tuple(candidates)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            parameter.arg
            for parameter in (
                *self.node.args.posonlyargs,
                *self.node.args.args,
                *self.node.args.kwonlyargs,
            )
        )


@dataclass(frozen=True)
class DispatchPolymorphismHost(ABC, metaclass=AutoRegisterMeta):
    """Lexical owner of one dispatch target and of its strategy family's position."""

    __registry__: ClassVar[
        dict[AstTargetNodeKind, type[DispatchPolymorphismHost]]
    ] = {}
    __registry_key__ = "target_kind"
    __skip_if_no_key__ = True
    target_kind: ClassVar[AstTargetNodeKind | None] = None

    node: ast.FunctionDef

    @classmethod
    def for_target(
        cls,
        target_digest: AstTargetDigest,
        node: AstTargetNode,
        module: ast.Module,
    ) -> DispatchPolymorphismHost:
        """The host owning a target, or a ValueError naming the refused premise."""

        host_type = cls.__registry__.get(target_digest.node_kind)
        if host_type is None:
            raise ValueError(
                "dispatch_to_polymorphism rewrites functions and methods, not "
                f"{target_digest.node_kind.value} target {target_digest.qualname!r}"
            )
        if not isinstance(node, ast.FunctionDef):
            raise ValueError(
                "dispatch_to_polymorphism requires a synchronous function target; "
                f"{target_digest.qualname!r} is not one"
            )
        return host_type.resolved(target_digest, node, module)

    @classmethod
    @abstractmethod
    def resolved(
        cls,
        target_digest: AstTargetDigest,
        node: ast.FunctionDef,
        module: ast.Module,
    ) -> DispatchPolymorphismHost:
        raise NotImplementedError

    @property
    @abstractmethod
    def insertion_anchor(self) -> ast.stmt:
        """Top-level statement the strategy family is declared immediately before."""

        raise NotImplementedError

    @property
    @abstractmethod
    def strategy_name_source(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class ModuleFunctionDispatchHost(DispatchPolymorphismHost):
    """A module function whose family is declared just above it."""

    target_kind = AstTargetNodeKind.FUNCTION

    @classmethod
    def resolved(
        cls,
        target_digest: AstTargetDigest,
        node: ast.FunctionDef,
        module: ast.Module,
    ) -> ModuleFunctionDispatchHost:
        del target_digest, module
        return cls(node)

    @property
    def insertion_anchor(self) -> ast.stmt:
        return self.node

    @property
    def strategy_name_source(self) -> str:
        return self.node.name


@dataclass(frozen=True)
class ClassMethodDispatchHost(DispatchPolymorphismHost):
    """A method of a top-level class whose family precedes the class.

    The case bodies leave the class body, so the method may not depend on
    class-body compilation: only standard descriptor decorators
    (``staticmethod``, ``classmethod``, ``property``, ``cached_property``) that
    leave the body semantics unchanged are allowed, with no ``__class__`` cell
    (which any ``super`` or ``__class__`` reference creates) and no private
    name that the class would mangle.
    """

    target_kind = AstTargetNodeKind.METHOD
    class_cell_names: ClassVar[frozenset[str]] = frozenset({"super", "__class__"})
    builtin_descriptor_decorators: ClassVar[frozenset[str]] = frozenset(
        {"staticmethod", "classmethod", "property"}
    )

    owner: ast.ClassDef

    @classmethod
    def resolved(
        cls,
        target_digest: AstTargetDigest,
        node: ast.FunctionDef,
        module: ast.Module,
    ) -> ClassMethodDispatchHost:
        owners = tuple(
            statement
            for statement in module.body
            if isinstance(statement, ast.ClassDef)
            and any(member is node for member in statement.body)
        )
        if len(owners) != 1:
            raise ValueError(
                f"method target {target_digest.qualname!r} is not declared directly "
                "on a top-level module class"
            )
        if node.decorator_list and not cls._all_safe_descriptor_decorators(
            node.decorator_list, module
        ):
            raise ValueError(
                f"method target {target_digest.qualname!r} is decorated; only plain "
                "or standard descriptor methods lift into a module strategy family"
            )
        identifiers = cls.identifiers(node)
        if identifiers & cls.class_cell_names:
            raise ValueError(
                f"method target {target_digest.qualname!r} uses the class cell "
                "(super or __class__), which a module strategy leaf cannot bind"
            )
        mangled = sorted_tuple(
            identifier
            for identifier in identifiers
            if identifier.startswith("__") and not identifier.endswith("__")
        )
        if mangled:
            raise ValueError(
                f"method target {target_digest.qualname!r} uses class-mangled names "
                f"{mangled!r}, which change meaning outside the class body"
            )
        return cls(node, owners[0])

    @classmethod
    def _all_safe_descriptor_decorators(
        cls,
        decorators: list[ast.expr],
        module: ast.Module,
    ) -> bool:
        imported_sources: dict[str, str] = {}
        non_import_bound: set[str] = set()
        for statement in module.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                if any(alias.name == "*" for alias in statement.names):
                    return False
                for name, source in ImportBoundNameProjection(statement).name_sources():
                    imported_sources[name] = source
            else:
                non_import_bound.update(
                    LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names((statement,))
                )
        for decorator in decorators:
            match decorator:
                case ast.Name(id=name) if name in cls.builtin_descriptor_decorators:
                    if name in imported_sources or name in non_import_bound:
                        return False
                case ast.Name(id=name):
                    if (
                        name in non_import_bound
                        or imported_sources.get(name)
                        != "from functools import cached_property\n"
                    ):
                        return False
                case ast.Attribute(
                    value=ast.Name(id=module_name),
                    attr="cached_property",
                ):
                    if (
                        module_name in non_import_bound
                        or imported_sources.get(module_name) != "import functools\n"
                    ):
                        return False
                case _:
                    return False
        return True

    @staticmethod
    def identifiers(node: ast.FunctionDef) -> frozenset[str]:
        """Every identifier the compiler resolves or mangles inside the method.

        Call keyword names are left out: the compiler does not mangle them.
        """

        names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
            elif isinstance(child, ast.Attribute):
                names.add(child.attr)
            elif isinstance(child, ast.arg):
                names.add(child.arg)
            elif isinstance(child, ast.alias):
                names.add(child.asname or child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
        return frozenset(names)

    @property
    def insertion_anchor(self) -> ast.stmt:
        return self.owner

    @property
    def strategy_name_source(self) -> str:
        return f"{self.owner.name}_{self.node.name}"


@dataclass(frozen=True, kw_only=True)
class DispatchPolymorphismSource:
    """Render an extracted dispatch family and replacement function body."""

    case_key_attribute: ClassVar[str] = "case"
    method_name: ClassVar[str] = "apply"
    support_import_sources: ClassVar[tuple[str, ...]] = (
        "from abc import ABC, abstractmethod\n",
        "from typing import ClassVar\n",
        "from metaclass_registry import AutoRegisterMeta\n",
    )
    host: DispatchPolymorphismHost
    dispatch_function: DispatchPolymorphismFunction

    @classmethod
    def from_host(
        cls,
        host: DispatchPolymorphismHost,
    ) -> DispatchPolymorphismSource | None:
        candidates = DispatchPolymorphismFunction.derived_from_function(host.node)
        if len(candidates) != 1:
            return None
        return cls(host=host, dispatch_function=candidates[0])

    @property
    def extraction(self) -> DispatchPolymorphismExtraction:
        return self.dispatch_function.extraction

    @property
    def base_name(self) -> str:
        return dispatch_strategy_base_name(self.host.strategy_name_source)

    @cached_property
    def class_names(self) -> tuple[str, ...]:
        return (
            self.base_name,
            *(case.class_name_for(self.base_name) for case in self.extraction.cases),
        )

    @property
    def call_parameter_names(self) -> tuple[str, ...]:
        return (
            *self.dispatch_function.parameter_names,
            *self.extraction.captured_locals,
        )

    @property
    def apply_signature(self) -> str:
        parameters = ", ".join(
            (
                self.generated_binding_name("_dispatch_strategy"),
                *self.call_parameter_names,
            )
        )
        return f"def {self.method_name}({parameters})"

    @property
    def apply_call_arguments(self) -> str:
        return ", ".join(self.call_parameter_names)

    def dispatch_call_lines(self) -> tuple[str, ...]:
        return self.extraction.dispatch_call_lines(
            case_type_binding=self.generated_binding_name("_dispatch_case_type"),
            base_name=self.base_name,
            axis_source=self.dispatch_function.axis.source,
            method_name=self.method_name,
            call_arguments=self.apply_call_arguments,
        )

    @cached_property
    def source_names(self) -> frozenset[str]:
        return frozenset(
            node.id
            for node in walk_function_body_nodes(self.dispatch_function.node.body)
            if isinstance(node, ast.Name)
        ) | frozenset(self.dispatch_function.parameter_names)

    def generated_binding_name(self, preferred_name: str) -> str:
        candidate = preferred_name
        suffix = 2
        while candidate in self.source_names:
            candidate = f"{preferred_name}_{suffix}"
            suffix += 1
        return candidate

    def support_binding_conflicts(self, module: ast.Module) -> tuple[str, ...]:
        required_sources = {
            name: source
            for import_source in self.support_import_sources
            for statement in ast.parse(import_source).body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
            for name, source in ImportBoundNameProjection(statement).name_sources()
        }
        conflicts: set[str] = set()
        for statement in module.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                if any(alias.name == "*" for alias in statement.names):
                    conflicts.update(required_sources)
                    continue
                bound_sources = dict(
                    ImportBoundNameProjection(statement).name_sources()
                )
                conflicts.update(
                    name
                    for name, source in bound_sources.items()
                    if name in required_sources and source != required_sources[name]
                )
                continue
            conflicts.update(
                required_sources.keys()
                & LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names((statement,))
            )
        return sorted_tuple(conflicts)

    def family_source(self, *, self_reference_resolves: bool) -> str:
        """The strategy family; the base names itself unquoted where that resolves."""

        return "\n\n\n".join(
            (
                self.base_source(self_reference_resolves=self_reference_resolves),
                *(self.case_source(case) for case in self.extraction.cases),
            )
        )

    def base_source(self, *, self_reference_resolves: bool) -> str:
        base_reference = (
            self.base_name if self_reference_resolves else f'"{self.base_name}"'
        )
        return "\n".join(
            (
                f"class {self.base_name}(ABC, metaclass=AutoRegisterMeta):",
                (
                    "    __registry__: ClassVar[dict[object, "
                    f"type[{base_reference}]]] = {{}}"
                ),
                f'    __registry_key__ = "{self.case_key_attribute}"',
                "    __skip_if_no_key__ = True",
                f"    {self.case_key_attribute}: ClassVar[object] = None",
                "",
                "    @abstractmethod",
                f"    {self.apply_signature}:",
                "        raise NotImplementedError",
            )
        )

    def case_source(self, dispatch_case: DispatchPolymorphismCase) -> str:
        arm_leaf = dispatch_case.arm.cases[0]
        header = (
            f"    {self.case_key_attribute} = {dispatch_case.literal_source}",
        )
        if dispatch_case != arm_leaf:
            return "\n".join(
                (
                    f"class {dispatch_case.class_name_for(self.base_name)}"
                    f"({arm_leaf.class_name_for(self.base_name)}):",
                    *header,
                )
            )
        return "\n".join(
            (
                f"class {dispatch_case.class_name_for(self.base_name)}({self.base_name}):",
                *header,
                "",
                f"    {self.apply_signature}:",
                *self.extraction.case_statement_lines(dispatch_case.arm.statements),
            )
        )


@dataclass(frozen=True, kw_only=True)
class DispatchToPolymorphismOperation(SourceReprovedOperation):
    """Re-derive one function's closed dispatch as strategy subclasses."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        target_identifier, target_digest, node = self.target_node_from_context(snapshot)
        return self.source_edits_for_target_node(
            snapshot,
            target_identifier,
            target_digest,
            node,
        )

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        _target_identifier, target_digest, node = self.target_node_from_context(context)
        source = self.required_source(context, target_digest, node)
        return (
            AuthorityClaim(
                claimed_symbol=source.base_name,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                file_path=target_digest.file_path,
                qualname=source.base_name,
            ),
        )

    def current_source_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        _target_identifier, target_digest, node = self.target_node_from_context(context)
        source = self.required_source(context, target_digest, node)
        source_file = context.source_index.file_by_id[target_digest.file_id]
        return (
            ArchitectureGuardRule(
                rule_id=f"{source.base_name}-declaration-owned-dispatch",
                constraints=(
                    ForbiddenDispatchArchitectureGuardConstraint(
                        (source.dispatch_function.axis.source,)
                    ),
                ),
                scopes=(
                    ArchitectureGuardTargetScope(
                        file_path=(
                            source_file.module_path_identity.declared_source_relative_path.as_posix()
                        ),
                        target_qualname=target_digest.qualname,
                    ),
                ),
                reason="dispatch cases execute on the generated nominal leaves",
            ),
        )

    def source_edits_for_target_node(
        self,
        context: CodemodSelectorContext,
        target_identifier: str,
        target_digest: AstTargetDigest,
        node: AstTargetNode,
    ) -> tuple[NominalSourceEdit, ...]:
        del target_identifier
        source = self.required_source(context, target_digest, node)
        support_conflicts = source.support_binding_conflicts(
            context.module_nodes_by_file_path[target_digest.file_path]
        )
        if support_conflicts:
            raise ValueError(
                "Dispatch support names already have incompatible bindings: "
                f"{support_conflicts!r}"
            )
        return (
            *self.import_mutations(
                context,
                target_digest.file_path,
                source,
            ),
            self.family_insertion_replacement(
                context,
                target_digest,
                source,
            ),
            self.function_body_replacement(
                target_digest,
                source,
                context.sources_by_file_path,
            ),
        )

    @staticmethod
    def required_source(
        context: CodemodSelectorContext,
        target_digest: AstTargetDigest,
        node: AstTargetNode,
    ) -> DispatchPolymorphismSource:
        host = DispatchPolymorphismHost.for_target(
            target_digest,
            node,
            context.module_nodes_by_file_path[target_digest.file_path],
        )
        source = DispatchPolymorphismSource.from_host(host)
        if source is None:
            raise ValueError(
                f"Target {target_digest.qualname!r} is not a supported literal dispatch"
            )
        return source

    def import_mutations(
        self,
        context: CodemodSelectorContext,
        source_path: str,
        source: DispatchPolymorphismSource,
    ) -> tuple[ModuleImportMutation, ...]:
        return tuple(
            replacement
            for import_source in source.support_import_sources
            for replacement in EnsureImportOperation(
                target=SourceRewriteTarget(file_path=source_path),
                import_source=import_source,
                rationale=self.rationale_text("Import dispatch strategy support."),
            ).source_edits(context)
        )

    def family_insertion_replacement(
        self,
        context: CodemodSelectorContext,
        target_digest: AstTargetDigest,
        source: DispatchPolymorphismSource,
    ) -> SourceInsertion:
        conflicts = self.class_name_conflicts(
            context,
            target_digest,
            source.class_names,
        )
        if conflicts:
            raise ValueError(f"Dispatch class names already exist: {conflicts!r}")
        if len(frozenset(source.class_names)) != len(source.class_names):
            raise ValueError(
                f"Dispatch literals derive duplicate class names: {source.class_names!r}"
            )
        insertion_line = NamedDeclarationSourceAuthority(
            source.host.insertion_anchor,
            context.sources_by_file_path[target_digest.file_path],
        ).declaration_line_span.start_line
        self_reference_resolves = ModuleReferenceScope(
            context.module_nodes_by_file_path[target_digest.file_path]
        ).resolves(
            source.base_name,
            insertion_line,
            annotation=True,
            interpreter=DeclaredInterpreter.of(Path(target_digest.file_path)),
        )
        return SourceInsertion(
            file_path=target_digest.file_path,
            insertion_line=insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(
                source.family_source(self_reference_resolves=self_reference_resolves)
                + "\n\n\n"
            ),
            rationale=self.rationale_text(
                f"Insert dispatch strategy family {source.base_name!r}."
            ),
        )

    def function_body_replacement(
        self,
        target_digest: AstTargetDigest,
        source: DispatchPolymorphismSource,
        source_by_path: Mapping[str, str],
    ) -> SourceSpanReplacement:
        replaced_statements = source.extraction.replaced_statements
        if not replaced_statements:
            raise ValueError("dispatch function has no dispatch statements")
        body_start = replaced_statements[0].lineno
        body_end = replaced_statements[-1].end_lineno or replaced_statements[-1].lineno
        body_indent = SourceTargetEditor(
            source_by_path,
            target_digest,
        ).indentation_for_line(body_start)
        return SourceSpanReplacement(
            file_path=target_digest.file_path,
            span=SourceLineSpan(body_start, body_end),
            replacement_lines=tuple(
                f"{body_indent}{line}\n" for line in source.dispatch_call_lines()
            ),
            rationale=self.rationale_text(
                f"Replace literal dispatch in {target_digest.qualname!r}."
            ),
        )

    @staticmethod
    def class_name_conflicts(
        context: CodemodSelectorContext,
        target: AstTargetDigest,
        class_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        return sorted_tuple(
            frozenset(class_names)
            & LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                context.module_nodes_by_file_path[target.file_path].body
            )
        )


class LiteralDispatchFindingRecipeSynthesizer(
    FindingRecipeSynthesizer,
    AutoRegisterStrategyFamilyConcept,
    ABC,
):
    """Build strategy-family recipes for simple literal dispatch findings."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "literal dispatch synthesis requires a source selector context"
            )
        try:
            target = self.dispatch_target(finding, context)
        except ValueError as refusal:
            return self.rejected_evaluation(f"literal dispatch: {refusal}")
        return self.executable_evaluation(self.recipe_from_target(finding, target))

    def dispatch_target(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext,
    ) -> AstTargetDigest:
        """The one rewritable dispatch function or method, or a ValueError naming the premise that failed."""

        action_keys = self.action_keys_for_finding(finding)
        if len(action_keys) != 1:
            raise ValueError(
                f"the finding has {len(action_keys)} source action keys, not one"
            )
        (action_key,) = action_keys
        target_ids = TargetSetExpressionSelector(
            include=(FindingEvidenceTargetSelector.from_findings((finding,)),),
            require=(
                SourceIndexTargetSelector(
                    node_kinds=(AstTargetNodeKind.FUNCTION, AstTargetNodeKind.METHOD),
                    file_paths=(action_key.file_path,),
                ),
            ),
        ).target_ids(context)
        targets = tuple(
            context.source_index.target_by_id[target_id] for target_id in target_ids
        )
        if len(targets) != 1:
            raise ValueError(
                f"{len(targets)} function or method targets matched dispatch action "
                f"{action_key.subject_name!r}, not one"
            )
        (target,) = targets
        host = DispatchPolymorphismHost.for_target(
            target,
            context.ast_target_nodes_by_id[target.target_id],
            context.module_nodes_by_file_path[target.file_path],
        )
        if DispatchPolymorphismSource.from_host(host) is None:
            raise ValueError(
                f"{target.qualname!r} is not a mechanically supported "
                "literal dispatch with terminal case bodies; extract the "
                "closed-axis authority with the replacement scaffold before simulating"
            )
        return target

    def recipe_from_target(
        self,
        finding: RefactorFinding,
        target: AstTargetDigest,
    ) -> RefactorRecipe:
        return RefactorRecipe(
            recipe_id=f"{finding.stable_id}-dispatch-to-polymorphism",
            reason="Replace literal dispatch with AutoRegisterMeta strategy family.",
        ).with_operation(
            DispatchToPolymorphismOperation(
                target=SourceRewriteTarget(target_id=target.target_id),
                rationale="",
            )
        )

    def action_keys_for_finding(
        self,
        finding: RefactorFinding,
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = finding.primary_evidence
        if evidence is None:
            return ()
        if not finding.metrics.plan_dispatch_axis:
            return ()
        if not finding.metrics.plan_literal_cases:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            ((evidence.file_path, evidence.subject_symbol),),
        )


class NumericLiteralDispatchFindingRecipeSynthesizer(
    LiteralDispatchFindingRecipeSynthesizer
):
    """Build recipes for closed numeric-literal dispatch functions."""


def dispatch_strategy_base_name(function_name: str) -> str:
    function_suffix = CLASS_NAME_ALGEBRA.pascal_identifier(function_name)
    if function_suffix:
        return f"{function_suffix}DispatchCase"
    return "DispatchCase"
