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

from ..ast_tools import (
    statements_without_docstring,
    walk_function_body_nodes,
)
from ..codemod import (
    EnsureImportOperation,
    FindingRecipeSynthesizer,
)
from ..codemod_architecture_guards import (
    ArchitectureGuardRule,
    ArchitectureGuardTargetScope,
    ForbiddenDispatchArchitectureGuardConstraint,
)
from ..codemod_declaration_source import NamedDeclarationSourceAuthority
from ..codemod_imports import ModuleImportMutation
from ..codemod_reproof import SourceReprovedOperation
from ..codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from ..codemod_selection_context import CodemodSelectorContext
from ..codemod_selector_models import SourceRewriteTarget
from ..codemod_source_edits import (
    NominalSourceEdit,
    SourceInsertion,
    SourceLineSpan,
    SourceSpanReplacement,
    SourceTargetEditor,
)
from ..codemod_target_selectors import (
    FindingEvidenceTargetSelector,
    SourceIndexTargetSelector,
    TargetSetExpressionSelector,
)
from ..collection_algebra import sorted_tuple
from ..finding_recipe_actions import FindingRecipeActionKey
from ..lexical_bindings import (
    ImportBoundNameProjection,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
)
from ..models import RefactorFinding
from ..name_algebra import CLASS_NAME_ALGEBRA
from ..quoted_reference import (
    DeclaredInterpreter,
    ModuleReferenceScope,
)
from ..refactor_concepts import AutoRegisterStrategyFamilyConcept
from ..semantic_descent import (
    AuthorityClaim,
    SemanticAuthorityKind,
)
from ..source_index import (
    AstTargetDigest,
    AstTargetNode,
    AstTargetNodeKind,
)


@dataclass(frozen=True)
class DispatchAxis:
    """Parameter-rooted chain of plain attribute loads that selects a dispatch case."""

    parameter: ast.arg
    attribute_path: tuple[str, ...]

    @classmethod
    def decoded(
        cls,
        expression: ast.expr,
        parameters: Iterable[ast.arg],
    ) -> DispatchAxis | None:
        attribute_path: list[str] = []
        current = expression
        while isinstance(current, ast.Attribute):
            attribute_path.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        parameter = next(
            (parameter for parameter in parameters if parameter.arg == current.id),
            None,
        )
        if parameter is None:
            return None
        return cls(parameter, tuple(reversed(attribute_path)))

    def selects(self, expression: ast.expr) -> bool:
        return self.decoded(expression, (self.parameter,)) == self

    @property
    def root_name(self) -> str:
        return self.parameter.arg

    @property
    def expression(self) -> ast.expr:
        expression: ast.expr = ast.Name(id=self.parameter.arg, ctx=ast.Load())
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
class DispatchPolymorphismExtraction:
    """AST-derived dispatch data for one mechanically convertible function."""

    arms: tuple[DispatchPolymorphismArm, ...]
    fallback_statements: tuple[ast.stmt, ...]

    @property
    def cases(self) -> DispatchPolymorphismCases:
        return tuple(case for arm in self.arms for case in arm.cases)


@dataclass(frozen=True, kw_only=True)
class DispatchPolymorphismFunction:
    """Strict recognizer for literal branch functions convertible to strategies.

    The dispatch is the first executable statement after binding-free terminal
    guards. Each case body is terminal on every path and reads no function
    local bound outside itself, so it executes unchanged as a strategy leaf.
    """

    node: ast.FunctionDef
    axis: DispatchAxis

    @classmethod
    def derived_from_function(
        cls,
        node: ast.FunctionDef,
    ) -> tuple[Self, ...]:
        """Recover supported parameter-rooted dispatches from one function."""

        axes = dict.fromkeys(
            axis
            for expression in cls.compared_expressions(node)
            if (axis := DispatchAxis.decoded(expression, node.args.args)) is not None
        )
        return tuple(
            candidate
            for axis in axes
            if (candidate := cls(node=node, axis=axis)).extraction is not None
        )

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

    @cached_property
    def extraction(self) -> DispatchPolymorphismExtraction | None:
        if self.unsupported_signature:
            return None
        extraction = self.branch_extraction()
        if extraction is None:
            extraction = self.match_extraction()
        if extraction is None:
            extraction = self.sequential_guard_extraction()
        if extraction is None:
            return None
        registry_keys = tuple(case.registry_key for case in extraction.cases)
        if len(registry_keys) < 2 or len(frozenset(registry_keys)) != len(
            registry_keys
        ):
            return None
        return extraction

    def selects_axis_within(self, statements: Iterable[ast.stmt]) -> bool:
        """Whether a residual statement still compares or matches the axis."""

        return any(
            self.axis.selects(expression)
            for statement in statements
            for expression in self.compared_expressions_within(statement)
        )

    @property
    def unsupported_signature(self) -> bool:
        return bool(
            self.node.args.vararg
            or self.node.args.kwarg
            or self.node.args.kwonlyargs
            or self.node.args.posonlyargs
            or self.axis.parameter not in self.node.args.args
            or any(
                isinstance(node, (ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal))
                for node in walk_function_body_nodes(self.node.body)
            )
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.arg for parameter in self.node.args.args)

    @property
    def executable_body(self) -> tuple[ast.stmt, ...]:
        return tuple(statements_without_docstring(self.node.body))

    @cached_property
    def guard_statements(self) -> tuple[ast.stmt, ...]:
        """Leading terminal guards that bind nothing and stay in the dispatcher."""

        guards: list[ast.stmt] = []
        for statement in self.executable_body:
            if not (
                isinstance(statement, ast.If)
                and not statement.orelse
                and not self.test_literals(statement.test)
                and self.terminal(statement.body)
                and not LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names((statement,))
            ):
                break
            guards.append(statement)
        return tuple(guards)

    @property
    def dispatch_body(self) -> tuple[ast.stmt, ...]:
        return self.executable_body[len(self.guard_statements) :]

    @cached_property
    def function_locals(self) -> frozenset[str]:
        return LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            self.executable_body
        ) - frozenset(self.parameter_names)

    def branch_extraction(self) -> DispatchPolymorphismExtraction | None:
        body = self.dispatch_body
        if not body or not isinstance(body[0], ast.If):
            return None
        arms: list[DispatchPolymorphismArm] = []
        current = body[0]
        fallback: tuple[ast.stmt, ...] = body[1:]
        while True:
            literals = self.test_literals(current.test)
            statements = self.case_statements(current.body)
            if not literals or statements is None:
                return None
            arms.append(DispatchPolymorphismArm(literals, statements))
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                continue
            fallback = (*current.orelse, *fallback)
            break
        if not self.is_preservable_fallback(fallback):
            return None
        return DispatchPolymorphismExtraction(tuple(arms), fallback)

    def match_extraction(self) -> DispatchPolymorphismExtraction | None:
        body = self.dispatch_body
        if not body or not isinstance(body[0], ast.Match):
            return None
        match_node = body[0]
        if not self.axis.selects(match_node.subject):
            return None
        arms: list[DispatchPolymorphismArm] = []
        fallback: tuple[ast.stmt, ...] = body[1:]
        for index, match_case in enumerate(match_node.cases):
            if match_case.guard is not None:
                return None
            if self.is_default_match_pattern(match_case.pattern):
                if index != len(match_node.cases) - 1 or fallback:
                    return None
                fallback = tuple(match_case.body)
                continue
            literals = self.pattern_literals(match_case.pattern)
            statements = self.case_statements(match_case.body)
            if not literals or statements is None:
                return None
            arms.append(DispatchPolymorphismArm(literals, statements))
        if not self.is_preservable_fallback(fallback):
            return None
        return DispatchPolymorphismExtraction(tuple(arms), fallback)

    def sequential_guard_extraction(self) -> DispatchPolymorphismExtraction | None:
        arms: list[DispatchPolymorphismArm] = []
        body = self.dispatch_body
        index = 0
        while index < len(body):
            statement = body[index]
            if not isinstance(statement, ast.If) or statement.orelse:
                break
            literals = self.test_literals(statement.test)
            statements = self.case_statements(statement.body)
            if not literals or statements is None:
                return None
            arms.append(DispatchPolymorphismArm(literals, statements))
            index += 1
        fallback = body[index:]
        if not arms or not self.is_preservable_fallback(fallback):
            return None
        return DispatchPolymorphismExtraction(tuple(arms), fallback)

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
            DispatchPolymorphismFunction.is_literal(element) for element in node.elts
        ) or len(literals) != len(node.elts):
            return ()
        return literals

    def case_statements(
        self,
        statements: list[ast.stmt],
    ) -> tuple[ast.stmt, ...] | None:
        """Statements that execute unchanged as one strategy leaf's body."""

        if not self.terminal(statements) or not self.preserves_bindings(statements):
            return None
        return tuple(statements)

    def preserves_bindings(self, statements: Iterable[ast.stmt]) -> bool:
        """Leave the axis root bound and read no local bound outside the statements."""

        statements = tuple(statements)
        bound_here = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(statements)
        loaded = frozenset(
            node.id
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )
        return self.axis.root_name not in bound_here and not (
            loaded & (self.function_locals - bound_here)
        )

    @staticmethod
    def terminal(statements: Iterable[ast.stmt]) -> bool:
        """Every path through the statements ends in a return or a raise."""

        statements = tuple(statements)
        if not statements:
            return False
        last = statements[-1]
        if isinstance(last, ast.If):
            return DispatchPolymorphismFunction.terminal(
                last.body
            ) and DispatchPolymorphismFunction.terminal(last.orelse)
        return isinstance(last, (ast.Return, ast.Raise))

    def is_preservable_fallback(self, statements: tuple[ast.stmt, ...]) -> bool:
        return (
            self.terminal(statements)
            and self.preserves_bindings(statements)
            and not self.selects_axis_within(statements)
        )

    @staticmethod
    def is_default_match_pattern(pattern: ast.pattern) -> bool:
        return isinstance(pattern, ast.MatchAs) and pattern.name is None

    @staticmethod
    def is_literal(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(
            node.value,
            (str, int, float),
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
    """A plain instance method of a top-level class; its family precedes the class.

    The case bodies leave the class body, so the method may not depend on
    class-body compilation: no decorator, no ``__class__`` cell (which any
    ``super`` or ``__class__`` reference creates), and no private name that
    the class would mangle.
    """

    target_kind = AstTargetNodeKind.METHOD
    class_cell_names: ClassVar[frozenset[str]] = frozenset({"super", "__class__"})

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
        if node.decorator_list:
            raise ValueError(
                f"method target {target_digest.qualname!r} is decorated; only plain "
                "instance methods lift into a module strategy family"
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
        function = candidates[0]
        if function.extraction is None:
            return None
        return cls(host=host, dispatch_function=function)

    @property
    def extraction(self) -> DispatchPolymorphismExtraction:
        extraction = self.dispatch_function.extraction
        if extraction is None:
            raise ValueError("Dispatch source no longer has a supported extraction")
        return extraction

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
    def apply_signature(self) -> str:
        parameters = ", ".join(
            (
                self.generated_binding_name("_dispatch_strategy"),
                *self.dispatch_function.parameter_names,
            )
        )
        return f"def {self.method_name}({parameters})"

    @property
    def apply_call_arguments(self) -> str:
        return ", ".join(self.dispatch_function.parameter_names)

    def dispatch_call_lines(self) -> tuple[str, ...]:
        case_type_binding = self.generated_binding_name("_dispatch_case_type")
        fallback_lines = tuple(
            line
            for statement in self.extraction.fallback_statements
            for line in ast.unparse(statement).splitlines()
        )
        return (
            (
                f"{case_type_binding} = {self.base_name}.__registry__.get"
                f"({self.dispatch_function.axis.source})"
            ),
            f"if {case_type_binding} is None:",
            *(f"    {line}" for line in fallback_lines),
            (
                f"return {case_type_binding}().{self.method_name}"
                f"({self.apply_call_arguments})"
            ),
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
                *self.case_statement_lines(dispatch_case.arm.statements),
            )
        )

    @staticmethod
    def case_statement_lines(statements: tuple[ast.stmt, ...]) -> tuple[str, ...]:
        return tuple(
            f"        {line}"
            for statement in statements
            for line in ast.unparse(statement).splitlines()
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
        dispatch_body = source.dispatch_function.dispatch_body
        if not dispatch_body:
            raise ValueError("dispatch function has no dispatch statements")
        body_start = dispatch_body[0].lineno
        body_end = dispatch_body[-1].end_lineno or dispatch_body[-1].lineno
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
        if finding.metrics.plan_dispatch_axis is None:
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
