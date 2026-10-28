"""False distinctions: code that tells cases apart and then treats them the same.

A branch whose arms all run the same code distinguishes cases only in its
test. The site families here collect each shape per module; the detectors in
`detectors/_false_distinctions.py` name the fact each one restates. Each shape
also knows how to collapse itself to its one arm, refusing when the tests it
would delete could have an effect.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property
from itertools import groupby
from operator import attrgetter
from pathlib import Path
from types import EllipsisType
from typing import Any, cast

from .ast_projection import ExpressionReference
from .ast_tools import (
    BuiltinCallName,
    CollectedFamily,
    ParsedModule,
    PythonModuleRootParser,
    PythonSourcePathDiscovery,
    PythonSourcePathPolicy,
    walk_function_body_nodes,
)
from .codemod_source_edits import (
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .class_index import (
    BoundNominalReference,
    CompactClassFamilyIndex,
    CompactNominalBinding,
    CompactNominalBindingKind,
)
from .codemod_foreign_family_source import RewrittenLineFit
from .enum_semantics import PythonEnumBaseKind, PythonEnumDeclarationAuthority
from .observation_families import ShapeFamily
from .restated_fact_observations import (
    RestatingSiteFamily,
    RestatingSiteModule,
)
from .observation_shapes import RestatingSite

StatementList = Sequence[ast.stmt]
ScopeNode = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def same_code(left: Sequence[ast.AST], right: Sequence[ast.AST]) -> bool:
    """Whether two node sequences are the same code, positions aside."""

    return len(left) == len(right) and all(
        ast.dump(one) == ast.dump(other) for one, other in zip(left, right)
    )


def node_source(geometry: SourceTextGeometry, node: ast.AST) -> str:
    """The exact source text of one parsed node."""

    start, end = geometry.required_node_offsets(node)
    return geometry.source[start:end]


def evaluates_without_effect(node: ast.AST) -> bool:
    """Whether evaluating `node` only reads: names, constants, attributes, `is`,
    `and`/`or`/`not`, tuples and calls of the bare `isinstance`/`issubclass`.

    `==`, `in`, operators, subscripts and every other call can run user code,
    so a test using them is not one a collapse may delete. Attribute reads are
    taken as reads; a property with an effect is outside this decision.
    """

    match node:
        case ast.Name() | ast.Constant():
            return True
        case ast.Attribute(value=value):
            return evaluates_without_effect(value)
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            return all(isinstance(op, (ast.Is, ast.IsNot)) for op in ops) and all(
                map(evaluates_without_effect, (left, *comparators))
            )
        case ast.BoolOp(values=values) | ast.Tuple(elts=values):
            return all(map(evaluates_without_effect, values))
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return evaluates_without_effect(operand)
        case ast.Call(func=callee, args=arguments, keywords=[]) if any(
            name.is_callee(callee) for name in BuiltinCallName.type_test_names()
        ):
            return all(map(evaluates_without_effect, arguments))
    return False


def matches_without_effect(pattern: ast.pattern) -> bool:
    """Whether matching `pattern` only tests types and reads attributes.

    A value pattern compares with `==`; sequence and mapping patterns call
    `len` and `__getitem__`; each can run user code.
    """

    match pattern:
        case ast.MatchAs(pattern=None) | ast.MatchSingleton():
            return True
        case ast.MatchAs(pattern=ast.pattern() as inner):
            return matches_without_effect(inner)
        case ast.MatchOr(patterns=patterns):
            return all(map(matches_without_effect, patterns))
        case ast.MatchClass(cls=cls, patterns=patterns, kwd_patterns=keywords):
            return evaluates_without_effect(cls) and all(
                map(matches_without_effect, (*patterns, *keywords))
            )
    return False


def captures_names(pattern: ast.pattern) -> bool:
    """Whether matching `pattern` binds a name the arm can read."""

    for node in ast.walk(pattern):
        match node:
            case (
                ast.MatchAs(name=str())
                | ast.MatchStar(name=str())
                | ast.MatchMapping(rest=str())
            ):
                return True
    return False


@dataclass(frozen=True)
class Scope:
    """One scope's own code: its body walked without entering nested definitions."""

    body: list[ast.stmt]

    @cached_property
    def nodes(self) -> tuple[ast.AST, ...]:
        return walk_function_body_nodes(self.body)

    def ladder_heads(self) -> Iterator[ast.If]:
        """Each if/elif ladder once, by its first `if`."""

        continuations = {
            id(node.orelse[0])
            for node in self.nodes
            if isinstance(node, ast.If) and IfLadderArms.continues(node)
        }
        for node in self.nodes:
            match node:
                case ast.If() if id(node) not in continuations:
                    yield node

    def blocks(self) -> Iterator[StatementList]:
        """The body and every statement block nested in it, definitions aside."""

        yield self.body
        for node in self.nodes:
            match node:
                case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                    continue
            for _name, value in ast.iter_fields(node):
                if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                    yield value

    @classmethod
    def of_module(cls, module: ast.Module) -> Iterator[Scope]:
        """The module's own scope and every definition's, each once."""

        yield cls(module.body)
        for node in ast.walk(module):
            match node:
                case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                    yield cls(node.body)


@dataclass(frozen=True)
class IdenticalArms(ABC):
    """A branch every arm of which runs the same code, so its tests decide nothing."""

    node: ast.stmt | ast.expr

    @property
    @abstractmethod
    def tests(self) -> tuple[ast.expr, ...]:
        """Every expression the branch evaluates only to choose an arm."""

        raise NotImplementedError

    @abstractmethod
    def collapsed(self, geometry: SourceTextGeometry) -> SourceTextSpanReplacement:
        """The branch's source replaced by its one arm."""

        raise NotImplementedError

    def refusals(self) -> Iterator[str]:
        """Why deleting the tests could change behaviour, one reason per test."""

        for test in self.tests:
            if not evaluates_without_effect(test):
                yield f"evaluating `{ast.unparse(test)}` can have an effect"

    def collapse(self, geometry: SourceTextGeometry) -> SourceTextSpanReplacement:
        """The collapse, or a ValueError naming why it would change behaviour."""

        refusals = tuple(self.refusals())
        if refusals:
            raise ValueError(f"Branch at line {self.node.lineno}: {'; '.join(refusals)}")
        return self.collapsed(geometry)

    @classmethod
    def in_scope(cls, scope: Scope) -> Iterator[IdenticalArms]:
        """Every branch of one scope, nested definitions left to their own scope."""

        for head in scope.ladder_heads():
            yield from IfLadderArms.found(head)
        for node in scope.nodes:
            match node:
                case ast.IfExp(body=arm, orelse=other) if same_code([arm], [other]):
                    yield ConditionalExpressionArms(node)
                case ast.Match():
                    yield from MatchArms.found(node)
        for statements in scope.blocks():
            yield from FallThroughArms.found(statements)


@dataclass(frozen=True)
class StatementArms(IdenticalArms, ABC):
    """A branch statement that collapses to the statements of its one arm."""

    node: ast.If | ast.Match

    @property
    @abstractmethod
    def kept(self) -> StatementList:
        raise NotImplementedError

    def collapsed(self, geometry: SourceTextGeometry) -> SourceTextSpanReplacement:
        node, kept = self.node, self.kept
        start, end = geometry.line_span_offsets(node.lineno, node.end_lineno or node.lineno)
        indentation = geometry.lines[node.lineno - 1][: node.col_offset]
        if indentation.strip():
            raise ValueError(f"Branch at line {node.lineno} does not begin its own line")
        kept_start = geometry.required_node_offsets(kept[0])[0]
        kept_end = geometry.required_node_offsets(kept[-1])[1]
        if geometry.span_contains_comment(
            SourceTextSpan(start, kept_start)
        ) or geometry.span_contains_comment(SourceTextSpan(kept_end, end)):
            raise ValueError(f"Collapsing the branch at line {node.lineno} would drop a comment")
        literal_lines = geometry.literal_continuation_lines(node)
        if any(
            line in literal_lines
            for line in range(kept[0].lineno + 1, (kept[-1].end_lineno or 0) + 1)
        ):
            raise ValueError(
                f"The arm at line {kept[0].lineno} holds a multi-line literal whose text dedenting would change"
            )
        step = kept[0].col_offset - node.col_offset
        first, *rest = geometry.source[kept_start:kept_end].split("\n")
        if any(line[:step].strip() for line in rest):
            raise ValueError(f"The arm at line {kept[0].lineno} is not indented as a block")
        text = "\n".join(
            (indentation + first, *(line[step:] if line.strip() else "" for line in rest))
        )
        trailer = "\n" if geometry.source[start:end].endswith("\n") else ""
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start, end_offset=end, replacement_source=text + trailer
        )


@dataclass(frozen=True)
class IfLadderArms(StatementArms):
    """`if a: S elif b: S else: S`: every arm, the final `else` included, is S."""

    node: ast.If

    @staticmethod
    def continues(node: ast.If) -> bool:
        """Whether `node`'s else is one `if`: an `elif`, or `else:` holding only an `if`."""

        return len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)

    @property
    def ladder(self) -> tuple[ast.If, ...]:
        rungs = [self.node]
        while True:
            match rungs[-1].orelse:
                case [ast.If() as rung]:
                    rungs.append(rung)
                case _:
                    return tuple(rungs)

    @property
    def tests(self) -> tuple[ast.expr, ...]:
        return tuple(rung.test for rung in self.ladder)

    @property
    def kept(self) -> StatementList:
        return self.node.body

    @classmethod
    def found(cls, node: ast.If) -> Iterator[IfLadderArms]:
        arms = cls(node)
        final = arms.ladder[-1].orelse
        if final and all(same_code(rung.body, final) for rung in arms.ladder):
            yield arms


@dataclass(frozen=True)
class MatchArms(StatementArms):
    """A `match` whose cases all run the same code and whose last case is `case _:`.

    A pattern that binds a name is left out: the same text in each arm then
    reads a different binding, which is a distinction, not a false one.
    """

    node: ast.Match

    @property
    def tests(self) -> tuple[ast.expr, ...]:
        return (self.node.subject, *self.guards())

    @property
    def kept(self) -> StatementList:
        return self.node.cases[0].body

    def guards(self) -> Iterator[ast.expr]:
        for case in self.node.cases:
            match case:
                case ast.match_case(guard=ast.expr() as guard):
                    yield guard

    def refusals(self) -> Iterator[str]:
        for case in self.node.cases:
            if not matches_without_effect(case.pattern):
                yield f"matching `{ast.unparse(case.pattern)}` can have an effect"
        yield from super().refusals()

    @classmethod
    def found(cls, node: ast.Match) -> Iterator[MatchArms]:
        match node.cases[-1]:
            case ast.match_case(
                pattern=ast.MatchAs(pattern=None, name=None), guard=None
            ) if all(
                same_code(case.body, node.cases[0].body)
                and not captures_names(case.pattern)
                for case in node.cases
            ):
                yield cls(node)


@dataclass(frozen=True)
class ConditionalExpressionArms(IdenticalArms):
    """`x if test else x`, collapsed to `x`."""

    node: ast.IfExp

    @property
    def tests(self) -> tuple[ast.expr, ...]:
        return (self.node.test,)

    def collapsed(self, geometry: SourceTextGeometry) -> SourceTextSpanReplacement:
        start, end = geometry.required_node_offsets(self.node)
        kept_start, kept_end = geometry.required_node_offsets(self.node.body)
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise ValueError(
                f"Collapsing the conditional at line {self.node.lineno} would drop a comment"
            )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start,
            end_offset=end,
            replacement_source=geometry.source[kept_start:kept_end],
        )


@dataclass(frozen=True)
class FallThroughArms(IdenticalArms):
    """`if test: S` followed directly by the same S, where S leaves the block.

    The collapse deletes the `if`: the statement after it already runs S.
    """

    node: ast.If

    @property
    def tests(self) -> tuple[ast.expr, ...]:
        return (self.node.test,)

    def collapsed(self, geometry: SourceTextGeometry) -> SourceTextSpanReplacement:
        node = self.node
        start, end = geometry.line_span_offsets(node.lineno, node.end_lineno or node.lineno)
        if geometry.lines[node.lineno - 1][: node.col_offset].strip():
            raise ValueError(f"Branch at line {node.lineno} does not begin its own line")
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise ValueError(f"Deleting the branch at line {node.lineno} would drop a comment")
        if geometry.source[geometry.required_node_offsets(node)[1] : end].strip():
            raise ValueError(f"Branch at line {node.lineno} shares its last line")
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start, end_offset=end, replacement_source=""
        )

    @classmethod
    def found(cls, statements: StatementList) -> Iterator[FallThroughArms]:
        for statement, following in zip(statements, statements[1:]):
            match statement:
                case ast.If(
                    body=[ast.Return() | ast.Raise() | ast.Continue() | ast.Break()],
                    orelse=[],
                ) if same_code(statement.body, [following]):
                    yield cls(statement)


class IdenticalArmsSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """Branches whose every arm is the same code, one site per branch."""

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        index = parsed_module.syntax_index
        node_indices = {
            id(node): node_index
            for node_type in (ast.If, ast.IfExp, ast.Match)
            for node_index, node in index.indexed_nodes_of_type(node_type)
        }
        for arms in (
            arms
            for scope in Scope.of_module(parsed_module.module)
            for arms in IdenticalArms.in_scope(scope)
        ):
            yield cls.site(
                parsed_module,
                arms.node,
                index.lexical_symbol(node_indices[id(arms.node)]),
                f"line {arms.node.lineno}",
                tuple(ast.unparse(test) for test in arms.tests),
            )


@dataclass(frozen=True)
class TypeTestRung:
    """One rung `isinstance(subject, classes)` of an if/elif ladder."""

    branch: ast.If
    subject: ast.expr
    classes: tuple[ast.expr, ...]

    @classmethod
    def decoded(cls, branch: ast.If) -> Iterator[TypeTestRung]:
        match branch.test:
            case ast.Call(
                func=callee, args=[subject, ast.Tuple(elts=classes)], keywords=[]
            ) if BuiltinCallName.ISINSTANCE.is_callee(callee):
                yield cls(branch, subject, tuple(classes))
            case ast.Call(func=callee, args=[subject, single], keywords=[]) if (
                BuiltinCallName.ISINSTANCE.is_callee(callee)
            ):
                yield cls(branch, subject, (single,))

    @property
    def body(self) -> StatementList:
        return self.branch.body


@dataclass(frozen=True)
class UndistinguishedTypeArms(ABC):
    """Rungs of one `isinstance` ladder whose outcomes the code treats identically."""

    rungs: tuple[TypeTestRung, ...]

    @property
    @abstractmethod
    def merged(self) -> tuple[TypeTestRung, ...]:
        """The rungs whose type tests make no difference."""

        raise NotImplementedError

    @abstractmethod
    def collapsed(
        self, geometry: SourceTextGeometry, fit: RewrittenLineFit
    ) -> tuple[SourceTextSpanReplacement, ...]:
        raise NotImplementedError

    @property
    def subject(self) -> ast.expr:
        return self.rungs[0].subject

    @property
    def line(self) -> int:
        return self.merged[0].branch.lineno

    def refusals(self) -> Iterator[str]:
        for expression in (
            self.subject,
            *(tested for rung in self.merged for tested in rung.classes),
        ):
            if not evaluates_without_effect(expression):
                yield f"evaluating `{ast.unparse(expression)}` can have an effect"

    def collapse(
        self, geometry: SourceTextGeometry, fit: RewrittenLineFit
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """The merge, or a ValueError naming why it would change behaviour."""

        refusals = tuple(self.refusals())
        if refusals:
            raise ValueError(f"Type tests at line {self.line}: {'; '.join(refusals)}")
        return self.collapsed(geometry, fit)

    @staticmethod
    def deleted(
        geometry: SourceTextGeometry, rung: TypeTestRung
    ) -> SourceTextSpanReplacement:
        """The `elif` rung's lines removed, test and body."""

        line = rung.branch.lineno
        if not geometry.lines[line - 1].lstrip().startswith("elif "):
            raise ValueError(f"The rung at line {line} is not written as `elif`")
        start, end = geometry.line_span_offsets(line, rung.body[-1].end_lineno or line)
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise ValueError(f"Deleting the rung at line {line} would drop a comment")
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start, end_offset=end, replacement_source=""
        )

    @staticmethod
    def distinct_prefix(rungs: tuple[TypeTestRung, ...], final: StatementList) -> int:
        """How many leading rungs remain once trailing rungs equal to `else` are dropped."""

        distinct = len(rungs)
        for rung in reversed(rungs):
            if not (final and same_code(rung.body, final)):
                break
            distinct -= 1
        return distinct

    @staticmethod
    def body_of(item: tuple[int, TypeTestRung]) -> tuple[str, ...]:
        return tuple(ast.dump(statement) for statement in item[1].body)

    @classmethod
    def in_ladder(cls, node: ast.If) -> Iterator[UndistinguishedTypeArms]:
        ladder = IfLadderArms(node).ladder
        rungs = tuple(rung for branch in ladder for rung in TypeTestRung.decoded(branch))
        if (
            len(rungs) < 2
            or len(rungs) != len(ladder)
            or not all(same_code([rung.subject], [rungs[0].subject]) for rung in rungs)
        ):
            return
        distinct = cls.distinct_prefix(rungs, ladder[-1].orelse)
        if not distinct:
            return
        if distinct < len(rungs):
            yield TypeArmsLikeElse(rungs, distinct)
        for _body, group in groupby(enumerate(rungs[:distinct]), key=cls.body_of):
            indices = [index for index, _rung in group]
            if len(indices) > 1:
                yield AdjacentTypeArms(rungs, indices[0], indices[-1])

    @classmethod
    def in_scope(cls, scope: Scope) -> Iterator[UndistinguishedTypeArms]:
        for head in scope.ladder_heads():
            yield from cls.in_ladder(head)


@dataclass(frozen=True)
class AdjacentTypeArms(UndistinguishedTypeArms):
    """Adjacent rungs `isinstance(x, A): S` / `isinstance(x, B): S`, merged into one test.

    Only adjacent rungs merge: moving a later class test above an intervening
    rung would change which arm an instance of both classes takes.
    """

    first: int
    last: int

    @property
    def merged(self) -> tuple[TypeTestRung, ...]:
        return self.rungs[self.first : self.last + 1]

    def collapsed(
        self, geometry: SourceTextGeometry, fit: RewrittenLineFit
    ) -> tuple[SourceTextSpanReplacement, ...]:
        start, end = geometry.required_node_offsets(self.merged[0].branch.test)
        subject = node_source(geometry, self.subject)
        classes = ", ".join(
            node_source(geometry, tested)
            for rung in self.merged
            for tested in rung.classes
        )
        test = fit.fitted(f"isinstance({subject}, ({classes}))", start, end)
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start, end_offset=end, replacement_source=test
            ),
            *(self.deleted(geometry, rung) for rung in self.merged[1:]),
        )



@dataclass(frozen=True)
class TypeArmsLikeElse(UndistinguishedTypeArms):
    """Trailing rungs whose body is the `else` body: their tests change nothing."""

    first: int

    @property
    def merged(self) -> tuple[TypeTestRung, ...]:
        return self.rungs[self.first :]

    def collapsed(
        self, geometry: SourceTextGeometry, fit: RewrittenLineFit
    ) -> tuple[SourceTextSpanReplacement, ...]:
        del fit
        return tuple(self.deleted(geometry, rung) for rung in self.merged)


class UndistinguishedTypeArmsSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """`isinstance` ladders with rungs that run the same code, one site per group."""

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        index = parsed_module.syntax_index
        node_indices = {
            id(node): node_index
            for node_index, node in index.indexed_nodes_of_type(ast.If)
        }
        for arms in (
            arms
            for scope in Scope.of_module(parsed_module.module)
            for arms in UndistinguishedTypeArms.in_scope(scope)
        ):
            branch = arms.merged[0].branch
            yield cls.site(
                parsed_module,
                branch,
                index.lexical_symbol(node_indices[id(branch)]),
                f"line {branch.lineno}",
                tuple(ast.unparse(tested) for rung in arms.merged for tested in rung.classes),
            )


class ExceptionMatcherCall(StrEnum):
    """Test calls that pass when their block raises the class they are given:
    an `except` clause spelled as a call."""

    RAISES = "raises"
    ASSERT_RAISES = "assertRaises"
    ASSERT_RAISES_REGEX = "assertRaisesRegex"

    @classmethod
    def catches_with(cls, call: ast.AST, argument: ast.AST) -> bool:
        """Whether `call` matches exceptions against `argument`, its first argument."""

        match call:
            case ast.Call(func=callee, args=[first, *_]) if first is argument:
                return ExpressionReference.decoded(callee).terminal_name in set(cls)
        return False


class NameUseKind(StrEnum):
    """What one spelling of a name does with the object it names."""

    CONSTRUCT = "construct"
    RAISE = "raise"
    CATCH = "catch"
    OTHER = "other"

    @classmethod
    def of(cls, node: ast.AST, parents: Mapping[int, ast.AST]) -> NameUseKind:
        parent = parents.get(id(node))
        match parent:
            case ast.Call(func=callee) if callee is node:
                return cls.CONSTRUCT
            case ast.Raise(exc=raised) if raised is node:
                return cls.RAISE
            case ast.Tuple() if cls.caught(parent, parents):
                return cls.CATCH
            case _ if cls.caught(node, parents):
                return cls.CATCH
        return cls.OTHER

    @staticmethod
    def caught(node: ast.AST, parents: Mapping[int, ast.AST]) -> bool:
        """Whether `node` is what an `except` clause or a raises-matcher catches."""

        match parents.get(id(node)):
            case ast.ExceptHandler(type=classes) if classes is node:
                return True
            case ast.Call() as call:
                return ExceptionMatcherCall.catches_with(call, node)
        return False

    @classmethod
    def only_builds(cls) -> frozenset[NameUseKind]:
        """The uses that only make an instance: calling the class, raising it."""

        return frozenset((cls.CONSTRUCT, cls.RAISE))


@dataclass(frozen=True)
class NameUse:
    """One name spelled somewhere in a module, and what that spelling does."""

    name: str
    kind: NameUseKind

    @classmethod
    def in_module(cls, parsed_module: ParsedModule) -> frozenset[NameUse]:
        """Every name the module spells, by what the spelling does."""

        return frozenset(
            cls(name, kind) for name, kind, _line in cls.spellings(parsed_module)
        )

    @staticmethod
    def spellings(parsed_module: ParsedModule) -> Iterator[tuple[str, NameUseKind, int]]:
        """Every name the module spells, with what it does and where: references
        by what they do; the capitalised words of strings and comments, and
        explicit re-exports (`from m import X as X`), as other uses."""

        module = parsed_module.module
        parents = {
            id(child): node for node in ast.walk(module) for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(module):
            match node:
                case ast.Name(id=name) | ast.Attribute(attr=name):
                    yield name, NameUseKind.of(node, parents), node.lineno
                case ast.Constant(value=str() as text):
                    for word in SPELLED_WORD.findall(text):
                        yield word, NameUseKind.OTHER, node.lineno
                case ast.alias(name=name, asname=exported) if exported == name:
                    yield name, NameUseKind.OTHER, node.lineno
        for token in tokenize.generate_tokens(io.StringIO(parsed_module.source).readline):
            if token.type == tokenize.COMMENT:
                for word in SPELLED_WORD.findall(token.string):
                    yield word, NameUseKind.OTHER, token.start[0]


@dataclass(frozen=True)
class SiblingTestModules:
    """The test modules of the repositories that scanned modules import from.

    They are consumer evidence: a test that catches a class or names a member
    tells it apart as surely as library code does. They are never where a
    finding is reported, and the scan itself still excludes them. They are
    found by the scan's own discovery (`PythonSourcePathDiscovery`, with
    tests included, kept when `PythonSourcePathPolicy.is_test_path` says so)
    under each scanned module's import root, and parsed by the scan's parser.
    """

    modules: tuple[ParsedModule, ...]

    @classmethod
    def under(cls, import_roots: Iterable[str]) -> SiblingTestModules:
        policy = PythonSourcePathPolicy(include_tests=True)
        modules: list[ParsedModule] = []
        for root in map(Path, sorted(set(import_roots))):
            paths = tuple(
                path
                for path in PythonSourcePathDiscovery(root, policy).paths()
                if PythonSourcePathPolicy.is_test_path(path.relative_to(root))
            )
            modules.extend(
                PythonModuleRootParser.for_root(root, source_policy=policy).parsed_source_paths(
                    paths
                )
            )
        return cls(tuple(modules))

    @staticmethod
    def holds(parsed_module: ParsedModule) -> bool:
        """Whether a module is itself a test: evidence only, never a finding's site."""

        identity = parsed_module.module_path_identity
        return PythonSourcePathPolicy.is_test_path(
            Path(parsed_module.file_path).resolve().relative_to(
                identity.declared_import_root.resolve()
            )
        )

    @classmethod
    def beside(cls, parsed_modules: Iterable[ParsedModule]) -> SiblingTestModules:
        """The test modules beside a set of parsed (scanned) modules."""

        return cls.under(
            str(module.module_path_identity.declared_import_root) for module in parsed_modules
        )


SPELLED_WORD = re.compile(r"(?<![\w])[A-Z]\w*")


@dataclass(frozen=True)
class EmptySubclass:
    """A top-level class with one base, spelled as a bare name, and no body.

    It has no decorators and no class keywords; its body is at most a
    docstring, `pass` or `...`.
    """

    node: ast.ClassDef
    base: ast.Name

    @classmethod
    def decoded(cls, node: ast.AST) -> Iterator[EmptySubclass]:
        match node:
            case ast.ClassDef(
                bases=[ast.Name() as base], keywords=[], decorator_list=[], body=body
            ) if all(cls.is_filler(statement) for statement in body):
                yield cls(node, base)

    @staticmethod
    def is_filler(statement: ast.stmt) -> bool:
        match statement:
            case ast.Pass() | ast.Expr(value=ast.Constant(value=str() | EllipsisType())):
                return True
        return False

    @property
    def name(self) -> str:
        return self.node.name


@dataclass(frozen=True)
class EmptySubclassSiteModule(RestatingSiteModule):
    """One module's empty subclasses as sites, and every name it spells."""

    import_root: str
    uses: tuple[NameUse, ...] = ()


class EmptySubclassSiteFamily(ShapeFamily[EmptySubclassSiteModule]):
    """Top-level empty subclasses, one site each, with the module's name uses."""

    item_type = EmptySubclassSiteModule

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[EmptySubclassSiteModule]:
        return [
            EmptySubclassSiteModule(
                module_name=parsed_module.module_name,
                sites=tuple(
                    RestatingSite(
                        parsed_module.file_path,
                        subclass.node.lineno,
                        subclass.name,
                        subclass.name,
                        (f"{parsed_module.module_name}.{subclass.name}", subclass.base.id),
                    )
                    for statement in parsed_module.module.body
                    if not SiblingTestModules.holds(parsed_module)
                    for subclass in EmptySubclass.decoded(statement)
                ),
                uses=tuple(
                    sorted(NameUse.in_module(parsed_module), key=attrgetter("name", "kind"))
                ),
                import_root=str(parsed_module.module_path_identity.declared_import_root),
            )
        ]


@dataclass(frozen=True)
class EmptySubclassIndex:
    """The scan's empty subclasses that differ from their base only by name.

    A class qualifies when its simple name is unique in the scan; every
    spelling of that name anywhere constructs or raises it; it and every
    ancestor resolve to scanned classes or builtins, none with a metaclass,
    class keywords, a registering metaclass or an `__init_subclass__` that
    stores the subclass (the class index's registration facts); and its base
    is concrete, so constructing the base instead still works.
    """

    class_index: CompactClassFamilyIndex
    kinds_by_name: Mapping[str, frozenset[NameUseKind]]
    bases_by_symbol: Mapping[str, str]

    @classmethod
    def from_projection_groups(
        cls, projections_by_family: dict[type[CollectedFamily[Any]], Sequence[object]]
    ) -> EmptySubclassIndex:
        class_index = CompactClassFamilyIndex.from_projection_groups(projections_by_family)
        modules = EmptySubclassSiteFamily.modules_in(projections_by_family)
        kinds: dict[str, set[NameUseKind]] = defaultdict(set)
        for module in modules:
            for use in module.uses:
                kinds[use.name].add(use.kind)
        for test_module in SiblingTestModules.under(module.import_root for module in modules).modules:
            for use in NameUse.in_module(test_module):
                kinds[use.name].add(use.kind)
        index = cls(class_index, {name: frozenset(found) for name, found in kinds.items()}, {})
        return replace(
            index,
            bases_by_symbol={
                site.detail[0]: base
                for module in modules
                for site in module.sites
                for base in index.undistinguished_base(site.detail[0], site.group)
            },
        )

    def undistinguished_base(self, symbol: str, name: str) -> Iterator[str]:
        """The base symbol, when `symbol` differs from it only by name."""

        indexed = self.class_index.classes_by_symbol.get(symbol)
        if indexed is None or not self.only_built(symbol, name):
            return
        match indexed.resolved_base_symbols:
            case ():
                yield indexed.declared_base_names[0]
            case (base,) if not self.abstract(base):
                yield base

    def only_built(self, symbol: str, name: str) -> bool:
        """Unique in the scan, built by every spelling, and with a closed, unregistered lineage."""

        lineage = (symbol, *self.class_index.ancestor_symbols(symbol))
        return (
            self.class_index.symbols_by_simple_name.get(name) == (symbol,)
            and self.kinds_by_name.get(name, frozenset()) <= NameUseKind.only_builds()
            and all(map(self.closed_and_unregistered, lineage))
        )

    def closed_and_unregistered(self, symbol: str) -> bool:
        indexed = self.class_index.classes_by_symbol.get(symbol)
        if indexed is None:
            return False
        builtins_bases = sum(
            1 for reference in indexed.base_references if self.builtin(reference)
        )
        return (
            builtins_bases + len(indexed.resolved_base_symbols) == len(indexed.base_references)
            and not indexed.declares_autoregister_meta
            and not indexed.is_registration_authority
            and not indexed.registry_facts.stores_subclass
            and not indexed.metaclass_names
            and not indexed.class_keyword_names
        )

    @staticmethod
    def builtin(reference: object) -> bool:
        match reference:
            case BoundNominalReference(
                root_binding=CompactNominalBinding(kind=CompactNominalBindingKind.BUILTIN)
            ):
                return True
        return False

    def abstract(self, symbol: str) -> bool:
        indexed = self.class_index.classes_by_symbol[symbol]
        return indexed.is_abstract or bool(indexed.abstract_method_names)

    def siblings(self, symbol: str) -> tuple[str, ...]:
        base = self.bases_by_symbol[symbol]
        return tuple(
            sorted(
                other.rpartition(".")[2]
                for other, other_base in self.bases_by_symbol.items()
                if other_base == base and other != symbol
            )
        )


@dataclass(frozen=True)
class EnumMemberUse(ABC):
    """One spelling of `Owner.MEMBER`, with the members spelled beside it."""

    owner: str
    member: str
    together: tuple[str, ...]
    line: int

    @abstractmethod
    def record(self, groups_by_member: dict[str, set[tuple[str, ...]]], alone: set[str]) -> None:
        """Count this spelling toward the member's groups, or mark it named alone."""

        raise NotImplementedError

    @abstractmethod
    def refusals(self, file_path: str, folded: tuple[str, ...]) -> Iterator[str]:
        """Why this spelling forbids folding `folded`."""

        raise NotImplementedError


@dataclass(frozen=True)
class GroupedMemberUse(EnumMemberUse):
    """A member spelled inside a grouped test."""

    def record(self, groups_by_member: dict[str, set[tuple[str, ...]]], alone: set[str]) -> None:
        groups_by_member[self.member].add(self.together)

    def refusals(self, file_path: str, folded: tuple[str, ...]) -> Iterator[str]:
        missing = [other for other in folded if other not in self.together]
        if self.member in folded and missing:
            yield (
                f"{file_path}:{self.line} names `{self.owner}.{self.member}` without "
                f"{', '.join(f'`{other}`' for other in missing)}"
            )


@dataclass(frozen=True)
class LoneMemberUse(EnumMemberUse):
    """A member spelled on its own: read, passed, compared or reached by attribute."""

    def record(self, groups_by_member: dict[str, set[tuple[str, ...]]], alone: set[str]) -> None:
        alone.add(self.member)

    def refusals(self, file_path: str, folded: tuple[str, ...]) -> Iterator[str]:
        if self.member in folded:
            yield (
                f"{file_path}:{self.line} names `{self.owner}.{self.member}` outside a grouped test"
            )


def spelled_through(member: ast.Attribute, owner: str) -> bool:
    """Whether `member` is an `owner.MEMBER` access."""

    match member.value:
        case ast.Name(id=spelled) | ast.Attribute(attr=spelled):
            return spelled == owner
    return False


@dataclass(frozen=True)
class MemberGroup(ABC):
    """Where enum members are spelled: together in one grouped test, or alone."""

    members: tuple[ast.Attribute, ...]

    @abstractmethod
    def use(self, owner: str, member: ast.Attribute) -> EnumMemberUse:
        raise NotImplementedError

    @abstractmethod
    def folded(
        self,
        geometry: SourceTextGeometry,
        owner: str,
        folded: frozenset[str],
        file_path: str,
    ) -> Iterator[SourceTextSpanReplacement]:
        """This group's source without the folded members' spellings."""

        raise NotImplementedError

    def names(self, owner: str) -> tuple[str, ...]:
        return tuple(
            sorted({member.attr for member in self.members if spelled_through(member, owner)})
        )

    @classmethod
    def of(cls, member: ast.Attribute, parents: Mapping[int, ast.AST]) -> MemberGroup:
        parent = parents.get(id(member))
        grandparent = None if parent is None else parents.get(id(parent))
        match parent, grandparent:
            case (
                ast.Tuple(elts=elements) | ast.List(elts=elements) | ast.Set(elts=elements),
                ast.Compare(ops=[ast.In() | ast.NotIn()], comparators=[literal]),
            ) if literal is parent:
                return GroupedMembers(tuple(cls.members_in(elements)), parent)
            case (ast.MatchValue(), ast.MatchOr(patterns=patterns)):
                return GroupedMembers(
                    tuple(
                        value
                        for pattern in patterns
                        for value in cls.members_in(
                            [pattern.value] if isinstance(pattern, ast.MatchValue) else []
                        )
                    ),
                    grandparent,
                )
            case (
                ast.Compare(ops=[ast.Is()], comparators=[compared]),
                ast.BoolOp(op=ast.Or(), values=values),
            ) if compared is member and cls.is_identity_chain(values):
                return GroupedMembers(
                    tuple(
                        comparator
                        for value in values
                        for comparator in cls.members_in(cast(ast.Compare, value).comparators)
                    ),
                    grandparent,
                )
        return LoneMember((member,))

    @staticmethod
    def members_in(nodes: Sequence[ast.AST]) -> Iterator[ast.Attribute]:
        for node in nodes:
            match node:
                case ast.Attribute(value=ast.Name() | ast.Attribute()):
                    yield node

    @staticmethod
    def is_identity_chain(values: Sequence[ast.expr]) -> bool:
        """`s is E.A or s is E.B ...`: every operand an `is` test of one subject."""

        subjects = [
            ast.dump(left)
            for value in values
            for left in (
                [value.left]
                if isinstance(value, ast.Compare) and isinstance(value.ops[0], ast.Is)
                and len(value.ops) == 1
                else []
            )
        ]
        return len(subjects) == len(values) and len(set(subjects)) == 1


@dataclass(frozen=True)
class GroupedMembers(MemberGroup):
    """`x in (E.A, E.B)`, `case E.A | E.B:` or `x is E.A or x is E.B`."""

    node: ast.expr | ast.pattern

    def use(self, owner: str, member: ast.Attribute) -> EnumMemberUse:
        return GroupedMemberUse(owner, member.attr, self.names(owner), member.lineno)

    def folded(
        self,
        geometry: SourceTextGeometry,
        owner: str,
        folded: frozenset[str],
        file_path: str,
    ) -> Iterator[SourceTextSpanReplacement]:
        if folded.isdisjoint(self.names(owner)):
            return
        start, end = geometry.required_node_offsets(self.node)
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise ValueError(f"{file_path}:{self.node.lineno} folding would drop a comment")
        yield SourceTextSpanReplacement.from_offsets(
            start_offset=start,
            end_offset=end,
            replacement_source=self.source_without(geometry, owner, folded),
        )

    def source_without(
        self, geometry: SourceTextGeometry, owner: str, folded: frozenset[str]
    ) -> str:
        """The grouped test's source with the folded members' spellings dropped."""

        match self.node:
            case ast.Tuple(elts=elements):
                remaining = self.kept(geometry, elements, owner, folded)
                return f"({remaining[0]},)" if len(remaining) == 1 else f"({', '.join(remaining)})"
            case ast.List(elts=elements):
                return f"[{', '.join(self.kept(geometry, elements, owner, folded))}]"
            case ast.Set(elts=elements):
                return f"{{{', '.join(self.kept(geometry, elements, owner, folded))}}}"
            case ast.MatchOr(patterns=patterns):
                return " | ".join(self.kept(geometry, patterns, owner, folded))
            case ast.BoolOp(values=values):
                return " or ".join(self.kept(geometry, values, owner, folded))
        raise ValueError("Folding met a node that is not a grouped test")

    @classmethod
    def kept(
        cls,
        geometry: SourceTextGeometry,
        parts: Sequence[ast.AST],
        owner: str,
        folded: frozenset[str],
    ) -> list[str]:
        """The source of each part that does not spell a folded member."""

        return [
            node_source(geometry, part)
            for part in parts
            if not cls.names_folded(part, owner, folded)
        ]

    @staticmethod
    def names_folded(part: ast.AST, owner: str, folded: frozenset[str]) -> bool:
        match part:
            case ast.MatchValue(value=ast.Attribute() as member) | (
                ast.Compare(comparators=[ast.Attribute() as member])
            ) | (ast.Attribute() as member):
                return member.attr in folded and spelled_through(member, owner)
        return False


@dataclass(frozen=True)
class LoneMember(MemberGroup):
    """One member spelled outside any grouped test."""

    def use(self, owner: str, member: ast.Attribute) -> EnumMemberUse:
        return LoneMemberUse(owner, member.attr, (member.attr,), member.lineno)

    def folded(
        self,
        geometry: SourceTextGeometry,
        owner: str,
        folded: frozenset[str],
        file_path: str,
    ) -> Iterator[SourceTextSpanReplacement]:
        """Nothing: a folded member spelled alone refuses the fold before any edit."""

        yield from ()


class EnumUseKind(StrEnum):
    """What one spelling of an enum class name does."""

    MEMBER = "member"
    TYPE = "type"
    OTHER = "other"

    @classmethod
    def of(cls, node: ast.AST, parents: Mapping[int, ast.AST]) -> EnumUseKind:
        """A member access, a type use (annotation or `isinstance`), or anything else.

        Calling, subscripting or iterating the class reaches members by value,
        name or order, which a fold would change.
        """

        parent = parents.get(id(node))
        match parent:
            case ast.Attribute(value=owner) if owner is node:
                return cls.MEMBER
            case ast.arg(annotation=annotation) | ast.AnnAssign(annotation=annotation) if (
                annotation is node
            ):
                return cls.TYPE
            case ast.FunctionDef(returns=returned) | ast.AsyncFunctionDef(returns=returned) if (
                returned is node
            ):
                return cls.TYPE
            case ast.Call(func=callee, args=[_, tested]) if tested is node and any(
                name.is_callee(callee) for name in BuiltinCallName.type_test_names()
            ):
                return cls.TYPE
        return cls.OTHER


@dataclass(frozen=True)
class EnumClassUse:
    """One spelling of a class name, by what it does."""

    owner: str
    kind: EnumUseKind
    line: int


@dataclass(frozen=True)
class EnumMembers:
    """A top-level class whose body only binds names to constants."""

    node: ast.ClassDef

    @classmethod
    def decoded(cls, node: ast.AST) -> Iterator[EnumMembers]:
        match node:
            case ast.ClassDef(body=body, decorator_list=[]) if body and all(
                cls.binds_member(statement) or EmptySubclass.is_filler(statement)
                for statement in body
            ):
                yield cls(node)

    @staticmethod
    def binds_member(statement: ast.stmt) -> bool:
        match statement:
            case ast.Assign(targets=[ast.Name()], value=ast.Constant()):
                return True
        return False

    @property
    def statements_by_member(self) -> dict[str, ast.stmt]:
        return {
            target.id: statement
            for statement in self.node.body
            if self.binds_member(statement)
            for target in cast(ast.Assign, statement).targets
            if isinstance(target, ast.Name)
        }


@dataclass(frozen=True)
class EnumUseModule(RestatingSiteModule):
    """One module's literal-valued classes as sites, and every enum-shaped spelling."""

    import_root: str
    member_uses: tuple[EnumMemberUse, ...] = ()
    class_uses: tuple[EnumClassUse, ...] = ()


def enum_uses(
    module: ast.Module, owners: Callable[[str], bool]
) -> tuple[list[EnumMemberUse], list[EnumClassUse], list[MemberGroup]]:
    """Every `Owner.MEMBER` and every other spelling of an owner that `owners` admits."""

    parents = {id(child): node for node in ast.walk(module) for child in ast.iter_child_nodes(node)}
    member_uses: list[EnumMemberUse] = []
    class_uses: list[EnumClassUse] = []
    groups: list[MemberGroup] = []
    for node in ast.walk(module):
        match node:
            case ast.Attribute(value=ast.Name(id=owner) | ast.Attribute(attr=owner)) if owners(
                owner
            ):
                group = MemberGroup.of(node, parents)
                groups.append(group)
                member_uses.append(group.use(owner, node))
        match node:
            case ast.Name(id=owner) | ast.Attribute(attr=owner) if owners(owner):
                class_uses.append(
                    EnumClassUse(owner, EnumUseKind.of(node, parents), node.lineno)
                )
    return member_uses, class_uses, groups


class EnumUseSiteFamily(ShapeFamily[EnumUseModule]):
    """Literal-valued classes, one site each, and the module's enum-shaped spellings.

    Owners are spelled names that some scanned class could be: capitalised,
    the shape a class name takes. The join keeps only plain Enum classes.
    """

    item_type = EnumUseModule

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[EnumUseModule]:
        member_uses, class_uses, _groups = enum_uses(
            parsed_module.module, cls.spelled_like_a_class
        )
        return [
            EnumUseModule(
                module_name=parsed_module.module_name,
                sites=tuple(
                    RestatingSite(
                        parsed_module.file_path,
                        members.node.lineno,
                        members.node.name,
                        members.node.name,
                        (f"{parsed_module.module_name}.{members.node.name}",),
                    )
                    for statement in parsed_module.module.body
                    if not SiblingTestModules.holds(parsed_module)
                    for members in EnumMembers.decoded(statement)
                ),
                member_uses=tuple(dict.fromkeys(member_uses)),
                class_uses=tuple(dict.fromkeys(class_uses)),
                import_root=str(parsed_module.module_path_identity.declared_import_root),
            )
        ]

    @staticmethod
    def spelled_like_a_class(name: str) -> bool:
        return name[:1].isupper()


@dataclass(frozen=True)
class EnumMemberIndex:
    """Plain-Enum members that every spelling in the scan names together.

    Members qualify when the class is a plain `Enum` with literal values and a
    simple name unique in the scan; every spelling of the class is a member
    access or a type use; every spelling of each member sits in a grouped
    form; and the members share exactly the same groups.
    """

    class_index: CompactClassFamilyIndex
    folds: Mapping[str, tuple[tuple[str, ...], ...]]

    @classmethod
    def from_projection_groups(
        cls, projections_by_family: dict[type[CollectedFamily[Any]], Sequence[object]]
    ) -> EnumMemberIndex:
        class_index = CompactClassFamilyIndex.from_projection_groups(projections_by_family)
        modules = EnumUseSiteFamily.modules_in(projections_by_family)
        member_uses: dict[str, list[EnumMemberUse]] = defaultdict(list)
        class_kinds: dict[str, set[EnumUseKind]] = defaultdict(set)
        tests = SiblingTestModules.under(module.import_root for module in modules)
        evidence = (
            *((module.member_uses, module.class_uses) for module in modules),
            *(
                enum_uses(test.module, EnumUseSiteFamily.spelled_like_a_class)[:2]
                for test in tests.modules
            ),
        )
        for module_member_uses, module_class_uses in evidence:
            for use in module_member_uses:
                member_uses[use.owner].append(use)
            for class_use in module_class_uses:
                class_kinds[class_use.owner].add(class_use.kind)
        folds = {
            site.detail[0]: found
            for module in modules
            for site in module.sites
            if cls.plain_enum(class_index, site.detail[0], site.group)
            and class_kinds.get(site.group, set()) <= {EnumUseKind.MEMBER, EnumUseKind.TYPE}
            for found in [cls.undistinguished(member_uses.get(site.group, []))]
            if found
        }
        return cls(class_index, folds)

    @staticmethod
    def plain_enum(class_index: CompactClassFamilyIndex, symbol: str, name: str) -> bool:
        indexed = class_index.classes_by_symbol.get(symbol)
        if indexed is None or class_index.symbols_by_simple_name.get(name) != (symbol,):
            return False
        match indexed.enum_declaration:
            case PythonEnumDeclarationAuthority(base=PythonEnumBaseKind.ENUM):
                return not indexed.method_names
        return False

    @staticmethod
    def undistinguished(uses: Sequence[EnumMemberUse]) -> tuple[tuple[str, ...], ...]:
        """Sets of two or more members spelled only in groups, and always together."""

        groups_by_member: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        alone: set[str] = set()
        for use in uses:
            use.record(groups_by_member, alone)
        by_groups: dict[frozenset[tuple[str, ...]], list[str]] = defaultdict(list)
        for member, groups in groups_by_member.items():
            if member not in alone:
                by_groups[frozenset(groups)].append(member)
        return tuple(
            sorted(tuple(sorted(members)) for members in by_groups.values() if len(members) > 1)
        )
