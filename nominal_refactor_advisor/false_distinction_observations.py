"""False distinctions: code that tells cases apart and then treats them the same.

A branch whose arms all run the same code distinguishes cases only in its
test. The site families here collect each shape per module; the detectors in
`detectors/_false_distinctions.py` name the fact each one restates. Each shape
also knows how to collapse itself to its one arm, refusing when the tests it
would delete could have an effect.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby

from .ast_tools import BuiltinCallName, ParsedModule, walk_function_body_nodes
from .codemod_source_edits import (
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .codemod_foreign_family_source import RewrittenLineFit
from .restated_fact_observations import (
    RestatingSite,
    RestatingSiteFamily,
    RestatingSiteModule,
)

StatementList = Sequence[ast.stmt]
ScopeNode = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def same_code(left: Sequence[ast.AST], right: Sequence[ast.AST]) -> bool:
    """Whether two node sequences are the same code, positions aside."""

    return len(left) == len(right) and all(
        ast.dump(one) == ast.dump(other) for one, other in zip(left, right)
    )


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
        subject = self.source_of(geometry, self.subject)
        classes = ", ".join(
            self.source_of(geometry, tested)
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

    @staticmethod
    def source_of(geometry: SourceTextGeometry, node: ast.expr) -> str:
        start, end = geometry.required_node_offsets(node)
        return geometry.source[start:end]


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
