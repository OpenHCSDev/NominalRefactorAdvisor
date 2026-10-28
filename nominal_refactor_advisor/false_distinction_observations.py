"""False distinctions: code that tells cases apart and then treats them the same.

A branch whose arms all run the same code distinguishes cases only in its
test. The site families here collect each shape per module; the detectors in
`detectors/_false_distinctions.py` name the fact each one restates.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .ast_tools import ParsedModule
from .restated_fact_observations import (
    RestatingSite,
    RestatingSiteFamily,
    RestatingSiteModule,
)

StatementList = Sequence[ast.stmt]


def same_code(left: Sequence[ast.AST], right: Sequence[ast.AST]) -> bool:
    """Whether two node sequences are the same code, positions aside."""

    return len(left) == len(right) and all(
        ast.dump(one) == ast.dump(other) for one, other in zip(left, right)
    )


def statement_lists(root: ast.AST) -> Iterator[StatementList]:
    """Every block of statements under `root`, nested scopes included."""

    for node in ast.walk(root):
        for _name, value in ast.iter_fields(node):
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                yield value


@dataclass(frozen=True)
class IdenticalArms(ABC):
    """A branch every arm of which runs the same code, so its tests decide nothing."""

    node: ast.stmt | ast.expr

    @property
    @abstractmethod
    def tests(self) -> tuple[ast.expr, ...]:
        """Every expression the branch evaluates only to choose an arm."""

        raise NotImplementedError

    @classmethod
    def within(cls, root: ast.AST) -> Iterator[IdenticalArms]:
        """Every branch under `root` whose arms are the same code."""

        continuations = {
            id(node.orelse[0])
            for node in ast.walk(root)
            if isinstance(node, ast.If) and IfLadderArms.continues(node)
        }
        for node in ast.walk(root):
            match node:
                case ast.If() if id(node) not in continuations:
                    yield from IfLadderArms.found(node)
                case ast.IfExp(body=body, orelse=orelse) if same_code(
                    [body], [orelse]
                ):
                    yield ConditionalExpressionArms(node)
                case ast.Match():
                    yield from MatchArms.found(node)
        for statements in statement_lists(root):
            yield from FallThroughArms.found(statements)


@dataclass(frozen=True)
class IfLadderArms(IdenticalArms):
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

    @classmethod
    def found(cls, node: ast.If) -> Iterator[IfLadderArms]:
        arms = cls(node)
        final = arms.ladder[-1].orelse
        if final and all(
            same_code(rung.body, final) for rung in arms.ladder
        ):
            yield arms


@dataclass(frozen=True)
class ConditionalExpressionArms(IdenticalArms):
    """`x if test else x`."""

    node: ast.IfExp

    @property
    def tests(self) -> tuple[ast.expr, ...]:
        return (self.node.test,)


@dataclass(frozen=True)
class MatchArms(IdenticalArms):
    """A `match` whose cases all run the same code and whose last case is `case _:`."""

    node: ast.Match

    @property
    def tests(self) -> tuple[ast.expr, ...]:
        return (self.node.subject, *self.guards())

    def guards(self) -> Iterator[ast.expr]:
        for case in self.node.cases:
            match case:
                case ast.match_case(guard=ast.expr() as guard):
                    yield guard

    @classmethod
    def found(cls, node: ast.Match) -> Iterator[MatchArms]:
        match node.cases[-1]:
            case ast.match_case(
                pattern=ast.MatchAs(pattern=None, name=None), guard=None
            ) if all(same_code(case.body, node.cases[0].body) for case in node.cases):
                yield cls(node)


@dataclass(frozen=True)
class FallThroughArms(IdenticalArms):
    """`if test: S` followed directly by the same S, where S leaves the block."""

    node: ast.If

    @property
    def tests(self) -> tuple[ast.expr, ...]:
        return (self.node.test,)

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
        for arms in IdenticalArms.within(parsed_module.module):
            yield cls.site(
                parsed_module,
                arms.node,
                index.lexical_symbol(node_indices[id(arms.node)]),
                f"line {arms.node.lineno}",
                tuple(ast.unparse(test) for test in arms.tests),
            )
