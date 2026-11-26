"""String keys that select behaviour, joined across sites into the family they spell.

A string compared with literals to pick a branch is a nominal key kept by hand
(paper1 Thm 3.2). One site is `StringKindDispatchDetector`'s; the family is the
cross-site object: every place that tests the same key with overlapping literals.

Per module, `StringKeySiteFamily` records each selecting site: the key it tests,
each literal's branch (normalized), the default branch, whether the default
refuses (a closed set) and whether the site is a boundary decoder (a raw mapping
read, or every arm building its own class). Across the scan, `StringFamily` joins
the sites of one key whose literals overlap or that share a module, and builds
the confusability graph:
two literals are joined when some site sends them to different branches, a
literal a site does not name taking that site's default. Literals no edge joins
agree at every site, so they are one member spelled several ways; the fewest
members that keep every site's answer is the graph's chromatic number, exactly
for small families and by greedy colouring, between the largest clique found and
the colouring, above that.
"""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property

from nominal_refactor_advisor.ast_field_decoders import match_guards, match_patterns
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.foreign_family import FunctionDefinition
from nominal_refactor_advisor.models import SourceLocation
from nominal_refactor_advisor.observation_families import ShapeFamily

FALLS_THROUGH = "<falls through>"
"""The branch of a literal a chain names no arm for and has no else: the code after it."""
REFUSES = "<refuses>"
"""An else that raises: the literal set is closed."""
EXACT_COLOURING_LIMIT = 12
"""Families up to this many literals are coloured exactly; larger ones greedily."""
DECODER_NAMES = frozenset({"decode", "parse", "from_wire", "from_json", "from_dict", "load", "loads", "member_named"})
"""Function names whose string tests read a foreign value: the one place strings belong."""


def _behaviour(body: list[ast.stmt]) -> str:
    """One branch, normalized: its statements without positions or comments; no
    statements is the code after the test."""

    if not body:
        return FALLS_THROUGH
    return ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False)


def _refuses(body: list[ast.stmt]) -> bool:
    return len(body) == 1 and isinstance(body[0], ast.Raise)


def _key(subject: ast.expr) -> str:
    """What a site tests, as sites elsewhere spell it: the attribute read, or the bare name."""

    match subject:
        case ast.Attribute(attr=attr):
            return attr
        case ast.Name(id=name):
            return name
        case ast.Call(func=ast.Attribute(attr="get"), args=[ast.Constant(value=str() as name), *_]):
            return name
        case ast.Subscript(slice=ast.Constant(value=str() as name)):
            return name
    return ""


def _raw_read(subject: ast.expr) -> bool:
    """A value read off a mapping (`d["kind"]`, `d.get("kind")`): foreign until decoded."""

    match subject:
        case ast.Subscript(slice=ast.Constant(value=str())):
            return True
        case ast.Call(func=ast.Attribute(attr="get"), args=[ast.Constant(value=str()), *_]):
            return True
    return False


def _builds_a_class(body: list[ast.stmt]) -> bool:
    """`return Name(...)` or `return Name`: an arm that decodes to its own class."""

    match body:
        case [ast.Return(value=ast.Call(func=ast.Name(id=name)) | ast.Name(id=name))]:
            return name[:1].isupper()
    return False


def _literals(test: ast.expr) -> tuple[ast.expr, tuple[str, ...], bool] | None:
    """`x == "a"`, `"a" == x`, `x in ("a", "b")`, `x.startswith("a")`: the subject,
    its literals, and whether the test is negated (`!=`, `not in`), so that its
    literals take the else branch and every other value the body."""

    match test:
        case ast.Compare(left=left, ops=[ast.Eq() | ast.NotEq() as op], comparators=[ast.Constant(value=str() as literal)]):
            return left, (literal,), isinstance(op, ast.NotEq)
        case ast.Compare(left=ast.Constant(value=str() as literal), ops=[ast.Eq() | ast.NotEq() as op], comparators=[right]):
            return right, (literal,), isinstance(op, ast.NotEq)
        case ast.Compare(left=left, ops=[ast.In() | ast.NotIn() as op], comparators=[ast.Tuple(elts=elements) | ast.Set(elts=elements) | ast.List(elts=elements)]):
            values = tuple(e.value for e in elements if isinstance(e, ast.Constant) and isinstance(e.value, str))
            if values and len(values) == len(elements):
                return left, values, isinstance(op, ast.NotIn)
        case ast.Call(func=ast.Attribute(value=subject, attr="startswith"), args=[ast.Constant(value=str() as prefix)]):
            return subject, (prefix + "*",), False
    return None


@dataclass(frozen=True)
class StringKeySite:
    """One place a string selects a branch: what it tests and where each literal goes."""

    file_path: str
    lineno: int
    symbol: str
    key: str
    subject: str
    arms: tuple[tuple[str, str], ...]
    """Each literal with its branch."""
    default: str
    """The branch of a literal the site names no arm for."""
    boundary: bool
    """The site decodes a foreign value: a raw mapping read, a decoder, or one class per arm."""

    @property
    def closed(self) -> bool:
        return self.default == REFUSES

    @property
    def location(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.lineno, self.symbol)

    @property
    def literals(self) -> frozenset[str]:
        return frozenset(literal for literal, _branch in self.arms)

    def branch(self, literal: str) -> str:
        """Where this site sends `literal`: its arm, or the default."""

        return dict(self.arms).get(literal, self.default)


@dataclass(frozen=True)
class StringKeySiteModule:
    """One module's selecting sites, kept even when empty."""

    module_name: str
    sites: tuple[StringKeySite, ...]


class StringKeySiteFamily(ShapeFamily[StringKeySiteModule]):
    """Every if/elif chain and match statement whose arms select by string literal."""

    item_type = StringKeySiteModule

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[StringKeySiteModule]:
        return [StringKeySiteModule(parsed_module.module_name, tuple(cls.sites(parsed_module)))]

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[StringKeySite]:
        for function in FunctionDefinition.within(ast.walk(parsed_module.module)):
            decoder = function.name in DECODER_NAMES
            nested: set[int] = set()
            for node in cls.own_nodes(function):
                if isinstance(node, ast.If) and id(node) not in nested:
                    site = cls.chain(parsed_module, function, node, nested, decoder)
                elif isinstance(node, ast.Match):
                    site = cls.matched(parsed_module, function, node, decoder)
                else:
                    continue
                if site is not None:
                    yield site

    @staticmethod
    def own_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
        """The function's own statements and expressions, not a nested function's or
        class's: each of those is a site's function of its own."""

        scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        stack: list[ast.AST] = [node for node in reversed(function.body) if not isinstance(node, scopes)]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(
                child for child in reversed(list(ast.iter_child_nodes(node))) if not isinstance(child, scopes)
            )

    @staticmethod
    def _site(
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        node: ast.stmt,
        subject: ast.expr,
        arms: list[tuple[str, list[ast.stmt]]],
        default: list[ast.stmt] | None,
        decoder: bool,
    ) -> StringKeySite | None:
        if not arms or not _key(subject):
            return None
        default_branch = FALLS_THROUGH if default is None else REFUSES if _refuses(default) else _behaviour(default)
        return StringKeySite(
            file_path=parsed_module.file_path,
            lineno=node.lineno,
            symbol=f"{parsed_module.module_name}.{function.name}",
            key=_key(subject),
            subject=ast.unparse(subject),
            arms=tuple(sorted((literal, _behaviour(body)) for literal, body in arms)),
            default=default_branch,
            boundary=decoder or _raw_read(subject) or all(_builds_a_class(body) for _literal, body in arms),
        )

    @classmethod
    def chain(
        cls,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        node: ast.If,
        nested: set[int],
        decoder: bool,
    ) -> StringKeySite | None:
        """An if/elif chain testing one subject: each `elif` is the next arm, the last `else` the default.

        A negated first test is a site of its own: its literals take the else, and
        everything else the body; a negated test later in a chain ends it."""

        arms: list[tuple[str, list[ast.stmt]]] = []
        subject_text: str | None = None
        subject: ast.expr | None = None
        current: ast.If | None = node
        default: list[ast.stmt] | None = None
        while current is not None:
            found = _literals(current.test)
            if found is None or (found[2] and arms):
                default = [current]
                break
            tested, literals, negated = found
            if subject_text is None:
                subject_text, subject = ast.unparse(tested), tested
            elif ast.unparse(tested) != subject_text:
                default = [current]
                break
            if negated:
                arms.extend((literal, current.orelse) for literal in literals)
                default = current.body
                break
            arms.extend((literal, current.body) for literal in literals)
            nested.add(id(current))
            match current.orelse:
                case [ast.If() as following]:
                    current = following
                case []:
                    current = None
                case orelse:
                    default, current = orelse, None
        if subject is None:
            return None
        return cls._site(parsed_module, function, node, subject, arms, default, decoder)

    @classmethod
    def matched(
        cls,
        parsed_module: ParsedModule,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        node: ast.Match,
        decoder: bool,
    ) -> StringKeySite | None:
        arms: list[tuple[str, list[ast.stmt]]] = []
        default: list[ast.stmt] | None = None
        for case in node.cases:
            literals = cls.pattern_literals(case.pattern)
            if literals:
                arms.extend((literal, case.body) for literal in literals)
            elif isinstance(case.pattern, ast.MatchAs) and not match_patterns(
                case.pattern
            ) and not match_guards(case):
                default = case.body
        return cls._site(parsed_module, function, node, node.subject, arms, default, decoder)

    @staticmethod
    def pattern_literals(pattern: ast.pattern) -> tuple[str, ...]:
        match pattern:
            case ast.MatchValue(value=ast.Constant(value=str() as literal)):
                return (literal,)
            case ast.MatchOr(patterns=patterns):
                found = tuple(
                    p.value.value for p in patterns
                    if isinstance(p, ast.MatchValue) and isinstance(p.value, ast.Constant) and isinstance(p.value.value, str)
                )
                return found if len(found) == len(patterns) else ()
        return ()


@dataclass(frozen=True)
class StringFamily:
    """The sites of one key whose literals overlap, and the members they need."""

    key: str
    sites: tuple[StringKeySite, ...]

    @classmethod
    def joined(cls, sites: Iterable[StringKeySite]) -> tuple[StringFamily, ...]:
        """Sites of one key, joined when their literals overlap or they share a module:
        one module's tests of a key are one family even split one literal per function;
        across modules, a shared literal is the evidence (union-find over sites)."""

        by_key: dict[str, list[StringKeySite]] = defaultdict(list)
        for site in sites:
            by_key[site.key].append(site)
        families = []
        for key, keyed in sorted(by_key.items()):
            parent = list(range(len(keyed)))

            def find(index: int) -> int:
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            owners: dict[tuple[str, str], int] = {}
            for index, site in enumerate(keyed):
                for anchor in (*(("literal", literal) for literal in site.literals), ("module", site.file_path)):
                    if anchor in owners:
                        parent[find(index)] = find(owners[anchor])
                    else:
                        owners[anchor] = index
            groups: dict[int, list[StringKeySite]] = defaultdict(list)
            for index, site in enumerate(keyed):
                groups[find(index)].append(site)
            families.extend(
                cls(key, tuple(sorted(group, key=lambda s: (s.file_path, s.lineno))))
                for _root, group in sorted(groups.items())
            )
        return tuple(families)

    @cached_property
    def literals(self) -> tuple[str, ...]:
        return tuple(sorted(frozenset().union(*(site.literals for site in self.sites))))

    @cached_property
    def edges(self) -> frozenset[tuple[str, str]]:
        """Literal pairs some site sends to different branches: they cannot share a member."""

        found = set()
        for index, first in enumerate(self.literals):
            for second in self.literals[index + 1 :]:
                if any(site.branch(first) != site.branch(second) for site in self.sites):
                    found.add((first, second))
        return frozenset(found)

    @cached_property
    def members(self) -> tuple[tuple[str, ...], ...]:
        """The fewest literal groups that keep every site's answer: a colouring of `edges`."""

        colours = self._colouring()
        groups: dict[int, list[str]] = defaultdict(list)
        for literal in self.literals:
            groups[colours[literal]].append(literal)
        return tuple(sorted(tuple(group) for group in groups.values()))

    @property
    def exact(self) -> bool:
        """Whether `members` is the minimum (paper1 Thm 3.2's chromatic number) or a greedy bound."""

        return len(self.literals) <= EXACT_COLOURING_LIMIT

    @cached_property
    def clique(self) -> int:
        """A lower bound on the member count: the largest clique the greedy search finds."""

        neighbours = self._neighbours()
        best = 1 if self.literals else 0
        for start in self.literals:
            clique = [start]
            for literal in sorted(neighbours[start], key=lambda l: -len(neighbours[l])):
                if all(literal in neighbours[member] for member in clique):
                    clique.append(literal)
            best = max(best, len(clique))
        return best

    def _neighbours(self) -> dict[str, set[str]]:
        neighbours: dict[str, set[str]] = {literal: set() for literal in self.literals}
        for first, second in self.edges:
            neighbours[first].add(second)
            neighbours[second].add(first)
        return neighbours

    def _colouring(self) -> dict[str, int]:
        neighbours = self._neighbours()
        order = sorted(self.literals, key=lambda literal: (-len(neighbours[literal]), literal))
        greedy: dict[str, int] = {}
        for literal in order:
            used = {greedy[other] for other in neighbours[literal] if other in greedy}
            greedy[literal] = next(colour for colour in range(len(order) + 1) if colour not in used)
        if not self.exact:
            return greedy
        bound = max(greedy.values(), default=-1) + 1
        for count in range(self.clique, bound):
            found = self._colour_with(order, neighbours, count, {})
            if found is not None:
                return found
        return greedy

    def _colour_with(
        self, order: list[str], neighbours: dict[str, set[str]], count: int, given: dict[str, int]
    ) -> dict[str, int] | None:
        if len(given) == len(order):
            return dict(given)
        literal = order[len(given)]
        used = {given[other] for other in neighbours[literal] if other in given}
        for colour in range(min(count, max(given.values(), default=-1) + 2)):
            if colour in used:
                continue
            given[literal] = colour
            found = self._colour_with(order, neighbours, count, given)
            if found is not None:
                return found
            del given[literal]
        return None

    @property
    def merges(self) -> tuple[tuple[str, ...], ...]:
        """Members spelled by more than one literal: one member, many spellings."""

        return tuple(member for member in self.members if len(member) > 1)

    @property
    def boundary_share(self) -> float:
        return sum(site.boundary for site in self.sites) / len(self.sites)

    @property
    def confidence(self) -> float:
        """How likely the strings are a hand-kept identity rather than a wire boundary,
        uncalibrated: more sites and a closed set raise it; decoder sites lower it."""

        log_odds = (
            1.0 * math.log(len(self.sites))
            + 0.5 * math.log(len(self.literals))
            + (0.5 if any(site.closed for site in self.sites) else 0.0)
            + (0.5 if self.merges else 0.0)
            - 3.0 * self.boundary_share
        )
        return 1.0 / (1.0 + math.exp(-log_odds))

    @property
    def weight(self) -> int:
        """Sites times literals: how much hand-kept identity the family carries."""

        return len(self.sites) * len(self.literals)


__all__ = (
    "StringFamily",
    "StringKeySite",
    "StringKeySiteFamily",
    "StringKeySiteModule",
)
