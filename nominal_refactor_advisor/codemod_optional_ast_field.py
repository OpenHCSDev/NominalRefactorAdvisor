"""Route absence probes of an optional ast field through its one boundary decoder.

An optional ast field (``keyword.arg``, ``arg.annotation`` ...) has exactly
two states, and a decoder owns the choice between them. A probe site that
tests the field for ``None`` restates that choice; this recipe rewrites the
site to read the decoder's answer instead, and refuses any site whose shape it
cannot carry without changing what the code means.

A project that does not declare a decoder gets it generated: the decoder's
source is copied from the owner this package declares, into one
``ast_field_decoders`` module per root package.
"""

from __future__ import annotations

import ast
import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Generic, Self, TypeGuard, TypeVar

from . import ast_field_decoders, ast_projection
from .ast_tools import ParsedModule
from .cases_family import handback
from .codemod import FindingEvidenceActionKeysMixin, FindingRecipeSynthesizer
from .codemod_foreign_family_source import (
    FunctionRewriteAuthority,
    ImportedClassReference,
)
from .codemod_payload import RequiredStringPayloadValueCodec, codemod_payload_field
from .codemod_reproof import SourceReprovedOperation
from .codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewriteTarget
from .codemod_source_edits import (
    NominalSourceEdit,
    SourceFileCreation,
    SourceLineSpan,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .codemod_target_selectors import SourceIndexTargetSelector
from .declared_root import DeclaredRoot
from .models import RefactorFinding
from .refactor_concepts import OptionalAstFieldDecoderConcept

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
FUNCTION_CLASSES = frozenset(("FunctionDef", "AsyncFunctionDef", "Lambda"))
GENERATED_MODULE = "ast_field_decoders"


def _same_expression(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left) == ast.dump(right)


def _none_probe(node: ast.AST) -> TypeGuard[ast.Compare]:
    """Whether ``node`` is ``X.f is None`` or ``X.f is not None``."""

    match node:
        case ast.Compare(
            left=ast.Attribute(),
            ops=[ast.Is() | ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        ):
            return True
    return False


# --- shapes -----------------------------------------------------------------


@dataclass(frozen=True)
class ProbeSite(ABC):
    """One probe of ``V.field is [not] None`` in a shape a decoder can carry."""

    decoder: type[OptionalAstFieldDecoder]
    probe: ast.Compare
    variable: str

    @classmethod
    @abstractmethod
    def at(
        cls, probe: ast.Compare, authority: OptionalAstFieldAuthority
    ) -> Self | None:
        """This shape's site for the probe, when the probe sits in this shape."""
        raise NotImplementedError

    @abstractmethod
    def replacements(
        self, authority: OptionalAstFieldAuthority
    ) -> tuple[SourceTextSpanReplacement, ...]:
        raise NotImplementedError

    @property
    def anchor(self) -> ast.AST:
        """The node whose span identifies the site when two probes share it."""
        return self.probe

    @staticmethod
    def probed(
        probe: ast.Compare, decoder: type[CollectionDecoder], absent: bool
    ) -> str | None:
        """The probed variable when the probe tests the decoder's field as asked."""

        match probe:
            case ast.Compare(
                left=ast.Attribute(value=ast.Name(id=variable), attr=attribute),
                ops=[operator],
                comparators=[ast.Constant(value=None)],
            ) if attribute in decoder.field_names() and isinstance(
                operator, ast.Is if absent else ast.IsNot
            ):
                return variable
        return None


@dataclass(frozen=True)
class CollectionProbeSite(ProbeSite, ABC):
    """A probe on the element variable of a decoded collection."""

    decoder: type[CollectionDecoder]

    def reads_field(self, node: ast.AST) -> TypeGuard[ast.Attribute]:
        """Whether ``node`` reads an attribute of the probed variable."""

        match node:
            case ast.Attribute(value=ast.Name(id=name)):
                return name == self.variable
        return False

    def renamed_field_reads(
        self, authority: OptionalAstFieldAuthority, scope: ast.AST, skip: ast.AST
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """Read ``V.field`` as the present element's field everywhere in scope.

        Refuses a scope that reads an attribute of ``V`` the present element
        does not answer, or uses ``V`` itself as an ast node.
        """

        skipped = {id(node) for node in ast.walk(skip)}
        field_reads = tuple(
            node
            for node in ast.walk(scope)
            if id(node) not in skipped and self.reads_field(node)
        )
        read_owners = {id(node.value) for node in field_reads}
        for node in ast.walk(scope):
            match node:
                case ast.Name(id=name, ctx=ast.Load()) if (
                    name == self.variable
                    and id(node) not in skipped
                    and id(node) not in read_owners
                ):
                    raise authority.refusal(
                        node, f"`{self.variable}` is used as an ast node, not by field"
                    )
        replacements = []
        for node in field_reads:
            if node.attr not in self.decoder.present_attributes:
                raise authority.refusal(
                    node,
                    f"`{self.variable}.{node.attr}` has no answer on the present "
                    f"element of {self.decoder.collection_owner}",
                )
            if node.attr in self.decoder.field_names():
                _, end = authority.geometry.required_node_offsets(node)
                replacements.append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=end - len(node.attr),
                        end_offset=end,
                        replacement_source=self.decoder.present_field,
                    )
                )
        return tuple(replacements)


@dataclass(frozen=True)
class QuantifiedProbeSite(CollectionProbeSite):
    """``any(V.f is None for V in ITER)`` or ``all(V.f is not None ... for V in ITER)``."""

    call: ast.Call
    iterable: ast.expr
    remainder: tuple[ast.expr, ...]

    @classmethod
    def at(cls, probe, authority):
        parents = authority.parents
        decoder = authority.decoder
        generator = parents.get(probe)
        elt_probe = probe
        remainder: tuple[ast.expr, ...] = ()
        if isinstance(generator, ast.BoolOp) and isinstance(generator.op, ast.And):
            if generator.values[0] is not probe:
                return None
            remainder = tuple(generator.values[1:])
            elt_probe = generator
            generator = parents.get(generator)
        match generator:
            case ast.GeneratorExp(
                elt=elt,
                generators=[
                    ast.comprehension(target=ast.Name(id=name), iter=iterable, ifs=[])
                ],
            ) if (
                elt is elt_probe
            ):
                pass
            case _:
                return None
        call = parents.get(generator)
        match call:
            case ast.Call(func=ast.Name(id="any"), args=[only], keywords=[]) if (
                only is generator and not remainder
            ):
                variable = cls.probed(probe, decoder, absent=True)
            case ast.Call(func=ast.Name(id="all"), args=[only], keywords=[]) if (
                only is generator
            ):
                variable = cls.probed(probe, decoder, absent=False)
            case _:
                return None
        if variable != name or not authority.iterates_elements(iterable):
            return None
        return cls(decoder, probe, variable, call, iterable, remainder)

    def replacements(self, authority):
        geometry = authority.geometry
        start, end = geometry.required_node_offsets(self.call)
        iterable_source = authority.source_of(self.iterable)
        absent = f"{self.decoder.collected(iterable_source)}.{self.decoder.absent}"
        if isinstance(self.call.func, ast.Name) and self.call.func.id == "any":
            candidate = authority.truth_of(self.call, absent)
        elif not self.remainder:
            candidate = f"not {absent}"
        else:
            rest = " and ".join(authority.source_of(part) for part in self.remainder)
            candidate = (
                f"not {absent} and all({rest} for {self.variable} in "
                f"{iterable_source})"
            )
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise authority.refusal(self.call, "the quantifier holds a comment")
        replacement = authority.fit.fitted(candidate, start, end)
        if candidate.startswith("not ") and not authority.holds_boolean_operand(
            self.call
        ):
            replacement = f"({replacement})"
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start, end_offset=end, replacement_source=replacement
            ),
        )


@dataclass(frozen=True)
class ComprehensionFilterProbeSite(CollectionProbeSite):
    """``for V in ITER if V.f is not None [and REST]`` in a comprehension."""

    comprehension_node: ast.AST
    clause: ast.comprehension
    condition: ast.expr

    @classmethod
    def at(cls, probe, authority):
        parents = authority.parents
        decoder = authority.decoder
        condition: ast.AST = probe
        parent = parents.get(probe)
        if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.And):
            if parent.values[0] is not probe:
                return None
            condition = parent
            parent = parents.get(parent)
        if not isinstance(parent, ast.comprehension) or not any(
            clause_if is condition for clause_if in parent.ifs
        ):
            return None
        variable = cls.probed(probe, decoder, absent=False)
        match parent.target:
            case ast.Name(id=name) if name == variable and authority.iterates_elements(
                parent.iter
            ):
                return cls(decoder, probe, name, parents[parent], parent, condition)
        return None

    def replacements(self, authority):
        geometry = authority.geometry
        iterable_start, iterable_end = geometry.required_node_offsets(self.clause.iter)
        condition_start, condition_end = geometry.required_node_offsets(self.condition)
        if geometry.span_contains_comment(
            SourceTextSpan(iterable_start, condition_end)
        ):
            raise authority.refusal(self.probe, "the filter holds a comment")
        edits = [
            SourceTextSpanReplacement.from_offsets(
                start_offset=iterable_start,
                end_offset=iterable_end,
                replacement_source=(
                    f"{self.decoder.collected(authority.source_of(self.clause.iter))}"
                    f".{self.decoder.present}"
                ),
            )
        ]
        if self.condition is self.probe:
            removal_start = geometry.required_node_offsets(self.previous_part())[1]
            separator = geometry.source[removal_start:condition_start]
            if re.fullmatch(r"\s+if\s+", separator) is None:
                raise authority.refusal(self.probe, "the filter is not a bare `if`")
            edits.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=removal_start,
                    end_offset=condition_end,
                    replacement_source="",
                )
            )
        else:
            assert isinstance(self.condition, ast.BoolOp)
            next_start, _ = geometry.required_node_offsets(self.condition.values[1])
            edits.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=condition_start,
                    end_offset=next_start,
                    replacement_source="",
                )
            )
        edits.extend(
            self.renamed_field_reads(authority, self.comprehension_node, self.probe)
        )
        return (authority.fitted_rewrite(self.comprehension_node, edits),)

    def previous_part(self) -> ast.AST:
        """The node just before the probe's `if`: the iterable or an earlier filter."""

        index = next(
            index
            for index, clause_if in enumerate(self.clause.ifs)
            if clause_if is self.condition
        )
        return self.clause.ifs[index - 1] if index else self.clause.iter


@dataclass(frozen=True)
class LoopGuardProbeSite(CollectionProbeSite):
    """``for V in ITER:`` whose first statement is ``if V.f is None: continue|return R``."""

    loop: ast.For
    guard: ast.If

    @classmethod
    def at(cls, probe, authority):
        parents = authority.parents
        guard = parents.get(probe)
        loop = parents.get(guard) if guard is not None else None
        match (guard, loop):
            case (
                ast.If(test=test, body=[ast.Continue() | ast.Return()], orelse=[]),
                ast.For(target=ast.Name(id=name), iter=iterable, body=[first, _, *_]),
            ) if (
                test is probe and first is guard
            ):
                pass
            case _:
                return None
        variable = cls.probed(probe, authority.decoder, absent=True)
        if variable != name or not authority.iterates_elements(iterable):
            return None
        return cls(authority.decoder, probe, variable, loop, guard)

    def replacements(self, authority):
        geometry = authority.geometry
        guard_span = SourceTextSpan(
            *geometry.line_span_offsets(
                self.guard.lineno, self.guard.end_lineno or self.guard.lineno
            )
        )
        if geometry.span_contains_comment(guard_span):
            raise authority.refusal(self.guard, "the guard holds a comment")
        deleted = geometry.statement_deletion_span(
            SourceLineSpan(
                self.guard.lineno, self.guard.end_lineno or self.guard.lineno
            )
        )
        deletion_start, deletion_end = geometry.line_span_offsets(
            deleted.start_line, deleted.end_line
        )
        iterable_start, iterable_end = geometry.required_node_offsets(self.loop.iter)
        iterable_source = authority.source_of(self.loop.iter)
        edits = [
            SourceTextSpanReplacement.from_offsets(
                start_offset=deletion_start,
                end_offset=deletion_end,
                replacement_source="",
            )
        ]
        (exit_statement,) = self.guard.body
        if isinstance(exit_statement, ast.Continue):
            edits.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=iterable_start,
                    end_offset=iterable_end,
                    replacement_source=(
                        f"{self.decoder.collected(iterable_source)}.{self.decoder.present}"
                    ),
                )
            )
        else:
            collection = self.decoder.collection_variable
            if collection in authority.bound_names:
                raise authority.refusal(
                    self.loop, f"`{collection}` is already bound in the function"
                )
            loop_line_start = geometry.line_offsets[self.loop.lineno - 1]
            indentation = geometry.source[
                loop_line_start : loop_line_start + self.loop.col_offset
            ]
            exit_source = authority.source_of(exit_statement)
            edits.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=loop_line_start,
                    end_offset=loop_line_start,
                    replacement_source=(
                        f"{indentation}{collection} = "
                        f"{self.decoder.collected(iterable_source)}\n"
                        f"{indentation}if {collection}.{self.decoder.absent}:\n"
                        f"{indentation}    {exit_source}\n"
                    ),
                )
            )
            edits.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=iterable_start,
                    end_offset=iterable_end,
                    replacement_source=f"{collection}.{self.decoder.present}",
                )
            )
        for statement in self.loop.body[1:]:
            edits.extend(self.renamed_field_reads(authority, statement, self.probe))
        return tuple(edits)


@dataclass(frozen=True)
class SingleNodeProbeSite(ProbeSite, ABC):
    """A probe on one node whose ast class the function proves."""

    subject: ast.expr
    field: str

    @staticmethod
    def proved_subject(
        probe: ast.Compare, authority: OptionalAstFieldAuthority, absent: bool
    ) -> tuple[ast.expr, str] | None:
        """``(V, f)`` when the probe asks the given state of a proved node's field."""

        match probe:
            case ast.Compare(
                left=ast.Attribute(value=subject, attr=field),
                ops=[operator],
            ) if isinstance(operator, ast.Is if absent else ast.IsNot) and (
                authority.proves(subject, field, probe)
            ):
                return subject, field
        return None


@dataclass(frozen=True)
class IfPresentProbeSite(SingleNodeProbeSite):
    """``if [GUARD and] V.f is not None: BODY`` reading ``V.f`` in BODY.

    BODY runs once per value the decoder answers: ``for N in DECODED(V): BODY``
    with ``V.f`` read as ``N``.
    """

    statement: ast.If
    guards: tuple[ast.expr, ...]

    @classmethod
    def at(cls, probe, authority):
        parent = authority.parents.get(probe)
        guards: tuple[ast.expr, ...] = ()
        test: ast.expr = probe
        if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.And):
            if parent.values[-1] is not probe:
                return None
            guards = tuple(parent.values[:-1])
            test = parent
            parent = authority.parents.get(parent)
        match parent:
            case ast.If(test=found, orelse=[]) if found is test:
                pass
            case _:
                return None
        proved = cls.proved_subject(probe, authority, absent=False)
        if proved is None:
            return None
        subject, field = proved
        return cls(
            authority.decoder,
            probe,
            ast.unparse(subject),
            subject,
            field,
            parent,
            guards,
        )

    def replacements(self, authority):
        geometry = authority.geometry
        statement = self.statement
        name = self.decoder.present_name()
        if name in authority.bound_names or any(
            isinstance(node, ast.Name) and node.id == name
            for node in ast.walk(statement)
        ):
            raise authority.refusal(statement, f"`{name}` is already used here")
        start, _ = geometry.required_node_offsets(statement)
        if not geometry.source.startswith("if", start):
            raise authority.refusal(statement, "an `elif` cannot become a loop")
        if statement.body[0].lineno == statement.lineno:
            raise authority.refusal(statement, "the body shares the `if` line")
        for node in authority.escaping_loop_jumps(statement.body):
            raise authority.refusal(node, "the body jumps an enclosing loop")
        probed_names = {
            part.id for part in ast.walk(self.subject) if isinstance(part, ast.Name)
        }
        for node in ast.walk(statement):
            match node:
                case ast.Name(id=bound, ctx=ast.Store() | ast.Del()) if (
                    bound in probed_names
                ):
                    raise authority.refusal(node, "the body rebinds the probed node")
        header_end = geometry.line_offsets[statement.body[0].lineno - 1]
        if geometry.span_contains_comment(SourceTextSpan(start, header_end)):
            raise authority.refusal(statement, "the `if` header holds a comment")
        reads = [
            node
            for body_statement in statement.body
            for node in ast.walk(body_statement)
            if isinstance(node, ast.Attribute)
            and node.attr == self.field
            and _same_expression(node.value, self.subject)
        ]
        decoded = self.decoder.single(authority.source_of(self.subject))
        if not self.guards:
            _, test_end = geometry.required_node_offsets(statement.test)
            header = SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=test_end,
                replacement_source=f"for {name} in {decoded}",
            )
            return (
                header,
                *(authority.renamed(node, name) for node in reads),
            )
        return (self.nested(authority, name, decoded, reads),)

    def nested(
        self,
        authority: OptionalAstFieldAuthority,
        name: str,
        decoded: str,
        reads: list[ast.Attribute],
    ) -> SourceTextSpanReplacement:
        """``if GUARD:`` over a loop holding the body, indented one level deeper."""

        geometry = authority.geometry
        statement = self.statement
        if any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.end_lineno != node.lineno
            for node in ast.walk(statement)
        ):
            raise authority.refusal(statement, "the body holds a multi-line string")
        start, end = geometry.required_node_offsets(statement)
        line_start = geometry.line_offsets[statement.lineno - 1]
        indentation = geometry.source[line_start:start]
        body_start = geometry.line_offsets[statement.body[0].lineno - 1]
        body = geometry.source[body_start:end]
        for node in sorted(
            reads,
            key=lambda read: geometry.required_node_offsets(read)[0],
            reverse=True,
        ):
            read_start, read_end = geometry.required_node_offsets(node)
            body = (
                body[: read_start - body_start] + name + body[read_end - body_start :]
            )
        guard = " and ".join(
            (
                f"({authority.source_of(part)})"
                if isinstance(part, ast.BoolOp | ast.NamedExpr | ast.IfExp | ast.Lambda)
                else authority.source_of(part)
            )
            for part in self.guards
        )
        indented = "".join(
            f"    {line}" if line.strip() else line
            for line in body.splitlines(keepends=True)
        )
        rewritten = (
            f"if {guard}:\n{indentation}    for {name} in {decoded}:\n{indented}"
        )
        if not authority.fit.fits(indentation, rewritten, ""):
            raise authority.refusal(statement, "the nested loop overflows the line")
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start, end_offset=end, replacement_source=rewritten
        )


@dataclass(frozen=True)
class SpliceProbeSite(SingleNodeProbeSite):
    """``*((V.f,) if V.f is not None else ())`` or ``() if V.f is None else (V.f,)``."""

    expression: ast.IfExp
    starred: bool

    @classmethod
    def at(cls, probe, authority):
        expression = authority.parents.get(probe)
        if not isinstance(expression, ast.IfExp) or expression.test is not probe:
            return None
        present = cls.proved_subject(probe, authority, absent=False)
        absent = cls.proved_subject(probe, authority, absent=True)
        if present is not None:
            subject, field = present
            kept, dropped = expression.body, expression.orelse
        elif absent is not None:
            subject, field = absent
            kept, dropped = expression.orelse, expression.body
        else:
            return None
        match (kept, dropped):
            case (ast.Tuple(elts=[only]), ast.Tuple(elts=[])) if _same_expression(
                only, probe.left
            ):
                pass
            case _:
                return None
        return cls(
            authority.decoder,
            probe,
            ast.unparse(subject),
            subject,
            field,
            expression,
            isinstance(authority.parents.get(expression), ast.Starred),
        )

    @property
    def node(self) -> ast.AST:
        return self.expression

    def replacements(self, authority):
        decoded = self.decoder.single(authority.source_of(self.subject))
        replaced: ast.AST = (
            authority.parents[self.expression] if self.starred else self.expression
        )
        start, end = authority.geometry.required_node_offsets(replaced)
        if authority.geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise authority.refusal(self.expression, "the splice holds a comment")
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=f"*{decoded}" if self.starred else decoded,
            ),
        )


@dataclass(frozen=True)
class PairedProbeSite(SingleNodeProbeSite):
    """Both fields of a pair asked together of one node.

    ``V.a is None and V.b is None`` reads ``not DECODED(V)``; ``V.a is not None
    or V.b is not None`` reads ``DECODED(V)``; adjacent splices
    ``*((V.a,) if ...), *((V.b,) if ...)`` read ``*DECODED(V)``.
    """

    first: ast.expr
    second: ast.expr
    joined: str

    @property
    def anchor(self) -> ast.AST:
        return self.first

    @classmethod
    def at(cls, probe, authority):
        parent = authority.parents.get(probe)
        match parent:
            case ast.BoolOp(op=ast.And() | ast.Or() as operator, values=values):
                absent = isinstance(operator, ast.And)
                proved = cls.proved_subject(probe, authority, absent=absent)
                return cls.adjacent(
                    authority, probe, proved, values, probe, "and" if absent else "or"
                )
            case ast.IfExp(test=test) if test is probe:
                splice = SpliceProbeSite.at(probe, authority)
                container = authority.parents.get(authority.parents.get(parent))
                if splice is None or not splice.starred:
                    return None
                match container:
                    case (
                        ast.Tuple(elts=items)
                        | ast.List(elts=items)
                        | ast.Set(elts=items)
                        | ast.Call(args=items)
                    ):
                        return cls.adjacent(
                            authority,
                            probe,
                            (splice.subject, splice.field),
                            items,
                            authority.parents[parent],
                            "*",
                        )
        return None

    @classmethod
    def adjacent(
        cls,
        authority: OptionalAstFieldAuthority,
        probe: ast.Compare,
        proved: tuple[ast.expr, str] | None,
        items: list[ast.expr],
        item: ast.AST,
        joined: str,
    ) -> Self | None:
        """The site when ``item``'s pair partner sits right next to it, in order."""

        if proved is None:
            return None
        subject, field = proved
        first_field, second_field = authority.decoder.paired_fields
        index = next(index for index, value in enumerate(items) if value is item)
        offset = 1 if field == first_field else -1
        if not 0 <= index + offset < len(items):
            return None
        partner = items[index + offset]
        partner_probe = (
            partner.value.test
            if isinstance(partner, ast.Starred) and isinstance(partner.value, ast.IfExp)
            else partner
        )
        match partner_probe:
            case ast.Compare(
                left=ast.Attribute(value=other, attr=other_field), ops=[operator]
            ) if (
                other_field == (second_field if offset == 1 else first_field)
                and _same_expression(other, subject)
                and type(operator) is type(probe.ops[0])
                and _none_probe(partner_probe)
            ):
                pass
            case _:
                return None
        if joined == "*" and SpliceProbeSite.at(partner_probe, authority) is None:
            return None
        first, second = (item, partner) if offset == 1 else (partner, item)
        return cls(
            authority.decoder,
            probe,
            ast.unparse(subject),
            subject,
            field,
            first,
            second,
            joined,
        )

    def replacements(self, authority):
        geometry = authority.geometry
        start, _ = geometry.required_node_offsets(self.first)
        _, end = geometry.required_node_offsets(self.second)
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise authority.refusal(self.probe, "the pair holds a comment")
        decoded = self.decoder.single(authority.source_of(self.subject))
        boolean = authority.parents[self.first]
        whole = isinstance(boolean, ast.BoolOp) and len(boolean.values) == 2
        match self.joined:
            case "*":
                replacement = f"*{decoded}"
                if (
                    isinstance(boolean, ast.Tuple)
                    and len(boolean.elts) == 2
                    and not geometry.source[end:].lstrip().startswith(",")
                ):
                    replacement += ","
            case "and":
                replacement = f"not {decoded}"
                if whole and not authority.holds_boolean_operand(boolean):
                    replacement = f"({replacement})"
            case _:
                replacement = authority.truth_of(boolean, decoded)
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start, end_offset=end, replacement_source=replacement
            ),
        )


# --- decoders ---------------------------------------------------------------


class OptionalAstFieldDecoder(ABC):
    """One registered decoder: the ast fields it owns and the shapes sites take.

    ``fields`` maps each ast class to its optional field. ``owners`` are the
    objects this package declares as the decoder; a project without them gets
    their source generated.
    """

    registry: ClassVar[dict[str, type[OptionalAstFieldDecoder]]] = {}
    fields: ClassVar[Mapping[type[ast.AST], str]]
    owners: ClassVar[tuple[type | object, ...]]
    shapes: ClassVar[tuple[type[ProbeSite], ...]]
    collection_elements: ClassVar[Mapping[str, type[ast.AST]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "fields" in cls.__dict__:
            OptionalAstFieldDecoder.registry[cls.__name__] = cls

    @classmethod
    def for_field(cls, field: str) -> tuple[type[OptionalAstFieldDecoder], ...]:
        return tuple(
            decoder
            for decoder in cls.registry.values()
            if field in decoder.field_names()
        )

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(cls.fields.values())

    @classmethod
    def classes_with(cls, field: str) -> frozenset[str]:
        """The ast classes on which ``field`` is the optional field this decoder owns."""

        return frozenset(
            node_type.__name__
            for node_type, owned in cls.fields.items()
            if owned == field
        )

    @classmethod
    def present_name(cls) -> str:
        """The name a decoded value is read under: the decoder's own, in snake case."""

        return re.sub(
            r"(?<!^)(?=[A-Z])", "_", cls.__name__.removesuffix("Decoder")
        ).lower()

    @classmethod
    def owner_names(cls) -> tuple[str, ...]:
        return tuple(owner.__name__ for owner in cls.owners)

    @classmethod
    @abstractmethod
    def single(cls, subject_source: str) -> str:
        """The values one node's field holds: none when absent."""
        raise NotImplementedError


class CollectionDecoder(OptionalAstFieldDecoder, ABC):
    """A decoder that also splits a collection into present and absent elements."""

    collection_owner: ClassVar[str]
    present: ClassVar[str]
    absent: ClassVar[str]
    present_field: ClassVar[str]
    present_attributes: ClassVar[frozenset[str]]
    collection_variable: ClassVar[str]

    @classmethod
    def collected(cls, iterable_source: str) -> str:
        return f"{cls.collection_owner}.of({iterable_source})"


class KeywordNameDecoder(CollectionDecoder):
    """``ast.keyword.arg``: a named keyword or a ``**`` splat (#75, f580f61)."""

    fields = {ast.keyword: "arg"}
    owners = (
        ast_projection.CallKeyword,
        *ast_projection.CallKeyword.__subclasses__(),
        ast_projection.CallKeywords,
    )
    shapes = (
        *CollectionProbeSite.__subclasses__(),
        IfPresentProbeSite,
        SpliceProbeSite,
    )
    collection_elements = {"keywords": ast.keyword}
    collection_owner = "CallKeywords"
    present = "named"
    absent = "splats"
    present_field = "name"
    present_attributes = frozenset(("arg", "value", "node"))
    collection_variable = "keywords"

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"CallKeyword.decoded({subject_source}).names"


class CaptureNameDecoder(OptionalAstFieldDecoder):
    """The name an except handler or capture pattern binds itself."""

    fields = {
        ast.ExceptHandler: "name",
        ast.MatchAs: "name",
        ast.MatchStar: "name",
        ast.MatchMapping: "rest",
    }
    owners = (ast_field_decoders.capture_names,)
    shapes = (IfPresentProbeSite, SpliceProbeSite)

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"capture_names({subject_source})"


class DeclaredAnnotationDecoder(OptionalAstFieldDecoder):
    """The annotation a parameter declares, or the return annotation of a function."""

    fields = {
        ast.arg: "annotation",
        ast.FunctionDef: "returns",
        ast.AsyncFunctionDef: "returns",
    }
    owners = (ast_field_decoders.declared_annotations,)
    shapes = (IfPresentProbeSite, SpliceProbeSite)

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"declared_annotations({subject_source})"


class VariadicParameterDecoder(OptionalAstFieldDecoder):
    """A signature's ``*args`` and ``**kwargs`` parameters, asked together."""

    fields = {ast.arguments: "vararg"}
    paired_fields: ClassVar[tuple[str, str]] = ("vararg", "kwarg")
    owners = (ast_field_decoders.variadic_parameters,)
    shapes = (PairedProbeSite,)

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(cls.paired_fields)

    @classmethod
    def classes_with(cls, field: str) -> frozenset[str]:
        return frozenset(("arguments",)) if field in cls.paired_fields else frozenset()

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"variadic_parameters({subject_source})"


# --- owners in a project ----------------------------------------------------


@dataclass(frozen=True)
class ProjectDecoderOwners:
    """Where one root package declares each decoder owner, and what it lacks.

    Owners declared outside the generated module are imported from where they
    are. The rest live in the generated module, whose source is copied from the
    owners this package declares.
    """

    snapshot: CodemodSourceSnapshot
    root: str

    @classmethod
    def of(cls, snapshot: CodemodSourceSnapshot, module_name: str) -> Self:
        root = module_name.partition(".")[0]
        return _PROJECT_OWNERS.for_snapshot(snapshot).setdefault(
            root, cls(snapshot, root)
        )

    @property
    def generated_module(self) -> str:
        return f"{self.root}.{GENERATED_MODULE}"

    @cached_property
    def top_level_names(self) -> Mapping[str, frozenset[str]]:
        """Module -> the owner names it declares at top level, in this root package."""

        names = {
            name
            for decoder in OptionalAstFieldDecoder.registry.values()
            for name in decoder.owner_names()
        }
        return {
            parsed.module_name: frozenset(
                statement.name
                for statement in parsed.module.body
                if isinstance(statement, ast.ClassDef | ast.FunctionDef)
                and statement.name in names
            )
            for parsed in self.snapshot.parsed_modules
            if parsed.module_name.partition(".")[0] == self.root
        }

    @cached_property
    def declared(self) -> Mapping[str, str]:
        """Owner name -> the one hand-declared module of this package defining it."""

        found: dict[str, list[str]] = {}
        for module, names in self.top_level_names.items():
            if module == self.generated_module:
                continue
            for name in names:
                found.setdefault(name, []).append(module)
        return {
            name: modules[0] for name, modules in found.items() if len(modules) == 1
        }

    @cached_property
    def pending(self) -> tuple[type[OptionalAstFieldDecoder], ...]:
        """Decoders the package probes whose owners only generation can supply."""

        probed = {
            node.left.attr
            for parsed in self.snapshot.parsed_modules
            if parsed.module_name.partition(".")[0] == self.root
            for node in ast.walk(parsed.module)
            if _none_probe(node)
        }
        return tuple(
            decoder
            for decoder in OptionalAstFieldDecoder.registry.values()
            if not all(name in self.declared for name in decoder.owner_names())
            and probed & decoder.field_names()
        )

    def generated_path(self) -> str:
        package = self.snapshot.module_import_graph.unique_source_file_for_module_name(
            self.root
        )
        if package is None or Path(package.file_path).name != "__init__.py":
            raise ValueError(
                f"`{self.root}` is not a package to generate {GENERATED_MODULE} in"
            )
        return str(Path(package.file_path).parent / f"{GENERATED_MODULE}.py")

    @property
    def needs_declaration(self) -> bool:
        """Whether a pending decoder has no generated module to live in yet."""

        return bool(self.pending) and (
            self.generated_path() not in self.snapshot.sources_by_file_path
        )

    def generated_source(self) -> str:
        """The pending decoders, copied from the owners this package declares."""

        body = "\n\n".join(
            inspect.getsource(owner)
            for decoder in self.pending
            for owner in decoder.owners
        )
        imports = [
            line
            for line, used in (
                ("import ast", "ast."),
                ("from abc import ABC, abstractmethod", "(ABC)"),
                ("from collections.abc import Iterable, Mapping", "Iterable["),
                ("from dataclasses import dataclass", "@dataclass"),
            )
            if used in body
        ]
        return (
            '"""The one boundary read of each optional ast field (generated by NRA)."""'
            "\n\nfrom __future__ import annotations\n\n"
            + "\n".join(imports)
            + "\n\n\n"
            + body.rstrip()
            + "\n"
        )

    def import_sources_for(
        self, parsed_module: ParsedModule, names: Iterable[str]
    ) -> tuple[str, ...]:
        """One import per owner name the rewrite reads, none inside its own module.

        A name the package does not declare must already be in the generated
        module: routing to an owner nothing declares would import nothing.
        """

        sources = []
        for name in names:
            module = self.declared.get(name, self.generated_module)
            if name not in self.top_level_names.get(module, frozenset()):
                raise ValueError(
                    f"`{name}` is not declared under `{self.root}`; the recipe "
                    f"declaring {self.generated_module} routes this function"
                )
            if module != parsed_module.module_name:
                sources.append(
                    ImportedClassReference(f"{module}.{name}").import_source_for(
                        self.snapshot, parsed_module
                    )
                )
        return tuple(sources)


CachedT = TypeVar("CachedT")


@dataclass
class SnapshotCache(Generic[CachedT]):
    """Per-root values for the one snapshot synthesis is reading now."""

    snapshot: CodemodSourceSnapshot | None = None
    values: dict[str, CachedT] = field(default_factory=dict)

    def for_snapshot(self, snapshot: CodemodSourceSnapshot) -> dict[str, CachedT]:
        if self.snapshot is not snapshot:
            self.snapshot = snapshot
            self.values = {}
        return self.values


_PROJECT_OWNERS = SnapshotCache[ProjectDecoderOwners]()


# --- authority --------------------------------------------------------------


@dataclass(frozen=True)
class OptionalAstFieldAuthority(FunctionRewriteAuthority):
    """Every routable probe of one decoder's fields inside one function."""

    decoder: type[OptionalAstFieldDecoder]
    rewrite_name = "Probe routing"

    @cached_property
    def parents(self) -> dict[ast.AST, ast.AST]:
        return {
            child: parent
            for parent in ast.walk(self.function)
            for child in ast.iter_child_nodes(parent)
        }

    @cached_property
    def bound_names(self) -> frozenset[str]:
        return frozenset(
            node.id
            for node in ast.walk(self.function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        ) | frozenset(
            argument.arg
            for argument in ast.walk(self.function.args)
            if isinstance(argument, ast.arg)
        )

    def ancestors(self, node: ast.AST) -> Iterator[tuple[ast.AST, ast.AST]]:
        """``(child, parent)`` pairs from the node up to the function."""

        child = node
        while (parent := self.parents.get(child)) is not None:
            yield child, parent
            child = parent

    # proof of a node's ast class ------------------------------------------

    @staticmethod
    def ast_classes(annotation: ast.expr) -> frozenset[str]:
        """The ``ast.X`` classes an annotation or isinstance target names, all or none."""

        match annotation:
            case ast.Attribute(value=ast.Name(id="ast"), attr=name):
                return frozenset((name,))
            case ast.BinOp(left=left, op=ast.BitOr(), right=right):
                left_classes = OptionalAstFieldAuthority.ast_classes(left)
                right_classes = OptionalAstFieldAuthority.ast_classes(right)
                if left_classes and right_classes:
                    return left_classes | right_classes
            case ast.Tuple(elts=items) if items:
                parts = [OptionalAstFieldAuthority.ast_classes(item) for item in items]
                if all(parts):
                    return frozenset().union(*parts)
        return frozenset()

    def annotated(self, name: str) -> frozenset[str]:
        arguments = self.function.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            if argument.arg == name:
                return frozenset().union(
                    *map(
                        self.ast_classes,
                        ast_field_decoders.declared_annotations(argument),
                    )
                )
        return frozenset()

    def guarded(self, subject: ast.expr, at: ast.AST) -> frozenset[str]:
        """Classes an enclosing ``isinstance(V, ...)`` guard proves at ``at``."""

        for child, parent in self.ancestors(at):
            match parent:
                case ast.BoolOp(op=ast.And(), values=values):
                    index = next(i for i, value in enumerate(values) if value is child)
                    tests = values[:index]
                case ast.If(test=test, body=body) if any(
                    statement is child for statement in body
                ):
                    tests = (
                        test.values
                        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
                        else [test]
                    )
                case _:
                    continue
            for test in tests:
                match test:
                    case ast.Call(
                        func=ast.Name(id="isinstance"), args=[checked, classes]
                    ) if _same_expression(checked, subject):
                        if found := self.ast_classes(classes):
                            return found
        return frozenset()

    def iterated(self, name: str, at: ast.AST) -> frozenset[str]:
        """Classes proved by iterating a decoder's collection attribute into ``name``."""

        elements = self.decoder.collection_elements
        for child, parent in self.ancestors(at):
            match parent:
                case ast.For(target=ast.Name(id=target), iter=ast.Attribute(attr=attr)):
                    if target == name and attr in elements and child is not parent.iter:
                        return frozenset((elements[attr].__name__,))
                case (
                    ast.ListComp() | ast.SetComp() | ast.GeneratorExp() | ast.DictComp()
                ):
                    for generator in parent.generators:
                        match generator:
                            case ast.comprehension(
                                target=ast.Name(id=target),
                                iter=ast.Attribute(attr=attr),
                            ) if (
                                target == name and attr in elements
                            ):
                                return frozenset((elements[attr].__name__,))
        return frozenset()

    def classes_of(self, subject: ast.expr, at: ast.AST) -> frozenset[str]:
        match subject:
            case ast.Name(id=name):
                return (
                    self.guarded(subject, at)
                    or self.annotated(name)
                    or self.iterated(name, at)
                )
            case ast.Attribute(value=owner, attr="args"):
                owner_classes = self.guarded(subject, at) or self.classes_of(owner, at)
                if owner_classes == frozenset(("arguments",)):
                    return owner_classes
                if owner_classes and owner_classes <= FUNCTION_CLASSES:
                    return frozenset(("arguments",))
        return self.guarded(subject, at)

    def proves(self, subject: ast.expr, field: str, at: ast.AST) -> bool:
        """Whether ``subject`` is proved to be an ast class owning ``field``."""

        classes = self.classes_of(subject, at)
        return bool(classes) and classes <= self.decoder.classes_with(field)

    def iterates_elements(self, iterable: ast.expr) -> bool:
        """Whether the iterable is proved a collection of the decoder's elements."""

        elements = self.decoder.collection_elements
        match iterable:
            case ast.Attribute(attr=attribute):
                return attribute in elements
            case ast.Name(id=name):
                arguments = self.function.args
                return any(
                    argument.arg == name
                    and any(
                        f"ast.{element.__name__}" in ast.unparse(argument)
                        for element in elements.values()
                    )
                    for argument in (
                        *arguments.posonlyargs,
                        *arguments.args,
                        *arguments.kwonlyargs,
                    )
                )
        return False

    # source -------------------------------------------------------------

    def escaping_loop_jumps(self, body: list[ast.stmt]) -> Iterator[ast.AST]:
        """``break``/``continue`` in ``body`` that leave it for an enclosing loop."""

        def walk(node: ast.AST) -> Iterator[ast.AST]:
            match node:
                case ast.Break() | ast.Continue():
                    yield node
                case (
                    ast.For(orelse=orelse)
                    | ast.AsyncFor(orelse=orelse)
                    | ast.While(orelse=orelse)
                ):
                    for part in orelse:
                        yield from walk(part)
                case (
                    ast.FunctionDef()
                    | ast.AsyncFunctionDef()
                    | ast.Lambda()
                    | ast.ClassDef()
                ):
                    pass
                case _:
                    for child in ast.iter_child_nodes(node):
                        yield from walk(child)

        for statement in body:
            yield from walk(statement)

    def holds_boolean_operand(self, node: ast.AST) -> bool:
        """Whether a `not`/`and` expression can replace the node without parentheses.

        Boolean operators bind looser than comparisons, arithmetic and
        attribute access, so only a statement slot or another boolean
        operator takes them bare.
        """

        parent = self.parents.get(node)
        return isinstance(
            parent,
            ast.Assign
            | ast.AnnAssign
            | ast.Return
            | ast.If
            | ast.While
            | ast.Assert
            | ast.BoolOp
            | ast.comprehension
            | ast.keyword
            | ast.Expr,
        ) or (isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not))

    def only_truth_tested(self, node: ast.AST) -> bool:
        """Whether the node's value is only tested for truth, never kept.

        There a collection answers for its own truth and needs no `bool()`.
        """

        parent = self.parents.get(node)
        match parent:
            case (
                ast.If(test=test)
                | ast.While(test=test)
                | ast.Assert(test=test)
                | ast.IfExp(test=test)
            ):
                return test is node
            case ast.comprehension(ifs=ifs):
                return any(clause is node for clause in ifs)
            case ast.UnaryOp(op=ast.Not()):
                return True
            case ast.BoolOp():
                return self.only_truth_tested(parent)
        return False

    def truth_of(self, node: ast.AST, collection: str) -> str:
        """The collection standing for ``node``'s truth: bare where only tested."""

        return collection if self.only_truth_tested(node) else f"bool({collection})"

    def renamed(self, node: ast.AST, name: str) -> SourceTextSpanReplacement:
        start, end = self.geometry.required_node_offsets(node)
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start, end_offset=end, replacement_source=name
        )

    def fitted_rewrite(
        self, node: ast.AST, edits: Iterable[SourceTextSpanReplacement]
    ) -> SourceTextSpanReplacement:
        """One replacement of the node with its inner edits applied, fit to the line."""

        start, end = self.geometry.required_node_offsets(node)
        rewritten = self.geometry.source[start:end]
        for edit in sorted(edits, key=lambda item: item.start_offset, reverse=True):
            rewritten = (
                rewritten[: edit.start_offset - start]
                + edit.replacement_source
                + rewritten[edit.end_offset - start :]
            )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start,
            end_offset=end,
            replacement_source=self.fit.fitted(rewritten, start, end),
        )

    def source_of(self, node: ast.AST) -> str:
        start, end = self.geometry.required_node_offsets(node)
        return self.geometry.source[start:end]

    def refusal(self, node: ast.AST, reason: str) -> ValueError:
        return handback(self.parsed_module, node, reason)

    # sites ----------------------------------------------------------------

    def probes(self) -> Iterator[ast.Compare]:
        fields = self.decoder.field_names()
        for node in ast.walk(self.function):
            if _none_probe(node) and node.left.attr in fields:
                yield node

    def sites(self) -> tuple[tuple[ast.Compare, ProbeSite | None], ...]:
        found: list[tuple[ast.Compare, ProbeSite | None]] = []
        anchors: set[tuple[type[ProbeSite], int]] = set()
        for probe in self.probes():
            site = next(
                (
                    site
                    for shape in self.decoder.shapes
                    if (site := shape.at(probe, self)) is not None
                ),
                None,
            )
            if site is not None:
                anchor = (type(site), id(site.anchor))
                if anchor in anchors:
                    continue
                anchors.add(anchor)
            found.append((probe, site))
        return tuple(found)

    def replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        """Edits for every routable probe; a function with none is handed back."""

        replacements: list[SourceTextSpanReplacement] = []
        refusals: list[ValueError] = []
        for probe, site in self.sites():
            if site is None:
                refusals.append(
                    self.refusal(
                        probe,
                        f"`{ast.unparse(probe)}` is in no shape the "
                        f"{self.decoder.__name__} can carry",
                    )
                )
                continue
            try:
                replacements.extend(site.replacements(self))
            except ValueError as error:
                refusals.append(error)
        if not replacements:
            raise (
                refusals[0]
                if refusals
                else ValueError(
                    f"No probe of {', '.join(sorted(self.decoder.field_names()))} "
                    f"for {self.decoder.__name__} in {self.function.name!r}"
                )
            )
        return tuple(sorted(replacements, key=lambda item: item.start_offset))


# --- operations -------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class RouteOptionalAstFieldProbesOperation(SourceReprovedOperation):
    """Rewrite one function's probes of an optional ast field onto its decoder."""

    decoder: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    @property
    def decoder_type(self) -> type[OptionalAstFieldDecoder]:
        decoder = OptionalAstFieldDecoder.registry.get(self.decoder)
        if decoder is None:
            raise ValueError(f"Unknown optional ast field decoder {self.decoder!r}")
        return decoder

    def owners(self, snapshot: CodemodSourceSnapshot) -> ProjectDecoderOwners:
        _, target, _ = self.target_node_from_context(snapshot)
        return ProjectDecoderOwners.of(
            snapshot,
            snapshot.parsed_module_for_source_path(target.file_path).module_name,
        )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        parsed_module = snapshot.parsed_module_for_source_path(target.file_path)
        decoder = self.decoder_type
        authority = OptionalAstFieldAuthority(
            parsed_module=parsed_module,
            geometry=SourceTextGeometry(
                snapshot.sources_by_file_path[target.file_path]
            ),
            line_length=DeclaredRoot.line_length_for(Path(target.file_path)),
            declaration=node,
            decoder=decoder,
        )
        replacements = authority.replacements()
        used = tuple(
            name
            for name in decoder.owner_names()
            if any(
                re.search(rf"\b{name}\b", replacement.replacement_source)
                for replacement in replacements
            )
        )
        imports = self.owners(snapshot).import_sources_for(parsed_module, used)
        return (
            *authority.geometry.physical_edits(
                file_path=target.file_path,
                replacements=replacements,
                rationale=self.rationale
                or f"Route {decoder.__name__} probes in {target.qualname!r}.",
            ),
            *(
                mutation
                for import_source in imports
                for mutation in self.required_import_mutations(
                    target.file_path,
                    import_source=import_source,
                    default_rationale=f"Import {decoder.__name__}'s owner.",
                )
            ),
        )


@dataclass(frozen=True, kw_only=True)
class DeclareAndRouteOptionalAstFieldProbesOperation(
    RouteOptionalAstFieldProbesOperation
):
    """Create a package's generated decoders, then route its first consumer.

    The rest of the package's consumers are routed in the same recipe, so the
    new module never exists without the code that reads it.
    """

    def creation(self, snapshot: CodemodSourceSnapshot) -> SourceFileCreation:
        owners = self.owners(snapshot)
        return SourceFileCreation.from_operation(
            self,
            requested_path=owners.generated_path(),
            source_index=snapshot.source_index,
            source=owners.generated_source(),
        )

    def source_file_creations(
        self, context: CodemodSelectorContext
    ) -> tuple[SourceFileCreation, ...]:
        snapshot = context.execution_snapshot()
        if not self.owners(snapshot).needs_declaration:
            return ()
        return (self.creation(snapshot),)

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        return (
            self.creation(snapshot),
            *super().source_edits_from_snapshot(snapshot),
        )


# --- synthesis --------------------------------------------------------------


@dataclass(frozen=True)
class PendingDecoderMigration:
    """A package lacking decoders: one recipe declares them and routes every consumer."""

    owners: ProjectDecoderOwners

    @classmethod
    def of(cls, owners: ProjectDecoderOwners) -> PendingDecoderMigration:
        return _MIGRATIONS.for_snapshot(owners.snapshot).setdefault(
            owners.root, cls(owners)
        )

    @cached_property
    def declared_snapshot(self) -> CodemodSourceSnapshot:
        """Current source with the generated module in place."""

        return self.owners.snapshot.with_virtual_sources(
            {self.owners.generated_path(): self.owners.generated_source()}
        )

    @cached_property
    def consumers(self) -> tuple[RoutedConsumer | Unroutable, ...]:
        """Every consumer probing a pending decoder's field, routed or refused."""

        snapshot = self.declared_snapshot
        found: list[RoutedConsumer | Unroutable] = []
        for parsed in sorted(
            self.owners.snapshot.parsed_modules, key=lambda module: str(module.path)
        ):
            if parsed.module_name.partition(".")[0] != self.owners.root:
                continue
            for qualname, function in _functions(parsed.module):
                for decoder in self.owners.pending:
                    if any(
                        _none_probe(node) and node.left.attr in decoder.field_names()
                        for node in ast.walk(function)
                    ):
                        found.append(
                            RoutedConsumer.attempt(
                                snapshot, str(parsed.path), qualname, decoder
                            )
                        )
        return tuple(found)

    @cached_property
    def routes(self) -> tuple[RoutedConsumer, ...]:
        return tuple(
            consumer
            for consumer in self.consumers
            if isinstance(consumer, RoutedConsumer)
        )

    def refusal_for(self, file_path: str, symbol: str) -> str:
        """Why this function is not in the declaring recipe, or who routes it."""

        reasons = [
            consumer.reason
            for consumer in self.consumers
            if isinstance(consumer, Unroutable)
            and (consumer.file_path, consumer.symbol) == (file_path, symbol)
        ]
        if reasons:
            return "; ".join(reasons)
        return (
            f"routed by the recipe declaring {self.owners.generated_module} "
            f"({self.holder})"
        )

    def recipe_operations(
        self, file_path: str, symbol: str
    ) -> tuple[RouteOptionalAstFieldProbesOperation, ...]:
        """Every routing, declaring on the first, when this function is the first."""

        if not self.routes or self.routes[0].key != (file_path, symbol):
            return ()
        holder, *rest = self.routes
        declaring = _routing(
            self.owners.snapshot,
            holder.file_path,
            holder.symbol,
            holder.decoder,
            DeclareAndRouteOptionalAstFieldProbesOperation,
        )
        return (
            declaring,
            *(
                _routing(
                    self.owners.snapshot,
                    consumer.file_path,
                    consumer.symbol,
                    consumer.decoder,
                )
                for consumer in rest
            ),
        )

    @property
    def holder(self) -> str:
        if not self.routes:
            return "no routable consumer"
        return f"{Path(self.routes[0].file_path).name}:{self.routes[0].symbol}"


_MIGRATIONS = SnapshotCache[PendingDecoderMigration]()


@dataclass(frozen=True)
class MigrationConsumer(ABC):
    """One function probing a pending decoder's field."""

    file_path: str
    symbol: str
    decoder: type[OptionalAstFieldDecoder]

    @property
    def key(self) -> tuple[str, str]:
        return self.file_path, self.symbol


@dataclass(frozen=True)
class RoutedConsumer(MigrationConsumer):
    """A consumer whose routing is proved once the decoders are declared."""

    @staticmethod
    def attempt(
        snapshot: CodemodSourceSnapshot,
        file_path: str,
        symbol: str,
        decoder: type[OptionalAstFieldDecoder],
    ) -> RoutedConsumer | Unroutable:
        """Prove the routing, or name why it is handed back.

        Handbacks and reproof refusals arrive as ValueError; anything else is
        a defect and propagates.
        """

        try:
            _routing(snapshot, file_path, symbol, decoder).source_edits_from_snapshot(
                snapshot
            )
        except ValueError as error:
            return Unroutable(file_path, symbol, decoder, str(error))
        return RoutedConsumer(file_path, symbol, decoder)


@dataclass(frozen=True)
class Unroutable(MigrationConsumer):
    """A consumer the migration hands back, with the reason."""

    reason: str


def _routing(
    snapshot: CodemodSourceSnapshot,
    file_path: str,
    symbol: str,
    decoder: type[OptionalAstFieldDecoder],
    operation_type: type[
        RouteOptionalAstFieldProbesOperation
    ] = RouteOptionalAstFieldProbesOperation,
) -> RouteOptionalAstFieldProbesOperation:
    (target_id,) = SourceIndexTargetSelector.for_function_or_method(
        file_path, symbol
    ).target_ids(snapshot)
    return operation_type(
        target=SourceRewriteTarget(target_id=target_id), decoder=decoder.__name__
    )


def _functions(
    module: ast.Module,
) -> Iterator[tuple[str, FunctionNode]]:
    """Every function with its dotted qualname, in source order."""

    def walk(node: ast.AST, prefix: str) -> Iterator[tuple[str, FunctionNode]]:
        for child in ast.iter_child_nodes(node):
            match child:
                case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                    yield f"{prefix}{name}", child
                    yield from walk(child, f"{prefix}{name}.")
                case ast.ClassDef(name=name):
                    yield from walk(child, f"{prefix}{name}.")

    yield from walk(module, "")


class OptionalAstFieldDecoderFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    OptionalAstFieldDecoderConcept,
    ABC,
):
    """One routing per evidenced function whose probes a registered decoder owns.

    In a package lacking a decoder, the first routable consumer's recipe
    declares the decoder and routes every consumer; the others point at it.
    """

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "optional ast field routing requires source context"
            )
        snapshot = context.execution_snapshot()
        operations: dict[tuple[str, str, str], RouteOptionalAstFieldProbesOperation] = (
            {}
        )
        refusals: list[str] = []
        probed_fields: set[str] = set()
        for evidence in finding.evidence:
            parsed_module = snapshot.parsed_module_for_source_path(evidence.file_path)
            fields = {
                probe.left.attr
                for probe in ast.walk(parsed_module.module)
                if _none_probe(probe) and probe.lineno == evidence.line
            }
            probed_fields |= fields
            owners = ProjectDecoderOwners.of(snapshot, parsed_module.module_name)
            for field_name in sorted(fields):
                for decoder in OptionalAstFieldDecoder.for_field(field_name):
                    try:
                        if decoder in owners.pending and owners.needs_declaration:
                            migration = PendingDecoderMigration.of(owners)
                            migrated = migration.recipe_operations(
                                evidence.file_path, evidence.symbol
                            )
                            if not migrated:
                                refusals.append(
                                    f"{decoder.__name__}: "
                                    + migration.refusal_for(
                                        evidence.file_path, evidence.symbol
                                    )
                                )
                            for operation in migrated:
                                operations[
                                    (
                                        operation.target.target_id or "",
                                        operation.decoder,
                                        type(operation).__name__,
                                    )
                                ] = operation
                            continue
                        operation = _routing(
                            snapshot, evidence.file_path, evidence.symbol, decoder
                        )
                        operation.source_edits_from_snapshot(snapshot)
                    except ValueError as error:
                        refusals.append(str(error))
                        continue
                    operations[
                        (evidence.file_path, evidence.symbol, decoder.__name__)
                    ] = operation
        if not operations:
            registered = sorted(
                {
                    name
                    for decoder in OptionalAstFieldDecoder.registry.values()
                    for name in decoder.field_names()
                }
            )
            return self.rejected_evaluation(
                f"probed fields {', '.join(sorted(probed_fields)) or 'none'}; "
                f"registered decoders own {', '.join(registered)}"
                + (f" ({'; '.join(refusals)})" if refusals else "")
            )
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-route-optional-ast-field",
            reason=(
                "Probes of an optional ast field read the decoder that owns the "
                "field's two states instead of testing it for None."
            ),
        )
        for operation in operations.values():
            recipe = recipe.with_operation(operation)
        return self.executable_evaluation(recipe)


__all__ = (
    "KeywordNameDecoder",
    "CaptureNameDecoder",
    "DeclareAndRouteOptionalAstFieldProbesOperation",
    "DeclaredAnnotationDecoder",
    "OptionalAstFieldAuthority",
    "OptionalAstFieldDecoder",
    "OptionalAstFieldDecoderFindingRecipeSynthesizer",
    "PendingDecoderMigration",
    "ProjectDecoderOwners",
    "RouteOptionalAstFieldProbesOperation",
    "VariadicParameterDecoder",
)
