from __future__ import annotations

import ast
import inspect
import re
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
)
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    ClassVar,
    Self,
    TypeGuard,
)

from metaclass_registry import (
    AutoRegisterMeta,
    extract_key_from_class_name,
)

from nominal_refactor_advisor.refactors.cases_family import handback
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
)
from nominal_refactor_advisor import (
    ast_field_decoders,
    ast_projection,
)
from nominal_refactor_advisor.ast_tools import AstParentIndex, ParsedModule
from nominal_refactor_advisor.codemod_declaration_source import (
    PythonExpressionSourceFormatter,
)
from nominal_refactor_advisor.codemod_foreign_family_source import (
    FunctionRewriteAuthority,
    ImportedClassReference,
)
from nominal_refactor_advisor.codemod_payload import (
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import SourceReprovedOperation
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceFileCreation,
    SourceLineSpan,
    SourceTextGeometry,
    SourceTextMutation,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.codemod_target_selectors import SourceIndexTargetSelector
from nominal_refactor_advisor.declared_receivers import (
    DeclaredInstance,
    ElementStep,
    ReceiverPath,
)
from nominal_refactor_advisor.declared_root import DeclaredRoot
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.probe_owner import ProbeOwnerResolver
from nominal_refactor_advisor.refactor_concepts import OptionalAstFieldDecoderConcept
from nominal_refactor_advisor.registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE
from nominal_refactor_advisor.source_index import AstTargetDigest

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

    @staticmethod
    def drops_cleanly(probe: ast.Compare, conjunction: ast.BoolOp) -> bool:
        """Whether the probe can leave the ``and`` without changing what runs.

        Operands before it ran for every element and will now run only for
        present ones; that is safe only when they call nothing.
        """

        index = next(i for i, value in enumerate(conjunction.values) if value is probe)
        return not any(
            isinstance(node, ast.Call)
            for value in conjunction.values[:index]
            for node in ast.walk(value)
        )

    @staticmethod
    def dropped_operand(
        geometry: SourceTextGeometry, probe: ast.Compare, conjunction: ast.BoolOp
    ) -> SourceTextSpanReplacement:
        """The edit removing the probe and one adjoining ``and``."""

        values = conjunction.values
        index = next(i for i, value in enumerate(values) if value is probe)
        start, end = geometry.required_node_offsets(probe)
        if index == 0:
            end, _ = geometry.required_node_offsets(values[1])
        else:
            _, start = geometry.required_node_offsets(values[index - 1])
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start, end_offset=end, replacement_source=""
        )

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
            if not cls.drops_cleanly(probe, parent):
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
            separator = re.fullmatch(
                r"((?:\s*\))*)\s+if\s+", geometry.source[removal_start:condition_start]
            )
            if separator is None:
                raise authority.refusal(self.probe, "the filter is not a bare `if`")
            # Closing parentheses of the iterable stay; only ` if PROBE` goes.
            removal_start += len(separator.group(1))
            edits.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=removal_start,
                    end_offset=condition_end,
                    replacement_source="",
                )
            )
        else:
            assert isinstance(self.condition, ast.BoolOp)
            edits.append(self.dropped_operand(geometry, self.probe, self.condition))
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
class AnyPresentProbeSite(CollectionProbeSite):
    """``any(A and V.f is not None and B for V in ITER)``: present elements only."""

    generator: ast.GeneratorExp
    conjunction: ast.BoolOp

    @classmethod
    def at(cls, probe, authority):
        parents = authority.parents
        conjunction = parents.get(probe)
        if not (
            isinstance(conjunction, ast.BoolOp)
            and isinstance(conjunction.op, ast.And)
            and cls.drops_cleanly(probe, conjunction)
        ):
            return None
        generator = parents.get(conjunction)
        match generator:
            case ast.GeneratorExp(
                elt=elt,
                generators=[
                    ast.comprehension(target=ast.Name(id=name), iter=iterable, ifs=[])
                ],
            ) if (
                elt is conjunction
            ):
                pass
            case _:
                return None
        match parents.get(generator):
            case ast.Call(func=ast.Name(id="any"), args=[only], keywords=[]) if (
                only is generator
            ):
                pass
            case _:
                return None
        variable = cls.probed(probe, authority.decoder, absent=False)
        if variable != name or not authority.iterates_elements(iterable):
            return None
        return cls(authority.decoder, probe, variable, generator, conjunction)

    def replacements(self, authority):
        geometry = authority.geometry
        start, end = geometry.required_node_offsets(self.generator)
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise authority.refusal(self.probe, "the quantifier holds a comment")
        (clause,) = self.generator.generators
        edits = [
            self.dropped_operand(geometry, self.probe, self.conjunction),
            SourceTextSpanReplacement.from_offsets(
                start_offset=geometry.required_node_offsets(clause.iter)[0],
                end_offset=geometry.required_node_offsets(clause.iter)[1],
                replacement_source=(
                    f"{self.decoder.collected(authority.source_of(clause.iter))}"
                    f".{self.decoder.present}"
                ),
            ),
            *self.renamed_field_reads(authority, self.generator, self.probe),
        ]
        return (authority.fitted_rewrite(self.generator, edits),)


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
                    replacement_source=authority.formatted(
                        f"{self.decoder.collected(iterable_source)}"
                        f".{self.decoder.present}",
                        iterable_start,
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


def _format_if_header(indentation: str, condition: str, line_length: int) -> str:
    """Format `if <condition>:` on one line when it fits, else parenthesized multi-line."""

    single = f"{indentation}if {condition}:"
    if len(single) <= line_length and "\n" not in condition:
        return single
    parsed = ast.parse(condition, mode="eval").body
    formatted = PythonExpressionSourceFormatter(
        max(20, line_length - len(indentation) - 4)
    ).replacement_source(parsed, line_prefix=indentation)
    if formatted.startswith("(") and formatted.endswith(")"):
        return f"{indentation}if {formatted}:"
    inner_indent = f"{indentation}    "
    inner = PythonExpressionSourceFormatter(
        max(20, line_length - len(inner_indent))
    ).replacement_source(parsed, line_prefix=inner_indent)
    return f"{indentation}if (\n{inner_indent}{inner}\n{indentation}):"


def _negated_guard_source(
    guard: ast.expr, authority: OptionalAstFieldAuthority
) -> str:
    """Boolean negation of `guard`, simplifying unary `not` and single comparisons."""

    match guard:
        case ast.UnaryOp(op=ast.Not(), operand=inner):
            text = authority.source_of(inner)
            if isinstance(inner, ast.BoolOp | ast.NamedExpr | ast.IfExp | ast.Lambda):
                return f"({text})"
            return text
        case ast.Compare(left=left, ops=[ast.NotEq()], comparators=[right]):
            return f"{authority.source_of(left)} == {authority.source_of(right)}"
        case ast.Compare(left=left, ops=[ast.Eq()], comparators=[right]):
            return f"{authority.source_of(left)} != {authority.source_of(right)}"
        case ast.Compare(left=left, ops=[ast.Is()], comparators=[right]):
            return f"{authority.source_of(left)} is not {authority.source_of(right)}"
        case ast.Compare(left=left, ops=[ast.IsNot()], comparators=[right]):
            return f"{authority.source_of(left)} is {authority.source_of(right)}"
    text = authority.source_of(guard)
    if isinstance(guard, ast.BoolOp | ast.NamedExpr | ast.IfExp | ast.Lambda):
        return f"not ({text})"
    return f"not {text}"


@dataclass(frozen=True)
class IfPresentProbeSite(SingleNodeProbeSite):
    """``if [GUARD and] V.f is not None [and POST]: BODY`` reading ``V.f``.

    BODY runs once per value the decoder answers: ``for N in DECODED(V): BODY``
    (with ``if POST:`` inside when post-guards follow) with ``V.f`` read as ``N``.
    Terminal ``elif`` branches rewrite under ``else:``.
    """

    statement: ast.If
    guards: tuple[ast.expr, ...]
    post_guards: tuple[ast.expr, ...] = ()

    @classmethod
    def at(cls, probe, authority):
        parent = authority.parents.get(probe)
        guards: tuple[ast.expr, ...] = ()
        post_guards: tuple[ast.expr, ...] = ()
        test: ast.expr = probe
        if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.And):
            idx = next(i for i, value in enumerate(parent.values) if value is probe)
            guards = tuple(parent.values[:idx])
            post_guards = tuple(parent.values[idx + 1 :])
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
            post_guards,
        )

    def replacements(self, authority):
        geometry = authority.geometry
        statement = self.statement
        start, _ = geometry.required_node_offsets(statement)
        is_elif = geometry.source.startswith("elif", start)
        if not is_elif and not geometry.source.startswith("if", start):
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
        post_reads = [
            node
            for guard in self.post_guards
            for node in ast.walk(guard)
            if isinstance(node, ast.Attribute)
            and node.attr == self.field
            and _same_expression(node.value, self.subject)
        ]
        decoded = self.decoder.single(authority.source_of(self.subject))
        if not reads and not post_reads:
            # Nothing reads the field: the probe only asks presence.
            probe_start, probe_end = geometry.required_node_offsets(self.probe)
            return (
                SourceTextSpanReplacement.from_offsets(
                    start_offset=probe_start,
                    end_offset=probe_end,
                    replacement_source=decoded,
                ),
            )
        if not is_elif and not self.guards and not self.post_guards:
            match statement.body:
                case [ast.Expr(value=ast.Yield(value=yielded))] if (
                    yielded is not None and _same_expression(yielded, self.probe.left)
                ):
                    _, stmt_end = geometry.required_node_offsets(statement)
                    return (
                        SourceTextSpanReplacement.from_offsets(
                            start_offset=start,
                            end_offset=stmt_end,
                            replacement_source=f"yield from {decoded}",
                        ),
                    )
        name = self.decoder.present_name()
        if name in authority.bound_names or any(
            isinstance(node, ast.Name) and node.id == name
            for node in ast.walk(statement)
        ):
            raise authority.refusal(statement, f"`{name}` is already used here")
        if not is_elif and not self.guards and not self.post_guards:
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
        return (self.nested(authority, name, decoded, reads, is_elif=is_elif),)

    def nested(
        self,
        authority: OptionalAstFieldAuthority,
        name: str,
        decoded: str,
        reads: list[ast.Attribute],
        *,
        is_elif: bool = False,
    ) -> SourceTextSpanReplacement:
        """``[else:] [if GUARD:] for N in DECODED(V): [if POST:] BODY`` indented cleanly."""

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

        def _render_conjunction(parts: tuple[ast.expr, ...], rename: bool) -> str:
            rendered_parts: list[str] = []
            for part in parts:
                part_start, part_end = geometry.required_node_offsets(part)
                text = geometry.source[part_start:part_end]
                if rename:
                    part_reads = [
                        node
                        for node in ast.walk(part)
                        if isinstance(node, ast.Attribute)
                        and node.attr == self.field
                        and _same_expression(node.value, self.subject)
                    ]
                    for rnode in sorted(
                        part_reads,
                        key=lambda read: geometry.required_node_offsets(read)[0],
                        reverse=True,
                    ):
                        rs, re_ = geometry.required_node_offsets(rnode)
                        text = text[: rs - part_start] + name + text[re_ - part_start :]
                if isinstance(part, ast.BoolOp | ast.NamedExpr | ast.IfExp | ast.Lambda):
                    text = f"({text})"
                rendered_parts.append(text)
            return " and ".join(rendered_parts)

        def _indent_lines(text: str, extra: str) -> str:
            return "".join(
                f"{extra}{line}" if line.strip() else line
                for line in text.splitlines(keepends=True)
            )

        base_indent = f"{indentation}    " if is_elif else indentation
        extra_levels = 1 if is_elif else 0
        if self.guards and self.post_guards:
            guard = _render_conjunction(self.guards, rename=False)
            post = _render_conjunction(self.post_guards, rename=True)
            guard_hdr = _format_if_header(base_indent, guard, authority.line_length)
            post_hdr = _format_if_header(
                f"{base_indent}        ", post, authority.line_length
            )
            indented = _indent_lines(body, "    " * (extra_levels + 2))
            inner = (
                f"{guard_hdr}\n"
                f"{base_indent}    for {name} in {decoded}:\n"
                f"{post_hdr}\n"
                f"{indented}"
            )
        elif self.guards:
            guard = _render_conjunction(self.guards, rename=False)
            guard_hdr = _format_if_header(base_indent, guard, authority.line_length)
            indented = _indent_lines(body, "    " * (extra_levels + 1))
            inner = (
                f"{guard_hdr}\n"
                f"{base_indent}    for {name} in {decoded}:\n"
                f"{indented}"
            )
        elif self.post_guards:
            post = _render_conjunction(self.post_guards, rename=True)
            post_hdr = _format_if_header(
                f"{base_indent}    ", post, authority.line_length
            )
            indented = _indent_lines(body, "    " * (extra_levels + 1))
            inner = (
                f"{base_indent}for {name} in {decoded}:\n"
                f"{post_hdr}\n"
                f"{indented}"
            )
        else:
            indented = _indent_lines(body, "    " * extra_levels)
            inner = f"{base_indent}for {name} in {decoded}:\n{indented}"
        rewritten = (
            f"else:\n{inner}" if is_elif else inner[len(indentation) :]
        )
        header_lines = "\n".join(
            line
            for line in rewritten.splitlines()
            if line.lstrip().startswith(("if ", "else:", "for "))
        )
        if header_lines and not authority.fit.fits(indentation, header_lines, ""):
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


def _is_pure_fallback(expression: ast.expr) -> bool:
    """Whether `expression` is side-effect-free and safe to evaluate eagerly as a
    default argument to `next(iter(...), fallback)`."""

    match expression:
        case ast.Constant() | ast.Name():
            return True
        case ast.Attribute(value=inner):
            return _is_pure_fallback(inner)
        case ast.UnaryOp(operand=inner):
            return _is_pure_fallback(inner)
        case ast.Tuple(elts=items) | ast.List(elts=items) | ast.Set(elts=items):
            return all(_is_pure_fallback(item) for item in items)
        case ast.Dict(keys=keys, values=values):
            return all(
                key is not None and _is_pure_fallback(key) and _is_pure_fallback(val)
                for key, val in zip(keys, values)
            )
        case ast.JoinedStr(values=parts):
            return all(
                isinstance(part, ast.Constant)
                or (
                    isinstance(part, ast.FormattedValue)
                    and _is_pure_fallback(part.value)
                )
                for part in parts
            )
    return False


def _reads_same_attribute(
    nodes: Iterable[ast.AST], subject: ast.expr, field: str
) -> bool:
    for root in nodes:
        for node in ast.walk(root):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == field
                and _same_expression(node.value, subject)
            ):
                return True
    return False


def _enclosing_block(
    parent: ast.AST, statement: ast.stmt
) -> tuple[list[ast.stmt], int] | None:
    """Return `(block, index)` for `statement` inside `parent`, if found."""

    block_lists: list[list[ast.stmt]] = []
    match parent:
        case ast.FunctionDef(body=b) | ast.AsyncFunctionDef(body=b) | ast.ClassDef(body=b):
            block_lists.append(b)
        case ast.If(body=b, orelse=o) | ast.For(body=b, orelse=o) | ast.AsyncFor(
            body=b, orelse=o
        ) | ast.While(body=b, orelse=o):
            block_lists.extend((b, o))
        case ast.With(body=b) | ast.AsyncWith(body=b):
            block_lists.append(b)
        case ast.Try(body=b, orelse=o, finalbody=f):
            block_lists.extend((b, o, f))
        case ast.ExceptHandler(body=b) | ast.match_case(body=b):
            block_lists.append(b)
    for statements in block_lists:
        for idx, candidate in enumerate(statements):
            if candidate is statement:
                return statements, idx
    return None


def _statements_after_in_block(
    parent: ast.AST, statement: ast.stmt
) -> tuple[ast.stmt, ...]:
    """Statements that follow `statement` in its immediately enclosing statement list."""

    located = _enclosing_block(parent, statement)
    if located is None:
        return ()
    statements, idx = located
    return tuple(statements[idx + 1 :])


def _always_terminates_block(statements: Iterable[ast.stmt]) -> bool:
    items = tuple(statements)
    if not items:
        return False
    match items[-1]:
        case ast.Return() | ast.Raise():
            return True
        case ast.If(body=b, orelse=o) if o:
            return _always_terminates_block(b) and _always_terminates_block(o)
    return False


def _is_pure_condition(expr: ast.expr) -> bool:
    """Whether `expr` is a side-effect-free boolean test safe to reorder in `and`/`or`."""

    match expr:
        case ast.Constant() | ast.Name():
            return True
        case ast.Attribute(value=inner):
            return _is_pure_condition(inner)
        case ast.UnaryOp(op=ast.Not(), operand=inner):
            return _is_pure_condition(inner)
        case ast.Compare(left=left, comparators=comparators):
            return _is_pure_condition(left) and all(
                _is_pure_condition(comp) for comp in comparators
            )
    return False


def _replace_attribute_in_ast(
    node: ast.AST, subject: ast.expr, field: str, replacement_name: str
) -> ast.AST:
    class _Replacer(ast.NodeTransformer):
        def visit_Attribute(self, attr: ast.Attribute) -> ast.AST:
            if attr.attr == field and _same_expression(attr.value, subject):
                return ast.Name(id=replacement_name, ctx=attr.ctx)
            return self.generic_visit(attr)

    return _Replacer().visit(ast.fix_missing_locations(node))


@dataclass(frozen=True)
class EarlyExitAbsentProbeSite(SingleNodeProbeSite):
    """``if [PRE or] V.f is None [or SUF]: continue|return|raise`` + guarded tail."""

    statement: ast.If
    tail: tuple[ast.stmt, ...]
    is_loop_continue: bool
    prefix_guards: tuple[ast.expr, ...] = ()
    suffix_guards: tuple[ast.expr, ...] = ()

    @property
    def anchor(self) -> ast.AST:
        return self.statement

    @classmethod
    def at(cls, probe, authority):
        parent = authority.parents.get(probe)
        prefix_guards: tuple[ast.expr, ...] = ()
        suffix_guards: tuple[ast.expr, ...] = ()
        test: ast.expr = probe
        if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or):
            idx = next((i for i, v in enumerate(parent.values) if v is probe), None)
            if idx is not None:
                prefix_guards = tuple(parent.values[:idx])
                suffix_guards = tuple(parent.values[idx + 1 :])
                test = parent
                parent = authority.parents.get(parent)
        match parent:
            case ast.If(test=found, body=body, orelse=[]) if found is test and body:
                pass
            case _:
                return None
        proved = cls.proved_subject(probe, authority, absent=True)
        if proved is None:
            return None
        subject, field = proved
        if (
            _reads_same_attribute(body, subject, field)
            or _reads_same_attribute(prefix_guards, subject, field)
            or _reads_same_attribute(suffix_guards, subject, field)
        ):
            return None
        enclosing = authority.parents.get(parent)
        if enclosing is None:
            return None
        tail = _statements_after_in_block(enclosing, parent)
        if not tail or not _reads_same_attribute(tail, subject, field):
            return None
        is_loop_continue = (
            isinstance(enclosing, (ast.For, ast.AsyncFor, ast.While))
            and len(body) == 1
            and isinstance(body[0], ast.Continue)
        )
        if not is_loop_continue and not (
            _always_terminates_block(body) and _always_terminates_block(tail)
        ):
            return None
        return cls(
            authority.decoder,
            probe,
            ast.unparse(subject),
            subject,
            field,
            parent,
            tail,
            is_loop_continue,
            prefix_guards,
            suffix_guards,
        )

    def replacements(self, authority):
        geometry = authority.geometry
        statement = self.statement
        name = self.decoder.present_name()
        if name in authority.bound_names or any(
            isinstance(node, ast.Name) and node.id == name
            for node in (statement, *self.tail)
            for node in ast.walk(node)
        ):
            raise authority.refusal(statement, f"`{name}` is already used here")
        start, _ = geometry.required_node_offsets(statement)
        if not geometry.source.startswith("if", start):
            raise authority.refusal(statement, "an `elif` cannot become a loop")
        if any(authority.escaping_loop_jumps(list(self.tail))):
            raise authority.refusal(statement, "the guarded tail jumps an enclosing loop")
        if not self.is_loop_continue and any(
            authority.escaping_loop_jumps(statement.body)
        ):
            raise authority.refusal(statement, "the absent branch jumps an enclosing loop")
        probed_names = {
            part.id for part in ast.walk(self.subject) if isinstance(part, ast.Name)
        }
        for stmt in self.tail:
            for node in ast.walk(stmt):
                match node:
                    case ast.Name(id=bound, ctx=ast.Store() | ast.Del()) if (
                        bound in probed_names
                    ):
                        raise authority.refusal(node, "the tail rebinds the probed node")
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.end_lineno != node.lineno
                ):
                    raise authority.refusal(node, "the tail holds a multi-line string")
        _, tail_end = geometry.required_node_offsets(self.tail[-1])
        if geometry.span_contains_comment(SourceTextSpan(start, tail_end)):
            raise authority.refusal(statement, "the early-exit region holds a comment")
        line_start = geometry.line_offsets[statement.lineno - 1]
        indentation = geometry.source[line_start:start]
        tail_line_start = geometry.line_offsets[self.tail[0].lineno - 1]
        tail_source = geometry.source[tail_line_start:tail_end]
        reads = [
            node
            for stmt in self.tail
            for node in ast.walk(stmt)
            if isinstance(node, ast.Attribute)
            and node.attr == self.field
            and _same_expression(node.value, self.subject)
        ]
        for rnode in sorted(
            reads,
            key=lambda read: geometry.required_node_offsets(read)[0],
            reverse=True,
        ):
            rs, re_ = geometry.required_node_offsets(rnode)
            tail_source = (
                tail_source[: rs - tail_line_start]
                + name
                + tail_source[re_ - tail_line_start :]
            )
        extra_levels = 1 + bool(self.prefix_guards) + bool(self.suffix_guards)
        extra_indent = "    " * extra_levels
        indented_tail = "".join(
            f"{extra_indent}{line}" if line.strip() else line
            for line in tail_source.splitlines(keepends=True)
        )
        decoded = self.decoder.single(authority.source_of(self.subject))
        loop_indent = f"{indentation}    " if self.prefix_guards else indentation
        if self.suffix_guards:
            suffix_negated = " and ".join(
                _negated_guard_source(guard, authority) for guard in self.suffix_guards
            )
            suffix_hdr = _format_if_header(
                f"{loop_indent}    ", suffix_negated, authority.line_length
            )
            inner_loop = (
                f"{loop_indent}for {name} in {decoded}:\n"
                f"{suffix_hdr}\n"
                f"{indented_tail}"
            )
        else:
            inner_loop = f"{loop_indent}for {name} in {decoded}:\n{indented_tail}"
        if self.prefix_guards:
            negated = " and ".join(
                _negated_guard_source(guard, authority) for guard in self.prefix_guards
            )
            guard_hdr = _format_if_header(indentation, negated, authority.line_length)
            loop_block = f"{guard_hdr[len(indentation) :]}\n{inner_loop}"
        else:
            loop_block = inner_loop[len(indentation) :]
        if self.is_loop_continue:
            rewritten = loop_block
        else:
            absent_start = geometry.line_offsets[statement.body[0].lineno - 1]
            _, absent_end = geometry.required_node_offsets(statement.body[-1])
            absent_raw = geometry.source[absent_start:absent_end]
            dedented_absent = "".join(
                line[4:] if line.startswith(f"{indentation}    ") else line
                for line in absent_raw.splitlines(keepends=True)
            )
            rewritten = f"{loop_block}\n{dedented_absent}"
        header_lines = "\n".join(
            line
            for line in loop_block.splitlines()
            if line.lstrip().startswith(("if ", "for "))
        )
        if header_lines and not authority.fit.fits(indentation, header_lines, ""):
            raise authority.refusal(statement, "the rewritten early-exit loop overflows the line")
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=tail_end,
                replacement_source=rewritten,
            ),
        )


@dataclass(frozen=True)
class FallbackTernaryProbeSite(SingleNodeProbeSite):
    """``EXPR(V.f) if [GUARD and] V.f is not None else FALLBACK`` or ``FALLBACK if V.f is None else EXPR(V.f)``.

    When ``FALLBACK`` is side-effect-free and does not read ``V.f``, bare ``V.f``
    becomes ``next(iter(DECODED(V)), FALLBACK)`` and a transforming ``EXPR(V.f)``
    becomes ``next((EXPR(N) for N in DECODED(V)), FALLBACK)``.
    """

    expression: ast.IfExp
    value_expr: ast.expr
    fallback: ast.expr
    guards: tuple[ast.expr, ...] = ()

    @classmethod
    def at(cls, probe, authority):
        parent = authority.parents.get(probe)
        guards: tuple[ast.expr, ...] = ()
        if (
            isinstance(parent, ast.BoolOp)
            and isinstance(parent.op, ast.And)
            and parent.values[-1] is probe
        ):
            guards = tuple(parent.values[:-1])
            expression = authority.parents.get(parent)
            if not isinstance(expression, ast.IfExp) or expression.test is not parent:
                return None
            present = cls.proved_subject(probe, authority, absent=False)
            if present is None:
                return None
            subject, field = present
            if _reads_same_attribute(guards, subject, field):
                return None
            value_expr, fallback = expression.body, expression.orelse
        else:
            expression = parent
            if not isinstance(expression, ast.IfExp) or expression.test is not probe:
                return None
            present = cls.proved_subject(probe, authority, absent=False)
            absent = cls.proved_subject(probe, authority, absent=True)
            if present is not None:
                subject, field = present
                value_expr, fallback = expression.body, expression.orelse
            elif absent is not None:
                subject, field = absent
                fallback, value_expr = expression.body, expression.orelse
            else:
                return None
        if not _same_expression(value_expr, probe.left):
            if not _reads_same_attribute((value_expr,), subject, field):
                return None
            if any(
                isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await))
                for node in ast.walk(value_expr)
            ):
                return None
        if not _is_pure_fallback(fallback) or _reads_same_attribute(
            (fallback,), subject, field
        ):
            return None
        return cls(
            authority.decoder,
            probe,
            ast.unparse(subject),
            subject,
            field,
            expression,
            value_expr,
            fallback,
            guards,
        )

    @property
    def anchor(self) -> ast.AST:
        return self.expression

    def replacements(self, authority):
        geometry = authority.geometry
        start, end = geometry.required_node_offsets(self.expression)
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise authority.refusal(self.expression, "the fallback ternary holds a comment")
        decoded = self.decoder.single(authority.source_of(self.subject))
        if self.guards:
            guard_parts = [
                f"({authority.source_of(guard)})"
                if isinstance(guard, ast.BoolOp | ast.NamedExpr | ast.IfExp | ast.Lambda)
                else authority.source_of(guard)
                for guard in self.guards
            ]
            guard_source = " and ".join(guard_parts)
            decoded = f"({decoded} if {guard_source} else ())"
        fallback_source = authority.source_of(self.fallback)
        if _same_expression(self.value_expr, self.probe.left):
            candidate = f"next(iter({decoded}), {fallback_source})"
        else:
            name = self.decoder.present_name()
            if name in authority.bound_names or any(
                isinstance(node, ast.Name) and node.id == name
                for node in ast.walk(self.value_expr)
            ):
                raise authority.refusal(
                    self.expression, f"`{name}` is already used here"
                )
            val_start, val_end = geometry.required_node_offsets(self.value_expr)
            val_source = geometry.source[val_start:val_end]
            reads = [
                node
                for node in ast.walk(self.value_expr)
                if isinstance(node, ast.Attribute)
                and node.attr == self.field
                and _same_expression(node.value, self.subject)
            ]
            for rnode in sorted(
                reads,
                key=lambda read: geometry.required_node_offsets(read)[0],
                reverse=True,
            ):
                rs, re_ = geometry.required_node_offsets(rnode)
                val_source = (
                    val_source[: rs - val_start] + name + val_source[re_ - val_start :]
                )
            candidate = f"next(({val_source} for {name} in {decoded}), {fallback_source})"
        replacement = authority.fit.fitted(candidate, start, end)
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=replacement,
            ),
        )


@dataclass(frozen=True)
class SingleNodeComprehensionFilterProbeSite(SingleNodeProbeSite):
    """``... if [PRE and] V.f is not None [and POST]`` in a comprehension reading ``V.f``.

    Becomes ``[if PRE] for N in DECODED(V) [if POST(V.f -> N)]`` with downstream
    reads of ``V.f`` rewritten to ``N``.
    """

    comprehension_node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp
    clause: ast.comprehension
    clause_index: int
    if_index: int
    condition: ast.expr
    pre_guards: tuple[ast.expr, ...]
    post_guards: tuple[ast.expr, ...]

    @property
    def anchor(self) -> ast.AST:
        return self.comprehension_node

    @classmethod
    def at(cls, probe, authority):
        parents = authority.parents
        condition: ast.expr = probe
        pre_guards: tuple[ast.expr, ...] = ()
        post_guards: tuple[ast.expr, ...] = ()
        parent = parents.get(probe)
        if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.And):
            idx = next(i for i, value in enumerate(parent.values) if value is probe)
            pre_guards = tuple(parent.values[:idx])
            post_guards = tuple(parent.values[idx + 1 :])
            condition = parent
            parent = parents.get(parent)
        if not isinstance(parent, ast.comprehension) or not any(
            clause_if is condition for clause_if in parent.ifs
        ):
            return None
        comp = parents.get(parent)
        if not isinstance(
            comp, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            return None
        proved = cls.proved_subject(probe, authority, absent=False)
        if proved is None:
            return None
        subject, field = proved
        clause_index = next(
            i for i, generator in enumerate(comp.generators) if generator is parent
        )
        if_index = next(
            i for i, clause_if in enumerate(parent.ifs) if clause_if is condition
        )
        probed_names = {
            part.id for part in ast.walk(subject) if isinstance(part, ast.Name)
        }
        for later_clause in comp.generators[clause_index + 1 :]:
            for node in ast.walk(later_clause.target):
                if isinstance(node, ast.Name) and node.id in probed_names:
                    return None
        downstream: list[ast.AST] = [
            *post_guards,
            *parent.ifs[if_index + 1 :],
            *comp.generators[clause_index + 1 :],
        ]
        match comp:
            case ast.DictComp(key=key, value=value):
                downstream.extend((key, value))
            case _:
                downstream.append(comp.elt)
        if not _reads_same_attribute(downstream, subject, field):
            return None
        return cls(
            authority.decoder,
            probe,
            ast.unparse(subject),
            subject,
            field,
            comp,
            parent,
            clause_index,
            if_index,
            condition,
            pre_guards,
            post_guards,
        )

    def replacements(self, authority):
        geometry = authority.geometry
        name = self.decoder.present_name()
        if name in authority.bound_names or any(
            isinstance(node, ast.Name) and node.id == name
            for node in ast.walk(self.comprehension_node)
        ):
            raise authority.refusal(self.probe, f"`{name}` is already used here")
        comp_start, comp_end = geometry.required_node_offsets(self.comprehension_node)
        if geometry.span_contains_comment(SourceTextSpan(comp_start, comp_end)):
            raise authority.refusal(self.probe, "the comprehension holds a comment")
        prev_node: ast.AST = (
            self.clause.ifs[self.if_index - 1]
            if self.if_index
            else self.clause.iter
        )
        prev_end = geometry.required_node_offsets(prev_node)[1]
        cond_start, cond_end = geometry.required_node_offsets(self.condition)
        separator = re.fullmatch(
            r"((?:\s*\))*)\s+if\s+", geometry.source[prev_end:cond_start]
        )
        if separator is None:
            raise authority.refusal(self.probe, "the filter is not a bare `if`")
        if_start = prev_end + len(separator.group(1))
        decoded = self.decoder.single(authority.source_of(self.subject))

        def _render_conjunction(parts: tuple[ast.expr, ...], rename: bool) -> str:
            rendered: list[str] = []
            for part in parts:
                part_start, part_end = geometry.required_node_offsets(part)
                text = geometry.source[part_start:part_end]
                if rename:
                    part_reads = [
                        node
                        for node in ast.walk(part)
                        if isinstance(node, ast.Attribute)
                        and node.attr == self.field
                        and _same_expression(node.value, self.subject)
                    ]
                    for rnode in sorted(
                        part_reads,
                        key=lambda read: geometry.required_node_offsets(read)[0],
                        reverse=True,
                    ):
                        rs, re_ = geometry.required_node_offsets(rnode)
                        text = text[: rs - part_start] + name + text[re_ - part_start :]
                if isinstance(part, ast.BoolOp | ast.NamedExpr | ast.IfExp | ast.Lambda):
                    text = f"({text})"
                rendered.append(text)
            return " and ".join(rendered)

        pieces: list[str] = []
        if self.pre_guards:
            pieces.append(f"if {_render_conjunction(self.pre_guards, rename=False)}")
        pieces.append(f"for {name} in {decoded}")
        if self.post_guards:
            pieces.append(f"if {_render_conjunction(self.post_guards, rename=True)}")
        edits: list[SourceTextSpanReplacement] = [
            SourceTextSpanReplacement.from_offsets(
                start_offset=if_start,
                end_offset=cond_end,
                replacement_source=" " + " ".join(pieces),
            )
        ]
        downstream_roots: list[ast.AST] = [
            *self.clause.ifs[self.if_index + 1 :],
            *self.comprehension_node.generators[self.clause_index + 1 :],
        ]
        match self.comprehension_node:
            case ast.DictComp(key=key, value=value):
                downstream_roots.extend((key, value))
            case _:
                downstream_roots.append(self.comprehension_node.elt)
        for root in downstream_roots:
            for node in ast.walk(root):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == self.field
                    and _same_expression(node.value, self.subject)
                ):
                    edits.append(authority.renamed(node, name))
        return (authority.fitted_rewrite(self.comprehension_node, edits),)


@dataclass(frozen=True)
class DisjunctiveAbsentProbeSite(SingleNodeProbeSite):
    """Truth-tested ``V.f is None or P(V.f)`` rewritten as ``all(P(N) for N in DECODED(V))``."""

    or_node: ast.BoolOp
    probe_index: int
    rest: tuple[ast.expr, ...]

    @property
    def anchor(self) -> ast.AST:
        return self.or_node

    @classmethod
    def at(cls, probe, authority):
        or_node = authority.parents.get(probe)
        if not isinstance(or_node, ast.BoolOp) or not isinstance(or_node.op, ast.Or):
            return None
        if not authority.only_truth_tested(or_node):
            return None
        probe_index = next(
            i for i, value in enumerate(or_node.values) if value is probe
        )
        rest = tuple(or_node.values[probe_index + 1 :])
        if not rest:
            return None
        proved = cls.proved_subject(probe, authority, absent=True)
        if proved is None:
            return None
        subject, field = proved
        if not _reads_same_attribute(rest, subject, field):
            return None
        if any(
            isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr))
            for part in rest
            for node in ast.walk(part)
        ):
            return None
        return cls(
            authority.decoder,
            probe,
            ast.unparse(subject),
            subject,
            field,
            or_node,
            probe_index,
            rest,
        )

    def replacements(self, authority):
        geometry = authority.geometry
        name = self.decoder.present_name()
        if name in authority.bound_names or any(
            isinstance(node, ast.Name) and node.id == name
            for part in self.rest
            for node in ast.walk(part)
        ):
            raise authority.refusal(self.probe, f"`{name}` is already used here")
        start, probe_end = geometry.required_node_offsets(self.probe)
        _, end = geometry.required_node_offsets(self.rest[-1])
        src = geometry.source
        # Include any closing parentheses balancing an opening `(` on `rest` operands.
        depth = 0
        for ch in src[probe_end:end]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        while depth > 0:
            pos = end
            while pos < len(src) and src[pos] in " \t\r\n":
                pos += 1
            if pos >= len(src) or src[pos] != ")":
                raise authority.refusal(
                    self.probe, "could not balance parentheses in disjunction"
                )
            end = pos + 1
            depth -= 1
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise authority.refusal(self.probe, "the disjunction holds a comment")
        rest_start = geometry.required_node_offsets(self.rest[0])[0]
        rest_end = geometry.required_node_offsets(self.rest[-1])[1]
        rest_source = src[rest_start:rest_end]
        reads = [
            node
            for part in self.rest
            for node in ast.walk(part)
            if isinstance(node, ast.Attribute)
            and node.attr == self.field
            and _same_expression(node.value, self.subject)
        ]
        for rnode in sorted(
            reads,
            key=lambda read: geometry.required_node_offsets(read)[0],
            reverse=True,
        ):
            rs, re_ = geometry.required_node_offsets(rnode)
            rest_source = (
                rest_source[: rs - rest_start] + name + rest_source[re_ - rest_start :]
            )
        decoded = self.decoder.single(authority.source_of(self.subject))
        candidate = f"all({rest_source} for {name} in {decoded})"
        replacement = (
            authority.formatted(candidate, start)
            if "\n" in candidate
            else authority.fit.fitted(candidate, start, end)
        )
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=replacement,
            ),
        )


@dataclass(frozen=True)
class PresenceTestProbeSite(SingleNodeProbeSite):
    """Pure absence or presence test ``V.f is None`` / ``V.f is not None`` when no
    narrowed downstream code reads ``V.f``.

    ``V.f is None`` becomes ``not DECODED(V)``; ``V.f is not None`` becomes
    ``DECODED(V)`` in truth-tested slots and ``bool(DECODED(V))`` elsewhere.
    """

    absent: bool

    @classmethod
    def at(cls, probe, authority):
        absent_match = cls.proved_subject(probe, authority, absent=True)
        present_match = (
            None
            if absent_match is not None
            else cls.proved_subject(probe, authority, absent=False)
        )
        if absent_match is not None:
            subject, field = absent_match
            absent = True
        elif present_match is not None:
            subject, field = present_match
            absent = False
        else:
            return None
        if cls._downstream_reads_narrowed_field(probe, authority, subject, field, absent):
            return None
        return cls(
            authority.decoder,
            probe,
            ast.unparse(subject),
            subject,
            field,
            absent,
        )

    @staticmethod
    def _downstream_reads_narrowed_field(
        probe: ast.Compare,
        authority: OptionalAstFieldAuthority,
        subject: ast.expr,
        field: str,
        absent: bool,
    ) -> bool:
        """Whether any code guarded by `probe` reads `subject.field` relying on narrowing."""

        effective_absent = absent
        narrows_true = True
        narrows_false = True
        for child, parent in authority.ancestors(probe):
            match parent:
                case ast.UnaryOp(op=ast.Not()):
                    effective_absent = not effective_absent
                    narrows_true, narrows_false = narrows_false, narrows_true
                case ast.BoolOp(op=ast.And(), values=values):
                    idx = next(i for i, v in enumerate(values) if v is child)
                    if (
                        not effective_absent
                        and narrows_true
                        and _reads_same_attribute(values[idx + 1 :], subject, field)
                    ):
                        return True
                    narrows_false = False
                case ast.BoolOp(op=ast.Or(), values=values):
                    idx = next(i for i, v in enumerate(values) if v is child)
                    if (
                        effective_absent
                        and narrows_false
                        and _reads_same_attribute(values[idx + 1 :], subject, field)
                    ):
                        return True
                    narrows_true = False
                case ast.If(test=test, body=body, orelse=orelse) if test is child:
                    if (
                        not effective_absent
                        and narrows_true
                        and _reads_same_attribute(body, subject, field)
                    ):
                        return True
                    if (
                        effective_absent
                        and narrows_false
                        and _reads_same_attribute(orelse, subject, field)
                    ):
                        return True
                    if effective_absent and narrows_false:
                        enclosing = authority.parents.get(parent)
                        if enclosing is not None and _reads_same_attribute(
                            _statements_after_in_block(enclosing, parent),
                            subject,
                            field,
                        ):
                            return True
                    break
                case ast.IfExp(test=test, body=body, orelse=orelse) if test is child:
                    if (
                        not effective_absent
                        and narrows_true
                        and _reads_same_attribute((body,), subject, field)
                    ):
                        return True
                    if (
                        effective_absent
                        and narrows_false
                        and _reads_same_attribute((orelse,), subject, field)
                    ):
                        return True
                    break
                case ast.comprehension(ifs=ifs) if any(c is child for c in ifs):
                    if not effective_absent and narrows_true:
                        comp = authority.parents.get(parent)
                        if comp is not None:
                            idx = next(i for i, c in enumerate(ifs) if c is child)
                            rest_nodes: list[ast.AST] = list(ifs[idx + 1 :])
                            match comp:
                                case ast.ListComp(elt=e) | ast.SetComp(elt=e) | ast.GeneratorExp(
                                    elt=e
                                ):
                                    rest_nodes.append(e)
                                case ast.DictComp(key=k, value=v):
                                    rest_nodes.extend((k, v))
                            if _reads_same_attribute(rest_nodes, subject, field):
                                return True
                    break
                case _:
                    break
        return False

    def replacements(self, authority):
        geometry = authority.geometry
        start, end = geometry.required_node_offsets(self.probe)
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise authority.refusal(self.probe, "the probe holds a comment")
        decoded = self.decoder.single(authority.source_of(self.subject))
        if self.absent:
            candidate = f"not {decoded}"
            replacement = authority.fit.fitted(candidate, start, end)
            if not authority.holds_boolean_operand(self.probe):
                replacement = f"({replacement})"
        else:
            candidate = authority.truth_of(self.probe, decoded)
            replacement = authority.fit.fitted(candidate, start, end)
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=replacement,
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
    adjacent_pair: bool = True

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
        """The site when ``item``'s pair partner appears in ``items`` in order."""

        if proved is None:
            return None
        subject, field = proved
        first_field, second_field = authority.decoder.paired_fields
        index = next(index for index, value in enumerate(items) if value is item)
        wanted_field = second_field if field == first_field else first_field
        candidate_indices = (
            range(index + 1, len(items))
            if field == first_field
            else range(index - 1, -1, -1)
        )
        partner_index: int | None = None
        for cand_idx in candidate_indices:
            if joined == "*" and abs(cand_idx - index) != 1:
                break
            partner = items[cand_idx]
            partner_probe = (
                partner.value.test
                if isinstance(partner, ast.Starred) and isinstance(partner.value, ast.IfExp)
                else partner
            )
            match partner_probe:
                case ast.Compare(
                    left=ast.Attribute(value=other, attr=other_field), ops=[operator]
                ) if (
                    other_field == wanted_field
                    and _same_expression(other, subject)
                    and type(operator) is type(probe.ops[0])
                    and _none_probe(partner_probe)
                ):
                    lo, hi = (index, cand_idx) if index < cand_idx else (cand_idx, index)
                    if all(_is_pure_condition(items[mid]) for mid in range(lo + 1, hi)):
                        partner_index = cand_idx
                        break
        if partner_index is None:
            return None
        partner = items[partner_index]
        partner_probe = (
            partner.value.test
            if isinstance(partner, ast.Starred) and isinstance(partner.value, ast.IfExp)
            else partner
        )
        if joined == "*" and SpliceProbeSite.at(partner_probe, authority) is None:
            return None
        first, second = (item, partner) if index < partner_index else (partner, item)
        return cls(
            authority.decoder,
            probe,
            ast.unparse(subject),
            subject,
            field,
            first,
            second,
            joined,
            abs(partner_index - index) == 1,
        )

    def replacements(self, authority):
        geometry = authority.geometry
        decoded = self.decoder.single(authority.source_of(self.subject))
        boolean = authority.parents[self.first]
        if not self.adjacent_pair and isinstance(boolean, ast.BoolOp):
            first_start, first_end = geometry.required_node_offsets(self.first)
            if geometry.span_contains_comment(SourceTextSpan(first_start, first_end)):
                raise authority.refusal(self.probe, "the pair holds a comment")
            replacement = (
                f"not {decoded}"
                if self.joined == "and"
                else authority.truth_of(boolean, decoded)
            )
            return (
                SourceTextSpanReplacement.from_offsets(
                    start_offset=first_start,
                    end_offset=first_end,
                    replacement_source=replacement,
                ),
                CollectionProbeSite.dropped_operand(geometry, self.second, boolean),
            )
        start, _ = geometry.required_node_offsets(self.first)
        _, end = geometry.required_node_offsets(self.second)
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise authority.refusal(self.probe, "the pair holds a comment")
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


@dataclass(frozen=True)
class PairedIfPresentProbeSite(SingleNodeProbeSite):
    """Adjacent ``if V.a is not None: BODY(V.a)`` + ``if V.b is not None: BODY(V.b)``."""

    first_stmt: ast.If
    second_stmt: ast.If
    first_field: str
    second_field: str

    @property
    def anchor(self) -> ast.AST:
        return self.first_stmt

    @classmethod
    def at(cls, probe, authority):
        parent = authority.parents.get(probe)
        match parent:
            case ast.If(test=found, body=body, orelse=[]) if found is probe and body:
                pass
            case _:
                return None
        proved = cls.proved_subject(probe, authority, absent=False)
        if proved is None:
            return None
        subject, field = proved
        first_field, second_field = authority.decoder.paired_fields
        enclosing = authority.parents.get(parent)
        if enclosing is None:
            return None
        located = _enclosing_block(enclosing, parent)
        if located is None:
            return None
        statements, idx = located
        offset = 1 if field == first_field else -1
        partner_idx = idx + offset
        if not 0 <= partner_idx < len(statements):
            return None
        partner = statements[partner_idx]
        wanted_field = second_field if offset == 1 else first_field
        match partner:
            case ast.If(
                test=ast.Compare(
                    left=ast.Attribute(value=other, attr=other_field),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
                body=partner_body,
                orelse=[],
            ) if (
                other_field == wanted_field
                and _same_expression(other, subject)
                and partner_body
            ):
                pass
            case _:
                return None
        first_stmt, second_stmt = (parent, partner) if offset == 1 else (partner, parent)
        # Bodies must be structurally identical once `V.first_field` / `V.second_field` map to the loop var.
        sentinel = "__nra_paired_param__"
        norm_first = [
            ast.dump(_replace_attribute_in_ast(ast.parse(ast.unparse(s)).body[0], subject, first_field, sentinel))
            for s in first_stmt.body
        ]
        norm_second = [
            ast.dump(_replace_attribute_in_ast(ast.parse(ast.unparse(s)).body[0], subject, second_field, sentinel))
            for s in second_stmt.body
        ]
        if norm_first != norm_second:
            return None
        return cls(
            authority.decoder,
            probe,
            ast.unparse(subject),
            subject,
            field,
            first_stmt,
            second_stmt,
            first_field,
            second_field,
        )

    def replacements(self, authority):
        geometry = authority.geometry
        name = self.decoder.present_name()
        if name in authority.bound_names or any(
            isinstance(node, ast.Name) and node.id == name
            for stmt in (self.first_stmt, self.second_stmt)
            for node in ast.walk(stmt)
        ):
            raise authority.refusal(self.first_stmt, f"`{name}` is already used here")
        start, _ = geometry.required_node_offsets(self.first_stmt)
        _, end = geometry.required_node_offsets(self.second_stmt)
        if not geometry.source.startswith("if", start):
            raise authority.refusal(self.first_stmt, "an `elif` cannot become a loop")
        if self.first_stmt.body[0].lineno == self.first_stmt.lineno:
            raise authority.refusal(self.first_stmt, "the body shares the `if` line")
        if any(authority.escaping_loop_jumps(self.first_stmt.body)):
            raise authority.refusal(self.first_stmt, "the body jumps an enclosing loop")
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise authority.refusal(self.first_stmt, "the paired `if` holds a comment")
        probed_names = {
            part.id for part in ast.walk(self.subject) if isinstance(part, ast.Name)
        }
        for node in ast.walk(self.first_stmt):
            match node:
                case ast.Name(id=bound, ctx=ast.Store() | ast.Del()) if bound in probed_names:
                    raise authority.refusal(node, "the body rebinds the probed node")
        body_start = geometry.line_offsets[self.first_stmt.body[0].lineno - 1]
        _, first_end = geometry.required_node_offsets(self.first_stmt)
        body_source = geometry.source[body_start:first_end]
        reads = [
            node
            for stmt in self.first_stmt.body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Attribute)
            and node.attr == self.first_field
            and _same_expression(node.value, self.subject)
        ]
        for rnode in sorted(
            reads,
            key=lambda read: geometry.required_node_offsets(read)[0],
            reverse=True,
        ):
            rs, re_ = geometry.required_node_offsets(rnode)
            body_source = (
                body_source[: rs - body_start] + name + body_source[re_ - body_start :]
            )
        decoded = self.decoder.single(authority.source_of(self.subject))
        rewritten = f"for {name} in {decoded}:\n{body_source}"
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=rewritten,
            ),
        )


@dataclass(frozen=True)
class PairedAnyPresentProbeSite(SingleNodeProbeSite):
    """Adjacent ``(V.a is not None and P(V.a)) or (V.b is not None and P(V.b))``."""

    first_branch: ast.BoolOp
    second_branch: ast.BoolOp
    first_field: str
    second_field: str

    @property
    def anchor(self) -> ast.AST:
        return self.first_branch

    @classmethod
    def at(cls, probe, authority):
        and_node = authority.parents.get(probe)
        if (
            not isinstance(and_node, ast.BoolOp)
            or not isinstance(and_node.op, ast.And)
            or len(and_node.values) < 2
            or and_node.values[0] is not probe
        ):
            return None
        or_node = authority.parents.get(and_node)
        if not isinstance(or_node, ast.BoolOp) or not isinstance(or_node.op, ast.Or):
            return None
        proved = cls.proved_subject(probe, authority, absent=False)
        if proved is None:
            return None
        subject, field = proved
        first_field, second_field = authority.decoder.paired_fields
        idx = next(i for i, v in enumerate(or_node.values) if v is and_node)
        offset = 1 if field == first_field else -1
        partner_idx = idx + offset
        if not 0 <= partner_idx < len(or_node.values):
            return None
        partner = or_node.values[partner_idx]
        wanted_field = second_field if offset == 1 else first_field
        match partner:
            case ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Compare(
                        left=ast.Attribute(value=other, attr=other_field),
                        ops=[ast.IsNot()],
                        comparators=[ast.Constant(value=None)],
                    ),
                    *partner_rest,
                ],
            ) if (
                partner_rest
                and other_field == wanted_field
                and _same_expression(other, subject)
            ):
                pass
            case _:
                return None
        first_branch, second_branch = (
            (and_node, partner) if offset == 1 else (partner, and_node)
        )
        sentinel = "__nra_paired_param__"
        first_rest = first_branch.values[1:]
        second_rest = second_branch.values[1:]
        if len(first_rest) != len(second_rest):
            return None
        norm_first = [
            ast.dump(_replace_attribute_in_ast(ast.parse(ast.unparse(e), mode="eval").body, subject, first_field, sentinel))
            for e in first_rest
        ]
        norm_second = [
            ast.dump(_replace_attribute_in_ast(ast.parse(ast.unparse(e), mode="eval").body, subject, second_field, sentinel))
            for e in second_rest
        ]
        if norm_first != norm_second:
            return None
        return cls(
            authority.decoder,
            probe,
            ast.unparse(subject),
            subject,
            field,
            first_branch,
            second_branch,
            first_field,
            second_field,
        )

    def replacements(self, authority):
        geometry = authority.geometry
        name = self.decoder.present_name()
        if name in authority.bound_names or any(
            isinstance(node, ast.Name) and node.id == name
            for branch in (self.first_branch, self.second_branch)
            for node in ast.walk(branch)
        ):
            raise authority.refusal(self.first_branch, f"`{name}` is already used here")
        start, _ = geometry.required_node_offsets(self.first_branch)
        _, end = geometry.required_node_offsets(self.second_branch)
        # Include surrounding parentheses around each BoolOp branch when present.
        src = geometry.source
        while start > 0 and src[start - 1] == "(":
            # Only consume matching outer parens on first_branch if second_branch ends with ")" or parens balance.
            break
        # Walk backwards/forwards cleanly for `(A and B)` parenthesized branches.
        open_idx = start - 1
        while open_idx >= 0 and src[open_idx] in " \t":
            open_idx -= 1
        close_first = geometry.required_node_offsets(self.first_branch)[1]
        if open_idx >= 0 and src[open_idx] == "(":
            probe_close = close_first
            while probe_close < len(src) and src[probe_close] in " \t\n":
                probe_close += 1
            if probe_close < len(src) and src[probe_close] == ")":
                start = open_idx
        close_idx = end
        while close_idx < len(src) and src[close_idx] in " \t\n":
            close_idx += 1
        sec_start = geometry.required_node_offsets(self.second_branch)[0] - 1
        while sec_start >= 0 and src[sec_start] in " \t\n":
            sec_start -= 1
        if sec_start >= 0 and src[sec_start] == "(" and close_idx < len(src) and src[close_idx] == ")":
            end = close_idx + 1
        if geometry.span_contains_comment(SourceTextSpan(start, end)):
            raise authority.refusal(self.first_branch, "the paired `or` holds a comment")
        rest_start = geometry.required_node_offsets(self.first_branch.values[1])[0]
        rest_end = geometry.required_node_offsets(self.first_branch.values[-1])[1]
        pred_source = src[rest_start:rest_end]
        reads = [
            node
            for expr in self.first_branch.values[1:]
            for node in ast.walk(expr)
            if isinstance(node, ast.Attribute)
            and node.attr == self.first_field
            and _same_expression(node.value, self.subject)
        ]
        for rnode in sorted(
            reads,
            key=lambda read: geometry.required_node_offsets(read)[0],
            reverse=True,
        ):
            rs, re_ = geometry.required_node_offsets(rnode)
            pred_source = (
                pred_source[: rs - rest_start] + name + pred_source[re_ - rest_start :]
            )
        decoded = self.decoder.single(authority.source_of(self.subject))
        candidate = f"any({pred_source} for {name} in {decoded})"
        replacement = authority.fit.fitted(candidate, start, end)
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=replacement,
            ),
        )


class OptionalAstFieldDecoder(ABC, metaclass=AutoRegisterMeta):
    """One registered decoder: the ast fields it owns and the shapes sites take.

    ``fields`` maps each ast class to its optional field. ``owners`` are the
    objects this package declares as the decoder; a project without them gets
    their source generated. Every concrete decoder registers under its class
    name, the name a recipe payload carries.
    """

    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = extract_key_from_class_name
    __registry__: ClassVar[dict[str, type[OptionalAstFieldDecoder]]]
    fields: ClassVar[Mapping[type[ast.AST], str]]
    owners: ClassVar[tuple[type | Callable[..., object], ...]]
    shapes: ClassVar[tuple[type[ProbeSite], ...]] = (
        IfPresentProbeSite,
        EarlyExitAbsentProbeSite,
        SpliceProbeSite,
        FallbackTernaryProbeSite,
        SingleNodeComprehensionFilterProbeSite,
        DisjunctiveAbsentProbeSite,
        PresenceTestProbeSite,
    )

    @classmethod
    def for_field(cls, field: str) -> tuple[type[OptionalAstFieldDecoder], ...]:
        matching = (
            decoder
            for decoder in cls.__registry__.values()
            if field in decoder.field_names()
        )
        return tuple(sorted(matching, key=lambda d: 0 if len(d.field_names()) > 1 else 1))

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(cls.fields.values())

    @staticmethod
    def _ast_subclass_names(root: type[ast.AST]) -> frozenset[str]:
        found: set[str] = set()
        stack: list[type[ast.AST]] = [root]
        while stack:
            cur = stack.pop()
            if cur.__name__ not in found:
                found.add(cur.__name__)
                stack.extend(cur.__subclasses__())
        return frozenset(found)

    @classmethod
    def classes_with(cls, field: str) -> frozenset[str]:
        """The ast classes (including ast subclasses) on which ``field`` is owned."""

        names: set[str] = set()
        for node_type, owned in cls.fields.items():
            if owned == field:
                names.update(cls._ast_subclass_names(node_type))
        return frozenset(names)

    @classmethod
    def collects(
        cls, collection: ast.Attribute, authority: OptionalAstFieldAuthority
    ) -> bool:
        """Whether reading ``collection`` yields this decoder's elements."""
        return False

    @classmethod
    def collected_classes(cls) -> frozenset[str]:
        """The ast classes a proved collection of this decoder holds."""
        return frozenset()

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

    element_type: ClassVar[type[ast.AST]]
    collection_owner: ClassVar[str]
    present: ClassVar[str]
    absent: ClassVar[str]
    present_field: ClassVar[str]
    present_attributes: ClassVar[frozenset[str]]
    collection_variable: ClassVar[str]
    shapes: ClassVar[tuple[type[ProbeSite], ...]] = (
        *CollectionProbeSite.__subclasses__(),
        IfPresentProbeSite,
        EarlyExitAbsentProbeSite,
        SpliceProbeSite,
        FallbackTernaryProbeSite,
        SingleNodeComprehensionFilterProbeSite,
        DisjunctiveAbsentProbeSite,
        PresenceTestProbeSite,
    )

    @classmethod
    def collected_classes(cls) -> frozenset[str]:
        return frozenset((cls.element_type.__name__,))

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
    element_type = ast.keyword
    collection_owner = "CallKeywords"
    present = "named"
    absent = "splats"
    present_field = "name"
    present_attributes = frozenset(("arg", "value", "node"))
    collection_variable = "keywords"

    @classmethod
    def collects(cls, collection, authority):
        return collection.attr == "keywords"

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"CallKeyword.decoded({subject_source}).names"


class ExceptHandlerDecoder(OptionalAstFieldDecoder, ABC):
    """Shared collection proof for optional fields of ``ast.ExceptHandler``."""

    @classmethod
    def collected_classes(cls) -> frozenset[str]:
        return frozenset(("ExceptHandler",))

    @classmethod
    def collects(cls, collection, authority) -> bool:
        if collection.attr != "handlers":
            return False
        owner_classes = authority.classes_of(collection.value, collection)
        return bool(owner_classes) and owner_classes <= frozenset(("Try", "TryStar"))


class CaptureNameDecoder(ExceptHandlerDecoder):
    """The name an except handler or capture pattern binds itself."""

    fields = {
        ast.ExceptHandler: "name",
        ast.MatchAs: "name",
        ast.MatchStar: "name",
        ast.MatchMapping: "rest",
    }
    owners = (ast_field_decoders.capture_names,)

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"capture_names({subject_source})"


class DeclaredAnnotationDecoder(CollectionDecoder):
    """The annotation a parameter declares, or the return annotation of a function.

    A parameter list splits into annotated and unannotated parameters.
    """

    fields = {
        ast.arg: "annotation",
        ast.FunctionDef: "returns",
        ast.AsyncFunctionDef: "returns",
    }
    owners = (
        ast_field_decoders.declared_annotations,
        ast_field_decoders.AnnotatedParameter,
        ast_field_decoders.ParameterAnnotations,
    )
    element_type = ast.arg
    collection_owner = "ParameterAnnotations"
    present = "annotated"
    absent = "unannotated"
    present_field = "annotation"
    present_attributes = frozenset(("arg", "annotation", "node"))
    collection_variable = "parameters"

    @classmethod
    def collects(cls, collection, authority):
        """``X.posonlyargs`` / ``X.kwonlyargs``, or ``args`` of a proved signature."""

        match collection:
            case ast.Attribute(attr="posonlyargs" | "kwonlyargs"):
                return True
            case ast.Attribute(attr="args", value=ast.Attribute(attr="args")):
                return True
            case ast.Attribute(attr="args", value=signature):
                return authority.classes_of(signature, collection) == frozenset(
                    ("arguments",)
                )
        return False

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"declared_annotations({subject_source})"


class VariadicParameterDecoder(OptionalAstFieldDecoder):
    """A signature's ``*args`` and ``**kwargs`` parameters, asked together."""

    fields = {ast.arguments: "vararg"}
    paired_fields: ClassVar[tuple[str, str]] = ("vararg", "kwarg")
    owners = (ast_field_decoders.variadic_parameters,)
    shapes = (PairedProbeSite, PairedIfPresentProbeSite, PairedAnyPresentProbeSite)

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(cls.paired_fields)

    @classmethod
    def classes_with(cls, field: str) -> frozenset[str]:
        return frozenset(("arguments",)) if field in cls.paired_fields else frozenset()

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"variadic_parameters({subject_source})"


class VarargParameterDecoder(OptionalAstFieldDecoder):
    """A signature's ``*args`` parameter asked on its own."""

    fields = {ast.arguments: "vararg"}
    owners = (ast_field_decoders.vararg_parameters,)

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"vararg_parameters({subject_source})"


class KwargParameterDecoder(OptionalAstFieldDecoder):
    """A signature's ``**kwargs`` parameter asked on its own."""

    fields = {ast.arguments: "kwarg"}
    owners = (ast_field_decoders.kwarg_parameters,)

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"kwarg_parameters({subject_source})"


class ImportAliasDecoder(OptionalAstFieldDecoder):
    """The explicit ``as ...`` alias name an import alias declares."""

    fields = {ast.alias: "asname"}
    owners = (ast_field_decoders.alias_names,)

    @classmethod
    def collected_classes(cls) -> frozenset[str]:
        return frozenset(("alias",))

    @classmethod
    def collects(cls, collection, authority) -> bool:
        if collection.attr != "names":
            return False
        owner_classes = authority.classes_of(collection.value, collection)
        return bool(owner_classes) and owner_classes <= frozenset(
            ("Import", "ImportFrom")
        )

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"alias_names({subject_source})"


class OptionalValueDecoder(OptionalAstFieldDecoder):
    """The optional expression value of a ``return``, annotated assignment, or ``yield``."""

    fields = {
        ast.Return: "value",
        ast.AnnAssign: "value",
        ast.Yield: "value",
    }
    owners = (ast_field_decoders.optional_values,)

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"optional_values({subject_source})"


class ImportModuleDecoder(OptionalAstFieldDecoder):
    """The optional module name spelled by ``ast.ImportFrom``."""

    fields = {ast.ImportFrom: "module"}
    owners = (ast_field_decoders.import_modules,)

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"import_modules({subject_source})"


class ExceptionTypeDecoder(ExceptHandlerDecoder):
    """The optional exception type expression on an ``except`` handler."""

    fields = {ast.ExceptHandler: "type"}
    owners = (ast_field_decoders.exception_types,)

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"exception_types({subject_source})"


class MatchGuardDecoder(OptionalAstFieldDecoder):
    """The optional ``if ...`` guard expression on an ``ast.match_case``."""

    fields = {ast.match_case: "guard"}
    owners = (ast_field_decoders.match_guards,)

    @classmethod
    def collected_classes(cls) -> frozenset[str]:
        return frozenset(("match_case",))

    @classmethod
    def collects(cls, collection, authority) -> bool:
        if collection.attr != "cases":
            return False
        owner_classes = authority.classes_of(collection.value, collection)
        return bool(owner_classes) and owner_classes <= frozenset(("Match",))

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"match_guards({subject_source})"


class MatchPatternDecoder(OptionalAstFieldDecoder):
    """The optional inner pattern wrapped by an ``ast.MatchAs``."""

    fields = {ast.MatchAs: "pattern"}
    owners = (ast_field_decoders.match_patterns,)

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"match_patterns({subject_source})"


_LOCATED_AST_BASES: tuple[type[ast.AST], ...] = (
    ast.stmt,
    ast.expr,
    ast.excepthandler,
    ast.arg,
    ast.keyword,
    ast.alias,
    ast.pattern,
    ast.type_param,
)


class EndLineDecoder(OptionalAstFieldDecoder):
    """The optional closing line number ``end_lineno`` on a located AST node."""

    fields = {base: "end_lineno" for base in _LOCATED_AST_BASES}
    owners = (ast_field_decoders.end_line_numbers,)

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"end_line_numbers({subject_source})"


class EndColumnDecoder(OptionalAstFieldDecoder):
    """The optional closing UTF-8 column offset ``end_col_offset`` on a located AST node."""

    fields = {base: "end_col_offset" for base in _LOCATED_AST_BASES}
    owners = (ast_field_decoders.end_column_offsets,)

    @classmethod
    def single(cls, subject_source: str) -> str:
        return f"end_column_offsets({subject_source})"


@dataclass(frozen=True)
class NamedOwnerDeclaration(ABC):
    """A top-level declaration carrying a decoder owner's name, decoded once by
    its source (`NativeDeclaration`), never by the name: the owner itself
    (`OwnerDeclaration`) or an unrelated namesake (`ForeignNamesake`)."""

    module_name: str
    file_path: str
    node: ast.ClassDef | ast.FunctionDef
    owner: NativeDeclaration

    @property
    def name(self) -> str:
        return self.node.name

    @staticmethod
    def decoded(
        module_name: str,
        file_path: str,
        node: ast.ClassDef | ast.FunctionDef,
        owner: NativeDeclaration,
    ) -> NamedOwnerDeclaration:
        kind = OwnerDeclaration if owner.source_matches(node) else ForeignNamesake
        return kind(module_name, file_path, node, owner)

    @abstractmethod
    def sort_into(
        self, owners: list[OwnerDeclaration], namesakes: list[ForeignNamesake]
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class OwnerDeclaration(NamedOwnerDeclaration):
    """The package's own copy of a decoder owner."""

    def sort_into(
        self, owners: list[OwnerDeclaration], namesakes: list[ForeignNamesake]
    ) -> None:
        owners.append(self)


@dataclass(frozen=True)
class ForeignNamesake(NamedOwnerDeclaration):
    """A different declaration under an owner's name."""

    def sort_into(
        self, owners: list[OwnerDeclaration], namesakes: list[ForeignNamesake]
    ) -> None:
        namesakes.append(self)

    @property
    def conflict(self) -> str:
        return (
            f"`{self.name}` at {self.file_path}:{self.node.lineno} is not the "
            f"decoder owner `{self.owner.qualified_name}` "
            f"({inspect.getsourcefile(self.owner.declaration)}:"
            f"{inspect.getsourcelines(self.owner.declaration)[1]}); "
            "the same name with a different declaration leaves the routing ambiguous"
        )


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
        return snapshot.derived((cls, root), lambda: cls(snapshot, root))

    @property
    def generated_module(self) -> str:
        return f"{self.root}.{GENERATED_MODULE}"

    @cached_property
    def named_declarations(self) -> tuple[NamedOwnerDeclaration, ...]:
        """Every top-level class or function in this root package that carries an
        owner's name, whether or not it is that owner."""

        owners = {
            owner.__name__: NativeDeclaration(owner)
            for decoder in OptionalAstFieldDecoder.__registry__.values()
            for owner in decoder.owners
        }
        return tuple(
            NamedOwnerDeclaration.decoded(
                parsed.module_name, str(parsed.path), statement, owner
            )
            for parsed in self.snapshot.parsed_modules
            if parsed.module_name.partition(".")[0] == self.root
            for statement in parsed.module.body
            if isinstance(statement, ast.ClassDef | ast.FunctionDef)
            and (owner := owners.get(statement.name)) is not None
        )

    @cached_property
    def sorted_declarations(
        self,
    ) -> tuple[tuple[OwnerDeclaration, ...], tuple[ForeignNamesake, ...]]:
        owners: list[OwnerDeclaration] = []
        namesakes: list[ForeignNamesake] = []
        for declaration in self.named_declarations:
            declaration.sort_into(owners, namesakes)
        return tuple(owners), tuple(namesakes)

    @property
    def owner_declarations(self) -> tuple[OwnerDeclaration, ...]:
        return self.sorted_declarations[0]

    @property
    def foreign_namesakes(self) -> tuple[ForeignNamesake, ...]:
        return self.sorted_declarations[1]

    @cached_property
    def top_level_names(self) -> Mapping[str, frozenset[str]]:
        """Module -> the owners it declares at top level, by declaration identity."""

        names: dict[str, set[str]] = {}
        for declaration in self.owner_declarations:
            names.setdefault(declaration.module_name, set()).add(declaration.name)
        return {module: frozenset(found) for module, found in names.items()}

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

    def require_unshadowed(self, names: Iterable[str]) -> None:
        """Refuse an owner name the package also gives a different declaration:
        importing or generating it would leave two meanings under one name."""

        names = frozenset(names)
        shadows = [
            namesake
            for namesake in self.foreign_namesakes
            if namesake.name in names
        ]
        if shadows:
            raise ValueError(
                "; ".join(declaration.conflict for declaration in shadows)
            )

    @cached_property
    def pending(self) -> tuple[type[OptionalAstFieldDecoder], ...]:
        """Decoders the package probes whose owners only generation can supply."""

        missing_decoders = tuple(
            decoder
            for decoder in OptionalAstFieldDecoder.__registry__.values()
            if not all(name in self.declared for name in decoder.owner_names())
        )
        if not missing_decoders:
            return ()
        needed_fields = frozenset(
            field_name
            for decoder in missing_decoders
            for field_name in decoder.field_names()
        )
        probed: set[str] = set()
        for parsed in self.snapshot.parsed_modules:
            if parsed.module_name.partition(".")[0] != self.root:
                continue
            source = parsed.source
            if "None" not in source or not any(
                field_name in source for field_name in needed_fields
            ):
                continue
            for _, node in parsed.syntax_index.indexed_nodes_of_type(ast.Compare):
                if _none_probe(node) and node.left.attr in needed_fields:
                    probed.add(node.left.attr)
        return tuple(
            decoder
            for decoder in missing_decoders
            if probed & decoder.field_names()
        )

    def generated_path(self) -> str:
        package = self.snapshot.module_import_graph.unique_source_file_for_module_name(
            self.root
        )
        if package is None or not package.is_package_init:
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

        self.require_unshadowed(
            name for decoder in self.pending for name in decoder.owner_names()
        )
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

        names = tuple(names)
        self.require_unshadowed(names)
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
                    ImportedClassReference(f"{module}.{name}").import_source
                )
        return tuple(sources)


@dataclass(frozen=True)
class OptionalAstFieldAuthority(FunctionRewriteAuthority):
    """Every routable probe of one decoder's fields inside one function."""

    decoder: type[OptionalAstFieldDecoder]
    snapshot: CodemodSourceSnapshot | None = None
    rewrite_name = "Probe routing"

    @cached_property
    def parents(self) -> AstParentIndex:
        """The function's own parent relation: its root has no parent here."""

        return AstParentIndex.of(self.function)

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

    @cached_property
    def _module_ast_aliases(self) -> Mapping[str, frozenset[str]]:
        """Module-level type aliases whose RHS resolves entirely to `ast.*` classes."""

        counts: dict[str, int] = {}
        raw: list[tuple[str, ast.expr]] = []
        for statement in self.parsed_module.module.body:
            match statement:
                case ast.TypeAlias(name=ast.Name(id=name), type_params=[], value=value):
                    counts[name] = counts.get(name, 0) + 1
                    raw.append((name, value))
                case ast.AnnAssign(target=ast.Name(id=name), value=ast.expr() as value):
                    counts[name] = counts.get(name, 0) + 1
                    raw.append((name, value))
                case ast.Assign(targets=targets, value=value):
                    for target in targets:
                        if isinstance(target, ast.Name):
                            counts[target.id] = counts.get(target.id, 0) + 1
                    if len(targets) == 1 and isinstance(targets[0], ast.Name):
                        raw.append((targets[0].id, value))
        resolved: dict[str, frozenset[str]] = {}
        progressed = True
        while progressed:
            progressed = False
            for name, value in raw:
                if name in resolved or counts.get(name, 0) != 1:
                    continue
                classes = self._ast_classes_with_aliases(value, resolved)
                if classes:
                    resolved[name] = classes
                    progressed = True
        return resolved

    @staticmethod
    def _ast_classes_with_aliases(
        annotation: ast.expr, aliases: Mapping[str, frozenset[str]]
    ) -> frozenset[str]:
        match annotation:
            case ast.Attribute(value=ast.Name(id="ast"), attr=name):
                return frozenset((name,))
            case ast.Name(id=name) if name in aliases:
                return aliases[name]
            case ast.BinOp(left=left, op=ast.BitOr(), right=right):
                left_classes = OptionalAstFieldAuthority._ast_classes_with_aliases(
                    left, aliases
                )
                right_classes = OptionalAstFieldAuthority._ast_classes_with_aliases(
                    right, aliases
                )
                if left_classes and right_classes:
                    return left_classes | right_classes
            case ast.Tuple(elts=items) if items:
                parts = [
                    OptionalAstFieldAuthority._ast_classes_with_aliases(item, aliases)
                    for item in items
                ]
                if all(parts):
                    return frozenset().union(*parts)
        return frozenset()

    def ast_classes(self, annotation: ast.expr) -> frozenset[str]:
        """The ``ast.X`` classes an annotation or isinstance target names, all or none."""

        return self._ast_classes_with_aliases(annotation, self._module_ast_aliases)

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

    def _match_pattern_ast_classes(
        self,
        pattern: ast.pattern,
        subject: ast.expr,
        match_subject: ast.expr,
    ) -> frozenset[str]:
        """AST classes proven for `subject` by a `match` case pattern."""

        def _pattern_classes(pat: ast.pattern) -> frozenset[str]:
            match pat:
                case ast.MatchClass(cls=cls_expr):
                    return self.ast_classes(cls_expr)
                case ast.MatchAs(pattern=ast.pattern() as inner):
                    return _pattern_classes(inner)
                case ast.MatchOr(patterns=alts) if alts:
                    parts = [_pattern_classes(alt) for alt in alts]
                    if all(parts):
                        return frozenset().union(*parts)
            return frozenset()

        if _same_expression(subject, match_subject):
            if found := _pattern_classes(pattern):
                return found

        if isinstance(subject, ast.Name):
            target_name = subject.id
            stack: list[ast.AST] = [pattern]
            found_aliases: list[frozenset[str]] = []
            while stack:
                cur = stack.pop()
                match cur:
                    case ast.MatchAs(pattern=ast.pattern() as inner, name=alias) if (
                        alias == target_name
                    ):
                        classes = _pattern_classes(inner)
                        if not classes:
                            return frozenset()
                        found_aliases.append(classes)
                    case ast.MatchClass(
                        cls=cls_expr, kwd_attrs=attrs, kwd_patterns=pats
                    ):
                        owner_classes = self.ast_classes(cls_expr)
                        for attr_name, kwd_pat in zip(attrs, pats):
                            match kwd_pat:
                                case ast.MatchAs(pattern=None, name=alias) if (
                                    alias == target_name
                                    and attr_name == "args"
                                    and owner_classes
                                    and owner_classes <= FUNCTION_CLASSES
                                ):
                                    found_aliases.append(frozenset(("arguments",)))
                                case _:
                                    stack.append(kwd_pat)
                        stack.extend(cur.patterns)
                    case _:
                        stack.extend(ast.iter_child_nodes(cur))
            if found_aliases:
                return frozenset().union(*found_aliases)
        return frozenset()

    def _negated_isinstance_classes(
        self, test: ast.expr, subject: ast.expr
    ) -> frozenset[str]:
        match test:
            case ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Call(
                    func=ast.Name(id="isinstance"), args=[checked, classes]
                ),
            ) if _same_expression(checked, subject):
                return self.ast_classes(classes)
        return frozenset()

    @staticmethod
    def _rebinds_subject(statements: Iterable[ast.stmt], subject: ast.expr) -> bool:
        names = {
            node.id for node in ast.walk(subject) if isinstance(node, ast.Name)
        }
        if not names:
            return False
        for stmt in statements:
            for node in ast.walk(stmt):
                if (
                    isinstance(node, ast.Name)
                    and node.id in names
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                ):
                    return True
        return False

    def _positive_isinstance_classes(
        self, test: ast.expr, subject: ast.expr
    ) -> frozenset[str]:
        parts = (
            test.values
            if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
            else [test]
        )
        for part in parts:
            match part:
                case ast.Call(
                    func=ast.Name(id="isinstance"), args=[checked, classes]
                ) if _same_expression(checked, subject):
                    if found := self.ast_classes(classes):
                        return found
        return frozenset()

    def guarded(self, subject: ast.expr, at: ast.AST) -> frozenset[str]:
        """Classes an enclosing `isinstance`, `not isinstance`, `assert`, comprehension `if`, or `match` proves at `at`."""

        for child, parent in self.ancestors(at):
            match parent:
                case ast.BoolOp(op=ast.And(), values=values):
                    index = next(i for i, value in enumerate(values) if value is child)
                    for test in values[:index]:
                        if found := self._positive_isinstance_classes(test, subject):
                            return found
                case ast.BoolOp(op=ast.Or(), values=values):
                    index = next(i for i, value in enumerate(values) if value is child)
                    for test in values[:index]:
                        if found := self._negated_isinstance_classes(test, subject):
                            return found
                case ast.If(test=test, body=body) if any(
                    statement is child for statement in body
                ):
                    if found := self._positive_isinstance_classes(test, subject):
                        return found
                case ast.match_case(pattern=pattern, guard=case_guard, body=body) if (
                    child is case_guard or any(statement is child for statement in body)
                ):
                    match_node = self.parents.get(parent)
                    if isinstance(match_node, ast.Match):
                        if found := self._match_pattern_ast_classes(
                            pattern, subject, match_node.subject
                        ):
                            return found
                case (
                    ast.ListComp() | ast.SetComp() | ast.GeneratorExp() | ast.DictComp()
                ):
                    if isinstance(child, ast.comprehension):
                        gen_index = next(
                            i for i, g in enumerate(parent.generators) if g is child
                        )
                        earlier_gens = parent.generators[:gen_index]
                    else:
                        earlier_gens = parent.generators
                    for gen in earlier_gens:
                        for clause_if in gen.ifs:
                            if found := self._positive_isinstance_classes(
                                clause_if, subject
                            ):
                                return found
                case ast.comprehension(ifs=ifs):
                    if any(clause_if is child for clause_if in ifs):
                        if_index = next(
                            i for i, clause_if in enumerate(ifs) if clause_if is child
                        )
                        for clause_if in ifs[:if_index]:
                            if found := self._positive_isinstance_classes(
                                clause_if, subject
                            ):
                                return found
            if isinstance(child, ast.stmt):
                located = _enclosing_block(parent, child)
                if located is not None:
                    block, stmt_idx = located
                    for offset, earlier in enumerate(reversed(block[:stmt_idx])):
                        between = block[stmt_idx - offset : stmt_idx]
                        if self._rebinds_subject(between, subject):
                            break
                        match earlier:
                            case ast.Assert(test=assert_test):
                                if found := self._positive_isinstance_classes(
                                    assert_test, subject
                                ):
                                    return found
                            case ast.If(test=guard_test, body=early_body, orelse=[]) if (
                                early_body
                                and isinstance(
                                    early_body[-1],
                                    (ast.Return, ast.Raise, ast.Continue, ast.Break),
                                )
                            ):
                                guard_parts = (
                                    guard_test.values
                                    if isinstance(guard_test, ast.BoolOp)
                                    and isinstance(guard_test.op, ast.Or)
                                    else [guard_test]
                                )
                                for part in guard_parts:
                                    if found := self._negated_isinstance_classes(
                                        part, subject
                                    ):
                                        return found
        return frozenset()

    def _iterated_source(
        self, target: ast.expr, iterable: ast.expr, name: str
    ) -> ast.expr | None:
        match target, iterable:
            case ast.Name(id=bound), _ if bound == name:
                return iterable
            case (
                ast.Tuple(elts=[_, ast.Name(id=bound)]),
                ast.Call(func=ast.Name(id="enumerate"), args=[inner, *_]),
            ) if bound == name:
                return inner
        return None

    def _iterable_ast_classes(
        self, iterable: ast.expr, at: ast.AST, seen: frozenset[str]
    ) -> frozenset[str]:
        """AST classes of elements produced by `iterable`, when statically proved."""

        if self.iterates_elements(iterable):
            return self.decoder.collected_classes()
        if declared := self._declared_iterable_ast_classes(iterable):
            return declared
        match iterable:
            case ast.Call(func=ast.Name(id="reversed" | "tuple" | "list"), args=[inner], keywords=[]):
                return self._iterable_ast_classes(inner, at, seen)
            case ast.Name(id=var_name):
                if found := self._annotated_collection_classes(var_name):
                    return found
            case ast.Tuple(elts=items) if items:
                combined: set[str] = set()
                for item in items:
                    if isinstance(item, ast.Starred):
                        part = self._iterable_ast_classes(item.value, at, seen)
                    else:
                        part = self.classes_of(item, at, seen)
                    if not part:
                        return frozenset()
                    combined.update(part)
                return frozenset(combined)
        return frozenset()

    def _indexed_iterated_classes(
        self, target: ast.expr, iterable: ast.expr, name: str
    ) -> frozenset[str]:
        match target, iterable:
            case (
                ast.Tuple(elts=[_, ast.Name(id=bound)]),
                ast.Call(
                    func=ast.Attribute(attr="indexed_nodes_of_type"),
                    args=[cls_expr],
                    keywords=[],
                ),
            ) if bound == name:
                return self.ast_classes(cls_expr)
        return frozenset()

    def iterated(
        self, name: str, at: ast.AST, seen: frozenset[str] = frozenset()
    ) -> frozenset[str]:
        """Classes proved by iterating a proved AST collection into ``name``."""

        for child, parent in self.ancestors(at):
            match parent:
                case ast.For(target=target, iter=iterable) if child is not iterable:
                    if indexed := self._indexed_iterated_classes(target, iterable, name):
                        return indexed
                    source = self._iterated_source(target, iterable, name)
                    if source is not None:
                        if found := self._iterable_ast_classes(source, parent, seen):
                            return found
                case (
                    ast.ListComp() | ast.SetComp() | ast.GeneratorExp() | ast.DictComp()
                ):
                    for generator in parent.generators:
                        if indexed := self._indexed_iterated_classes(
                            generator.target, generator.iter, name
                        ):
                            return indexed
                        source = self._iterated_source(
                            generator.target, generator.iter, name
                        )
                        if source is not None:
                            if found := self._iterable_ast_classes(
                                source, generator, seen
                            ):
                                return found
        return frozenset()

    def _bound_ast_classes(
        self, name: str, seen: frozenset[str]
    ) -> frozenset[str]:
        """Classes proved when every assignment to `name` in this function yields AST classes."""

        if name in seen:
            return frozenset()
        param_names = {
            arg.arg
            for arg in ast.walk(self.function.args)
            if isinstance(arg, ast.arg)
        }
        if name in param_names and not self.annotated(name):
            return frozenset()
        bindings = [
            node
            for node in ast.walk(self.function)
            if isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Store)
        ]
        if not bindings:
            return frozenset()
        next_seen = seen | {name}
        proved: set[str] = set(self.annotated(name))
        for binding in bindings:
            found: frozenset[str] = frozenset()
            match self.parents.get(binding):
                case ast.AnnAssign(target=target, annotation=ann, value=val) as assign if (
                    target is binding
                ):
                    found = self.ast_classes(ann) or (
                        self.classes_of(val, assign, next_seen)
                        if isinstance(val, ast.expr)
                        else frozenset()
                    )
                case ast.Assign(targets=[target], value=val) as assign if (
                    target is binding
                ):
                    found = self.classes_of(val, assign, next_seen)
                case ast.NamedExpr(target=target, value=val) as expr if (
                    target is binding
                ):
                    found = self.classes_of(val, expr, next_seen)
            if not found:
                return frozenset()
            proved.update(found)
        return frozenset(proved)

    def _declared_ast_classes(self, subject: ast.expr) -> frozenset[str]:
        """AST classes proved by the snapshot's `DeclaredReceiverTyping`."""

        if self.snapshot is None:
            return frozenset()
        declared = (
            ProbeOwnerResolver.of(self.parsed_module)
            .builder.path_of(subject)
            .declared(self.snapshot.declared_receiver_typing)
        )
        return self._ast_classes_from_declared(declared)

    def _declared_iterable_ast_classes(self, iterable: ast.expr) -> frozenset[str]:
        """Element AST classes proved for `iterable` by `DeclaredReceiverTyping`."""

        if self.snapshot is None:
            return frozenset()
        path = ProbeOwnerResolver.of(self.parsed_module).builder.path_of(iterable)
        element_path = ReceiverPath(path.root, (*path.steps, ElementStep()))
        declared = element_path.declared(self.snapshot.declared_receiver_typing)
        return self._ast_classes_from_declared(declared)

    @staticmethod
    def _ast_classes_from_declared(declared: object) -> frozenset[str]:
        if not isinstance(declared, DeclaredInstance) or not declared.symbols:
            return frozenset()
        names: list[str] = []
        for sym in declared.symbols:
            if sym.startswith(("ast.", "_ast.")):
                names.append(sym.split(".", 1)[1])
            else:
                return frozenset()
        return frozenset(names)

    def classes_of(
        self, subject: ast.expr, at: ast.AST, _seen: frozenset[str] = frozenset()
    ) -> frozenset[str]:
        if guarded := self.guarded(subject, at):
            return guarded
        match subject:
            case ast.Name(id=name):
                return (
                    self.annotated(name)
                    or self.iterated(name, at, _seen)
                    or self._bound_ast_classes(name, _seen)
                    or self._declared_ast_classes(subject)
                )
            case ast.Attribute(value=owner, attr="args"):
                owner_classes = self.classes_of(owner, at, _seen)
                if owner_classes == frozenset(("arguments",)):
                    return owner_classes
                if owner_classes and owner_classes <= FUNCTION_CLASSES:
                    return frozenset(("arguments",))
            case ast.Call(
                func=ast.Name(id="next"),
                args=[ast.GeneratorExp(elt=elt) as gen],
                keywords=[],
            ):
                if found := self.classes_of(elt, gen, _seen):
                    return found
            case ast.Call(
                func=ast.Name(id="as_ast") | ast.Attribute(attr="as_ast"),
                args=[_, cls_expr],
                keywords=[],
            ):
                if found := self.ast_classes(cls_expr):
                    return found
            case ast.Subscript(value=seq):
                if found := self._iterable_ast_classes(seq, at, _seen):
                    return found
        return self._declared_ast_classes(subject)

    def proves(self, subject: ast.expr, field: str, at: ast.AST) -> bool:
        """Whether ``subject`` is proved to be an ast class owning ``field``."""

        classes = self.classes_of(subject, at)
        return bool(classes) and classes <= self.decoder.classes_with(field)

    def _match_bound_collection(self, name: str, at: ast.AST) -> bool:
        """Whether an enclosing `match` case binds `name` to a collected attribute."""

        collected = self.decoder.collected_classes()
        if not collected:
            return False
        attr_owners: dict[str, tuple[frozenset[str], str]] = {
            "names": (frozenset(("Import", "ImportFrom")), "alias"),
            "handlers": (frozenset(("Try", "TryStar")), "ExceptHandler"),
            "keywords": (frozenset(("Call",)), "keyword"),
            "cases": (frozenset(("Match",)), "match_case"),
        }
        for child, parent in self.ancestors(at):
            match parent:
                case ast.match_case(pattern=pattern, guard=case_guard, body=body) if (
                    child is case_guard or any(statement is child for statement in body)
                ):
                    stack: list[ast.AST] = [pattern]
                    while stack:
                        cur = stack.pop()
                        match cur:
                            case ast.MatchClass(
                                cls=cls_expr, kwd_attrs=attrs, kwd_patterns=pats
                            ):
                                owner_classes = self.ast_classes(cls_expr)
                                for attr_name, kwd_pat in zip(attrs, pats):
                                    expected = attr_owners.get(attr_name)
                                    match kwd_pat:
                                        case ast.MatchAs(pattern=None, name=alias) if (
                                            alias == name
                                            and expected is not None
                                            and owner_classes
                                            and owner_classes <= expected[0]
                                            and frozenset((expected[1],)) <= collected
                                        ):
                                            return True
                                        case _:
                                            stack.append(kwd_pat)
                                stack.extend(cur.patterns)
                            case _:
                                stack.extend(ast.iter_child_nodes(cur))
        return False

    def _is_variadic_arg_filter(self, iterable: ast.expr) -> bool:
        """Whether `iterable` filters non-None `vararg`/`kwarg` of a proved `ast.arguments`."""

        if self.decoder is not DeclaredAnnotationDecoder:
            return False
        match iterable:
            case ast.Call(
                func=ast.Name(id="filter"),
                args=[
                    ast.Constant(value=None),
                    ast.Tuple(
                        elts=[
                            ast.Attribute(value=owner1, attr="vararg" | "kwarg"),
                            ast.Attribute(value=owner2, attr="vararg" | "kwarg"),
                        ]
                    ),
                ],
                keywords=[],
            ) if _same_expression(owner1, owner2) and (
                self.classes_of(owner1, iterable) == frozenset(("arguments",))
            ):
                return True
            case ast.IfExp(
                test=test,
                body=ast.Tuple(
                    elts=[ast.Attribute(value=owner, attr="vararg" | "kwarg") as attr]
                ),
                orelse=ast.Tuple(elts=[]),
            ) if (
                _same_expression(test, attr)
                or (
                    isinstance(test, ast.Compare)
                    and _same_expression(test.left, attr)
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.IsNot)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value is None
                )
            ) and self.classes_of(owner, iterable) == frozenset(("arguments",)):
                return True
            case ast.GeneratorExp(
                elt=ast.Name(id=elt_name),
                generators=[
                    ast.comprehension(
                        target=ast.Name(id=target_name),
                        iter=ast.Tuple(
                            elts=[
                                ast.Attribute(value=owner1, attr="vararg" | "kwarg"),
                                ast.Attribute(value=owner2, attr="vararg" | "kwarg"),
                            ]
                        ),
                        ifs=[
                            ast.Compare(
                                left=ast.Name(id=if_name),
                                ops=[ast.IsNot()],
                                comparators=[ast.Constant(value=None)],
                            )
                        ],
                    )
                ],
            ) if (
                elt_name == target_name == if_name
                and _same_expression(owner1, owner2)
                and self.classes_of(owner1, iterable) == frozenset(("arguments",))
            ):
                return True
        return False

    def iterates_elements(self, iterable: ast.expr) -> bool:
        """Whether the iterable is proved a collection of the decoder's elements.

        Proved by a collecting attribute, a tuple or concatenation of proved
        collections, a parameter annotated with the element type, a decoder call
        yielding those elements, a match-bound collection, declared element
        typing, or a local bound once to a proved collection.
        """

        collected = self.decoder.collected_classes()
        match iterable:
            case ast.Attribute():
                if self.decoder.collects(iterable, self):
                    return True
                if collected:
                    declared = self._declared_iterable_ast_classes(iterable)
                    if declared and declared <= collected:
                        return True
                return False
            case ast.Tuple(elts=items) if items:
                return all(
                    isinstance(item, ast.Starred) and self.iterates_elements(item.value)
                    for item in items
                )
            case ast.BinOp(left=left, op=ast.Add(), right=right):
                return self.iterates_elements(left) and self.iterates_elements(right)
            case ast.Call(
                func=ast.Name(id="tuple" | "list" | "reversed"),
                args=[only],
                keywords=[],
            ):
                return self.iterates_elements(only)
            case ast.Call(func=ast.Name(id="variadic_parameters"), args=[arg], keywords=[]) if (
                self.decoder is DeclaredAnnotationDecoder
                and self.classes_of(arg, iterable) == frozenset(("arguments",))
            ):
                return True
            case _ if self._is_variadic_arg_filter(iterable):
                return True
            case ast.Name(id=name):
                if (
                    self.annotated_collection(name)
                    or self.bound_collection(name)
                    or self._match_bound_collection(name, iterable)
                ):
                    return True
                if collected:
                    declared = self._declared_iterable_ast_classes(iterable)
                    if declared and declared <= collected:
                        return True
                return False
        return False

    def _annotated_collection_classes(self, name: str) -> frozenset[str]:
        """AST element classes declared in a parameter's container annotation (e.g. `tuple[FunctionNode, ...]`)."""

        arguments = self.function.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            if argument.arg != name:
                continue
            for annotation in ast_field_decoders.declared_annotations(argument):
                match annotation:
                    case ast.Subscript(slice=ast.Tuple(elts=[elem, ast.Constant(value=Ellipsis)])):
                        if found := self.ast_classes(elem):
                            return found
                    case ast.Subscript(slice=elem) if not isinstance(elem, ast.Tuple):
                        if found := self.ast_classes(elem):
                            return found
        return frozenset()

    def annotated_collection(self, name: str) -> bool:
        classes = self.decoder.collected_classes()
        if not classes:
            return False
        if (declared := self._annotated_collection_classes(name)) and declared <= classes:
            return True
        arguments = self.function.args
        return any(
            argument.arg == name
            and any(
                re.search(rf"\bast\.{element}\b", ast.unparse(annotation))
                for element in classes
                for annotation in ast_field_decoders.declared_annotations(argument)
            )
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        )

    def bound_collection(self, name: str) -> bool:
        """Whether ``name`` is bound exactly once, to a proved collection."""

        bindings = [
            node
            for node in ast.walk(self.function)
            if isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Store)
        ]
        if len(bindings) != 1:
            return False
        match self.parents.get(bindings[0]):
            case ast.Assign(targets=[target], value=value) if target is bindings[0]:
                return self.iterates_elements(value)
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
            case ast.GeneratorExp(elt=elt) if elt is node:
                call = self.parents.get(parent)
                return (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in ("all", "any")
                    and len(call.args) == 1
                    and call.args[0] is parent
                    and not call.keywords
                )
            case ast.Call(func=ast.Name(id="bool"), args=[only], keywords=[]) if (
                only is node
            ):
                return True
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
            replacement_source=(
                self.formatted(rewritten, start)
                if "\n" in rewritten
                else self.fit.fitted(rewritten, start, end)
            ),
        )

    def formatted(self, expression: str, start: int) -> str:
        """A multi-line expression laid out by the formatter, from its own column.

        Edits inside a multi-line expression keep the old line breaks around
        new text; the formatter lays the whole expression out again.
        """

        if "\n" not in expression:
            return expression
        line_start = self.geometry.line_offsets[
            self.geometry.line_number_for_offset(start) - 1
        ]
        prefix = self.geometry.source[line_start:start]
        indentation = prefix[: len(prefix) - len(prefix.lstrip())]
        return PythonExpressionSourceFormatter(
            self.line_length - (len(prefix) - len(indentation))
        ).replacement_source(
            ast.parse(expression.strip(), mode="eval").body, line_prefix=indentation
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

    def line_span_of(self, replacement: SourceTextSpanReplacement) -> tuple[int, int]:
        start_line = self.geometry.line_number_for_offset(replacement.start_offset)
        end_offset = max(replacement.start_offset, replacement.end_offset - 1)
        end_line = self.geometry.line_number_for_offset(end_offset)
        return start_line, end_line

    def resolve_replacements(
        self,
    ) -> tuple[tuple[SourceTextSpanReplacement, ...], dict[int, ValueError]]:
        """Collect all routable probe edits alongside the first refusal per unrouted line."""

        replacements: list[SourceTextSpanReplacement] = []
        refusals: list[ValueError] = []
        refusals_by_line: dict[int, ValueError] = {}
        for probe, site in self.sites():
            if site is None:
                error = self.refusal(
                    probe,
                    f"`{ast.unparse(probe)}` is in no shape the "
                    f"{self.decoder.__name__} can carry",
                )
                refusals.append(error)
                refusals_by_line.setdefault(probe.lineno, error)
                continue
            try:
                replacements.extend(site.replacements(self))
            except ValueError as error:
                refusals.append(error)
                refusals_by_line.setdefault(probe.lineno, error)
        if not replacements:
            raise (
                refusals[0]
                if refusals
                else ValueError(
                    f"No probe of {', '.join(sorted(self.decoder.field_names()))} "
                    f"for {self.decoder.__name__} in {self.function.name!r}"
                )
            )
        return (
            tuple(sorted(replacements, key=lambda item: item.start_offset)),
            refusals_by_line,
        )

    def replacements(
        self, *, required_lines: Collection[int] = ()
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """Edits for every routable probe; a function with none (or missing required lines) is handed back."""

        ordered, refusals_by_line = self.resolve_replacements()
        if required_lines:
            spans = tuple(self.line_span_of(item) for item in ordered)
            for line in sorted(set(required_lines)):
                if not any(start <= line <= end for start, end in spans):
                    if line in refusals_by_line:
                        raise refusals_by_line[line]
                    raise ValueError(
                        f"No routable probe of {', '.join(sorted(self.decoder.field_names()))} "
                        f"for {self.decoder.__name__} at {self.parsed_module.path}:{line} "
                        f"in {self.function.name!r}"
                    )
        return ordered


@dataclass(frozen=True, kw_only=True)
class RouteOptionalAstFieldProbesOperation(SourceReprovedOperation):
    """Rewrite one function's probes of an optional ast field onto its decoder."""

    decoder: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    @property
    def decoder_type(self) -> type[OptionalAstFieldDecoder]:
        decoder = OptionalAstFieldDecoder.__registry__.get(self.decoder)
        if decoder is None:
            raise ValueError(f"Unknown optional ast field decoder {self.decoder!r}")
        return decoder

    def owners(self, snapshot: CodemodSourceSnapshot) -> ProjectDecoderOwners:
        _, target, _ = self.target_node_from_context(snapshot)
        return ProjectDecoderOwners.of(
            snapshot,
            snapshot.parsed_module_for_source_path(target.file_path).module_name,
        )

    def _authority(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[AstTargetDigest, ParsedModule, OptionalAstFieldAuthority]:
        _, target, node = self.target_node_from_context(snapshot)
        parsed_module = snapshot.parsed_module_for_source_path(target.file_path)
        authority = OptionalAstFieldAuthority(
            parsed_module=parsed_module,
            geometry=SourceTextGeometry(
                snapshot.sources_by_file_path[target.file_path]
            ),
            line_length=DeclaredRoot.line_length_for(Path(target.file_path)),
            declaration=node,
            decoder=self.decoder_type,
            snapshot=snapshot,
        )
        return target, parsed_module, authority

    def _edits_from_replacements(
        self,
        snapshot: CodemodSourceSnapshot,
        target: AstTargetDigest,
        parsed_module: ParsedModule,
        authority: OptionalAstFieldAuthority,
        replacements: tuple[SourceTextSpanReplacement, ...],
    ) -> tuple[NominalSourceEdit, ...]:
        decoder = self.decoder_type
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
            authority.geometry.nominal_edit(
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

    def verify_probe_lines(
        self,
        snapshot: CodemodSourceSnapshot,
        required_lines: Collection[int],
    ) -> tuple[NominalSourceEdit, ...]:
        """Prove routing covers every evidenced probe line and imports resolve cleanly."""

        target, parsed_module, authority = self._authority(snapshot)
        replacements = authority.replacements(required_lines=required_lines)
        return self._edits_from_replacements(
            snapshot, target, parsed_module, authority, replacements
        )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        target, parsed_module, authority = self._authority(snapshot)
        replacements = authority.replacements()
        return self._edits_from_replacements(
            snapshot, target, parsed_module, authority, replacements
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


@dataclass(frozen=True)
class PendingDecoderMigration:
    """A package lacking decoders: one recipe declares them and routes every consumer."""

    owners: ProjectDecoderOwners

    @classmethod
    def of(cls, owners: ProjectDecoderOwners) -> PendingDecoderMigration:
        return owners.snapshot.derived((cls, owners.root), lambda: cls(owners))

    @cached_property
    def consumers(self) -> tuple[RoutedConsumer | Unroutable, ...]:
        """Every consumer probing a pending decoder's field, routed or needing more input.

        Each routing is proved on the current source with the generated module
        in place. That state is needed only here, so the migration, which its
        source state holds, does not keep it."""

        pending_with_fields = tuple(
            (decoder, decoder.field_names()) for decoder in self.owners.pending
        )
        if not pending_with_fields:
            return ()
        all_fields = frozenset(
            field_name
            for _decoder, fields in pending_with_fields
            for field_name in fields
        )
        snapshot = self.owners.snapshot.with_virtual_sources(
            {self.owners.generated_path(): self.owners.generated_source()}
        )
        found: list[RoutedConsumer | Unroutable] = []
        for parsed in sorted(
            self.owners.snapshot.parsed_modules, key=lambda module: str(module.path)
        ):
            if parsed.module_name.partition(".")[0] != self.owners.root:
                continue
            source = parsed.source
            if "None" not in source or not any(
                field_name in source for field_name in all_fields
            ):
                continue
            if not any(
                _none_probe(node) and node.left.attr in all_fields
                for _, node in parsed.syntax_index.indexed_nodes_of_type(ast.Compare)
            ):
                continue
            for qualname, function in _functions(parsed.module):
                fn_attrs = {
                    node.left.attr
                    for node in ast.walk(function)
                    if _none_probe(node) and node.left.attr in all_fields
                }
                if not fn_attrs:
                    continue
                paired_first = sorted(
                    pending_with_fields,
                    key=lambda item: 0 if len(item[1]) > 1 else 1,
                )
                claimed_offsets: list[tuple[int, int]] = []
                for decoder, fields in paired_first:
                    if fn_attrs & fields:
                        attempt = RoutedConsumer.attempt(
                            snapshot, str(parsed.path), qualname, decoder
                        )
                        if isinstance(attempt, RoutedConsumer):
                            if any(
                                start < o_end and o_start < end
                                for start, end in attempt.routed_offset_spans
                                for o_start, o_end in claimed_offsets
                            ):
                                continue
                            claimed_offsets.extend(attempt.routed_offset_spans)
                        found.append(attempt)
        return tuple(found)

    @cached_property
    def routes(self) -> tuple[RoutedConsumer, ...]:
        return tuple(
            consumer
            for consumer in self.consumers
            if isinstance(consumer, RoutedConsumer)
        )

    def route_for(
        self,
        file_path: str,
        symbol: str,
        decoder: type[OptionalAstFieldDecoder],
    ) -> RoutedConsumer | None:
        return next(
            (
                route
                for route in self.routes
                if route.key == (file_path, symbol) and route.decoder is decoder
            ),
            None,
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

    routed_line_spans: tuple[tuple[int, int], ...] = ()
    routed_offset_spans: tuple[tuple[int, int], ...] = ()
    unrouted_refusals: tuple[tuple[int, ValueError], ...] = ()

    def covers_line(self, line: int) -> bool:
        return any(start <= line <= end for start, end in self.routed_line_spans)

    def refusal_for_line(self, line: int) -> ValueError | None:
        return next(
            (
                error
                for error_line, error in self.unrouted_refusals
                if error_line == line
            ),
            None,
        )

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
            operation = _routing(snapshot, file_path, symbol, decoder)
            target, parsed_module, authority = operation._authority(snapshot)
            replacements, refusals_by_line = authority.resolve_replacements()
            operation._edits_from_replacements(
                snapshot, target, parsed_module, authority, replacements
            )
        except ValueError as error:
            return Unroutable(file_path, symbol, decoder, str(error))
        return RoutedConsumer(
            file_path,
            symbol,
            decoder,
            routed_line_spans=tuple(
                authority.line_span_of(item) for item in replacements
            ),
            routed_offset_spans=tuple(
                (item.start_offset, item.end_offset) for item in replacements
            ),
            unrouted_refusals=tuple(sorted(refusals_by_line.items())),
        )


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
        claimed_spans: dict[str, list[tuple[int, int]]] = {}
        refusals: list[str] = []
        needs: list[NeededInput] = []
        probed_fields: set[str] = set()
        all_evidence_covered = bool(finding.evidence)
        for evidence in finding.evidence:
            parsed_module = snapshot.parsed_module_for_source_path(evidence.file_path)
            fields = {
                probe.left.attr
                for probe in ast.walk(parsed_module.module)
                if _none_probe(probe) and probe.lineno == evidence.line
            }
            probed_fields |= fields
            owners = ProjectDecoderOwners.of(snapshot, parsed_module.module_name)
            evidence_covered = False
            for field_name in sorted(fields):
                for decoder in OptionalAstFieldDecoder.for_field(field_name):
                    try:
                        if decoder in owners.pending and owners.needs_declaration:
                            migration = PendingDecoderMigration.of(owners)
                            routed = migration.route_for(
                                evidence.file_path, evidence.symbol, decoder
                            )
                            if routed is not None and not routed.covers_line(
                                evidence.line
                            ):
                                line_error = routed.refusal_for_line(evidence.line)
                                if line_error is not None:
                                    refusals.append(str(line_error))
                                    needs.extend(NeedsInput.needs_of(line_error))
                                else:
                                    refusals.append(
                                        f"{decoder.__name__}: no routable probe at "
                                        f"{evidence.file_path}:{evidence.line}"
                                    )
                                continue
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
                            else:
                                evidence_covered = True
                            for operation in migrated:
                                operations[
                                    (
                                        operation.target.target_id or "",
                                        operation.decoder,
                                        type(operation).__name__,
                                    )
                                ] = operation
                            if evidence_covered:
                                break
                            continue
                        operation = _routing(
                            snapshot, evidence.file_path, evidence.symbol, decoder
                        )
                        edits = operation.verify_probe_lines(snapshot, (evidence.line,))
                    except ValueError as error:
                        refusals.append(str(error))
                        needs.extend(NeedsInput.needs_of(error))
                        continue
                    op_spans = tuple(
                        (replacement.start_offset, replacement.end_offset)
                        for edit in edits
                        if isinstance(edit, SourceTextMutation)
                        for replacement in edit.replacements
                    )
                    file_spans = claimed_spans.setdefault(evidence.file_path, [])
                    evidence_covered = True
                    if not any(
                        start < o_end and o_start < end
                        for start, end in op_spans
                        for o_start, o_end in file_spans
                    ):
                        file_spans.extend(op_spans)
                        operations[
                            (evidence.file_path, evidence.symbol, decoder.__name__)
                        ] = operation
                    break
            if not evidence_covered:
                all_evidence_covered = False
        if not operations or not all_evidence_covered:
            registered = sorted(
                {
                    name
                    for decoder in OptionalAstFieldDecoder.__registry__.values()
                    for name in decoder.field_names()
                }
            )
            return self.rejected_evaluation(
                f"needs a decoder for fields {', '.join(sorted(probed_fields)) or 'none'}; "
                f"registered decoders cover {', '.join(registered)}"
                + (f" ({'; '.join(refusals)})" if refusals else ""),
                NeededInput.distinct(needs) if refusals else (NeededInput.OPERATION,),
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
    "DisjunctiveAbsentProbeSite",
    "EarlyExitAbsentProbeSite",
    "EndColumnDecoder",
    "EndLineDecoder",
    "ExceptionTypeDecoder",
    "FallbackTernaryProbeSite",
    "ImportAliasDecoder",
    "ImportModuleDecoder",
    "KwargParameterDecoder",
    "MatchGuardDecoder",
    "MatchPatternDecoder",
    "OptionalAstFieldAuthority",
    "OptionalAstFieldDecoder",
    "OptionalAstFieldDecoderFindingRecipeSynthesizer",
    "OptionalValueDecoder",
    "PairedAnyPresentProbeSite",
    "PairedIfPresentProbeSite",
    "PairedProbeSite",
    "PendingDecoderMigration",
    "PresenceTestProbeSite",
    "ProjectDecoderOwners",
    "RouteOptionalAstFieldProbesOperation",
    "SingleNodeComprehensionFilterProbeSite",
    "VarargParameterDecoder",
    "VariadicParameterDecoder",
)
