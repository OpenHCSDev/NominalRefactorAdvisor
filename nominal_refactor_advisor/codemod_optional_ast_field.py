"""Route absence probes of an optional ast field through its one boundary decoder.

An optional ast field (``keyword.arg``, ``AnnAssign.value`` ...) has exactly
two states, and a decoder owns the choice between them. A probe site that
tests the field for ``None`` restates that choice; this recipe rewrites the
site to read the decoder's answer instead, and refuses any site whose shape it
cannot carry without changing what the code means.
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Self, TypeGuard

from .ast_tools import ParsedModule, module_syntax_index
from .cases_family import handback
from .codemod import FindingEvidenceActionKeysMixin, FindingRecipeSynthesizer
from .codemod_foreign_family_source import FunctionRewriteAuthority
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


class OptionalAstFieldDecoder(ABC):
    """One registered decoder: the (ast class, field) it owns and how sites read it.

    The decoder splits a collection of ``element_type`` nodes into the present
    and absent states of ``field``. A present element keeps every attribute in
    ``present_attributes``; ``field`` itself is read as ``present_field``.
    """

    registry: ClassVar[dict[str, type[OptionalAstFieldDecoder]]] = {}
    element_type: ClassVar[type[ast.AST]]
    field: ClassVar[str]
    collection_attributes: ClassVar[frozenset[str]]
    import_path: ClassVar[str]
    present: ClassVar[str]
    absent: ClassVar[str]
    present_field: ClassVar[str]
    present_attributes: ClassVar[frozenset[str]]
    collection_variable: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        OptionalAstFieldDecoder.registry[cls.__name__] = cls

    @classmethod
    def for_field(cls, field: str) -> tuple[type[OptionalAstFieldDecoder], ...]:
        return tuple(
            decoder for decoder in cls.registry.values() if decoder.field == field
        )

    @classmethod
    def class_name(cls) -> str:
        return cls.import_path.rpartition(".")[2]

    @classmethod
    def module_name(cls) -> str:
        return cls.import_path.rpartition(".")[0]

    @classmethod
    def collected(cls, iterable_source: str) -> str:
        return f"{cls.class_name()}.of({iterable_source})"

    @classmethod
    def element_type_name(cls) -> str:
        return f"ast.{cls.element_type.__name__}"

    @classmethod
    def iterates_elements(cls, iterable: ast.expr, function: FunctionNode) -> bool:
        """Whether the iterable is a collection of this decoder's element type.

        Proved by the collecting attribute (``call.keywords``) or by the
        function's own parameter annotation naming the element type.
        """

        match iterable:
            case ast.Attribute(attr=attribute):
                return attribute in cls.collection_attributes
            case ast.Name(id=name):
                arguments = function.args
                return any(
                    argument.arg == name
                    and cls.element_type_name() in ast.unparse(argument)
                    for argument in (
                        *arguments.posonlyargs,
                        *arguments.args,
                        *arguments.kwonlyargs,
                    )
                )
        return False

    @classmethod
    def import_source_for(
        cls, snapshot: CodemodSourceSnapshot, parsed_module: ParsedModule
    ) -> str:
        """Import in the importing module's own style: relative when it is."""

        decoder_file = snapshot.module_import_graph.unique_source_file_for_module_name(
            cls.module_name()
        )
        uses_relative_imports = any(
            statement.level
            for _, statement in module_syntax_index(
                parsed_module.module
            ).indexed_nodes_of_type(ast.ImportFrom)
        )
        reference = (
            snapshot.module_import_graph.import_module_reference(
                importing_file_path=parsed_module.file_path,
                imported_file_path=decoder_file.file_path,
                imported_name=cls.class_name(),
            )
            if decoder_file is not None and uses_relative_imports
            else None
        )
        return f"from {reference or cls.module_name()} import {cls.class_name()}"


class CallKeywordArgDecoder(OptionalAstFieldDecoder):
    """``ast.keyword.arg``: a named keyword or a ``**`` splat (#75, f580f61)."""

    element_type = ast.keyword
    field = "arg"
    collection_attributes = frozenset(("keywords",))
    import_path = "nominal_refactor_advisor.ast_projection.CallKeywords"
    present = "named"
    absent = "splats"
    present_field = "name"
    present_attributes = frozenset(("arg", "value", "node"))
    collection_variable = "keywords"


@dataclass(frozen=True)
class ProbeSite(ABC):
    """One probe of ``V.field is [not] None`` on an element of a decoded collection."""

    decoder: type[OptionalAstFieldDecoder]
    probe: ast.Compare
    variable: str

    @classmethod
    @abstractmethod
    def at(
        cls,
        probe: ast.Compare,
        parents: dict[ast.AST, ast.AST],
        function: FunctionNode,
        decoder: type[OptionalAstFieldDecoder],
    ) -> Self | None:
        """This shape's site for the probe, when the probe sits in this shape."""
        raise NotImplementedError

    @abstractmethod
    def replacements(
        self, authority: OptionalAstFieldAuthority
    ) -> tuple[SourceTextSpanReplacement, ...]:
        raise NotImplementedError

    @staticmethod
    def probed(
        probe: ast.Compare, decoder: type[OptionalAstFieldDecoder], absent: bool
    ) -> str | None:
        """The probed variable when the probe tests the decoder's field as asked."""

        match probe:
            case ast.Compare(
                left=ast.Attribute(value=ast.Name(id=variable), attr=attribute),
                ops=[operator],
                comparators=[ast.Constant(value=None)],
            ) if attribute == decoder.field and isinstance(
                operator, ast.Is if absent else ast.IsNot
            ):
                return variable
        return None

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
                    f"element of {self.decoder.class_name()}",
                )
            if node.attr == self.decoder.field:
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
class QuantifiedProbeSite(ProbeSite):
    """``any(V.f is None for V in ITER)`` or ``all(V.f is not None ... for V in ITER)``."""

    call: ast.Call
    iterable: ast.expr
    remainder: tuple[ast.expr, ...]

    @classmethod
    def at(cls, probe, parents, function, decoder):
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
                elt=elt, generators=[ast.comprehension(target=ast.Name(id=name), iter=iterable, ifs=[])]
            ) if elt is elt_probe:
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
        if variable != name or not decoder.iterates_elements(iterable, function):
            return None
        return cls(decoder, probe, variable, call, iterable, remainder)

    def replacements(self, authority):
        geometry = authority.geometry
        start, end = geometry.required_node_offsets(self.call)
        iterable_source = authority.source_of(self.iterable)
        absent = f"{self.decoder.collected(iterable_source)}.{self.decoder.absent}"
        if isinstance(self.call.func, ast.Name) and self.call.func.id == "any":
            candidate = f"bool({absent})"
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
        if not authority.holds_boolean_operand(self.call) and not candidate.startswith(
            "bool("
        ):
            replacement = f"({replacement})"
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start, end_offset=end, replacement_source=replacement
            ),
        )


@dataclass(frozen=True)
class ComprehensionFilterProbeSite(ProbeSite):
    """``for V in ITER if V.f is not None [and REST]`` in a comprehension."""

    comprehension_node: ast.AST
    clause: ast.comprehension
    condition: ast.expr

    @classmethod
    def at(cls, probe, parents, function, decoder):
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
            case ast.Name(id=name) if name == variable and decoder.iterates_elements(
                parent.iter, function
            ):
                return cls(decoder, probe, name, parents[parent], parent, condition)
        return None

    def replacements(self, authority):
        geometry = authority.geometry
        iterable_start, iterable_end = geometry.required_node_offsets(self.clause.iter)
        condition_start, condition_end = geometry.required_node_offsets(self.condition)
        if geometry.span_contains_comment(SourceTextSpan(iterable_start, condition_end)):
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
            separator = geometry.source[
                geometry.required_node_offsets(self.previous_part())[1] : condition_start
            ]
            if re.fullmatch(r"\s+if\s+", separator) is None:
                raise authority.refusal(self.probe, "the filter is not a bare `if`")
            removal_start = geometry.required_node_offsets(self.previous_part())[1]
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
        return tuple(edits)

    def previous_part(self) -> ast.AST:
        """The node just before the probe's `if`: the iterable or an earlier filter."""

        index = next(
            index
            for index, clause_if in enumerate(self.clause.ifs)
            if clause_if is self.condition
        )
        return self.clause.ifs[index - 1] if index else self.clause.iter


@dataclass(frozen=True)
class LoopGuardProbeSite(ProbeSite):
    """``for V in ITER:`` whose first statement is ``if V.f is None: continue|return R``."""

    loop: ast.For
    guard: ast.If

    @classmethod
    def at(cls, probe, parents, function, decoder):
        guard = parents.get(probe)
        loop = parents.get(guard) if guard is not None else None
        match (guard, loop):
            case (
                ast.If(test=test, body=[ast.Continue() | ast.Return()], orelse=[]),
                ast.For(target=ast.Name(id=name), iter=iterable, body=[first, _, *_]),
            ) if test is probe and first is guard:
                pass
            case _:
                return None
        variable = cls.probed(probe, decoder, absent=True)
        if variable != name or not decoder.iterates_elements(iterable, function):
            return None
        return cls(decoder, probe, variable, loop, guard)

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
            SourceLineSpan(self.guard.lineno, self.guard.end_lineno or self.guard.lineno)
        )
        deletion_start, deletion_end = geometry.line_span_offsets(
            deleted.start_line, deleted.end_line
        )
        iterable_start, iterable_end = geometry.required_node_offsets(self.loop.iter)
        iterable_source = authority.source_of(self.loop.iter)
        edits = [
            SourceTextSpanReplacement.from_offsets(
                start_offset=deletion_start, end_offset=deletion_end, replacement_source=""
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
            indentation = geometry.source[loop_line_start : loop_line_start + self.loop.col_offset]
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
class OptionalAstFieldAuthority(FunctionRewriteAuthority):
    """Every routable probe of one decoder's field inside one function."""

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
            argument.arg for argument in ast.walk(self.function.args) if isinstance(argument, ast.arg)
        )

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

    def source_of(self, node: ast.AST) -> str:
        start, end = self.geometry.required_node_offsets(node)
        return self.geometry.source[start:end]

    def refusal(self, node: ast.AST, reason: str) -> ValueError:
        return handback(self.parsed_module, node, reason)

    def probes(self) -> Iterator[ast.Compare]:
        for node in ast.walk(self.function):
            match node:
                case ast.Compare(
                    left=ast.Attribute(attr=attribute),
                    ops=[ast.Is() | ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ) if attribute == self.decoder.field:
                    yield node

    def sites(self) -> tuple[tuple[ast.Compare, ProbeSite | None], ...]:
        return tuple(
            (
                probe,
                next(
                    (
                        site
                        for shape in ProbeSite.__subclasses__()
                        if (site := shape.at(probe, self.parents, self.function, self.decoder))
                        is not None
                    ),
                    None,
                ),
            )
            for probe in self.probes()
        )

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
                        f"{self.decoder.class_name()} decoder can carry",
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
                    f"No `{self.decoder.field}` probe in {self.function.name!r}"
                )
            )
        return tuple(sorted(replacements, key=lambda item: item.start_offset))


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

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        parsed_module = snapshot.parsed_module_for_source_path(target.file_path)
        decoder = self.decoder_type
        authority = OptionalAstFieldAuthority(
            parsed_module=parsed_module,
            geometry=SourceTextGeometry(snapshot.sources_by_file_path[target.file_path]),
            line_length=DeclaredRoot.line_length_for(Path(target.file_path)),
            declaration=node,
            decoder=decoder,
        )
        return (
            *authority.geometry.physical_edits(
                file_path=target.file_path,
                replacements=authority.replacements(),
                rationale=self.rationale
                or f"Route `{decoder.field}` probes in {target.qualname!r} through "
                f"{decoder.class_name()}.",
            ),
            *self.required_import_mutations(
                target.file_path,
                import_source=decoder.import_source_for(snapshot, parsed_module),
                default_rationale=f"Import {decoder.class_name()}.",
            ),
        )


class OptionalAstFieldDecoderFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    OptionalAstFieldDecoderConcept,
    ABC,
):
    """One operation per evidenced function whose probes a registered decoder owns."""

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
        operations: dict[tuple[str, str, str], RouteOptionalAstFieldProbesOperation] = {}
        refusals: list[str] = []
        for evidence in finding.evidence:
            parsed_module = snapshot.parsed_module_for_source_path(evidence.file_path)
            fields = {
                probe.left.attr
                for probe in ast.walk(parsed_module.module)
                if isinstance(probe, ast.Compare)
                and probe.lineno == evidence.line
                and isinstance(probe.left, ast.Attribute)
            }
            for field in sorted(fields):
                for decoder in OptionalAstFieldDecoder.for_field(field):
                    key = (evidence.file_path, evidence.symbol, decoder.__name__)
                    if key in operations:
                        continue
                    try:
                        (target_id,) = SourceIndexTargetSelector.for_function_or_method(
                            evidence.file_path, evidence.symbol
                        ).target_ids(snapshot)
                        operation = RouteOptionalAstFieldProbesOperation(
                            target=SourceRewriteTarget(target_id=target_id),
                            decoder=decoder.__name__,
                        )
                        operation.source_edits_from_snapshot(snapshot)
                    except ValueError as error:
                        refusals.append(str(error))
                        continue
                    operations[key] = operation
        if not operations:
            return self.rejected_evaluation(
                "no evidenced probe is a registered optional ast field"
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
    "CallKeywordArgDecoder",
    "OptionalAstFieldAuthority",
    "OptionalAstFieldDecoder",
    "OptionalAstFieldDecoderFindingRecipeSynthesizer",
    "RouteOptionalAstFieldProbesOperation",
)
