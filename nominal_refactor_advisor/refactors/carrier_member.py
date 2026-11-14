"""A text field that only carries a family member's declared name becomes the member (BOUND-8)."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from functools import cached_property

from nominal_refactor_advisor.codemod_payload import (
    PayloadRecordValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import (
    PreflightByReproof,
    RepositorySourceReprovedOperation,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.refactors.answered_method import (
    AnsweredField,
    AnsweredUse,
    AttributeLookup,
)

DECLARED_NAME = "declared_name"


@dataclass(frozen=True)
class CarriedMember:
    """`Member.declared_name`, one value the field carries: the member class
    expression as written, and the member it resolves to."""

    expression: ast.expr
    symbol: str

    @property
    def source(self) -> str:
        return ast.unparse(self.expression)


@dataclass(frozen=True)
class CarrierField:
    """The field, its family and the members it may carry, read once from the snapshot."""

    snapshot: CodemodSourceSnapshot
    answered: AnsweredField
    family: ResolvedClassTarget

    @cached_property
    def family_symbol(self) -> str:
        return self.family.required_symbol(self.snapshot)

    @cached_property
    def members(self) -> frozenset[str]:
        index = self.snapshot.required_class_family_index
        return frozenset((self.family_symbol, *index.descendant_symbols(self.family_symbol)))

    @cached_property
    def optional(self) -> bool:
        """Whether the annotation admits None: `str | None` or `Optional[str]`."""

        text = ast.unparse(self.annotation)
        if text == "str":
            return False
        if text in {"str | None", "None | str", "Optional[str]", "typing.Optional[str]"}:
            return True
        raise ValueError(f"{self.answered.symbol}.{self.answered.name} is typed {text!r}, not text")

    @property
    def annotation(self) -> ast.expr:
        declaration = self.answered.method
        assert isinstance(declaration, ast.AnnAssign)
        return declaration.annotation

    def carried(self, file_path: str, value: ast.expr, parents: dict[ast.AST, ast.AST], where: str) -> CarriedMember | None:
        """The member a written value carries; None for an admitted None; refuses anything else."""

        if isinstance(value, ast.Constant) and value.value is None and self.optional:
            return None
        if isinstance(value, ast.Attribute) and value.attr == DECLARED_NAME:
            symbol = self.answered.class_symbol(file_path, value.value, parents)
            if symbol in self.members:
                return CarriedMember(value.value, symbol)
        raise ValueError(
            f"{where} writes {ast.unparse(value)!r} to {self.answered.name!r}, not a "
            f"{self.family.qualname} member's declared_name"
        )


@dataclass(frozen=True, kw_only=True)
class CarrierFieldBecomesMemberOperation(
    PreflightByReproof, RepositorySourceReprovedOperation
):
    """Retype a text field that only ever carries a family member's declared name
    to the member class itself, so consumers ask the member instead of comparing
    its spelling (BOUND-8).

    The target is `Class.field`, declared `field: str` or `field: str | None` in
    the class body; `family` is the declared family whose members it carries.
    The proof is closed-world over the scan, with every receiver typed:

    - every write is `Member.declared_name` for a member of `family` (or None
      where the annotation admits it): the field's default, constructor keywords
      of the class or a descendant, and attribute stores on a family receiver.
      A positional constructor argument, a `**` splat into a family constructor,
      a `replace(...)` naming the field or any other value refuses;
    - every read on a family receiver is rewritten: `== M.declared_name` and
      `!= M.declared_name` become `is M` and `is not M`, a membership test over
      a literal of them becomes one over the members, and any other read reads
      `.declared_name` off the member, the text it always was.

    The annotation becomes `type[Family]`, keeping `| None`. Encoding is the
    decoder's: a codec that spells a member class by its declared name (agent-
    comms FieldCodec's `_FamilyNameDecoder`) keeps every encoded byte. The
    operation does not see the codec; naming `family` asserts it.
    """

    family: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )

    preflight_passed_message = (
        "Every write carries a family member and every read is rewritten"
    )

    def carrier(self, snapshot: CodemodSourceSnapshot) -> CarrierField:
        source_path = self.required_source_path(snapshot, self.operation_key())
        qualname = self.target.qualname
        if qualname is None or "." not in qualname:
            raise ValueError("carrier_field_becomes_member requires a Class.field target_qualname")
        class_name, field_name = qualname.rsplit(".", 1)
        source_class = ResolvedClassTarget.from_rewrite_target(
            snapshot, SourceRewriteTarget(file_path=source_path, qualname=class_name)
        )
        declaration = next(
            (
                statement
                for statement in source_class.node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == field_name
            ),
            None,
        )
        if declaration is None:
            raise ValueError(f"{qualname!r} is not a field declared in {source_path!r}")
        carrier = CarrierField(
            snapshot,
            AnsweredField(snapshot, source_class, declaration),
            ResolvedClassTarget.from_rewrite_target(snapshot, self.family),
        )
        carrier.optional  # the annotation must be text before anything is read
        return carrier

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        carrier = self.carrier(snapshot)
        answered = carrier.answered
        declaration = answered.method
        assert isinstance(declaration, ast.AnnAssign)
        source_path = answered.source_class.file_path
        replacements: dict[str, list[tuple[ast.AST, str]]] = {}

        def replace(file_path: str, node: ast.AST, text: str) -> None:
            replacements.setdefault(file_path, []).append((node, text))

        source_parents = snapshot.parsed_module_for_source_path(source_path).fact(ModuleNodeIndex).parents
        family_name = carrier.family.node.name
        replace(
            source_path,
            declaration.annotation,
            f"type[{family_name}] | None" if carrier.optional else f"type[{family_name}]",
        )
        if declaration.value is not None:
            member = carrier.carried(source_path, declaration.value, source_parents, f"{source_path}:{declaration.lineno}")
            if member is not None:
                replace(source_path, declaration.value, member.source)
        index = snapshot.required_class_family_index
        spellings = {index.classes_by_symbol[symbol].simple_name for symbol in answered.family}
        for parsed in snapshot.parsed_modules:
            if not any(spelling in parsed.source for spelling in spellings):
                continue
            parents = parsed.fact(ModuleNodeIndex).parents
            for node in parsed.syntax_index.instances_of(ast.Call):
                self.constructor_writes(carrier, parsed.file_path, node, parents, replace)
        for use in answered.answering_calls():
            self.rewritten_use(carrier, use, replace)
        edits: list[NominalSourceEdit] = []
        for file_path, file_replacements in replacements.items():
            geometry = SourceTextGeometry(snapshot.sources_by_file_path[file_path])
            edits.append(
                geometry.nominal_edit(
                    file_path=file_path,
                    replacements=tuple(
                        SourceTextSpanReplacement.from_offsets(
                            start_offset=start, end_offset=end, replacement_source=text
                        )
                        for node, text in file_replacements
                        for start, end in (geometry.required_node_offsets(node),)
                    ),
                    rationale=self.rationale
                    or f"{answered.symbol}.{answered.name} carries the {family_name} member itself.",
                )
            )
        edits.extend(self.family_import(snapshot, carrier, source_path))
        return tuple(edits)

    def constructor_writes(self, carrier, file_path, call, parents, replace) -> None:
        """Rewrite the field's keyword in a constructor of the class family; refuse a
        write the keywords cannot show."""

        answered = carrier.answered
        where = f"{file_path}:{call.lineno}"
        if AttributeLookup.callee_name(call.func) == "replace" and any(
            keyword.arg == answered.name for keyword in call.keywords
        ):
            raise ValueError(f"{where} replaces {answered.name!r} through replace()")
        if answered.class_symbol(file_path, call.func, parents) not in answered.family:
            return
        if call.args:
            raise ValueError(f"{where} constructs {answered.symbol} positionally; keyword it first")
        if any(keyword.arg is None for keyword in call.keywords):
            raise ValueError(f"{where} splats keywords into {answered.symbol}")
        for keyword in call.keywords:
            if keyword.arg == answered.name:
                member = carrier.carried(file_path, keyword.value, parents, where)
                if member is not None:
                    replace(file_path, keyword.value, member.source)

    def rewritten_use(self, carrier: CarrierField, use: AnsweredUse, replace) -> None:
        snapshot = carrier.snapshot
        parents = snapshot.parsed_module_for_source_path(use.file_path).fact(ModuleNodeIndex).parents
        attribute = use.attribute
        if isinstance(attribute.ctx, ast.Store):
            assigned = parents.get(attribute)
            if not isinstance(assigned, (ast.Assign, ast.AnnAssign)) or assigned.value is None:
                raise ValueError(f"{use.where} writes {carrier.answered.name!r} other than by assignment")
            member = carrier.carried(use.file_path, assigned.value, parents, use.where)
            if member is not None:
                replace(use.file_path, assigned.value, member.source)
            return
        compare = parents.get(attribute)
        if isinstance(compare, ast.Compare):
            rewritten = self.compared(carrier, use.file_path, compare, attribute, parents)
            if rewritten is not None:
                replace(use.file_path, compare, rewritten)
                return
        replace(use.file_path, attribute, f"{ast.unparse(attribute)}.{DECLARED_NAME}")

    def compared(self, carrier, file_path, compare: ast.Compare, attribute, parents) -> str | None:
        """The comparison asked of the member instead of its spelling, when every
        other side spells members; None leaves the read to read the spelling."""

        if len(compare.ops) != 1:
            return None
        (op,) = compare.ops
        (other,) = [side for side in (compare.left, *compare.comparators) if side is not attribute]
        try:
            match op, other:
                case ast.Eq() | ast.NotEq(), ast.Attribute(attr="declared_name"):
                    member = carrier.carried(file_path, other, parents, f"{file_path}:{compare.lineno}")
                    if member is None:
                        return None
                    rewritten = copy.deepcopy(compare)
                    rewritten.ops = [ast.Is() if isinstance(op, ast.Eq) else ast.IsNot()]
                    rewritten.left, rewritten.comparators = (
                        (copy.deepcopy(attribute), [copy.deepcopy(member.expression)])
                    )
                    return ast.unparse(rewritten)
                case ast.In() | ast.NotIn(), ast.Tuple(elts=elements) | ast.List(elts=elements) | ast.Set(elts=elements) if (
                    compare.left is attribute
                ):
                    members = [carrier.carried(file_path, element, parents, f"{file_path}:{compare.lineno}") for element in elements]
                    if any(member is None for member in members):
                        return None
                    rewritten = copy.deepcopy(compare)
                    rewritten.comparators = [
                        type(other)(elts=[copy.deepcopy(member.expression) for member in members if member is not None])
                    ]
                    return ast.unparse(rewritten)
        except ValueError:
            return None
        return None

    def family_import(
        self, snapshot: CodemodSourceSnapshot, carrier: CarrierField, source_path: str
    ) -> tuple[ModuleImportMutation, ...]:
        """The family's import into the field's module, when it does not bind it."""

        family_path = carrier.family.file_path
        name = carrier.family.node.name
        if family_path == source_path:
            return ()
        module = snapshot.module_nodes_by_file_path[source_path]
        binding = carrier.answered.named_class(source_path, name, (module,))
        if binding is not None and binding.symbol == carrier.family_symbol:
            return ()
        return self.required_import_mutations(
            source_path,
            import_source=snapshot.module_import_graph.required_import_source(
                importing_file_path=source_path, imported_file_path=family_path, imported_name=name,
            ),
            default_rationale=f"The field is typed by {name}.",
        )
