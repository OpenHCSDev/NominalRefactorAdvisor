"""A roster naming some of a family's members, declared once as the base they share.

`after = (BusDeliveryReadersPhase, BusHistoryReadersPhase, BusCarryPhase)` names 3
of `VirtualBusPhase`'s members. Which three is a fact the source states only as that
literal, so a fourth reader phase is one more edit to it, forgotten or not. Declaring
the choice as a base between the family and those members (paper1 Prop 4.2) gives it
one owner: each member inherits the base, a value every member declares alike moves
onto it, and the roster reads the base's members, `tuple(BusReaderPhase.__subclasses__())`.

The base's name and docstring are a new identity, so the picker supplies them
(`OWNER_PICK`); everything else is proved from source:

- the members are direct subclasses of the authority in one module, so the base goes
  just above the first of them and each member's authority base becomes the base;
- the authority's own header keywords set to None (`declared_name=None`) are how its
  creation hook keeps an unnamed intermediate out of the family, so the base repeats
  them; a registering family with no such keyword needs that proof as input;
- a class attribute moves onto the base only when every member declares it with the
  same value, an ancestor declares the slot (so lookup reads it), and every name the
  value reads is bound before the base;
- the roster becomes the base's direct subclasses, which Python lists in definition
  order: an ordered roster whose order differs from the declarations keeps that order
  as its own fact and asks for the order instead.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from functools import cached_property
from typing import cast

from nominal_refactor_advisor.assignment_projection import (
    SingleAssignmentAndValueNameProjection,
)
from nominal_refactor_advisor.class_families.parsed_index import IndexedClass
from nominal_refactor_advisor.codemod_authority_claims import AstTargetAuthorityClaim
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot, RefactorRecipe
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceLineSpan,
    SourceSpanDeletion,
    SourceSpanReplacement,
    SourceTargetEditor,
)
from nominal_refactor_advisor.codemod_payload import (
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.class_symbols import ClassSymbolResolutionAuthority
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.refactors.authority_extraction import (
    SourceDerivedAuthorityProjectionOperation,
)
from nominal_refactor_advisor.refactors.class_family_collection import (
    ClassFamilyCollectionCandidate,
    ClassFamilyCollectionMembershipProjection,
    ClassFamilyCollectionProjection,
    ClassFamilyCollectionSemanticMirrorRecipeBuilder,
    ClassFamilyCollectionSite,
    PositionalCollectionReads,
)
from nominal_refactor_advisor.semantic_descent import SemanticAuthorityKind


@dataclass(frozen=True)
class HoistedAttribute:
    """A class attribute every subset member declares alike: one statement per member,
    the same value in each."""

    name: str
    statements: tuple[ast.Assign | ast.AnnAssign, ...]
    """Each member's declaration, in member order."""


@dataclass(frozen=True)
class SubsetFamilyBase:
    """The proved move declaring one roster's members as a base of their own."""

    site: ClassFamilyCollectionSite
    statement: ast.Assign | ast.AnnAssign
    collection: ClassFamilyCollectionProjection
    members: tuple[IndexedClass, ...]
    """The roster's members in declaration order."""
    member_bases: tuple[ast.expr, ...]
    """Each member's base naming the authority, in member order."""
    module_path: str
    source_lines: tuple[str, ...]
    base_name: str
    docstring: str
    creation_keywords: tuple[str, ...]
    hoisted: tuple[HoistedAttribute, ...]

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
        base_name: str,
        docstring: str,
    ) -> SubsetFamilyBase:
        if not base_name.isidentifier():
            raise NeedsInput(
                f"`{base_name}` is not a class name: the pick names the base",
                NeededInput.OWNER_PICK,
            )
        if not docstring.strip():
            raise NeedsInput(
                f"`{base_name}` needs a one-line docstring saying what its members share",
                NeededInput.OWNER_PICK,
            )
        site = ClassFamilyCollectionSite.from_context(
            context, authority_reference, projection_reference
        )
        proof = site.proof
        statement, collection = cls.roster(site)
        index = proof.class_index
        indexed = tuple(index.class_for(symbol) for symbol in collection.projected_symbols)
        if any(member is None for member in indexed):
            raise ValueError("a roster member has no indexed declaration")
        members = tuple(
            sorted((member for member in indexed if member is not None), key=lambda m: m.line)
        )
        paths = {member.file_path for member in members}
        if len(paths) != 1:
            raise NeedsInput(
                "the roster's members are declared in more than one module, so no one "
                "place declares their base before all of them",
                NeededInput.SOURCE_CHANGE,
            )
        (module_path,) = paths
        if module_path != site.projection_module.file_path:
            raise NeedsInput(
                "the roster sits in another module than its members: declaring their "
                "base there also needs the roster's import of it",
                NeededInput.OPERATION,
            )
        declared_order = tuple(member.symbol for member in members)
        if not collection.factory.preserves_member_sequence(
            collection.projected_symbols, declared_order
        ):
            raise NeedsInput(
                f"`{SingleAssignmentAndValueNameProjection(statement).required_name}` lists "
                "its members out of declaration order, and the base's members come in "
                "declaration order: the order is its own fact, so reorder the roster or "
                "the declarations first",
                NeededInput.SOURCE_CHANGE,
            )
        if any("." in member.qualname for member in members):
            raise ValueError("a roster member is nested in another class")
        cls.require_no_direct_subclass_reads(context, site.authority.name)
        last = max(member.node.end_lineno or member.line for member in members)
        if statement.lineno <= last:
            raise ValueError(
                "the roster runs before its last member is declared, so the base's "
                "members at that point are not all of them"
            )
        module = context.module_nodes_by_file_path[module_path]
        if base_name in cls.module_bindings(module):
            raise NeedsInput(
                f"`{base_name}` is already bound in {module_path}", NeededInput.OWNER_PICK
            )
        authority_node = site.authority.node
        creation_keywords = tuple(
            keyword.arg
            for keyword in authority_node.keywords
            if keyword.arg is not None
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is None
        )
        if proof.authority_declaration.declares_autoregister_meta and not creation_keywords:
            raise NeedsInput(
                f"`{site.authority.name}` registers each class its creation makes, and its "
                "header declares no keyword that keeps an unnamed base out",
                NeededInput.PROOF,
            )
        PositionalCollectionReads.require_none(
            context,
            ClassFamilyCollectionCandidate(
                statement=statement,
                collection=collection,
                membership=ClassFamilyCollectionMembershipProjection.DIRECT_SUBCLASSES,
            ),
        )
        source_lines = tuple(context.sources_by_file_path[module_path].splitlines())
        insertion_line = cls.header_start(members[0].node)
        return cls(
            site=site,
            statement=statement,
            collection=collection,
            members=members,
            member_bases=tuple(cls.authority_base(member, proof.authority_symbol) for member in members),
            module_path=module_path,
            source_lines=source_lines,
            base_name=base_name,
            docstring=docstring.strip(),
            creation_keywords=creation_keywords,
            hoisted=cls.hoisted_attributes(
                members,
                tuple(
                    sibling
                    for symbol in proof.class_index.children_by_symbol.get(
                        proof.authority_symbol, ()
                    )
                    if symbol not in declared_order
                    and (sibling := index.class_for(symbol)) is not None
                ),
                index,
                proof.authority_symbol,
                module,
                insertion_line,
            ),
        )

    @staticmethod
    def roster(
        site: ClassFamilyCollectionSite,
    ) -> tuple[ast.Assign | ast.AnnAssign, ClassFamilyCollectionProjection]:
        """The one literal naming at least two, and not all, of the family's direct
        members. A holder that is itself a member counts in the family, so a roster of
        every other member is a subset: deriving it from the whole family is the other
        reading, and the pick says which one the roster means."""

        proof = site.proof
        family = frozenset(proof.class_index.children_by_symbol.get(proof.authority_symbol, ()))
        rosters = tuple(
            (statement, collection)
            for statement in site.statements
            if (pair := SingleAssignmentAndValueNameProjection(statement).pair) is not None
            and pair[0] != "__all__"
            for collection in ClassFamilyCollectionProjection.from_value(
                pair[1],
                proof.reference.unavailable_builtin_names,
                proof.reference.resolver,
                proof.descendant_symbols,
            )
            if len(set(collection.projected_symbols)) == len(collection.projected_symbols) >= 2
            and frozenset(collection.projected_symbols) < family
            and proof.holder_symbol not in collection.projected_symbols
        )
        if len(rosters) != 1:
            raise ValueError(
                f"needs exactly one literal naming some of the `{proof.authority_name}` "
                f"members; found {len(rosters)}"
            )
        statement, collection = rosters[0]
        return cast(ast.Assign | ast.AnnAssign, statement), collection

    @staticmethod
    def authority_base(member: IndexedClass, authority_symbol: str) -> ast.expr:
        if authority_symbol not in member.resolved_base_symbols:
            raise ValueError(
                f"`{member.simple_name}` is not a direct subclass of the authority"
            )
        simple = authority_symbol.rsplit(".", 1)[-1]
        bases = tuple(
            base
            for base in member.node.bases
            if (declared := ClassSymbolResolutionAuthority.declared_base_name(base)) is not None
            and declared.rsplit(".", 1)[-1] == simple
        )
        if len(bases) != 1:
            raise ValueError(
                f"`{member.simple_name}` does not name the authority by exactly one base"
            )
        return bases[0]

    @staticmethod
    def require_no_direct_subclass_reads(
        context: CodemodSelectorContext, authority_name: str
    ) -> None:
        """Rebasing the members moves them out of the authority's direct subclasses,
        so a read of those would lose them; a registry over every descendant keeps them."""

        for path, module in context.module_nodes_by_file_path.items():
            for node in ast.walk(module):
                match node:
                    case ast.Attribute(attr="__subclasses__", value=value) if (
                        ast.unparse(value).rsplit(".", 1)[-1] == authority_name
                    ):
                        raise NeedsInput(
                            f"`{authority_name}.__subclasses__()` is read at "
                            f"{path}:{node.lineno}, and the rebased members leave it",
                            NeededInput.SOURCE_CHANGE,
                        )

    @staticmethod
    def header_start(node: ast.ClassDef) -> int:
        return min((decorator.lineno for decorator in node.decorator_list), default=node.lineno)

    @staticmethod
    def module_bindings(module: ast.Module) -> dict[str, int]:
        """Each name the module binds at its top level, with the line that first binds it."""

        bound: dict[str, int] = {}
        for statement in module.body:
            names: tuple[str, ...] = ()
            match statement:
                case ast.ClassDef(name=name) | ast.FunctionDef(name=name) | ast.AsyncFunctionDef(
                    name=name
                ):
                    names = (name,)
                case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                    names = tuple(
                        (alias.asname or alias.name).split(".")[0] for alias in aliases
                    )
                case ast.Assign(targets=targets):
                    names = tuple(
                        node.id
                        for target in targets
                        for node in ast.walk(target)
                        if isinstance(node, ast.Name)
                    )
                case ast.AnnAssign(target=ast.Name(id=name)):
                    names = (name,)
            for name in names:
                bound.setdefault(name, statement.lineno)
        return bound

    @classmethod
    def hoisted_attributes(
        cls,
        members: tuple[IndexedClass, ...],
        siblings: tuple[IndexedClass, ...],
        index,
        authority_symbol: str,
        module: ast.Module,
        insertion_line: int,
    ) -> tuple[HoistedAttribute, ...]:
        bindings = cls.module_bindings(module)
        slots = cls.ancestor_attributes(index, authority_symbol)
        declared = [cls.class_attributes(member.node) for member in members]
        outside = [cls.class_attributes(sibling.node) for sibling in siblings]
        hoisted: list[HoistedAttribute] = []
        for name, first in declared[0].items():
            statements = tuple(attributes.get(name) for attributes in declared)
            if any(statement is None for statement in statements):
                continue
            if name.startswith("__"):
                continue
            keys = {cls.declaration_key(statement) for statement in statements}
            if len(keys) != 1:
                continue
            if name not in slots and any(
                name in attributes and cls.declaration_key(attributes[name]) in keys
                for attributes in outside
            ):
                continue
            reads = {
                node.id
                for node in ast.walk(first.value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            if not all(
                (name_read in bindings and bindings[name_read] < insertion_line)
                or (name_read not in bindings and hasattr(builtins, name_read))
                for name_read in reads
            ):
                continue
            hoisted.append(HoistedAttribute(name, statements))  # type: ignore[arg-type]
        return tuple(hoisted)

    @staticmethod
    def class_attributes(node: ast.ClassDef) -> dict[str, ast.Assign | ast.AnnAssign]:
        """The plain class attributes a body declares once: `name = value` or
        `name: T = value`, never a statement the class header shares a line with."""

        found: dict[str, ast.Assign | ast.AnnAssign] = {}
        repeated: set[str] = set()
        for statement in node.body:
            match statement:
                case ast.Assign(targets=[ast.Name(id=name)]) | ast.AnnAssign(
                    target=ast.Name(id=name), value=ast.expr()
                ) if statement.lineno > node.lineno:
                    if name in found:
                        repeated.add(name)
                    found[name] = statement
        return {name: statement for name, statement in found.items() if name not in repeated}

    @staticmethod
    def declaration_key(statement: ast.Assign | ast.AnnAssign) -> str:
        annotation = statement.annotation if isinstance(statement, ast.AnnAssign) else None
        return ast.dump(statement.value) + "|" + (ast.dump(annotation) if annotation else "")

    @staticmethod
    def ancestor_attributes(index, authority_symbol: str) -> frozenset[str]:
        """The names the authority or a scanned ancestor binds in its class body: the
        slots a member's attribute overrides and lookup reads."""

        names: set[str] = set()
        frontier = [authority_symbol]
        seen: set[str] = set()
        while frontier:
            symbol = frontier.pop()
            if symbol in seen:
                continue
            seen.add(symbol)
            declaration = index.class_for(symbol)
            if declaration is None:
                continue
            names.update(declaration.member_binding_names)
            frontier.extend(declaration.resolved_base_symbols)
        return frozenset(names)

    # ---- rendering

    @property
    def authority_spelling(self) -> str:
        return ast.unparse(self.member_bases[0])

    @property
    def body_indent(self) -> str:
        first = self.members[0].node.body[0]
        return " " * first.col_offset

    def base_lines(self) -> tuple[str, ...]:
        indent = self.body_indent
        keywords = "".join(f", {keyword}=None" for keyword in self.creation_keywords)
        lines = [
            f"class {self.base_name}({self.authority_spelling}{keywords}):",
            f'{indent}"""{self.docstring}"""',
        ]
        for attribute in self.hoisted:
            statement = attribute.statements[0]
            lines.append("")
            lines.extend(self.source_lines[statement.lineno - 1 : statement.end_lineno])
        return (*lines, "", "")

    def rebased_header(self, member: IndexedClass, base: ast.expr) -> tuple[int, int, tuple[str, ...]]:
        """The member's header lines, its authority base renamed to the new base."""

        start = self.header_start(member.node)
        end = member.node.body[0].lineno - 1
        if end < member.node.lineno or base.end_lineno is None or base.end_col_offset is None:
            raise ValueError(f"`{member.simple_name}` declares its body on its header line")
        lines = list(self.source_lines[start - 1 : end])
        if base.lineno != base.end_lineno:
            raise ValueError(f"`{member.simple_name}` spells its base over several lines")
        row = base.lineno - start
        encoded = lines[row].encode("utf-8")
        lines[row] = (
            encoded[: base.col_offset] + self.base_name.encode("utf-8") + encoded[base.end_col_offset :]
        ).decode("utf-8")
        return start, end, tuple(lines)

    @staticmethod
    def physical(lines: tuple[str, ...]) -> tuple[str, ...]:
        return SourceTargetEditor.source_lines("\n".join(lines))

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        edits: list[NominalSourceEdit] = []
        for position, (member, base) in enumerate(zip(self.members, self.member_bases)):
            start, end, header = self.rebased_header(member, base)
            edits.append(
                SourceSpanReplacement(
                    file_path=self.module_path,
                    span=SourceLineSpan(start, end),
                    replacement_lines=self.physical(
                        (*self.base_lines(), *header) if position == 0 else header
                    ),
                    rationale=rationale,
                )
            )
            removed = tuple(attribute.statements[position] for attribute in self.hoisted)
            body_left = [
                statement
                for statement in member.node.body
                if statement not in removed
                and not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                    and statement is member.node.body[0]
                )
            ]
            has_docstring = (
                isinstance(member.node.body[0], ast.Expr)
                and isinstance(member.node.body[0].value, ast.Constant)
                and isinstance(member.node.body[0].value.value, str)
            )
            for number, statement in enumerate(removed):
                span = SourceLineSpan(statement.lineno, statement.end_lineno or statement.lineno)
                if number == 0 and not body_left and not has_docstring:
                    edits.append(
                        SourceSpanReplacement(
                            file_path=self.module_path,
                            span=span,
                            replacement_lines=self.physical((" " * statement.col_offset + "pass",)),
                            rationale=rationale,
                        )
                    )
                else:
                    edits.append(
                        SourceSpanDeletion(file_path=self.module_path, span=span, rationale=rationale)
                    )
        statement = self.statement
        value = self.collection.value_source(
            self.base_name, ClassFamilyCollectionMembershipProjection.DIRECT_SUBCLASSES
        )
        name = SingleAssignmentAndValueNameProjection(statement).required_name
        indent = " " * statement.col_offset
        replacement = (
            f"{indent}{name}: {ast.unparse(statement.annotation)} = {value}"
            if isinstance(statement, ast.AnnAssign)
            else f"{indent}{name} = {value}"
        )
        edits.append(
            SourceSpanReplacement(
                file_path=self.module_path,
                span=SourceLineSpan(statement.lineno, statement.end_lineno or statement.lineno),
                replacement_lines=self.physical((replacement,)),
                rationale=rationale,
            )
        )
        return tuple(edits)


@dataclass(frozen=True, kw_only=True)
class DeclareSubsetFamilyBaseOperation(SourceDerivedAuthorityProjectionOperation):
    """Declare a roster's members as a base of their own and read the roster from it."""

    base_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    docstring: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        move = self.required_move(snapshot)
        return move.source_edits(
            self.rationale_text(
                f"Declare `{self.base_name}` for the members "
                f"`{SingleAssignmentAndValueNameProjection(move.statement).required_name}` names."
            )
        )

    def required_move(self, context: CodemodSelectorContext) -> SubsetFamilyBase:
        return SubsetFamilyBase.from_context(
            context, self.target, self.projection_target, self.base_name, self.docstring
        )


class SubsetFamilyBaseRecipeBuilder(ClassFamilyCollectionSemanticMirrorRecipeBuilder):
    """Build the declared-base recipe for a roster whose finding carries a picked base."""

    def is_applicable(self) -> bool:
        return self.base_pick is not None

    @cached_property
    def base_operation(self) -> DeclareSubsetFamilyBaseOperation | None:
        pick = self.base_pick
        if self.targets is None or pick is None:
            return None
        return DeclareSubsetFamilyBaseOperation(
            target=SourceRewriteTarget(target_id=self.targets.authority.target.target_id),
            projection_target=SourceRewriteTarget(target_id=self.projection_target_id),
            base_name=pick.name,
            docstring=pick.judgement,
        )

    @cached_property
    def refusal(self) -> ValueError | None:
        operation = self.base_operation
        if operation is None:
            return ValueError(
                "semantic mirror finding does not resolve one class authority and "
                "one projection module"
            )
        try:
            operation.required_move(self)
        except ValueError as error:
            return error
        return None

    def recipe(self) -> RefactorRecipe | None:
        operation = self.base_operation
        if operation is None or self.targets is None or self.refusal is not None:
            return None
        return (
            RefactorRecipe(
                recipe_id=f"{self.finding.stable_id}-declare-subset-family-base",
                reason="Declare the roster's members as a base of their own.",
            )
            .with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    self.targets.authority.target,
                    authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                )
            )
            .with_operation(operation)
        )

    def rejection_reason(self) -> str:
        return "subset family base is available" if self.refusal is None else str(self.refusal)


__all__ = (
    "DeclareSubsetFamilyBaseOperation",
    "HoistedAttribute",
    "SubsetFamilyBase",
    "SubsetFamilyBaseRecipeBuilder",
)
