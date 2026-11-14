"""Drop type parameters a class family no longer reads.

Serving a declaration moves its members down, and a type parameter that only those members read
moves with them: after serving, a wire declaration generic over its server's context names that
context in every subclass header (``RuntimeRequest[AttachedRuntimeRequestContext]``) while
nothing it declares reads it. Dropping the parameter is one family-wide move: the parameter
leaves each named class, and every subscript of that class in the repository loses the argument
at its position. A class whose own parameter was only passed on to a dropped position becomes
droppable in the same operation, so a chain (``ResultRuntimeRequest[Ctx](RuntimeRequest[Ctx])``)
leaves together."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

from nominal_refactor_advisor.ast_projection import name_nodes
from nominal_refactor_advisor.class_families.parsed_index import (
    ModuleClassReferenceResolver,
)
from nominal_refactor_advisor.codemod_payload import (
    CodemodPayloadRecord,
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    PhysicalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)


@dataclass(frozen=True, kw_only=True)
class DroppedTypeParameter(CodemodPayloadRecord):
    """One class and the type parameter it stops declaring."""

    target: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    parameter: str = codemod_payload_field(RequiredStringPayloadValueCodec())


@dataclass(frozen=True)
class GenericDeclaration:
    """A class losing type parameters: its arity before the drop and the positions that go."""

    source_class: ResolvedClassTarget
    symbol: str
    dropped: frozenset[str]

    @classmethod
    def require(
        cls,
        snapshot: CodemodSourceSnapshot,
        target: SourceRewriteTarget,
        parameters: frozenset[str],
    ) -> GenericDeclaration:
        source_class = ResolvedClassTarget.from_rewrite_target(snapshot, target)
        declared = {parameter.name: parameter for parameter in source_class.node.type_params}
        for name in parameters:
            if name not in declared:
                raise ValueError(f"{source_class.qualname!r} declares no type parameter {name!r}")
            if not isinstance(declared[name], ast.TypeVar):
                raise ValueError(
                    f"{source_class.qualname!r}'s {name!r} is not a plain type variable; a "
                    "variadic or parameter-spec argument has no fixed position to drop"
                )
        return cls(source_class, source_class.required_symbol(snapshot), parameters)

    @property
    def arity(self) -> int:
        return len(self.source_class.node.type_params)

    @property
    def kept_positions(self) -> tuple[int, ...]:
        return tuple(
            position for position, parameter in enumerate(self.source_class.node.type_params)
            if parameter.name not in self.dropped
        )

    def prefix_replacement(self, geometry: SourceTextGeometry) -> SourceTextSpanReplacement:
        """The bracketed parameter list after the class name, without the dropped names."""

        node = self.source_class.node
        first = geometry.required_node_offsets(node.type_params[0])[0]
        last = geometry.required_node_offsets(node.type_params[-1])[1]
        opening = geometry.source.rindex("[", 0, first)
        closing = geometry.source.index("]", last)
        if geometry.source[opening + 1:first].strip() or geometry.source[last:closing].strip(" \t\n,"):
            raise ValueError(f"{self.source_class.qualname!r}'s type parameters carry comments")
        kept = tuple(
            geometry.source[slice(*geometry.required_node_offsets(node.type_params[position]))]
            for position in self.kept_positions
        )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=opening,
            end_offset=closing + 1,
            replacement_source=f"[{', '.join(kept)}]" if kept else "",
        )


@dataclass(frozen=True)
class DroppedArgument:
    """A subscript of a class losing parameters, rewritten without the dropped arguments."""

    subscript: ast.Subscript
    declaration: GenericDeclaration

    @property
    def arguments(self) -> tuple[ast.expr, ...]:
        index = self.subscript.slice
        return tuple(index.elts) if isinstance(index, ast.Tuple) else (index,)

    @property
    def removed(self) -> tuple[ast.expr, ...]:
        kept = self.declaration.kept_positions
        return tuple(
            argument for position, argument in enumerate(self.arguments) if position not in kept
        )

    def replacement(self, geometry: SourceTextGeometry, where: str) -> SourceTextSpanReplacement:
        if len(self.arguments) != self.declaration.arity:
            raise ValueError(
                f"{where} passes {len(self.arguments)} arguments to "
                f"{self.declaration.source_class.qualname!r}, which takes {self.declaration.arity}"
            )
        start, end = geometry.required_node_offsets(self.subscript)
        value = geometry.source[slice(*geometry.required_node_offsets(self.subscript.value))]
        kept = tuple(
            geometry.source[slice(*geometry.required_node_offsets(self.arguments[position]))]
            for position in self.declaration.kept_positions
        )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start,
            end_offset=end,
            replacement_source=f"{value}[{', '.join(kept)}]" if kept else value,
        )


@dataclass(frozen=True)
class ModuleTypeArguments:
    """One module's subscripts of the classes losing parameters."""

    snapshot: CodemodSourceSnapshot
    file_path: str
    declarations: Mapping[str, GenericDeclaration]

    @cached_property
    def geometry(self) -> SourceTextGeometry:
        return SourceTextGeometry(self.snapshot.sources_by_file_path[self.file_path])

    @cached_property
    def arguments(self) -> tuple[DroppedArgument, ...]:
        resolver = ModuleClassReferenceResolver(
            self.snapshot.parsed_module_for_source_path(self.file_path),
            self.snapshot.required_class_family_index,
        )
        found = tuple(
            DroppedArgument(node, self.declarations[symbol])
            for node in ast.walk(self.snapshot.module_nodes_by_file_path[self.file_path])
            if isinstance(node, ast.Subscript)
            and (symbol := resolver.symbol_for_reference(node.value)) in self.declarations
        )
        nested = {
            id(inner.subscript)
            for outer in found
            for argument in outer.arguments
            for inner in found
            if inner.subscript in ast.walk(argument)
        }
        if nested:
            raise ValueError(
                f"{self.file_path} subscripts a class losing parameters inside another one's "
                "arguments; drop them in separate operations"
            )
        return found

    @property
    def removed_reads(self) -> frozenset[int]:
        """Names the rewrite deletes: those inside a dropped argument."""

        return frozenset(
            id(name)
            for argument in self.arguments
            for removed in argument.removed
            for name in name_nodes(removed)
        )

    def replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        return tuple(
            argument.replacement(
                self.geometry, f"{self.file_path}:{argument.subscript.lineno}",
            )
            for argument in self.arguments
        )

    def string_spellings(self) -> tuple[str, ...]:
        """String annotations that subscript a class losing parameters: no resolver reads them."""

        names = "|".join(
            re.escape(declaration.source_class.name) for declaration in self.declarations.values()
        )
        pattern = re.compile(rf"\b(?:{names})\s*\[")
        return tuple(
            f"{self.file_path}:{node.lineno}"
            for node in ast.walk(self.snapshot.module_nodes_by_file_path[self.file_path])
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and pattern.search(node.value)
        )


@dataclass(frozen=True, kw_only=True)
class DropTypeParametersOperation(RepositorySourceReprovedOperation):
    """Drop type parameters: each named class stops declaring them, and every subscript of
    those classes in the repository loses the argument at the dropped position. Refused when a
    class still reads a dropped parameter outside an argument this operation deletes, when a
    subscript's argument count does not match the class's parameters, or when a string
    annotation spells a subscript no resolver can read."""

    parameters: tuple[DroppedTypeParameter, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(DroppedTypeParameter)
    )

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("Dropping type parameters requires at least one parameter")

    def declarations(self, snapshot: CodemodSourceSnapshot) -> dict[str, GenericDeclaration]:
        by_target: dict[SourceRewriteTarget, set[str]] = {}
        for dropped in self.parameters:
            by_target.setdefault(dropped.target, set()).add(dropped.parameter)
        resolved = tuple(
            GenericDeclaration.require(snapshot, target, frozenset(names))
            for target, names in by_target.items()
        )
        return {declaration.symbol: declaration for declaration in resolved}

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        declarations = self.declarations(snapshot)
        names = tuple(declaration.source_class.name for declaration in declarations.values())
        modules = tuple(
            ModuleTypeArguments(snapshot, file_path, declarations)
            for file_path, source in snapshot.sources_by_file_path.items()
            if any(name in source for name in names)
        )
        spelled = tuple(spelling for module in modules for spelling in module.string_spellings())
        if spelled:
            raise ValueError(
                "String annotations subscript a class losing parameters: " + ", ".join(spelled)
            )
        removed = frozenset().union(*(module.removed_reads for module in modules))
        failures = tuple(
            f"{declaration.source_class.file_path}:{name.lineno} "
            f"{declaration.source_class.qualname} reads {name.id!r}"
            for declaration in declarations.values()
            for part in (
                *declaration.source_class.node.bases,
                *declaration.source_class.node.keywords,
                *declaration.source_class.node.body,
            )
            for name in name_nodes(part)
            if name.id in declaration.dropped
            and id(name) not in removed
        )
        if failures:
            raise ValueError("Type parameters are still read: " + "; ".join(failures))
        edits: list[PhysicalSourceEdit] = []
        for module in modules:
            replacements = [
                *module.replacements(),
                *(
                    declaration.prefix_replacement(module.geometry)
                    for declaration in declarations.values()
                    if declaration.source_class.file_path == module.file_path
                ),
            ]
            if replacements:
                edits.extend(module.geometry.physical_edits(
                    file_path=module.file_path,
                    replacements=sorted(replacements, key=lambda each: each.start_offset),
                    rationale=self.rationale or "Drop type parameters the family no longer reads.",
                ))
        return tuple(edits)
