from __future__ import annotations

import ast
import builtins
import copy
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from nominal_refactor_advisor.refactors.answered_method import AnsweredMethod
from nominal_refactor_advisor.refactors.declaration_operations import (
    DeclarationDecoratorsPayload,
    DeclarationMutationOperationABC,
)
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod_call_source import (
    DeclaredCallArgumentsRewrite,
    DeclaredCallExpressionRewrite,
    DeclaredCallRewriteABC,
    DeclaredCallTargetRewrite,
)
from nominal_refactor_advisor.codemod_declaration_source import (
    FunctionAliasSourceAuthority,
    FunctionBindingProjectionSourceAuthority,
    FunctionBodyPrefixSourceAuthority,
    FunctionBodySourceAuthority,
    FunctionDecoratorsSourceAuthority,
    FunctionRegionSourceAuthority,
    FunctionSignatureSourceAuthority,
)
from nominal_refactor_advisor.codemod_foreign_adapter_source import (
    AdapterBindChainAuthority,
)
from nominal_refactor_advisor.codemod_foreign_family_source import (
    ForeignFamilyFilterAuthority,
    ForeignFamilyReference,
)
from nominal_refactor_advisor.codemod_import_scopes import ModuleImportScope
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_payload import (
    BooleanPayloadValueCodec,
    EmptyDefaultStringPayloadValueCodec,
    OptionalStringArrayPayloadValueCodec,
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
)
from nominal_refactor_advisor.codemod_reproof import (
    RepositorySourceReprovedOperation,
    SourceReproofDiagnostic,
    SourceReprovedOperation,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_import_graph import SourceModuleImportGraph
from nominal_refactor_advisor.codemod_module_declarations import ModuleSymbolTable
from nominal_refactor_advisor.codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from nominal_refactor_advisor.codemod_selector_models import (
    DeclaredCallSelection,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_semantics import CodemodPreflightStatus
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    PhysicalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.codemod_static_projection_source import (
    StaticProjection,
    StaticProjectionAuthority,
)
from nominal_refactor_advisor.declaration_dependencies import (
    FunctionBindingABC,
    FunctionLocalBinding,
    FunctionParameterBinding,
)
from nominal_refactor_advisor.declared_root import DeclaredRoot
from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.product_flow import BareCallTargetReference
from nominal_refactor_advisor.source_index import AstTargetNodeKind
from nominal_refactor_advisor.value_expression import LexicalValueReference


@dataclass(frozen=True, kw_only=True)
class FunctionMutationOperationABC(DeclarationMutationOperationABC, ABC):
    """Mutation family whose source authorities require a function declaration."""

    source_authority: ClassVar[type[FunctionRegionSourceAuthority]]


@dataclass(frozen=True, kw_only=True)
class AliasFunctionOperation(RepositorySourceReprovedOperation):
    """Replace a function with an alias to a source-proved same-scope implementation."""

    implementation: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[PhysicalSourceEdit, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        implementation_id = self.implementation.required_target_id(
            snapshot.source_index
        )
        implementation = snapshot.source_index.target_by_id[implementation_id]
        implementation_node = snapshot.ast_target_nodes_by_id[implementation_id]
        authority = FunctionAliasSourceAuthority(
            node=node,
            source=snapshot.sources_by_file_path[target.file_path],
        )
        if not isinstance(implementation_node, ast.FunctionDef | ast.AsyncFunctionDef):
            raise ValueError("Alias implementation must be a function declaration")
        if (target.file_path, target.qualname.rpartition(".")[0]) != (
            implementation.file_path,
            implementation.qualname.rpartition(".")[0],
        ):
            raise ValueError("Function alias declarations must share a lexical scope")
        target_symbol = snapshot.source_index.symbol_for_target(target)
        scope_symbol = target_symbol.rpartition(".")[0]
        repository = snapshot.product_flow_repository
        context = repository.flow_contexts_by_owner_symbol.get(scope_symbol)
        if context is None:
            raise ValueError("Function alias has no unique enclosing flow authority")
        target_mutations = tuple(
            mutation
            for mutation in context.flow.mutations_by_root_name.get(node.name, ())
            if mutation.line == node.lineno and mutation.kind.is_definition_binding
        )
        if len(target_mutations) != 1:
            raise ValueError("Function alias target has no unique definition binding")
        if not repository.resolve_function_target(
            context,
            BareCallTargetReference(implementation_node.name),
            target_mutations[0].position,
        ).declares_symbol(snapshot.source_index.symbol_for_target(implementation)):
            raise ValueError(
                "Alias implementation is not the selected preceding definition"
            )
        return authority.geometry.physical_edits(
            file_path=target.file_path,
            replacements=(authority.replacement(implementation_node.name),),
            rationale=self.rationale
            or f"Alias {target.qualname!r} to {implementation.qualname!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceFunctionDecoratorsOperation(
    DeclarationDecoratorsPayload, FunctionMutationOperationABC
):
    """Function-only decorator mutation using the shared source and payload authority."""

    source_authority = FunctionDecoratorsSourceAuthority


@dataclass(frozen=True, kw_only=True)
class ReplaceFunctionSignatureOperation(FunctionMutationOperationABC):
    """Replace function parameters and return annotation, preserving its body."""

    signature_suffix: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    source_authority = FunctionSignatureSourceAuthority
    replacement_source = AliasProperty[str]("signature_suffix")


@dataclass(frozen=True, kw_only=True)
class FunctionBodySourcePayload:
    """Authored statements shared by function-suite mutation operations."""

    body_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    replacement_source = AliasProperty[str]("body_source")


@dataclass(frozen=True, kw_only=True)
class ReplaceFunctionBodyOperation(
    FunctionBodySourcePayload, FunctionMutationOperationABC
):
    """Replace a function or method body while preserving its signature."""

    source_authority = FunctionBodySourceAuthority


@dataclass(frozen=True, kw_only=True)
class PrependFunctionBodyOperation(
    FunctionBodySourcePayload, FunctionMutationOperationABC
):
    """Insert statements after a function's docstring, retaining its existing body."""

    source_authority = FunctionBodyPrefixSourceAuthority


@dataclass(frozen=True, kw_only=True)
class DeclaredCallMutationOperationABC(
    DeclaredCallSelection, RepositorySourceReprovedOperation, ABC
):
    """Re-prove declaration-selected calls before deriving one authored mutation."""

    source_authority: ClassVar[type[DeclaredCallRewriteABC]]

    @property
    @abstractmethod
    def replacement_source(self) -> str:
        """Project the selected operation's authored payload."""
        raise NotImplementedError

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _identifier, caller = self.target_digest(snapshot)
        callee_id = self.callee.required_target_id(snapshot.source_index)
        authority = type(self).source_authority(
            snapshot,
            caller,
            snapshot.source_index.target_by_id[callee_id],
            self.replacement_source,
            self.selection_count,
        )
        return authority.geometry.physical_edits(
            file_path=caller.file_path,
            replacements=authority.replacements(),
            rationale=self.rationale
            or f"{self.operation_key()} in {caller.qualname!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceDeclaredCallArgumentsOperation(DeclaredCallMutationOperationABC):
    """Replace arguments of declaration-resolved calls in one selected scope."""

    arguments_source: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(), default=""
    )
    source_authority = DeclaredCallArgumentsRewrite
    replacement_source = AliasProperty[str]("arguments_source")


@dataclass(frozen=True, kw_only=True)
class CallExpressionSourcePayload:
    """Authored expression shared by whole-call and callable-only edits."""

    expression_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    replacement_source = AliasProperty[str]("expression_source")


@dataclass(frozen=True, kw_only=True)
class ReplaceDeclaredCallOperation(
    CallExpressionSourcePayload, DeclaredCallMutationOperationABC
):
    """Replace resolved calls with an authored expression; equivalence is author-owned."""

    source_authority = DeclaredCallExpressionRewrite


@dataclass(frozen=True, kw_only=True)
class ReplaceDeclaredCallTargetOperation(
    CallExpressionSourcePayload, DeclaredCallMutationOperationABC
):
    """Replace a resolved call's callable; new behaviour and binding are author-owned."""

    source_authority = DeclaredCallTargetRewrite


@dataclass(frozen=True, kw_only=True)
class FunctionBindingProjectionOperationABC(SourceReprovedOperation, ABC):
    """Rewrite owned reads through the selected lexical binding authority."""

    projection_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    binding_type: ClassVar[type[FunctionBindingABC]]

    attribute_path: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(), default=()
    )

    @property
    @abstractmethod
    def binding_name(self) -> str:
        """Name the binding selected by this operation's payload."""
        raise NotImplementedError

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _identifier, target, node = self.target_node_from_context(snapshot)
        authority = FunctionBindingProjectionSourceAuthority(
            node=node,
            source=snapshot.sources_by_file_path[target.file_path],
        )
        expression = ast.parse(self.projection_source, mode="eval").body
        reference = LexicalValueReference.from_expression(expression)
        if reference is None:
            raise ValueError("Binding projection requires a Name/Attribute access path")
        return authority.geometry.physical_edits(
            file_path=target.file_path,
            replacements=authority.replacements_for(
                type(self).binding_type(node, self.binding_name),
                reference,
                attribute_path=self.attribute_path,
            ),
            rationale=self.rationale
            or f"Project binding {self.binding_name!r} in {target.qualname!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class ProjectFunctionParameterOperation(FunctionBindingProjectionOperationABC):
    """Rewrite parameter reads; signature and caller migrations remain explicit."""

    parameter_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    binding_type = FunctionParameterBinding
    binding_name = AliasProperty[str]("parameter_name")


@dataclass(frozen=True, kw_only=True)
class ProjectFunctionLocalOperation(FunctionBindingProjectionOperationABC):
    """Rewrite a single-assignment local's reads onto a parameter access path."""

    local_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    binding_type = FunctionLocalBinding
    binding_name = AliasProperty[str]("local_name")


@dataclass(frozen=True, kw_only=True)
class RouteForeignFamilyFilterOperation(SourceReprovedOperation):
    """Route foreign-class filters in one function through a declared family.

    Rewrites `for V in ITER if isinstance(V, T)` comprehension clauses and
    `if not isinstance(V, T): continue` loop guards to `Family.within(ITER)`.
    A filter testing only part of the family would widen; that needs
    `accept_widening`.
    """

    family: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    accept_widening: bool = codemod_payload_field(
        BooleanPayloadValueCodec(), default=False
    )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        parsed_module = snapshot.parsed_module_for_source_path(target.file_path)
        reference = ForeignFamilyReference(self.family)
        authority = ForeignFamilyFilterAuthority(
            parsed_module=parsed_module,
            geometry=SourceTextGeometry(
                snapshot.sources_by_file_path[target.file_path]
            ),
            line_length=DeclaredRoot.line_length_for(Path(target.file_path)),
            declaration=node,
            reference=reference,
            accept_widening=self.accept_widening,
        )
        return (
            *authority.geometry.physical_edits(
                file_path=target.file_path,
                replacements=authority.replacements(),
                rationale=self.rationale
                or f"Route foreign-class filters in {target.qualname!r} through "
                f"{reference.class_name}.",
            ),
            *self.required_import_mutations(
                target.file_path,
                import_source=reference.import_source,
                default_rationale=f"Import {reference.class_name}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class FoldAdapterBindChainOperation(SourceReprovedOperation):
    """Fold bind chains in one function into reads of a declared adapter family.

    A chain of `isinstance` arms that each bind the same locals becomes
    `fresh = Family.decoded(subject)` followed by one property read per local,
    when the arms test exactly the family's adapted classes and every arm
    expression is that property's body for the arm's class.
    """

    family: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        parsed_module = snapshot.parsed_module_for_source_path(target.file_path)
        reference = ForeignFamilyReference(self.family)
        authority = AdapterBindChainAuthority(
            parsed_module=parsed_module,
            geometry=SourceTextGeometry(
                snapshot.sources_by_file_path[target.file_path]
            ),
            line_length=DeclaredRoot.line_length_for(Path(target.file_path)),
            declaration=node,
            reference=reference,
        )
        return (
            *authority.geometry.physical_edits(
                file_path=target.file_path,
                replacements=authority.replacements(),
                rationale=self.rationale
                or f"Fold bind chains in {target.qualname!r} into "
                f"{reference.class_name} property reads.",
            ),
            *self.required_import_mutations(
                target.file_path,
                import_source=reference.import_source,
                default_rationale=f"Import {reference.class_name}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ProjectStaticCallsToAdapterOperation(SourceReprovedOperation):
    """Rewrite a module's static projection calls into reads of an adapter family.

    `Projector.question(argument)` becomes `Family.decoded(argument).question`
    for every question the projector's public callables share by name with the
    family's properties. The family must be the unique one whose properties
    cover all of the projector's public callables. Bindings of the projector
    that no read keeps alive are removed from the module's imports.
    """

    projector: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    family: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        _, target = self.target_digest(snapshot)
        target.require_kind(
            AstTargetNodeKind.MODULE,
            "Static projection rewriting requires a module target",
        )
        projection = StaticProjection.declared_for(self.projector)
        reference = ForeignFamilyReference(self.family)
        if projection is None or projection.family is not reference.family:
            raise ValueError(
                f"{self.projector} is not a loaded static projection of {self.family}"
            )
        parsed_module = snapshot.parsed_module_for_source_path(target.file_path)
        authority = StaticProjectionAuthority(
            parsed_module=parsed_module,
            geometry=SourceTextGeometry(
                snapshot.sources_by_file_path[target.file_path]
            ),
            line_length=DeclaredRoot.line_length_for(Path(target.file_path)),
            projection=projection,
            family=reference,
            other_modules=tuple(
                snapshot.parsed_module_for_source_path(file_path)
                for file_path in sorted(snapshot.sources_by_file_path)
                if file_path != target.file_path
            ),
        )
        replacements = authority.replacements()
        if not replacements and not authority.retired_names:
            raise ValueError(
                f"No use of {self.projector} in {target.file_path!r} can move to "
                f"{self.family}"
            )
        imports = (
            ()
            if not replacements or parsed_module.module_name == reference.module_name
            else self.required_import_mutations(
                target.file_path,
                import_source=reference.import_source,
                default_rationale=f"Import {reference.class_name}.",
            )
        )
        retired = (
            ModuleImportMutation.remove_bound_names(
                file_path=target.file_path,
                names=authority.retired_names,
                rationale=self.rationale_text(
                    f"Drop the unused {self.projector} import."
                ),
            ),
        )
        return (
            *authority.geometry.physical_edits(
                file_path=target.file_path,
                replacements=replacements,
                rationale=self.rationale
                or f"Read {reference.class_name} properties in place of "
                f"{self.projector} calls.",
            ),
            *imports,
            *(retired if authority.retired_names else ()),
        )


@dataclass(frozen=True)
class _ForwarderBody:
    """A method whose whole body is one call built from its own parameters, after
    any imports that bind what it calls. An async forwarder awaits that call, so
    its callers' `await` reaches the forwarded call itself."""

    method: ast.FunctionDef | ast.AsyncFunctionDef
    call: ast.Call
    returns: bool
    imports: tuple[ast.Import | ast.ImportFrom, ...] = ()

    @classmethod
    def of(cls, method: ast.FunctionDef | ast.AsyncFunctionDef, qualname: str) -> _ForwarderBody:
        body = [
            statement
            for statement in method.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        imports = tuple(
            statement for statement in body if isinstance(statement, (ast.Import, ast.ImportFrom))
        )
        arguments = method.args
        if (
            method.decorator_list
            or arguments.vararg
            or arguments.kwarg
            or arguments.posonlyargs
            or not arguments.args
            or len(body) != len(imports) + 1
            or tuple(body[: len(imports)]) != imports
        ):
            raise ValueError(f"{qualname!r} is not a plain one-call forwarder")
        statement = body[-1]
        value = statement.value if isinstance(statement, (ast.Return, ast.Expr)) else None
        if isinstance(method, ast.AsyncFunctionDef):
            value = value.value if isinstance(value, ast.Await) else None
        if not isinstance(value, ast.Call):
            raise ValueError(f"{qualname!r} is not a plain one-call forwarder")
        return cls(method, value, isinstance(statement, ast.Return), imports)

    @property
    def free_names(self) -> frozenset[str]:
        """The names the forwarded call reads besides the forwarder's parameters."""

        return frozenset(
            node.id
            for node in ast.walk(self.call)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        ) - frozenset(self.parameter_names)

    def value_used(self, call: ast.Call, parents: Mapping[ast.AST, ast.AST]) -> bool:
        """Whether the call site reads the value of the call (or of its await)."""

        outer = parents.get(call)
        if isinstance(self.method, ast.AsyncFunctionDef) and isinstance(outer, ast.Await):
            outer = parents.get(outer)
        return not isinstance(outer, ast.Expr)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(argument.arg for argument in (*self.method.args.args, *self.method.args.kwonlyargs))

    def inlined(self, call_site: ast.Call, receiver: ast.expr, file_path: str) -> str:
        """The forwarded call with self and each parameter replaced by the call
        site's receiver and arguments."""

        if any(isinstance(argument, ast.Starred) for argument in call_site.args) or any(
            keyword.arg is None for keyword in call_site.keywords
        ):
            raise ValueError(f"{file_path!r}:{call_site.lineno} passes unpacked arguments")
        positional = self.method.args.args[1:]
        if len(call_site.args) > len(positional):
            raise ValueError(f"{file_path!r}:{call_site.lineno} passes too many arguments")
        bound: dict[str, ast.expr] = {self.method.args.args[0].arg: receiver}
        bound.update(
            (parameter.arg, argument) for parameter, argument in zip(positional, call_site.args)
        )
        bound.update((keyword.arg, keyword.value) for keyword in call_site.keywords)
        missing = tuple(name for name in self.parameter_names if name not in bound)
        if missing:
            raise ValueError(
                f"{file_path!r}:{call_site.lineno} relies on defaults for {missing!r}"
            )
        uses = [
            node.id
            for node in ast.walk(self.call)
            if isinstance(node, ast.Name) and node.id in bound
        ]
        unbalanced = tuple(name for name in self.parameter_names if uses.count(name) != 1)
        if unbalanced:
            raise ValueError(
                f"Inlining would not evaluate {unbalanced!r} exactly once at "
                f"{file_path!r}:{call_site.lineno}"
            )
        site_order = [self.method.args.args[0].arg, *(
            parameter.arg for parameter in positional[: len(call_site.args)]
        ), *(keyword.arg for keyword in call_site.keywords)]
        inlined_order = sorted(uses, key=lambda name: _source_position(self.call, name))
        if inlined_order != site_order and not all(
            _plain_read(bound[name]) for name in bound
        ):
            raise ValueError(
                f"Inlining reorders the evaluation of arguments with effects at "
                f"{file_path!r}:{call_site.lineno}"
            )

        class Substitute(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.expr:
                return bound.get(node.id, node)

        return ast.unparse(Substitute().visit(copy.deepcopy(self.call)))


@dataclass(frozen=True, kw_only=True)
class InlineForwarderOperation(RepositorySourceReprovedOperation):
    """Inline a method whose body only forwards one call, at every call site, and
    delete it (paper1 Thm 4.6: a forwarder supplies a pair outside the lattice).

    Without receiver types the call sites are found by name, so the proof is
    closed-world: no other indexed class declares or assigns the name, every use
    of the name is a direct call, and no string names it. Each call site becomes
    the forwarded call with self and the parameters replaced by the receiver and
    arguments; imports only the method used leave its module."""

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            self.required_reproof(lambda: self.source_edits_from_snapshot(context))
        except CodemodOperationPreflightError as error:
            return (error.report,)
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.PASSED,
                message="Every use of the forwarder is a call that inlines",
                detail=SourceReproofDiagnostic(self.target, ()),
            ),
        )

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        source_path = self.required_source_path(snapshot, self.operation_key())
        qualname = self.target.qualname
        if qualname is None or "." not in qualname:
            raise ValueError("inline_forwarder requires a Class.method target_qualname")
        class_name, method_name = qualname.rsplit(".", 1)
        source_class = ResolvedClassTarget.from_rewrite_target(
            snapshot, SourceRewriteTarget(file_path=source_path, qualname=class_name),
        )
        method = next(
            (
                node for node in source_class.node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
            ),
            None,
        )
        if method is None:
            raise ValueError(f"{qualname!r} is not declared in {source_path!r}")
        forwarder = _ForwarderBody.of(method, qualname)
        answered = AnsweredMethod(snapshot, source_class, method)
        answered.require_sole_dispatch("inlining it")
        _MethodNameStrings.require_none(snapshot, method_name)
        replacements: dict[str, list[tuple[ast.AST, str]]] = {source_path: [(method, "")]}
        for call in answered.answering_calls():
            parents = snapshot.parsed_module_for_source_path(call.file_path).fact(ModuleNodeIndex).parents
            if not forwarder.returns and forwarder.value_used(call.call, parents):
                raise ValueError(f"{call.where} uses the value of {method_name!r}, which returns None")
            replacements.setdefault(call.file_path, []).append(
                (call.call, forwarder.inlined(call.call, call.attribute.value, call.file_path))
            )
        callee = _ForwardedNames(snapshot, source_path, forwarder)
        edits: list[NominalSourceEdit] = []
        for file_path, file_replacements in replacements.items():
            geometry = SourceTextGeometry(snapshot.sources_by_file_path[file_path])
            edits.append(
                geometry.nominal_edit(
                    file_path=file_path,
                    replacements=tuple(
                        SourceTextSpanReplacement.from_offsets(
                            start_offset=start,
                            end_offset=end,
                            replacement_source=replacement,
                        )
                        for node, replacement in file_replacements
                        for start, end in (
                            method_deletion_offsets(geometry, node)
                            if node is method
                            else geometry.required_node_offsets(node),
                        )
                    ),
                    rationale=self.rationale or f"Inline the forwarder {qualname!r}.",
                )
            )
            if any(node is not method for node, _replacement in file_replacements):
                edits.extend(callee.imports_into(file_path))
        edits.extend(imports_only_read_by(snapshot.module_nodes_by_file_path[source_path], source_path, method))
        return tuple(edits)


class _MethodNameStrings:
    """A string spelling the forwarder's name can reach it past every typed call
    only through an attribute lookup by name. The inline deletes the method, so a
    string that may feed such a lookup refuses: one held as data (a table entry,
    a comparison, an assignment) or passed to a lookup itself. A string that is
    only part of a message (an f-string, a concatenation, an argument to a call
    that is no lookup) cannot, unless the scan has a function that looks an
    attribute up by one of its own parameters, which turns a name argument into
    a lookup."""

    OBJECT_LOOKUPS: ClassVar[frozenset[str]] = frozenset({"getattr", "hasattr", "setattr", "delattr"})
    """Lookups that take an object, then the attribute name."""
    NAME_LOOKUPS: ClassVar[frozenset[str]] = frozenset({"attrgetter", "methodcaller"})
    """Lookups that take only attribute names."""
    LOOKUPS: ClassVar[frozenset[str]] = OBJECT_LOOKUPS | NAME_LOOKUPS

    @classmethod
    def name_arguments(cls, call: ast.Call) -> list[ast.expr]:
        """The arguments a lookup reads as attribute names; none for any other call."""

        callee = cls.callee_name(call.func)
        if callee in cls.OBJECT_LOOKUPS:
            return call.args[1:2]
        if callee in cls.NAME_LOOKUPS:
            return call.args[:1] if callee == "methodcaller" else list(call.args)
        return []

    @classmethod
    def require_none(cls, snapshot: CodemodSourceSnapshot, name: str) -> None:
        lookups: dict[str, str] | None = None
        for parsed in snapshot.parsed_modules:
            if name not in parsed.source:
                continue
            index = parsed.fact(ModuleNodeIndex)
            for node in index.string_constants:
                if node.value != name or cls.docstring(node, index.parents):
                    continue
                callee = cls.prose_callee(node, index.parents)
                if callee is None:
                    raise ValueError(f"{parsed.file_path!r}:{node.lineno} names {name!r} in a string")
                if lookups is None:
                    lookups = cls.parameter_lookups(snapshot)
                if callee in lookups:
                    raise ValueError(
                        f"{parsed.file_path!r}:{node.lineno} passes {name!r} to {callee!r}, "
                        f"which looks an attribute up by it at {lookups[callee]}"
                    )

    @staticmethod
    def docstring(node: ast.Constant, parents: Mapping[ast.AST, ast.AST]) -> bool:
        statement = parents.get(node)
        owner = parents.get(statement) if isinstance(statement, ast.Expr) else None
        return (
            isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and bool(owner.body)
            and owner.body[0] is statement
        )

    @classmethod
    def prose_callee(cls, node: ast.AST, parents: Mapping[ast.AST, ast.AST]) -> str | None:
        """Where a string that only reaches text goes: "" for a message part, the
        callee's name for an argument to a call that is no lookup; None for a
        string held as data or handed to a lookup."""

        parent = parents.get(node)
        while isinstance(parent, (ast.FormattedValue, ast.JoinedStr)) or (
            isinstance(parent, ast.BinOp) and isinstance(parent.op, (ast.Add, ast.Mod))
        ):
            if isinstance(parent, ast.JoinedStr):
                return ""
            node, parent = parent, parents.get(parent)
        call = parents.get(parent) if isinstance(parent, ast.keyword) else parent
        if not isinstance(call, ast.Call) or node is call.func:
            return None
        callee = cls.callee_name(call.func)
        if callee is None or callee in cls.LOOKUPS:
            return None
        return callee

    @staticmethod
    def callee_name(func: ast.expr) -> str | None:
        match func:
            case ast.Name(id=name) | ast.Attribute(attr=name):
                return name
        return None

    @classmethod
    def parameter_lookups(cls, snapshot: CodemodSourceSnapshot) -> dict[str, str]:
        """Each function that looks an attribute up by one of its own parameters, by
        name, with where: a name passed to it is a lookup by that name."""

        found: dict[str, str] = {}
        for parsed in snapshot.parsed_modules:
            for function in ast.walk(parsed.module):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                arguments = function.args
                parameters = {
                    argument.arg
                    for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
                }
                for call in ast.walk(function):
                    if isinstance(call, ast.Call) and any(
                        isinstance(argument, ast.Name) and argument.id in parameters
                        for argument in cls.name_arguments(call)
                    ):
                        found.setdefault(function.name, f"{parsed.file_path}:{call.lineno}")
        return found


@dataclass(frozen=True)
class _ForwardedNames:
    """What the forwarded call reads besides the forwarder's parameters, bound
    where each call now lands the way the forwarder binds it: its own local
    imports first, then the source module's imports and declarations. One the
    call site binds to another object refuses."""

    snapshot: CodemodSourceSnapshot
    source_path: str
    forwarder: _ForwarderBody

    @property
    def tables(self) -> tuple[ModuleSymbolTable, ...]:
        source = self.snapshot.sources_by_file_path[self.source_path]
        module = self.snapshot.module_nodes_by_file_path[self.source_path]
        local = ast.Module(body=list(self.forwarder.imports), type_ignores=[])
        return (
            ModuleSymbolTable(file_path=self.source_path, source=source, module=local),
            ModuleSymbolTable(file_path=self.source_path, source=source, module=module),
        )

    def imports_into(self, file_path: str) -> tuple[ModuleImportMutation, ...]:
        graph = self.snapshot.module_import_graph
        source_module = self.snapshot.module_nodes_by_file_path[self.source_path]
        unshadowed = frozenset(dir(builtins)) - LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(source_module.body)
        consumer = ModuleSymbolTable(
            file_path=file_path,
            source=self.snapshot.sources_by_file_path[file_path],
            module=self.snapshot.module_nodes_by_file_path[file_path],
        )
        return tuple(
            ModuleImportMutation.from_source(
                file_path=file_path,
                import_source=import_source,
                rationale=f"The inlined call reads {name!r}.",
            )
            for name in sorted(self.forwarder.free_names - unshadowed)
            for import_source in self._imports_for(graph, consumer, name)
        )

    def _imports_for(self, graph: SourceModuleImportGraph, consumer: ModuleSymbolTable, name: str) -> Iterator[str]:
        """The import `consumer` needs for `name`: one, or none when it already binds it."""

        for table in self.tables:
            resolved = table.import_binding_for_dependency(
                name, import_graph=graph, permits_guarded_import=False,
            )
            if resolved is None:
                continue
            binding, identity = resolved
            if identity.is_destination_declaration(
                graph, destination_path=consumer.file_path, bound_name=name,
            ) or consumer.satisfies_import_binding(name, identity, binding.scope, import_graph=graph):
                return
            if consumer.conflicts_with_import_binding(name, identity, import_graph=graph):
                raise ValueError(f"{consumer.file_path!r} binds {name!r}, which the inlined call reads, to another object")
            yield identity.source_for(
                graph, importing_file_path=consumer.file_path, scope=binding.scope, bound_name=name,
            )
            return
        if name not in self.tables[1].declared_names:
            raise ValueError(f"The forwarded call reads {name!r}, which {self.source_path!r} does not bind")
        if consumer.file_path == self.source_path:
            return
        if name in consumer.top_level_names:
            raise ValueError(f"{consumer.file_path!r} binds {name!r}, which the inlined call reads, to another object")
        yield graph.required_import_source(
            importing_file_path=consumer.file_path, imported_file_path=self.source_path, imported_name=name,
        )


def method_deletion_offsets(
    geometry: SourceTextGeometry, method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[int, int]:
    """A method's whole lines, decorators included, plus one blank line after it if there is one."""

    lines = geometry.source.splitlines(keepends=True)
    first_line = min((decorator.lineno for decorator in method.decorator_list), default=method.lineno)
    start = sum(len(line) for line in lines[: first_line - 1])
    end_line = method.end_lineno or method.lineno
    if end_line < len(lines) and not lines[end_line].strip():
        end_line += 1
    end = sum(len(line) for line in lines[:end_line])
    return start, end


def imports_only_read_by(
    module: ast.Module, source_path: str, node: ast.AST,
) -> Iterator[ModuleImportMutation]:
    """Removals of the module's import bindings that only this node reads: they leave with it."""

    used_by_node = {name.id for name in ast.walk(node) if isinstance(name, ast.Name)}
    used_elsewhere = {
        name.id
        for name in ast.walk(module)
        if isinstance(name, ast.Name) and not _within(name, node)
    }
    for scope, names in ModuleImportScope.bound_in(module, used_by_node - used_elsewhere):
        yield ModuleImportMutation.remove_bound_names(
            file_path=source_path,
            names=names,
            scope=scope,
        )


def _source_position(call: ast.Call, name: str) -> tuple[int, int]:
    return next(
        (node.lineno, node.col_offset)
        for node in ast.walk(call)
        if isinstance(node, ast.Name) and node.id == name
    )


def _plain_read(expression: ast.expr) -> bool:
    """A name, a constant, or attribute and constant-index reads over a name."""

    if isinstance(expression, (ast.Name, ast.Constant)):
        return True
    if isinstance(expression, ast.Attribute):
        return _plain_read(expression.value)
    if isinstance(expression, ast.Subscript):
        return isinstance(expression.slice, ast.Constant) and _plain_read(expression.value)
    return False


def _within(node: ast.AST, container: ast.AST) -> bool:
    return (
        getattr(container, "lineno", 0) <= getattr(node, "lineno", -1) <= (getattr(container, "end_lineno", 0) or 0)
    )
