from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


from .codemod_call_source import (
    DeclaredCallArgumentsRewrite,
    DeclaredCallExpressionRewrite,
    DeclaredCallRewriteABC,
    DeclaredCallTargetRewrite,
)
from .codemod_declaration_operations import (
    DeclarationDecoratorsPayload as DeclarationDecoratorsPayload,
    DeclarationMutationOperationABC as DeclarationMutationOperationABC,
    ReplaceDeclarationDecoratorsOperation as ReplaceDeclarationDecoratorsOperation,
)
from .codemod_declaration_source import (
    FunctionBindingProjectionSourceAuthority,
    FunctionAliasSourceAuthority,
    FunctionBodyPrefixSourceAuthority,
    FunctionBodySourceAuthority,
    FunctionDecoratorsSourceAuthority,
    FunctionRegionSourceAuthority,
    FunctionSignatureSourceAuthority,
)
from .codemod_foreign_adapter_source import AdapterBindChainAuthority
from .codemod_foreign_family_source import (
    ForeignFamilyFilterAuthority,
    ForeignFamilyReference,
)
from .codemod_static_projection_source import (
    StaticProjection,
    StaticProjectionAuthority,
)
from .codemod_payload import (
    BooleanPayloadValueCodec,
    EmptyDefaultStringPayloadValueCodec,
    OptionalStringArrayPayloadValueCodec,
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_reproof import (
    RepositorySourceReprovedOperation,
    SourceReprovedOperation,
)
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selector_models import (
    DeclaredCallSelection,
    SourceRewriteTarget,
)
from .codemod_imports import ModuleImportMutation
from .codemod_preflight import CodemodOperationPreflightError, CodemodOperationPreflightReport
from .codemod_semantics import CodemodPreflightStatus
from .codemod_selection_context import CodemodSelectorContext
from .codemod_reproof import SourceReproofDiagnostic
from .codemod_source_edits import SourceTextGeometry, SourceTextSpanReplacement, NominalSourceEdit
from .codemod_import_scopes import ModuleImportScope
from collections.abc import Iterator
from .codemod_source_edits import (
    NominalSourceEdit,
    PhysicalSourceEdit,
    SourceTextGeometry,
)
from .declaration_dependencies import (
    FunctionBindingABC,
    FunctionLocalBinding,
    FunctionParameterBinding,
)
from .declared_root import DeclaredRoot
from .descriptor_algebra import AliasProperty
from .source_index import AstTargetNodeKind
from .product_flow import BareCallTargetReference
from .value_expression import LexicalValueReference


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
                import_source=reference.import_source_for(snapshot, parsed_module),
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
                import_source=reference.import_source_for(snapshot, parsed_module),
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
                import_source=reference.import_source_for(snapshot, parsed_module),
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
    """A method whose whole body is one call built from its own parameters."""

    method: ast.FunctionDef
    call: ast.Call
    returns: bool

    @classmethod
    def of(cls, method: ast.FunctionDef, qualname: str) -> _ForwarderBody:
        body = [
            statement
            for statement in method.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        arguments = method.args
        if (
            method.decorator_list
            or arguments.vararg
            or arguments.kwarg
            or arguments.posonlyargs
            or not arguments.args
            or len(body) != 1
        ):
            raise ValueError(f"{qualname!r} is not a plain one-call forwarder")
        statement = body[0]
        call = statement.value if isinstance(statement, (ast.Return, ast.Expr)) else None
        if not isinstance(call, ast.Call):
            raise ValueError(f"{qualname!r} is not a plain one-call forwarder")
        return cls(method, call, isinstance(statement, ast.Return))

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
        uses = {name: 0 for name in self.parameter_names}
        for node in ast.walk(self.call):
            if isinstance(node, ast.Name) and node.id in uses:
                uses[node.id] += 1
        repeated = tuple(name for name, count in uses.items() if count > 1)
        if repeated:
            raise ValueError(
                f"Inlining would evaluate {repeated!r} more than once at "
                f"{file_path!r}:{call_site.lineno}"
            )

        class Substitute(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.expr:
                return bound.get(node.id, node)

        import copy

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
        module = snapshot.module_nodes_by_file_path[source_path]
        owner = next(
            (node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name),
            None,
        )
        method = None if owner is None else next(
            (node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == method_name),
            None,
        )
        if method is None:
            raise ValueError(f"{qualname!r} is not declared in {source_path!r}")
        forwarder = _ForwarderBody.of(method, qualname)
        edits: list[NominalSourceEdit] = []
        for source_file in snapshot.source_index.files:
            file_path = source_file.file_path
            consumer = snapshot.module_nodes_by_file_path.get(file_path)
            if consumer is None:
                continue
            replacements = self._call_site_replacements(
                consumer, file_path, method_name, forwarder, method if file_path == source_path else None
            )
            if file_path == source_path:
                replacements.append((method, ""))
            if not replacements:
                continue
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
                        for node, replacement in replacements
                        for start, end in (
                            self._deletion_offsets(geometry, node)
                            if node is method
                            else geometry.required_node_offsets(node),
                        )
                    ),
                    rationale=self.rationale or f"Inline the forwarder {qualname!r}.",
                )
            )
        edits.extend(self._unused_import_removals(module, source_path, method))
        return tuple(edits)

    @staticmethod
    def _call_site_replacements(
        module: ast.Module,
        file_path: str,
        method_name: str,
        forwarder: _ForwarderBody,
        declaration: ast.FunctionDef | None,
    ) -> list[tuple[ast.AST, str]]:
        parents = {child: parent for parent in ast.walk(module) for child in ast.iter_child_nodes(parent)}
        replacements: list[tuple[ast.AST, str]] = []
        for node in ast.walk(module):
            if declaration is not None and node is declaration:
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == method_name:
                raise ValueError(f"{file_path!r}:{node.lineno} declares another {method_name!r}")
            if isinstance(node, ast.Constant) and node.value == method_name:
                raise ValueError(f"{file_path!r}:{node.lineno} names {method_name!r} in a string")
            if not (isinstance(node, ast.Attribute) and node.attr == method_name):
                continue
            call = parents.get(node)
            if not (isinstance(call, ast.Call) and call.func is node) or not isinstance(node.ctx, ast.Load):
                raise ValueError(f"{file_path!r}:{node.lineno} uses {method_name!r} other than by calling it")
            if not forwarder.returns and not isinstance(parents.get(call), ast.Expr):
                raise ValueError(
                    f"{file_path!r}:{call.lineno} uses the value of {method_name!r}, which returns None"
                )
            replacements.append((call, forwarder.inlined(call, node.value, file_path)))
        return replacements

    @staticmethod
    def _deletion_offsets(geometry: SourceTextGeometry, method: ast.FunctionDef) -> tuple[int, int]:
        """The method's whole lines, plus one blank line after it if there is one."""

        lines = geometry.source.splitlines(keepends=True)
        start = sum(len(line) for line in lines[: method.lineno - 1])
        end_line = method.end_lineno or method.lineno
        if end_line < len(lines) and not lines[end_line].strip():
            end_line += 1
        end = sum(len(line) for line in lines[:end_line])
        return start, end

    @staticmethod
    def _unused_import_removals(
        module: ast.Module,
        source_path: str,
        method: ast.FunctionDef,
    ) -> Iterator[ModuleImportMutation]:
        used_by_method = {node.id for node in ast.walk(method) if isinstance(node, ast.Name)}
        used_elsewhere = {
            node.id
            for node in ast.walk(module)
            if isinstance(node, ast.Name) and not _within(node, method)
        }
        for scope in ModuleImportScope:
            for statement in scope.import_statements(module):
                names = tuple(
                    alias.asname or alias.name
                    for alias in statement.names
                    if (alias.asname or alias.name) in used_by_method - used_elsewhere
                )
                if names:
                    yield ModuleImportMutation.remove_bound_names(
                        file_path=source_path,
                        names=names,
                        scope=scope,
                    )


def _within(node: ast.AST, container: ast.AST) -> bool:
    return (
        getattr(container, "lineno", 0) <= getattr(node, "lineno", -1) <= (getattr(container, "end_lineno", 0) or 0)
    )
