"""Move a method onto another class: the old receiver becomes the new method's first parameter.

A method that only builds and asks another family's object belongs to that family; left where it
is, it makes its own class import the other family, and when that family imports back, a local
import hides the cycle (agent-comms Thread.native_environment builds a ProjectRuntimeRequest).

Call sites are found by the method's name and typed by their receiver: the receiver's declared
path (parameters, fields, declared returns) where the declarations type it, the type checker's
answer where they do not. A call whose receiver is the method's class or a descendant becomes a
call on the destination; a call whose receiver is another class is that class's own method and
stays; a call this proof cannot type refuses the move."""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from functools import cached_property

from ..codemod_imports import ModuleImportMutation
from ..codemod_payload import (
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from ..codemod_reproof import RepositorySourceReprovedOperation
from ..codemod_runtime import CodemodSourceSnapshot
from ..codemod_selection_context import ResolvedClassTarget
from ..codemod_selector_models import SourceRewriteTarget
from ..codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from ..lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from ..probe_owner import InferredReceiverType, ProbeOwnerResolver, ReceiverSpan
from ..type_checker_session import SourceTexts, TypeCheckerSession
from .function_operations import imports_only_read_by, method_deletion_offsets


@dataclass(frozen=True)
class MovedMethod:
    """The method as it leaves its class, and the classes whose instances answer it."""

    snapshot: CodemodSourceSnapshot
    source_class: ResolvedClassTarget
    method: ast.FunctionDef | ast.AsyncFunctionDef
    destination: ResolvedClassTarget
    receiver_parameter: str

    @classmethod
    def require(
        cls,
        snapshot: CodemodSourceSnapshot,
        source_path: str,
        qualname: str,
        destination: SourceRewriteTarget,
        receiver_parameter: str,
    ) -> MovedMethod:
        class_name, _, method_name = qualname.rpartition(".")
        if not class_name:
            raise ValueError("move_method_to_class requires a Class.method target_qualname")
        source_class = ResolvedClassTarget.from_rewrite_target(
            snapshot, SourceRewriteTarget(file_path=source_path, qualname=class_name),
        )
        method = next(
            (
                statement for statement in source_class.node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == method_name
            ),
            None,
        )
        if method is None:
            raise ValueError(f"{qualname!r} is not a method declared in {source_path!r}")
        if method.decorator_list:
            raise ValueError(f"{qualname!r} is decorated; only a plain method moves")
        positional = method.args.posonlyargs + method.args.args
        if not positional or positional[0].arg != "self":
            raise ValueError(f"{qualname!r} takes no self receiver")
        bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(method.body) | frozenset(
            argument.arg for argument in ast.walk(method.args) if isinstance(argument, ast.arg)
        )
        if receiver_parameter in bound:
            raise ValueError(f"{qualname!r} already binds {receiver_parameter!r}")
        target = ResolvedClassTarget.from_rewrite_target(snapshot, destination)
        if method_name in target.bound_names:
            raise ValueError(f"{target.qualname!r} already binds {method_name!r}")
        moved = cls(snapshot, source_class, method, target, receiver_parameter)
        overriding = tuple(
            snapshot.required_class_family_index.classes_by_symbol[symbol].qualname
            for symbol in moved.descendants
            if symbol in snapshot.required_class_family_index.classes_by_symbol
            and method_name in _class_node(snapshot, symbol).bound_names_of_body
        )
        if overriding:
            raise ValueError(f"{overriding!r} override {method_name!r}; moving it breaks their dispatch")
        return moved

    @property
    def name(self) -> str:
        return self.method.name

    @cached_property
    def symbol(self) -> str:
        return self.source_class.required_symbol(self.snapshot)

    @cached_property
    def descendants(self) -> frozenset[str]:
        return frozenset(self.snapshot.required_class_family_index.descendant_symbols(self.symbol))

    @cached_property
    def family(self) -> frozenset[str]:
        """Every class whose instances answer the method by inheriting it."""

        return self.descendants | {self.symbol}

    def answers(self, symbols: tuple[str, ...], where: str) -> bool:
        """Whether a receiver of these classes calls the moved method: all of the family, or
        none of it (another class's own method of that name)."""

        inside = tuple(symbol for symbol in symbols if symbol in self.family)
        if symbols and len(inside) == len(symbols):
            return True
        if symbols and not inside:
            return False
        raise ValueError(f"{where} calls {self.name!r} on a receiver typed {symbols!r}")

    def destination_source(self) -> str:
        """The method as a static method of the destination: self reads become the receiver."""

        source = self.snapshot.sources_by_file_path[self.source_class.file_path]
        geometry = SourceTextGeometry(source)
        start, end = method_deletion_offsets(geometry, self.method)
        method_start = geometry.required_node_offsets(self.method)[0]
        renames = sorted(
            (
                geometry.required_node_offsets(node)
                for node in ast.walk(self.method)
                if (isinstance(node, ast.Name) and node.id == "self")
                or (isinstance(node, ast.arg) and node.arg == "self")
            ),
            reverse=True,
        )
        text = source[method_start:end]
        for node_start, node_end in renames:
            text = (
                text[: node_start - method_start] + self.receiver_parameter
                + text[node_end - method_start:]
            )
        source_indent = source[start:method_start]
        destination_indent = " " * (self.destination.node.body[0].col_offset)
        lines = [
            line.removeprefix(source_indent) if line.strip() else line
            for line in (source_indent + text).splitlines(keepends=True)
        ]
        dropped = self.destination_self_imports
        kept = [
            line for number, line in enumerate(lines, start=self.method.lineno)
            if number not in dropped
        ]
        body = "".join(
            destination_indent + line if line.strip() else line for line in kept
        )
        return f"\n{destination_indent}@staticmethod\n{body.rstrip()}\n"

    @cached_property
    def destination_self_imports(self) -> frozenset[int]:
        """Lines of local imports that only bind the destination itself: in its own module the
        destination is already bound, and the local import was the hidden cycle."""

        lines: set[int] = set()
        for node in ast.walk(self.method):
            if isinstance(node, ast.ImportFrom) and all(
                (alias.asname or alias.name) == self.destination.name for alias in node.names
            ):
                if node.lineno != (node.end_lineno or node.lineno):
                    raise ValueError(f"{self.name!r}'s import of the destination spans lines")
                lines.add(node.lineno)
        return frozenset(lines)

    def free_names(self) -> frozenset[str]:
        """Module-level names of the source the method reads, which its new module must bind."""

        local = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.method.body) | frozenset(
            argument.arg for argument in ast.walk(self.method.args) if isinstance(argument, ast.arg)
        )
        reads = frozenset(
            node.id for node in ast.walk(self.method)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )
        return frozenset(
            name for name in reads - local - {"self", self.destination.name}
            if not hasattr(builtins, name)
        )

    def destination_imports(self) -> tuple[ModuleImportMutation, ...]:
        """Each free name, imported into the destination's module the way the source binds it."""

        snapshot = self.snapshot
        source_module = snapshot.module_nodes_by_file_path[self.source_class.file_path]
        destination_path = self.destination.file_path
        destination_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            snapshot.module_nodes_by_file_path[destination_path].body
        )
        graph = snapshot.module_import_graph
        imports: list[ModuleImportMutation] = []
        for name in sorted(self.free_names()):
            statement = next(
                (
                    statement for statement in source_module.body
                    if isinstance(statement, ast.ImportFrom)
                    and any((alias.asname or alias.name) == name for alias in statement.names)
                ),
                None,
            )
            if statement is not None and not statement.level:
                alias = next(alias for alias in statement.names if (alias.asname or alias.name) == name)
                spelled = alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
                import_source = f"from {statement.module} import {spelled}\n"
            elif name in destination_names:
                raise ValueError(
                    f"{self.name!r} reads {name!r}, which {destination_path} binds to something else"
                )
            else:
                import_source = graph.required_import_source(
                    importing_file_path=destination_path,
                    imported_file_path=self.source_class.file_path if statement is None
                    else _imported_file(snapshot, self.source_class.file_path, statement),
                    imported_name=name,
                )
            imports.append(ModuleImportMutation.from_source(
                file_path=destination_path,
                import_source=import_source,
                rationale=f"{self.name!r} reads {name!r} in its new class.",
            ))
        return tuple(imports)


@dataclass(frozen=True)
class _ClassBody:
    bound_names_of_body: frozenset[str]


def _class_node(snapshot: CodemodSourceSnapshot, symbol: str) -> _ClassBody:
    declaration = snapshot.required_class_family_index.classes_by_symbol[symbol]
    scope: ast.AST = snapshot.module_nodes_by_file_path[declaration.file_path]
    for part in declaration.qualname.split("."):
        scope = next(
            statement for statement in scope.body  # type: ignore[attr-defined]
            if isinstance(statement, ast.ClassDef) and statement.name == part
        )
    assert isinstance(scope, ast.ClassDef)
    return _ClassBody(LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(scope.body))


def _imported_file(snapshot: CodemodSourceSnapshot, file_path: str, statement: ast.ImportFrom) -> str:
    graph = snapshot.module_import_graph
    source_file = graph.source_file_for_path(file_path)
    module = None if source_file is None else source_file.module_path_identity.resolve_import_from_module(
        imported_module=statement.module, level=statement.level,
    )
    provider = None if module is None else graph.unique_source_file_for_module_name(module)
    if provider is None:
        raise ValueError(f"{file_path}:{statement.lineno} imports from a module outside the scan")
    return provider.file_path


@dataclass(frozen=True)
class MethodCall:
    """One call by the method's name, and the receiver it is called on."""

    file_path: str
    call: ast.Call
    attribute: ast.Attribute

    @property
    def where(self) -> str:
        return f"{self.file_path}:{self.call.lineno}"

    def replacement(self, geometry: SourceTextGeometry, destination: str) -> SourceTextSpanReplacement:
        start, end = geometry.required_node_offsets(self.call)
        receiver = geometry.source[slice(*geometry.required_node_offsets(self.attribute.value))]
        arguments = geometry.source[geometry.required_node_offsets(self.attribute)[1]:end]
        inner = arguments.strip()[1:-1]
        joined = f"{receiver}, {inner.lstrip()}" if inner.strip() else receiver
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start,
            end_offset=end,
            replacement_source=f"{destination}.{self.attribute.attr}({joined})",
        )


@dataclass(frozen=True, kw_only=True)
class MoveMethodToClassOperation(RepositorySourceReprovedOperation):
    """Move `Class.method` onto `destination` as a static method whose first parameter is the
    old receiver, and rewrite every call on an instance of the method's class family into a call
    on the destination. Calls of the same name on another class's instances stay; a call whose
    receiver neither the declarations nor the type checker types refuses the move, as do a
    descendant overriding the method, a reference that is not a call, a super() call, and a
    string naming the method."""

    destination: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    receiver_parameter: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        source_path = self.required_source_path(snapshot, self.operation_key())
        if self.target.qualname is None:
            raise ValueError("move_method_to_class requires a Class.method target_qualname")
        moved = MovedMethod.require(
            snapshot, source_path, self.target.qualname, self.destination, self.receiver_parameter,
        )
        rewritten = self.answering_calls(snapshot, moved)
        replacements: dict[str, list[SourceTextSpanReplacement]] = {}
        imports: list[ModuleImportMutation] = list(moved.destination_imports())
        graph = snapshot.module_import_graph
        for call in rewritten:
            geometry = SourceTextGeometry(snapshot.sources_by_file_path[call.file_path])
            replacements.setdefault(call.file_path, []).append(
                call.replacement(geometry, moved.destination.name)
            )
            module_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                snapshot.module_nodes_by_file_path[call.file_path].body
            )
            if call.file_path != moved.destination.file_path and moved.destination.name not in module_names:
                imports.append(ModuleImportMutation.from_source(
                    file_path=call.file_path,
                    import_source=graph.required_import_source(
                        importing_file_path=call.file_path,
                        imported_file_path=moved.destination.file_path,
                        imported_name=moved.destination.name,
                    ),
                    rationale=f"{call.where} calls {moved.name!r} on its new class.",
                ))
        source_geometry = SourceTextGeometry(snapshot.sources_by_file_path[source_path])
        start, end = method_deletion_offsets(source_geometry, moved.method)
        replacements.setdefault(source_path, []).append(
            SourceTextSpanReplacement.from_offsets(start_offset=start, end_offset=end, replacement_source="")
        )
        destination_end = moved.destination.node.end_lineno or moved.destination.node.lineno
        destination_geometry = SourceTextGeometry(
            snapshot.sources_by_file_path[moved.destination.file_path]
        )
        lines = destination_geometry.source.splitlines(keepends=True)
        offset = sum(len(line) for line in lines[:destination_end])
        replacements.setdefault(moved.destination.file_path, []).append(
            SourceTextSpanReplacement.from_offsets(
                start_offset=offset, end_offset=offset, replacement_source=moved.destination_source(),
            )
        )
        edits: list[NominalSourceEdit] = []
        for file_path, file_replacements in replacements.items():
            geometry = SourceTextGeometry(snapshot.sources_by_file_path[file_path])
            edits.extend(geometry.physical_edits(
                file_path=file_path,
                replacements=sorted(file_replacements, key=lambda each: each.start_offset),
                rationale=self.rationale or f"Move {moved.name!r} onto {moved.destination.qualname!r}.",
            ))
        edits.extend(imports)
        edits.extend(imports_only_read_by(
            snapshot.module_nodes_by_file_path[source_path], source_path, moved.method,
        ))
        return tuple(edits)

    def answering_calls(
        self, snapshot: CodemodSourceSnapshot, moved: MovedMethod,
    ) -> tuple[MethodCall, ...]:
        """The calls the moved method answers, each receiver typed by declarations or the checker."""

        typing = snapshot.declared_receiver_typing
        answering: list[MethodCall] = []
        undeclared: dict[ReceiverSpan, MethodCall] = {}
        for parsed in snapshot.parsed_modules:
            if moved.name not in parsed.source:
                continue
            parents = {
                child: parent for parent in ast.walk(parsed.module)
                for child in ast.iter_child_nodes(parent)
            }
            resolver = ProbeOwnerResolver.of(parsed)
            for node in ast.walk(parsed.module):
                where = f"{parsed.file_path}:{getattr(node, 'lineno', 0)}"
                if isinstance(node, ast.Constant) and node.value == moved.name:
                    raise ValueError(f"{where} names {moved.name!r} in a string")
                if node is moved.method or not (
                    isinstance(node, ast.Attribute) and node.attr == moved.name
                ):
                    continue
                call = parents.get(node)
                if not (isinstance(call, ast.Call) and call.func is node):
                    raise ValueError(f"{where} uses {moved.name!r} other than by calling it")
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and (
                    node.value.func.id == "super"
                ):
                    raise ValueError(f"{where} calls {moved.name!r} through super()")
                site = MethodCall(parsed.file_path, call, node)
                owner = resolver.owner(node.value).resolved(typing)
                spans = owner.type_checker_spans()
                if spans:
                    undeclared[spans[0]] = site
                elif moved.answers(tuple(getattr(owner, "class_symbols", tuple)()), where):
                    answering.append(site)
        if undeclared:
            with TypeCheckerSession.session(SourceTexts(snapshot.parsed_modules)) as session:
                answers = session.receiver_types(undeclared)
            for span, site in undeclared.items():
                answer = answers.get(span)
                if not isinstance(answer, InferredReceiverType):
                    raise ValueError(f"{site.where}: the type checker gives no type for the receiver")
                if moved.answers(answer.classes, site.where):
                    answering.append(site)
        nested = [
            site for site in answering
            for other in answering
            if other is not site and other.call in ast.walk(site.call)
        ]
        if nested:
            raise ValueError(f"{nested[0].where} nests one moved call inside another")
        return tuple(answering)
