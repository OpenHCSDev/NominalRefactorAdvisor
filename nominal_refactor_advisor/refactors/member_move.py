"""Move a method onto another class, where it becomes a real method of that class.

A method that only builds or asks another family's object belongs to that family; left where it
is, it makes its own class import the other family, and when that family imports back, a local
import hides the cycle (agent-comms Thread.native_environment builds a ProjectRuntimeRequest).

Two facts decide the move. The method's source shape (a plain method, a staticmethod or a
classmethod) says what its call sites are spelled on before the move: a receiver of its class
family, or the class itself. Its body says which member of the destination it becomes. One that
takes a destination instance as a parameter becomes an instance method of the destination: that
parameter becomes self, and each call is spelled on the destination argument. One that builds a
destination becomes a classmethod constructing through cls. A method that does neither is not
the destination's, and the move refuses.

Call sites are found by the method's name and typed by their receiver: the class binding itself
for a staticmethod or classmethod, the receiver's declared path (parameters, fields, declared
returns) where the declarations type it, the type checker's answer where they do not. A call
whose receiver is the method's class family becomes a call on the destination; a call whose
receiver is another class is that class's own method and stays; a call this proof cannot type
refuses the move."""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, NamedTuple

from nominal_refactor_advisor.ast_projection import name_nodes, names_read
from nominal_refactor_advisor.refactors.answered_method import (
    AnsweredMethod as AnsweredMethod,
    ClassBinding as ClassBinding,
    DeclaredClass as DeclaredClass,
    ForeignClass as ForeignClass,
    ImportedModule as ImportedModule,
    ImportedName as ImportedName,
    MethodCall as MethodCall,
    MethodReference as MethodReference,
    NameBinding as NameBinding,
    NamedClass as NamedClass,
    OtherClass as OtherClass,
    OtherModule as OtherModule,
    OutsideScan as OutsideScan,
    ReceiverBinding as ReceiverBinding,
)
from nominal_refactor_advisor.ast_tools import ModuleAnnotationEvaluationMode
from nominal_refactor_advisor.codemod_import_scopes import ModuleImportScope
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_payload import (
    OptionalStringPayloadValueCodec,
    PayloadRecordValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from nominal_refactor_advisor.lexical_bindings import ImportBoundNameProjection
from nominal_refactor_advisor.refactors.function_operations import (
    imports_only_read_by,
    method_deletion_offsets,
)


@dataclass(frozen=True)
class MovedMethod(AnsweredMethod):
    """The method as it leaves its class for another."""

    destination: ResolvedClassTarget
    receiver_parameter: str | None
    destination_name: str | None

    @classmethod
    def require(
        cls,
        snapshot: CodemodSourceSnapshot,
        source_path: str,
        qualname: str,
        destination: SourceRewriteTarget,
        receiver_parameter: str | None,
        destination_name: str | None,
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
        if method.body[0].lineno == method.lineno:
            raise ValueError(f"{qualname!r} keeps its body on the def line")
        target = ResolvedClassTarget.from_rewrite_target(snapshot, destination)
        moved = cls(snapshot, source_class, method, target, receiver_parameter, destination_name)
        if moved.new_name in target.bound_names:
            raise ValueError(f"{target.qualname!r} already binds {moved.new_name!r}")
        moved.require_sole_dispatch("moving it")
        moved.source, moved.member  # the decorators and the body must both decide the move
        return moved

    @property
    def new_name(self) -> str:
        return self.destination_name or self.method.name

    @cached_property
    def locals(self) -> frozenset[str]:
        return LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.method.body) | frozenset(
            argument.arg for argument in ast.walk(self.method.args) if isinstance(argument, ast.arg)
        )

    def bound_class(
        self, receiver: ast.expr, file_path: str, enclosing: str | None, scopes: tuple[ast.AST, ...],
    ) -> ClassBinding | None:
        return self.source.bound_class(receiver, file_path, enclosing, scopes)

    def admits(self, binding: ReceiverBinding, where: str) -> None:
        self.source.admits(binding, where)

    @cached_property
    def source(self) -> SourceShape:
        """What the method is on its own class, read from its decorators."""

        decorators = tuple(ast.unparse(decorator) for decorator in self.method.decorator_list)
        shape = SourceShape.members.get(decorators)
        if shape is None:
            raise ValueError(
                f"{self.name!r} is decorated {decorators!r}; only a plain, static or class method moves"
            )
        return shape(self)

    @cached_property
    def member(self) -> DestinationMember:
        """The member of the destination this method becomes, read from its body."""

        destination = self.destination.name
        annotated = tuple(
            argument for argument in self.source.call_parameters
            if _annotation_names(argument.annotation, destination)
        )
        if len(annotated) > 1:
            raise ValueError(
                f"{self.name!r} takes several {destination} parameters "
                f"{tuple(argument.arg for argument in annotated)!r}; none is clearly its self"
            )
        if annotated:
            return DestinationInstanceMethod(self, annotated[0])
        if self.destination_reads:
            return DestinationClassMethod(self)
        raise ValueError(
            f"{self.name!r} neither takes a {destination} nor builds one; it is not {destination}'s"
        )

    @cached_property
    def destination_reads(self) -> tuple[ast.Name, ...]:
        return self.body_reads(self.destination.name)

    def body_reads(self, name: str) -> tuple[ast.Name, ...]:
        return tuple(
            node for statement in self.method.body for node in name_nodes(statement)
            if node.id == name and isinstance(node.ctx, ast.Load)
        )

    def destination_source(self) -> str:
        """The method as its destination member: its own header and body text, with only what
        the member changes rewritten, re-indented for the destination class."""

        source = self.snapshot.sources_by_file_path[self.source_class.file_path]
        geometry = SourceTextGeometry(source)
        definition = geometry.required_node_offsets(self.method)[0]
        body_start = geometry.line_offsets[self.method.body[0].lineno - 1]
        end = method_deletion_offsets(geometry, self.method)[1]
        signature = SignatureText(geometry, self.method)
        header = _edited(source, definition, body_start, (
            *self.member.signature_edits(signature),
            *signature.renamed(self.new_name),
        ))
        body_renames = tuple(
            (*geometry.required_node_offsets(node), replacement, node)
            for node, replacement in self.member.body_renames()
        )
        body = _edited(source, body_start, end, tuple(edit[:3] for edit in body_renames))
        shifts = self.continuation_shifts(body_renames)
        source_indent = " " * self.method.col_offset
        destination_indent = " " * (self.destination.node.body[0].col_offset)
        dropped = self.destination_self_imports
        kept: list[str] = []
        previous_dropped = False
        for number, line in enumerate(body.splitlines(keepends=True), start=self.method.body[0].lineno):
            if number in dropped or (previous_dropped and not line.strip()):
                previous_dropped = number in dropped
                continue
            previous_dropped = False
            kept.append(_shifted(line, shifts.get(number, 0)))
        lines = [
            destination_indent + line.removeprefix(source_indent) if line.strip() else line
            for line in (*header.splitlines(keepends=True), *kept)
        ]
        return f"\n{self.member.decorator(destination_indent)}{''.join(lines).rstrip()}\n"

    def continuation_shifts(self, renames: tuple[tuple[int, int, str, ast.AST], ...]) -> dict[int, int]:
        """Lines aligned under a call whose head a rename changed move by the same columns: a
        call whose first argument shares its opening line aligns its continuation to it."""

        shifts: dict[int, int] = {}
        calls = tuple(node for node in ast.walk(self.method) if isinstance(node, ast.Call))
        for start, end, replacement, node in renames:
            delta = len(replacement) - (end - start)
            if not delta:
                continue
            for call in calls:
                passed: tuple[ast.expr | ast.keyword, ...] = (*call.args, *call.keywords)
                if not (
                    passed and call.end_lineno and call.end_lineno > call.lineno
                    and min(each.lineno for each in passed) == call.lineno
                    and any(each is node for each in ast.walk(call.func))
                ):
                    continue
                for line in range(call.lineno + 1, call.end_lineno + 1):
                    shifts[line] = shifts.get(line, 0) + delta
        return shifts

    @cached_property
    def destination_self_imports(self) -> frozenset[int]:
        """Lines of local imports that only bind the destination itself: in its own module the
        destination is already bound, and the local import was the hidden cycle."""

        lines: set[int] = set()
        for node in ast.walk(self.method):
            if isinstance(node, ast.ImportFrom) and set(
                ImportBoundNameProjection(node).names()
            ) == {self.destination.name}:
                if node.lineno != (node.end_lineno or node.lineno):
                    raise ValueError(f"{self.name!r}'s import of the destination spans lines")
                lines.add(node.lineno)
        return frozenset(lines)

    @cached_property
    def runtime_reads(self) -> frozenset[str]:
        """Names the method reads when it runs or is defined: everything but its parameter and
        return annotations, which a module postponing annotations never evaluates."""

        annotations = tuple(
            annotation
            for annotation in (
                *(argument.annotation for argument in ast.walk(self.method.args) if isinstance(argument, ast.arg)),
                self.method.returns,
            )
            if annotation is not None
        )
        annotated = frozenset(id(node) for annotation in annotations for node in ast.walk(annotation))
        return frozenset(
            node.id for node in name_nodes(self.method)
            if isinstance(node.ctx, ast.Load) and id(node) not in annotated
        )

    def free_names(self) -> frozenset[str]:
        """Module-level names of the source the method reads, which its new module must bind."""

        reads = names_read(self.method)
        return frozenset(
            name for name in (reads - self.locals - {self.destination.name}) | self.source.free_names()
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
        postponed = not ModuleAnnotationEvaluationMode.from_module(
            snapshot.module_nodes_by_file_path[destination_path]
        ).annotations_execute_at_declaration
        imports: list[ModuleImportMutation] = []
        for name in sorted(self.free_names()):
            # A TYPE_CHECKING binding answers only a read nothing evaluates when the code runs.
            needed = (
                ModuleImportScope.TYPE_CHECKING
                if postponed and name not in self.runtime_reads else ModuleImportScope.RUNTIME
            )
            source = _binding(snapshot, self.source_class.file_path, name)
            if name in destination_names:
                destination = _binding(snapshot, destination_path, name)
                if destination.declaration != source.declaration:
                    raise ValueError(
                        f"{self.name!r} reads {name!r}, which {destination_path} binds to something else"
                    )
                if destination.scope.canonical_rank > needed.canonical_rank:
                    raise ValueError(
                        f"{self.name!r} reads {name!r} when it runs, which {destination_path} "
                        f"binds only under TYPE_CHECKING"
                    )
                continue
            if source.scope.canonical_rank > needed.canonical_rank:
                raise ValueError(
                    f"{self.name!r} reads {name!r} when it runs, which {self.source_class.file_path} "
                    f"binds only under TYPE_CHECKING; moving it would promote that import"
                )
            statement = next(
                (
                    statement for statement in source.scope.import_statements(source_module)
                    if isinstance(statement, ast.ImportFrom)
                    and name in ImportBoundNameProjection(statement).names()
                ),
                None,
            )
            if statement is not None and not statement.level:
                alias = next(
                    alias for alias in statement.names
                    if ImportBoundNameProjection(statement).alias_bound_name(alias) == name
                )
                spelled = alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
                import_source = f"from {statement.module} import {spelled}\n"
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
                scope=source.scope,
                rationale=f"{self.name!r} reads {name!r} in its new class.",
            ))
        return tuple(imports)


def _annotation_names(annotation: ast.expr | None, name: str) -> bool:
    """Whether a parameter's annotation is exactly the named class, bare or quoted."""

    return annotation is not None and ast.unparse(annotation) in {name, repr(name)}


def _calls(node: ast.AST) -> bool:
    return any(isinstance(each, (ast.Call, ast.Await)) for each in ast.walk(node))


@dataclass(frozen=True)
class SourceShape:
    """The method on its own class: what leads its parameters and what its calls are spelled on.
    Members register under the decorators that declare them."""

    moved: MovedMethod

    members: ClassVar[dict[tuple[str, ...], type[SourceShape]]] = {}
    decorators: ClassVar[tuple[str, ...] | None] = None
    leading_name: ClassVar[str | None] = None
    describes: ClassVar[str] = "a plain method"
    keeps_receiver: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.decorators is not None:
            SourceShape.members[cls.decorators] = cls

    def __post_init__(self) -> None:
        positional = self.moved.method.args.posonlyargs + self.moved.method.args.args
        if self.leading_name is not None and (not positional or positional[0].arg != self.leading_name):
            raise ValueError(f"{self.moved.name!r} takes no {self.leading_name} parameter")
        if self.keeps_receiver != (self.moved.receiver_parameter is not None):
            raise ValueError(
                f"{self.moved.name!r} is {self.describes}; receiver_parameter is "
                + ("required" if self.keeps_receiver else "not taken")
            )

    @property
    def positional_parameters(self) -> tuple[ast.arg, ...]:
        """The positional parameters a call passes: every one after the leading self or cls."""

        positional = self.moved.method.args.posonlyargs + self.moved.method.args.args
        return tuple(positional[0 if self.leading_name is None else 1:])

    @property
    def call_parameters(self) -> tuple[ast.arg, ...]:
        return self.positional_parameters + tuple(self.moved.method.args.kwonlyargs)

    def body_renames(self) -> tuple[tuple[ast.AST, str], ...]:
        return ()

    def free_names(self) -> frozenset[str]:
        return frozenset()

    def bound_class(
        self, receiver: ast.expr, file_path: str, enclosing: str | None, scopes: tuple[ast.AST, ...],
    ) -> ClassBinding | None:
        """The class a call receiver names by its binding, when it names one: a family class,
        or a class, module or import from outside the scan, which no family class is, whatever
        its attribute is called."""

        return AnsweredMethod.bound_class(self.moved, receiver, file_path, enclosing, scopes)

    def admits(self, binding: ReceiverBinding, where: str) -> None:
        """Refuse a call whose receiver would change what the moved method means."""


@dataclass(frozen=True)
class InstanceSource(SourceShape):
    """A plain method: calls are spelled on instances of the family, and the receiver stays as
    the destination member's receiver parameter."""

    decorators: ClassVar[tuple[str, ...] | None] = ()
    leading_name: ClassVar[str | None] = "self"
    keeps_receiver: ClassVar[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.moved.receiver_parameter in self.moved.locals:
            raise ValueError(f"{self.moved.name!r} already binds {self.moved.receiver_parameter!r}")

    def body_renames(self) -> tuple[tuple[ast.AST, str], ...]:
        return tuple((node, str(self.moved.receiver_parameter)) for node in self.moved.body_reads("self"))


@dataclass(frozen=True)
class ClassBoundSource(SourceShape):
    """A staticmethod or classmethod: calls are spelled on the class itself, by its binding, or
    inside it on cls or self; the destination member takes no receiver from them."""

    def bound_class(
        self, receiver: ast.expr, file_path: str, enclosing: str | None, scopes: tuple[ast.AST, ...],
    ) -> ClassBinding | None:
        if isinstance(receiver, ast.Name) and receiver.id in {"cls", "self"}:
            return None if enclosing not in self.moved.family else ImplicitClass(enclosing)
        return super().bound_class(receiver, file_path, enclosing, scopes)


@dataclass(frozen=True)
class StaticSource(ClassBoundSource):
    """A staticmethod: nothing leads its parameters."""

    decorators: ClassVar[tuple[str, ...] | None] = ("staticmethod",)
    describes: ClassVar[str] = "a staticmethod"


@dataclass(frozen=True)
class ClassSource(ClassBoundSource):
    """A classmethod: on the destination its cls reads become the source class, so a call that
    could bind cls to a descendant refuses (`admits`)."""

    decorators: ClassVar[tuple[str, ...] | None] = ("classmethod",)
    leading_name: ClassVar[str | None] = "cls"
    describes: ClassVar[str] = "a classmethod"

    def admits(self, binding: ReceiverBinding, where: str) -> None:
        """cls becomes the source class on the destination, which changes the method only where
        its body reads cls and a call may bind cls to a descendant: one naming a descendant, or
        through cls, self or an instance, whose class is known only at run time."""

        if self.moved.body_reads("cls") and self.moved.descendants and not binding.names(self.moved.symbol):
            raise ValueError(
                f"{where} calls {self.moved.name!r} {binding.describes}, which may bind cls to a "
                f"descendant of {self.moved.source_class.name}; move that caller first or call the class"
            )

    def body_renames(self) -> tuple[tuple[ast.AST, str], ...]:
        return tuple((node, self.moved.source_class.name) for node in self.moved.body_reads("cls"))

    def free_names(self) -> frozenset[str]:
        return frozenset({self.moved.source_class.name}) if self.moved.body_reads("cls") else frozenset()


@dataclass(frozen=True)
class ImplicitClass(ClassBinding):
    """cls or self inside a family class: the runtime class is that class or any descendant."""

    describes: ClassVar[str] = "through cls or self"


@dataclass(frozen=True)
class PassedArgument:
    """The destination as a call passes it positionally: the receiver takes its place as is."""

    node: ast.AST
    value: ast.expr

    def spelled(self, parameter: str, receiver: str) -> str:
        return receiver


@dataclass(frozen=True)
class KeywordArgument(PassedArgument):
    """The destination passed by keyword: the receiver takes its place under its own name."""

    def spelled(self, parameter: str, receiver: str) -> str:
        return f"{parameter}={receiver}"


@dataclass(frozen=True)
class DestinationMember:
    """What the moved method becomes on its destination, and how each call is spelled there."""

    moved: MovedMethod

    def signature_edits(self, signature: SignatureText) -> tuple[_Edit, ...]:
        """What the member changes in the method's own parameter list."""

        raise NotImplementedError

    def body_renames(self) -> tuple[tuple[ast.AST, str], ...]:
        """Each body node of the method and the text it becomes on the destination."""

        return self.moved.source.body_renames()

    def decorator(self, indent: str) -> str:
        return ""

    @property
    def callers_import_destination(self) -> bool:
        """Whether a rewritten call spells the destination class, so its module must bind it."""

        return False

    def replacement(
        self, geometry: SourceTextGeometry, call: MethodCall,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        raise NotImplementedError

    def renamed_attribute(self, geometry: SourceTextGeometry, call: MethodCall) -> SourceTextSpanReplacement:
        end = geometry.required_node_offsets(call.attribute)[1]
        return SourceTextSpanReplacement.from_offsets(
            start_offset=end - len(call.attribute.attr), end_offset=end,
            replacement_source=self.moved.new_name,
        )

    def dropped_receiver(self, call: MethodCall) -> None:
        if not self.moved.source.keeps_receiver and _calls(call.attribute.value):
            raise ValueError(f"{call.where}: the receiver calls, and the moved call no longer evaluates it")


@dataclass(frozen=True)
class DestinationInstanceMethod(DestinationMember):
    """The method takes a destination: that parameter becomes self. A plain method's receiver
    takes its place, so `receiver.m(..., destination, ...)` becomes `destination.m(..., receiver,
    ...)`; a staticmethod or classmethod has none, so `Source.m(..., destination, ...)` becomes
    `destination.m(...)`."""

    parameter: ast.arg

    def __post_init__(self) -> None:
        method = self.moved.method
        if self.parameter in method.args.kwonlyargs:
            defaults = dict(zip(method.args.kwonlyargs, method.args.kw_defaults))
            has_default = defaults[self.parameter] is not None
        else:
            positional = method.args.posonlyargs + method.args.args
            has_default = positional.index(self.parameter) >= len(positional) - len(method.args.defaults)
        if has_default:
            raise ValueError(f"{self.moved.name!r}'s {self.parameter.arg!r} has a default; calls may omit it")

    def signature_edits(self, signature: SignatureText) -> tuple[_Edit, ...]:
        receiver = self.moved.receiver_parameter
        if receiver is not None:
            return (signature.replaced(self.parameter, receiver),)
        leading = signature.leading(self.moved.source.leading_name)
        return (
            signature.removed(self.parameter),
            signature.replaced(leading, "self") if leading is not None
            else signature.prepended("self", without=self.parameter),
        )

    def body_renames(self) -> tuple[tuple[ast.AST, str], ...]:
        return super().body_renames() + tuple(
            (node, "self") for node in self.moved.body_reads(self.parameter.arg)
        )

    def argument(self, call: MethodCall) -> PassedArgument:
        if any(isinstance(each, ast.Starred) for each in call.call.args) or any(
            keyword.arg is None for keyword in call.call.keywords
        ):
            raise ValueError(f"{call.where} unpacks arguments into {self.moved.name!r}")
        keyword = next(
            (each for each in call.call.keywords if each.arg == self.parameter.arg), None,
        )
        if keyword is not None:
            return KeywordArgument(keyword, keyword.value)
        positional = self.moved.source.positional_parameters
        if self.parameter in positional and positional.index(self.parameter) < len(call.call.args):
            passed = call.call.args[positional.index(self.parameter)]
            return PassedArgument(passed, passed)
        raise ValueError(f"{call.where} passes no {self.parameter.arg!r} to {self.moved.name!r}")

    def replacement(
        self, geometry: SourceTextGeometry, call: MethodCall,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        self.dropped_receiver(call)
        receiver = call.attribute.value
        argument = self.argument(call)
        value = argument.value
        passed: list[ast.AST] = sorted(
            (*call.call.args, *call.call.keywords),
            key=lambda each: geometry.required_node_offsets(each)[0],
        )
        index = passed.index(argument.node)
        earlier = ([receiver] if self.moved.source.keeps_receiver else []) + passed[:index]
        if _calls(value) and any(_calls(each) for each in earlier):
            raise ValueError(
                f"{call.where}: {self.parameter.arg!r} and an earlier argument both call, and "
                "moving it ahead would reorder their effects"
            )
        value_text = geometry.source[slice(*geometry.required_node_offsets(value))]
        atom = isinstance(value, (ast.Name, ast.Attribute, ast.Subscript, ast.Call, ast.Constant))
        receiver_start, receiver_end = geometry.required_node_offsets(receiver)
        argument_start, argument_end = geometry.required_node_offsets(argument.node)
        replacements = [
            SourceTextSpanReplacement.from_offsets(
                start_offset=receiver_start,
                end_offset=receiver_end,
                replacement_source=value_text if atom else f"({value_text})",
            ),
            self.renamed_attribute(geometry, call),
        ]
        if self.moved.source.keeps_receiver:
            replacements.append(SourceTextSpanReplacement.from_offsets(
                start_offset=argument_start,
                end_offset=argument_end,
                replacement_source=argument.spelled(
                    str(self.moved.receiver_parameter), geometry.source[receiver_start:receiver_end],
                ),
            ))
        elif index + 1 < len(passed):
            replacements.append(SourceTextSpanReplacement.from_offsets(
                start_offset=argument_start,
                end_offset=geometry.required_node_offsets(passed[index + 1])[0],
                replacement_source="",
            ))
        elif index:
            replacements.append(SourceTextSpanReplacement.from_offsets(
                start_offset=geometry.required_node_offsets(passed[index - 1])[1],
                end_offset=argument_end,
                replacement_source="",
            ))
        else:
            # The only argument: the parentheses close on it, with no line left for it.
            call_end = geometry.required_node_offsets(call.call)[1]
            attribute_end = geometry.required_node_offsets(call.attribute)[1]
            opening = attribute_end + geometry.source[attribute_end:call_end].index("(") + 1
            replacements.append(SourceTextSpanReplacement.from_offsets(
                start_offset=opening, end_offset=call_end - 1, replacement_source="",
            ))
        return tuple(replacements)


@dataclass(frozen=True)
class DestinationClassMethod(DestinationMember):
    """The method builds a destination: it becomes a classmethod constructing through cls. A
    plain method's receiver becomes its first parameter, so `receiver.m(...)` becomes
    `Destination.m(receiver, ...)`; a staticmethod or classmethod call is respelled on the
    destination with its arguments as they are."""

    def __post_init__(self) -> None:
        if "cls" in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.moved.method.body):
            raise ValueError(f"{self.moved.name!r} already binds 'cls'")

    def signature_edits(self, signature: SignatureText) -> tuple[_Edit, ...]:
        receiver = self.moved.receiver_parameter
        leading = signature.leading(self.moved.source.leading_name)
        if receiver is not None and leading is not None:
            return (signature.replaced(leading, f"cls, {receiver}"),)
        return () if leading is not None else (signature.prepended("cls"),)

    def body_renames(self) -> tuple[tuple[ast.AST, str], ...]:
        return super().body_renames() + tuple((node, "cls") for node in self.moved.destination_reads)

    def decorator(self, indent: str) -> str:
        return f"{indent}@classmethod\n"

    @property
    def callers_import_destination(self) -> bool:
        return True

    def replacement(
        self, geometry: SourceTextGeometry, call: MethodCall,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        self.dropped_receiver(call)
        receiver_start, receiver_end = geometry.required_node_offsets(call.attribute.value)
        replacements = [
            SourceTextSpanReplacement.from_offsets(
                start_offset=receiver_start, end_offset=receiver_end,
                replacement_source=self.moved.destination.name,
            ),
            self.renamed_attribute(geometry, call),
        ]
        if not self.moved.source.keeps_receiver:
            return tuple(replacements)
        receiver = geometry.source[receiver_start:receiver_end]
        end = geometry.required_node_offsets(call.call)[1]
        attribute_end = geometry.required_node_offsets(call.attribute)[1]
        opening = attribute_end + geometry.source[attribute_end:end].index("(") + 1
        inner = geometry.source[opening:end - 1]
        lead = inner[: len(inner) - len(inner.lstrip())]
        if "\n" in lead:
            inserted, span_end = f"{lead}{receiver},", opening
        elif inner.strip():
            inserted, span_end = f"{receiver}, ", opening + len(lead)
        else:
            inserted, span_end = receiver, opening + len(lead)
        replacements.append(SourceTextSpanReplacement.from_offsets(
            start_offset=opening, end_offset=span_end, replacement_source=inserted,
        ))
        return tuple(replacements)


type _Edit = tuple[int, int, str]


def _edited(source: str, start: int, end: int, edits: tuple[_Edit, ...]) -> str:
    """source[start:end] with each edit's offsets replaced, the latest first."""

    text = source[start:end]
    for edit_start, edit_end, replacement in sorted(edits, reverse=True):
        text = text[: edit_start - start] + replacement + text[edit_end - start:]
    return text


def _shifted(line: str, shift: int) -> str:
    if shift >= 0:
        return " " * shift + line if line.strip() else line
    return line[min(-shift, len(line) - len(line.lstrip(" "))):]


@dataclass(frozen=True)
class SignatureText:
    """A method's header as written: edits replace, remove or add one parameter at a time and
    leave every other character, line break and default as the source has it."""

    geometry: SourceTextGeometry
    method: ast.FunctionDef | ast.AsyncFunctionDef

    @property
    def source(self) -> str:
        return self.geometry.source

    @cached_property
    def name_span(self) -> tuple[int, int]:
        start = self.source.index(self.method.name, self.geometry.required_node_offsets(self.method)[0])
        return start, start + len(self.method.name)

    @cached_property
    def opening(self) -> int:
        return self.source.index("(", self.name_span[1]) + 1

    @cached_property
    def parameters(self) -> tuple[ast.arg, ...]:
        arguments = self.method.args
        return tuple(
            argument for argument in (
                *arguments.posonlyargs, *arguments.args, arguments.vararg,
                *arguments.kwonlyargs, arguments.kwarg,
            ) if argument is not None
        )

    def leading(self, name: str | None) -> ast.arg | None:
        """The self or cls the source shape leads with, when it leads with one."""

        return None if name is None else self.parameters[0]

    def renamed(self, name: str) -> tuple[_Edit, ...]:
        return () if name == self.method.name else ((*self.name_span, name),)

    def replaced(self, parameter: ast.arg, text: str) -> _Edit:
        return (*self.geometry.required_node_offsets(parameter), text)

    def removed(self, parameter: ast.arg) -> _Edit:
        """A parameter and its separator: its whole line when it stands alone on one."""

        start, end = self.geometry.required_node_offsets(parameter)
        line_start = self.source.rfind("\n", 0, start) + 1
        rest = re.match(r"[ \t]*,?[ \t]*\n", self.source[end:])
        if not self.source[line_start:start].strip() and rest is not None:
            return line_start, end + rest.end(), ""
        following = re.match(r"\s*,\s*", self.source[end:])
        if following is not None and self.source[end + following.end()] != ")":
            return start, end + following.end(), ""
        comma = self.source.rfind(",", self.opening, start)
        return (comma if comma >= 0 else start), end, ""

    def prepended(self, text: str, without: ast.arg | None = None) -> _Edit:
        """A new first parameter, on its own line when the first one stands on its own."""

        remaining = tuple(parameter for parameter in self.parameters if parameter is not without)
        if not remaining:
            return self.opening, self.opening, text
        first = remaining[0]
        first_start = self.geometry.required_node_offsets(first)[0]
        if "\n" in self.source[self.opening:first_start]:
            return first_start, first_start, f"{text},\n{' ' * first.col_offset}"
        return first_start, first_start, f"{text}, "


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


class ModuleBinding(NamedTuple):
    """What a module-level name resolves to: the declaration it names (the module and name it
    imports or declares, None when neither) and the scope that binds it. Two modules binding one
    object compare equal by declaration, whatever statement or scope binds it in each."""

    declaration: tuple[str, str] | None
    scope: ModuleImportScope


def _binding(snapshot: CodemodSourceSnapshot, file_path: str, name: str) -> ModuleBinding:
    source_file = snapshot.module_import_graph.source_file_for_path(file_path)
    if source_file is None:
        raise ValueError(f"{file_path} is outside the scan")
    identity = source_file.module_path_identity
    module = snapshot.module_nodes_by_file_path[file_path]

    def imported(statement: ast.ImportFrom) -> tuple[str, str] | None:
        for alias in statement.names:
            if ImportBoundNameProjection(statement).alias_bound_name(alias) == name:
                module_name = identity.resolve_import_from_module(
                    imported_module=statement.module, level=statement.level,
                )
                return str(module_name), alias.name
        return None

    for statement in module.body:
        if isinstance(statement, ast.ImportFrom) and (declaration := imported(statement)):
            return ModuleBinding(declaration, ModuleImportScope.RUNTIME)
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == name:
            return ModuleBinding(
                (str(snapshot.module_import_graph.module_name_for_file_path(file_path)), name),
                ModuleImportScope.RUNTIME,
            )
    for statement in ModuleImportScope.TYPE_CHECKING.import_statements(module):
        if isinstance(statement, ast.ImportFrom) and (declaration := imported(statement)):
            return ModuleBinding(declaration, ModuleImportScope.TYPE_CHECKING)
    return ModuleBinding(None, ModuleImportScope.RUNTIME)


@dataclass(frozen=True, kw_only=True)
class MoveMethodToClassOperation(RepositorySourceReprovedOperation):
    """Move `Class.method` (a plain method, staticmethod or classmethod) onto `destination` as
    the member its body makes it, an instance method when it takes a destination and a
    classmethod when it builds one, named `destination_name` when given, and rewrite every call
    on the method's class family into a call on the destination. Calls of the same name on
    another class's instances stay; a call whose receiver neither the class binding, the
    declarations nor the type checker types refuses the move, as do a descendant overriding the
    method, a reference that is not a call, a super() call, and a string naming the method.
    `receiver_parameter` names the old receiver's parameter on the destination; only a plain
    method has a receiver to keep."""

    destination: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    receiver_parameter: str | None = codemod_payload_field(OptionalStringPayloadValueCodec(), default=None)
    destination_name: str | None = codemod_payload_field(OptionalStringPayloadValueCodec(), default=None)

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        source_path = self.required_source_path(snapshot, self.operation_key())
        if self.target.qualname is None:
            raise ValueError("move_method_to_class requires a Class.method target_qualname")
        moved = MovedMethod.require(
            snapshot, source_path, self.target.qualname, self.destination,
            self.receiver_parameter, self.destination_name,
        )
        rewritten = moved.answering_calls()
        replacements: dict[str, list[SourceTextSpanReplacement]] = {}
        imports: list[ModuleImportMutation] = list(moved.destination_imports())
        graph = snapshot.module_import_graph
        for call in rewritten:
            geometry = SourceTextGeometry(snapshot.sources_by_file_path[call.file_path])
            replacements.setdefault(call.file_path, []).extend(
                moved.member.replacement(geometry, call)
            )
            if not moved.member.callers_import_destination:
                continue
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
