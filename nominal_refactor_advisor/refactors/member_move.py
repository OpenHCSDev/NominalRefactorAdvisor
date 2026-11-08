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
import copy
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

from ..codemod_imports import ModuleImportMutation
from ..codemod_payload import (
    OptionalStringPayloadValueCodec,
    PayloadRecordValueCodec,
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
        overriding = tuple(
            snapshot.required_class_family_index.classes_by_symbol[symbol].qualname
            for symbol in moved.descendants
            if symbol in snapshot.required_class_family_index.classes_by_symbol
            and method_name in _class_node(snapshot, symbol).bound_names_of_body
        )
        if overriding:
            raise ValueError(f"{overriding!r} override {method_name!r}; moving it breaks their dispatch")
        moved.source, moved.member  # the decorators and the body must both decide the move
        return moved

    @property
    def name(self) -> str:
        return self.method.name

    @property
    def new_name(self) -> str:
        return self.destination_name or self.method.name

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

    @cached_property
    def locals(self) -> frozenset[str]:
        return LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.method.body) | frozenset(
            argument.arg for argument in ast.walk(self.method.args) if isinstance(argument, ast.arg)
        )

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
            node for statement in self.method.body for node in ast.walk(statement)
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
        )

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
        """The method as its destination member: a header built from its new parameters, and
        its body with the member's renames applied."""

        source = self.snapshot.sources_by_file_path[self.source_class.file_path]
        geometry = SourceTextGeometry(source)
        body_start = geometry.line_offsets[self.method.body[0].lineno - 1]
        end = method_deletion_offsets(geometry, self.method)[1]
        renames = sorted(
            (
                (*geometry.required_node_offsets(node), replacement)
                for node, replacement in self.member.body_renames()
            ),
            reverse=True,
        )
        text = source[body_start:end]
        for node_start, node_end, replacement in renames:
            text = text[: node_start - body_start] + replacement + text[node_end - body_start:]
        source_indent = " " * self.method.col_offset
        destination_indent = " " * (self.destination.node.body[0].col_offset)
        lines = [
            line.removeprefix(source_indent) if line.strip() else line
            for line in text.splitlines(keepends=True)
        ]
        dropped = self.destination_self_imports
        kept: list[str] = []
        previous_dropped = False
        for number, line in enumerate(lines, start=self.method.body[0].lineno):
            if number in dropped or (previous_dropped and not line.strip()):
                previous_dropped = number in dropped
                continue
            previous_dropped = False
            kept.append(line)
        body = "".join(destination_indent + line if line.strip() else line for line in kept)
        return (
            f"\n{self.member.decorator(destination_indent)}"
            f"{destination_indent}{self.header()}\n{body.rstrip()}\n"
        )

    def header(self) -> str:
        definition = type(self.method)(
            name=self.new_name,
            args=self.member.arguments(),
            body=[ast.Pass()],
            decorator_list=[],
            returns=copy.deepcopy(self.method.returns),
            type_params=list(getattr(self.method, "type_params", [])),
        )
        return ast.unparse(ast.fix_missing_locations(definition)).splitlines()[0]

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

        reads = frozenset(
            node.id for node in ast.walk(self.method)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )
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

    def class_receiver(self, receiver: ast.expr, file_path: str, enclosing: str | None) -> bool:
        """Whether a call receiver is the source class itself."""

        return False


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

    def class_receiver(self, receiver: ast.expr, file_path: str, enclosing: str | None) -> bool:
        if not isinstance(receiver, ast.Name):
            return False
        if receiver.id in {"cls", "self"}:
            return enclosing == self.moved.symbol
        snapshot = self.moved.snapshot
        name = self.moved.source_class.name
        if file_path == self.moved.source_class.file_path:
            return receiver.id == name
        return any(
            isinstance(statement, ast.ImportFrom)
            and any(alias.name == name and (alias.asname or alias.name) == receiver.id for alias in statement.names)
            and _imported_file(snapshot, file_path, statement) == self.moved.source_class.file_path
            for statement in ast.walk(snapshot.module_nodes_by_file_path[file_path])
        )


@dataclass(frozen=True)
class StaticSource(ClassBoundSource):
    """A staticmethod: nothing leads its parameters."""

    decorators: ClassVar[tuple[str, ...] | None] = ("staticmethod",)
    describes: ClassVar[str] = "a staticmethod"


@dataclass(frozen=True)
class ClassSource(ClassBoundSource):
    """A classmethod: its cls is the source class, which a descendant would change, so the move
    requires a class without descendants; on the destination cls reads become that class."""

    decorators: ClassVar[tuple[str, ...] | None] = ("classmethod",)
    leading_name: ClassVar[str | None] = "cls"
    describes: ClassVar[str] = "a classmethod"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.moved.descendants:
            raise ValueError(
                f"{self.moved.name!r} is a classmethod of a class with descendants; "
                "cls would stop naming them"
            )

    def body_renames(self) -> tuple[tuple[ast.AST, str], ...]:
        return tuple((node, self.moved.source_class.name) for node in self.moved.body_reads("cls"))

    def free_names(self) -> frozenset[str]:
        return frozenset({self.moved.source_class.name}) if self.moved.body_reads("cls") else frozenset()


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

    def arguments(self) -> ast.arguments:
        """The member's parameters, as its header declares them."""

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

    def parameters(self, leading: list[ast.arg], kept: tuple[ast.arg, ...]) -> ast.arguments:
        """The method's arguments with its leading parameters replaced and only `kept` left."""

        original = self.moved.method.args
        everything = original.posonlyargs + original.args
        defaults = dict(zip(everything[len(everything) - len(original.defaults):], original.defaults))
        positional = [argument for argument in everything if argument in kept]
        keywords = [argument for argument in original.kwonlyargs if argument in kept]
        keyword_defaults = dict(zip(original.kwonlyargs, original.kw_defaults))
        return ast.arguments(
            posonlyargs=[],
            args=leading + [copy.deepcopy(argument) for argument in positional],
            vararg=copy.deepcopy(original.vararg),
            kwonlyargs=[copy.deepcopy(argument) for argument in keywords],
            kw_defaults=[copy.deepcopy(keyword_defaults[argument]) for argument in keywords],
            kwarg=copy.deepcopy(original.kwarg),
            defaults=[copy.deepcopy(defaults[argument]) for argument in positional if argument in defaults],
        )


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

    def arguments(self) -> ast.arguments:
        source = self.moved.source
        receiver = self.moved.receiver_parameter
        if receiver is None:
            return self.parameters(
                [ast.arg("self")],
                tuple(argument for argument in source.call_parameters if argument is not self.parameter),
            )
        arguments = self.parameters([ast.arg("self")], source.call_parameters)
        for argument in arguments.args + arguments.kwonlyargs:
            if argument.arg == self.parameter.arg:
                argument.arg, argument.annotation = receiver, None
        return arguments

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
        else:
            replacements.append(SourceTextSpanReplacement.from_offsets(
                start_offset=geometry.required_node_offsets(passed[index - 1])[1] if index else argument_start,
                end_offset=argument_end,
                replacement_source="",
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

    def arguments(self) -> ast.arguments:
        leading = [ast.arg("cls")]
        if self.moved.receiver_parameter is not None:
            leading.append(ast.arg(self.moved.receiver_parameter))
        return self.parameters(leading, self.moved.source.call_parameters)

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


def _enclosing_class(
    snapshot: CodemodSourceSnapshot, file_path: str, node: ast.AST, parents: dict[ast.AST, ast.AST],
) -> str | None:
    """The symbol of the class whose body a node sits in, when the index knows it."""

    names: list[str] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.ClassDef):
            names.append(current.name)
        current = parents.get(current)
    qualname = ".".join(reversed(names))
    return next(
        (
            symbol for symbol, declaration in snapshot.required_class_family_index.classes_by_symbol.items()
            if names and declaration.file_path == file_path and declaration.qualname == qualname
        ),
        None,
    )


@dataclass(frozen=True)
class MethodCall:
    """One call by the method's name, and the receiver it is called on."""

    file_path: str
    call: ast.Call
    attribute: ast.Attribute

    @property
    def where(self) -> str:
        return f"{self.file_path}:{self.call.lineno}"


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
        rewritten = self.answering_calls(snapshot, moved)
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

    def answering_calls(
        self, snapshot: CodemodSourceSnapshot, moved: MovedMethod,
    ) -> tuple[MethodCall, ...]:
        """The calls the moved method answers, each receiver typed by the class binding, the
        declarations or the checker."""

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
                enclosing = _enclosing_class(snapshot, parsed.file_path, node, parents)
                if moved.source.class_receiver(node.value, parsed.file_path, enclosing):
                    answering.append(site)
                    continue
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
