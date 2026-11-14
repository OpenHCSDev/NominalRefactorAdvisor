"""What a class's own syntax declares: its bases, members, keyed tables and projection surfaces, read once per class."""

from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import (
    dataclass,
    field,
)
from heapq import merge
from typing import (
    ClassVar,
    Iterable,
    NamedTuple,
    Self,
    cast,
)

from nominal_refactor_advisor.annotation_semantics import (
    CLASSVAR_ANNOTATION_AUTHORITY,
    NOMINAL_ANNOTATION_SOURCE_AUTHORITY,
    TypeWrapper,
)
from nominal_refactor_advisor.ast_projection import (
    CallKeywords,
    ExpressionReference,
    StatementValue,
    StatementValues,
    single_name_targets,
)
from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    ParsedModuleFact,
    PythonSourcePathPolicy,
)
from nominal_refactor_advisor.class_symbols import (
    ClassSymbolResolutionAuthority,
    _referenced_bases,
    _terminal_reference_name,
    ModuleClassDefinitions,
)
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.foreign_family import AssignmentStatement
from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from nominal_refactor_advisor.models import SourceLineReference
from nominal_refactor_advisor.value_expression import CompactValueExpression


class CompactClassMemberDeclaration(NamedTuple):
    """One direct class binding from which all member projections descend."""

    name: str
    line: int
    expression: str | None
    value: CompactValueExpression
    constructor_name: str | None
    constructor_keyword_names: tuple[str, ...]
    annotation_expression: str | None

    @property
    def annotation_reference_parts(self) -> tuple[str, ...] | None:
        """Project the one nominal type named by this member annotation."""

        if self.annotation_expression is None:
            return None
        try:
            annotation = ast.parse(self.annotation_expression, mode="eval").body
        except SyntaxError:
            return None
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                annotation = ast.parse(annotation.value, mode="eval").body
            except SyntaxError:
                return None
        if isinstance(annotation, ast.Subscript) and (
            CLASSVAR_ANNOTATION_AUTHORITY.matches(annotation)
        ):
            annotation = annotation.slice
        return NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_parts_or_none(annotation)


@dataclass(frozen=True)
class CompactClassSyntaxFacets:
    """Class-family views derived together from the shared module traversal."""

    closed_axis_branch_functions: tuple[CompactClosedAxisBranchFunction, ...] = ()
    exact_type_guards: tuple[CompactExactTypeGuard, ...] = ()
    autoregister_reference_index: CompactAutoRegisterReferenceIndex | None = None


@dataclass(frozen=True)
class CompactNamedProjectionSurface(SourceLineReference):
    """Top-level tuple/list/dict references used by registry projections."""

    surface_name: str
    sequence_references: tuple[tuple[str, str | None], ...] = ()
    dict_key_references: tuple[tuple[str, str | None], ...] = ()
    dict_value_references: tuple[tuple[str, str | None], ...] = ()


@dataclass(frozen=True)
class CompactSourceLocation(SourceLineReference):
    """Source position shared by compact structural observations."""



@dataclass(frozen=True)
class CompactAutoRegisterReferenceIndex:
    """Interned receiver-attribute edges for registry consumer resolution."""

    function_qualnames: tuple[str, ...]
    receiver_names: tuple[str, ...]
    attribute_names: tuple[str, ...]
    encoded_edges: str


@dataclass(frozen=True)
class CompactSortedKeyCall(CompactSourceLocation):
    """One sorted call and the semantic attributes used by its key."""

    registry_owner_names: tuple[str, ...]
    key_attribute_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactKeyedTableAxis(CompactSourceLocation):
    """AST-free module-level dictionary keyed by one enum-like axis."""

    table_name: str
    key_type_name: str
    case_names: tuple[str, ...]
    value_shape_name: str | None


@dataclass(frozen=True)
class CompactClosedAxisBranchFact:
    key_type_name: str
    branch_site_count: int
    case_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactReceiverFieldBranchRead:
    """``receiver.field_name.attribute`` reads in one method's branch tests."""

    field_name: str
    branch_site_count: int
    attribute_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactQualifiedSourceLocation(CompactSourceLocation):
    """Qualified source position shared by compact structural observations."""

    qualname: str


@dataclass(frozen=True)
class CompactClosedAxisBranchFunction(CompactQualifiedSourceLocation):
    axes: tuple[CompactClosedAxisBranchFact, ...]
    receiver_field_reads: tuple[CompactReceiverFieldBranchRead, ...]


@dataclass(frozen=True)
class CompactExactTypeGuard(CompactQualifiedSourceLocation):
    subject_expression: str
    type_reference_expression: str
    type_reference_parts: tuple[str, ...]
    matches_exact_type_when_true: bool
    expression: str

    @property
    def structural_membership_expression(self) -> str:
        membership = (
            f"isinstance({self.subject_expression}, {self.type_reference_expression})"
        )
        return membership if self.matches_exact_type_when_true else f"not {membership}"


def _compact_base_argument_sources(node: ast.ClassDef) -> tuple[tuple[str, ...], ...]:
    """The subscript arguments of each referenced base, aligned with its base references."""

    sources: list[tuple[str, ...]] = []
    for base, _parts in _referenced_bases(node):
        match base:
            case ast.Subscript(slice=ast.Tuple(elts=arguments)):
                sources.append(tuple(ast.unparse(argument) for argument in arguments))
            case ast.Subscript(slice=argument):
                sources.append((ast.unparse(argument),))
            case _:
                sources.append(())
    return tuple(sources)


def _compact_base_key_type_names(node: ast.ClassDef) -> tuple[str | None, ...]:
    """The key type each referenced base's subscript names first (`Family[Key]`), aligned with its base references."""

    return tuple(
        next(iter(annotation_type_names(base.slice)), None)
        if isinstance(base, ast.Subscript)
        else None
        for base, _parts in _referenced_bases(node)
    )


def _class_nominal_reference_root_names(
    parsed_module: ParsedModule,
) -> frozenset[str]:
    """Return module bindings queried while projecting class declarations."""

    return frozenset(
        name
        for _qualname, class_node in parsed_module.fact(ModuleClassDefinitions).definitions
        for name in _class_reference_root_names(class_node)
    )


def _class_reference_root_names(class_node: ast.ClassDef) -> frozenset[str]:
    """The module bindings one class declaration reads while it is projected."""

    reference_nodes: list[ast.AST] = []
    for class_node in (class_node,):
        reference_nodes.extend(class_node.bases)
        reference_nodes.extend(class_node.decorator_list)
        reference_nodes.extend(keyword.value for keyword in class_node.keywords)
        for statement in class_node.body:
            if isinstance(statement, ast.AnnAssign):
                reference_nodes.append(statement.annotation)
                reference_nodes.extend(StatementValue.of(statement).walk())
            elif isinstance(statement, ast.Assign):
                reference_nodes.append(statement.value)
    return frozenset(
        node.id
        for reference_node in reference_nodes
        for node in ast.walk(reference_node)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )


def _trim_leading_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _compact_named_projection_surfaces(
    parsed_module: ParsedModule,
) -> tuple[CompactNamedProjectionSurface, ...]:
    named_values: dict[str, tuple[int, ast.AST]] = {}
    named_sequences: dict[str, tuple[int, ast.Tuple | ast.List]] = {}
    for (statement, target), assigned in StatementValues.keyed(
        ((statement, target), statement)
        for statement in _trim_leading_docstring(list(parsed_module.module.body))
        if isinstance(statement, ast.Assign | ast.AnnAssign)
        for target in single_name_targets(statement)
    ).supplied:
        value = assigned.value
        named_values[target.id] = (statement.lineno, value)
        if isinstance(value, (ast.Tuple, ast.List)):
            named_sequences[target.id] = (statement.lineno, value)

    surfaces: list[CompactNamedProjectionSurface] = []
    for surface_name, (line, value) in named_sequences.items():
        surfaces.append(
            CompactNamedProjectionSurface(
                file_path=parsed_module.file_path,
                surface_name=surface_name,
                line=line,
                sequence_references=tuple(
                    reference
                    for element in value.elts
                    if (reference := _compact_projection_reference(element)) is not None
                ),
            )
        )
    for surface_name, (line, value) in named_values.items():
        if isinstance(value, ast.Dict):
            surfaces.append(
                CompactNamedProjectionSurface(
                    file_path=parsed_module.file_path,
                    surface_name=surface_name,
                    line=line,
                    dict_key_references=tuple(
                        reference
                        for key in value.keys
                        if key is not None
                        if (reference := _compact_projection_reference(key)) is not None
                    ),
                    dict_value_references=tuple(
                        reference
                        for item in value.values
                        if (reference := _compact_projection_reference(item))
                        is not None
                    ),
                )
            )
    return tuple(surfaces)


def _compact_projection_reference(
    node: ast.AST,
) -> tuple[str, str | None] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, None
    if isinstance(node, ast.Name):
        return node.id, node.id
    if isinstance(node, ast.Attribute):
        parts = ExpressionReference.decoded(node).attribute_chain
        if parts is None:
            return ast.unparse(node), None
        return ".".join(parts), parts[0]
    return None


def annotation_type_names(node: ast.AST | None) -> tuple[str, ...]:
    if node is None:
        return ()
    if CompactValueExpression.project(cast(ast.expr, node)).value_is_none_literal:
        return ()
    if isinstance(node, ast.Name):
        return () if node.id == "None" else (node.id,)
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    if isinstance(node, ast.Tuple):
        return sorted_tuple(
            {name for element in node.elts for name in annotation_type_names(element)}
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return sorted_tuple(
            {*annotation_type_names(node.left), *annotation_type_names(node.right)}
        )
    if isinstance(node, ast.Subscript):
        base_name = _terminal_reference_name(node.value)
        if TypeWrapper.wraps(base_name):
            return annotation_type_names(node.slice)
        if base_name == "Annotated":
            if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                return annotation_type_names(node.slice.elts[0])
            return annotation_type_names(node.slice)
    return ()


def _compact_keyed_table_axes(
    parsed_module: ParsedModule,
) -> tuple[CompactKeyedTableAxis, ...]:
    axes: list[CompactKeyedTableAxis] = []
    for statement in parsed_module.module.body:
        table_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                table_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            table_name = statement.target.id
            value = statement.value
        if table_name is None or not isinstance(value, ast.Dict):
            continue
        if len(value.keys) < 2 or any(key is None for key in value.keys):
            continue
        case_names = tuple(ast.unparse(key) for key in value.keys if key is not None)
        key_type_names = {
            case_name.split(".", 1)[0] for case_name in case_names if "." in case_name
        }
        if len(key_type_names) != 1:
            continue
        value_shape_name: str | None = None
        value_constructor_names = {
            ast.unparse(item.func)
            for item in value.values
            if isinstance(item, ast.Call)
        }
        if (
            all(isinstance(item, ast.Call) for item in value.values)
            and len(value_constructor_names) == 1
        ):
            value_shape_name = next(iter(value_constructor_names))
        axes.append(
            CompactKeyedTableAxis(
                file_path=parsed_module.file_path,
                line=statement.lineno,
                table_name=table_name,
                key_type_name=next(iter(key_type_names)),
                case_names=sorted_tuple(case_names),
                value_shape_name=value_shape_name,
            )
        )
    return tuple(axes)


def _exact_type_predicate(
    node: ast.AST,
) -> tuple[ast.AST, ast.AST, bool, str] | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        negated = True
        comparison = node.operand
    else:
        negated = False
        comparison = node
    if not (
        isinstance(comparison, ast.Compare)
        and len(comparison.ops) == 1
        and len(comparison.comparators) == 1
    ):
        return None
    operator = comparison.ops[0]
    if not isinstance(operator, (ast.Is, ast.Eq, ast.IsNot, ast.NotEq)):
        return None
    left = comparison.left
    right = comparison.comparators[0]
    subject = _type_call_subject(left)
    type_reference = right
    if subject is None:
        subject = _type_call_subject(right)
        type_reference = left
    if subject is None:
        return None
    matches_exact_type = isinstance(operator, (ast.Is, ast.Eq))
    return (
        subject,
        type_reference,
        not matches_exact_type if negated else matches_exact_type,
        ast.unparse(node),
    )


def _type_call_subject(node: ast.AST) -> ast.AST | None:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "type"
        and len(node.args) == 1
        and not node.keywords
    ):
        return None
    return node.args[0]


def _fail_loud_block(statements: list[ast.stmt]) -> bool:
    non_terminating_prefix_types = (
        ast.AnnAssign,
        ast.Assign,
        ast.AugAssign,
        ast.Delete,
        ast.Expr,
        ast.Import,
        ast.ImportFrom,
        ast.Pass,
    )
    return (
        bool(statements)
        and isinstance(statements[-1], ast.Raise)
        and all(
            isinstance(statement, non_terminating_prefix_types)
            for statement in statements[:-1]
        )
    )


@dataclass
class _BranchReferenceTally(ABC):
    """Per-function branch-site counts for one projection of attribute chains."""

    fact_type: ClassVar[type]
    site_counts: dict[int, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    names: dict[int, dict[str, set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )

    @abstractmethod
    def reference(
        self, parts: tuple[str, ...], receiver: str | None
    ) -> tuple[str, str] | None:
        """The key and name one attribute chain contributes, if any."""

    def add(
        self,
        function_id: int,
        branch: Iterable[ast.AST],
        receiver: str | None,
    ) -> None:
        """Count one branch site whose tests and patterns are ``branch``."""

        names_by_key: dict[str, set[str]] = defaultdict(set)
        for node in branch:
            for subnode in self.referenced_nodes(node):
                parts = ExpressionReference.decoded(subnode).attribute_chain
                reference = None if parts is None else self.reference(parts, receiver)
                if reference is not None:
                    names_by_key[reference[0]].add(reference[1])
        for key, names in names_by_key.items():
            self.site_counts[function_id][key] += 1
            self.names[function_id][key].update(names)

    def referenced_nodes(self, branch_node: ast.AST) -> Iterator[ast.AST]:
        """Every node of one test or pattern whose chain may name a reference."""

        return ast.walk(branch_node)

    def facts(self, function_id: int) -> tuple:
        return tuple(
            self.fact_type(key, site_count, sorted_tuple(self.names[function_id][key]))
            for key, site_count in sorted(self.site_counts[function_id].items())
        )


class _EnumMemberBranchTally(_BranchReferenceTally):
    """``Key.CASE`` references, keyed by the chain's penultimate name.

    A case is recovered only where a branch compares a value against it or
    matches it as a pattern. A member used as a call receiver or argument is a
    constant query about something else, not a decision between cases.
    """

    fact_type = CompactClosedAxisBranchFact

    def referenced_nodes(self, branch_node: ast.AST) -> Iterator[ast.AST]:
        for node in ast.walk(branch_node):
            match node:
                case ast.Compare(left=left, comparators=comparators):
                    for operand in (left, *comparators):
                        yield from ast.walk(operand)
                case ast.MatchValue(value=value):
                    yield from ast.walk(value)

    def reference(
        self, parts: tuple[str, ...], receiver: str | None
    ) -> tuple[str, str] | None:
        if len(parts) < 2:
            return None
        return parts[-2], f"{parts[-2]}.{parts[-1]}"


class _ReceiverFieldBranchTally(_BranchReferenceTally):
    """``receiver.field.attribute`` reads, keyed by the receiver's field."""

    fact_type = CompactReceiverFieldBranchRead

    def reference(
        self, parts: tuple[str, ...], receiver: str | None
    ) -> tuple[str, str] | None:
        if receiver is None or len(parts) != 3 or parts[0] != receiver:
            return None
        return parts[1], parts[2]


def _method_receiver(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """The first positional parameter, which binds the receiver of a method."""

    if any(
        ExpressionReference.decoded(decorator).attribute_chain == ("staticmethod",)
        for decorator in function.decorator_list
    ):
        return None
    positional = (*function.args.posonlyargs, *function.args.args)
    return positional[0].arg if positional else None


@dataclass
class _CompactAutoRegisterFunctionReferenceBuilder:
    qualname: str
    receiver_attribute_refs: set[tuple[str, str]]


def _compact_autoregister_reference_projection(
    builders: tuple[_CompactAutoRegisterFunctionReferenceBuilder, ...],
) -> CompactAutoRegisterReferenceIndex | None:
    consumer_builders = tuple(
        builder for builder in builders if builder.receiver_attribute_refs
    )
    if not consumer_builders:
        return None
    receiver_names = sorted_tuple(
        {
            receiver_name
            for builder in consumer_builders
            for receiver_name, _attr_name in builder.receiver_attribute_refs
        }
    )
    attribute_names = sorted_tuple(
        {
            attr_name
            for builder in consumer_builders
            for _receiver_name, attr_name in builder.receiver_attribute_refs
        }
    )
    receiver_indexes = {name: index for index, name in enumerate(receiver_names)}
    attribute_indexes = {name: index for index, name in enumerate(attribute_names)}
    return CompactAutoRegisterReferenceIndex(
        function_qualnames=tuple(builder.qualname for builder in consumer_builders),
        receiver_names=receiver_names,
        attribute_names=attribute_names,
        encoded_edges=";".join(
            f"{function_index},{receiver_index},{attribute_index}"
            for function_index, receiver_index, attribute_index in sorted(
                {
                    (
                        function_index,
                        receiver_indexes[receiver_name],
                        attribute_indexes[attr_name],
                    )
                    for function_index, builder in enumerate(consumer_builders)
                    for receiver_name, attr_name in builder.receiver_attribute_refs
                }
            )
        ),
    )


@dataclass(frozen=True)
class ModuleClassSyntaxFacets(ParsedModuleFact):
    """One module's class syntax facets, autoregister references included."""

    collects_autoregister: ClassVar[bool] = True
    facets: CompactClassSyntaxFacets

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        return cls(
            _compact_class_syntax_facets(
                parsed_module, collect_autoregister=cls.collects_autoregister
            )
        )


@dataclass(frozen=True)
class ModuleClassSyntaxFacetsWithoutAutoregister(ModuleClassSyntaxFacets):
    """One module's class syntax facets, for a demand without autoregister."""

    collects_autoregister: ClassVar[bool] = False


def _compact_class_syntax_facets(
    parsed_module: ParsedModule,
    *,
    collect_autoregister: bool = True,
) -> CompactClassSyntaxFacets:
    syntax_index = parsed_module.syntax_index
    file_path = parsed_module.file_path
    collect_autoregister = (
        collect_autoregister
        and not PythonSourcePathPolicy.is_test_path(parsed_module.path)
    )
    builders_by_function_id = (
        {
            id(function): _CompactAutoRegisterFunctionReferenceBuilder(
                qualname=qualname,
                receiver_attribute_refs=set(),
            )
            for qualname, function in syntax_index.named_functions
        }
        if collect_autoregister
        else {}
    )
    function_bindings_by_id = {
        id(function): LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(function.body)
        | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(function)
        for _qualname, function in syntax_index.named_functions
    }
    module_binds_type = "type" in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
        parsed_module.module.body
    )
    active_function_ids_by_scope = tuple(
        tuple(
            id(syntax_index.depth_first_nodes[function_index])
            for function_index in scope.function_node_indices
        )
        for scope in syntax_index.scopes
    )
    active_builders_by_scope = tuple(
        (
            tuple(
                builders_by_function_id[function_id]
                for function_id in active_function_ids
            )
            if builders_by_function_id
            else ()
        )
        for active_function_ids in active_function_ids_by_scope
    )
    scope_binds_type = tuple(
        module_binds_type
        or any(
            "type" in function_bindings_by_id[function_id]
            for function_id in active_function_ids
        )
        for active_function_ids in active_function_ids_by_scope
    )
    scope_qualnames = tuple(".".join(scope.names) for scope in syntax_index.scopes)
    enum_member_tally = _EnumMemberBranchTally()
    receiver_field_tally = _ReceiverFieldBranchTally()
    receivers_by_function_id = {
        id(function): _method_receiver(function)
        for _qualname, function in syntax_index.named_functions
    }
    exact_type_guards: list[CompactExactTypeGuard] = []
    indices_by_type = syntax_index.node_indices_by_type

    if builders_by_function_id:
        for node_index in indices_by_type.get(ast.Attribute, ()):
            node = syntax_index.depth_first_nodes[node_index]
            if not isinstance(node.value, ast.Name):
                continue
            receiver_reference = (node.value.id, node.attr)
            for builder in active_builders_by_scope[syntax_index.scope_ids[node_index]]:
                builder.receiver_attribute_refs.add(receiver_reference)

    for node_index in indices_by_type.get(ast.If, ()):
        active_function_ids = active_function_ids_by_scope[
            syntax_index.scope_ids[node_index]
        ]
        if not active_function_ids:
            continue
        active_function_id = active_function_ids[-1]
        node = syntax_index.depth_first_nodes[node_index]
        for tally in (enum_member_tally, receiver_field_tally):
            tally.add(
                active_function_id,
                (node.test,),
                receivers_by_function_id[active_function_id],
            )

    for node_index in indices_by_type.get(ast.Match, ()):
        active_function_ids = active_function_ids_by_scope[
            syntax_index.scope_ids[node_index]
        ]
        if not active_function_ids:
            continue
        active_function_id = active_function_ids[-1]
        node = syntax_index.depth_first_nodes[node_index]
        branch = tuple(
            part
            for case in node.cases
            for part in (case.pattern, case.guard)
            if part is not None
        )
        for tally in (enum_member_tally, receiver_field_tally):
            tally.add(
                active_function_id,
                branch,
                receivers_by_function_id[active_function_id],
            )

    for node_index in merge(
        indices_by_type.get(ast.If, ()),
        indices_by_type.get(ast.Assert, ()),
    ):
        scope_id = syntax_index.scope_ids[node_index]
        if not active_function_ids_by_scope[scope_id]:
            continue
        node = syntax_index.depth_first_nodes[node_index]
        predicate = _exact_type_predicate(node.test)
        if predicate is None:
            continue
        if isinstance(node, ast.If):
            matches_exact_type_when_true = predicate[2]
            rejects_descendants = (
                not matches_exact_type_when_true and _fail_loud_block(node.body)
            ) or (matches_exact_type_when_true and _fail_loud_block(node.orelse))
            if not rejects_descendants:
                continue
        elif not predicate[2]:
            continue
        if scope_binds_type[scope_id]:
            continue
        subject, type_reference, matches_exact_type_when_true, expression = predicate
        reference_node = ClassSymbolResolutionAuthority.reference_node(type_reference)
        parts = ExpressionReference.decoded(reference_node).attribute_chain
        if parts is None:
            continue
        exact_type_guards.append(
            CompactExactTypeGuard(
                file_path=file_path,
                line=node.lineno,
                qualname=scope_qualnames[scope_id],
                subject_expression=ast.unparse(subject),
                type_reference_expression=ast.unparse(type_reference),
                type_reference_parts=parts,
                matches_exact_type_when_true=matches_exact_type_when_true,
                expression=expression,
            )
        )

    builders = (
        tuple(
            builders_by_function_id[id(function)]
            for _qualname, function in syntax_index.named_functions
        )
        if collect_autoregister
        else ()
    )
    autoregister_index = _compact_autoregister_reference_projection(builders)
    closed_axis_functions: list[CompactClosedAxisBranchFunction] = []
    for qualname, function in syntax_index.named_functions:
        axes = enum_member_tally.facts(id(function))
        receiver_field_reads = receiver_field_tally.facts(id(function))
        if axes or receiver_field_reads:
            closed_axis_functions.append(
                CompactClosedAxisBranchFunction(
                    file_path=file_path,
                    line=function.lineno,
                    qualname=qualname,
                    axes=axes,
                    receiver_field_reads=receiver_field_reads,
                )
            )
    return CompactClassSyntaxFacets(
        autoregister_reference_index=autoregister_index,
        closed_axis_branch_functions=tuple(closed_axis_functions),
        exact_type_guards=tuple(exact_type_guards),
    )


def _compact_class_member_declarations(
    node: ast.ClassDef,
) -> tuple[CompactClassMemberDeclaration, ...]:
    declarations: list[CompactClassMemberDeclaration] = []
    for statement in node.body:
        assignment_statement = AssignmentStatement.decoded(statement)
        if assignment_statement is None:
            continue
        targets = assignment_statement.targets
        value = assignment_statement.value
        annotation = assignment_statement.annotation
        constructor_name = (
            _terminal_reference_name(value.func)
            if isinstance(value, ast.Call)
            else None
        )
        constructor_keyword_names = (
            sorted_tuple(
                CallKeywords.of(value.keywords).names
            )
            if isinstance(value, ast.Call) and constructor_name is not None
            else ()
        )
        compact_value = StatementValue.of(assignment_statement.node).compact_value()
        declarations.extend(
            CompactClassMemberDeclaration(
                name=target.id,
                line=statement.lineno,
                expression=ast.unparse(value) if value is not None else None,
                value=compact_value,
                constructor_name=constructor_name,
                constructor_keyword_names=constructor_keyword_names,
                annotation_expression=(
                    ast.unparse(annotation) if annotation is not None else None
                ),
            )
            for target in targets
            if isinstance(target, ast.Name)
        )
    return tuple(declarations)


def _compact_sorted_key_calls(
    parsed_module: ParsedModule,
) -> tuple[CompactSortedKeyCall, ...]:
    calls: list[CompactSortedKeyCall] = []
    for node in parsed_module.syntax_index.breadth_first_nodes:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sorted"
        ):
            continue
        registry_owner_names = sorted_tuple(
            {
                child.value.id
                for argument in node.args
                for child in ast.walk(argument)
                if isinstance(child, ast.Attribute)
                and child.attr == "__registry__"
                and isinstance(child.value, ast.Name)
            }
        )
        key_attribute_names: set[str] = set()
        for keyword in node.keywords:
            if keyword.arg != "key":
                continue
            key_attribute_names.update(
                child.attr
                for child in ast.walk(keyword.value)
                if isinstance(child, ast.Attribute)
            )
            for child in ast.walk(keyword.value):
                if not (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "attrgetter"
                ):
                    continue
                key_attribute_names.update(
                    argument.value
                    for argument in child.args
                    if isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                )
        if key_attribute_names:
            calls.append(
                CompactSortedKeyCall(
                    file_path=parsed_module.file_path,
                    line=node.lineno,
                    registry_owner_names=registry_owner_names,
                    key_attribute_names=sorted_tuple(key_attribute_names),
                )
            )
    return tuple(calls)
