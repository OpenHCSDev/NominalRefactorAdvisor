"""Efficiency detectors: patterns where runtime work restates a fact the structure already holds.

Each detector here finds code that re-derives information a declaration already provided.
The cost is runtime (O(n^2) loops, repeated lookups) and maintenance (the restated route
drifts from its source). Paper1 Corollary 4.10: passing runs cannot reveal the outside
supply; the structure that makes the dependence checkable must be preserved.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from operator import attrgetter

from collections.abc import Iterator

from nominal_refactor_advisor.ast_projection import (
    name_nodes,
    names_mentioned,
    nodes_of,
)
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_families.parsed_index import ModuleClassStatements
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.foreign_family import FunctionDefinition
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.observation_families import ShapeFamily
from nominal_refactor_advisor.observation_shapes import RestatingSite, SymbolSite
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.relocation_observations import (
    ClassShapeFamily,
    ModuleClassReadings,
    declares_abstract,
)
from nominal_refactor_advisor.restated_fact_observations import (
    RestatingSiteFamily,
    RestatingSiteModule,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag
from nominal_refactor_advisor.detectors._relocations import MemberFamilies
from nominal_refactor_advisor.detectors._base import (
    CompactMultiModuleProjectionDetectorMixin,
    CompactProjectionGroups,
    DetectorConfig,
    IssueDetector,
    PerModuleIssueDetector,
    high_confidence_spec,
)


# ---------------------------------------------------------------------------
# Detector 1: callback parameter threaded through intermediate signatures
# ---------------------------------------------------------------------------

_MIN_THREADING_FUNCTIONS = 2


@dataclass(frozen=True)
class _ThreadedParameter:
    function_name: str
    param_name: str
    line: int


def _forwarded_only_params(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[_ThreadedParameter]:
    """Return parameters that have a default and are only used as same-name keyword forwards."""

    params_with_defaults: dict[str, int] = {}
    args = func.args
    num_args = len(args.args)
    num_defaults = len(args.defaults)
    offset = num_args - num_defaults
    for i, default in enumerate(args.defaults):
        arg = args.args[offset + i]
        params_with_defaults[arg.arg] = arg.lineno

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            params_with_defaults[arg.arg] = arg.lineno

    if not params_with_defaults:
        return []

    keyword_forwards = _keyword_forward_names(func)
    name_uses: dict[str, int] = defaultdict(int)
    forward_uses: dict[str, int] = defaultdict(int)

    for node in name_nodes(func):
        if node.id in params_with_defaults:
            name_uses[node.id] += 1
            if (node.lineno, node.col_offset, node.id) in keyword_forwards:
                forward_uses[node.id] += 1

    result = []
    for param_name, line in params_with_defaults.items():
        total = name_uses.get(param_name, 0)
        forwarded = forward_uses.get(param_name, 0)
        if total > 0 and total == forwarded:
            result.append(
                _ThreadedParameter(
                    function_name=func.name,
                    param_name=param_name,
                    line=line,
                )
            )

    return result


def _keyword_forward_names(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[tuple[int, int, str]]:
    """Collect (line, col, name) of Name nodes used as same-name keyword arguments."""

    forwards: set[tuple[int, int, str]] = set()
    for node in nodes_of(ast.Call, func):
        for keyword in node.keywords:
            if (
                keyword.arg is not None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == keyword.arg
            ):
                forwards.add((
                    keyword.value.lineno,
                    keyword.value.col_offset,
                    keyword.value.id,
                ))
    return forwards


class CallbackThreadingDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.CALLBACK_THREADING,
        "Callback parameter threaded through intermediate signatures",
        (
            "A parameter with a default is forwarded unchanged as a same-name keyword "
            "argument through two or more function signatures. Each intermediate "
            "signature restates the route; the parameter belongs to the layer that "
            "uses it (paper1 Theorem 4.6: helpers cannot lower d-down)."
        ),
        "parameter owned by its consumer, not threaded through intermediaries",
        "same-name defaulted parameter forwarded without inspection across functions",
        (CapabilityTag.SHARED_ALGORITHM_AUTHORITY,),
        (ObservationTag.DATAFLOW_ROOT,),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        threaded_by_name: dict[str, list[_ThreadedParameter]] = defaultdict(list)
        index = module.syntax_index

        for _node_index, node in (
            *index.indexed_nodes_of_type(ast.FunctionDef),
            *index.indexed_nodes_of_type(ast.AsyncFunctionDef),
        ):
            for tp in _forwarded_only_params(node):
                threaded_by_name[tp.param_name].append(tp)

        findings: list[RefactorFinding] = []
        for param_name, sites in sorted(threaded_by_name.items()):
            if len(sites) < _MIN_THREADING_FUNCTIONS:
                continue
            evidence = tuple(
                SourceLocation(
                    str(module.path),
                    site.line,
                    f"{site.function_name}({site.param_name}=)",
                )
                for site in sites
            )
            findings.append(
                self.build_finding(
                    (
                        f"parameter `{param_name}` threaded through "
                        f"{len(sites)} signatures: "
                        f"{', '.join(site.function_name for site in sites)}"
                    ),
                    evidence,
                )
            )
        return findings


# ---------------------------------------------------------------------------
# Detector 2: loop element discarded and re-looked up
# ---------------------------------------------------------------------------


def _for_target_names(target: ast.expr) -> set[str]:
    """All simple names bound by a for-loop target, including unpacking."""
    names: set[str] = set()
    match target:
        case ast.Name(id=name):
            names.add(name)
        case ast.Tuple(elts=elements) | ast.List(elts=elements):
            for elt in elements:
                names.update(_for_target_names(elt))
        case ast.Starred(value=inner):
            names.update(_for_target_names(inner))
    return names


def _unparse_short(node: ast.expr) -> str:
    """Short readable form for evidence labels."""
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=inner, attr=attr):
            return f"{_unparse_short(inner)}.{attr}"
        case ast.Call(func=func):
            return f"{_unparse_short(func)}(...)"
    return "?"


def _element_attr_args(
    call: ast.Call, target_names: set[str],
) -> list[tuple[str, str]]:
    """Return (target_name, attr) pairs for call arguments that are target.attr."""
    hits: list[tuple[str, str]] = []
    for arg in (*call.args, *(kw.value for kw in call.keywords)):
        match arg:
            case ast.Attribute(value=ast.Name(id=var), attr=attr) if var in target_names:
                hits.append((var, attr))
    return hits


@dataclass(frozen=True)
class _LoopRelookup:
    """One site where a loop body uses element.attr as a lookup key."""
    loop_line: int
    iterator_label: str
    call_line: int
    call_label: str
    element_attr: str
    target_name: str
    enclosing_function: str


def _find_relookups_in_for(
    for_node: ast.For | ast.AsyncFor,
    enclosing_function: str,
) -> list[_LoopRelookup]:
    """Find calls in a for-loop body that look up a value using element.attr.

    The general pattern:
        for element in source:
            result = lookup(..., element.attr)

    The element was yielded by the iterator. The body uses element.attr as a key
    to recover related information. When lookup and source share a name (same
    class, same variable, same first argument), it strongly suggests the same
    collection is queried twice — the iterator already had what the lookup
    re-derives.

    This is not specific to any particular API. It catches any loop where:
    1. The target binds one or more names
    2. A call in the body takes target.attr as an argument
    3. That call shares at least one name with the iterator expression
       (the "same source" signal)
    """

    target_names = _for_target_names(for_node.target)
    if not target_names:
        return []

    source_names = names_mentioned(for_node.iter)
    if not source_names:
        return []

    iter_label = _unparse_short(for_node.iter)
    relookups: list[_LoopRelookup] = []

    for node in nodes_of(ast.Call, for_node):
        hits = _element_attr_args(node, target_names)
        if not hits:
            continue

        call_names = set(names_mentioned(node.func))
        for arg in node.args:
            if isinstance(arg, ast.Name):
                call_names.add(arg.id)
        for kw in node.keywords:
            if isinstance(kw.value, ast.Name):
                call_names.add(kw.value.id)

        if not (call_names & source_names - target_names):
            continue

        call_label = _unparse_short(node.func)
        for target_name, attr in hits:
            relookups.append(_LoopRelookup(
                loop_line=for_node.lineno,
                iterator_label=iter_label,
                call_line=node.lineno,
                call_label=call_label,
                element_attr=attr,
                target_name=target_name,
                enclosing_function=enclosing_function,
            ))

    return relookups


def _collect_loop_relookups(module: ParsedModule) -> list[_LoopRelookup]:
    """Walk every function in the module for for-loops with relookup calls."""

    results: list[_LoopRelookup] = []
    index = module.syntax_index

    for _node_index, func_node in (
        *index.indexed_nodes_of_type(ast.FunctionDef),
        *index.indexed_nodes_of_type(ast.AsyncFunctionDef),
    ):
        for node in ast.walk(func_node):
            if isinstance(node, (ast.For, ast.AsyncFor)):
                results.extend(_find_relookups_in_for(node, func_node.name))

    return results


# ---------------------------------------------------------------------------
# Detector 3: linear scan by attribute where a dict lookup suffices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LinearScan:
    """One site where a sequence is scanned to find an element by attribute."""
    line: int
    spelling: str
    sequence_label: str
    attribute: str
    enclosing_function: str


def _linear_scan_sites(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[_LinearScan]:
    """Find `next(x for x in SEQ if x.ATTR == KEY)` and `[x for x in SEQ if x.ATTR == KEY][0]`.

    Both are O(n) scans finding one element by attribute comparison. A dict keyed
    by that attribute answers the same question in O(1). The scan is the linear
    search the dict replaces.
    """
    results: list[_LinearScan] = []

    for node in ast.walk(func):
        match node:
            case ast.Call(
                func=ast.Name(id="next"),
                args=[ast.GeneratorExp() as genexp],
            ):
                hit = _scan_from_comprehension(genexp, "next(...)")
                if hit is not None:
                    results.append(_LinearScan(
                        line=node.lineno,
                        spelling="next(... for ... if .attr == key)",
                        sequence_label=hit[0],
                        attribute=hit[1],
                        enclosing_function=func.name,
                    ))

            case ast.Subscript(
                value=ast.ListComp() as listcomp,
                slice=ast.Constant(value=0),
            ):
                hit = _scan_from_comprehension(listcomp, "[...][0]")
                if hit is not None:
                    results.append(_LinearScan(
                        line=node.lineno,
                        spelling="[... for ... if .attr == key][0]",
                        sequence_label=hit[0],
                        attribute=hit[1],
                        enclosing_function=func.name,
                    ))

    return results


def _scan_from_comprehension(
    comp: ast.GeneratorExp | ast.ListComp,
    label: str,
) -> tuple[str, str] | None:
    """Extract (sequence_label, attribute) from a comprehension that filters by attribute equality.

    Matches:  x for x in SEQ if x.ATTR == KEY   (one generator, one if, equality test)
    Returns (readable SEQ label, ATTR name) or None.
    """
    if len(comp.generators) != 1:
        return None
    gen = comp.generators[0]
    if len(gen.ifs) != 1:
        return None

    match gen.target:
        case ast.Name(id=var_name):
            pass
        case _:
            return None

    condition = gen.ifs[0]
    attr_name = _equality_attribute_filter(condition, var_name)
    if attr_name is None:
        return None

    seq_label = _unparse_short(gen.iter)
    return (seq_label, attr_name)


def _equality_attribute_filter(condition: ast.expr, var_name: str) -> str | None:
    """If condition is `var.ATTR == expr` or `expr == var.ATTR`, return ATTR."""
    match condition:
        case ast.Compare(
            left=left,
            ops=[ast.Eq()],
            comparators=[right],
        ):
            attr = _attr_of_var(left, var_name) or _attr_of_var(right, var_name)
            return attr
    return None


def _attr_of_var(node: ast.expr, var_name: str) -> str | None:
    """If node is `var_name.ATTR`, return ATTR."""
    match node:
        case ast.Attribute(value=ast.Name(id=name), attr=attr) if name == var_name:
            return attr
    return None


# ---------------------------------------------------------------------------
# Detector 4: method parameter ignored in every override
# ---------------------------------------------------------------------------


def _candidate_parameters(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, int]:
    """Each parameter a method could be asked to justify, with its line.

    Leaves out `self`, `cls`, `args`, `kwargs`, names starting with `_` (the
    convention for intentionally unused) and every parameter of a dunder method.
    """

    if func.name.startswith("__") and func.name.endswith("__"):
        return {}
    arguments = (
        *func.args.args,
        *func.args.kwonlyargs,
        *(
            argument
            for argument in (func.args.vararg, func.args.kwarg)
            if argument is not None
        ),
    )
    return {
        argument.arg: argument.lineno
        for argument in arguments
        if argument.arg not in {"self", "cls", "args", "kwargs"}
        and not argument.arg.startswith("_")
    }


@dataclass(frozen=True)
class MethodParameterReading(SymbolSite):
    """One method declaration that does more than ask, and which of its
    parameters its body names."""

    class_qualname: str
    method_name: str
    ignored: tuple[tuple[str, int], ...]
    """Each parameter the body never names, with its line."""
    read: tuple[str, ...]
    """The parameters the body names."""


@dataclass(frozen=True)
class MethodParametersModule:
    module_name: str
    methods: tuple[MethodParameterReading, ...]


class MethodParameterFamily(ShapeFamily[MethodParametersModule]):
    """Each method that does more than ask, with the parameters it reads and ignores.

    A declaration that only asks (`declares_abstract`) fixes a signature it does
    not use, and is left out.
    """

    item_type = MethodParametersModule

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[MethodParametersModule]:
        methods: list[MethodParameterReading] = []
        for reading in parsed_module.fact(ModuleClassReadings).classes:
            for member in FunctionDefinition.within(reading.node.body):
                candidates = _candidate_parameters(member)
                if not candidates or declares_abstract(member, reading.bindings):
                    continue
                named = names_mentioned(*member.body)
                methods.append(
                    MethodParameterReading(
                        file_path=parsed_module.file_path,
                        lineno=member.lineno,
                        symbol=f"{reading.qualname}.{member.name}",
                        class_qualname=reading.qualname,
                        method_name=member.name,
                        ignored=tuple(
                            (name, line)
                            for name, line in candidates.items()
                            if name not in named
                        ),
                        read=tuple(name for name in candidates if name in named),
                    )
                )
        return [MethodParametersModule(parsed_module.module_name, tuple(methods))]


class UnusedMethodParameterDetector(
    CompactMultiModuleProjectionDetectorMixin, IssueDetector
):
    """A parameter that no declaration of a method reads.

    Declarations are grouped by the topmost classes that declare the method
    above them, so a default that ignores a parameter its overrides read is not
    a finding. A class with an ancestor the scan does not declare is left out:
    that ancestor may fix the signature.
    """

    module_projection_families = (
        MethodParameterFamily,
        CompactModuleClassProjectionFamily,
        ClassShapeFamily,
    )
    compact_shared_group_context_builder = staticmethod(
        MemberFamilies.from_projection_groups
    )
    finding_spec = high_confidence_spec(
        PatternId.UNUSED_METHOD_PARAMETER,
        "Method parameter ignored in every override",
        (
            "A method parameter is unused in the body. When the method is on a "
            "compiled or frozen object and called in a loop, the result is "
            "recomputed from self alone on every call. The parameter should be "
            "removed, or the result cached at construction."
        ),
        "parameter removed or result cached at construction",
        "method parameter unused in the body on a concrete class",
        (CapabilityTag.EXACT_LOOKUP,),
        (ObservationTag.DATAFLOW_ROOT,),
    )

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        families = MemberFamilies.require(context)
        declarations: dict[tuple[str, str], list[MethodParameterReading]] = (
            defaultdict(list)
        )
        for module in MethodParameterFamily.modules_in(projections_by_family):
            for method in module.methods:
                symbol = f"{module.module_name}.{method.class_qualname}"
                if not families.declares_all_ancestors(symbol):
                    continue
                owner = families.family(symbol, method.method_name) or symbol
                declarations[(owner, method.method_name)].append(method)
        findings: list[RefactorFinding] = []
        for (owner, method_name), members in sorted(declarations.items()):
            read = {name for member in members for name in member.read}
            ignored = tuple(
                (member, name, line)
                for member in sorted(members, key=attrgetter("file_path", "lineno"))
                for name, line in member.ignored
                if name not in read
            )
            if not ignored:
                continue
            names = ", ".join(
                f"`{name}`" for name in dict.fromkeys(name for _, name, _ in ignored)
            )
            match members:
                case [only]:
                    where = only.symbol
                case _:
                    where = (
                        f"`{method_name}`, in all {len(members)} declarations "
                        f"under {owner},"
                    )
            findings.append(
                self.build_finding(
                    f"{where} ignores parameter(s) {names} — "
                    f"the result depends only on self",
                    tuple(
                        SourceLocation(
                            member.file_path, line, f"{member.symbol}({name})"
                        )
                        for member, name, line in ignored
                    ),
                )
            )
        return findings


class LinearScanByNameDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.LINEAR_SCAN_BY_NAME,
        "Linear scan by attribute where a dict lookup suffices",
        (
            "A sequence is scanned with next(x for x in SEQ if x.attr == key) or "
            "the [... if ...][0] spelling to find one element by attribute. A dict "
            "keyed by that attribute answers the same question in O(1). When the "
            "sequence is a field of a frozen or cached plan, the dict can be built "
            "once at construction."
        ),
        "dict keyed by the attribute, built once; O(1) lookup",
        "linear scan of a sequence filtering by attribute equality",
        (CapabilityTag.EXACT_LOOKUP,),
        (ObservationTag.DATAFLOW_ROOT,),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        index = module.syntax_index
        scans: list[_LinearScan] = []

        for _node_index, func_node in (
            *index.indexed_nodes_of_type(ast.FunctionDef),
            *index.indexed_nodes_of_type(ast.AsyncFunctionDef),
        ):
            scans.extend(_linear_scan_sites(func_node))

        if not scans:
            return []

        findings: list[RefactorFinding] = []
        for scan in scans:
            evidence = (
                SourceLocation(
                    str(module.path),
                    scan.line,
                    f"{scan.spelling} over {scan.sequence_label}",
                ),
            )
            findings.append(
                self.build_finding(
                    (
                        f"linear scan of `{scan.sequence_label}` by "
                        f"`.{scan.attribute}` in {scan.enclosing_function} — "
                        f"a dict keyed by `{scan.attribute}` is O(1)"
                    ),
                    evidence,
                )
            )
        return findings


class LoopRelookupDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.LOOP_RELOOKUP,
        "Loop element discarded and re-looked up by attribute",
        (
            "A for-loop body calls a function passing element.attr as a lookup key, "
            "where element was already yielded by the iterator. The call shares a "
            "name with the iterator source, indicating the same collection is queried "
            "twice. When the lookup scans the collection this is O(n^2). The fix: "
            "the iterator yields the needed value directly, or a dict keyed by the "
            "attribute is built once before the loop."
        ),
        "iterator yields the needed value directly; no second lookup",
        "loop body uses element.attr to re-derive a value the iterator provided",
        (CapabilityTag.EXACT_LOOKUP,),
        (ObservationTag.DATAFLOW_ROOT,),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        relookups = _collect_loop_relookups(module)
        if not relookups:
            return []

        grouped: dict[tuple[int, str], list[_LoopRelookup]] = defaultdict(list)
        for rl in relookups:
            grouped[(rl.loop_line, rl.iterator_label)].append(rl)

        findings: list[RefactorFinding] = []
        for (loop_line, iterator), sites in sorted(grouped.items()):
            evidence = [
                SourceLocation(
                    str(module.path),
                    loop_line,
                    f"for ... in {iterator}",
                ),
            ]
            for site in sites:
                evidence.append(SourceLocation(
                    str(module.path),
                    site.call_line,
                    f"{site.call_label}(..., {site.target_name}.{site.element_attr})",
                ))

            call_names = sorted({s.call_label for s in sites})
            findings.append(
                self.build_finding(
                    (
                        f"loop over `{iterator}` re-looks up {len(sites)} "
                        f"element(s) via {', '.join(f'`{c}`' for c in call_names)} "
                        f"in {sites[0].enclosing_function}"
                    ),
                    tuple(evidence),
                )
            )
        return findings


# ---------------------------------------------------------------------------
# Detector 5: same function/method name in 2+ modules, no shared ancestor
# ---------------------------------------------------------------------------


_MIN_PHANTOM_MODULES = 2
_SKIP_COMMON_NAMES = frozenset({
    "setup", "teardown", "main", "run", "execute", "process", "handle",
    "validate", "serialize", "deserialize", "parse", "render", "format",
    "reset", "close", "open", "start", "stop", "configure", "initialize",
})


class PublicDefinitionSiteFamily(RestatingSiteFamily[RestatingSiteModule]):
    """Each public top-level function and each public method of a top-level
    class, one site per def: `group` is its name, `detail` its class."""

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        for declared in parsed_module.fact(ModuleClassStatements).top_level:
            for method in declared.methods:
                if not method.name.startswith("_"):
                    yield cls.site(
                        parsed_module,
                        method.node,
                        f"{declared.name}.{method.name}",
                        method.name,
                        (declared.name,),
                    )
        for function in FunctionDefinition.within(parsed_module.module.body):
            if not function.name.startswith("_"):
                yield cls.site(parsed_module, function, function.name, function.name, ())


class PhantomFamilyDetector(CompactMultiModuleProjectionDetectorMixin, IssueDetector):
    """One public name declared by owners that share no declaring ancestor.

    A method's owner is the topmost classes declaring its name above it, so an
    override and the declaration it overrides are one owner. A module function
    is its own owner. A class with an ancestor the scan does not declare is left
    out: that ancestor may declare the name, so nothing is claimed.
    """

    module_projection_families = (
        PublicDefinitionSiteFamily,
        CompactModuleClassProjectionFamily,
        ClassShapeFamily,
    )
    compact_shared_group_context_builder = staticmethod(
        MemberFamilies.from_projection_groups
    )
    finding_spec = high_confidence_spec(
        PatternId.PHANTOM_FAMILY,
        "Same name across modules with no shared declaration",
        (
            "A function or method name appears in multiple modules of one package "
            "with no shared ancestor declaring it. The name promises a family that "
            "does not exist. Each body is independently maintained and can drift; "
            "callers cannot dispatch through a common type."
        ),
        "one declared owner (ABC, base method, or family) that each site overrides",
        "same non-private name defined in 2+ modules with no common declaring ancestor",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (ObservationTag.METHOD_ROLE,),
    )

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        families = MemberFamilies.require(context)
        owners_by_name: dict[str, dict[str, list[RestatingSite]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for module in PublicDefinitionSiteFamily.modules_in(projections_by_family):
            for site in module.sites:
                if site.group in _SKIP_COMMON_NAMES:
                    continue
                match site.detail:
                    case (class_name,):
                        symbol = f"{module.module_name}.{class_name}"
                        if not families.declares_all_ancestors(symbol):
                            continue
                        owner = families.family(symbol, site.group) or symbol
                    case _:
                        owner = f"{module.module_name}.{site.group}"
                owners_by_name[site.group][owner].append(site)
        return [
            self._finding_for_name(name, owners)
            for name, owners in sorted(owners_by_name.items())
            if len(owners) >= 2
            and len({site.file_path for sites in owners.values() for site in sites})
            >= _MIN_PHANTOM_MODULES
        ]

    def _finding_for_name(
        self, name: str, owners: dict[str, list[RestatingSite]]
    ) -> RefactorFinding:
        first_sites = sorted(
            (min(sites, key=attrgetter("file_path", "lineno")) for sites in owners.values()),
            key=attrgetter("file_path", "lineno"),
        )
        module_count = len(
            {site.file_path for sites in owners.values() for site in sites}
        )
        declared_by = ", ".join(f"`{owner}`" for owner in sorted(owners))
        summary = (
            f"`{name}` is declared by {len(owners)} owners across {module_count} "
            f"modules with no shared declaring ancestor: {declared_by}"
        )
        return self.build_finding(summary, tuple(site.location for site in first_sites))
