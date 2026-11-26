from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor import ast_field_decoders
from nominal_refactor_advisor.detectors._registration_owners import HandRolledRegistrationDetector
from nominal_refactor_advisor.detectors._restated_facts import ForeignAbsenceProbeDetector
from nominal_refactor_advisor.detectors._runtime import ManualClassRegistrationDetector
from nominal_refactor_advisor.ast_tools import SourceModule, parse_python_modules
from nominal_refactor_advisor.refactors.optional_ast_field import (
    KeywordNameDecoder,
    CaptureNameDecoder,
    DeclaredAnnotationDecoder,
    ExceptionTypeDecoder,
    ImportAliasDecoder,
    ImportModuleDecoder,
    OptionalAstFieldAuthority,
    OptionalAstFieldDecoder,
    OptionalValueDecoder,
    ProjectDecoderOwners,
    VariadicParameterDecoder,
)
from nominal_refactor_advisor.codemod import CodemodPlanSequence, codemod_plan_from_findings
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_semantics import FindingRecipeSynthesisStatus
from nominal_refactor_advisor.codemod_source_edits import SourceTextGeometry
from nominal_refactor_advisor.detectors._base import DetectorConfig


def _rewritten(
    source: str, decoder: type[OptionalAstFieldDecoder] = KeywordNameDecoder
) -> str:
    parsed = SourceModule.from_source_path(Path("m.py"), source).parse()
    function = next(
        node for node in parsed.module.body if isinstance(node, ast.FunctionDef)
    )
    authority = OptionalAstFieldAuthority(
        parsed_module=parsed,
        geometry=SourceTextGeometry(source),
        line_length=88,
        declaration=function,
        decoder=decoder,
    )
    rewritten = source
    for replacement in sorted(
        authority.replacements(), key=lambda item: item.start_offset, reverse=True
    ):
        rewritten = (
            rewritten[: replacement.start_offset]
            + replacement.replacement_source
            + rewritten[replacement.end_offset :]
        )
    ast.parse(rewritten)
    return rewritten


@pytest.mark.parametrize(
    ("statement", "expected"),
    (
        (
            "a = any(keyword.arg is None for keyword in call.keywords)",
            "a = bool(CallKeywords.of(call.keywords).splats)",
        ),
        (
            "if x or any(keyword.arg is None for keyword in call.keywords): pass",
            "if x or CallKeywords.of(call.keywords).splats: pass",
        ),
        (
            "a = (x, x or any(keyword.arg is None for keyword in call.keywords))",
            "a = (x, x or bool(CallKeywords.of(call.keywords).splats))",
        ),
        (
            "b = all(keyword.arg is not None for keyword in call.keywords)",
            "b = not CallKeywords.of(call.keywords).splats",
        ),
        (
            "c = [kw.arg for kw in call.keywords if kw.arg is not None and kw.value]",
            "c = [kw.name for kw in CallKeywords.of(call.keywords).named if kw.value]",
        ),
        (
            "d = all(keyword.arg is not None for keyword in call.keywords) == x",
            "d = (not CallKeywords.of(call.keywords).splats) == x",
        ),
    ),
)
def test_expression_probes_read_the_decoder(statement: str, expected: str) -> None:
    rewritten = _rewritten(f"def f(call, x):\n    {statement}\n")
    assert rewritten == f"def f(call, x):\n    {expected}\n"


def test_loop_guards_read_the_decoder() -> None:
    source = (
        "def f(call, projected: tuple[ast.keyword, ...]):\n"
        "    for keyword in call.keywords:\n"
        "        if keyword.arg is None:\n"
        "            continue\n"
        "        use(keyword.arg)\n"
        "    for keyword in projected:\n"
        "        if keyword.arg is None:\n"
        "            return None\n"
        "        use(keyword.arg, keyword.value)\n"
    )
    assert _rewritten(source) == (
        "def f(call, projected: tuple[ast.keyword, ...]):\n"
        "    for keyword in CallKeywords.of(call.keywords).named:\n"
        "        use(keyword.name)\n"
        "    keywords = CallKeywords.of(projected)\n"
        "    if keywords.splats:\n"
        "        return None\n"
        "    for keyword in keywords.named:\n"
        "        use(keyword.name, keyword.value)\n"
    )


def test_probe_outside_a_decoded_collection_is_handed_back() -> None:
    with pytest.raises(ValueError, match="no shape the KeywordNameDecoder"):
        _rewritten("def f(keyword):\n    if keyword.arg is None:\n        return 1\n")


def test_element_used_as_an_ast_node_is_handed_back() -> None:
    with pytest.raises(ValueError, match="used as an ast node"):
        _rewritten(
            "def f(call):\n"
            "    return [ast.dump(k) for k in call.keywords if k.arg is not None]\n"
        )


def test_unknown_collection_is_not_a_site() -> None:
    with pytest.raises(ValueError, match="no shape"):
        _rewritten(
            "def f(items):\n    return [k for k in items if k.arg is not None]\n"
        )


def test_filtered_comprehension_is_fit_to_the_line() -> None:
    source = (
        "def f(call_node):\n"
        "    return tuple(\n"
        "        (keyword.arg, keyword.value) for keyword in call_node.keywords"
        " if keyword.arg is not None\n"
        "    )\n"
    )
    rewritten = _rewritten(source)
    assert "CallKeywords.of(call_node.keywords).named" in rewritten
    assert max(len(line) for line in rewritten.splitlines()) <= 88


def test_present_guard_on_a_proved_node_loops_over_its_decoded_names() -> None:
    source = (
        "def visit_MatchAs(self, node: ast.MatchAs):\n"
        "    if node.name is not None:\n"
        "        self.names.add(node.name)\n"
        "    self.generic_visit(node)\n"
    )
    assert _rewritten(source, CaptureNameDecoder) == (
        "def visit_MatchAs(self, node: ast.MatchAs):\n"
        "    for capture_name in capture_names(node):\n"
        "        self.names.add(capture_name)\n"
        "    self.generic_visit(node)\n"
    )


@pytest.mark.parametrize(
    ("decoder", "param", "field", "expected_call", "loop_var"),
    (
        (ImportAliasDecoder, "alias: ast.alias", "asname", "alias_names(alias)", "import_alias"),
        (OptionalValueDecoder, "node: ast.Return", "value", "optional_values(node)", "optional_value"),
        (OptionalValueDecoder, "node: ast.AnnAssign", "value", "optional_values(node)", "optional_value"),
        (ImportModuleDecoder, "node: ast.ImportFrom", "module", "import_modules(node)", "import_module"),
        (ExceptionTypeDecoder, "handler: ast.ExceptHandler", "type", "exception_types(handler)", "exception_type"),
    ),
)
def test_additional_optional_ast_field_decoders_rewrite_present_guards(
    decoder: type[OptionalAstFieldDecoder],
    param: str,
    field: str,
    expected_call: str,
    loop_var: str,
) -> None:
    subject = param.split(":")[0]
    source = (
        f"def f({param}, out):\n"
        f"    if {subject}.{field} is not None:\n"
        f"        out.append({subject}.{field})\n"
    )
    assert _rewritten(source, decoder) == (
        f"def f({param}, out):\n"
        f"    for {loop_var} in {expected_call}:\n"
        f"        out.append({loop_var})\n"
    )


def test_iterated_enumerate_handlers_and_import_names_prove_element_class() -> None:
    try_source = (
        "def visit_Try(self, node: ast.Try):\n"
        "    for index, handler in enumerate(node.handlers):\n"
        "        if handler.type is not None:\n"
        "            self.visit(handler.type)\n"
    )
    assert _rewritten(try_source, ExceptionTypeDecoder) == (
        "def visit_Try(self, node: ast.Try):\n"
        "    for index, handler in enumerate(node.handlers):\n"
        "        for exception_type in exception_types(handler):\n"
        "            self.visit(exception_type)\n"
    )
    import_source = (
        "def aliases(node: ast.Import, out):\n"
        "    for alias in node.names:\n"
        "        if alias.asname is not None:\n"
        "            out.append(alias.asname)\n"
    )
    assert _rewritten(import_source, ImportAliasDecoder) == (
        "def aliases(node: ast.Import, out):\n"
        "    for alias in node.names:\n"
        "        for import_alias in alias_names(alias):\n"
        "            out.append(import_alias)\n"
    )


def test_guarded_present_probe_nests_the_loop_under_the_guard() -> None:
    source = (
        "def f(child, parts):\n"
        "    if isinstance(child, ast.keyword) and child.arg is not None:\n"
        "        parts.add(child.arg)\n"
    )
    assert _rewritten(source) == (
        "def f(child, parts):\n"
        "    if isinstance(child, ast.keyword):\n"
        "        for keyword_name in CallKeyword.decoded(child).names:\n"
        "            parts.add(keyword_name)\n"
    )


def test_unproved_node_is_not_a_site() -> None:
    with pytest.raises(ValueError, match="no shape"):
        _rewritten(
            "def f(self, node):\n"
            "    if node.name is not None:\n"
            "        self.names.add(node.name)\n",
            CaptureNameDecoder,
        )


def test_loop_jump_inside_the_guard_is_handed_back() -> None:
    with pytest.raises(ValueError, match="jumps an enclosing loop"):
        _rewritten(
            "def h(node: ast.arg):\n"
            "    while True:\n"
            "        if node.annotation is not None:\n"
            "            break\n",
            DeclaredAnnotationDecoder,
        )


@pytest.mark.parametrize(
    ("statement", "expected"),
    (
        (
            "a = (*((node.returns,) if node.returns is not None else ()), x)",
            "a = (*declared_annotations(node), x)",
        ),
        (
            "a = () if node.returns is None else (node.returns,)",
            "a = declared_annotations(node)",
        ),
    ),
)
def test_splices_read_the_decoded_values(statement: str, expected: str) -> None:
    source = f"def f(node: ast.FunctionDef, x):\n    {statement}\n"
    assert _rewritten(source, DeclaredAnnotationDecoder) == (
        f"def f(node: ast.FunctionDef, x):\n    {expected}\n"
    )


@pytest.mark.parametrize(
    ("statement", "expected"),
    (
        (
            "if x and node.args.vararg is None and node.args.kwarg is None: pass",
            "if x and not variadic_parameters(node.args): pass",
        ),
        (
            "if node.args.vararg is not None or node.args.kwarg is not None: pass",
            "if variadic_parameters(node.args): pass",
        ),
        (
            "a = x or node.args.vararg is not None or node.args.kwarg is not None",
            "a = x or bool(variadic_parameters(node.args))",
        ),
        (
            "a = (*((node.args.vararg,) if node.args.vararg is not None else ()),"
            " *((node.args.kwarg,) if node.args.kwarg is not None else ()))",
            "a = (*variadic_parameters(node.args),)",
        ),
    ),
)
def test_variadic_pairs_read_the_decoded_parameters(
    statement: str, expected: str
) -> None:
    source = f"def f(node: ast.FunctionDef, x):\n    {statement}\n"
    assert _rewritten(source, VariadicParameterDecoder) == (
        f"def f(node: ast.FunctionDef, x):\n    {expected}\n"
    )


def test_lone_variadic_probe_is_handed_back() -> None:
    with pytest.raises(ValueError, match="no shape"):
        _rewritten(
            "def f(node: ast.FunctionDef):\n"
            "    if node.args.vararg is not None:\n"
            "        return 1\n",
            VariadicParameterDecoder,
        )


def test_new_file_takes_the_import_root_of_its_package(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    script = package / "tools" / "loose.py"
    sources = {
        str(package / "__init__.py"): "",
        str(package / "mod.py"): "X = 1\n",
        str(script): "Y = 2\n",
    }
    for path, source in sources.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(source)
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        sources, analysis_roots=(package,)
    )
    created = str(package / "ast_field_decoders.py")
    projected = snapshot.with_virtual_sources({created: "Z = 3\n"})
    assert {
        module.module_name
        for module in projected.parsed_modules
        if str(module.path) == created
    } == {"pkg.ast_field_decoders"}


@pytest.mark.parametrize(
    ("statement", "expected"),
    (
        (
            "a = {p.arg: p.annotation for p in (*node.args.posonlyargs,"
            " *node.args.args) if p.arg not in skip and p.annotation is not None}",
            "a = {p.arg: p.annotation for p in ParameterAnnotations.of((*node.args."
            "posonlyargs, *node.args.args)).annotated if p.arg not in skip}",
        ),
        (
            "a = any(p.arg == x and p.annotation is not None and check(p.annotation)"
            " for p in node.args.kwonlyargs)",
            "a = any(p.arg == x and check(p.annotation) for p in "
            "ParameterAnnotations.of(node.args.kwonlyargs).annotated)",
        ),
    ),
)
def test_parameter_lists_split_into_annotated_parameters(
    statement: str, expected: str
) -> None:
    rewritten = _rewritten(
        f"def f(node, skip, x):\n    {statement}\n", DeclaredAnnotationDecoder
    )
    assert ast.dump(ast.parse(rewritten)) == ast.dump(
        ast.parse(f"def f(node, skip, x):\n    {expected}\n")
    )


def test_call_before_the_probe_keeps_the_filter() -> None:
    """A call before the probe cannot hoist `ParameterAnnotations.of(...)` past it,
    so the comprehension keeps the filter and checks `declared_annotations(p)` in place."""

    rewritten = _rewritten(
        "def f(node):\n"
        "    return [p for p in node.args.kwonlyargs"
        " if seen(p) and p.annotation is not None]\n",
        DeclaredAnnotationDecoder,
    )
    assert "ParameterAnnotations" not in rewritten
    assert (
        rewritten
        == "def f(node):\n"
        "    return [p for p in node.args.kwonlyargs"
        " if seen(p) and declared_annotations(p)]\n"
    )


def test_loop_over_a_once_bound_parameter_tuple_drops_unannotated() -> None:
    source = (
        "def f(node):\n"
        "    parameters = (*node.args.posonlyargs, *node.args.kwonlyargs)\n"
        "    for parameter in parameters:\n"
        "        if parameter.annotation is None:\n"
        "            continue\n"
        "        use(parameter.arg, parameter.annotation)\n"
    )
    assert _rewritten(source, DeclaredAnnotationDecoder) == (
        "def f(node):\n"
        "    parameters = (*node.args.posonlyargs, *node.args.kwonlyargs)\n"
        "    for parameter in ParameterAnnotations.of(parameters).annotated:\n"
        "        use(parameter.arg, parameter.annotation)\n"
    )


def test_filter_after_a_parenthesized_iterable_keeps_the_parenthesis() -> None:
    source = (
        "def f(node):\n"
        "    return [\n"
        "        p.arg\n"
        "        for p in (\n"
        "            tuple(node.args.posonlyargs) + tuple(node.args.kwonlyargs)\n"
        "        )\n"
        "        if p.annotation is not None\n"
        "    ]\n"
    )
    rewritten = _rewritten(source, DeclaredAnnotationDecoder)
    assert "if p.annotation" not in rewritten
    assert "ParameterAnnotations.of(" in rewritten and ".annotated" in rewritten
    assert max(len(line) for line in rewritten.splitlines()) <= 88


def test_generated_owners_register_nothing_by_hand(tmp_path: Path) -> None:
    """The module the recipe generates passes NRA's own registration detectors,
    in a project that declares an installed registry owner it should use."""

    package = tmp_path / "pkg"
    (tmp_path / "pyproject.toml").write_text(
        '[tool.nominal-refactor-advisor]\nowner-packages = ["metaclass-registry"]\n'
    )
    sources = {
        str(package / "__init__.py"): "",
        str(package / "mod.py"): (
            "def f(k, h, a, p, fn, al, ret, imp):\n"
            "    return (k.arg is None, h.name is None, h.type is None, a.vararg is None,"
            " a.kwarg is None, p.annotation is None, fn.returns is None,"
            " al.asname is None, ret.value is None, imp.module is None)\n"
        ),
    }
    for path, source in sources.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(source)
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        sources, analysis_roots=(package,)
    )
    owners = ProjectDecoderOwners.of(snapshot, "pkg")
    assert set(owners.pending) == set(OptionalAstFieldDecoder.__registry__.values())
    Path(owners.generated_path()).write_text(owners.generated_source())

    modules = parse_python_modules(tmp_path)
    assert [
        finding.summary
        for detector in (
            HandRolledRegistrationDetector,
            ManualClassRegistrationDetector,
        )
        for finding in detector().detect(modules, DetectorConfig())
    ] == []


def _capture_name_records(tmp_path: Path, other_source: str = ""):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(
        "import ast\n\n\n"
        "def bound(handler: ast.ExceptHandler, names: list[str]) -> None:\n"
        "    if handler.name is not None:\n"
        "        names.append(handler.name)\n"
    )
    if other_source:
        (package / "other.py").write_text(other_source)
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in ForeignAbsenceProbeDetector().detect(
            modules, DetectorConfig()
        )
        if finding.detector_id == "foreign_absence_probe"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
    return codemod_plan_from_findings(findings, selector_context=snapshot).records


def test_an_unrelated_declaration_with_an_owner_name_is_refused(
    tmp_path: Path,
) -> None:
    """An owner is recognised by its declaration, not its name: a different
    `capture_names` in the package refuses the routing and names both."""

    (record,) = _capture_name_records(
        tmp_path,
        "def capture_names(handler):\n    return [handler]\n",
    )

    assert record.status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    assert f"`capture_names` at {tmp_path / 'pkg' / 'other.py'}:1 is not" in (
        record.reason
    )
    assert "decoder owner `nominal_refactor_advisor.ast_field_decoders.capture_names` (" in (
        record.reason
    )
    assert f"{Path(ast_field_decoders.__file__)}:" in record.reason


def test_without_a_same_name_declaration_the_capture_names_routing_is_executable(
    tmp_path: Path,
) -> None:
    (record,) = _capture_name_records(tmp_path)

    assert record.status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE, (
        record.reason
    )


def test_routes_fallback_ternary_and_presence_absence_probes(tmp_path: Path) -> None:
    """`alias.name if alias.asname is None else alias.asname`, `if alias.asname is None:`,
    and `any(alias.asname is None and ...)` route through `alias_names` cleanly."""

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(
        "import ast\n\n\n"
        "def bound_name(alias: ast.alias) -> str:\n"
        "    return alias.name if alias.asname is None else alias.asname\n\n\n"
        "def unaliased_names(node: ast.Import) -> list[str]:\n"
        "    found: list[str] = []\n"
        "    for alias in node.names:\n"
        "        if alias.asname is None and alias.name != 'os':\n"
        "            found.append(alias.name)\n"
        "    return found\n\n\n"
        "def has_plain_import(node: ast.Import, target: str) -> bool:\n"
        "    return any(alias.asname is None and alias.name == target for alias in node.names)\n"
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in ForeignAbsenceProbeDetector().detect(modules, DetectorConfig())
        if finding.detector_id == "foreign_absence_probe"
    )
    assert len(findings) == 3
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    assert len(plan.document.recipes) == 1
    simulation = plan.document.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    after = (package / "mod.py").read_text()
    assert "next(iter(alias_names(alias)), alias.name)" in after
    assert "if not alias_names(alias) and alias.name != 'os':" in after
    assert "any(not alias_names(alias) and alias.name == target for alias in node.names)" in after
    assert "is None" not in after


def test_refuses_presence_rewrite_when_downstream_relies_on_none_narrowing(
    tmp_path: Path,
) -> None:
    """`if alias.asname is None: ...; use(alias.asname)` with a non-terminating guard
    relies on mypy narrowing, and impure fallbacks cannot be eagerly evaluated."""

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(
        "import ast\n\n\n"
        "def compute_default() -> str:\n"
        "    return 'x'\n\n\n"
        "def narrowed(alias: ast.alias) -> str:\n"
        "    if alias.asname is None:\n"
        "        return alias.name\n"
        "    out = alias.asname.upper()\n"
        "    if out:\n"
        "        return out\n"
        "    return alias.name\n\n\n"
        "def non_terminating_narrowed(alias: ast.alias, log: list[str]) -> str:\n"
        "    if alias.asname is None:\n"
        "        log.append(alias.name)\n"
        "    else:\n"
        "        return alias.asname.upper()\n"
        "    return alias.name\n\n\n"
        "def impure_fallback(alias: ast.alias) -> str:\n"
        "    return alias.asname if alias.asname is not None else compute_default()\n"
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in ForeignAbsenceProbeDetector().detect(modules, DetectorConfig())
        if finding.detector_id == "foreign_absence_probe"
    )
    # `narrowed` rewrites cleanly via EarlyExitAbsentProbeSite; `non_terminating_narrowed`
    # and `impure_fallback` stay refused.
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    assert len(plan.document.recipes) == 1
    simulation = plan.document.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    after = (package / "mod.py").read_text()
    assert "for import_alias in alias_names(alias):" in after
    assert "if alias.asname is None:\n        log.append(alias.name)" in after
    assert "else compute_default()" in after


def test_routes_transforming_ternary_paired_variadic_and_local_binding_proofs(
    tmp_path: Path,
) -> None:
    """Transforming fallback ternaries, paired variadic if/or probes, yield-from,
    and local/match/not-isinstance AST class proofs route cleanly."""

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(
        "import ast\n"
        "from collections.abc import Iterator\n\n\n"
        "def format_return(fn: ast.FunctionDef) -> str | None:\n"
        "    return None if fn.returns is None else ast.unparse(fn.returns)\n\n\n"
        "def collect_variadic(node: ast.AST) -> set[str]:\n"
        "    bound: set[str] = set()\n"
        "    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):\n"
        "        args = node.args\n"
        "        if args.vararg is not None:\n"
        "            bound.add(args.vararg.arg)\n"
        "        if args.kwarg is not None:\n"
        "            bound.add(args.kwarg.arg)\n"
        "    elif isinstance(node, ast.Lambda):\n"
        "        args = node.args\n"
        "        if (args.vararg is not None and args.vararg.arg == 'x') or (\n"
        "            args.kwarg is not None and args.kwarg.arg == 'x'\n"
        "        ):\n"
        "            bound.add('x')\n"
        "    return bound\n\n\n"
        "def yield_param_annotations(fn: ast.FunctionDef) -> Iterator[ast.expr]:\n"
        "    arguments = fn.args\n"
        "    for argument in (\n"
        "        *arguments.posonlyargs,\n"
        "        *arguments.args,\n"
        "        *(a for a in (arguments.vararg, arguments.kwarg) if a is not None),\n"
        "    ):\n"
        "        if argument.annotation is not None:\n"
        "            yield argument.annotation\n\n\n"
        "def require_assigned_value(node: ast.AST) -> bool:\n"
        "    if not isinstance(node, ast.AnnAssign) or node.value is None:\n"
        "        return False\n"
        "    return True\n"
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in ForeignAbsenceProbeDetector().detect(modules, DetectorConfig())
        if finding.detector_id == "foreign_absence_probe"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    assert len(plan.document.recipes) == 1
    simulation = plan.document.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    after = (package / "mod.py").read_text()
    assert "next(" in after and "declared_annotations(fn)" in after
    assert "for variadic_parameter in variadic_parameters(args):" in after
    assert "any(" in after and "variadic_parameters(args)" in after
    assert "yield from declared_annotations(argument)" in after
    assert "not optional_values(node)" in after
    # Execute the rewritten module to verify runtime parity!
    ns: dict[str, object] = {}
    sys.path.insert(0, str(tmp_path))
    try:
        exec(compile(after, str(package / "mod.py"), "exec"), ns)
        fn = ast.parse("def f(a: int, *v: str, **k: bytes) -> float:\n    pass").body[0]
        lam = ast.parse("lambda *x: x").body[0].value  # type: ignore[attr-defined]
        ann_none = ast.parse("x: int").body[0]
        ann_val = ast.parse("x: int = 1").body[0]
        assert ns["format_return"](fn) == "float"  # type: ignore[operator]
        assert ns["collect_variadic"](fn) == {"v", "k"}  # type: ignore[operator]
        assert ns["collect_variadic"](lam) == {"x"}  # type: ignore[operator]
        assert [ast.unparse(e) for e in ns["yield_param_annotations"](fn)] == [  # type: ignore[operator]
            "int",
            "str",
            "bytes",
        ]
        assert ns["require_assigned_value"](ann_none) is False  # type: ignore[operator]
        assert ns["require_assigned_value"](ann_val) is True  # type: ignore[operator]
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in [m for m in sys.modules if m == "pkg_shapes" or m.startswith("pkg_shapes.")]:
            sys.modules.pop(mod_name, None)


def _write_declared_decoder_pkg(tmp_path: Path, mod_source: str) -> tuple[Path, list]:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    dec_src = Path(ast_field_decoders.__file__).read_text()
    (package / "ast_field_decoders.py").write_text(dec_src)
    mod_path = package / "mod.py"
    mod_path.write_text(mod_source)
    modules = parse_python_modules(tmp_path)
    findings = [
        finding
        for finding in ForeignAbsenceProbeDetector().detect(modules, DetectorConfig())
        if finding.detector_id == "foreign_absence_probe"
        and Path(finding.primary_evidence.file_path) == mod_path
    ]
    return mod_path, findings


def test_same_function_different_decoders_compose_without_conflict(
    tmp_path: Path,
) -> None:
    """Two distinct AST decoders in one function (`native_call.py:588,597` shape)
    must not falsely collide on a function-level action key."""

    mod_path, findings = _write_declared_decoder_pkg(
        tmp_path,
        "import ast\n\n\n"
        "def has_header_inputs(closure: ast.FunctionDef) -> bool:\n"
        "    return bool(\n"
        "        closure.returns is not None\n"
        "        or closure.args.vararg is not None\n"
        "        or closure.args.kwarg is not None\n"
        "    )\n",
    )
    assert len(findings) == 2
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path), findings
    )
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    assert plan.candidate_count == 2
    assert all(
        record.status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
        for record in plan.records
    )
    simulation = plan.document.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    after = mod_path.read_text()
    assert "declared_annotations(closure)" in after
    assert "variadic_parameters(closure.args)" in after
    assert "is not None" not in after


def test_unroutable_probe_does_not_piggyback_on_routable_sibling_in_same_function(
    tmp_path: Path,
) -> None:
    """An unroutable probe (`semantic_descent.py:5736` shape) must stay refused
    even when another probe of the same decoder (`semantic_descent.py:5745`) in
    the same function is routable."""

    mod_path, findings = _write_declared_decoder_pkg(
        tmp_path,
        "import ast\n\n\n"
        "def extract(node: ast.FunctionDef, argument: ast.arg, out: dict[str, ast.expr]) -> tuple[ast.expr, ...]:\n"
        "    if argument.annotation is None:\n"
        "        out.pop(argument.arg, None)\n"
        "    else:\n"
        "        out[argument.arg] = argument.annotation\n"
        "    return () if node.returns is None else (node.returns,)\n",
    )
    assert len(findings) == 2
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path), findings
    )
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    assert plan.candidate_count == 1
    by_line = {record.finding.primary_evidence.line: record for record in plan.records}
    assert by_line[5].status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    assert "argument.annotation is None" in by_line[5].reason
    assert by_line[9].status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    simulation = plan.document.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    after = mod_path.read_text()
    assert "if argument.annotation is None:" in after
    assert "return declared_annotations(node)" in after


def test_same_function_same_decoder_both_routable_compose_cleanly(
    tmp_path: Path,
) -> None:
    """Two routable probes of the same decoder in one function compose in one batch."""

    mod_path, findings = _write_declared_decoder_pkg(
        tmp_path,
        "import ast\n\n\n"
        "def both(node: ast.FunctionDef, argument: ast.arg) -> tuple[tuple[ast.expr, ...], tuple[ast.expr, ...]]:\n"
        "    a = () if argument.annotation is None else (argument.annotation,)\n"
        "    b = () if node.returns is None else (node.returns,)\n"
        "    return a, b\n",
    )
    assert len(findings) == 2
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path), findings
    )
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    assert plan.candidate_count == 2
    simulation = plan.document.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    after = mod_path.read_text()
    assert "a = declared_annotations(argument)" in after
    assert "b = declared_annotations(node)" in after
    assert "is None" not in after


def test_routes_comprehension_filter_disjunction_prefix_early_exit_guarded_ternary_and_terminal_elif(
    tmp_path: Path,
) -> None:
    """Exercise all new sound shapes end-to-end with runtime parity:
    - SingleNodeComprehensionFilterProbeSite (with pre/post guards)
    - DisjunctiveAbsentProbeSite (`V.f is None or P(V.f)` inside `if` and `all(...)`)
    - EarlyExitAbsentProbeSite with prefix `or` (`if not isinstance(...) or V.f is None:`)
    - FallbackTernaryProbeSite with prefix `and` (`EXPR if guard and V.f is not None else fallback`)
    - Terminal `elif` in IfPresentProbeSite + PresenceTestProbeSite `and`-narrowing fix
    - Multi-line `if (...)` header wrapping in IfPresentProbeSite
    """

    package = tmp_path / "pkg_shapes"
    package.mkdir()
    (package / "__init__.py").write_text("")
    mod_path = package / "mod.py"
    mod_path.write_text(
        "from __future__ import annotations\n"
        "import ast\n"
        "from collections.abc import Iterator\n\n\n"
        "def aliased_for(stmt: ast.Import, target: str) -> tuple[str, ...]:\n"
        "    return tuple(\n"
        "        alias.asname\n"
        "        for alias in stmt.names\n"
        "        if alias.asname is not None and alias.name == target\n"
        "    )\n\n\n"
        "def is_none_return(stmt: ast.stmt) -> bool:\n"
        "    return all(\n"
        "        isinstance(s, ast.Return)\n"
        "        and (\n"
        "            s.value is None\n"
        "            or (\n"
        "                isinstance(s.value, ast.Constant)\n"
        "                and s.value.value is None\n"
        "            )\n"
        "        )\n"
        "        for s in (stmt,)\n"
        "    )\n\n\n"
        "def import_from_module(stmt: ast.stmt) -> tuple[str, int] | None:\n"
        "    if not isinstance(stmt, ast.ImportFrom) or stmt.module is None:\n"
        "        return None\n"
        "    return (stmt.module, stmt.level)\n\n\n"
        "def maybe_return(fn: ast.FunctionDef, include: bool) -> str:\n"
        "    return (\n"
        "        ast.unparse(fn.returns)\n"
        "        if include and fn.returns is not None\n"
        "        else 'None'\n"
        "    )\n\n\n"
        "def import_bindings(stmt: ast.Import, target: str) -> Iterator[str]:\n"
        "    for alias in stmt.names:\n"
        "        if alias.asname is None and alias.name == target:\n"
        "            yield alias.name\n"
        "        elif alias.asname is not None and alias.name == target:\n"
        "            yield alias.asname\n\n\n"
        "def matching_kw_names(call: ast.Call) -> list[str]:\n"
        "    out: list[str] = []\n"
        "    for keyword in call.keywords:\n"
        "        value = keyword.value\n"
        "        if (\n"
        "            keyword.arg is not None\n"
        "            and isinstance(value, ast.Attribute)\n"
        "            and value.attr == keyword.arg\n"
        "            and isinstance(value.value, ast.Name)\n"
        "        ):\n"
        "            out.append(keyword.arg)\n"
        "    return out\n"
    )
    modules = parse_python_modules(tmp_path)
    findings = [
        finding
        for finding in ForeignAbsenceProbeDetector().detect(modules, DetectorConfig())
        if finding.detector_id == "foreign_absence_probe"
        and Path(finding.primary_evidence.file_path) == mod_path
    ]
    assert len(findings) >= 6
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    assert len(plan.document.recipes) == 1, [
        (r.finding.primary_evidence.line, r.status, r.reason) for r in plan.records
    ]
    simulation = plan.document.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    after = mod_path.read_text()
    assert "is None" not in after or after.count("is None") == 1  # Constant.value is None stays
    assert "s.value is None" not in after
    assert "stmt.module is None" not in after
    assert "fn.returns is not None" not in after
    assert "alias.asname is None" not in after
    assert "alias.asname is not None" not in after
    assert "keyword.arg is not None" not in after
    assert "for import_alias in alias_names(alias)" in after
    assert "for optional_value in optional_values(s)" in after
    assert "for import_module in import_modules(stmt):" in after
    assert "declared_annotations(fn) if include else ()" in after
    assert max(len(line) for line in after.splitlines()) <= 88

    ns: dict[str, object] = {}
    sys.path.insert(0, str(tmp_path))
    try:
        exec(compile(after, str(mod_path), "exec"), ns)
        imp = ast.parse("import os as operating_system, sys").body[0]
        imp_from = ast.parse("from pkg.mod import x").body[0]
        rel_from = ast.parse("from . import x").body[0]
        fn = ast.parse("def g() -> int: return 1").body[0]
        fn_bare = ast.parse("def g(): return").body[0]
        call = ast.parse("f(a=x.a, b=x.c, **kw)").body[0].value  # type: ignore[attr-defined]
        assert ns["aliased_for"](imp, "os") == ("operating_system",)  # type: ignore[operator]
        assert ns["aliased_for"](imp, "sys") == ()  # type: ignore[operator]
        assert ns["is_none_return"](ast.parse("return").body[0]) is True  # type: ignore[operator]
        assert ns["is_none_return"](ast.parse("return None").body[0]) is True  # type: ignore[operator]
        assert ns["is_none_return"](ast.parse("return 1").body[0]) is False  # type: ignore[operator]
        assert ns["import_from_module"](imp_from) == ("pkg.mod", 0)  # type: ignore[operator]
        assert ns["import_from_module"](rel_from) is None  # type: ignore[operator]
        assert ns["import_from_module"](imp) is None  # type: ignore[operator]
        assert ns["maybe_return"](fn, True) == "int"  # type: ignore[operator]
        assert ns["maybe_return"](fn, False) == "None"  # type: ignore[operator]
        assert ns["maybe_return"](fn_bare, True) == "None"  # type: ignore[operator]
        assert list(ns["import_bindings"](imp, "sys")) == ["sys"]  # type: ignore[operator]
        assert list(ns["import_bindings"](imp, "os")) == ["operating_system"]  # type: ignore[operator]
        assert ns["matching_kw_names"](call) == ["a"]  # type: ignore[operator]
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in [m for m in sys.modules if m == "pkg_shapes" or m.startswith("pkg_shapes.")]:
            sys.modules.pop(mod_name, None)


def test_proves_ast_types_through_assert_type_alias_match_subpattern_comprehension_and_typed_attr(
    tmp_path: Path,
) -> None:
    """Exercise all new AST type proofs in OptionalAstFieldAuthority:
    - Module-level TypeAlias (`FunctionNode`) + `for fn in (function, *reversed(enclosing)):`
    - `assert isinstance(node, ast.ExceptHandler)`
    - `match` keyword subpattern `args=arguments` + `filter(None, (arguments.vararg, arguments.kwarg))`
    - Earlier comprehension `if isinstance(node, ast.Import)` clause
    - Typed class attribute collection (`self.fixed_keywords: tuple[ast.keyword, ...]`)
    """

    package = tmp_path / "pkg_proofs"
    package.mkdir()
    (package / "__init__.py").write_text("")
    mod_path = package / "mod.py"
    mod_path.write_text(
        "from __future__ import annotations\n"
        "import ast\n"
        "from dataclasses import dataclass\n\n"
        "FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef\n\n\n"
        "def has_variadic_named(\n"
        "    function: FunctionNode, enclosing: tuple[FunctionNode, ...], name: str\n"
        ") -> bool:\n"
        "    for fn in (function, *reversed(enclosing)):\n"
        "        if (fn.args.vararg is not None and fn.args.vararg.arg == name) or (\n"
        "            fn.args.kwarg is not None and fn.args.kwarg.arg == name\n"
        "        ):\n"
        "            return True\n"
        "    return False\n\n\n"
        "def assert_handler_name(node: ast.AST) -> list[str]:\n"
        "    out: list[str] = []\n"
        "    assert isinstance(node, ast.ExceptHandler)\n"
        "    if node.name is not None:\n"
        "        out.append(node.name)\n"
        "    return out\n\n\n"
        "def match_arg_annotations(node: ast.stmt) -> list[ast.expr]:\n"
        "    out: list[ast.expr] = []\n"
        "    match node:\n"
        "        case ast.FunctionDef(args=arguments) | ast.AsyncFunctionDef(args=arguments):\n"
        "            for argument in (\n"
        "                *arguments.posonlyargs,\n"
        "                *arguments.args,\n"
        "                *arguments.kwonlyargs,\n"
        "                *filter(None, (arguments.vararg, arguments.kwarg)),\n"
        "            ):\n"
        "                if argument.annotation is not None:\n"
        "                    out.append(argument.annotation)\n"
        "    return out\n\n\n"
        "def aliased_imports(stmts: list[ast.stmt], target: str) -> list[ast.alias]:\n"
        "    return [\n"
        "        alias\n"
        "        for node in stmts\n"
        "        if isinstance(node, ast.Import)\n"
        "        for alias in node.names\n"
        "        if alias.asname is not None and alias.name == target\n"
        "    ]\n\n\n"
        "@dataclass(frozen=True)\n"
        "class FixedCall:\n"
        "    fixed_keywords: tuple[ast.keyword, ...]\n\n"
        "    def named_keys(self) -> list[str]:\n"
        "        return [kw.arg for kw in self.fixed_keywords if kw.arg is not None]\n"
    )
    modules = parse_python_modules(tmp_path)
    findings = [
        finding
        for finding in ForeignAbsenceProbeDetector().detect(modules, DetectorConfig())
        if finding.detector_id == "foreign_absence_probe"
        and Path(finding.primary_evidence.file_path) == mod_path
    ]
    assert len(findings) == 5
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    assert len(plan.document.recipes) == 1, [
        (r.finding.primary_evidence.line, r.status, r.reason) for r in plan.records
    ]
    simulation = plan.document.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    after = mod_path.read_text()
    assert "is not None" not in after
    assert "variadic_parameters(fn.args)" in after
    assert "capture_names(node)" in after
    assert "declared_annotations(argument)" in after
    assert "alias_names(alias)" in after
    assert "CallKeywords.of(self.fixed_keywords).named" in after

    ns: dict[str, object] = {}
    sys.path.insert(0, str(tmp_path))
    try:
        exec(compile(after, str(mod_path), "exec"), ns)
        fn1 = ast.parse("def f(a: int, *v: str): pass").body[0]
        fn2 = ast.parse("async def g(**k: bytes): pass").body[0]
        handler = ast.parse("try:\n    pass\nexcept Exception as err:\n    pass").body[0].handlers[0]  # type: ignore[attr-defined]
        imp = ast.parse("import os as o, sys").body[0]
        call = ast.parse("f(a=1, **d)").body[0].value  # type: ignore[attr-defined]
        assert ns["has_variadic_named"](fn1, (fn2,), "k") is True  # type: ignore[operator]
        assert ns["has_variadic_named"](fn1, (), "k") is False  # type: ignore[operator]
        assert ns["assert_handler_name"](handler) == ["err"]  # type: ignore[operator]
        assert [ast.unparse(e) for e in ns["match_arg_annotations"](fn1)] == ["int", "str"]  # type: ignore[operator]
        assert [a.asname for a in ns["aliased_imports"]([imp], "os")] == ["o"]  # type: ignore[operator]
        assert ns["FixedCall"](tuple(call.keywords)).named_keys() == ["a"]  # type: ignore[operator]
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in [m for m in sys.modules if m == "pkg_proofs" or m.startswith("pkg_proofs.")]:
            sys.modules.pop(mod_name, None)

