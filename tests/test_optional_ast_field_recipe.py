from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor import ast_field_decoders, detectors
from nominal_refactor_advisor.ast_tools import SourceModule, parse_python_modules
from nominal_refactor_advisor.codemod_optional_ast_field import (
    KeywordNameDecoder,
    CaptureNameDecoder,
    DeclaredAnnotationDecoder,
    OptionalAstFieldAuthority,
    OptionalAstFieldDecoder,
    ProjectDecoderOwners,
    VariadicParameterDecoder,
)
from nominal_refactor_advisor.codemod import codemod_plan_from_findings
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_semantics import FindingRecipeSynthesisStatus
from nominal_refactor_advisor.codemod_source_edits import SourceTextGeometry
from nominal_refactor_advisor.detectors import DetectorConfig


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
    with pytest.raises(ValueError, match="no shape"):
        _rewritten(
            "def f(node):\n"
            "    return [p for p in node.args.kwonlyargs"
            " if seen(p) and p.annotation is not None]\n",
            DeclaredAnnotationDecoder,
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
            "def f(k, h, a, p, fn):\n"
            "    return (k.arg is None, h.name is None, a.vararg is None,"
            " a.kwarg is None, p.annotation is None, fn.returns is None)\n"
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
            detectors.HandRolledRegistrationDetector,
            detectors.ManualClassRegistrationDetector,
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
        for finding in detectors.ForeignAbsenceProbeDetector().detect(
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
