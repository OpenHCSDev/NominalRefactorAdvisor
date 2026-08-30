from __future__ import annotations

import ast
from collections import Counter
import json
import signal
import sys
import time
from pathlib import Path

import pytest

from nominal_refactor_advisor import cli as cli_module
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.analysis_cache import GlobalModuleContextSignature
from nominal_refactor_advisor.detectors import _runtime as runtime_detectors
from nominal_refactor_advisor.detectors import _systemic as systemic_detectors
from nominal_refactor_advisor.detectors._base import (
    CrossModuleCollectorCandidateDetector,
    DetectorConfig,
)
from nominal_refactor_advisor.deadline import (
    ScanDeadline,
    ScanDeadlineExceeded,
    enforce_scan_deadline,
)


def _write_module(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_cross_module_preparation_reuses_exact_candidate_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyCandidateSnapshotProbe(CrossModuleCollectorCandidateDetector[str]):
        candidate_collector = staticmethod(lambda modules: ())

    candidate_calls = 0
    finding_calls = 0
    candidates = ("first", "second")

    def counted_candidates(self, modules, config):
        nonlocal candidate_calls
        del self, modules, config
        candidate_calls += 1
        return candidates

    def counted_findings(self, prepared_candidates, config):
        nonlocal finding_calls
        del self, config
        finding_calls += 1
        assert tuple(prepared_candidates) == candidates
        return []

    # Exercise the base full-AST candidate snapshot contract independently of
    # the production registry. All production contextual-global detectors now
    # prepare from persisted compact projection families.
    detector_type = LegacyCandidateSnapshotProbe
    monkeypatch.setattr(detector_type, "_candidate_items", counted_candidates)
    monkeypatch.setattr(detector_type, "_findings_for_candidates", counted_findings)

    prepared = detector_type().prepare_analysis((), DetectorConfig())
    assert candidate_calls == 1
    assert prepared.findings() == []
    assert candidate_calls == 1
    assert finding_calls == 1


def test_grouped_shape_preparation_reuses_exact_shape_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_calls = 0
    finding_calls = 0
    shapes = ("alpha", "beta")

    def counted_shapes(self, modules, config):
        nonlocal collection_calls
        del self, modules, config
        collection_calls += 1
        return list(shapes)

    def group_key(self, shape):
        del self
        return shape

    def counted_findings(self, prepared_shapes, config):
        nonlocal finding_calls
        del self, config
        finding_calls += 1
        assert tuple(prepared_shapes) == shapes
        return []

    detector_type = runtime_detectors.RepeatedExportDictDetector
    monkeypatch.setattr(detector_type, "_collect_shapes", counted_shapes)
    monkeypatch.setattr(detector_type, "_group_key", group_key)
    monkeypatch.setattr(detector_type, "_findings_for_shapes", counted_findings)

    prepared = detector_type().prepare_analysis((), DetectorConfig())
    assert collection_calls == 1
    assert prepared.findings() == []
    assert collection_calls == 1
    assert finding_calls == 1


def test_private_reference_module_index_matches_independent_ast_projections(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/sample.py",
        '''
"""module docs"""

DECORATOR_NAME = "decorator-literal"


def decorate(function):
    return function


@decorate
def _outer(value: "argument-literal") -> "return-literal":
    """function docs"""
    local = value.member

    if isinstance(value, Owner):
        local = value._method

    def nested():
        if isinstance(value, Nested):
            return value.field
        return "nested-literal", local

    class Nested:
        field = "class-literal"

    return nested(), Nested


class Owner:
    @decorate
    def _method(self, row):
        return row.member, "_method"
''',
    )
    module = tuple(parse_python_modules(tmp_path))[0]
    runtime_detectors._private_reference_module_index.cache_clear()

    def symbol_counts(root: ast.AST) -> Counter[str]:
        counts: Counter[str] = Counter()
        for node in runtime_detectors._walk_nodes(root):
            if isinstance(node, ast.Name):
                counts[node.id] += 1
            elif isinstance(node, ast.Attribute):
                counts[node.attr] += 1
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                counts[node.value] += 1
        return counts

    index = runtime_detectors.PrivateReferenceModuleIndex.from_module(module)
    assert index.total_counts == symbol_counts(module.module)

    indexed_functions = {
        id(indexed_function.function): indexed_function
        for indexed_function in index.functions
    }
    for _, function in runtime_detectors.SurfaceFunctionIndex.from_module(
        module.module
    ).functions:
        indexed_function = indexed_functions[id(function)]
        assert index.function_counts_by_id[id(function)] == symbol_counts(function)
        assert (
            indexed_function.symbol_references
            == runtime_detectors._function_symbol_references(function)
        )
        assert indexed_function.body_digest == runtime_detectors._stable_text_digest(
            f"{module.semantic_hash}\0{indexed_function.qualname}"
        )

    assert index.class_surface_members_by_type_name == {
        "Nested": ("field",),
        "Owner": ("_method",),
        "pkg.sample.Nested": ("field",),
        "pkg.sample.Owner": ("_method",),
    }
    assert index.role_guarded_accesses == (
        runtime_detectors._compact_role_guarded_access_facts_for_module(module)
    )
    legacy_named_functions = systemic_detectors._iter_named_functions(module)
    assert len(index.named_functions) == len(legacy_named_functions)
    for indexed_function, (qualname, function) in zip(
        index.named_functions,
        legacy_named_functions,
        strict=True,
    ):
        assert indexed_function.qualname == qualname
        assert indexed_function.function is function
        assert indexed_function.isinstance_calls == tuple(
            node
            for node in systemic_detectors._walk_nodes(function)
            if isinstance(node, ast.Call)
            and len(node.args) == 2
            and not node.keywords
            and systemic_detectors._ast_terminal_name(node.func) == "isinstance"
        )
    legacy_reference_sites = systemic_detectors._local_symbol_reference_sites((module,))
    assert index.reference_summaries_by_symbol == tuple(
        (
            symbol,
            len(sites),
            tuple(sorted({site.symbol for site in sites})),
        )
        for symbol, sites in legacy_reference_sites.items()
    )
    declarations = systemic_detectors._public_top_level_declarations(module)
    public_names = frozenset(declarations)
    assert index.public_declaration_reference_names_by_name == {
        name: tuple(
            sorted(
                systemic_detectors._public_declaration_reference_names(
                    node,
                    public_names,
                )
            )
        )
        for name, node in sorted(declarations.items())
    }


def test_private_reference_compact_families_share_one_module_projection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/sample.py",
        "\ndef _render(value):\n"
        "    return str(value)\n"
        "\n"
        "class Renderer:\n"
        "    def render(self, value):\n"
        "        return _render(value)\n",
    )
    modules = tuple(parse_python_modules(tmp_path))
    runtime_detectors._private_reference_module_index.cache_clear()
    private_projections = tuple(
        runtime_detectors.CompactPrivateReferenceModuleProjectionFamily.collect(
            modules[0]
        )
    )
    first_cache_state = runtime_detectors._private_reference_module_index.cache_info()
    role_surfaces = dict(
        runtime_detectors.CompactRoleGuardedSurfaceModuleProjectionFamily.collect(
            modules[0]
        )[0].class_surface_members_by_type_name
    )
    second_cache_state = runtime_detectors._private_reference_module_index.cache_info()

    assert role_surfaces == {
        "Renderer": ("render",),
        "pkg.sample.Renderer": ("render",),
    }
    assert private_projections[0].functions
    assert first_cache_state.misses == len(modules)
    assert second_cache_state.misses == first_cache_state.misses


def test_process_cli_hard_exits_after_publishing_deadline_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    deadline = ScanDeadline.start(1.0)
    deadline.stage = "test_projection"
    error = ScanDeadlineExceeded(deadline)
    exit_codes: list[int] = []
    terminated_children: list[str] = []

    class ActiveChild:
        def __init__(self, name: str) -> None:
            self.name = name

        def terminate(self) -> None:
            terminated_children.append(self.name)

    def raise_deadline() -> int:
        raise error

    def hard_exit(exit_code: int) -> None:
        exit_codes.append(exit_code)
        raise SystemExit(exit_code)

    monkeypatch.setattr(cli_module, "_main_without_deadline", raise_deadline)
    monkeypatch.setattr(cli_module.os, "_exit", hard_exit)
    monkeypatch.setattr(
        cli_module.multiprocessing,
        "active_children",
        lambda: (ActiveChild("alpha"), ActiveChild("beta")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["nominal-refactor-advisor", "--json", "sample.py"],
    )

    with pytest.raises(SystemExit, match="124"):
        cli_module.process_main()

    assert exit_codes == [124]
    assert terminated_children == ["alpha", "beta"]
    assert json.loads(capsys.readouterr().out)["scan_status"] == {
        "complete": False,
        "deadline_exceeded": True,
        "stage": "test_projection",
        "budget_seconds": 1.0,
        "elapsed_seconds": pytest.approx(error.elapsed_seconds, abs=0.001),
    }


@pytest.mark.skipif(
    not hasattr(signal, "setitimer"),
    reason="hard wall-clock signals are unavailable on this platform",
)
def test_hard_deadline_terminates_at_signal_boundary() -> None:
    deadline = ScanDeadline.start(0.01)
    deadline.stage = "process_pool_wait"
    observed_errors: list[ScanDeadlineExceeded] = []

    def terminate(error: ScanDeadlineExceeded) -> None:
        observed_errors.append(error)
        raise SystemExit(124)

    with pytest.raises(SystemExit, match="124"):
        with enforce_scan_deadline(deadline, hard_timeout=terminate):
            time.sleep(1.0)

    assert len(observed_errors) == 1
    assert observed_errors[0].stage == "process_pool_wait"


def test_repository_semantic_signature_changes_for_contextual_source_edit(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/sample.py", "\nVALUE = 'before'\n")
    before_modules = tuple(parse_python_modules(tmp_path))
    before = GlobalModuleContextSignature.from_modules(before_modules).cache_token

    _write_module(tmp_path, "pkg/sample.py", "\nVALUE = 'after'\n")
    after_modules = tuple(parse_python_modules(tmp_path))
    after = GlobalModuleContextSignature.from_modules(after_modules).cache_token

    assert after != before


def test_empty_derived_contract_projection_is_not_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "pkg/sample.py",
        "\ndef _helper(value):\n"
        "    return str(value)\n"
        "\ndef render(value):\n"
        "    return _helper(value)\n",
    )
    modules = tuple(parse_python_modules(tmp_path))
    private_projections = tuple(
        runtime_detectors.CompactPrivateReferenceModuleProjectionFamily.collect(
            modules[0]
        )
    )
    assert private_projections[0].derived_candidate_collector_contract_names == ()

    monkeypatch.setattr(
        runtime_detectors.DERIVED_CANDIDATE_COLLECTOR_CONTRACTS,
        "names",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("empty cached contract projection was recomputed")
        ),
    )

    runtime_detectors._compact_unreferenced_private_function_candidates(
        private_projections,
        DetectorConfig(),
    )
    runtime_detectors._compact_private_helper_semantic_cluster_candidates(
        private_projections,
        DetectorConfig(),
    )
