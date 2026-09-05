"""Native base replacement must follow declarations, not familiar class names."""

import ast
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor import native_declarations
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DeriveCandidateCollectorOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.detectors._base import (
    CrossModuleCandidateDetector,
    CrossModuleCollectorCandidateDetector,
)


def test_batched_collectors_inspect_each_native_base_once(
    tmp_path: Path,
    native_collector_module: ParsedModule,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_names = tuple(f"Owner{index}" for index in range(20))
    source = (
        "from nominal_refactor_advisor.detectors._base import CrossModuleCandidateDetector\n"
        "def collect(modules): return tuple(modules)\n"
        + "".join(
            f"class {name}(CrossModuleCandidateDetector[int]):\n"
            "    def _candidate_items(self, modules, config): return collect(modules)\n"
            for name in owner_names
        )
        + "print(("
        + ",".join(f"{name}()._candidate_items([1], None)" for name in owner_names)
        + "))\n"
    )
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    module = ParsedModule(path, "probe", False, ast.parse(source), source)
    snapshot = CodemodSourceSnapshot.from_modules((module, native_collector_module))
    recipe = RefactorRecipe(
        "derive-twenty-collectors",
        operations=tuple(
            DeriveCandidateCollectorOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(),
                    qualname=f"{name}._candidate_items",
                )
            )
            for name in owner_names
        ),
    )
    native_calls = []
    getsource = native_declarations.inspect.getsource

    def counted_source(declaration):
        if (
            declaration is CrossModuleCandidateDetector
            or declaration is CrossModuleCollectorCandidateDetector
        ):
            native_calls.append(declaration)
        return getsource(declaration)

    monkeypatch.setattr(native_declarations.inspect, "getsource", counted_source)
    simulation = recipe.simulate(snapshot)
    assert simulation.is_clean
    # Warm projections may already exist; no base may be inspected twice.
    assert len(native_calls) <= 2
    assert len(set(native_calls)) == len(native_calls)
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before


@pytest.mark.parametrize(
    "competing_base",
    (
        "ConfiguredCrossModuleCollectorCandidateDetector",
        "OtherCollector",
    ),
)
def test_competing_collector_base_does_not_change_forwarded_arguments(
    tmp_path: Path,
    native_collector_module: ParsedModule,
    competing_base: str,
) -> None:
    source = (
        "from nominal_refactor_advisor.detectors._base import CrossModuleCandidateDetector\n"
        f"from nominal_refactor_advisor.detectors._base import ConfiguredCrossModuleCollectorCandidateDetector as {competing_base}\n"
        "def collect(modules): return ('collected',)\n"
        f"class Owner({competing_base}[int], CrossModuleCandidateDetector[int]):\n"
        "    def _candidate_items(self, modules, config): return collect(modules)\n"
        "print(Owner()._candidate_items([], None))\n"
    )
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8", newline="")
    assert (
        subprocess.check_output([sys.executable, str(path)], text=True).strip()
        == "('collected',)"
    )
    module = ParsedModule(path, "probe", False, ast.parse(source), source)
    snapshot = CodemodSourceSnapshot.from_modules((module, native_collector_module))
    plan = CodemodPlanSequence.from_operations(
        (
            DeriveCandidateCollectorOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Owner._candidate_items"
                )
            ),
        )
    )
    with pytest.raises(ValueError, match="competing native collector"):
        plan.simulate(snapshot)
    assert path.read_bytes() == source.encode()


@pytest.mark.parametrize(
    "declaration",
    (
        CrossModuleCandidateDetector,
        CrossModuleCollectorCandidateDetector,
    ),
)
def test_canonical_name_with_modified_source_is_not_native_authority(
    tmp_path: Path,
    native_collector_module: ParsedModule,
    declaration: type,
) -> None:
    native_ast = ast.parse(native_collector_module.source)
    native_node = next(
        node
        for node in native_ast.body
        if isinstance(node, ast.ClassDef) and node.name == declaration.__name__
    )
    native_node.body.append(ast.parse("marker = 'different implementation'").body[0])
    modified_source = ast.unparse(native_ast)
    modified_module = replace(
        native_collector_module,
        source=modified_source,
        module=ast.parse(modified_source),
    )
    source = (
        "from nominal_refactor_advisor.detectors._base import CrossModuleCandidateDetector\n"
        "def collect(modules): return ('collected',)\n"
        "class Owner(CrossModuleCandidateDetector[int]):\n"
        "    def _candidate_items(self, modules, config): return collect(modules)\n"
    )
    path = tmp_path / "probe.py"
    module = ParsedModule(path, "probe", False, ast.parse(source), source)
    snapshot = CodemodSourceSnapshot.from_modules((module, modified_module))
    plan = CodemodPlanSequence.from_operations(
        (
            DeriveCandidateCollectorOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Owner._candidate_items"
                )
            ),
        )
    )
    with pytest.raises(ValueError, match="Source does not match native declaration"):
        plan.simulate(snapshot)


@pytest.mark.parametrize(
    "replacement_member",
    (
        "marker = 'replacement'",
        "def marker(self): return 'replacement'",
        "def __init__(self): self.marker = 'replacement'",
    ),
)
def test_unrelated_same_name_base_cannot_change_inherited_behaviour(
    tmp_path: Path, replacement_member: str
) -> None:
    source = (
        "from typing import Generic, TypeVar\n"
        "T = TypeVar('T')\n"
        "class CrossModuleCandidateDetector(Generic[T]):\n    marker = 'original'\n"
        "class CrossModuleCollectorCandidateDetector(CrossModuleCandidateDetector[T]):\n"
        f"    {replacement_member}\n"
        "    def _candidate_items(self, modules, config): return self.candidate_collector(modules)\n"
        "def collect(modules): return ('collected',)\n"
        "class Owner(CrossModuleCandidateDetector[int]):\n"
        "    def _candidate_items(self, modules, config): return collect(modules)\n"
        "owner = Owner()\n"
        "print(owner.marker() if callable(owner.marker) else owner.marker, owner._candidate_items([], None))\n"
    )
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    assert before.strip() == "original ('collected',)"
    module = ParsedModule(path, "probe", False, ast.parse(source), source)
    plan = CodemodPlanSequence.from_operations(
        (
            DeriveCandidateCollectorOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Owner._candidate_items"
                )
            ),
        )
    )
    with pytest.raises(ValueError, match="native|authority"):
        plan.simulate(CodemodSourceSnapshot.from_modules((module,)))
    assert path.read_bytes() == source.encode()


def test_registered_collector_base_preserves_native_execution(
    tmp_path: Path, native_collector_module: ParsedModule
) -> None:
    source = (
        "from nominal_refactor_advisor.detectors._base import CrossModuleCandidateDetector\n"
        "def collect(modules): return ('collected',)\n"
        "class Owner(CrossModuleCandidateDetector[int]):\n"
        "    def _candidate_items(self, modules, config): return collect(modules)\n"
        "print(issubclass(Owner, CrossModuleCandidateDetector), Owner()._candidate_items([], None))\n"
    )
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    module = ParsedModule(path, "probe", False, ast.parse(source), source)
    snapshot = CodemodSourceSnapshot.from_modules((module, native_collector_module))
    plan = CodemodPlanSequence.from_operations(
        (
            DeriveCandidateCollectorOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Owner._candidate_items"
                )
            ),
        )
    )
    simulation = plan.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
