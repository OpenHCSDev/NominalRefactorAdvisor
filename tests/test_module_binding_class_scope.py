"""Module moves consume the existing ordered class/function lookup decisions."""
import ast
from collections.abc import Callable
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from nominal_refactor_advisor.codemod import CodemodPlanRoot, CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_preflight import CodemodOperationPreflightError
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex


type CapturedModule = Callable[[str], tuple[Path, CodemodSourceSnapshot, ModuleNodeIndex]]


@pytest.fixture
def captured_module(tmp_path: Path) -> CapturedModule:
    def capture(body: str) -> tuple[Path, CodemodSourceSnapshot, ModuleNodeIndex]:
        path = tmp_path / "pkg/presentation.py"
        sources = {
            str(tmp_path / "pkg/__init__.py"): "",
            str(tmp_path / "pkg/relationships.py"): (
                "class RelationshipCatalogSource:\n    pass\n\n"
                "class RelationshipStore:\n    pass\n"
            ),
            str(path): "from pkg import relationships\n" + textwrap.dedent(body),
        }
        snapshot = CodemodSourceSnapshot.from_source_mapping(sources, analysis_roots=(tmp_path,))
        return path, snapshot, snapshot.parsed_module_for_source_path(str(path)).fact(ModuleNodeIndex)
    return capture


@pytest.mark.parametrize("body,loads", (
    ("class WireRevision:\n    relationships: int\n    value = lambda self: relationships.RelationshipStore\n", 1),
    ("class WireRevision:\n    relationships: int = 3\n    value = lambda self: relationships.RelationshipStore\n", 1),
    ("class WireRevision:\n    relationships: int\n    value = relationships.RelationshipStore\n", 1),
    ("class WireRevision:\n    value = relationships.RelationshipStore\n    relationships = 3\n", 1),
    ("class WireRevision:\n    relationships = object()\n    value = relationships.attribute\n", 0),
    ("class WireRevision:\n    relationships = 3\n    def value(self):\n        return relationships.RelationshipStore\n", 1),
    ("class WireRevision:\n    relationships = 3\n    value = [relationships.RelationshipStore for _ in (1,)]\n", 1),
    ("class WireRevision:\n    relationships = (1,)\n    value = [item for item in relationships]\n", 0),
    ("class WireRevision:\n    relationships = relationships.RelationshipStore\n", 1),
    ("def outer(relationships):\n    class WireRevision:\n        relationships: int\n        value = lambda self: relationships.RelationshipStore\n    return WireRevision\n", 0),
))
def test_class_fields_do_not_rebind_module_and_reads_keep_python_lookup(
    captured_module: CapturedModule, body: str, loads: int,
) -> None:
    _, _, index = captured_module(body)
    assert index.module_scope_bindings("relationships") == 1
    assert index.module_scope_loads("relationships") == loads


@pytest.mark.parametrize("body", (
    "relationships = object()\n",
    "class WireRevision:\n    global relationships\n    relationships = object()\n",
    "def change():\n    global relationships\n    relationships = object()\n",
))
def test_true_module_rebindings_remain_counted(captured_module: CapturedModule, body: str) -> None:
    _, _, index = captured_module(body)
    assert index.module_scope_bindings("relationships") == 2


def test_nonlocal_write_skips_intervening_classes_and_unbound_functions(
    captured_module: CapturedModule,
) -> None:
    _, _, index = captured_module('''
        def outer():
            from pkg import relationships
            class OuterClass:
                relationships = object()
                def middle(self):
                    class InnerClass:
                        nonlocal relationships
                        relationships = object()
    ''')
    outer = next(node for node in index.parents.values()
                 if isinstance(node, ast.FunctionDef) and node.name == "outer")
    assert index.bindings_in("relationships", outer) == 2
    assert index.module_scope_bindings("relationships") == 1


def test_path_dependent_class_read_stays_unproved(captured_module: CapturedModule) -> None:
    _, _, index = captured_module("class WireRevision:\n    if flag:\n        relationships = object()\n    value = relationships.attribute\n")
    assert index.module_scope_bindings("relationships") == 1
    with pytest.raises(ValueError, match="Unproved class namespace binding"):
        index.module_scope_names("relationships")


@pytest.mark.parametrize("member", ("RelationshipCatalogSource", "RelationshipStore"))
@pytest.mark.parametrize("read", (
    "value = lambda self: relationships.MEMBER",
    "def value(self):\n    return relationships.MEMBER",
    "value = relationships.MEMBER\ndef read(self):\n    return self.value",
))
def test_native_move_preserves_class_field_and_actual_module_reads(
    tmp_path: Path, captured_module: CapturedModule, member: str, read: str,
) -> None:
    body = "class WireRevision:\n    relationships: int\n" + textwrap.indent(read.replace("MEMBER", member), "    ") + "\n"
    path, snapshot, _ = captured_module(body)
    operation = {
        "operation": "extract_symbols_to_new_module",
        "file_path": str(tmp_path / "pkg/relationships.py"),
        "symbol_qualnames": ["RelationshipCatalogSource"],
        "destination_path": str(tmp_path / "pkg/catalog_source.py"),
    }
    plan = CodemodPlanRoot.from_json_value({"recipes": [{"recipe_id": "class-scope-move", "operations": [operation]}]})
    result = plan.simulate(snapshot)
    assert result.is_clean
    after = result.required_after_snapshot
    for name, source in after.sources_by_file_path.items():
        destination = Path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
    expression = "row.read()" if "def read(" in read else "row.value()"
    subprocess.run([sys.executable, "-c", (
        "from pkg.presentation import WireRevision\n"
        "row = WireRevision()\n"
        f"assert {expression}.__name__ == {member!r}\n"
        "assert WireRevision.__annotations__['relationships'] is int\n"
    )], cwd=tmp_path, check=True, capture_output=True, text=True)


def test_native_move_still_refuses_real_module_rebinding(
    tmp_path: Path, captured_module: CapturedModule,
) -> None:
    _, snapshot, _ = captured_module("relationships = object()\n")
    operation = {
        "operation": "extract_symbols_to_new_module",
        "file_path": str(tmp_path / "pkg/relationships.py"),
        "symbol_qualnames": ["RelationshipCatalogSource"],
        "destination_path": str(tmp_path / "pkg/catalog_source.py"),
    }
    plan = CodemodPlanRoot.from_json_value({"recipes": [{"recipe_id": "reject-rebind", "operations": [operation]}]})
    with pytest.raises(CodemodOperationPreflightError, match="rebind"):
        plan.simulate(snapshot)
