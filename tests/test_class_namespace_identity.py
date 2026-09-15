"""A duplicate declaration entry cannot claim a second original namespace."""

from copy import copy
from pathlib import Path
import runpy

import pytest

from test_native_source_class_preparation import prepared_execution


def rebuild(entry):
    return type(entry)(entry.execution, entry.node)


@pytest.mark.parametrize("duplicate", (copy, rebuild), ids=("copy", "rebuild"))
@pytest.mark.parametrize("warm", (False, True))
@pytest.mark.parametrize(
    "source",
    ("class Family: pass\n", "class Family(metaclass=Creator): pass\n"),
    ids=("plain", "native-metaclass"),
)
def test_duplicate_entry_does_not_own_the_canonical_prepared_namespace(
    source, warm, duplicate
):
    environment, (node,) = prepared_execution(source, conditions=False)
    entry = environment.class_entry(node)
    if warm:
        _ = entry.frame
        entry.require_admitted(environment.initial)
    other = duplicate(entry)
    assert other is not entry
    assert other.execution is environment and other.node is node
    assert environment.class_entry(node) is entry
    with pytest.raises(ValueError, match="canonical"):
        other.require_admitted(environment.initial)
    entry.require_admitted(environment.initial)


def test_canonical_namespace_dsl_preserves_input_and_declaration_signature():
    from nominal_refactor_advisor.codemod import CodemodSourceSnapshot

    path = "nominal_refactor_advisor/source_execution.py"
    source = """class SourceClassBodyEntryABC:
    def require_admitted(self, initial) -> None:
        if initial is not self.initial:
            raise ValueError("Source class belongs to a foreign native admission")
"""
    original = CodemodSourceSnapshot.from_source_mapping({path: source})
    builder = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "docs/examples/canonical_class_namespace_owner.py"
        )
    )["canonical_namespace_plan"]
    result = builder(original).simulate(original)
    assert result.is_clean and result.stage_count == 1
    assert original.sources_by_file_path[path] == source
    final = result.final_snapshot.sources_by_file_path[path]
    assert "def require_admitted(self, initial) -> None:" in final
    assert "self.execution.class_entry(self.node) is not self" in final
