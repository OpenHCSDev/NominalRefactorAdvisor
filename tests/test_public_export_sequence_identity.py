"""Export membership and rename sites share one valid native literal sequence."""

import ast
from pathlib import Path
import sys
from types import ModuleType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    CompactExplicitPublicExportContract,
    CompactUnresolvedPublicExportContract,
    ModulePublicExportSourceAuthority,
    module_public_export_contract,
)


def _module(expression):
    source = f"Known = 1\n__all__ = {expression}\n"
    return ParsedModule(Path("exports.py"), "exports", False, ast.parse(source), source)


@pytest.mark.parametrize(
    "expression", ("['Known']", "('Known',)", "['Known', 'Known']", "[]", "()")
)
def test_valid_literal_sequences_retain_original_rename_sites(expression, monkeypatch):
    module = _module(expression)
    native = ModuleType("_nra_export_sequence_probe")
    exec(module.source, vars(native))
    monkeypatch.setitem(sys.modules, native.__name__, native)
    namespace = {}
    exec("from _nra_export_sequence_probe import *", namespace)
    assert ("Known" in namespace) is bool(native.__all__)
    if native.__all__:
        assert namespace["Known"] == 1
    policy = module_public_export_contract(module)
    assert isinstance(policy, CompactExplicitPublicExportContract)
    assert policy.exported_names == tuple(sorted(set(native.__all__)))
    declaration = ModulePublicExportSourceAuthority.from_module(module.module)
    references = declaration.name_references("Known")
    assert all(
        selected is original
        for selected, original in zip(
            references, declaration.literal_references, strict=True
        )
    )
    assert declaration.literal_references is declaration.literal_references
    assert declaration.name_references("Absent") == ()
    assert tuple(id(reference.literal) for reference in references) == tuple(
        id(element) for element in declaration.value.elts
    )


@pytest.mark.parametrize("expression", ("{'Known'}", "{'Known': 1}", "['Known', 1]"))
def test_invalid_native_export_sequences_do_not_supply_membership_or_rename_sites(
    expression, monkeypatch
):
    module = _module(expression)
    native = ModuleType("_nra_invalid_export_sequence_probe")
    exec(module.source, vars(native))
    monkeypatch.setitem(sys.modules, native.__name__, native)
    with pytest.raises(TypeError):
        exec("from _nra_invalid_export_sequence_probe import *", {})
    assert isinstance(
        module_public_export_contract(module), CompactUnresolvedPublicExportContract
    )
    declaration = ModulePublicExportSourceAuthority.from_module(module.module)
    assert declaration.name_references("Known") == ()
