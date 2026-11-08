"""Own-state sites reuse their class's declaration-bound module snapshot."""

from pathlib import Path
from unittest.mock import patch

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_index import ModuleNominalBindingAuthority
from nominal_refactor_advisor.relocation_observations import OwnStateBoolSiteFamily


def test_own_state_sites_keep_binding_answers_without_individual_snapshots(tmp_path: Path) -> None:
    (tmp_path / "records.py").write_text(
        "from dataclasses import dataclass as record\n"
        "from typing import NamedTuple as TupleRecord\n"
        "from builtins import property as prop\n"
        "@record(frozen=True)\n"
        "class Frozen:\n"
        "    value: object\n"
        "    @prop\n"
        "    def present(self) -> bool: return self.value is not None\n"
        "    def kind(self) -> bool: return isinstance(self.value, str)\n"
        "class Tuple(TupleRecord):\n"
        "    value: object\n"
        "    def present(self) -> bool: return self.value is not None\n"
        "class GenericTuple(TupleRecord[object]):\n"
        "    def present(self) -> bool: return self.value is not None\n"
        "def factory():\n"
        "    @record(frozen=True)\n"
        "    class Hidden:\n"
        "        def present(self) -> bool: return self.value is not None\n"
        "if condition:\n"
        "    @record(frozen=True)\n"
        "    class Conditional:\n"
        "        def present(self) -> bool: return self.value is not None\n"
        "isinstance = foreign\n"
        "prop = foreign\n"
        "@record(frozen=True)\n"
        "class ShadowedCalls:\n"
        "    @prop\n"
        "    def present(self) -> bool: return self.value is not None\n"
        "    def kind(self) -> bool: return isinstance(self.value, str)\n"
        "record = foreign\n"
        "TupleRecord = foreign\n"
        "@record(frozen=True)\n"
        "class UnknownDecorator:\n"
        "    def present(self) -> bool: return self.value is not None\n"
        "class UnknownBase(TupleRecord):\n"
        "    def present(self) -> bool: return self.value is not None\n",
        encoding="utf-8",
    )
    (module,) = parse_python_modules(tmp_path, use_parse_cache=False)
    with patch.object(
        ModuleNominalBindingAuthority,
        "snapshot_before",
        autospec=True,
        side_effect=ModuleNominalBindingAuthority.snapshot_before,
    ) as individual_snapshots:
        sites = tuple(OwnStateBoolSiteFamily.sites(module))

    assert {(site.symbol, site.detail[1]) for site in sites} == {
        ("Frozen.present", "property"),
        ("Frozen.kind", "method"),
        ("Tuple.present", "method"),
        ("GenericTuple.present", "method"),
        ("factory.Hidden.present", "method"),
        ("Conditional.present", "method"),
        ("ShadowedCalls.present", "method"),
    }
    # bound_classes already supplies every declaration's snapshot in one module pass.
    assert individual_snapshots.call_count == 0
