"""Native creation probes distinguish stored arguments from inspected arguments."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_index import RepositoryModuleBindingProof
from nominal_refactor_advisor.class_namespace import ClassNamespaceExecutionEvidence


def _verify_namespace(
    tmp_path: Path, source: str, *, safe: bool, injected: bool
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        source + "\nprint('injected' in vars(Owner))\n", encoding="utf-8", newline=""
    )
    assert subprocess.check_output(
        [sys.executable, str(path)], text=True
    ).strip() == str(injected)
    modules = parse_python_modules(tmp_path)
    module = modules[0]
    owner = next(
        node
        for node in module.module.body
        if isinstance(node, ast.ClassDef) and node.name == "Owner"
    )
    evidence = ClassNamespaceExecutionEvidence.from_class(owner)
    bindings = RepositoryModuleBindingProof(modules)
    if safe:
        evidence.require_closed(bindings, module, owner)
    else:
        with pytest.raises(ValueError):
            evidence.require_closed(bindings, module, owner)


@pytest.mark.parametrize(
    "annotation,safe",
    (
        ("CV[str]", True),
        ("CV[Text]", True),
        ("CV[dict[str, tuple[int, ...]]]", True),
        ("CV['Poison']", True),
        ("CV[Poison]", False),
        ("CV[list[Poison]]", False),
    ),
)
def test_subscription_argument_hashing_uses_native_provenance(
    tmp_path: Path, annotation: str, safe: bool
) -> None:
    source = (
        "from typing import ClassVar as CV\n"
        "from builtins import str as Text\n"
        "class Meta(type):\n"
        "    def __hash__(cls):\n"
        "        import sys\n"
        "        frame = sys._getframe(1)\n"
        "        while frame is not None:\n"
        "            if frame.f_code.co_name == 'Owner': frame.f_locals['injected'] = True\n"
        "            frame = frame.f_back\n"
        "        return 17\n"
        "class Poison(metaclass=Meta): pass\n"
        f"class Owner:\n    field: {annotation} = 1\n"
    )
    eager = sys.version_info < (3, 14)
    _verify_namespace(
        tmp_path, source, safe=safe or not eager, injected=not safe and eager
    )


@pytest.mark.parametrize("constructor", ("staticmethod", "classmethod", "property"))
@pytest.mark.parametrize("argument,safe", (("payload", False), ("lambda: None", True)))
def test_native_descriptor_arguments_can_execute_metadata_hooks(
    tmp_path: Path, constructor: str, argument: str, safe: bool
) -> None:
    source = (
        "class Metadata:\n"
        "    def __getattribute__(self, name):\n"
        "        import sys\n"
        "        frame = sys._getframe(1)\n"
        "        while frame is not None:\n"
        "            if frame.f_code.co_name == 'Owner': frame.f_locals['injected'] = True\n"
        "            frame = frame.f_back\n"
        "        return object.__getattribute__(self, name)\n"
        "payload = Metadata()\n"
        f"class Owner:\n    field = {constructor}({argument})\n"
    )
    _verify_namespace(tmp_path, source, safe=safe, injected=not safe)


def test_computed_subscription_reference_remains_explicitly_unproved(
    tmp_path: Path,
) -> None:
    # Native execution succeeds, but the computed origin has no captured nominal
    # reference. Report an unproved effect rather than leaking a lookup KeyError.
    _verify_namespace(
        tmp_path,
        "class Owner:\n    field = property((list if True else dict)[str])\n",
        safe=False,
        injected=False,
    )
