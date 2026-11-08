"""The CLI entry point sees every codemod operation the refactors package declares.

Operations register themselves when their module is imported, so a structural
move that drops the last import of a module silently shrinks the registry
while every name still resolves. The operation registry discovers the
refactors package itself; this compares what the CLI reaches with what the
package's modules declare.
"""

from __future__ import annotations

import json
import subprocess
import sys

_DECLARED_IN_REFACTORS = (
    "import importlib, inspect, json, pkgutil\n"
    "import nominal_refactor_advisor.refactors as refactors\n"
    "from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation\n"
    "declared = sorted(\n"
    "    value.operation_key()\n"
    "    for info in pkgutil.iter_modules(refactors.__path__, refactors.__name__ + '.')\n"
    "    for value in vars(importlib.import_module(info.name)).values()\n"
    "    if inspect.isclass(value) and value.__module__ == info.name\n"
    "    and issubclass(value, RefactorRecipeOperation) and not inspect.isabstract(value)\n"
    ")\n"
    "print(json.dumps(declared))\n"
)
_REACHED_FROM_CLI = (
    "import json\n"
    "import nominal_refactor_advisor.cli\n"
    "from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation\n"
    "print(json.dumps(sorted(RefactorRecipeOperation.__registry__)))\n"
)


def _keys(program: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    return json.loads(completed.stdout.splitlines()[-1])


def test_cli_entry_point_registers_every_refactors_operation() -> None:
    declared = _keys(_DECLARED_IN_REFACTORS)

    assert declared
    assert set(declared) <= set(_keys(_REACHED_FROM_CLI))
