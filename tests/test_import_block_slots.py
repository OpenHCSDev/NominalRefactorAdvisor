"""An added import lands in its family's slot, past type-checking guards."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModuleSet
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    EnsureImportOperation,
    SourceRewriteTarget,
)


def ensured(initial: str, requested: str) -> str:
    module = ParsedModule(
        path=Path("package/slots.py"),
        module_name="package.slots",
        is_package_init=False,
        module=ast.parse(initial),
        source=initial,
    )
    plan = CodemodPlanSequence.from_operations(
        (
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=module.file_path),
                import_source=requested,
            ),
        )
    )
    result = plan.simulate(CodemodSourceSnapshot.from_modules(ParsedModuleSet.of((module,))))
    assert result.is_clean
    return result.final_snapshot.sources_by_file_path[module.file_path]


@pytest.mark.parametrize(
    "initial, requested, expected",
    (
        pytest.param(
            "from typing import TYPE_CHECKING\n"
            "\n"
            "from .alpha import Journal\n"
            "\n"
            "if TYPE_CHECKING:\n"
            "    from .alpha import Proof\n"
            "\n"
            "from .omega import Changed\n"
            "\n"
            "VALUE = 1\n",
            "from .middle import Violation",
            "from typing import TYPE_CHECKING\n"
            "\n"
            "from .alpha import Journal\n"
            "\n"
            "if TYPE_CHECKING:\n"
            "    from .alpha import Proof\n"
            "\n"
            "from .middle import Violation\n"
            "from .omega import Changed\n"
            "\n"
            "VALUE = 1\n",
            id="relative-past-guard",
        ),
        pytest.param(
            "import os\n\nfrom .omega import Changed\n\nVALUE = 1\n",
            "from .middle import Violation",
            "import os\n"
            "\n"
            "from .middle import Violation\n"
            "from .omega import Changed\n"
            "\n"
            "VALUE = 1\n",
            id="start-of-next-family",
        ),
        pytest.param(
            "import os\n\nfrom .omega import Changed\n\nVALUE = 1\n",
            "import sys",
            "import os\n"
            "import sys\n"
            "\n"
            "from .omega import Changed\n"
            "\n"
            "VALUE = 1\n",
            id="end-of-previous-family",
        ),
    ),
)
def test_added_import_joins_its_family(
    initial: str, requested: str, expected: str
) -> None:
    assert ensured(initial, requested) == expected
