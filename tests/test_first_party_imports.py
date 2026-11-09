"""The importing repository's own modules sort as first party: after third-party
imports, in a block of their own, as ruff's import sort keeps them."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule, ParsedModuleSet
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_imports import ImportSourceGroup, RequestedImportStatement
from nominal_refactor_advisor.declared_root import DeclaredRoot
from nominal_refactor_advisor.refactors.source_insertion import EnsureImportOperation


def _project(root: Path) -> Path:
    (root / "pyproject.toml").write_text('[tool.ruff]\nsrc = ["src"]\n')
    (root / "src/app").mkdir(parents=True)
    (root / "src/app/__init__.py").write_text("")
    return root / "src/app/mod.py"


def _ensured(path: Path, source: str, requested: str) -> str:
    path.write_text(source)
    module = ParsedModule(
        path=path, module_name="app.mod", module=ast.parse(source), source=source
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


def test_ruff_src_names_the_first_party_modules(tmp_path: Path) -> None:
    path = _project(tmp_path)
    assert "app" in DeclaredRoot.first_party_for(path)
    statement = ast.parse("from app import core").body[0]
    assert isinstance(statement, ast.ImportFrom)
    assert RequestedImportStatement(statement).in_file(str(path)).source_group is (
        ImportSourceGroup.FIRST_PARTY
    )
    assert RequestedImportStatement(statement).source_group is ImportSourceGroup.THIRD_PARTY


@pytest.mark.parametrize(
    "source, requested, expected",
    (
        (
            "import os\n\nimport pydantic\n\nfrom app import core\n\nVALUE = 1\n",
            "from app import extra",
            "import os\n\nimport pydantic\n\nfrom app import core, extra\n\nVALUE = 1\n",
        ),
        (
            "import os\n\nimport pydantic\n\nVALUE = 1\n",
            "from app import extra",
            "import os\n\nimport pydantic\n\nfrom app import extra\n\nVALUE = 1\n",
        ),
        (
            "import os\n\nfrom app import core\n\nVALUE = 1\n",
            "import pydantic",
            "import os\n\nimport pydantic\n\nfrom app import core\n\nVALUE = 1\n",
        ),
    ),
)
def test_a_first_party_import_lands_after_the_third_party_block(
    tmp_path: Path, source: str, requested: str, expected: str
) -> None:
    assert _ensured(_project(tmp_path), source, requested) == expected
