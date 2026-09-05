from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodBackend,
    CodemodPlanDocument,
    DeleteModuleCallDeclarationsOperation,
    ModuleCallDeclarationSelector,
    RefactorRecipe,
    SelectionCountExpectation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.json_reports import json_report_object


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(parse_python_modules(root), ())


def test_module_call_declaration_operation_round_trips_and_deletes_exact_call(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "pkg" / "rules.py"
    module_path.parent.mkdir()
    module_path.write_text(
        "declare_rule(AlphaCandidate, summary=alpha_summary)\n\n"
        "declare_rule(BetaCandidate, summary=beta_summary)\n\n"
        "def retain():\n"
        "    declare_rule(AlphaCandidate)\n",
        encoding="utf-8",
    )
    operation = DeleteModuleCallDeclarationsOperation(
        target=SourceRewriteTarget(file_path=module_path.as_posix()),
        declaration_selector=ModuleCallDeclarationSelector(
            callee_qualname="declare_rule",
            positional_argument_prefix=("AlphaCandidate",),
        ),
        selection_count=SelectionCountExpectation(exact=1),
    )
    restored = DeleteModuleCallDeclarationsOperation.from_json_value(
        json_report_object(operation)
    )
    document = CodemodPlanDocument(
        recipes=(RefactorRecipe("delete-alpha-rule").with_operation(restored),)
    )

    simulation = document.simulate(
        _snapshot(tmp_path),
        backend=CodemodBackend.AST_SPAN,
    )

    assert simulation.is_clean
    simulation.apply()
    rewritten = module_path.read_text(encoding="utf-8")
    assert "declare_rule(AlphaCandidate, summary=alpha_summary)" not in rewritten
    assert "declare_rule(BetaCandidate, summary=beta_summary)" in rewritten
    assert "    declare_rule(AlphaCandidate)" in rewritten
    assert json_report_object(operation)["operation"] == (
        "delete_module_call_declarations"
    )


def test_module_call_declaration_operation_fails_closed_on_ambiguous_selection(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "rules.py"
    module_path.write_text(
        "registry.declare(Alpha)\nregistry.declare(Alpha, enabled=True)\n",
        encoding="utf-8",
    )
    operation = DeleteModuleCallDeclarationsOperation(
        target=SourceRewriteTarget(file_path=module_path.as_posix()),
        declaration_selector=ModuleCallDeclarationSelector(
            callee_qualname="registry.declare",
            positional_argument_prefix=("Alpha",),
        ),
        selection_count=SelectionCountExpectation(exact=1),
    )
    document = CodemodPlanDocument(
        recipes=(RefactorRecipe("ambiguous-call").with_operation(operation),)
    )

    with pytest.raises(ValueError, match="expected exactly 1 target"):
        document.simulate(_snapshot(tmp_path), backend=CodemodBackend.AST_SPAN)
