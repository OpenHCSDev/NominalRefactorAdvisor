from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    ReplaceDirectClassBaseOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.cli import CodemodRecipePlanFastSourceSnapshot
from nominal_refactor_advisor.json_reports import json_report_object


def _write_module(root: Path, relative_path: str, source: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _operation(root: Path) -> ReplaceDirectClassBaseOperation:
    return ReplaceDirectClassBaseOperation(
        target=SourceRewriteTarget(
            file_path=(root / "pkg/legacy.py").as_posix(),
            qualname="LegacyRecord",
        ),
        replacement_base=SourceRewriteTarget(
            file_path=(root / "pkg/records.py").as_posix(),
            qualname="SemanticRecord",
        ),
    )


def test_replaces_complete_direct_child_cohort_from_two_class_targets(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    _write_module(
        tmp_path,
        "pkg/records.py",
        "class SemanticRecord:\n"
        "    def to_dict(self):\n"
        "        return vars(self)\n",
    )
    _write_module(
        tmp_path,
        "pkg/legacy.py",
        "class LegacyRecord:\n"
        "    def to_dict(self):\n"
        "        return vars(self)\n",
    )
    consumer_path = _write_module(
        tmp_path,
        "pkg/consumers.py",
        "from .legacy import LegacyRecord as StoredRecord\n\n"
        "class LocalRole:\n"
        "    pass\n\n"
        "class Alpha(StoredRecord):\n"
        "    pass\n\n"
        "class Beta(StoredRecord, LocalRole):\n"
        "    pass\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _operation(tmp_path)
    replayed = RefactorRecipeOperation.from_dict(json_report_object(operation))

    simulation = (
        RefactorRecipe(recipe_id="replace-direct-class-base")
        .with_operation(replayed)
        .simulate(snapshot)
    )
    rewritten = simulation.simulation.rewritten_sources[consumer_path.as_posix()]
    payload = json_report_object(operation)

    assert simulation.is_clean is True
    assert payload["operation"] == "replace_direct_class_base"
    assert payload["target_qualname"] == "LegacyRecord"
    assert payload["replacement_base"]["target_qualname"] == "SemanticRecord"
    assert "class_names" not in payload
    assert "import_source" not in payload
    assert "from .records import SemanticRecord" in rewritten
    assert "class Alpha(SemanticRecord):" in rewritten
    assert "class Beta(SemanticRecord, LocalRole):" in rewritten
    assert "StoredRecord" not in "\n".join(
        line for line in rewritten.splitlines() if line.startswith("class ")
    )


def test_rejects_replacement_import_cycle_before_emitting_edits(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    _write_module(
        tmp_path,
        "pkg/records.py",
        "from .legacy import LEGACY_MARKER\n\n" "class SemanticRecord:\n" "    pass\n",
    )
    _write_module(
        tmp_path,
        "pkg/legacy.py",
        "LEGACY_MARKER = object()\n\n"
        "class LegacyRecord:\n"
        "    pass\n\n"
        "class Alpha(LegacyRecord):\n"
        "    pass\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(CodemodOperationPreflightError, match="module cycle"):
        RefactorRecipe(recipe_id="cycle").with_operation(_operation(tmp_path)).simulate(
            snapshot
        )


def test_rejects_replacement_related_sibling_base(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    _write_module(
        tmp_path,
        "pkg/records.py",
        "class SemanticRecord:\n"
        "    pass\n\n"
        "class SpecializedRecord(SemanticRecord):\n"
        "    pass\n",
    )
    _write_module(
        tmp_path,
        "pkg/legacy.py",
        "from .records import SpecializedRecord\n\n"
        "class LegacyRecord:\n"
        "    pass\n\n"
        "class Alpha(LegacyRecord, SpecializedRecord):\n"
        "    pass\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="replacement-related sibling base",
    ):
        RefactorRecipe(recipe_id="mro-conflict").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_replacement_descending_from_displaced_authority(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    _write_module(
        tmp_path,
        "pkg/records.py",
        "class LegacyRecord:\n"
        "    pass\n\n"
        "class SemanticRecord(LegacyRecord):\n"
        "    pass\n\n"
        "class Alpha(LegacyRecord):\n"
        "    pass\n",
    )
    operation = ReplaceDirectClassBaseOperation(
        target=SourceRewriteTarget(
            file_path=(tmp_path / "pkg/records.py").as_posix(),
            qualname="LegacyRecord",
        ),
        replacement_base=SourceRewriteTarget(
            file_path=(tmp_path / "pkg/records.py").as_posix(),
            qualname="SemanticRecord",
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="cannot use a related class authority",
    ):
        RefactorRecipe(recipe_id="related-authorities").with_operation(
            operation
        ).simulate(snapshot)


def test_rejects_incomplete_nominal_base_graph(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    _write_module(
        tmp_path,
        "pkg/records.py",
        "class SemanticRecord:\n" "    pass\n",
    )
    _write_module(
        tmp_path,
        "pkg/legacy.py",
        "from external_package import ExternalRole\n\n"
        "class LegacyRecord:\n"
        "    pass\n\n"
        "class Alpha(LegacyRecord, ExternalRole):\n"
        "    pass\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="unresolved nominal bases",
    ):
        RefactorRecipe(recipe_id="open-family").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_repository_derived_operation_disables_explicit_target_snapshot(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    _write_module(tmp_path, "pkg/records.py", "class SemanticRecord:\n    pass\n")
    _write_module(tmp_path, "pkg/legacy.py", "class LegacyRecord:\n    pass\n")
    sequence = CodemodPlanSequence.from_document(
        CodemodPlanDocument(
            recipes=(
                RefactorRecipe(recipe_id="repository-class-graph").with_operation(
                    _operation(tmp_path)
                ),
            )
        )
    )

    snapshot = CodemodRecipePlanFastSourceSnapshot(
        sequence=sequence,
        roots=(tmp_path,),
        cwd=tmp_path,
    ).optional_snapshot()

    assert sequence.requires_complete_source_snapshot is True
    assert snapshot is None
