from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    CollapseRedundantClassAuthorityOperation,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def _write_module(root: Path, relative_path: str, source: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _write_authority_fixture(
    root: Path,
    *,
    method_return: str = "asdict(self)",
    trailing_source: str = "",
) -> Path:
    _write_module(root, "pkg/__init__.py", "")
    _write_module(
        root,
        "pkg/records.py",
        "from abc import ABC\n"
        "from dataclasses import asdict\n"
        "from typing import Any\n\n"
        "class SemanticRecord(ABC):\n"
        "    def to_dict(self) -> dict[str, object]:\n"
        "        record: Any = self\n"
        "        return asdict(record)\n",
    )
    return _write_module(
        root,
        "pkg/source_index.py",
        "from abc import ABC\n"
        "from dataclasses import asdict, dataclass\n\n"
        "class SourceIndexRecord(ABC):\n"
        "    def to_dict(self) -> dict[str, object]:\n"
        f"        return {method_return}\n\n"
        "@dataclass(frozen=True)\n"
        "class SourceFileDigest(SourceIndexRecord):\n"
        "    file_id: str\n\n"
        "@dataclass(frozen=True)\n"
        "class AstTargetDigest(SourceIndexRecord):\n"
        "    target_id: str\n"
        f"{trailing_source}",
    )


def _operation(root: Path) -> CollapseRedundantClassAuthorityOperation:
    return CollapseRedundantClassAuthorityOperation(
        target=SourceRewriteTarget(
            file_path=(root / "pkg/source_index.py").as_posix(),
            qualname="SourceIndexRecord",
        ),
        replacement_base=SourceRewriteTarget(
            file_path=(root / "pkg/records.py").as_posix(),
            qualname="SemanticRecord",
        ),
    )


def _runtime_output(root: Path) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; "
            "from pkg.source_index import AstTargetDigest, SourceFileDigest; "
            "print(json.dumps([SourceFileDigest('file').to_dict(), "
            "AstTargetDigest('target').to_dict()], sort_keys=True))",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_collapses_historical_semantic_record_authority_as_one_reproved_batch(
    tmp_path: Path,
) -> None:
    source_index_path = _write_authority_fixture(tmp_path)
    expected_runtime_output = _runtime_output(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _operation(tmp_path)
    replayed = RefactorRecipeOperation.from_dict(json_report_object(operation))

    result = (
        RefactorRecipe(recipe_id="collapse-semantic-record-authority")
        .with_operation(replayed)
        .simulate(snapshot)
    )
    rewritten = result.simulation.rewritten_sources[source_index_path.as_posix()]
    payload = json_report_object(operation)

    assert result.is_clean is True
    assert payload["operation"] == "collapse_redundant_class_authority"
    assert payload["target_qualname"] == "SourceIndexRecord"
    assert payload["replacement_base"]["target_qualname"] == "SemanticRecord"
    assert "class_names" not in payload
    assert "method_names" not in payload
    assert "SourceIndexRecord" not in rewritten
    assert "from .records import SemanticRecord" in rewritten
    assert "class SourceFileDigest(SemanticRecord):" in rewritten
    assert "class AstTargetDigest(SemanticRecord):" in rewritten
    assert "from abc import ABC" not in rewritten
    assert "from dataclasses import dataclass" in rewritten
    assert "asdict" not in rewritten

    for file_path, rewritten_source in result.simulation.rewritten_sources.items():
        Path(file_path).write_text(rewritten_source, encoding="utf-8")
    assert json.loads(_runtime_output(tmp_path)) == json.loads(expected_runtime_output)


def test_rejects_non_equivalent_class_method_behavior(tmp_path: Path) -> None:
    _write_authority_fixture(tmp_path, method_return="{'different': True}")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="does not have equivalent behavior",
    ):
        RefactorRecipe(recipe_id="different-method").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_matching_method_syntax_with_different_global_binding(
    tmp_path: Path,
) -> None:
    source_index_path = _write_authority_fixture(tmp_path)
    source_index_path.write_text(
        source_index_path.read_text(encoding="utf-8").replace(
            "from dataclasses import asdict, dataclass",
            "from alternate_records import asdict\nfrom dataclasses import dataclass",
        ),
        encoding="utf-8",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="does not have equivalent behavior",
    ):
        RefactorRecipe(recipe_id="different-binding").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_different_neutral_base_mechanics(tmp_path: Path) -> None:
    source_index_path = _write_authority_fixture(tmp_path)
    source_index_path.write_text(
        source_index_path.read_text(encoding="utf-8").replace(
            "class SourceIndexRecord(ABC):",
            "class SourceIndexRecord(object):",
        ),
        encoding="utf-8",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="do not have equivalent base mechanics",
    ):
        RefactorRecipe(recipe_id="different-base-mechanics").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_equivalent_source_that_depends_on_defining_class(
    tmp_path: Path,
) -> None:
    source_index_path = _write_authority_fixture(tmp_path)
    records_path = tmp_path / "pkg/records.py"
    records_path.write_text(
        records_path.read_text(encoding="utf-8").replace(
            "        record: Any = self\n        return asdict(record)",
            "        return super().__repr__()",
        ),
        encoding="utf-8",
    )
    source_index_path.write_text(
        source_index_path.read_text(encoding="utf-8").replace(
            "        return asdict(self)",
            "        return super().__repr__()",
        ),
        encoding="utf-8",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="ownership-sensitive behavior",
    ):
        RefactorRecipe(recipe_id="class-cell-method").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_import_cleanup_that_would_remove_a_retained_alias(
    tmp_path: Path,
) -> None:
    source_index_path = _write_authority_fixture(tmp_path)
    source_index_path.write_text(
        source_index_path.read_text(encoding="utf-8")
        .replace(
            "from dataclasses import asdict, dataclass",
            "from dataclasses import asdict, asdict as retained_asdict, dataclass",
        )
        .replace(
            "class SourceIndexRecord(ABC):",
            "RETAINED_SERIALIZER = retained_asdict\n\nclass SourceIndexRecord(ABC):",
        ),
        encoding="utf-8",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="cannot partially clean a shared import binding",
    ):
        RefactorRecipe(recipe_id="shared-import-alias").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_non_base_reference_to_displaced_authority(tmp_path: Path) -> None:
    _write_authority_fixture(
        tmp_path,
        trailing_source="\nSOURCE_RECORD_TYPE = SourceIndexRecord\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="non-base repository reference",
    ):
        RefactorRecipe(recipe_id="escaping-authority").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_imported_reference_to_displaced_authority(tmp_path: Path) -> None:
    _write_authority_fixture(tmp_path)
    _write_module(
        tmp_path,
        "pkg/consumer.py",
        "from .source_index import SourceIndexRecord\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="imported repository reference",
    ):
        RefactorRecipe(recipe_id="imported-authority").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)
