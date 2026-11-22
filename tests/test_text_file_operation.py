from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.refactors.text_file import PatchTextFileOperation


def operation(path: Path, old: str = "beta", new: str = "delta") -> PatchTextFileOperation:
    return PatchTextFileOperation(target=SourceRewriteTarget(file_path=str(path)),
                                  old_source=old, new_source=new)


def test_patch_text_file_compiles_one_exact_non_python_edit(tmp_path):
    target = tmp_path / "pyproject.toml"
    target.write_text("alpha\nbeta\ngamma\n")
    change, = operation(target).text_file_changes(None)
    assert change.rewritten == "alpha\ndelta\ngamma\n"
    assert change.file_path == str(target)


def test_patch_text_file_rejects_missing_or_ambiguous_span(tmp_path):
    target = tmp_path / "requirements.txt"
    target.write_text("beta\nbeta\n")
    with pytest.raises(ValueError, match="found 2"):
        operation(target).text_file_changes(None)
    with pytest.raises(ValueError, match="found 0"):
        operation(target, old="missing").text_file_changes(None)
