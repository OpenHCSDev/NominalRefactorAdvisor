from pathlib import Path

from nominal_refactor_advisor.cli import analyze_path


def test_opaque_object_annotations_preserve_python_equality_contract(
    tmp_path: Path,
) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "mod.py").write_text(
        '''
class ExactValue:
    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExactValue)

    def __ne__(self, other: object) -> bool:
        return not self == other

    def accepts_opaque_value(self, other: object) -> bool:
        return other is self
''',
        encoding="utf-8",
    )

    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "opaque_object_annotation"
    ]
    summaries = "\n".join(finding.summary for finding in findings)
    assert "ExactValue.__eq__" not in summaries
    assert "ExactValue.__ne__" not in summaries
    assert "ExactValue.accepts_opaque_value" in summaries
