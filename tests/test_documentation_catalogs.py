"""Documentation catalogues derive their membership from live declarations."""

from pathlib import Path
import runpy

import pytest

from nominal_refactor_advisor.codemod import RefactorRecipeOperation
from nominal_refactor_advisor.detectors import IssueDetector
from nominal_refactor_advisor.native_declarations import NativeDeclaration


def _generator():
    return runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "docs/source/_ext/catalog_generation.py"
        )
    )


def test_operation_catalog_tracks_registered_declarations_and_is_repeatable(
    tmp_path: Path,
) -> None:
    generate = _generator()["generate_api_reference_pages"]
    generate(tmp_path)
    catalog = tmp_path / "api/_generated/codemod_catalog.rst"
    text = catalog.read_text(encoding="utf-8")
    assert text.count(".. autoclass::") == len(RefactorRecipeOperation.__registry__)
    for operation in RefactorRecipeOperation.__registry__.values():
        assert (
            f":Declaration: ``{NativeDeclaration(operation).qualified_name}``" in text
        )
        assert f".. autoclass:: {NativeDeclaration(operation).qualified_name}" in text
        assert f":Operation key: ``{operation.operation_key()}``" in text
        assert (
            f":Source proof scope: ``{operation.source_dependency_scope.value}``"
            in text
        )
    before = catalog.stat().st_mtime_ns
    generate(tmp_path)
    assert catalog.stat().st_mtime_ns == before


def test_retired_detector_disappears_from_derived_catalogues(tmp_path: Path) -> None:
    reference_dir = tmp_path / "api/detector_reference"
    reference_dir.mkdir(parents=True)
    stale_page = reference_dir / "autoregister_meta_under_rented.rst"
    stale_page.write_text("Obsolete generated detector documentation")
    _generator()["generate_api_reference_pages"](tmp_path)
    assert not stale_page.exists()
    for path in (tmp_path / "api/_generated").glob("*.rst"):
        assert "autoregister_meta_under_rented" not in path.read_text()
        assert "AutoRegisterMetaUnderRented" not in path.read_text()


def test_new_operation_appears_without_a_second_catalogue_declaration(
    tmp_path: Path,
) -> None:
    class DocumentationProbeOperation(RefactorRecipeOperation):
        """A test-only registered operation."""

        def source_edits(self, context):
            return ()

    try:
        generate = _generator()["generate_api_reference_pages"]
        generate(tmp_path)
        catalog = tmp_path / "api/_generated/codemod_catalog.rst"
        assert NativeDeclaration(
            DocumentationProbeOperation
        ).qualified_name in catalog.read_text(encoding="utf-8")
    finally:
        del RefactorRecipeOperation.__registry__[
            DocumentationProbeOperation.operation_key()
        ]
    generate(tmp_path)
    assert NativeDeclaration(
        DocumentationProbeOperation
    ).qualified_name not in catalog.read_text(encoding="utf-8")


def test_detector_reference_uses_actual_declaration_module() -> None:
    render = _generator()["_render_detector_reference_page"]
    for detector in IssueDetector.registered_detector_types():
        assert f".. autoclass:: {NativeDeclaration(detector).qualified_name}" in render(
            detector
        )


@pytest.mark.parametrize("newline", ("\n", "\r\n"), ids=("lf", "crlf"))
def test_generated_reference_writer_preserves_unicode_and_unchanged_files(
    tmp_path: Path,
    newline: str,
) -> None:
    write = _generator()["_write_if_changed"]
    path = tmp_path / "reference.rst"
    text = "caf\u00e9" + newline
    write(path, text)
    assert path.read_bytes() == text.encode("utf-8")
    before = path.stat().st_mtime_ns
    write(path, text)
    assert path.stat().st_mtime_ns == before
