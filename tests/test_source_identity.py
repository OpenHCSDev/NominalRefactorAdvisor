from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import build_class_family_index
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.source_identity import source_path_text
from nominal_refactor_advisor.source_index import build_source_index


def test_source_identity_joins_native_windows_paths_to_source_evidence() -> None:
    native_path = r"C:\repo\pkg\mod.py"
    source = "class Alpha:\n    pass\n"
    module = ParsedModule(
        path=Path(native_path),
        module_name="pkg.mod",
        is_package_init=False,
        module=ast.parse(source, filename=native_path),
        source=source,
    )
    location = SourceLocation(native_path, 1, "Alpha")
    finding = RefactorFinding(
        detector_id="source-identity-probe",
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Source identity probe",
        summary="Source identity probe",
        why="Source projections must join across native path separators.",
        capability_gap="one canonical source identity",
        relation_context="parsed source and evidence path",
        evidence=(location,),
    )

    source_index = build_source_index((module,), (finding,))
    evidence = source_index.evidence[0]

    assert module.file_path == "C:/repo/pkg/mod.py"
    assert location.file_path == module.file_path
    assert source_path_text(native_path) == module.file_path
    assert evidence.file_id == source_index.files[0].file_id
    assert evidence.target_ids
    assert (
        build_class_family_index((module,)).symbol_for(
            file_path=location.file_path,
            qualname="Alpha",
        )
        == "pkg.mod.Alpha"
    )


def test_codemod_snapshot_canonicalizes_source_mapping_identity() -> None:
    native_path = r"pkg\mod.py"
    source = "class Alpha:\n    pass\n"

    snapshot = CodemodSourceSnapshot.from_source_mapping({native_path: source})

    canonical_path = "pkg/mod.py"
    assert snapshot.sources_by_file_path == {canonical_path: source}
    assert snapshot.source_index.files[0].file_path == canonical_path
    assert snapshot.class_family_index.symbol_for(
        file_path=canonical_path,
        qualname="Alpha",
    ) == "pkg.mod.Alpha"


def test_canonical_source_mapping_rejects_duplicate_path_identities() -> None:
    source = "class Alpha:\n    pass\n"

    try:
        CodemodSourceSnapshot.from_source_mapping(
            {
                r"C:\repo\pkg\mod.py": source,
                "C:/repo/pkg/mod.py": source,
            }
        )
    except ValueError as error:
        assert "same canonical identity" in str(error)
    else:
        raise AssertionError("duplicate source identity was accepted")
