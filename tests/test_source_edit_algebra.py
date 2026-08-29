from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    EnsureImportOperation,
    ImportNameRemoval,
    ImportFromModuleName,
    ModuleImportMutation,
    NominalSourceEdit,
    PhysicalSourceEdit,
    RefactorRecipe,
    ReplaceFieldsWithCarrierOperation,
    SourceEditOrigin,
    SourceFileCreation,
    SourceInsertion,
    SourcePathCandidateSet,
    SourcePathResolutionAuthority,
    SourceSpanReplacement,
    SourceRewriteTarget,
)


def _snapshot(tmp_path: Path, source: str) -> tuple[Path, CodemodSourceSnapshot]:
    module_path = tmp_path / "pkg/mod.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(source, encoding="utf-8")
    return module_path, CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path)
    )


def test_source_path_resolution_mro_preserves_stronger_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_candidate = (first_root / "pkg/mod.py").as_posix()
    second_candidate = (second_root / "pkg/mod.py").as_posix()
    authority = SourcePathResolutionAuthority(
        requested_path="pkg/mod.py",
        candidate_set=SourcePathCandidateSet.from_paths(
            (first_candidate, second_candidate)
        ),
    )

    monkeypatch.chdir(first_root)
    assert authority.required_path() == first_candidate
    monkeypatch.chdir(second_root)
    assert authority.required_path() == second_candidate


def test_source_path_resolution_fails_closed_on_suffix_ambiguity() -> None:
    authority = SourcePathResolutionAuthority(
        requested_path="pkg/mod.py",
        candidate_set=SourcePathCandidateSet.from_paths(
            ("/first/pkg/mod.py", "/second/pkg/mod.py")
        ),
    )

    assert authority.optional_path() is None
    with pytest.raises(ValueError, match="resolved to multiple indexed source files"):
        authority.required_path()


def test_span_replacement_owns_deduplication_and_conflict_proof(
    tmp_path: Path,
) -> None:
    _, context = _snapshot(tmp_path, "FIRST = 1\nSECOND = 2\n")
    first_origin = SourceEditOrigin("recipe", "FirstOperation", 0)
    second_origin = SourceEditOrigin("recipe", "SecondOperation", 1)
    first = SourceSpanReplacement(
        file_path=next(iter(context.sources_by_file_path)),
        start_line=1,
        end_line=1,
        replacement_lines=("FIRST = 3\n",),
        origins=(first_origin,),
    )
    equivalent = SourceSpanReplacement(
        file_path=first.file_path,
        start_line=1,
        end_line=1,
        replacement_lines=first.replacement_lines,
        origins=(second_origin,),
    )

    coalesced = NominalSourceEdit.coalesced_by_declaration((first, equivalent), context)

    assert len(coalesced) == 1
    assert coalesced[0].origins == (first_origin, second_origin)
    with pytest.raises(ValueError, match="Conflicting source span replacements"):
        NominalSourceEdit.coalesced_by_declaration(
            (
                first,
                SourceSpanReplacement(
                    file_path=first.file_path,
                    start_line=1,
                    end_line=1,
                    replacement_lines=("FIRST = 4\n",),
                ),
            ),
            context,
        )
    with pytest.raises(ValueError, match="Physical source edits conflict"):
        PhysicalSourceEdit.require_compatible(
            (
                first,
                SourceSpanReplacement(
                    file_path=first.file_path,
                    start_line=1,
                    end_line=2,
                ),
            )
        )


def test_insertion_owns_exact_deduplication_and_order(tmp_path: Path) -> None:
    _, context = _snapshot(tmp_path, "VALUE = 1\n")
    file_path = next(iter(context.sources_by_file_path))
    first = SourceInsertion(
        file_path=file_path,
        insertion_line=1,
        inserted_lines=("ALPHA = 1\n",),
    )
    second = SourceInsertion(
        file_path=file_path,
        insertion_line=1,
        inserted_lines=("BETA = 2\n",),
    )

    coalesced = NominalSourceEdit.coalesced_by_declaration(
        (first, first, second), context
    )

    assert len(coalesced) == 1
    assert coalesced[0].replacement_lines == ("ALPHA = 1\n", "BETA = 2\n")


def test_import_mutation_owns_add_remove_union_without_parsing_insertions(
    tmp_path: Path,
) -> None:
    module_path, context = _snapshot(
        tmp_path,
        "from pkg.types import Alpha, Gamma\n\nVALUE = 1\n",
    )
    addition = ModuleImportMutation.from_source(
        file_path=module_path.as_posix(),
        import_source="from pkg.types import Beta\n",
    )
    removal = ModuleImportMutation(
        file_path=module_path.as_posix(),
        removals=(
            ImportNameRemoval(
                module_name=ImportFromModuleName("pkg.types"),
                names=("Gamma",),
            ),
        ),
    )
    ordinary_insertion = SourceInsertion(
        file_path=module_path.as_posix(),
        insertion_line=1,
        inserted_lines=("from pkg.types import Delta\n",),
    )

    semantic = NominalSourceEdit.coalesced_by_declaration(
        (addition, removal, ordinary_insertion), context
    )
    mutation = next(edit for edit in semantic if type(edit) is ModuleImportMutation)
    physical = mutation.resolved_edits(context)

    assert any("Beta" in "".join(edit.replacement_lines) for edit in physical)
    assert all("Gamma" not in "".join(edit.replacement_lines) for edit in physical)
    assert ordinary_insertion in semantic
    with pytest.raises(ValueError, match="both add and remove"):
        NominalSourceEdit.coalesced_by_declaration(
            (
                addition,
                ModuleImportMutation.remove_names(
                    file_path=module_path.as_posix(),
                    module_name="pkg.types",
                    names=("Beta",),
                ),
            ),
            context,
        )


def test_file_creation_is_explicit_and_has_one_authority(tmp_path: Path) -> None:
    module_path, context = _snapshot(tmp_path, "")
    creation = SourceFileCreation(
        file_path=module_path.as_posix(),
        source="VALUE = 1\n",
    )

    resolved = creation.resolved_edits(context)
    assert len(resolved) == 1
    assert resolved[0].file_path == module_path.as_posix()
    assert resolved[0].insertion_line == 1
    assert resolved[0].inserted_lines == ("VALUE = 1\n",)
    with pytest.raises(ValueError, match="one creation authority"):
        NominalSourceEdit.coalesced_by_declaration((creation, creation), context)


def test_compiler_unions_imports_and_carrier_projection_stays_granular(
    tmp_path: Path,
) -> None:
    module_path, snapshot = _snapshot(
        tmp_path,
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Stats:\n"
        "    count: int\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Candidate:\n"
        "    name: str\n"
        "    count: int\n\n\n"
        "def build(stats: Stats):\n"
        "    return Candidate(name='x', count=stats.count)\n",
    )
    recipe = (
        RefactorRecipe("typed-source-edits")
        .with_operation(
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=module_path.as_posix()),
                payload_value="from pkg.types import Alpha\n",
            )
        )
        .with_operation(
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=module_path.as_posix()),
                payload_value="from pkg.types import Beta\n",
            )
        )
        .with_operation(
            ReplaceFieldsWithCarrierOperation(
                target=SourceRewriteTarget(file_path=module_path.as_posix()),
                class_name="Candidate",
                carrier_field_declaration="stats: Stats",
                field_projection_pairs=("count=count",),
            )
        )
    )
    carrier_edits = recipe.operations[-1].source_edits(
        snapshot.source_index,
        snapshot.sources_by_file_path,
    )

    assert len(carrier_edits) > 1
    assert all(
        not (
            isinstance(edit, SourceSpanReplacement)
            and edit.start_line == 1
            and edit.end_line
            == len(snapshot.sources_by_file_path[module_path.as_posix()].splitlines())
        )
        for edit in carrier_edits
    )
    rewritten = recipe.simulate_snapshot(snapshot).simulation.rewritten_sources[
        module_path.as_posix()
    ]
    assert rewritten.count("from pkg.types import") == 1
    assert "Alpha" in rewritten
    assert "Beta" in rewritten
    assert "stats: Stats" in rewritten


def test_plan_parser_rejects_obsolete_parallel_rewrite_surface() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unsupported refactor recipe field\(s\): 'rewrites'",
    ):
        CodemodPlanDocument.from_json_value(
            {
                "recipes": [
                    {
                        "recipe_id": "obsolete-rewrite-lane",
                        "rewrites": [
                            {
                                "target_qualname": "Alpha.run",
                                "replacement_source": "def run(self): pass\n",
                            }
                        ],
                    }
                ]
            }
        )

    with pytest.raises(
        ValueError,
        match=r"Unsupported replace_target operation field\(s\): 'legacy_target'",
    ):
        CodemodPlanDocument.from_json_value(
            {
                "recipes": [
                    {
                        "recipe_id": "unknown-operation-field",
                        "operations": [
                            {
                                "operation": "replace_target",
                                "target_qualname": "Alpha.run",
                                "replacement_source": "def run(self): pass\n",
                                "legacy_target": "Alpha.run",
                            }
                        ],
                    }
                ]
            }
        )
