from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CarrierFieldProjection,
    CodemodPlanDocument,
    CodemodPlanRoot,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    CreateFileOperation,
    EnsureImportOperation,
    RefactorRecipe,
    RefactorRecipeOperationCompiler,
    ReplaceFieldsWithCarrierOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_import_scopes import ModuleImportScope
from nominal_refactor_advisor.codemod_imports import (
    ImportFromModuleName,
    ImportNameRemoval,
    ModuleImportMutation,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.codemod_paths import (
    SourcePathCandidateSet,
    SourcePathResolutionAuthority,
)
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    PhysicalSourceEdit,
    SourceEditOrigin,
    SourceFileCreation,
    SourceInsertion,
    SourceSpanDeletion,
    SourceSpanEdit,
    SourceSpanReplacement,
)
from nominal_refactor_advisor.codemod_spacing import DestinationInsertionSpacing


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


def test_document_compiler_supplies_one_context_authority_to_every_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path, snapshot = _snapshot(tmp_path, "VALUE = 1\n")
    observed_contexts: list[RefactorRecipeOperationCompiler] = []
    original_source_edits = EnsureImportOperation.source_edits

    def recording_source_edits(
        operation: EnsureImportOperation,
        context: RefactorRecipeOperationCompiler,
    ) -> tuple[ModuleImportMutation, ...]:
        observed_contexts.append(context)
        return original_source_edits(operation, context)

    monkeypatch.setattr(EnsureImportOperation, "source_edits", recording_source_edits)
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe("one-context")
            .with_operation(
                EnsureImportOperation(
                    target=SourceRewriteTarget(file_path=module_path.as_posix()),
                    import_source="from pkg.alpha import Alpha\n",
                )
            )
            .with_operation(
                EnsureImportOperation(
                    target=SourceRewriteTarget(file_path=module_path.as_posix()),
                    import_source="from pkg.beta import Beta\n",
                )
            ),
        )
    )

    document.source_rewrite_batch(snapshot)

    assert len(observed_contexts) == 2
    assert observed_contexts[0] is observed_contexts[1]
    assert observed_contexts[0].execution_snapshot() is observed_contexts[0]
    assert observed_contexts[0].module_node_cache is snapshot.module_node_cache
    assert observed_contexts[0].ast_target_node_cache is snapshot.ast_target_node_cache


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
    with pytest.raises(
        ValueError,
        match="use SourceSpanDeletion",
    ):
        SourceSpanReplacement(
            file_path=first.file_path,
            start_line=1,
            end_line=1,
            replacement_lines=(),
        )
    assert isinstance(
        SourceSpanEdit.from_replacement_lines(
            file_path=first.file_path,
            start_line=1,
            end_line=1,
            replacement_lines=(),
        ),
        SourceSpanDeletion,
    )
    with pytest.raises(ValueError, match="Physical source edits conflict"):
        PhysicalSourceEdit.require_compatible(
            (
                first,
                SourceSpanDeletion(
                    file_path=first.file_path,
                    start_line=1,
                    end_line=2,
                ),
            )
        )


def test_span_deletion_coalesces_overlapping_spans(tmp_path: Path) -> None:
    _, context = _snapshot(tmp_path, "FIRST = 1\nSECOND = 2\nTHIRD = 3\n")
    file_path = next(iter(context.sources_by_file_path))
    first_origin = SourceEditOrigin("recipe", "FirstDeletion", 0)
    second_origin = SourceEditOrigin("recipe", "SecondDeletion", 1)
    first = SourceSpanDeletion(
        file_path=file_path,
        start_line=1,
        end_line=2,
        origins=(first_origin,),
    )
    second = SourceSpanDeletion(
        file_path=file_path,
        start_line=2,
        end_line=3,
        origins=(second_origin,),
    )

    coalesced = NominalSourceEdit.coalesced_by_declaration((first, second), context)

    assert coalesced == (
        SourceSpanDeletion(
            file_path=file_path,
            start_line=1,
            end_line=3,
            origins=(first_origin, second_origin),
        ),
    )


def test_span_deletion_preserves_adjacent_insertion_boundary(tmp_path: Path) -> None:
    _, context = _snapshot(tmp_path, "FIRST = 1\nSECOND = 2\n")
    file_path = next(iter(context.sources_by_file_path))
    deletions = (
        SourceSpanDeletion(file_path=file_path, start_line=1, end_line=1),
        SourceSpanDeletion(file_path=file_path, start_line=2, end_line=2),
    )

    coalesced = NominalSourceEdit.coalesced_by_declaration(deletions, context)

    assert coalesced == deletions
    assert PhysicalSourceEdit.require_compatible(
        (
            *coalesced,
            SourceInsertion(
                file_path=file_path,
                insertion_line=2,
                inserted_lines=("INSERTED = 3\n",),
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


def test_import_mutation_orders_forms_by_nominal_ast_declaration(
    tmp_path: Path,
) -> None:
    module_path, context = _snapshot(tmp_path, "VALUE = 1\n")
    mutations = (
        ModuleImportMutation.from_source(
            file_path=module_path.as_posix(),
            import_source="from collections.abc import Iterable\n",
        ),
        ModuleImportMutation.from_source(
            file_path=module_path.as_posix(),
            import_source="import ast\n",
        ),
        ModuleImportMutation.from_source(
            file_path=module_path.as_posix(),
            import_source="from .types import LocalType\n",
        ),
    )
    mutation = NominalSourceEdit.coalesced_by_declaration(mutations, context)[0]
    physical = mutation.resolved_edits(context)

    assert "".join(physical[0].inserted_lines).startswith(
        "import ast\n"
        "from collections.abc import Iterable\n\n"
        "from .types import LocalType\n"
    )


def test_import_mutation_renders_bare_imports_as_independent_statements(
    tmp_path: Path,
) -> None:
    module_path, context = _snapshot(tmp_path, "VALUE = 1\n")
    mutations = tuple(
        ModuleImportMutation.from_source(
            file_path=module_path.as_posix(),
            import_source=f"import {module_name}\n",
        )
        for module_name in ("tokenize", "ast", "io")
    )

    mutation = NominalSourceEdit.coalesced_by_declaration(mutations, context)[0]
    physical = mutation.resolved_edits(context)

    assert "".join(physical[0].inserted_lines).startswith(
        "import ast\nimport io\nimport tokenize\n"
    )


def test_import_mutation_derives_canonical_python_source_groups(
    tmp_path: Path,
) -> None:
    module_path, context = _snapshot(tmp_path, "VALUE = 1\n")
    mutation = ModuleImportMutation.from_source(
        file_path=module_path.as_posix(),
        import_source=(
            "from .types import LocalType\n"
            "from metaclass_registry import AutoRegisterMeta\n"
            "from typing import Iterable\n"
            "from __future__ import annotations\n"
            "from dataclasses import dataclass\n"
        ),
    )

    physical = mutation.resolved_edits(context)

    assert "".join(physical[0].inserted_lines).startswith(
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n"
        "from typing import Iterable\n\n"
        "from metaclass_registry import AutoRegisterMeta\n\n"
        "from .types import LocalType\n"
    )


def test_import_mutation_splits_direct_imports_at_source_group_boundaries(
    tmp_path: Path,
) -> None:
    module_path, context = _snapshot(tmp_path, "VALUE = 1\n")
    mutation = ModuleImportMutation.from_source(
        file_path=module_path.as_posix(),
        import_source="import metaclass_registry, ast\n",
    )

    physical = mutation.resolved_edits(context)

    assert "".join(physical[0].inserted_lines).startswith(
        "import ast\n\nimport metaclass_registry\n"
    )


def test_import_mutation_rejects_existing_statement_with_mixed_source_groups(
    tmp_path: Path,
) -> None:
    module_path, context = _snapshot(
        tmp_path,
        "import ast, metaclass_registry\n\nVALUE = 1\n",
    )
    mutation = ModuleImportMutation.from_source(
        file_path=module_path.as_posix(),
        import_source="from collections import defaultdict\n",
    )

    with pytest.raises(ValueError, match="spans multiple canonical source groups"):
        mutation.resolved_edits(context)


def test_import_mutation_orders_addition_within_type_checking_guard(
    tmp_path: Path,
) -> None:
    module_path, context = _snapshot(
        tmp_path,
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    from .zeta import Zeta\n\n\n"
        "VALUE = 1\n",
    )
    mutation = ModuleImportMutation.from_source(
        file_path=module_path.as_posix(),
        import_source="from .alpha import Alpha\n",
        scope=ModuleImportScope.TYPE_CHECKING,
    )

    physical = mutation.resolved_edits(context)

    assert len(physical) == 1
    assert physical[0].insertion_line == 4
    assert physical[0].replacement_lines == ("    from .alpha import Alpha\n",)


def test_import_mutation_inserts_absolute_dependency_before_relative_group(
    tmp_path: Path,
) -> None:
    module_path, context = _snapshot(
        tmp_path,
        "import ast\n"
        "from typing import ClassVar\n\n"
        "from .types import LocalType\n\n\n"
        "VALUE = 1\n",
    )
    mutation = ModuleImportMutation.from_source(
        file_path=module_path.as_posix(),
        import_source="from collections import defaultdict\n",
    )

    physical = mutation.resolved_edits(context)

    assert len(physical) == 1
    assert physical[0].insertion_line == 2
    assert physical[0].replacement_lines == ("from collections import defaultdict\n",)


def test_import_mutation_canonically_merges_existing_from_import_aliases(
    tmp_path: Path,
) -> None:
    module_path, context = _snapshot(
        tmp_path,
        "from .types import Zebra, Alpha\n\n\nVALUE = 1\n",
    )
    mutation = ModuleImportMutation.from_source(
        file_path=module_path.as_posix(),
        import_source="from .types import Middle\n",
    )

    physical = mutation.resolved_edits(context)

    assert len(physical) == 1
    assert physical[0].replacement_lines == (
        "from .types import (\n",
        "    Alpha,\n",
        "    Middle,\n",
        "    Zebra,\n",
        ")\n",
    )


def test_import_mutation_refuses_to_erase_import_comments(tmp_path: Path) -> None:
    module_path, context = _snapshot(
        tmp_path,
        "from .types import (\n"
        "    Alpha,  # retained explanation\n"
        ")\n\n\n"
        "VALUE = 1\n",
    )
    mutation = ModuleImportMutation.from_source(
        file_path=module_path.as_posix(),
        import_source="from .types import Beta\n",
    )

    with pytest.raises(ValueError, match="Cannot rewrite commented import"):
        mutation.resolved_edits(context)


def test_import_mutation_preserves_group_boundaries_for_multiple_additions(
    tmp_path: Path,
) -> None:
    module_path, context = _snapshot(
        tmp_path,
        "from .types import LocalType\n\n\nVALUE = 1\n",
    )
    mutations = (
        ModuleImportMutation.from_source(
            file_path=module_path.as_posix(),
            import_source="from __future__ import annotations\n",
        ),
        ModuleImportMutation.from_source(
            file_path=module_path.as_posix(),
            import_source="import ast\n",
        ),
    )
    mutation = NominalSourceEdit.coalesced_by_declaration(mutations, context)[0]

    physical = mutation.resolved_edits(context)

    assert len(physical) == 1
    assert physical[0].insertion_line == 1
    assert physical[0].replacement_lines == (
        "from __future__ import annotations\n",
        "\n",
        "import ast\n",
        "\n",
    )


@pytest.mark.parametrize(
    ("source", "expected_separator"),
    (
        ("import ast\nVALUE = 1\n", "\n\n"),
        ("import ast\n\nVALUE = 1\n", "\n"),
        ("import ast\n\n\nVALUE = 1\n", ""),
    ),
)
def test_import_insertion_derives_only_missing_separator_lines(
    source: str,
    expected_separator: str,
) -> None:
    spacing = DestinationInsertionSpacing.from_source(
        source,
        2,
        inserted_source_is_import_block=True,
    )

    assert spacing.trailing_separator == expected_separator


def test_file_creation_is_explicit_and_has_one_authority(tmp_path: Path) -> None:
    module_path, context = _snapshot(tmp_path, "")
    creation = SourceFileCreation(
        operation_type=CreateFileOperation,
        file_path=module_path.as_posix(),
        source="VALUE = 1\n",
    )
    virtual_context = context.with_virtual_sources(
        {module_path.as_posix(): creation.source}
    )

    resolved = creation.resolved_edits(virtual_context)
    assert len(resolved) == 1
    assert resolved[0].file_path == module_path.as_posix()
    assert resolved[0].insertion_line == 1
    assert resolved[0].inserted_lines == ()
    with pytest.raises(ValueError, match="one creation authority"):
        NominalSourceEdit.coalesced_by_declaration(
            (creation, creation), virtual_context
        )


def test_same_document_operations_resolve_against_created_initial_source(
    tmp_path: Path,
) -> None:
    _existing_path, snapshot = _snapshot(tmp_path, "EXISTING = 1\n")
    generated_path = tmp_path / "pkg/generated.py"
    initial_source = (
        '"""Generated declarations."""\n\n'
        "from __future__ import annotations\n\n\n"
        "VALUE = 1\n"
    )
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe("create-and-import")
            .with_operation(
                CreateFileOperation(
                    target=SourceRewriteTarget(file_path=generated_path.as_posix()),
                    source=initial_source,
                )
            )
            .with_operation(
                EnsureImportOperation(
                    target=SourceRewriteTarget(file_path=generated_path.as_posix()),
                    import_source="import ast\n",
                )
            ),
            RefactorRecipe("add-relative-import").with_operation(
                EnsureImportOperation(
                    target=SourceRewriteTarget(file_path=generated_path.as_posix()),
                    import_source="from .source import Thing\n",
                )
            ),
        )
    )

    simulation = document.simulate(snapshot)
    rewritten = simulation.simulation.rewritten_sources[generated_path.as_posix()]

    assert rewritten == (
        '"""Generated declarations."""\n\n'
        "from __future__ import annotations\n\n"
        "import ast\n\n"
        "from .source import Thing\n\n\n"
        "VALUE = 1\n"
    )
    assert simulation.simulation.base_revisions[0].source_hash is None


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
                import_source="from pkg.types import Alpha\n",
            )
        )
        .with_operation(
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=module_path.as_posix()),
                import_source="from pkg.types import Beta\n",
            )
        )
        .with_operation(
            ReplaceFieldsWithCarrierOperation(
                target=SourceRewriteTarget(
                    file_path=module_path.as_posix(),
                    qualname="Candidate",
                ),
                carrier_field_declaration="stats: Stats",
                field_projections=(CarrierFieldProjection("count", "count"),),
            )
        )
    )
    carrier_edits = recipe.operations[-1].source_edits(snapshot)

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
    rewritten = recipe.simulate(snapshot).simulation.rewritten_sources[
        module_path.as_posix()
    ]
    assert rewritten.count("from pkg.types import") == 1
    assert "Alpha" in rewritten
    assert "Beta" in rewritten
    assert "stats: Stats" in rewritten


def test_plan_declarations_reject_obsolete_or_unknown_fields() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unsupported RefactorRecipe payload field\(s\): 'rewrites'",
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
        match=r"Unsupported AuthorityClaim payload field\(s\): 'legacy_authority'",
    ):
        CodemodPlanDocument.from_json_value(
            {
                "recipes": [
                    {
                        "recipe_id": "unknown-authority-claim-field",
                        "authority_claims": [
                            {
                                "claimed_symbol": "AlphaAuthority",
                                "legacy_authority": "AlphaAuthority",
                            }
                        ],
                    }
                ]
            }
        )

    with pytest.raises(
        ValueError,
        match=(
            r"Unsupported ReplaceTargetOperation payload field\(s\): "
            r"'legacy_target'"
        ),
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


def test_plan_root_owns_document_sequence_input_algebra() -> None:
    document = CodemodPlanDocument(
        recipes=(RefactorRecipe(recipe_id="one-stage"),),
    )
    sequence = CodemodPlanSequence.from_document(document)

    assert CodemodPlanRoot.from_json_value(json_report_object(document)) == document
    assert CodemodPlanRoot.from_json_value(json_report_object(sequence)) == sequence
    assert document.as_sequence() == sequence
    assert sequence.as_sequence() is sequence

    with pytest.raises(
        ValueError,
        match=(
            r"Unsupported CodemodPlanSequence payload field\(s\): "
            r"'architecture_guards', 'recipes'"
        ),
    ):
        CodemodPlanSequence.from_json_value(json_report_object(document))
