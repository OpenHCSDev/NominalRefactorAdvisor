"""Candidate declarations and retained reads are not runtime object identity."""

import ast

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    PatchTargetOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_source_correspondence import (
    SourceReadCorrespondence,
)
from nominal_refactor_advisor.codemod_source_edits import SourceTextReplacement
from test_registry_policy_integrity import _execute

SOURCE = """REGISTRY = {}
class Alpha:
    def kind(self):
        return object
class Beta:
    pass
REGISTRY['alpha'] = Alpha
REGISTRY['beta'] = Beta
"""


def _convert(tmp_path, *, change_body=False):
    path = (tmp_path / "candidate.py").as_posix()
    before = CodemodSourceSnapshot.from_source_mapping({path: SOURCE})
    operation = ConvertManualRegistryToAutoregisterOperation(
        target=SourceRewriteTarget(file_path=path, qualname="Alpha")
    )
    original_targets = operation.required_targets(before).registered_classes.targets
    operations = (operation,)
    if change_body:
        operations += (
            PatchTargetOperation(
                target=SourceRewriteTarget(file_path=path, qualname="Alpha.kind"),
                replacements=(SourceTextReplacement("return object", "return type"),),
            ),
        )
    simulation = CodemodPlanDocument(
        recipes=(RefactorRecipe("candidate-classes", operations=operations),)
    ).simulate(before)
    assert simulation.simulation.parse_valid
    assert not simulation.is_clean
    return path, before, original_targets, simulation


def _projection(snapshot, path):
    return snapshot.product_flow_repository.source_projection(
        snapshot.parsed_module_for_source_path(path)
    )


def _object_read(projection):
    (read,) = tuple(
        node
        for node in projection.reference_reads_by_node
        if isinstance(node, ast.Name) and node.id == "object"
    )
    return read


@pytest.mark.parametrize("change_body", (False, True))
def test_candidate_class_selection_does_not_prove_retained_body_or_runtime_identity(
    tmp_path, change_body
):
    path, before, originals, simulation = _convert(tmp_path, change_body=change_body)
    after = simulation.required_after_snapshot
    candidates = tuple(
        ResolvedClassTarget.from_rewrite_target(
            after, SourceRewriteTarget.from_semantic_target(original.target)
        )
        for original in originals
    )
    assert tuple(target.qualname for target in candidates) == ("Alpha", "Beta")
    assert all(
        candidate.node is after.ast_target_nodes_by_id[candidate.target.target_id]
        for candidate in candidates
    )
    assert all(
        original.node is not candidate.node
        for original, candidate in zip(originals, candidates, strict=True)
    )

    original_source = _projection(before, path)
    candidate_source = _projection(after, path)
    original_read = _object_read(original_source)
    correspondence = SourceReadCorrespondence(
        simulation, original_source, candidate_source
    )
    if change_body:
        with pytest.raises(
            ValueError,
            match="Opaque source window has no unchanged-text correspondence",
        ):
            correspondence.corresponding_reads((original_read,))
    else:
        (candidate_read,) = correspondence.corresponding_reads((original_read,))
        assert candidate_read is _object_read(candidate_source)
        assert candidate_read in ast.walk(candidates[0].node)

    # Execute only this authored fixture; neither target selection nor retained
    # text is substituted for native source-execution evidence.
    original_runtime = _execute(SOURCE)
    candidate_runtime = _execute(after.sources_by_file_path[path])
    assert (
        tuple(original_runtime.REGISTRY)
        == tuple(candidate_runtime.REGISTRY)
        == (
            "alpha",
            "beta",
        )
    )
    assert original_runtime.REGISTRY["alpha"] is original_runtime.Alpha
    assert candidate_runtime.REGISTRY["alpha"] is candidate_runtime.Alpha
    assert original_runtime.REGISTRY["beta"] is original_runtime.Beta
    assert candidate_runtime.REGISTRY["beta"] is candidate_runtime.Beta
    assert original_runtime.Alpha is not candidate_runtime.Alpha
    assert original_runtime.Beta is not candidate_runtime.Beta
    assert original_runtime.Alpha().kind() is object
    assert candidate_runtime.Alpha().kind() is (type if change_body else object)


def test_actual_later_rename_refuses_old_class_selector_even_with_retained_body(
    tmp_path,
):
    path, _, originals, conversion = _convert(tmp_path)
    converted = conversion.required_after_snapshot
    renamed = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                "rename-candidate",
                operations=(
                    PatchTargetOperation(
                        target=SourceRewriteTarget(file_path=path, qualname="Alpha"),
                        replacements=(
                            SourceTextReplacement("class Alpha(", "class Gamma("),
                        ),
                    ),
                ),
            ),
        )
    ).simulate(converted)
    assert renamed.is_clean
    after = renamed.required_after_snapshot
    with pytest.raises(ValueError, match="exactly one eligible source-index target"):
        ResolvedClassTarget.from_rewrite_target(
            after, SourceRewriteTarget.from_semantic_target(originals[0].target)
        )

    gamma = ResolvedClassTarget.from_rewrite_target(
        after, SourceRewriteTarget(file_path=path, qualname="Gamma")
    )
    assert gamma.node is after.ast_target_nodes_by_id[gamma.target.target_id]
    previous_source = _projection(converted, path)
    current_source = _projection(after, path)
    correspondence = SourceReadCorrespondence(renamed, previous_source, current_source)
    (retained,) = correspondence.corresponding_reads((_object_read(previous_source),))
    assert retained is _object_read(current_source)
    assert retained in ast.walk(gamma.node)

    converted_runtime = _execute(converted.sources_by_file_path[path])
    renamed_runtime = _execute(after.sources_by_file_path[path])
    assert "Alpha" not in vars(renamed_runtime)
    assert renamed_runtime.REGISTRY["alpha"] is renamed_runtime.Gamma
    assert converted_runtime.REGISTRY["alpha"] is converted_runtime.Alpha
    assert renamed_runtime.Gamma is not converted_runtime.Alpha
    assert converted_runtime.Alpha().kind() is renamed_runtime.Gamma().kind() is object
