"""Exact read origins do not transfer lookup, execution or authored acceptance."""

import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CreateFileOperation,
    DescendTypeKeyedBehaviorProjectionOperation,
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    RefactorRecipe,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_runtime import RefactorRecipeOperationCompiler
from nominal_refactor_advisor.codemod_native_requirements import (
    DeclaredNativeUseInvariants,
    NativeUseRequirement,
    NativeUseProvenance,
)
from nominal_refactor_advisor.codemod_source_correspondence import (
    SourceReadCorrespondence,
)
from nominal_refactor_advisor.codemod_source_edits import (
    CodemodSourceRevision,
    SourceTextMutation,
    SourceTextSpanReplacement,
    SourceInsertion,
    SourceSpanReplacement,
    SourceTextGeometry,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import SourceModuleExecution

PATH = "/repo/probe.py"


def document_for(source, edits, path=PATH):
    """Run real preflight, lowering, rendering and document revision capture."""

    class ReadCorrespondenceFixtureOperation(RefactorRecipeOperation):
        def source_edits(self, context):
            return edits

    try:
        snapshot = CodemodSourceSnapshot.from_source_mapping({path: source})
        document = CodemodPlanDocument(
            recipes=(
                RefactorRecipe(
                    "read-correspondence",
                    operations=(
                        ReadCorrespondenceFixtureOperation(
                            target=SourceRewriteTarget(file_path=path),
                        ),
                    ),
                ),
            )
        )
        result = document.simulate(snapshot)
        assert result.is_clean
        return result
    finally:
        del RefactorRecipeOperation.__registry__[
            ReadCorrespondenceFixtureOperation.operation_key()
        ]


def join(
    source="value = object\n",
    prefix="padding = None\n",
    *,
    rewritten=None,
    path=PATH,
    mutation_source=None,
):
    before_module = ParsedModule(Path(PATH), "probe", False, ast.parse(source), source)
    after_source = prefix + source if rewritten is None else rewritten
    after_module = ParsedModule(
        Path(path), "probe", False, ast.parse(after_source), after_source
    )
    before = SourceModuleExecution.from_module(before_module)
    after = SourceModuleExecution.from_module(after_module)
    document_source = source if mutation_source is None else mutation_source
    mutation = SourceTextMutation(
        revision=CodemodSourceRevision(
            PATH,
            CodemodSourceRevision.hash_source(document_source),
        ),
        replacements=(SourceTextSpanReplacement(0, 0, replacement_source=prefix),),
    )
    return (
        before,
        after,
        SourceReadCorrespondence(
            document_for(document_source, (mutation,)), before.source, after.source
        ),
    )


def requirement(env, node):
    return NativeUseRequirement(
        DescendTypeKeyedBehaviorProjectionOperation,
        node,
        (NativeDeclaration(object),),
        env,
    )


def transport(use, relation, after):
    (result,) = NativeUseRequirement.after_edits((use,), relation, after)
    return result


@pytest.mark.parametrize("prefix", ("padding = None\n", "object = object\n"))
def test_native_lookup_is_independently_proved_even_when_its_path_changes(prefix):
    before, after, relation = join(prefix=prefix)
    old = before.module.module.body[-1].value
    initial = requirement(before, old)
    transported = transport(initial, relation, after)
    assert transported.node is after.module.module.body[-1].value
    assert transported.node is not old
    assert transported.operation is initial.operation
    assert transported.declarations is initial.declarations
    for use in (initial, transported):
        inspected = use.inspect()
        assert inspected.provenance is NativeUseProvenance.CAPTURED_IDENTITY
        assert not inspected.provenance.is_admitted


def test_new_binding_can_change_an_unchanged_read():
    before, after, relation = join(prefix="object = type\n")
    initial = requirement(before, before.module.module.body[-1].value)
    transported = transport(initial, relation, after)
    assert initial.inspect().provenance is NativeUseProvenance.CAPTURED_IDENTITY
    with pytest.raises(ValueError, match="required native"):
        transported.inspect()


def test_unproved_intervening_execution_remains_unresolved():
    before, after, relation = join(prefix="unknown_call()\n")
    initial = requirement(before, before.module.module.body[-1].value)
    transported = transport(initial, relation, after)
    assert initial.inspect().provenance is NativeUseProvenance.CAPTURED_IDENTITY
    assert transported.inspect().provenance is NativeUseProvenance.UNRESOLVED


def test_original_authored_acceptance_is_not_a_rewritten_use_receipt():
    before, after, relation = join()
    initial = requirement(before, before.module.module.body[-1].value)
    accepted = DeclaredNativeUseInvariants.from_requirements(
        (initial,), rationale="Fixture's original supported execution contract."
    )
    transported = transport(initial, relation, after)
    assert initial.receipt.revision != transported.receipt.revision
    with pytest.raises(ValueError, match="stale, foreign"):
        accepted.resolve((transported,))
    assert not transported.inspect().provenance.is_admitted


def test_copied_node_coordinates_do_not_authenticate_original_read():
    before, _, relation = join()
    with pytest.raises(ValueError, match="canonical read"):
        relation.corresponding_reads((deepcopy(before.module.module.body[-1].value),))


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"rewritten": "extra = None\nvalue = object\n"}, "exact rewritten"),
        ({"path": "foreign.py"}, "module identity"),
        ({"mutation_source": "value = type\n"}, "original document revision"),
    ),
)
def test_source_authority_is_not_a_coordinate_match(kwargs, message):
    with pytest.raises(ValueError, match=message):
        join(**kwargs)


@pytest.mark.parametrize(
    "field, value", (("module_name", "foreign"), ("is_package_init", True))
)
def test_import_identity_is_part_of_the_two_source_join(field, value):
    _, _, relation = join()
    wrong_module = replace(relation.after.module, **{field: value})
    with pytest.raises(ValueError, match="module identity"):
        replace(relation, after=replace(relation.after, module=wrong_module))


@pytest.mark.parametrize("field", ("before", "after"))
def test_equal_source_projections_cannot_replace_actual_requirement_environments(field):
    before, after, relation = join()
    foreign_before, foreign_after, _ = join()
    initial = requirement(
        foreign_before if field == "before" else before,
        (foreign_before if field == "before" else before).module.module.body[-1].value,
    )
    with pytest.raises(ValueError, match="actual source environments"):
        transport(initial, relation, foreign_after if field == "after" else after)


def test_repeated_lexemes_preserve_individual_read_origins_and_batch_order():
    before, after, relation = join(source="left = object; right = object\n")
    originals = tuple(item.value for item in reversed(before.module.module.body))
    projected = relation.corresponding_reads(originals)
    assert projected == tuple(
        item.value for item in reversed(after.module.module.body[1:])
    )
    assert relation.corresponding_reads(()) == ()
    uses = tuple(requirement(before, node) for node in originals)
    transported = NativeUseRequirement.after_edits(uses, relation, after)
    assert tuple(use.node for use in transported) == projected
    assert all(use.environment is after for use in transported)
    assert NativeUseRequirement.after_edits((), relation, after) == ()


def test_unicode_and_changed_scope_do_not_create_execution_equivalence():
    before, after, relation = join(prefix="class Ω:\n    ")
    old = before.module.module.body[-1].value
    (new,) = relation.corresponding_reads((old,))
    assert new is after.module.module.body[0].body[0].value
    assert (
        before.source.reference_reads_by_node[old].context.flow.owner.kind
        != after.source.reference_reads_by_node[new].context.flow.owner.kind
    )


def test_replaced_identical_text_has_no_retained_read_origin():
    before, after, relation = join(prefix="")
    mutation = SourceTextMutation(
        revision=CodemodSourceRevision(
            PATH, CodemodSourceRevision.hash_source(before.module.source)
        ),
        replacements=(SourceTextSpanReplacement(8, 14, replacement_source="object"),),
    )
    relation = replace(
        relation, document_simulation=document_for(before.module.source, (mutation,))
    )
    with pytest.raises(ValueError, match="unchanged-text correspondence"):
        relation.corresponding_reads((before.module.module.body[-1].value,))


def test_ambiguous_after_operation_does_not_select_a_coordinate_lookalike():
    before, after, relation = join()
    node = after.module.module.body[-1].value
    read = after.source.reference_reads_by_node[node]
    operation = after.source.source_operation(read.context, read.use)
    ambiguous = replace(after.source, operations=(*after.source.operations, operation))
    relation = replace(relation, after=ambiguous)
    with pytest.raises(ValueError, match="unique corresponding read"):
        relation.corresponding_reads((before.module.module.body[-1].value,))


def test_real_mixed_dsl_batch_cannot_claim_its_partial_exact_edit_is_the_whole_change():
    source = """REGISTRY = {}
class Alpha:
    pass
class Beta:
    pass
REGISTRY['alpha'] = Alpha
REGISTRY['beta'] = Beta
"""
    path = "/repo/registry_join.py"
    compiler = RefactorRecipeOperationCompiler.from_context(
        CodemodSourceSnapshot.from_source_mapping({path: source})
    )
    recipe = RefactorRecipe(
        "registry",
        operations=(
            ConvertManualRegistryToAutoregisterOperation(
                target=SourceRewriteTarget(file_path=path, qualname="Alpha"),
            ),
        ),
    )
    batch = compiler.edit_batch_for_recipes((recipe,))
    exact = tuple(edit for edit in batch.edits if isinstance(edit, SourceTextMutation))
    assert exact
    assert len(exact) < len(batch.edits)
    simulation = compiler.simulate_rewrites(batch.planned_rewrites)
    after_snapshot = compiler.with_virtual_sources(simulation.rewritten_sources)
    before = SourceModuleExecution.from_module(
        compiler.parsed_module_for_source_path(path)
    )
    after = SourceModuleExecution.from_module(
        after_snapshot.parsed_module_for_source_path(path)
    )
    assert "AutoRegisterMeta" in after.module.source
    # This tests real compiler provenance, not semantic approval of the registry
    # conversion, whose execution-motion obligations remain independently open.
    for partial in exact:
        with pytest.raises(ValueError, match="exact rewritten source"):
            SourceReadCorrespondence(
                document_for(source, (partial,), path), before.source, after.source
            )


def source_environment(source, path=PATH):
    return SourceModuleExecution.from_module(
        ParsedModule(Path(path), "probe", False, ast.parse(source), source)
    )


def read_named(environment, name):
    (node,) = tuple(
        node
        for node in environment.source.reference_reads_by_node
        if isinstance(node, ast.Name) and node.id == name
    )
    return node


def test_mixed_document_projects_multiple_windows_and_gap_reads_once():
    source = "first = object\nleft = type; right = sorted\nlast = len\n"
    geometry = SourceTextGeometry(source)
    edits = (
        SourceInsertion(
            file_path=PATH, insertion_line=1, inserted_lines=("import typing\n",)
        ),
        geometry.nominal_edit(
            file_path=PATH,
            replacements=(
                SourceTextSpanReplacement(0, 5, replacement_source="renamed_first"),
                SourceTextSpanReplacement(
                    source.index("left"),
                    source.index("left") + 4,
                    replacement_source="renamed_left",
                ),
            ),
        ),
    )
    document = document_for(source, edits)
    before = source_environment(source)
    after = source_environment(
        document.required_after_snapshot.sources_by_file_path[PATH]
    )
    relation = SourceReadCorrespondence(document, before.source, after.source)
    names = ("sorted", "object", "len", "sorted", "type")
    originals = tuple(read_named(before, name) for name in names)
    expected = tuple(read_named(after, name) for name in names)
    index = relation.retained_span_index
    assert relation.corresponding_reads(originals) == expected
    assert relation.corresponding_reads(originals) == expected
    assert relation.retained_span_index is index
    assert all(
        any(window is actual for actual in document.edit_batch.windows)
        for window in index.ordered_inputs
    )


@pytest.mark.parametrize("reverse", (False, True))
def test_equal_opaque_document_output_does_not_create_exact_read_provenance(reverse):
    source = "left = object; right = type\n"
    exact = SourceTextGeometry(source).nominal_edit(
        file_path=PATH,
        replacements=(SourceTextSpanReplacement(0, 4, replacement_source="renamed"),),
    )
    opaque = SourceSpanReplacement(
        file_path=PATH,
        start_line=1,
        end_line=1,
        replacement_lines=(source.replace("left", "renamed"),),
    )
    document = document_for(source, (opaque, exact) if reverse else (exact, opaque))
    before = source_environment(source)
    after = source_environment(
        document.required_after_snapshot.sources_by_file_path[PATH]
    )
    relation = SourceReadCorrespondence(document, before.source, after.source)
    with pytest.raises(ValueError, match="correspondence"):
        relation.corresponding_reads((read_named(before, "type"),))


def test_unchanged_module_can_join_through_another_files_document_edit():
    changed = "/repo/changed.py"
    source = "value = object\n"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {PATH: source, changed: "def run(): return 1\n"}
    )
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                "other-file",
                operations=(
                    ReplaceFunctionBodyOperation(
                        target=SourceRewriteTarget(file_path=changed, qualname="run"),
                        body_source="return 2",
                    ),
                ),
            ),
        )
    ).simulate(snapshot)
    assert PATH not in document.simulation.rewritten_sources
    before, after = source_environment(source), source_environment(source)
    relation = SourceReadCorrespondence(document, before.source, after.source)
    assert relation.corresponding_reads((read_named(before, "object"),)) == (
        read_named(after, "object"),
    )
    assert relation.retained_span_index.ordered_inputs == ()


def test_created_module_has_no_original_reads_but_next_stage_does():
    other = "/repo/other.py"
    source = "value = object\n"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {other: "def run(): return 1\n"}
    )
    result = CodemodPlanSequence.from_operations(
        (
            CreateFileOperation(
                target=SourceRewriteTarget(file_path=PATH), source=source
            ),
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(file_path=other, qualname="run"),
                body_source="return 2",
            ),
        )
    ).simulate(snapshot)
    first, second = (stage.document_simulation for stage in result.stage_reports)
    before, after = source_environment(source), source_environment(source)
    with pytest.raises(ValueError, match="original document revision"):
        SourceReadCorrespondence(first, before.source, after.source)
    relation = SourceReadCorrespondence(second, before.source, after.source)
    assert relation.corresponding_reads((read_named(before, "object"),)) == (
        read_named(after, "object"),
    )
    assert result.simulation.base_revision_by_file_path[PATH].source_hash is None


def test_uncovered_original_module_is_not_recovered_from_another_authority():
    document = document_for("value = object\n", ())
    before = source_environment("value = object\n", "/repo/uncovered.py")
    after = source_environment("value = object\n", "/repo/uncovered.py")
    with pytest.raises(KeyError):
        SourceReadCorrespondence(document, before.source, after.source)


@pytest.mark.parametrize("shadowed_metaclass", (False, True))
def test_failed_candidate_native_gate_preserves_read_origin_but_blocks_application(
    tmp_path, shadowed_metaclass
):
    source = ("AutoRegisterMeta = int\n" if shadowed_metaclass else "") + (
        "REGISTRY = {}\n"
        "class Alpha:\n"
        "    def kind(self):\n"
        "        return object\n"
        "class Beta:\n"
        "    pass\n"
        "REGISTRY['alpha'] = Alpha\n"
        "REGISTRY['beta'] = Beta\n"
    )
    path = tmp_path / "failed_native_candidate.py"
    path.write_text(source)
    snapshot = CodemodSourceSnapshot.from_source_mapping({path.as_posix(): source})
    operation = ConvertManualRegistryToAutoregisterOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Alpha")
    )
    document = CodemodPlanDocument(
        recipes=(RefactorRecipe("native-candidate", operations=(operation,)),)
    ).simulate(snapshot)
    assert document.simulation.parse_valid
    assert not document.preflight_report.is_clean
    assert not document.is_clean
    assert "AutoRegisterMeta" in document.simulation.rewritten_sources[path.as_posix()]

    before_module = snapshot.parsed_module_for_source_path(path.as_posix())
    after_snapshot = document.required_after_snapshot
    after_module = after_snapshot.parsed_module_for_source_path(path.as_posix())
    before = snapshot.product_flow_repository.source_projection(before_module)
    after = after_snapshot.product_flow_repository.source_projection(after_module)
    (original,) = tuple(
        node
        for node in before.reference_reads_by_node
        if isinstance(node, ast.Name) and node.id == "object"
    )
    (expected,) = tuple(
        node
        for node in after.reference_reads_by_node
        if isinstance(node, ast.Name) and node.id == "object"
    )
    correspondence = SourceReadCorrespondence(document, before, after)
    assert correspondence.corresponding_reads((original,)) == (expected,)
    assert original is not expected

    for require_clean in (True, False):
        with pytest.raises(ValueError):
            document.apply(require_clean=require_clean)
        assert path.read_text() == source


def test_invalid_renderer_parse_evidence_cannot_authenticate_source_correspondence():
    _, _, relation = join()
    original_document = relation.document_simulation
    invalid_simulation = replace(
        original_document.simulation,
        parse_validation=replace(
            original_document.simulation.parse_validation,
            parse_valid=False,
        ),
    )
    invalid_document = replace(original_document, simulation=invalid_simulation)
    assert invalid_document.preflight_report.is_clean
    with pytest.raises(ValueError):
        SourceReadCorrespondence(invalid_document, relation.before, relation.after)
