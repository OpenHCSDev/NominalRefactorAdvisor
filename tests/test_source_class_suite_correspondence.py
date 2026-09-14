"""Complete retained class suites are origins, not native result equivalence."""

import ast
from copy import deepcopy

import pytest

from nominal_refactor_advisor.ast_tools import ModuleSyntaxIndex, module_syntax_index
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_source_correspondence import (
    SourceClassSuiteCorrespondence,
    SourceDocumentCorrespondence,
    SourceReadCorrespondence,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from test_registry_candidate_correspondence import _projection
from test_registry_policy_integrity import _execute
from test_source_read_correspondence import document_for

SOURCE = """REGISTRY = {}
class Alpha:
    def kind(self, value=1):
        return value
class Beta:
    pass
REGISTRY['alpha'] = Alpha
REGISTRY['beta'] = Beta
"""


def conversion(tmp_path, source=SOURCE, *, edits=()):
    path = (tmp_path / "candidate.py").as_posix()
    before = CodemodSourceSnapshot.from_source_mapping({path: source})
    converter = ConvertManualRegistryToAutoregisterOperation(
        target=SourceRewriteTarget(file_path=path, qualname="Alpha")
    )
    originals = converter.required_targets(before).registered_classes.targets

    class ClassSuiteFixtureEditOperation(RefactorRecipeOperation):
        def source_edits(self, context):
            geometry = SourceTextGeometry(context.sources_by_file_path[path])
            return (
                geometry.nominal_edit(
                    file_path=path,
                    replacements=tuple(
                        SourceTextSpanReplacement.from_offsets(
                            start_offset=start,
                            end_offset=end,
                            replacement_source=text,
                        )
                        for start, end, text in edits
                    ),
                    rationale="Actual additional document edit",
                ),
            )

    try:
        operations = (converter,)
        if edits:
            operations += (
                ClassSuiteFixtureEditOperation(
                    target=SourceRewriteTarget(file_path=path)
                ),
            )
        simulation = CodemodPlanDocument(
            recipes=(RefactorRecipe("suite-origins", operations=operations),)
        ).simulate(before)
    finally:
        del RefactorRecipeOperation.__registry__[
            ClassSuiteFixtureEditOperation.operation_key()
        ]
    assert simulation.simulation.parse_valid
    assert not simulation.is_clean, "Origin evidence must not admit registration"
    after = simulation.required_after_snapshot
    relation = SourceClassSuiteCorrespondence(
        simulation, _projection(before, path), _projection(after, path)
    )
    candidates = {
        node.name: node
        for node in relation.after.module.module.body
        if isinstance(node, ast.ClassDef)
    }
    return relation, tuple(target.node for target in originals), candidates


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("unicode", (False, True))
def test_actual_conversion_pairs_complete_block_suites(tmp_path, newline, unicode):
    source = SOURCE.replace("kind", "café") if unicode else SOURCE
    source = source.replace("\n", newline)
    relation, originals, candidates = conversion(tmp_path, source)
    assert isinstance(relation, SourceDocumentCorrespondence)
    for original in originals:
        expected = candidates[original.name]
        candidate = relation.corresponding_class(
            original, added_statements=(expected.body[-1],)
        )
        assert candidate is expected
        assert candidate is relation.after.definition_operation(candidate).node
        assert original is relation.before.definition_operation(original).node
        assert original is not candidate
    reader = SourceReadCorrespondence(
        relation.document_simulation, relation.before, relation.after
    )
    assert reader.before_geometry.source == relation.before_geometry.source
    assert reader.retained_span_index.project(()) == ()


def test_identical_class_bodies_pair_by_original_sites_not_text(tmp_path):
    source = SOURCE.replace(
        "    def kind(self, value=1):\n        return value", "    pass"
    )
    relation, originals, candidates = conversion(tmp_path, source)
    actual = tuple(
        relation.corresponding_class(
            original, added_statements=(candidates[original.name].body[-1],)
        )
        for original in originals
    )
    assert actual == (candidates["Alpha"], candidates["Beta"])
    assert ast.dump(actual[0].body[0]) == ast.dump(actual[1].body[0])
    assert actual[0].body[0] is not actual[1].body[0]


def test_suite_origins_do_not_claim_cross_execution_class_identity(tmp_path):
    relation, originals, candidates = conversion(tmp_path)
    original_runtime = _execute(SOURCE)
    candidate_runtime = _execute(relation.after.module.source)
    assert original_runtime.Alpha is not candidate_runtime.Alpha
    assert original_runtime.Alpha().kind() == candidate_runtime.Alpha().kind() == 1
    assert (
        relation.corresponding_class(
            originals[0], added_statements=(candidates["Alpha"].body[-1],)
        )
        is candidates["Alpha"]
    )
    assert not relation.document_simulation.is_clean


@pytest.mark.parametrize("foreign", ("copy", "other-projection"))
def test_original_definition_requires_canonical_identity(tmp_path, foreign):
    relation, originals, candidates = conversion(tmp_path)
    original = deepcopy(originals[0])
    if foreign == "other-projection":
        other, other_originals, _ = conversion(tmp_path)
        assert other.before.module.source == relation.before.module.source
        original = other_originals[0]
    with pytest.raises(ValueError):
        relation.corresponding_class(
            original, added_statements=(candidates["Alpha"].body[-1],)
        )


@pytest.mark.parametrize(
    "foreign", ("copy", "original", "other-class", "other-projection", "duplicate")
)
def test_added_statements_require_actual_selected_candidate_owners(tmp_path, foreign):
    relation, originals, candidates = conversion(tmp_path)
    added = candidates["Alpha"].body[-1]
    additions = (added,)
    if foreign == "copy":
        additions = (deepcopy(added),)
    elif foreign == "original":
        additions = (originals[0].body[0],)
    elif foreign == "other-class":
        additions = (candidates["Beta"].body[-1],)
    elif foreign == "other-projection":
        _, _, other_candidates = conversion(tmp_path)
        additions = (other_candidates["Alpha"].body[-1],)
    elif foreign == "duplicate":
        additions = (added, added)
    with pytest.raises(ValueError, match="actual distinct candidate owners"):
        relation.corresponding_class(originals[0], added_statements=additions)


def test_generated_key_is_an_explicit_delta_not_implicitly_accepted(tmp_path):
    relation, originals, _ = conversion(tmp_path)
    with pytest.raises(ValueError, match="unaccounted"):
        relation.corresponding_class(originals[0])


@pytest.mark.parametrize("side", ("original", "candidate"))
def test_canonical_class_cannot_launder_copied_members(tmp_path, side):
    relation, originals, candidates = conversion(tmp_path)
    owner = originals[0] if side == "original" else candidates["Alpha"]
    owner.body[0] = deepcopy(owner.body[0])
    with pytest.raises(ValueError, match="foreign|reparented"):
        relation.corresponding_class(
            originals[0], added_statements=(candidates["Alpha"].body[-1],)
        )


@pytest.mark.parametrize("side", ("original", "candidate"))
@pytest.mark.parametrize("placement", ("body", "decorator"))
def test_same_tree_reparenting_cannot_redefine_class_suite_ownership(
    tmp_path, side, placement
):
    relation, originals, candidates = conversion(tmp_path)
    owner, other = (
        originals if side == "original" else (candidates["Alpha"], candidates["Beta"])
    )
    moved = other.body[0]
    syntax = module_syntax_index(
        relation.before.module.module
        if side == "original"
        else relation.after.module.module
    )
    assert moved in syntax.node_membership
    assert syntax.parent_by_node[moved] is other
    if placement == "body":
        owner.body.append(moved)
    else:
        # An actual node of the same tree is still foreign to this declaration.
        owner.decorator_list.append(moved)
    with pytest.raises(ValueError, match="reparented"):
        relation.corresponding_class(
            originals[0], added_statements=(candidates["Alpha"].body[-1],)
        )


def test_class_suite_queries_reuse_syntax_index_without_tree_walks(
    tmp_path, monkeypatch
):
    relation, originals, candidates = conversion(tmp_path)
    before_syntax = module_syntax_index(relation.before.module.module)
    after_syntax = module_syntax_index(relation.after.module.module)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "A class-suite query must reuse its original syntax traversal"
        )

    monkeypatch.setattr(ast, "walk", forbidden)
    monkeypatch.setattr(ast, "iter_child_nodes", forbidden)
    monkeypatch.setattr(ModuleSyntaxIndex, "build", forbidden)
    for _ in range(3):
        for original in originals:
            candidate = candidates[original.name]
            assert (
                relation.corresponding_class(
                    original, added_statements=(candidate.body[-1],)
                )
                is candidate
            )
    assert relation.after_syntax is after_syntax
    assert relation.after_syntax.parent_by_node is after_syntax.parent_by_node
    assert before_syntax.parent_by_node[originals[0].body[0]] is originals[0]


def test_syntax_parent_projection_is_derived_and_refuses_ambiguous_nodes(monkeypatch):
    module = ast.parse("class First:\n    pass\nclass Second:\n    pass\n")
    first, second = module.body
    shared = first.body[0]
    second.body[0] = shared
    syntax = module_syntax_index(module)

    def forbidden(*args, **kwargs):
        raise AssertionError("Parent projection must consume the indexed arrays")

    monkeypatch.setattr(ast, "walk", forbidden)
    monkeypatch.setattr(ast, "iter_child_nodes", forbidden)
    assert syntax.parent_by_node[first] is module
    assert syntax.parent_by_node[second] is module
    assert shared not in syntax.parent_by_node
    assert syntax.parent_by_node is syntax.parent_by_node


@pytest.mark.parametrize(
    "part", ("default", "body", "method-decorator", "class-decorator")
)
def test_complete_document_edits_cannot_hide_changed_original_parts(tmp_path, part):
    if part == "default":
        start = SOURCE.index("value=1") + len("value=")
        edit = (start, start + 1, "2")
    elif part == "body":
        start = SOURCE.index("return value") + len("return ")
        edit = (start, start + len("value"), "3")
    elif part == "method-decorator":
        start = SOURCE.index("    def kind")
        edit = (start, start, "    @staticmethod\n")
    else:
        start = SOURCE.index("class Alpha")
        edit = (start, start, "@staticmethod\n")
    relation, originals, candidates = conversion(tmp_path, edits=(edit,))
    with pytest.raises(ValueError):
        relation.corresponding_class(
            originals[0], added_statements=(candidates["Alpha"].body[-1],)
        )
    if part in ("default", "body"):
        runtime = _execute(relation.after.module.source)
        assert runtime.Alpha().kind() == (2 if part == "default" else 3)
        assert _execute(SOURCE).Alpha().kind() == 1


@pytest.mark.parametrize("addition", ("__registry__ = {}", "registry_key = 'wrong'"))
def test_retained_methods_do_not_conceal_extra_candidate_statements(tmp_path, addition):
    start = SOURCE.index("    def kind")
    relation, originals, candidates = conversion(
        tmp_path, edits=((start, start, f"    {addition}\n"),)
    )
    candidate = candidates["Alpha"]
    with pytest.raises(ValueError, match="unaccounted"):
        relation.corresponding_class(
            originals[0], added_statements=(candidate.body[-1],)
        )
    # An explicit delta accounts for source origin only. It never approves a
    # registry policy, and the unchanged converter gate still rejects applying.
    assert (
        relation.corresponding_class(
            originals[0], added_statements=(candidate.body[0], candidate.body[-1])
        )
        is candidate
    )
    assert not relation.document_simulation.is_clean


def test_inline_suite_expansion_has_no_invented_retained_origin(tmp_path):
    source = SOURCE.replace("class Beta:\n    pass", "class Beta: pass")
    relation, originals, candidates = conversion(tmp_path, source)
    with pytest.raises(ValueError, match="unchanged-text correspondence"):
        relation.corresponding_class(
            originals[1], added_statements=(candidates["Beta"].body[-1],)
        )


def test_stale_document_after_revision_is_not_a_correspondence(tmp_path):
    relation, _, _ = conversion(tmp_path)
    changed = CodemodSourceSnapshot.from_source_mapping(
        {relation.after.module.file_path: relation.after.module.source + "extra = 1\n"}
    )
    with pytest.raises(ValueError, match="exact rewritten source"):
        SourceClassSuiteCorrespondence(
            relation.document_simulation,
            relation.before,
            _projection(changed, relation.after.module.file_path),
        )


@pytest.mark.parametrize("rename", (False, True))
def test_class_names_are_not_the_correspondence_and_decorators_are_retained(
    tmp_path, rename
):
    path = (tmp_path / "decorated.py").as_posix()
    source = (
        "def identity(value): return value\n"
        "@identity\n"
        "class Before:\n"
        "    pass\n"
    )
    start = source.index("Before")
    replacement = (
        (start, start + len("Before"), "After") if rename else (0, 0, "padding = 1\n")
    )
    geometry = SourceTextGeometry(source)
    edit = geometry.nominal_edit(
        file_path=path,
        replacements=(
            SourceTextSpanReplacement.from_offsets(
                start_offset=replacement[0],
                end_offset=replacement[1],
                replacement_source=replacement[2],
            ),
        ),
        rationale="Authored source-only class-header control",
    )
    simulation = document_for(source, (edit,), path=path)
    before = simulation.after_snapshot_projection.base_snapshot
    relation = SourceClassSuiteCorrespondence(
        simulation,
        _projection(before, path),
        _projection(simulation.required_after_snapshot, path),
    )
    original = next(
        node
        for node in relation.before.module.module.body
        if isinstance(node, ast.ClassDef)
    )
    candidate = relation.corresponding_class(original)
    assert original.name == "Before"
    assert candidate.name == ("After" if rename else "Before")
    assert candidate is relation.after.definition_operation(candidate).node
    assert len(candidate.decorator_list) == len(original.decorator_list) == 1
