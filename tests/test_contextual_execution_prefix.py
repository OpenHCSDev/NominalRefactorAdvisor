"""Cross-flow slot observations under a checked, explicitly admitted fixture."""

import ast
import builtins
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceEffectsABC,
    CapturedReferenceKernel,
    CapturedReferenceViolation,
    CapturedSlotQuery,
    ChildExecutionPrefix,
    ContextualMutation,
    InitialNativeFrame,
    InitialNativeIsland,
    OpenCapturedReference,
    SequentialExecutionPrefix,
    SingleFlowPrefix,
)
from nominal_refactor_advisor.product_flow import (
    CompactDefinitionTarget,
    CompactEvaluationBranch,
    CompactFlowPosition,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


@dataclass(frozen=True)
class PlainChildFixtureEffects(CapturedReferenceEffectsABC):
    """The fixture admits only an ordinary, header-free class and inert values.

    The fixture explicitly supplies native frames and native builtin/import
    assumptions. This is not a production source-entry proof. Definitions have
    no bases, keywords, decorators or methods; names are exact-string namespace
    accesses, stored values native types/modules, indices exact strings. No
    arbitrary calls/operators/finalizers/import hooks are in this fixture.
    """

    module: object
    parent: object
    child: object
    definition: object
    parent_frame: InitialNativeFrame
    child_frame: InitialNativeFrame

    def admit(self, context, position):
        allowed = {
            ast.Module,
            ast.Import,
            ast.alias,
            ast.ClassDef,
            ast.Assign,
            ast.Name,
            ast.Attribute,
            ast.Load,
            ast.Store,
            ast.Global,
            ast.Subscript,
            ast.Constant,
        }
        for node in ast.walk(self.module.module):
            if type(node) not in allowed:
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.ClassDef) and (
                node.bases or node.keywords or node.decorator_list
            ):
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.Import) and any(
                a.name != "builtins" for a in node.names
            ):
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.Constant) and type(node.value) is not str:
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
        cut = self.definition.target.header_position
        parent = SingleFlowPrefix(self.parent, self.parent_frame, cut)
        if context is self.child:
            return ChildExecutionPrefix(
                parent,
                self.definition,
                SingleFlowPrefix(self.child, self.child_frame, position),
            )
        if context is not self.parent:
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_EFFECTS)
        if not self.definition.position.may_precede(position):
            return SingleFlowPrefix(context, self.parent_frame, position)
        completed = ChildExecutionPrefix(
            parent,
            self.definition,
            SingleFlowPrefix(self.child, self.child_frame, None),
        )
        return SequentialExecutionPrefix(
            (
                completed,
                SingleFlowPrefix(context, self.parent_frame, position, cut),
            )
        )


def fixture(source, globals_storage=None):
    module = ParsedModule(
        Path("contextual_fixture.py"),
        "contextual_fixture",
        False,
        ast.parse(source),
        source,
    )
    projection = compact_product_flow_projection(module)
    parent, child = projection.flow_contexts
    definition = next(
        m
        for m in parent.flow.mutations
        if isinstance(m.target, CompactDefinitionTarget)
    )
    globals_storage = {} if globals_storage is None else globals_storage
    local_storage = {}
    island = InitialNativeIsland((builtins,), (globals_storage, local_storage))
    globals_namespace = island.namespace_for_storage(globals_storage)
    builtins_namespace = island.namespace_for_storage(vars(builtins))
    parent_frame = InitialNativeFrame(
        globals_namespace, globals_namespace, builtins_namespace
    )
    child_frame = InitialNativeFrame(
        island.namespace_for_storage(local_storage),
        globals_namespace,
        builtins_namespace,
    )
    effects = PlainChildFixtureEffects(
        module, parent, child, definition, parent_frame, child_frame
    )
    result = next(
        n
        for n in ast.walk(module.module)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "result"
    )
    read = projection.reference_reads_by_span[SourceByteSpan.require_node(result.value)]
    return CapturedReferenceKernel(island, effects), read


def native(source, expected):
    script = (
        "import builtins\noriginal = builtins.property\n"
        + source
        + "\nprint("
        + expected
        + ")\n"
    )
    return subprocess.check_output([sys.executable, "-c", script], text=True).strip()


@pytest.mark.parametrize("result", ("builtins.property", "saved"))
def test_completed_child_slot_write_distinguishes_saved_object_from_late_lookup(result):
    source = (
        "import builtins\nsaved = builtins.property\n"
        "class Child:\n    builtins.property = object\n"
        f"result = {result}\n"
    )
    kernel, read = fixture(source)
    outcome = kernel.read(read)
    if result == "saved":
        assert outcome.value is property
        assert native(source, "result is original") == "True"
    else:
        assert outcome.violation is CapturedReferenceViolation.POSSIBLE_SLOT_WRITE
        assert outcome.mutation is kernel.effects.child.flow.mutations[0]
        assert native(source, "result is object") == "True"


def test_child_read_uses_parent_positioned_import_before_initial_global_value():
    source = "import builtins\nclass Child:\n    result = builtins.property\n"
    kernel, read = fixture(source, {"builtins": object})
    assert kernel.read(read).value is property
    assert native(source, "Child.result is original") == "True"


def test_completed_child_unrelated_slot_write_preserves_native_member():
    source = "import builtins\nclass Child:\n    builtins.unrelated = object\nresult = builtins.property\n"
    kernel, read = fixture(source)
    assert kernel.read(read).value is property
    assert native(source, "result is original") == "True"


def test_child_global_binding_is_not_misclassified_as_its_local_namespace():
    source = (
        "import builtins\nsaved = builtins.property\n"
        "class Child:\n    global saved\n    saved = object\nresult = saved\n"
    )
    kernel, read = fixture(source)
    assert kernel.read(read).value is object
    assert native(source, "result is object") == "True"


def test_same_spelling_child_local_does_not_replace_parent_capture():
    source = (
        "import builtins\nsaved = builtins.property\n"
        "class Child:\n    saved = object\nresult = saved\n"
    )
    kernel, read = fixture(source)
    assert kernel.read(read).value is property
    assert native(source, "result is original") == "True"


def test_completed_child_raw_globals_write_reaches_parent_binding_slot():
    source = (
        "import builtins\nsaved = builtins.property\n"
        "class Child:\n    ns['saved'] = object\nresult = saved\n"
    )
    globals_storage = {}
    globals_storage["ns"] = globals_storage
    kernel, read = fixture(source, globals_storage)
    assert kernel.read(read).violation is CapturedReferenceViolation.POSSIBLE_SLOT_WRITE


def test_child_entry_keeps_context_and_owner_identity():
    kernel, read = fixture(
        "import builtins\nclass Child:\n    result = builtins.property\n"
    )
    prefix = kernel.effects.admit(read.context, read.use.position)
    assert prefix.endpoint.context is read.context
    assert prefix.definition is kernel.effects.definition
    assert prefix.intervals[0].context is kernel.effects.parent
    assert prefix.intervals[1].context is kernel.effects.child
    with pytest.raises(ValueError, match="actual flow owner"):
        replace(
            prefix,
            child=SingleFlowPrefix(
                kernel.effects.parent, kernel.effects.parent_frame, read.use.position
            ),
        )


def test_composition_rejects_missing_same_activation_interval():
    kernel, read = fixture(
        "import builtins\nclass Child:\n    saved = object\nresult = builtins.property\n"
    )
    prefix = kernel.effects.admit(read.context, read.use.position)
    continuation = prefix.parts[-1]
    with pytest.raises(ValueError, match="omit"):
        continuation.require_admitted(kernel.initial)
    with pytest.raises(ValueError, match="join"):
        replace(
            prefix,
            parts=(prefix.parts[0], replace(continuation, after=read.use.position)),
        ).require_admitted(kernel.initial)


def test_unknown_child_builtin_fallback_does_not_block_qualified_global():
    kernel, read = fixture(
        "import builtins\nclass Child:\n    result = builtins.property\n"
    )
    effects = replace(
        kernel.effects,
        child_frame=replace(
            kernel.effects.child_frame,
            builtins=OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING),
        ),
    )
    assert replace(kernel, effects=effects).read(read).value is property


def test_child_exported_capture_uses_its_historical_cut():
    source = (
        "import builtins\nclass Child:\n    global saved\n"
        "    saved = builtins.property\nbuiltins.property = object\nresult = saved\n"
    )
    kernel, read = fixture(source)
    assert kernel.read(read).value is property
    assert native(source, "result is original") == "True"


def test_parent_refresh_after_completed_child_raw_write_restores_binding_proof():
    source = (
        "import builtins\nclass Child:\n    ns['saved'] = object\n"
        "import builtins\nsaved = builtins.property\nresult = saved\n"
    )
    storage = {}
    storage["ns"] = storage
    kernel, read = fixture(source, storage)
    assert kernel.read(read).value is property


def test_child_receiver_alias_is_resolved_in_child_context_not_parent():
    source = (
        "import builtins\nclass Child:\n    alias = builtins\n"
        "    alias.property = object\nresult = builtins.property\n"
    )
    kernel, read = fixture(source)
    outcome = kernel.read(read)
    assert outcome.violation is CapturedReferenceViolation.POSSIBLE_SLOT_WRITE
    assert outcome.mutation is kernel.effects.child.flow.mutations[1]
    assert native(source, "result is object") == "True"


def test_child_and_parent_positions_are_never_flattened():
    source = (
        "import builtins\nclass Child:\n"
        + "".join(f"    value_{i} = object\n" for i in range(20))
        + "    builtins.property = object\nresult = builtins.property\n"
    )
    kernel, read = fixture(source)
    outcome = kernel.read(read)
    assert outcome.violation is CapturedReferenceViolation.POSSIBLE_SLOT_WRITE
    assert outcome.mutation.position.statement_index > read.use.position.statement_index


def test_shared_actual_mutation_cut_has_one_installation_occurrence():
    source = "import builtins\nclass Child:\n    saved = object\nresult = builtins.property\n"
    kernel, read = fixture(source)
    context = kernel.effects.parent
    frame = kernel.effects.parent_frame
    installation = context.flow.mutations[0]
    first = SingleFlowPrefix(context, frame, installation.position)
    second = SingleFlowPrefix(context, frame, read.use.position, installation.position)
    prefix = SequentialExecutionPrefix((first, second))
    prefix.require_admitted(kernel.initial)
    assert not first.contains(installation)
    assert second.contains(installation)
    assert (
        sum(event.mutation is installation for event in prefix.mutation_occurrences())
        == 1
    )
    query = CapturedSlotQuery(
        frame.globals,
        "builtins",
        prefix,
        frozenset(),
        ContextualMutation(second, installation),
    )
    assert all(
        not isinstance(event, OpenCapturedReference)
        for event in query.mutations_after_installation()
    )


def test_interval_rejects_reversed_cuts_without_sorting_them():
    kernel, read = fixture(
        "import builtins\nclass Child:\n    saved = object\nresult = builtins.property\n"
    )
    with pytest.raises(ValueError, match="ordered cuts"):
        SingleFlowPrefix(
            read.context,
            kernel.effects.parent_frame,
            kernel.effects.definition.target.header_position,
            read.use.position,
        )


def test_equal_cuts_are_empty_not_a_replayed_activation():
    kernel, _read = fixture(
        "import builtins\nclass Child:\n    saved = object\nresult = builtins.property\n"
    )
    position = kernel.effects.parent.flow.mutations[0].position
    empty = SingleFlowPrefix(
        kernel.effects.parent, kernel.effects.parent_frame, position, position
    )
    assert tuple(empty.mutation_occurrences()) == ()


def test_incomparable_evaluation_cuts_do_not_manufacture_interval_order():
    kernel, read = fixture(
        "import builtins\nclass Child:\n    saved = object\nresult = builtins.property\n"
    )
    first = CompactFlowPosition((), 0, 1, (CompactEvaluationBranch(0, 0),))
    second = CompactFlowPosition((), 0, 2, (CompactEvaluationBranch(0, 1),))
    assert first.may_precede(second)
    with pytest.raises(ValueError, match="ordered cuts"):
        SingleFlowPrefix(read.context, kernel.effects.parent_frame, second, first)
