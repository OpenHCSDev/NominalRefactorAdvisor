"""Actual native frame dictionaries under explicitly checked fixture effects."""

import ast
import builtins
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceEffectsABC,
    CapturedReferenceKernel,
    CapturedReferenceViolation,
    CapturedSlotQuery,
    InitialNativeFrame,
    InitialNativeIsland,
    NativeNamespace,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import (
    CompactEvaluationBranch,
    CompactFlowContext,
    CompactFlowPosition,
    CompactItemTarget,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


@dataclass(frozen=True)
class SimpleFrameEffects(CapturedReferenceEffectsABC):
    """One actual exec module frame and a narrow, inert fixture language.

    Fixture seeds are exact dictionaries/modules and native type values, never
    user instances/finalizers. Item indices are exact string constants. No calls,
    operators, definitions, closures or custom imports are admitted. Tests that
    exercise adversarial namespace admission do not use this effect provider.
    """

    module: ParsedModule
    context: CompactFlowContext
    frame: InitialNativeFrame
    admissions: list[CompactFlowPosition] = field(default_factory=list)

    def admit(self, context, position):
        self.admissions.append(position)
        if context is not self.context:
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_EFFECTS)
        allowed = {
            ast.Module,
            ast.Import,
            ast.alias,
            ast.Assign,
            ast.Name,
            ast.Attribute,
            ast.Subscript,
            ast.Constant,
            ast.Global,
            ast.Load,
            ast.Store,
        }
        prefix = ast.Module(
            body=self.module.module.body[: position.statement_index + 1],
            type_ignores=[],
        )
        for node in ast.walk(prefix):
            if type(node) not in allowed:
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.Import) and any(
                alias.name != "builtins" for alias in node.names
            ):
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.Subscript) and not (
                isinstance(node.slice, ast.Constant) and type(node.slice.value) is str
            ):
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.Constant) and type(node.value) is not str:
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.Attribute) and node.attr != "property":
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
        return self.frame


def _fixture(source, globals_storage=None, builtin_storage=None, locals_storage=None):
    if globals_storage is None:
        globals_storage = {}
    if builtin_storage is None:
        builtin_storage = vars(builtins)
    if locals_storage is None:
        locals_storage = globals_storage
    module = ParsedModule(
        path=Path("native_frames.py"),
        module_name="native_frames",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    projection = compact_product_flow_projection(module)
    context = projection.flow_contexts[0]
    island = InitialNativeIsland(
        (builtins,),
        (locals_storage, globals_storage, builtin_storage),
    )
    frame = InitialNativeFrame(
        island.namespace_for_storage(locals_storage),
        island.namespace_for_storage(globals_storage),
        island.namespace_for_storage(builtin_storage),
    )
    effects = SimpleFrameEffects(module, context, frame)
    kernel = CapturedReferenceKernel(island, effects)
    result = next(
        node
        for node in module.module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "result"
    )
    read = projection.reference_reads_by_span[SourceByteSpan.require_node(result.value)]
    return kernel, read


def _native(source, setup, comparison):
    code = (
        "import builtins\n" + setup + f"\nexec({source!r}, g, l)\nprint({comparison})\n"
    )
    return subprocess.check_output([sys.executable, "-c", code], text=True).strip()


@pytest.mark.parametrize(
    "local_value,global_value,expected",
    (
        (None, None, property),
        (None, object, object),
        (property, object, property),
    ),
)
def test_actual_initial_local_global_and_builtin_lookup(
    local_value, global_value, expected
):
    globals_storage = {} if global_value is None else {"property": global_value}
    locals_storage = {} if local_value is None else {"property": local_value}
    source = "result = property\n"
    kernel, read = _fixture(source, globals_storage, locals_storage=locals_storage)
    assert kernel.read(read).value is expected
    setup = (
        "g = {'__builtins__': vars(builtins)}\nl = {}\n"
        + ("g['property'] = object\n" if global_value is object else "")
        + ("l['property'] = property\n" if local_value is property else "")
    )
    assert _native(source, setup, f"l['result'] is {expected.__name__}") == "True"


@pytest.mark.parametrize("copied", (False, True))
def test_module_attribute_write_meets_actual_frame_storage_not_module_name(copied):
    source = "import builtins\nbuiltins.property = object\nresult = property\n"
    builtin_storage = vars(builtins).copy() if copied else vars(builtins)
    kernel, read = _fixture(source, builtin_storage=builtin_storage)
    resolution = kernel.read(read)
    if copied:
        assert resolution.value is property
    else:
        assert resolution.violation is CapturedReferenceViolation.POSSIBLE_SLOT_WRITE
    setup = (
        "original = property\n"
        + ("b = vars(builtins).copy()\n" if copied else "b = vars(builtins)\n")
        + "g = {'__builtins__': b}\nl = g\n"
    )
    assert _native(source, setup, "l['result'] is original") == str(copied)


@pytest.mark.parametrize("same_storage", (False, True))
@pytest.mark.parametrize("through_alias", (False, True))
def test_raw_item_write_uses_actual_dictionary_identity(same_storage, through_alias):
    builtin_storage = vars(builtins).copy()
    written_storage = builtin_storage if same_storage else {"property": property}
    prefix = "alias = storage\n" if through_alias else ""
    receiver = "alias" if through_alias else "storage"
    source = prefix + f"{receiver}['property'] = object\nresult = property\n"
    kernel, read = _fixture(source, {"storage": written_storage}, builtin_storage)
    resolution = kernel.read(read)
    if same_storage:
        assert resolution.violation is CapturedReferenceViolation.POSSIBLE_SLOT_WRITE
    else:
        assert resolution.value is property
    setup = (
        "b = vars(builtins).copy()\n"
        + ("s = b\n" if same_storage else "s = {'property': property}\n")
        + "g = {'storage': s, '__builtins__': b}\nl = g\n"
    )
    assert _native(source, setup, "l['result'] is property") == str(not same_storage)


def test_saved_value_is_before_later_raw_namespace_write():
    builtin_storage = vars(builtins).copy()
    source = "saved = property\nstorage['property'] = object\nresult = saved\n"
    kernel, read = _fixture(source, {"storage": builtin_storage}, builtin_storage)
    assert (
        kernel.read(read)
        .require_native_identity(NativeDeclaration(property))
        .declaration
        is property
    )
    assert (
        _native(
            source,
            "b = vars(builtins).copy()\ng = {'storage': b, '__builtins__': b}\nl = g\n",
            "l['result'] is property",
        )
        == "True"
    )


def test_rebinding_global_builtin_name_does_not_replace_captured_frame():
    original = vars(builtins).copy()
    other = {"property": object}
    source = "__builtins__ = other\nother['property'] = object\nresult = property\n"
    kernel, read = _fixture(
        source, {"__builtins__": original, "other": other}, original
    )
    assert kernel.read(read).value is property
    assert (
        _native(
            source,
            "b = vars(builtins).copy()\ng = {'other': {'property': object}, '__builtins__': b}\nl = g\n",
            "l['result'] is property",
        )
        == "True"
    )


@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("global_declared", (False, True))
def test_lexical_write_reaches_only_its_actual_namespace(shared, global_declared):
    builtin_storage = vars(builtins).copy()
    globals_storage = builtin_storage if shared else {}
    # A separate exec locals mapping lets `global` change the actual destination.
    locals_storage = {} if global_declared else globals_storage
    globals_storage["space"] = builtin_storage
    prefix = "global property\n" if global_declared else ""
    source = prefix + "property = object\nresult = space\n"
    kernel, read = _fixture(source, globals_storage, builtin_storage, locals_storage)
    # Resolve an actual initial slot through its retained namespace obligation;
    # ordinary source lookup correctly selects the new lexical object instead.
    resolution = kernel._slot(
        kernel.effects.frame.builtins,
        "property",
        read.context,
        read.use.position,
        frozenset(),
    )
    if shared:
        assert resolution.violation is CapturedReferenceViolation.POSSIBLE_SLOT_WRITE
    else:
        assert resolution.value is property
    setup = (
        "b = vars(builtins).copy()\n"
        + ("g = b\n" if shared else "g = {}\n")
        + "g.update(space=b, __builtins__=b)\n"
        + ("l = {}\n" if global_declared else "l = g\n")
    )
    assert _native(source, setup, "b['property'] is object") == str(shared)


def test_unproved_globals_slot_does_not_fall_through_to_builtin():
    globals_storage = {"property": property}
    globals_storage["space"] = globals_storage
    kernel, read = _fixture(
        "space['property'] = object\nresult = property\n", globals_storage
    )
    resolution = kernel.read(read)
    assert isinstance(resolution, OpenCapturedReference)
    assert resolution.violation is CapturedReferenceViolation.POSSIBLE_SLOT_WRITE


def test_namespace_snapshot_and_frame_owners_are_one_initial_admission():
    storage = {"property": property}
    island = InitialNativeIsland((), (storage, storage))
    namespace = island.namespace_for_storage(storage)
    assert len(island.namespaces) == 1
    assert namespace.storage is storage
    storage["property"] = object
    assert namespace.member("property").value is property
    replacement_snapshot = NativeNamespace(storage)
    assert replacement_snapshot.member("property").value is object
    with pytest.raises(ValueError, match="different admission"):
        island.require_frame(
            InitialNativeFrame(namespace, namespace, replacement_snapshot)
        )
    with pytest.raises(ValueError, match="not admitted"):
        island.namespace_for_storage(storage.copy())
    with pytest.raises(TypeError):
        namespace.initial_entries["property"] = object


def test_kernel_rejects_foreign_frame_snapshot_before_using_it():
    kernel, read = _fixture("result = property\n")
    frame = kernel.effects.frame
    foreign = InitialNativeFrame(
        frame.locals, frame.globals, NativeNamespace(frame.builtins.storage)
    )
    effects = SimpleFrameEffects(kernel.effects.module, kernel.effects.context, foreign)
    with pytest.raises(ValueError, match="different admission"):
        CapturedReferenceKernel(kernel.initial, effects).read(read)


def test_exact_dict_with_hostile_foreign_key_is_rejected_without_equality():
    events = []

    class ForeignKey:
        def __hash__(self):
            return hash("property")

        def __eq__(self, other):
            events.append(other)
            raise AssertionError("Admission must not compare this key")

    storage = {ForeignKey(): property}
    with pytest.raises(TypeError, match="exact string keys"):
        NativeNamespace(storage)
    assert events == []


def test_foreign_query_key_and_dict_subclass_are_rejected_without_lookup():
    class ForeignString(str):
        def __hash__(self):
            raise AssertionError("Query must be rejected before hashing")

    class ForeignDict(dict):
        def __iter__(self):
            raise AssertionError("Custom storage must not be inspected")

    namespace = NativeNamespace({"property": property})
    with pytest.raises(TypeError, match="exact string key"):
        namespace.member(ForeignString("property"))
    with pytest.raises(TypeError, match="exact dictionary"):
        NativeNamespace(ForeignDict())


def test_foreign_string_storage_key_is_rejected_before_copy():
    class ForeignString(str):
        pass

    with pytest.raises(TypeError, match="exact string keys"):
        NativeNamespace({ForeignString("property"): property})


def test_module_and_frame_reuse_the_same_namespace_owner():
    module = ModuleType("native_namespace_owner")
    island = InitialNativeIsland((module,), (vars(module), vars(module)))
    namespace = island.namespace_for_storage(vars(module))
    assert island.namespaces == (namespace,)
    assert (
        island.attribute_namespace(CapturedNativeObject(module), "property")
        is namespace
    )


def test_prefix_admission_count_does_not_scale_with_unrelated_lexical_writes():
    source = (
        "".join(f"alias_{index} = object\n" for index in range(40))
        + "result = property\n"
    )
    kernel, read = _fixture(source)
    assert kernel.read(read).value is property
    # Root admission plus initial globals and builtins slot obligations. Each
    # already-admitted lexical effect shares those fixed frame handles.
    assert len(kernel.effects.admissions) == 3


def test_nonlocal_destinations_remain_open_and_other_contexts_are_not_admitted():
    source = "def outer():\n    value = object\n    def inner():\n        nonlocal value\n        value = property\n"
    module = ParsedModule(
        path=Path("closure.py"),
        module_name="closure",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    projection = compact_product_flow_projection(module)
    inner = next(
        context
        for context in projection.flow_contexts
        if context.flow.owner.qualname == "outer.inner"
    )
    kernel, read = _fixture("result = property\n")
    assert (
        kernel.effects.frame.binding_namespace(inner, "value").violation
        is CapturedReferenceViolation.UNPROVED_BINDING
    )
    assert (
        kernel.effects.admit(inner, read.use.position).violation
        is CapturedReferenceViolation.UNPROVED_EFFECTS
    )


@pytest.mark.parametrize(
    "source,expected",
    (
        ("saved = property\nns['saved'] = object\nresult = saved\n", False),
        (
            "import builtins\nns['builtins'] = other\nresult = builtins.property\n",
            False,
        ),
        ("saved = property\nunrelated['saved'] = object\nresult = saved\n", True),
        (
            "ns['saved'] = object\nimport builtins\nsaved = builtins.property\nresult = saved\n",
            True,
        ),
        ("saved = property\ncopy = saved\nsaved = object\nresult = copy\n", True),
    ),
)
def test_selected_binding_requires_its_namespace_slot_to_survive(source, expected):
    other = ModuleType("other_native_module")
    other.property = object
    globals_storage = {"other": other, "unrelated": {}}
    globals_storage["ns"] = globals_storage
    kernel, read = _fixture(source, globals_storage)
    resolution = kernel.read(read)
    if expected:
        assert resolution.value is property
    else:
        assert isinstance(resolution, OpenCapturedReference)
        assert resolution.violation is CapturedReferenceViolation.POSSIBLE_SLOT_WRITE
    setup = (
        "from types import ModuleType\nother = ModuleType('other_native_module')\n"
        "other.property = object\ng = {'other': other, 'unrelated': {}, '__builtins__': vars(builtins)}\n"
        "g['ns'] = g\nl = g\n"
    )
    assert _native(source, setup, "l['result'] is property") == str(expected)


@pytest.mark.parametrize("repeat", (False, True))
def test_slot_interval_preserves_loop_ambiguity_and_dominating_refresh(repeat):
    body = "ns['saved'] = object\nimport builtins\nsaved = builtins.property\nresult = saved\n"
    source = (
        "for entry in entries:\n"
        + "".join("    " + line + "\n" for line in body.splitlines())
        if repeat
        else body
    )
    module = ParsedModule(
        path=Path("slot_interval.py"),
        module_name="slot_interval",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    projection = compact_product_flow_projection(module)
    context = projection.flow_contexts[0]
    installed = next(
        mutation
        for mutation in context.flow.mutations
        if mutation.target.bound_name == "saved"
    )
    item_write = next(
        mutation
        for mutation in context.flow.mutations
        if isinstance(mutation.target, CompactItemTarget)
    )
    result_node = next(
        node
        for node in ast.walk(module.module)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "result"
    )
    read = projection.reference_reads_by_span[
        SourceByteSpan.require_node(result_node.value)
    ]
    island = InitialNativeIsland((), ({},))
    namespace = island.namespaces[0]
    frame = InitialNativeFrame(namespace, namespace, namespace)
    query = CapturedSlotQuery(
        namespace, "saved", context, frame, frozenset(), installed
    )
    assert (item_write in tuple(query.mutations_before(read.use.position))) is repeat
    # Actual native loop executions refresh this slot each iteration. The
    # compact interval deliberately retains an obligation, not a false proof
    # about repeated execution order.
    assert (
        _native(
            source,
            "g = {'entries': (True, True), '__builtins__': vars(builtins)}\ng['ns'] = g\nl = g\n",
            "l['result'] is property",
        )
        == "True"
    )


def test_slot_interval_does_not_order_unproved_evaluation_siblings():
    kernel, read = _fixture("first = object\nsaved = property\nresult = saved\n")
    first, installed, _ = read.context.flow.mutations
    first = replace(
        first, position=CompactFlowPosition((), 0, 1, (CompactEvaluationBranch(0, 0),))
    )
    installed = replace(
        installed,
        position=CompactFlowPosition((), 0, 2, (CompactEvaluationBranch(0, 1),)),
    )
    flow = replace(read.context.flow, mutations=(first, installed))
    context = replace(read.context, flow=flow)
    frame = kernel.effects.frame
    query = CapturedSlotQuery(
        frame.globals, "saved", context, frame, frozenset(), installed
    )
    assert tuple(query.mutations_before(CompactFlowPosition((), 1, 0))) == (first,)
