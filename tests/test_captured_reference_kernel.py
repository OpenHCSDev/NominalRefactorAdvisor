"""Native capture under a checked fixture language, not a general effect proof."""

import ast
import builtins
import math
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceEffectsABC,
    CapturedReferenceKernel,
    CapturedReferenceViolation,
    InitialNativeFrame,
    InitialNativeIsland,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import (
    CompactFlowContext,
    CompactFlowPosition,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


@dataclass(frozen=True)
class FixtureEffects(CapturedReferenceEffectsABC):
    """Only fresh module execution over native modules, types and scalar values.

    No external globals, user instances, operator calls, imported user code,
    destruction hooks or definitions are admitted. Literal-bool branches and
    plain-module writes are allowed. Production must provide its own proof.
    """

    module: ParsedModule
    context: CompactFlowContext
    frame: InitialNativeFrame

    def admit(
        self, context: CompactFlowContext, position: CompactFlowPosition
    ) -> InitialNativeFrame | OpenCapturedReference:
        if context is not self.context:
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_EFFECTS)
        statement = (
            position.branch_path[0].parent_statement_index
            if position.branch_path
            else position.statement_index
        )
        prefix = ast.Module(
            body=self.module.module.body[: statement + 1], type_ignores=[]
        )
        allowed = {
            ast.Module,
            ast.Import,
            ast.ImportFrom,
            ast.alias,
            ast.Assign,
            ast.Name,
            ast.Attribute,
            ast.Load,
            ast.Store,
            ast.Constant,
            ast.If,
        }
        for node in ast.walk(prefix):
            if type(node) not in allowed:
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.Import) and any(
                alias.name not in {"builtins", "math"} for alias in node.names
            ):
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.ImportFrom) and (
                node.level
                or node.module not in {"builtins", "math"}
                or any(
                    alias.name not in {"property", "object", "pi"}
                    for alias in node.names
                )
            ):
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.Attribute) and (
                not isinstance(node.value, ast.Name)
                or node.attr not in {"property", "unrelated", "pi", "__class__"}
            ):
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.Constant) and type(node.value) not in {
                bool,
                int,
                type(None),
            }:
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
            if isinstance(node, ast.If) and not (
                isinstance(node.test, ast.Constant) and type(node.test.value) is bool
            ):
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
        return self.frame


def _fixture(source):
    module = ParsedModule(
        path=Path("capture.py"),
        module_name="capture",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    projection = compact_product_flow_projection(module)
    context = projection.flow_contexts[0]
    globals_storage = {}
    island = InitialNativeIsland((builtins, math), (globals_storage,))
    globals_namespace = island.namespace_for_storage(globals_storage)
    frame = InitialNativeFrame(
        globals_namespace,
        globals_namespace,
        island.namespace_for_storage(vars(builtins)),
    )
    effects = FixtureEffects(module, context, frame)
    kernel = CapturedReferenceKernel(island, effects)
    result = next(
        node
        for node in module.module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "result"
    )
    read = projection.reference_reads_by_span[SourceByteSpan.require_node(result.value)]
    return kernel, read, projection


SLOT_CASES = (
    (
        "import builtins\nimport math\nmodule = builtins\nbuiltins = math\nresult = module.property\n",
        True,
    ),
    ("import builtins\nresult = builtins.property\n", True),
    (
        "import builtins\nbuiltins.property = object\nresult = builtins.property\n",
        False,
    ),
    ("import builtins\nbuiltins.property = object\nresult = property\n", False),
    (
        "import builtins\nmodule = builtins\nbuiltins.property = object\nresult = module.property\n",
        False,
    ),
    (
        "import builtins\nmodule = builtins\nmodule.property = object\nresult = builtins.property\n",
        False,
    ),
    (
        "import builtins\nsaved = builtins.property\nbuiltins.property = object\nresult = saved\n",
        True,
    ),
    (
        "import builtins\nsaved = property\nbuiltins.property = object\nresult = saved\n",
        True,
    ),
    (
        "import builtins\nfrom builtins import property as saved\nbuiltins.property = object\nresult = saved\n",
        True,
    ),
    (
        "import builtins\nbuiltins.property = object\nfrom builtins import property as saved\nresult = saved\n",
        False,
    ),
    (
        "import builtins\nbuiltins.unrelated = object\nresult = builtins.property\n",
        True,
    ),
    (
        "import builtins\nimport math\nmath.pi = object\nresult = builtins.property\n",
        True,
    ),
    (
        "import builtins\nif True:\n    builtins.property = object\nresult = builtins.property\n",
        False,
    ),
    ("import builtins\nresult = builtins.property\nbuiltins.property = object\n", True),
)


@pytest.mark.parametrize("source, unchanged", SLOT_CASES)
def test_capture_and_later_slot_access_agree_with_isolated_native_execution(
    source, unchanged
):
    kernel, read, _ = _fixture(source)
    resolution = kernel.read(read)
    if unchanged:
        assert isinstance(resolution, CapturedNativeObject)
        assert resolution.value is builtins.property
        declaration = NativeDeclaration(property)
        assert resolution.require_native_identity(declaration) is declaration
    else:
        assert isinstance(resolution, OpenCapturedReference)
        assert resolution.violation is CapturedReferenceViolation.POSSIBLE_SLOT_WRITE
        assert resolution.mutation is not None
        with pytest.raises(ValueError, match="identity remains open"):
            resolution.require_native_identity(NativeDeclaration(property))
    native = (
        "import builtins\noriginal = builtins.property\n"
        + source
        + "\nprint(result is original)\n"
    )
    observed = subprocess.check_output(
        [sys.executable, "-c", native], text=True
    ).strip()
    assert observed == str(unchanged)


@pytest.mark.parametrize(
    "prefix",
    (
        "class Earlier: pass\n",
        "def earlier(): pass\n",
        "call()\n",
        "left + right\n",
        "from unknown import *\n",
        "__builtins__ = object\n",
        "del previous\n",
    ),
)
def test_missing_or_unsupported_source_effects_never_become_native_proof(prefix):
    kernel, read, _ = _fixture(prefix + "result = property\n")
    resolution = kernel.read(read)
    assert isinstance(resolution, OpenCapturedReference)
    assert resolution.violation is CapturedReferenceViolation.UNPROVED_EFFECTS


def test_effect_admission_is_tied_to_actual_capture_prefix_not_future_source():
    kernel, read, _ = _fixture("result = property\nlater()\n")
    assert (
        kernel.read(read)
        .require_native_identity(NativeDeclaration(property))
        .declaration
        is property
    )


def test_module_layout_write_is_not_reduced_to_an_unrelated_dictionary_slot():
    kernel, read, _ = _fixture(
        "import builtins\nbuiltins.__class__ = object\nresult = builtins.property\n"
    )
    resolution = kernel.read(read)
    assert isinstance(resolution, OpenCapturedReference)
    assert resolution.violation is CapturedReferenceViolation.UNPROVED_EFFECTS
    assert resolution.mutation.target.attribute_name == "__class__"


def test_unknown_receiver_does_not_prove_a_distinct_slot():
    kernel, read, _ = _fixture(
        "import builtins\nunknown.property = object\nresult = builtins.property\n"
    )
    resolution = kernel.read(read)
    assert isinstance(resolution, OpenCapturedReference)
    assert resolution.violation is CapturedReferenceViolation.UNKNOWN_RECEIVER
    assert (
        resolution.mutation.target.receiver_use.lexical_reference.root_name == "unknown"
    )


def test_effect_authority_has_no_default_or_cross_context_admission():
    with pytest.raises(TypeError):
        CapturedReferenceEffectsABC()
    kernel, read, _ = _fixture("result = property\n")
    _, other, _ = _fixture("result = property\n")
    assert kernel.effects.admit(read.context, read.use.position) is kernel.effects.frame
    assert (
        kernel.effects.admit(other.context, other.use.position).violation
        is CapturedReferenceViolation.UNPROVED_EFFECTS
    )


def test_initial_island_requires_plain_native_modules_and_admitted_storage():
    class CustomModule(type(builtins)):
        pass

    with pytest.raises(TypeError, match="plain native"):
        InitialNativeIsland((CustomModule("custom"),))
    with pytest.raises(ValueError, match="not admitted"):
        InitialNativeIsland((math,)).namespace_for_storage(vars(builtins))
    with pytest.raises(ValueError, match="objects must be unique"):
        InitialNativeIsland((builtins, builtins))
    assert (
        InitialNativeIsland((builtins,)).module("missing").violation
        is CapturedReferenceViolation.UNADMITTED_IMPORT
    )


def test_native_identity_does_not_accept_matching_source_names():
    replacement = type("property", (), {})
    replacement.__module__ = "builtins"
    with pytest.raises(ValueError, match="not the required"):
        CapturedNativeObject(replacement).require_native_identity(
            NativeDeclaration(property)
        )


def test_saved_import_does_not_license_an_unproved_later_call():
    source = "from builtins import property as saved\nmutate()\nresult = saved\n"
    kernel, read, _ = _fixture(source)
    resolution = kernel.read(read)
    assert isinstance(resolution, OpenCapturedReference)
    assert resolution.violation is CapturedReferenceViolation.UNPROVED_EFFECTS
    native = (
        "import builtins\noriginal = builtins.property\n"
        "def mutate(): builtins.property = object\n"
        + source
        + "\nprint(result is original)\n"
    )
    assert (
        subprocess.check_output([sys.executable, "-c", native], text=True).strip()
        == "True"
    )


def test_native_identity_alone_does_not_authenticate_mutable_implementation():
    def original():
        return 1

    def replacement():
        return 2

    declaration = NativeDeclaration(original)
    captured = CapturedNativeObject(original)
    original.__code__ = replacement.__code__
    assert captured.require_native_identity(declaration) is declaration
    assert declaration.declaration() == 2


@dataclass(frozen=True)
class PlainModuleFixtureEffects(CapturedReferenceEffectsABC):
    """Controlled native-only seeds and explicit simple source, not a default.

    Callers below construct fresh plain modules whose custom values are native
    types or plain modules. No user hooks, finalizers, import execution, frame
    replacements or implicit operators exist in these particular fixtures.
    """

    module: ParsedModule
    context: CompactFlowContext
    frame: InitialNativeFrame

    def admit(self, context, position):
        allowed = {
            ast.Module,
            ast.Import,
            ast.ImportFrom,
            ast.alias,
            ast.Assign,
            ast.Name,
            ast.Attribute,
            ast.Load,
            ast.Store,
        }
        if context is not self.context or any(
            type(node) not in allowed for node in ast.walk(self.module.module)
        ):
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_EFFECTS)
        for node in ast.walk(self.module.module):
            if isinstance(node, ast.ImportFrom) and (
                node.level or any(alias.name == "*" for alias in node.names)
            ):
                return OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_EFFECTS
                )
        return self.frame


@contextmanager
def _module_registry(entries):
    previous = {name: sys.modules[name] for name in entries if name in sys.modules}
    sys.modules.update(entries)
    try:
        yield
    finally:
        for name in entries:
            if name in previous:
                sys.modules[name] = previous[name]
            else:
                del sys.modules[name]


def _plain_module_fixture(source, modules, builtin_storage=None, registry_entries=None):
    module = ParsedModule(
        path=Path("native_module_fixture.py"),
        module_name="native_module_fixture",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    projection = compact_product_flow_projection(module)
    context = projection.flow_contexts[0]
    entries = (
        {module.__name__: module for module in modules}
        if registry_entries is None
        else registry_entries
    )
    with _module_registry(entries):
        globals_storage = {}
        if builtin_storage is None:
            builtin_storage = vars(builtins)
        island = InitialNativeIsland(modules, (globals_storage, builtin_storage))
        globals_namespace = island.namespace_for_storage(globals_storage)
        frame = InitialNativeFrame(
            globals_namespace,
            globals_namespace,
            island.namespace_for_storage(builtin_storage),
        )
        kernel = CapturedReferenceKernel(
            island,
            PlainModuleFixtureEffects(module, context, frame),
        )
        read = projection.reference_reads_by_span[
            SourceByteSpan.require_node(module.module.body[-1].value)
        ]
        return kernel.read(read)


@pytest.mark.parametrize("attribute", ("__class__", "__dict__"))
@pytest.mark.parametrize("from_import", (False, True))
def test_module_attribute_descriptor_wins_over_colliding_namespace_entry(
    attribute, from_import
):
    module = ModuleType("descriptor_fixture")
    vars(module)[attribute] = property
    source = (
        f"from descriptor_fixture import {attribute} as selected\nresult = selected\n"
        if from_import
        else f"import descriptor_fixture\nresult = descriptor_fixture.{attribute}\n"
    )
    resolution = _plain_module_fixture(source, (builtins, module))
    assert isinstance(resolution, OpenCapturedReference)
    assert resolution.violation is CapturedReferenceViolation.UNPROVED_ACCESS
    native = (
        "import sys\nfrom types import ModuleType\n"
        "module = ModuleType('descriptor_fixture')\n"
        f"vars(module)[{attribute!r}] = property\n"
        "sys.modules['descriptor_fixture'] = module\n"
        + source
        + "\nprint(result is property)\n"
    )
    assert (
        subprocess.check_output([sys.executable, "-c", native], text=True).strip()
        == "False"
    )


@pytest.mark.parametrize("attribute", ("__class__", "__dict__"))
def test_bare_builtin_lookup_uses_dictionary_despite_module_descriptor(attribute):
    module = ModuleType("frame_fixture")
    vars(module)[attribute] = property
    source = f"result = {attribute}\n"
    resolution = _plain_module_fixture(source, (module,), vars(module))
    assert (
        resolution.require_native_identity(NativeDeclaration(property)).declaration
        is property
    )
    native = (
        "from types import ModuleType\n"
        "module = ModuleType('frame_fixture')\n"
        f"vars(module)[{attribute!r}] = property\n"
        "namespace = {}\n"
        f"exec({source!r}, {{'__builtins__': vars(module)}}, namespace)\n"
        "print(namespace['result'] is property)\n"
    )
    assert (
        subprocess.check_output([sys.executable, "-c", native], text=True).strip()
        == "True"
    )


@pytest.mark.parametrize("rebind_parent", (False, True))
@pytest.mark.parametrize("alias", ("chosen", "pkg"))
def test_aliased_dotted_import_needs_actual_package_attribute_traversal(
    rebind_parent, alias
):
    package = ModuleType("pkg")
    package.__path__ = []
    child = ModuleType("pkg.child")
    child.property = property
    alternate = ModuleType("alternate")
    alternate.property = object
    package.child = child if rebind_parent else alternate
    prefix = (
        "import pkg\nimport alternate\npkg.child = alternate\n" if rebind_parent else ""
    )
    source = prefix + f"import pkg.child as {alias}\nresult = {alias}.property\n"
    resolution = _plain_module_fixture(source, (builtins, package, child, alternate))
    assert isinstance(resolution, OpenCapturedReference)
    assert resolution.violation is CapturedReferenceViolation.UNPROVED_IMPORT_TRAVERSAL
    native = (
        "import sys\nfrom types import ModuleType\n"
        "pkg = ModuleType('pkg')\npkg.__path__ = []\n"
        "child = ModuleType('pkg.child')\nchild.property = property\n"
        "alternate = ModuleType('alternate')\nalternate.property = object\n"
        + ("pkg.child = child\n" if rebind_parent else "pkg.child = alternate\n")
        + "sys.modules.update({'pkg': pkg, 'pkg.child': child, 'alternate': alternate})\n"
        + source
        + "\nprint(result is object)\n"
    )
    assert (
        subprocess.check_output([sys.executable, "-c", native], text=True).strip()
        == "True"
    )


def test_unaliased_dotted_import_still_binds_admitted_root_module():
    package = ModuleType("pkg")
    package.__path__ = []
    child = ModuleType("pkg.child")
    package.child = child
    resolution = _plain_module_fixture(
        "import pkg.child\nresult = pkg\n", (builtins, package, child)
    )
    assert isinstance(resolution, CapturedNativeObject)
    assert resolution.value is package
    native = (
        "import sys\nfrom types import ModuleType\n"
        "package = ModuleType('pkg')\npackage.__path__ = []\n"
        "child = ModuleType('pkg.child')\npackage.child = child\n"
        "sys.modules.update({'pkg': package, 'pkg.child': child})\n"
        "import pkg.child\nprint(pkg is package)\n"
    )
    assert (
        subprocess.check_output([sys.executable, "-c", native], text=True).strip()
        == "True"
    )


@pytest.mark.parametrize("registered_name", ("registered_one", "registered_two"))
def test_import_registry_aliases_authenticate_object_independently_of_display_name(
    registered_name,
):
    module = ModuleType("original_display_name")
    module.property = property
    module.__name__ = "changed_display_name"
    source = f"import {registered_name} as selected\nresult = selected.property\n"
    resolution = _plain_module_fixture(
        source,
        (builtins, module),
        registry_entries={"registered_one": module, "registered_two": module},
    )
    assert (
        resolution.require_native_identity(NativeDeclaration(property)).declaration
        is property
    )
    native = (
        "import sys\nfrom types import ModuleType\n"
        "module = ModuleType('original_display_name')\nmodule.property = property\n"
        "sys.modules.update({'registered_one': module, 'registered_two': module})\n"
        "module.__name__ = 'changed_display_name'\n"
        + source
        + "\nprint(result is property)\n"
    )
    assert (
        subprocess.check_output([sys.executable, "-c", native], text=True).strip()
        == "True"
    )


def test_display_name_cannot_impersonate_actual_builtin_import():
    frame = ModuleType("builtins")
    frame.property = object
    frame.__import__ = builtins.__import__
    source = "import builtins\nresult = builtins.property\n"
    resolution = _plain_module_fixture(
        source,
        (frame,),
        vars(frame),
        registry_entries={"registered_frame_fixture": frame},
    )
    assert isinstance(resolution, OpenCapturedReference)
    assert resolution.violation is CapturedReferenceViolation.UNADMITTED_IMPORT
    native = (
        "import sys\nfrom types import ModuleType\n"
        "frame = ModuleType('builtins')\nframe.property = object\nframe.__import__ = __import__\n"
        "sys.modules['registered_frame_fixture'] = frame\n"
        "namespace = {}\n"
        f"exec({source!r}, {{'__builtins__': vars(frame)}}, namespace)\n"
        "print(namespace['result'] is property)\n"
    )
    assert (
        subprocess.check_output([sys.executable, "-c", native], text=True).strip()
        == "True"
    )


def test_unregistered_display_names_never_manufacture_import_handles():
    module = ModuleType("unregistered_display_name")
    with _module_registry({"real_registration": module}):
        island = InitialNativeIsland((builtins, module))
        assert island.module("real_registration").value is module
        assert (
            island.module("unregistered_display_name").violation
            is CapturedReferenceViolation.UNADMITTED_IMPORT
        )
    dangling = InitialNativeIsland((builtins, module))
    assert (
        dangling.module("real_registration").violation
        is CapturedReferenceViolation.UNADMITTED_IMPORT
    )
    assert (
        dangling.module("unregistered_display_name").violation
        is CapturedReferenceViolation.UNADMITTED_IMPORT
    )


def test_unregistered_module_can_still_supply_actual_frame_builtin_dictionary():
    frame = ModuleType("unregistered_frame_fixture")
    frame.property = property
    resolution = _plain_module_fixture(
        "result = property\n",
        (frame,),
        vars(frame),
        registry_entries={},
    )
    assert (
        resolution.require_native_identity(NativeDeclaration(property)).declaration
        is property
    )


def test_import_registry_capture_is_eager_immutable_and_does_not_compare_values():
    class HostileValue:
        def __eq__(self, other):
            raise AssertionError("Registry values must not be compared")

        def __hash__(self):
            raise AssertionError("Registry values must not be hashed")

    initial = ModuleType("display")
    replacement = ModuleType("display")
    with _module_registry(
        {"stable_handle": initial, "irrelevant_value": HostileValue()}
    ):
        island = InitialNativeIsland((builtins, initial))
        sys.modules["stable_handle"] = replacement
        # Inspect the admission receipt, not a query claiming the changed ambient
        # registry still satisfies the mandatory effect authority's contract.
        assert island.modules_by_name["stable_handle"] is initial
        assert "irrelevant_value" not in island.modules_by_name
        with pytest.raises(TypeError):
            island.modules_by_name["stable_handle"] = replacement


def test_separate_initial_registry_admissions_are_distinct_query_authorities():
    module = ModuleType("display")
    with _module_registry({"first_admission": module, "later_alias": None}):
        before = InitialNativeIsland((builtins, module))
        sys.modules["later_alias"] = module
        after = InitialNativeIsland((builtins, module))
        assert before != after
        assert len({before, after}) == 2
        assert "later_alias" not in before.modules_by_name
        assert after.modules_by_name["later_alias"] is module
