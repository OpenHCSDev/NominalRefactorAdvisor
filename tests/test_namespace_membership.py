"""Complete namespace membership comes from admitted storage and ordered writes."""

import ast
import builtins
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceViolation,
    CapturedSlotQuery,
    OpenCapturedReference,
)
from nominal_refactor_advisor.product_flow import (
    CompactAttributeTarget,
    CompactMutation,
)
from nominal_refactor_advisor.source_entry import SourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.class_namespace import SourceExecutionEffectEvidence


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("membership.py"), "membership", False, ast.parse(source), source
        )
    )


def observe(environment, node=None):
    if node is None:
        node = environment.module.module.body[-1].value
    read = environment.source.value_reads_by_node[node]
    value = environment.kernel.read(read)
    namespace = value.dictionary_namespace(environment.initial)
    assert not isinstance(namespace, OpenCapturedReference)
    return environment.kernel.namespace_names(
        namespace, read.context, read.use.position
    )


def test_membership_includes_completed_writes_but_not_the_result_binding():
    environment = execution("first = 1\n_private = 2\nresult = globals()\n")
    assert observe(environment) == frozenset(environment.entry.initial_entries) | {
        "first",
        "_private",
    }


def test_membership_at_an_earlier_read_does_not_include_later_writes():
    environment = execution("first = globals()\nlater = 1\nresult = globals()\n")
    initial = frozenset(environment.entry.initial_entries)
    assert observe(environment, environment.module.module.body[0].value) == initial
    assert observe(environment) == initial | {"first", "later"}


def test_initial_presence_does_not_require_the_values_identity():
    standard = execution("result = globals()\n")
    unknown = OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
    entry = SourceModuleEntryPremise(
        standard.source, standard.initial, {"unknown": unknown}, standard.entry.builtins
    )
    environment = SourceModuleExecution(entry)
    assert observe(environment) == {"unknown"}
    assert entry.member("unknown") is unknown


def test_native_module_attribute_store_updates_only_its_own_namespace():
    environment = execution(
        "import builtins\nbuiltins.NEW_MEMBER = property\nresult = vars(builtins)\n"
    )
    native = environment.initial.namespace_for_storage(vars(builtins))
    assert observe(environment) == frozenset(native.initial_entries) | {"NEW_MEMBER"}
    other = execution(
        "import builtins\nbuiltins.NEW_MEMBER = property\nresult = globals()\n"
    )
    assert observe(other) == frozenset(other.entry.initial_entries) | {"builtins"}
    assert "NEW_MEMBER" not in vars(builtins)


def test_copy_membership_uses_its_creation_cut_and_keyword_keys():
    environment = execution(
        "before = property\ncopied = dict(globals(), extra=object)\n"
        "later = object\nresult = copied\n"
    )
    assert observe(environment) == frozenset(environment.entry.initial_entries) | {
        "before",
        "extra",
    }


def test_class_body_and_completed_child_globals_use_their_original_frames():
    environment = execution(
        "before = property\nclass Body:\n    global exported\n"
        "    exported = property\n    local_only = object\n"
        "    inside = globals()\nresult = globals()\n"
    )
    initial = frozenset(environment.entry.initial_entries)
    owner = environment.module.module.body[1]
    assert observe(environment, owner.body[-1].value) == initial | {
        "before",
        "exported",
    }
    assert observe(environment) == initial | {"before", "exported", "Body"}


def test_initial_key_projection_is_cached_on_its_original_owner():
    environment = execution("copied = dict(globals())\nresult = copied\n")
    result = environment.capture_value(environment.module.module.body[-1].value)
    namespace = result.dictionary_namespace(environment.initial)
    assert namespace.initial_names is namespace.initial_names
    assert environment.entry.initial_names is environment.entry.initial_names


@pytest.mark.parametrize("depth", (2, 3))
def test_nested_class_global_writes_survive_completed_outer_activation(depth):
    lines = []
    for level in range(depth):
        lines.append("    " * level + f"class Owner{level}:")
    lines.extend(
        (
            "    " * depth + "global exported",
            "    " * depth + "exported = property",
            "result = globals()",
        )
    )
    environment = execution("\n".join(lines) + "\n")
    native = subprocess.run(
        [
            sys.executable,
            "-c",
            environment.module.source.replace(
                "result = globals()", "result = tuple(globals())"
            )
            + "print(sorted(name for name in result if not name.startswith('__')))\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert ast.literal_eval(native.stdout) == ["Owner0", "exported"]
    assert observe(environment) == frozenset(environment.entry.initial_entries) | {
        "exported",
        "Owner0",
    }


def test_completed_class_prefix_is_the_canonical_admission_receipt():
    environment = execution("class Owner:\n    member = 1\n")
    owner = environment.class_entry(environment.module.module.body[0])
    assert owner.completion_prefix is environment.required_prefix(owner.context, None)


def test_completed_class_prefix_requires_the_body_effects_to_be_closed():
    environment = execution("class Owner:\n    unknown()\n")
    owner = environment.class_entry(environment.module.module.body[0])
    with pytest.raises(ValueError):
        _ = owner.completion_prefix


@pytest.mark.parametrize(
    "source",
    (
        "class Body:\n    global exported\n    exported = 1\n",
        "def outer():\n    selected = 1\n    class Body:\n        nonlocal selected\n        selected = 2\n",
    ),
)
def test_binding_directives_do_not_create_runtime_value_obligations(source):
    module = ast.parse(source)
    evidence = SourceExecutionEffectEvidence.from_source(module)
    directives = tuple(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.Global, ast.Nonlocal))
    )
    assert directives
    assert all(site.trigger not in directives for site in evidence.sites)


def test_namespace_from_a_different_entry_cannot_supply_membership():
    environment = execution("result = globals()\n")
    other = SourceModuleEntryPremise(
        environment.source, environment.initial, {}, environment.entry.builtins
    )
    read = environment.source.value_reads_by_node[
        environment.module.module.body[0].value
    ]
    with pytest.raises(ValueError):
        environment.kernel.namespace_names(other, read.context, read.use.position)


def test_future_copy_cannot_supply_keys_at_an_earlier_execution_cut():
    environment = execution(
        "first = property\ncopied = dict(globals())\nresult = copied\n"
    )
    result = environment.capture_value(environment.module.module.body[-1].value)
    namespace = result.dictionary_namespace(environment.initial)
    read = environment.source.value_reads_by_node[
        environment.module.module.body[0].value
    ]
    with pytest.raises(ValueError):
        environment.kernel.namespace_names(namespace, read.context, read.use.position)


@pytest.mark.parametrize(
    "prefix",
    (
        "if property:\n    conditional = 1\n",
        "class First: pass\ndel First\n",
        "globals()[unknown_key()] = 1\n",
        "unknown()\n",
    ),
)
def test_unproved_membership_never_becomes_a_partial_key_set(prefix):
    environment = execution(prefix + "result = globals()\n")
    node = environment.module.module.body[-1].value
    read = environment.source.value_reads_by_node[node]
    with pytest.raises(ValueError):
        environment.kernel.namespace_names(
            environment.entry, read.context, read.use.position
        )


@pytest.mark.parametrize(
    "prefix",
    (
        "first = 1\n_private = 2\n",
        "first = 1\ndel first\n",
        "globals()['injected'] = 1\n",
        "namespace = globals()\nnamespace['injected'] = 1\n",
    ),
)
def test_membership_matches_native_module_execution(tmp_path, prefix):
    source = prefix + "observed = tuple(globals())\n"
    path = tmp_path / "membership.py"
    path.write_text(
        source
        + "print(sorted(name for name in observed if not name.startswith('__')))\n"
    )
    native = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, check=True
    )
    environment = execution(source)
    node = environment.module.module.body[-1].value.args[0]
    names = observe(environment, node)
    assert sorted(
        name for name in names if not name.startswith("__")
    ) == ast.literal_eval(native.stdout)


@pytest.mark.parametrize("operation", ("read", "store", "effect"))
def test_attribute_consumers_use_the_receivers_declared_namespace(operation):
    class RestrictedAttributeCapture(CapturedNativeObject):
        def attribute_namespace(self, initial, attribute):
            raise ValueError("receiver-owned namespace restriction")

    environment = execution("import builtins\nbuiltins.marker = property\n")
    mutation = next(
        site.event
        for site in environment.source.operations
        if isinstance(site.event, CompactMutation)
        and isinstance(site.event.target, CompactAttributeTarget)
    )
    context = environment.entry.context
    prefix = environment.required_prefix(context, mutation.position)
    query = CapturedSlotQuery(
        environment.initial.namespace_for_storage(vars(builtins)),
        "marker",
        prefix,
        frozenset(),
    )
    capture = RestrictedAttributeCapture(builtins)
    with pytest.raises(ValueError, match="receiver-owned namespace restriction"):
        if operation == "read":
            capture.access(
                environment.kernel, "marker", context, mutation.position, frozenset()
            )
        elif operation == "store":
            capture.require_attribute_write(
                environment.kernel, "marker", context, mutation.position
            )
        else:
            capture.write_effect(environment.kernel, query, mutation)
