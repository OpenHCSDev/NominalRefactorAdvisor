"""Real native root preparation retains the source namespace and open result law."""

import ast
from copy import copy, deepcopy
from dataclasses import replace
from types import FunctionType

import pytest
from metaclass_registry import AutoRegisterMeta

from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    NamespaceMemberInventory,
    NativeTypePremise,
)
from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.native_subscription import ClassVariableSubscription
from nominal_refactor_advisor.source_entry import (
    DeclaredNativeOperationBehavior,
    DeclaredOperationCompletion,
)
from nominal_refactor_advisor.source_execution import (
    AutoRegisterClassEntry,
    SourceCreatedFunctionCapture,
    SourceModuleExecution,
)
from test_documentation_store import execution
from test_registry_native_registration_controls import native_control

SOURCE = """REGISTRY = {}
class Family(metaclass=Creator):
    __registry__ = REGISTRY
    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key = None
    def example(self, value=1):
        raise AssertionError("method body must never run")
class Alpha(Family):
    registry_key = "alpha"
"""


def prepared_execution(source=SOURCE, *, metaclass=AutoRegisterMeta, conditions=True):
    original = execution(source)
    nodes = tuple(
        node for node in original.module.module.body if isinstance(node, ast.ClassDef)
    )
    entry = replace(
        original.entry,
        bindings={
            **original.entry.initial_entries,
            "Creator": CapturedNativeObject(metaclass),
        },
        declared_operation_conditions=(
            tuple(
                DeclaredNativeOperationBehavior(
                    original.source.definition_operation(node), AutoRegisterClassEntry
                )
                for node in nodes
            )
            if conditions
            else ()
        ),
    )
    return SourceModuleExecution(entry), nodes


def test_actual_root_uses_canonical_factory_and_complete_prepared_namespace():
    environment, (root, leaf) = prepared_execution()
    entry = environment.class_entry(root)
    assert type(entry) is AutoRegisterClassEntry
    assert environment.class_entry(root) is entry
    assert entry.metaclass_declaration.declaration is AutoRegisterMeta
    assert entry.frame.locals is entry
    assert entry.completion_prefix.endpoint.frame is entry.frame
    names = NamespaceMemberInventory(
        environment.kernel, entry, entry.completion_prefix
    ).names
    assert {
        "__registry__",
        "__registry_key__",
        "__skip_if_no_key__",
        "registry_key",
        "example",
    } <= names
    for name in names:
        entry.completion_member(name).require_closed()
    assert (
        entry.completion_member("__registry_key__").require_native_text()
        == "registry_key"
    )
    method = entry.completion_member("example")
    assert isinstance(method, SourceCreatedFunctionCapture)
    assert method.native_type is FunctionType
    module = entry.completion_member("__module__")
    assert isinstance(module, NativeTypePremise) and module.native_type is str
    with pytest.raises(ValueError, match="scalar contents remain unproved"):
        module.require_native_text()
    assert "completed" not in vars(entry)
    assert not environment._pending
    # The original implicit base read cannot skip the root's missing result law.
    with pytest.raises(ValueError):
        environment.capture_value(leaf.bases[0]).require_closed()
    assert "completed" not in vars(entry)
    assert not environment._pending


@pytest.mark.parametrize(
    "condition_kind", ("absent", "completion_only", "wrong_protocol")
)
def test_selection_is_not_driven_by_supplied_conditions(condition_kind):
    environment, (root, _) = prepared_execution(conditions=False)
    operation = environment.source.definition_operation(root)
    conditions = {
        "absent": (),
        "completion_only": (DeclaredOperationCompletion(operation),),
        "wrong_protocol": (
            DeclaredNativeOperationBehavior(operation, ClassVariableSubscription),
        ),
    }
    environment = SourceModuleExecution(
        replace(
            environment.entry,
            bindings=dict(environment.entry.initial_entries),
            declared_operation_conditions=conditions[condition_kind],
        )
    )
    entry = environment.class_entry(root)
    assert type(entry) is AutoRegisterClassEntry
    with pytest.raises(ValueError):
        _ = entry.frame
    assert "frame" not in vars(entry)


@pytest.mark.parametrize("wrong", (type, int, object))
def test_explicit_metaclass_identity_cannot_be_replaced_by_source_spelling(wrong):
    environment, (root, _) = prepared_execution(metaclass=wrong)
    with pytest.raises(ValueError, match="required native declaration"):
        environment.class_entry(root)
    assert root not in environment._class_entries


@pytest.mark.parametrize(
    "header", ("object, metaclass=Creator", "metaclass=Creator, option=True")
)
def test_native_preparation_does_not_supply_unproved_header_binding(header):
    environment, (root,) = prepared_execution(f"class Family({header}): pass\n")
    entry = environment.class_entry(root)
    with pytest.raises(ValueError, match="header binding remains unproved"):
        entry.require_preparation()
    with pytest.raises(ValueError, match="header binding remains unproved"):
        _ = entry.frame


@pytest.mark.parametrize("copier", (copy, deepcopy))
@pytest.mark.parametrize("source", (SOURCE, "class Family: pass\n"))
def test_factory_only_caches_original_definitions(copier, source):
    environment, nodes = prepared_execution(source)
    node = copier(nodes[0])
    with pytest.raises(ValueError):
        environment.class_entry(node)
    assert node not in environment._class_entries


def test_unknown_body_effect_is_not_closed_by_native_preparation():
    environment, (root,) = prepared_execution(
        "class Family(metaclass=Creator):\n    unknown()\n"
    )
    entry = environment.class_entry(root)
    assert entry.frame.locals is entry
    with pytest.raises(ValueError):
        _ = entry.completion_prefix
    assert "completed" not in vars(entry)
    assert not environment._pending


def test_prepared_body_never_supplies_a_plain_type_result():
    environment, (root, _) = prepared_execution()
    entry = environment.class_entry(root)
    _ = entry.completion_prefix
    with pytest.raises(ValueError, match="External source interference"):
        entry.result()
    with pytest.raises(ValueError, match="registration class result"):
        entry._created_result()


def test_missing_backend_capability_keeps_prepared_frame_open(monkeypatch):
    environment, (root, _) = prepared_execution()
    entry = environment.class_entry(root)
    # Keep actual compilation receipts, but withdraw only backend preparation.
    _ = entry.initial_entries
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    with pytest.raises(ValueError, match="fresh class namespace remains unproved"):
        _ = entry.frame
    assert "frame" not in vars(entry)


@pytest.mark.parametrize("native_metaclass", (False, True))
def test_direct_preparation_authenticates_original_builder_for_every_family(
    native_metaclass,
):
    outcome = native_control(
        """
import builtins
import json
import sys
sys.path.insert(0, 'tests')
from test_native_source_class_preparation import prepared_execution

original = builtins.__build_class__
calls = []
def replacement(*args, **kwargs):
    calls.append(True)
    raise AssertionError('Analyzer must never invoke the target builder')

builtins.__build_class__ = replacement
try:
    header = '(metaclass=Creator)' if bool(int(sys.argv[1])) else ''
    environment, (root,) = prepared_execution('class Family' + header + ': pass\\n')
    entry = environment.class_entry(root)
    outcomes = []
    for query in (entry.require_preparation, lambda: entry.frame):
        try:
            query()
        except ValueError:
            outcomes.append(True)
        else:
            outcomes.append(False)
finally:
    builtins.__build_class__ = original
print(json.dumps({'rejected': outcomes, 'calls': calls}))
""",
        native_metaclass,
    )
    assert outcome == {"rejected": [True, True], "calls": []}
