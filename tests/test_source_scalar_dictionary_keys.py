"""Dictionary slots retain native scalar equality, not lexical-name restrictions."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    NativeNamespace,
    NamespaceMemberInventory,
)
from nominal_refactor_advisor.native_compilation import NativeCreationBackend
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from test_source_distinct_item_stores import (
    BASE,
    assert_closed_stores,
    execution,
    item_statements,
)
from test_source_item_installation import tail_query
from test_registry_copied_class_mapping import _component, _copy
from test_namespace_contents import positioned_contents


@pytest.mark.parametrize("key", (None, False, True, 0, 1, 7, 2**80, "", "alpha"))
def test_scalar_item_installation_preserves_original_value_and_inventory(key):
    environment = execution(BASE + f"REGISTRY[{key!r}] = Alpha\ntail = REGISTRY\n")
    assert_closed_stores(environment)
    query = tail_query(environment, key)
    original = environment.class_entry(environment.module.module.body[1]).definition
    assert query.resolve(environment.kernel).source_definition()[1] is original
    contents = NamespaceMemberInventory(
        environment.kernel, query.namespace, query.prefix
    )
    assert contents.names == frozenset((key,))
    assert contents.require_member(key).source_definition()[1] is original


@pytest.mark.parametrize(
    "stored, queried", ((False, 0), (0, False), (True, 1), (1, True))
)
def test_boolean_integer_aliases_select_the_same_slot(stored, queried):
    environment = execution(
        f"REGISTRY = {{}}\nREGISTRY[{stored!r}] = property\ntail = REGISTRY\n"
    )
    tail_query(environment, queried).resolve(
        environment.kernel
    ).require_native_identity(NativeDeclaration(property))


@pytest.mark.parametrize(
    "stored, replacement", ((False, 0), (0, False), (True, 1), (1, True))
)
def test_equal_scalar_keys_release_the_actual_previous_class(stored, replacement):
    environment = execution(
        BASE
        + f"REGISTRY[{stored!r}] = Alpha\nREGISTRY[{replacement!r}] = Beta\ntail = REGISTRY\n"
    )
    first, second = item_statements(environment)
    environment.require_item_write(first.targets[0])
    environment.require_item_write(second.targets[0])
    query = tail_query(environment, stored)
    beta = environment.class_entry(environment.module.module.body[2]).definition
    assert query.resolve(environment.kernel).source_definition()[1] is beta
    assert (
        replace(query, key=replacement)
        .resolve(environment.kernel)
        .source_definition()[1]
        is beta
    )
    with pytest.raises(ValueError):
        environment.capture_value(first.value).require_release()


@pytest.mark.parametrize("key", (None, False, True, 7))
def test_native_copy_keeps_scalar_keys_at_original_copy_cut(key):
    environment = execution(
        BASE + f"REGISTRY[{key!r}] = Alpha\ncopied = dict(REGISTRY)\n"
        "REGISTRY['later'] = Beta\ntail = copied\n"
    )
    query = tail_query(environment, key)
    original = environment.class_entry(environment.module.module.body[1]).definition
    assert query.resolve(environment.kernel).source_definition()[1] is original
    assert query.namespace.initial_names == frozenset((key,))
    assert replace(query, key="later").resolve(environment.kernel) is None


def test_initial_scalar_keys_use_dictionary_equality_without_lexical_coercion():
    namespace = NativeNamespace({False: property, "False": object, None: str})
    assert namespace.initial_names == frozenset((False, "False", None))
    namespace.member(0).require_native_identity(NativeDeclaration(property))
    namespace.member("False").require_native_identity(NativeDeclaration(object))


@pytest.mark.parametrize("key", (None, False, 0, True, 1, 7))
def test_registry_mapping_derives_key_support_from_complete_content_owner(key):
    environment = execution(
        BASE + f"REGISTRY[{key!r}] = Alpha\nREGISTRY['beta'] = Beta\n"
        "observed = dict(REGISTRY)\ntail = REGISTRY\n"
    )
    component = _component(environment)
    copied, positioned = _copy(environment), positioned_contents(environment)
    copied.require_same_mapping(positioned)
    for contents in (copied, positioned):
        component.require_class_mapping(contents, environment)


def test_integer_subclasses_never_run_key_or_release_protocols():
    events = []

    class Foreign(int):
        def __hash__(self):
            events.append("hash")
            return 0

        def __eq__(self, other):
            events.append("equality")
            return True

    key = Foreign(0)
    namespace = NativeNamespace({0: property})
    with pytest.raises(TypeError, match="exact scalar key"):
        namespace.member(key)
    with pytest.raises(ValueError, match="exact scalar key"):
        NativeCreationBackend.current().require_dictionary_scalar_store(key)
    with pytest.raises(ValueError, match="lifetime"):
        NativeCreationBackend.current().require_inert_instance_release(Foreign)
    assert events == []


@pytest.mark.parametrize("key", ((), (False,), 0.0, b"alpha", Ellipsis))
def test_unproved_key_families_remain_outside_scalar_storage(key):
    with pytest.raises(TypeError, match="exact scalar key"):
        NativeNamespace({key: property})
    with pytest.raises(ValueError, match="exact scalar key"):
        NativeCreationBackend.current().require_dictionary_scalar_store(key)
