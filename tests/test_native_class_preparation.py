"""Fresh preparation follows native descriptor identity, never arbitrary hooks."""

from abc import ABCMeta
import subprocess
import sys
from textwrap import dedent

from metaclass_registry import AutoRegisterMeta
import pytest

from nominal_refactor_advisor.native_class_mro import NativeClassMroDeclaration
from nominal_refactor_advisor.native_compilation import (
    CPython311CreationBackend,
    CPython314CreationBackend,
    CPythonClassConstruction,
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)


def require_preparation(metaclass):
    return NativeCreationBackend.current().require_fresh_class_namespace(
        NativeClassMroDeclaration(metaclass)
    )


@pytest.mark.parametrize("metaclass", (type, ABCMeta, AutoRegisterMeta))
def test_loaded_native_metaclasses_select_the_exact_type_descriptor(metaclass):
    assert require_preparation(metaclass) is None


def test_exact_descriptor_alias_is_native_identity_not_owner_spelling():
    alias = type(
        "NativePrepareAlias", (type,), {"__prepare__": vars(type)["__prepare__"]}
    )
    declaration = NativeClassMroDeclaration(alias)
    assert declaration.member_owner("__prepare__") is alias
    assert (
        declaration.stored_namespace(alias)["__prepare__"] is vars(type)["__prepare__"]
    )
    assert require_preparation(alias) is None
    # Retained aliases deeper in a metaclass hierarchy use the same native binding.
    child = type("NativePrepareAliasChild", (alias,), {})
    assert require_preparation(child) is None


def test_source_class_instance_getter_does_not_replace_its_metaclass_lookup():
    events = []

    class Meta(type):
        def __getattribute__(cls, name):
            events.append(name)
            raise AssertionError("Only instances use this getter")

    assert require_preparation(Meta) is None
    assert events == []


def test_reimplemented_prepare_is_not_called_or_admitted():
    events = []

    class Meta(type):
        @classmethod
        def __prepare__(cls, *args, **kwargs):
            events.append("prepare")
            return {}

    with pytest.raises(ValueError, match="exact type descriptor"):
        require_preparation(Meta)
    assert events == []


def test_custom_prepare_descriptor_is_not_executed():
    events = []

    class Descriptor:
        def __get__(self, instance, owner):
            events.append("descriptor")
            raise AssertionError("Analyzer cannot invoke custom descriptors")

    meta = type("DescriptorMeta", (type,), {"__prepare__": Descriptor()})
    with pytest.raises(ValueError, match="exact type descriptor"):
        require_preparation(meta)
    assert events == []


def test_custom_metameta_getter_is_rejected_without_invocation():
    events = []

    class MetaMeta(type):
        def __getattribute__(cls, name):
            events.append(name)
            return type.__getattribute__(cls, name)

    class Meta(type, metaclass=MetaMeta):
        pass

    events.clear()
    with pytest.raises(ValueError, match="ordinary metaclass attribute lookup"):
        require_preparation(Meta)
    assert events == []


def test_metameta_data_descriptor_cannot_hide_behind_raw_member_owner():
    events = []

    class MetaMeta(type):
        pass

    class Meta(type, metaclass=MetaMeta):
        pass

    class Descriptor:
        def __get__(self, instance, owner):
            events.append("get")
            return lambda *args, **kwargs: {"injected": True}

        def __set__(self, instance, value):
            raise AssertionError("unused")

    MetaMeta.__prepare__ = Descriptor()
    assert NativeClassMroDeclaration(Meta).member_owner("__prepare__") is type
    with pytest.raises(ValueError, match="ordinary metaclass attribute lookup"):
        require_preparation(Meta)
    assert events == []


def test_nonstring_namespace_keys_are_rejected_before_name_lookup_callbacks():
    events = []

    class Key:
        def __hash__(self):
            return hash("__prepare__")

        def __eq__(self, other):
            events.append(other)
            return False

    meta = type("ForeignKeyMeta", (type,), {Key(): 1})
    events.clear()
    with pytest.raises(ValueError, match="unproved member keys"):
        require_preparation(meta)
    assert events == []


def test_mutating_prepare_invalidates_a_previous_success_without_stale_cache():
    meta = type("MutablePrepareMeta", (type,), {})
    declaration = NativeClassMroDeclaration(meta)
    backend = NativeCreationBackend.current()
    backend.require_fresh_class_namespace(declaration)
    meta.__prepare__ = classmethod(lambda cls, *args, **kwargs: {})
    with pytest.raises(ValueError, match="exact type descriptor"):
        backend.require_fresh_class_namespace(declaration)
    del meta.__prepare__
    backend.require_fresh_class_namespace(declaration)


@pytest.mark.parametrize("non_metaclass", (object, int, lambda: None, object()))
def test_a_class_or_callable_is_not_automatically_a_native_metaclass(non_metaclass):
    with pytest.raises(ValueError, match="ordinary metaclass attribute lookup"):
        require_preparation(non_metaclass)


def test_unsupported_or_wrong_runtime_backend_fails_closed():
    declaration = NativeClassMroDeclaration(AutoRegisterMeta)
    with pytest.raises(ValueError, match="fresh class namespace remains unproved"):
        SpanOnlyCreationBackend().require_fresh_class_namespace(declaration)
    other = (
        CPython314CreationBackend
        if sys.version_info[:2] == (3, 11)
        else CPython311CreationBackend
    )
    with pytest.raises(ValueError, match="actual native interpreter backend"):
        object.__new__(other).require_fresh_class_namespace(declaration)


def test_supported_backends_reuse_the_class_construction_owner():
    for backend in (CPython311CreationBackend, CPython314CreationBackend):
        assert (
            backend.require_fresh_class_namespace
            is CPythonClassConstruction.require_fresh_class_namespace
        )


def test_native_preparation_controls_in_disposable_process():
    program = dedent("""
        from abc import ABCMeta
        from metaclass_registry import AutoRegisterMeta
        alias = type('Alias', (type,), {'__prepare__': vars(type)['__prepare__']})
        for meta in (type, ABCMeta, AutoRegisterMeta, alias):
            first = meta.__prepare__('First', ())
            second = meta.__prepare__('Second', (), unused_keyword=object())
            assert type(first) is type(second) is dict
            assert not first and not second and first is not second
            first['retained'] = 1
            assert not second

        class HookMeta(type):
            @classmethod
            def __prepare__(cls, *args, **kwargs):
                return {'injected': True}
        assert HookMeta.__prepare__('Target', ()) == {'injected': True}

        class MetaMeta(type):
            pass
        class Meta(type, metaclass=MetaMeta):
            pass
        class Descriptor:
            def __get__(self, instance, owner):
                return lambda *args, **kwargs: {'metameta': True}
            def __set__(self, instance, value):
                pass
        MetaMeta.__prepare__ = Descriptor()
        assert Meta.__prepare__('Target', ()) == {'metameta': True}
    """)
    result = subprocess.run(
        [sys.executable, "-c", program], text=True, capture_output=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
