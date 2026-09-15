"""Native class MRO projection reuses Python C3 without rerunning user hooks."""

from abc import ABC
from typing import Generic, TypeVar

import pytest

from nominal_refactor_advisor.class_mro import DeclarationMroType
from nominal_refactor_advisor.detectors._base import (
    CrossModuleCandidateDetector,
    CrossModuleCollectorCandidateDetector,
)
from nominal_refactor_advisor.native_class_mro import NativeClassMroDeclaration
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.source_native_mro import SourceNativeClassMro

T = TypeVar("T")


class Root(Generic[T], ABC):
    pass


class Left(Root[T]):
    pass


class Right(Root[T]):
    pass


class Diamond(Left[int], Right[int]):
    pass


@pytest.mark.parametrize(
    "declaration",
    (
        object,
        ABC,
        Generic,
        Root,
        Diamond,
        CrossModuleCandidateDetector,
        CrossModuleCollectorCandidateDetector,
    ),
)
def test_projected_mro_matches_loaded_python_identity(declaration: type) -> None:
    native = NativeClassMroDeclaration(declaration)
    projected = native.mro_type
    actual = tuple(
        (
            owner.declaration.declaration
            if isinstance(owner, DeclarationMroType)
            else owner
        )
        for owner in projected.__mro__
    )
    assert len(actual) == len(declaration.__mro__)
    assert all(
        left is right for left, right in zip(actual, declaration.__mro__, strict=True)
    )
    assert NativeClassMroDeclaration(declaration).mro_type is projected


def test_native_class_creation_hook_is_not_executed_again() -> None:
    calls = []

    class Hook:
        def __init_subclass__(cls):
            calls.append(cls)

    class Leaf(Hook):
        pass

    assert calls == [Leaf]
    _ = NativeClassMroDeclaration(Leaf).mro_type
    assert calls == [Leaf]


def test_native_generic_origin_requires_native_subscription() -> None:
    NativeClassMroDeclaration(Root).require_generic_origin()

    class Custom(Root[T]):
        def __class_getitem__(cls, item):
            return Root[item]

    with pytest.raises(ValueError, match="Custom class subscription"):
        NativeClassMroDeclaration(Custom).require_generic_origin()

    class IndexedMeta(type):
        def __getitem__(cls, item):
            return Root[item]

    class Indexed(Generic[T], metaclass=IndexedMeta):
        pass

    with pytest.raises(ValueError, match="Metaclass subscription"):
        NativeClassMroDeclaration(Indexed).require_generic_origin()


def test_native_roots_with_one_name_do_not_overwrite_identity() -> None:
    first = type("Root", (), {"__module__": "same"})
    second = type("Root", (), {"__module__": "same"})
    context = CodemodSourceSnapshot.from_modules(())
    with pytest.raises(ValueError):
        _ = SourceNativeClassMro(context, (first, second)).native_declarations
    indexed = SourceNativeClassMro(context, (first, first)).native_declarations
    assert indexed["same.Root"].declaration is first


def test_python_constructor_class_cell_belongs_to_selected_ancestor() -> None:
    class Creator(type):
        def __new__(mcls, name, bases, namespace):
            return super().__new__(mcls, name, bases, namespace)

    class Left(Creator):
        pass

    class Right(Creator):
        pass

    class Diamond(Left, Right):
        pass

    native = NativeClassMroDeclaration(Diamond)
    function = vars(Creator)["__new__"].__func__
    assert native.python_constructor() is function
    assert function.__closure__[0].cell_contents is Creator


def test_constructor_class_cell_is_revalidated_on_the_same_owner(monkeypatch) -> None:
    class Creator(type):
        def __new__(mcls, name, bases, namespace):
            return super().__new__(mcls, name, bases, namespace)

    native = NativeClassMroDeclaration(Creator)
    function = native.python_constructor()
    cell = function.__closure__[0]
    with monkeypatch.context() as mutation:
        mutation.setattr(cell, "cell_contents", type)
        assert vars(Creator)["__new__"].__func__ is function
        with pytest.raises(ValueError, match="selected MRO owner"):
            native.python_constructor()
    assert native.python_constructor() is function


def test_empty_constructor_class_cell_stays_unproved(monkeypatch) -> None:
    class Creator(type):
        def __new__(mcls, name, bases, namespace):
            return super().__new__(mcls, name, bases, namespace)

    native = NativeClassMroDeclaration(Creator)
    function = native.python_constructor()
    with monkeypatch.context() as mutation:
        mutation.delattr(function.__closure__[0], "cell_contents")
        with pytest.raises(ValueError, match="class cell is empty"):
            native.python_constructor()
    assert native.python_constructor() is function


def test_foreign_class_cell_protocols_are_not_invoked(monkeypatch) -> None:
    events = []

    class Foreign:
        def __eq__(self, other):
            events.append("equality")
            return True

        def __repr__(self):
            events.append("representation")
            return "Creator"

    class Creator(type):
        def __new__(mcls, name, bases, namespace):
            return super().__new__(mcls, name, bases, namespace)

    native = NativeClassMroDeclaration(Creator)
    function = native.python_constructor()
    with monkeypatch.context() as mutation:
        mutation.setattr(function.__closure__[0], "cell_contents", Foreign())
        with pytest.raises(ValueError, match="selected MRO owner"):
            native.python_constructor()
    assert events == []


def test_constructor_with_other_closure_roles_stays_unproved() -> None:
    def factory(callback):
        class Creator(type):
            def __new__(mcls, name, bases, namespace):
                callback()
                return super().__new__(mcls, name, bases, namespace)

        return Creator

    events = []
    creator = factory(lambda: events.append("callback"))
    with pytest.raises(ValueError, match="closure roles remain unproved"):
        NativeClassMroDeclaration(creator).python_constructor()
    assert events == []
