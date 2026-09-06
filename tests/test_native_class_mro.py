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


def test_custom_metaclass_mro_is_not_assumed_to_be_c3() -> None:
    class Custom(type):
        def mro(cls):
            return super().mro()

    class Leaf(metaclass=Custom):
        pass

    with pytest.raises(ValueError, match="custom MRO"):
        _ = NativeClassMroDeclaration(Leaf).mro_type


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
