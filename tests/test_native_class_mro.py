"""Native class MRO projection reuses Python C3 without rerunning user hooks."""

from abc import ABC, ABCMeta
from pathlib import Path
import runpy
from typing import Generic, TypeVar

import pytest
from metaclass_registry import AutoRegisterMeta

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


def test_actual_metaclass_parent_constructor_uses_current_c3_owner() -> None:
    native = NativeClassMroDeclaration(AutoRegisterMeta)
    assert native.member_owner("__new__", start_after=AutoRegisterMeta) is ABCMeta
    selected = native.python_constructor(start_after=AutoRegisterMeta)
    assert selected is vars(ABCMeta)["__new__"].__func__
    assert selected is super(AutoRegisterMeta, AutoRegisterMeta).__new__
    assert selected.__closure__[0].cell_contents is ABCMeta
    assert native.member_owner("__new__", start_after=ABCMeta) is type
    # Selecting the native parent is not a Python body or construction proof.
    with pytest.raises(ValueError, match="exact Python constructor source"):
        native.python_constructor(start_after=ABCMeta)


def test_super_lookup_follows_sibling_before_common_ancestor() -> None:
    class Root(type):
        marker = "root"

    class Left(Root):
        marker = "left"

    class Right(Root):
        marker = "right"

    class Diamond(Left, Right):
        pass

    native = NativeClassMroDeclaration(Diamond)
    assert native.member_owner("marker") is Left
    assert native.member_owner("marker", start_after=Left) is Right
    assert native.member_owner("marker", start_after=Right) is Root
    assert super(Left, Diamond).marker == vars(Right)["marker"]
    assert native.member_owner("marker", start_after=object) is None


def test_super_lookup_rejoins_current_mro_on_same_owner() -> None:
    class Left(type):
        marker = "left"

    class Right(type):
        marker = "right"

    class Diamond(Left, Right):
        pass

    native = NativeClassMroDeclaration(Diamond)
    assert native.member_owner("marker", start_after=Left) is Right
    original_bases = Diamond.__bases__
    try:
        Diamond.__bases__ = (Right, Left)
        assert native.member_owner("marker", start_after=Left) is None
        assert native.member_owner("marker", start_after=Right) is Left
    finally:
        Diamond.__bases__ = original_bases
    assert native.member_owner("marker", start_after=Left) is Right


def test_foreign_super_start_does_not_run_identity_protocols() -> None:
    events = []

    class EqualMeta(type):
        def __eq__(cls, other):
            events.append("equality")
            return True

        def __repr__(cls):
            events.append("representation")
            return "AutoRegisterMeta"

    foreign = EqualMeta("AutoRegisterMeta", (), {})
    with pytest.raises(ValueError, match="start owner is absent"):
        NativeClassMroDeclaration(AutoRegisterMeta).member_owner(
            "__new__", start_after=foreign
        )
    assert events == []


def test_super_member_selection_does_not_invoke_descriptor() -> None:
    events = []

    class Descriptor:
        def __get__(self, instance, owner):
            events.append("descriptor")
            raise AssertionError("Only stored member selection is being proved")

    class Root(type):
        marker = Descriptor()

    class Leaf(Root):
        pass

    assert (
        NativeClassMroDeclaration(Leaf).member_owner("marker", start_after=Leaf) is Root
    )
    assert events == []


def test_member_lookup_rejects_active_name_before_hashing() -> None:
    events = []

    class ActiveName(str):
        def __hash__(self):
            events.append("hash")
            raise AssertionError("Query names must be exact native strings")

    with pytest.raises(ValueError, match="exact native member name"):
        NativeClassMroDeclaration(AutoRegisterMeta).member_owner(ActiveName("__new__"))
    assert events == []


def test_parent_constructor_revalidates_selected_sibling_cell(monkeypatch) -> None:
    class Left(type):
        def __new__(mcls, name, bases, namespace):
            return super().__new__(mcls, name, bases, namespace)

    class Right(type):
        def __new__(mcls, name, bases, namespace):
            return super().__new__(mcls, name, bases, namespace)

    class Diamond(Left, Right):
        pass

    native = NativeClassMroDeclaration(Diamond)
    selected = native.python_constructor(start_after=Left)
    assert selected is vars(Right)["__new__"].__func__
    with monkeypatch.context() as mutation:
        mutation.setattr(selected.__closure__[0], "cell_contents", Left)
        with pytest.raises(ValueError, match="selected MRO owner"):
            native.python_constructor(start_after=Left)
    assert native.python_constructor(start_after=Left) is selected


def test_parent_constructor_revalidates_current_descriptor(monkeypatch) -> None:
    class Parent(type):
        def __new__(mcls, name, bases, namespace):
            return super().__new__(mcls, name, bases, namespace)

    class Leaf(Parent):
        pass

    native = NativeClassMroDeclaration(Leaf)
    selected = native.python_constructor(start_after=Leaf)
    with monkeypatch.context() as mutation:
        mutation.setattr(Parent, "__new__", object())
        with pytest.raises(ValueError, match="exact Python constructor source"):
            native.python_constructor(start_after=Leaf)
    assert native.python_constructor(start_after=Leaf) is selected


def test_current_super_mro_dsl_preserves_input_and_chains_signature_edits() -> None:
    path = "nominal_refactor_advisor/native_class_mro.py"
    source = (
        "class NativeClassMroDeclaration:\n"
        "    def member_owner(self, name): return None\n"
        "    def python_constructor(self): return self.member_owner('__new__')\n"
    )
    snapshot = CodemodSourceSnapshot.from_source_mapping({path: source})
    plan = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "docs/examples/current_super_mro_lookup.py"
        )
    )["current_super_mro_plan"](snapshot)
    result = plan.simulate(snapshot)
    assert result.is_clean and result.stage_count == 4
    module = result.final_snapshot.parsed_module_for_source_path(path).module
    lookup, constructor = module.body[0].body
    assert lookup.args.kwonlyargs[0].arg == "start_after"
    assert constructor.args.kwonlyargs[0].arg == "start_after"
    call = constructor.body[0].value
    assert call.keywords[0].arg == "start_after"
    assert call.keywords[0].value.id == "start_after"
    assert snapshot.sources_by_file_path[path] == source
    assert result.simulation.changed_file_paths == (path,)
