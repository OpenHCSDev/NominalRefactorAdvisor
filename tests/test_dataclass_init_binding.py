"""The class index records how each dataclass field binds in `__init__` (#86)."""

import inspect
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.class_index import (
    CompactClassFamilyIndex,
    DataclassInitBinding,
)
from nominal_refactor_advisor.scan_cache import ScanCache


SOURCE = """\
from dataclasses import KW_ONLY, dataclass, field

@dataclass(frozen=True)
class Base:
    a: int
    b: int = field(kw_only=True)
    c: int = 0

@dataclass(frozen=True)
class Child(Base):
    d: int = 1
    _: KW_ONLY
    e: int
    f: int = field(default=2, kw_only=False)

@dataclass(frozen=True, kw_only=True)
class AllKeyword:
    g: int
    h: int = field(kw_only=False)

FLAG = True

@dataclass(kw_only=FLAG)
class Dynamic:
    i: int
"""


def authority(index, symbol):
    return index.product_authority_resolutions_by_symbol[symbol].authority


def test_each_field_records_its_init_binding_from_field_sentinel_and_decorator():
    modules = (SourceModule(Path("records.py"), "records", SOURCE).parse(),)
    with ScanCache.scope():
        index = CompactClassFamilyIndex.from_modules(modules)
        child = authority(index, "records.Child")
        assert {field.name: field.binding for field in child.fields} == {
            "a": DataclassInitBinding.POSITIONAL,
            "b": DataclassInitBinding.KEYWORD_ONLY,
            "c": DataclassInitBinding.POSITIONAL,
            "d": DataclassInitBinding.POSITIONAL,
            "e": DataclassInitBinding.KEYWORD_ONLY,
            "f": DataclassInitBinding.POSITIONAL,
        }
        assert index.construction_parameters("records.Child") == ("a", "c", "d", "f")
        assert index.construction_parameters("records.AllKeyword") == ("h",)
        dynamic = authority(index, "records.Dynamic")
        assert [field.binding for field in dynamic.fields] == [
            DataclassInitBinding.UNKNOWN
        ]
        with pytest.raises(ValueError, match="non-literal"):
            index.construction_parameters("records.Dynamic")


CONSTRUCTION_SOURCE = """\
from dataclasses import InitVar, dataclass, field

class Named:
    def label(self) -> str:
        return type(self).__name__

@dataclass
class Root:
    a: int
    b: int = 0

@dataclass
class Left(Root):
    c: int = 0

@dataclass
class Right(Root):
    a: int = 5
    d: int = 0

@dataclass
class Joined(Left, Right):
    e: int = 0

@dataclass
class Labelled(Named, Root):
    seed: InitVar[int] = 0
    cached: int = field(init=False, default=0)

    def __post_init__(self, seed: int) -> None:
        self.cached = seed

class Plain(Joined):
    pass
"""


def test_construction_parameters_follow_the_generated_init_along_the_mro():
    """Two dataclass parents, a mixin, `__post_init__`, an `InitVar` and a plain
    subclass all bind exactly what Python's own `__init__` takes."""

    namespace: dict[str, object] = {}
    exec(CONSTRUCTION_SOURCE, namespace)
    modules = (SourceModule(Path("shapes.py"), "shapes", CONSTRUCTION_SOURCE).parse(),)
    with ScanCache.scope():
        index = CompactClassFamilyIndex.from_modules(modules)
        for name in ("Root", "Left", "Right", "Joined", "Labelled", "Plain"):
            parameters = inspect.signature(namespace[name]).parameters.values()
            assert index.construction_parameters(f"shapes.{name}") == tuple(
                parameter.name
                for parameter in parameters
                if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            ), name


@pytest.mark.parametrize(
    ("body", "refusal"),
    (
        ("    def __init__(self, a: int) -> None:\n        self.a = a\n", "defines the `__init__`"),
        ("    def __new__(cls, *args, **kwargs):\n        return super().__new__(cls)\n", "`__new__`"),
        ("    def __init_subclass__(cls) -> None:\n        pass\n", "no closed runtime MRO"),
    ),
)
def test_construction_parameters_refuse_what_rebinds_the_call(body, refusal):
    source = f"from dataclasses import dataclass\n\n@dataclass\nclass Built:\n    a: int\n{body}"
    modules = (SourceModule(Path("built.py"), "built", source).parse(),)
    with ScanCache.scope():
        index = CompactClassFamilyIndex.from_modules(modules)
        with pytest.raises(ValueError, match=refusal):
            index.construction_parameters("built.Built")
