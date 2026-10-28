"""The class index records how each dataclass field binds in `__init__` (#86)."""

from pathlib import Path

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
        assert child.positional_parameter_names() == ("a", "c", "d", "f")
        assert authority(index, "records.AllKeyword").positional_parameter_names() == (
            "h",
        )
        dynamic = authority(index, "records.Dynamic")
        assert [field.binding for field in dynamic.fields] == [
            DataclassInitBinding.UNKNOWN
        ]
        try:
            dynamic.positional_parameter_names()
        except ValueError as error:
            assert "not a literal" in str(error)
        else:
            raise AssertionError("an unknown binding must refuse")
