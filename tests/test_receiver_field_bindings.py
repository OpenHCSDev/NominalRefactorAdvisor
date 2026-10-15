"""Nominal constructor fields and public query lookup use the existing owners."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.class_index import (
    FunctionNominalParameterBindingAuthority,
    ModuleNominalBindingAuthority,
)
from nominal_refactor_advisor.product_flow_authority import SourceProductFlowRepository


def module(source, name="pkg.models"):
    return SourceModule(Path(name.replace(".", "/") + ".py"), name, source).parse()


@pytest.mark.parametrize(
    "body, expected",
    [
        ("self.field = field", "pkg.declarations.DeclaredField"),
        ("field = other\nself.field = field", None),
        ("self = other\nself.field = field", None),
        ("self.field = field\nself.field = other", None),
        ("self.field = field\ndel self.field", None),
        ("if other:\n    self.field = field", None),
        ("self.field = field\ndef mutate():\n    self.field = other", None),
        ("self.field = field\nself.field += other", None),
    ],
)
def test_constructor_fields_require_one_stable_binding(body, expected):
    source = "from .declarations import DeclaredField as Field\nclass View:\n    def __init__(self, field: Field, other):\n"
    source += "\n".join("        " + line for line in body.splitlines()) + "\n"
    parsed = module(source)
    function = parsed.module.body[1].body[0]
    authority = FunctionNominalParameterBindingAuthority(
        ModuleNominalBindingAuthority(parsed), function
    )
    assert authority.type_name_for_receiver_field("field") == expected
    assert "field" in authority.receiver_field_type_names


def test_public_query_and_inherited_field_use_qualified_bindings():
    declarations = module(
        """
from builtins import property as query, bool as Boolean
class DeclaredField:
    value: str
    @query
    def has_default(self) -> Boolean:
        return True
    @property
    def _private(self) -> bool:
        return True
""",
        "pkg.declarations",
    )
    views = module("""
from .declarations import DeclaredField as Field
class FieldView(UnknownWidget):
    def __init__(self, field: Field):
        self.field = field
class ValueView(FieldView):
    pass
class ReboundView(FieldView):
    def __init__(self, field):
        self.field = field
class AmbiguousView(UnknownWidget, FieldView):
    pass
""")
    repository = SourceProductFlowRepository.from_modules((declarations, views))
    assert (
        repository.receiver_field_type_name("pkg.models.ValueView", "field")
        == "pkg.declarations.DeclaredField"
    )
    assert (
        repository.receiver_field_type_name("pkg.models.ReboundView", "field") is None
    )
    assert (
        repository.receiver_field_type_name("pkg.models.AmbiguousView", "field") is None
    )
    assert (
        repository.public_boolean_property(
            "pkg.declarations.DeclaredField", "has_default"
        )
        is True
    )
    assert (
        repository.public_boolean_property("pkg.declarations.DeclaredField", "value")
        is False
    )
    assert (
        repository.public_boolean_property("pkg.declarations.DeclaredField", "_private")
        is False
    )
    assert (
        repository.public_boolean_property("pkg.declarations.DeclaredField", "unknown")
        is None
    )


def test_closed_multiple_inheritance_uses_existing_c3_order():
    parsed = module("""
class DeclaredField:
    @property
    def ready(self) -> bool:
        return True
class Root:
    pass
class First(Root):
    pass
class Second(Root):
    def __init__(self, field: DeclaredField):
        self.field = field
class View(First, Second):
    pass
""")
    repository = SourceProductFlowRepository.from_modules((parsed,))
    assert (
        repository.receiver_field_type_name("pkg.models.View", "field")
        == "pkg.models.DeclaredField"
    )


def test_class_shadowed_bool_is_not_a_proven_boolean_query():
    parsed = module("""
class Field:
    bool = object
    @property
    def ready(self) -> bool:
        return True
""")
    repository = SourceProductFlowRepository.from_modules((parsed,))
    assert repository.public_boolean_property("pkg.models.Field", "ready") is None
