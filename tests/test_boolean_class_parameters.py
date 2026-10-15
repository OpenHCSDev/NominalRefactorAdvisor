import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import SourceModule, module_syntax_index
from nominal_refactor_advisor.product_flow_authority import SourceProductFlowRepository


def test_nested_class_parameter_inverts_only_a_declared_public_boolean_classvar():
    source = """
from typing import ClassVar as Flag
class ThemeColour(metaclass=UnknownRegistry):
    background: Flag[bool] = False
    raw: bool = False
    _private: Flag[bool] = False
    maybe: Flag[bool | None] = None
class Derived(ThemeColour):
    pass
class Colours:
    def changed(self):
        def read(member: type[Derived]):
            return (not member.background, not member.raw,
                    not member._private, not member.maybe)
        return read
"""
    parsed = SourceModule(Path("pkg/colours.py"), "pkg.colours", source).parse()
    repository = SourceProductFlowRepository.from_modules((parsed,))
    attributes = [
        node.operand
        for node in ast.walk(parsed.module)
        if isinstance(node, ast.UnaryOp)
    ]
    outcomes = {
        attribute.attr: repository.declared_boolean_inversion(
            "pkg.colours.Colours.changed.read", attribute
        )
        for attribute in attributes
    }
    assert outcomes == {
        "background": True,
        "raw": False,
        "_private": False,
        "maybe": None,
    }
    reference = attributes[0].value
    assert (
        repository.function_parameter_type_name(
            "pkg.colours.Colours.changed.read", reference
        )
        is None
    )


def test_type_factory_and_outer_bindings_must_not_be_guessed():
    for shadow in ("type = object", "ThemeColour = object"):
        source = f"""
from typing import ClassVar
class ThemeColour:
    background: ClassVar[bool] = False
class Colours:
    def changed(self):
        {shadow}
        def read(member: type[ThemeColour]):
            return not member.background
        return read
"""
        parsed = SourceModule(Path("pkg/colours.py"), "pkg.colours", source).parse()
        repository = SourceProductFlowRepository.from_modules((parsed,))
        attribute = next(
            node.operand
            for node in ast.walk(parsed.module)
            if isinstance(node, ast.UnaryOp)
        )
        assert (
            repository.declared_boolean_inversion(
                "pkg.colours.Colours.changed.read", attribute
            )
            is None
        )


def test_property_getter_keeps_nominal_receiver_ownership():
    parsed = SourceModule(
        Path("pkg/views.py"),
        "pkg.views",
        """
class View:
    @property
    def order(self):
        return self.form if self.form is not None else None
""",
    ).parse()
    repository = SourceProductFlowRepository.from_modules((parsed,))
    reference = next(
        node
        for node in ast.walk(parsed.module)
        if isinstance(node, ast.Name) and node.id == "self"
    )
    assert (
        repository.function_parameter_type_name("pkg.views.View.order", reference)
        == "pkg.views.View"
    )


def test_class_receivers_do_not_borrow_instance_properties():
    parsed = SourceModule(
        Path("pkg/views.py"),
        "pkg.views",
        """
from typing import ClassVar
class View:
    background: ClassVar[bool] = False
    @property
    def ready(self) -> bool:
        return True
    @classmethod
    def query(cls, instance: View):
        return not cls.background, not cls.ready, not instance.ready
""",
    ).parse()
    repository = SourceProductFlowRepository.from_modules((parsed,))
    attributes = [
        node.operand
        for node in ast.walk(parsed.module)
        if isinstance(node, ast.UnaryOp)
    ]
    assert [
        repository.declared_boolean_inversion("pkg.views.View.query", attribute)
        for attribute in attributes
    ] == [True, False, True]


def test_lambda_and_comprehension_rebinding_remain_unresolved():
    for expression in (
        "lambda member: not member.ready",
        "[not member.ready for member in others]",
    ):
        source = f"""
class Field:
    @property
    def ready(self) -> bool:
        return True
class View:
    def query(self, member: Field, others):
        return {expression}
"""
        parsed = SourceModule(Path("pkg/views.py"), "pkg.views", source).parse()
        repository = SourceProductFlowRepository.from_modules((parsed,))
        attribute = next(
            node.operand
            for node in ast.walk(parsed.module)
            if isinstance(node, ast.UnaryOp)
        )
        assert (
            repository.declared_boolean_inversion("pkg.views.View.query", attribute)
            is None
        )
