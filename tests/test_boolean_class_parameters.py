import ast
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("expression", [
    "[not role.background for role in frame.stops]",
    "{not role.background for role in frame.stops}",
    "{role: not role.background for role in dict.fromkeys(frame.stops)}",
    "(not role.background for role in dict.fromkeys(frame.stops))",
    "[[not role.background for other in ()] for role in frame.stops]",
    "[not role.background for frame in (frame,) for role in frame.stops]",
])
def test_declared_tuple_class_elements_share_external_member_lookup(expression):
    # The last case must remain open: the inner `frame` is an iteration target,
    # not the annotated parameter. Its outer-evaluated first iterable is separate.
    expected = None if "for frame in" in expression else True
    assert _comprehension_query(expression) is expected


@pytest.mark.parametrize(
    "expression, before, body, expected",
    [
        ("[not role.raw for role in frame.stops]", "", "", False),
        ("[not role._private for role in frame.stops]", "", "", False),
        ("[not role.background for role in others]", "", "", None),
        (
            "[not role.background for role in dict.fromkeys(frame.stops)]",
            "dict = factory",
            "",
            None,
        ),
        (
            "[not role.background for role in dict.fromkeys(frame.stops)]",
            "",
            "dict = factory",
            None,
        ),
        (
            "[not role.background for role in dict.fromkeys(frame.stops)]",
            "",
            "frame = other",
            None,
        ),
        (
            "[not role.background for dict in others for role in dict.fromkeys(frame.stops)]",
            "",
            "",
            None,
        ),
        (
            "[not role.background for role in frame.stops for role in others]",
            "",
            "",
            None,
        ),
        ("[not role.background for role, other in frame.stops]", "", "", None),
        ("[not role.background async for role in frame.stops]", "", "", None),
        ("[not role.background for role in dict.fromkeys(*frame.stops)]", "", "", None),
        (
            "[not role.background for role in factory.fromkeys(frame.stops)]",
            "",
            "",
            None,
        ),
        (
            "[[not role.background for role in others] for role in frame.stops]",
            "",
            "",
            None,
        ),
    ],
)
def test_comprehension_nonqueries_and_unresolved_bindings_remain_countable(
    expression, before, body, expected
):
    assert _comprehension_query(expression, before=before, body=body) is expected


@pytest.mark.parametrize(
    "annotation, expected",
    [
        ("tuple[type[Derived], ...]", True),
        ('"tuple[type[Derived], ...]"', True),
        ("tuple[type[Derived], type[Derived]]", True),
        ("tuple[type[Derived], type[ThemeColour]]", None),
        ("tuple[type[Derived] | None, ...]", None),
        ("tuple[Derived, ...]", None),
        ("list[type[Derived]]", None),
    ],
)
def test_collection_annotation_preserves_class_identity_and_uncertainty(
    annotation, expected
):
    assert (
        _comprehension_query(
            "[not role.background for role in frame.stops]", annotation=annotation
        )
        is expected
    )


def test_class_parameter_capture_is_not_shadowed_by_an_unrelated_comprehension():
    assert _comprehension_query("[not role.background for other in ()]") is True


def test_constructor_collection_annotation_uses_the_same_field_projection():
    assert (
        _comprehension_query(
            "[not role.background for role in dict.fromkeys(frame.stops)]",
            constructor=True,
        )
        is True
    )


def _comprehension_query(
    expression,
    *,
    before="",
    body="",
    annotation="tuple[type[Derived], ...]",
    constructor=False,
):
    declarations = SourceModule(
        Path("pkg/declarations.py"),
        "pkg.declarations",
        """
from typing import ClassVar
class ThemeColour(metaclass=UnknownRegistry):
    background: ClassVar[bool] = False
    raw: bool = False
    _private: ClassVar[bool] = False
class Derived(ThemeColour):
    pass
class BaseFrame:
    stops: tuple[type[Derived], ...]
class Frame(BaseFrame):
    pass
""",
    ).parse()
    parsed = SourceModule(
        Path("pkg/paint.py"),
        "pkg.paint",
        f"""
from pkg.declarations import Frame, Derived
{before}
class Painter:
    def paint(self, frame: Frame, role: type[Derived]):
        {body or "pass"}
        return {expression}
""",
    ).parse()
    # Keep fixture declarations external and immutable without executing them.
    declared_source = declarations.source.replace(
        "tuple[type[Derived], ...]", annotation
    )
    if constructor:
        declared_source = declared_source.replace(
            f"stops: {annotation}",
            f"def __init__(self, stops: {annotation}):\n        self.stops = stops",
        )
    declarations = SourceModule(
        Path("pkg/declarations.py"), "pkg.declarations", declared_source
    ).parse()
    repository = SourceProductFlowRepository.from_modules((parsed, declarations))
    attribute = next(
        node.operand
        for node in ast.walk(parsed.module)
        if isinstance(node, ast.UnaryOp)
    )
    return repository.declared_boolean_inversion("pkg.paint.Painter.paint", attribute)


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
