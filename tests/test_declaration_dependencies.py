from __future__ import annotations

import ast
import sys

import pytest

from nominal_refactor_advisor.declaration_dependencies import (
    DeclarationDependencyProjection,
    ModuleLexicalDependencyProjection,
)


def _projection(source: str) -> DeclarationDependencyProjection:
    module = ast.parse(source)
    declarations = tuple(
        statement
        for statement in module.body
        if isinstance(
            statement,
            (
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Assign,
                ast.AnnAssign,
            ),
        )
    )
    return DeclarationDependencyProjection.from_declarations(declarations)


def test_function_local_binding_does_not_hide_header_dependency() -> None:
    projection = _projection(
        "def build(value: External = Default()) -> Result:\n"
        "    External = object()\n"
        "    return Runtime(value)\n"
    )

    assert projection.annotation_names == frozenset(("External", "Result"))
    assert projection.execution_names == frozenset(("Default", "Runtime", "object"))


def test_nested_scope_bindings_do_not_hide_class_dependencies() -> None:
    projection = _projection(
        "@decorate\n"
        "class Helper(Base):\n"
        "    value: FieldType\n"
        "\n"
        "    def build(self):\n"
        "        Base = object()\n"
        "        return Runtime(Base)\n"
    )

    assert projection.annotation_names == frozenset(("FieldType",))
    assert projection.execution_names == frozenset(
        ("Base", "Runtime", "decorate", "object")
    )


def test_annotation_projection_includes_function_local_type_dependencies() -> None:
    projection = _projection(
        "class Container:\n"
        "    field: FieldType\n"
        "\n"
        "    def build(self, value: ArgumentType) -> ReturnType:\n"
        "        local: LocalType\n"
        "\n"
        "        class Nested:\n"
        "            field: NestedType\n"
        "\n"
        "        return value\n"
    )

    assert projection.annotation_names == frozenset(
        ("ArgumentType", "FieldType", "LocalType", "NestedType", "ReturnType")
    )
    assert projection.annotation_count == 4


def test_enclosing_function_binding_satisfies_nested_closure_dependency() -> None:
    projection = _projection(
        "def outer():\n"
        "    captured = 1\n"
        "\n"
        "    def inner():\n"
        "        return captured + External\n"
        "\n"
        "    return inner()\n"
    )

    assert projection.execution_names == frozenset(("External",))


def test_class_load_before_binding_remains_an_external_dependency() -> None:
    projection = _projection(
        "class Before:\n"
        "    value = External\n"
        "    External = 1\n"
        "\n"
        "class After:\n"
        "    Internal = 1\n"
        "    value = Internal\n"
    )

    assert projection.execution_names == frozenset(("External",))


def test_comprehension_bindings_do_not_escape_their_lexical_scope() -> None:
    projection = _projection(
        "def build(values):\n"
        "    selected = [transform(value) for value in values if predicate(value)]\n"
        "    return value, selected\n"
    )

    assert projection.execution_names == frozenset(("predicate", "transform", "value"))


def test_comprehension_walrus_binding_belongs_to_containing_function() -> None:
    projection = _projection(
        "def resolve(values):\n"
        "    return tuple(\n"
        "        (value, resolved)\n"
        "        for value in values\n"
        "        if (resolved := lookup(value)) is not None\n"
        "    )\n"
    )

    assert projection.execution_names == frozenset(("lookup", "tuple"))


def test_external_reference_projection_preserves_lexical_name_identity() -> None:
    module = ast.parse(
        "class Authority:\n"
        "    pass\n\n"
        "def external() -> Authority:\n"
        "    return Authority()\n\n"
        "def shadowed(Authority):\n"
        "    return Authority\n\n"
        "class Consumer:\n"
        "    before = Authority\n"
        "    Authority = object\n"
        "    after = Authority\n"
    )

    references = ModuleLexicalDependencyProjection.from_module(
        module
    ).external_references_named("Authority")

    assert tuple(reference.lineno for reference in references) == (4, 5, 11)


def test_assignment_dependencies_preserve_value_and_annotation_contexts() -> None:
    projection = _projection(
        "BASE = make_base()\n" "VALUE: ValueType = transform(BASE)\n"
    )

    assert projection.execution_names == frozenset(("BASE", "make_base", "transform"))
    assert projection.annotation_names == frozenset(("ValueType",))
    assert projection.annotation_count == 1


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 syntax")
def test_type_parameter_scope_owns_declaration_references() -> None:
    projection = _projection(
        "class Container[T: Bound = Default](Base[T]):\n"
        "    value: T\n"
        "\n"
        "    def resolve(self) -> T:\n"
        "        return T\n"
    )

    assert projection.execution_names == frozenset(("Base",))
    assert projection.annotation_names == frozenset(("Bound", "Default"))
    assert projection.annotation_count == 4
