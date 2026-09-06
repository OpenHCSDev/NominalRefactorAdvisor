"""Memoized projections belong to each immutable call declaration, not its callers."""

import ast
from dataclasses import fields, replace
from pathlib import Path
import runpy

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.product_flow import (
    CompactCallArguments,
    CompactFunctionBindingKind,
    CompactFunctionDeclaration,
    CompactFunctionIdentity,
)
from nominal_refactor_advisor.call_binding import (
    CompactCallBindingViolation,
    CompactFunctionSignature,
)
from nominal_refactor_advisor.value_expression import (
    CompactValueExpression,
    LexicalValueReference,
)


def _declaration() -> CompactFunctionDeclaration:
    node = ast.parse("def consume(self, value): pass").body[0]
    return CompactFunctionDeclaration(
        identity=CompactFunctionIdentity("probe", "Owner.consume"),
        line=1,
        end_line=1,
        owner_class_qualname="Owner",
        signature=CompactFunctionSignature.from_arguments(node.args),
    )


def test_repeated_binding_uses_one_derived_signature() -> None:
    declaration = _declaration()
    arguments = CompactCallArguments.from_call(
        ast.parse("consume(item)", mode="eval").body, CompactValueExpression.project
    )
    signature = declaration.call_signature
    first = arguments.bind_to(declaration)
    for _ in range(10):
        assert arguments.bind_to(declaration) == first
        assert declaration.call_signature is signature
    assert first.is_exact
    assert (
        vars(declaration)["binding_kind"] is CompactFunctionBindingKind.INSTANCE_METHOD
    )
    assert vars(declaration)["signature_decorator_hazard"] is False
    assert {field.name for field in fields(declaration)}.isdisjoint(
        {"binding_kind", "call_signature", "signature_decorator_hazard"}
    )


def test_replacing_a_declaration_recomputes_its_projections() -> None:
    declaration = _declaration()
    signature = declaration.call_signature
    assert not declaration.signature_decorator_hazard
    changed = replace(declaration, decorators=(LexicalValueReference("staticmethod"),))
    assert "call_signature" not in vars(changed)
    assert changed.binding_kind is CompactFunctionBindingKind.STATIC_METHOD
    assert len(changed.call_signature.parameters) == 2
    assert len(signature.parameters) == 1
    assert declaration.call_signature is signature
    unsafe = replace(declaration, decorators=(LexicalValueReference("unknown"),))
    assert unsafe.signature_decorator_hazard
    assert (
        unsafe.bind_call((), ()).violation
        is CompactCallBindingViolation.SIGNATURE_DECORATOR_HAZARD
    )
    assert not declaration.signature_decorator_hazard


def test_recorded_memoization_plan_only_changes_decorators(tmp_path: Path) -> None:
    plan = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "docs/examples/cache_call_declaration_projections.py"
        )
    )["PLAN"]
    path = tmp_path / "nominal_refactor_advisor/product_flow.py"
    path.parent.mkdir()
    source = (
        "from functools import cached_property\n"
        "class CompactFunctionDeclaration:\n"
        "    @property\n    def binding_kind(self): return 1\n"
        "    @property\n    def signature_decorator_hazard(self): return False\n"
        "    @property\n    def call_signature(self): return (1, 2)\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    simulation = plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert simulation.is_clean
    simulation.apply()
    assert path.read_text(encoding="utf-8") == source.replace(
        "@property", "@cached_property"
    )
