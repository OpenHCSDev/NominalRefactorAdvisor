"""A supplied target-code contract is not an inferred runtime observation."""

import ast
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from types import CodeType
from typing import cast

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import CapturedNativeObject
from nominal_refactor_advisor.lexical_bindings import DefaultedParameterSource, FunctionParameterSource
from nominal_refactor_advisor.source_entry import (
    CompiledSourceModuleEntryPremise,
    NoninterferingSourceModuleEntryPremise,
)
from nominal_refactor_advisor.source_execution import (
    SourceCreatedFunctionCapture,
    SourceFunctionDefault,
    SourceModuleExecution,
)

pytestmark = pytest.mark.native_proof


class NoninterferingCompiledEntry(
    CompiledSourceModuleEntryPremise, NoninterferingSourceModuleEntryPremise
):
    pass


@dataclass(frozen=True)
class TargetDefault:
    join: SourceFunctionDefault
    code: CodeType
    ordinary: SourceModuleExecution

    @classmethod
    def declared(
        cls, header: str,
        entry_kind: type[CompiledSourceModuleEntryPremise] = NoninterferingCompiledEntry,
    ) -> "TargetDefault":
        text = f"def chosen({header}): pass\n"
        module = ParsedModule(Path("target-entry.py"), "target_entry", ast.parse(text), text)
        ordinary = SourceModuleExecution.from_module(module)
        original = ordinary.entry
        code = module.native_compilation.compile()
        entry = entry_kind(
            source=original.source, native_island=original.initial,
            bindings=dict(original.initial_entries), builtins=original.builtins,
            target_code=code,
        )
        execution = SourceModuleExecution(entry)
        function = SourceCreatedFunctionCapture(execution, module.module.body[0])
        parameter = cast(
            DefaultedParameterSource,
            FunctionParameterSource.from_arguments(function.node.args)[0],
        )
        return cls(
            SourceFunctionDefault(
                function, parameter, execution.required_prefix(entry.context, None)
            ), code, ordinary,
        )


@pytest.mark.parametrize("header,expected", (
    ("value=5000", 5000), ("*, value=5000", 5000),
    ("value=(5000, True)", (5000, True)), ("*, value=None", None),
))
def test_declared_birth_object_belongs_to_the_held_target_pool(header: str, expected: object) -> None:
    target = TargetDefault.declared(header)
    captured = target.join.require_target_birth_object()
    assert captured.value == expected
    # The query must name an object in the exact held code graph, independently
    # of the analyzer's separately compiled specimen or AST literal object.
    roots = list(target.code.co_consts)
    while roots:
        original = roots.pop()
        if type(original) is tuple:
            roots.extend(original)
        if captured.proves_same_object(CapturedNativeObject(original)):
            break
    else:
        pytest.fail("birth object is outside the supplied target's constant pool")
    with pytest.raises(ValueError, match="target compilation remains unproved"):
        target.ordinary.entry.target_store(target.join.store)


def test_target_code_contract_does_not_also_declare_noninterference() -> None:
    target = TargetDefault.declared("value=5000", CompiledSourceModuleEntryPremise)
    target.join.result().require_constant_contents(5000)
    with pytest.raises(ValueError, match="External source interference remains unproved"):
        target.join.require_target_birth_object()
    target.join.result().require_constant_contents(5000)


def test_target_projection_requires_original_source_store_and_cut() -> None:
    target = TargetDefault.declared("value=5000")
    join = target.join
    with pytest.raises(ValueError, match="original source store"):
        join.execution.entry.target_store(copy(join.store))
    with pytest.raises(ValueError, match="canonical closed source cut"):
        SourceFunctionDefault(join.function, join.parameter, copy(join.prefix)).require_target_birth_object()


def test_foreign_code_cannot_borrow_the_source_target_association() -> None:
    target = TargetDefault.declared("value=5000")
    compilation = target.join.execution.module.native_compilation
    with pytest.raises(ValueError, match="differs from its source"):
        compilation.project_code(target.code.replace(co_filename="foreign.py"))
    with pytest.raises(TypeError, match="exact native code object"):
        compilation.project_code(cast(CodeType, object()))
