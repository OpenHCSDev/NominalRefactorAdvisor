"""Runtime consumers cannot substitute equal-looking source value receipts."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution():
    source = "value = property\n"
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("read_identity.py"), "read_identity", False, ast.parse(source), source
        )
    )


@pytest.mark.parametrize("value_capture", (False, True), ids=("lexical", "evaluated"))
def test_actual_read_preserves_capture(value_capture):
    environment = execution()
    node = environment.module.module.body[0].value
    reads = (
        environment.source.value_reads_by_node
        if value_capture
        else environment.source.reference_reads_by_node
    )
    environment.kernel.read(reads[node]).require_native_identity(
        NativeDeclaration(property)
    )


@pytest.mark.parametrize("value_capture", (False, True), ids=("lexical", "evaluated"))
@pytest.mark.parametrize("forgery", ("use", "context", "reparsed"))
def test_foreign_read_cannot_borrow_source_prefix(value_capture, forgery):
    environment = execution()
    node = environment.module.module.body[0].value
    reads = (
        environment.source.value_reads_by_node
        if value_capture
        else environment.source.reference_reads_by_node
    )
    read = reads[node]
    if forgery == "use":
        foreign = replace(read, use=replace(read.use))
    elif forgery == "context":
        foreign = replace(read, context=replace(read.context))
    else:
        other = execution()
        other_node = other.module.module.body[0].value
        other_reads = (
            other.source.value_reads_by_node
            if value_capture
            else other.source.reference_reads_by_node
        )
        foreign = other_reads[other_node]
    with pytest.raises(
        ValueError, match="unique original operation|different canonical context"
    ):
        environment.kernel.read(foreign)
