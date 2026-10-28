"""A debug string is not a graph traversal: the reprs of an execution, its
source projection, its parsed module and its flow contexts do not grow with
the module.

An interrupted pytest run reprs every argument of every frame
(``excinfo.getrepr(funcargs=True)``). When these reprs walked the whole
product-flow graph, a 200-class module made one execution prefix repr 90 MB,
and a Ctrl-C took the process past 2 GB instead of stopping it.
"""

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def _execution(count: int) -> SourceModuleExecution:
    source = "\n".join(
        f"class C{index}:\n    def value(self):\n        return {index}\n"
        for index in range(count)
    )
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("/repo/bounded_repr.py"), "bounded_repr", ast.parse(source), source
        )
    )


def _reprs(execution: SourceModuleExecution) -> dict[str, int]:
    source = execution.entry.source
    return {
        "execution": len(repr(execution)),
        "projection": len(repr(source)),
        "module": len(repr(source.module)),
        "largest context": max(
            len(repr(context)) for context in source.compact.flow_contexts
        ),
    }


def test_graph_node_reprs_do_not_grow_with_the_module():
    small = _reprs(_execution(10))
    large = _reprs(_execution(200))

    for name, size in large.items():
        assert size < 10_000, (name, size)
        # The 200-class module only adds digits to what the repr names.
        assert size - small[name] < 100, (name, small[name], size)
