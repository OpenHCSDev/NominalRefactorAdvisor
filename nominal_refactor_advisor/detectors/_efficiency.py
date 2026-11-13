"""Efficiency detectors: patterns where runtime work restates a fact the structure already holds."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag
from nominal_refactor_advisor.detectors._base import (
    DetectorConfig,
    PerModuleIssueDetector,
    high_confidence_spec,
)


_MIN_THREADING_FUNCTIONS = 2


@dataclass(frozen=True)
class _ThreadedParameter:
    function_name: str
    param_name: str
    line: int


def _forwarded_only_params(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[_ThreadedParameter]:
    """Return parameters that have a default and are only used as same-name keyword forwards."""

    params_with_defaults: dict[str, int] = {}
    args = func.args
    num_args = len(args.args)
    num_defaults = len(args.defaults)
    offset = num_args - num_defaults
    for i, default in enumerate(args.defaults):
        arg = args.args[offset + i]
        params_with_defaults[arg.arg] = arg.lineno

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            params_with_defaults[arg.arg] = arg.lineno

    if not params_with_defaults:
        return []

    keyword_forwards = _keyword_forward_names(func)
    name_uses: dict[str, int] = defaultdict(int)
    forward_uses: dict[str, int] = defaultdict(int)

    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id in params_with_defaults:
            name_uses[node.id] += 1
            if (node.lineno, node.col_offset, node.id) in keyword_forwards:
                forward_uses[node.id] += 1

    result = []
    for param_name, line in params_with_defaults.items():
        total = name_uses.get(param_name, 0)
        forwarded = forward_uses.get(param_name, 0)
        if total > 0 and total == forwarded:
            result.append(
                _ThreadedParameter(
                    function_name=func.name,
                    param_name=param_name,
                    line=line,
                )
            )

    return result


def _keyword_forward_names(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[tuple[int, int, str]]:
    """Collect (line, col, name) of Name nodes used as same-name keyword arguments."""

    forwards: set[tuple[int, int, str]] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg is not None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == keyword.arg
            ):
                forwards.add((
                    keyword.value.lineno,
                    keyword.value.col_offset,
                    keyword.value.id,
                ))
    return forwards


class CallbackThreadingDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.CALLBACK_THREADING,
        "Callback parameter threaded through intermediate signatures",
        (
            "A parameter with a default is forwarded unchanged as a same-name keyword "
            "argument through two or more function signatures. Each intermediate "
            "signature restates the route; the parameter belongs to the layer that "
            "uses it (paper1 Theorem 4.6: helpers cannot lower d-down)."
        ),
        "parameter owned by its consumer, not threaded through intermediaries",
        "same-name defaulted parameter forwarded without inspection across functions",
        (CapabilityTag.SHARED_ALGORITHM_AUTHORITY,),
        (ObservationTag.DATAFLOW_ROOT,),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        threaded_by_name: dict[str, list[_ThreadedParameter]] = defaultdict(list)
        index = module.syntax_index

        for _node_index, node in (
            *index.indexed_nodes_of_type(ast.FunctionDef),
            *index.indexed_nodes_of_type(ast.AsyncFunctionDef),
        ):
            for tp in _forwarded_only_params(node):
                threaded_by_name[tp.param_name].append(tp)

        findings: list[RefactorFinding] = []
        for param_name, sites in sorted(threaded_by_name.items()):
            if len(sites) < _MIN_THREADING_FUNCTIONS:
                continue
            evidence = tuple(
                SourceLocation(
                    str(module.path),
                    site.line,
                    f"{site.function_name}({site.param_name}=)",
                )
                for site in sites
            )
            findings.append(
                self.build_finding(
                    (
                        f"parameter `{param_name}` threaded through "
                        f"{len(sites)} signatures: "
                        f"{', '.join(site.function_name for site in sites)}"
                    ),
                    evidence,
                )
            )
        return findings
