"""Direct reflection boundary detectors."""

from __future__ import annotations

import ast

from ..ast_tools import LEXICAL_SCOPE_BINDING_AUTHORITY
from ._base import *

_NATIVE_BARE_OR_ATTRIBUTE_CALL_QUERY = """
(call function: (identifier) @callee) @call
(call function: (attribute attribute: (identifier) @attribute)) @call
"""


@dataclass(frozen=True)
class DirectReflectiveSiteCandidate:
    owner: str
    evidence: tuple[SourceLocation, ...]
@dataclass(frozen=True)
class BuiltinLocalsCallCandidate(DirectReflectiveSiteCandidate):
    pass


def _builtin_locals_call_candidates(
    module: ParsedModule,
) -> tuple[BuiltinLocalsCallCandidate, ...]:
    candidates: list[BuiltinLocalsCallCandidate] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope_bindings: list[frozenset[str]] = [
                LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.module.body)
            ]
            self.scope_names: list[str] = ["module"]

        def _visit_scope(
            self,
            node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
            name: str,
        ) -> None:
            body = node.body if not isinstance(node, ast.Lambda) else ()
            self.scope_bindings.append(
                LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(body)
                | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(node)
            )
            self.scope_names.append(name)
            self.generic_visit(node)
            self.scope_names.pop()
            self.scope_bindings.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_scope(node, node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_scope(node, node.name)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            self._visit_scope(node, "lambda")

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "locals"
                and not any("locals" in bindings for bindings in self.scope_bindings)
            ):
                owner = ".".join(self.scope_names)
                candidates.append(
                    BuiltinLocalsCallCandidate(
                        owner=owner,
                        evidence=(
                            SourceLocation(module.file_path, node.lineno, f"{owner}:locals"),
                        ),
                    )
                )
            self.generic_visit(node)

    Visitor().visit(module.module)
    return tuple(candidates)


def _source_builtin_locals_call_candidates(
    module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    config: DetectorConfig,
) -> tuple[BuiltinLocalsCallCandidate, ...] | None:
    del module, config
    if not syntax_index.is_complete:
        return None
    captures = syntax_index.captures(_NATIVE_BARE_OR_ATTRIBUTE_CALL_QUERY)
    if any(
        syntax_index.source_for(callee) == b"locals"
        for callee in captures.get("callee", ())
    ):
        return None
    return ()


declare_candidate_rule_detector(
    BuiltinLocalsCallCandidate,
    high_confidence_certified_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Builtin locals calls hide lexical dependencies",
        "Calls to Python's built-in locals() convert lexical dependencies into a string-keyed, untyped registry and hide semantic coupling. Production code should use explicit typed fields or parameters.",
        "no calls to Python's built-in locals() in production execution code",
        "runtime code captures its lexical namespace through the built-in locals() mapping",
        (
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        (ObservationTag.PARTIAL_VIEW, ObservationTag.NORMALIZED_AST),
    ),
    summary=lambda candidate: (
        f"`{candidate.owner}` calls Python's built-in `locals()`, hiding lexical dependencies."
    ),
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: (
        "# Replace locals() with explicit typed parameters or a nominal runtime contract."
    ),
    codemod_patch=lambda candidate: (
        f"# Remove built-in locals() capture in `{candidate.owner}` and pass the required values explicitly."
    ),
    metrics=lambda candidate: ProbeCountMetrics(probe_site_count=len(candidate.evidence)),
    candidate_collector=_builtin_locals_call_candidates,
    source_candidate_collector=_source_builtin_locals_call_candidates,
    detector_base=SourceModuleCollectorCandidateDetector,
)
