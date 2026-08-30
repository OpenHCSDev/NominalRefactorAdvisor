"""Direct reflection boundary detectors."""

from __future__ import annotations

import ast

from ..ast_tools import LEXICAL_SCOPE_BINDING_AUTHORITY
from ._base import *

_DIRECT_REFLECTION_ATTRIBUTE_HOOKS = frozenset(
    (
        "__delattr__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
    )
)
_NATIVE_BARE_OR_ATTRIBUTE_CALL_QUERY = """
(call function: (identifier) @callee) @call
(call function: (attribute attribute: (identifier) @attribute)) @call
"""


def _direct_reflection_owner(
    class_stack: Sequence[str],
    function_stack: Sequence[str],
) -> str:
    owner_parts = (*tuple(class_stack), *tuple(function_stack))
    return ".".join(owner_parts) if owner_parts else "module"


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
                            SourceLocation(str(module.path), node.lineno, f"{owner}:locals"),
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


@dataclass(frozen=True)
class DirectReflectiveAttributeHookCandidate(DirectReflectiveSiteCandidate):
    hook_names: tuple[str, ...]


def _direct_reflective_attribute_hook_candidates(
    module: ParsedModule,
) -> tuple[DirectReflectiveAttributeHookCandidate, ...]:
    sites_by_owner: dict[str, list[tuple[int, str]]] = {}

    class Visitor(ClassFunctionStackNodeVisitor):
        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            if node.name in _DIRECT_REFLECTION_ATTRIBUTE_HOOKS:
                owner = _direct_reflection_owner(
                    self.class_stack,
                    self.function_stack,
                )
                sites_by_owner.setdefault(owner, []).append((node.lineno, node.name))
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.visit_FunctionDef(node)

    Visitor().visit(module.module)
    candidates: list[DirectReflectiveAttributeHookCandidate] = []
    for owner, sites in sorted(sites_by_owner.items()):
        candidates.append(
            DirectReflectiveAttributeHookCandidate(
                owner=owner,
                hook_names=tuple(sorted({name for _line, name in sites})),
                evidence=tuple(
                    SourceLocation(str(module.path), line, f"{owner}:{hook_name}")
                    for line, hook_name in sites
                ),
            )
        )
    return tuple(candidates)


def _source_direct_reflective_attribute_hook_candidates(
    module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    config: DetectorConfig,
) -> tuple[DirectReflectiveAttributeHookCandidate, ...] | None:
    del config
    if not syntax_index.is_complete:
        return None
    sites_by_owner: dict[str, list[tuple[int, str]]] = {}
    functions = syntax_index.common_captures().get("function", ())
    for function in sorted(functions, key=lambda node: node.start_byte):
        hook_name = syntax_index.declared_name(function)
        if hook_name not in _DIRECT_REFLECTION_ATTRIBUTE_HOOKS:
            continue
        scopes = syntax_index.named_scope_nodes(function)
        owner = _direct_reflection_owner(
            tuple(
                syntax_index.declared_name(scope)
                for scope in scopes
                if scope.type == "class_definition"
            ),
            tuple(
                syntax_index.declared_name(scope)
                for scope in scopes
                if scope.type == "function_definition"
            ),
        )
        sites_by_owner.setdefault(owner, []).append(
            (function.start_point.row + 1, hook_name)
        )
    return tuple(
        DirectReflectiveAttributeHookCandidate(
            owner=owner,
            hook_names=tuple(sorted({name for _line, name in sites})),
            evidence=tuple(
                SourceLocation(str(module.path), line, f"{owner}:{hook_name}")
                for line, hook_name in sites
            ),
        )
        for owner, sites in sorted(sites_by_owner.items())
    )


declare_candidate_rule_detector(
    DirectReflectiveAttributeHookCandidate,
    high_confidence_certified_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Reflective attribute hooks bypass nominal boundaries",
        "Definitions of __getattr__, __getattribute__, __setattr__, or __delattr__ turn missing fields and dynamic mutation into runtime semantics. Production execution code should use explicit fields, mapping protocols, generated accessors, or fail-loud formal authorities instead.",
        "no reflective attribute hooks in production execution code",
        "runtime code handles missing fields or mutation through Python magic hooks",
        (
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.ATTRIBUTE_PROBE,
            ObservationTag.PARTIAL_VIEW,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.owner}` defines {len(candidate.evidence)} reflective "
        f"attribute hook(s): {', '.join(candidate.hook_names)}."
    ),
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: (
        "class DeclaredRuntimeSource(ABC):\n"
        "    @abstractmethod\n"
        "    def value_for(self, name: RuntimeFieldName) -> TypedPayload:\n"
        "        raise NotImplementedError\n\n"
        "# Replace magic hooks with explicit ABC APIs, Mapping implementations "
        "with typed keys, or generated "
        "nominal accessors."
    ),
    codemod_patch=lambda candidate: (
        f"# Remove reflective attribute hook(s) in `{candidate.owner}`.\n"
        "# Convert call sites to explicit value()/set_value(), Mapping access, "
        "or generated formal-boundary constants."
    ),
    metrics=lambda candidate: ProbeCountMetrics(
        probe_site_count=len(candidate.evidence)
    ),
    candidate_collector=_direct_reflective_attribute_hook_candidates,
    source_candidate_collector=_source_direct_reflective_attribute_hook_candidates,
    detector_base=SourceModuleCollectorCandidateDetector,
)
