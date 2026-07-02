"""Direct reflection boundary detectors."""

from __future__ import annotations

import ast

from ._base import *

_DIRECT_REFLECTION_BUILTINS = frozenset(
    (
        "delattr",
        "getattr",
        "hasattr",
        "setattr",
    )
)
_DIRECT_REFLECTION_DUNDER_METHODS = frozenset(
    (
        "__delattr__",
        "__getattribute__",
        "__setattr__",
    )
)
_DIRECT_REFLECTION_ATTRIBUTE_HOOKS = frozenset(
    (
        "__delattr__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
    )
)
_UPPERCASE_SEMANTIC_DECLARATION_TOKENS = frozenset(
    (
        "CACHE",
        "REGISTRY",
    )
)
_UPPERCASE_SEMANTIC_DECLARATION_CALLS = frozenset(
    (
        "ContextVar",
        "RegistryConfig",
        "dict",
        "set",
        "list",
    )
)


def _direct_reflection_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name) and node.func.id in _DIRECT_REFLECTION_BUILTINS:
        return node.func.id
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in _DIRECT_REFLECTION_DUNDER_METHODS
    ):
        return ast.unparse(node.func)
    return None


def _direct_reflection_owner(
    class_stack: Sequence[str],
    function_stack: Sequence[str],
) -> str:
    owner_parts = (*tuple(class_stack), *tuple(function_stack))
    return ".".join(owner_parts) if owner_parts else "module"


def _uppercase_declaration_name(name: str) -> bool:
    bare_name = name.removeprefix("_")
    return bool(bare_name) and bare_name.upper() == bare_name


def _semantic_declaration_name(name: str) -> bool:
    tokens = frozenset(name.removeprefix("_").split("_"))
    return bool(tokens & _UPPERCASE_SEMANTIC_DECLARATION_TOKENS)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _semantic_declaration_value_role(node: ast.AST) -> str | None:
    if isinstance(node, ast.Dict):
        return "mutable dict"
    if isinstance(node, (ast.List, ast.Set)):
        return "mutable collection"
    if isinstance(node, ast.Call):
        call_name = _call_name(node)
        if call_name in _UPPERCASE_SEMANTIC_DECLARATION_CALLS:
            return f"{call_name} declaration"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "string protocol declaration"
    if isinstance(node, ast.Name):
        return "alias declaration"
    return None


def _assignment_target_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    if isinstance(node, ast.AnnAssign):
        return (node.target.id,) if isinstance(node.target, ast.Name) else ()
    return tuple(target.id for target in node.targets if isinstance(target, ast.Name))


@dataclass(frozen=True)
class DirectReflectiveSiteCandidate:
    owner: str
    evidence: tuple[SourceLocation, ...]


@dataclass(frozen=True)
class UppercaseSemanticDeclarationCandidate:
    module_path: str
    declaration_names: tuple[str, ...]
    value_roles: tuple[str, ...]
    evidence: tuple[SourceLocation, ...]


def _uppercase_semantic_declaration_candidates(
    module: ParsedModule,
) -> tuple[UppercaseSemanticDeclarationCandidate, ...]:
    declaration_rows: list[tuple[str, int, str]] = []
    for statement in module.module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if value is None:
            continue
        value_role = _semantic_declaration_value_role(value)
        if value_role is None:
            continue
        for target_name in _assignment_target_names(statement):
            if not _uppercase_declaration_name(target_name):
                continue
            if not _semantic_declaration_name(target_name):
                continue
            declaration_rows.append((target_name, statement.lineno, value_role))
    if not declaration_rows:
        return ()
    return (
        UppercaseSemanticDeclarationCandidate(
            module_path=str(module.path),
            declaration_names=tuple(name for name, _line, _role in declaration_rows),
            value_roles=tuple(sorted({role for _name, _line, role in declaration_rows})),
            evidence=tuple(
                SourceLocation(str(module.path), line, name)
                for name, line, _role in declaration_rows
            ),
        ),
    )


declare_candidate_rule_detector(
    UppercaseSemanticDeclarationCandidate,
    high_confidence_certified_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Uppercase semantic declarations should become nominal authorities",
        "Module-level uppercase cache/registry declarations expose mutable or protocol identity as anonymous globals. Production code should route those through nominal cache catalogs, StrEnum field authorities, or AutoRegisterMeta-backed families so ownership is typed and centralized.",
        "nominal cache catalogs and registry/field authorities own runtime semantic identity",
        "runtime semantic cache or registry identity is declared as an uppercase module global",
        (
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.module_path}` declares {len(candidate.declaration_names)} "
        "uppercase semantic cache/registry global(s): "
        f"{', '.join(candidate.declaration_names[:6])}"
        + ("." if len(candidate.declaration_names) <= 6 else ", ...")
    ),
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: (
        "class RuntimeCacheCatalog:\n"
        "    compiled_result: ClassVar[dict[CacheKey, CompiledResult]] = {}\n\n"
        "class RuntimeRegistryAxis(StrEnum):\n"
        "    case_key = 'case_key'\n\n"
        "# Replace uppercase mutable/protocol globals with catalog fields, "
        "nominal enum members, or AutoRegisterMeta-backed family roots."
    ),
    codemod_patch=lambda candidate: (
        f"# Move uppercase semantic declarations in `{candidate.module_path}` "
        "behind a nominal cache catalog, StrEnum authority, or "
        "AutoRegisterMeta family root."
    ),
    metrics=lambda candidate: RegistrationMetrics(
        registration_site_count=len(candidate.declaration_names),
        registry_name="uppercase semantic declarations",
        class_key_pairs=candidate.declaration_names,
    ),
    candidate_collector=_uppercase_semantic_declaration_candidates,
)


@dataclass(frozen=True)
class DirectReflectiveBuiltinCallCandidate(DirectReflectiveSiteCandidate):
    builtin_names: tuple[str, ...]


def _direct_reflective_builtin_call_candidates(
    module: ParsedModule,
) -> tuple[DirectReflectiveBuiltinCallCandidate, ...]:
    sites_by_owner: dict[str, list[tuple[int, str]]] = {}

    class Visitor(ClassFunctionStackNodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            builtin_name = _direct_reflection_call_name(node)
            if builtin_name is not None:
                owner = _direct_reflection_owner(
                    self.class_stack,
                    self.function_stack,
                )
                sites_by_owner.setdefault(owner, []).append((node.lineno, builtin_name))
            self.generic_visit(node)

    Visitor().visit(module.module)
    candidates: list[DirectReflectiveBuiltinCallCandidate] = []
    for owner, sites in sorted(sites_by_owner.items()):
        candidates.append(
            DirectReflectiveBuiltinCallCandidate(
                owner=owner,
                builtin_names=tuple(sorted({name for _line, name in sites})),
                evidence=tuple(
                    SourceLocation(str(module.path), line, f"{owner}:{builtin_name}")
                    for line, builtin_name in sites
                ),
            )
        )
    return tuple(candidates)


declare_candidate_rule_detector(
    DirectReflectiveBuiltinCallCandidate,
    high_confidence_certified_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Direct reflective builtin calls bypass nominal boundaries",
        "Calls to getattr, hasattr, setattr, delattr, and reflective dunder methods recover behavior from partial structural views at runtime. Production code should expose a typed nominal contract, generated accessor, explicit protocol method, or fail-loud formal boundary instead.",
        "no direct reflective builtin calls in production execution code",
        "runtime code probes or mutates attributes through Python reflection",
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
        f"`{candidate.owner}` uses {len(candidate.evidence)} direct reflective "
        f"builtin call(s): {', '.join(candidate.builtin_names)}."
    ),
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: (
        "class DeclaredRuntimeContract(ABC):\n"
        "    @abstractmethod\n"
        "    def required_value(self) -> TypedPayload:\n"
        "        raise NotImplementedError\n\n"
        "# Replace reflection with typed fields, explicit ABC methods, "
        "generated accessors, or formal-boundary fail-loud payload authorities."
    ),
    codemod_patch=lambda candidate: (
        f"# Remove direct reflection in `{candidate.owner}`.\n"
        "# Introduce one typed/nominal authority for the accessed field or "
        "operation, then call it directly."
    ),
    metrics=lambda candidate: ProbeCountMetrics(
        probe_site_count=len(candidate.evidence)
    ),
    candidate_collector=_direct_reflective_builtin_call_candidates,
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
)
