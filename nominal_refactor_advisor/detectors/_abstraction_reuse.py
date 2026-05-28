"""Generic detection for local reimplementation of available abstractions."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from ..collection_algebra import sorted_tuple
from ..models import MappingMetrics
from ..patterns import PatternId
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    DetectorConfig,
    IssueDetector,
    ParsedModule,
    RefactorFinding,
    SourceLocation,
    high_confidence_spec,
)


_MIN_AUTHORITY_ATOMS = 7
_MIN_LOCAL_ATOMS = 6
_MIN_OVERLAP_ATOMS = 5
_MIN_OVERLAP_SCORE = 9
_MIN_AUTHORITY_COVERAGE = 0.40
_MIN_LOCAL_COVERAGE = 0.35
_MAX_FOCUSED_AUTHORITY_ATOMS = 32

_AUTHORITY_PATH_PARTS = frozenset(
    {
        "common",
        "component",
        "components",
        "factory",
        "factories",
        "scaffold",
        "scaffolds",
        "shared",
        "support",
        "utils",
    }
)
_AUTHORITY_NAME_SUFFIXES = (
    "Adapter",
    "Authority",
    "Base",
    "Builder",
    "Catalog",
    "Factory",
    "Formatter",
    "Manager",
    "Mixin",
    "Panel",
    "Parser",
    "Renderer",
    "Resolver",
    "Scaffold",
    "Strategy",
)
_HIGH_SIGNAL_ATOM_PREFIXES = ("construct:", "method:", "signal:", "store:", "control:")
_STRUCTURAL_ATOM_PREFIXES = ("construct:", "method:", "signal:", "store:", "control:")
_IDENTITY_FIELD_TERMINALS = frozenset(
    {"id", "ids", "key", "keys", "name", "names", "path", "paths", "root", "roots"}
)
_MIN_PARALLEL_PRIMITIVE_FIELDS = 3
_MIN_PARALLEL_PRIMITIVE_RECORDS = 2


@dataclass(frozen=True, slots=True)
class ParallelPrimitiveFieldBundle:
    file_path: str
    module_name: str
    line: int
    class_name: str
    field_names: tuple[str, ...]
    semantic_roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParallelPrimitiveCarrierCandidate:
    semantic_roles: tuple[str, ...]
    bundles: tuple[ParallelPrimitiveFieldBundle, ...]


@dataclass(frozen=True, slots=True)
class CapabilitySignature:
    atoms: frozenset[str]
    call_names: frozenset[str]

    @property
    def high_signal_atoms(self) -> frozenset[str]:
        return frozenset(
            atom
            for atom in self.atoms
            if atom.startswith(_HIGH_SIGNAL_ATOM_PREFIXES)
        )


def _snake_tokens(name: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for chunk in name.replace("-", "_").split("_"):
        if chunk:
            tokens.append(chunk.lower())
    return tuple(tokens)


def _semantic_role_for_identity_field(name: str) -> str | None:
    tokens = _snake_tokens(name)
    if len(tokens) < 2 or tokens[-1] not in _IDENTITY_FIELD_TERMINALS:
        return None
    role_tokens = tokens[:-1]
    if not role_tokens:
        return None
    return "_".join(role_tokens)


def _annotation_is_primitive_carrier(node: ast.AST | None) -> bool:
    if node is None:
        return True
    text = ast.unparse(node)
    return any(
        token in text
        for token in (
            "str",
            "Path",
            "Any",
            "Optional",
            "None",
            "Union",
        )
    )


def _class_identity_field_names(node: ast.ClassDef) -> tuple[str, ...]:
    names: list[str] = []
    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign):
            continue
        if not isinstance(statement.target, ast.Name):
            continue
        field_name = statement.target.id
        if _semantic_role_for_identity_field(field_name) is None:
            continue
        if not _annotation_is_primitive_carrier(statement.annotation):
            continue
        names.append(field_name)
    return tuple(names)


def _module_parallel_primitive_bundles(
    module: ParsedModule,
) -> tuple[ParallelPrimitiveFieldBundle, ...]:
    bundles: list[ParallelPrimitiveFieldBundle] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        field_names = _class_identity_field_names(node)
        semantic_roles = tuple(
            role
            for field_name in field_names
            if (role := _semantic_role_for_identity_field(field_name)) is not None
        )
        if len(semantic_roles) < _MIN_PARALLEL_PRIMITIVE_FIELDS:
            continue
        bundles.append(
            ParallelPrimitiveFieldBundle(
                file_path=str(module.path),
                module_name=module.module_name,
                line=node.lineno,
                class_name=node.name,
                field_names=field_names,
                semantic_roles=semantic_roles,
            )
        )
    return tuple(bundles)


def _parallel_primitive_carrier_candidates(
    modules: list[ParsedModule],
) -> tuple[ParallelPrimitiveCarrierCandidate, ...]:
    grouped: dict[tuple[str, ...], list[ParallelPrimitiveFieldBundle]] = defaultdict(
        list
    )
    for module in modules:
        for bundle in _module_parallel_primitive_bundles(module):
            for role_count in range(
                _MIN_PARALLEL_PRIMITIVE_FIELDS, len(bundle.semantic_roles) + 1
            ):
                for semantic_roles in combinations(bundle.semantic_roles, role_count):
                    grouped[semantic_roles].append(bundle)
    candidates: list[ParallelPrimitiveCarrierCandidate] = []
    for semantic_roles, bundles in grouped.items():
        ordered = sorted_tuple(
            bundles, key=lambda item: (item.file_path, item.line, item.class_name)
        )
        if len(ordered) < _MIN_PARALLEL_PRIMITIVE_RECORDS:
            continue
        candidates.append(
            ParallelPrimitiveCarrierCandidate(
                semantic_roles=semantic_roles,
                bundles=ordered,
            )
        )
    ordered_candidates = sorted_tuple(
        candidates,
        key=lambda item: (
            -len(item.bundles),
            -len(item.semantic_roles),
            item.semantic_roles,
            item.bundles[0].file_path,
        ),
    )
    selected: list[ParallelPrimitiveCarrierCandidate] = []
    selected_bundle_sets: list[frozenset[tuple[str, int, str]]] = []
    for candidate in ordered_candidates:
        candidate_bundle_set = frozenset(
            (bundle.file_path, bundle.line, bundle.class_name)
            for bundle in candidate.bundles
        )
        if any(
            candidate_bundle_set <= selected_bundle_set
            for selected_bundle_set in selected_bundle_sets
        ):
            continue
        selected.append(candidate)
        selected_bundle_sets.append(candidate_bundle_set)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class AbstractionAuthoritySignature:
    file_path: str
    module_name: str
    line: int
    name: str
    symbol: str
    signature: CapabilitySignature
    shared_path_authority: bool


@dataclass(frozen=True, slots=True)
class LocalImplementationSignature:
    file_path: str
    module_name: str
    line: int
    symbol: str
    signature: CapabilitySignature
    imported_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class AvailableAbstractionReuseCandidate:
    local: LocalImplementationSignature
    authority: AbstractionAuthoritySignature
    overlap_atoms: tuple[str, ...]
    overlap_score: int


class _CapabilityAtomVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.atoms: set[str] = set()
        self.call_names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self.atoms.add(f"param:{argument.arg}")
        if node.args.vararg is not None:
            self.atoms.add(f"param:{node.args.vararg.arg}")
        if node.args.kwarg is not None:
            self.atoms.add(f"param:{node.args.kwarg.arg}")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_terminal_name(node.func)
        if call_name is not None:
            self.call_names.update(_call_reference_names(node.func))
            self.atoms.add(f"call:{call_name}")
            if _looks_like_constructor_name(call_name):
                self.atoms.add(f"construct:{call_name}")
        if isinstance(node.func, ast.Attribute):
            self.atoms.add(f"method:{node.func.attr}")
            if node.func.attr == "connect":
                signal_name = _terminal_name(node.func.value)
                if signal_name is not None:
                    self.atoms.add(f"signal:{signal_name}.connect")
        for keyword in node.keywords:
            if keyword.arg is not None:
                self.atoms.add(f"keyword:{keyword.arg}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_store_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_store_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_store_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.atoms.add("control:for")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_If(self, node: ast.If) -> None:
        self.atoms.add("control:if")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.atoms.add("control:try")
        self.generic_visit(node)

    def _record_store_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.atoms.add(f"store:{target.id}")
            return
        if isinstance(target, ast.Attribute):
            self.atoms.add(f"store:{target.attr}")
            return
        if isinstance(target, ast.Subscript):
            target_name = _terminal_name(target.value)
            if target_name is not None:
                self.atoms.add(f"store:{target_name}")
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._record_store_target(element)


class _LocalSignatureCollector(ast.NodeVisitor):
    def __init__(self, module: ParsedModule) -> None:
        self.module = module
        self.class_stack: list[str] = []
        self.locals: list[LocalImplementationSignature] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        symbol = ".".join((*self.class_stack, node.name))
        signature = _signature_for_node(node)
        if len(signature.high_signal_atoms) >= _MIN_LOCAL_ATOMS:
            self.locals.append(
                LocalImplementationSignature(
                    file_path=str(self.module.path),
                    module_name=self.module.module_name,
                    line=node.lineno,
                    symbol=symbol,
                    signature=signature,
                    imported_names=frozenset(_imported_local_names(self.module)),
                )
            )


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _terminal_name(node.value)
    return None


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain(node.value)
        if parent is None:
            return None
        return (*parent, node.attr)
    return None


def _call_terminal_name(node: ast.AST) -> str | None:
    return _terminal_name(node)


def _call_reference_names(node: ast.AST) -> frozenset[str]:
    chain = _attribute_chain(node)
    if chain is None:
        terminal = _call_terminal_name(node)
        return frozenset(() if terminal is None else (terminal,))
    names: set[str] = set(chain)
    for start in range(len(chain)):
        suffix = chain[start:]
        if len(suffix) > 1:
            names.add(".".join(suffix))
    return frozenset(names)


def _looks_like_constructor_name(name: str) -> bool:
    return bool(name) and name[:1].isupper()


def _signature_for_node(node: ast.AST) -> CapabilitySignature:
    visitor = _CapabilityAtomVisitor()
    visitor.visit(node)
    return CapabilitySignature(
        atoms=frozenset(visitor.atoms),
        call_names=frozenset(visitor.call_names),
    )


def _module_path_parts(module: ParsedModule) -> frozenset[str]:
    return frozenset(part.lower() for part in Path(module.path).with_suffix("").parts)


def _is_shared_authority_location(module: ParsedModule) -> bool:
    return bool(_module_path_parts(module) & _AUTHORITY_PATH_PARTS)


def _looks_like_reusable_authority_name(name: str) -> bool:
    return name.endswith(_AUTHORITY_NAME_SUFFIXES)


def _public_name(name: str) -> bool:
    return not name.startswith("_")


def _imported_local_names(module: ParsedModule) -> tuple[str, ...]:
    names: list[str] = []
    for statement in module.module.body:
        if isinstance(statement, ast.Import):
            names.extend(alias.asname or alias.name.split(".", 1)[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            names.extend(
                alias.asname or alias.name
                for alias in statement.names
                if alias.name != "*"
            )
    return sorted_tuple(set(names))


def _class_method_nodes(node: ast.ClassDef) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    return tuple(
        statement
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _combined_class_signature(node: ast.ClassDef) -> CapabilitySignature:
    atoms: set[str] = set()
    call_names: set[str] = set()
    for method in _class_method_nodes(node):
        signature = _signature_for_node(method)
        atoms.update(signature.atoms)
        call_names.update(signature.call_names)
    return CapabilitySignature(frozenset(atoms), frozenset(call_names))


def _module_authorities(module: ParsedModule) -> tuple[AbstractionAuthoritySignature, ...]:
    shared_path_authority = _is_shared_authority_location(module)
    authorities: list[AbstractionAuthoritySignature] = []
    for statement in module.module.body:
        if isinstance(statement, ast.ClassDef):
            if not _public_name(statement.name):
                continue
            signature = _combined_class_signature(statement)
            if len(signature.high_signal_atoms) < _MIN_AUTHORITY_ATOMS:
                continue
            if len(signature.high_signal_atoms) > _MAX_FOCUSED_AUTHORITY_ATOMS:
                continue
            if not _looks_like_reusable_authority_name(statement.name):
                continue
            authorities.append(
                AbstractionAuthoritySignature(
                    file_path=str(module.path),
                    module_name=module.module_name,
                    line=statement.lineno,
                    name=statement.name,
                    symbol=statement.name,
                    signature=signature,
                    shared_path_authority=shared_path_authority,
                )
            )
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _public_name(statement.name):
                continue
            signature = _signature_for_node(statement)
            if len(signature.high_signal_atoms) < _MIN_AUTHORITY_ATOMS:
                continue
            if len(signature.high_signal_atoms) > _MAX_FOCUSED_AUTHORITY_ATOMS:
                continue
            if not _looks_like_reusable_authority_name(statement.name):
                continue
            authorities.append(
                AbstractionAuthoritySignature(
                    file_path=str(module.path),
                    module_name=module.module_name,
                    line=statement.lineno,
                    name=statement.name,
                    symbol=statement.name,
                    signature=signature,
                    shared_path_authority=shared_path_authority,
                )
            )
    return sorted_tuple(
        authorities,
        key=lambda authority: (authority.file_path, authority.line, authority.name),
    )


def _module_locals(module: ParsedModule) -> tuple[LocalImplementationSignature, ...]:
    collector = _LocalSignatureCollector(module)
    collector.visit(module.module)
    return sorted_tuple(
        collector.locals,
        key=lambda local: (local.file_path, local.line, local.symbol),
    )


def _top_level_package(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def _authority_available_to_local(
    authority: AbstractionAuthoritySignature, local: LocalImplementationSignature
) -> bool:
    if authority.name in local.imported_names:
        return True
    if not authority.shared_path_authority:
        return False
    return _top_level_package(authority.module_name) == _top_level_package(
        local.module_name
    )


def _structural_overlap(atoms: Iterable[str]) -> tuple[str, ...]:
    return sorted_tuple(
        atom for atom in atoms if atom.startswith(_STRUCTURAL_ATOM_PREFIXES)
    )


def _overlap_score(atoms: Sequence[str]) -> int:
    score = 0
    for atom in atoms:
        if atom.startswith("construct:"):
            score += 3
        elif atom.startswith(("method:", "signal:", "store:")):
            score += 2
        elif atom.startswith("control:"):
            score += 1
        else:
            score += 1
    return score


def _local_declares_authority_name(
    local: LocalImplementationSignature, authority: AbstractionAuthoritySignature
) -> bool:
    return local.symbol == authority.name or local.symbol.startswith(f"{authority.name}.")


def _reimplements_authority(
    local: LocalImplementationSignature, authority: AbstractionAuthoritySignature
) -> AvailableAbstractionReuseCandidate | None:
    if local.file_path == authority.file_path:
        return None
    if not _authority_available_to_local(authority, local):
        return None
    if (
        authority.name in local.signature.call_names
        and not _local_declares_authority_name(local, authority)
    ):
        return None
    overlap = local.signature.high_signal_atoms & authority.signature.high_signal_atoms
    if len(overlap) < _MIN_OVERLAP_ATOMS:
        return None
    authority_coverage = len(overlap) / max(len(authority.signature.high_signal_atoms), 1)
    if authority_coverage < _MIN_AUTHORITY_COVERAGE:
        return None
    local_coverage = len(overlap) / max(len(local.signature.high_signal_atoms), 1)
    if local_coverage < _MIN_LOCAL_COVERAGE:
        return None
    structural_overlap = _structural_overlap(overlap)
    if len(structural_overlap) < _MIN_OVERLAP_ATOMS:
        return None
    if not any(atom.startswith("construct:") for atom in structural_overlap):
        if len(
            tuple(
                atom
                for atom in structural_overlap
                if atom.startswith(("method:", "signal:", "store:"))
            )
        ) < 4:
            return None
    score = _overlap_score(structural_overlap)
    if score < _MIN_OVERLAP_SCORE:
        return None
    return AvailableAbstractionReuseCandidate(
        local=local,
        authority=authority,
        overlap_atoms=structural_overlap,
        overlap_score=score,
    )


def _available_abstraction_reuse_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[AvailableAbstractionReuseCandidate, ...]:
    authorities = tuple(
        authority for module in modules for authority in _module_authorities(module)
    )
    if not authorities:
        return ()
    candidates_by_local: dict[
        tuple[str, int, str], list[AvailableAbstractionReuseCandidate]
    ] = defaultdict(list)
    for module in modules:
        for local in _module_locals(module):
            for authority in authorities:
                candidate = _reimplements_authority(local, authority)
                if candidate is not None:
                    candidates_by_local[(local.file_path, local.line, local.symbol)].append(
                        candidate
                    )
    best_candidates = [
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate.overlap_score,
                candidate.authority.file_path,
                candidate.authority.line,
                candidate.authority.name,
            ),
        )[0]
        for candidates in candidates_by_local.values()
    ]
    return sorted_tuple(
        best_candidates,
        key=lambda candidate: (
            candidate.local.file_path,
            candidate.local.line,
            candidate.local.symbol,
            candidate.authority.name,
        ),
    )


class AvailableAbstractionReuseDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Local implementation should reuse the available abstraction authority",
        "A local function or method rebuilds the construction/orchestration shape already owned by an available abstraction. The docs prefer routing through the existing authority instead of recreating its internal mechanics at the call site.",
        "reuse of the available abstraction authority instead of local reconstruction",
        "local code and an available abstraction share the same high-signal capability atoms",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.PROVENANCE,
        ),
        (ObservationTag.NORMALIZED_AST, ObservationTag.METHOD_ROLE),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _available_abstraction_reuse_candidates(modules):
            overlap_preview = ", ".join(candidate.overlap_atoms[:8])
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.local.symbol}` locally rebuilds `{candidate.authority.name}` "
                        f"with shared capability atoms {overlap_preview}."
                    ),
                    (
                        SourceLocation(
                            candidate.local.file_path,
                            candidate.local.line,
                            candidate.local.symbol,
                        ),
                        SourceLocation(
                            candidate.authority.file_path,
                            candidate.authority.line,
                            candidate.authority.symbol,
                        ),
                    ),
                    scaffold=(
                        f"# Replace local reconstruction in `{candidate.local.symbol}` with `{candidate.authority.name}`.\n"
                        f"{candidate.authority.name}(...)"
                    ),
                    codemod_patch=(
                        f"# Import and call `{candidate.authority.name}` instead of rebuilding its internals.\n"
                        "# Keep local residue as configuration, callback, or adapter arguments passed into the authority."
                    ),
                )
            )
        return findings


class ParallelPrimitiveCarrierDetector(IssueDetector):
    detector_id = "parallel_primitive_carrier"
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Parallel primitive fields should become a nominal carrier",
        "Several record/request classes carry the same correlated primitive identity fields. The docs prefer one nominal carrier with local invariants over repeatedly threading adjacent strings or paths that must describe one semantic object.",
        "single nominal carrier for correlated identity/path roles",
        "same primitive identity role bundle is repeated across record classes",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _parallel_primitive_carrier_candidates(modules):
            bundles = candidate.bundles
            role_summary = ", ".join(candidate.semantic_roles)
            class_summary = ", ".join(bundle.class_name for bundle in bundles[:5])
            field_summary = ", ".join(bundles[0].field_names)
            findings.append(
                self.build_finding(
                    (
                        f"Primitive identity roles ({role_summary}) are repeated "
                        f"across records {class_summary}."
                    ),
                    tuple(
                        SourceLocation(
                            bundle.file_path,
                            bundle.line,
                            bundle.class_name,
                        )
                        for bundle in bundles[:6]
                    ),
                    scaffold=(
                        "@dataclass(frozen=True)\n"
                        "class NominalIdentityCarrier:\n"
                        f"    # roles: {role_summary}\n"
                        "    ...\n\n"
                        f"# Replace parallel primitive fields ({field_summary}) "
                        "with one nominal carrier and project it at the transport boundary."
                    ),
                    codemod_patch=(
                        "# Introduce one dataclass/record for the repeated role bundle.\n"
                        "# Store invariants on that record; pass the carrier internally; "
                        "serialize primitive fields only at external protocol boundaries."
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(bundles),
                        mapping_name="parallel_primitive_carrier",
                        field_names=bundles[0].field_names,
                        identity_field_names=candidate.semantic_roles,
                    ),
                )
            )
        return findings


__all__ = tuple(name for name in globals() if not name.startswith("_"))
