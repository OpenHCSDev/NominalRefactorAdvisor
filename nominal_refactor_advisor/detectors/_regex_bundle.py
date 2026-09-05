"""Repeated regex syntax observed through lexical and imported declarations."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property, lru_cache
from inspect import Signature, signature
from itertools import combinations
import re

from ..ast_projection import AstExpressionProjection
from ..ast_tools import ParsedModule, named_function_nodes, walk_function_body_nodes
from ..class_index import (
    ModuleNominalBindingAuthority,
    ModuleNominalBindingSnapshot,
    nominal_reference_root,
)
from ..codemod_module_declarations import SourceTopLevelDeclarationIndex
from ..collection_algebra import sorted_tuple
from ..declaration_dependencies import ModuleLexicalDependencyProjection
from ..lexical_scopes import LexicalNameResolution
from ..models import MappingMetrics, RefactorFinding
from ..patterns import PatternId
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    ConfiguredModuleCollectorCandidateDetector,
    DetectorConfig,
    FunctionEvidenceLocationsCandidate,
    high_confidence_spec,
)


class RegexPatternOperation(StrEnum):
    """Standard-library declarations whose first parameter consumes regex syntax."""

    COMPILE = (re.compile,)
    FINDALL = (re.findall,)
    FINDITER = (re.finditer,)
    SEARCH = (re.search,)
    MATCH = (re.match,)
    FULLMATCH = (re.fullmatch,)
    SPLIT = (re.split,)
    SUB = (re.sub,)
    SUBN = (re.subn,)

    function: Callable[..., object]

    def __new__(cls, function: Callable[..., object]) -> RegexPatternOperation:
        value = f"{function.__module__}.{function.__qualname__}"
        member = str.__new__(cls, value)
        member._value_ = value
        member.function = function
        return member

    @cached_property
    def call_signature(self) -> Signature:
        return signature(self.function)

    @classmethod
    def for_qualified_name(cls, name: str | None) -> RegexPatternOperation | None:
        return next((operation for operation in cls if operation.value == name), None)

    def pattern_literal(self, call: ast.Call) -> str | None:
        if (
            any(isinstance(argument, ast.Starred) for argument in call.args)
            or any(keyword.arg is None for keyword in call.keywords)
            or len({keyword.arg for keyword in call.keywords}) != len(call.keywords)
        ):
            return None
        try:
            bound = self.call_signature.bind(
                *call.args, **{keyword.arg: keyword.value for keyword in call.keywords}
            )
        except TypeError:
            return None
        pattern_name = next(iter(self.call_signature.parameters))
        expression = bound.arguments[pattern_name]
        return (
            expression.value
            if isinstance(expression, ast.Constant)
            and isinstance(expression.value, str)
            else None
        )


@dataclass(frozen=True)
class FunctionRegexBundle:
    function_name: str
    literal_lines: dict[str, int]

    @property
    def owner_name(self) -> str:
        return self.function_name.rpartition(".")[0] or "<module>"


@dataclass(frozen=True)
class RepeatedLocalRegexBundleCandidate(FunctionEvidenceLocationsCandidate):
    owner_name: str
    regex_literals: tuple[str, ...]

    @property
    def witness_name(self) -> str:
        return "repeated local regex bundle"


@dataclass(frozen=True)
class RegexBundleModuleProjection:
    module: ParsedModule

    @classmethod
    @lru_cache(maxsize=32768)
    def from_module(cls, module: ParsedModule) -> RegexBundleModuleProjection:
        """Share source facts across scans; analysis memory release clears this cache."""
        return cls(module)

    @cached_property
    def external_reference_ids(self) -> frozenset[int]:
        return frozenset(
            id(surface.reference)
            for surface in ModuleLexicalDependencyProjection.from_module(
                self.module.module
            ).direct_name_surfaces
            if surface.resolution is LexicalNameResolution.EXTERNAL
        )

    @cached_property
    def stable_binding_names(self) -> frozenset[str]:
        declarations = SourceTopLevelDeclarationIndex(
            source_path=self.module.file_path, module=self.module.module
        )
        return frozenset(
            name
            for name, statements in declarations.binding_statements_by_name.items()
            if len(statements) == 1
        )

    @cached_property
    def module_bindings(self) -> ModuleNominalBindingSnapshot:
        return ModuleNominalBindingAuthority(self.module).snapshot_before()

    @cached_property
    def regex_binding_names(self) -> frozenset[str]:
        origins = frozenset(
            origin
            for operation in RegexPatternOperation
            for origin in (operation.function.__module__, operation.value)
        )
        return frozenset(
            name
            for name, binding in self.module_bindings.bindings_by_name.items()
            if binding.qualified_name in origins
        )

    def pattern_literal(self, call: ast.Call) -> str | None:
        root = nominal_reference_root(call.func)
        if (
            root is None
            or root.id not in self.regex_binding_names
            or root.id not in self.stable_binding_names
            or id(root) not in self.external_reference_ids
        ):
            return None
        parts = AstExpressionProjection.attribute_chain(call.func)
        if parts is None:
            return None
        operation = RegexPatternOperation.for_qualified_name(
            self.module_bindings.reference_for(parts).qualified_name
        )
        return None if operation is None else operation.pattern_literal(call)

    def literals_for(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> dict[str, int]:
        literals: dict[str, int] = {}
        for node in walk_function_body_nodes(function):
            if not isinstance(node, ast.Call):
                continue
            literal = self.pattern_literal(node)
            if literal is not None and _is_substantial_regex_literal(literal):
                literals.setdefault(literal, node.lineno)
        return literals

    @classmethod
    def collect(
        cls, module: ParsedModule, config: DetectorConfig
    ) -> tuple[RepeatedLocalRegexBundleCandidate, ...]:
        projection = cls.from_module(module)
        if not projection.regex_binding_names:
            return ()
        grouped: dict[str, list[FunctionRegexBundle]] = defaultdict(list)
        for qualname, function in named_function_nodes(module.module):
            literals = projection.literals_for(function)
            if literals:
                bundle = FunctionRegexBundle(qualname, literals)
                grouped[bundle.owner_name].append(bundle)
        candidates = []
        for owner_name, bundles in grouped.items():
            for left, right in combinations(bundles, 2):
                shared = sorted_tuple(
                    left.literal_lines.keys() & right.literal_lines.keys()
                )
                if len(shared) < config.min_repeated_local_regex_literals:
                    continue
                lines = tuple(
                    min(bundle.literal_lines[pattern] for pattern in shared)
                    for bundle in (left, right)
                )
                candidates.append(
                    RepeatedLocalRegexBundleCandidate(
                        file_path=module.file_path,
                        line=min(lines),
                        owner_name=owner_name,
                        function_names=(left.function_name, right.function_name),
                        regex_literals=shared,
                        line_numbers=lines,
                    )
                )
        return sorted_tuple(
            candidates,
            key=lambda candidate: (
                candidate.file_path,
                candidate.line,
                candidate.function_names,
                candidate.regex_literals,
            ),
        )


def _is_substantial_regex_literal(literal: str) -> bool:
    return (
        len(literal) >= 12
        and any(token in literal for token in ("\\", "[", "(", "{", "^", "$"))
        and sum(char.isalpha() for char in literal) >= 3
    )


class RepeatedLocalRegexBundleDetector(
    ConfiguredModuleCollectorCandidateDetector[RepeatedLocalRegexBundleCandidate]
):
    candidate_collector = RegexBundleModuleProjection.collect
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Sibling functions repeat regex patterns",
        "Calls to imported standard-library regex declarations repeat substantial pattern literals. These sites provide evidence for consolidating syntax; the shared semantic owner remains a practitioner decision.",
        "one declared owner for pattern syntax that is intended to change together",
        "lexically external calls resolve through stable module bindings to regex declarations",
        (CapabilityTag.AUTHORITATIVE_MAPPING, CapabilityTag.PROVENANCE),
        (ObservationTag.NORMALIZED_AST, ObservationTag.DATAFLOW_ROOT),
    )

    def _finding_for_candidate(
        self, candidate: RepeatedLocalRegexBundleCandidate
    ) -> RefactorFinding:
        return self.build_finding(
            f"{candidate.file_path} repeats {len(candidate.regex_literals)} "
            f"regex pattern literals across {', '.join(candidate.function_names)}.",
            candidate.evidence_locations,
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(candidate.function_names),
                mapping_name="repeated regex patterns",
                field_names=candidate.regex_literals,
                source_name=candidate.owner_name,
                identity_field_names=(),
            ),
        )
