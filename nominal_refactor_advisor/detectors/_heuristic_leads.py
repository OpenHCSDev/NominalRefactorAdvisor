"""High-recall source leads; no inferred owner or executable migration."""

from __future__ import annotations

import ast
from collections import defaultdict

from ..ast_tools import ModuleSyntaxIndex, ParsedModule, module_syntax_index
from ..models import RefactorFinding, SourceLocation
from ..patterns import PatternId
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import DetectorConfig, PerModuleIssueDetector, finding_spec_template


def _lexical_symbol(index: ModuleSyntaxIndex, node_index: int) -> str:
    owners = (
        ancestor.name
        for ancestor in index.ancestor_nodes(node_index)
        if isinstance(ancestor, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return ".".join(owners) or "<module>"


class LongConditionFactoringLeadDetector(PerModuleIssueDetector):
    """Expose original decision terms, never classify a long test as debt by length."""

    finding_spec = finding_spec_template(
        PatternId.SOURCE_BACKED_CONDITION_LEAD,
        "Long boolean decision raises a factoring question",
        "A long decision can repeat a fact, probe another object's state, or combine genuinely independent predicates. Trace its terms to their owners before changing it; retain Python's original short-circuit order and effects.",
        "source-backed question only; owner, equivalence and migration recipe OPEN",
        "four or more direct boolean terms in one original if/while decision test",
        (CapabilityTag.PROVENANCE,),
        (ObservationTag.PARTIAL_VIEW,),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        index = module_syntax_index(module.module)
        findings: list[RefactorFinding] = []
        for node_index, node in sorted(
            (
                *index.indexed_nodes_of_type(ast.If),
                *index.indexed_nodes_of_type(ast.While),
            )
        ):
            test = node.test
            if not isinstance(test, ast.BoolOp) or len(test.values) < 4:
                continue
            symbol = _lexical_symbol(index, node_index)
            original_terms = tuple(
                ast.get_source_segment(module.source, term) or ast.unparse(term)
                for term in test.values
            )
            findings.append(
                self.build_finding(
                    f"{module.path} decision in {symbol} at line {node.lineno} has "
                    f"{len(original_terms)} direct {'and' if isinstance(test.op, ast.And) else 'or'} "
                    f"terms in source order {original_terms}; common fact or owner, "
                    "reachability, short-circuit effects and replacement OPEN.",
                    (SourceLocation(module.file_path, node.lineno, symbol),),
                )
            )
        return findings


class RepeatedLiteralRosterLeadDetector(PerModuleIssueDetector):
    """Find exact repeated literal containers without inferring a domain family."""

    finding_spec = finding_spec_template(
        PatternId.SOURCE_BACKED_ROSTER_LEAD,
        "Repeated literal roster raises a membership authority question",
        "Identical local literal containers may restate one family, capability or external schema, or be unrelated values. Inspect the use sites and domain authority before deriving a roster or proposing case classes.",
        "exact source container repetition only; family, authority and migration recipe OPEN",
        "two or more literal string containers of at least three values with identical values and container kind within a module",
        (CapabilityTag.PROVENANCE,),
        (ObservationTag.PARTIAL_VIEW,),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        index = module_syntax_index(module.module)
        groups: dict[tuple[str, tuple[str, ...]], list[SourceLocation]] = defaultdict(
            list
        )
        for node_index, node in sorted(
            (
                *index.indexed_nodes_of_type(ast.Set),
                *index.indexed_nodes_of_type(ast.Tuple),
                *index.indexed_nodes_of_type(ast.List),
            )
        ):
            values = node.elts
            if len(values) < 3 or not all(
                isinstance(value, ast.Constant) and isinstance(value.value, str)
                for value in values
            ):
                continue
            literals = tuple(value.value for value in values)
            # A set is a membership view; tuples and lists retain source order.
            if isinstance(node, ast.Set):
                if len(set(literals)) != len(literals):
                    continue
                literals = tuple(sorted(literals))
            kind = type(node).__name__
            groups[(kind, literals)].append(
                SourceLocation(
                    module.file_path, node.lineno, _lexical_symbol(index, node_index)
                )
            )
        findings: list[RefactorFinding] = []
        for (kind, literals), sites in groups.items():
            if len(sites) < 2:
                continue
            findings.append(
                self.build_finding(
                    f"{module.path} repeats {kind} literals {literals} at original "
                    f"source lines {tuple(site.line for site in sites)}; whether these "
                    "sites express one family, capability, schema or unrelated "
                    "values, and whether order/effects can change, remains OPEN.",
                    tuple(sites),
                )
            )
        return findings
