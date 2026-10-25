"""High-recall source leads; no inferred owner or executable migration."""

from __future__ import annotations

import ast
from collections import defaultdict

from ..ast_tools import ParsedModule, module_syntax_index
from ..models import RefactorFinding, SourceLocation
from ..observation_families import ModuleExpressionShapeFamily
from ..observation_shapes import ModuleExpressionShape
from ..patterns import PatternId
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    CANDIDATE_COLLECTION_AUTHORITY,
    CompactGroupedShapeIssueDetector,
    DetectorConfig,
    PerModuleIssueDetector,
    finding_spec_template,
)


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
            symbol = index.lexical_symbol(node_index)
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
                    module.file_path, node.lineno, index.lexical_symbol(node_index)
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


class RepeatedModuleExpressionLeadDetector(
    CompactGroupedShapeIssueDetector[ModuleExpressionShape, str]
):
    """Find one module-attribute expression written out across modules."""

    module_projection_family = ModuleExpressionShapeFamily
    compact_report_context_requires_target_projection = True
    finding_spec = finding_spec_template(
        PatternId.SOURCE_BACKED_MODULE_EXPRESSION_LEAD,
        "Repeated module expression raises a value-owner question",
        "An expression computed only from imported modules and constants, written out at several sites, restates one value each time. Inspect what the value means at each site before declaring one owner for it.",
        "exact source expression repetition only; owner, meaning and migration recipe OPEN",
        "one maximal expression over imported-module attributes and constants at or above the configured site count across the scanned modules",
        (CapabilityTag.PROVENANCE,),
        (ObservationTag.PARTIAL_VIEW,),
    )

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[ModuleExpressionShape]:
        return [
            shape
            for module in modules
            for shape in CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module, ModuleExpressionShapeFamily, ModuleExpressionShape
            )
        ]

    def _group_key(self, shape: ModuleExpressionShape) -> str:
        return shape.expression_key

    def _finding_from_group(
        self, shapes: tuple[ModuleExpressionShape, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        if len(shapes) < config.min_repeated_module_expression_sites:
            return None
        sites = sorted(shapes, key=lambda shape: (shape.file_path, shape.lineno))
        modules = {shape.file_path for shape in sites}
        return self.build_finding(
            f"`{sites[0].expression_text}` is written out at {len(sites)} sites in "
            f"{len(modules)} module(s); whether one declaration should own this "
            "value, and what each site means by it, remains OPEN.",
            tuple(
                SourceLocation(shape.file_path, shape.lineno, shape.symbol)
                for shape in sites
            ),
        )
