"""High-recall source leads; no inferred owner or executable migration."""

from __future__ import annotations

import ast
import inspect
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Self, TypeVar

from ..ast_tools import ParsedModuleSet
from ..ast_tools import ParsedModule
from ..collection_algebra import sorted_tuple
from ..models import RefactorFinding, SourceLocation
from ..observation_families import (
    TypeTestModuleFamily,
    ModuleExpressionShapeFamily,
)
from ..observation_shapes import (
    TypeTestModule,
    TypeTestShape,
    ModuleExpressionShape,
)
from ..patterns import PatternId
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    CANDIDATE_COLLECTION_AUTHORITY,
    CompactGroupedShapeIssueDetector,
    DetectorConfig,
    CompactModuleProjectionDetectorMixin,
    IssueDetector,
    PerModuleIssueDetector,
    finding_spec_template,
)


T = TypeVar("T")

# Findings list this many type sets or divergence pairs; the rest are counted.
DIVERGENCE_LISTING_LIMIT = 8


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
        index = module.syntax_index
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
                    f"decision in {symbol} has "
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
        index = module.syntax_index
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
        self, modules: ParsedModuleSet, config: DetectorConfig
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


class ForeignTypeSwitchDetector(
    CompactModuleProjectionDetectorMixin[TypeTestModule], IssueDetector
):
    """Find `isinstance` tests on one foreign module's classes across the scan.

    Every site counts: a module may legitimately hold exactly one decoder for a
    family, but this detector does not try to recognise decoders.
    """

    module_projection_family = TypeTestModuleFamily

    finding_spec = finding_spec_template(
        PatternId.FOREIGN_TYPE_SWITCH_LEAD,
        "Type tests on foreign classes raise a closed-set question",
        "Several sites test instances against classes owned by one module outside this repository. Each site restates which of that module's classes it handles; a site that tests a strict subset of another site's set may have missed a case.",
        "isinstance/issubclass sites whose type argument resolves through imports or builtins to classes of one non-repository module; closedness, owner and migration recipe OPEN",
        "isinstance/issubclass sites on classes of one foreign module at or above the configured site count across the scanned modules",
        (CapabilityTag.PROVENANCE,),
        (ObservationTag.PARTIAL_VIEW,),
    )

    def _findings_from_compact_projections(
        self,
        projections: Sequence[TypeTestModule],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        scanned = tuple(projection.module_name for projection in projections)
        groups: dict[str, list[TypeTestShape]] = defaultdict(list)
        for projection in projections:
            for shape in projection.sites:
                if len(shape.owners) == 1 and shape.is_foreign_to(scanned):
                    (owner,) = shape.owners
                    if owner:
                        groups[owner].append(shape)
        findings = (
            self._finding_from_group(owner, tuple(shapes), config)
            for owner, shapes in groups.items()
        )
        return [finding for finding in findings if finding is not None]

    def _finding_from_group(
        self, owner: str, shapes: tuple[TypeTestShape, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        if len(shapes) < config.min_foreign_type_switch_sites:
            return None
        sites = sorted(shapes, key=lambda shape: (shape.file_path, shape.lineno))
        type_sets = Counter(shape.type_names for shape in sites)
        modules = {shape.file_path for shape in sites}
        lines = [
            f"{len(sites)} isinstance/issubclass sites in {len(modules)} module(s) "
            f"test classes of `{owner}`, which this repository does not "
            "declare. Type sets seen: "
            + _counted_list(
                type_sets.most_common(),
                lambda type_set: _type_set_text(type_set),
                "type sets",
            )
            + ". Whether each set is meant to be closed, and which site should own "
            "it, remains OPEN."
        ]
        misses = ForeignClassTwins.from_type_sets(type_sets).misses(type_sets)
        if misses:
            lines.append(
                f"{sum(type_sets[type_set] for type_set, _twins in misses)} sites "
                "test a class without its same-shape twin (possible missed case): "
                + _counted_list(
                    sorted(
                        ((miss, type_sets[miss[0]]) for miss in misses),
                        key=lambda item: -item[1],
                    ),
                    lambda miss: (
                        f"{_type_set_text(miss[0])} without {_type_set_text(miss[1])}"
                    ),
                    "type sets",
                )
            )
        return self.build_finding(
            "\n".join(lines),
            tuple(
                SourceLocation(shape.file_path, shape.lineno, shape.symbol)
                for shape in sites
            ),
        )


def _counted_list(
    entries: Sequence[tuple[T, int]], render: Callable[[T], str], noun: str
) -> str:
    """The most frequent entries with their site counts, then how many were cut."""

    shown = entries[:DIVERGENCE_LISTING_LIMIT]
    text = "; ".join(f"{render(item)} x{count}" for item, count in shown)
    hidden = len(entries) - len(shown)
    return f"{text}; and {hidden} more {noun}" if hidden else text


def _type_set_text(type_set: tuple[str, ...]) -> str:
    return "`" + " | ".join(type_set) + "`"


@dataclass(frozen=True)
class ForeignClassTwins:
    """Foreign classes sharing a base, match fields and field types.

    Twins are one shape under two names (``ast.FunctionDef`` and
    ``ast.AsyncFunctionDef``), so a type test naming one without the other is a
    likely missed case. Classes are read only from modules already imported;
    nothing is imported for analysis, and an unreadable class has no twins.
    """

    twins_by_name: Mapping[str, frozenset[str]]

    @classmethod
    def from_type_sets(cls, type_sets: Iterable[tuple[str, ...]]) -> Self:
        names_by_shape: dict[tuple[object, ...], set[str]] = defaultdict(set)
        for name in {name for type_set in type_sets for name in type_set}:
            shape = cls._shape(name)
            if shape is not None:
                names_by_shape[shape].add(name)
        return cls(
            {
                name: frozenset(names - {name})
                for names in names_by_shape.values()
                if len(names) > 1
                for name in names
            }
        )

    @staticmethod
    def _shape(qualified_name: str) -> tuple[object, ...] | None:
        module_name, _, attribute = qualified_name.rpartition(".")
        module = sys.modules.get(module_name)
        declared = None if module is None else vars(module).get(attribute)
        if not isinstance(declared, type):
            return None
        match_fields = vars(declared).get("__match_args__", ())
        if not match_fields:
            return None
        field_types = tuple(
            (field_name, repr(field_type))
            for field_name, field_type in inspect.get_annotations(declared).items()
        )
        return (declared.__bases__, match_fields, field_types)

    def misses(
        self, type_sets: Iterable[tuple[str, ...]]
    ) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
        """Each type set paired with the twins of its members that it omits."""

        return tuple(
            (type_set, missing)
            for type_set in sorted(type_sets)
            for missing in (
                sorted_tuple(
                    {
                        twin
                        for name in type_set
                        for twin in self.twins_by_name.get(name, ())
                    }
                    - set(type_set)
                ),
            )
            if missing
        )