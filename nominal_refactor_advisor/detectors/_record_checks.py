"""Declared attribute-contract detector over the shared semantic projection."""

from __future__ import annotations

from ..class_index import CompactClassFamilyIndex
from ..models import ProbeCountMetrics, RefactorFinding
from ..patterns import PatternId
from ..record_checks import DeclaredAttributeContractResolver
from ..semantic_descent import CompactSemanticDescentRepository
from ._base import CompactProjectionGroups, DetectorConfig, high_confidence_spec
from ._semantic_descent import SemanticProjectionDetector


class RedundantTypeCheckDetector(SemanticProjectionDetector):
    """Source-backed leads for rechecking an already declared attribute type."""

    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Type check repeats a declared attribute contract",
        "An annotated parameter or self already names the field's declared owner. "
        "Validate external values at decoding, then trust that contract. Python does "
        "not enforce annotations; exact-type checks can further exclude subclasses, "
        "so this lead does not certify deleting a runtime check.",
        "one type-validation boundary and trusted nominal field consumers",
        "attribute type is redeclared by a type or isinstance check",
    )

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        repository = CompactSemanticDescentRepository.from_projection_groups(
            projections_by_family, class_index=CompactClassFamilyIndex.require(context)
        )
        resolver = DeclaredAttributeContractResolver(
            repository.class_index,
            repository.class_reference_resolver,
            {
                module.module_name: module.type_checks
                for module in repository.semantic_projections
            },
        )
        findings = []
        for module in repository.semantic_projections:
            for check in module.type_checks.checks:
                resolved = resolver.resolve(module.module_name, check)
                if resolved is None:
                    continue
                declaration, annotation = resolved
                findings.append(
                    self.build_finding(
                        f"`{check.expression}` repeats declared attribute type `{annotation}`. "
                        "Boundary validation and exact-type/subclass intent remain to be checked before removal.",
                        (check.location, declaration),
                        metrics=ProbeCountMetrics(1),
                    )
                )
        return findings
