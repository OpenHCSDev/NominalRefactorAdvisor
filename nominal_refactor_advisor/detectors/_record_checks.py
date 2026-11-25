"""Declared attribute-contract detector over the shared semantic projection."""

from __future__ import annotations

from nominal_refactor_advisor.class_index import (
    CompactClassFamilyIndex,
    CompactModuleClassProjection,
)
from nominal_refactor_advisor.models import ProbeCountMetrics, RefactorFinding
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.record_checks import DeclaredAttributeContractResolver
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.semantic_descent import CompactSemanticModuleProjection
from nominal_refactor_advisor.detectors._base import (
    CompactProjectionGroups,
    DetectorConfig,
    high_confidence_spec,
)
from nominal_refactor_advisor.detectors._semantic_descent import (
    SemanticProjectionDetector,
)


@ScanCache.cached
def _base_redundant_type_checks_slot() -> list[
    tuple[
        dict[str, CompactSemanticModuleProjection],
        dict[str, CompactModuleClassProjection],
        CompactClassFamilyIndex,
        dict[str, tuple[RefactorFinding, ...]],
        dict[str, frozenset[str]],
        dict[str, frozenset[str]],
    ]
    | None
]:
    return [None]


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
        repository = self._semantic_repository(projections_by_family, context)
        sem_projs = repository.semantic_projections
        class_projs = repository.class_projections
        if ScanCache.active() is not None:
            slot = _base_redundant_type_checks_slot()
            base_entry = slot[0]
            if base_entry is not None:
                (
                    base_sem_by_path,
                    base_class_by_path,
                    base_index,
                    base_findings_by_path,
                    base_deps_by_path,
                    base_unresolved_by_type_name,
                ) = base_entry
                changed_sem_paths = {
                    m.file_path
                    for m in sem_projs
                    if base_sem_by_path.get(m.file_path) is not m
                }
                cur_sem_paths = {m.file_path for m in sem_projs}
                removed_sem_paths = set(base_sem_by_path) - cur_sem_paths
                changed_class_paths = {
                    p.file_path
                    for p in class_projs
                    if base_class_by_path.get(p.file_path) is not p
                }
                cur_class_paths = {p.file_path for p in class_projs}
                removed_class_paths = set(base_class_by_path) - cur_class_paths
                if (
                    len(changed_sem_paths)
                    + len(removed_sem_paths)
                    + len(changed_class_paths)
                    + len(removed_class_paths)
                    <= 64
                ):
                    cur_index = repository.class_index
                    changed_class_syms: set[str] = set()
                    added_class_simple_names: set[str] = set()
                    for path in changed_class_paths | removed_class_paths:
                        old_p = base_class_by_path.get(path)
                        if old_p is not None:
                            changed_class_syms.update(c.symbol for c in old_p.classes)
                    for p in class_projs:
                        if p.file_path in changed_class_paths:
                            old_p = base_class_by_path.get(p.file_path)
                            old_syms = {c.symbol for c in old_p.classes} if old_p else set()
                            for c in p.classes:
                                changed_class_syms.add(c.symbol)
                                if c.symbol not in old_syms:
                                    added_class_simple_names.add(c.simple_name)
                    # Include any class declared in a module whose bindings changed
                    for m in sem_projs:
                        if m.file_path in changed_sem_paths:
                            old_m = base_sem_by_path.get(m.file_path)
                            if old_m is None or old_m.type_checks.bindings != m.type_checks.bindings:
                                cp = base_class_by_path.get(m.file_path)
                                if cp is not None:
                                    changed_class_syms.update(c.symbol for c in cp.classes)
                    affected_syms = set(changed_class_syms)
                    for sym in changed_class_syms:
                        affected_syms.update(base_index.descendant_symbols(sym))
                        affected_syms.update(cur_index.descendant_symbols(sym))
                    dirty_paths = set(changed_sem_paths)
                    if added_class_simple_names:
                        for type_name in added_class_simple_names:
                            dirty_paths.update(
                                base_unresolved_by_type_name.get(type_name, ())
                            )
                    if affected_syms:
                        for path, deps in base_deps_by_path.items():
                            if path in cur_sem_paths and not deps.isdisjoint(affected_syms):
                                dirty_paths.add(path)
                    resolver = (
                        DeclaredAttributeContractResolver(
                            cur_index,
                            repository.class_reference_resolver,
                            {m.module_name: m.type_checks for m in sem_projs},
                        )
                        if dirty_paths
                        else None
                    )
                    findings: list[RefactorFinding] = []
                    for module in sem_projs:
                        if module.file_path not in dirty_paths:
                            cached = base_findings_by_path.get(module.file_path)
                            if cached:
                                findings.extend(cached)
                            continue
                        assert resolver is not None
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

        resolver = DeclaredAttributeContractResolver(
            repository.class_index,
            repository.class_reference_resolver,
            {module.module_name: module.type_checks for module in sem_projs},
        )
        findings = []
        findings_by_path: dict[str, tuple[RefactorFinding, ...]] = {}
        deps_by_path: dict[str, frozenset[str]] = {}
        unresolved_by_type_name_mut: dict[str, set[str]] = {}
        for module in sem_projs:
            mod_findings: list[RefactorFinding] = []
            mod_deps: set[str] = set()
            for check in module.type_checks.checks:
                sym = repository.class_reference_resolver.symbol_for(
                    module_name=module.module_name,
                    reference_parts=check.subject_type,
                    allow_unique_unqualified=False,
                )
                if sym is not None:
                    mod_deps.add(sym)
                elif check.subject_type:
                    unresolved_by_type_name_mut.setdefault(
                        check.subject_type[-1], set()
                    ).add(module.file_path)
                resolved = resolver.resolve(module.module_name, check)
                if resolved is None:
                    continue
                declaration, annotation = resolved
                if declaration.symbol:
                    mod_deps.add(declaration.symbol)
                finding = self.build_finding(
                    f"`{check.expression}` repeats declared attribute type `{annotation}`. "
                    "Boundary validation and exact-type/subclass intent remain to be checked before removal.",
                    (check.location, declaration),
                    metrics=ProbeCountMetrics(1),
                )
                mod_findings.append(finding)
                findings.append(finding)
            if mod_findings:
                findings_by_path[module.file_path] = tuple(mod_findings)
            if mod_deps:
                deps_by_path[module.file_path] = frozenset(mod_deps)
        if ScanCache.active() is not None:
            slot = _base_redundant_type_checks_slot()
            if slot[0] is None:
                slot[0] = (
                    {m.file_path: m for m in sem_projs},
                    {p.file_path: p for p in class_projs},
                    repository.class_index,
                    findings_by_path,
                    deps_by_path,
                    {
                        name: frozenset(paths)
                        for name, paths in unresolved_by_type_name_mut.items()
                    },
                )
        return findings
