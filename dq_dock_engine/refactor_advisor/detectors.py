from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .ast_tools import (
    BuilderCallShape,
    ExportDictShape,
    MethodShape,
    ParsedModule,
    RegistrationShape,
    collect_builder_call_shapes,
    collect_export_dict_shapes,
    collect_method_shapes,
    collect_registration_shapes,
)
from .models import RefactorFinding, SourceLocation


@dataclass(frozen=True)
class DetectorConfig:
    min_duplicate_statements: int = 3
    min_string_cases: int = 3
    min_attribute_probes: int = 2
    min_builder_keywords: int = 3
    min_export_keys: int = 3
    min_registration_sites: int = 2

    @classmethod
    def from_namespace(cls, namespace: Any) -> "DetectorConfig":
        return cls(
            min_duplicate_statements=int(namespace.min_duplicate_statements),
            min_string_cases=int(namespace.min_string_cases),
            min_attribute_probes=int(namespace.min_attribute_probes),
            min_builder_keywords=int(namespace.min_builder_keywords),
            min_export_keys=int(namespace.min_export_keys),
            min_registration_sites=int(namespace.min_registration_sites),
        )


class IssueDetector(ABC):
    detector_id: str

    def detect(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings = self._collect_findings(modules, config)
        return sorted(
            findings,
            key=lambda finding: (finding.pattern_id, finding.title, finding.summary),
        )

    @abstractmethod
    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        raise NotImplementedError

    def _collect_rule_locations(
        self,
        module: ParsedModule,
        rules: tuple["NodeRule", ...],
    ) -> list[SourceLocation]:
        locations: list[SourceLocation] = []
        for node in ast.walk(module.module):
            for rule in rules:
                location = rule.maybe_collect(module, node)
                if location is not None:
                    locations.append(location)
        return locations


class PerModuleIssueDetector(IssueDetector):
    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for module in modules:
            findings.extend(self._findings_for_module(module, config))
        return findings

    @abstractmethod
    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        raise NotImplementedError


class GroupedShapeIssueDetector(IssueDetector):
    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        groups: dict[object, list[object]] = defaultdict(list)
        for shape in self._collect_shapes(modules, config):
            groups[self._group_key(shape)].append(shape)

        findings: list[RefactorFinding] = []
        for shapes in groups.values():
            finding = self._finding_from_group(tuple(shapes), config)
            if finding is not None:
                findings.append(finding)
        return findings

    @abstractmethod
    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        raise NotImplementedError

    @abstractmethod
    def _group_key(self, shape: object) -> object:
        raise NotImplementedError

    @abstractmethod
    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        raise NotImplementedError


def _as_method_shape(shape: object) -> MethodShape:
    if not isinstance(shape, MethodShape):
        raise TypeError(f"Expected MethodShape, got {type(shape)!r}")
    return shape


def _as_builder_shape(shape: object) -> BuilderCallShape:
    if not isinstance(shape, BuilderCallShape):
        raise TypeError(f"Expected BuilderCallShape, got {type(shape)!r}")
    return shape


def _as_registration_shape(shape: object) -> RegistrationShape:
    if not isinstance(shape, RegistrationShape):
        raise TypeError(f"Expected RegistrationShape, got {type(shape)!r}")
    return shape


def _as_export_shape(shape: object) -> ExportDictShape:
    if not isinstance(shape, ExportDictShape):
        raise TypeError(f"Expected ExportDictShape, got {type(shape)!r}")
    return shape


class NodeRule(ABC):
    @abstractmethod
    def maybe_collect(
        self, module: ParsedModule, node: ast.AST
    ) -> SourceLocation | None:
        raise NotImplementedError


@dataclass(frozen=True)
class NamedCallProbeRule(NodeRule):
    function_name: str
    min_args: int
    symbol: str

    def maybe_collect(
        self, module: ParsedModule, node: ast.AST
    ) -> SourceLocation | None:
        if not isinstance(node, ast.Call):
            return None
        if not isinstance(node.func, ast.Name):
            return None
        if node.func.id != self.function_name:
            return None
        if len(node.args) < self.min_args:
            return None
        return SourceLocation(str(module.path), node.lineno, self.symbol)


@dataclass(frozen=True)
class AttributeErrorTryRule(NodeRule):
    symbol: str = "AttributeError"

    def maybe_collect(
        self, module: ParsedModule, node: ast.AST
    ) -> SourceLocation | None:
        if not isinstance(node, ast.Try):
            return None
        for handler in node.handlers:
            if (
                isinstance(handler.type, ast.Name)
                and handler.type.id == "AttributeError"
            ):
                return SourceLocation(str(module.path), handler.lineno, self.symbol)
        return None


class RepeatedPrivateMethodDetector(GroupedShapeIssueDetector):
    detector_id = "repeated_private_methods"

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        return list(_collect_repeated_method_shapes(modules, config))

    def _group_key(self, shape: object) -> object:
        method = _as_method_shape(shape)
        return (method.is_private, method.param_count, method.fingerprint)

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        methods = tuple(
            sorted(
                (_as_method_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        class_names = {method.class_name for method in methods}
        if len(methods) < 2 or len(class_names) < 2:
            return None
        evidence = tuple(
            SourceLocation(method.file_path, method.lineno, method.symbol)
            for method in methods[:6]
        )
        relation = (
            "same private helper role across sibling classes"
            if methods[0].is_private
            else "same method role across sibling classes"
        )
        return RefactorFinding(
            detector_id=self.detector_id,
            pattern_id=5,
            title="Repeated non-orthogonal method skeleton across classes",
            summary=(
                f"{len(methods)} methods across {len(class_names)} classes share the same normalized AST shape."
            ),
            why=(
                "Shared orchestration logic is duplicated across a behavior family. The docs say this shared "
                "non-orthogonal logic should move into an ABC with a concrete template method, leaving only "
                "orthogonal hooks in subclasses."
            ),
            capability_gap="single authoritative algorithm for a nominal behavior family",
            confidence="high",
            relation_context=relation,
            evidence=evidence,
        )


class InheritanceHierarchyCandidateDetector(IssueDetector):
    detector_id = "inheritance_hierarchy_candidate"

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        repeated_groups = _group_repeated_methods(modules, config)
        family_groups: dict[frozenset[str], list[tuple[MethodShape, ...]]] = (
            defaultdict(list)
        )
        for methods in repeated_groups:
            class_names = frozenset(
                method.class_name for method in methods if method.class_name is not None
            )
            if len(class_names) < 2:
                continue
            family_groups[class_names].append(methods)

        findings: list[RefactorFinding] = []
        for class_names, groups in family_groups.items():
            method_count_by_class: dict[str, int] = defaultdict(int)
            for methods in groups:
                for method in methods:
                    if method.class_name is not None:
                        method_count_by_class[method.class_name] += 1
            supports_family = (
                len(groups) >= 2
                or sum(1 for count in method_count_by_class.values() if count >= 2) >= 2
            )
            if not supports_family:
                continue
            evidence: list[SourceLocation] = []
            for methods in groups:
                for method in methods:
                    evidence.append(
                        SourceLocation(method.file_path, method.lineno, method.symbol)
                    )
            findings.append(
                RefactorFinding(
                    detector_id=self.detector_id,
                    pattern_id=5,
                    title="Classes cluster into an ABC hierarchy candidate",
                    summary=(
                        f"Classes {', '.join(sorted(class_names))} share {len(groups)} repeated method-shape groups and repeated method roles that likely want one ABC family."
                    ),
                    why=(
                        "The same set of classes repeats multiple non-orthogonal method skeletons. The docs say this is a "
                        "strong signal that the family should be factored into an ABC with one concrete template method "
                        "layer; orthogonal reusable concerns can then live in mixins so MRO preserves declared precedence."
                    ),
                    capability_gap="single authoritative inheritance hierarchy for a duplicated behavior family",
                    confidence="high",
                    relation_context="same class set repeats several method roles across the same family boundary",
                    evidence=tuple(evidence[:8]),
                )
            )
        return findings


class RepeatedBuilderCallDetector(GroupedShapeIssueDetector):
    detector_id = "repeated_builder_calls"

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        shapes: list[object] = []
        for module in modules:
            for shape in collect_builder_call_shapes(module):
                if len(shape.keyword_names) < config.min_builder_keywords:
                    continue
                shapes.append(shape)
        return shapes

    def _group_key(self, shape: object) -> object:
        builder = _as_builder_shape(shape)
        return (builder.callee_name, builder.keyword_names, builder.value_fingerprint)

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        builders = tuple(
            sorted(
                (_as_builder_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        if len(builders) < 2:
            return None
        owner_symbols = {builder.symbol for builder in builders}
        if len(owner_symbols) < 2:
            return None
        evidence = tuple(
            SourceLocation(builder.file_path, builder.lineno, builder.symbol)
            for builder in builders[:6]
        )
        same_source = all(builder.source_arity == 1 for builder in builders)
        return RefactorFinding(
            detector_id=self.detector_id,
            pattern_id=14,
            title="Repeated field assignment should become an authoritative builder",
            summary=(
                f"Call `{builders[0].callee_name}` repeats the same keyword-mapping shape across {len(builders)} sites."
            ),
            why=(
                "The docs say repeated manual field assignment is an SSOT violation: the mapping should be declared once "
                "in an authoritative constructor, classmethod, or shared builder rather than copied across call sites."
            ),
            capability_gap=(
                "single authoritative data-to-record mapping"
                if same_source
                else "single authoritative record-builder mapping for a repeated constructor family"
            ),
            confidence="medium",
            relation_context=(
                "same builder role repeated across sibling functions or methods"
            ),
            evidence=evidence,
        )


class RepeatedExportDictDetector(GroupedShapeIssueDetector):
    detector_id = "repeated_export_dicts"

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        return [
            shape
            for module in modules
            for shape in collect_export_dict_shapes(module)
            if len(shape.key_names) >= config.min_export_keys
        ]

    def _group_key(self, shape: object) -> object:
        export_shape = _as_export_shape(shape)
        return (export_shape.key_names, export_shape.value_fingerprint)

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        export_shapes = tuple(
            sorted(
                (_as_export_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        if len(export_shapes) < 2:
            return None
        owner_symbols = {shape.symbol for shape in export_shapes}
        if len(owner_symbols) < 2:
            return None
        evidence = tuple(
            SourceLocation(shape.file_path, shape.lineno, shape.symbol)
            for shape in export_shapes[:6]
        )
        return RefactorFinding(
            detector_id=self.detector_id,
            pattern_id=14,
            title="Repeated export dict should become a declarative export schema",
            summary=(
                f"Export dict with keys {export_shapes[0].key_names} repeats across {len(export_shapes)} sites."
            ),
            why=(
                "The docs say repeated JSON/CSV/export dict builders should become one authoritative row dataclass or "
                "declarative export schema instead of many hand-maintained dict literals."
            ),
            capability_gap="single authoritative export projection for a repeated record family",
            confidence="medium",
            relation_context="same export role repeated across sibling functions or methods",
            evidence=evidence,
        )


class ManualClassRegistrationDetector(GroupedShapeIssueDetector):
    detector_id = "manual_class_registration"

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        return [
            shape for module in modules for shape in collect_registration_shapes(module)
        ]

    def _group_key(self, shape: object) -> object:
        registration = _as_registration_shape(shape)
        return registration.registry_name

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        registrations = tuple(
            sorted(
                (_as_registration_shape(shape) for shape in shapes),
                key=lambda item: (item.file_path, item.lineno),
            )
        )
        if len(registrations) < config.min_registration_sites:
            return None
        class_names = {item.registered_class for item in registrations}
        if len(class_names) < config.min_registration_sites:
            return None
        evidence = tuple(
            SourceLocation(item.file_path, item.lineno, item.symbol)
            for item in registrations[:6]
        )
        registry_name = registrations[0].registry_name
        return RefactorFinding(
            detector_id=self.detector_id,
            pattern_id=6,
            title="Manual class registration should become AutoRegisterMeta",
            summary=(
                f"Registry `{registry_name}` is populated manually for {len(class_names)} classes across {len(registrations)} sites."
            ),
            why=(
                "The docs say repeated class-level registration boilerplate is a class-level non-orthogonal algorithm. "
                "It should move into one authoritative metaclass or registry base so abstract-class skipping, uniqueness, "
                "and inheritance behavior are enforced in one place."
            ),
            capability_gap="single authoritative class-registration algorithm with nominal class identity",
            confidence="medium",
            relation_context="same registry key family repeated through manual class-level registration assignments",
            evidence=evidence,
        )


class AttributeProbeDetector(PerModuleIssueDetector):
    detector_id = "attribute_probes"
    probe_rules = (
        NamedCallProbeRule("hasattr", 2, "hasattr"),
        NamedCallProbeRule("getattr", 3, "getattr"),
        AttributeErrorTryRule(),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        evidence = tuple(self._collect_rule_locations(module, self.probe_rules)[:6])
        total = len(evidence)
        if total < config.min_attribute_probes:
            return []
        return [
            RefactorFinding(
                detector_id=self.detector_id,
                pattern_id=5,
                title="Semantic role recovered from attribute probing",
                summary=f"{module.path} contains {total} attribute-probe sites.",
                why=(
                    "Repeated hasattr/getattr/AttributeError logic means the code is recovering identity from a "
                    "partial structural view. The documented fix is to migrate this region toward an ABC contract "
                    "with direct method calls and fail-loud guarantees."
                ),
                capability_gap="declared semantic role identity and import-time enforcement",
                confidence="medium",
                relation_context="same module-level probing layer across multiple call sites",
                evidence=evidence,
            )
        ]


class InlineLiteralDispatchDetector(PerModuleIssueDetector):
    detector_id = "inline_literal_dispatch"

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for owner_name, block in _iter_statement_blocks(module.module):
            for relation_key, evidence in _group_inline_literal_dispatch(
                module, block
            ).items():
                if len(evidence) < config.min_attribute_probes:
                    continue
                findings.append(
                    RefactorFinding(
                        detector_id=self.detector_id,
                        pattern_id=3,
                        title="Inline literal dispatch should be a registry",
                        summary=(
                            f"{module.path} repeats literal-case dispatch over `{relation_key}` in {len(evidence)} sibling branches."
                        ),
                        why=(
                            "When the same observed value is split across several sibling literal branches, the docs "
                            "say the local rule family should be moved into an authoritative registry, dataclass table, "
                            "or another closed dispatch object instead of repeating inline branch logic."
                        ),
                        capability_gap="single authoritative dispatch representation for a closed local rule family",
                        confidence="medium",
                        relation_context=(
                            f"same branch role repeated inline inside {owner_name or 'module block'}"
                        ),
                        evidence=tuple(evidence[:6]),
                    )
                )
        return findings


class StringDispatchDetector(PerModuleIssueDetector):
    detector_id = "string_dispatch"

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        evidence: list[SourceLocation] = []
        for node in ast.walk(module.module):
            if isinstance(node, ast.If):
                string_tests = _count_string_comparisons(node)
                if string_tests >= config.min_string_cases:
                    evidence.append(
                        SourceLocation(
                            str(module.path), node.lineno, "if-string-dispatch"
                        )
                    )
            if isinstance(node, ast.Dict):
                string_keys = sum(
                    1
                    for key in node.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                )
                if string_keys >= config.min_string_cases:
                    evidence.append(
                        SourceLocation(
                            str(module.path), node.lineno, "dict-string-dispatch"
                        )
                    )

        if not evidence:
            return []
        return [
            RefactorFinding(
                detector_id=self.detector_id,
                pattern_id=3,
                title="Closed-family dispatch expressed through strings",
                summary=(
                    f"{module.path} contains {len(evidence)} string-dispatch sites that look like closed variant logic."
                ),
                why=(
                    "The docs prefer enum- or type-keyed O(1) dispatch for closed families. Repeated string branches "
                    "suggest the code is using a weaker representation than the domain requires."
                ),
                capability_gap="closed-family dispatch with stable nominal keys",
                confidence="medium",
                relation_context="same dispatch role repeated through string comparisons or string-key registries",
                evidence=tuple(evidence[:6]),
            )
        ]


class BidirectionalRegistryDetector(PerModuleIssueDetector):
    detector_id = "bidirectional_registry"

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for node in ast.walk(module.module):
            if not isinstance(node, ast.ClassDef):
                continue
            dict_attrs = _collect_dict_attrs(node)
            mirrored_pairs = _collect_mirrored_assignments(node)
            if len(dict_attrs) < 2 or not mirrored_pairs:
                continue
            evidence = tuple(
                SourceLocation(str(module.path), lineno, f"{node.name}.{label}")
                for lineno, label in mirrored_pairs[:6]
            )
            findings.append(
                RefactorFinding(
                    detector_id=self.detector_id,
                    pattern_id=13,
                    title="Bidirectional registry maintained manually",
                    summary=(
                        f"Class {node.name} appears to maintain mirrored forward/reverse registry assignments."
                    ),
                    why=(
                        "The docs prescribe a single authoritative bidirectional type registry when exact companion "
                        "normalization and reverse lookup matter. Manual mirrored assignments are drift-prone and "
                        "should be centralized."
                    ),
                    capability_gap="exact bijection and O(1) reverse lookup on nominal keys",
                    confidence="medium",
                    relation_context="same class maintains forward and reverse registry state",
                    evidence=evidence,
                )
            )
        return findings


def default_detectors() -> tuple[IssueDetector, ...]:
    return (
        RepeatedPrivateMethodDetector(),
        InheritanceHierarchyCandidateDetector(),
        RepeatedBuilderCallDetector(),
        RepeatedExportDictDetector(),
        ManualClassRegistrationDetector(),
        AttributeProbeDetector(),
        InlineLiteralDispatchDetector(),
        StringDispatchDetector(),
        BidirectionalRegistryDetector(),
    )


def _count_string_comparisons(node: ast.If) -> int:
    count = 0
    current: ast.stmt | None = node
    while isinstance(current, ast.If):
        count += _string_comparisons_in_test(current.test)
        current = current.orelse[0] if len(current.orelse) == 1 else None
    return count


def _string_comparisons_in_test(node: ast.AST) -> int:
    if isinstance(node, ast.Compare):
        values = [node.left] + list(node.comparators)
        if any(
            isinstance(value, ast.Constant) and isinstance(value.value, str)
            for value in values
        ):
            return 1
    if isinstance(node, ast.BoolOp):
        return sum(_string_comparisons_in_test(value) for value in node.values)
    return 0


def _collect_dict_attrs(node: ast.ClassDef) -> set[str]:
    dict_attrs: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        if not isinstance(child.value, ast.Dict):
            continue
        for target in child.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                dict_attrs.add(target.attr)
    return dict_attrs


def _collect_mirrored_assignments(node: ast.ClassDef) -> list[tuple[int, str]]:
    mirrored: list[tuple[int, str]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        for target in child.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Attribute):
                continue
            if (
                not isinstance(target.value.value, ast.Name)
                or target.value.value.id != "self"
            ):
                continue
            if isinstance(child.value, ast.Name):
                mirrored.append((child.lineno, target.value.attr))
    return mirrored


def _iter_statement_blocks(
    module: ast.Module,
) -> list[tuple[str | None, list[ast.stmt]]]:
    blocks: list[tuple[str | None, list[ast.stmt]]] = [(None, module.body)]

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            blocks.append((node.name, node.body))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            blocks.append((node.name, node.body))
            self.generic_visit(node)

    Visitor().visit(module)
    return blocks


def _group_inline_literal_dispatch(
    module: ParsedModule,
    block: list[ast.stmt],
) -> dict[str, list[SourceLocation]]:
    groups: dict[str, list[SourceLocation]] = defaultdict(list)
    for stmt in block:
        if not isinstance(stmt, ast.If):
            continue
        key = _literal_dispatch_key(stmt.test)
        if key is None:
            continue
        groups[key].append(
            SourceLocation(str(module.path), stmt.lineno, "inline-literal-dispatch")
        )
    return {key: items for key, items in groups.items() if len(items) >= 2}


def _literal_dispatch_key(test: ast.AST) -> str | None:
    if not isinstance(test, ast.Compare):
        return None
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    if len(test.comparators) != 1:
        return None
    comparator = test.comparators[0]
    if not (isinstance(comparator, ast.Constant) and isinstance(comparator.value, str)):
        return None
    return ast.dump(test.left, include_attributes=False)


def _collect_repeated_method_shapes(
    modules: list[ParsedModule], config: DetectorConfig
) -> tuple[MethodShape, ...]:
    return tuple(
        method
        for module in modules
        for method in collect_method_shapes(module)
        if method.class_name
        and method.statement_count >= config.min_duplicate_statements
    )


def _group_repeated_methods(
    modules: list[ParsedModule], config: DetectorConfig
) -> list[tuple[MethodShape, ...]]:
    groups: dict[tuple[bool, int, str], list[MethodShape]] = defaultdict(list)
    for method in _collect_repeated_method_shapes(modules, config):
        key = (method.is_private, method.param_count, method.fingerprint)
        groups[key].append(method)
    return [
        tuple(sorted(methods, key=lambda item: (item.file_path, item.lineno)))
        for methods in groups.values()
        if len(methods) >= 2 and len({method.class_name for method in methods}) >= 2
    ]
