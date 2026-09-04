"""Declaration-owned architecture guard constraints."""

from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import (
    dataclass,
    replace,
)
from enum import StrEnum
from functools import cached_property
from typing import (
    ClassVar,
    Self,
    cast,
)

from metaclass_registry import AutoRegisterMeta

from .ast_tools import AstExpressionProjection
from .codemod_paths import SourcePathCandidateSet, SourcePathResolutionAuthority
from .codemod_payload import (
    CodemodJsonReport,
    CodemodPayloadRecord,
    DataclassJsonReport,
    DiscriminatedPayloadRecord,
    EmptyDefaultStringPayloadValueCodec,
    JSON_REPORT_VALUE_PROJECTION,
    JsonObject,
    JsonValue,
    OptionalStringPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    PayloadValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
    json_report_field,
    json_report_property,
)
from .collection_algebra import sorted_tuple
from .models import SourceLocation
from .source_index import (
    AstTargetDigest,
    SourceIndex,
)


@dataclass(frozen=True)
class ArchitectureGuardMatch:
    """One declaration-owned guard match before repository context is attached."""

    node: ast.expr | ast.stmt
    constraint_type: type[ArchitectureGuardConstraint]
    symbol: str
    message: str


@dataclass(frozen=True)
class ArchitectureGuardConstraint(
    DiscriminatedPayloadRecord,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered owner of one architecture invariant and its AST evidence."""

    __registry__: ClassVar[dict[str, type["ArchitectureGuardConstraint"]]] = {}
    __registry_key__ = "constraint_key_value"
    __skip_if_no_key__ = True
    discriminator_field_name: ClassVar[str] = "constraint"
    constraint_key_value: ClassVar[str]

    @classmethod
    def record_type_for_discriminator(cls, discriminator: str) -> type[Self]:
        constraint_type = cls.__registry__.get(discriminator)
        if constraint_type is None or not issubclass(constraint_type, cls):
            raise ValueError(
                f"Unsupported architecture guard constraint: {discriminator}"
            )
        return cast(type[Self], constraint_type)

    @classmethod
    def discriminator_key(cls) -> str:
        return cls.constraint_key_value

    def match(
        self,
        node: ast.expr | ast.stmt,
        symbol: str,
        message: str,
    ) -> ArchitectureGuardMatch:
        return ArchitectureGuardMatch(
            node=node,
            constraint_type=type(self),
            symbol=symbol,
            message=message,
        )

    @abstractmethod
    def matches(self, module: ast.Module) -> tuple[ArchitectureGuardMatch, ...]:
        """Return every source node violating this declared constraint."""

        raise NotImplementedError


@dataclass(frozen=True)
class ForbiddenNameArchitectureGuardConstraint(
    ArchitectureGuardConstraint,
    ABC,
):
    """Shared declared source names for call and attribute constraints."""

    names: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())


@dataclass(frozen=True)
class ForbiddenCallArchitectureGuardConstraint(
    ForbiddenNameArchitectureGuardConstraint,
):
    """Forbid calls whose canonical source expression is declared here."""

    constraint_key_value: ClassVar[str] = "forbidden_calls"

    def matches(self, module: ast.Module) -> tuple[ArchitectureGuardMatch, ...]:
        return tuple(
            self.match(
                node,
                call_name,
                f"Forbidden call {call_name!r}",
            )
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            for call_name in (AstExpressionProjection.qualified_name(node.func),)
            if call_name in self.names
        )


@dataclass(frozen=True)
class ForbiddenAttributeArchitectureGuardConstraint(
    ForbiddenNameArchitectureGuardConstraint,
):
    """Forbid attribute access whose terminal name is declared here."""

    constraint_key_value: ClassVar[str] = "forbidden_attributes"

    def matches(self, module: ast.Module) -> tuple[ArchitectureGuardMatch, ...]:
        return tuple(
            self.match(
                node,
                node.attr,
                f"Forbidden attribute {node.attr!r}",
            )
            for node in ast.walk(module)
            if isinstance(node, ast.Attribute) and node.attr in self.names
        )


@dataclass(frozen=True)
class ArchitectureGuardDispatchSubject:
    """Parsed semantic dispatch axis supplied at the JSON boundary as source."""

    source: str

    @cached_property
    def expression(self) -> ast.expr:
        return ast.parse(self.source, mode="eval").body

    @cached_property
    def fingerprint(self) -> str:
        return ast.dump(self.expression, include_attributes=False)

    def matches(self, expression: ast.AST) -> bool:
        return ast.dump(expression, include_attributes=False) == self.fingerprint

    def matches_runtime_type_call(self, expression: ast.AST) -> bool:
        return (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "type"
            and len(expression.args) == 1
            and not expression.keywords
            and self.matches(expression.args[0])
        )

    def matches_dispatch_expression(self, expression: ast.AST) -> bool:
        return self.matches(expression) or self.matches_runtime_type_call(expression)

    @classmethod
    def accepts_value_case(cls, expression: ast.AST) -> bool:
        return (
            (isinstance(expression, ast.Constant) and expression.value is not None)
            or isinstance(expression, ast.Attribute)
            or (
                isinstance(expression, (ast.Tuple, ast.List, ast.Set))
                and any(cls.accepts_value_case(item) for item in expression.elts)
            )
        )

    @classmethod
    def accepts_runtime_type_case(cls, expression: ast.AST) -> bool:
        return isinstance(expression, (ast.Name, ast.Attribute)) or (
            isinstance(expression, ast.Tuple)
            and any(cls.accepts_runtime_type_case(item) for item in expression.elts)
        )

    def comparison_relation_matches(
        self,
        left: ast.AST,
        right: ast.AST,
    ) -> bool:
        return (
            (self.matches(left) and self.accepts_value_case(right))
            or (self.matches(right) and self.accepts_value_case(left))
            or (
                self.matches_runtime_type_call(left)
                and self.accepts_runtime_type_case(right)
            )
            or (
                self.matches_runtime_type_call(right)
                and self.accepts_runtime_type_case(left)
            )
        )

    def comparison_matches(self, comparison: ast.Compare) -> bool:
        operands = (comparison.left, *comparison.comparators)
        return any(
            isinstance(
                operator,
                (ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.In, ast.NotIn),
            )
            and self.comparison_relation_matches(left, right)
            for left, operator, right in zip(
                operands,
                comparison.ops,
                operands[1:],
            )
        )

    def isinstance_call_matches(self, call: ast.Call) -> bool:
        return (
            isinstance(call.func, ast.Name)
            and call.func.id == "isinstance"
            and len(call.args) >= 2
            and self.matches(call.args[0])
        )

    def conditional_matches(self, test: ast.AST) -> bool:
        candidates = tuple(ast.walk(test))
        return any(
            self.comparison_matches(candidate)
            for candidate in candidates
            if isinstance(candidate, ast.Compare)
        ) or any(
            self.isinstance_call_matches(candidate)
            for candidate in candidates
            if isinstance(candidate, ast.Call)
        )


class ArchitectureGuardDispatchSiteKind(StrEnum):
    """Syntax forms that recover a forbidden semantic case downstream."""

    CONDITIONAL = "conditional"
    MATCH = "match/case"
    INLINE_MAPPING = "inline mapping"

    def message(self, subject: ArchitectureGuardDispatchSubject) -> str:
        return f"Forbidden {self.value} dispatch over {subject.source!r}"


class _ForbiddenDispatchCollector(ast.NodeVisitor):
    """AST-native dispatch over syntax leaves for one forbidden semantic axis."""

    def __init__(
        self,
        constraint: ForbiddenDispatchArchitectureGuardConstraint,
    ) -> None:
        self.constraint = constraint
        self.subjects = tuple(
            ArchitectureGuardDispatchSubject(item) for item in constraint.subjects
        )
        self.matches: list[ArchitectureGuardMatch] = []

    def _append(
        self,
        node: ast.expr | ast.stmt,
        subject: ArchitectureGuardDispatchSubject,
        site_kind: ArchitectureGuardDispatchSiteKind,
    ) -> None:
        self.matches.append(
            self.constraint.match(
                node,
                subject.source,
                site_kind.message(subject),
            )
        )

    def visit_If(self, node: ast.If) -> None:
        for subject in self.subjects:
            if subject.conditional_matches(node.test):
                self._append(
                    node,
                    subject,
                    ArchitectureGuardDispatchSiteKind.CONDITIONAL,
                )
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        for subject in self.subjects:
            if subject.matches_dispatch_expression(node.subject):
                self._append(node, subject, ArchitectureGuardDispatchSiteKind.MATCH)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Dict):
            for subject in self.subjects:
                if subject.matches(node.slice):
                    self._append(
                        node,
                        subject,
                        ArchitectureGuardDispatchSiteKind.INLINE_MAPPING,
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Dict)
            and node.args
        ):
            for subject in self.subjects:
                if subject.matches(node.args[0]):
                    self._append(
                        node,
                        subject,
                        ArchitectureGuardDispatchSiteKind.INLINE_MAPPING,
                    )
        self.generic_visit(node)


@dataclass(frozen=True)
class ForbiddenDispatchArchitectureGuardConstraint(ArchitectureGuardConstraint):
    """Forbid downstream case recovery for declared semantic subjects."""

    constraint_key_value: ClassVar[str] = "forbidden_dispatch"
    subjects: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())

    def matches(self, module: ast.Module) -> tuple[ArchitectureGuardMatch, ...]:
        collector = _ForbiddenDispatchCollector(self)
        collector.visit(module)
        return tuple(collector.matches)


@dataclass(frozen=True)
class ArchitectureGuardTargetScope(CodemodPayloadRecord):
    """One source path and optional nominal target guarded as a unit."""

    file_path: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    target_qualname: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )

    def resolve(
        self,
        source_index: SourceIndex,
    ) -> ResolvedArchitectureGuardTargetScope:
        file_path = SourcePathResolutionAuthority(
            requested_path=self.file_path,
            candidate_set=SourcePathCandidateSet.from_paths(
                source_index.target_file_paths
            ),
        ).required_path()
        if self.target_qualname is not None:
            matching_targets = tuple(
                target
                for target in source_index.targets_by_file[file_path]
                if target.qualname == self.target_qualname
            )
            if len(matching_targets) != 1:
                raise ValueError(
                    f"Architecture guard target qualname {self.target_qualname!r} "
                    f"did not resolve exactly once in {file_path!r}"
                )
        return ResolvedArchitectureGuardTargetScope(
            file_path=file_path,
            target_qualname=self.target_qualname,
        )


@dataclass(frozen=True)
class ResolvedArchitectureGuardTargetScope:
    """One unambiguous indexed source target selected by a guard scope."""

    file_path: str
    target_qualname: str | None

    def includes_target(self, file_path: str, target_qualname: str | None) -> bool:
        return self.file_path == file_path and (
            self.target_qualname is None or self.target_qualname == target_qualname
        )


@dataclass(frozen=True)
class ArchitectureGuardRule(CodemodPayloadRecord):
    """Caller-supplied invariant for a completed authority-boundary refactor."""

    rule_id: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    constraints: tuple[ArchitectureGuardConstraint, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(ArchitectureGuardConstraint),
        default=(),
    )
    scopes: tuple[ArchitectureGuardTargetScope, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(ArchitectureGuardTargetScope),
        default=(),
    )
    reason: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(),
        default="",
    )

    def resolve(
        self,
        source_index: SourceIndex,
    ) -> ArchitectureGuardRuleResolution:
        return ArchitectureGuardRuleResolution(
            rule=self,
            scopes=tuple(scope.resolve(source_index) for scope in self.scopes),
        )

    def matches(self, module: ast.Module) -> tuple[ArchitectureGuardMatch, ...]:
        return tuple(
            match
            for constraint in self.constraints
            for match in constraint.matches(module)
        )


@dataclass(frozen=True)
class ArchitectureGuardRuleResolution:
    """One guard rule with every source scope resolved exactly once."""

    rule: ArchitectureGuardRule
    scopes: tuple[ResolvedArchitectureGuardTargetScope, ...]

    def applies_to_file(self, file_path: str) -> bool:
        return not self.scopes or any(
            scope.file_path == file_path for scope in self.scopes
        )

    def includes_target(self, file_path: str, target_qualname: str | None) -> bool:
        return not self.scopes or any(
            scope.includes_target(file_path, target_qualname) for scope in self.scopes
        )

    def matches(self, module: ast.Module) -> tuple[ArchitectureGuardMatch, ...]:
        return self.rule.matches(module)


@dataclass(frozen=True)
class ArchitectureGuardViolationTarget(DataclassJsonReport):
    """Source-index target context for one architecture guard violation."""

    target_id: str | None = None
    file_path: str | None = None
    qualname: str | None = "<module>"

    @classmethod
    def from_location_target(
        cls,
        location: SourceLocation,
        target: AstTargetDigest | None,
    ) -> "ArchitectureGuardViolationTarget":
        if target is None:
            return cls(file_path=location.file_path)
        return cls(
            target_id=target.target_id,
            qualname=target.qualname,
            file_path=target.file_path,
        )


@dataclass(frozen=True)
class ArchitectureGuardViolation(DataclassJsonReport):
    """One concrete source location that violates an architecture guard rule."""

    rule_id: str
    constraint_type: type[ArchitectureGuardConstraint] = json_report_field(
        included=False
    )
    location: SourceLocation = json_report_field(included=False)
    target_context: "ArchitectureGuardViolationTarget" = json_report_field(
        flattened=True
    )
    detail: str = ""

    @json_report_property()
    def violation_kind(self) -> str:
        return self.constraint_type.discriminator_key()

    @json_report_property()
    def line(self) -> int:
        return self.location.line

    @json_report_property()
    def symbol(self) -> str:
        return self.location.symbol


@dataclass(frozen=True)
class ArchitectureGuardReport(DataclassJsonReport):
    """Result of checking caller-supplied codemod architecture invariants."""

    rules: tuple[ArchitectureGuardRule, ...]
    violations: tuple[ArchitectureGuardViolation, ...]

    @json_report_property()
    def is_clean(self) -> bool:
        return not self.violations

    @json_report_property()
    def violation_count(self) -> int:
        return len(self.violations)


@dataclass(frozen=True)
class ArchitectureGuardSuite:
    """Nominal carrier for post-refactor architecture guard rules."""

    rules: tuple[ArchitectureGuardRule, ...] = ()

    @classmethod
    def from_json_value(cls, value: JsonValue | None) -> "ArchitectureGuardSuite":
        if value is None:
            return cls()
        if not isinstance(value, (list, tuple)):
            raise ValueError("architecture_guards must be an array")
        return cls(tuple(ArchitectureGuardRule.from_json_value(row) for row in value))

    @property
    def is_empty(self) -> bool:
        return not self.rules

    def with_rule(self, rule: ArchitectureGuardRule) -> "ArchitectureGuardSuite":
        return replace(self, rules=(*self.rules, rule))

    def merge(self, *suites: "ArchitectureGuardSuite") -> "ArchitectureGuardSuite":
        return replace(
            self,
            rules=tuple(
                dict.fromkeys(
                    (
                        *self.rules,
                        *(rule for suite in suites for rule in suite.rules),
                    )
                )
            ),
        )

    def evaluate(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
    ) -> ArchitectureGuardReport:
        return evaluate_architecture_guards(source_index, source_by_path, self.rules)

    def clean_report(self) -> ArchitectureGuardReport:
        """Return the canonical clean report for this suite without source work."""

        return ArchitectureGuardReport(self.rules, ())

    def to_tuple(self) -> tuple[ArchitectureGuardRule, ...]:
        return self.rules

@dataclass(frozen=True)
class ArchitectureGuardSuitePayloadValueCodec(
    PayloadValueCodec[ArchitectureGuardSuite]
):
    """Optional architecture-guard array owned by its nominal suite."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> ArchitectureGuardSuite:
        return ArchitectureGuardSuite.from_json_value(payload.get(field_name))

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, ArchitectureGuardSuite):
            raise TypeError(
                "architecture-guard payload codec requires ArchitectureGuardSuite"
            )
        return JSON_REPORT_VALUE_PROJECTION.project(value.rules)


def evaluate_architecture_guards(
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
    rules: Iterable[ArchitectureGuardRule],
) -> ArchitectureGuardReport:
    """Evaluate caller-supplied codemod invariants over current source text."""

    rule_tuple = tuple(rules)
    resolved_rules = tuple(rule.resolve(source_index) for rule in rule_tuple)
    violations: list[ArchitectureGuardViolation] = []
    for file_path, source in source_by_path.items():
        active_rules = tuple(
            resolution
            for resolution in resolved_rules
            if resolution.applies_to_file(file_path)
        )
        if not active_rules:
            continue
        module = ast.parse(source, filename=file_path)
        for resolution in active_rules:
            for match in resolution.matches(module):
                location = SourceLocation(file_path, match.node.lineno, match.symbol)
                target = source_index.targets_by_file.smallest_enclosing_target(
                    file_path,
                    match.node.lineno,
                    match.node.lineno,
                )
                if not resolution.includes_target(
                    file_path,
                    None if target is None else target.qualname,
                ):
                    continue
                rule = resolution.rule
                violations.append(
                    ArchitectureGuardViolation(
                        rule_id=rule.rule_id,
                        constraint_type=match.constraint_type,
                        location=location,
                        target_context=(
                            ArchitectureGuardViolationTarget.from_location_target(
                                location,
                                target,
                            )
                        ),
                        detail=(
                            f"{match.message}: {rule.reason}"
                            if rule.reason
                            else match.message
                        ),
                    )
                )
    return ArchitectureGuardReport(
        rules=rule_tuple,
        violations=sorted_tuple(
            violations,
            key=lambda item: (
                item.location.file_path,
                item.location.line,
                item.rule_id,
                item.violation_kind,
                item.location.symbol,
            ),
        ),
    )
