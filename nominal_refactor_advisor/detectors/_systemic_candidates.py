"""Candidate records built and read by the systemic detectors.

Moved out of the _base god file. They ride on the detector and collector
base machinery that stays in _base, imported here by name.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Self

from nominal_refactor_advisor.ast_projection import CallKeywords
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_families.parsed_index import ModuleClassStatements
from nominal_refactor_advisor.descriptor_algebra import AliasProperty, ConstantProperty
from nominal_refactor_advisor.detectors._base import (
    CandidateCollectorBaseReference,
    CandidateCollectorBaseShape,
    CandidateCollectorScope,
    CandidateFindingDetector,
    ClassMethodLineWitnessCandidate,
    CollectsItself,
    DerivedCandidateCollectorMixin,
    DetectorDeclaration,
    DispatchAlgebraAuthority,
    EnumCaseFamilyMixin,
    FieldFamilyLineSurface,
    IssueDetector,
    ParameterizedBaseSource,
    PositionalKeywordCallSurface,
    SyntaxProjectionAuthority,
    _detector_id_value_from_class_name,
    _self_attr_name,
)
from nominal_refactor_advisor.detectors._candidates import (
    LineWitnessCandidate,
    WitnessCarrierCandidate,
)
from nominal_refactor_advisor.detectors._registry_candidates import (
    ClassLineWitnessCandidate,
)
from nominal_refactor_advisor.detectors._substrate_support import _constant_string
from nominal_refactor_advisor.models import SourceLocation
from nominal_refactor_advisor.positional_forwarding import PositionalForwardingCall
from nominal_refactor_advisor.semantic_description_length import CompressionCertificate
from nominal_refactor_advisor.semantic_match import (
    name_id,
    named_value_binding,
    return_call,
)


class ModuleCollectedLineWitnessCandidate(CollectsItself, LineWitnessCandidate, ABC):
    """A line witness collected from the top-level classes of the current source.

    A member says what one class contributes; the walk over the module's classes is here.
    """

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        return tuple(
            candidate
            for declared in module.fact(ModuleClassStatements).top_level
            for candidate in cls.from_class(module, declared.node)
        )

    @classmethod
    @abstractmethod
    def from_class(cls, module: ParsedModule, node: ast.ClassDef) -> tuple[Self, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class EnumProjectionTableCandidate(EnumCaseFamilyMixin, LineWitnessCandidate):
    table_name: str
    witness_name = AliasProperty[str]("table_name")


@dataclass(frozen=True)
class ManualStructuralRecordMechanicsClassCandidate(ClassLineWitnessCandidate):
    base_names: tuple[str, ...]
    validation_method_name: str
    projection_method_names: tuple[str, ...]
    roundtrip_method_names: tuple[str, ...]
    transform_method_names: tuple[str, ...]

    @property
    def method_names(self) -> tuple[str, ...]:
        return (
            self.validation_method_name,
            *self.projection_method_names,
            *self.roundtrip_method_names,
            *self.transform_method_names,
        )


@dataclass(frozen=True)
class CandidateCollectorBoilerplateCandidate(
    ClassMethodLineWitnessCandidate,
    ModuleCollectedLineWitnessCandidate,
):
    collector_declaration_name: ClassVar[str] = "candidate_collector"

    forwarding: PositionalForwardingCall
    collector_scope: CandidateCollectorScope
    candidate_type_source: str

    collector_descriptor_type = ConstantProperty[type[staticmethod]](staticmethod)

    @classmethod
    def from_class(
        cls,
        module: ParsedModule,
        node: ast.ClassDef,
    ) -> tuple[CandidateCollectorBoilerplateCandidate, ...]:
        detector_shape = cls.detector_shape(node)
        if detector_shape is None:
            return ()
        forwarding_base_name, candidate_type_source = detector_shape
        method = next(
            (
                statement
                for statement in node.body
                if isinstance(statement, ast.FunctionDef)
                and statement.name == CandidateFindingDetector._candidate_items.__name__
            ),
            None,
        )
        if method is None:
            return ()
        collector_calls = tuple(
            (scope, call)
            for scope in CandidateCollectorScope.for_forwarding_base_name(
                forwarding_base_name
            )
            for call in (cls.collector_call(method, scope),)
            if call is not None
        )
        if len(collector_calls) != 1:
            return ()
        collector_scope, forwarding = collector_calls[0]
        candidate = cls(
            file_path=module.file_path,
            line=method.lineno,
            class_name=node.name,
            method_name=method.name,
            forwarding=forwarding,
            collector_scope=collector_scope,
            candidate_type_source=candidate_type_source,
        )
        return () if candidate.recommended_base_name == node.name else (candidate,)

    @staticmethod
    def detector_shape(
        node: ast.ClassDef,
    ) -> tuple[str, str] | None:
        for base in node.bases:
            parameterized_base = ParameterizedBaseSource.from_node(base)
            if parameterized_base is None:
                continue
            if CandidateCollectorScope.for_forwarding_base_name(
                parameterized_base.base_name
            ):
                return parameterized_base.base_name, parameterized_base.parameter_source
        return None

    @staticmethod
    def collector_call(
        method: ast.FunctionDef, collector_scope: CandidateCollectorScope
    ) -> PositionalForwardingCall | None:
        forwarding = PositionalForwardingCall.from_function(method)
        if forwarding is None:
            return None
        return (
            forwarding
            if any(
                forwarding.parameter_names[1:] == declared.parameter_names[1:]
                and forwarding.argument_names == declared.argument_names
                for declared in collector_scope.direct_forwarding_calls
            )
            else None
        )

    @property
    def collector_name(self) -> str:
        return ast.unparse(self.forwarding.callee)

    @property
    def uses_config(self) -> bool:
        return len(self.forwarding.argument_names) == 2

    @property
    def collector_declaration_source(self) -> str:
        return (
            f"{self.collector_declaration_name} = "
            f"{self.collector_descriptor_type.__name__}({self.collector_name})"
        )

    @property
    def recommended_base_type(self) -> type[DerivedCandidateCollectorMixin]:
        return DerivedCandidateCollectorMixin.collector_base_types_by_shape()[
            CandidateCollectorBaseShape(self.collector_scope, self.uses_config)
        ]

    @property
    def recommended_base_name(self) -> str:
        return self.recommended_base_type.__name__


@dataclass(frozen=True)
class ConcreteCandidateDetectorShape:
    base_name: str

    @classmethod
    def from_class(
        cls,
        module: ParsedModule,
        node: ast.ClassDef,
    ) -> ConcreteCandidateDetectorShape | None:
        declares_detector_id = any(
            binding is not None
            and binding.name == DetectorDeclaration.detector_id_field_name
            for statement in node.body
            for binding in (named_value_binding(statement),)
        )
        if not declares_detector_id:
            return None
        base_names = tuple(
            reference.source_name
            for base in node.bases
            for reference in (
                CandidateCollectorBaseReference.from_node(
                    module, base, line=node.lineno
                ),
            )
            if reference is not None and reference.is_candidate
        )
        if len(base_names) != 1:
            return None
        return cls(base_names[0])


@dataclass(frozen=True)
class DirectBuildFindingRendererCandidate(
    PositionalKeywordCallSurface,
    ClassMethodLineWitnessCandidate,
    ModuleCollectedLineWitnessCandidate,
):
    base_name: str
    parameter_name: str

    @staticmethod
    def build_call(method: ast.FunctionDef) -> ast.Call | None:
        if (
            method.decorator_list
            or len(method.body) != 1
            or method.args.posonlyargs
            or method.args.vararg is not None
            or method.args.kwarg is not None
            or method.args.kwonlyargs
            or method.args.defaults
            or method.args.kw_defaults
            or len(method.args.args) != 2
            or method.args.args[0].arg != "self"
        ):
            return None
        call = return_call(method.body[0])
        function = call.func if call is not None else None
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == IssueDetector.build_finding.__name__
            and name_id(function.value) == "self"
        ):
            return None
        return call

    @classmethod
    def from_class(
        cls,
        module: ParsedModule,
        node: ast.ClassDef,
    ) -> tuple[DirectBuildFindingRendererCandidate, ...]:
        shape = ConcreteCandidateDetectorShape.from_class(module, node)
        if shape is None:
            return ()
        return tuple(
            cls(
                file_path=module.file_path,
                line=method.lineno,
                class_name=node.name,
                method_name=method.name,
                base_name=shape.base_name,
                parameter_name=method.args.args[1].arg,
                positional_arg_count=len(call.args),
                keyword_names=CallKeywords.of(call.keywords).names,
            )
            for method in node.body
            if isinstance(method, ast.FunctionDef)
            and method.name == CandidateFindingDetector._finding_for_candidate.__name__
            for call in (cls.build_call(method),)
            if call is not None
        )


@dataclass(frozen=True)
class DeclarativeDetectorClassCandidate(
    ClassLineWitnessCandidate,
    ModuleCollectedLineWitnessCandidate,
):
    base_name: str
    candidate_type_name: str
    assignment_names: tuple[str, ...]
    line_count: int

    @classmethod
    def assignment_values(cls, node: ast.ClassDef) -> dict[str, ast.expr] | None:
        if node.decorator_list:
            return None
        values: dict[str, ast.expr] = {}
        for statement in node.body:
            binding = named_value_binding(statement)
            if (
                binding is None
                or binding.value is None
                or binding.name in values
                or binding.name not in DetectorDeclaration.class_shell_field_names()
            ):
                return None
            values[binding.name] = binding.value
        required_names = frozenset(
            DetectorDeclaration.required_class_shell_field_names()
        )
        if not required_names <= values.keys():
            return None
        detector_id_field_name = DetectorDeclaration.detector_id_field_name
        if detector_id_field_name in values:
            detector_id = values[detector_id_field_name]
            if not (
                isinstance(detector_id, ast.Constant)
                and detector_id.value == _detector_id_value_from_class_name(node.name)
            ):
                return None
        return values

    @classmethod
    def from_class(
        cls,
        module: ParsedModule,
        node: ast.ClassDef,
    ) -> tuple[DeclarativeDetectorClassCandidate, ...]:
        if node.end_lineno is None or len(node.bases) != 1:
            return ()
        base = ParameterizedBaseSource.from_node(node.bases[0])
        assignment_values = cls.assignment_values(node)
        if (
            base is None
            or not base.parameter_source.isidentifier()
            or assignment_values is None
        ):
            return ()
        reference = CandidateCollectorBaseReference.from_node(
            module,
            node.bases[0],
            line=node.lineno,
        )
        if reference is None or not reference.is_candidate:
            return ()
        return (
            cls(
                file_path=module.file_path,
                line=node.lineno,
                class_name=node.name,
                base_name=base.base_name,
                candidate_type_name=base.parameter_source,
                assignment_names=tuple(assignment_values),
                line_count=node.end_lineno - node.lineno + 1,
            ),
        )


@dataclass(frozen=True)
class FieldFamilyCompressionSurface(FieldFamilyLineSurface):
    line_numbers: tuple[int, ...]
    compression_certificate: CompressionCertificate


@dataclass(frozen=True)
class ManualFiberTagCandidate(
    ModuleCollectedLineWitnessCandidate, WitnessCarrierCandidate
):
    _TYPE_NAME_LITERAL = "type"
    _EVAL_PARSE_MODE = "eval"
    _TAG_PARAM_NAMES = frozenset({"kind", "mode", _TYPE_NAME_LITERAL, "tag", "backend"})
    init_line: int
    method_name: str
    tag_name: str
    assigned_field_names: tuple[str, ...]
    method_line = AliasProperty[int]("line")
    case_names = AliasProperty[tuple[str, ...]]("name_family")

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(
                self.file_path, self.init_line, f"{self.class_name}.__init__"
            ),
            SourceLocation(
                self.file_path,
                self.method_line,
                f"{self.class_name}.{self.method_name}",
            ),
        )

    @classmethod
    def from_class(cls, module: ParsedModule, node: ast.ClassDef) -> tuple[Self, ...]:
        """Each method of the class that dispatches on a tag its `__init__` copies
        from a parameter, for each such tag."""

        candidates: list[Self] = []
        methods = {
            item.name: item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        init_method = methods.get("__init__")
        if init_method is None:
            return ()
        assigned_from_param = SyntaxProjectionAuthority.assigned_self_attr_from_param(
            init_method
        )
        tag_names = tuple(
            (
                attr_name
                for attr_name, param_name in assigned_from_param.items()
                if param_name in cls._TAG_PARAM_NAMES
            )
        )
        if not tag_names:
            return ()
        assigned_field_names = cls._assigned_self_attrs(init_method)
        for method_name, method in methods.items():
            if method_name == "__init__":
                continue
            if not method.body:
                continue
            for tag_name in tag_names:
                case_names = cls._string_dispatch_cases_from_body(
                    method.body, f"self.{tag_name}"
                )
                if len(case_names) < 2:
                    continue
                if len(assigned_field_names) <= len(case_names) + 1:
                    continue
                candidates.append(
                    cls(
                        file_path=module.file_path,
                        line=method.lineno,
                        subject_name=node.name,
                        name_family=case_names,
                        init_line=init_method.lineno,
                        method_name=method_name,
                        tag_name=tag_name,
                        assigned_field_names=assigned_field_names,
                    )
                )
        return tuple(candidates)

    @staticmethod
    def _assigned_self_attrs(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[str, ...]:
        assigned: list[str] = []
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Assign):
                for target in subnode.targets:
                    attr_name = _self_attr_name(target)
                    if attr_name is not None:
                        assigned.append(attr_name)
            elif isinstance(subnode, ast.AnnAssign):
                attr_name = _self_attr_name(subnode.target)
                if attr_name is not None:
                    assigned.append(attr_name)
        return tuple(dict.fromkeys(assigned))

    @staticmethod
    def _string_dispatch_cases_from_body(
        body: list[ast.stmt],
        dispatch_axis_expression: str,
    ) -> tuple[str, ...]:
        cases: list[str] = []
        if not body:
            return ()
        current = body[0]
        while isinstance(current, ast.If):
            dispatch_case = DispatchAlgebraAuthority.comparison_dispatch_case(
                current.test
            )
            if dispatch_case is None:
                return ()
            current_axis, case_name = dispatch_case
            if current_axis != dispatch_axis_expression:
                return ()
            if (
                _constant_string(
                    ast.parse(
                        case_name, mode=ManualFiberTagCandidate._EVAL_PARSE_MODE
                    ).body
                )
                is None
            ):
                return ()
            cases.append(case_name)
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                continue
            break
        return tuple(cases)
