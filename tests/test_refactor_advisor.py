from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import cast

from nominal_refactor_advisor.ast_tools import (
    AccessorWrapperObservationFamily,
    AttributeProbeObservationFamily,
    BuilderCallShapeFamily,
    ClassMarkerObservationFamily,
    ConfigDispatchObservationFamily,
    DualAxisResolutionObservationFamily,
    DynamicMethodInjectionObservationFamily,
    ExportDictShapeFamily,
    FieldObservationSpec,
    FieldObservationFamily,
    InlineStringLiteralDispatchObservationFamily,
    InterfaceGenerationObservationFamily,
    LineageMappingObservationFamily,
    MethodShapeFamily,
    ProjectionHelperObservationFamily,
    RegistrationShapeSpec,
    RegistrationShapeFamily,
    RuntimeTypeGenerationObservationFamily,
    ScopedShapeWrapperFunctionFamily,
    ScopedShapeWrapperSpecFamily,
    SentinelTypeObservationFamily,
    StringLiteralDispatchObservationFamily,
    NumericLiteralDispatchObservationFamily,
    TypedLiteralObservationSpec,
    collect_family_items,
    collect_scoped_observations,
    parse_python_modules,
)
from nominal_refactor_advisor.cli import _CLI_ARGUMENT_SPECS
from nominal_refactor_advisor.cli import _format_markdown
from nominal_refactor_advisor.cli import _json_payload
from nominal_refactor_advisor.cli import analyze_path
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.models import DispatchCountMetrics
from nominal_refactor_advisor.observation_graph import (
    ObservationGraph,
    ObservationKind,
    StructuralExecutionLevel,
    build_observation_graph,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.planner import build_refactor_plans


def _write_module(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_detects_repeated_private_method_shape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def _build(self, item):
        prepared = self.normalize(item)
        checked = self.validate(prepared)
        return self.finish(checked)


class Beta:
    def _assemble(self, value):
        prepared = self.normalize(value)
        checked = self.validate(prepared)
        return self.finish(checked)
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 5 for finding in findings)
    assert any(finding.pattern_id == 5 and finding.scaffold for finding in findings)
    assert any(
        finding.pattern_id == 5 and finding.codemod_patch for finding in findings
    )


def test_detects_sibling_role_helper_symmetry(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from pathlib import Path


class PathPlanner:
    def _input_dir_for_step(self, snapshot, step_index):
        if step_index in self.plans and self.plans[step_index].input_dir is not None:
            return Path(self.plans[step_index].input_dir)
        if step_index == 0 or snapshot.input_source == "pipeline_start":
            return self.initial_input
        return Path(self.plans[step_index - 1].output_dir)

    def _output_dir_for_step(self, snapshot, step_index, work_in_place_dir):
        if step_index in self.plans and self.plans[step_index].output_dir is not None:
            return Path(self.plans[step_index].output_dir)
        if step_index == 0 or snapshot.input_source == "pipeline_start":
            return self._build_output_path()
        return work_in_place_dir
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "sibling_role_helper_symmetry"
    )

    assert finding.pattern_id == PatternId.LOCAL_VALUE_AUTHORITY
    assert "_input_dir_for_step" in finding.summary
    assert "_output_dir_for_step" in finding.summary
    assert "one local authority" in finding.title
    assert "record only if this result crosses a boundary" in (finding.scaffold or "")


def test_detects_oversized_orchestration_hub(tmp_path: Path) -> None:
    branch_body = "\n".join(
        f"""
    if branch_{index}(request):
        value = phase_{index}(value)
    else:
        value = fallback_{index}(value)
    audit_{index}(value)
""".rstrip()
        for index in range(12)
    )
    _write_module(
        tmp_path,
        "pkg/mod.py",
        (
            "def orchestrate(request):\n"
            "    value = start(request)\n"
            f"{branch_body}\n"
            "    finalized = finalize(value)\n"
            "    publish(finalized)\n"
            "    return finalized\n"
        ),
    )

    findings = analyze_path(
        tmp_path,
        DetectorConfig(
            min_orchestration_function_lines=40,
            min_orchestration_branches=10,
            min_orchestration_calls=24,
        ),
    )

    assert any(
        finding.pattern_id == PatternId.STAGED_ORCHESTRATION for finding in findings
    )


def test_detects_private_cohort_should_be_module(tmp_path: Path) -> None:
    filler = "# filler\n" * 240
    repeated_lines = "\n".join(
        f"    detail_{index} = selection['winner']" for index in range(60)
    )
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        (
            f"{filler}\n"
            "class _ReturnedPoseSelection:\n"
            "    def __init__(self, winner, support):\n"
            "        self.winner = winner\n"
            "        self.support = support\n\n"
            "class _ReturnedPoseProofContext:\n"
            "    def __init__(self, scores):\n"
            "        self.scores = scores\n\n"
            "def _returned_pose_support_indices(context):\n"
            "    support = []\n"
            "    for index, _score in enumerate(context.scores):\n"
            "        if index < 2:\n"
            "            support.append(index)\n"
            "    return tuple(support)\n\n"
            "def _returned_pose_selection(context):\n"
            "    support = _returned_pose_support_indices(context)\n"
            "    winner = support[0] if support else 0\n"
            "    return _ReturnedPoseSelection(winner, support)\n\n"
            "def _returned_pose_proof_plan(context):\n"
            "    selection = _returned_pose_selection(context)\n"
            f"{repeated_lines}\n"
            "    return {'winner': selection.winner, 'support': selection.support}\n\n"
            "def _returned_pose_certification(context):\n"
            "    plan = _returned_pose_proof_plan(context)\n"
            "    return plan['winner'], plan['support']\n\n"
            "def run_pipeline(scores):\n"
            "    context = _ReturnedPoseProofContext(scores)\n"
            "    return _returned_pose_certification(context)\n"
        ),
    )

    findings = analyze_path(
        tmp_path,
        DetectorConfig(
            min_orchestration_function_lines=20,
            min_registration_sites=2,
        ),
    )
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "private_cohort_should_be_module"
    )

    assert "returned, pose" in finding.summary
    assert "_returned_pose_proof_plan" in finding.summary
    assert "pipeline_returned_pose" in (finding.codemod_patch or "")


def test_ignores_private_helpers_without_cohesive_cohort(tmp_path: Path) -> None:
    filler = "# filler\n" * 240
    _write_module(
        tmp_path,
        "pkg/helpers.py",
        (
            f"{filler}\n"
            "def _build_payload(value):\n"
            "    return {'value': value}\n\n"
            "def _load_registry(name):\n"
            "    return {'name': name}\n\n"
            "def _write_audit(event):\n"
            "    return event\n\n"
            "def run_helpers(value):\n"
            "    return _write_audit(_build_payload(value))\n"
        ),
    )

    findings = analyze_path(
        tmp_path,
        DetectorConfig(
            min_orchestration_function_lines=20,
            min_registration_sites=2,
        ),
    )

    assert not any(
        finding.detector_id == "private_cohort_should_be_module"
        for finding in findings
    )


def test_private_cohort_ignores_generic_analyzer_vocabulary(tmp_path: Path) -> None:
    filler = "# filler\n" * 240
    helper_blocks = "\n\n".join(
        (
            f"def _parallel_keyed_family_candidate_{name}(value):\n"
            + "\n".join(f"    step_{index} = value" for index in range(30))
            + "\n    return value\n"
        )
        for name in (
            "alpha",
            "bravo",
            "charlie",
            "delta",
            "echo",
            "foxtrot",
        )
    )
    _write_module(
        tmp_path,
        "pkg/analyzer_helpers.py",
        f"{filler}\n{helper_blocks}\n",
    )

    findings = analyze_path(
        tmp_path,
        DetectorConfig(
            min_orchestration_function_lines=20,
            min_registration_sites=2,
        ),
    )

    assert not any(
        finding.detector_id == "private_cohort_should_be_module"
        for finding in findings
    )


def test_detects_repeated_threaded_parameter_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def score_exact(
    request,
    scoring_context,
    electrostatics,
    receptor_coords,
    receptor_radii,
    quaternion,
    translation,
    candidate_coords,
):
    posed = rigid(candidate_coords, quaternion, translation)
    audited = audit_pose(posed, receptor_coords)
    return compute_exact(
        request,
        scoring_context,
        electrostatics,
        receptor_coords,
        receptor_radii,
        audited,
    )


def score_softened(
    request,
    scoring_context,
    electrostatics,
    receptor_coords,
    receptor_radii,
    quaternion,
    translation,
    candidate_coords,
):
    posed = rigid(candidate_coords, quaternion, translation)
    audited = audit_pose(posed, receptor_coords)
    return compute_softened(
        request,
        scoring_context,
        electrostatics,
        receptor_coords,
        receptor_radii,
        audited,
    )


def certify_pose(
    request,
    scoring_context,
    electrostatics,
    receptor_coords,
    receptor_radii,
    quaternion,
    translation,
    pose_index,
):
    posed = derive_pose(pose_index, quaternion, translation)
    audited = audit_pose(posed, receptor_coords)
    return certify(
        request,
        scoring_context,
        electrostatics,
        receptor_coords,
        receptor_radii,
        audited,
    )
""",
    )

    findings = analyze_path(
        tmp_path,
        DetectorConfig(
            min_shared_parameters=5,
            min_parameter_family_function_lines=8,
        ),
    )

    assert any(
        finding.pattern_id == PatternId.AUTHORITATIVE_CONTEXT for finding in findings
    )


def test_detects_suffix_axis_compatibility_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Compiler:
    @staticmethod
    def declare_for_context(context, steps, runner):
        names = [step.name for step in steps]
        return declare(context, steps, runner, names)

    @staticmethod
    def declare_for_session(session):
        return declare(session.context, session.steps, session.runner, session.names)

    @staticmethod
    def validate_for_context(context, steps, runner):
        names = [step.name for step in steps]
        return validate(context, steps, runner, names)

    @staticmethod
    def validate_for_session(session):
        return validate(session.context, session.steps, session.runner, session.names)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "suffix_axis_compatibility_surface"
    )

    assert finding.pattern_id == PatternId.AUTHORITATIVE_CONTEXT
    assert "context / session" in finding.summary
    assert "declare" in finding.summary
    assert "validate" in finding.summary
    assert "OperationContext" in (finding.scaffold or "")


def test_detects_enum_strategy_dispatch_with_abc_guidance(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from enum import Enum


class Mode(Enum):
    OBSERVED = "observed"
    CERTIFIED = "certified"


def run_mode(mode, inputs, steps):
    if mode == Mode.OBSERVED:
        return run_observed(inputs, steps)
    elif mode == Mode.CERTIFIED:
        return run_certified(inputs, steps)
    else:
        raise ValueError(mode)
""",
    )

    findings = analyze_path(tmp_path)

    strategy_finding = next(
        finding
        for finding in findings
        if finding.pattern_id == PatternId.NOMINAL_STRATEGY_FAMILY
    )
    assert "Mode.OBSERVED" in strategy_finding.summary
    assert strategy_finding.scaffold is not None
    assert "from metaclass_registry import AutoRegisterMeta" in strategy_finding.scaffold
    assert "class ModeRunner(ABC, metaclass=AutoRegisterMeta):" in strategy_finding.scaffold
    assert strategy_finding.codemod_patch is not None
    assert "runner = ModeRunner.for_mode(mode)" in strategy_finding.codemod_patch


def test_detects_residual_closed_axis_indirection(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from enum import Enum
from types import MappingProxyType


class Direction(Enum):
    INPUT = "input"
    OUTPUT = "output"


DIRECTION_READERS = MappingProxyType(
    {
        Direction.INPUT: lambda plan: plan.input_dir,
        Direction.OUTPUT: lambda plan: plan.output_dir,
    }
)


def resolve_dir(plan, direction, fallback):
    existing = DIRECTION_READERS[direction](plan)
    if existing is not None:
        return existing
    if direction is Direction.INPUT:
        return plan.initial_input
    return fallback
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "residual_closed_axis_indirection"
    )

    assert finding.pattern_id == PatternId.NOMINAL_STRATEGY_FAMILY
    assert "DIRECTION_READERS" in finding.summary
    assert "Direction" in finding.summary
    assert "INPUT" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "class AxisPolicy(ABC, metaclass=AutoRegisterMeta)" in (
        finding.scaffold or ""
    )


def test_detects_repeated_concrete_type_case_analysis(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class MissingState:
    note: str


@dataclass(frozen=True)
class ReadyState:
    value: int


@dataclass(frozen=True)
class FailedState:
    error: str


State = MissingState | ReadyState | FailedState


@dataclass(frozen=True)
class Record:
    state: State


def state_status(record):
    state = record.state
    if isinstance(state, ReadyState):
        return "ready"
    if isinstance(state, FailedState):
        return "failed"
    return "missing"


def state_value(record):
    state = record.state
    if isinstance(state, ReadyState):
        return state.value
    if isinstance(state, FailedState):
        return None
    return None


def state_message(record):
    state = record.state
    if isinstance(state, MissingState):
        return state.note
    if isinstance(state, FailedState):
        return state.error
    return "ok"
""",
    )

    findings = analyze_path(tmp_path)

    case_finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_concrete_type_case_analysis"
    )
    assert case_finding.pattern_id == PatternId.NOMINAL_INTERFACE_WITNESS
    assert "state" in case_finding.summary
    assert "ReadyState" in case_finding.summary
    assert "State" in case_finding.summary
    assert case_finding.scaffold is not None
    assert "class StateFamily(ABC)" in case_finding.scaffold


def test_detects_repeated_enum_strategy_dispatch_across_owners(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from enum import Enum


class SamplingStrategy(Enum):
    RANDOM = "random"
    GUIDED = "guided"
    HYBRID = "hybrid"


def run_sampling(strategy, sampler, request, guided_fn):
    if strategy == SamplingStrategy.GUIDED:
        return guided_fn(request)
    if strategy == SamplingStrategy.HYBRID:
        guided, random = sampler.hybrid(request, guided_fn)
        return guided + random
    return sampler.random(request)


class Sampler:
    def sample(self, strategy, request, guided_fn):
        match strategy:
            case SamplingStrategy.RANDOM:
                return self.random(request)
            case SamplingStrategy.GUIDED:
                return guided_fn(request)
            case SamplingStrategy.HYBRID:
                guided, random = self.hybrid(request, guided_fn)
                return guided + random
            case _:
                raise ValueError(strategy)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_enum_strategy_dispatch"
    )

    assert "SamplingStrategy" in finding.summary
    assert "run_sampling" in finding.summary
    assert "Sampler.sample" in finding.summary


def test_detects_split_dispatch_authority(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC, abstractmethod
from functools import singledispatch


class ModeRunner(ABC):
    @abstractmethod
    def run(self, *, random_fn, source_fn):
        raise NotImplementedError

    @classmethod
    def for_mode(cls, mode):
        return _MODE_RUNNERS[mode]


class RandomRunner(ModeRunner):
    def run(self, *, random_fn, source_fn):
        return random_fn()


class GuidedRunner(ModeRunner):
    def run(self, *, random_fn, source_fn):
        return source_fn()


_MODE_RUNNERS = {
    Mode.RANDOM: RandomRunner(),
    Mode.GUIDED: GuidedRunner(),
}


@singledispatch
def source_for_item(item):
    raise TypeError(type(item).__name__)


@source_for_item.register
def _(item: FileItem):
    return item.path


@source_for_item.register
def _(item: MemoryItem):
    return item.payload


def orchestrate(request):
    runner = ModeRunner.for_mode(request.mode)

    def _source():
        return source_for_item(request.item)

    return runner.run(
        random_fn=lambda: request.default_source,
        source_fn=_source,
    )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "split_dispatch_authority"
    )

    assert "ModeRunner.for_mode(request.mode)" in finding.summary
    assert "source_for_item(request.item)" in finding.summary
    assert "ProductPolicy" in (finding.scaffold or "")


def test_detects_closed_constant_selector(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from enum import Enum


class Mode(Enum):
    DIRECT = "direct"
    FALLBACK = "fallback"


class Plan:
    def __init__(self, *, mode_name):
        self.mode_name = mode_name


class Runner:
    def __init__(self, plan):
        self.plan = plan


PRIMARY_PLAN = Plan(mode_name="primary")
FALLBACK_PLAN = Plan(mode_name="fallback")
SAFE_PLAN = Plan(mode_name="safe")

DIRECT_CONTRACT = "direct"
FALLBACK_CONTRACT = "fallback"


def build_runner(mode: Mode, *, enabled: bool):
    if mode == Mode.DIRECT and enabled:
        return Runner(PRIMARY_PLAN)
    if enabled:
        return Runner(FALLBACK_PLAN)
    return Runner(SAFE_PLAN)


def active_contract(mode: Mode):
    if mode == Mode.DIRECT:
        return DIRECT_CONTRACT
    return FALLBACK_CONTRACT
""",
    )

    findings = analyze_path(tmp_path)
    selector_findings = [
        finding
        for finding in findings
        if finding.detector_id == "closed_constant_selector"
    ]

    assert len(selector_findings) == 2
    assert any("build_runner" in finding.summary for finding in selector_findings)
    assert any("Runner(...)" in finding.summary for finding in selector_findings)
    assert any("active_contract" in finding.summary for finding in selector_findings)
    assert any("SelectorRule" in (finding.scaffold or "") for finding in selector_findings)


def test_detects_derived_wrapper_spec_shadow(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass, field


class AlphaRequest:
    pass


class BetaRequest:
    pass


def run_alpha(request):
    return request


def run_beta(request):
    return request


@dataclass(frozen=True)
class ExecutionSpec:
    request_type: type
    runner: object


ALPHA_EXECUTION_SPEC = ExecutionSpec(request_type=AlphaRequest, runner=run_alpha)
BETA_EXECUTION_SPEC = ExecutionSpec(request_type=BetaRequest, runner=run_beta)
EXECUTION_SPECS = (ALPHA_EXECUTION_SPEC, BETA_EXECUTION_SPEC)


@dataclass(frozen=True)
class WrapperRule:
    name: str
    execution: ExecutionSpec
    defaults: dict[str, object] = field(default_factory=dict)


def build_wrapper(rule: WrapperRule):
    def wrapper():
        return rule.execution.runner(rule.execution.request_type())
    wrapper.__name__ = rule.name
    return wrapper


WRAPPER_RULES = (
    WrapperRule(name="run_alpha", execution=ALPHA_EXECUTION_SPEC),
    WrapperRule(name="run_beta", execution=BETA_EXECUTION_SPEC, defaults={"key": None}),
)

globals().update({rule.name: build_wrapper(rule) for rule in WRAPPER_RULES})
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "derived_wrapper_spec_shadow"
    )

    assert "WRAPPER_RULES" in finding.summary
    assert "EXECUTION_SPECS" in finding.summary
    assert "execution" in finding.summary
    assert "build_wrapper" in finding.summary
    assert "wrapper_name" in (finding.scaffold or "")


def test_detects_module_keyed_selection_helper(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Sequence, TypeVar


KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


class Mode(Enum):
    ALPHA = "alpha"
    BETA = "beta"


@dataclass(frozen=True)
class SelectionRule(Generic[KeyT, ValueT]):
    key: KeyT
    selected: ValueT


def build_index(rules: Sequence[SelectionRule[KeyT, ValueT]]) -> dict[KeyT, ValueT]:
    return {rule.key: rule.selected for rule in rules}


def choose(index: dict[KeyT, ValueT], key: KeyT, *, family_name: str) -> ValueT:
    try:
        return index[key]
    except KeyError as error:
        raise ValueError(f"No {family_name} registered for {key!r}.") from error


VALUE_RULES = (
    SelectionRule(key=Mode.ALPHA, selected="a"),
    SelectionRule(key=Mode.BETA, selected="b"),
)

HANDLER_RULES = (
    SelectionRule(key=Mode.ALPHA, selected=int),
    SelectionRule(key=Mode.BETA, selected=str),
)

VALUE_BY_MODE = build_index(VALUE_RULES)
HANDLER_BY_MODE = build_index(HANDLER_RULES)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "module_keyed_selection_helper"
    )

    assert "SelectionRule" in finding.summary
    assert "build_index" in finding.summary
    assert "choose" in finding.summary
    assert "VALUE_RULES" in finding.summary
    assert "HANDLER_RULES" in finding.summary
    assert "KeyedRecordTable" in (finding.scaffold or "")


def test_detects_cross_module_axis_shadow_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/core.py",
        """
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import ClassVar, Generic, TypeVar


KeyT = TypeVar("KeyT")


class AutoRegisterByClassVar:
    registry_key_attr: ClassVar[str]
    _registry: ClassVar[dict[object, object]]

    def __init_subclass__(cls, **kwargs):
        if "registry_key_attr" in cls.__dict__ and "_registry" not in cls.__dict__:
            cls._registry = {}
        super().__init_subclass__(**kwargs)
        key_attr = getattr(cls, "registry_key_attr", None)
        if key_attr is None:
            return
        registry = getattr(cls, "_registry", None)
        if not isinstance(registry, dict):
            return
        key = cls.__dict__.get(key_attr)
        if key is not None:
            registry[key] = cls()


class KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):
    @classmethod
    def for_key(cls, key: KeyT):
        return cls._registry[key]


class Mode(Enum):
    ALPHA = auto()
    BETA = auto()


class ModePolicy(KeyedNominalFamily[Mode], ABC):
    registry_key_attr = "mode"
    _registry = {}
    mode: ClassVar[Mode]

    @abstractmethod
    def ratio(self) -> float:
        raise NotImplementedError


class AlphaModePolicy(ModePolicy):
    mode = Mode.ALPHA

    def ratio(self) -> float:
        return 0.0


class BetaModePolicy(ModePolicy):
    mode = Mode.BETA

    def ratio(self) -> float:
        return 1.0
""",
    )
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        """
from abc import ABC, abstractmethod
from pkg.core import Mode


class ModeRunner(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError

    @classmethod
    def for_mode(cls, mode: Mode):
        return _MODE_RUNNERS[mode]


class AlphaModeRunner(ModeRunner):
    def run(self):
        return "alpha"


class BetaModeRunner(ModeRunner):
    def run(self):
        return "beta"


_MODE_RUNNERS = {
    Mode.ALPHA: AlphaModeRunner(),
    Mode.BETA: BetaModeRunner(),
}
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "cross_module_axis_shadow_family"
    )

    assert "Mode" in finding.summary
    assert "ModePolicy" in finding.summary
    assert "ModeRunner.for_mode" in finding.summary
    assert "AxisPolicy" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "return cls.__registry__[key]()" in (finding.scaffold or "")


def test_detects_parallel_keyed_axis_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/specs.py",
        """
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import ClassVar, Generic, TypeVar


KeyT = TypeVar("KeyT")


class AutoRegisterByClassVar:
    registry_key_attr: ClassVar[str]
    _registry: ClassVar[dict[object, object]]

    def __init_subclass__(cls, **kwargs):
        if "registry_key_attr" in cls.__dict__ and "_registry" not in cls.__dict__:
            cls._registry = {}
        super().__init_subclass__(**kwargs)
        key_attr = getattr(cls, "registry_key_attr", None)
        if key_attr is None:
            return
        registry = getattr(cls, "_registry", None)
        if not isinstance(registry, dict):
            return
        key = cls.__dict__.get(key_attr)
        if key is not None:
            registry[key] = cls()


class KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):
    @classmethod
    def for_key(cls, key: KeyT):
        return cls._registry[key]


class Mode(Enum):
    ALPHA = auto()
    BETA = auto()
    GAMMA = auto()


class ModeSpecPolicy(KeyedNominalFamily[Mode], ABC):
    registry_key_attr = "mode"
    family_label = "mode case"
    _registry = {}
    mode: ClassVar[Mode]

    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError


class AlphaModeSpec(ModeSpecPolicy):
    mode = Mode.ALPHA

    def describe(self) -> str:
        return "alpha"


class BetaModeSpec(ModeSpecPolicy):
    mode = Mode.BETA

    def describe(self) -> str:
        return "beta"


class GammaModeSpec(ModeSpecPolicy):
    mode = Mode.GAMMA

    def describe(self) -> str:
        return "gamma"
""",
    )
    _write_module(
        tmp_path,
        "pkg/runtime.py",
        """
from abc import ABC, abstractmethod
from typing import ClassVar

from pkg.specs import KeyedNominalFamily, Mode


class ModeAssemblyPolicy(KeyedNominalFamily[Mode], ABC):
    registry_key_attr = "mode"
    family_label = "mode case"
    _registry = {}
    mode: ClassVar[Mode]

    @abstractmethod
    def build(self) -> str:
        raise NotImplementedError


class AlphaModeAssembly(ModeAssemblyPolicy):
    mode = Mode.ALPHA

    def build(self) -> str:
        return "build-alpha"


class BetaModeAssembly(ModeAssemblyPolicy):
    mode = Mode.BETA

    def build(self) -> str:
        return "build-beta"


class GammaModeAssembly(ModeAssemblyPolicy):
    mode = Mode.GAMMA

    def build(self) -> str:
        return "build-gamma"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "parallel_keyed_axis_family"
    )

    assert "Mode" in finding.summary
    assert "ModeSpecPolicy" in finding.summary
    assert "ModeAssemblyPolicy" in finding.summary
    assert "AxisPolicy" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "return cls.__registry__[key]()" in (finding.scaffold or "")


def test_detects_parallel_keyed_table_and_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, Generic, TypeVar


KeyT = TypeVar("KeyT")


class AutoRegisterByClassVar:
    registry_key_attr: ClassVar[str]
    _registry: ClassVar[dict[object, object]]

    def __init_subclass__(cls, **kwargs):
        if "registry_key_attr" in cls.__dict__ and "_registry" not in cls.__dict__:
            cls._registry = {}
        super().__init_subclass__(**kwargs)
        key_attr = getattr(cls, "registry_key_attr", None)
        if key_attr is None:
            return
        registry = getattr(cls, "_registry", None)
        if not isinstance(registry, dict):
            return
        key = cls.__dict__.get(key_attr)
        if key is not None:
            registry[key] = cls()


class KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):
    @classmethod
    def for_key(cls, key: KeyT):
        return cls._registry[key]


class Mode(Enum):
    ALPHA = auto()
    BETA = auto()
    GAMMA = auto()


@dataclass(frozen=True)
class ModeConfig:
    mode: Mode
    weight: float


MODE_CONFIGS = {
    Mode.ALPHA: ModeConfig(mode=Mode.ALPHA, weight=0.0),
    Mode.BETA: ModeConfig(mode=Mode.BETA, weight=0.5),
    Mode.GAMMA: ModeConfig(mode=Mode.GAMMA, weight=1.0),
}


class ModeRunner(KeyedNominalFamily[Mode], ABC):
    registry_key_attr = "mode"
    mode: ClassVar[Mode]

    @abstractmethod
    def run(self):
        raise NotImplementedError


class AlphaModeRunner(ModeRunner):
    mode = Mode.ALPHA

    def run(self):
        return "alpha"


class BetaModeRunner(ModeRunner):
    mode = Mode.BETA

    def run(self):
        return "beta"


class GammaModeRunner(ModeRunner):
    mode = Mode.GAMMA

    def run(self):
        return "gamma"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "parallel_keyed_table_and_family"
    )

    assert "Mode" in finding.summary
    assert "MODE_CONFIGS" in finding.summary
    assert "ModeRunner" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "build_axis_rows" in (finding.scaffold or "")


def test_detects_parallel_keyed_table_axis(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/specs.py",
        """
from dataclasses import dataclass
from enum import Enum, auto


class Mode(Enum):
    ALPHA = auto()
    BETA = auto()
    GAMMA = auto()


@dataclass(frozen=True)
class ModeSpec:
    mode: Mode
    label: str


MODE_SPECS = {
    Mode.ALPHA: ModeSpec(Mode.ALPHA, "alpha"),
    Mode.BETA: ModeSpec(Mode.BETA, "beta"),
    Mode.GAMMA: ModeSpec(Mode.GAMMA, "gamma"),
}
""",
    )
    _write_module(
        tmp_path,
        "pkg/plans.py",
        """
from dataclasses import dataclass

from pkg.specs import Mode


@dataclass(frozen=True)
class ModePlan:
    mode: Mode
    priority: int


MODE_PLANNING_SPECS = {
    Mode.ALPHA: ModePlan(Mode.ALPHA, 1),
    Mode.BETA: ModePlan(Mode.BETA, 2),
    Mode.GAMMA: ModePlan(Mode.GAMMA, 3),
}
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "parallel_keyed_table_axis"
    )

    assert "Mode" in finding.summary
    assert "MODE_SPECS" in finding.summary
    assert "MODE_PLANNING_SPECS" in finding.summary
    assert "AxisRow" in (finding.scaffold or "")


def test_detects_derived_query_index_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
ITEMS = ()


def _registered_items():
    return ITEMS


def item_for_type(item_type):
    for item in _registered_items():
        if item.item_type is item_type:
            return item
    raise KeyError(item_type)


def item_for_kind(kind):
    for item in _registered_items():
        if item.kind is kind:
            return item
    raise KeyError(kind)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "derived_query_index_surface"
    )

    assert "item_for_type" in finding.summary
    assert "item_for_kind" in finding.summary
    assert "_registered_items()" in finding.summary
    assert "ITEM_BY_KEY" in (finding.scaffold or "")


def test_detects_runtime_adapter_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass
from enum import Enum, auto


class StrategyId(Enum):
    ALPHA = auto()


class ActionId(Enum):
    DEFAULT = auto()


class AlphaStrategy:
    pass


class DefaultAction:
    pass


@dataclass(frozen=True)
class BaseSpec:
    priority: int
    dependencies: tuple[str, ...] = ()
    strategy_id: StrategyId | None = None
    action_id: ActionId | None = None


@dataclass(frozen=True)
class RuntimeSpec:
    priority: int = 0
    dependencies: tuple[str, ...] = ()
    strategy: object | None = None
    action: object | None = None


STRATEGY_BY_ID = {StrategyId.ALPHA: AlphaStrategy()}
ACTION_BY_ID = {ActionId.DEFAULT: DefaultAction()}


def runtime_spec_for(spec: BaseSpec | None) -> RuntimeSpec:
    if spec is None:
        return RuntimeSpec()
    return RuntimeSpec(
        priority=spec.priority,
        dependencies=spec.dependencies,
        strategy=STRATEGY_BY_ID.get(spec.strategy_id)
        if spec.strategy_id is not None
        else None,
        action=ACTION_BY_ID.get(spec.action_id) if spec.action_id is not None else None,
    )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "runtime_adapter_shell"
    )

    assert "runtime_spec_for" in finding.summary
    assert "RuntimeSpec" in finding.summary
    assert "STRATEGY_BY_ID" in finding.summary
    assert "ACTION_BY_ID" in finding.summary
    assert "resolve_strategy" in (finding.scaffold or "")


def test_detects_keyword_bag_adapter_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class OptionSpec:
    help: str
    action: str | None = None
    default: object | None = None
    dest: str | None = None


def option_kwargs(spec: OptionSpec) -> dict[str, object]:
    kwargs = {"help": spec.help}
    if spec.action is not None:
        kwargs["action"] = spec.action
    if spec.default is not None:
        kwargs["default"] = spec.default
    if spec.dest is not None:
        kwargs["dest"] = spec.dest
    return kwargs
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "keyword_bag_adapter_shell"
    )

    assert "option_kwargs" in finding.summary
    assert "help" in finding.summary
    assert "action" in finding.summary
    assert "as_kwargs" in (finding.scaffold or "")


def test_detects_enum_keyed_table_class_axis_shadow(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from enum import Enum
from typing import ClassVar


class RouteKind(Enum):
    DIRECT = "direct"
    MULTI_STAGE = "multi_stage"


class NominalRequest:
    route_kind: ClassVar[RouteKind | None] = None


class DirectRequest(NominalRequest):
    route_kind: ClassVar[RouteKind] = RouteKind.DIRECT


class MultiStageRequest(NominalRequest):
    route_kind: ClassVar[RouteKind] = RouteKind.MULTI_STAGE


class DirectRoute:
    pass


class MultiStageRoute:
    pass


ROUTE_REGISTRY = {
    RouteKind.DIRECT: DirectRoute,
    RouteKind.MULTI_STAGE: MultiStageRoute,
}
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "enum_keyed_table_class_axis_shadow"
    )

    assert finding.pattern_id == PatternId.AUTHORITATIVE_SCHEMA
    assert "ROUTE_REGISTRY" in finding.summary
    assert "RouteKind" in finding.summary
    assert "route_kind" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "AXIS_BY_KEY" in (finding.scaffold or "")


def test_detects_manual_structural_record_mechanics(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


class StructuralRecordTransportMixin:
    def encode(self):
        return (self.payload_fields(), self.metadata_fields())


@dataclass(frozen=True)
class AlphaSpec(StructuralRecordTransportMixin):
    left: object
    right: object
    cutoff: float

    def validate(self):
        if self.left.ndim != 1:
            raise ValueError
        if self.right.ndim != 1:
            raise ValueError
        if self.cutoff <= 0:
            raise ValueError

    def payload_fields(self):
        return (self.left, self.right)

    def metadata_fields(self):
        return (self.cutoff,)

    @classmethod
    def from_payload(cls, metadata, payload):
        return cls(*payload, *metadata)

    def subsetted(self, indices):
        return AlphaSpec(
            left=self.left[indices],
            right=self.right,
            cutoff=self.cutoff,
        )


@dataclass(frozen=True)
class BetaSpec(StructuralRecordTransportMixin):
    left: object
    right: object
    beta: float
    cutoff: float

    def validate(self):
        if self.left.ndim != 1:
            raise ValueError
        if self.right.ndim != 1:
            raise ValueError
        if self.beta <= 0:
            raise ValueError
        if self.cutoff <= 0:
            raise ValueError

    def payload_fields(self):
        return (self.left, self.right)

    def metadata_fields(self):
        return (self.beta, self.cutoff)

    @classmethod
    def from_payload(cls, metadata, payload):
        return cls(*payload, *metadata)

    def subsetted(self, indices):
        return BetaSpec(
            left=self.left[indices],
            right=self.right,
            beta=self.beta,
            cutoff=self.cutoff,
        )

    def zeroed(self):
        return BetaSpec(
            left=zeros_like(self.left),
            right=zeros_like(self.right),
            beta=self.beta,
            cutoff=self.cutoff,
        )


@dataclass(frozen=True)
class GammaSpec(StructuralRecordTransportMixin):
    left: object
    right: object
    width: float

    def validate(self):
        if self.left.ndim != 1:
            raise ValueError
        if self.right.ndim != 1:
            raise ValueError
        if self.left.shape[0] != self.right.shape[0]:
            raise ValueError
        if self.width <= 0:
            raise ValueError

    def payload_fields(self):
        return (self.left, self.right)

    def metadata_fields(self):
        return (self.width,)

    @classmethod
    def from_payload(cls, metadata, payload):
        return cls(*payload, *metadata)

    def zeroed(self):
        return GammaSpec(
            left=zeros_like(self.left),
            right=zeros_like(self.right),
            width=self.width,
        )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "manual_structural_record_mechanics"
    )

    assert "AlphaSpec" in finding.summary
    assert "BetaSpec" in finding.summary
    assert "StructuralRecordBase" in (finding.scaffold or "")


def test_detects_prefixed_role_field_bundle(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


class ChildrenAuxDataPyTreeMixin:
    pass


@dataclass(frozen=True)
class DirectionalBatchInputs(ChildrenAuxDataPyTreeMixin):
    receptor_coords: object
    poses_coords: object
    receptor_anchor_indices: object
    receptor_directions: object
    ligand_anchor_indices: object
    ligand_local_directions: object
    ligand_frame_coords: object
    receptor_strengths: object
    ligand_strengths: object
    receptor_alignment_sign: float
    ligand_alignment_sign: float
    ideal_distance: float
    distance_width: float

    def _tree_children(self):
        return (
            self.receptor_coords,
            self.poses_coords,
            self.receptor_anchor_indices,
            self.receptor_directions,
            self.ligand_anchor_indices,
            self.ligand_local_directions,
            self.ligand_frame_coords,
            self.receptor_strengths,
            self.ligand_strengths,
        )

    def _tree_aux_data(self):
        return (
            self.receptor_alignment_sign,
            self.ligand_alignment_sign,
            self.ideal_distance,
            self.distance_width,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            receptor_coords=children[0],
            poses_coords=children[1],
            receptor_anchor_indices=children[2],
            receptor_directions=children[3],
            ligand_anchor_indices=children[4],
            ligand_local_directions=children[5],
            ligand_frame_coords=children[6],
            receptor_strengths=children[7],
            ligand_strengths=children[8],
            receptor_alignment_sign=aux_data[0],
            ligand_alignment_sign=aux_data[1],
            ideal_distance=aux_data[2],
            distance_width=aux_data[3],
        )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "prefixed_role_field_bundle"
    )

    assert "DirectionalBatchInputs" in finding.summary
    assert "receptor" in finding.summary
    assert "ligand" in finding.summary
    assert "anchor_indices" in finding.summary
    assert "alignment_sign" in finding.summary
    assert "Protocol" not in (finding.scaffold or "")


def test_detects_repeated_guard_validator_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def contains_group(handles, required):
    return all(handle in handles for handle in required)


def alpha_handles():
    return ("A1", "A2")


def beta_handles():
    return ("B1",)


def gamma_handles():
    return ("C1",)


def has_alpha_chain(plan):
    witness = plan.witness
    if not isinstance(witness, AlphaWitness):
        return False
    if plan.case != "alpha":
        return False
    if plan.total_gap is None:
        return False
    if plan.total_gap > witness.bound:
        return False
    return contains_group(plan.theorem_handles, alpha_handles())


def has_beta_chain(plan):
    witness = plan.witness
    if not isinstance(witness, BetaWitness):
        return False
    if plan.case != "beta":
        return False
    if plan.total_gap is None:
        return False
    if plan.total_gap > witness.bound:
        return False
    return contains_group(plan.theorem_handles, beta_handles())


def has_gamma_chain(plan):
    witness = plan.witness
    if not isinstance(witness, GammaWitness):
        return False
    if plan.case != "gamma":
        return False
    if plan.total_gap is None:
        return False
    if plan.total_gap > witness.bound:
        return False
    return contains_group(plan.theorem_handles, gamma_handles())
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_guard_validator_family"
    )

    assert "has_alpha_chain" in finding.summary
    assert "has_beta_chain" in finding.summary
    assert "ValidationCasePolicy" in (finding.scaffold or "")


def test_detects_repeated_validate_shape_guard_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class AnchoredArray:
    def __init__(self, positions, vectors, strengths):
        self.positions = positions
        self.vectors = vectors
        self.strengths = strengths

    def validate(self):
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.vectors.ndim != 2 or self.vectors.shape[1] != 3:
            raise ValueError("vectors must have shape (N, 3)")
        if self.strengths.ndim != 1:
            raise ValueError("strengths must be 1D")
        if self.positions.shape[0] != self.vectors.shape[0]:
            raise ValueError("positions and vectors must align")
        if self.positions.shape[0] != self.strengths.shape[0]:
            raise ValueError("positions and strengths must align")


class IndexedArray:
    def __init__(self, atom_rows, reference_rows, weights):
        self.atom_rows = atom_rows
        self.reference_rows = reference_rows
        self.weights = weights

    def validate(self):
        if self.atom_rows.ndim != 2 or self.atom_rows.shape[1] != 3:
            raise ValueError("rows must have shape (N, 3)")
        if self.reference_rows.ndim != 2 or self.reference_rows.shape[1] != 3:
            raise ValueError("references must have shape (N, 3)")
        if self.weights.ndim != 1:
            raise ValueError("weights must be 1D")
        if self.atom_rows.shape[0] != self.reference_rows.shape[0]:
            raise ValueError("row families must align")
        if self.atom_rows.shape[0] != self.weights.shape[0]:
            raise ValueError("rows and weights must align")
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_validate_shape_guard_family"
    )

    assert "AnchoredArray.validate" in finding.summary
    assert "IndexedArray.validate" in finding.summary
    assert "ShapeValidatedRecord" in (finding.scaffold or "")


def test_detects_cross_module_repeated_validate_shape_guard_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/chemistry.py",
        """
class AnchoredArray:
    def __init__(self, positions, vectors, strengths):
        self.positions = positions
        self.vectors = vectors
        self.strengths = strengths

    def validate(self):
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.vectors.ndim != 2 or self.vectors.shape[1] != 3:
            raise ValueError("vectors must have shape (N, 3)")
        if self.strengths.ndim != 1:
            raise ValueError("strengths must be 1D")
        if self.positions.shape[0] != self.vectors.shape[0]:
            raise ValueError("positions and vectors must align")
""",
    )
    _write_module(
        tmp_path,
        "pkg/scoring.py",
        """
class ReceptorGrid:
    def __init__(self, centers, normals, weights):
        self.centers = centers
        self.normals = normals
        self.weights = weights

    def validate(self):
        if self.centers.ndim != 2 or self.centers.shape[1] != 3:
            raise ValueError("centers must have shape (N, 3)")
        if self.normals.ndim != 2 or self.normals.shape[1] != 3:
            raise ValueError("normals must have shape (N, 3)")
        if self.weights.ndim != 1:
            raise ValueError("weights must be 1D")
        if self.centers.shape[0] != self.normals.shape[0]:
            raise ValueError("centers and normals must align")
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_validate_shape_guard_family"
        and "AnchoredArray.validate" in finding.summary
        and "ReceptorGrid.validate" in finding.summary
    )

    assert "repeat 4 shared shape/ndim guard forms" in finding.summary
    assert "ShapeValidatedRecord" in (finding.scaffold or "")


def test_detects_pairwise_validate_shape_guard_family_without_full_intersection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/a.py",
        """
class AnchoredArray:
    def __init__(self, positions, strengths):
        self.positions = positions
        self.strengths = strengths

    def validate(self):
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.strengths.ndim != 1:
            raise ValueError("strengths must be 1D")
""",
    )
    _write_module(
        tmp_path,
        "pkg/b.py",
        """
class IndexedArray:
    def __init__(self, rows, mask, strengths):
        self.rows = rows
        self.mask = mask
        self.strengths = strengths

    def validate(self):
        if self.rows.ndim != 2 or self.mask.ndim != 2:
            raise ValueError("rows and masks must be 2D")
        if self.strengths.ndim != 1:
            raise ValueError("strengths must be 1D")
        if self.rows.shape != self.mask.shape:
            raise ValueError("rows and masks must match")
""",
    )
    _write_module(
        tmp_path,
        "pkg/c.py",
        """
class ReceptorGrid:
    def __init__(self, coords, mask):
        self.coords = coords
        self.mask = mask

    def validate(self):
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3)")
        if self.coords.shape != self.mask.shape:
            raise ValueError("coords and mask must match")
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_validate_shape_guard_family"
        and "AnchoredArray.validate" in finding.summary
        and "IndexedArray.validate" in finding.summary
        and "ReceptorGrid.validate" in finding.summary
    )

    assert "repeat 4 shared shape/ndim guard forms" in finding.summary
    assert "ShapeValidatedRecord" in (finding.scaffold or "")


def test_detects_transport_shell_template_method(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC, abstractmethod
from typing import Generic, TypeVar


class ArtifactBase:
    pass


class AlphaArtifact(ArtifactBase):
    pass


class BetaArtifact(ArtifactBase):
    pass


ArtifactT = TypeVar("ArtifactT", bound=ArtifactBase)
ResultT = TypeVar("ResultT")


def materialize_artifact(artifact_cls, source, **kwargs):
    del source, kwargs
    return artifact_cls()


class ArtifactShell(ABC, Generic[ArtifactT, ResultT]):
    artifact_cls: type[ArtifactT]

    def execute(self, source):
        artifact = materialize_artifact(
            self.artifact_cls,
            source,
            **self.options(source),
        )
        return self.package(self.operate(artifact))

    def options(self, source):
        del source
        return {}

    @abstractmethod
    def operate(self, artifact: ArtifactT) -> ResultT:
        raise NotImplementedError

    @abstractmethod
    def package(self, result: ResultT):
        raise NotImplementedError


class AlphaShell(ArtifactShell[AlphaArtifact, AlphaArtifact]):
    artifact_cls = AlphaArtifact

    def operate(self, artifact: AlphaArtifact) -> AlphaArtifact:
        return artifact

    def package(self, result: AlphaArtifact):
        return result


class BetaShell(ArtifactShell[BetaArtifact, BetaArtifact]):
    artifact_cls = BetaArtifact

    def operate(self, artifact: BetaArtifact) -> BetaArtifact:
        return artifact

    def package(self, result: BetaArtifact):
        return result
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "transport_shell_template_method"
    )

    assert "ArtifactShell.execute" in finding.summary
    assert "AlphaArtifact" in finding.summary
    assert "BetaArtifact" in finding.summary
    assert "operate" in finding.summary
    assert "package" in finding.summary


def test_detects_cross_module_spec_axis_authority(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        """
class AlphaArtifact:
    pass


class BetaArtifact:
    pass


def execute_alpha(artifact):
    return artifact


def execute_beta(artifact):
    return artifact


class GeneratedWrapperRule:
    def __init__(self, *, name, artifact_cls, executor):
        self.name = name
        self.artifact_cls = artifact_cls
        self.executor = executor


WRAPPER_RULES = (
    GeneratedWrapperRule(
        name="wrap_alpha",
        artifact_cls=AlphaArtifact,
        executor=execute_alpha,
    ),
    GeneratedWrapperRule(
        name="wrap_beta",
        artifact_cls=BetaArtifact,
        executor=execute_beta,
    ),
)
""",
    )
    _write_module(
        tmp_path,
        "pkg/benchmark.py",
        """
from pkg.pipeline import (
    AlphaArtifact,
    BetaArtifact,
    execute_alpha,
    execute_beta,
)


def package_outcome(result):
    return result


class BenchmarkRoute:
    def __init__(self, *, path_name, artifact_cls, executor, outcome_builder):
        self.path_name = path_name
        self.artifact_cls = artifact_cls
        self.executor = executor
        self.outcome_builder = outcome_builder


ALPHA_ROUTE = BenchmarkRoute(
    path_name="alpha",
    artifact_cls=AlphaArtifact,
    executor=execute_alpha,
    outcome_builder=package_outcome,
)

BETA_ROUTE = BenchmarkRoute(
    path_name="beta",
    artifact_cls=BetaArtifact,
    executor=execute_beta,
    outcome_builder=package_outcome,
)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "cross_module_spec_axis_authority"
    )

    assert "WRAPPER_RULES" in finding.summary
    assert "ALPHA_ROUTE" in finding.summary
    assert "AlphaArtifact->execute_alpha" in finding.summary
    assert "BetaArtifact->execute_beta" in finding.summary


def test_detects_parallel_registry_projection_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class AlphaAuthority:
    @classmethod
    def declared_variants(cls):
        return ()


class BetaAuthority:
    @classmethod
    def declared_variants(cls):
        return ()


class AlphaProjection:
    def __init__(self, *, sites):
        self.sites = sites


class BetaProjection:
    def __init__(self, *, sites):
        self.sites = sites


def _collect_sites(structure, extractor_types):
    return tuple(extractor_types)


def projection_from_alpha(source):
    return AlphaProjection(
        sites=_collect_sites(source, AlphaAuthority.declared_variants())
    )


def projection_from_beta(source):
    return BetaProjection(
        sites=_collect_sites(source, BetaAuthority.declared_variants())
    )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "parallel_registry_projection_family"
    )

    assert "projection_from_alpha" in finding.summary
    assert "projection_from_beta" in finding.summary
    assert "AlphaAuthority" in finding.summary
    assert "BetaAuthority" in finding.summary


def test_detects_repeated_keyed_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/a.py",
        """
from abc import ABC, abstractmethod


class AutoRegisterByClassVar:
    pass


class SamplingStrategyPolicy(AutoRegisterByClassVar, ABC):
    registry_key_attr = "strategy"
    _registry = {}

    @classmethod
    def for_strategy(cls, strategy):
        try:
            return cls._registry[strategy]
        except KeyError as error:
            raise ValueError(f"Unsupported sampling strategy: {strategy}") from error

    @abstractmethod
    def keep_ratio(self):
        raise NotImplementedError


class CertificationDecisionSummaryPolicy(AutoRegisterByClassVar, ABC):
    registry_key_attr = "decision"
    _registry = {}

    @classmethod
    def for_decision(cls, decision):
        try:
            return cls._registry[decision]
        except KeyError as error:
            raise ValueError(f"Unsupported decision: {decision}") from error

    @abstractmethod
    def format(self, value):
        raise NotImplementedError
""",
    )
    _write_module(
        tmp_path,
        "pkg/b.py",
        """
from abc import ABC, abstractmethod


class AutoRegisterByClassVar:
    pass


class ScoringBackendFactory(AutoRegisterByClassVar, ABC):
    registry_key_attr = "family"
    _registry = {}

    @classmethod
    def for_family(cls, family):
        try:
            return cls._registry[family]
        except KeyError as error:
            raise ValueError(f"Unsupported family: {family}") from error

    @abstractmethod
    def create_backend(self):
        raise NotImplementedError
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_keyed_family"
    )

    assert "SamplingStrategyPolicy" in finding.summary
    assert "CertificationDecisionSummaryPolicy" in finding.summary
    assert "ScoringBackendFactory" in finding.summary
    assert "KeyedNominalFamily" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "cls.__registry__[key]" in (finding.scaffold or "")


def test_detects_manual_keyed_record_table(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class MetalChargeCompatibility:
    charge_method: str
    incompatibility_reasons: tuple[str, ...] = ()
    _registry = {}

    @classmethod
    def register(cls, *, charge_method, incompatibility_reasons=()):
        if charge_method in cls._registry:
            raise TypeError(charge_method)
        cls._registry[charge_method] = cls(
            charge_method=charge_method,
            incompatibility_reasons=incompatibility_reasons,
        )

    @classmethod
    def for_charge_method(cls, charge_method):
        if charge_method not in cls._registry:
            raise TypeError(charge_method)
        return cls._registry[charge_method]


@dataclass(frozen=True)
class ScoringFamilyCompatibility:
    scoring_family: str
    reasons: tuple[str, ...] = ()
    _registry = {}

    @classmethod
    def register(cls, *, scoring_family, reasons=()):
        if scoring_family in cls._registry:
            raise TypeError(scoring_family)
        cls._registry[scoring_family] = cls(
            scoring_family=scoring_family,
            reasons=reasons,
        )

    @classmethod
    def for_scoring_family(cls, scoring_family):
        if scoring_family not in cls._registry:
            raise TypeError(scoring_family)
        return cls._registry[scoring_family]


@dataclass(frozen=True)
class ComponentCompatibilityRule:
    role: str
    projector: object
    _registry = {}

    @classmethod
    def register(cls, *, role, projector):
        if role in cls._registry:
            raise TypeError(role)
        cls._registry[role] = cls(role=role, projector=projector)

    @classmethod
    def for_role(cls, role):
        if role not in cls._registry:
            raise TypeError(role)
        return cls._registry[role]
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "manual_keyed_record_table"
    )

    assert "MetalChargeCompatibility" in finding.summary
    assert "ScoringFamilyCompatibility" in finding.summary
    assert "ComponentCompatibilityRule" in finding.summary
    assert "KeyedRecordTable" in (finding.scaffold or "")


def test_detects_repeated_result_assembly_pipeline(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Sampler:
    def sample_from_certified(self, key, n_poses, pocket):
        templates, template_weights = self.certified_templates(pocket)
        key_trans, key_rot = random.split(key)
        indices = select_template_indices(key_trans, template_weights, n_poses)
        translations = sample_biased_translations(
            key_trans, templates, template_weights, n_poses
        )
        quaternions = sample_biased_rotations(key_rot, templates, indices, n_poses)
        return SamplingResult(
            translations=translations,
            quaternions=quaternions,
            strategy=SamplingStrategy.GUIDED,
            n_guided=n_poses,
            n_random=0,
            templates_used=len(templates),
        )

    def sample_from_analysis(self, request):
        templates, template_weights = self.analysis_templates(
            request.coords, request.shape, request.features
        )
        key_trans, key_rot = random.split(request.key)
        indices = select_template_indices(key_trans, template_weights, request.n_poses)
        translations = sample_biased_translations(
            key_trans, templates, template_weights, request.n_poses
        )
        quaternions = sample_biased_rotations(
            key_rot, templates, indices, request.n_poses
        )
        return SamplingResult(
            translations=translations,
            quaternions=quaternions,
            strategy=SamplingStrategy.GUIDED,
            n_guided=request.n_poses,
            n_random=0,
            templates_used=len(templates),
        )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_result_assembly_pipeline"
    )

    assert "sample_from_certified" in finding.summary
    assert "sample_from_analysis" in finding.summary
    assert "sample_biased_rotations" in finding.summary


def test_detects_nested_builder_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class SearchRequest:
    @classmethod
    def from_inputs(
        cls,
        *,
        key,
        ligand_com,
        strategy,
        n_poses=None,
        n_poses_override=None,
    ):
        return cls(
            key=key,
            ligand_com=ligand_com,
            strategy=strategy,
            n_poses=n_poses,
            n_poses_override=n_poses_override,
        )


class ExecutionRequest:
    @classmethod
    def from_detected_site(
        cls,
        site,
        *,
        key,
        ligand_com,
        strategy,
        n_poses=None,
        n_poses_override=None,
    ):
        return cls(
            search=SearchRequest.from_inputs(
                key=key,
                ligand_com=ligand_com,
                strategy=strategy,
                n_poses=n_poses,
                n_poses_override=n_poses_override,
            ),
            center=site.center,
            box_size=max(site.radius, extent(site)),
        )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "nested_builder_shell"
    )

    assert "ExecutionRequest.from_detected_site" in finding.summary
    assert "SearchRequest.from_inputs" in finding.summary
    assert "key, ligand_com, strategy, n_poses, n_poses_override" in finding.summary


def test_detects_manual_fiber_tag_with_abc_fix(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Notification:
    def __init__(self, kind, recipient, subject=None, body=None, phone=None, device_token=None):
        self.kind = kind
        self.recipient = recipient
        self.subject = subject
        self.body = body
        self.phone = phone
        self.device_token = device_token

    def send(self):
        if self.kind == "email":
            return smtp_send(self.recipient, self.subject, self.body)
        elif self.kind == "sms":
            return twilio_send(self.phone, self.body)
        elif self.kind == "push":
            return apns_send(self.device_token, self.body)
        raise ValueError(self.kind)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(item for item in findings if item.detector_id == "manual_fiber_tag")
    assert "self.kind" in finding.summary
    assert finding.scaffold is not None
    assert "class Notification(ABC)" in finding.scaffold


def test_detects_descriptor_derived_view_drift(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Model:
    def __init__(self, table_name):
        self.table_name = table_name
        self.select_query = f"SELECT * FROM {self.table_name}"
        self.insert_query = f"INSERT INTO {self.table_name}"
        self.count_query = f"SELECT COUNT(*) FROM {self.table_name}"

    def rename_table(self, new_name):
        self.table_name = new_name
        self.select_query = f"SELECT * FROM {self.table_name}"
        self.insert_query = f"INSERT INTO {self.table_name}"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item for item in findings if item.detector_id == "descriptor_derived_view"
    )
    assert "count_query" in finding.summary
    assert finding.scaffold is not None
    assert "class DerivedField" in finding.scaffold


def test_detects_deferred_class_registration(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
HANDLERS = {}


def register_handler(event_type):
    def decorator(cls):
        HANDLERS[event_type] = cls
        return cls
    return decorator


@register_handler("user.created")
class UserCreatedHandler:
    def handle(self, event):
        return event


@register_handler("order.placed")
class OrderPlacedHandler:
    def handle(self, event):
        return event


class PaymentFailedHandler:
    def handle(self, event):
        return event
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item for item in findings if item.detector_id == "deferred_class_registration"
    )
    assert "HANDLERS" in finding.summary
    assert finding.scaffold is not None
    assert "from metaclass_registry import AutoRegisterMeta" in finding.scaffold
    assert "type_for_event_type" in finding.scaffold
    assert "cls.__registry__[event_type]" in finding.scaffold


def test_detects_structural_confusability_without_abc_witness(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def process_batch(items, backend):
    for item in items:
        backend.store(item)
    backend.flush()


class DatabaseBackend:
    def store(self, item):
        return item

    def flush(self):
        return None


class CacheBackend:
    def store(self, item):
        return item

    def flush(self):
        return None

    def invalidate(self):
        return None
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item for item in findings if item.detector_id == "structural_confusability"
    )
    assert "process_batch" in finding.summary
    assert finding.scaffold is not None
    assert "class BackendInterface(ABC)" in finding.scaffold


def test_ignores_structural_confusability_when_abstract_witness_exists(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC, abstractmethod


def process_batch(items, backend):
    for item in items:
        backend.store(item)
    backend.flush()


class BackendInterface(ABC):
    @abstractmethod
    def store(self, item):
        raise NotImplementedError

    @abstractmethod
    def flush(self):
        raise NotImplementedError


class DatabaseBackend(BackendInterface):
    def store(self, item):
        return item

    def flush(self):
        return None


class CacheBackend(BackendInterface):
    def store(self, item):
        return item

    def flush(self):
        return None
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(
        item.detector_id == "structural_confusability" for item in findings
    )


def test_detects_semantic_witness_family_with_abc_base(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionTrace:
    file_path: str
    function_name: str
    line: int
    helper_names: tuple[str, ...]


@dataclass(frozen=True)
class RegistryTrace:
    source_path: str
    registry_name: str
    init_line: int
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class ExportTrace:
    artifact_path: str
    subject_name: str
    method_line: int
    export_names: tuple[str, ...]
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item for item in findings if item.detector_id == "semantic_witness_family"
    )
    assert "FunctionTrace" in finding.summary
    assert finding.scaffold is not None
    assert "class SemanticCarrier(ABC)" in finding.scaffold


def test_detects_mixin_enforcement_for_renamed_semantic_roles(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionTrace:
    file_path: str
    function_name: str
    method_line: int
    helper_names: tuple[str, ...]


@dataclass(frozen=True)
class RegistryTrace:
    source_path: str
    registry_name: str
    line: int
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class ExportTrace:
    artifact_path: str
    subject_name: str
    init_line: int
    export_names: tuple[str, ...]
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        item
        for item in findings
        if item.detector_id == "mixin_enforcement"
        and "function_name" in item.summary
        and "class_names" in item.summary
    )
    assert finding.scaffold is not None
    assert "class PrimaryNameMixin(ABC)" in finding.scaffold
    assert "(SemanticCarrier, PrimaryNameMixin" in finding.scaffold
    assert finding.codemod_patch is not None
    assert "multiple inheritance" in finding.codemod_patch


def test_detects_sentinel_attribute_simulation(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    sigma = "alpha"


class Beta:
    sigma = "beta"


def choose(obj):
    if obj.sigma == "alpha":
        return 1
    return 2
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 1 for finding in findings)


def test_detects_predicate_factory_chain(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def build(param_type):
    if is_optional(param_type):
        return OptionalInfo()
    elif is_dataclass(param_type):
        return DataclassInfo()
    return GenericInfo()
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 2 for finding in findings)


def test_detects_config_attribute_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(config):
    if hasattr(config, "napari_port"):
        return config.napari_port
    if getattr(config, "viewer_type", None) == "fiji":
        return 2
    return 0
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 4 for finding in findings)


def test_detects_concrete_config_field_probe(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class VinardoConfig:
    gaussians: tuple[tuple[float, float], ...] = ()
    repulsion: float = 0.0
    hydrophobic_low: float = 0.0
    cutoff: float = 8.0


@dataclass(frozen=True)
class SoftLJConfig:
    repulsion_exp: int = 8
    attraction_exp: int = 4
    repulsion_weight: float = 4.0
    attraction_weight: float = 2.0
    cutoff: float = 8.0


class ScoringBackend(ABC):
    _config: VinardoConfig | SoftLJConfig


class SoftLJBackend(ScoringBackend):
    def __init__(self, config: SoftLJConfig | None = None):
        self._config = config if config is not None else SoftLJConfig()

    def score(self):
        cfg = self._config
        return (
            getattr(cfg, "gaussians"),
            getattr(cfg, "repulsion"),
            getattr(cfg, "hydrophobic_low"),
            cfg.cutoff,
        )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "concrete_config_field_probe"
    )

    assert "SoftLJBackend.score" in finding.summary
    assert "SoftLJConfig" in finding.summary
    assert "gaussians" in finding.summary
    assert "repulsion" in finding.summary


def test_collects_config_dispatch_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(config):
    if hasattr(config, "napari_port"):
        return config.napari_port
    if getattr(config, "viewer_type", None) == "fiji":
        return 2
    return 0
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, ConfigDispatchObservationFamily)

    assert {item.observed_attribute for item in observations} == {
        "napari_port",
        "viewer_type",
    }


def test_ignores_single_generic_name_sentinel_branch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    name = "alpha"


class Beta:
    name = "beta"


def choose(obj):
    if obj.name == "alpha":
        return 1
    return 2
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(finding.pattern_id == 1 for finding in findings)


def test_detects_generated_type_lineage(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
BASE_TO_LAZY = {}


class Base:
    pass


LazyBase = type("LazyBase", (Base,), {})
BASE_TO_LAZY[Base] = LazyBase
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 7 for finding in findings)


def test_collects_generated_type_lineage_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
BASE_TO_LAZY = {}


class Base:
    pass


LazyBase = type("LazyBase", (Base,), {})
BASE_TO_LAZY[Base] = LazyBase
""",
    )

    module = parse_python_modules(tmp_path)[0]
    generation = collect_family_items(module, RuntimeTypeGenerationObservationFamily)
    lineage = collect_family_items(module, LineageMappingObservationFamily)

    assert [item.generator_name for item in generation] == ["type"]
    assert [item.mapping_name for item in lineage] == ["BASE_TO_LAZY"]


def test_ignores_type_introspection_for_generated_type_lineage(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Box:
    def clone(self):
        return type(self)()
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(finding.detector_id == "generated_type_lineage" for finding in findings)

    module = parse_python_modules(tmp_path)[0]
    generation = collect_family_items(module, RuntimeTypeGenerationObservationFamily)
    assert generation == []


def test_detects_dual_axis_resolution(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(scope_stack, obj):
    for scope in scope_stack:
        for mro_type in type(obj).__mro__:
            if scope and mro_type:
                return scope, mro_type
    return None
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 8 for finding in findings)


def test_collects_dual_axis_resolution_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(scope_stack, obj):
    for scope in scope_stack:
        for mro_type in type(obj).__mro__:
            if scope and mro_type:
                return scope, mro_type
    return None
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, DualAxisResolutionObservationFamily)

    assert len(observations) == 1
    assert observations[0].outer_axis_name == "scope"
    assert observations[0].inner_axis_name == "mro_type"


def test_detects_manual_virtual_membership(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def check(instance):
    if hasattr(instance.__class__, "_is_global_config"):
        return instance.__class__._is_global_config
    return False
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 9 for finding in findings)


def test_collects_class_marker_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def check(instance):
    if hasattr(instance.__class__, "_is_global_config"):
        return instance.__class__._is_global_config
    return False
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, ClassMarkerObservationFamily)

    assert any(item.marker_name == "_is_global_config" for item in observations)


def test_detects_dynamic_interface_generation(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


def make_interface(name):
    return type(name, (ABC,), {})
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 10 for finding in findings)


def test_collects_interface_generation_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


def make_interface(name):
    return type(name, (ABC,), {})
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, InterfaceGenerationObservationFamily)

    assert [item.generator_name for item in observations] == ["type"]


def test_detects_sentinel_type_marker(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
SENTINEL = type("Sentinel", (), {})()


def present(registry):
    return SENTINEL in registry
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 11 for finding in findings)


def test_collects_sentinel_type_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
SENTINEL = type("Sentinel", (), {})()


def present(registry):
    return SENTINEL in registry
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, SentinelTypeObservationFamily)

    assert any(item.sentinel_name == "SENTINEL" for item in observations)
    assert len(observations) >= 2


def test_detects_dynamic_method_injection(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def inject(target_type, method_name, method_impl):
    setattr(target_type, method_name, method_impl)
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 12 for finding in findings)


def test_collects_dynamic_method_injection_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def inject(target_type, method_name, method_impl):
    setattr(target_type, method_name, method_impl)
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, DynamicMethodInjectionObservationFamily)

    assert [item.mutator_name for item in observations] == ["setattr"]


def test_markdown_output_includes_prescription_details(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def build(param_type):
    if is_optional(param_type):
        return OptionalInfo()
    elif is_dataclass(param_type):
        return DataclassInfo()
    return GenericInfo()
""",
    )

    findings = analyze_path(tmp_path)
    output = _format_markdown(findings)
    assert "Prescription:" in output
    assert "Canonical shape:" in output
    assert "First move:" in output
    assert "Example skeleton:" in output


def test_markdown_output_handles_multiple_example_skeletons(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def _prepare(self, item):
        ready = self.normalize(item)
        checked = self.validate(ready)
        return self.finish(checked)


class Beta:
    def _build(self, value):
        ready = self.normalize(value)
        checked = self.validate(ready)
        return self.finish(checked)
""",
    )

    findings = analyze_path(tmp_path)
    output = _format_markdown(findings)
    assert output.count("Example skeleton:") >= 2
    assert "Suggested scaffold:" in output
    assert "Suggested patch:" in output


def test_clusters_redundant_methods_into_abc_candidate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def _prepare(self, item):
        ready = self.normalize(item)
        checked = self.validate(ready)
        return self.finish(checked)

    def _score(self, item):
        scored = self.compute(item)
        bounded = self.bound(scored)
        packaged = self.package(bounded)
        return self.finish(packaged)


class Beta:
    def _build(self, value):
        ready = self.normalize(value)
        checked = self.validate(ready)
        return self.finish(checked)

    def _evaluate(self, value):
        scored = self.compute(value)
        bounded = self.bound(scored)
        packaged = self.package(bounded)
        return self.finish(packaged)
""",
    )

    findings = analyze_path(tmp_path)
    assert any(
        finding.detector_id == "inheritance_hierarchy_candidate" for finding in findings
    )


def test_observation_graph_recovers_method_coherence_cohort(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def _prepare(self, item):
        ready = self.normalize(item)
        checked = self.validate(ready)
        return self.finish(checked)

    def _score(self, item):
        scored = self.compute(item)
        bounded = self.bound(scored)
        packaged = self.package(bounded)
        return self.finish(packaged)


class Beta:
    def _build(self, value):
        ready = self.normalize(value)
        checked = self.validate(ready)
        return self.finish(checked)

    def _evaluate(self, value):
        scored = self.compute(value)
        bounded = self.bound(scored)
        packaged = self.package(bounded)
        return self.finish(packaged)


class Gamma:
    def _render(self, payload):
        ready = self.normalize(payload)
        checked = self.validate(ready)
        return self.finish(checked)
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, MethodShapeFamily)
    graph = ObservationGraph(
        tuple(item.structural_observation for item in observations)
    )

    cohorts = graph.coherence_cohorts_for(
        ObservationKind.METHOD_SHAPE,
        StructuralExecutionLevel.FUNCTION_BODY,
        minimum_witnesses=2,
        minimum_fibers=2,
    )

    cohort = next(
        item for item in cohorts if item.nominal_witnesses == ("Alpha", "Beta")
    )
    assert len(cohort.fibers) == 2


def test_detects_attribute_probe_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(widget):
    if hasattr(widget, \"isChecked\"):
        return widget.isChecked()
    return getattr(widget, \"value\", None)
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.detector_id == "attribute_probes" for finding in findings)


def test_collects_attribute_probe_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(widget):
    if hasattr(widget, "checked"):
        return widget.checked
    try:
        return getattr(widget, "value", None)
    except AttributeError:
        return None
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, AttributeProbeObservationFamily)

    assert {item.probe_kind for item in observations} == {
        "hasattr",
        "getattr",
        "attribute_error",
    }
    assert any(item.observed_attribute == "checked" for item in observations)


def test_ignores_array_protocol_attribute_probes(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def validate(value):
    shape = getattr(value, "shape", None)
    ndim = getattr(value, "ndim", None)
    dtype = getattr(value, "dtype", None)
    return shape, ndim, dtype
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(finding.detector_id == "attribute_probes" for finding in findings)


def test_collects_literal_dispatch_observations_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def convert(kind, value):
    if kind == "numpy":
        return value
    elif kind == "cupy":
        return value
    return value


def walk(node):
    if node.kind == "alpha":
        return 1
    if node.kind == "beta":
        return 2
    return 0
""",
    )

    module = parse_python_modules(tmp_path)[0]
    chains = collect_family_items(module, StringLiteralDispatchObservationFamily)
    inline_groups = collect_family_items(
        module, InlineStringLiteralDispatchObservationFamily
    )

    assert any(item.axis_expression == "kind" for item in chains)
    assert any(item.axis_expression == "node.kind" for item in inline_groups)


def test_detects_string_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def convert(kind, value):
    if kind == \"numpy\":
        return value
    elif kind == \"cupy\":
        return value
    elif kind == \"torch\":
        return value
    return value
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "string_dispatch"
    )
    assert finding.pattern_id == 3
    assert "`kind`" in finding.summary
    assert "'numpy'" in finding.summary
    assert finding.certification == "certified"


def test_detects_inline_literal_dispatch_registry_smell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def walk(node):
    if node.kind == "alpha":
        return 1
    if node.kind == "beta":
        return 2
    return 0
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.detector_id == "inline_literal_dispatch" for finding in findings)


def test_detects_bidirectional_registry(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Registry:
    def __init__(self):
        self._forward = {}
        self._reverse = {}

    def register(self, left, right):
        self._forward[left] = right
        self._reverse[right] = left
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 13 for finding in findings)


def test_detects_repeated_builder_call_shape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def build(self, candidate):
        return RuntimePlan(
            pose_id=candidate.pose_id,
            score=candidate.score,
            theorem_handles=tuple(candidate.theorem_handles),
        )


class Beta:
    def build(self, entry):
        return RuntimePlan(
            pose_id=entry.pose_id,
            score=entry.score,
            theorem_handles=tuple(entry.theorem_handles),
        )
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 14 for finding in findings)
    assert any(finding.pattern_id == 14 and finding.scaffold for finding in findings)
    assert any(
        finding.pattern_id == 14 and finding.codemod_patch for finding in findings
    )


def test_detects_single_owner_builder_call_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    parser.add_argument(
        "--include-plans",
        action="store_true",
        help="Include planning details",
    )
    parser.add_argument(
        "--min-builder-keywords",
        type=int,
        default=3,
        help="Minimum builder keywords",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        dest="excluded_pattern_ids",
        default=[],
        help="Exclude one pattern id",
    )
    return parser
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_builder_calls"
        and "main" in finding.summary
        and "add_argument" in finding.summary
    )

    assert "InvocationSpec" in (finding.scaffold or "")
    assert "declarative invocation table" in (finding.codemod_patch or "")


def test_cli_argument_specs_build_parser_for_flag_actions() -> None:
    parser = argparse.ArgumentParser()
    for spec in _CLI_ARGUMENT_SPECS:
        spec.add_to_parser(parser)

    args = parser.parse_args(
        [
            "--json",
            "--include-plans",
            "--exclude-pattern",
            "14",
            "nominal_refactor_advisor",
        ]
    )

    assert args.json is True
    assert args.include_plans is True
    assert args.excluded_pattern_ids == [14]
    assert args.path == "nominal_refactor_advisor"


def test_detects_manual_class_registration(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
REGISTRY = {}


class AlphaHandler:
    pass


class BetaHandler:
    pass


REGISTRY["alpha"] = AlphaHandler
REGISTRY["beta"] = BetaHandler
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 6 for finding in findings)
    assert any(
        finding.pattern_id == 6
        and "from metaclass_registry import AutoRegisterMeta"
        in (finding.scaffold or "")
        for finding in findings
    )
    assert any(
        finding.pattern_id == 6
        and "__key_extractor__" in (finding.scaffold or "")
        for finding in findings
    )
    assert any(
        finding.pattern_id == 6
        and "__registry__" in (finding.codemod_patch or "")
        for finding in findings
    )


def test_detects_manual_concrete_subclass_roster_with_abstract_filter(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
import inspect
from abc import ABC, abstractmethod


class Extractor(ABC):
    _registered_types = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls._registered_types.append(cls)

    @classmethod
    def registered_types(cls):
        return tuple(cls._registered_types)

    @abstractmethod
    def extract(self):
        raise NotImplementedError


class HydrogenExtractor(Extractor):
    def extract(self):
        return ("H",)


class DonorExtractor(Extractor):
    def extract(self):
        return ("D",)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "manual_concrete_subclass_roster"
    )

    assert "Extractor" in finding.summary
    assert "_registered_types" in finding.summary
    assert "registered_types" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__key_extractor__" in (finding.scaffold or "")
    assert "AutoRegisteredFamily.__registry__.values()" in (finding.scaffold or "")


def test_detects_manual_concrete_subclass_roster_with_selector_guard(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class RoutedRequest(ABC):
    route_name = None
    _registered_types = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("route_name") is not None:
            cls._registered_types.append(cls)

    @classmethod
    def concrete_types(cls):
        return tuple(cls._registered_types)


class DirectRequest(RoutedRequest):
    route_name = "direct"


class GuidedRequest(RoutedRequest):
    route_name = "guided"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "manual_concrete_subclass_roster"
    )

    assert "route_name" in finding.summary
    assert "DirectRequest" in finding.summary
    assert "GuidedRequest" in finding.summary
    assert "metaclass-registry" in (finding.codemod_patch or "")


def test_detects_manual_concrete_subclass_roster_with_root_qualified_append(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
import inspect
from abc import ABC, abstractmethod


class HandlerBase(ABC):
    _registered_handlers = []
    _registration_index = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        cls._registration_index = HandlerBase._registration_index
        HandlerBase._registration_index += 1
        HandlerBase._registered_handlers.append(cls)

    @classmethod
    def registered_handlers(cls):
        return tuple(
            sorted(
                HandlerBase._registered_handlers,
                key=lambda item: item._registration_index,
            )
        )

    @abstractmethod
    def run(self):
        raise NotImplementedError


class AlphaHandler(HandlerBase):
    def run(self):
        return "alpha"


class BetaHandler(HandlerBase):
    def run(self):
        return "beta"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "manual_concrete_subclass_roster"
    )

    assert "HandlerBase" in finding.summary
    assert "_registered_handlers" in finding.summary
    assert "registered_handlers" in finding.summary
    assert "AlphaHandler" in finding.summary
    assert "BetaHandler" in finding.summary


def test_detects_predicate_selected_concrete_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC, abstractmethod


class AutoRegisterConcreteTypes:
    pass


class RenderRule(AutoRegisterConcreteTypes, ABC):
    _registered_types = []

    @classmethod
    def registered_types(cls):
        return (AlphaRenderRule, BetaRenderRule)

    @classmethod
    def resolve(cls, artifact):
        matches = [
            candidate
            for candidate in cls.registered_types()
            if candidate.matches_context(artifact)
        ]
        if not matches:
            raise ValueError(type(artifact).__name__)
        if len(matches) != 1:
            raise TypeError([candidate.__name__ for candidate in matches])
        return matches[0]()

    @classmethod
    @abstractmethod
    def matches_context(cls, artifact):
        raise NotImplementedError


class AlphaRenderRule(RenderRule):
    @classmethod
    def matches_context(cls, artifact):
        return artifact.kind == "alpha"


class BetaRenderRule(RenderRule):
    @classmethod
    def matches_context(cls, artifact):
        return artifact.kind == "beta"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "predicate_selected_concrete_family"
    )

    assert "RenderRule.resolve" in finding.summary
    assert "matches_context(artifact)" in finding.summary
    assert "AlphaRenderRule" in finding.summary
    assert "BetaRenderRule" in finding.summary
    assert "PredicateSelectedConcreteFamily" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__key_extractor__" in (finding.scaffold or "")
    assert "cls.__registry__.values()" in (finding.scaffold or "")


def test_detects_manual_concrete_subclass_roster_across_modules(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/base.py",
        """
from abc import ABC


class RoutedRequest(ABC):
    route_name = None
    _registered_types = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("route_name") is not None:
            cls._registered_types.append(cls)

    @classmethod
    def concrete_types(cls):
        return tuple(cls._registered_types)
""",
    )
    _write_module(
        tmp_path,
        "pkg/routes.py",
        """
from .base import RoutedRequest


class DirectRequest(RoutedRequest):
    route_name = "direct"


class GuidedRequest(RoutedRequest):
    route_name = "guided"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "manual_concrete_subclass_roster"
    )

    assert "DirectRequest" in finding.summary
    assert "GuidedRequest" in finding.summary
    assert "route_name" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__registry_key__ = \"route_name\"" in (finding.scaffold or "")


def test_detects_manual_concrete_subclass_roster_with_module_level_consumer(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC
from typing import cast


class FamilyGeneratingSpec(ABC):
    family_specs = ()
    _declaring_spec_types = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("family_specs"):
            FamilyGeneratingSpec._declaring_spec_types.append(
                cast(type[FamilyGeneratingSpec], cls)
            )


class AlphaSpec(FamilyGeneratingSpec):
    family_specs = ("alpha",)


class BetaSpec(FamilyGeneratingSpec):
    family_specs = ("beta",)


def materialize_declared_families():
    return tuple(
        spec_type.__name__
        for spec_type in FamilyGeneratingSpec._declaring_spec_types
    )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "manual_concrete_subclass_roster"
    )

    assert "FamilyGeneratingSpec" in finding.summary
    assert "_declaring_spec_types" in finding.summary
    assert "materialize_declared_families" in finding.summary
    assert "AlphaSpec" in finding.summary
    assert "BetaSpec" in finding.summary


def test_detects_predicate_selected_concrete_family_across_modules(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/base.py",
        """
from abc import ABC, abstractmethod
from .alpha import AlphaRenderRule
from .beta import BetaRenderRule


class RenderRule(ABC):
    _registered_types = []

    @classmethod
    def registered_types(cls):
        return (AlphaRenderRule, BetaRenderRule)

    @classmethod
    def resolve(cls, artifact):
        matches = [
            candidate
            for candidate in cls.registered_types()
            if candidate.matches_context(artifact)
        ]
        if not matches:
            raise ValueError(type(artifact).__name__)
        if len(matches) != 1:
            raise TypeError([candidate.__name__ for candidate in matches])
        return matches[0]()

    @classmethod
    @abstractmethod
    def matches_context(cls, artifact):
        raise NotImplementedError
""",
    )
    _write_module(
        tmp_path,
        "pkg/alpha.py",
        """
from .base import RenderRule


class AlphaRenderRule(RenderRule):
    @classmethod
    def matches_context(cls, artifact):
        return artifact.kind == "alpha"
""",
    )
    _write_module(
        tmp_path,
        "pkg/beta.py",
        """
from .base import RenderRule


class BetaRenderRule(RenderRule):
    @classmethod
    def matches_context(cls, artifact):
        return artifact.kind == "beta"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "predicate_selected_concrete_family"
    )

    assert "RenderRule.resolve" in finding.summary
    assert "AlphaRenderRule" in finding.summary
    assert "BetaRenderRule" in finding.summary
    assert "PredicateSelectedConcreteFamily" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__key_extractor__" in (finding.scaffold or "")
    assert "cls.__registry__.values()" in (finding.scaffold or "")


def test_detects_parallel_mirrored_leaf_families(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC, abstractmethod


class InvoiceFieldEmitter(ABC):
    _registered_types = []

    @abstractmethod
    def emit(self, artifact):
        raise NotImplementedError


class ReceiptFieldEmitter(ABC):
    _registered_types = []

    @abstractmethod
    def emit(self, artifact):
        raise NotImplementedError


class InvoiceAlphaEmitter(InvoiceFieldEmitter):
    def emit(self, artifact):
        return artifact.alpha


class InvoiceBetaEmitter(InvoiceFieldEmitter):
    def emit(self, artifact):
        return artifact.beta


class InvoiceGammaEmitter(InvoiceFieldEmitter):
    def emit(self, artifact):
        return artifact.gamma


class ReceiptAlphaEmitter(ReceiptFieldEmitter):
    def emit(self, artifact):
        return artifact.alpha


class ReceiptBetaEmitter(ReceiptFieldEmitter):
    def emit(self, artifact):
        return artifact.beta


class ReceiptGammaEmitter(ReceiptFieldEmitter):
    def emit(self, artifact):
        return artifact.gamma
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "parallel_mirrored_leaf_family"
    )

    assert "InvoiceFieldEmitter" in finding.summary
    assert "ReceiptFieldEmitter" in finding.summary
    assert "alpha emitter" in finding.summary
    assert "GeneratedLeafFamily" in (finding.scaffold or "")


def test_detects_helper_registration_call(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Registry:
    def register(self, cls, key):
        return cls


registry = Registry()


class Alpha:
    pass


class Beta:
    pass


registry.register(Alpha, "alpha")
registry.register(Beta, "beta")
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 6 for finding in findings)


def test_detects_decorator_registration(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def register(registry, key):
    def deco(cls):
        return cls
    return deco


REGISTRY = {}


@register(REGISTRY, "alpha")
class Alpha:
    pass


@register(REGISTRY, "beta")
class Beta:
    pass
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 6 for finding in findings)


def test_detects_auto_register_decorator_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def auto_register(registry, key):
    def deco(cls):
        return cls
    return deco


REGISTRY = {}


@auto_register(REGISTRY, "alpha")
class Alpha:
    pass


@auto_register(REGISTRY, "beta")
class Beta:
    pass
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 6 for finding in findings)


def test_collects_scoped_call_observations(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def build(self, result):
        return transform(result)
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_scoped_observations(module, (ast.Call,))
    call_observation = next(
        item
        for item in observations
        if isinstance(item.node, ast.Call)
        and getattr(item.node.func, "id", None) == "transform"
    )

    assert call_observation.class_name == "Alpha"
    assert call_observation.function_name == "build"


def test_spec_families_use_autoregistration() -> None:
    registration_specs = {
        type(spec).__name__ for spec in RegistrationShapeSpec.registered_specs()
    }
    field_specs = {
        type(spec).__name__ for spec in FieldObservationSpec.registered_specs()
    }

    assert registration_specs == {
        "AssignmentRegistrationShapeSpec",
        "CallRegistrationShapeSpec",
        "DecoratorRegistrationShapeSpec",
    }
    assert field_specs == {
        "DataclassBodyFieldObservationSpec",
        "InitAssignmentFieldObservationSpec",
    }


def test_typed_literal_specs_are_derived_from_canonical_registry() -> None:
    all_typed_specs = {
        type(spec).__name__
        for spec in TypedLiteralObservationSpec.registered_specs_for_literal_type()
    }
    string_typed_specs = {
        type(spec).__name__
        for spec in TypedLiteralObservationSpec.registered_specs_for_literal_type(str)
    }

    assert all_typed_specs == {
        "StringLiteralDispatchObservationSpec",
        "NumericLiteralDispatchObservationSpec",
        "InlineStringLiteralDispatchObservationSpec",
    }
    assert string_typed_specs == {
        "StringLiteralDispatchObservationSpec",
        "InlineStringLiteralDispatchObservationSpec",
    }


def test_detects_parallel_scoped_shape_wrappers(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass
import ast


@dataclass(frozen=True)
class NodeWrapperSpec:
    node_types: tuple[type[ast.AST], ...]
    builder: object


def _build_function_projection(parsed_module, observation):
    node = observation.node
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return (parsed_module, node, observation.class_name)


def _build_call_projection(parsed_module, observation):
    node = observation.node
    if not isinstance(node, ast.Call):
        return None
    return (parsed_module, node, observation.function_name)


_FUNCTION_PROJECTION_SPEC = NodeWrapperSpec(
    node_types=(ast.FunctionDef, ast.AsyncFunctionDef),
    builder=_build_function_projection,
)


_CALL_PROJECTION_SPEC = NodeWrapperSpec(
    node_types=(ast.Call,),
    builder=_build_call_projection,
)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "scoped_shape_wrapper"
    )

    assert "polymorphic family" in finding.title
    assert "NodeFamilySpec" in (finding.scaffold or "")


def test_detects_manual_indexed_family_expansion(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class FieldObservationSpec: ...
class FieldObservation: ...
class ConfigDispatchObservationSpec: ...
class ConfigDispatchObservation: ...


def collect_field_observations(parsed_module):
    return [
        item
        for item in _collect_items_from_spec_root(
            FieldObservationSpec, parsed_module, FieldObservation
        )
        if isinstance(item, FieldObservation)
    ]


def collect_config_dispatch_observations(parsed_module):
    return [
        item
        for item in _collect_items_from_spec_root(
            ConfigDispatchObservationSpec, parsed_module, ConfigDispatchObservation
        )
        if isinstance(item, ConfigDispatchObservation)
    ]
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.detector_id == "manual_indexed_family" for finding in findings)


def test_collects_scoped_shape_wrapper_observations_via_spec_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
import ast


def _build_method_shape_from_observation(parsed_module, observation):
    node = observation.node
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return (parsed_module, node)


_METHOD_SHAPE_SPEC = ScopedShapeSpec(
    node_types=(ast.FunctionDef, ast.AsyncFunctionDef),
    build_shape=_build_method_shape_from_observation,
)
""",
    )

    module = parse_python_modules(tmp_path)[0]
    functions = collect_family_items(module, ScopedShapeWrapperFunctionFamily)
    specs = collect_family_items(module, ScopedShapeWrapperSpecFamily)

    assert [item.function_name for item in functions] == [
        "_build_method_shape_from_observation"
    ]
    assert [item.spec_name for item in specs] == ["_METHOD_SHAPE_SPEC"]


def test_detects_namespaced_auto_register_decorator_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Plugins:
    def auto_register(self, registry, key):
        def deco(cls):
            return cls
        return deco


plugins = Plugins()
REGISTRY = {}


@plugins.auto_register(REGISTRY, "alpha")
class Alpha:
    pass


@plugins.auto_register(REGISTRY, "beta")
class Beta:
    pass
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.pattern_id == 6 for finding in findings)


def test_collects_registration_shapes_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Plugins:
    def auto_register(self, registry, key):
        def deco(cls):
            return cls
        return deco


plugins = Plugins()
REGISTRY = {}


@plugins.auto_register(REGISTRY, "alpha")
class Alpha:
    pass


REGISTRY["beta"] = Alpha
""",
    )

    module = parse_python_modules(tmp_path)[0]
    shapes = collect_family_items(module, RegistrationShapeFamily)

    assert {shape.registration_style for shape in shapes} == {
        "decorator_registration",
        "subscript_assignment",
    }


def test_detects_repeated_export_dict_shape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def export(self, result):
        return {
            "pose_id": result.pose_id,
            "score": result.score,
            "label": result.label,
        }


class Beta:
    def export(self, item):
        return {
            "pose_id": item.pose_id,
            "score": item.score,
            "label": item.label,
        }
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.detector_id == "repeated_export_dicts" for finding in findings)
    assert any("projection dict" in finding.title.lower() for finding in findings)
    assert any(
        finding.detector_id == "repeated_export_dicts" and finding.scaffold
        for finding in findings
    )
    assert any(
        finding.detector_id == "repeated_export_dicts" and finding.codemod_patch
        for finding in findings
    )


def test_collects_projection_helper_shapes_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def labels(items):
    return tuple(sorted(item.label for item in items))


def scores(items):
    return tuple(sorted(item.score for item in items))
""",
    )

    module = parse_python_modules(tmp_path)[0]
    shapes = collect_family_items(module, ProjectionHelperObservationFamily)

    assert {shape.projected_attribute for shape in shapes} == {"label", "score"}


def test_collects_accessor_wrapper_candidates_via_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Sample:
    def current(self):
        return self._current

    def update(self, current):
        self._current = current
""",
    )

    module = parse_python_modules(tmp_path)[0]
    candidates = collect_family_items(module, AccessorWrapperObservationFamily)

    assert {candidate.accessor_kind for candidate in candidates} == {"getter", "setter"}


def test_collects_field_observation_fibers_for_dataclass_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    score: float
    label: str


@dataclass
class BetaResult:
    pose_id: int
    score: float
    label: str
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)
    graph = ObservationGraph(
        tuple(item.structural_observation for item in observations)
    )
    fibers = graph.fibers_for(
        ObservationKind.FIELD, StructuralExecutionLevel.CLASS_BODY
    )

    pose_fiber = next(fiber for fiber in fibers if fiber.observed_name == "pose_id")
    assert len(pose_fiber.observations) == 2


def test_ignores_classvar_fields_via_generic_annotation_matcher(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class AlphaResult:
    pose_id: int
    cache: ClassVar[dict[str, int]] = {}


@dataclass
class BetaResult:
    pose_id: int
    cache: ClassVar[dict[str, int]] = {}
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)

    assert all(item.field_name != "cache" for item in observations)


def test_observation_graph_recovers_field_coherence_cohort(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    score: float
    label: str
    rank: int
    alpha_only: int


@dataclass
class BetaResult:
    pose_id: int
    score: float
    label: str
    rank: int
    beta_only: int


@dataclass
class GammaResult:
    pose_id: int
    score: float
    gamma_only: int
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)
    graph = ObservationGraph(
        tuple(item.structural_observation for item in observations)
    )

    cohorts = graph.coherence_cohorts_for(
        ObservationKind.FIELD,
        StructuralExecutionLevel.CLASS_BODY,
        minimum_witnesses=2,
        minimum_fibers=2,
    )

    cohort = next(
        item
        for item in cohorts
        if item.nominal_witnesses == ("AlphaResult", "BetaResult")
    )
    assert set(cohort.observed_names) == {"pose_id", "score", "label", "rank"}


def test_ignores_namespaced_classvar_fields_via_family_matcher(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
import typing
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    cache: typing.ClassVar[dict[str, int]] = {}
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)

    assert all(item.field_name != "cache" for item in observations)


def test_collects_namespaced_dataclass_fields_via_name_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
import dataclasses as dc


@dc.dataclass
class AlphaResult:
    pose_id: int
    score: float
""",
    )

    module = parse_python_modules(tmp_path)[0]
    observations = collect_family_items(module, FieldObservationFamily)

    assert {item.field_name for item in observations} == {"pose_id", "score"}


def test_detects_repeated_field_family_in_dataclasses(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    score: float
    label: str
    alpha_only: int


@dataclass
class BetaResult:
    pose_id: int
    score: float
    label: str
    beta_only: int
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_field_family"
    )

    assert finding.pattern_id == 5
    assert "pose_id" in finding.summary
    assert "ResultBase" in (finding.scaffold or "")
    assert "pose_id: int" in (finding.scaffold or "")


def test_does_not_merge_dataclass_fields_with_conflicting_types(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    score: float
    alpha_only: int


@dataclass
class BetaResult:
    pose_id: str
    score: float
    beta_only: int
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(
        finding.detector_id == "repeated_field_family" for finding in findings
    )


def test_plan_extracts_shared_fields_to_abc_base(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class AlphaController:
    def __init__(self, pose_id, score, label, alpha_only):
        self.pose_id = pose_id
        self.score = score
        self.label = label
        self.alpha_only = alpha_only


class BetaController:
    def __init__(self, pose_id, score, label, beta_only):
        self.pose_id = pose_id
        self.score = score
        self.label = label
        self.beta_only = beta_only
""",
    )

    findings = analyze_path(tmp_path)
    plans = build_refactor_plans(findings, tmp_path)
    plan = next(plan for plan in plans if plan.primary_pattern_id == 5)

    assert any(action.kind == "extract_shared_fields" for action in plan.actions)
    field_action = next(
        action for action in plan.actions if action.kind == "extract_shared_fields"
    )
    assert field_action.statement_operation == "move"
    assert "pose_id" in field_action.description


def test_json_payload_exposes_observation_graph(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass
class AlphaResult:
    pose_id: int
    score: float


def convert(kind, value):
    if kind == "numpy":
        return value
    elif kind == "cupy":
        return value
    return value
""",
    )

    modules = parse_python_modules(tmp_path)
    findings = analyze_path(tmp_path)
    payload = _json_payload(findings, [], modules)
    observations = cast(list[dict[str, object]], payload["observations"])
    fibers = cast(list[dict[str, object]], payload["fibers"])

    assert "observations" in payload
    assert "fibers" in payload
    assert any(item["observation_kind"] == "field" for item in observations)
    assert any(item["observation_kind"] == "literal_dispatch" for item in fibers)


def test_observation_graph_auto_includes_registered_observation_families(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
BASE_TO_LAZY = {}
SENTINEL = type("Sentinel", (), {})()


class Base:
    pass


LazyBase = type("LazyBase", (Base,), {})
BASE_TO_LAZY[Base] = LazyBase


def resolve(config, obj):
    if hasattr(config, "kind"):
        return config.kind
    for scope in [1]:
        for mro_type in type(obj).__mro__:
            if scope and mro_type:
                return scope, mro_type
    return SENTINEL
""",
    )

    graph = build_observation_graph(parse_python_modules(tmp_path))
    kinds = {item.observation_kind for item in graph.observations}

    assert ObservationKind.CONFIG_DISPATCH in kinds
    assert ObservationKind.RUNTIME_TYPE_GENERATION in kinds
    assert ObservationKind.LINEAGE_MAPPING in kinds
    assert ObservationKind.DUAL_AXIS_RESOLUTION in kinds
    assert ObservationKind.SENTINEL_TYPE in kinds


def test_ignores_constant_string_maps_for_pattern_three(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
LOOKUP = {
    "alpha": 1,
    "beta": 2,
    "gamma": 3,
}
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(finding.detector_id == "string_dispatch" for finding in findings)


def test_detects_module_level_dispatch_dict_with_callable_targets(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def alpha():
    return 1


def beta():
    return 2


def gamma():
    return 3


DISPATCH = {
    "alpha": alpha,
    "beta": beta,
    "gamma": gamma,
}
""",
    )

    findings = analyze_path(tmp_path)
    assert any(finding.detector_id == "string_dispatch" for finding in findings)


def test_ignores_non_branch_config_reads(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def resolve(config):
    port = config.napari_port
    return port
""",
    )

    findings = analyze_path(tmp_path)
    assert not any(finding.pattern_id == 4 for finding in findings)


def test_detects_numeric_literal_dispatch(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def render(pattern_id):
    if pattern_id == 3:
        return "dispatch"
    elif pattern_id == 5:
        return "abc"
    elif pattern_id == 14:
        return "schema"
    return "other"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "numeric_literal_dispatch"
    )
    assert "`pattern_id`" in finding.summary
    assert "3" in finding.summary
    assert finding.certification == "certified"


def test_detects_repeated_hardcoded_semantic_string(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
DEFAULT_CERTIFICATION = "strong_heuristic"


def first():
    return configure(certification="strong_heuristic")


def second():
    return configure(certification="strong_heuristic")
""",
    )

    findings = analyze_path(tmp_path)
    assert any(
        finding.detector_id == "repeated_hardcoded_strings" for finding in findings
    )


def test_detects_repeated_projection_helper_wrappers(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def dedupe(items):
    return items


def capability_labels(capabilities):
    return tuple(dedupe(tag.label for tag in capabilities))


def capability_distinctions(capabilities):
    return tuple(dedupe(tag.distinction for tag in capabilities))


def observation_labels(observations):
    return tuple(dedupe(tag.label for tag in observations))
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_projection_helpers"
    )

    assert "_render_projection" in (finding.scaffold or "")


def test_detects_accessor_wrapper_smell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Sample:
    def get_status(self):
        return self.status

    def set_status(self, status):
        self.status = status
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "accessor_wrapper"
    )

    assert "structural accessor wrapper" in finding.title
    assert "replace `Sample.get_status()` with `status`" in (finding.scaffold or "")


def test_detects_structural_accessor_wrappers_without_naming_convention(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Sample:
    def status(self):
        return self._status

    def store(self, status):
        self._status = status
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "accessor_wrapper"
    )

    assert "structural accessor wrapper" in finding.summary
    assert "read through" in finding.relation_context
    assert "replace `Sample.status()` with `status`" in (finding.scaffold or "")


def test_detects_single_structural_computed_property_candidate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Sample:
    def labels(self):
        return tuple(self._labels)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "accessor_wrapper"
    )

    assert "computed tuple" in finding.relation_context
    assert "an `@property` exposing `tuple(self._labels)`" in (finding.scaffold or "")


def test_detects_flattened_projection_property_local_minimum(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class AtomSet:
    coords: object
    radii: object
    elements: object


@dataclass(frozen=True)
class PreparedComplex:
    ligand: AtomSet
    pocket: AtomSet

    @property
    def ligand_coords(self):
        return self.ligand.coords

    @property
    def ligand_radii(self):
        return self.ligand.radii

    @property
    def pocket_coords(self):
        return self.pocket.coords

    @property
    def pocket_elements(self):
        return self.pocket.elements
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "flattened_projection_property"
    )

    assert "PreparedComplex" in finding.summary
    assert "ligand_coords" in finding.summary
    assert "pocket_elements" in finding.summary
    assert "obj.ligand.coords" in (finding.scaffold or "")
    assert "obj.pocket.elements" in (finding.scaffold or "")


def test_detects_transport_wrapper_chain(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class PocketRegion:
    coords: object
    elements: object


def extract_local_pocket_region_view(protein_coords, receptor_elements, box_center, box_size):
    return PocketRegion(coords=protein_coords, elements=receptor_elements)


def extract_local_pocket_region(protein_coords, receptor_elements, box_center, box_size):
    region = extract_local_pocket_region_view(
        protein_coords,
        receptor_elements,
        box_center,
        box_size,
    )
    return region.coords, region.elements


def _extract_local_pocket_coords_and_elements(
    *,
    protein_coords,
    receptor_elements,
    box_center,
    box_size,
):
    return extract_local_pocket_region(
        protein_coords=protein_coords,
        receptor_elements=receptor_elements,
        box_center=box_center,
        box_size=box_size,
    )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "wrapper_chain"
    )

    assert "extract_local_pocket_region" in finding.summary
    assert "_extract_local_pocket_coords_and_elements" in finding.summary
    assert "extract_local_pocket_region_view" in (finding.scaffold or "")


def test_uses_nominal_metric_dataclasses(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def render(pattern_id):
    if pattern_id == 3:
        return "dispatch"
    elif pattern_id == 5:
        return "abc"
    elif pattern_id == 14:
        return "schema"
    return "other"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "numeric_literal_dispatch"
    )

    assert isinstance(finding.metrics, DispatchCountMetrics)
    assert finding.metrics.dispatch_site_count == 3
    assert finding.metrics.dispatch_axis == "pattern_id"


def test_detects_semantic_metrics_dict_bag_and_recommends_nominal_class(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class RefactorFinding:
    metrics: object


def build():
    return RefactorFinding(metrics={"dispatch_site_count": len([1, 2, 3])})
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "semantic_dict_bag"
    )

    assert "DispatchCountMetrics" in (finding.scaffold or "")
    assert "CountedDispatchMetrics" in (finding.scaffold or "")


def test_detects_local_impact_dict_bag_and_recommends_impact_delta(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
def estimate():
    impact = {
        "lower_bound_removable_loc": 0,
        "upper_bound_removable_loc": 0,
        "loci_of_change_before": 0,
        "loci_of_change_after": 0,
        "repeated_mappings_centralized": 0,
        "dispatch_sites_eliminated": 0,
        "registration_sites_removed": 0,
        "shared_algorithm_sites_centralized": 0,
    }
    impact["dispatch_sites_eliminated"] = 2
    return impact
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "semantic_dict_bag"
    )

    assert "ImpactDelta" in (finding.scaffold or "")


def test_builds_composed_subsystem_plan(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
REGISTRY = {}


class RuntimePlan:
    def __init__(self, pose_id, score, label):
        self.pose_id = pose_id
        self.score = score
        self.label = label


class Alpha:
    def _prepare(self, item):
        ready = self.normalize(item)
        checked = self.validate(ready)
        return self.finish(checked)

    def build(self, candidate):
        return RuntimePlan(
            pose_id=candidate.pose_id,
            score=candidate.score,
            label=candidate.label,
        )


class Beta:
    def _assemble(self, value):
        ready = self.normalize(value)
        checked = self.validate(ready)
        return self.finish(checked)

    def build(self, entry):
        return RuntimePlan(
            pose_id=entry.pose_id,
            score=entry.score,
            label=entry.label,
        )


REGISTRY["alpha"] = Alpha
REGISTRY["beta"] = Beta
""",
    )

    findings = analyze_path(tmp_path)
    plans = build_refactor_plans(findings, tmp_path)

    assert plans
    plan = plans[0]
    assert plan.primary_pattern_id == 5
    assert 6 in plan.secondary_pattern_ids
    assert 14 in plan.secondary_pattern_ids
    assert plan.outcome.loci_of_change_before > plan.outcome.loci_of_change_after
    assert plan.outcome.registration_sites_removed == 2
    assert plan.outcome.repeated_mappings_centralized >= 3
    assert any(action.kind == "create_abc_base" for action in plan.actions)
    assert any(action.kind == "create_metaclass" for action in plan.actions)
    extract_action = next(
        action for action in plan.actions if action.kind == "extract_template_method"
    )
    assert extract_action.statement_operation == "move"
    assert extract_action.statement_sites
    assert "self.normalize" in extract_action.description
    mapping_action = next(
        action
        for action in plan.actions
        if action.kind == "create_authoritative_schema"
    )
    assert mapping_action.create_symbol == "RuntimePlan.from_source"
    assert "name-for-name boilerplate" in mapping_action.description
    replace_action = next(
        action for action in plan.actions if action.kind == "replace_mapping_sites"
    )
    assert replace_action.statement_operation == "replace"
    assert replace_action.replace_with == "RuntimePlan.from_source(candidate)"


def test_markdown_output_can_include_subsystem_plans(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    def _prepare(self, item):
        ready = self.normalize(item)
        checked = self.validate(ready)
        return self.finish(checked)


class Beta:
    def _build(self, value):
        ready = self.normalize(value)
        checked = self.validate(ready)
        return self.finish(checked)
""",
    )

    findings = analyze_path(tmp_path)
    plans = build_refactor_plans(findings, tmp_path)
    output = _format_markdown(findings, plans)

    assert "Subsystem plans:" in output
    assert "Primary pattern:" in output
    assert "Outcome:" in output
    assert "Action:" in output
    assert "Action sites:" in output


def test_detects_manual_family_roster_for_detector_registry(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class IssueDetector(ABC):
    pass


class AlphaDetector(IssueDetector):
    pass


class BetaDetector(IssueDetector):
    pass


def default_detectors():
    return (AlphaDetector(), BetaDetector())
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding for finding in findings if finding.detector_id == "manual_family_roster"
    )

    assert "default_detectors" in finding.summary
    assert "IssueDetector" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__key_extractor__" in (finding.scaffold or "")
    assert "RegisteredIssueDetector.__registry__.values()" in (finding.scaffold or "")


def test_detects_fragmented_pattern_planning_tables(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class PatternId:
    ABC_TEMPLATE_METHOD = "abc"
    AUTHORITATIVE_SCHEMA = "schema"
    AUTO_REGISTER_META = "auto"


_PATTERN_DEPENDENCIES = {
    PatternId.ABC_TEMPLATE_METHOD: {PatternId.AUTHORITATIVE_SCHEMA},
    PatternId.AUTHORITATIVE_SCHEMA: {PatternId.AUTO_REGISTER_META},
    PatternId.AUTO_REGISTER_META: set(),
}


_PATTERN_PRIORITY = {
    PatternId.ABC_TEMPLATE_METHOD: 80,
    PatternId.AUTHORITATIVE_SCHEMA: 60,
    PatternId.AUTO_REGISTER_META: 50,
}


_PATTERN_BUILDERS = {
    PatternId.ABC_TEMPLATE_METHOD: build_abc,
    PatternId.AUTHORITATIVE_SCHEMA: build_schema,
    PatternId.AUTO_REGISTER_META: build_registry,
}
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "fragmented_family_authority"
    )

    assert "_PATTERN_DEPENDENCIES" in finding.summary
    assert "PatternId" in finding.summary
    assert "class PatternIdSpec" in (finding.scaffold or "")


def test_detects_existing_nominal_authority_reuse(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class EventCarrierBase(ABC):
    file_path: str
    line: int
    subject_name: str
    payload: tuple[str, ...]


@dataclass(frozen=True)
class DetachedEventCarrier:
    file_path: str
    line: int
    subject_name: str
    payload: tuple[str, ...]
    status: str
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "existing_nominal_authority_reuse"
    )

    assert "DetachedEventCarrier" in finding.summary
    assert "EventCarrierBase" in finding.summary
    assert "EventCarrierBase" in (finding.scaffold or "")


def test_detects_pass_through_nominal_wrapper(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProbeRoute(ABC):
    @abstractmethod
    def generate(self, request):
        raise NotImplementedError

    @abstractmethod
    def score(self, request, batch):
        raise NotImplementedError


@dataclass(frozen=True)
class ProbeRouteWitness:
    route: ProbeRoute

    def generate(self, request):
        return self.route.generate(request)

    def score(self, request, batch):
        return self.route.score(request, batch)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "pass_through_nominal_wrapper"
    )

    assert "ProbeRouteWitness" in finding.summary
    assert "ProbeRoute" in finding.summary
    assert "type consumers against `ProbeRoute` directly" in (finding.scaffold or "")


def test_detects_trivial_forwarding_wrapper(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class ModeRunner:
    @classmethod
    def for_mode(cls, mode):
        return cls()

    def attempt_modes(self):
        return ("fast", "safe")


class Owner:
    def __init__(self, mode):
        self.mode = mode

    def attempt_modes(self):
        return ModeRunner.for_mode(self.mode).attempt_modes()


def refinement_mode_attempt_chain(mode):
    return ModeRunner.for_mode(mode).attempt_modes()
""",
    )

    findings = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "trivial_forwarding_wrapper"
    ]

    assert len(findings) == 2
    assert any("Owner.attempt_modes" in finding.summary for finding in findings)
    assert any(
        "refinement_mode_attempt_chain" in finding.summary for finding in findings
    )
    assert all(
        "call `ModeRunner.for_mode.attempt_modes` directly" in (finding.scaffold or "")
        for finding in findings
    )


def test_detects_public_api_private_delegate_shell(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/scoring.py",
        """
class _Router:
    @classmethod
    def for_engine(cls, engine):
        return cls()

    def score(self, kwargs):
        return kwargs["value"]


def route_scoring(engine, **kwargs):
    return _Router.for_engine(engine).score(kwargs)
""",
    )
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        """
from pkg.scoring import route_scoring as score_route


def run_pipeline():
    return score_route("fast", value=1.0)
""",
    )
    _write_module(
        tmp_path,
        "pkg/api.py",
        """
import pkg.scoring as scoring


def score_request():
    return scoring.route_scoring("safe", value=2.0)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "public_api_private_delegate_shell"
    )

    assert "route_scoring" in finding.summary
    assert "_Router" in finding.summary
    assert "2 external call site(s)" in finding.summary
    assert "public facade/ABC/policy authority" in (finding.codemod_patch or "")


def test_detects_public_api_private_delegate_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/scoring.py",
        """
class _Router:
    @classmethod
    def for_engine(cls, engine):
        return cls()

    def score(self, payload):
        return payload["value"]

    def requires_electrostatics(self):
        return True


def route_scoring(engine, **payload):
    return _Router.for_engine(engine).score(payload)


def scoring_engine_requires_electrostatics(engine):
    return _Router.for_engine(engine).requires_electrostatics()
""",
    )
    _write_module(
        tmp_path,
        "pkg/pipeline.py",
        """
from pkg.scoring import route_scoring, scoring_engine_requires_electrostatics


def run_pipeline():
    if scoring_engine_requires_electrostatics("fast"):
        return route_scoring("fast", value=1.0)
    return 0.0
""",
    )
    _write_module(
        tmp_path,
        "pkg/api.py",
        """
import pkg.scoring as scoring


def score_request():
    if scoring.scoring_engine_requires_electrostatics("safe"):
        return scoring.route_scoring("safe", value=2.0)
    return 0.0
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "public_api_private_delegate_family"
    )

    assert "route_scoring" in finding.summary
    assert "scoring_engine_requires_electrostatics" in finding.summary
    assert "_Router" in finding.summary
    assert "public facade" in (finding.codemod_patch or "")


def test_detects_nominal_policy_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class ProofCasePolicy:
    @classmethod
    def for_case(cls, proof_case):
        return cls()

    def decision(self):
        return "certified"

    def certificate_chain_error(self):
        return None


class CertifiedPlan:
    def __init__(self, proof_case):
        self.proof_case = proof_case

    @property
    def decision(self):
        return ProofCasePolicy.for_case(self.proof_case).decision()

    @property
    def certificate_chain_error(self):
        return ProofCasePolicy.for_case(self.proof_case).certificate_chain_error()
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "nominal_policy_surface"
    )

    assert "CertifiedPlan" in finding.summary
    assert "decision" in finding.summary
    assert "certificate_chain_error" in finding.summary
    assert "ProofCasePolicy.for_case" in finding.summary
    assert "explicit policy accessor" in (finding.scaffold or "")


def test_detects_repeated_finding_assembly_pipeline(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class PerModuleIssueDetector:
    pass


class AlphaDetector(PerModuleIssueDetector):
    def _findings_for_module(self, module, config):
        findings = []
        for candidate in alpha_candidates(module):
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    summarize_alpha(candidate),
                    alpha_evidence(candidate),
                    scaffold=alpha_scaffold(candidate),
                    codemod_patch=alpha_patch(candidate),
                    metrics=AlphaMetrics(site_count=1),
                )
            )
        return findings


class BetaDetector(PerModuleIssueDetector):
    def _findings_for_module(self, module, config):
        findings = []
        for entry in beta_candidates(module):
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    summarize_beta(entry),
                    beta_evidence(entry),
                    scaffold=beta_scaffold(entry),
                    codemod_patch=beta_patch(entry),
                    metrics=BetaMetrics(site_count=1),
                )
            )
        return findings


class GammaDetector(PerModuleIssueDetector):
    def _findings_for_module(self, module, config):
        findings = []
        for witness in gamma_candidates(module):
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    summarize_gamma(witness),
                    gamma_evidence(witness),
                    scaffold=gamma_scaffold(witness),
                    codemod_patch=gamma_patch(witness),
                    metrics=GammaMetrics(site_count=1),
                )
            )
        return findings
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "finding_assembly_pipeline"
    )

    assert "AlphaDetector" in finding.summary
    assert "CandidateFindingDetector" in (finding.scaffold or "")


def test_detects_guarded_delegator_spec_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class FunctionObservationSpec:
    pass


class ProjectionObservationSpec(FunctionObservationSpec):
    def build_from_function(self, parsed_module, function, observation):
        if observation.class_name is not None:
            return None
        return _projection_helper_shape_from_function(parsed_module, function)


class AccessorObservationSpec(FunctionObservationSpec):
    def build_from_function(self, parsed_module, function, observation):
        if observation.class_name is None:
            return None
        return _accessor_wrapper_candidate_from_function(parsed_module, observation.class_name, function)


class SpecAssignmentObservationSpec(FunctionObservationSpec):
    def build_from_function(self, parsed_module, function, observation):
        if observation.function_name is None:
            return None
        return _spec_candidate_from_function(parsed_module, function)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "guarded_delegator_spec"
    )

    assert "Observation specs" in finding.summary
    assert "ScopeFilteredSpec" in (finding.scaffold or "")


def test_detects_projection_style_builder_authority(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class SearchContext:
    def __init__(
        self,
        *,
        base_coords,
        score_fn,
        batch_fn,
        pruning_energy,
        local_mask,
        score_is_exact,
    ):
        self.base_coords = base_coords
        self.score_fn = score_fn
        self.batch_fn = batch_fn
        self.pruning_energy = pruning_energy
        self.local_mask = local_mask
        self.score_is_exact = score_is_exact


def build_from_runtime(prepared, runtime):
    return SearchContext(
        base_coords=prepared.base_coords,
        score_fn=prepared.score_fn,
        batch_fn=prepared.batch_fn,
        pruning_energy=None if runtime is None else runtime.pruning_energy,
        local_mask=None if runtime is None else runtime.local_mask,
        score_is_exact=True if runtime is None else runtime.score_is_exact,
    )


def build_from_request(request, runtime):
    return SearchContext(
        base_coords=request.base_coords,
        score_fn=request.score_fn,
        batch_fn=request.batch_fn,
        pruning_energy=runtime.pruning_energy,
        local_mask=runtime.local_mask,
        score_is_exact=runtime.score_is_exact,
    )


def build_sequential(prepared):
    return SearchContext(
        base_coords=prepared.base_coords,
        score_fn=prepared.score_fn,
        batch_fn=prepared.batch_fn,
        pruning_energy=None,
        local_mask=None,
        score_is_exact=True,
    )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "projection_builder_authority"
    )

    assert "SearchContext" in finding.summary
    assert "projection sites" in finding.summary
    assert "SearchContextBuilder" in (finding.scaffold or "")


def test_detects_repeated_structural_observation_projection(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class ProjectionRecord:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MethodShape:
    @property
    def projection_record(self):
        return ProjectionRecord(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            primary_name=self.class_name,
            line=self.lineno,
            category=self.observation_kind,
            observed_name=self.method_name,
            fiber_key=self.method_name,
        )


class BuilderShape:
    @property
    def projection_record(self):
        return ProjectionRecord(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            primary_name=self.class_name,
            line=self.lineno,
            category=self.observation_kind,
            observed_name=self.builder_name,
            fiber_key=self.builder_name,
        )


class ExportShape:
    @property
    def projection_record(self):
        return ProjectionRecord(
            file_path=self.file_path,
            owner_symbol=self.symbol,
            primary_name=self.class_name,
            line=self.lineno,
            category=self.observation_kind,
            observed_name=self.export_name,
            fiber_key=self.export_name,
        )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "structural_observation_projection"
    )

    assert "ProjectionRecord" in finding.summary
    assert "projection_record" in finding.summary
    assert "ProjectionTemplate" in (finding.scaffold or "")


def test_detects_repeated_property_alias_hooks_across_subclasses(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class ProjectionTemplate(ABC):
    @property
    def observation_kind(self):
        raise NotImplementedError


class AlphaProjection(ProjectionTemplate):
    @property
    def observation_line(self):
        return self.lineno


class BetaProjection(ProjectionTemplate):
    @property
    def observation_line(self):
        return self.lineno
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "repeated_property_alias_hooks"
    )

    assert "ProjectionTemplate" in finding.summary
    assert "observation_line" in finding.summary
    assert "self.lineno" in finding.summary


def test_detects_constant_property_hooks_across_subclasses(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class ObservationKind:
    FIELD = "field"
    METHOD = "method"


class ProjectionTemplate(ABC):
    @property
    def observation_kind(self):
        raise NotImplementedError


class AlphaProjection(ProjectionTemplate):
    @property
    def observation_kind(self):
        return ObservationKind.FIELD


class BetaProjection(ProjectionTemplate):
    @property
    def observation_kind(self):
        return ObservationKind.METHOD
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "constant_property_hooks"
    )

    assert "ProjectionTemplate" in finding.summary
    assert "observation_kind" in finding.summary
    assert "ObservationKind.FIELD" in finding.summary
    assert "ObservationKind.METHOD" in finding.summary


def test_detects_reflective_self_attribute_escape(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class ProjectionTemplate(ABC):
    @property
    def path_text(self):
        return getattr(self, "file_path")
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "reflective_self_attribute_escape"
    )

    assert "getattr(self, 'file_path')" in finding.summary
    assert "file_path" in (finding.scaffold or "")


def test_detects_helper_backed_observation_spec_wrappers(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class TaskAdapter(ABC):
    pass


class HelperBackedTaskAdapter(TaskAdapter, ABC):
    pass


class ClassTaskAdapter(HelperBackedTaskAdapter):
    def build(self, parsed_module, function, observation):
        return tuple(class_marker_events(parsed_module, function))


class InterfaceTaskAdapter(HelperBackedTaskAdapter):
    def build(self, parsed_module, function, observation):
        return interface_event(parsed_module, function)


class DynamicTaskAdapter(HelperBackedTaskAdapter):
    def build(self, parsed_module, function, observation):
        return tuple(dynamic_events(parsed_module, function))


class ProjectionTaskAdapter(HelperBackedTaskAdapter):
    def build(self, parsed_module, function, observation):
        return projection_event(parsed_module, function)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "helper_backed_observation_spec"
    )

    assert "ClassTaskAdapter" in finding.summary
    assert "HelperBackedTaskAdapter" in finding.summary
    assert "HelperBackedTemplate" in (finding.scaffold or "")


def test_detects_dynamic_self_field_selection(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class CountedDispatchMetrics(ABC):
    count_field_name = "branch_site_count"

    def _count_value(self):
        return int(getattr(self, self.count_field_name))
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "dynamic_self_field_selection"
    )

    assert "getattr(self, self.count_field_name)" in finding.summary
    assert "count_value" in (finding.scaffold or "")


def test_detects_string_backed_reflective_nominal_lookup_via_globals(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class Route:
    pass


class DirectRoute(Route):
    pass


class GuidedRoute(Route):
    pass


class RoutedRequest(ABC):
    route_type_name = None

    def create_route(self):
        return globals()[self.route_type_name]()


class DirectRequest(RoutedRequest):
    route_type_name = "DirectRoute"


class GuidedRequest(RoutedRequest):
    route_type_name = "GuidedRoute"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "string_backed_reflective_nominal_lookup"
    )

    assert "route_type_name" in finding.summary
    assert "globals[]" in finding.summary


def test_detects_string_backed_reflective_nominal_lookup_via_getattr(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class BackendFamily:
    ALPHA = object()
    BETA = object()


class Router(ABC):
    backend_name = None

    def resolve(self):
        return getattr(BackendFamily, self.backend_name)


class AlphaRouter(Router):
    backend_name = "ALPHA"


class BetaRouter(Router):
    backend_name = "BETA"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "string_backed_reflective_nominal_lookup"
    )

    assert "backend_name" in finding.summary
    assert "getattr" in finding.summary


def test_detects_string_backed_reflective_nominal_lookup_via_dict_get(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class WitnessSelector(ABC):
    witness_field_name = None

    def witness(self, state):
        return state.__dict__.get(type(self).witness_field_name)


class AlphaWitnessSelector(WitnessSelector):
    witness_field_name = "alpha"


class BetaWitnessSelector(WitnessSelector):
    witness_field_name = "beta"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "string_backed_reflective_nominal_lookup"
    )

    assert "witness_field_name" in finding.summary
    assert "dict.get" in finding.summary


def test_detects_classvar_only_sibling_leaf(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class ProjectionLeaf(ABC):
    pass


class AlphaProjection(ProjectionLeaf):
    payload_cls = Alpha
    renderer_cls = AlphaRenderer


class BetaProjection(ProjectionLeaf):
    payload_cls = Beta
    renderer_cls = BetaRenderer


class GammaProjection(ProjectionLeaf):
    payload_cls = Gamma
    renderer_cls = GammaRenderer
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "classvar_only_sibling_leaf"
    )

    assert "AlphaProjection" in finding.summary
    assert "payload_cls" in finding.summary
    assert "renderer_cls" in finding.summary
    assert "declarative family-definition table" in (finding.codemod_patch or "")


def test_detects_type_indexed_definition_boilerplate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class CollectedFamily(ABC):
    pass


class RegisteredObservationFamilyDefinition(ABC):
    pass


class AlphaFamilyDefinition(RegisteredObservationFamilyDefinition):
    item_type = Alpha
    spec_root = AlphaSpec


AlphaFamily = AlphaFamilyDefinition.family_type


class BetaFamilyDefinition(RegisteredObservationFamilyDefinition):
    item_type = Beta
    spec_root = BetaSpec


BetaFamily = BetaFamilyDefinition.family_type


class GammaFamilyDefinition(RegisteredObservationFamilyDefinition):
    item_type = Gamma
    spec_root = GammaSpec


GammaFamily = GammaFamilyDefinition.family_type
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "type_indexed_definition_boilerplate"
    )

    assert "AlphaFamilyDefinition" in finding.summary
    assert "AlphaFamily" in finding.summary
    assert "typed declaration table" in (finding.codemod_patch or "")


def test_detects_manual_derived_export_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class PublicSpecRoot(ABC):
    pass


class HandlerFamilyRoot(ABC):
    pass


class AlphaSpec(PublicSpecRoot):
    pass


class BetaSpec(PublicSpecRoot):
    pass


class GammaSpec(PublicSpecRoot):
    pass


class DeltaHandler(HandlerFamilyRoot):
    pass


class EpsilonHandler(HandlerFamilyRoot):
    pass


class ZetaHandler(HandlerFamilyRoot):
    pass


_STATIC_EXPORT_NAMES = (
    "AlphaSpec",
    "BetaSpec",
    "GammaSpec",
    "DeltaHandler",
    "EpsilonHandler",
    "ZetaHandler",
)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "derived_export_surface"
    )

    assert "_STATIC_EXPORT_NAMES" in finding.summary
    assert (
        "PublicSpecRoot" in finding.summary
        or "HandlerFamilyRoot" in finding.summary
    )
    assert "public_exports" in (finding.scaffold or "")


def test_detects_manual_derived_index_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC


class CommandRoot(ABC):
    pass


class AlphaCommand(CommandRoot):
    pass


class BetaCommand(CommandRoot):
    pass


class GammaCommand(CommandRoot):
    pass


COMMAND_BY_NAME = {
    "alpha": AlphaCommand,
    "beta": BetaCommand,
    "gamma": GammaCommand,
}
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "derived_indexed_surface"
    )

    assert "COMMAND_BY_NAME" in finding.summary
    assert "CommandRoot" in finding.summary
    assert "derived_index" in (finding.scaffold or "")


def test_detects_manual_public_api_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class Alpha:
    pass


class Beta:
    pass


def gamma():
    return 1


def delta():
    return 2


__all__ = ["Alpha", "Beta", "gamma", "delta"]
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "manual_public_api_surface"
    )

    assert "__all__" in finding.summary
    assert "public API" in finding.title
    assert "is_public_api_export" in (finding.scaffold or "")


def test_detects_repeated_export_policy_predicates(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/alpha.py",
        """
class Root:
    pass


def _is_public_alpha_export(name, value):
    if name.startswith("_"):
        return False
    if not isinstance(value, type) or value.__module__ != __name__:
        return False
    return issubclass(value, Root)


__all__ = sorted(
    name for name, value in globals().items() if _is_public_alpha_export(name, value)
)
""",
    )
    _write_module(
        tmp_path,
        "pkg/beta.py",
        """
class Root:
    pass


def _is_public_beta_export(name, value):
    if name.startswith("_"):
        return False
    if not isinstance(value, type) or value.__module__ != __name__:
        return False
    return issubclass(value, Root)


__all__ = sorted(
    name for name, value in globals().items() if _is_public_beta_export(name, value)
)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "export_policy_predicate"
    )

    assert "_is_public_alpha_export" in finding.summary
    assert "_is_public_beta_export" in finding.summary
    assert "DerivedSurfacePolicy" in (finding.scaffold or "")


def test_detects_manual_registered_union_surface(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class PluginRegistry:
    @classmethod
    def registered_plugins(cls):
        return ()


class HandlerRegistry:
    @classmethod
    def registered_plugins(cls):
        return ()


def collect_everything():
    for item in PluginRegistry.registered_plugins() + HandlerRegistry.registered_plugins():
        yield item
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "registered_union_surface"
    )

    assert "collect_everything" in finding.summary
    assert "registered_plugins" in finding.summary
    assert "UnifiedRegistryRoot" in (finding.scaffold or "")
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "__key_extractor__" in (finding.scaffold or "")
    assert "UnifiedRegistryRoot.__registry__.values()" in (finding.scaffold or "")


def test_detects_repeated_registry_traversal_substrate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class PluginRegistry:
    @classmethod
    def all_registered_plugins(cls):
        seen = set()
        ordered = []
        queue = list(cls.__subclasses__())
        while queue:
            current = queue.pop(0)
            queue.extend(current.__subclasses__())
            registry = current.__dict__.get("_registered_plugin_types")
            if registry is None:
                continue
            for plugin_type in registry:
                if plugin_type in seen:
                    continue
                seen.add(plugin_type)
                ordered.append(plugin_type())
        return tuple(ordered)


class HandlerRegistry:
    @classmethod
    def all_registered_handlers(cls):
        seen = set()
        ordered = []
        queue = list(cls.__subclasses__())
        while queue:
            current = queue.pop(0)
            queue.extend(current.__subclasses__())
            registry = current.__dict__.get("_registered_handler_types")
            if registry is None:
                continue
            for handler_type in registry:
                if handler_type in seen:
                    continue
                seen.add(handler_type)
                ordered.append(handler_type)
        return tuple(ordered)
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "registry_traversal_substrate"
    )

    assert "all_registered_plugins" in finding.summary
    assert "all_registered_handlers" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "materialize_family" in (finding.scaffold or "")
    assert "root.__registry__.values()" in (finding.scaffold or "")


def test_detects_cross_module_registry_traversal_substrate(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/plugins.py",
        """
class PluginBase:
    pass


def all_plugins():
    seen = set()
    ordered = []
    queue = list(PluginBase.__subclasses__())
    while queue:
        current = queue.pop(0)
        queue.extend(current.__subclasses__())
        if not current.__dict__.get("plugin_name"):
            continue
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
    return tuple(sorted(ordered, key=lambda item: item.__name__))
""",
    )
    _write_module(
        tmp_path,
        "pkg/metrics.py",
        """
from dataclasses import is_dataclass


class MetricBase:
    pass


def all_metrics():
    discovered = []
    queue = list(MetricBase.__subclasses__())
    while queue:
        current = queue.pop(0)
        queue.extend(current.__subclasses__())
        if not is_dataclass(current):
            continue
        discovered.append(current)
    return tuple(sorted(discovered, key=lambda item: item.__name__))
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "registry_traversal_substrate"
        and "all_plugins" in finding.summary
        and "all_metrics" in finding.summary
    )

    assert "materialize_family" in (finding.scaffold or "")
    assert "root.__registry__.values()" in (finding.scaffold or "")


def test_detects_alternate_constructor_family(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
class RegistrationShape:
    @classmethod
    def from_assignment(cls, parsed_module, node: Assign, registry_name, key_fingerprint):
        return cls(
            file_path=parsed_module.path,
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=node.value.id,
            key_fingerprint=key_fingerprint,
            key_expression=node.target,
            registration_style="assignment",
        )

    @classmethod
    def from_registration_call(cls, parsed_module, node: Call, registry_name, key_fingerprint):
        return cls(
            file_path=parsed_module.path,
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=node.func.id,
            key_fingerprint=key_fingerprint,
            key_expression=node.args[0],
            registration_style="call",
        )

    @classmethod
    def from_decorator(cls, parsed_module, node: ClassDef, registry_name, key_fingerprint):
        return cls(
            file_path=parsed_module.path,
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=node.name,
            key_fingerprint=key_fingerprint,
            key_expression=node.name,
            registration_style="decorator",
        )
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "alternate_constructor_family"
    )

    assert "RegistrationShape" in finding.summary
    assert "from_assignment" in finding.summary
    assert "@singledispatchmethod" in (finding.scaffold or "")


def test_detects_implicit_self_contract_mixins(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


class RequestContract:
    payload: object
    cache: object


class PreparationBase:
    pass


class PayloadPreparationMixin:
    def prepare(self):
        request = cast(Any, self)
        payload = request.payload
        return ("prepared", payload, request.cache)

    def prepare_typed(self):
        request = cast(RequestContract, self)
        return ("typed", request.payload, request.cache)


@dataclass(frozen=True)
class AlphaPreparation(PayloadPreparationMixin, PreparationBase):
    payload: object
    cache: object


@dataclass(frozen=True)
class BetaPreparation(PayloadPreparationMixin, PreparationBase):
    payload: object
    cache: object
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "implicit_self_contract_mixin"
    )

    assert "PayloadPreparationMixin" in finding.summary
    assert "cast(..., self)" in (finding.codemod_patch or "")
    assert "RequestContract" in finding.summary
    assert "AlphaPreparation" in finding.summary


def test_detects_empty_leaf_product_families(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/mod.py",
        """
from abc import ABC, abstractmethod


class DispatchFamily(ABC):
    @classmethod
    @abstractmethod
    def matches_mode(cls, request) -> bool:
        raise NotImplementedError

    @abstractmethod
    def run(self, request):
        raise NotImplementedError


class GuidedPolicy(DispatchFamily, ABC):
    @classmethod
    def matches_mode(cls, request) -> bool:
        return request.mode == "guided"


class HybridPolicy(DispatchFamily, ABC):
    @classmethod
    def matches_mode(cls, request) -> bool:
        return request.mode == "hybrid"


class LocalTemplatesMixin(ABC):
    def templates(self, request):
        return request.local_templates


class RemoteTemplatesMixin(ABC):
    def templates(self, request):
        return request.remote_templates


class LocalGuidedPolicy(LocalTemplatesMixin, GuidedPolicy):
    pass


class RemoteGuidedPolicy(RemoteTemplatesMixin, GuidedPolicy):
    pass


class LocalHybridPolicy(LocalTemplatesMixin, HybridPolicy):
    pass


class RemoteHybridPolicy(RemoteTemplatesMixin, HybridPolicy):
    pass
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "empty_leaf_product_family"
    )

    assert "LocalTemplatesMixin" in finding.summary
    assert "GuidedPolicy" in finding.summary
    assert "Cartesian-product leaf classes" in (finding.codemod_patch or "")


def test_detects_residual_closed_axis_branching(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "pkg/authority.py",
        """
from abc import ABC
from enum import Enum
from typing import ClassVar


class KeyedNominalFamily(ABC):
    registry_key_attr: ClassVar[str]


class ScoringFamily(Enum):
    FAST = "fast"
    ACCURATE = "accurate"


class ScoringPolicy(KeyedNominalFamily[ScoringFamily], ABC):
    registry_key_attr = "scoring_family"
    scoring_family: ClassVar[ScoringFamily]


class FastPolicy(ScoringPolicy):
    scoring_family = ScoringFamily.FAST


class AccuratePolicy(ScoringPolicy):
    scoring_family = ScoringFamily.ACCURATE
""",
    )
    _write_module(
        tmp_path,
        "pkg/consumer.py",
        """
from pkg.authority import ScoringFamily


def resolve_backend(scoring_family: ScoringFamily) -> str:
    if scoring_family == ScoringFamily.FAST:
        return "jit"
    return "exact"
""",
    )

    findings = analyze_path(tmp_path)
    finding = next(
        finding
        for finding in findings
        if finding.detector_id == "residual_closed_axis_branching"
    )

    assert "resolve_backend" in finding.summary
    assert "ScoringFamily" in finding.summary
    assert "ScoringPolicy" in finding.summary
    assert "from metaclass_registry import AutoRegisterMeta" in (finding.scaffold or "")
    assert "return cls.__registry__[key]()" in (finding.scaffold or "")
